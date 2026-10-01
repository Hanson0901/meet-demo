from __future__ import annotations

import gc
import hashlib
import json
import os
import re
from threading import RLock
from typing import Callable, Mapping, Protocol
from pathlib import Path

from .logging import EventLogger
from .normalize import convert_traditional


class AnalysisCheckpointStore(Protocol):
    def get_analysis_checkpoint(
        self,
        meeting_id: str,
        run_key: str,
        phase: str,
        round_number: int,
        item_key: str,
        input_hash: str,
    ) -> dict | None: ...

    def save_analysis_checkpoint(
        self,
        meeting_id: str,
        run_key: str,
        phase: str,
        round_number: int,
        item_key: str,
        input_hash: str,
        result: dict,
    ) -> None: ...

    def retain_analysis_checkpoint_run(self, meeting_id: str, run_key: str) -> None: ...


class OutputTokenLimitError(RuntimeError):
    """Raised when the model stopped before completing a JSON response."""


class LLMOrchestrator:
    """Local llama.cpp meeting analysis with token-budgeted map/reduce for long transcripts."""

    CONTEXT_MARGIN_TOKENS = 512
    MIN_CONTEXT_TOKENS = 8192
    MAX_CONTEXT_TOKENS = 16384
    CHECKPOINT_VERSION = "hierarchical-checkpoints-v1"
    TOKEN_SETTING_SPECS = {
        "direct_max_tokens": {
            "env": "LLAMA_CPP_DIRECT_MAX_TOKENS", "default": 2400,
            "min": 256, "max": 8192, "label": "直接摘要判斷門檻",
        },
        "chunk_tokens": {
            "env": "LLAMA_CPP_CHUNK_TOKENS", "default": 1800,
            "min": 256, "max": 8192, "label": "每段逐字稿輸入目標",
        },
        "map_max_tokens": {
            "env": "LLAMA_CPP_MAP_MAX_TOKENS", "default": 3072,
            "min": 512, "max": 8192, "label": "每段整理輸出上限",
        },
        "reduce_input_tokens": {
            "env": "LLAMA_CPP_REDUCE_INPUT_TOKENS", "default": 4096,
            "min": 512, "max": 8192, "label": "每批合併輸入目標",
        },
        "reduce_max_tokens": {
            "env": "LLAMA_CPP_REDUCE_MAX_TOKENS", "default": 4096,
            "min": 512, "max": 8192, "label": "每批合併輸出上限",
        },
        "final_max_tokens": {
            "env": "LLAMA_CPP_MAX_TOKENS", "default": 4096,
            "min": 512, "max": 8192, "label": "最終摘要輸出上限",
        },
    }

    def __init__(
        self,
        logger: EventLogger,
        context_tokens: int | None = None,
        *,
        token_settings: Mapping[str, int] | None = None,
        checkpoint_store: AnalysisCheckpointStore | None = None,
    ):
        self.logger = logger
        self.model = None
        self.model_path: str | None = None
        self._model_lock = RLock()
        self._token_settings_lock = RLock()
        self.checkpoint_store = checkpoint_store
        self._token_settings = self._environment_token_settings()
        if token_settings:
            self._token_settings.update(self.validate_token_settings(token_settings))
        if context_tokens is None:
            configured_context = self._int_setting(
                "LLAMA_CPP_CONTEXT", self.MAX_CONTEXT_TOKENS, self.MIN_CONTEXT_TOKENS
            )
            if configured_context > self.MAX_CONTEXT_TOKENS:
                self.logger.emit(
                    "LLM",
                    f"LLAMA_CPP_CONTEXT exceeds the supported safe limit; using {self.MAX_CONTEXT_TOKENS}",
                    level="WARNING",
                )
                configured_context = self.MAX_CONTEXT_TOKENS
        else:
            configured_context = context_tokens
        self.context_tokens = self._validate_context_tokens(configured_context)

    @classmethod
    def _environment_token_settings(cls) -> dict[str, int]:
        settings = {}
        for key, spec in cls.TOKEN_SETTING_SPECS.items():
            try:
                value = int(os.getenv(spec["env"], str(spec["default"])))
            except (TypeError, ValueError):
                value = spec["default"]
            settings[key] = min(spec["max"], max(spec["min"], value))
        return settings

    @classmethod
    def validate_token_settings(cls, values: Mapping[str, int]) -> dict[str, int]:
        if not isinstance(values, Mapping):
            raise ValueError("token_settings 必須是物件")
        unknown = set(values) - set(cls.TOKEN_SETTING_SPECS)
        if unknown:
            raise ValueError(f"未知的 token 設定：{', '.join(sorted(unknown))}")
        validated = {}
        for key, value in values.items():
            spec = cls.TOKEN_SETTING_SPECS[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} 必須是整數")
            if not spec["min"] <= value <= spec["max"]:
                raise ValueError(f"{key} 必須介於 {spec['min']} 到 {spec['max']}")
            validated[key] = value
        return validated

    def get_token_settings(self) -> dict[str, int]:
        with self._token_settings_lock:
            return dict(self._token_settings)

    def update_token_settings(self, values: Mapping[str, int]) -> dict[str, int]:
        validated = self.validate_token_settings(values)
        with self._token_settings_lock:
            self._token_settings.update(validated)
            return dict(self._token_settings)

    @classmethod
    def token_settings_schema(cls) -> dict[str, dict[str, int | str]]:
        return {
            key: {name: spec[name] for name in ("min", "max", "default", "label")}
            for key, spec in cls.TOKEN_SETTING_SPECS.items()
        }

    @staticmethod
    def _model_path() -> str | None:
        configured = os.getenv("LLAMA_CPP_MODEL") or os.getenv("MODEL_PATH")
        if configured:
            return configured
        candidates = []
        for folder in (Path.cwd() / "models", Path.home()):
            if folder.exists():
                candidates.extend(sorted(folder.glob("*.gguf")))
        return str(candidates[0]) if candidates else None

    @staticmethod
    def _int_setting(name: str, default: int, minimum: int = 1) -> int:
        try:
            return max(minimum, int(os.getenv(name, str(default))))
        except ValueError:
            return default

    @classmethod
    def _validate_context_tokens(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("context tokens 必須是整數")
        if not cls.MIN_CONTEXT_TOKENS <= value <= cls.MAX_CONTEXT_TOKENS:
            raise ValueError(
                f"context tokens 必須介於 {cls.MIN_CONTEXT_TOKENS} 到 {cls.MAX_CONTEXT_TOKENS}"
            )
        return value

    def _load(self):
        if self.model is not None:
            return self.model
        model_path = self.model_path or self._model_path()
        if not model_path:
            raise RuntimeError("找不到 GGUF 模型，請設定 LLAMA_CPP_MODEL=/path/to/model.gguf")
        if not os.path.isfile(model_path):
            raise RuntimeError(f"找不到 llama.cpp 模型：{model_path}")
        self.model_path = model_path
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError("目前 Python 環境沒有 llama_cpp，請使用 py3_10") from exc

        self.logger.emit("LLM", f"loading llama.cpp model={model_path} n_ctx={self.context_tokens}")
        self.model = Llama(
            model_path=model_path,
            n_ctx=self.context_tokens,
            n_threads=self._int_setting("LLAMA_CPP_THREADS", max(1, (os.cpu_count() or 4) - 1)),
            n_gpu_layers=self._int_setting("LLAMA_CPP_GPU_LAYERS", 0, -1),
            verbose=False,
        )
        self.logger.emit("LLM", "llama.cpp model loaded")
        return self.model

    def set_context_tokens(self, context_tokens: int) -> dict:
        requested = self._validate_context_tokens(context_tokens)
        with self._model_lock:
            previous = self.context_tokens
            if requested == previous:
                return {"context_tokens": previous, "model_reloaded": False, "unchanged": True}

            was_loaded = self.model is not None
            if was_loaded:
                old_model = self.model
                self.model = None
                close = getattr(old_model, "close", None)
                try:
                    if callable(close):
                        close()
                except Exception as exc:
                    self.logger.emit("LLM", f"model close warning during context change: {exc}", level="WARNING")
                finally:
                    del old_model
                    gc.collect()

            self.context_tokens = requested
            self.logger.emit("LLM", f"llama.cpp context change old={previous} new={requested}")
            if was_loaded:
                try:
                    self._load()
                except Exception as exc:
                    self.model = None
                    self.context_tokens = previous
                    gc.collect()
                    try:
                        self._load()
                    except Exception as restore_exc:
                        self.logger.emit(
                            "LLM",
                            f"failed to restore model context={previous}: {restore_exc}",
                            level="ERROR",
                        )
                        raise RuntimeError(
                            f"context {requested} 載入失敗，且原 context {previous} 也無法恢復：{restore_exc}"
                        ) from restore_exc
                    raise RuntimeError(
                        f"context {requested} 載入失敗，已恢復原 context {previous}：{exc}"
                    ) from exc

            return {"context_tokens": requested, "model_reloaded": was_loaded, "unchanged": False}

    @staticmethod
    def _parse_object(content: str) -> dict:
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I)
        start, end = content.find("{"), content.rfind("}")
        if start >= 0 and end > start:
            content = content[start : end + 1]
        result = json.loads(content)
        if not isinstance(result, dict):
            raise ValueError("模型回應不是 JSON object")
        return result

    @classmethod
    def _normalize_result(cls, result: dict) -> dict:
        return {
            "title": str(result.get("title", "")).strip(),
            "meeting_time": result.get("meeting_time"),
            "thread_name": str(result.get("thread_name", "")).strip(),
            "summary": str(result.get("summary", "")),
            "actions": result.get("actions", []) if isinstance(result.get("actions", []), list) else [],
            "decisions": result.get("decisions", []) if isinstance(result.get("decisions", []), list) else [],
            "follow_ups": result.get("follow_ups", []) if isinstance(result.get("follow_ups", []), list) else [],
            "related_meeting_ids": result.get("related_meeting_ids", []) if isinstance(result.get("related_meeting_ids", []), list) else [],
        }

    @classmethod
    def _json(cls, content: str) -> dict:
        return cls._normalize_result(cls._parse_object(content))

    @staticmethod
    def transcript_text(rows: list[dict]) -> str:
        return "\n".join(
            f"{row.get('speaker_name', row['speaker_id'])} [{row['start_time']:.1f}-{row['end_time']:.1f}s]: "
            f"{convert_traditional(row['normalized_text'])}"
            for row in rows
        )

    @staticmethod
    def _analysis_requirements() -> str:
        return """你是繁體中文會議秘書。分析逐字稿及相關舊會議，只輸出合法 JSON，不要 Markdown。
生成符合內容的簡短會議主題 title。若逐字稿明確提到本場會議日期或時間，meeting_time 用 ISO 格式日期/時間；若沒提到則填 null，不要自行猜測，也不要把待辦期限當成會議時間。
summary 是條列式摘要。actions 格式為 [{"item":"待辦","owner":"負責人或空字串","due_date":"期限或空字串"}]。
decisions 格式為 [{"decision":"決議內容","context":"背景或空字串"}]。
follow_ups 格式為 [{"previous_meeting_id":"舊會議 ID","previous_action_id":0,"item":"舊待辦","status":"completed|in_progress|not_mentioned","conclusion":"依本次逐字稿做的簡短判斷"}]。只檢查提供的舊待辦；沒有提及就標 not_mentioned，不可把沒有提及當作已完成。
related_meeting_ids 只列出確實與本次主題相關的候選舊會議 ID。thread_name 在至少兩場相關會議時，將各會議主題合併成簡潔主軸名稱；否則留空。
若沒有相關舊會議，follow_ups、related_meeting_ids 為空陣列，thread_name 為空字串。不要捏造逐字稿或舊會議資料中沒有的資訊。
JSON 格式：{"title":"主題","meeting_time":null,"thread_name":"","related_meeting_ids":[],"summary":"- 摘要","actions":[],"decisions":[],"follow_ups":[]}"""

    @classmethod
    def analysis_prompt(cls, transcript: str, related_context: list[dict] | None = None) -> str:
        related = json.dumps(related_context or [], ensure_ascii=False)
        return (
            cls._analysis_requirements()
            + "\n\n相關會議檢索資料（只可引用其中的候選 ID 和待辦）：\n"
            + related
            + "\n\n本次逐字稿：\n"
            + transcript
        )

    @classmethod
    def _map_prompt(cls, segment_number: int, segment_count: int, transcript_segment: str) -> str:
        return f"""你是會議逐字稿的分段紀錄員。這是第 {segment_number}/{segment_count} 段。
請完整列出本段中可核對的資訊，不要撰寫整場會議的總結，也不要猜測其他段落內容。把不同事項拆成獨立項目；人名、數字、日期、條件、否決意見與待確認事項要保留。evidence 使用逐字稿中的短句，並保留 speaker 與時間。
只有在本段明確說出「本場會議」日期／時間時才填 meeting_time_mentions；一般事件日期或待辦期限不要當作會議時間。
只輸出 JSON，格式：{{"topic_candidates":[],"facts":[{{"claim":"具體資訊","evidence":"原文短句","speaker":"說話人","time":"逐字稿時間"}}],"decisions":[{{"decision":"決定","context":"條件或背景","evidence":"原文短句","speaker":"說話人","time":"逐字稿時間"}}],"actions":[{{"item":"待辦","owner":"明確負責人或空字串","due_date":"明確期限或空字串","evidence":"原文短句","speaker":"說話人","time":"逐字稿時間"}}],"meeting_time_mentions":[{{"value":"日期時間原文","evidence":"原文短句"}}]}}
若某類沒有內容，回傳空陣列。不要把不同人的意見合併成同一個事實。

本段逐字稿：
{transcript_segment}"""

    @staticmethod
    def _reduce_prompt(notes_json: str) -> str:
        return """你是會議紀錄的分層合併器。輸入是多個逐字稿分段紀錄。
請合併重複項目，保留每一項不重複的事實、數字、人名、決議、待辦、日期與 source_segments；有矛盾的說法要並列，不能自行裁決。evidence 保留最短且可核對的原文摘錄，不要重複整段說明。不可只輸出概述或刪除少數意見。不得新增輸入中不存在的資訊。
只輸出 JSON，欄位為 source_segments、topic_candidates、facts、decisions、actions、meeting_time_mentions；陣列內容沿用輸入項目的欄位。

分段紀錄：
""" + notes_json

    @classmethod
    def _final_ledger_prompt(cls, notes: list[dict], related_context: list[dict] | None) -> str:
        related = json.dumps(related_context or [], ensure_ascii=False)
        ledger = json.dumps(notes, ensure_ascii=False)
        return (
            cls._analysis_requirements()
            + "\n\n以下是本場逐字稿逐段抽取並分層合併的完整事實紀錄。每個 source_segments 都代表已處理的原文段落；請只根據這些紀錄整理最終輸出，保留重要且不重複的資訊。若不同段落有矛盾，需在摘要中如實指出，不要自行補完。\n"
            + ledger
            + "\n\n相關舊會議檢索資料（只可引用其中的候選 ID 和待辦）：\n"
            + related
        )

    def _token_count(self, text: str) -> int:
        tokenizer = getattr(self._load(), "tokenize", None)
        if tokenizer is None:
            # UTF-8 byte count is a conservative fallback if a compatible
            # tokenizer is not exposed by a llama.cpp wrapper.
            return len(text.encode("utf-8"))
        try:
            return len(tokenizer(text.encode("utf-8"), add_bos=False))
        except TypeError:
            return len(tokenizer(text.encode("utf-8")))

    def _prompt_fits(self, prompt: str, output_tokens: int) -> bool:
        return self._token_count(prompt) + output_tokens + self.CONTEXT_MARGIN_TOKENS <= self.context_tokens

    def _request_json(self, prompt: str, meeting_id: str, stage: str, max_tokens: int) -> dict:
        prompt_tokens = self._token_count(prompt)
        if prompt_tokens + max_tokens + self.CONTEXT_MARGIN_TOKENS > self.context_tokens:
            raise RuntimeError(
                f"{stage} 超出模型上下文預算（輸入約 {prompt_tokens} tokens，"
                f"上下文 {self.context_tokens}）；請降低該階段的輸入目標或輸出上限。"
            )
        self.logger.emit(
            "LLM",
            f"{stage} input_tokens={prompt_tokens} output_limit_tokens={max_tokens} context_tokens={self.context_tokens}",
            meeting_id=meeting_id,
        )
        response = self._load().create_chat_completion(
            messages=[{"role": "system", "content": "你只輸出合法 JSON。"}, {"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        choice = response["choices"][0]
        if choice.get("finish_reason") == "length":
            raise OutputTokenLimitError(
                f"{stage} 輸出達到 token 上限，為避免保存不完整內容已中止；"
                "請調低該階段輸入目標或提高輸出上限。"
            )
        try:
            return self._parse_object(choice["message"]["content"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{stage} 沒有產生完整有效的 JSON；為避免遺漏，本次摘要未保存。") from exc

    def _split_long_utterance(self, row: dict, max_output_tokens: int, target_tokens: int) -> list[str]:
        speaker = row.get("speaker_name", row["speaker_id"])
        start, end = row["start_time"], row["end_time"]
        text = convert_traditional(row.get("normalized_text", row.get("text", "")))
        parts = []
        offset = 0
        part_number = 1
        while offset < len(text):
            low, high, best = 1, len(text) - offset, 0
            while low <= high:
                middle = (low + high) // 2
                line = f"{speaker} [{start:.1f}-{end:.1f}s] (續段 {part_number}): {text[offset:offset + middle]}"
                prompt = self._map_prompt(999999, 999999, line)
                if self._token_count(line) <= target_tokens and self._prompt_fits(prompt, max_output_tokens):
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if best == 0:
                raise RuntimeError(
                    "單一逐字稿的最小片段仍超過模型上下文；請在管理頁降低「每段整理輸出上限」，"
                    "或提高 Context 上限後重試。"
                )
            parts.append(f"{speaker} [{start:.1f}-{end:.1f}s] (續段 {part_number}): {text[offset:offset + best]}")
            offset += best
            part_number += 1
        return parts

    def _pack_transcript(
        self,
        rows: list[dict],
        max_output_tokens: int,
        target_tokens: int | None = None,
    ) -> list[str]:
        if target_tokens is None:
            target_tokens = self._int_setting("LLAMA_CPP_CHUNK_TOKENS", 1800, 256)
        lines = []
        for row in rows:
            speaker = row.get("speaker_name", row["speaker_id"])
            line = (
                f"{speaker} [{row['start_time']:.1f}-{row['end_time']:.1f}s]: "
                f"{convert_traditional(row.get('normalized_text', row.get('text', '')))}"
            )
            if self._token_count(line) <= target_tokens and self._prompt_fits(self._map_prompt(999999, 999999, line), max_output_tokens):
                lines.append(line)
            else:
                lines.extend(self._split_long_utterance(row, max_output_tokens, target_tokens))

        chunks, current = [], []
        for line in lines:
            candidate = "\n".join(current + [line])
            prompt = self._map_prompt(999999, 999999, candidate)
            within_chunk_target = self._token_count(candidate) <= target_tokens
            if current and (not within_chunk_target or not self._prompt_fits(prompt, max_output_tokens)):
                chunks.append("\n".join(current))
                current = [line]
            else:
                current.append(line)
            single_prompt = self._map_prompt(999999, 999999, "\n".join(current))
            if not self._prompt_fits(single_prompt, max_output_tokens):
                raise RuntimeError(
                    "逐字稿分段仍超出模型上下文；請在管理頁降低「每段逐字稿輸入目標」或「每段整理輸出上限」，"
                    "也可視情況提高 Context 上限後重試。"
                )
        if current:
            chunks.append("\n".join(current))
        return chunks

    @staticmethod
    def _clean_note(value: dict, source_segments: list[int], *, annotate_items: bool = False) -> dict:
        fields = ("topic_candidates", "facts", "decisions", "actions", "meeting_time_mentions")
        note = {field: value.get(field, []) if isinstance(value.get(field, []), list) else [] for field in fields}
        if annotate_items:
            for field in ("facts", "decisions", "actions", "meeting_time_mentions"):
                for item in note[field]:
                    if isinstance(item, dict):
                        item.setdefault("source_segments", source_segments)
        note["source_segments"] = source_segments
        return note

    def _checkpoint_run_key(self, transcript: str) -> str:
        model_path = self.model_path or self._model_path() or ""
        model_signature: dict[str, str | int] = {"path": model_path}
        try:
            model_stat = Path(model_path).stat()
            model_signature.update(size=model_stat.st_size, mtime_ns=model_stat.st_mtime_ns)
        except OSError:
            pass
        source = {
            "version": self.CHECKPOINT_VERSION,
            "transcript_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
            "model": model_signature,
            "map_prompt": self._map_prompt(1, 2, "<segment>"),
            "reduce_prompt": self._reduce_prompt("<notes>"),
            "final_requirements": self._analysis_requirements(),
        }
        encoded = json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _checkpoint_input_hash(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _load_checkpoint(
        self,
        meeting_id: str,
        run_key: str,
        phase: str,
        round_number: int,
        item_key: str,
        prompt: str,
    ) -> dict | None:
        if self.checkpoint_store is None:
            return None
        return self.checkpoint_store.get_analysis_checkpoint(
            meeting_id,
            run_key,
            phase,
            round_number,
            item_key,
            self._checkpoint_input_hash(prompt),
        )

    def _save_checkpoint(
        self,
        meeting_id: str,
        run_key: str,
        phase: str,
        round_number: int,
        item_key: str,
        prompt: str,
        result: dict,
    ) -> None:
        if self.checkpoint_store is None:
            return
        self.checkpoint_store.save_analysis_checkpoint(
            meeting_id,
            run_key,
            phase,
            round_number,
            item_key,
            self._checkpoint_input_hash(prompt),
            result,
        )

    @staticmethod
    def _split_note_items(note: dict) -> tuple[dict, dict] | None:
        fields = ("topic_candidates", "facts", "decisions", "actions", "meeting_time_mentions")
        entries = []
        for field in fields:
            values = note.get(field, [])
            if isinstance(values, list):
                entries.extend((field, item) for item in values)
        if len(entries) < 2:
            return None

        split_at = (len(entries) + 1) // 2
        pieces = []
        for batch in (entries[:split_at], entries[split_at:]):
            piece = {field: [] for field in fields}
            for field, item in batch:
                piece[field].append(item)
            piece["source_segments"] = list(note.get("source_segments", []))
            pieces.append(piece)
        return pieces[0], pieces[1]

    def _pack_note_groups(
        self,
        notes: list[dict],
        max_output_tokens: int,
        max_input_tokens: int,
    ) -> list[list[dict]]:
        groups, current = [], []
        for note in notes:
            candidate = current + [note]
            prompt = self._reduce_prompt(json.dumps(candidate, ensure_ascii=False))
            over_input_target = self._token_count(prompt) > max_input_tokens
            if current and (over_input_target or not self._prompt_fits(prompt, max_output_tokens)):
                groups.append(current)
                current = [note]
                prompt = self._reduce_prompt(json.dumps(current, ensure_ascii=False))
            else:
                current = candidate
            if not self._prompt_fits(prompt, max_output_tokens):
                raise RuntimeError(
                    "單批整理紀錄超出合併階段可用上下文；請在管理頁降低「每段整理輸出上限」或「每批合併輸出上限」，"
                    "也可視情況提高 Context 上限後重試。"
                )
        if current:
            groups.append(current)
        return groups

    def _reduce_to_final_budget(
        self,
        notes: list[dict],
        meeting_id: str,
        related_context: list[dict] | None,
        final_tokens: int,
        reduce_tokens: int,
        reduce_input_tokens: int,
        run_key: str,
        progress: Callable[[str], None] | None = None,
    ) -> tuple[list[dict], str]:
        input_limit = min(reduce_input_tokens, max(512, self.context_tokens // 3))

        def reduce_group(
            group: list[dict],
            round_number: int,
            group_number: int,
            path: str = "",
        ) -> list[dict]:
            prompt = self._reduce_prompt(json.dumps(group, ensure_ascii=False))
            item_key = f"{group_number:04d}{path or 'root'}"
            stage = f"分層合併 round={round_number} group={group_number}/{len(groups)}{path}"
            cached = self._load_checkpoint(
                meeting_id, run_key, "reduce", round_number, item_key, prompt
            )
            if cached is not None:
                if progress:
                    progress(f"恢復合併檢查點（第 {round_number} 輪，第 {group_number} 批{path}）")
                return [cached]

            group_segments = sorted({
                segment
                for note in group
                for segment in note.get("source_segments", [])
            })
            try:
                result = self._request_json(prompt, meeting_id, stage, reduce_tokens)
            except OutputTokenLimitError as exc:
                if len(group) > 1:
                    self.logger.emit(
                        "LLM",
                        f"{stage} 輸出達上限，拆分批次後重試",
                        meeting_id=meeting_id,
                        level="WARNING",
                    )
                    if progress:
                        progress(f"合併輸出達上限，拆分第 {round_number} 輪第 {group_number} 批重試")
                    midpoint = len(group) // 2
                    return reduce_group(group[:midpoint], round_number, group_number, path + "L") + reduce_group(
                        group[midpoint:], round_number, group_number, path + "R"
                    )

                split_note = self._split_note_items(group[0]) if group else None
                if split_note is not None:
                    self.logger.emit(
                        "LLM",
                        f"{stage} 單筆輸出達上限，拆分紀錄項目後重試",
                        meeting_id=meeting_id,
                        level="WARNING",
                    )
                    if progress:
                        progress(f"合併輸出達上限，拆分單筆紀錄重試（第 {round_number} 輪第 {group_number} 批）")
                    return reduce_group([split_note[0]], round_number, group_number, path + "L") + reduce_group(
                        [split_note[1]], round_number, group_number, path + "R"
                    )
                raise RuntimeError(
                    f"{stage} 輸出仍超過 token 上限；已保留其他完成批次，"
                    "請在管理頁提高合併輸出上限或降低每段整理輸出上限後重試。"
                ) from exc

            reduced_note = self._clean_note(result, group_segments)
            self._save_checkpoint(
                meeting_id, run_key, "reduce", round_number, item_key, prompt, reduced_note
            )
            return [reduced_note]

        for round_number in range(1, 9):
            final_prompt = self._final_ledger_prompt(notes, related_context)
            if self._prompt_fits(final_prompt, final_tokens):
                return notes, final_prompt

            groups = self._pack_note_groups(notes, reduce_tokens, input_limit)
            before_size = self._token_count(json.dumps(notes, ensure_ascii=False))
            reduced = []
            for group_number, group in enumerate(groups, start=1):
                if progress:
                    progress(f"合併分段（第 {round_number} 輪，{group_number}/{len(groups)} 批）")
                reduced.extend(reduce_group(group, round_number, group_number))

            after_size = self._token_count(json.dumps(reduced, ensure_ascii=False))
            reduced_prompt = self._final_ledger_prompt(reduced, related_context)
            if after_size >= before_size and not self._prompt_fits(reduced_prompt, final_tokens):
                raise RuntimeError(
                    "分層合併未能縮小紀錄到最終摘要可用的 context；已保存完成批次，"
                    "可調整管理頁的合併輸入／輸出 token 後重試。"
                )
            notes = reduced

        raise RuntimeError("逐字稿分層合併超過 8 輪仍無法符合模型上下文；本次摘要未保存。")

    def analyze(
        self,
        meeting_id: str,
        rows: list[dict],
        related_context: list[dict] | None = None,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> dict:
        with self._model_lock:
            return self._analyze(meeting_id, rows, related_context, progress=progress)

    def _analyze(
        self,
        meeting_id: str,
        rows: list[dict],
        related_context: list[dict] | None = None,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> dict:
        transcript = self.transcript_text(rows)
        if not transcript.strip():
            return {
                "title": "",
                "meeting_time": None,
                "thread_name": "",
                "summary": "- 尚無可分析內容",
                "actions": [],
                "decisions": [],
                "follow_ups": [],
                "related_meeting_ids": [],
            }

        token_settings = self.get_token_settings()
        self._load()
        final_tokens = token_settings["final_max_tokens"]
        direct_prompt = self.analysis_prompt(transcript, related_context)
        direct_limit = token_settings["direct_max_tokens"]
        if self._token_count(direct_prompt) <= direct_limit and self._prompt_fits(direct_prompt, final_tokens):
            if progress:
                progress("分析逐字稿")
            result = self._request_json(direct_prompt, meeting_id, "直接摘要", final_tokens)
            self.logger.emit("LLM", "llama.cpp analysis completed mode=direct", meeting_id=meeting_id)
            return self._normalize_result(result)

        map_tokens = token_settings["map_max_tokens"]
        transcript_segments = self._pack_transcript(rows, map_tokens, token_settings["chunk_tokens"])
        if not transcript_segments:
            raise RuntimeError("逐字稿無法建立摘要分段。")
        run_key = self._checkpoint_run_key(transcript)
        if self.checkpoint_store is not None:
            self.checkpoint_store.retain_analysis_checkpoint_run(meeting_id, run_key)
        self.logger.emit(
            "LLM",
            f"hierarchical summary started segments={len(transcript_segments)}",
            meeting_id=meeting_id,
        )

        notes = []
        resumed_segments = 0
        for segment_number, segment in enumerate(transcript_segments, start=1):
            prompt = self._map_prompt(segment_number, len(transcript_segments), segment)
            cached = self._load_checkpoint(
                meeting_id, run_key, "map", 0, f"{segment_number:06d}", prompt
            )
            if cached is not None:
                notes.append(self._clean_note(cached, [segment_number]))
                resumed_segments += 1
                if progress:
                    progress(f"恢復已完成分段 {segment_number}/{len(transcript_segments)}")
                continue

            if progress:
                progress(f"逐段整理 {segment_number}/{len(transcript_segments)}")
            result = self._request_json(
                prompt,
                meeting_id,
                f"分段整理 {segment_number}/{len(transcript_segments)}",
                map_tokens,
            )
            note = self._clean_note(result, [segment_number], annotate_items=True)
            self._save_checkpoint(
                meeting_id, run_key, "map", 0, f"{segment_number:06d}", prompt, note
            )
            notes.append(note)

        if resumed_segments:
            self.logger.emit(
                "LLM",
                f"resumed analysis checkpoints map={resumed_segments}/{len(transcript_segments)}",
                meeting_id=meeting_id,
            )

        notes, final_prompt = self._reduce_to_final_budget(
            notes,
            meeting_id,
            related_context,
            final_tokens,
            token_settings["reduce_max_tokens"],
            token_settings["reduce_input_tokens"],
            run_key,
            progress,
        )
        result = self._load_checkpoint(meeting_id, run_key, "final", 0, "final", final_prompt)
        if result is not None:
            if progress:
                progress("恢復最終摘要檢查點")
        else:
            if progress:
                progress("產生最終摘要")
            result = self._request_json(final_prompt, meeting_id, "最終摘要", final_tokens)
            self._save_checkpoint(
                meeting_id, run_key, "final", 0, "final", final_prompt, result
            )
        self.logger.emit(
            "LLM",
            f"llama.cpp analysis completed mode=hierarchical source_segments={len(transcript_segments)}",
            meeting_id=meeting_id,
        )
        return self._normalize_result(result)
