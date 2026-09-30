from __future__ import annotations
import json, os, re
from pathlib import Path
from .logging import EventLogger
from .normalize import convert_traditional

class LLMOrchestrator:
    def __init__(self, logger: EventLogger): self.logger, self.model = logger, None
    @staticmethod
    def _model_path() -> str | None:
        configured = os.getenv("LLAMA_CPP_MODEL") or os.getenv("MODEL_PATH")
        if configured: return configured
        candidates = []
        for folder in (Path.cwd() / "models", Path.home()):
            if folder.exists(): candidates.extend(sorted(folder.glob("*.gguf")))
        return str(candidates[0]) if candidates else None
    def _load(self):
        if self.model is not None: return self.model
        model_path = self._model_path()
        if not model_path: raise RuntimeError("找不到 GGUF 模型，請設定 LLAMA_CPP_MODEL=/path/to/model.gguf")
        if not os.path.isfile(model_path): raise RuntimeError(f"找不到 llama.cpp 模型：{model_path}")
        try: from llama_cpp import Llama
        except ImportError as exc: raise RuntimeError("目前 Python 環境沒有 llama_cpp，請使用 py3_10") from exc
        self.logger.emit("LLM", f"loading llama.cpp model={model_path}")
        self.model = Llama(model_path=model_path, n_ctx=int(os.getenv("LLAMA_CPP_CONTEXT", "8192")), n_threads=int(os.getenv("LLAMA_CPP_THREADS", str(max(1, (os.cpu_count() or 4)-1)))), n_gpu_layers=int(os.getenv("LLAMA_CPP_GPU_LAYERS", "0")), verbose=False)
        self.logger.emit("LLM", "llama.cpp model loaded"); return self.model
    @staticmethod
    def _json(content: str) -> dict:
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I); start, end = content.find("{"), content.rfind("}")
        if start >= 0 and end > start: content = content[start:end+1]
        result = json.loads(content)
        return {"title": str(result.get("title", "")).strip(), "meeting_time": result.get("meeting_time"), "thread_name": str(result.get("thread_name", "")).strip(), "summary": str(result.get("summary", "")), "actions": result.get("actions", []) if isinstance(result.get("actions", []), list) else [], "decisions": result.get("decisions", []) if isinstance(result.get("decisions", []), list) else [], "follow_ups": result.get("follow_ups", []) if isinstance(result.get("follow_ups", []), list) else [], "related_meeting_ids": result.get("related_meeting_ids", []) if isinstance(result.get("related_meeting_ids", []), list) else []}
    @staticmethod
    def transcript_text(rows: list[dict]) -> str:
        return "\n".join(f"{r.get('speaker_name', r['speaker_id'])} [{r['start_time']:.1f}-{r['end_time']:.1f}s]: {convert_traditional(r['normalized_text'])}" for r in rows)
    @staticmethod
    def analysis_prompt(transcript: str, related_context: list[dict] | None = None) -> str:
        related = json.dumps(related_context or [], ensure_ascii=False)
        return """你是繁體中文會議秘書。分析逐字稿及相關舊會議，只輸出合法 JSON，不要 Markdown。
生成符合內容的簡短會議主題 title。若逐字稿明確提到會議日期或時間，meeting_time 用 ISO 格式日期/時間；若沒提到則填 null，不要自行猜測。
summary 是條列式摘要。actions 格式為 [{"item":"待辦","owner":"負責人或空字串","due_date":"期限或空字串"}]。
decisions 格式為 [{"decision":"決議內容","context":"背景或空字串"}]。
follow_ups 格式為 [{"previous_meeting_id":"舊會議 ID","previous_action_id":0,"item":"舊待辦","status":"completed|in_progress|not_mentioned","conclusion":"依本次逐字稿做的簡短判斷"}]。只檢查提供的舊待辦；沒有提及就標 not_mentioned，不可把沒有提及當作已完成。
related_meeting_ids 只列出確實與本次主題相關的候選舊會議 ID。thread_name 在至少兩場相關會議時，將各會議主題合併成簡潔主軸名稱；否則留空。
若沒有相關舊會議，follow_ups、related_meeting_ids 為空陣列，thread_name 為空字串。不要捏造逐字稿或舊會議資料中沒有的資訊。
JSON 格式：{"title":"主題","meeting_time":null,"thread_name":"","related_meeting_ids":[],"summary":"- 摘要","actions":[],"decisions":[],"follow_ups":[]}

相關會議檢索資料（只可引用其中的候選 ID 和待辦）：
""" + related + "\n\n本次逐字稿：\n" + transcript
    def analyze(self, meeting_id: str, rows: list[dict], related_context: list[dict] | None = None) -> dict:
        transcript = self.transcript_text(rows)
        if not transcript.strip(): return {"title": "", "meeting_time": None, "thread_name": "", "summary": "- 尚無可分析內容", "actions": [], "decisions": [], "follow_ups": [], "related_meeting_ids": []}
        prompt = self.analysis_prompt(transcript, related_context)
        self.logger.emit("LLM", f"llama.cpp analyzing chars={len(transcript)}", meeting_id=meeting_id)
        response = self._load().create_chat_completion(messages=[{"role":"system","content":"你只輸出 JSON。"},{"role":"user","content":prompt}], temperature=0.2, max_tokens=int(os.getenv("LLAMA_CPP_MAX_TOKENS", "2048")), response_format={"type":"json_object"})
        result = self._json(response["choices"][0]["message"]["content"]); self.logger.emit("LLM", "llama.cpp analysis completed", meeting_id=meeting_id); return result
