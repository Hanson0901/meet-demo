from __future__ import annotations
import uuid
from datetime import datetime
from typing import Callable
from .audio import AudioIngest
from .asr import FunASRAdapter
from .db import MeetingDB
from .logging import EventLogger
from .normalize import convert_traditional
from .orchestrator import LLMOrchestrator
from .evaluation import MeetingEvaluator
from .rag import transcript_chunks
from .speaker_identity import suggest_names_from_transcript, voiceprint_match_threshold

class MeetingTools:
    def __init__(self, db: MeetingDB, logger: EventLogger):
        self.db, self.logger = db, logger
        original_sink = logger.sink
        def persist(event):
            original_sink(event)
            db.add_log(event)
        logger.sink = persist
        stored_context = db.get_setting("llama_cpp_context")
        context_tokens = None
        if stored_context is not None:
            try:
                candidate = int(stored_context)
                context_tokens = LLMOrchestrator._validate_context_tokens(candidate)
            except (TypeError, ValueError):
                logger.emit("LLM", "ignoring invalid persisted context setting", level="WARNING")
        token_settings = {}
        for key in LLMOrchestrator.TOKEN_SETTING_SPECS:
            stored_value = db.get_setting(f"llama_cpp_{key}")
            if stored_value is None:
                continue
            try:
                token_settings[key] = int(stored_value)
                LLMOrchestrator.validate_token_settings({key: token_settings[key]})
            except (TypeError, ValueError):
                token_settings.pop(key, None)
                logger.emit("LLM", f"ignoring invalid persisted token setting: {key}", level="WARNING")
        self.audio, self.asr = AudioIngest(logger), FunASRAdapter(logger)
        self.llm = LLMOrchestrator(
            logger,
            context_tokens=context_tokens,
            token_settings=token_settings,
            checkpoint_store=db,
        )
        self.evaluator = MeetingEvaluator(logger)
    @staticmethod
    def _started_at(started_at: str | None) -> datetime:
        try:
            return datetime.fromisoformat((started_at or "").replace("Z", "+00:00")) if started_at else datetime.now().astimezone()
        except ValueError:
            return datetime.now().astimezone()

    def prepare_meeting(self, title: str | None = None, started_at: str | None = None) -> str:
        """Create the temporary record before a potentially long ASR job starts."""
        meeting_id = str(uuid.uuid4())
        started = self._started_at(started_at)
        created_at = started.isoformat(timespec="seconds")
        temporary_title = f"整理中 · {started.astimezone().strftime('%Y-%m-%d %H:%M')}"
        self.db.create_meeting(meeting_id, temporary_title, created_at)
        self.db.set_transcription_status(meeting_id, "queued", "等待 FunASR")
        return meeting_id

    def transcribe_meeting(
        self,
        audio: str,
        title: str | None = None,
        started_at: str | None = None,
        *,
        meeting_id: str | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> str:
        if meeting_id is None:
            meeting_id = self.prepare_meeting(title, started_at)
        elif self.db.meeting(meeting_id) is None:
            raise ValueError("找不到要保存語音辨識結果的會議")

        segments = list(self.audio.segments(audio))
        if not segments:
            raise RuntimeError("音訊沒有可辨識的片段")
        self.logger.emit("ASR", f"transcription started segments={len(segments)}", meeting_id=meeting_id)
        if progress:
            progress(f"載入 FunASR，準備辨識 {len(segments)} 段音訊")

        recognized_count = 0
        placeholder_count = 0
        for index, segment in enumerate(segments, 1):
            if progress:
                progress(f"語音辨識中 {index}/{len(segments)}")
            for item in self.asr.transcribe(meeting_id, segment):
                if item.text.startswith("["):
                    placeholder_count += 1
                else:
                    recognized_count += 1
                self.db.save_utterance(item, convert_traditional(item.text))
                self.logger.emit("DB", f"utterance saved id={item.utterance_id}", meeting_id=meeting_id)

        if placeholder_count and not recognized_count:
            raise RuntimeError("FunASR 沒有完成辨識，請確認 py3_10 環境、FunASR 模型與音訊格式")

        if progress:
            progress("正在建立說話人聲紋資料")
        for speaker_id, embedding in self.asr.speaker_embeddings(audio).items():
            match = self.db.match_speaker_embedding(
                meeting_id,
                speaker_id,
                embedding,
                embedding_model=self.asr.speaker_model_name,
                threshold=voiceprint_match_threshold(),
            )
            if match["matched"]:
                self.logger.emit(
                    "ASR",
                    f"CAM++ 聲紋匹配 speaker={speaker_id} score={match['score']:.3f} name={match['display_name']}",
                    meeting_id=meeting_id,
                )
            elif match.get("score") is not None:
                self.logger.emit(
                    "ASR",
                    f"CAM++ 聲紋候選 speaker={speaker_id} score={match['score']:.3f}，等待確認",
                    meeting_id=meeting_id,
                )
        if progress:
            progress("正在整理姓名候選")
        text_suggestions = suggest_names_from_transcript(self.db.transcript(meeting_id))
        self.db.save_speaker_suggestions(meeting_id, text_suggestions)
        self.logger.emit("ASR", f"transcription completed utterances={recognized_count}", meeting_id=meeting_id)
        return meeting_id
    def summarize_meeting(
        self,
        meeting_id: str,
        *,
        progress: Callable[[str], None] | None = None,
        model_event: Callable[[dict], None] | None = None,
    ):
        rows = self.db.transcript(meeting_id)
        query = self.llm.transcript_text(rows)
        related = self.db.related_context(query, meeting_id)
        result = self.llm.analyze(
            meeting_id,
            rows,
            related,
            progress=progress,
            model_event=model_event,
        )
        meeting = self.db.meeting(meeting_id) or {}
        generated_title = result.get("title", "").strip()
        has_summary = bool(result.get("summary", "").strip()) and result.get("summary", "").strip() != "- 尚無可分析內容"
        finalized = bool(generated_title and has_summary)
        title = generated_title if finalized else meeting.get("title", "")
        meeting_time = result.get("meeting_time")
        if not isinstance(meeting_time, str) or not meeting_time.strip() or meeting_time.strip().lower() in ("null", "none"):
            meeting_time = meeting.get("created_at", "")
        else:
            try:
                datetime.fromisoformat(meeting_time.strip().replace("Z", "+00:00"))
            except ValueError:
                meeting_time = meeting.get("created_at", "")
        candidate_ids = {item["meeting_id"] for item in related}
        selected_ids = [item for item in result.get("related_meeting_ids", []) if item in candidate_ids]
        # A follow-up necessarily establishes a relationship even if the model
        # omitted that meeting from its separate related_meeting_ids field.
        for item in result.get("follow_ups", []):
            previous_id = item.get("previous_meeting_id")
            if previous_id in candidate_ids and previous_id not in selected_ids:
                selected_ids.append(previous_id)
        result["follow_ups"] = [item for item in result.get("follow_ups", []) if item.get("previous_meeting_id") in selected_ids]
        thread = self.db.link_topic(meeting_id, title, selected_ids, result.get("thread_name", "")) if finalized else None
        self.db.save_analysis(meeting_id, result.get("summary", ""), result.get("actions", []), result.get("decisions", []), title=title, meeting_time=meeting_time, follow_ups=result["follow_ups"])
        if finalized:
            self.db.index_rag(meeting_id, title, transcript_chunks(rows), result.get("summary", ""), result.get("actions", []))
        self.db.delete_analysis_checkpoints(meeting_id)
        result.update({"title": title, "meeting_time": meeting_time, "thread": thread, "related_context": related})
        return result
    def query_meeting_memory(self, query: str, category: str = "all"): return self.db.rag_search(query, category)
    def related_meeting_context(self, meeting_id: str):
        rows = self.db.transcript(meeting_id)
        return self.db.related_context(self.llm.transcript_text(rows), meeting_id)
    def list_meetings(self): return self.db.meetings()
    def meeting_logs(self, meeting_id: str | None = None, limit: int | None = None): return self.db.logs(meeting_id, limit)
    def speakers(self, meeting_id: str): return self.db.speakers(meeting_id)
    def update_speakers(self, meeting_id: str, mapping: dict[str, str]):
        self.db.update_speakers(meeting_id, mapping)
        meeting = self.db.meeting(meeting_id) or {}
        title = meeting.get("title", "")
        summary = meeting.get("summary", "")
        if summary.strip() and title and not title.startswith("整理中"):
            actions = [
                {"item": row["item"], "owner": row["owner"], "due_date": row["due_date"]}
                for row in self.db.connection.execute(
                    "SELECT item,owner,due_date FROM actions WHERE meeting_id=? ORDER BY id", (meeting_id,)
                )
            ]
            self.db.index_rag(meeting_id, title, transcript_chunks(self.db.transcript(meeting_id)), summary, actions)
    def suggest_speaker_names(self, meeting_id: str, *, model_event: Callable[[dict], None] | None = None):
        rows = self.db.transcript(meeting_id)
        suggestions = self.llm.suggest_speaker_names(meeting_id, rows, model_event=model_event)
        self.db.save_speaker_suggestions(
            meeting_id,
            {item["speaker_id"]: item for item in suggestions},
        )
        return {"suggestions": suggestions, "speakers": self.db.speakers(meeting_id)}
    def evaluate_meeting(self, meeting_id: str):
        rows = self.db.transcript(meeting_id)
        related = self.db.related_context(self.llm.transcript_text(rows), meeting_id)
        result = self.evaluator.evaluate(meeting_id, rows, self.db.analysis(meeting_id), related)
        self.db.save_evaluation(result)
        return result
    def meeting_evaluations(self, meeting_id: str): return self.db.evaluations(meeting_id)
    def delete_meeting(self, meeting_id: str): return self.db.delete_meeting(meeting_id)
    def purge_database(self): return self.db.purge()
    def database_counts(self): return self.db.counts()
    def export_minutes(self, meeting_id: str, format: str = "json"):
        data = self.db.export(meeting_id)
        if format != "json": raise ValueError("目前只支援 json")
        return data
