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
    def transcribe_meeting(self, audio: str, title: str | None = None, started_at: str | None = None) -> str:
        meeting_id = str(uuid.uuid4())
        try:
            started = datetime.fromisoformat((started_at or "").replace("Z", "+00:00")) if started_at else datetime.now().astimezone()
        except ValueError:
            started = datetime.now().astimezone()
        created_at = started.isoformat(timespec="seconds")
        temporary_title = f"整理中 · {started.astimezone().strftime('%Y-%m-%d %H:%M')}"
        self.db.create_meeting(meeting_id, temporary_title, created_at)
        for segment in self.audio.segments(audio):
            for item in self.asr.transcribe(meeting_id, segment):
                self.db.save_utterance(item, convert_traditional(item.text)); self.logger.emit("DB", f"utterance saved id={item.utterance_id}", meeting_id=meeting_id)
        return meeting_id
    def summarize_meeting(self, meeting_id: str, *, progress: Callable[[str], None] | None = None):
        rows = self.db.transcript(meeting_id)
        query = self.llm.transcript_text(rows)
        related = self.db.related_context(query, meeting_id)
        result = self.llm.analyze(meeting_id, rows, related, progress=progress)
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
    def meeting_logs(self, meeting_id: str | None = None): return self.db.logs(meeting_id)
    def speakers(self, meeting_id: str): return self.db.speakers(meeting_id)
    def update_speakers(self, meeting_id: str, mapping: dict[str, str]): return self.db.update_speakers(meeting_id, mapping)
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
