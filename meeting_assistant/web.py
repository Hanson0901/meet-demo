from __future__ import annotations
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import parse_qs, urlparse
from .tools import MeetingTools

WEB_ROOT = Path(__file__).parent / "web"
UPLOAD_ROOT = Path("data/uploads")

def save_upload(body: bytes, content_type: str, upload_root: Path = UPLOAD_ROOT) -> tuple[str, str | None, str | None]:
    if not content_type or not content_type.lower().startswith("multipart/form-data"):
        raise ValueError("上傳內容不是有效的 multipart/form-data")
    upload_root.mkdir(parents=True, exist_ok=True)
    headers = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
    message = BytesParser(policy=policy.default).parsebytes(headers + body)
    # Browser FormData file parts use Content-Disposition: form-data rather
    # than attachment. Walk all nested parts to handle both forms.
    file_part = next((p for p in message.walk() if p is not message and p.get_filename()), None)
    if file_part is None:
        raise ValueError("請選擇音訊檔案")
    name = Path(file_part.get_filename() or "audio.bin").name
    safe_name = re.sub(r"[^\w.\-]", "_", name)
    target = upload_root / f"{uuid.uuid4().hex}_{safe_name}"
    payload = file_part.get_payload(decode=True)
    if not payload:
        raise ValueError("上傳的音訊檔案是空的")
    target.write_bytes(payload)

    def field_value(field_name: str) -> str | None:
        part = next((p for p in message.walk() if p is not message and p.get_content_disposition() == "form-data" and p.get_param("name", header="content-disposition") == field_name), None)
        if part is None:
            return None
        raw = part.get_payload(decode=True)
        return (raw.decode("utf-8", errors="replace") if raw else str(part.get_content())).strip()

    title = field_value("title") or Path(name).stem
    return str(target), title, field_value("started_at")


def read_json_body(body: bytes) -> dict:
    """Decode a JSON request body and return a JSON object with clear errors."""
    if not body or not body.strip():
        raise ValueError("請求內容是空的，請提供 JSON 物件")
    try:
        value = json.loads(body.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError("JSON 必須使用 UTF-8 編碼") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"請求內容不是有效 JSON（第 {exc.lineno} 行，第 {exc.colno} 欄）") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON 內容必須是物件")
    return value

def serve(tools: MeetingTools, host: str = "0.0.0.0", port: int = 8080):
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    admin_token = os.getenv("MEETING_ADMIN_TOKEN", "")
    transcription_jobs: dict[str, dict] = {}
    analysis_jobs: dict[str, dict] = {}
    speaker_suggestion_jobs: dict[str, dict] = {}
    analysis_jobs_lock = Lock()
    transcription_run_lock = Lock()

    terminal_job_statuses = {"completed", "failed", "canceled"}

    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def make_job(status: str, stage: str, **extra) -> dict:
        timestamp = now_iso()
        return {
            "status": status,
            "stage": stage,
            "error": None,
            "created_at": timestamp,
            "updated_at": timestamp,
            "started_at": None,
            "finished_at": None,
            "elapsed_seconds": 0.0,
            "steps": [],
            "model_stage": None,
            "model_output": "",
            "model_output_chars": 0,
            "model_output_complete": False,
            "model_output_started_at": None,
            "model_output_finished_at": None,
            "model_elapsed_seconds": None,
            "model_input_tokens": None,
            "model_output_limit_tokens": None,
            "model_finish_reason": None,
            "model_output_error": None,
            "_started_monotonic": None,
            **extra,
        }

    def close_current_step(job: dict, timestamp: str, monotonic_now: float) -> None:
        steps = job.get("steps") or []
        if not steps:
            return
        step = steps[-1]
        if step.get("finished_at") is not None:
            return
        step["finished_at"] = timestamp
        step["elapsed_seconds"] = round(
            monotonic_now - step.get("_started_monotonic", monotonic_now), 3
        )

    def update_job(job: dict, changes: dict) -> bool:
        """Update a live job and return whether its visible stage changed."""
        changes = dict(changes)
        timestamp = now_iso()
        monotonic_now = time.monotonic()
        status = changes.get("status")
        stage = changes.get("stage")
        stage_changed = bool(stage and stage != job.get("stage"))

        if status in ("running", "reloading") and job.get("started_at") is None:
            job["started_at"] = timestamp
            job["_started_monotonic"] = monotonic_now
        if stage_changed:
            close_current_step(job, timestamp, monotonic_now)
            job.setdefault("steps", []).append({
                "name": stage,
                "started_at": timestamp,
                "finished_at": None,
                "elapsed_seconds": None,
                "_started_monotonic": monotonic_now,
            })

        if changes.pop("model_output_reset", False):
            job["model_output"] = ""
            job["model_output_chars"] = 0
            job["model_output_complete"] = False
            job["model_output_finished_at"] = None
            job["model_output_error"] = None
        appended = changes.pop("model_output_append", None)
        if appended:
            job["model_output"] = f"{job.get('model_output', '')}{appended}"
            job["model_output_chars"] = len(job["model_output"])

        job.update(changes)
        if status in terminal_job_statuses:
            close_current_step(job, timestamp, monotonic_now)
            job["finished_at"] = timestamp
            if job.get("_started_monotonic") is not None:
                job["elapsed_seconds"] = round(monotonic_now - job["_started_monotonic"], 3)
        elif job.get("_started_monotonic") is not None:
            job["elapsed_seconds"] = round(monotonic_now - job["_started_monotonic"], 3)
        job["updated_at"] = timestamp
        return stage_changed

    def snapshot_job(job: dict) -> dict:
        """Return a JSON-safe job view without internal monotonic clock values."""
        result = {key: value for key, value in job.items() if not key.startswith("_") and key != "steps"}
        result["steps"] = []
        monotonic_now = time.monotonic()
        if job.get("_started_monotonic") is not None and job.get("status") not in terminal_job_statuses:
            result["elapsed_seconds"] = round(monotonic_now - job["_started_monotonic"], 3)
        for step in job.get("steps", []):
            public_step = {key: value for key, value in step.items() if not key.startswith("_")}
            if step.get("finished_at") is None and step.get("_started_monotonic") is not None:
                public_step["elapsed_seconds"] = round(monotonic_now - step["_started_monotonic"], 3)
            result["steps"].append(public_step)
        return result

    context_job = make_job(
        "idle",
        "尚未調整 context",
        context_tokens=tools.llm.context_tokens,
        model_reloaded=False,
    )

    def update_transcription_job(meeting_id: str, **changes) -> None:
        stage_changed = False
        state = {}
        with analysis_jobs_lock:
            job = transcription_jobs.get(meeting_id)
            if job is not None:
                stage_changed = update_job(job, changes)
                state = snapshot_job(job)
        if stage_changed:
            tools.logger.emit("STEP", f"ASR stage started: {changes.get('stage')}", meeting_id=meeting_id)
        if any(key in changes for key in ("status", "stage", "error")):
            try:
                tools.db.set_transcription_status(
                    meeting_id,
                    changes.get("status", "running"),
                    changes.get("stage", ""),
                    changes.get("error"),
                )
            except Exception as exc:
                tools.logger.emit("DB", f"transcription status save failed: {exc}", meeting_id=meeting_id, level="ERROR")
        if changes.get("status") in terminal_job_statuses:
            tools.logger.emit(
                "STEP",
                f"ASR stage finished status={changes['status']} elapsed_seconds={state.get('elapsed_seconds', 0)}",
                meeting_id=meeting_id,
                level="ERROR" if changes["status"] == "failed" else "INFO",
            )

    def update_analysis_job(meeting_id: str, **changes) -> None:
        stage_changed = False
        state = {}
        with analysis_jobs_lock:
            job = analysis_jobs.get(meeting_id)
            if job is not None:
                stage_changed = update_job(job, changes)
                state = snapshot_job(job)
        if stage_changed:
            tools.logger.emit("STEP", f"摘要 stage started: {changes.get('stage')}", meeting_id=meeting_id)
        if changes.get("status") in terminal_job_statuses:
            tools.logger.emit(
                "STEP",
                f"摘要 stage finished status={changes['status']} elapsed_seconds={state.get('elapsed_seconds', 0)}",
                meeting_id=meeting_id,
                level="ERROR" if changes["status"] == "failed" else "INFO",
            )

    def update_context_job(**changes) -> None:
        stage_changed = False
        with analysis_jobs_lock:
            stage_changed = update_job(context_job, changes)
            state = snapshot_job(context_job)
        if stage_changed:
            tools.logger.emit("STEP", f"context stage started: {changes.get('stage')}")
        if changes.get("status") in terminal_job_statuses:
            tools.logger.emit(
                "STEP",
                f"context stage finished status={changes['status']} elapsed_seconds={state.get('elapsed_seconds', 0)}",
                level="ERROR" if changes["status"] == "failed" else "INFO",
            )

    def run_context_update(context_tokens: int) -> None:
        while True:
            with analysis_jobs_lock:
                active_analysis = any(
                    job["status"] in ("queued", "running") for job in analysis_jobs.values()
                )
                active_speaker_suggestion = any(
                    job["status"] in ("queued", "running") for job in speaker_suggestion_jobs.values()
                )
            if not active_analysis and not active_speaker_suggestion:
                break
            update_context_job(stage="等待目前的模型工作完成")
            time.sleep(0.5)

        update_context_job(status="reloading", stage="正在套用設定並重新載入模型")
        try:
            result = tools.llm.set_context_tokens(context_tokens)
            tools.db.set_setting("llama_cpp_context", str(context_tokens))
        except Exception as exc:
            tools.logger.emit("LLM", f"context update failed: {exc}", level="ERROR")
            update_context_job(status="failed", stage="context 調整失敗", error=str(exc))
        else:
            if result["unchanged"]:
                stage = f"context 維持 {context_tokens}，設定已保存"
            elif result["model_reloaded"]:
                stage = f"context 已調整為 {context_tokens}，模型重新載入完成"
            else:
                stage = f"context 已調整為 {context_tokens}，模型下次使用時載入"
            update_context_job(
                status="completed",
                stage=stage,
                context_tokens=context_tokens,
                model_reloaded=result["model_reloaded"],
                error=None,
            )

    def run_transcription(meeting_id: str, audio_path: str) -> None:
        with transcription_run_lock:
            update_transcription_job(meeting_id, status="running", stage="準備語音辨識")
            started_monotonic = time.monotonic()
            tools.logger.emit("STEP", "ASR workflow started", meeting_id=meeting_id)
            try:
                tools.transcribe_meeting(
                    audio_path,
                    meeting_id=meeting_id,
                    progress=lambda stage: update_transcription_job(meeting_id, stage=stage),
                )
            except Exception as exc:
                tools.logger.emit("ASR", f"background transcription failed: {exc}", meeting_id=meeting_id, level="ERROR")
                update_transcription_job(
                    meeting_id,
                    status="failed",
                    stage="語音辨識失敗",
                    error=str(exc),
                    utterance_count=len(tools.db.transcript(meeting_id)),
                )
                tools.logger.emit(
                    "STEP",
                    f"ASR workflow failed elapsed_seconds={round(time.monotonic() - started_monotonic, 3)}",
                    meeting_id=meeting_id,
                    level="ERROR",
                )
            else:
                update_transcription_job(
                    meeting_id,
                    status="completed",
                    stage="語音辨識完成",
                    error=None,
                    utterance_count=len(tools.db.transcript(meeting_id)),
                )
                tools.logger.emit(
                    "STEP",
                    f"ASR workflow completed elapsed_seconds={round(time.monotonic() - started_monotonic, 3)}",
                    meeting_id=meeting_id,
                )

    def start_transcription(meeting_id: str, audio_path: str) -> tuple[dict, int]:
        with analysis_jobs_lock:
            existing = transcription_jobs.get(meeting_id)
            if existing and existing["status"] in ("queued", "running"):
                return {"meeting_id": meeting_id, **snapshot_job(existing)}, 202
            job = {
                **make_job("queued", "等待 FunASR"),
                "utterance_count": 0,
            }
            transcription_jobs[meeting_id] = job
            worker = Thread(
                target=run_transcription,
                args=(meeting_id, audio_path),
                daemon=True,
                name=f"transcription-{meeting_id[:8]}",
            )
            worker.start()
            return {"meeting_id": meeting_id, **snapshot_job(job)}, 202

    def handle_model_event(updater, meeting_id: str, event: dict) -> None:
        event_type = event.get("type")
        if event_type == "start":
            updater(
                meeting_id,
                model_stage=event.get("stage"),
                model_output_reset=True,
                model_output_started_at=event.get("started_at") or event.get("timestamp"),
                model_input_tokens=event.get("input_tokens"),
                model_output_limit_tokens=event.get("output_limit_tokens"),
            )
        elif event_type == "token":
            updater(
                meeting_id,
                model_output_append=str(event.get("text", "")),
                model_output_complete=False,
            )
        elif event_type == "complete":
            updater(
                meeting_id,
                model_stage=event.get("stage"),
                model_output_complete=True,
                model_output_finished_at=event.get("timestamp"),
                model_elapsed_seconds=event.get("elapsed_seconds"),
                model_finish_reason=event.get("finish_reason"),
                model_output_chars=event.get("output_chars", 0),
            )
        elif event_type == "error":
            updater(
                meeting_id,
                model_output_complete=False,
                model_output_finished_at=event.get("timestamp"),
                model_elapsed_seconds=event.get("elapsed_seconds"),
                model_finish_reason=event.get("finish_reason"),
                model_output_error=event.get("error"),
                model_output_chars=event.get("output_chars", 0),
            )

    def run_analysis(meeting_id: str) -> None:
        update_analysis_job(meeting_id, status="running", stage="準備摘要")
        started_monotonic = time.monotonic()
        tools.logger.emit("STEP", "摘要 workflow started", meeting_id=meeting_id)
        try:
            tools.summarize_meeting(
                meeting_id,
                progress=lambda stage: update_analysis_job(meeting_id, stage=stage),
                model_event=lambda event: handle_model_event(update_analysis_job, meeting_id, event),
            )
        except Exception as exc:
            tools.logger.emit("LLM", f"llama.cpp request failed: {exc}", meeting_id=meeting_id, level="ERROR")
            update_analysis_job(meeting_id, status="failed", stage="摘要失敗", error=str(exc))
            tools.logger.emit(
                "STEP",
                f"摘要 workflow failed elapsed_seconds={round(time.monotonic() - started_monotonic, 3)}",
                meeting_id=meeting_id,
                level="ERROR",
            )
        else:
            update_analysis_job(meeting_id, status="completed", stage="摘要完成", error=None)
            tools.logger.emit(
                "STEP",
                f"摘要 workflow completed elapsed_seconds={round(time.monotonic() - started_monotonic, 3)}",
                meeting_id=meeting_id,
            )

    def update_speaker_suggestion_job(meeting_id: str, **changes) -> None:
        stage_changed = False
        state = {}
        with analysis_jobs_lock:
            job = speaker_suggestion_jobs.get(meeting_id)
            if job is not None:
                stage_changed = update_job(job, changes)
                state = snapshot_job(job)
        if stage_changed:
            tools.logger.emit("STEP", f"姓名推測 stage started: {changes.get('stage')}", meeting_id=meeting_id)
        if changes.get("status") in terminal_job_statuses:
            tools.logger.emit(
                "STEP",
                f"姓名推測 stage finished status={changes['status']} elapsed_seconds={state.get('elapsed_seconds', 0)}",
                meeting_id=meeting_id,
                level="ERROR" if changes["status"] == "failed" else "INFO",
            )

    def run_speaker_suggestions(meeting_id: str) -> None:
        update_speaker_suggestion_job(meeting_id, status="running", stage="本地模型正在分析說話者線索")
        started_monotonic = time.monotonic()
        tools.logger.emit("STEP", "姓名推測 workflow started", meeting_id=meeting_id)
        try:
            result = tools.suggest_speaker_names(
                meeting_id,
                model_event=lambda event: handle_model_event(update_speaker_suggestion_job, meeting_id, event),
            )
        except Exception as exc:
            tools.logger.emit("ASR", f"speaker name suggestion failed: {exc}", meeting_id=meeting_id, level="ERROR")
            update_speaker_suggestion_job(meeting_id, status="failed", stage="姓名推測失敗", error=str(exc))
            tools.logger.emit(
                "STEP",
                f"姓名推測 workflow failed elapsed_seconds={round(time.monotonic() - started_monotonic, 3)}",
                meeting_id=meeting_id,
                level="ERROR",
            )
        else:
            update_speaker_suggestion_job(
                meeting_id, status="completed", stage="姓名候選已產生", error=None, result=result
            )
            tools.logger.emit(
                "STEP",
                f"姓名推測 workflow completed elapsed_seconds={round(time.monotonic() - started_monotonic, 3)}",
                meeting_id=meeting_id,
            )

    def start_speaker_suggestions(meeting_id: str) -> tuple[dict, int]:
        with analysis_jobs_lock:
            active_transcription = next(
                ((active_id, job) for active_id, job in transcription_jobs.items() if job["status"] in ("queued", "running")),
                None,
            )
            if active_transcription:
                return {"error": "語音辨識尚未完成，請稍候再推測姓名。"}, 409
            if context_job["status"] in ("queued", "reloading"):
                return {"error": "正在套用模型 context 設定，請稍後再試。"}, 409
            active_analysis = next(
                ((active_id, job) for active_id, job in analysis_jobs.items() if job["status"] in ("queued", "running")),
                None,
            )
            if active_analysis:
                return {"error": "本地摘要正在使用模型，請完成後再推測姓名。"}, 409
            active_suggestion = next(
                ((active_id, job) for active_id, job in speaker_suggestion_jobs.items() if job["status"] in ("queued", "running")),
                None,
            )
            if active_suggestion:
                active_id, job = active_suggestion
                if active_id == meeting_id:
                    return {"meeting_id": meeting_id, **snapshot_job(job)}, 202
                return {"error": "另一場會議正在進行姓名推測，請稍後再試。"}, 409

            job = make_job("queued", "等待本地模型")
            speaker_suggestion_jobs[meeting_id] = job
            worker = Thread(
                target=run_speaker_suggestions,
                args=(meeting_id,),
                daemon=True,
                name=f"speaker-names-{meeting_id[:8]}",
            )
            worker.start()
            return {"meeting_id": meeting_id, **snapshot_job(job)}, 202

    def start_analysis(meeting_id: str) -> tuple[dict, int]:
        with analysis_jobs_lock:
            active_transcription = next(
                ((active_id, job) for active_id, job in transcription_jobs.items() if job["status"] in ("queued", "running")),
                None,
            )
            if active_transcription:
                return {"error": "語音辨識尚未完成，請等待逐字稿建立後再摘要。"}, 409
            if context_job["status"] in ("queued", "reloading"):
                return {"error": "正在套用模型 context 設定，請稍後再開始摘要。"}, 409
            active_speaker_suggestion = next(
                ((active_id, job) for active_id, job in speaker_suggestion_jobs.items() if job["status"] in ("queued", "running")),
                None,
            )
            if active_speaker_suggestion:
                return {"error": "本地姓名推測正在使用模型，請稍後再開始摘要。"}, 409
            active = next(
                ((active_id, job) for active_id, job in analysis_jobs.items() if job["status"] in ("queued", "running")),
                None,
            )
            if active:
                active_id, job = active
                if active_id == meeting_id:
                    return {"meeting_id": meeting_id, **snapshot_job(job)}, 202
                return {
                    "error": f"另一場會議正在摘要中（{active_id}），請稍後再試。",
                    "active_meeting_id": active_id,
                }, 409

            job = make_job("queued", "等待摘要工作")
            analysis_jobs[meeting_id] = job
            worker = Thread(target=run_analysis, args=(meeting_id,), daemon=True, name=f"summary-{meeting_id[:8]}")
            worker.start()
            return {"meeting_id": meeting_id, **snapshot_job(job)}, 202

    def start_context_update(context_tokens: int) -> tuple[dict, int]:
        with analysis_jobs_lock:
            if context_job["status"] in ("queued", "reloading"):
                return {"error": "另一個 context 調整正在進行，請稍候。"}, 409
            context_job.clear()
            context_job.update(make_job(
                "queued",
                "等待套用 context 設定",
                context_tokens=context_tokens,
                model_reloaded=False,
            ))
            worker = Thread(
                target=run_context_update,
                args=(context_tokens,),
                daemon=True,
                name="llama-context-update",
            )
            worker.start()
            return snapshot_job(context_job), 202

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, body: bytes, content_type: str, status: int = 200):
            try:
                self.send_response(status); self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body))); self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def send_json(self, value, status: int = 200):
            self.send_bytes(json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

        def read_body(self) -> bytes:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("Content-Length 不是有效數字") from exc
            if length < 0:
                raise ValueError("Content-Length 不能是負數")
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("請求內容不完整，請重新上傳")
            return body

        def do_GET(self):
            parsed = urlparse(self.path)
            transcription_status_prefix, transcription_status_suffix = "/api/meetings/", "/transcription-status"
            if parsed.path.startswith(transcription_status_prefix) and parsed.path.endswith(transcription_status_suffix):
                meeting_id = parsed.path[len(transcription_status_prefix):-len(transcription_status_suffix)]
                if tools.db.meeting(meeting_id) is None:
                    return self.send_json({"error": "meeting not found"}, 404)
                with analysis_jobs_lock:
                    job = transcription_jobs.get(meeting_id)
                    state = snapshot_job(job) if job else None
                if state is None:
                    count = len(tools.db.transcript(meeting_id))
                    meeting = tools.db.meeting(meeting_id) or {}
                    stored_status = meeting.get("transcription_status", "not_started")
                    if stored_status in ("queued", "running"):
                        stored_status = "failed"
                        stored_stage = "服務重啟後辨識未完成"
                        stored_error = "原語音辨識工作已中止，請重新上傳音檔。"
                    else:
                        stored_stage = meeting.get("transcription_stage") or ("逐字稿已存在" if count else "尚未開始語音辨識")
                        stored_error = meeting.get("transcription_error") or None
                    state = snapshot_job(make_job(
                        stored_status if stored_status != "not_started" else ("completed" if count else "idle"),
                        stored_stage,
                        error=stored_error,
                        utterance_count=count,
                    ))
                state["logs"] = tools.meeting_logs(meeting_id, limit=200)
                return self.send_json({"meeting_id": meeting_id, **state})
            speaker_status_prefix, speaker_status_suffix = "/api/meetings/", "/speaker-suggestions-status"
            if parsed.path.startswith(speaker_status_prefix) and parsed.path.endswith(speaker_status_suffix):
                meeting_id = parsed.path[len(speaker_status_prefix):-len(speaker_status_suffix)]
                with analysis_jobs_lock:
                    job = speaker_suggestion_jobs.get(meeting_id)
                    state = snapshot_job(job) if job else snapshot_job(make_job("idle", "尚未推測姓名"))
                state["logs"] = tools.meeting_logs(meeting_id, limit=200)
                return self.send_json({"meeting_id": meeting_id, **state})
            status_prefix, status_suffix = "/api/meetings/", "/analysis-status"
            if parsed.path.startswith(status_prefix) and parsed.path.endswith(status_suffix):
                meeting_id = parsed.path[len(status_prefix) : -len(status_suffix)]
                with analysis_jobs_lock:
                    job = analysis_jobs.get(meeting_id)
                    state = snapshot_job(job) if job else snapshot_job(make_job("idle", "尚未開始摘要"))
                state["logs"] = tools.meeting_logs(meeting_id, limit=200)
                return self.send_json({"meeting_id": meeting_id, **state})
            if parsed.path == "/api/admin/config":
                if not self.authorized(): return self.send_json({"error": "需要管理員 token"}, 401)
                with analysis_jobs_lock:
                    job = snapshot_job(context_job)
                return self.send_json({
                    "context_tokens": tools.llm.context_tokens,
                    "model_loaded": tools.llm.model is not None,
                    "min_context_tokens": tools.llm.MIN_CONTEXT_TOKENS,
                    "max_context_tokens": tools.llm.MAX_CONTEXT_TOKENS,
                    "token_settings": tools.llm.get_token_settings(),
                    "token_setting_specs": tools.llm.token_settings_schema(),
                    "job": job,
                })
            if parsed.path in ("/", "/index.html"): return self.send_bytes((WEB_ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
            if parsed.path == "/evaluate": return self.send_bytes((WEB_ROOT / "evaluation.html").read_bytes(), "text/html; charset=utf-8")
            if parsed.path == "/admin": return self.send_bytes((WEB_ROOT / "admin.html").read_bytes(), "text/html; charset=utf-8")
            if parsed.path == "/app.css": return self.send_bytes((WEB_ROOT / "app.css").read_bytes(), "text/css; charset=utf-8")
            if parsed.path == "/app.js": return self.send_bytes((WEB_ROOT / "app.js").read_bytes(), "application/javascript; charset=utf-8")
            if parsed.path == "/evaluation.css": return self.send_bytes((WEB_ROOT / "evaluation.css").read_bytes(), "text/css; charset=utf-8")
            if parsed.path == "/evaluation.js": return self.send_bytes((WEB_ROOT / "evaluation.js").read_bytes(), "application/javascript; charset=utf-8")
            if parsed.path == "/admin.css": return self.send_bytes((WEB_ROOT / "admin.css").read_bytes(), "text/css; charset=utf-8")
            if parsed.path == "/admin.js": return self.send_bytes((WEB_ROOT / "admin.js").read_bytes(), "application/javascript; charset=utf-8")
            if parsed.path == "/health": return self.send_json({"status": "ok"})
            if parsed.path in ("/api/meetings", "/meetings"): return self.send_json(tools.list_meetings())
            if parsed.path == "/api/admin/stats": return self.send_json(tools.database_counts())
            if parsed.path == "/api/evaluations":
                meeting_id = parse_qs(parsed.query).get("meeting_id", [""])[0]
                return self.send_json(tools.meeting_evaluations(meeting_id) if meeting_id else {"error": "meeting_id is required"}, 200 if meeting_id else 400)
            if parsed.path in ("/api/search", "/search"):
                params = parse_qs(parsed.query)
                return self.send_json(tools.query_meeting_memory(params.get("q", [""])[0], params.get("type", ["all"])[0]))
            if parsed.path.startswith(("/api/meetings/", "/meetings/")):
                meeting_id = parsed.path.rsplit("/", 1)[-1]; data = tools.export_minutes(meeting_id)
                if data["meeting"] is None: return self.send_json({"error": "meeting not found"}, 404)
                data["logs"] = tools.meeting_logs(meeting_id); return self.send_json(data)
            return self.send_json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path == "/api/evaluations":
                try:
                    payload = read_json_body(self.read_body())
                    if not payload.get("meeting_id"): return self.send_json({"error": "meeting_id is required"}, 400)
                    return self.send_json(tools.evaluate_meeting(payload["meeting_id"]), 201)
                except ValueError as exc:
                    return self.send_json({"error": str(exc)}, 400)
                except Exception as exc:
                    meeting_id = payload.get("meeting_id", "") if "payload" in locals() else ""
                    tools.logger.emit("EVAL", f"evaluation failed: {exc}", meeting_id=meeting_id or None, level="ERROR")
                    return self.send_json({"error": str(exc)}, 502)
            speaker_suggestions_suffix = "/speaker-suggestions"
            if self.path.startswith("/api/meetings/") and self.path.endswith(speaker_suggestions_suffix):
                meeting_id = self.path[len("/api/meetings/"):-len(speaker_suggestions_suffix)]
                if tools.db.meeting(meeting_id) is None:
                    return self.send_json({"error": "meeting not found"}, 404)
                payload, status = start_speaker_suggestions(meeting_id)
                return self.send_json(payload, status)
            if self.path.startswith("/api/meetings/") and self.path.endswith("/analyze"):
                meeting_id = self.path[len("/api/meetings/"):-len("/analyze")]
                if tools.db.meeting(meeting_id) is None:
                    return self.send_json({"error": "meeting not found"}, 404)
                payload, status = start_analysis(meeting_id)
                return self.send_json(payload, status)
            if self.path == "/api/admin/purge":
                if not self.authorized(): return self.send_json({"error": "需要管理員 token"}, 401)
                count = tools.purge_database()
                for upload in UPLOAD_ROOT.iterdir():
                    if upload.is_file(): upload.unlink()
                return self.send_json({"deleted_meetings": count})
            if self.path not in ("/api/transcribe", "/transcribe"): return self.send_json({"error": "not found"}, 404)
            try:
                body = self.read_body()
                content_type = self.headers.get("Content-Type", "")
                if content_type.lower().startswith("multipart/form-data"):
                    audio_path, title, started_at = save_upload(body, content_type)
                else:
                    payload = read_json_body(body)
                    audio_path = payload.get("audio")
                    if not audio_path:
                        raise ValueError("JSON 缺少 audio 音檔路徑")
                    title, started_at = payload.get("title"), payload.get("started_at")
                meeting_id = tools.prepare_meeting(title, started_at)
                payload, status = start_transcription(meeting_id, audio_path)
                return self.send_json(payload, status)
            except ValueError as exc: return self.send_json({"error": str(exc)}, 400)
            except Exception as exc:
                tools.logger.emit("UI", f"request failed: {exc}", level="ERROR"); return self.send_json({"error": "處理失敗，請查看 log"}, 500)

        def do_PUT(self):
            if self.path == "/api/admin/config":
                if not self.authorized(): return self.send_json({"error": "需要管理員 token"}, 401)
                try:
                    payload = read_json_body(self.read_body())
                except ValueError as exc:
                    return self.send_json({"error": str(exc)}, 400)
                context_tokens = payload.get("context_tokens")
                token_settings = payload.get("token_settings")
                if context_tokens is not None and token_settings is not None:
                    return self.send_json({"error": "請分開套用 context 與摘要 token 設定"}, 400)
                if token_settings is not None:
                    try:
                        validated = tools.llm.validate_token_settings(token_settings)
                    except ValueError as exc:
                        return self.send_json({"error": str(exc)}, 400)
                    tools.db.set_settings({
                        f"llama_cpp_{key}": str(value) for key, value in validated.items()
                    })
                    current = tools.llm.update_token_settings(validated)
                    return self.send_json({
                        "token_settings": current,
                        "message": "摘要 token 設定已保存，下一次分析生效；不需重啟服務。",
                    })
                if isinstance(context_tokens, bool) or not isinstance(context_tokens, int):
                    return self.send_json({"error": "context_tokens 必須是整數"}, 400)
                if not tools.llm.MIN_CONTEXT_TOKENS <= context_tokens <= tools.llm.MAX_CONTEXT_TOKENS:
                    return self.send_json({
                        "error": f"context_tokens 必須介於 {tools.llm.MIN_CONTEXT_TOKENS} 到 {tools.llm.MAX_CONTEXT_TOKENS}"
                    }, 400)
                result, status = start_context_update(context_tokens)
                return self.send_json(result, status)

            prefix, suffix = "/api/meetings/", "/speakers"
            if not (self.path.startswith(prefix) and self.path.endswith(suffix)): return self.send_json({"error": "not found"}, 404)
            meeting_id = self.path[len(prefix):-len(suffix)]
            try:
                payload = read_json_body(self.read_body())
                tools.update_speakers(meeting_id, payload.get("speakers", {}))
                return self.send_json({"speakers": tools.speakers(meeting_id)})
            except ValueError as exc: return self.send_json({"error": str(exc)}, 400)

        def do_DELETE(self):
            prefix = "/api/admin/meetings/"
            if not self.path.startswith(prefix): return self.send_json({"error": "not found"}, 404)
            if not self.authorized(): return self.send_json({"error": "需要管理員 token"}, 401)
            meeting_id = self.path[len(prefix):]
            if not tools.delete_meeting(meeting_id): return self.send_json({"error": "meeting not found"}, 404)
            return self.send_json({"deleted": meeting_id})

        def authorized(self):
            return not admin_token or self.headers.get("X-Admin-Token", "") == admin_token

        def handle(self):
            try:
                super().handle()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_args): pass

    server = ThreadingHTTPServer((host, port), Handler); tools.logger.emit("UI", f"server listening http://{host}:{port}"); server.serve_forever()
