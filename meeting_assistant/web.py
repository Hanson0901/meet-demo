from __future__ import annotations
import json
import os
import re
import time
import uuid
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
    upload_root.mkdir(parents=True, exist_ok=True)
    headers = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
    message = BytesParser(policy=policy.default).parsebytes(headers + body)
    file_part = next((p for p in message.iter_attachments() if p.get_filename()), None)
    if file_part is None:
        raise ValueError("請選擇音訊檔案")
    name = Path(file_part.get_filename() or "audio.bin").name
    safe_name = re.sub(r"[^\w.\-]", "_", name)
    target = upload_root / f"{uuid.uuid4().hex}_{safe_name}"
    target.write_bytes(file_part.get_payload(decode=True) or b"")

    def field_value(field_name: str) -> str | None:
        part = next((p for p in message.iter_parts() if p.get_content_disposition() == "form-data" and p.get_param("name", header="content-disposition") == field_name), None)
        if part is None:
            return None
        raw = part.get_payload(decode=True)
        return (raw.decode("utf-8", errors="replace") if raw else str(part.get_content())).strip()

    title = field_value("title") or Path(name).stem
    return str(target), title, field_value("started_at")

def serve(tools: MeetingTools, host: str = "0.0.0.0", port: int = 8080):
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    admin_token = os.getenv("MEETING_ADMIN_TOKEN", "")
    analysis_jobs: dict[str, dict] = {}
    analysis_jobs_lock = Lock()
    context_job = {
        "status": "idle",
        "stage": "尚未調整 context",
        "context_tokens": tools.llm.context_tokens,
        "model_reloaded": False,
        "error": None,
    }

    def update_analysis_job(meeting_id: str, **changes) -> None:
        with analysis_jobs_lock:
            job = analysis_jobs.get(meeting_id)
            if job is not None:
                job.update(changes)

    def update_context_job(**changes) -> None:
        with analysis_jobs_lock:
            context_job.update(changes)

    def run_context_update(context_tokens: int) -> None:
        while True:
            with analysis_jobs_lock:
                active_analysis = any(
                    job["status"] in ("queued", "running") for job in analysis_jobs.values()
                )
            if not active_analysis:
                break
            update_context_job(stage="等待目前的會議摘要完成")
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

    def run_analysis(meeting_id: str) -> None:
        update_analysis_job(meeting_id, status="running", stage="準備摘要")
        try:
            tools.summarize_meeting(meeting_id, progress=lambda stage: update_analysis_job(meeting_id, stage=stage))
        except Exception as exc:
            tools.logger.emit("LLM", f"llama.cpp request failed: {exc}", meeting_id=meeting_id, level="ERROR")
            update_analysis_job(meeting_id, status="failed", stage="摘要失敗", error=str(exc))
        else:
            update_analysis_job(meeting_id, status="completed", stage="摘要完成", error=None)

    def start_analysis(meeting_id: str) -> tuple[dict, int]:
        with analysis_jobs_lock:
            if context_job["status"] in ("queued", "reloading"):
                return {"error": "正在套用模型 context 設定，請稍後再開始摘要。"}, 409
            active = next(
                ((active_id, job) for active_id, job in analysis_jobs.items() if job["status"] in ("queued", "running")),
                None,
            )
            if active:
                active_id, job = active
                if active_id == meeting_id:
                    return {"meeting_id": meeting_id, **job}, 202
                return {
                    "error": f"另一場會議正在摘要中（{active_id}），請稍後再試。",
                    "active_meeting_id": active_id,
                }, 409

            job = {"status": "queued", "stage": "等待摘要工作", "error": None}
            analysis_jobs[meeting_id] = job
            worker = Thread(target=run_analysis, args=(meeting_id,), daemon=True, name=f"summary-{meeting_id[:8]}")
            worker.start()
            return {"meeting_id": meeting_id, **job}, 202

    def start_context_update(context_tokens: int) -> tuple[dict, int]:
        with analysis_jobs_lock:
            if context_job["status"] in ("queued", "reloading"):
                return {"error": "另一個 context 調整正在進行，請稍候。"}, 409
            context_job.update(
                status="queued",
                stage="等待套用 context 設定",
                context_tokens=context_tokens,
                model_reloaded=False,
                error=None,
            )
            worker = Thread(
                target=run_context_update,
                args=(context_tokens,),
                daemon=True,
                name="llama-context-update",
            )
            worker.start()
            return {**context_job}, 202

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, body: bytes, content_type: str, status: int = 200):
            self.send_response(status); self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def send_json(self, value, status: int = 200):
            self.send_bytes(json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

        def do_GET(self):
            parsed = urlparse(self.path)
            status_prefix, status_suffix = "/api/meetings/", "/analysis-status"
            if parsed.path.startswith(status_prefix) and parsed.path.endswith(status_suffix):
                meeting_id = parsed.path[len(status_prefix) : -len(status_suffix)]
                with analysis_jobs_lock:
                    job = analysis_jobs.get(meeting_id)
                    state = dict(job) if job else {"status": "idle", "stage": "尚未開始摘要", "error": None}
                return self.send_json({"meeting_id": meeting_id, **state})
            if parsed.path == "/api/admin/config":
                if not self.authorized(): return self.send_json({"error": "需要管理員 token"}, 401)
                with analysis_jobs_lock:
                    job = dict(context_job)
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
                    length = int(self.headers.get("Content-Length", "0")); payload = json.loads(self.rfile.read(length) or b"{}")
                    if not payload.get("meeting_id"): return self.send_json({"error": "meeting_id is required"}, 400)
                    return self.send_json(tools.evaluate_meeting(payload["meeting_id"]), 201)
                except (json.JSONDecodeError, ValueError) as exc:
                    return self.send_json({"error": str(exc)}, 400)
                except Exception as exc:
                    meeting_id = payload.get("meeting_id", "") if "payload" in locals() else ""
                    tools.logger.emit("EVAL", f"evaluation failed: {exc}", meeting_id=meeting_id or None, level="ERROR")
                    return self.send_json({"error": str(exc)}, 502)
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
                length = int(self.headers.get("Content-Length", "0")); body = self.rfile.read(length); content_type = self.headers.get("Content-Type", "")
                if content_type.startswith("multipart/form-data"): audio_path, title, started_at = save_upload(body, content_type)
                else:
                    payload = json.loads(body or b"{}"); audio_path, title, started_at = payload["audio"], payload.get("title"), payload.get("started_at")
                return self.send_json({"meeting_id": tools.transcribe_meeting(audio_path, title, started_at=started_at)}, 202)
            except (KeyError, json.JSONDecodeError, ValueError) as exc: return self.send_json({"error": str(exc)}, 400)
            except Exception as exc:
                tools.logger.emit("UI", f"request failed: {exc}", level="ERROR"); return self.send_json({"error": "處理失敗，請查看 log"}, 500)

        def do_PUT(self):
            if self.path == "/api/admin/config":
                if not self.authorized(): return self.send_json({"error": "需要管理員 token"}, 401)
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length) or b"{}")
                except (json.JSONDecodeError, ValueError):
                    return self.send_json({"error": "設定內容不是有效 JSON"}, 400)
                if not isinstance(payload, dict):
                    return self.send_json({"error": "設定內容必須是 JSON 物件"}, 400)
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
                length = int(self.headers.get("Content-Length", "0")); payload = json.loads(self.rfile.read(length) or b"{}")
                tools.update_speakers(meeting_id, payload.get("speakers", {}))
                return self.send_json({"speakers": tools.speakers(meeting_id)})
            except (json.JSONDecodeError, ValueError) as exc: return self.send_json({"error": str(exc)}, 400)

        def do_DELETE(self):
            prefix = "/api/admin/meetings/"
            if not self.path.startswith(prefix): return self.send_json({"error": "not found"}, 404)
            if not self.authorized(): return self.send_json({"error": "需要管理員 token"}, 401)
            meeting_id = self.path[len(prefix):]
            if not tools.delete_meeting(meeting_id): return self.send_json({"error": "meeting not found"}, 404)
            return self.send_json({"deleted": meeting_id})

        def authorized(self):
            return not admin_token or self.headers.get("X-Admin-Token", "") == admin_token

        def log_message(self, *_args): pass

    server = ThreadingHTTPServer((host, port), Handler); tools.logger.emit("UI", f"server listening http://{host}:{port}"); server.serve_forever()
