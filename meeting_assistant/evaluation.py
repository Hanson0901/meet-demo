from __future__ import annotations
import json
import os
import secrets
import urllib.error
import urllib.request
from datetime import datetime, timezone
from .orchestrator import LLMOrchestrator

DEFAULT_BASE_URL = "https://air.cgu.edu.tw/cgullmapi/v1"
DEFAULT_MODEL = "gpt-6-luna"

class ResponsesClient:
    def __init__(self):
        self.base_url = os.getenv("CGULLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        self.model = os.getenv("CGULLM_MODEL", DEFAULT_MODEL)
        self.api_key = os.getenv("CGULLM_API_KEY", "")
    def create(self, instructions: str, input_text: str, *, max_output_tokens: int = 2400) -> str:
        if not self.api_key:
            raise RuntimeError("未設定 CGULLM_API_KEY，請在 ~/.bashrc 設定 API key 後重新載入 shell")
        payload = json.dumps({"model": self.model, "instructions": instructions, "input": input_text, "max_output_tokens": max_output_tokens}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(f"{self.base_url}/responses", data=payload, headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, method="POST")
        timeout = int(os.getenv("CGULLM_TIMEOUT", "180"))
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"Responses API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"無法連線至 Responses API：{exc.reason}") from exc
        text = data.get("output_text")
        if text:
            return text
        fragments = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in ("output_text", "text") and content.get("text"):
                    fragments.append(content["text"])
        if not fragments:
            raise RuntimeError("Responses API 回應中沒有可讀取的文字輸出")
        return "\n".join(fragments)

def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].removesuffix("```").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("模型沒有回傳有效 JSON")
    return json.loads(text[start:end + 1])

class MeetingEvaluator:
    def __init__(self, logger):
        self.logger = logger
        self.remote = ResponsesClient()
    def evaluate(self, meeting_id: str, rows: list[dict], local: dict | None, related_context: list[dict] | None = None) -> dict:
        transcript = LLMOrchestrator.transcript_text(rows)
        if not transcript.strip():
            raise ValueError("這場會議沒有逐字稿可供評估")
        if not local or not str(local.get("summary", "")).strip():
            raise ValueError("請先在會議頁完成一次 llama.cpp 分析，再執行模型評估")
        prompt = LLMOrchestrator.analysis_prompt(transcript, related_context)
        self.logger.emit("EVAL", f"開始呼叫遠端摘要模型={self.remote.model}", meeting_id=meeting_id)
        remote_text = self.remote.create("你是繁體中文會議秘書。請依照使用者要求輸出合法 JSON，不要使用 Markdown。", prompt)
        remote = _parse_json(remote_text)
        label_map = {"A": "local", "B": "remote"}
        if secrets.randbelow(2): label_map = {"A": "remote", "B": "local"}
        labeled = {label: local if source == "local" else remote for label, source in label_map.items()}
        judge_prompt = """請以原始逐字稿為依據，評估兩份會議整理結果。候選 A、B 使用匿名標籤，請分別獨立評分，不要因為文筆或長度偏好而加分。
評分維度均為 0 到 100：factual_accuracy（是否符合逐字稿、是否捏造）、coverage（重要內容是否涵蓋）、action_decision_accuracy（待辦與決議的內容、負責人、期限是否正確）。overall_accuracy 為三項平均值。錯誤、遺漏、無根據內容請以短字串陣列列出。逐字稿沒有提供的細節不能算正確。
只輸出 JSON，格式：{"A":{"factual_accuracy":0,"coverage":0,"action_decision_accuracy":0,"overall_accuracy":0,"issues":[]},"B":{"factual_accuracy":0,"coverage":0,"action_decision_accuracy":0,"overall_accuracy":0,"issues":[]},"winner":"A|B|tie","reason":"繁體中文簡短比較"}
winner 必須使用 A、B 或 tie。請勿推測候選模型來源。

原始逐字稿：
""" + transcript + "\n\n候選 A：\n" + json.dumps(labeled["A"], ensure_ascii=False) + "\n\n候選 B：\n" + json.dumps(labeled["B"], ensure_ascii=False)
        self.logger.emit("EVAL", "遠端評審比較兩份摘要", meeting_id=meeting_id)
        judged = _parse_json(self.remote.create("你是嚴謹、中立的會議摘要評審。只能根據提供的逐字稿給分。", judge_prompt, max_output_tokens=1800))
        def normalized_score(value):
            value = value if isinstance(value, dict) else {}
            dimensions = ("factual_accuracy", "coverage", "action_decision_accuracy")
            scores = {key: max(0, min(100, int(float(value.get(key, 0))))) for key in dimensions}
            try: overall = max(0, min(100, int(float(value.get("overall_accuracy")))))
            except (TypeError, ValueError): overall = round(sum(scores.values()) / len(dimensions))
            issues = value.get("issues", [])
            if isinstance(issues, str): issues = [issues]
            return {**scores, "overall_accuracy": overall, "issues": [str(issue) for issue in issues]}
        winner_label = str(judged.get("winner", "tie")).upper()
        winner = label_map.get(winner_label, "tie") if winner_label != "TIE" else "tie"
        scores_by_source = {source: normalized_score(judged.get(label)) for label, source in label_map.items()}
        return {"meeting_id": meeting_id, "created_at": datetime.now(timezone.utc).isoformat(), "model": self.remote.model, "local": local, "remote": remote, "scores": scores_by_source, "winner": winner, "reason": str(judged.get("reason", ""))}
