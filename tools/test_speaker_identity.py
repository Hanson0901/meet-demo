from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from meeting_assistant.asr import FunASRAdapter, _Recognition
from meeting_assistant.db import MeetingDB
from meeting_assistant.models import Utterance
from meeting_assistant.normalize import convert_traditional
from meeting_assistant.orchestrator import LLMOrchestrator
from meeting_assistant.speaker_identity import suggest_names_from_transcript


class QuietLogger:
    def emit(self, *_args, **_kwargs):
        pass


class FakeSpeakerModel:
    def eval(self):
        return self

    def inference(self, *, data_in, **_kwargs):
        return [
            {"spk_embedding": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)}
            for _ in data_in
        ], {}


class FakeASRModel:
    kwargs = {"device": "cpu"}
    spk_model = FakeSpeakerModel()


class SpeakerIdentityTests(unittest.TestCase):
    def test_opencc_uses_taiwan_traditional_conversion(self):
        self.assertEqual(convert_traditional("软件开发"), "軟體開發")

    def test_explicit_self_introduction_is_only_a_suggestion(self):
        result = suggest_names_from_transcript([
            {"speaker_id": "0", "start_time": 0, "normalized_text": "大家好，我叫王小明，很高興認識大家。"},
            {"speaker_id": "1", "start_time": 2, "normalized_text": "請問今天進度如何？"},
        ])
        self.assertEqual(result["0"]["name"], "王小明")
        self.assertEqual(result["0"]["source"], "self_introduction")
        self.assertNotIn("1", result)

    def test_local_model_name_suggestions_keep_id_and_evidence(self):
        orchestrator = LLMOrchestrator(QuietLogger())
        orchestrator._request_json = lambda prompt, meeting_id, stage, max_tokens: {
            "speakers": [{
                "speaker_id": "0", "name": "王小明", "evidence": "我叫王小明。", "confidence": 0.9,
            }]
        }
        suggestions = orchestrator.suggest_speaker_names("meeting", [
            {"speaker_id": "0", "start_time": 0, "end_time": 1, "normalized_text": "我叫王小明。"},
        ])
        self.assertEqual(suggestions[0]["speaker_id"], "0")
        self.assertEqual(suggestions[0]["name"], "王小明")
        self.assertEqual(suggestions[0]["source"], "local_llm")

    def test_cam_embedding_extraction_reuses_loaded_speaker_model(self):
        adapter = FunASRAdapter(QuietLogger())
        adapter.model = FakeASRModel()
        items = [_Recognition(0.0, 2.0, "測試", "0", 0.9)]
        with patch("funasr.utils.load_utils.load_audio_text_image_video", return_value=np.ones(32000, dtype=np.float32)):
            embeddings = adapter._extract_speaker_embeddings("sample.wav", items)
        self.assertEqual(embeddings["0"], [1.0, 0.0, 0.0])

    def test_confirmed_voiceprint_is_reused_and_deleted_when_orphaned(self):
        with tempfile.TemporaryDirectory() as folder:
            db = MeetingDB(str(Path(folder) / "meetings.sqlite3"))
            first_id = "meeting-one"
            db.create_meeting(first_id, "第一場", "2026-10-01T09:00:00+08:00")
            db.save_utterance(Utterance(first_id, "u1", "0", 0, 2, "您好", 0.9), "您好")
            first = db.match_speaker_embedding(first_id, "0", [1.0, 0.0, 0.0], threshold=0.74)
            self.assertFalse(first["matched"])
            db.update_speakers(first_id, {"0": "王小明"})
            self.assertEqual(db.counts()["speaker_profiles"], 1)

            second_id = "meeting-two"
            db.create_meeting(second_id, "第二場", "2026-10-02T09:00:00+08:00")
            db.save_utterance(Utterance(second_id, "u2", "1", 0, 2, "早安", 0.9), "早安")
            second = db.match_speaker_embedding(second_id, "1", [0.99, 0.01, 0.0], threshold=0.74)
            self.assertTrue(second["matched"])
            self.assertEqual(second["display_name"], "王小明")
            speaker = db.speakers(second_id)[0]
            self.assertEqual(speaker["name_source"], "voiceprint")

            third_id = "meeting-three"
            db.create_meeting(third_id, "第三場", "2026-10-03T09:00:00+08:00")
            db.save_utterance(Utterance(third_id, "u3", "2", 0, 2, "你好", 0.9), "你好")
            candidate = db.match_speaker_embedding(
                third_id, "2", [0.70, (1 - 0.70**2) ** 0.5, 0.0], threshold=0.74
            )
            self.assertFalse(candidate["matched"])
            self.assertEqual(candidate["candidate_name"], "王小明")
            db.update_speakers(third_id, {"2": "王小明"})
            self.assertEqual(db.counts()["speaker_profiles"], 1)

            db.delete_meeting(first_id)
            self.assertEqual(db.counts()["speaker_profiles"], 1)
            db.delete_meeting(second_id)
            self.assertEqual(db.counts()["speaker_profiles"], 1)
            db.delete_meeting(third_id)
            self.assertEqual(db.counts()["speaker_profiles"], 0)
            db.connection.close()


if __name__ == "__main__":
    unittest.main()
