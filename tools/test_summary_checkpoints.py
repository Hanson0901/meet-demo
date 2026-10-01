#!/usr/bin/env python3
"""Offline tests for resumable local-summary checkpoints and merge retries."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from meeting_assistant.db import MeetingDB
from meeting_assistant.orchestrator import LLMOrchestrator, OutputTokenLimitError


class NullLogger:
    def emit(self, *args, **kwargs):
        pass


TOKEN_SETTINGS = {
    "direct_max_tokens": 256,
    "chunk_tokens": 1800,
    "map_max_tokens": 3072,
    "reduce_input_tokens": 4096,
    "reduce_max_tokens": 4096,
    "final_max_tokens": 4096,
}


class FakeOrchestrator(LLMOrchestrator):
    def __init__(self, database: MeetingDB, *, fail_segment: int | None = None):
        super().__init__(
            NullLogger(),
            context_tokens=16384,
            token_settings=TOKEN_SETTINGS,
            checkpoint_store=database,
        )
        self.model_path = "offline-test-model.gguf"
        self.fail_segment = fail_segment
        self.requested_stages: list[str] = []

    def _load(self):
        return object()

    def _token_count(self, text: str) -> int:
        return max(1, len(text.encode("utf-8")) // 4)

    def _pack_transcript(self, rows, max_output_tokens, target_tokens=None):
        return ["segment one", "segment two", "segment three"]

    def _request_json(self, prompt, meeting_id, stage, max_tokens):
        self.requested_stages.append(stage)
        if stage.startswith("分段整理"):
            segment_number = int(stage.split()[1].split("/")[0])
            if segment_number == self.fail_segment:
                raise RuntimeError("simulated interruption")
            return {
                "topic_candidates": [],
                "facts": [{"claim": f"fact-{segment_number}", "evidence": f"quote-{segment_number}"}],
                "decisions": [],
                "actions": [],
                "meeting_time_mentions": [],
            }
        if stage == "最終摘要":
            return {
                "title": "測試主題",
                "summary": "- 測試摘要",
                "actions": [],
                "decisions": [],
                "follow_ups": [],
            }
        raise AssertionError(f"Unexpected fake model stage: {stage}")


class AdaptiveReducer(FakeOrchestrator):
    def __init__(self, database: MeetingDB, *, fail_right_once: bool = False):
        super().__init__(database)
        self.context_tokens = 8192
        self.fail_right_once = fail_right_once
        self.merge_prompts: list[str] = []

    def _token_count(self, text: str) -> int:
        if text.startswith("FINAL_LEDGER:"):
            notes = json.loads(text.removeprefix("FINAL_LEDGER:"))
            return 9000 if len(notes) >= 3 else 1000
        return max(1, len(text.encode("utf-8")) // 4)

    @classmethod
    def _final_ledger_prompt(cls, notes, related_context):
        return "FINAL_LEDGER:" + json.dumps(notes, ensure_ascii=False)

    def _request_json(self, prompt, meeting_id, stage, max_tokens):
        if not stage.startswith("分層合併"):
            return super()._request_json(prompt, meeting_id, stage, max_tokens)
        self.merge_prompts.append(prompt)
        notes = json.loads(prompt.split("分段紀錄：\n", 1)[1])
        if len(notes) >= 3:
            raise OutputTokenLimitError("simulated merge output truncation")
        if self.fail_right_once and stage.endswith("R"):
            self.fail_right_once = False
            raise RuntimeError("simulated interruption after left merge batch")

        merged = {
            "topic_candidates": [],
            "facts": [],
            "decisions": [],
            "actions": [],
            "meeting_time_mentions": [],
        }
        for note in notes:
            for field in merged:
                for item in note.get(field, []):
                    copied = dict(item) if isinstance(item, dict) else item
                    if field == "facts" and isinstance(copied, dict):
                        copied["evidence"] = str(copied.get("evidence", ""))[:12]
                    merged[field].append(copied)
        return merged


class SummaryCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "meetings.sqlite3"
        self.database = MeetingDB(str(self.db_path))

    def tearDown(self):
        self.database.connection.close()
        self.tempdir.cleanup()

    @staticmethod
    def rows():
        return [{
            "speaker_id": "speaker-1",
            "speaker_name": "測試說話人",
            "start_time": 0.0,
            "end_time": 60.0,
            "normalized_text": "測試逐字稿內容。" * 400,
        }]

    @staticmethod
    def oversized_notes():
        notes = []
        for index in range(1, 4):
            notes.append({
                "topic_candidates": [],
                "facts": [{"claim": f"fact-{index}", "evidence": f"長證據{index}" * 80}],
                "decisions": [],
                "actions": [],
                "meeting_time_mentions": [],
                "source_segments": [index],
            })
        return notes

    def test_map_checkpoints_resume_after_interruption(self):
        meeting_id = "resume-test"
        first_run = FakeOrchestrator(self.database, fail_segment=3)
        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            first_run.analyze(meeting_id, self.rows())

        completed_map_count = self.database.connection.execute(
            "SELECT COUNT(*) FROM analysis_checkpoints WHERE meeting_id=? AND phase='map'",
            (meeting_id,),
        ).fetchone()[0]
        self.assertEqual(completed_map_count, 2)

        resumed = FakeOrchestrator(self.database)
        result = resumed.analyze(meeting_id, self.rows())
        resumed_map_stages = [
            stage for stage in resumed.requested_stages if stage.startswith("分段整理")
        ]
        self.assertEqual(resumed_map_stages, ["分段整理 3/3"])
        self.assertEqual(result["title"], "測試主題")

    def test_merge_overflow_splits_batch_and_resume_reuses_completed_child(self):
        meeting_id = "merge-resume-test"
        notes = self.oversized_notes()
        interrupted = AdaptiveReducer(self.database, fail_right_once=True)
        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            interrupted._reduce_to_final_budget(
                notes, meeting_id, [], 512, 4096, 4096, "stable-run-key"
            )

        left_prompt = interrupted._reduce_prompt(json.dumps([notes[0]], ensure_ascii=False))
        saved_left = self.database.connection.execute(
            "SELECT COUNT(*) FROM analysis_checkpoints "
            "WHERE meeting_id=? AND phase='reduce' AND item_key='0001L'",
            (meeting_id,),
        ).fetchone()[0]
        self.assertEqual(saved_left, 1)

        resumed = AdaptiveReducer(self.database)
        reduced, final_prompt = resumed._reduce_to_final_budget(
            notes, meeting_id, [], 512, 4096, 4096, "stable-run-key"
        )
        self.assertNotIn(left_prompt, resumed.merge_prompts)
        self.assertEqual(len(reduced), 2)
        self.assertTrue(final_prompt.startswith("FINAL_LEDGER:"))

    def test_checkpoint_and_admin_settings_survive_database_reopen(self):
        self.database.set_settings({"llama_cpp_chunk_tokens": "2048"})
        self.database.save_analysis_checkpoint(
            "persist-test", "run", "map", 0, "000001", "input-hash", {"facts": []}
        )
        self.database.connection.close()
        self.database = MeetingDB(str(self.db_path))

        checkpoint = self.database.get_analysis_checkpoint(
            "persist-test", "run", "map", 0, "000001", "input-hash"
        )
        self.assertEqual(checkpoint, {"facts": []})
        self.assertEqual(self.database.get_setting("llama_cpp_chunk_tokens"), "2048")

    def test_deleting_meeting_removes_its_checkpoints_and_keeps_app_settings(self):
        self.database.create_meeting("delete-test", "測試", "2026-10-01T00:00:00+00:00")
        self.database.set_setting("llama_cpp_chunk_tokens", "2048")
        self.database.save_analysis_checkpoint(
            "delete-test", "run", "map", 0, "000001", "input-hash", {"facts": []}
        )

        self.assertTrue(self.database.delete_meeting("delete-test"))
        checkpoint_count = self.database.connection.execute(
            "SELECT COUNT(*) FROM analysis_checkpoints WHERE meeting_id=?", ("delete-test",)
        ).fetchone()[0]
        self.assertEqual(checkpoint_count, 0)
        self.assertEqual(self.database.get_setting("llama_cpp_chunk_tokens"), "2048")


if __name__ == "__main__":
    unittest.main()
