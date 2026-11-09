#!/usr/bin/env python3
"""Offline tests for resumable local-summary checkpoints and merge retries."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from meeting_assistant.db import MeetingDB
from meeting_assistant.orchestrator import (
    LLMOrchestrator,
    MalformedModelJSONError,
    OutputTokenLimitError,
)


class NullLogger:
    def emit(self, *args, **kwargs):
        pass


TOKEN_SETTINGS = {
    "direct_max_tokens": 256,
    "chunk_tokens": 1800,
    "map_max_tokens": 3072,
    "verify_max_tokens": 768,
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


class VerifyRetryOrchestrator(LLMOrchestrator):
    def __init__(
        self,
        database: MeetingDB,
        *,
        always_truncate: bool = False,
        malformed_first: bool = False,
    ):
        super().__init__(
            NullLogger(),
            context_tokens=16384,
            token_settings=TOKEN_SETTINGS,
            checkpoint_store=database,
        )
        self.model_path = "offline-test-model.gguf"
        self.always_truncate = always_truncate
        self.malformed_first = malformed_first
        self.requested_stages: list[tuple[str, int]] = []

    def _load(self):
        return object()

    def _token_count(self, text: str) -> int:
        return max(1, len(text.encode("utf-8")) // 4)

    def _request_json(self, prompt, meeting_id, stage, max_tokens):
        self.requested_stages.append((stage, max_tokens))
        if stage.startswith("重要段落核對"):
            if "精簡重試" not in stage:
                if self.malformed_first:
                    raise MalformedModelJSONError("simulated malformed verifier JSON")
                raise OutputTokenLimitError("simulated verifier truncation")
            if self.always_truncate:
                raise OutputTokenLimitError("simulated verifier truncation")
        if stage.startswith("重要段落核對"):
            return {
                "gist": "確認重要資訊",
                "topic_candidates": [],
                "named_entities": [],
                "key_points": ["確認後的重要重點"],
                "decisions": [],
                "actions": [],
                "open_questions": [],
                "risks": [],
                "meeting_time_mentions": [],
            }
        raise AssertionError(f"Unexpected fake model stage: {stage}")


class VerifyCheckpointOrchestrator(FakeOrchestrator):
    def _request_json(self, prompt, meeting_id, stage, max_tokens):
        if stage.startswith("重要段落核對"):
            self.requested_stages.append(stage)
            return {
                "gist": "核對後的重點",
                "topic_candidates": [],
                "named_entities": [],
                "key_points": ["核對後的重要資訊"],
                "decisions": [],
                "actions": [],
                "open_questions": [],
                "risks": [],
                "meeting_time_mentions": [],
            }
        result = super()._request_json(prompt, meeting_id, stage, max_tokens)
        if stage.startswith("分段整理"):
            result["actions"] = [{"item": "保存這個待辦", "owner": "", "due_date": ""}]
        return result


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

    def test_verify_checkpoints_resume_with_map_annotations(self):
        meeting_id = "verify-resume-test"
        first_run = VerifyCheckpointOrchestrator(self.database, fail_segment=3)
        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            first_run.analyze(meeting_id, self.rows())

        resumed = VerifyCheckpointOrchestrator(self.database)
        resumed.analyze(meeting_id, self.rows())
        self.assertEqual(
            resumed.requested_stages,
            ["分段整理 3/3", "重要段落核對 3/3", "最終摘要"],
        )

    def test_compact_notes_drop_evidence_and_detect_high_signal(self):
        note = LLMOrchestrator._clean_note(
            {
                "gist": "活動時程與預算安排",
                "facts": [
                    {
                        "claim": "距離活動剩下約 2 個月",
                        "evidence": "距離活動，剩下約兩個月。",
                        "speaker": "speaker-1",
                    }
                ],
                "actions": [{"item": "下週提交預算", "owner": "王小明", "due_date": "下週"}],
            },
            [4],
            annotate_items=True,
        )

        self.assertNotIn("facts", note)
        self.assertEqual(note["key_points"], ["距離活動剩下約 2 個月"])
        self.assertEqual(note["actions"][0]["source_segments"], [4])
        self.assertTrue(LLMOrchestrator._needs_verification(note))
        self.assertFalse(LLMOrchestrator._needs_verification({"key_points": ["一般背景說明"]}))

    def test_parser_repairs_common_local_json_formatting_and_removes_emoji(self):
        result = LLMOrchestrator._parse_object(
            "模型說明\n```json\n"
            "{'gist': '重點✅', 'key_points': ['確認日期📌',], 'actions': [],}\n"
            "```"
        )

        self.assertEqual(result, {
            "gist": "重點",
            "key_points": ["確認日期"],
            "actions": [],
        })

    def test_verifier_retries_with_compact_schema_after_token_limit(self):
        orchestrator = VerifyRetryOrchestrator(self.database)
        result, completed = orchestrator._verify_with_retry(
            3,
            4,
            "本段討論重要數字與待辦。",
            {
                "gist": "討論重要數字",
                "key_points": ["重要數字需要核對"],
                "actions": [],
            },
            "verify-retry-test",
            768,
        )

        self.assertTrue(completed)
        self.assertEqual(result["gist"], "確認重要資訊")
        self.assertEqual(
            orchestrator.requested_stages,
            [
                ("重要段落核對 3/4", 768),
                ("重要段落核對 3/4（精簡重試）", 1536),
            ],
        )

    def test_verifier_falls_back_to_map_note_after_two_truncations(self):
        orchestrator = VerifyRetryOrchestrator(self.database, always_truncate=True)
        result, completed = orchestrator._verify_with_retry(
            3,
            4,
            "本段討論重要數字與待辦。",
            {"gist": "保留分段結果", "key_points": ["重要重點"]},
            "verify-fallback-test",
            768,
        )

        self.assertFalse(completed)
        self.assertIsNone(result)
        self.assertEqual(len(orchestrator.requested_stages), 2)

    def test_verifier_retries_after_malformed_json(self):
        orchestrator = VerifyRetryOrchestrator(self.database, malformed_first=True)
        result, completed = orchestrator._verify_with_retry(
            3,
            4,
            "本段討論重要數字與待辦。",
            {"gist": "保留分段結果", "key_points": ["重要重點"]},
            "verify-malformed-test",
            1536,
        )

        self.assertTrue(completed)
        self.assertEqual(result["gist"], "確認重要資訊")
        self.assertEqual(orchestrator.requested_stages[1][0], "重要段落核對 3/4（精簡重試）")

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
