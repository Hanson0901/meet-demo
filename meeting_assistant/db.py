from __future__ import annotations
import json, sqlite3, uuid
from datetime import datetime
from threading import RLock
from pathlib import Path
from .models import Utterance
from .rag import embed, similarity, transcript_chunks
from .speaker_identity import cosine_similarity, normalize_embedding

SCHEMA = """
    CREATE TABLE IF NOT EXISTS meetings (id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL, meeting_time TEXT DEFAULT '', summary TEXT DEFAULT '', transcription_status TEXT NOT NULL DEFAULT 'not_started', transcription_stage TEXT DEFAULT '', transcription_error TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS speakers (meeting_id TEXT NOT NULL, speaker_id TEXT NOT NULL, display_name TEXT, PRIMARY KEY (meeting_id, speaker_id));
CREATE TABLE IF NOT EXISTS speaker_profiles (profile_id TEXT PRIMARY KEY, display_name TEXT NOT NULL, embedding_json TEXT NOT NULL, embedding_model TEXT NOT NULL, sample_count INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS utterances (meeting_id TEXT NOT NULL, utterance_id TEXT PRIMARY KEY, speaker_id TEXT NOT NULL, start_time REAL NOT NULL, end_time REAL NOT NULL, text TEXT NOT NULL, normalized_text TEXT NOT NULL, confidence REAL NOT NULL);
CREATE TABLE IF NOT EXISTS actions (id INTEGER PRIMARY KEY AUTOINCREMENT, meeting_id TEXT NOT NULL, item TEXT NOT NULL, owner TEXT, due_date TEXT);
CREATE TABLE IF NOT EXISTS decisions (id INTEGER PRIMARY KEY AUTOINCREMENT, meeting_id TEXT NOT NULL, decision TEXT NOT NULL, context TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS logs (id INTEGER PRIMARY KEY AUTOINCREMENT, meeting_id TEXT, timestamp TEXT NOT NULL, module TEXT NOT NULL, level TEXT NOT NULL, message TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evaluations (id INTEGER PRIMARY KEY AUTOINCREMENT, meeting_id TEXT NOT NULL, created_at TEXT NOT NULL, result_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS threads (thread_id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS topics (topic_key TEXT PRIMARY KEY, display_name TEXT NOT NULL, thread_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS meeting_topics (meeting_id TEXT PRIMARY KEY, topic_key TEXT NOT NULL, thread_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS rag_content (id INTEGER PRIMARY KEY AUTOINCREMENT, topic_key TEXT NOT NULL, meeting_id TEXT NOT NULL, chunk_index INTEGER NOT NULL, content TEXT NOT NULL, vector_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS rag_summaries (id INTEGER PRIMARY KEY AUTOINCREMENT, topic_key TEXT NOT NULL, meeting_id TEXT NOT NULL, content TEXT NOT NULL, vector_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS rag_actions (id INTEGER PRIMARY KEY AUTOINCREMENT, topic_key TEXT NOT NULL, meeting_id TEXT NOT NULL, action_id INTEGER, content TEXT NOT NULL, vector_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS follow_ups (id INTEGER PRIMARY KEY AUTOINCREMENT, meeting_id TEXT NOT NULL, previous_meeting_id TEXT NOT NULL, previous_action_id INTEGER, item TEXT NOT NULL, status TEXT NOT NULL, conclusion TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS analysis_checkpoints (meeting_id TEXT NOT NULL, run_key TEXT NOT NULL, phase TEXT NOT NULL, round_number INTEGER NOT NULL DEFAULT 0, item_key TEXT NOT NULL, input_hash TEXT NOT NULL, result_json TEXT NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (meeting_id, run_key, phase, round_number, item_key));
CREATE INDEX IF NOT EXISTS idx_analysis_checkpoints_meeting ON analysis_checkpoints(meeting_id);
"""

class MeetingDB:
    def __init__(self, path: str = "data/meetings.sqlite3"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.lock = RLock()
        self.connection.executescript(SCHEMA)
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(meetings)")}
        if "meeting_time" not in columns:
            self.connection.execute("ALTER TABLE meetings ADD COLUMN meeting_time TEXT DEFAULT ''")
            self.connection.execute("UPDATE meetings SET meeting_time=created_at WHERE meeting_time IS NULL OR meeting_time=''")
        meeting_migrations = {
            "transcription_status": "TEXT NOT NULL DEFAULT 'not_started'",
            "transcription_stage": "TEXT DEFAULT ''",
            "transcription_error": "TEXT DEFAULT ''",
        }
        for column, definition in meeting_migrations.items():
            if column not in columns:
                self.connection.execute(f"ALTER TABLE meetings ADD COLUMN {column} {definition}")
        self.connection.execute(
            "UPDATE meetings SET transcription_status='completed', transcription_stage='逐字稿已建立' "
            "WHERE transcription_status='not_started' AND EXISTS "
            "(SELECT 1 FROM utterances WHERE utterances.meeting_id=meetings.id)"
        )
        speaker_columns = {row[1] for row in self.connection.execute("PRAGMA table_info(speakers)")}
        speaker_migrations = {
            "embedding_json": "TEXT",
            "embedding_model": "TEXT NOT NULL DEFAULT ''",
            "profile_id": "TEXT",
            "candidate_profile_id": "TEXT",
            "profile_enrolled": "INTEGER NOT NULL DEFAULT 0",
            "match_score": "REAL",
            "name_source": "TEXT NOT NULL DEFAULT ''",
            "suggested_name": "TEXT NOT NULL DEFAULT ''",
            "suggestion_source": "TEXT NOT NULL DEFAULT ''",
            "suggestion_evidence": "TEXT NOT NULL DEFAULT ''",
            "suggestion_confidence": "REAL",
        }
        for column, definition in speaker_migrations.items():
            if column not in speaker_columns:
                self.connection.execute(f"ALTER TABLE speakers ADD COLUMN {column} {definition}")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_speakers_profile_id ON speakers(profile_id)")
        self.connection.commit()
        self.backfill_rag_indexes()

    def create_meeting(self, meeting_id: str, title: str, created_at: str) -> None:
        self.connection.execute("INSERT INTO meetings(id,title,created_at,meeting_time) VALUES(?,?,?,?)", (meeting_id, title, created_at, created_at))
        self.connection.commit()

    def set_transcription_status(
        self,
        meeting_id: str,
        status: str,
        stage: str = "",
        error: str | None = None,
    ) -> None:
        with self.lock:
            self.connection.execute(
                "UPDATE meetings SET transcription_status=?,transcription_stage=?,transcription_error=? WHERE id=?",
                (status, stage, error or "", meeting_id),
            )
            self.connection.commit()

    def update_meeting_metadata(self, meeting_id: str, *, title: str, meeting_time: str) -> None:
        self.connection.execute("UPDATE meetings SET title=?,meeting_time=? WHERE id=?", (title, meeting_time, meeting_id))
        self.connection.commit()

    def save_utterance(self, item: Utterance, normalized_text: str) -> None:
        self.connection.execute("INSERT OR REPLACE INTO utterances VALUES(?,?,?,?,?,?,?,?)", (item.meeting_id,item.utterance_id,item.speaker_id,item.start_time,item.end_time,item.text,normalized_text,item.confidence))
        self.connection.execute("INSERT OR IGNORE INTO speakers(meeting_id,speaker_id) VALUES(?,?)", (item.meeting_id,item.speaker_id))
        self.connection.commit()

    def save_analysis(self, meeting_id: str, summary: str, actions: list[dict], decisions: list[dict], *, title: str | None = None, meeting_time: str | None = None, follow_ups: list[dict] | None = None) -> None:
        if title is not None or meeting_time is not None:
            current = self.meeting(meeting_id) or {}
            self.connection.execute("UPDATE meetings SET title=?,meeting_time=?,summary=? WHERE id=?", (title or current.get("title", ""), meeting_time or current.get("meeting_time", current.get("created_at", "")), summary, meeting_id))
        else:
            self.connection.execute("UPDATE meetings SET summary=? WHERE id=?", (summary, meeting_id))
        self.connection.execute("DELETE FROM actions WHERE meeting_id=?", (meeting_id,)); self.connection.execute("DELETE FROM decisions WHERE meeting_id=?", (meeting_id,))
        self.connection.execute("DELETE FROM follow_ups WHERE meeting_id=?", (meeting_id,))
        self.connection.executemany("INSERT INTO actions(meeting_id,item,owner,due_date) VALUES(?,?,?,?)", [(meeting_id,a.get("item",""),a.get("owner"),a.get("due_date")) for a in actions])
        self.connection.executemany("INSERT INTO decisions(meeting_id,decision,context) VALUES(?,?,?)", [(meeting_id,d.get("decision",""),d.get("context","")) for d in decisions]); self.connection.commit()
        for item in follow_ups or []:
            self.connection.execute("INSERT INTO follow_ups(meeting_id,previous_meeting_id,previous_action_id,item,status,conclusion) VALUES(?,?,?,?,?,?)", (meeting_id,item.get("previous_meeting_id", ""),item.get("previous_action_id"),item.get("item", ""),item.get("status", "unknown"),item.get("conclusion", "")))
        self.connection.commit()

    def add_log(self, event) -> None:
        self.connection.execute("INSERT INTO logs(meeting_id,timestamp,module,level,message) VALUES(?,?,?,?,?)", (event.meeting_id,event.timestamp,event.module,event.level,event.message)); self.connection.commit()

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        with self.lock:
            row = self.connection.execute("SELECT value FROM app_settings WHERE key=?", (key,)).fetchone()
            return row[0] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self.lock:
            self.connection.execute(
                "INSERT INTO app_settings(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            self.connection.commit()

    def set_settings(self, values: dict[str, str]) -> None:
        if not values:
            return
        with self.lock:
            self.connection.executemany(
                "INSERT INTO app_settings(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                list(values.items()),
            )
            self.connection.commit()

    def get_analysis_checkpoint(
        self,
        meeting_id: str,
        run_key: str,
        phase: str,
        round_number: int,
        item_key: str,
        input_hash: str,
    ) -> dict | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT result_json FROM analysis_checkpoints "
                "WHERE meeting_id=? AND run_key=? AND phase=? AND round_number=? AND item_key=? AND input_hash=?",
                (meeting_id, run_key, phase, round_number, item_key, input_hash),
            ).fetchone()
        if row is None:
            return None
        try:
            result = json.loads(row["result_json"])
        except (TypeError, json.JSONDecodeError):
            return None
        return result if isinstance(result, dict) else None

    def save_analysis_checkpoint(
        self,
        meeting_id: str,
        run_key: str,
        phase: str,
        round_number: int,
        item_key: str,
        input_hash: str,
        result: dict,
    ) -> None:
        with self.lock:
            self.connection.execute(
                "INSERT INTO analysis_checkpoints(meeting_id,run_key,phase,round_number,item_key,input_hash,result_json) "
                "VALUES(?,?,?,?,?,?,?) ON CONFLICT(meeting_id,run_key,phase,round_number,item_key) "
                "DO UPDATE SET input_hash=excluded.input_hash,result_json=excluded.result_json,updated_at=CURRENT_TIMESTAMP",
                (
                    meeting_id,
                    run_key,
                    phase,
                    round_number,
                    item_key,
                    input_hash,
                    json.dumps(result, ensure_ascii=False, separators=(",", ":")),
                ),
            )
            self.connection.commit()

    def retain_analysis_checkpoint_run(self, meeting_id: str, run_key: str) -> None:
        with self.lock:
            self.connection.execute(
                "DELETE FROM analysis_checkpoints WHERE meeting_id=? AND run_key<>?",
                (meeting_id, run_key),
            )
            self.connection.commit()

    def delete_analysis_checkpoints(self, meeting_id: str, run_key: str | None = None) -> None:
        with self.lock:
            if run_key is None:
                self.connection.execute(
                    "DELETE FROM analysis_checkpoints WHERE meeting_id=?", (meeting_id,)
                )
            else:
                self.connection.execute(
                    "DELETE FROM analysis_checkpoints WHERE meeting_id=? AND run_key=?",
                    (meeting_id, run_key),
                )
            self.connection.commit()

    def transcript(self, meeting_id: str) -> list[dict]:
        return [dict(row) for row in self.connection.execute("SELECT u.*, COALESCE(NULLIF(s.display_name,''), u.speaker_id) AS speaker_name FROM utterances u LEFT JOIN speakers s ON s.meeting_id=u.meeting_id AND s.speaker_id=u.speaker_id WHERE u.meeting_id=? ORDER BY u.start_time", (meeting_id,))]

    def meeting(self, meeting_id: str) -> dict | None:
        row = self.connection.execute("SELECT * FROM meetings WHERE id=?", (meeting_id,)).fetchone(); return dict(row) if row else None

    def analysis(self, meeting_id: str) -> dict | None:
        meeting = self.meeting(meeting_id)
        if meeting is None: return None
        actions = [dict(row) for row in self.connection.execute("SELECT item,owner,due_date FROM actions WHERE meeting_id=? ORDER BY id", (meeting_id,))]
        decisions = [dict(row) for row in self.connection.execute("SELECT decision,context FROM decisions WHERE meeting_id=? ORDER BY id", (meeting_id,))]
        follow_ups = self.follow_ups(meeting_id)
        return {"title": meeting.get("title", ""), "meeting_time": meeting.get("meeting_time", meeting.get("created_at", "")), "summary": meeting.get("summary", ""), "actions": actions, "decisions": decisions, "follow_ups": follow_ups}

    def search(self, query: str) -> list[dict]:
        needle = f"%{query}%"; return [dict(r) for r in self.connection.execute("SELECT u.*, COALESCE(NULLIF(s.display_name,''), u.speaker_id) AS speaker_name FROM utterances u LEFT JOIN speakers s ON s.meeting_id=u.meeting_id AND s.speaker_id=u.speaker_id WHERE u.normalized_text LIKE ? OR s.display_name LIKE ? ORDER BY u.start_time", (needle, needle))]

    def speakers(self, meeting_id: str) -> list[dict]:
        return [dict(row) for row in self.connection.execute(
            "SELECT meeting_id,speaker_id,COALESCE(display_name,'') AS display_name,"
            "profile_id,match_score,name_source,suggested_name,suggestion_source,suggestion_evidence,suggestion_confidence,"
            "CASE WHEN embedding_json IS NOT NULL AND embedding_json<>'' THEN 1 ELSE 0 END AS has_voiceprint "
            "FROM speakers WHERE meeting_id=? ORDER BY speaker_id", (meeting_id,)
        )]

    @staticmethod
    def _timestamp_now() -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")

    def match_speaker_embedding(
        self,
        meeting_id: str,
        speaker_id: str,
        embedding: list[float],
        *,
        embedding_model: str = "cam++",
        threshold: float = 0.74,
        ambiguity_margin: float = 0.06,
    ) -> dict:
        """Persist a meeting vector and apply a previously confirmed identity if unambiguous."""
        vector = normalize_embedding(embedding)
        if vector is None:
            return {"matched": False, "score": None, "display_name": ""}
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("聲紋比對門檻必須介於 0 與 1")
        vector_json = json.dumps(vector, separators=(",", ":"))
        with self.lock:
            self.connection.execute(
                "INSERT OR IGNORE INTO speakers(meeting_id,speaker_id) VALUES(?,?)",
                (meeting_id, speaker_id),
            )
            profiles = self.connection.execute(
                "SELECT profile_id,display_name,embedding_json,sample_count FROM speaker_profiles "
                "WHERE embedding_model=?", (embedding_model,),
            ).fetchall()
            scored = []
            for profile in profiles:
                try:
                    profile_vector = json.loads(profile["embedding_json"])
                except (TypeError, json.JSONDecodeError):
                    continue
                score = cosine_similarity(vector, profile_vector)
                if score >= 0:
                    scored.append((score, profile))
            scored.sort(key=lambda item: item[0], reverse=True)
            best_score, best_profile = scored[0] if scored else (None, None)
            second_score = scored[1][0] if len(scored) > 1 else -1.0
            unambiguous = best_profile is not None and best_score - second_score >= ambiguity_margin
            matched = bool(best_profile is not None and best_score >= threshold and unambiguous)
            current = self.connection.execute(
                "SELECT display_name,name_source FROM speakers WHERE meeting_id=? AND speaker_id=?",
                (meeting_id, speaker_id),
            ).fetchone()
            current_name = (current["display_name"] or "").strip() if current else ""
            candidate_name = ""
            suggestion_source = ""
            suggestion_evidence = ""
            profile_id = None
            candidate_profile_id = None
            display_name = current_name
            name_source = current["name_source"] or "" if current else ""
            if matched:
                profile_id = best_profile["profile_id"]
                if not current_name:
                    display_name = best_profile["display_name"]
                    name_source = "voiceprint"
            elif best_profile is not None and unambiguous and best_score >= threshold - 0.10:
                candidate_profile_id = best_profile["profile_id"]
                candidate_name = best_profile["display_name"]
                suggestion_source = "voiceprint_candidate"
                suggestion_evidence = f"相似度 {best_score:.0%}，尚未達自動套用門檻 {threshold:.0%}"

            self.connection.execute(
                "UPDATE speakers SET embedding_json=?,embedding_model=?,profile_id=?,candidate_profile_id=?,match_score=?,"
                "display_name=?,name_source=?,suggested_name=?,suggestion_source=?,suggestion_evidence=? "
                "WHERE meeting_id=? AND speaker_id=?",
                (
                    vector_json, embedding_model, profile_id, candidate_profile_id, best_score, display_name or None,
                    name_source, candidate_name, suggestion_source, suggestion_evidence,
                    meeting_id, speaker_id,
                ),
            )
            self.connection.commit()
            return {
                "matched": matched,
                "display_name": display_name,
                "score": best_score,
                "candidate_name": candidate_name,
                "profile_id": profile_id,
            }

    def save_speaker_suggestions(self, meeting_id: str, suggestions: dict[str, dict]) -> None:
        with self.lock:
            for speaker_id, suggestion in suggestions.items():
                if not isinstance(suggestion, dict):
                    continue
                name = str(suggestion.get("name", "")).strip()[:80]
                if not name:
                    continue
                self.connection.execute(
                    "UPDATE speakers SET suggested_name=?,suggestion_source=?,suggestion_evidence=?,suggestion_confidence=? "
                    "WHERE meeting_id=? AND speaker_id=? AND COALESCE(display_name,'')=''",
                    (
                        name, str(suggestion.get("source", "text_context"))[:40],
                        str(suggestion.get("evidence", ""))[:500],
                        float(suggestion.get("confidence", 0.0)), meeting_id, speaker_id,
                    ),
                )
            self.connection.commit()

    def update_speakers(self, meeting_id: str, mapping: dict[str, str]) -> None:
        if not isinstance(mapping, dict):
            raise ValueError("speakers 必須是物件")
        cleaned_mapping = {}
        for speaker_id, display_name in mapping.items():
            if not isinstance(display_name, str):
                raise ValueError("說話者姓名必須是文字")
            cleaned_mapping[str(speaker_id)] = display_name.strip()[:80]
        with self.lock:
            for speaker_id, display_name in cleaned_mapping.items():
                row = self.connection.execute(
                    "SELECT embedding_json,embedding_model,profile_id,candidate_profile_id,profile_enrolled FROM speakers "
                    "WHERE meeting_id=? AND speaker_id=?", (meeting_id, speaker_id),
                ).fetchone()
                if row is None:
                    continue

                profile_id = row["profile_id"] or row["candidate_profile_id"]
                enrolled = int(row["profile_enrolled"] or 0)
                vector = None
                if row["embedding_json"]:
                    try:
                        vector = normalize_embedding(json.loads(row["embedding_json"]))
                    except (TypeError, json.JSONDecodeError):
                        vector = None

                if display_name and vector is not None:
                    profile = self.connection.execute(
                        "SELECT * FROM speaker_profiles WHERE profile_id=?", (profile_id,)
                    ).fetchone() if profile_id else None
                    if profile and display_name == profile["display_name"]:
                        if not enrolled:
                            try:
                                old_vector = normalize_embedding(json.loads(profile["embedding_json"]))
                            except (TypeError, json.JSONDecodeError):
                                old_vector = None
                            if old_vector is not None and len(old_vector) == len(vector):
                                count = max(1, int(profile["sample_count"] or 1))
                                combined = normalize_embedding(
                                    [old * count + new for old, new in zip(old_vector, vector)]
                                ) or vector
                                self.connection.execute(
                                    "UPDATE speaker_profiles SET embedding_json=?,sample_count=?,updated_at=? WHERE profile_id=?",
                                    (json.dumps(combined, separators=(",", ":")), count + 1, self._timestamp_now(), profile_id),
                                )
                            self.connection.execute(
                                "UPDATE speakers SET profile_enrolled=1 WHERE meeting_id=? AND speaker_id=?",
                                (meeting_id, speaker_id),
                            )
                    else:
                        # A user correcting an automatic match creates a separate
                        # identity instead of renaming the previously confirmed profile.
                        profile_id = str(uuid.uuid4())
                        now = self._timestamp_now()
                        self.connection.execute(
                            "INSERT INTO speaker_profiles(profile_id,display_name,embedding_json,embedding_model,sample_count,created_at,updated_at) "
                            "VALUES(?,?,?,?,1,?,?)",
                            (profile_id, display_name, json.dumps(vector, separators=(",", ":")), row["embedding_model"] or "cam++", now, now),
                        )
                        self.connection.execute(
                            "UPDATE speakers SET profile_enrolled=1 WHERE meeting_id=? AND speaker_id=?",
                            (meeting_id, speaker_id),
                        )
                elif display_name and profile_id:
                    profile = self.connection.execute(
                        "SELECT display_name FROM speaker_profiles WHERE profile_id=?", (profile_id,)
                    ).fetchone()
                    if profile and profile["display_name"] != display_name:
                        profile_id = None

                self.connection.execute(
                    "UPDATE speakers SET display_name=?,profile_id=?,candidate_profile_id=NULL,name_source=?,suggested_name='',"
                    "suggestion_source='',suggestion_evidence='',suggestion_confidence=NULL WHERE meeting_id=? AND speaker_id=?",
                    (display_name or None, profile_id, "confirmed" if display_name else "", meeting_id, speaker_id),
                )
            self.connection.execute(
                "DELETE FROM speaker_profiles WHERE NOT EXISTS "
                "(SELECT 1 FROM speakers WHERE speakers.profile_id=speaker_profiles.profile_id)"
            )
            self.connection.commit()

    def export(self, meeting_id: str) -> dict:
        topic = self.connection.execute("SELECT t.topic_key,t.display_name,t.thread_id,th.name AS thread_name FROM meeting_topics mt JOIN topics t ON t.topic_key=mt.topic_key JOIN threads th ON th.thread_id=mt.thread_id WHERE mt.meeting_id=?", (meeting_id,)).fetchone()
        return {"meeting": self.meeting(meeting_id), "topic": dict(topic) if topic else None, "related_meetings": self.thread_meetings(meeting_id), "speakers": self.speakers(meeting_id), "transcript": self.transcript(meeting_id), "actions": [dict(r) for r in self.connection.execute("SELECT * FROM actions WHERE meeting_id=?",(meeting_id,))], "decisions": [dict(r) for r in self.connection.execute("SELECT * FROM decisions WHERE meeting_id=?",(meeting_id,))], "follow_ups": self.follow_ups(meeting_id)}

    def meetings(self) -> list[dict]:
        return [dict(row) for row in self.connection.execute("SELECT m.*,mt.thread_id,th.name AS thread_name FROM meetings m LEFT JOIN meeting_topics mt ON mt.meeting_id=m.id LEFT JOIN threads th ON th.thread_id=mt.thread_id ORDER BY m.created_at DESC")]

    def logs(self, meeting_id: str | None = None, limit: int | None = None) -> list[dict]:
        if meeting_id:
            if limit is None:
                rows = self.connection.execute("SELECT * FROM logs WHERE meeting_id=? ORDER BY id", (meeting_id,))
            else:
                rows = self.connection.execute(
                    "SELECT * FROM logs WHERE meeting_id=? ORDER BY id DESC LIMIT ?",
                    (meeting_id, max(1, int(limit))),
                )
                return [dict(row) for row in reversed(rows.fetchall())]
        elif limit is None:
            rows = self.connection.execute("SELECT * FROM logs ORDER BY id DESC LIMIT 200")
        else:
            rows = self.connection.execute("SELECT * FROM logs ORDER BY id DESC LIMIT ?", (max(1, int(limit)),))
        return [dict(row) for row in rows]

    def delete_meeting(self, meeting_id: str) -> bool:
        with self.lock:
            exists = self.connection.execute("SELECT 1 FROM meetings WHERE id=?", (meeting_id,)).fetchone()
            if not exists:
                return False
            for table in ("rag_content", "rag_summaries", "rag_actions", "analysis_checkpoints", "speakers", "utterances", "actions", "decisions", "logs", "evaluations"):
                self.connection.execute(f"DELETE FROM {table} WHERE meeting_id=?", (meeting_id,))
            self.connection.execute(
                "DELETE FROM speaker_profiles WHERE NOT EXISTS "
                "(SELECT 1 FROM speakers WHERE speakers.profile_id=speaker_profiles.profile_id)"
            )
            self.connection.execute("DELETE FROM follow_ups WHERE meeting_id=? OR previous_meeting_id=?", (meeting_id, meeting_id))
            self.connection.execute("DELETE FROM meeting_topics WHERE meeting_id=?", (meeting_id,))
            self.connection.execute("DELETE FROM meetings WHERE id=?", (meeting_id,))
            self.connection.execute("DELETE FROM topics WHERE NOT EXISTS (SELECT 1 FROM meeting_topics mt WHERE mt.topic_key=topics.topic_key)")
            self.connection.execute("DELETE FROM threads WHERE NOT EXISTS (SELECT 1 FROM meeting_topics mt WHERE mt.thread_id=threads.thread_id)")
            self.connection.commit()
            return True

    def purge(self) -> int:
        with self.lock:
            count = self.connection.execute("SELECT COUNT(*) FROM meetings").fetchone()[0]
            for table in ("rag_content", "rag_summaries", "rag_actions", "analysis_checkpoints", "follow_ups", "meeting_topics", "topics", "threads", "speakers", "speaker_profiles", "utterances", "actions", "decisions", "logs", "evaluations", "meetings"):
                self.connection.execute(f"DELETE FROM {table}")
            self.connection.commit()
            return count

    def counts(self) -> dict[str, int]:
        return {table: self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in ("meetings", "speakers", "speaker_profiles", "utterances", "actions", "decisions", "follow_ups", "logs", "evaluations", "threads", "rag_content", "rag_summaries", "rag_actions", "analysis_checkpoints")}

    def save_evaluation(self, result: dict) -> int:
        cursor = self.connection.execute("INSERT INTO evaluations(meeting_id,created_at,result_json) VALUES(?,?,?)", (result["meeting_id"], result["created_at"], json.dumps(result, ensure_ascii=False)))
        self.connection.commit()
        return cursor.lastrowid

    def evaluations(self, meeting_id: str) -> list[dict]:
        return [{"id": row["id"], **json.loads(row["result_json"])} for row in self.connection.execute("SELECT id,result_json FROM evaluations WHERE meeting_id=? ORDER BY id DESC", (meeting_id,))]

    def link_topic(self, meeting_id: str, topic_name: str, related_meeting_ids: list[str], suggested_thread_name: str = "") -> dict:
        topic_key = topic_name.strip()
        now = self.meeting(meeting_id)
        created_at = now.get("created_at", "") if now else ""
        with self.lock:
            ids = set(related_meeting_ids)
            ids.add(meeting_id)
            placeholders = ",".join("?" for _ in ids)
            existing = [row[0] for row in self.connection.execute(f"SELECT DISTINCT thread_id FROM meeting_topics WHERE meeting_id IN ({placeholders})", tuple(ids))]
            topic_thread = self.connection.execute("SELECT thread_id FROM topics WHERE topic_key=?", (topic_key,)).fetchone()
            if topic_thread: existing.append(topic_thread[0])
            existing = list(dict.fromkeys(existing))
            thread_id = existing[0] if existing else str(__import__("uuid").uuid4())
            prior_topics = []
            if existing:
                marks = ",".join("?" for _ in existing)
                prior_topics = [row[0] for row in self.connection.execute(f"SELECT display_name FROM topics WHERE thread_id IN ({marks})", tuple(existing))]
                for old_thread in existing[1:]:
                    self.connection.execute("UPDATE meeting_topics SET thread_id=? WHERE thread_id=?", (thread_id, old_thread))
                    self.connection.execute("UPDATE topics SET thread_id=? WHERE thread_id=?", (thread_id, old_thread))
                    self.connection.execute("DELETE FROM threads WHERE thread_id=?", (old_thread,))
            topic_names = list(dict.fromkeys([name for name in prior_topics if name] + [topic_key]))
            self.connection.execute("INSERT OR REPLACE INTO meeting_topics(meeting_id,topic_key,thread_id) VALUES(?,?,?)", (meeting_id, topic_key, thread_id))
            self.connection.execute("INSERT OR REPLACE INTO topics(topic_key,display_name,thread_id) VALUES(?,?,?)", (topic_key, topic_key, thread_id))
            member_count = self.connection.execute("SELECT COUNT(*) FROM meeting_topics WHERE thread_id=?", (thread_id,)).fetchone()[0]
            if member_count > 1:
                thread_name = suggested_thread_name.strip() or " / ".join(topic_names)
            else:
                thread_name = suggested_thread_name.strip() or topic_key
            self.connection.execute("INSERT OR REPLACE INTO threads(thread_id,name,created_at) VALUES(?,?,COALESCE((SELECT created_at FROM threads WHERE thread_id=?),?))", (thread_id, thread_name, thread_id, created_at))
            self.connection.commit()
            return {"topic_key": topic_key, "thread_id": thread_id, "thread_name": thread_name, "related_count": max(0, member_count - 1)}

    def index_rag(self, meeting_id: str, topic_key: str, transcript_chunks: list[str], summary: str, actions: list[dict]) -> None:
        with self.lock:
            for table in ("rag_content", "rag_summaries", "rag_actions"):
                self.connection.execute(f"DELETE FROM {table} WHERE meeting_id=?", (meeting_id,))
            self.connection.executemany("INSERT INTO rag_content(topic_key,meeting_id,chunk_index,content,vector_json) VALUES(?,?,?,?,?)", [(topic_key, meeting_id, index, chunk, json.dumps(embed(chunk))) for index, chunk in enumerate(transcript_chunks)])
            if summary.strip(): self.connection.execute("INSERT INTO rag_summaries(topic_key,meeting_id,content,vector_json) VALUES(?,?,?,?)", (topic_key, meeting_id, summary, json.dumps(embed(summary))))
            saved_actions = self.connection.execute("SELECT id,item,owner,due_date FROM actions WHERE meeting_id=? ORDER BY id", (meeting_id,)).fetchall()
            for action in saved_actions:
                body = f"{action['item']} 負責人:{action['owner'] or ''} 期限:{action['due_date'] or ''}"
                self.connection.execute("INSERT INTO rag_actions(topic_key,meeting_id,action_id,content,vector_json) VALUES(?,?,?,?,?)", (topic_key, meeting_id, action["id"], body, json.dumps(embed(body))))
            self.connection.execute("DELETE FROM topics WHERE NOT EXISTS (SELECT 1 FROM meeting_topics mt WHERE mt.topic_key=topics.topic_key) AND NOT EXISTS (SELECT 1 FROM rag_content rc WHERE rc.topic_key=topics.topic_key) AND NOT EXISTS (SELECT 1 FROM rag_summaries rs WHERE rs.topic_key=topics.topic_key) AND NOT EXISTS (SELECT 1 FROM rag_actions ra WHERE ra.topic_key=topics.topic_key)")
            self.connection.execute("DELETE FROM threads WHERE NOT EXISTS (SELECT 1 FROM meeting_topics mt WHERE mt.thread_id=threads.thread_id)")
            self.connection.commit()

    def backfill_rag_indexes(self) -> None:
        legacy = self.connection.execute("SELECT id,title,summary FROM meetings WHERE summary IS NOT NULL AND TRIM(summary)<>'' AND title NOT LIKE '整理中%'").fetchall()
        for meeting in legacy:
            indexed = self.connection.execute("SELECT 1 FROM rag_summaries WHERE meeting_id=? LIMIT 1", (meeting["id"],)).fetchone()
            if indexed: continue
            self.link_topic(meeting["id"], meeting["title"], [])
            rows = self.transcript(meeting["id"])
            actions = [dict(row) for row in self.connection.execute("SELECT item,owner,due_date FROM actions WHERE meeting_id=?", (meeting["id"],))]
            self.index_rag(meeting["id"], meeting["title"], transcript_chunks(rows), meeting["summary"], actions)

    def rag_search(self, query: str, category: str = "all", limit: int = 20) -> list[dict]:
        tables = {"content": "rag_content", "summary": "rag_summaries", "action": "rag_actions"}
        selected = list(tables.items()) if category == "all" else [(category, tables[category])] if category in tables else []
        vector = embed(query)
        matches = []
        for label, table in selected:
            rows = self.connection.execute(f"SELECT r.*,m.title,m.meeting_time FROM {table} r JOIN meetings m ON m.id=r.meeting_id").fetchall()
            for row in rows:
                score = similarity(vector, json.loads(row["vector_json"]))
                matches.append({"category": label, "meeting_id": row["meeting_id"], "topic_key": row["topic_key"], "title": row["title"], "meeting_time": row["meeting_time"], "content": row["content"], "score": round(score, 4)})
        matches.sort(key=lambda item: item["score"], reverse=True)
        return matches[:limit]

    def related_context(self, query: str, exclude_meeting_id: str, limit: int = 4) -> list[dict]:
        matches = [item for item in self.rag_search(query, "all", 80) if item["meeting_id"] != exclude_meeting_id and item["score"] >= 0.08]
        grouped = {}
        for item in matches:
            current = grouped.get(item["meeting_id"])
            if current is None or item["score"] > current["score"]: grouped[item["meeting_id"]] = item
        result = []
        for item in sorted(grouped.values(), key=lambda row: row["score"], reverse=True)[:limit]:
            meeting = self.meeting(item["meeting_id"])
            actions = [dict(row) for row in self.connection.execute("SELECT id,item,owner,due_date FROM actions WHERE meeting_id=? ORDER BY id", (item["meeting_id"],))]
            analysis = self.analysis(item["meeting_id"]) or {}
            result.append({"meeting_id": item["meeting_id"], "topic": item["title"], "meeting_time": item["meeting_time"], "score": item["score"], "summary": analysis.get("summary", ""), "actions": actions})
        return result

    def thread_meetings(self, meeting_id: str) -> list[dict]:
        row = self.connection.execute("SELECT thread_id FROM meeting_topics WHERE meeting_id=?", (meeting_id,)).fetchone()
        if not row: return []
        return [dict(item) for item in self.connection.execute("SELECT m.id,m.title,m.meeting_time FROM meeting_topics mt JOIN meetings m ON m.id=mt.meeting_id WHERE mt.thread_id=? AND mt.meeting_id<>? ORDER BY m.meeting_time", (row[0], meeting_id))]

    def follow_ups(self, meeting_id: str) -> list[dict]:
        return [dict(row) for row in self.connection.execute("SELECT f.*,m.title AS previous_title FROM follow_ups f LEFT JOIN meetings m ON m.id=f.previous_meeting_id WHERE f.meeting_id=? ORDER BY f.id", (meeting_id,))]
