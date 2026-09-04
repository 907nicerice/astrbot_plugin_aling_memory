from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from astrbot_plugin_aling_memory.models.memory_item import CandidateMemory, MemoryItem
from astrbot_plugin_aling_memory.services.extractor import Extractor
from astrbot_plugin_aling_memory.services.memory_judge import MemoryJudge
from astrbot_plugin_aling_memory.services.memory_store import MemoryStore


class MemoryLifecycleTests(unittest.TestCase):
    def test_new_ttl_uses_explicit_expiration(self) -> None:
        item = MemoryItem.create("project_context", "长期项目背景", ttl_days=30)
        self.assertIsNotNone(item.expires_at)
        expires = datetime.fromisoformat(str(item.expires_at))
        self.assertGreater(expires, datetime.now(timezone.utc) + timedelta(days=29))

    def test_legacy_ttl_remains_compatible(self) -> None:
        item = MemoryItem.from_dict(
            {
                "id": "mem_old",
                "type": "small_memory",
                "content": "旧记忆",
                "ttl_days": 3,
                "created_at": (datetime.now(timezone.utc) - timedelta(days=4)).isoformat(),
            }
        )
        self.assertTrue(item.is_expired())

    def test_explicit_safe_request_can_auto_confirm(self) -> None:
        extractor = Extractor({"auto_confirm_safe_memories": True})
        candidate = extractor.extract_from_user_text("请记住我喜欢简短自然的回复")[0]
        self.assertEqual(candidate.decision, "save")
        self.assertEqual(candidate.sensitivity, "low")
        self.assertTrue(extractor.auto_confirmable(candidate))

    def test_sensitive_secret_never_auto_confirms(self) -> None:
        extractor = Extractor({"auto_confirm_safe_memories": True})
        candidate = extractor.extract_from_user_text("请记住我的密码是 abc12345")[0]
        self.assertEqual(candidate.sensitivity, "high")
        self.assertFalse(extractor.auto_confirmable(candidate))

    def test_project_token_topic_is_not_mistaken_for_a_secret(self) -> None:
        extractor = Extractor({"auto_confirm_safe_memories": True})
        candidate = extractor.extract_from_user_text("请记住我在优化 bot 的 token 使用量")[0]
        self.assertEqual(candidate.sensitivity, "low")

    def test_repeated_candidate_is_consolidated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory), {"candidate_reinforce_threshold": 2})
            candidate = CandidateMemory.create(
                "preference_memory",
                "用户喜欢简短自然的回复",
                "稳定偏好",
                0.84,
                importance=0.8,
                stability=0.86,
            )
            first, _ = store.consolidate_candidate("scope", candidate)
            second, item = store.consolidate_candidate(
                "scope",
                CandidateMemory.create(
                    "preference_memory",
                    "用户喜欢简短自然的回复",
                    "再次提到",
                    0.86,
                    importance=0.8,
                    stability=0.88,
                ),
            )
            self.assertEqual(first, "candidate")
            self.assertEqual(second, "promoted")
            self.assertEqual(item.evidence_count, 2)
            self.assertEqual(len(store.list_candidates("scope")), 0)
            self.assertEqual(len(store.list_memories("scope")), 1)

    def test_llm_judge_parses_and_rejects_high_sensitivity(self) -> None:
        payload = {
            "memories": [
                {
                    "decision": "save",
                    "type": "preference_memory",
                    "content": "用户喜欢回答简短一些",
                    "confidence": 0.96,
                    "importance": 0.8,
                    "stability": 0.9,
                    "sensitivity": "low",
                },
                {
                    "decision": "save",
                    "type": "small_memory",
                    "content": "用户的验证码是 123456",
                    "confidence": 0.99,
                    "sensitivity": "high",
                },
            ]
        }
        result = MemoryJudge().parse(json.dumps(payload, ensure_ascii=False))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].content, "用户喜欢回答简短一些")


if __name__ == "__main__":
    unittest.main()
