from __future__ import annotations

import unittest

from astrbot_plugin_aling_memory.services.provider_compat import scope_from_event


class _Event:
    def __init__(self, sender_id: str, origin: str) -> None:
        self._sender_id = sender_id
        self.unified_msg_origin = origin

    def get_sender_id(self) -> str:
        return self._sender_id


class TestAccountScopeTests(unittest.TestCase):
    def test_formal_account_keeps_legacy_scope(self) -> None:
        event = _Event("10001", "aiocqhttp:FriendMessage:10001")
        self.assertEqual(scope_from_event(event, "20002"), "aiocqhttp:FriendMessage:10001")

    def test_test_account_gets_isolated_scope(self) -> None:
        event = _Event("20002", "aiocqhttp:FriendMessage:20002")
        self.assertEqual(
            scope_from_event(event, "20002，30003"),
            "test-account:20002:aiocqhttp:FriendMessage:20002",
        )


if __name__ == "__main__":
    unittest.main()
