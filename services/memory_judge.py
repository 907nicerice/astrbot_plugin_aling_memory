from __future__ import annotations

import json
import re
from typing import Any

from ..models.memory_item import ALLOWED_MEMORY_TYPES, CandidateMemory
from .defaults import DEFAULT_CONFIG
from .extractor import contains_sensitive_secret


SYSTEM_PROMPT = """你是阿绫的长期记忆整理器。你的任务不是尽量多记，而是谨慎判断用户信息是否值得跨会话保留。

原则：
1. 一次性情绪、当天状态、寒暄、待办和催办事项不进入长期记忆。
2. 稳定偏好、明确互动边界、长期身份背景和有意义的共同经历可以进入候选。
3. 密码、验证码、证件号、银行卡、精确住址、Cookie、Token、密钥等敏感信息绝不保存。
4. 不推断疾病、诊断、政治倾向、性取向等敏感属性。
5. content 只写一条独立、简短、可更新的事实，不复制整段原话。
6. 用户明确说“记住”且内容安全稳定时可 decision=save；其他有价值信息通常 decision=candidate。
7. 项目进度和未完成任务不是长期人格记忆；稳定的项目背景可以有期限地保存。

只输出 JSON，不要解释。格式：
{"memories":[{"decision":"save|candidate|ignore","type":"small_memory|preference_memory|relationship_memory|life_signal|project_context","content":"...","reason":"...","confidence":0.0,"importance":0.0,"stability":0.0,"sensitivity":"low|medium|high","ttl_days":null,"tags":["..."],"use_rule":"..."}]}
没有值得处理的内容时输出 {"memories":[]}。"""


class MemoryJudge:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = {**DEFAULT_CONFIG, **(config or {})}

    def should_call(self, text: str, has_rule_candidate: bool) -> bool:
        mode = str(self.config.get("memory_judge_mode") or "hybrid").lower()
        if mode == "rule":
            return False
        if mode == "llm":
            return True
        if has_rule_candidate:
            return True
        personal_signals = (
            "我", "我的", "以后", "一直", "通常", "平时", "喜欢", "讨厌", "不能", "不要", "叫我", "正在做",
        )
        return any(signal in text for signal in personal_signals)

    def build_prompt(self, text: str, history: list[dict[str, str]]) -> str:
        compact_history = []
        for row in history[-8:]:
            role = str(row.get("role") or "")
            content = re.sub(r"\s+", " ", str(row.get("content") or "")).strip()[:500]
            if role in {"user", "assistant"} and content:
                compact_history.append({"role": role, "content": content})
        payload = {"current_user_message": text[:1000], "recent_context": compact_history}
        return json.dumps(payload, ensure_ascii=False)

    def parse(self, text: str) -> list[CandidateMemory]:
        cleaned = re.sub(r"^```(?:json)?\s*", "", str(text or "").strip(), flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("memory judge did not return a JSON object")
        raw = json.loads(cleaned[start : end + 1])
        rows = raw.get("memories", []) if isinstance(raw, dict) else []
        if not isinstance(rows, list):
            raise ValueError("memory judge memories must be a list")
        result: list[CandidateMemory] = []
        for row in rows[:3]:
            if not isinstance(row, dict) or row.get("decision") == "ignore":
                continue
            content = re.sub(r"\s+", " ", str(row.get("content") or "")).strip()[:300]
            if not content:
                continue
            if contains_sensitive_secret(content):
                continue
            memory_type = str(row.get("type") or "small_memory")
            if memory_type not in ALLOWED_MEMORY_TYPES or memory_type == "context_summary":
                memory_type = "small_memory"
            sensitivity = str(row.get("sensitivity") or "low")
            if sensitivity not in {"low", "medium", "high"}:
                sensitivity = "medium"
            decision = str(row.get("decision") or "candidate")
            if sensitivity == "high":
                continue
            if sensitivity == "medium":
                decision = "candidate"
            result.append(
                CandidateMemory.create(
                    memory_type,
                    content,
                    reason=str(row.get("reason") or "LLM 判断为可能有长期价值。")[:300],
                    confidence=self._score(row.get("confidence"), 0.7),
                    importance=self._score(row.get("importance"), 0.5),
                    stability=self._score(row.get("stability"), 0.5),
                    sensitivity=sensitivity,
                    decision=decision,
                    ttl_days=self._ttl(row.get("ttl_days")),
                    tags=self._tags(row.get("tags")),
                    use_rule=str(row.get("use_rule") or "")[:500],
                )
            )
        return result

    @staticmethod
    def _score(value: Any, default: float) -> float:
        try:
            return round(max(0.0, min(1.0, float(value))), 2)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _ttl(value: Any) -> int | None:
        if value in (None, "", 0, "0"):
            return None
        try:
            return max(1, min(3650, int(value)))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _tags(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in value:
            tag = str(item).strip()[:40]
            if tag and tag not in result:
                result.append(tag)
        return result[:10]
