from __future__ import annotations

import re
from typing import Any

from ..models.memory_item import CandidateMemory
from .defaults import DEFAULT_CONFIG


STRONG_SIGNALS = ("记住", "请记得", "以后别", "我希望", "以后不要", "以后可以", "别再")
PREFERENCE_SIGNALS = ("我喜欢", "我讨厌", "我不喜欢", "我更喜欢", "我习惯", "我不想", "我不能吃", "我不吃")
STABLE_SIGNALS = (
    "我叫", "我是", "我一直", "我通常", "我平时", "每次都", "我正在做", "我在做", "我的专业", "我的工作",
)
EPHEMERAL_SIGNALS = ("今天", "今晚", "刚才", "现在", "这次", "暂时", "有点", "刚刚", "这两天")
SENSITIVE_PATTERNS = (
    re.compile(r"密码|验证码|身份证|银行卡|家庭住址|手机号", re.I),
    re.compile(r"(?:api[_ -]?key|cookie|access[_ -]?token|refresh[_ -]?token|密钥)\s*(?:是|为|[:=])\s*\S+", re.I),
)
TRIVIAL_MESSAGES = {"嗯", "哦", "好", "好的", "哈哈", "谢谢", "晚安", "早安", "在吗", "？", "?"}


def contains_sensitive_secret(text: str) -> bool:
    return any(pattern.search(str(text or "")) for pattern in SENSITIVE_PATTERNS)


class Extractor:
    """Conservative local judge and fallback for the optional LLM judge."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = {**DEFAULT_CONFIG, **(config or {})}

    def eligible_for_llm(self, text: str) -> bool:
        clean = self._clean(text)
        minimum = int(self.config.get("memory_judge_min_chars") or 8)
        if len(clean) < minimum or clean.lower().startswith(("/", "／")):
            return False
        return clean not in TRIVIAL_MESSAGES

    def should_extract_from_user_text(self, text: str) -> bool:
        clean = self._clean(text)
        return self.eligible_for_llm(clean) and any(
            signal in clean for signal in (*STRONG_SIGNALS, *PREFERENCE_SIGNALS, *STABLE_SIGNALS)
        )

    def extract_from_user_text(self, text: str) -> list[CandidateMemory]:
        if not self.should_extract_from_user_text(text):
            return []
        clean = self._clean(text)
        memory_type = self._type_for(clean)
        explicit = any(signal in clean for signal in STRONG_SIGNALS)
        ephemeral = any(signal in clean for signal in EPHEMERAL_SIGNALS)
        sensitivity = "high" if contains_sensitive_secret(clean) else "low"
        stability = self._stability(memory_type, explicit, ephemeral)
        importance = self._importance(memory_type, explicit)
        confidence = 0.95 if explicit else 0.82
        decision = "save" if explicit and sensitivity == "low" and stability >= 0.7 else "candidate"
        if sensitivity == "high":
            confidence = min(confidence, 0.65)
        return [
            CandidateMemory.create(
                memory_type,
                self._candidate_content(clean),
                reason="用户明确要求记住。" if explicit else "消息包含可能稳定的个人事实或偏好。",
                confidence=confidence,
                importance=importance,
                stability=stability,
                sensitivity=sensitivity,
                decision=decision,
                ttl_days=self._ttl_for(memory_type, ephemeral),
                tags=self._tags_for(clean),
                use_rule=self._use_rule_for(memory_type),
            )
        ]

    def extract_from_summary(self, summary: str) -> list[CandidateMemory]:
        text = self._clean(summary)
        if not text:
            return []
        if any(word in text for word in ("插件", "AstrBot", "prompt", "QQ bot", "token", "Napcat")):
            return [
                CandidateMemory.create(
                    "project_context",
                    "用户正在做阿绫 QQ bot，关注人格自然度、长期记忆、上下文压缩和 token 控制。",
                    reason="摘要中反复出现稳定项目背景。",
                    confidence=0.76,
                    importance=0.72,
                    stability=0.68,
                    sensitivity="low",
                    decision="candidate",
                    ttl_days=90,
                    tags=["project", "bot"],
                    use_rule="只有用户主动提到 bot、插件、prompt、AstrBot、QQ 空间、上下文或 token 时使用。",
                )
            ]
        return []

    def auto_confirmable(self, candidate: CandidateMemory) -> bool:
        enabled = bool(
            self.config.get("auto_confirm_safe_memories", self.config.get("auto_confirm_safe_preferences", False))
        )
        threshold = float(self.config.get("auto_confirm_min_confidence") or 0.92)
        return (
            enabled
            and candidate.decision == "save"
            and candidate.sensitivity == "low"
            and candidate.confidence >= threshold
            and candidate.stability >= 0.7
        )

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    @staticmethod
    def _candidate_content(text: str) -> str:
        clean = re.sub(r"^(请)?(帮我)?记住[：,:，\s]*", "", text).strip()
        return f"用户表达过：{clean[:240]}"

    @staticmethod
    def _type_for(text: str) -> str:
        if any(word in text for word in ("亲近", "边界", "表白", "关系", "偏心", "称呼我", "不要叫我")):
            return "relationship_memory"
        if any(word in text for word in (*PREFERENCE_SIGNALS, "回复短", "回复详细", "留白")):
            return "preference_memory"
        if any(word in text for word in ("插件", "AstrBot", "prompt", "Napcat", "QQ空间", "token", "bot")):
            return "project_context"
        if any(word in text for word in ("每天", "平时", "通常", "作息", "工作", "专业")):
            return "life_signal"
        return "small_memory"

    @staticmethod
    def _stability(memory_type: str, explicit: bool, ephemeral: bool) -> float:
        base = {
            "preference_memory": 0.84, "relationship_memory": 0.9, "project_context": 0.68,
            "life_signal": 0.65, "small_memory": 0.58,
        }.get(memory_type, 0.55)
        if explicit:
            base += 0.08
        if ephemeral:
            base -= 0.35
        return round(max(0.0, min(1.0, base)), 2)

    @staticmethod
    def _importance(memory_type: str, explicit: bool) -> float:
        base = {
            "relationship_memory": 0.86, "preference_memory": 0.76, "project_context": 0.7,
            "life_signal": 0.62, "small_memory": 0.55,
        }.get(memory_type, 0.5)
        return round(min(1.0, base + (0.1 if explicit else 0.0)), 2)

    @staticmethod
    def _ttl_for(memory_type: str, ephemeral: bool) -> int | None:
        if ephemeral:
            return 14
        return {
            "project_context": 90, "life_signal": 180, "small_memory": 365,
            "preference_memory": None, "relationship_memory": None,
        }.get(memory_type)

    @staticmethod
    def _tags_for(text: str) -> list[str]:
        tags: list[str] = []
        if any(word in text for word in ("课", "考试", "物理", "高数", "题")):
            tags.append("study")
        if any(word in text for word in ("插件", "AstrBot", "prompt", "bot", "token")):
            tags.append("project")
        if any(word in text for word in ("短", "自然", "助手", "留白", "详细")):
            tags.append("style")
        if any(word in text for word in ("记住", "记忆", "记得")):
            tags.append("memory")
        return tags or ["daily"]

    @staticmethod
    def _use_rule_for(memory_type: str) -> str:
        if memory_type == "project_context":
            return "只有用户主动提到项目、bot、插件、prompt、AstrBot 或 token 时使用。"
        if memory_type in {"preference_memory", "relationship_memory"}:
            return "优先隐性影响回复方式；除非用户询问，否则不要显式复述这条记忆。"
        return "只有当前话题明显相关时自然参考，不要主动展开。"
