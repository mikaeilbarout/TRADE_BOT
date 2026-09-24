from enum import StrEnum


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class TradingMode(StrEnum):
    PAPER = "paper"
    SHADOW = "shadow"
    LIVE = "live"
    MANUAL = "manual"


class GateDecision(StrEnum):
    """Decision emitted by a gating agent (news/sentiment)."""

    PASS = "PASS"
    WARNING = "WARNING"
    BLOCK = "BLOCK"


# 2026-09-19: MODIFY removed from both live vocabularies by explicit user
# decision -- the AI may approve or refuse a trade, never redraw it. A
# 190-signal replay of the deployed configuration showed the AI's level
# changes cut the modified trades' result from +20.0R to +14.6R. WARNING
# takes MODIFY's old place as the technical agent's "concerns, not a
# veto" middle state (it maps to WARN in the policy core exactly as
# MODIFY did).
class TechnicalDecision(StrEnum):
    PASS = "PASS"
    WARNING = "WARNING"
    BLOCK = "BLOCK"


class FinalDecision(StrEnum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    WAIT = "WAIT"


class Bias(StrEnum):
    VERY_BULLISH = "VERY_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    VERY_BEARISH = "VERY_BEARISH"


class ImpactLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class MarketEnvironment(StrEnum):
    SUPPORTIVE = "SUPPORTIVE"
    NEUTRAL = "NEUTRAL"
    HOSTILE = "HOSTILE"
    UNCERTAIN = "UNCERTAIN"


class SentimentStrength(StrEnum):
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"


class ContradictionLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class TradeCompatibility(StrEnum):
    STRONG_SUPPORT = "STRONG_SUPPORT"
    SUPPORT = "SUPPORT"
    NEUTRAL = "NEUTRAL"
    CONFLICT = "CONFLICT"
    STRONG_CONFLICT = "STRONG_CONFLICT"


class AssetType(StrEnum):
    COMMODITY = "commodity"
    CRYPTO = "crypto"
    FX = "fx"
    INDEX = "index"
    EQUITY = "equity"


class PipelineStage(StrEnum):
    VALIDATION = "validation"
    HARD_RISK_PRECHECK = "hard_risk_precheck"
    MARKET_DATA = "market_data"
    NEWS_AGENT = "news_agent"
    SENTIMENT_AGENT = "sentiment_agent"
    TECHNICAL_AGENT = "technical_agent"
    FINAL_DECISION_AGENT = "final_decision_agent"
    DECISION_POLICY = "decision_policy"
    EXECUTION_GUARD = "execution_guard"
    COMPLETE = "complete"


class AgentAgreement(StrEnum):
    """Each downstream agent must state explicitly how its own independent
    analysis relates to the upstream chain, so blind agreement is visible
    in the audit trail rather than hidden (section 18)."""

    AGREE = "AGREE"
    PARTIAL = "PARTIAL"
    DISAGREE = "DISAGREE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ApprovalStatus(StrEnum):
    """Human-in-the-loop state for MANUAL mode (section 30)."""

    NOT_REQUIRED = "NOT_REQUIRED"
    AWAITING_HUMAN = "AWAITING_HUMAN"
    HUMAN_APPROVED = "HUMAN_APPROVED"
    HUMAN_REJECTED = "HUMAN_REJECTED"
