from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

"""Compact agent outputs.

Every schema here is deliberately small: enums and reason CODES instead of
prose. Output tokens are the most expensive tokens in the request, and a
backtest needs machine-comparable decisions far more than it needs essays.
`note` is capped hard so a verbose model cannot quietly inflate the bill.
"""

MAX_NOTE_CHARS = 180


class Gate(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"
    UNAVAILABLE = "UNAVAILABLE"  # data genuinely absent; never invented


class FinalAction(StrEnum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    MODIFY = "MODIFY"
    WAIT = "WAIT"


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Bias(StrEnum):
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"


# Closed reason-code vocabularies. A closed set keeps outputs tiny, makes
# "which agent rejected the most trades and why" a groupby instead of text
# mining, and stops the model from inventing a new rationale every call.
TECHNICAL_CODES = [
    "TREND_ALIGNED", "TREND_CONFLICT", "HTF_ALIGNED", "HTF_CONFLICT",
    "STRUCTURE_BOS", "STRUCTURE_CHOCH", "AT_RESISTANCE", "AT_SUPPORT",
    "OVEREXTENDED", "MOMENTUM_OK", "MOMENTUM_WEAK", "RSI_EXTENDED",
    "MACD_CONFIRMS", "MACD_DIVERGES", "BREAKOUT_CONFIRMED", "FALSE_BREAKOUT_RISK",
    "PULLBACK_ENTRY", "VOLATILITY_OK", "VOLATILITY_HIGH", "VOLATILITY_LOW",
    "GOOD_RR", "POOR_RR", "SL_LOGICAL", "SL_ILLOGICAL", "TP_REALISTIC",
    "TP_UNREALISTIC", "SPREAD_OK", "SPREAD_WIDE", "LIQUIDITY_THIN",
]

NEWS_CODES = [
    "NO_HIGH_IMPACT_NEWS", "HIGH_IMPACT_PENDING", "HIGH_IMPACT_RECENT",
    "NEWS_SUPPORTS", "NEWS_CONTRADICTS", "NEWS_NEUTRAL", "FOMC_WINDOW",
    "CPI_WINDOW", "NFP_WINDOW", "RATE_DECISION", "GEOPOLITICAL_RISK",
    "USD_DRIVER", "YIELD_DRIVER", "DATA_UNAVAILABLE", "DATA_STALE",
]

SENTIMENT_CODES = [
    "SENTIMENT_SUPPORTS", "SENTIMENT_CONTRADICTS", "SENTIMENT_NEUTRAL",
    "SENTIMENT_STRONG", "SENTIMENT_WEAK", "SENTIMENT_MIXED",
    "CROWDED_POSITIONING", "RISK_ON", "RISK_OFF", "USD_BULLISH", "USD_BEARISH",
    "LOW_QUALITY_SOURCES", "MANIPULATION_SUSPECTED", "DATA_UNAVAILABLE",
]

FINAL_CODES = [
    "ALL_ALIGNED", "MAJORITY_ALIGNED", "TECHNICAL_VETO", "NEWS_VETO",
    "SENTIMENT_VETO", "EVENT_RISK", "CHAIN_CONFLICT", "WEAK_EVIDENCE",
    "INSUFFICIENT_DATA", "BETTER_ENTRY_AVAILABLE", "RR_IMPROVED",
    "STOP_TOO_TIGHT", "STOP_TOO_WIDE", "GOOD_SETUP", "POOR_SETUP",
]


class AgentVerdict(BaseModel):
    """Shared shape for the three analyst agents."""

    decision: Gate
    # Anthropic's strict tool schema rejects minimum/maximum/maxLength/
    # maxItems on the wire (see research.ai.client._strip_unsupported_
    # constraints), so the bound is stated here in the description instead --
    # Pydantic still enforces it when the response is parsed back.
    confidence: float = Field(ge=0.0, le=1.0, description="Between 0.0 and 1.0.")
    bias: Bias = Bias.NEUTRAL
    risk_level: RiskLevel = RiskLevel.MEDIUM
    # Mirrors app.models.agent_decision.AgentDecisionBase.is_degraded. UNAVAILABLE
    # is for "I could not analyze this at all" (no PIT coverage); is_degraded is
    # for "I rendered a verdict, but my inputs were thin/stale enough that I do
    # not fully trust it" -- the live policy's "degraded input forces WAIT" rule
    # (app/services/decision_policy.py, app/services/policy_core.py) keys off
    # exactly this, not off UNAVAILABLE (which carries no veto or WAIT of its
    # own). Without this field the research replay could never trigger that
    # rule, making the backtest measurably more permissive than production.
    is_degraded: bool = False
    reason_codes: list[str] = Field(
        default_factory=list, max_length=6, description="At most 6 codes."
    )
    note: str = Field(
        default="",
        max_length=MAX_NOTE_CHARS,
        description=f"At most {MAX_NOTE_CHARS} characters.",
    )


class TechnicalVerdict(AgentVerdict):
    htf_aligned: bool = False
    entry_quality: RiskLevel = RiskLevel.MEDIUM  # LOW = poor entry
    suggest_entry: float | None = None
    suggest_sl: float | None = None
    suggest_tp: float | None = None


class NewsVerdict(AgentVerdict):
    high_impact_within_window: bool = False


class SentimentVerdict(AgentVerdict):
    pass


class FinalVerdict(BaseModel):
    action: FinalAction
    confidence: float = Field(ge=0.0, le=1.0, description="Between 0.0 and 1.0.")
    risk_level: RiskLevel = RiskLevel.MEDIUM
    reason_codes: list[str] = Field(
        default_factory=list, max_length=6, description="At most 6 codes."
    )
    note: str = Field(
        default="",
        max_length=MAX_NOTE_CHARS,
        description=f"At most {MAX_NOTE_CHARS} characters.",
    )
    # Required when action == MODIFY; validated by the runner against the
    # deterministic guard, never trusted as-is.
    entry: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    # Per-dimension scores, 0-100. These exist so the research path can apply
    # the SAME weighted-score floor the live service applies -- without them
    # the backtest would be measuring a policy with one rule missing. Four
    # integers is a negligible output-token cost for that parity. Left None
    # the rule is skipped and the omission is recorded, never treated as a
    # pass.
    news_score: int | None = Field(default=None, ge=0, le=100, description="0 to 100.")
    sentiment_score: int | None = Field(default=None, ge=0, le=100, description="0 to 100.")
    technical_score: int | None = Field(default=None, ge=0, le=100, description="0 to 100.")
    risk_score: int | None = Field(default=None, ge=0, le=100, description="0 to 100.")

    def component_scores(self) -> tuple[float, float, float, float] | None:
        values = (
            self.news_score,
            self.sentiment_score,
            self.technical_score,
            self.risk_score,
        )
        if any(value is None for value in values):
            return None
        return tuple(float(value) for value in values)  # type: ignore[return-value]


AGENT_SCHEMAS: dict[str, type[BaseModel]] = {
    "technical": TechnicalVerdict,
    "news": NewsVerdict,
    "sentiment": SentimentVerdict,
    "final": FinalVerdict,
}

AGENT_CODES: dict[str, list[str]] = {
    "technical": TECHNICAL_CODES,
    "news": NEWS_CODES,
    "sentiment": SENTIMENT_CODES,
    "final": FINAL_CODES,
}
