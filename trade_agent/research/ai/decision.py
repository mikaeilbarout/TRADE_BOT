from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from research.ai.cost import TokenUsage
from research.ai.schemas import FinalAction

"""The AI layer's per-signal verdict.

Deliberately a leaf module: the execution bridge
(`research.backtest.executor`) needs this type, and importing it from the
runner would drag the Anthropic client and the whole agent stack into the
backtest engine's import graph. Keeping the data model separate from the
machinery that produces it is also what lets the executor be tested with
hand-written decisions and no LLM at all.
"""


class SignalDecision(BaseModel):
    """The AI layer's verdict on one historical signal, with the full chain."""

    signal_id: str
    signal_time: datetime
    action: FinalAction
    confidence: float
    reason: str
    reason_codes: list[str] = Field(default_factory=list)

    technical: dict | None = None
    news: dict | None = None
    sentiment: dict | None = None
    final: dict | None = None
    skip_reasons: dict = Field(default_factory=dict)

    modified_entry: float | None = None
    modified_sl: float | None = None
    modified_tp: float | None = None
    was_modified: bool = False

    gate_passed: bool = True
    gate_violations: list[str] = Field(default_factory=list)
    guard_violations: list[str] = Field(default_factory=list)
    failed_closed: bool = False
    agents_called: list[str] = Field(default_factory=list)
    cost_usd: float = 0.0
    usage: TokenUsage = Field(default_factory=TokenUsage)

    # --- policy provenance -------------------------------------------------
    # Which shared-policy rule produced this action, the weighted evidence
    # score it was judged on, and which agent -- if any -- is answerable for
    # a non-approval. The last one is what makes "which agent caused the most
    # false rejections?" a groupby rather than text mining.
    deciding_rule: str = "AGENT_DECISION"
    weighted_score: float | None = None
    blocking_agent: str | None = None

    @property
    def is_executable(self) -> bool:
        return self.action in (FinalAction.APPROVE, FinalAction.MODIFY)

    def agent_verdict(self, agent: str) -> dict | None:
        return getattr(self, agent, None)

    def chain(self) -> dict:
        """The full agent chain, for the trade log's audit trail."""
        return {
            "technical": self.technical,
            "news": self.news,
            "sentiment": self.sentiment,
            "final": self.final,
            "skip_reasons": self.skip_reasons,
            "agents_called": list(self.agents_called),
            "deciding_rule": self.deciding_rule,
            "weighted_score": self.weighted_score,
            "reason_codes": list(self.reason_codes),
            "gate_violations": list(self.gate_violations),
            "guard_violations": list(self.guard_violations),
            "failed_closed": self.failed_closed,
        }


def decisions_by_signal(decisions: list[SignalDecision]) -> dict[str, SignalDecision]:
    """Index decisions for the execution join.

    Raises on a duplicate signal_id rather than silently keeping the last
    one: two decisions for one signal means the checkpoint or the run was
    merged wrongly, and quietly picking one would produce an unexplainable
    equity curve.
    """
    index: dict[str, SignalDecision] = {}
    for decision in decisions:
        if decision.signal_id in index:
            raise ValueError(
                f"duplicate decision for signal {decision.signal_id}: refusing to "
                "guess which verdict applies"
            )
        index[decision.signal_id] = decision
    return index
