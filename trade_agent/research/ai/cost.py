from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from research.ai.models import (
    BATCH_DISCOUNT,
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER,
    PRICING_SNAPSHOT_DATE,
    get_pricing,
)


class BudgetExceeded(Exception):
    """Raised when the configured USD budget is reached.

    The runner catches this, checkpoints, and stops cleanly -- it never keeps
    spending past the limit (spec section 18).
    """


class TokenUsage(BaseModel):
    """One API call's token accounting, mirroring the four usage fields the
    Messages API reports. Kept separate rather than summed because the whole
    point of caching is to see the split."""

    input_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_input_equivalent(self) -> int:
        return self.input_tokens + self.cache_creation_tokens + self.cache_read_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


def compute_cost(usage: TokenUsage, model_id: str, batch: bool = False) -> float:
    """USD cost of one call.

    Cached tokens are billed at their own multipliers (writes above the input
    rate, reads far below it), so a report that only summed raw input tokens
    would misstate the bill in both directions.
    """
    pricing = get_pricing(model_id)
    input_rate = pricing.input_per_mtok / 1_000_000
    output_rate = pricing.output_per_mtok / 1_000_000

    cost = (
        usage.input_tokens * input_rate
        + usage.cache_creation_tokens * input_rate * CACHE_WRITE_MULTIPLIER
        + usage.cache_read_tokens * input_rate * CACHE_READ_MULTIPLIER
        + usage.output_tokens * output_rate
    )
    if batch:
        cost *= BATCH_DISCOUNT
    return cost


class CallRecord(BaseModel):
    """Per-call metrics the spec requires for every AI request."""

    signal_id: str
    agent: str
    model: str
    usage: TokenUsage
    cost_usd: float
    latency_seconds: float
    batch: bool = False
    prompt_version: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def cache_hit(self) -> bool:
        return self.usage.cache_read_tokens > 0


class CostLedger:
    """Running cost total with a hard budget stop.

    `reserve` is called BEFORE a call is made, using a conservative estimate,
    so the budget can halt the run before spending rather than after
    discovering it overspent.
    """

    def __init__(self, budget_usd: float | None = None) -> None:
        self._budget = budget_usd
        self._records: list[CallRecord] = []
        self._spent = 0.0

    @property
    def spent_usd(self) -> float:
        return self._spent

    @property
    def budget_usd(self) -> float | None:
        return self._budget

    @property
    def remaining_usd(self) -> float | None:
        return None if self._budget is None else max(0.0, self._budget - self._spent)

    @property
    def records(self) -> list[CallRecord]:
        return list(self._records)

    def check_budget(self, projected_cost: float = 0.0) -> None:
        if self._budget is None:
            return
        if self._spent + projected_cost > self._budget:
            raise BudgetExceeded(
                f"cost limit reached: spent ${self._spent:.4f} of ${self._budget:.2f} budget"
                + (
                    f", next call projected at ${projected_cost:.4f}"
                    if projected_cost
                    else ""
                )
                + ". Stopping safely; raise AI_COST_LIMIT_USD to continue."
            )

    def record(self, record: CallRecord) -> CallRecord:
        self._records.append(record)
        self._spent += record.cost_usd
        return record

    def add_external(self, cost_usd: float) -> None:
        """Account for spend recorded elsewhere (e.g. a resumed checkpoint),
        so the budget survives a restart instead of resetting to zero."""
        self._spent += cost_usd


class CostSummary(BaseModel):
    """Measured cost report + projections. Every field here comes from
    observed token counts -- nothing is estimated in advance (spec 12)."""

    pricing_snapshot_date: str = PRICING_SNAPSHOT_DATE
    signals_processed: int
    api_calls: int
    failed_calls: int
    total_cost_usd: float

    avg_cost_per_signal: float
    median_cost_per_signal: float
    min_cost_per_signal: float
    max_cost_per_signal: float

    avg_input_tokens: float
    avg_cache_creation_tokens: float
    avg_cache_read_tokens: float
    avg_output_tokens: float
    cache_read_ratio: float  # cache_read / total input-equivalent tokens
    cache_hit_rate: float    # fraction of calls served any cached tokens

    avg_latency_seconds: float
    cost_by_agent: dict[str, float] = Field(default_factory=dict)
    cost_by_model: dict[str, float] = Field(default_factory=dict)
    calls_by_agent: dict[str, int] = Field(default_factory=dict)
    batch_used: bool = False

    projected_1k: float
    projected_5k: float
    projected_10k: float
    projected_full_run: float | None = None
    full_run_signal_count: int | None = None


def summarize_costs(
    records: list[CallRecord],
    signals_processed: int,
    full_run_signal_count: int | None = None,
) -> CostSummary:
    """Build the measured cost report from actual call records."""
    import statistics

    successful = [r for r in records if r.error is None]
    failed = [r for r in records if r.error is not None]

    per_signal: dict[str, float] = {}
    for record in records:
        per_signal[record.signal_id] = per_signal.get(record.signal_id, 0.0) + record.cost_usd
    signal_costs = sorted(per_signal.values()) or [0.0]

    total_cost = sum(r.cost_usd for r in records)
    n = max(len(successful), 1)

    total_input_equivalent = sum(r.usage.total_input_equivalent for r in successful) or 1
    total_cache_read = sum(r.usage.cache_read_tokens for r in successful)

    cost_by_agent: dict[str, float] = {}
    cost_by_model: dict[str, float] = {}
    calls_by_agent: dict[str, int] = {}
    for record in records:
        cost_by_agent[record.agent] = cost_by_agent.get(record.agent, 0.0) + record.cost_usd
        cost_by_model[record.model] = cost_by_model.get(record.model, 0.0) + record.cost_usd
        calls_by_agent[record.agent] = calls_by_agent.get(record.agent, 0) + 1

    avg_per_signal = total_cost / signals_processed if signals_processed else 0.0

    return CostSummary(
        signals_processed=signals_processed,
        api_calls=len(records),
        failed_calls=len(failed),
        total_cost_usd=total_cost,
        avg_cost_per_signal=avg_per_signal,
        median_cost_per_signal=statistics.median(signal_costs),
        min_cost_per_signal=min(signal_costs),
        max_cost_per_signal=max(signal_costs),
        avg_input_tokens=sum(r.usage.input_tokens for r in successful) / n,
        avg_cache_creation_tokens=sum(r.usage.cache_creation_tokens for r in successful) / n,
        avg_cache_read_tokens=sum(r.usage.cache_read_tokens for r in successful) / n,
        avg_output_tokens=sum(r.usage.output_tokens for r in successful) / n,
        cache_read_ratio=total_cache_read / total_input_equivalent,
        cache_hit_rate=sum(1 for r in successful if r.cache_hit) / n,
        avg_latency_seconds=sum(r.latency_seconds for r in successful) / n,
        cost_by_agent=cost_by_agent,
        cost_by_model=cost_by_model,
        calls_by_agent=calls_by_agent,
        batch_used=any(r.batch for r in records),
        projected_1k=avg_per_signal * 1_000,
        projected_5k=avg_per_signal * 5_000,
        projected_10k=avg_per_signal * 10_000,
        projected_full_run=(
            avg_per_signal * full_run_signal_count if full_run_signal_count else None
        ),
        full_run_signal_count=full_run_signal_count,
    )
