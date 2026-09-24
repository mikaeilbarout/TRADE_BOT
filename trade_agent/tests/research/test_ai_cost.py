from __future__ import annotations

import pytest

from research.ai.cost import (
    BudgetExceeded,
    CallRecord,
    CostLedger,
    TokenUsage,
    compute_cost,
    summarize_costs,
)
from research.ai.models import (
    BATCH_DISCOUNT,
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER,
    UnknownModelError,
    get_pricing,
)
from research.ai.settings import AISettings


def usage(**kwargs) -> TokenUsage:
    return TokenUsage(**kwargs)


def record(signal_id="s1", agent="technical", model="claude-haiku-4-5", **kwargs) -> CallRecord:
    token_usage = kwargs.pop("usage", usage(input_tokens=500, output_tokens=60))
    return CallRecord(
        signal_id=signal_id,
        agent=agent,
        model=model,
        usage=token_usage,
        cost_usd=kwargs.pop("cost_usd", compute_cost(token_usage, model)),
        latency_seconds=kwargs.pop("latency_seconds", 1.0),
        **kwargs,
    )


# --- pricing ---------------------------------------------------------------


def test_haiku_is_cheaper_than_sonnet_on_both_directions():
    haiku = get_pricing("claude-haiku-4-5")
    sonnet = get_pricing("claude-sonnet-5")
    assert haiku.input_per_mtok < sonnet.input_per_mtok
    assert haiku.output_per_mtok < sonnet.output_per_mtok


def test_haiku_does_not_advertise_effort_support():
    """Haiku 4.5 rejects output_config.effort -- the client relies on this
    flag to avoid sending a parameter that would 400 on the cheap path."""
    assert get_pricing("claude-haiku-4-5").supports_effort is False
    assert get_pricing("claude-sonnet-5").supports_effort is True


def test_unknown_model_is_fatal():
    with pytest.raises(UnknownModelError, match="no pricing entry"):
        get_pricing("claude-imaginary-9")


# --- cost arithmetic -------------------------------------------------------


def test_plain_input_output_cost():
    cost = compute_cost(usage(input_tokens=1_000_000, output_tokens=0), "claude-haiku-4-5")
    assert cost == pytest.approx(1.00)
    cost = compute_cost(usage(input_tokens=0, output_tokens=1_000_000), "claude-haiku-4-5")
    assert cost == pytest.approx(5.00)


def test_cache_writes_cost_more_and_reads_cost_far_less():
    written = compute_cost(usage(cache_creation_tokens=1_000_000), "claude-haiku-4-5")
    read = compute_cost(usage(cache_read_tokens=1_000_000), "claude-haiku-4-5")
    plain = compute_cost(usage(input_tokens=1_000_000), "claude-haiku-4-5")

    assert written == pytest.approx(plain * CACHE_WRITE_MULTIPLIER)
    assert read == pytest.approx(plain * CACHE_READ_MULTIPLIER)
    assert read < plain < written


def test_caching_is_cheaper_across_repeated_calls():
    """The economic claim behind prompt caching: paying 1.25x once then 0.1x
    thereafter beats paying 1.0x every time."""
    static_tokens = 3000
    calls = 50

    uncached = calls * compute_cost(usage(input_tokens=static_tokens), "claude-haiku-4-5")
    cached = compute_cost(
        usage(cache_creation_tokens=static_tokens), "claude-haiku-4-5"
    ) + (calls - 1) * compute_cost(usage(cache_read_tokens=static_tokens), "claude-haiku-4-5")

    assert cached < uncached
    assert cached / uncached < 0.20  # better than a 5x saving on the static prefix


def test_batch_halves_the_cost():
    standard = compute_cost(usage(input_tokens=100_000, output_tokens=1_000), "claude-sonnet-5")
    batched = compute_cost(
        usage(input_tokens=100_000, output_tokens=1_000), "claude-sonnet-5", batch=True
    )
    assert batched == pytest.approx(standard * BATCH_DISCOUNT)


def test_cheap_analyst_routing_costs_less_than_all_sonnet():
    """The routing decision: three analysts on Haiku + one Sonnet
    adjudicator, versus Sonnet everywhere."""
    per_agent = usage(input_tokens=2_000, output_tokens=80)
    routed = 3 * compute_cost(per_agent, "claude-haiku-4-5") + compute_cost(
        per_agent, "claude-sonnet-5"
    )
    all_sonnet = 4 * compute_cost(per_agent, "claude-sonnet-5")
    assert routed < all_sonnet


# --- ledger and budget ----------------------------------------------------


def test_ledger_accumulates_spend():
    ledger = CostLedger(budget_usd=10.0)
    ledger.record(record(cost_usd=1.5))
    ledger.record(record(cost_usd=2.5))
    assert ledger.spent_usd == pytest.approx(4.0)
    assert ledger.remaining_usd == pytest.approx(6.0)


def test_budget_stops_before_exceeding_not_after():
    ledger = CostLedger(budget_usd=1.0)
    ledger.record(record(cost_usd=0.95))
    # A projected call that would breach the cap must raise BEFORE spending.
    with pytest.raises(BudgetExceeded, match="cost limit reached"):
        ledger.check_budget(projected_cost=0.10)
    assert ledger.spent_usd == pytest.approx(0.95)  # nothing extra was spent


def test_no_budget_means_no_limit():
    ledger = CostLedger(budget_usd=None)
    ledger.record(record(cost_usd=9_999.0))
    ledger.check_budget(projected_cost=1_000.0)  # must not raise
    assert ledger.remaining_usd is None


def test_resumed_spend_counts_against_the_budget():
    """A restart must not reset the budget to zero."""
    ledger = CostLedger(budget_usd=5.0)
    ledger.add_external(4.9)  # recovered from the checkpoint
    with pytest.raises(BudgetExceeded):
        ledger.check_budget(projected_cost=0.2)


# --- cost summary and projections ----------------------------------------


def test_summary_computes_measured_stats_and_projections():
    records = [
        record(signal_id="s1", agent="technical", usage=usage(input_tokens=1000, output_tokens=50)),
        record(signal_id="s1", agent="news", usage=usage(input_tokens=800, output_tokens=40)),
        record(
            signal_id="s2", agent="technical",
            usage=usage(input_tokens=1000, cache_read_tokens=3000, output_tokens=50),
        ),
    ]
    summary = summarize_costs(records, signals_processed=2, full_run_signal_count=4_000)

    assert summary.api_calls == 3
    assert summary.signals_processed == 2
    assert summary.total_cost_usd == pytest.approx(sum(r.cost_usd for r in records))
    assert summary.avg_cost_per_signal == pytest.approx(summary.total_cost_usd / 2)
    # Projections scale the MEASURED per-signal average.
    assert summary.projected_1k == pytest.approx(summary.avg_cost_per_signal * 1_000)
    assert summary.projected_10k == pytest.approx(summary.avg_cost_per_signal * 10_000)
    assert summary.projected_full_run == pytest.approx(summary.avg_cost_per_signal * 4_000)
    assert summary.full_run_signal_count == 4_000


def test_summary_tracks_cache_effectiveness():
    """cache_read_ratio near zero is the signal that caching never engaged --
    usually because the static prefix is below the model's minimum."""
    cold = [record(usage=usage(input_tokens=4000, output_tokens=50))]
    warm = [
        record(usage=usage(input_tokens=500, cache_read_tokens=3500, output_tokens=50)),
        record(usage=usage(input_tokens=500, cache_read_tokens=3500, output_tokens=50)),
    ]
    assert summarize_costs(cold, 1).cache_read_ratio == pytest.approx(0.0)
    assert summarize_costs(cold, 1).cache_hit_rate == pytest.approx(0.0)

    warm_summary = summarize_costs(warm, 2)
    assert warm_summary.cache_read_ratio > 0.8
    assert warm_summary.cache_hit_rate == pytest.approx(1.0)


def test_summary_breaks_cost_down_by_agent_and_model():
    records = [
        record(agent="technical", model="claude-haiku-4-5"),
        record(agent="news", model="claude-haiku-4-5"),
        record(agent="final", model="claude-sonnet-5"),
    ]
    summary = summarize_costs(records, signals_processed=1)
    assert set(summary.cost_by_agent) == {"technical", "news", "final"}
    assert set(summary.cost_by_model) == {"claude-haiku-4-5", "claude-sonnet-5"}
    assert summary.calls_by_agent["technical"] == 1


def test_summary_counts_failed_calls_separately():
    records = [record(), record(error="timeout", cost_usd=0.0)]
    summary = summarize_costs(records, signals_processed=1)
    assert summary.api_calls == 2
    assert summary.failed_calls == 1


def test_summary_handles_no_records():
    summary = summarize_costs([], signals_processed=0)
    assert summary.total_cost_usd == 0.0
    assert summary.projected_1k == 0.0


# --- settings -------------------------------------------------------------


def test_settings_default_to_cheap_analysts_and_sonnet_adjudicator():
    settings = AISettings(anthropic_api_key="test")
    assert settings.model_for("technical") == "claude-haiku-4-5"
    assert settings.model_for("news") == "claude-haiku-4-5"
    assert settings.model_for("sentiment") == "claude-haiku-4-5"
    assert settings.model_for("final") == "claude-sonnet-5"


def test_settings_models_are_configurable():
    settings = AISettings(
        anthropic_api_key="test",
        ai_technical_model="claude-sonnet-5",
        ai_final_model="claude-opus-5",
    )
    assert settings.model_for("technical") == "claude-sonnet-5"
    assert settings.model_for("final") == "claude-opus-5"


def test_settings_reject_an_unpriced_model():
    with pytest.raises(Exception, match="no pricing entry"):
        AISettings(anthropic_api_key="test", ai_news_model="not-a-model")


def test_settings_default_fail_closed_to_no_trade():
    assert AISettings(anthropic_api_key="t").ai_fail_closed_policy.value == "NO_TRADE"


def test_output_tokens_are_capped_by_default():
    """Short structured outputs are a cost requirement, not a preference."""
    assert AISettings(anthropic_api_key="t").ai_max_output_tokens <= 500
