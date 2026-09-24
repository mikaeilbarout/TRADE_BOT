from __future__ import annotations

import pytest

from research.experiment import (
    AI_ONLY_MANIFEST_FIELDS,
    EXECUTION_ASSUMPTIONS,
    ExperimentConfig,
    ManifestMismatch,
    assert_ab_identical,
    assert_manifests_match,
    load_experiment,
    save_experiment,
)
from tests.conftest import make_settings

"""A and B must be the same experiment apart from the AI layer."""


# --- the shared fingerprint -------------------------------------------------
def test_fingerprint_covers_every_required_category():
    """The categories the experiment design requires to be identical."""
    fingerprint = ExperimentConfig().shared_fingerprint(make_settings())
    for category in (
        "initial_capital",
        "risk",
        "spread",
        "slippage",
        "commission",
        "position_limits",
        "daily_limits",
        "instrument",
        "execution_assumptions",
        "strategy_parameters",
    ):
        assert category in fingerprint, category
        assert fingerprint[category], f"{category} is empty"


def test_identical_configs_pass():
    config = ExperimentConfig()
    assert_ab_identical(config, config, make_settings())


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda c: setattr(c.backtest.risk, "initial_balance", 50_000.0), "initial_capital"),
        (lambda c: setattr(c.backtest.risk, "risk_per_trade_pct", 1.0), "risk"),
        (lambda c: setattr(c.backtest.costs, "fallback_spread_price", 0.9), "spread"),
        (lambda c: setattr(c.backtest.costs, "slippage_price", 0.5), "slippage"),
        (lambda c: setattr(c.backtest.costs, "stop_slippage_price", 0.9), "slippage"),
        (
            lambda c: setattr(c.backtest.costs, "commission_per_lot_per_side", 9.0),
            "commission",
        ),
        (
            lambda c: setattr(c.backtest.risk, "max_concurrent_positions", 3),
            "position_limits",
        ),
        (lambda c: setattr(c.backtest.risk, "max_trades_per_day", 20), "daily_limits"),
        (lambda c: setattr(c.backtest.instrument, "contract_size", 10.0), "instrument"),
        (
            lambda c: setattr(c.backtest.risk, "modified_limit_expiry_bars", 99),
            "execution_assumptions",
        ),
        (lambda c: setattr(c.backtest, "timeframe_minutes", 5), "execution_assumptions"),
        (
            lambda c: setattr(c.strategy_params, "atr_stop_mult", 2.5),
            "strategy_parameters",
        ),
        (lambda c: setattr(c.policy, "min_confidence", 0.5), "policy"),
        (lambda c: setattr(c.backtest, "random_seed", 999), "random_seed"),
    ],
)
def test_any_shared_difference_is_caught(mutate, expected):
    """Each category of drift is detected and named."""
    baseline = ExperimentConfig()
    other = baseline.model_copy(deep=True)
    mutate(other)
    with pytest.raises(ManifestMismatch) as error:
        assert_ab_identical(baseline, other, make_settings())
    assert expected in str(error.value)


def test_fingerprint_hash_changes_when_anything_shared_changes():
    baseline = ExperimentConfig()
    before = baseline.fingerprint_hash(make_settings())
    changed = baseline.model_copy(deep=True)
    changed.backtest.costs.commission_per_lot_per_side = 4.0
    assert changed.fingerprint_hash(make_settings()) != before


def test_execution_assumptions_are_recorded_and_versioned():
    """The fill model is part of the comparison, so it is pinned."""
    fingerprint = ExperimentConfig().shared_fingerprint(make_settings())
    assumptions = fingerprint["execution_assumptions"]
    assert assumptions["semantics_version"] == EXECUTION_ASSUMPTIONS["semantics_version"]
    assert assumptions["entry_delay"] == "next_bar_open"
    assert assumptions["stop_assumed_first_when_both_touched"] is True


# --- derived objects --------------------------------------------------------
def test_both_arms_get_identically_configured_executors():
    config = ExperimentConfig()
    first, second = config.executor(make_settings()), config.executor(make_settings())
    assert first.config == second.config
    assert first.allow_modify == second.allow_modify
    assert first.counterfactual_balance == second.counterfactual_balance
    assert first.counterfactual_balance == config.backtest.risk.initial_balance


def test_risk_settings_are_derived_not_read_independently():
    """Changing the research risk model moves the risk engine's limits too."""
    config = ExperimentConfig()
    config.backtest.risk.max_concurrent_positions = 4
    config.backtest.risk.max_trades_per_day = 11
    config.backtest.risk.risk_per_trade_pct = 0.25

    settings = config.risk_settings(make_settings())
    assert settings.max_simultaneous_positions == 4
    assert settings.max_trades_per_day == 11
    assert settings.max_risk_per_trade_pct == 0.25


def test_replay_only_neutralizes_the_two_wall_clock_rules():
    settings = ExperimentConfig().risk_settings(make_settings())
    assert settings.max_signal_age_seconds >= 10**9
    assert settings.market_data_max_staleness_seconds >= 10**9
    # Everything monetary stays live.
    assert settings.min_risk_reward_ratio == 1.0
    assert settings.max_stop_loss_distance_pct > 0
    assert settings.max_spread_pct > 0
    assert settings.max_leverage > 0


# --- manifests --------------------------------------------------------------
def test_manifests_for_the_two_arms_match():
    config = ExperimentConfig()
    assert_manifests_match(
        config.manifest("a", "baseline"), config.manifest("b", "ai")
    )


def test_ai_fields_are_allowed_to_differ():
    config = ExperimentConfig()
    ai = config.manifest("b", "ai")
    assert ai.llm_model  # the AI arm names a model
    assert ai.agent_settings
    baseline = config.manifest("a", "baseline")
    assert baseline.llm_model is None
    assert baseline.agent_settings is None
    # ...and that difference does not fail the check.
    assert_manifests_match(baseline, ai)
    assert "llm_model" in AI_ONLY_MANIFEST_FIELDS


def test_manifest_mismatch_outside_ai_fields_is_fatal():
    config = ExperimentConfig()
    baseline = config.manifest("a", "baseline")

    drifted = config.model_copy(deep=True)
    drifted.backtest.risk.initial_balance = 25_000.0
    ai = drifted.manifest("b", "ai")

    with pytest.raises(ManifestMismatch, match="initial_balance"):
        assert_manifests_match(baseline, ai)


def test_strategy_param_drift_between_arms_is_fatal():
    config = ExperimentConfig()
    baseline = config.manifest("a", "baseline")
    drifted = config.model_copy(deep=True)
    drifted.strategy_params.n_period = 33
    with pytest.raises(ManifestMismatch, match="n_period"):
        assert_manifests_match(baseline, drifted.manifest("b", "ai"))


def test_provenance_fields_are_not_compared():
    """Run id, timestamp, git revision and host always differ."""
    config = ExperimentConfig()
    first = config.manifest("run-one", "baseline")
    second = config.manifest("run-two", "baseline")
    second.git_revision = "deadbeef"
    second.platform = "some-other-host"
    assert_manifests_match(first, second)


def test_manifest_carries_limitations():
    from research.report.limitations import as_dicts, limitations_for

    config = ExperimentConfig()
    manifest = config.manifest(
        "b", "ai", limitations=as_dicts(limitations_for("ai", ["news"]))
    )
    codes = [entry["code"] for entry in manifest.known_limitations]
    assert "MODEL_KNOWLEDGE_LEAKAGE" in codes
    assert "DATASETS_MISSING" in codes


# --- persistence ------------------------------------------------------------
def test_config_round_trips_through_disk(tmp_path):
    config = ExperimentConfig()
    config.policy.min_confidence = 0.77
    config.backtest.risk.max_trades_per_day = 9
    path = save_experiment(config, tmp_path / "experiment.json")

    loaded = load_experiment(path)
    assert loaded.policy.min_confidence == 0.77
    assert loaded.backtest.risk.max_trades_per_day == 9
    assert loaded.fingerprint_hash(make_settings()) == config.fingerprint_hash(
        make_settings()
    )


def test_missing_config_file_yields_defaults(tmp_path):
    assert load_experiment(tmp_path / "absent.json").name == ExperimentConfig().name


def test_policy_weights_must_sum_to_one():
    from research.experiment import PolicyConfig

    with pytest.raises(ValueError, match="must sum to 1.0"):
        PolicyConfig(weight_news=0.9, weight_sentiment=0.9)
