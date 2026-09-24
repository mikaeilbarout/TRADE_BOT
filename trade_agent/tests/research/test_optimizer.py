from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from research.config import BacktestConfig
from research.data.split import DataSplit, LeakageError, StrategySeal
from research.strategy.optimizer import (
    OptimizerConfig,
    WalkForwardOptimizer,
    freeze_strategy,
)
from research.strategy.donchian_scalp import DonchianParams, parameter_grid

UTC = timezone.utc
BASE = datetime(2021, 1, 4, 0, 0, tzinfo=UTC)

"""Tests for development-only parameter selection.

These use a seeded random-walk FIXTURE, not market data. The point is to
exercise the selection machinery and the leakage guards -- no conclusion about
the strategy is drawn from them, and none could be.
"""


def fixture_bars(n: int = 9000, seed: int = 11, drift: float = 0.03) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 1800 + np.cumsum(rng.normal(drift, 1.4, n))
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [BASE + timedelta(minutes=15 * i) for i in range(n)], utc=True
            ),
            "open": closes,
            "high": closes + rng.uniform(0.5, 3.0, n),
            "low": closes - rng.uniform(0.5, 3.0, n),
            "close": closes,
            "volume": rng.uniform(50, 200, n),
            "tick_count": rng.integers(40, 200, n),
            "bid_close": closes - 0.15,
            "ask_close": closes + 0.15,
            "spread_mean": np.full(n, 0.3),
            "spread_max": np.full(n, 0.5),
            "is_partial": [False] * n,
        }
    )


def small_grid() -> list[DonchianParams]:
    """A handful of the production bot's own tuning axes."""
    return parameter_grid(
        n_period=(10, 20),
        ema_trend_period=(30,),
        min_trend_strength_pct=(0.0, 0.5),
        atr_stop_mult=(2.0, 3.0),
        reward_risk_ratio=(1.5, 3.0),
    )


def permissive_config() -> OptimizerConfig:
    """Eligibility loose enough that the fixture yields eligible candidates."""
    return OptimizerConfig(
        folds=3,
        purge_bars=30,
        min_trades_per_fold=2,
        min_trades_total=8,
        min_profitable_fold_fraction=0.5,
        max_fold_drawdown_pct=40.0,
    )


@pytest.fixture
def split(tmp_path) -> DataSplit:
    return DataSplit(
        fixture_bars(), development_fraction=0.70, embargo_bars=200,
        seal_path=tmp_path / "seal.json",
    )


@pytest.fixture
def optimizer() -> WalkForwardOptimizer:
    return WalkForwardOptimizer(
        config=BacktestConfig(), optimizer_config=permissive_config()
    )


# --- development-only access ------------------------------------------------
def test_optimizer_only_ever_sees_development_bars(split, optimizer):
    """Every fold's evaluation window ends at or before the boundary."""
    development = split.development()
    folds = optimizer.folds(len(development))
    assert folds
    for fold in folds:
        assert fold.eval_end <= len(development)
        last = development["timestamp"].iloc[fold.eval_end - 1].to_pydatetime()
        assert last <= split.development_end


def test_optimizer_refuses_a_frame_reaching_past_the_boundary(split, optimizer):
    """Even a hand-built frame cannot smuggle test-period bars in."""
    with pytest.raises(LeakageError, match="past the development boundary"):
        optimizer._assert_development_only(fixture_bars(), split)


def test_optimize_cannot_be_handed_out_of_sample_data(split, optimizer):
    """The signature takes a DataSplit, so there is no argument for bars.

    This is the structural half of the guarantee: the check above can only
    fire if someone reaches past the public API.
    """
    import inspect

    parameters = inspect.signature(optimizer.optimize).parameters
    assert "split" in parameters
    assert not any(
        name in parameters for name in ("bars", "candles", "frame", "prepared")
    )


def test_folds_do_not_overlap_and_are_purged(split, optimizer):
    folds = optimizer.folds(split.boundary_index)
    for earlier, later in zip(folds, folds[1:]):
        assert later.eval_start >= earlier.eval_end
        assert later.eval_start - earlier.eval_end >= 0
    # The first window starts after a purge gap, never at bar 0.
    assert folds[0].eval_start >= optimizer.optimizer_config.purge_bars


def test_too_few_bars_for_the_fold_plan_is_an_error(optimizer):
    with pytest.raises(ValueError, match="too short"):
        optimizer.folds(50)


# --- robustness objective ---------------------------------------------------
def test_objective_prefers_consistency_at_equal_median():
    """Same median, different spread: the steady candidate wins.

    This is the property that makes the objective robustness-oriented rather
    than return-oriented.
    """
    optimizer = WalkForwardOptimizer(
        config=BacktestConfig(),
        optimizer_config=OptimizerConfig(stability_penalty=0.5, drawdown_floor_pct=1.0),
    )
    steady = [1.0, 1.0, 1.0, 1.0]
    spiky = [0.0, 1.0, 1.0, 2.0]
    assert statistics.median(steady) == statistics.median(spiky)
    assert optimizer.objective_from_scores(steady) > optimizer.objective_from_scores(spiky)


def test_objective_uses_the_median_so_one_outlier_fold_cannot_carry_a_candidate():
    optimizer = WalkForwardOptimizer(
        config=BacktestConfig(), optimizer_config=OptimizerConfig(stability_penalty=0.0)
    )
    # A single enormous fold raises the MEAN far above the median; the
    # objective must ignore it.
    assert optimizer.objective_from_scores([0.1, 0.1, 0.1, 50.0]) == pytest.approx(0.1)


def test_objective_of_no_folds_is_the_ineligible_sentinel():
    # Not literal -inf: that value round-trips through JSON as null (JSON has
    # no Infinity) and then fails to re-validate as a float. The sentinel is
    # finite but still sorts below every real objective score.
    from research.strategy.optimizer import INELIGIBLE_OBJECTIVE_SCORE

    optimizer = WalkForwardOptimizer(config=BacktestConfig())
    assert optimizer.objective_from_scores([]) == INELIGIBLE_OBJECTIVE_SCORE
    assert optimizer.objective_from_scores([]) < 0.0


def test_fold_score_is_return_per_unit_of_drawdown(optimizer):
    modest = optimizer.fold_score(return_pct=8.0, max_drawdown_pct=4.0)
    reckless = optimizer.fold_score(return_pct=12.0, max_drawdown_pct=20.0)
    assert modest > reckless  # higher return, worse score


def test_drawdown_floor_bounds_the_score(optimizer):
    """A fold with no drawdown cannot score infinitely well."""
    assert optimizer.fold_score(5.0, 0.0) == pytest.approx(
        5.0 / optimizer.optimizer_config.drawdown_floor_pct
    )


def test_selection_is_not_by_highest_total_profit(split, optimizer):
    report = optimizer.optimize(split, candidates=small_grid())
    eligible = [c for c in report.candidates if c.eligible]
    if len(eligible) < 2:
        pytest.skip("fixture produced too few eligible candidates to compare ranking")

    winner = eligible[0]
    best_raw_profit = max(
        eligible, key=lambda c: sum(f.net_profit for f in c.folds)
    )
    # The winner is chosen by the objective. It MAY coincide with the highest
    # raw profit, but the ranking must follow the objective, not profit.
    assert winner.objective_score == max(c.objective_score for c in eligible)
    assert winner.rank == 1
    assert best_raw_profit.objective_score <= winner.objective_score


# --- eligibility filters ----------------------------------------------------
def test_thin_candidates_are_excluded_with_a_reason(split):
    strict = WalkForwardOptimizer(
        config=BacktestConfig(),
        optimizer_config=OptimizerConfig(
            folds=3, purge_bars=30, min_trades_per_fold=500, min_trades_total=5000
        ),
    )
    report = strict.optimize(split, candidates=small_grid())
    assert report.eligible_candidates == 0
    assert report.selected_params is None
    assert "no candidate met the eligibility filters" in report.selection_reason
    assert all(c.ineligible_reasons for c in report.candidates)
    assert any("trades" in r for c in report.candidates for r in c.ineligible_reasons)


def test_every_candidate_is_recorded_even_when_ineligible(split, optimizer):
    grid = small_grid()
    report = optimizer.optimize(split, candidates=grid)
    assert report.candidates_evaluated == len(grid)
    assert len(report.candidates) == len(grid)
    for candidate in report.candidates:
        assert candidate.params_hash
        assert candidate.folds  # per-fold metrics for all of them
        for fold in candidate.folds:
            assert fold.trades >= 0
            assert fold.start <= fold.end


def test_empty_grid_is_an_error(split, optimizer):
    with pytest.raises(ValueError, match="parameter grid is empty"):
        optimizer.optimize(split, candidates=[])


# --- sealing ----------------------------------------------------------------
def test_freeze_records_everything_the_seal_must_carry(split, optimizer, tmp_path):
    report = optimizer.optimize(split, candidates=small_grid(), dataset_hash="a" * 64)
    if report.selected_params is None:
        pytest.skip("fixture produced no eligible candidate")

    seal_path = tmp_path / "seal.json"
    seal = freeze_strategy(report, split, seal_path, optimizer.optimizer_config)

    assert seal.params == report.selected_params
    assert seal.params_hash
    assert seal.development_start == split.development_start
    assert seal.development_end == split.development_end
    assert seal.development_metrics["per_fold"]
    assert seal.optimizer_config["folds"] == optimizer.optimizer_config.folds
    assert seal.candidates_evaluated == report.candidates_evaluated
    assert seal.eligible_candidates == report.eligible_candidates
    assert seal.selection_criterion
    assert seal.selection_reason
    assert seal.dataset_hash == "a" * 64
    assert seal_path.exists()
    # Round-trips.
    assert StrategySeal.load(seal_path).params_hash == seal.params_hash


def test_freeze_refuses_when_nothing_was_selected(split, tmp_path):
    strict = WalkForwardOptimizer(
        config=BacktestConfig(),
        optimizer_config=OptimizerConfig(folds=3, purge_bars=30, min_trades_total=99999),
    )
    report = strict.optimize(split, candidates=small_grid())
    with pytest.raises(ValueError, match="cannot freeze"):
        freeze_strategy(report, split, tmp_path / "seal.json", strict.optimizer_config)


def test_out_of_sample_stays_locked_until_the_seal_exists(split, optimizer, tmp_path):
    with pytest.raises(LeakageError, match="sealed"):
        split.out_of_sample()

    report = optimizer.optimize(split, candidates=small_grid())
    if report.selected_params is None:
        pytest.skip("fixture produced no eligible candidate")
    freeze_strategy(report, split, tmp_path / "seal.json", optimizer.optimizer_config)

    frame = split.out_of_sample(report.selected_params)
    assert not frame.empty
    assert "is_test" in frame.columns


def test_out_of_sample_refuses_unsealed_parameters(split, optimizer, tmp_path):
    report = optimizer.optimize(split, candidates=small_grid())
    if report.selected_params is None:
        pytest.skip("fixture produced no eligible candidate")
    freeze_strategy(report, split, tmp_path / "seal.json", optimizer.optimizer_config)

    tampered = dict(report.selected_params)
    tampered["reward_risk_ratio"] = 9.0
    with pytest.raises(LeakageError, match="differ from the sealed strategy"):
        split.out_of_sample(tampered)


def test_seal_refuses_a_different_dataset(split, optimizer, tmp_path):
    report = optimizer.optimize(split, candidates=small_grid(), dataset_hash="a" * 64)
    if report.selected_params is None:
        pytest.skip("fixture produced no eligible candidate")
    seal = freeze_strategy(report, split, tmp_path / "seal.json", optimizer.optimizer_config)

    seal.verify_dataset("a" * 64)  # same data: fine
    with pytest.raises(LeakageError, match="does not match the sealed one"):
        seal.verify_dataset("b" * 64)


def test_reseal_is_refused_unless_explicitly_allowed(split, optimizer, tmp_path):
    from research.data.split import guard_reseal

    report = optimizer.optimize(split, candidates=small_grid())
    if report.selected_params is None:
        pytest.skip("fixture produced no eligible candidate")
    seal_path = tmp_path / "seal.json"
    freeze_strategy(report, split, seal_path, optimizer.optimizer_config)

    with pytest.raises(LeakageError, match="already exists"):
        guard_reseal(seal_path, reason="tuning", allow=False)

    guard_reseal(seal_path, reason="accepted invalidation", allow=True)
    assert StrategySeal.load(seal_path).reseal_history[0]["reason"] == (
        "accepted invalidation"
    )
