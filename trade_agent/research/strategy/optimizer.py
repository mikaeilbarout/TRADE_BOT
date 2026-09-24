from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
from pydantic import BaseModel, Field

from research.backtest.engine import BacktestEngine
from research.backtest.metrics import compute_metrics
from research.config import BacktestConfig
from research.data.split import DataSplit, LeakageError, StrategySeal
from research.manifest import sha256_obj
from research.strategy.donchian_scalp import (
    DonchianParams,
    DonchianScalpStrategy,
    parameter_grid,
)

# Sentinel for "no usable objective" (an ineligible or scoreless candidate).
# Deliberately finite: actual -inf round-trips through JSON as null (standard
# JSON has no Infinity), which then fails to re-validate as a float. This
# value sorts below every real objective score, same as -inf would.
INELIGIBLE_OBJECTIVE_SCORE = -1e18

"""Walk-forward parameter selection, structurally confined to the first 70%.

Two things this module refuses to do, both enforced in code rather than
promised in a comment:

1. **It cannot see out-of-sample data.** The optimizer takes a `DataSplit`
   and calls `split.development()` itself. It never accepts a bare frame, so
   there is no argument through which test-period bars could arrive. A frame
   whose last timestamp is past the development boundary raises
   `LeakageError`.

2. **It does not pick the highest total profit.** Maximum profit over one
   fixed window is the single easiest thing to overfit: with a few hundred
   candidates, some parameter set always wins by luck. Selection is by a
   robustness objective across chronological folds, with hard eligibility
   filters applied first, and the winner must be decent in most folds rather
   than spectacular in one.

The fold design is expanding-window walk-forward WITHIN the development set:

    fold 1:  [=== train-equivalent ===][ eval 1 ]
    fold 2:  [=== train-equivalent =============][ eval 2 ]
    fold 3:  [=== train-equivalent =======================][ eval 3 ]

The strategy has no fitted state -- parameters are evaluated, not learned --
so a fold is an out-of-sample-shaped evaluation window, and the bars before
it serve only as indicator warm-up. Each evaluation window is preceded by an
embargo gap so indicator state cannot straddle the boundary, and the windows
never overlap, so a candidate that only works in one regime cannot pass by
appearing in several folds at once.
"""


class OptimizerConfig(BaseModel):
    """Recorded verbatim in the seal, so the selection can be re-run."""

    folds: int = 4
    # Bars dropped between the warm-up region and each evaluation window.
    purge_bars: int = 50
    # Eligibility: a candidate with too few trades has an unmeasurable edge,
    # however good its numbers look.
    min_trades_per_fold: int = 5
    min_trades_total: int = 40
    # A candidate must be profitable in at least this fraction of folds.
    min_profitable_fold_fraction: float = 0.6
    # Reject candidates whose worst fold drawdown is beyond this, whatever
    # their return.
    max_fold_drawdown_pct: float = 25.0
    # Objective: median fold score minus a penalty on the spread between
    # folds. The penalty is what makes this a robustness objective rather
    # than an average-return objective.
    stability_penalty: float = 0.5
    # Floor on the drawdown used in the MAR-style ratio, so a fold that
    # happened not to draw down cannot produce an unbounded score.
    drawdown_floor_pct: float = 1.0
    objective: str = "median_mar_minus_stability_penalty"

    def describe(self) -> str:
        return (
            f"{self.objective}: folds={self.folds}, purge={self.purge_bars} bars, "
            f"min trades/fold={self.min_trades_per_fold}, "
            f"min trades total={self.min_trades_total}, "
            f"min profitable folds={self.min_profitable_fold_fraction:.0%}, "
            f"max fold drawdown={self.max_fold_drawdown_pct}%, "
            f"stability penalty={self.stability_penalty}"
        )


class FoldMetrics(BaseModel):
    fold: int
    start: datetime
    end: datetime
    bars: int
    trades: int
    net_profit: float
    return_pct: float
    win_rate: float
    profit_factor: float | None
    max_drawdown_pct: float
    expectancy_r: float
    fold_score: float


class CandidateResult(BaseModel):
    """One parameter set's full record. Every candidate is kept, eligible or
    not, so the selection can be audited rather than taken on trust."""

    rank: int = 0
    params: dict
    params_hash: str
    folds: list[FoldMetrics] = Field(default_factory=list)
    total_trades: int = 0
    profitable_folds: int = 0
    median_fold_score: float = 0.0
    mean_fold_score: float = 0.0
    fold_score_stdev: float = 0.0
    worst_fold_drawdown_pct: float = 0.0
    objective_score: float = INELIGIBLE_OBJECTIVE_SCORE
    eligible: bool = False
    ineligible_reasons: list[str] = Field(default_factory=list)


class OptimizationReport(BaseModel):
    strategy_name: str
    development_start: datetime
    development_end: datetime
    development_bars: int
    candidates_evaluated: int
    eligible_candidates: int
    optimizer_config: dict
    selection_criterion: str
    selected_params: dict | None = None
    selection_reason: str = ""
    selected_development_metrics: dict = Field(default_factory=dict)
    candidates: list[CandidateResult] = Field(default_factory=list)
    dataset_hash: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def top(self, n: int = 10) -> list[CandidateResult]:
        return self.candidates[:n]


@dataclass
class _Fold:
    index: int
    eval_start: int
    eval_end: int  # exclusive
    warmup_start: int


@dataclass
class WalkForwardOptimizer:
    """Selects strategy parameters using development data only."""

    config: BacktestConfig
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)

    def folds(self, total_bars: int) -> list[_Fold]:
        """Expanding-window folds with non-overlapping evaluation windows.

        The development set is divided into `folds + 1` blocks; block 0 is
        warm-up only and each later block is one evaluation window, preceded
        by a purge gap. Dividing by bar count rather than by date keeps the
        folds comparable when the data has gaps.
        """
        cfg = self.optimizer_config
        blocks = cfg.folds + 1
        block = total_bars // blocks
        if block <= cfg.purge_bars:
            raise ValueError(
                f"development set of {total_bars} bars is too short for "
                f"{cfg.folds} folds with a {cfg.purge_bars}-bar purge"
            )

        result: list[_Fold] = []
        for i in range(cfg.folds):
            eval_start = block * (i + 1) + cfg.purge_bars
            eval_end = block * (i + 2) if i < cfg.folds - 1 else total_bars
            if eval_start >= eval_end:
                continue
            result.append(
                _Fold(index=i + 1, eval_start=eval_start, eval_end=eval_end, warmup_start=0)
            )
        return result

    # --- scoring ----------------------------------------------------------
    def fold_score(self, return_pct: float, max_drawdown_pct: float) -> float:
        """A MAR-style ratio: return per unit of drawdown suffered.

        Return alone rewards whoever took the most risk. Dividing by the
        drawdown the fold actually produced makes a candidate that made 8%
        through a 4% dip score above one that made 12% through a 20% dip,
        which is the preference an account actually has.
        """
        floor = self.optimizer_config.drawdown_floor_pct
        return return_pct / max(max_drawdown_pct, floor)

    def objective_from_scores(self, scores: list[float]) -> float:
        """Combine per-fold scores into the selection objective.

        Median rather than mean, so one spectacular fold cannot carry a
        candidate, minus a penalty on the spread between folds. Two candidates
        with the same median are separated by consistency, which is the
        property that survives out of sample.
        """
        if not scores:
            return INELIGIBLE_OBJECTIVE_SCORE
        spread = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        return statistics.median(scores) - self.optimizer_config.stability_penalty * spread

    def evaluate_candidate(
        self, prepared_by_params: pd.DataFrame, params: DonchianParams, folds: list[_Fold]
    ) -> CandidateResult:
        cfg = self.optimizer_config
        strategy = DonchianScalpStrategy(params, symbol=self.config.symbol)
        engine = BacktestEngine(self.config)
        signals = strategy.generate(prepared_by_params)

        fold_metrics: list[FoldMetrics] = []
        for fold in folds:
            window = prepared_by_params.iloc[fold.warmup_start : fold.eval_end].reset_index(
                drop=True
            )
            offset = fold.warmup_start
            in_fold = [
                s.model_copy(update={"bar_index": s.bar_index - offset})
                for s in signals
                if fold.eval_start <= s.bar_index < fold.eval_end
            ]
            result = engine.run(
                window, in_fold, run_id=f"fold{fold.index}", run_kind="optimization"
            )
            metrics = compute_metrics(result)
            score = self.fold_score(metrics.return_pct, metrics.max_drawdown_pct)
            fold_metrics.append(
                FoldMetrics(
                    fold=fold.index,
                    start=prepared_by_params["timestamp"]
                    .iloc[fold.eval_start]
                    .to_pydatetime(),
                    end=prepared_by_params["timestamp"]
                    .iloc[fold.eval_end - 1]
                    .to_pydatetime(),
                    bars=fold.eval_end - fold.eval_start,
                    trades=metrics.total_trades,
                    net_profit=metrics.net_profit,
                    return_pct=metrics.return_pct,
                    win_rate=metrics.win_rate,
                    profit_factor=metrics.profit_factor,
                    max_drawdown_pct=metrics.max_drawdown_pct,
                    expectancy_r=metrics.expectancy_r,
                    fold_score=score,
                )
            )

        scores = [f.fold_score for f in fold_metrics] or [0.0]
        total_trades = sum(f.trades for f in fold_metrics)
        profitable = sum(1 for f in fold_metrics if f.net_profit > 0)
        stdev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        median = statistics.median(scores)
        worst_dd = max((f.max_drawdown_pct for f in fold_metrics), default=0.0)

        reasons: list[str] = []
        if total_trades < cfg.min_trades_total:
            reasons.append(
                f"only {total_trades} trades across folds (minimum "
                f"{cfg.min_trades_total})"
            )
        thin = [f.fold for f in fold_metrics if f.trades < cfg.min_trades_per_fold]
        if thin:
            reasons.append(
                f"fewer than {cfg.min_trades_per_fold} trades in fold(s) "
                f"{thin}: edge is unmeasurable there"
            )
        if fold_metrics and profitable / len(fold_metrics) < cfg.min_profitable_fold_fraction:
            reasons.append(
                f"profitable in only {profitable}/{len(fold_metrics)} folds "
                f"(minimum {cfg.min_profitable_fold_fraction:.0%})"
            )
        if worst_dd > cfg.max_fold_drawdown_pct:
            reasons.append(
                f"worst fold drawdown {worst_dd:.1f}% exceeds "
                f"{cfg.max_fold_drawdown_pct}%"
            )

        eligible = not reasons
        objective = self.objective_from_scores(scores) if eligible else INELIGIBLE_OBJECTIVE_SCORE

        return CandidateResult(
            params=params.to_dict(),
            params_hash=sha256_obj(params.to_dict()),
            folds=fold_metrics,
            total_trades=total_trades,
            profitable_folds=profitable,
            median_fold_score=median,
            mean_fold_score=sum(scores) / len(scores),
            fold_score_stdev=stdev,
            worst_fold_drawdown_pct=worst_dd,
            objective_score=objective,
            eligible=eligible,
            ineligible_reasons=reasons,
        )

    # --- the run ----------------------------------------------------------
    def optimize(
        self,
        split: DataSplit,
        candidates: list[DonchianParams] | None = None,
        dataset_hash: str | None = None,
    ) -> OptimizationReport:
        """Search the grid on development data and report every candidate.

        Takes the `DataSplit`, not a frame: the development slice is obtained
        here, so no caller can hand this method out-of-sample bars.
        """
        development = split.development()
        self._assert_development_only(development, split)

        grid = candidates if candidates is not None else parameter_grid()
        if not grid:
            raise ValueError("parameter grid is empty: nothing to optimize")

        folds = self.folds(len(development))
        results: list[CandidateResult] = []
        for params in grid:
            strategy = DonchianScalpStrategy(params, symbol=self.config.symbol)
            prepared = strategy.prepare(development)
            results.append(self.evaluate_candidate(prepared, params, folds))

        results.sort(key=lambda c: (c.objective_score, c.total_trades), reverse=True)
        for rank, candidate in enumerate(results, start=1):
            candidate.rank = rank

        eligible = [c for c in results if c.eligible]
        report = OptimizationReport(
            strategy_name=DonchianScalpStrategy.name,
            development_start=split.development_start,
            development_end=split.development_end,
            development_bars=len(development),
            candidates_evaluated=len(results),
            eligible_candidates=len(eligible),
            optimizer_config=self.optimizer_config.model_dump(),
            selection_criterion=self.optimizer_config.describe(),
            dataset_hash=dataset_hash,
            candidates=results,
        )

        if not eligible:
            report.selection_reason = (
                "no candidate met the eligibility filters (trade counts, profitable-fold "
                "fraction, drawdown cap). Nothing is selected and no seal is written: "
                "an ineligible best-of-a-bad-grid choice is not a strategy."
            )
            return report

        winner = eligible[0]
        report.selected_params = winner.params
        report.selected_development_metrics = self._development_metrics(winner)
        runner_up = eligible[1] if len(eligible) > 1 else None
        report.selection_reason = (
            f"highest robustness objective {winner.objective_score:.3f} "
            f"(median fold MAR {winner.median_fold_score:.3f} minus "
            f"{self.optimizer_config.stability_penalty} x fold spread "
            f"{winner.fold_score_stdev:.3f}); profitable in "
            f"{winner.profitable_folds}/{len(winner.folds)} folds on "
            f"{winner.total_trades} trades, worst fold drawdown "
            f"{winner.worst_fold_drawdown_pct:.1f}%"
            + (
                f". Runner-up scored {runner_up.objective_score:.3f}."
                if runner_up
                else ". No other candidate was eligible."
            )
        )
        return report

    @staticmethod
    def _development_metrics(candidate: CandidateResult) -> dict:
        return {
            "objective_score": candidate.objective_score,
            "median_fold_score": candidate.median_fold_score,
            "mean_fold_score": candidate.mean_fold_score,
            "fold_score_stdev": candidate.fold_score_stdev,
            "total_trades": candidate.total_trades,
            "profitable_folds": candidate.profitable_folds,
            "folds": len(candidate.folds),
            "worst_fold_drawdown_pct": candidate.worst_fold_drawdown_pct,
            "per_fold": [f.model_dump(mode="json") for f in candidate.folds],
        }

    @staticmethod
    def _assert_development_only(frame: pd.DataFrame, split: DataSplit) -> None:
        """Refuse to run on anything that reaches past the boundary.

        `split.development()` cannot produce such a frame, so this guards
        against a future caller passing a hand-built frame -- the check is
        cheap and the failure it prevents would silently invalidate the whole
        experiment.
        """
        if frame.empty:
            raise ValueError("development set is empty")
        last = frame["timestamp"].iloc[-1].to_pydatetime()
        boundary = split.development_end
        if last > boundary:
            raise LeakageError(
                f"optimizer received bars up to {last.isoformat()}, past the "
                f"development boundary {boundary.isoformat()}. Parameter selection "
                "may only see the first "
                f"{split.summary()['development_fraction_actual']:.0%} of history."
            )


def freeze_strategy(
    report: OptimizationReport,
    split: DataSplit,
    seal_path,
    optimizer_config: OptimizerConfig,
) -> StrategySeal:
    """Write the seal that unlocks out-of-sample access.

    Called only after `optimize`, and only with that run's report: the seal
    records the parameters, their hash, the development window and metrics,
    the optimizer configuration, how many candidates were considered, the
    selection criterion and the dataset hash -- everything needed to show
    the choice was made before the test period was ever read.
    """
    if report.selected_params is None:
        raise ValueError(
            "cannot freeze: the optimization selected no parameters. "
            f"{report.selection_reason}"
        )

    seal = StrategySeal.create(
        strategy_name=report.strategy_name,
        params=report.selected_params,
        split=split,
        development_metrics=report.selected_development_metrics,
        selection_criterion=report.selection_criterion,
        candidates_evaluated=report.candidates_evaluated,
        optimizer_config=optimizer_config.model_dump(),
        dataset_hash=report.dataset_hash,
        selection_reason=report.selection_reason,
        eligible_candidates=report.eligible_candidates,
    )
    seal.save(seal_path)
    return seal
