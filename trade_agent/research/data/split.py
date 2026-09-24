from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from research.manifest import sha256_obj


class LeakageError(Exception):
    """Raised on any attempt to touch out-of-sample data before the strategy
    is frozen, or to re-optimize after seeing test results.

    This is the mechanism that makes "no data leakage" a property of the code
    rather than a promise in a document.
    """


class StrategySeal(BaseModel):
    """An immutable record that the strategy was frozen BEFORE the
    out-of-sample period was ever read.

    The seal stores the parameters and their hash. Out-of-sample access is
    refused unless a seal exists, and any later re-optimization has to
    explicitly break the seal -- which is recorded in `reseal_history`, so
    "we tuned it after seeing the test set" can never happen silently.
    """

    strategy_name: str
    params: dict
    params_hash: str
    sealed_at: datetime
    development_start: datetime
    development_end: datetime
    out_of_sample_start: datetime
    out_of_sample_end: datetime
    development_metrics: dict = Field(default_factory=dict)
    selection_criterion: str = ""
    candidates_evaluated: int = 0
    # How many candidates actually cleared the eligibility filters. A seal
    # written from a grid where only one candidate qualified is a weaker
    # claim than one chosen from fifty, and the report should say so.
    eligible_candidates: int = 0
    # Why THIS candidate, in words, so the choice can be challenged without
    # re-running the search.
    selection_reason: str = ""
    # The optimizer's own configuration: fold count, purge, eligibility
    # filters, objective. Without it "median MAR minus stability penalty" is
    # not reproducible.
    optimizer_config: dict = Field(default_factory=dict)
    # Content hash of the candle dataset the selection ran on. A seal that
    # verifies against different data is not a seal.
    dataset_hash: str | None = None
    reseal_history: list[dict] = Field(default_factory=list)

    @classmethod
    def create(
        cls,
        strategy_name: str,
        params: dict,
        split: "DataSplit",
        development_metrics: dict | None = None,
        selection_criterion: str = "",
        candidates_evaluated: int = 0,
        optimizer_config: dict | None = None,
        dataset_hash: str | None = None,
        selection_reason: str = "",
        eligible_candidates: int = 0,
    ) -> "StrategySeal":
        return cls(
            strategy_name=strategy_name,
            params=params,
            params_hash=sha256_obj(params),
            sealed_at=datetime.now(timezone.utc),
            development_start=split.development_start,
            development_end=split.development_end,
            out_of_sample_start=split.out_of_sample_start,
            out_of_sample_end=split.out_of_sample_end,
            development_metrics=development_metrics or {},
            selection_criterion=selection_criterion,
            candidates_evaluated=candidates_evaluated,
            eligible_candidates=eligible_candidates,
            selection_reason=selection_reason,
            optimizer_config=optimizer_config or {},
            dataset_hash=dataset_hash,
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, default=str), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "StrategySeal":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    def verify(self, params: dict) -> None:
        """Confirm the params about to be run are exactly the sealed ones."""
        if sha256_obj(params) != self.params_hash:
            raise LeakageError(
                "parameters differ from the sealed strategy: the out-of-sample test "
                "must run the frozen parameters. Sealed hash "
                f"{self.params_hash[:12]}, provided {sha256_obj(params)[:12]}."
            )

    def verify_dataset(self, dataset_hash: str | None) -> None:
        """Confirm the data is the data the strategy was sealed against.

        Skipped when the seal carries no dataset hash (older seals), because
        refusing to run is worse than reporting the gap -- but a MISMATCH is
        always fatal: parameters frozen on one candle set and tested on
        another prove nothing.
        """
        if self.dataset_hash is None or dataset_hash is None:
            return
        if dataset_hash != self.dataset_hash:
            raise LeakageError(
                "candle dataset does not match the sealed one: the strategy was "
                f"frozen against {self.dataset_hash[:12]} but the run supplied "
                f"{dataset_hash[:12]}. Re-import the data or re-run development."
            )


class DataSplit:
    """Chronological 70/30 split over a bar series.

    The boundary is computed from the data itself, by bar count, so the
    development period always contains exactly `development_fraction` of the
    available bars regardless of gaps. An embargo of N bars is dropped at the
    boundary so indicator state warmed up on development data cannot bleed
    into the first out-of-sample trades.

    Access rules:
      * `development()` is always available.
      * `out_of_sample()` raises unless a StrategySeal exists, and verifies
        the params being tested match the sealed ones.
    """

    def __init__(
        self,
        candles: pd.DataFrame,
        development_fraction: float = 0.70,
        embargo_bars: int = 200,
        seal_path: Path | None = None,
    ) -> None:
        if candles.empty:
            raise ValueError("cannot split an empty candle series")
        if not candles["timestamp"].is_monotonic_increasing:
            raise ValueError("candles must be chronologically ordered before splitting")

        self._candles = candles.reset_index(drop=True)
        self._fraction = development_fraction
        self._embargo = embargo_bars
        self._seal_path = seal_path

        total = len(self._candles)
        self._boundary_index = int(total * development_fraction)
        if self._boundary_index <= embargo_bars:
            raise ValueError(
                f"development period ({self._boundary_index} bars) is too short for an "
                f"embargo of {embargo_bars} bars"
            )
        if self._boundary_index >= total:
            raise ValueError("split leaves no out-of-sample data")

    # --- boundaries -----------------------------------------------------
    @property
    def development_start(self) -> datetime:
        return self._candles["timestamp"].iloc[0].to_pydatetime()

    @property
    def development_end(self) -> datetime:
        return self._candles["timestamp"].iloc[self._boundary_index - 1].to_pydatetime()

    @property
    def out_of_sample_start(self) -> datetime:
        return self._candles["timestamp"].iloc[self._boundary_index].to_pydatetime()

    @property
    def out_of_sample_end(self) -> datetime:
        return self._candles["timestamp"].iloc[-1].to_pydatetime()

    @property
    def boundary_index(self) -> int:
        return self._boundary_index

    # --- data access ----------------------------------------------------
    def development(self) -> pd.DataFrame:
        """The first 70%: free to analyze, fit and optimize on."""
        return self._candles.iloc[: self._boundary_index].reset_index(drop=True)

    def out_of_sample(self, params: dict | None = None) -> pd.DataFrame:
        """The last 30%: only accessible once the strategy is sealed.

        The returned frame includes the preceding `embargo_bars` bars as
        warm-up context so indicators can be computed at the very start of
        the test period, but `is_test` marks which rows are actually
        tradeable, and the executor must only trade those.
        """
        seal = self.require_seal()
        if params is not None:
            seal.verify(params)

        warmup_start = max(0, self._boundary_index - self._embargo)
        frame = self._candles.iloc[warmup_start:].reset_index(drop=True)
        frame["is_test"] = frame["timestamp"] >= pd.Timestamp(self.out_of_sample_start)
        return frame

    def require_seal(self) -> StrategySeal:
        if self._seal_path is None or not Path(self._seal_path).exists():
            raise LeakageError(
                "out-of-sample data is sealed: finalize and freeze the strategy on the "
                "development period first (writes the strategy seal). This guard exists "
                "so test-period data cannot influence strategy development."
            )
        return StrategySeal.load(Path(self._seal_path))

    def is_sealed(self) -> bool:
        return self._seal_path is not None and Path(self._seal_path).exists()

    def summary(self) -> dict:
        return {
            "total_bars": len(self._candles),
            "development_bars": self._boundary_index,
            "out_of_sample_bars": len(self._candles) - self._boundary_index,
            "development_fraction_actual": round(
                self._boundary_index / len(self._candles), 4
            ),
            "development_start": self.development_start,
            "development_end": self.development_end,
            "out_of_sample_start": self.out_of_sample_start,
            "out_of_sample_end": self.out_of_sample_end,
            "embargo_bars": self._embargo,
            "sealed": self.is_sealed(),
        }


def guard_reseal(seal_path: Path, reason: str, allow: bool) -> None:
    """Called before re-optimizing when a seal already exists.

    Re-optimization after the test set has been run is exactly the leakage
    the methodology forbids. It is therefore refused unless explicitly
    allowed, and when allowed it is appended to the seal's history so the
    final report can disclose that the strategy was not, in fact, frozen
    first.
    """
    path = Path(seal_path)
    if not path.exists():
        return
    if not allow:
        raise LeakageError(
            "a strategy seal already exists: re-optimizing now would let "
            "out-of-sample results influence the strategy. Pass allow_reseal=True "
            "only if you accept that this invalidates the out-of-sample claim; it "
            "will be recorded in the seal history and the final report."
        )

    seal = StrategySeal.load(path)
    seal.reseal_history.append(
        {
            "resealed_at": datetime.now(timezone.utc).isoformat(),
            "previous_params_hash": seal.params_hash,
            "reason": reason,
        }
    )
    seal.save(path)
