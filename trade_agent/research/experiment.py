from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from app.config.settings import Settings
from app.services.decision_policy import thresholds_from_settings
from app.services.policy_core import PolicyThresholds
from app.services.risk_service import RiskService
from research.ai.gate import DeterministicGate, replay_risk_settings
from research.ai.settings import AISettings
from research.backtest.engine import BacktestEngine
from research.backtest.executor import DecisionExecutor
from research.config import BacktestConfig
from research.manifest import DatasetVersion, RunManifest, sha256_obj
from research.strategy.optimizer import OptimizerConfig
from research.strategy.donchian_scalp import DonchianParams, DonchianScalpStrategy

"""One configuration for both arms of the A/B experiment.

The failure this prevents: Experiment A and Experiment B each built their own
config, and the two drifted -- different position caps, different daily caps,
different confidence thresholds. A comparison between two differently
configured runs measures the configuration difference, and the AI's apparent
edge is then partly an artifact of it.

So there is exactly one `ExperimentConfig`. It builds the engine, the risk
service, the executor and the policy thresholds for BOTH arms, and
`assert_ab_identical` proves the two arms received the same numbers rather
than relying on the convention that they should have.

It is also the point where the old live/research conflicts are resolved
structurally: the live `Settings` used by the research risk engine is DERIVED
from this config's risk model, so "max 1 position in the engine but max 5 in
the risk service" cannot recur -- there is only one number, and both readers
read it.
"""

# The engine's fill assumptions, recorded in every manifest. Bump the version
# when any of these change: two runs with different execution semantics are
# not comparable, and a silent change to the fill model is the kind of thing
# that makes an old result quietly wrong.
EXECUTION_ASSUMPTIONS: dict = {
    "semantics_version": 2,
    "entry_delay": "next_bar_open",
    "entry_crosses_spread": True,
    "spread_source": "bar spread_mean, falling back to configured fallback_spread_price",
    "stop_assumed_first_when_both_touched": True,
    "stop_fills_worse_than_trigger": True,
    "target_fills_at_level": True,
    "gap_through_level_fills_at_open": True,
    "modified_entry_away_from_price": "resting limit, filled at the limit price only "
    "when the bar's spread-adjusted extreme reaches it",
    "modified_limit_cancelled_if_target_first": True,
    "position_sizing": "risk_per_trade_pct of running balance, floored to volume_step",
    "commission_charged": "both sides at fill",
    "counterfactual_sizing": "fixed initial balance, never added to any equity curve",
}

# Manifest fields that are ALLOWED to differ between the two arms. Everything
# else differing is a bug in the experiment, not a finding about the AI.
AI_ONLY_MANIFEST_FIELDS: frozenset[str] = frozenset(
    {
        "run_kind",
        "llm_model",
        "llm_provider",
        "prompts_hash",
        "agent_settings",
        # The AI arm consumes news/sentiment/calendar data the baseline never
        # touches, and carries limitations (model hindsight, agent data
        # availability) that do not apply to a run with no model in it. These
        # describe the run rather than configure it.
        #
        # Not a loophole: `assert_manifests_match` separately requires that any
        # dataset named in BOTH manifests -- the candle series above all -- be
        # byte-identical. The arms may use different inputs; they may not run
        # on different bars.
        "datasets",
        "known_limitations",
    }
)


class PolicyConfig(BaseModel):
    """The deterministic policy numbers, in one place for both paths."""

    min_confidence: float = 0.70
    min_weighted_score: float = 60.0
    veto_on_news_block: bool = True
    veto_on_sentiment_block: bool = True
    veto_on_technical_block: bool = True
    allow_modify: bool = True
    weight_news: float = 0.25
    weight_sentiment: float = 0.20
    weight_technical: float = 0.35
    weight_risk: float = 0.20
    high_impact_news_blackout_minutes: int = 15

    @model_validator(mode="after")
    def check_weights(self) -> "PolicyConfig":
        total = (
            self.weight_news
            + self.weight_sentiment
            + self.weight_technical
            + self.weight_risk
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"policy weights must sum to 1.0, got {total:.4f}")
        return self

    def thresholds(self) -> PolicyThresholds:
        return PolicyThresholds(
            min_confidence=self.min_confidence,
            min_weighted_score=self.min_weighted_score,
            veto_on_news_block=self.veto_on_news_block,
            veto_on_sentiment_block=self.veto_on_sentiment_block,
            veto_on_technical_block=self.veto_on_technical_block,
            allow_modify=self.allow_modify,
            weight_news=self.weight_news,
            weight_sentiment=self.weight_sentiment,
            weight_technical=self.weight_technical,
            weight_risk=self.weight_risk,
        )


class ExperimentConfig(BaseModel):
    """Everything both arms of the experiment share, plus the AI-only parts."""

    name: str = "xauusd_m15_70_30"
    # How an economic release's clock time is established. SCHEDULED_LOCAL uses
    # the publisher's long-standing release time (08:30 America/New_York for BLS
    # prints) converted to UTC with DST handled; every record produced that way
    # is marked IMPUTED_FROM_SCHEDULE, because the time is an assumption about
    # the schedule rather than something ALFRED published. Recorded here so it
    # reaches the run manifest and both arms see the same calendar.
    calendar_release_time_policy: str = "SCHEDULED_LOCAL"
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    # The production bot's parameters. Replaced the placeholder strategy once
    # scalp-sample-v2 became available.
    strategy_params: DonchianParams = Field(default_factory=DonchianParams)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    # AI-only. Present in the config because the AI arm needs it; excluded
    # from the shared fingerprint because it is the intended difference.
    ai: AISettings | None = None

    # --- derived objects, built identically for both arms ------------------
    def policy_thresholds(self) -> PolicyThresholds:
        return self.policy.thresholds()

    def risk_settings(self, base: Settings | None = None) -> Settings:
        """The live `Settings` the research risk engine runs on.

        Derived from THIS config, not read independently from the environment:
        that is what makes the position and daily caps single-valued. Only the
        two wall-clock rules are neutralized, by `replay_risk_settings`, and
        the reason is documented there.
        """
        source = base or Settings()
        risk = self.backtest.risk
        return replay_risk_settings(
            source.model_copy(
                update={
                    "max_risk_per_trade_pct": risk.risk_per_trade_pct,
                    "max_simultaneous_positions": risk.max_concurrent_positions,
                    "max_trades_per_day": risk.max_trades_per_day,
                    "min_confidence": self.policy.min_confidence,
                    "min_weighted_score": self.policy.min_weighted_score,
                    "veto_on_news_block": self.policy.veto_on_news_block,
                    "veto_on_sentiment_block": self.policy.veto_on_sentiment_block,
                    "veto_on_technical_block": self.policy.veto_on_technical_block,
                    "weight_news": self.policy.weight_news,
                    "weight_sentiment": self.policy.weight_sentiment,
                    "weight_technical": self.policy.weight_technical,
                    "weight_risk": self.policy.weight_risk,
                    "high_impact_news_blackout_minutes": (
                        self.policy.high_impact_news_blackout_minutes
                    ),
                }
            )
        )

    def ai_settings(self) -> AISettings:
        """AI settings with the policy numbers forced to match this config.

        The AI layer's own defaults are overridden here so a stale
        `AI_MIN_FINAL_CONFIDENCE` in an env file cannot make the research path
        apply a looser policy than the live path.
        """
        source = self.ai or AISettings()
        return source.model_copy(
            update={
                "ai_min_final_confidence": self.policy.min_confidence,
                "ai_min_weighted_score": self.policy.min_weighted_score,
                "ai_veto_on_news_block": self.policy.veto_on_news_block,
                "ai_veto_on_sentiment_block": self.policy.veto_on_sentiment_block,
                "ai_veto_on_technical_block": self.policy.veto_on_technical_block,
                "ai_allow_modify": self.policy.allow_modify,
                "ai_weight_news": self.policy.weight_news,
                "ai_weight_sentiment": self.policy.weight_sentiment,
                "ai_weight_technical": self.policy.weight_technical,
                "ai_weight_risk": self.policy.weight_risk,
            }
        )

    def engine(self) -> BacktestEngine:
        return BacktestEngine(self.backtest)

    def risk_service(self, base: Settings | None = None) -> RiskService:
        return RiskService(self.risk_settings(base))

    def strategy(self) -> DonchianScalpStrategy:
        """The production bot's signal logic.

        Always built with the leakage-free trend alignment; the legacy
        look-ahead mode the bot's own backtest used is deliberately not
        reachable from the experiment config.
        """
        return DonchianScalpStrategy(self.strategy_params, symbol=self.backtest.symbol)

    def gate(self, base: Settings | None = None) -> DeterministicGate:
        return DeterministicGate(self.backtest, self.risk_service(base), self.engine())

    def executor(self, base: Settings | None = None) -> DecisionExecutor:
        """The executor. Both arms get one built by this method, so neither
        can be handed a differently configured engine or risk service."""
        return DecisionExecutor(
            config=self.backtest,
            engine=self.engine(),
            risk_service=self.risk_service(base),
            allow_modify=self.policy.allow_modify,
            counterfactual_balance=self.backtest.risk.initial_balance,
        )

    # --- the shared fingerprint -------------------------------------------
    def shared_fingerprint(self, base: Settings | None = None) -> dict:
        """Every setting that MUST be identical in both arms.

        Grouped by the categories the experiment design names, so a mismatch
        report says which category drifted rather than only which key.
        """
        risk = self.backtest.risk
        costs = self.backtest.costs
        settings = self.risk_settings(base)
        return {
            "initial_capital": {"initial_balance": risk.initial_balance},
            "risk": {
                "risk_per_trade_pct": risk.risk_per_trade_pct,
                "max_risk_per_trade_pct": settings.max_risk_per_trade_pct,
                "max_stop_loss_distance_pct": settings.max_stop_loss_distance_pct,
                "min_risk_reward_ratio": settings.min_risk_reward_ratio,
                "max_daily_loss_pct": settings.max_daily_loss_pct,
                "max_leverage": settings.max_leverage,
                "max_exposure_per_asset_pct": settings.max_exposure_per_asset_pct,
                "max_volatility_atr_multiple": settings.max_volatility_atr_multiple,
            },
            "spread": {
                "fallback_spread_price": costs.fallback_spread_price,
                "max_spread_pct": settings.max_spread_pct,
            },
            "slippage": {
                "slippage_price": costs.slippage_price,
                "stop_slippage_price": costs.stop_slippage_price,
                "max_slippage_pct": settings.max_slippage_pct,
                "max_pending_entry_distance_pct": settings.max_pending_entry_distance_pct,
            },
            "commission": {
                "commission_per_lot_per_side": costs.commission_per_lot_per_side
            },
            "position_limits": {
                "max_concurrent_positions": risk.max_concurrent_positions,
                "max_simultaneous_positions": settings.max_simultaneous_positions,
            },
            "daily_limits": {
                "max_trades_per_day": risk.max_trades_per_day,
                "settings_max_trades_per_day": settings.max_trades_per_day,
            },
            "instrument": self.backtest.instrument.model_dump(),
            "execution_assumptions": {
                **EXECUTION_ASSUMPTIONS,
                "timeframe_minutes": self.backtest.timeframe_minutes,
                "skip_if_below_min_volume": risk.skip_if_below_min_volume,
                "modified_limit_expiry_bars": risk.modified_limit_expiry_bars,
            },
            "strategy_parameters": self.strategy_params.to_dict(),
            "split": self.backtest.split.model_dump(),
            "policy": self.policy.model_dump(),
            "random_seed": self.backtest.random_seed,
            "symbol": self.backtest.symbol,
            "calendar_release_time_policy": self.calendar_release_time_policy,
        }

    def fingerprint_hash(self, base: Settings | None = None) -> str:
        return sha256_obj(self.shared_fingerprint(base))

    # --- manifests ---------------------------------------------------------
    def manifest(
        self,
        run_id: str,
        run_kind: str,
        datasets: list[DatasetVersion] | None = None,
        seal_hash: str | None = None,
        prompts_hash: str | None = None,
        limitations: list[dict] | None = None,
    ) -> RunManifest:
        ai = self.ai_settings() if run_kind == "ai" else None
        return RunManifest(
            run_id=run_id,
            run_kind=run_kind,
            backtest_config=self.shared_fingerprint(),
            strategy_name=DonchianScalpStrategy.name,
            strategy_params=self.strategy_params.to_dict(),
            strategy_seal_hash=seal_hash,
            datasets=datasets or [],
            llm_model=ai.ai_final_model if ai else None,
            llm_provider="anthropic" if ai else None,
            prompts_hash=prompts_hash if ai else None,
            agent_settings=(
                {
                    "models": {
                        agent: ai.model_for(agent)
                        for agent in ("technical", "news", "sentiment", "final")
                    },
                    "max_output_tokens": ai.ai_max_output_tokens,
                    "use_prompt_cache": ai.ai_use_prompt_cache,
                    "cache_ttl": ai.ai_cache_ttl,
                    "use_batch": ai.ai_use_batch,
                    "fail_closed_policy": ai.ai_fail_closed_policy.value,
                    "skip_agent_when_data_unavailable": (
                        ai.ai_skip_agent_when_data_unavailable
                    ),
                    "strip_dates": ai.ai_strip_dates,
                }
                if ai
                else None
            ),
            random_seed=self.backtest.random_seed,
            known_limitations=limitations or [],
        )


class ManifestMismatch(Exception):
    """Raised when the two arms of the experiment were not equivalent.

    Deliberately fatal. A comparison run on mismatched configs produces a
    number that looks like a measurement of the AI layer and is not one.
    """


def assert_ab_identical(
    config_a: ExperimentConfig,
    config_b: ExperimentConfig,
    base: Settings | None = None,
) -> None:
    """Prove both arms are configured identically outside the AI layer."""
    fa, fb = config_a.shared_fingerprint(base), config_b.shared_fingerprint(base)
    differences = _diff(fa, fb)
    if differences:
        raise ManifestMismatch(
            "Experiment A and B are not configured identically; the comparison "
            "would measure the configuration difference, not the AI layer:\n  - "
            + "\n  - ".join(differences)
        )


def assert_manifests_match(
    manifest_a: RunManifest,
    manifest_b: RunManifest,
    allowed: frozenset[str] = AI_ONLY_MANIFEST_FIELDS,
) -> None:
    """Compare two run manifests, allowing only the AI-specific fields to differ.

    `run_id`, `created_at` and the host/git fields are excluded: they are
    provenance, not configuration, and always differ between two runs.
    """
    a = manifest_a.model_dump(mode="json")
    b = manifest_b.model_dump(mode="json")
    for provenance in ("run_id", "created_at", "git_revision", "python_version", "platform"):
        a.pop(provenance, None)
        b.pop(provenance, None)

    differences: list[str] = []

    # Datasets used by BOTH arms must be identical, even though the set of
    # datasets may differ. This is what stops "the arms may use different
    # inputs" from becoming "the arms may run on different price data".
    if "datasets" in allowed:
        differences.extend(_shared_dataset_differences(a, b))

    for key in sorted(set(a) | set(b)):
        if key in allowed:
            continue
        if a.get(key) != b.get(key):
            nested = (
                _diff(a[key], b[key])
                if isinstance(a.get(key), dict) and isinstance(b.get(key), dict)
                else []
            )
            differences.extend(nested or [f"{key}: {a.get(key)!r} != {b.get(key)!r}"])

    if differences:
        raise ManifestMismatch(
            "run manifests differ outside the allowed AI-only fields "
            f"({', '.join(sorted(allowed))}):\n  - " + "\n  - ".join(differences)
        )


def _shared_dataset_differences(a: dict, b: dict) -> list[str]:
    """Differences in datasets that appear in both manifests."""
    by_name_a = {entry["name"]: entry for entry in a.get("datasets") or []}
    by_name_b = {entry["name"]: entry for entry in b.get("datasets") or []}
    shared = set(by_name_a) & set(by_name_b)
    out: list[str] = []
    for name in sorted(shared):
        out.extend(_diff(by_name_a[name], by_name_b[name], f"datasets.{name}"))
    return out


def _diff(a, b, path: str = "") -> list[str]:
    """Flat list of differing leaf paths between two nested structures."""
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for key in sorted(set(a) | set(b)):
            child = f"{path}.{key}" if path else str(key)
            if key not in a:
                out.append(f"{child}: missing in A, {b[key]!r} in B")
            elif key not in b:
                out.append(f"{child}: {a[key]!r} in A, missing in B")
            else:
                out.extend(_diff(a[key], b[key], child))
        return out
    if a != b:
        return [f"{path}: {a!r} != {b!r}"]
    return []


def load_experiment(path: Path | None = None) -> ExperimentConfig:
    """Load the experiment config from JSON, or build the defaults.

    One file, referenced by every CLI command, so a run cannot be executed
    against numbers nobody wrote down.
    """
    if path is None or not Path(path).exists():
        return ExperimentConfig()
    return ExperimentConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_experiment(config: ExperimentConfig, path: Path) -> Path:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(config.model_dump_json(indent=2), encoding="utf-8")
    return Path(path)


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat()
