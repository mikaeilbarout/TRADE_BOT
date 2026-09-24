from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from research.ai.models import CHEAPEST_ANALYST_MODEL, DEFAULT_FINAL_MODEL, get_pricing


class FailClosedPolicy(StrEnum):
    """What to do when an agent cannot produce a trustworthy verdict."""

    NO_TRADE = "NO_TRADE"  # default: missing information never becomes approval
    WAIT = "WAIT"


class AISettings(BaseSettings):
    """AI layer configuration. Every model is an env var -- no model name is
    hard-coded at a call site (spec section 3)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ai_enabled: bool = True

    # --- model routing: cheap analysts, capable adjudicator ---------------
    ai_technical_model: str = CHEAPEST_ANALYST_MODEL
    ai_news_model: str = CHEAPEST_ANALYST_MODEL
    ai_sentiment_model: str = CHEAPEST_ANALYST_MODEL
    ai_final_model: str = DEFAULT_FINAL_MODEL

    anthropic_api_key: str | None = None

    # --- cost controls ----------------------------------------------------
    ai_cost_limit_usd: float = 10.0
    ai_max_signals_per_run: int = 100
    ai_use_batch: bool = True
    ai_use_prompt_cache: bool = True
    ai_max_output_tokens: int = 400
    # 1h keeps the static prefix warm across a long backtest; 5m would expire
    # between batch polls on a slow run.
    ai_cache_ttl: str = "1h"

    # --- request behavior --------------------------------------------------
    ai_timeout_seconds: float = 60.0
    ai_max_retries: int = 2
    ai_concurrency: int = 4  # only used for the non-batch path

    # --- context sizing (fewer tokens = lower cost) ----------------------
    ai_technical_lookback_bars: int = 60
    ai_htf_lookback_bars: int = 30
    ai_news_max_items: int = 8
    ai_sentiment_max_items: int = 6

    # --- fail-closed policy ------------------------------------------------
    ai_fail_closed_policy: FailClosedPolicy = FailClosedPolicy.NO_TRADE
    # When a point-in-time store has no data for a timestamp, the agent can
    # only answer UNAVAILABLE -- so skip the call entirely rather than paying
    # for a guaranteed non-answer. The skip is recorded as UNAVAILABLE.
    ai_skip_agent_when_data_unavailable: bool = True
    # Whether a MODIFY from the final agent may be executed at all.
    ai_allow_modify: bool = True

    # --- deterministic policy (shared with the live service) ---------------
    # These mirror the live service's MIN_CONFIDENCE / MIN_WEIGHTED_SCORE /
    # VETO_ON_* / WEIGHT_* settings. Both sides are normally populated from
    # one `research.experiment.ExperimentConfig`, which is what stops the
    # backtest from measuring a policy the live system does not run; the
    # defaults here only matter for standalone use.
    ai_min_final_confidence: float = 0.60
    ai_min_weighted_score: float = 60.0
    ai_veto_on_news_block: bool = True
    ai_veto_on_sentiment_block: bool = True
    ai_veto_on_technical_block: bool = True
    ai_weight_news: float = 0.25
    ai_weight_sentiment: float = 0.20
    ai_weight_technical: float = 0.35
    ai_weight_risk: float = 0.20

    # --- planned placebo control (NOT enabled by default) ------------------
    # Strips absolute dates from agent payloads so a future control arm can
    # measure how much of the AI layer's edge comes from recognising the
    # period rather than reading the setup. Designed now, deliberately not
    # part of the pilot; see docs/METHODOLOGY.md.
    ai_strip_dates: bool = False

    # --- persistence -------------------------------------------------------
    ai_checkpoint_path: Path = Path("results/ai_checkpoint.sqlite")
    ai_results_dir: Path = Path("results/ai")

    @model_validator(mode="after")
    def validate_weights(self) -> "AISettings":
        total = (
            self.ai_weight_news
            + self.ai_weight_sentiment
            + self.ai_weight_technical
            + self.ai_weight_risk
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"AI scoring weights must sum to 1.0, got {total:.4f} "
                "(ai_weight_news + ai_weight_sentiment + ai_weight_technical + "
                "ai_weight_risk)"
            )
        return self

    @model_validator(mode="after")
    def validate_models(self) -> "AISettings":
        # Fail fast on an unpriced model: the budget guard depends on pricing.
        for field in (
            "ai_technical_model",
            "ai_news_model",
            "ai_sentiment_model",
            "ai_final_model",
        ):
            get_pricing(getattr(self, field))
        return self

    def model_for(self, agent: str) -> str:
        return {
            "technical": self.ai_technical_model,
            "news": self.ai_news_model,
            "sentiment": self.ai_sentiment_model,
            "final": self.ai_final_model,
        }[agent]
