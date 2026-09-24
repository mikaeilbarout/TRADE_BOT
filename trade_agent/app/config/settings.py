from __future__ import annotations

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.models.enums import TradingMode


class Settings(BaseSettings):
    """All tunables live here. Nothing risk-related is hard-coded elsewhere.

    Every field is overridable via environment variable or .env file using
    the same name (case-insensitive).
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Trading mode -------------------------------------------------
    trading_mode: TradingMode = TradingMode.PAPER
    # "four_agent" (news/sentiment/technical/final, app/services/pipeline.py)
    # or "single_agent" (one UnifiedAgent call,
    # app/services/single_agent_pipeline.py). Switched here, not by code
    # change, so a bad rollout is one env var away from reverting.
    #
    # Default flipped to single_agent 2026-09-19 after both live profiles
    # it matters for were validated with REAL tick fills (not the bar-close
    # approximation used earlier, which was missing breakeven-at-2R and
    # understated the single-agent result badly for M30):
    #   donchian_m15: raw $1,876.82 -> AI $2,102.06 (190-signal sample,
    #     2026-03-25..09-11, +12%, 26 avoided losses / 7 false rejections)
    #   donchian_m30: raw $290.87 -> AI $1,435.13 (full 235-signal
    #     population, same window, +394%, 32 avoided losses / 1 false
    #     rejection)
    # donchian_h1 has no validated ADX threshold (see
    # single_agent_strong_trend_adx_thresholds below) and is not currently
    # traded live -- single_agent still runs for it (default-heavy-APPROVE,
    # no deterministic trend override), just without that one backstop.
    pipeline_mode: str = "single_agent"

    # --- LLM ------------------------------------------------------------
    llm_provider: str = "mock"  # "anthropic" | "openai" | "mock"
    llm_model: str = "claude-sonnet-5"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    agent_timeout_seconds: float = 10.0
    llm_max_retries: int = 1
    # Each agent's system prompt + tool schema is byte-identical call to call
    # (same .md file, same response_model every time); only the per-signal
    # user message actually changes. Caching that static prefix is cost/
    # latency only -- same mechanism already used in research/ai/client.py,
    # ported here since the live service calls this far more often per day.
    llm_use_prompt_cache: bool = True
    llm_cache_ttl: str = "1h"  # "5m" | "1h", per Anthropic's cache_control

    # --- News -------------------------------------------------------------
    news_provider: str = "mock"  # "finnhub" | "fred" | "finnhub+fred" | "newsapi" | "mock"
    newsapi_api_key: str | None = None
    # finnhub.io free tier: no publish-time embargo (unlike the newsapi plan
    # above, which refuses anything from the last ~24h and made news_lookback
    # windows always resolve empty -- see FinnhubNewsProvider's docstring).
    finnhub_api_key: str | None = None
    # FRED (Federal Reserve Economic Data): free key at
    # fredaccount.stlouisfed.org/apikeys. Surfaces actual macro releases
    # (CPI, NFP, Fed funds rate, ...) that move gold/USD directly -- content
    # Finnhub's generic business-news feed rarely covers. "finnhub+fred"
    # merges both (see CompositeNewsProvider) rather than choosing one,
    # since they cover genuinely different content, not overlapping ones.
    fred_api_key: str | None = None
    # Widened 60->180->720 (2026-09-18): with finnhub, news_agent's own
    # prompt already distinguishes breaking/recent/old by each item's
    # age_minutes and is told to "never treat old news as breaking"
    # (app/prompts/news_agent.md), so a wider window feeds it real,
    # relevant items instead of an empty bundle most of the time --
    # confirmed live, 60min returned 0 items at a routine moment. 180->720
    # specifically to catch FRED releases: unlike headlines, FRED updates
    # follow a real publish calendar (CPI/NFP/etc. release a few times a
    # month each), not a continuous stream -- confirmed live, a 180min
    # window caught 0 of the last 24h's 5 real releases, 720min caught
    # most of them. A same-day economic print (e.g. Fed funds rate from 12h
    # ago) stays genuinely decision-relevant far longer than a generic
    # headline would, so this isn't the same staleness risk widening
    # Finnhub-only lookback would be.
    news_lookback_minutes: int = 720
    news_cache_ttl_seconds: int = 120
    # A cached news bundle older than this is considered unusable even as a
    # degraded fallback -- better to fail closed than to decide on ancient news.
    news_max_fallback_age_seconds: int = 900
    high_impact_news_blackout_minutes: int = 15

    # --- Sentiment ---------------------------------------------------------
    sentiment_provider: str = "mock"
    sentiment_lookback_minutes: int = 180  # widened 60->180 alongside news_lookback_minutes, see its comment
    sentiment_cache_ttl_seconds: int = 120
    # Mirrors news_max_fallback_age_seconds. Until this existed the
    # sentiment fallback had no age bound at all and could serve an
    # arbitrarily old cached bundle when the provider was down.
    sentiment_max_fallback_age_seconds: int = 900

    # --- Market data --------------------------------------------------
    market_data_provider: str = "mock"
    market_data_max_staleness_seconds: float = 30.0
    oanda_api_key: str | None = None
    oanda_account_id: str | None = None
    oanda_environment: str = "practice"  # "practice" | "live"
    default_higher_timeframes: list[str] = Field(default_factory=lambda: ["D1", "H4"])
    default_medium_timeframes: list[str] = Field(default_factory=lambda: ["H1", "M30"])
    default_entry_timeframes: list[str] = Field(default_factory=lambda: ["M15", "M5"])

    # --- Risk limits (section 11) --------------------------------------
    max_risk_per_trade_pct: float = 0.5
    # Monetary risk per trade cannot be computed without the account balance,
    # so by default a signal that omits it is rejected rather than having the
    # limit silently skipped. Set False only if the bot owns position sizing
    # and you accept that max_risk_per_trade_pct is then unenforceable here.
    require_account_balance: bool = True
    max_daily_loss_pct: float = 2.0
    max_trades_per_day: int = 20
    max_simultaneous_positions: int = 5
    max_exposure_per_asset_pct: float = 5.0
    max_leverage: float = 20.0
    max_stop_loss_distance_pct: float = 3.0
    # 1.0, not 1.5: the H1 live profile's own tick-verified backtest found
    # rr=1.0 to be its best parameter (profiles/profile_h1.py, 2026-09-08
    # update) -- a 1.5 floor here rejected every H1 signal unconditionally,
    # before AI review ever ran. M1 (rr=1.5), M30/M15 (rr=3.0) stay unaffected.
    min_risk_reward_ratio: float = 1.0
    max_spread_pct: float = 0.15
    # Applies to market orders only: how far the live price may have moved
    # from the intended entry before the fill is considered bad.
    max_slippage_pct: float = 0.1
    # Applies to pending/limit entries (e.g. an agent proposing a pullback
    # entry): how far from the live price the resting order may sit. Bounds
    # how far an agent can move an entry away from the market.
    max_pending_entry_distance_pct: float = 1.0
    # Rejects setups whose stop is small relative to current volatility:
    # ATR must not exceed this multiple of the trade's stop distance.
    max_volatility_atr_multiple: float = 4.0
    max_signal_age_seconds: float = 120.0

    # --- Decision thresholds --------------------------------------------
    min_confidence: float = 0.70
    min_weighted_score: float = 60.0
    # SingleAgentPipeline only: a deterministic override the LLM cannot
    # skip, on top of its own judgment -- REJECT whenever the trade runs
    # against the daily trend AND the D1 market is strongly trending (D1
    # ADX(14) at or above the threshold for THIS signal's own strategy).
    #
    # Keyed by `signal.strategy` ("donchian_m15", "donchian_m30", ...) --
    # NOT one global number -- because the pattern itself is
    # profile-specific: it was validated separately for each profile with
    # its OWN full-history sweep (2026-09-19), using that profile's own
    # live params and its own "one level above its own trend filter"
    # timeframe (D1 for M15/M30, since both use H4 as their own filter).
    # A profile with no entry here gets NO override -- never a borrowed
    # threshold from a different profile -- because the pattern does not
    # automatically transfer: H1 was tested (its own filter is already D1,
    # so its blind spot is weekly/monthly, not D1) and neither weekly nor
    # monthly held up across years, so H1 is deliberately left out rather
    # than given an unvalidated number.
    #
    # donchian_m15 = 34.0: full 5-year sweep, lowest threshold with zero
    # contradicting years (5 of 5 favor). An earlier pass landed on 37 from
    # a single top-quartile cut and a boundary bug (D1 aggregated at
    # midnight UTC instead of the 21:00-UTC/NY-close roll the live
    # market-data provider, OANDA, actually uses -- confirmed up to
    # ~$60/day close differences on XAUUSD); corrected and re-swept.
    # Backtested for real at 37 (176/190 taken, +47.0R vs raw +37.0R;
    # only 4 of the 14 rejections were this rule, 3 correct). Lowering
    # toward ~25-28 looked even better on that SAME 190-signal window, but
    # that was curve-fit to it: the 5-year sweep showed 2023 consistently
    # reversing below ~30 (counter-trend trades at moderate ADX did well
    # that year, 38-46% WR) -- 34 was chosen from the sweep, not the
    # paid backtest, precisely to avoid re-fitting to one window.
    #
    # donchian_m30 = 38.0: same full 5-year sweep methodology (this
    # profile's own N=55/EMA=50/H4-filter params), lowest threshold with
    # zero contradicting years (4 of 4 usable years favor; 2024
    # consistently reversed below 38, same failure shape as M15's 2023).
    # Not yet backtested for real -- do that before trusting the exact
    # number, same as M15 was.
    single_agent_strong_trend_adx_thresholds: dict[str, float] = Field(
        default_factory=lambda: {"donchian_m15": 34.0, "donchian_m30": 38.0}
    )
    short_circuit_on_critical_news: bool = True
    # 2026-09-18: turned off news/technical hard vetoes by explicit user
    # decision. Previously a single gating agent's BLOCK force-rejected the
    # trade regardless of what final_decision_agent itself concluded --
    # even if the other two dimensions strongly supported it and final_agent
    # judged the BLOCK to be minor. That made final_agent's synthesis moot
    # in exactly the cases it exists to handle. Now every specialist's
    # decision is just strong input to final_agent's own weighing (see
    # app/prompts/final_agent.md's domain-ownership/hard-veto-priority
    # sections), and its own APPROVE/MODIFY/REJECT/WAIT stands. The
    # deterministic backstops that remain regardless of these flags:
    # min_confidence, min_weighted_score (technical still carries the
    # heaviest weight, weight_technical=0.45), the news blackout window,
    # degraded-data handling, and all hard account/risk limits below.
    veto_on_news_block: bool = False
    veto_on_sentiment_block: bool = False
    veto_on_technical_block: bool = False

    # --- Scoring weights (section 10); must sum to 1.0 -------------------
    weight_news: float = 0.25
    weight_sentiment: float = 0.20
    weight_technical: float = 0.35
    weight_risk: float = 0.20

    # --- Persistence ------------------------------------------------------
    database_url: str = "sqlite+aiosqlite:///./trade_agent.db"
    # When set, news/sentiment caches use Redis so multiple API workers share
    # one cache. Any Redis failure degrades to the in-process cache rather
    # than failing a trade decision.
    redis_url: str | None = None

    # --- API ---------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # When set, callers must present it as the X-API-Key header.
    internal_api_key: str | None = None

    @model_validator(mode="after")
    def check_weights_sum_to_one(self) -> "Settings":
        total = self.weight_news + self.weight_sentiment + self.weight_technical + self.weight_risk
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"scoring weights must sum to 1.0, got {total:.4f} "
                "(weight_news + weight_sentiment + weight_technical + weight_risk)"
            )
        return self

    @property
    def all_timeframes(self) -> list[str]:
        """Every timeframe the market-data layer must supply, highest to
        lowest. The last entry is treated as the entry timeframe."""
        return [
            *self.default_higher_timeframes,
            *self.default_medium_timeframes,
            *self.default_entry_timeframes,
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
