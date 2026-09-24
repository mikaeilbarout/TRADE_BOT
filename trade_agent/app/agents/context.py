from __future__ import annotations

from datetime import datetime

from app.models.agent_decision import AgentDecisionBase
from app.models.market_data import MarketSnapshot


def market_conditions(market: MarketSnapshot) -> dict:
    """Compact view of live market conditions given to EVERY agent.

    Even the news and sentiment agents need to know where price is, how
    wide the spread is, which session is active and how volatile things
    are -- judging whether a trade fits the environment is impossible
    without it (section: each agent must consider current market
    conditions).
    """
    return {
        "bid": market.bid,
        "ask": market.ask,
        "last_price": market.last_price,
        "spread": market.spread,
        "session": market.session,
        "atr_entry_timeframe": market.atr,
        "volatility_pct": market.volatility_pct,
        "quote_timestamp": market.quote_timestamp,
        "data_age_seconds": round(market.freshness_seconds, 2),
        "data_is_stale": market.is_stale,
        "trend_by_timeframe": {
            tf: ind.trend for tf, ind in market.indicators.items()
        },
    }


def full_technical_conditions(market: MarketSnapshot, candles_per_timeframe: int = 60) -> dict:
    """Everything the technical agent needs: indicators plus the actual
    OHLCV tail per timeframe, so it can read structure (highs/lows, breaks
    of structure, candle behavior) itself rather than trusting a summary.

    Widened 30->60 (2026-09-18, user request): doubles the structural
    lookback (e.g. D1 30d->60d, M5 2.5h->5h) for more support/resistance
    and swing context. Only technical_agent receives this raw OHLCV tail
    directly -- the other 3 agents only see its summarized chain_link
    output -- so the extra prompt size/latency cost is contained to that
    one call, not the whole chain. market_data.py already fetches 220
    candles/timeframe, so this stays well within what's already pulled.
    """
    indicators = {
        tf: {
            "trend": ind.trend,
            "ema_50": ind.ema_50,
            "ema_200": ind.ema_200,
            "rsi_14": ind.rsi_14,
            "macd": ind.macd,
            "macd_signal": ind.macd_signal,
            "macd_hist": ind.macd_hist,
            "atr_14": ind.atr_14,
            "adx_14": ind.adx_14,
            "recent_high": ind.recent_high,
            "recent_low": ind.recent_low,
        }
        for tf, ind in market.indicators.items()
    }
    # Positional rows, not keyed dicts: the same 360 candles cost ~13.6k
    # tokens as {"t","o","h","l","c","v"} objects and ~5.9k as
    # [t,o,h,l,c,v] rows -- the keys and indentation were 55% of the
    # technical agent's input, i.e. most of the most expensive call in the
    # chain, for zero information. The column order is stated once in
    # `ohlcv_columns` and in technical_agent.md.
    ohlcv = {
        tf: [
            [
                c.timestamp.strftime("%Y-%m-%dT%H:%M"),
                round(c.open, 5),
                round(c.high, 5),
                round(c.low, 5),
                round(c.close, 5),
                round(c.volume, 2),
            ]
            for c in series.candles[-candles_per_timeframe:]
        ]
        for tf, series in market.timeframes.items()
    }
    return {
        **market_conditions(market),
        "entry_timeframe": market.entry_timeframe,
        "indicators_by_timeframe": indicators,
        "ohlcv_columns": ["time_utc", "open", "high", "low", "close", "volume"],
        "recent_ohlcv_by_timeframe": ohlcv,
    }


def chain_link(result: AgentDecisionBase) -> dict:
    """One upstream agent's findings, as handed to the next agent in the
    chain. Deliberately includes the agent's reasoning and warnings (not
    just its verdict) so the next agent can evaluate the ARGUMENT rather
    than inherit the conclusion."""
    link: dict = {
        "agent": result.agent,
        "decision": getattr(result, "decision", None),
        "confidence": result.confidence,
        "summary": getattr(result, "summary", ""),
        "reasoning": result.reasoning,
        "warnings": result.warnings,
        "analysis_degraded": result.is_degraded,
    }

    # Domain-specific detail, so the next agent sees the substance of the
    # upstream analysis rather than an opaque verdict.
    for extra in (
        "news_bias",
        "market_environment",
        "major_events_detected",
        "high_impact_event_within_minutes",
        "risk_of_news_reversal",
        "overall_sentiment",
        "sentiment_score",
        "sentiment_strength",
        "sentiment_momentum",
        "contradiction_level",
        "trade_compatibility",
        "manipulation_suspected",
        "low_quality_source_ratio",
        "higher_tf_trend",
        "aligned_with_higher_tf",
        "entry_valid",
        "is_overextended",
        "breakout_confirmed",
        "false_breakout_risk",
        "risk_reward_ratio",
        "stop_loss_logical",
        "take_profit_realistic",
        "volatility_acceptable",
        "confluence_factors",
        "conflicting_factors",
        "agreement_with_previous",
        "independent_finding",
    ):
        if hasattr(result, extra):
            link[extra] = getattr(result, extra)

    return link


def signal_payload(signal, now: datetime | None = None) -> dict:
    return {
        "signal_id": signal.signal_id,
        "symbol": signal.symbol,
        "side": signal.side.value,
        "entry": signal.entry,
        "stop_loss": signal.stop_loss,
        "take_profit": signal.take_profit,
        "volume": signal.volume,
        "timeframe": signal.timeframe,
        "strategy": signal.strategy,
        "risk_reward_ratio": round(signal.risk_reward_ratio, 3),
        "age_seconds": round(signal.age_seconds(now), 1),
    }
