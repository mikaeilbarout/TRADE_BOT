from __future__ import annotations

import pandas as pd

from app.models.market_data import MarketSnapshot
from research.ai.client import AgentRequest
from research.ai.pit_store import PitQueryResult
from research.ai.prompts import PROMPTS, prompt_version
from research.ai.schemas import (
    AgentVerdict,
    FinalVerdict,
    NewsVerdict,
    SentimentVerdict,
    TechnicalVerdict,
)
from research.ai.settings import AISettings
from research.strategy.base import StrategySignal

"""Per-agent payload construction.

This module is where token cost is actually decided. Three rules:

1. **Only the signal bar's state, plus a downsampled tail.** A full 60-bar
   OHLCV dump is thousands of tokens per call for information an indicator
   summary already encodes. We send computed values plus a short, rounded
   bar tail.
2. **Round aggressively.** Gold at 5 decimal places costs tokens per digit
   and changes no verdict. Prices go to 2dp, indicators to 2-4dp.
3. **Nothing static in the payload.** Rules, schema, and reason codes live in
   the cached system prompt. Repeating them per signal would pay full price
   for them on every call.
"""


def _round(value, digits: int = 2):
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return round(float(value), digits)


def signal_block(signal: StrategySignal, volume: float | None = None) -> dict:
    return {
        "side": signal.side.value,
        "entry": _round(signal.entry),
        "sl": _round(signal.stop_loss),
        "tp": _round(signal.take_profit),
        "rr": _round(signal.risk_reward_ratio),
        "risk_pts": _round(signal.risk_distance),
        "volume": _round(volume, 2) if volume else None,
        "ts": signal.signal_time.isoformat(),
    }


def market_block(market: MarketSnapshot) -> dict:
    return {
        "bid": _round(market.bid),
        "ask": _round(market.ask),
        "spread": _round(market.spread, 3),
        "session": market.session,
        "atr": _round(market.atr, 3),
        "atr_pct": _round(market.volatility_pct, 3),
    }


def technical_block(bars: pd.DataFrame, bar_index: int, settings: AISettings) -> dict:
    """Indicator state at the signal bar plus a compact bar tail.

    The tail is downsampled: the most recent 12 bars at full resolution, then
    every 4th bar before that, which preserves the shape of the move at a
    fraction of the tokens of a raw dump.
    """
    bar = bars.iloc[bar_index]
    start = max(0, bar_index - settings.ai_technical_lookback_bars + 1)
    window = bars.iloc[start : bar_index + 1]

    recent = window.tail(12)
    older = window.iloc[:-12:4] if len(window) > 12 else window.iloc[0:0]
    tail = pd.concat([older, recent])

    # Backtesting found tick volume at the signal bar a clean, monotonic
    # predictor on this strategy (480 development trades: ~23% win rate in
    # the lowest volume quartile rising to ~33% in the highest) -- but the
    # payload previously never carried volume at all, so no agent could ever
    # have used it. `volume_pctile` is the signal bar's volume rank within
    # its own lookback window, since raw tick counts are not comparable
    # across different periods.
    signal_volume = bar.get("volume")
    volume_pctile = None
    if signal_volume is not None and len(window) > 1:
        volume_pctile = float((window["volume"] < signal_volume).mean())

    return {
        "indicators": {
            "ema50": _round(bar.get("ema_50")),
            "ema200": _round(bar.get("ema_200")),
            "ema_fast": _round(bar.get("ema_fast")),
            "ema_slow": _round(bar.get("ema_slow")),
            "rsi": _round(bar.get("rsi"), 1),
            "macd": _round(bar.get("macd"), 3),
            "macd_sig": _round(bar.get("macd_signal"), 3),
            "macd_hist": _round(bar.get("macd_hist"), 3),
            "atr": _round(bar.get("atr"), 3),
            "atr_pctile": _round(bar.get("atr_percentile"), 2),
            "trend": str(bar.get("trend") or "RANGE"),
            "structure": str(bar.get("structure") or "RANGE"),
            "prior_high": _round(bar.get("prior_high")),
            "prior_low": _round(bar.get("prior_low")),
            "breakout_high": _round(bar.get("breakout_high")),
            "breakout_low": _round(bar.get("breakout_low")),
            "volume": _round(signal_volume, 0),
            "volume_pctile": _round(volume_pctile, 2),
        },
        "bars_ohlc": [
            [
                row["timestamp"].strftime("%m-%dT%H:%M"),
                _round(row["open"]),
                _round(row["high"]),
                _round(row["low"]),
                _round(row["close"]),
            ]
            for _, row in tail.iterrows()
        ],
        "bar_count": int(len(tail)),
        "lookback_bars": settings.ai_technical_lookback_bars,
    }


def htf_block(htf_bars: pd.DataFrame | None, as_of, settings: AISettings) -> dict:
    """Higher-timeframe context, if a higher-timeframe series was supplied.

    Only bars that CLOSED at or before the signal timestamp are included -- a
    partially-formed higher-timeframe bar contains future information.
    """
    if htf_bars is None or htf_bars.empty:
        return {"available": False}

    closed = htf_bars[htf_bars["timestamp"] <= pd.Timestamp(as_of)]
    if closed.empty:
        return {"available": False}

    last = closed.iloc[-1]
    tail = closed.tail(min(settings.ai_htf_lookback_bars, 10))
    return {
        "available": True,
        "trend": str(last.get("trend") or "RANGE"),
        "ema50": _round(last.get("ema_50")),
        "ema200": _round(last.get("ema_200")),
        "rsi": _round(last.get("rsi"), 1),
        "closes": [_round(c) for c in tail["close"].tolist()],
    }


def news_block(result: PitQueryResult, blackout_minutes: int, as_of) -> dict:
    """News and scheduled events, with ages and minutes-to-release so the
    agent can reason about timing without being told the outcome."""
    if not result.available:
        return {"available": False, "reason": result.reason}

    items = [
        {
            "ts": record.timestamp.isoformat(),
            "age_min": int((as_of - record.timestamp).total_seconds() / 60),
            "src": record.source,
            "headline": record.headline[:180],
            "cat": record.category,
        }
        for record in result.records
    ]
    events = [
        {
            "scheduled": event.scheduled_at.isoformat(),
            "mins_until": int((event.scheduled_at - as_of).total_seconds() / 60),
            "name": event.name,
            "importance": event.importance,
            "ccy": event.currency,
            "forecast": event.forecast_value,
            # Present only when the release time has already passed.
            "actual": event.released_value,
        }
        for event in result.events
    ]
    return {
        "available": True,
        "blackout_window_min": blackout_minutes,
        "items": items,
        "scheduled_events": events,
        "dataset": result.dataset_name,
    }


def sentiment_block(result: PitQueryResult, as_of) -> dict:
    """The only admissible point-in-time sentiment source (GDELT tone) is a
    numeric per-slot average, never article text -- there is no headline to
    show. `PitRecord.headline` is always "" for this dataset, so building
    "items" from it produced N entries that all read as empty commentary,
    which is indistinguishable from "no data" to the reading agent. The
    actual number (`value`/`raw_value`/`article_count`) was already being
    computed and then discarded before it ever reached the prompt.
    """
    if not result.available:
        return {"available": False, "reason": result.reason}
    return {
        "available": True,
        "items": [
            {
                "ts": record.timestamp.isoformat(),
                "age_min": int((as_of - record.timestamp).total_seconds() / 60),
                "src": record.source,
                "tone": _round(record.payload.get("value"), 3),
                "tone_raw": _round(record.payload.get("raw_value"), 2),
                "article_count": record.payload.get("article_count"),
            }
            for record in result.records
        ],
        "dataset": result.dataset_name,
    }


def verdict_block(agent: str, verdict: AgentVerdict | None, skipped_reason: str | None) -> dict:
    """One upstream verdict as handed to the next agent.

    A skipped agent appears explicitly as UNAVAILABLE with the reason, so the
    final agent can tell 'no conflict found' apart from 'never examined'.
    """
    if verdict is None:
        return {"agent": agent, "decision": "UNAVAILABLE", "reason": skipped_reason}

    block = {
        "agent": agent,
        "decision": verdict.decision.value,
        "confidence": verdict.confidence,
        "bias": verdict.bias.value,
        "risk": verdict.risk_level.value,
        "codes": verdict.reason_codes,
    }
    if verdict.note:
        block["note"] = verdict.note
    if isinstance(verdict, TechnicalVerdict):
        block["htf_aligned"] = verdict.htf_aligned
        block["entry_quality"] = verdict.entry_quality.value
        if verdict.suggest_entry is not None:
            block["suggested"] = {
                "entry": verdict.suggest_entry,
                "sl": verdict.suggest_sl,
                "tp": verdict.suggest_tp,
            }
    if isinstance(verdict, NewsVerdict):
        block["high_impact_window"] = verdict.high_impact_within_window
    return block


def build_technical_request(
    signal: StrategySignal,
    market: MarketSnapshot,
    bars: pd.DataFrame,
    htf_bars: pd.DataFrame | None,
    settings: AISettings,
    volume: float | None = None,
) -> AgentRequest:
    return AgentRequest(
        agent="technical",
        signal_id=signal.signal_id,
        static_system=PROMPTS["technical"],
        user_payload={
            "signal": signal_block(signal, volume),
            "market": market_block(market),
            "technical": technical_block(bars, signal.bar_index, settings),
            "higher_timeframe": htf_block(htf_bars, signal.signal_time, settings),
        },
        response_model=TechnicalVerdict,
        prompt_version=prompt_version("technical"),
        model=settings.model_for("technical"),
        max_output_tokens=settings.ai_max_output_tokens,
    )


def build_news_request(
    signal: StrategySignal,
    market: MarketSnapshot,
    news: PitQueryResult,
    blackout_minutes: int,
    settings: AISettings,
) -> AgentRequest:
    return AgentRequest(
        agent="news",
        signal_id=signal.signal_id,
        static_system=PROMPTS["news"],
        user_payload={
            "signal": signal_block(signal),
            "market": market_block(market),
            "news": news_block(news, blackout_minutes, signal.signal_time),
        },
        response_model=NewsVerdict,
        prompt_version=prompt_version("news"),
        model=settings.model_for("news"),
        max_output_tokens=settings.ai_max_output_tokens,
    )


def build_sentiment_request(
    signal: StrategySignal,
    market: MarketSnapshot,
    sentiment: PitQueryResult,
    news_verdict: NewsVerdict | None,
    news_skip_reason: str | None,
    settings: AISettings,
) -> AgentRequest:
    return AgentRequest(
        agent="sentiment",
        signal_id=signal.signal_id,
        static_system=PROMPTS["sentiment"],
        user_payload={
            "signal": signal_block(signal),
            "market": market_block(market),
            "sentiment": sentiment_block(sentiment, signal.signal_time),
            "upstream": [verdict_block("news", news_verdict, news_skip_reason)],
        },
        response_model=SentimentVerdict,
        prompt_version=prompt_version("sentiment"),
        model=settings.model_for("sentiment"),
        max_output_tokens=settings.ai_max_output_tokens,
    )


def build_final_request(
    signal: StrategySignal,
    market: MarketSnapshot,
    technical: TechnicalVerdict | None,
    news: NewsVerdict | None,
    sentiment: SentimentVerdict | None,
    skip_reasons: dict[str, str | None],
    settings: AISettings,
    risk_context: dict,
    volume: float | None = None,
) -> AgentRequest:
    return AgentRequest(
        agent="final",
        signal_id=signal.signal_id,
        static_system=PROMPTS["final"],
        user_payload={
            "signal": signal_block(signal, volume),
            "market": market_block(market),
            "chain": [
                verdict_block("technical", technical, skip_reasons.get("technical")),
                verdict_block("news", news, skip_reasons.get("news")),
                verdict_block("sentiment", sentiment, skip_reasons.get("sentiment")),
            ],
            "risk": risk_context,
            "modify_allowed": settings.ai_allow_modify,
        },
        response_model=FinalVerdict,
        prompt_version=prompt_version("final"),
        model=settings.model_for("final"),
        max_output_tokens=settings.ai_max_output_tokens,
    )
