from __future__ import annotations

from app.models.market_data import Candle, TechnicalIndicators


def ema(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    k = 2.0 / (period + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def rsi(values: list[float], period: int = 14) -> float | None:
    if len(values) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def macd(
    values: list[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[float | None, float | None, float | None]:
    if len(values) < slow + signal:
        return None, None, None
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = ema(macd_line, signal)
    hist = macd_line[-1] - signal_line[-1]
    return macd_line[-1], signal_line[-1], hist


def atr(candles: list[Candle], period: int = 14) -> float | None:
    if len(candles) < period + 1:
        return None
    trs = []
    for i in range(1, len(candles)):
        c = candles[i]
        prev_close = candles[i - 1].close
        tr = max(
            c.high - c.low,
            abs(c.high - prev_close),
            abs(c.low - prev_close),
        )
        trs.append(tr)
    return sum(trs[-period:]) / period


def adx(candles: list[Candle], period: int = 14) -> float | None:
    """Wilder's Average Directional Index: how STRONGLY a market is
    trending, independent of direction (0-100; >25 is conventionally
    "trending", >40 "strong"). Distinct from `trend`/`detect_trend`, which
    says which way, not how hard.

    A 2026-09-19 stability check (5 years of D1 XAUUSD, tested against a
    simple return/volatility ratio too) found ADX(14) on D1 was the one
    trend-strength measure whose direction held in every year with enough
    data: a counter-daily-trend trade during a high-ADX(14) D1 regime lost
    noticeably more often than the same trade during a low-ADX regime, with
    zero years contradicting it -- unlike the return/vol ratio, which
    reversed direction depending on the lookback window chosen. See
    `app.services.risk_service.AccountState` / `unified_agent.md` for where
    this is used.
    """
    if len(candles) < period * 2 + 1:
        return None
    n = len(candles)
    plus_dm = [0.0] * n
    minus_dm = [0.0] * n
    tr = [0.0] * n
    for i in range(1, n):
        up = candles[i].high - candles[i - 1].high
        down = candles[i - 1].low - candles[i].low
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
        tr[i] = max(
            candles[i].high - candles[i].low,
            abs(candles[i].high - candles[i - 1].close),
            abs(candles[i].low - candles[i - 1].close),
        )

    def wilder_smooth(values: list[float]) -> list[float]:
        smoothed = [0.0] * n
        smoothed[period] = sum(values[1 : period + 1])
        for i in range(period + 1, n):
            smoothed[i] = smoothed[i - 1] - smoothed[i - 1] / period + values[i]
        return smoothed

    atr_s = wilder_smooth(tr)
    pdm_s = wilder_smooth(plus_dm)
    mdm_s = wilder_smooth(minus_dm)

    dx = [0.0] * n
    for i in range(period, n):
        if atr_s[i] == 0:
            continue
        plus_di = 100 * pdm_s[i] / atr_s[i]
        minus_di = 100 * mdm_s[i] / atr_s[i]
        denom = plus_di + minus_di
        dx[i] = 100 * abs(plus_di - minus_di) / denom if denom else 0.0

    start = period * 2
    if start >= n:
        return None
    adx_val = sum(dx[period + 1 : start + 1]) / period
    for i in range(start + 1, n):
        adx_val = (adx_val * (period - 1) + dx[i]) / period
    return adx_val


def detect_trend(closes: list[float]) -> str:
    """Very small, explainable trend heuristic: compares EMA50 vs EMA200
    slope-adjusted position. Not a substitute for a full market-structure
    engine, but good enough as a confluence input alongside the LLM's own
    structural read of the swings."""
    if len(closes) < 60:
        return "RANGE"
    e50 = ema(closes, min(50, len(closes) - 1))
    e200 = ema(closes, min(200, len(closes) - 1))
    fast, slow = e50[-1], e200[-1]
    fast_prev = e50[-5] if len(e50) > 5 else e50[0]
    if fast > slow and fast > fast_prev:
        return "UPTREND"
    if fast < slow and fast < fast_prev:
        return "DOWNTREND"
    return "RANGE"


def compute_indicators(timeframe: str, candles: list[Candle]) -> TechnicalIndicators:
    closes = [c.close for c in candles]
    ema50 = ema(closes, 50)[-1] if len(closes) >= 2 else None
    ema200 = ema(closes, 200)[-1] if len(closes) >= 2 else None
    macd_line, macd_sig, macd_hist = macd(closes)
    recent = candles[-20:] if len(candles) >= 20 else candles
    return TechnicalIndicators(
        timeframe=timeframe,
        ema_50=ema50,
        ema_200=ema200,
        rsi_14=rsi(closes),
        macd=macd_line,
        macd_signal=macd_sig,
        macd_hist=macd_hist,
        atr_14=atr(candles),
        adx_14=adx(candles),
        recent_high=max((c.high for c in recent), default=None),
        recent_low=min((c.low for c in recent), default=None),
        trend=detect_trend(closes),
    )


def higher_tf_alignment(side: str, higher_tf_trend: str) -> bool:
    if higher_tf_trend == "UPTREND":
        return side.upper() == "BUY"
    if higher_tf_trend == "DOWNTREND":
        return side.upper() == "SELL"
    return False  # RANGE never counts as aligned; be conservative
