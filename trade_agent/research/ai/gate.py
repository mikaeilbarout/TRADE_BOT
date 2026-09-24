from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.config.settings import Settings
from app.models.market_data import MarketSnapshot
from app.models.signal import TradeSignal
from app.services.risk_service import AccountState, RiskService
from research.backtest.engine import BacktestEngine
from research.config import BacktestConfig
from research.strategy.base import StrategySignal


def replay_risk_settings(settings: Settings) -> Settings:
    """Neutralize the two wall-clock rules that cannot apply during replay.

    Signal age and feed staleness are live-execution safety rules; on
    historical data they are always tripped and would reject every signal
    before any real check ran. EVERY other limit -- risk per trade, R:R, stop
    distance, exposure, leverage, spread, volatility, daily caps -- stays
    fully active, and the same object is used for the post-decision guard so
    baseline and AI runs are screened identically.
    """
    return settings.model_copy(
        update={
            "max_signal_age_seconds": float(10**9),
            "market_data_max_staleness_seconds": float(10**9),
        }
    )


def snapshot_at(
    config: BacktestConfig, bars: pd.DataFrame, bar_index: int, symbol: str
) -> MarketSnapshot:
    """MarketSnapshot from the signal bar, using only bars up to and
    including it. The quote timestamp is the bar's own close time, so the
    snapshot never carries a time the strategy could not have seen.

    Module-level so the pre-AI gate and the execution guard build the snapshot
    from identical code -- a decision screened against one view of the market
    and executed against another would be its own source of bias.
    """
    bar = bars.iloc[bar_index]
    bar_time = bar["timestamp"].to_pydatetime()
    spread = _float_or(bar.get("spread_mean"), config.costs.fallback_spread_price)
    mid = float(bar["close"])
    bid = _float_or(bar.get("bid_close"), mid - spread / 2)
    ask = _float_or(bar.get("ask_close"), mid + spread / 2)
    atr_value = _float_or(bar.get("atr"), None)

    return MarketSnapshot(
        symbol=symbol,
        bid=bid,
        ask=ask,
        last_price=mid,
        spread=spread,
        session=str(bar.get("session") or "UNKNOWN"),
        atr=atr_value,
        volatility_pct=(atr_value / mid * 100) if atr_value and mid else None,
        generated_at=bar_time,
        source="historical_bars",
        quote_timestamp=bar_time,
        is_stale=False,
        freshness_seconds=0.0,
        entry_timeframe=f"M{config.timeframe_minutes}",
    )


@dataclass
class GateResult:
    passed: bool
    violations: list[str] = field(default_factory=list)
    market_snapshot: MarketSnapshot | None = None
    sized_volume: float = 0.0
    trade_signal: TradeSignal | None = None

    @property
    def reason(self) -> str:
        return "; ".join(self.violations)


class DeterministicGate:
    """Every check that can be made without an LLM, made before any LLM call.

    This is the primary cost lever in the system: a signal rejected here
    costs zero tokens. It reuses the same `RiskService` the live trading path
    uses -- so the backtest cannot drift from production behavior -- and
    sizes the position with the same engine that will execute it, so the
    risk-per-trade limit is checked against the real volume rather than a
    placeholder.
    """

    def __init__(
        self,
        config: BacktestConfig,
        risk_service: RiskService,
        engine: BacktestEngine,
    ) -> None:
        self._config = config
        self._risk = risk_service
        self._engine = engine

    def snapshot_at(self, bars: pd.DataFrame, bar_index: int, symbol: str) -> MarketSnapshot:
        return snapshot_at(self._config, bars, bar_index, symbol)

    def check(
        self, signal: StrategySignal, bars: pd.DataFrame, account: AccountState
    ) -> GateResult:
        market = self.snapshot_at(bars, signal.bar_index, signal.symbol)

        # Size with the real engine so max_risk_per_trade_pct is enforced
        # against the volume that would actually be traded.
        balance = account.balance or self._config.risk.initial_balance
        volume, _risk_amount = self._engine.position_size(
            balance, signal.entry, signal.stop_loss
        )

        if volume < self._config.instrument.min_volume:
            return GateResult(
                passed=False,
                violations=[
                    f"position size {volume} below instrument minimum "
                    f"{self._config.instrument.min_volume}"
                ],
                market_snapshot=market,
                sized_volume=volume,
            )

        trade_signal = TradeSignal(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            side=signal.side,
            entry=signal.entry,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            volume=volume,
            timeframe=f"M{self._config.timeframe_minutes}",
            strategy="seventy_thirty",
            timestamp=signal.signal_time,
        )

        result = self._risk.pre_check(trade_signal, account, market=market)
        return GateResult(
            passed=result.passed,
            violations=result.violations,
            market_snapshot=market,
            sized_volume=volume,
            trade_signal=trade_signal,
        )


def _float_or(value, default):
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return default
    return float(value)
