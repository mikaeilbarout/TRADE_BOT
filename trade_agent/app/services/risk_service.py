from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from app.config.assets import get_asset_meta
from app.config.settings import Settings
from app.models.enums import Side
from app.models.market_data import MarketSnapshot
from app.models.signal import TradeSignal
from app.models.trade import ModifiedTrade


@dataclass
class AccountState:
    """Deterministic account/exposure facts the bot must supply. The AI
    agents never see broker credentials (section 31); this is just the
    numeric state needed to enforce hard limits. Defaults are conservative
    (assume nothing is open) so the service degrades safely if the bot
    doesn't yet report this."""

    balance: float | None = None
    daily_loss_pct: float = 0.0
    trades_today: int = 0
    open_positions: int = 0
    exposure_by_asset_pct: dict[str, float] = field(default_factory=dict)
    current_leverage: float = 0.0
    market_open: bool = True
    # Consecutive losing trades immediately before this one, same count the
    # live bot's own CooldownGuard tracks (live_bot/mt5/live_bot_mt5.py).
    # A full-history check (2022-06..2026-09, live M15 params, no AI) found
    # this is the one feature whose direction held in every single year:
    # win rate keeps dropping as this rises (~29% base -> ~10-20% at 1 loss
    # -> under 10% at 4+), unlike daily-trend alignment or breakout size,
    # which reversed direction across years and are not included here.
    recent_loss_streak: int = 0


@dataclass
class RiskCheckResult:
    passed: bool
    violations: list[str] = field(default_factory=list)

    def add(self, condition: bool, message: str) -> None:
        if condition:
            self.passed = False
            self.violations.append(message)


def monetary_risk(symbol: str, entry: float, stop_loss: float, volume: float) -> float:
    """Money at risk if the stop is hit, in the quote currency.

    stop distance x volume x contract size. Contract size comes from the
    asset registry, so adding an instrument means adding metadata, not
    changing this rule.
    """
    contract_size = get_asset_meta(symbol).contract_size
    return abs(entry - stop_loss) * volume * contract_size


class RiskService:
    """Deterministic, non-LLM safety rules (sections 11 and 32).

    Runs in two places:
      1. As a cheap pre-check right after signal validation, before any
         (costly) LLM agent runs -- so an obviously bad signal never reaches
         an agent.
      2. As the final execution guard after the AI decision, re-validated
         against the (possibly modified) trade -- AI approval never
         bypasses these rules.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def pre_check(
        self,
        signal: TradeSignal,
        account: AccountState,
        market: MarketSnapshot | None = None,
        now: datetime | None = None,
    ) -> RiskCheckResult:
        s = self._settings
        result = RiskCheckResult(passed=True)

        age = signal.age_seconds(now)
        result.add(
            age > s.max_signal_age_seconds,
            f"signal is stale: {age:.1f}s old "
            f"(max {s.max_signal_age_seconds}s)",
        )
        result.add(
            signal.risk_reward_ratio < s.min_risk_reward_ratio,
            f"risk/reward {signal.risk_reward_ratio:.2f} below minimum "
            f"{s.min_risk_reward_ratio}",
        )
        sl_distance_pct = (signal.risk_distance / signal.entry) * 100
        result.add(
            sl_distance_pct > s.max_stop_loss_distance_pct,
            f"stop-loss distance {sl_distance_pct:.2f}% exceeds max "
            f"{s.max_stop_loss_distance_pct}%",
        )
        result.add(
            account.daily_loss_pct >= s.max_daily_loss_pct,
            f"daily loss {account.daily_loss_pct:.2f}% at/over limit "
            f"{s.max_daily_loss_pct}%",
        )
        result.add(
            account.trades_today >= s.max_trades_per_day,
            f"trade count today {account.trades_today} at/over limit "
            f"{s.max_trades_per_day}",
        )
        result.add(
            account.open_positions >= s.max_simultaneous_positions,
            f"open positions {account.open_positions} at/over limit "
            f"{s.max_simultaneous_positions}",
        )
        result.add(
            account.exposure_by_asset_pct.get(signal.symbol, 0.0)
            >= s.max_exposure_per_asset_pct,
            f"exposure on {signal.symbol} at/over limit "
            f"{s.max_exposure_per_asset_pct}%",
        )
        result.add(
            account.current_leverage > s.max_leverage,
            f"current leverage {account.current_leverage} exceeds max {s.max_leverage}",
        )
        result.add(not account.market_open, "market is closed")

        self._check_risk_per_trade(
            result, signal.symbol, signal.entry, signal.stop_loss, signal.volume, account
        )

        if market is not None:
            self._check_market_conditions(
                result,
                market,
                entry=signal.entry,
                risk_distance=signal.risk_distance,
                check_slippage=False,
            )

        return result

    def final_guard(
        self,
        trade: TradeSignal | ModifiedTrade,
        account: AccountState,
        market: MarketSnapshot | None = None,
        original_signal: TradeSignal | None = None,
    ) -> RiskCheckResult:
        """Re-validate the exact trade about to be executed (original or
        AI-modified) with the same deterministic rules, plus the checks that
        only make sense at execution time (slippage vs. the live price, and
        coherence with the signal the bot actually sent). Nothing produced
        by an LLM can skip this -- see section 32."""
        s = self._settings
        result = RiskCheckResult(passed=True)

        risk_distance = abs(trade.entry - trade.stop_loss)
        reward_distance = abs(trade.take_profit - trade.entry)
        rr = (reward_distance / risk_distance) if risk_distance else 0.0
        sl_distance_pct = (risk_distance / trade.entry) * 100 if trade.entry else 100.0

        result.add(risk_distance == 0, "stop-loss equals entry: no defined risk")
        result.add(rr < s.min_risk_reward_ratio, f"final RR {rr:.2f} below minimum")
        result.add(
            sl_distance_pct > s.max_stop_loss_distance_pct,
            f"final SL distance {sl_distance_pct:.2f}% exceeds max",
        )
        result.add(not account.market_open, "market is closed")
        result.add(
            account.open_positions >= s.max_simultaneous_positions,
            "position limit reached",
        )
        result.add(
            account.exposure_by_asset_pct.get(trade.symbol, 0.0)
            >= s.max_exposure_per_asset_pct,
            "asset exposure limit reached",
        )

        # Directional coherence: a modification must stay the same trade.
        if trade.side == Side.BUY:
            result.add(
                not (trade.stop_loss < trade.entry < trade.take_profit),
                "BUY levels incoherent: require stop_loss < entry < take_profit",
            )
        else:
            result.add(
                not (trade.take_profit < trade.entry < trade.stop_loss),
                "SELL levels incoherent: require take_profit < entry < stop_loss",
            )

        if original_signal is not None:
            result.add(
                trade.symbol.upper() != original_signal.symbol.upper(),
                f"modified symbol {trade.symbol} does not match signal "
                f"{original_signal.symbol}",
            )
            result.add(
                trade.side != original_signal.side,
                f"modified side {trade.side.value} does not match signal "
                f"{original_signal.side.value}",
            )

        # ModifiedTrade.volume is optional, and an agent that omits it used
        # to skip this check entirely -- the one rule standing between a
        # widened stop and an oversized loss. Fall back to the volume the
        # caller actually intends to trade (the original signal's), since
        # that is what the money at risk will really be computed from.
        volume = getattr(trade, "volume", None) or (
            original_signal.volume if original_signal is not None else None
        )
        if volume:
            self._check_risk_per_trade(
                result, trade.symbol, trade.entry, trade.stop_loss, volume, account
            )
        else:
            result.add(
                self._settings.require_account_balance,
                "no volume on the trade to be executed: cannot verify "
                f"max_risk_per_trade_pct ({self._settings.max_risk_per_trade_pct}%)",
            )

        if market is not None:
            # A market order must fill near the intended price (slippage); a
            # pending/limit entry is deliberately away from price and is
            # bounded by distance instead. Treating a pullback entry as
            # "slippage" would reject exactly the modifications agents are
            # supposed to propose.
            order_type = getattr(trade, "order_type", "MARKET").upper()
            self._check_market_conditions(
                result,
                market,
                entry=trade.entry,
                risk_distance=risk_distance,
                check_slippage=order_type == "MARKET",
                check_pending_distance=order_type != "MARKET",
            )

        return result

    def _check_risk_per_trade(
        self,
        result: RiskCheckResult,
        symbol: str,
        entry: float,
        stop_loss: float,
        volume: float,
        account: AccountState,
    ) -> None:
        """Enforce max_risk_per_trade_pct. Requires the bot to report account
        balance; without it the monetary risk is unknowable, which is itself
        a violation when a limit is configured (fail closed, section 14)."""
        s = self._settings
        if account.balance is None:
            result.add(
                s.require_account_balance,
                "account balance not reported: cannot verify "
                f"max_risk_per_trade_pct ({s.max_risk_per_trade_pct}%)",
            )
            return

        result.add(account.balance <= 0, "account balance is zero or negative")
        if account.balance <= 0:
            return

        risk_money = monetary_risk(symbol, entry, stop_loss, volume)
        risk_pct = (risk_money / account.balance) * 100
        result.add(
            risk_pct > s.max_risk_per_trade_pct,
            f"risk per trade {risk_pct:.3f}% of balance exceeds max "
            f"{s.max_risk_per_trade_pct}%",
        )

    def _check_market_conditions(
        self,
        result: RiskCheckResult,
        market: MarketSnapshot,
        entry: float,
        risk_distance: float,
        check_slippage: bool,
        check_pending_distance: bool = False,
    ) -> None:
        s = self._settings
        spread_pct = (market.spread / market.last_price) * 100 if market.last_price else 0.0
        result.add(
            spread_pct > s.max_spread_pct,
            f"spread {spread_pct:.3f}% exceeds max {s.max_spread_pct}%",
        )
        result.add(
            market.is_stale,
            f"market data is stale ({market.freshness_seconds:.1f}s old, max "
            f"{s.market_data_max_staleness_seconds}s)",
        )

        # Volatility gate: if ATR dwarfs the stop distance, the stop sits
        # inside the noise band and will likely be hit at random.
        if market.atr and risk_distance > 0:
            atr_multiple = market.atr / risk_distance
            result.add(
                atr_multiple > s.max_volatility_atr_multiple,
                f"volatility too high for this stop: ATR is {atr_multiple:.2f}x the "
                f"stop distance (max {s.max_volatility_atr_multiple}x)",
            )

        if market.last_price:
            distance_pct = abs(market.last_price - entry) / market.last_price * 100
            if check_slippage:
                result.add(
                    distance_pct > s.max_slippage_pct,
                    f"price moved {distance_pct:.3f}% away from intended entry "
                    f"{entry} (max slippage {s.max_slippage_pct}%)",
                )
            if check_pending_distance:
                result.add(
                    distance_pct > s.max_pending_entry_distance_pct,
                    f"pending entry {entry} sits {distance_pct:.3f}% from the live price "
                    f"(max {s.max_pending_entry_distance_pct}%)",
                )
