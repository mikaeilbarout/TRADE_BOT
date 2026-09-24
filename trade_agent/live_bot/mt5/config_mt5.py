"""
Bot configuration for MetaTrader5 (Pepperstone - XAUUSD).
This file must live on the same machine where the MT5 terminal is installed (Windows).
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()  # reads values from the .env file next to this project (live_bot/.env)


@dataclass
class MT5Config:
    login: int = int(os.getenv("MT5_LOGIN", "0"))         # demo account number
    password: str = os.getenv("MT5_PASSWORD", "")
    server: str = os.getenv("MT5_SERVER", "Pepperstone-Demo")  # get the exact server name from MT5
    symbol: str = "XAUUSD"
    deviation: int = 20            # allowed slippage (points)
    # NOTE: timeframe and magic-number fields used to live here
    # (higher_tf_minutes/lower_tf_minutes/magic_number) but were dead --
    # every profile module hardcodes its own ENTRY_TIMEFRAME/TREND_TIMEFRAME/
    # MAGIC_NUMBER instead. Removed 2026-09-06 during a structural cleanup.


@dataclass
class RiskConfig:
    # Updated 2026-09-09: set for a $25k prop-firm-style evaluation (8% phase-1 / 5%
    # phase-2 targets, 4% daily loss cap, 9% max loss cap). First checked against a
    # bar-based worst-case (-22R non-compounding combined across all 4 live profiles):
    # at 0.4%, worst combined drawdown looked like 8.8% of the 9% cap (97% occupancy).
    # Re-checked against REAL TICK-VERIFIED trade-by-trade data (mt5.copy_ticks_range,
    # same methodology as each profile's own tick re-verification -- see their
    # docstrings): the tick-verified worst combined drawdown was actually -10.80% at
    # 0.4% risk -- ABOVE the 9% cap, meaning 0.4% would have breached Maximum Loss on
    # this account's own worst historical stretch (2024-01-08 to 2024-02-29, mostly
    # M15+M30 losses). Lowered to 0.35%: worst combined drawdown scales linearly
    # (this metric is non-compounding by construction) to 10.80 * 0.35/0.4 = 9.45% --
    # still marginally above 9%, so this is not a hard guarantee, just a closer margin
    # than 0.4% gave; user chose 0.35% after being shown this. Worst single day scales
    # the same way: -2.40% * 0.35/0.4 = -2.10% (well under the 4% daily cap).
    #
    # Update 2026-09-09 (re-sized for the REAL FundedNext account, $6000, not the earlier
    # $25k hypothetical -- Daily Loss $300/5%, Max Loss $600/10%, Profit Target $300/5%,
    # min 3 trading days). Re-ran the worst-case check with a full COMPOUNDING equity
    # curve (not the earlier non-compounding R-multiple approximation) on the CURRENT
    # tick-verified live parameters (post EMA re-tune + spread-bug fix) for all 4
    # profiles combined on this $6000 base: 0.35% breaches badly (worst combined
    # drawdown $1380, 230% of the $600 cap -- $6000 is much smaller than $25k, so the
    # same % risk is proportionally far more dangerous here). Binary-searched the max
    # risk that stays under BOTH caps on the worst historical stretch: ~0.24% (zero
    # margin at that exact value). Chose 0.20% instead for a real safety margin: worst
    # single day $140 (47% of the $300 daily cap), worst combined drawdown $425 (71% of
    # the $600 max-loss cap).
    risk_per_trade_pct: float = 0.20
    reward_risk_ratio: float = 1.5
    atr_stop_multiplier: float = 1.5
    max_daily_loss_pct: float = 100.0  # temporarily disabled (2026-09-07) -- was 3.0, restore when asked
    time_stop_minutes: int = 45
    max_consecutive_losses: int = 0  # 0 = disabled; if set, pauses new entries for the rest of the day after this many losses in a row
    # 0 = disabled. When set (e.g. 2.0), once an open position's favorable
    # move reaches this many multiples of its ORIGINAL stop distance, the
    # stop-loss is moved to breakeven (entry price) once, server-side, and
    # left there for the rest of the trade -- never re-widened, never moved
    # again. Backtested (bar-based, 5y XAUUSD, live_bot/strategy/donchian.py's
    # canonical engine) 2026-09-17 against several trigger/lock combinations
    # (1R/2R trigger x breakeven/partial-lock) -- every earlier trigger or
    # tighter lock made results WORSE (cuts off big trend-following winners
    # too often); only trigger=2R + lock=breakeven helped, and only for M15
    # (+$1659 on $27877 baseline over 805 trades). M30 got worse (-$622) at
    # this same setting but is enabled anyway per explicit user decision,
    # H1/M1 left at 0 (disabled) since their target R is <=1.5, well under
    # where this would ever trigger.
    breakeven_at_r: float = 0.0
    # NOTE: max_open_positions used to live here but was dead -- the live bot
    # hardcodes "1 open position at a time" via an if/else on
    # get_open_position(), never reads this field. An empirical test
    # (2026-09-06, see chat history) found allowing more than 1 concurrent
    # position mostly hurts walk-forward performance, so this wasn't wired
    # up -- just removed as a misleading, unused knob.


@dataclass
class StrategyConfig:
    ema_fast: int = 50
    ema_slow: int = 200
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    require_macd_confirmation: bool = True
    atr_period: int = 14
    min_atr_pct: float = 0.02      # gold is less volatile than crypto, relatively
    max_atr_pct: float = 1.5
    min_trend_strength_pct: float = 0.0   # see strategy/signals.py::trend_direction
    require_rsi_turning: bool = False     # see strategy/signals.py::entry_signal


# --- Telegram notifications (optional) ---
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")


MT5 = MT5Config()
RISK = RiskConfig()
STRATEGY = StrategyConfig()
