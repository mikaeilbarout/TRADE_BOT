from __future__ import annotations

from datetime import datetime, timezone

from app.config.settings import Settings
from app.models.enums import ApprovalStatus, FinalDecision, Side, TradingMode
from app.models.pipeline_result import PipelineResult


def _executes(result: PipelineResult) -> bool:
    """Whether this decision results in a trade -- always the bot's own
    signal, never a redrawn one (MODIFY was removed 2026-09-19).

    `execution_blocked` is checked FIRST and unconditionally: an
    evidence-thin WAIT is traded through (the bot's stated policy), a
    safety WAIT (imminent high-impact event, stale quote) never is.
    """
    if result.execution_blocked:
        return False
    return result.decision in (FinalDecision.APPROVE, FinalDecision.WAIT)


class ExecutionService:
    """Decides what the API actually tells the bot to do, based on
    TRADING_MODE. This service never talks to a broker -- the bot owns
    order execution (section 13); this only shapes the response the bot
    receives and records what *would* happen.

    - PAPER: full pipeline runs and a hypothetical fill is recorded, so
      paper runs are analyzable later; the bot should simulate, not send.
    - SHADOW: pipeline runs and is recorded, but the response always tells
      the bot to follow its OWN original behavior -- the AI is observed,
      not obeyed, so its decisions can be compared against the live bot
      before it is trusted.
    - LIVE: the decision is returned as-is and gates real execution.
    - MANUAL: the decision is computed and parked awaiting human review;
      the bot must not act until a human approves it.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def finalize(self, result: PipelineResult) -> dict:
        mode = self._settings.trading_mode
        response = result.to_api_response()
        response["trading_mode"] = mode.value

        if mode == TradingMode.SHADOW:
            response["ai_decision"] = result.decision.value
            response["ai_confidence"] = result.confidence
            response["decision"] = "SHADOW_OBSERVE_ONLY"
            response["reason"] = (
                "SHADOW mode: AI decision recorded for comparison only; bot "
                f"should follow its own original logic. AI would have said "
                f"{result.decision.value}: {result.reason}"
            )
            response["modified_trade"] = None
            return response

        if mode == TradingMode.MANUAL:
            actionable = _executes(result)
            response["requires_human_approval"] = actionable
            response["approval_status"] = (
                ApprovalStatus.AWAITING_HUMAN.value
                if actionable
                else ApprovalStatus.NOT_REQUIRED.value
            )
            if actionable:
                response["reason"] = (
                    "MANUAL mode: awaiting human review before execution. "
                    + response["reason"]
                )
                response["review_url"] = f"/api/v1/decisions/{result.signal_id}"
            return response

        if mode == TradingMode.PAPER:
            response["paper_trade"] = True
            fill = self.simulate_fill(result)
            if fill is not None:
                response["hypothetical_execution"] = fill
            return response

        return response

    def simulate_fill(self, result: PipelineResult) -> dict | None:
        """Record what the fill would have been, for PAPER-mode analysis.

        Uses the ask for a BUY and the bid for a SELL from the same snapshot
        the decision was made on, and reports the gap versus the intended
        price so paper results aren't silently assumed to be perfect fills.
        """
        if not _executes(result):
            return None

        trade = result.signal
        market = result.market_snapshot
        intended = trade.entry
        if market is None:
            return {
                "filled": False,
                "reason": "no market snapshot available to price a hypothetical fill",
                "intended_entry": intended,
            }

        fill_price = market.ask if trade.side == Side.BUY else market.bid
        slippage = fill_price - intended if trade.side == Side.BUY else intended - fill_price
        slippage_pct = (abs(slippage) / intended * 100) if intended else 0.0
        volume = getattr(trade, "volume", None) or result.signal.volume

        return {
            "filled": True,
            "side": trade.side.value,
            "symbol": trade.symbol,
            "intended_entry": intended,
            "fill_price": fill_price,
            "slippage": round(slippage, 6),
            "slippage_pct": round(slippage_pct, 5),
            "volume": volume,
            "stop_loss": trade.stop_loss,
            "take_profit": trade.take_profit,
            "spread_at_fill": market.spread,
            "filled_at": datetime.now(timezone.utc).isoformat(),
            "quote_timestamp": (
                market.quote_timestamp.isoformat() if market.quote_timestamp else None
            ),
        }
