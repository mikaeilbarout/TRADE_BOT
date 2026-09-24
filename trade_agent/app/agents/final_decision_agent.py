from __future__ import annotations

from datetime import datetime

from app.agents.base import AgentRun, BaseAgent
from app.agents.context import chain_link, market_conditions, signal_payload
from app.config.settings import Settings
from app.models.agent_decision import (
    FinalDecisionResult,
    NewsAgentResult,
    SentimentAgentResult,
    TechnicalAgentResult,
)
from app.models.market_data import MarketSnapshot
from app.models.signal import TradeSignal
from app.services.risk_service import AccountState


def _enforced_vetoes(settings: Settings) -> list[str]:
    """Which specialist BLOCKs are still enforced deterministically.

    Derived from the live flags, never hardcoded: telling the model a veto
    will catch something it will not is worse than telling it nothing --
    it invites the model to defer an objection to a backstop that has been
    turned off.
    """
    enforced = []
    if settings.veto_on_news_block:
        enforced.append("news_agent BLOCK")
    if settings.veto_on_sentiment_block:
        enforced.append("sentiment_agent BLOCK")
    if settings.veto_on_technical_block:
        enforced.append("technical_agent BLOCK")
    return enforced


def _policy_note(settings: Settings) -> str:
    enforced = _enforced_vetoes(settings)
    veto_sentence = (
        f"{', '.join(enforced)} is vetoed deterministically after your decision "
        "regardless of scores, so do not try to work around it."
        if enforced
        else (
            "No specialist BLOCK is vetoed deterministically. A BLOCK from news, "
            "sentiment or technical is evidence for YOU to weigh -- if you do not "
            "act on it, nothing downstream will. This is the whole reason you "
            "receive all three findings."
        )
    )
    return (
        f"{veto_sentence} Regardless of the above, these are always enforced after "
        "your decision: the confidence and weighted-score minimums listed here, a "
        "high-impact event inside the blackout window, stale market data, and the "
        "hard account/risk limits."
    )


class FinalDecisionAgent(BaseAgent):
    """Link 4 of the chain: the complete chain of findings and
    recommendations from all three upstream agents, plus the latest market
    conditions and the deterministic risk context, resolved into one
    decision."""

    prompt_file = "final_agent.md"
    response_model = FinalDecisionResult
    agent_name = "final_decision_agent"
    task_instruction = (
        "Weigh the complete chain of agent findings below against the latest market "
        "conditions and risk context, then decide: APPROVE, REJECT or WAIT. "
        "Identify where the chain agrees and where it conflicts, and say which "
        "evidence you weighted most heavily and why."
    )

    def build_payload(
        self,
        signal: TradeSignal,
        market: MarketSnapshot,
        news_result: NewsAgentResult,
        sentiment_result: SentimentAgentResult,
        technical_result: TechnicalAgentResult,
        account: AccountState,
        settings: Settings,
        now: datetime | None = None,
    ) -> dict:
        return {
            "chain_position": "4 of 4 (final decision & risk)",
            "signal": signal_payload(signal, now),
            "market_conditions": market_conditions(market),
            "agent_chain": [
                chain_link(news_result),
                chain_link(sentiment_result),
                chain_link(technical_result),
            ],
            "risk_context": {
                "account_balance_reported": account.balance is not None,
                "daily_loss_pct": account.daily_loss_pct,
                "trades_today": account.trades_today,
                "open_positions": account.open_positions,
                "exposure_by_asset_pct": account.exposure_by_asset_pct,
                "current_leverage": account.current_leverage,
                "market_open": account.market_open,
            },
            "policy": {
                "min_confidence_required": settings.min_confidence,
                "min_weighted_score_required": settings.min_weighted_score,
                "high_impact_news_blackout_minutes": settings.high_impact_news_blackout_minutes,
                "dimension_weights": {
                    "news": settings.weight_news,
                    "sentiment": settings.weight_sentiment,
                    "technical": settings.weight_technical,
                    "risk": settings.weight_risk,
                },
                "min_risk_reward_ratio": settings.min_risk_reward_ratio,
                "hard_vetoes_enforced_after_your_decision": _enforced_vetoes(settings),
                "note": _policy_note(settings),
            },
        }

    async def run(
        self,
        signal: TradeSignal,
        market: MarketSnapshot,
        news_result: NewsAgentResult,
        sentiment_result: SentimentAgentResult,
        technical_result: TechnicalAgentResult,
        account: AccountState,
        settings: Settings,
        now: datetime | None = None,
    ) -> AgentRun:
        return await self._call(
            self.build_payload(
                signal,
                market,
                news_result,
                sentiment_result,
                technical_result,
                account,
                settings,
                now,
            )
        )
