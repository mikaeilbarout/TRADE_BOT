from __future__ import annotations

from pydantic import BaseModel

"""Model registry and pricing.

Prices are USD per 1M tokens, first-party Anthropic API rates. They are a
CACHED SNAPSHOT (source: the bundled claude-api skill reference, cached
2026-06-24) -- not fetched at runtime, because a backtest's cost report must
be reproducible rather than silently re-priced later. `pricing_snapshot_date`
is recorded in every cost report so a number can always be traced to the
rate card that produced it.

Refresh procedure when rates change: update the table below and bump
PRICING_SNAPSHOT_DATE. Cost reports from earlier runs keep their own
recorded snapshot date, so historical reports stay interpretable.
"""

PRICING_SNAPSHOT_DATE = "2026-06-24"

# Multipliers applied to the INPUT rate.
CACHE_WRITE_MULTIPLIER = 1.25  # writing a cache entry costs ~1.25x input
CACHE_READ_MULTIPLIER = 0.10   # reading from cache costs ~0.1x input
BATCH_DISCOUNT = 0.50          # Message Batches run at 50% of standard rates


class ModelPricing(BaseModel):
    model_id: str
    input_per_mtok: float
    output_per_mtok: float
    context_window: int
    # Haiku 4.5 rejects `output_config.effort` and adaptive thinking; Sonnet 5
    # accepts both. Encoding it here stops the client from sending a parameter
    # that would 400 for a given model.
    supports_effort: bool
    supports_adaptive_thinking: bool


REGISTRY: dict[str, ModelPricing] = {
    "claude-haiku-4-5": ModelPricing(
        model_id="claude-haiku-4-5",
        input_per_mtok=1.00,
        output_per_mtok=5.00,
        context_window=200_000,
        supports_effort=False,
        supports_adaptive_thinking=False,
    ),
    "claude-sonnet-5": ModelPricing(
        model_id="claude-sonnet-5",
        input_per_mtok=2.00,
        output_per_mtok=10.00,
        context_window=1_000_000,
        supports_effort=True,
        supports_adaptive_thinking=True,
    ),
    "claude-opus-5": ModelPricing(
        model_id="claude-opus-5",
        input_per_mtok=5.00,
        output_per_mtok=25.00,
        context_window=1_000_000,
        supports_effort=True,
        supports_adaptive_thinking=True,
    ),
}

# The cheapest model that is still suitable for compact structured
# classification, used as the default for the three analyst agents.
CHEAPEST_ANALYST_MODEL = "claude-haiku-4-5"
DEFAULT_FINAL_MODEL = "claude-sonnet-5"


class UnknownModelError(Exception):
    """Raised when a configured model has no pricing entry.

    Deliberately fatal: running a cost-controlled experiment against a model
    whose price is unknown would make the budget guard meaningless.
    """


def get_pricing(model_id: str) -> ModelPricing:
    pricing = REGISTRY.get(model_id)
    if pricing is None:
        raise UnknownModelError(
            f"no pricing entry for model {model_id!r}. Add it to research/ai/models.py "
            f"(known: {sorted(REGISTRY)}) -- the cost budget cannot be enforced for an "
            "unpriced model."
        )
    return pricing
