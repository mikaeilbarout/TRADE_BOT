from __future__ import annotations

from pydantic import BaseModel

from app.models.enums import AssetType


class AssetMeta(BaseModel):
    symbol: str
    asset_type: AssetType
    base_asset: str
    quote_asset: str
    relevant_factors: list[str]
    news_categories: list[str]
    pip_size: float = 0.0001
    typical_spread_pct: float = 0.02
    # Units of the base asset per 1.0 volume/lot. Used to convert a signal's
    # stop distance into money so max_risk_per_trade_pct can be enforced.
    contract_size: float = 100_000.0


_REGISTRY: dict[str, AssetMeta] = {
    "XAUUSD": AssetMeta(
        symbol="XAUUSD",
        asset_type=AssetType.COMMODITY,
        base_asset="XAU",
        quote_asset="USD",
        relevant_factors=[
            "USD strength",
            "US real yields",
            "Federal Reserve policy",
            "inflation (CPI/PPI)",
            "interest rate expectations",
            "geopolitical risk",
            "central bank gold reserves",
        ],
        news_categories=["fed", "inflation", "geopolitics", "usd", "yields"],
        pip_size=0.01,
        typical_spread_pct=0.02,
        contract_size=100.0,  # 1.0 lot = 100 oz
    ),
    "BTCUSD": AssetMeta(
        symbol="BTCUSD",
        asset_type=AssetType.CRYPTO,
        base_asset="BTC",
        quote_asset="USD",
        relevant_factors=[
            "crypto regulation",
            "spot ETF flows",
            "Bitcoin-specific breaking news",
            "exchange security incidents",
            "risk-on/risk-off sentiment",
            "USD liquidity",
            "large institutional / whale activity",
        ],
        news_categories=["crypto_regulation", "etf", "exchange", "security", "macro_risk"],
        pip_size=1.0,
        typical_spread_pct=0.05,
        contract_size=1.0,  # 1.0 volume = 1 BTC
    ),
    "ETHUSD": AssetMeta(
        symbol="ETHUSD",
        asset_type=AssetType.CRYPTO,
        base_asset="ETH",
        quote_asset="USD",
        relevant_factors=[
            "crypto regulation",
            "ETF flows",
            "network upgrades",
            "DeFi activity",
            "risk sentiment",
            "USD liquidity",
        ],
        news_categories=["crypto_regulation", "etf", "network", "macro_risk"],
        pip_size=1.0,
        typical_spread_pct=0.06,
        contract_size=1.0,  # 1.0 volume = 1 ETH
    ),
    "EURUSD": AssetMeta(
        symbol="EURUSD",
        asset_type=AssetType.FX,
        base_asset="EUR",
        quote_asset="USD",
        relevant_factors=[
            "ECB policy",
            "Federal Reserve policy",
            "eurozone inflation",
            "US inflation",
            "employment data (NFP)",
            "GDP / PMI",
            "political/economic events in EU/US",
        ],
        news_categories=["ecb", "fed", "inflation", "employment", "pmi"],
        pip_size=0.0001,
        typical_spread_pct=0.01,
    ),
    "GBPUSD": AssetMeta(
        symbol="GBPUSD",
        asset_type=AssetType.FX,
        base_asset="GBP",
        quote_asset="USD",
        relevant_factors=[
            "BoE policy",
            "Federal Reserve policy",
            "UK inflation",
            "US inflation",
            "employment data",
            "GDP / PMI",
            "UK political events",
        ],
        news_categories=["boe", "fed", "inflation", "employment", "pmi"],
        pip_size=0.0001,
        typical_spread_pct=0.015,
    ),
    "USDJPY": AssetMeta(
        symbol="USDJPY",
        asset_type=AssetType.FX,
        base_asset="USD",
        quote_asset="JPY",
        relevant_factors=[
            "BoJ policy",
            "Federal Reserve policy",
            "US-Japan yield differential",
            "US inflation",
            "risk sentiment / carry trade flows",
            "Japan intervention risk",
        ],
        news_categories=["boj", "fed", "yields", "intervention"],
        pip_size=0.01,
        typical_spread_pct=0.01,
    ),
}

_DEFAULT_META = AssetMeta(
    symbol="UNKNOWN",
    asset_type=AssetType.FX,
    base_asset="UNKNOWN",
    quote_asset="USD",
    relevant_factors=["USD strength", "general risk sentiment"],
    news_categories=["macro_risk"],
)


def get_asset_meta(symbol: str) -> AssetMeta:
    """Look up asset metadata; falls back to a conservative generic profile
    for unmapped symbols rather than raising, so the pipeline can still run
    (agents will simply have less specific guidance)."""
    return _REGISTRY.get(symbol.strip().upper(), _DEFAULT_META)


def register_asset(meta: AssetMeta) -> None:
    """Extend the registry at runtime (e.g. from a config file on startup)."""
    _REGISTRY[meta.symbol.upper()] = meta


def all_symbols() -> list[str]:
    return list(_REGISTRY.keys())
