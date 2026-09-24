"""Automated historical-data ingestion for the XAUUSD backtest.

A thin command-line front end over `research.data.ingest`. Kept as its own
top-level package so the documented commands work verbatim:

    python -m research_data fetch-news
    python -m research_data fetch-calendar
    python -m research_data build-sentiment
    python -m research_data validate
    python -m research_data status
    python -m research_data sources
"""

from research_data.cli import main

__all__ = ["main"]
