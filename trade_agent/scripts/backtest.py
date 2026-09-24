#!/usr/bin/env python3
"""Replay historical trading-bot signals through the AI decision pipeline
and compare: raw strategy vs. strategy+AI-filter vs. strategy+AI-filter+
modification. See app/backtest/harness.py for the mechanics and
docs/backtesting.md-equivalent notes in the README for the fixture format.

Usage:
    python scripts/backtest.py fixtures/sample_backtest.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.backtest.harness import load_fixtures, run_backtest  # noqa: E402
from app.config.settings import Settings  # noqa: E402
from app.providers.llm.factory import build_llm_provider  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_path", help="Path to a JSON fixture file")
    parser.add_argument(
        "--llm-provider",
        default=None,
        help="Override LLM_PROVIDER for this run (defaults to env/.env setting)",
    )
    args = parser.parse_args()

    settings = Settings(llm_provider=args.llm_provider) if args.llm_provider else Settings()
    llm = build_llm_provider(settings)
    fixtures = load_fixtures(args.fixture_path)

    report = asyncio.run(run_backtest(fixtures, settings, llm))
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
