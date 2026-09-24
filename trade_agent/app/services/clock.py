from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

Clock = Callable[[], datetime]


def system_clock() -> datetime:
    return datetime.now(timezone.utc)


# Every "how old is this" computation on the decision path takes its "now"
# from an injected Clock rather than calling datetime.now() itself. Live,
# the clock is the system clock and nothing changes. In a historical
# replay the clock is pinned to the signal's decision time -- without that,
# a signal from a year ago reads as a year old, every news/sentiment item
# is filtered out as stale, the session is computed from today's wall
# clock, and the agents are told in their own input that the data is
# ancient. Found by diffing a replayed payload against a live one.
