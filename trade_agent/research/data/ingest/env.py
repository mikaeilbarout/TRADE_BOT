from __future__ import annotations

import os
from pathlib import Path

"""Load `.env` into the process environment.

The ingestion sources read credentials from `os.environ`, and the documentation
tells people to put them in `.env`. Without this, those two statements disagree
and a correctly-configured key reads as missing.

Values already present in the real environment WIN over the file: an explicitly
exported variable is a deliberate override, and silently replacing it with a
stale file value is the kind of surprise that costs an afternoon.

Nothing here logs a value. Callers that want to report configuration state use
`describe_presence`, which reports only whether a variable is set.
"""


def load_env_file(path: Path | None = None, override: bool = False) -> list[str]:
    """Load KEY=VALUE lines from `.env`. Returns the names that were set.

    Deliberately minimal rather than pulling in a parser: it handles comments,
    blank lines, `export` prefixes and surrounding quotes, which is the whole of
    what this project's `.env` contains.
    """
    path = Path(path or ".env")
    if not path.exists():
        return []

    applied: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip().strip("\"'")
        if not name or not value:
            continue
        if not override and os.environ.get(name):
            continue  # an exported value wins
        os.environ[name] = value
        applied.append(name)
    return applied


def describe_presence(names: list[str]) -> dict[str, bool]:
    """Whether each variable is set. Never returns or logs a value."""
    return {name: bool(os.environ.get(name)) for name in names}
