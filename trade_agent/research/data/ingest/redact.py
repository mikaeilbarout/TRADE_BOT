from __future__ import annotations

import re

"""Redaction of secrets from anything that becomes text.

A credential passed in a query string reaches further than people expect: into
exception messages, into the resumable checkpoint's `error` column, into the
cache's sidecar metadata, and from there into any report generated from them.
An API key in a URL is therefore redacted at every point where a URL, or an
error containing one, is turned into text -- not at the point where it is
logged, because there is always one more place that logs.

The patterns cover the query-parameter names credentialed data APIs actually
use. `redact` is deliberately applied broadly and is cheap; missing a secret is
expensive and unrecoverable once it is committed or shipped in a report.
"""

# Query parameters whose values are secrets.
_SECRET_PARAMS = (
    "api_key", "apikey", "api-key", "key", "token", "access_token",
    "auth", "auth_token", "password", "secret", "client_secret",
    "signature", "sig",
)

_QUERY_PATTERN = re.compile(
    r"(?i)\b(" + "|".join(re.escape(name) for name in _SECRET_PARAMS) + r")=([^&\s\"']*)"
)

# Authorization-style headers, in case one is ever echoed into a message.
_HEADER_PATTERN = re.compile(
    r"(?i)\b(authorization|x-api-key|x-auth-token)\s*[:=]\s*\S+"
)

REDACTED = "<REDACTED>"


def redact(text: str | None) -> str | None:
    """Mask secret values in a string, preserving its shape for diagnosis.

    `api_key=053cd0...` becomes `api_key=<REDACTED>`, so the message still says
    WHICH parameter was supplied -- which is the part that helps debugging --
    without disclosing the value.
    """
    if not text:
        return text
    masked = _QUERY_PATTERN.sub(lambda m: f"{m.group(1)}={REDACTED}", text)
    return _HEADER_PATTERN.sub(lambda m: f"{m.group(1)}={REDACTED}", masked)


def redact_url(url: str) -> str:
    """Redacted form of a URL, safe to log, store or report."""
    return redact(url) or url


def contains_secret(text: str | None) -> bool:
    """Whether a string still carries an unredacted secret.

    Used by tests and by the purge tool to assert the redaction actually held.
    """
    if not text:
        return False
    for match in _QUERY_PATTERN.finditer(text):
        if match.group(2) and match.group(2) != REDACTED:
            return True
    return bool(_HEADER_PATTERN.search(text)) and REDACTED not in text
