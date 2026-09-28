"""Read-only web dashboard (Persian, RTL) served at "/" and "/dashboard".

The HTML itself holds no data and needs no key; it calls the protected
/api/v1 endpoints from the browser with the X-API-Key the user enters.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["dashboard"])
_PAGE = Path(__file__).resolve().parents[2] / "static" / "dashboard.html"


@lru_cache(maxsize=1)
def _html() -> str:
    return _PAGE.read_text(encoding="utf-8")


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> HTMLResponse:
    return HTMLResponse(_html(), headers={"Cache-Control": "no-cache"})
