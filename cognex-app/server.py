"""
Cognex — FastAPI app. This is the whole deployed service: it serves the
frontend (GET /) and the API the frontend actually calls (POST /api/ask,
POST /api/complete-story/extract — see live.py).

Run with:  uvicorn server:app --reload --port 8787
Requires:  ANTHROPIC_API_KEY set in the environment.

The /health, /personas, /decisions, /ask, /complete-story/extract endpoints
below (no /api prefix) are the ORIGINAL single-tenant reference
implementation against one hardcoded company (data.py) — kept exactly as
built so test_offline.py and demo_cli.py keep working unchanged. They're not
what the deployed frontend calls day to day; see REFERENCE_BACKEND.md.

Endpoints intentionally mirror the front-end prototype's three surfaces
(Ask Cognex, Decision Memory, Complete the Story) so it's easy to see what
maps to what.

SECURITY NOTE ON AUTH (read before deploying anything like this for real):
Every endpoint below takes `persona_id` as a query/body parameter for demo
convenience, exactly like the front-end prototype's persona switcher. That is
NOT how identity should work in production — persona_id here is a stand-in
for "whatever your auth middleware resolved from the request's session token /
SSO assertion." A real deployment resolves the caller's identity and clearance
from a verified session (e.g. an `Authorization` header validated against your
IdP) BEFORE this code ever runs, and nothing in the request body should be
able to change who the caller is. The reason the demo takes it as a parameter
is only so you can `curl` different personas without standing up real auth.
"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from data import PERSONAS, DECISIONS
from permissions import serialize_decision_for
from agent import run_agent_turn
from extract import draft_decision_from_story
import live
import slack_integration
import persistence
from db import init_db
from db_postgres import init_db as init_postgres_db

# ============ Pre-launch security pass (2026-09-04) ============
# Data in this app is confidential per company (financials, layoffs,
# acquisition talks — the whole pitch is "chain of command protection"), so
# the bar here is "would I be comfortable pointing a real customer's
# confidential data at this," not just "does it work." Three things below:
# a real CORS allowlist instead of a wildcard, response security headers on
# every request, and the interactive API docs turned off by default. See
# also live.py (the file-download Content-Disposition sanitization) and the
# Procfile (uvicorn now trusts Railway's own proxy headers).

# CORS: this is a single deployable service — the browser loads the page
# AND calls the API from the exact same origin, so a same-origin request
# never needs a CORS header to succeed at all. The allowlist below only
# matters for the handful of legitimate cross-origin cases (local dev
# against a separately-served frontend, a future integration calling the
# API directly) and, more importantly, for what it now REFUSES: the
# wildcard this replaces would let literally any website on the internet
# read responses from a signed-in user's browser via a background fetch,
# which is real data exfiltration surface on an app whose entire premise is
# per-company confidentiality — this is the fix for audit Low #33 (the
# "tighten this before deploying anywhere real" comment sitting right above
# the wildcard, since 2026-08-31, until now). Configurable via ALLOWED_ORIGINS
# (comma-separated) so the founder isn't stuck editing code to add a real
# custom domain; defaults to the known production domain plus local dev.
_DEFAULT_ALLOWED_ORIGINS = [
    "https://cognex.arviahstudio.com",
    "http://localhost:8787",
    "http://127.0.0.1:8787",
]
_allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "").strip()
ALLOWED_ORIGINS = (
    [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]
    if _allowed_origins_env
    else _DEFAULT_ALLOWED_ORIGINS
)

# Interactive API docs (/docs, /redoc) and the raw OpenAPI schema
# (/openapi.json) are FastAPI defaults meant for building against an API,
# not for a production app protecting confidential per-company data — left
# on, they hand anyone the complete route/parameter inventory of a
# multi-tenant system with real customer data behind it, for free, with no
# auth check of their own. Off unless explicitly opted into (local dev,
# or a deliberate choice to publish the API for integration partners).
_enable_api_docs = os.environ.get("ENABLE_API_DOCS", "").strip().lower() in ("1", "true", "yes")

app = FastAPI(
    title="Cognex",
    docs_url="/docs" if _enable_api_docs else None,
    redoc_url="/redoc" if _enable_api_docs else None,
    openapi_url="/openapi.json" if _enable_api_docs else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# Content-Security-Policy for the frontend this app itself serves. The whole
# frontend is one self-contained HTML file with no external script/style/
# font/image host anywhere in it (confirmed by grep before writing this —
# zero <script src>, <link rel=stylesheet>, or https:// asset references) —
# which is exactly what makes a real CSP practical here without breaking
# anything: default-src 'self' plus connect-src 'self' means even a
# successful injection can't load a remote script or exfiltrate data to an
# attacker's own server, which is the actual high-value part of this policy.
# 'unsafe-inline' on script-src/style-src is a real, known gap — this file's
# markup and logic live in large inline <script>/<style> blocks (a nonce-
# based CSP would mean restructuring how the frontend ships, out of scope
# for this pass) — so this does not stop a DOM-based XSS payload from
# EXECUTING; it stops one from phoning home or loading a second-stage
# payload from anywhere but this app's own origin. Defense in depth on top
# of (not instead of) the actual XSS fixes already shipped (esc() escaping
# quotes, the offline-answer-engine removal).
_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "font-src 'self'; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'; "
    "object-src 'none'"
)


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    """Response security headers, on every request. Each one closes a
    specific, well-known class of attack rather than being boilerplate:
    X-Frame-Options + frame-ancestors (CSP, above) both stop this app from
    being loaded inside another site's <iframe> for clickjacking; X-Content-
    Type-Options stops a browser from "helpfully" re-sniffing an uploaded
    document's content-type into something executable; Referrer-Policy keeps
    this app's own URLs (which can carry no secrets today, but shouldn't
    ever be relied on to stay that way) out of the Referer header sent to
    any third-party link; Permissions-Policy denies browser device/payment
    APIs this app has no legitimate use for, so an injected script can't
    invoke them even if it got that far; Strict-Transport-Security tells the
    browser to never downgrade to plain HTTP for this domain again, once
    it's been loaded over HTTPS once (Railway serves the production domain
    over HTTPS already — this doesn't add a redirect, which would risk a
    loop if the app ever misjudges its own scheme behind a proxy; it only
    strengthens what the browser does on its own next visit).
    """
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = (
        "geolocation=(), microphone=(), camera=(), payment=(), usb=(), "
        "magnetometer=(), gyroscope=(), interest-cohort=()"
    )
    response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    response.headers["Content-Security-Policy"] = _CSP
    return response


@app.on_event("startup")
def _startup():
    # Creates the SQLite tables (Slack connection/messages/candidates) if
    # they don't exist yet. Safe to run on every boot. See db.py.
    init_db()
    # Creates the Postgres tables (companies/personas/decisions/goals) if
    # they don't exist yet, and seeds the Cognex Labs demo company on a
    # brand-new database. Safe to run on every boot — the seed only fires
    # once, when the companies table is empty. See db_postgres.py.
    #
    # Deliberately non-fatal: on 2026-08-27 a misconfigured Postgres volume
    # made this call hang forever (no timeout on the underlying connect),
    # which took the ENTIRE app down — including the static frontend and the
    # SQLite-backed Slack integration, neither of which actually need
    # Postgres — because the whole startup hook never completed. db_postgres
    # now fails fast (an 8s connect timeout) instead of hanging, and this is
    # caught here so a Postgres outage degrades to "persistence endpoints
    # return errors" rather than "the whole site is down." See the build log
    # for the root-cause writeup.
    try:
        init_postgres_db()
    except Exception as e:
        print(f"[startup] WARNING: Postgres init failed, continuing without persistence: {e}")


# The multi-tenant, real-Claude endpoints the deployed frontend actually
# calls (POST /api/ask, POST /api/complete-story/extract, GET /api/health).
# See live.py for why this is a separate module from the single-tenant
# reference endpoints below.
app.include_router(live.router)

# Slack OAuth connect + ingestion + decision-extraction endpoints. See
# slack_integration.py for the full pipeline and its design rationale.
app.include_router(slack_integration.router)

# Real persistence — companies/personas/decisions/goals in Postgres instead
# of only in the browser tab. See persistence.py for scope and design notes.
app.include_router(persistence.router)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
def index():
    """Serves the Cognex frontend. One deployable service: this FastAPI app
    is both the API and the site — no separate static host needed.

    no-store: this is a single HTML file that changes on every deploy, with
    no filename versioning/hashing. Without an explicit no-cache directive,
    browsers can (and did, in practice — see the 2026-08-29 build log entry)
    keep serving a stale cached copy after a push, which looks exactly like
    "I pushed the fix but nothing changed" from the outside.
    """
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


def _get_persona(persona_id: str):
    persona = PERSONAS.get(persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail=f"Unknown persona_id '{persona_id}'")
    return persona


@app.get("/health")
def health():
    return {"ok": True, "api_key_configured": bool(os.environ.get("ANTHROPIC_API_KEY"))}


@app.get("/personas")
def list_personas():
    return [
        {"id": p.id, "name": p.name, "title": p.title, "level": p.level, "dept": p.dept}
        for p in PERSONAS.values()
    ]


@app.get("/decisions")
def list_decisions(persona_id: str):
    """Mirrors the Decision Memory tab: same underlying records, filtered per viewer."""
    persona = _get_persona(persona_id)
    return [serialize_decision_for(persona, d) for d in DECISIONS]


class AskRequest(BaseModel):
    persona_id: str
    message: str
    history: list[dict] = []  # prior turns in Anthropic Messages format; [] for a fresh thread


@app.post("/ask")
def ask(req: AskRequest):
    """Mirrors the Ask Cognex tab. This is the one that actually calls Claude."""
    persona = _get_persona(req.persona_id)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY is not set — see README.md to configure it.",
        )
    result = run_agent_turn(persona, req.message, req.history)
    return result


class StoryExtractRequest(BaseModel):
    work_item: str
    qa_pairs: list[dict]  # [{"question": str, "field": str, "answer": str}, ...]


@app.post("/complete-story/extract")
def complete_story_extract(req: StoryExtractRequest):
    """
    Mirrors the Complete the Story tab's draft-preview step. Returns a structured
    draft for the employee to review — does NOT write to Decision Memory. A
    separate, explicit save step (left as an exercise: it's the same
    DECISIONS.append(...) the front-end prototype already does, just server-side
    with the same visibility=company-wide / via_story=True fields) should require
    the employee to confirm the draft first.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY is not set — see README.md to configure it.",
        )
    return draft_decision_from_story(req.work_item, req.qa_pairs)
