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

from fastapi import FastAPI, HTTPException
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

app = FastAPI(title="Cognex")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before deploying anywhere real
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    is both the API and the site — no separate static host needed."""
    return FileResponse(STATIC_DIR / "index.html")


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
