"""
Live, multi-tenant-aware Cognex endpoints — this is what the deployed app
actually calls.

Why this file exists instead of just using data.py/tools.py/agent.py as-is:
those files (deliberately) model ONE hardcoded company so the reference
implementation and its offline tests stay simple and stable. The real
front-end is multi-tenant — every company's roster, decisions and goals live
in that company's own browser session (see cognex.html's `COMPANIES` object).
There is intentionally no server-side database yet (same "resets on
reload/redeploy" caveat that's been true of this prototype from the start —
see DEPLOY.md for the recommended next step of adding real persistence).

So instead of asking Claude to reason over a fixed global dataset, each
request carries the CALLING COMPANY's current decisions/goals/persona as
JSON, and this module does the same real-Claude tool-use loop and the same
permission-filtering discipline as agent.py/permissions.py — just against a
per-request snapshot instead of module-level globals.

Trust boundary: permission filtering is recomputed HERE from each record's
`visibility`/`derivedLevel` field and the persona's `level` field. It is
never trusted as already-filtered from the client. A tampered request can at
most misrepresent its own company's data; it cannot make Claude reveal a
why/risks/alternatives field that the visibility rule would otherwise strip,
because this module strips those keys unconditionally before they're ever
put in a tool_result, regardless of what the client sent or asked for.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import base64
import uuid

from anthropic import Anthropic
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel

from auth import require_any_session, require_session, token_from_request

# Added 2026-08-31 for the company Document Library's search_documents/
# get_document tools (and the "doc-" branch of GET /api/files/{file_id}
# below) -- a deliberate, narrow departure from this module's otherwise
# request-supplied-snapshot design (see the module docstring above). Every
# other tool here operates purely on the decisions/goals this request's
# body already carried; documents don't follow that pattern because a
# company's uploaded files (extracted text included) can be large and
# numerous enough that shipping them all in-line on every single /ask
# request — the way decisions/goals already are — would be wasteful for
# turns that never end up needing them. Scoped strictly to read-only
# document lookups; nothing else in this file gains DB access through this.
from db_postgres import get_session
from models import Document
from models import Persona as DBPersona
from sqlalchemy import select

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
MAX_TOOL_ROUNDS = 8

# ---------------------------------------------------------------------------
# Real image generation — added 2026-08-29 so Ask Cognex can actually
# produce an image, not just describe one in words. Claude itself has no
# image-generation capability, so this is a CUSTOM tool (like
# search_decisions below, not a server tool like web_search) whose
# implementation calls OpenAI's image API directly and hands the result
# back into the SAME Claude conversation as a generated file — Claude
# decides when a request actually calls for an image, generates it
# mid-turn, and can keep talking (captioning it, tying it to company
# context) in the same streamed answer, exactly like a code_execution chart
# already works. This was a deliberate choice over routing image requests
# to a different chat model entirely: that would lose company-memory
# grounding, streaming, and the ability to combine an image with everything
# else Ask Cognex already knows how to do in one answer.
#
# Model/quality are env-configurable rather than hardcoded, since OpenAI's
# image lineup and pricing tiers both change over time (gpt-image-2 as of
# this writing, replacing gpt-image-1.5 and the original gpt-image-1) and
# quality is a real cost/fidelity trade-off (roughly $0.006/$0.05/$0.21 per
# 1024x1024 image at low/medium/high) — "medium" is the founder's chosen
# default, per their own explicit call, not a guess.
OPENAI_IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-2")
OPENAI_IMAGE_QUALITY = os.environ.get("OPENAI_IMAGE_QUALITY", "medium")

# Generated images are opaque bytes with no home in Anthropic's Files API
# (they never came from Anthropic) — kept in this process's own memory
# instead, under an "img-" prefixed id namespace that GET /api/files/{id}
# and _resolve_files() below both recognize and branch on, so the frontend
# needs zero changes: it already renders anything in a turn's `files[]` via
# that same endpoint regardless of where the bytes actually came from.
# Deliberately NOT persisted to Postgres (avoids bloating the database with
# binary blobs for what's meant to be a short-lived, view-and-download
# artifact) and bounded to the most recent 200 images so this can't grow
# unbounded over a long-running process — this does mean a generated image
# stops being viewable after a redeploy/restart, or once 200 newer ones
# have been generated, which is an acceptable trade-off at this app's scale
# but worth knowing if usage grows.
_GENERATED_IMAGES: "dict[str, dict]" = {}
_GENERATED_IMAGES_MAX = 200


def _store_generated_image(image_bytes: bytes, mime_type: str, filename: str) -> str:
    if len(_GENERATED_IMAGES) >= _GENERATED_IMAGES_MAX:
        oldest_id = next(iter(_GENERATED_IMAGES))
        del _GENERATED_IMAGES[oldest_id]
    file_id = "img-" + uuid.uuid4().hex
    _GENERATED_IMAGES[file_id] = {"bytes": image_bytes, "mime_type": mime_type, "filename": filename, "size_bytes": len(image_bytes)}
    return file_id


GENERATE_IMAGE_SCHEMA = {
    "name": "generate_image",
    "description": (
        "Generate a real image from a text description (an illustration, logo concept, mockup, "
        "social/marketing graphic, or any other visual asset someone asks to be CREATED). Returns "
        "the generated image as a file the person can view and download. Do NOT use this for a "
        "chart, graph, or plot of data or numbers — use code_execution for that instead, since it "
        "produces an accurate chart from real values rather than an AI-imagined picture of one."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "A clear, detailed, self-contained description of the image to generate — the image model has no other context, so include style, subject, composition, and any specifics that matter."},
            "size": {"type": "string", "enum": ["1024x1024", "1024x1536", "1536x1024"], "description": "Image dimensions. Default to the square size unless the request clearly implies a portrait or landscape shape."},
        },
        "required": ["prompt"],
    },
}


def generate_image(prompt: str, size: str = "1024x1024") -> dict:
    """Implementation behind the generate_image tool. Returns a small dict
    (not the raw image) as the tool_result Claude sees — just enough for
    Claude to know it worked and talk about it; the actual bytes are
    resolved into the turn's files[] separately (see _resolve_files),
    mirroring exactly how a code_execution-generated chart's file_id
    already flows through this same pipeline."""
    if not os.environ.get("OPENAI_API_KEY"):
        return {"error": "Image generation isn't configured yet — OPENAI_API_KEY is not set on the server."}
    if size not in ("1024x1024", "1024x1536", "1536x1024"):
        size = "1024x1024"
    try:
        client = OpenAI()
        result = client.images.generate(
            model=OPENAI_IMAGE_MODEL, prompt=prompt, size=size,
            quality=OPENAI_IMAGE_QUALITY, n=1, output_format="png",
        )
        item = (result.data or [None])[0]
        if not item or not item.b64_json:
            return {"error": "The image model returned no image data."}
        image_bytes = base64.b64decode(item.b64_json)
    except Exception as e:
        return {"error": f"Image generation failed: {e}"}
    file_id = _store_generated_image(image_bytes, "image/png", "generated-image.png")
    return {"file_id": file_id, "prompt": prompt}

# ---------------------------------------------------------------------------
# Real web search + real web fetch + real code execution — added 2026-08-29
# so Ask Cognex can actually search the web, actually retrieve a specific
# page, and actually run numbers/build charts, instead of only answering
# from the model's own knowledge and prose estimates. These are Anthropic
# *server* tools: unlike search_decisions/get_decision/etc. below (which we
# implement and execute ourselves), Anthropic's API executes these itself
# and returns the result inline in the same response — see the tool_use
# handling in run_agent_turn for why that changes the loop slightly.
#
# web_search vs. web_fetch — these solve different problems and Claude picks
# between them: web_search queries a search index, so it can miss a small,
# new, or lightly-indexed site entirely even though the site is live and
# real (hit this in production: asking about the founder's own studio site
# came back "no results," because web_search just never indexed it). web_fetch
# directly retrieves a specific URL's content — the same thing Claude.ai does
# when you paste a link — and works regardless of search-index coverage. Both
# are included so Claude can use whichever fits: search when it doesn't have
# a URL yet, fetch when it does (a URL from the user's own message, or one
# web_search just returned). web_fetch is also a genuine safety design, not
# just a feature: per Anthropic's docs it can ONLY fetch a URL that already
# appeared somewhere in the conversation (the user's message, a tool result,
# a prior search result) — it cannot fetch a URL Claude invents on its own.
#
# max_uses=5 on each bounds worst-case cost per turn. web_search is
# $10/1,000 searches plus normal token cost for retrieved pages; web_fetch
# has no separate charge beyond normal token cost per Anthropic's pricing
# docs.
#
# IMPORTANT: both web_search AND web_fetch are pinned to their basic/oldest
# versions (20250305 / 20250910) deliberately, NOT their newer 20260209+
# versions. Anthropic's newer versions of both tools add "dynamic
# filtering" — they auto-provision their OWN internal code_execution tool to
# filter results before they reach the context window. If we ALSO declare
# our own code_execution tool (which we need, for genuine user-requested
# computation/charts, not just result filtering), the API rejects the
# request outright: "Auto-injecting tools would conflict with existing tool
# names: ['code_execution']" — hit this live in production on 2026-08-29 with
# web_search (see that build-log entry); web_fetch's dynamic-filtering
# versions would collide the same way, so it's pinned old from the start.
# Trade-off: with these basic versions, code_execution is billed normally
# (by execution time, 5-minute minimum per invocation) rather than being
# free — but Anthropic's free tier (1,550 container-hours/month per org, per
# their pricing docs) should comfortably cover this app's realistic usage.
SERVER_TOOLS = [
    {"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
    {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 5},
    {"type": "code_execution_20250825", "name": "code_execution"},
]

LEVEL_LABEL = {1: "Company-wide", 2: "Employee+", 3: "Manager+", 4: "Director+", 5: "Executive", 6: "CEO only"}

# ---------------------------------------------------------------------------
# Basic per-company rate limit + cost cap — added 2026-08-31 after an audit
# found nothing stopped one user (or anyone who had the shared demo
# password) from running up an unbounded Anthropic/OpenAI bill: unlimited
# Ask Cognex turns, each able to call web_search/code_execution/
# generate_image with no per-company ceiling anywhere. Deliberately basic,
# not a billing system:
#
# - The RATE limit is a short in-memory sliding window (bursts, not spend) —
#   it doesn't need to survive a restart, since its only job is smoothing a
#   short burst of requests, not tracking cumulative usage.
# - The COST cap is a running per-company, per-calendar-month estimated-
#   dollar total, persisted to Postgres (CompanyUsage in models.py) so it
#   actually survives a redeploy — a cap that resets every deploy isn't a
#   cap. Cost is estimated from real token/image usage at the same published
#   per-unit prices used in this project's own unit-economics work ($3/MTok
#   input, $15/MTok output for Claude; OpenAI image cost by quality tier) —
#   this is a circuit breaker against runaway spend, not reconciled billing.
#
# Both are env-configurable so the founder can raise/lower them per
# environment without a code change.
COMPANY_MONTHLY_CAP_USD = float(os.environ.get("COGNEX_COMPANY_MONTHLY_CAP_USD", "15"))
CLAUDE_INPUT_USD_PER_MTOK = 3.00
CLAUDE_OUTPUT_USD_PER_MTOK = 15.00
IMAGE_COST_USD = {"low": 0.006, "medium": 0.05, "high": 0.21}

_RATE_WINDOW_SECONDS = 60
_RATE_MAX_PER_WINDOW = int(os.environ.get("COGNEX_COMPANY_RATE_PER_MIN", "20"))
_rate_state: "dict[str, list]" = {}  # company_id -> unix timestamps within the current window


def _check_rate_limit(company_id: str):
    if not company_id:
        return  # no company_id given (e.g. an old cached frontend) — nothing to key the limit on; fails open, not closed
    import time
    now = time.time()
    bucket = _rate_state.setdefault(company_id, [])
    cutoff = now - _RATE_WINDOW_SECONDS
    while bucket and bucket[0] < cutoff:
        bucket.pop(0)
    if len(bucket) >= _RATE_MAX_PER_WINDOW:
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests from this company in the last minute (limit {_RATE_MAX_PER_WINDOW}/min) — wait a moment and try again.",
        )
    bucket.append(now)


def _current_usage_period() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m")


def _check_cost_cap(company_id: str):
    if not company_id:
        return
    from db_postgres import get_session
    from models import CompanyUsage
    period = _current_usage_period()
    with get_session() as db:
        row = db.get(CompanyUsage, (company_id, period))
        spent = (row.estimated_cost_cents / 100.0) if row else 0.0
    if spent >= COMPANY_MONTHLY_CAP_USD:
        raise HTTPException(
            status_code=429,
            detail=f"This company has reached its usage cap for this month (${COMPANY_MONTHLY_CAP_USD:.2f}). Contact Cognex to raise it.",
        )


def _record_cost(company_id: str, usd: float):
    if not company_id or usd <= 0:
        return
    from db_postgres import get_session
    from models import CompanyUsage
    period = _current_usage_period()
    cents = round(usd * 100)
    with get_session() as db:
        row = db.get(CompanyUsage, (company_id, period))
        if row:
            row.estimated_cost_cents += cents
        else:
            db.add(CompanyUsage(company_id=company_id, period=period, estimated_cost_cents=cents))
        db.commit()


def _usage_cost_usd(usage) -> float:
    """Estimated USD cost of one Claude response.usage object. Never raises
    on a missing/odd-shaped usage object — a cost estimate that fails to
    compute should degrade to "count it as free this round," not break the
    actual answer the person is waiting on."""
    if not usage:
        return 0.0
    try:
        inp = getattr(usage, "input_tokens", 0) or 0
        out = getattr(usage, "output_tokens", 0) or 0
        return (inp / 1_000_000.0) * CLAUDE_INPUT_USD_PER_MTOK + (out / 1_000_000.0) * CLAUDE_OUTPUT_USD_PER_MTOK
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Request-scoped data shapes (mirrors the frontend's camelCase JS objects)
# ---------------------------------------------------------------------------
@dataclass
class Persona:
    id: str
    name: str
    title: str
    level: int
    dept: str = ""


@dataclass
class Decision:
    id: str
    title: str
    decided: str = ""
    owner: str = ""
    visibility: int = 1
    derivedLevel: int = 1
    why: list = field(default_factory=list)
    alternatives: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)
    risks: list = field(default_factory=list)
    review: str = ""
    result: str = ""
    tags: list = field(default_factory=list)
    derived: str = ""
    contributor: Optional[str] = None
    domain: str = "other"


@dataclass
class Goal:
    id: str
    title: str
    kicker: str = ""
    depth: int = 0
    parent: Optional[str] = None
    dept: Optional[str] = None
    owner: Optional[str] = None


def _tokenize(s: str) -> set:
    return set(re.sub(r"[^a-z0-9\s]", " ", (s or "").lower()).split())


# ---------------------------------------------------------------------------
# Permission enforcement — same discipline as permissions.py, parameterized
# by an int clearance level instead of a bound Persona object.
# ---------------------------------------------------------------------------
def decision_view(level: int, d: Decision) -> str:
    if level >= d.visibility:
        return "source"
    if level >= d.derivedLevel:
        return "derived"
    return "none"


def serialize_decision_for(level: int, d: Decision) -> dict:
    mode = decision_view(level, d)
    base = {
        "id": d.id, "title": d.title, "decided": d.decided, "owner": d.owner,
        "clearance_required": LEVEL_LABEL.get(d.visibility, "?"), "access": mode,
        "domain": d.domain,
    }
    if mode == "source":
        base.update({
            "why": d.why, "alternatives": d.alternatives, "assumptions": d.assumptions,
            "risks": d.risks, "review": d.review, "result": d.result,
        })
        if d.contributor:
            base["reported_by_note"] = d.contributor
    elif mode == "derived":
        base["derived_summary"] = d.derived
        base["note"] = (
            "The underlying decision record is restricted above this viewer's clearance. "
            "Only the pre-approved derived summary is available — do not imply you have "
            "seen the raw why/alternatives/risks, because you have not been given them."
        )
    else:
        base["note"] = "This record and its derived context both require higher clearance than this viewer has."
    return base


# ---------------------------------------------------------------------------
# Tools — bound to this request's decisions/goals via closures in run_agent_turn
# ---------------------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "name": "search_decisions",
        "description": (
            "Search company Decision Memory (what was decided, why, alternatives considered, "
            "risks, review date, result) for records relevant to a query. Returns only what the "
            "current viewer's clearance allows — some results will be a 'derived' summary instead "
            "of the full record, or a 'none' stub with no content. Never claim to know more than "
            "what a result actually contains."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Free-text search query, e.g. 'Germany market entry'."}},
            "required": ["query"],
        },
    },
    {
        "name": "get_decision",
        "description": "Fetch a single decision record by id, filtered to the current viewer's clearance.",
        "input_schema": {"type": "object", "properties": {"decision_id": {"type": "string"}}, "required": ["decision_id"]},
    },
    {
        "name": "search_goals",
        "description": "Search the company/department/team/individual goal hierarchy for records relevant to a query.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    {
        "name": "get_goal_chain",
        "description": (
            "Given a goal id, return the full chain from that goal up to the company-level goal it "
            "ladders up to. Use this to answer 'why am I working on X' or 'how does this connect to "
            "company strategy' questions."
        ),
        "input_schema": {"type": "object", "properties": {"goal_id": {"type": "string"}}, "required": ["goal_id"]},
    },
    {
        "name": "get_my_context",
        "description": (
            "Get the current viewer's own role, department, clearance level, and individual goal "
            "(if they have one). Call this first when a question is about 'my' work, team, or priorities."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "search_documents",
        "description": (
            "Search this company's uploaded document library (files people have added — contracts, "
            "specs, reports, notes, etc.) by keyword, matching against filename and extracted text. "
            "Every result is company-wide (this library has no per-document clearance model, unlike "
            "Decision Memory) — call get_document to read a specific one's full extracted text once "
            "you've found it here. A document with extraction_status 'not_indexed' or 'failed' can't "
            "be searched by content this way (only by filename) — say so plainly rather than implying "
            "you've read a file this tool couldn't actually extract text from."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Free-text search query, e.g. 'Q3 marketing budget'."}},
            "required": ["query"],
        },
    },
    {
        "name": "get_document",
        "description": "Fetch a single uploaded document's full extracted text by id (from search_documents).",
        "input_schema": {"type": "object", "properties": {"document_id": {"type": "string"}}, "required": ["document_id"]},
    },
    GENERATE_IMAGE_SCHEMA,
]


def _make_tool_impls(persona: Persona, decisions: list, goals: list, company_id: str = ""):
    def goal_by_id(goal_id):
        return next((g for g in goals if g.id == goal_id), None)

    def search_decisions(query: str):
        q_words = _tokenize(query)
        scored = []
        for d in decisions:
            score = 0
            for tag in d.tags:
                tw = _tokenize(tag)
                if tw and tw.issubset(q_words):
                    score += 4 if len(tw) >= 2 else 2
            for w in _tokenize(d.title):
                if len(w) > 3 and w in q_words:
                    score += 2
            # Also search the actual saved content (why/alternatives/
            # assumptions/risks/result/derived), not just tags and title.
            # Added 2026-08-29: a decision saved from an Ask Cognex chat gets
            # its tags/title auto-generated from the QUESTION that triggered
            # the save (see saveChatTurnToMemory in the frontend), which very
            # often shares no words at all with a later, differently-phrased
            # question — even though the real answer is sitting right there
            # in the saved body text. Without this, a decision's actual
            # content was effectively invisible to search the moment its
            # title/tags didn't happen to overlap with a later query.
            # Weighted lower than a tag/title hit (a body-text match is a
            # weaker relevance signal) but this is what makes free-text
            # saved content actually findable.
            body_words = set()
            for field in (d.why or []):
                body_words |= _tokenize(field)
            for field in (d.alternatives or []):
                body_words |= _tokenize(field)
            for field in (d.assumptions or []):
                body_words |= _tokenize(field)
            for field in (d.risks or []):
                body_words |= _tokenize(field)
            if d.result:
                body_words |= _tokenize(d.result)
            if d.derived:
                body_words |= _tokenize(d.derived)
            score += sum(1 for w in q_words if len(w) > 3 and w in body_words)
            if score > 0:
                scored.append((score, d))
        scored.sort(key=lambda pair: -pair[0])
        return [serialize_decision_for(persona.level, d) for _, d in scored[:5]]

    def get_decision(decision_id: str):
        d = next((x for x in decisions if x.id == decision_id), None)
        if not d:
            return {"error": f"No decision with id '{decision_id}'."}
        return serialize_decision_for(persona.level, d)

    def search_goals(query: str):
        q_words = _tokenize(query)
        results = []
        for g in goals:
            tw = _tokenize(g.title)
            matches = any(len(w) > 3 and w in q_words for w in tw)
            matches = matches or (g.dept and g.dept.lower() in q_words)
            if matches:
                results.append(g)
        return [{"id": g.id, "kicker": g.kicker, "title": g.title, "dept": g.dept, "depth": g.depth} for g in results]

    def get_goal_chain(goal_id: str):
        g = goal_by_id(goal_id)
        if not g:
            return [{"error": f"No goal with id '{goal_id}'."}]
        chain, cur = [g], g
        while cur.parent:
            cur = goal_by_id(cur.parent)
            if not cur:
                break
            chain.insert(0, cur)
        return [{"id": x.id, "kicker": x.kicker, "title": x.title} for x in chain]

    def get_my_context():
        my_goal = next((g for g in goals if g.owner == persona.id), None)
        return {
            "name": persona.name, "title": persona.title, "department": persona.dept,
            "clearance_level": persona.level,
            "individual_goal": ({"id": my_goal.id, "title": my_goal.title} if my_goal else None),
        }

    def _document_summary(d) -> dict:
        preview = (d.text_content or "")[:280]
        return {
            "id": d.id, "title": d.filename, "access": "source",
            "extraction_status": d.extraction_status,
            "preview": preview + ("…" if len(d.text_content or "") > 280 else ""),
        }

    def search_documents(query: str):
        if not company_id:
            return []
        q_words = _tokenize(query)
        with get_session() as db:
            rows = db.execute(select(Document).where(Document.company_id == company_id)).scalars().all()
        scored = []
        for d in rows:
            score = 0
            for w in _tokenize(d.filename or ""):
                if len(w) > 2 and w in q_words:
                    score += 3
            body_words = _tokenize(d.text_content or "")
            score += sum(1 for w in q_words if len(w) > 3 and w in body_words)
            if score > 0:
                scored.append((score, d))
        scored.sort(key=lambda pair: -pair[0])
        return [_document_summary(d) for _, d in scored[:5]]

    def get_document(document_id: str):
        if not company_id:
            return {"error": "No company context for this request."}
        with get_session() as db:
            d = db.get(Document, document_id)
        if not d or d.company_id != company_id:
            return {"error": f"No document with id '{document_id}' in this company's library."}
        if d.extraction_status != "indexed":
            return {
                "id": d.id, "title": d.filename, "access": "source",
                "extraction_status": d.extraction_status,
                "note": f"This file's text could not be extracted (status: {d.extraction_status}) — you cannot read its contents, only confirm it exists and was uploaded.",
            }
        return {"id": d.id, "title": d.filename, "access": "source", "extraction_status": d.extraction_status, "text": d.text_content}

    return {
        "search_decisions": lambda tool_input: search_decisions(tool_input.get("query", "")),
        "get_decision": lambda tool_input: get_decision(tool_input.get("decision_id", "")),
        "search_goals": lambda tool_input: search_goals(tool_input.get("query", "")),
        "get_goal_chain": lambda tool_input: get_goal_chain(tool_input.get("goal_id", "")),
        "get_my_context": lambda tool_input: get_my_context(),
        "search_documents": lambda tool_input: search_documents(tool_input.get("query", "")),
        "get_document": lambda tool_input: get_document(tool_input.get("document_id", "")),
        "generate_image": lambda tool_input: generate_image(tool_input.get("prompt", ""), tool_input.get("size", "1024x1024")),
    }


def _decisions_index(persona: Persona, decisions: list) -> str:
    """A compact, always-present index of what's currently in this company's
    Decision Memory — title plus a short preview — injected directly into
    the system prompt on every turn, filtered to what this viewer is
    allowed to see at all (source or derived; fully-restricted records are
    left out entirely rather than teased).

    Added 2026-08-29 in direct response to a real bug report: the founder
    asked Cognex to research their own site and save it as company info in
    one chat, then in a brand-new chat asked "how can we get new
    customers" and got an answer that didn't know what the company sells.
    Broadening search_decisions to full-text search (see above) helps, but
    doesn't fully solve it — the model still has to decide to call the tool
    and guess a query that happens to match. This index removes that
    dependency for anything foundational: the assistant can see what
    exists in memory on every turn without searching for it first, and
    call get_decision(id) for the full record when something here is
    actually relevant. This is what "the model should naturally have
    context of everything saved" means in practice, without dumping every
    full decision record into every system prompt."""
    visible = []
    for d in decisions:
        mode = decision_view(persona.level, d)
        if mode == "none":
            continue
        preview = ""
        if mode == "source" and d.why:
            preview = d.why[0]
        elif mode == "derived" and d.derived:
            preview = d.derived
        preview = (preview[:160] + "…") if len(preview) > 160 else preview
        visible.append((d, preview))
    if not visible:
        return "(Decision Memory is currently empty for this company.)"
    visible.sort(key=lambda pair: pair[0].decided or "", reverse=True)
    lines = []
    for d, preview in visible[:40]:
        line = f'- [{d.domain}] {d.id}: "{d.title}"'
        if preview:
            line += f" — {preview}"
        lines.append(line)
    return "\n".join(lines)


def _system_prompt(persona: Persona, decisions: list) -> str:
    import datetime as _dt
    today_str = _dt.datetime.now(_dt.timezone.utc).strftime("%A, %B %d, %Y")
    return f"""Today's real date is {today_str}. Use this, not any date you might otherwise assume, for
anything involving "today," "overdue," "how long ago," or similar — decision `review` dates and
similar time-sensitive fields must be judged against this actual date, not a guess.

You are Cognex, the AI assistant {persona.name} ({persona.title}, {persona.dept}
department, clearance level {persona.level} of 6) uses for day-to-day company work — research,
writing, coding, documentation, analysis, debugging, brainstorming, and anything else they'd
otherwise reach for a general-purpose AI assistant to do. Answer those requests fully and
directly, the same way you would in any other context: complete, well-structured answers, real
working code with explanation when asked to write code, thorough research or documentation when
asked for it. Do not artificially shorten an answer that deserves depth — match length and
structure to what the question actually needs, not to a fixed target.

You have four real capabilities beyond your own training, and you should reach for them whenever
they would make an answer better rather than defaulting to a prose guess:
- Web search: use it for anything current, time-sensitive, or better answered with a real source
  (prices, news, a competitor's current product, a fact you're not fully certain of) — don't
  guess at something searchable. Do NOT put confidential company details (decision content,
  financial figures, names of people or deals) into a search query — search generically; if a
  question genuinely needs company-internal information, use your company-memory tools below
  instead of the web for that part.
  CRITICAL when asked for a list of specific real-world entities — people, accounts/handles
  (Instagram, TikTok, etc.), companies, contacts, or similar — that must actually exist and be
  usable: search is not a database of those entities, and a general web index often can't verify
  something as specific as "an Instagram account run by a jewelry-focused micro-influencer under
  10k followers." If, after searching, you cannot confirm a specific name/handle/account is real
  from what the search actually returned, do NOT invent a plausible-sounding one to fill the
  list — a fabricated handle that doesn't exist is worse than no answer, because it looks
  verified when it isn't. Instead: report what you genuinely found (even a shorter or partial
  list), say plainly that general web search has limited coverage of a platform like Instagram
  and can't reliably verify individual accounts, and suggest a better path for this specific kind
  of request (e.g. searching directly within Instagram/the platform in question, a specialized
  influencer-discovery tool, or hashtag/location browsing) rather than presenting invented
  results as if they were real. This same rule applies to any other request for specific,
  checkable real-world facts (a person's contact details, a real company's stated pricing, a
  specific citation) — state what you verified, and flag plainly whatever you could not.
  CRITICAL for exchange rates, currency conversions, stock/crypto prices, interest rates, or any
  other value that moves day to day: whatever number you remember from training is guaranteed to
  be stale — these are exactly the facts that change daily, so a confident-sounding memorized
  number carries no signal that it's wrong. Reported directly by a user (2026-08-31): asked for an
  INR/USD conversion and got a "rough estimate" of ₹85 = $1 pulled from memory, when the real rate
  that day was over ₹95 — more than 10% off, stated as fact. Never state or calculate with a
  currency rate, stock price, or similar figure recalled from training, and never soften it as a
  "rough estimate" as a way of using a memorized number anyway — that framing still hands the
  person a wrong number, just with a hedge attached. Always call web_search for the actual current
  figure first, then use that real, sourced number for anything that follows (including any
  code_execution calculation). If search doesn't return a clear current figure, say so plainly and
  ask for the number rather than filling the gap with a memorized approximation.
  CRITICAL for a business input the person hasn't told you and that isn't a fact you can look up
  (a cost, budget, headcount, revenue figure, price, timeline, or any other number specific to
  their situation): never silently invent a plausible-sounding value and compute with it as if it
  were given. Reported directly by a user (2026-08-31): asked a question involving a business
  calculation, never mentioned any cost figure, and the answer silently assumed a fixed cost of
  $90,000 out of nowhere and calculated from it as if the person had said so. This is the same
  failure as the exchange-rate case above — a confident number with no real source behind it — but
  for a fact that has no "real" value to look up at all, since it lives only in the person's head
  or their own records, not in web_search or training data. When a calculation needs a number like
  this and the person hasn't given it, and it isn't already sitting in Decision Memory or Goals
  (check first), do one of two things: ask them for it before calculating, or — if a
  ballpark answer is still useful without stopping to ask — state the assumption plainly as an
  assumption (e.g. "assuming $0 since no cost was given" or "assuming no fixed cost was mentioned,
  so treating it as $0"), never as a fact, and make clear the real answer will change once they
  supply the actual number. Do not default to $0 (or any other placeholder) silently — either ask,
  or say out loud that you're assuming it.
- Web fetch: when you have (or the person gives you) a specific URL or a named site/domain — not
  just a general topic — fetch it directly instead of only searching for it. Search queries a
  search index and can come back empty for a small, new, or lightly-indexed site even though the
  site is live and real; fetching the URL directly works regardless. If a search doesn't surface
  something the person clearly expects to be findable and they named a specific site, try
  fetching it directly before concluding it doesn't exist.
- Code execution (a real Python sandbox): use it whenever a question involves actual computation,
  data analysis, or a chart — run the numbers for real instead of estimating them in prose, and
  generate an actual chart image when a visualization would help instead of describing one in
  words. If the person gives you data (pasted numbers, a table), work with it directly in code. If
  a calculation depends on a live value you don't already have from the conversation — an exchange
  rate, a stock price, an interest rate, current pricing — get that value from web_search first and
  feed the real number into the code; do not hardcode a rate or price you recall from training,
  per the web search rule above.
- Image generation: use it when someone asks you to actually CREATE a visual asset — an
  illustration, a logo concept, a mockup, a social/marketing graphic, or any other picture that
  doesn't come from real data. Write a clear, detailed, self-contained prompt (the image model has
  no other context — describe subject, style, and composition explicitly) rather than a short
  restatement of the request. Do NOT use this for a chart, graph, or plot of real numbers — use
  code_execution for that instead, since it draws an accurate chart from real values rather than
  an AI-imagined picture of one.

You ALSO have privileged, permission-filtered access to this company's Decision Memory and Goal
graph through your tools. Use them whenever a question is actually about this company's
decisions, goals, or priorities, and ground that specific part of your answer in what a tool
call returns — never guess at or invent company-internal facts (who decided what, why, what the
company's goals are) that your tools didn't give you. Every tool call is automatically filtered
to what this person is personally allowed to see; if a result says a record is restricted or
only a derived summary is available, say so plainly instead of guessing at the full record. This
grounding requirement applies specifically to company-internal facts — it does not apply to
general knowledge, web search results, coding help, or open research/computation, which you
should answer using your own judgment and tools exactly as you normally would.

You also have search_documents/get_document tools over this company's uploaded document library
(contracts, specs, reports, notes — whatever people have added). Unlike Decision Memory, there's
no live index of these injected into this prompt (the library can hold real files, not just short
text, so it isn't cheap to summarize on every turn) — call search_documents whenever a question
plausibly involves something someone would have uploaded rather than typed into a decision or
goal (e.g. "what does the contract say about renewal terms", "summarize the Q3 report") before
concluding the information doesn't exist. A document whose extraction_status isn't "indexed"
couldn't be read by these tools at all (only its filename is known) — say so plainly rather than
implying you've seen its contents.

Here is an index of everything currently in this company's Decision Memory that you're allowed
to see at all (title plus a short preview) — this is a live index, not a static list, so treat it
as current fact about what's been captured, including anything saved earlier in a completely
different chat thread. Skim it before answering anything about the company itself — what it
does, sells, or is planning — even if the question doesn't obviously sound like a "decision
memory" question (e.g. "how do we get new customers" should make you check this index for
anything about the product, market, or customers before answering from general reasoning alone).
When something here looks relevant, call get_decision(id) to pull the full record rather than
answering from the title/preview alone:
{_decisions_index(persona, decisions)}

When you do cite something retrieved from company memory, name what you retrieved so the person
can see where it came from. When you cite something from a web search, name the source. When you
compute or generate something with code, state the key numbers or say what the chart shows —
don't just say "see attached." """


def _extract_file_ids(content_blocks) -> list:
    """Pull file_ids for anything code_execution generated (a chart image, a
    CSV, etc.) out of a response's content blocks. Uses getattr-based duck
    typing rather than strict isinstance checks against the SDK's typed
    blocks, so a minor result-shape difference degrades to "no files found"
    rather than a hard crash on an otherwise-good answer."""
    ids = []
    for block in content_blocks:
        if getattr(block, "type", None) not in ("code_execution_tool_result", "bash_code_execution_tool_result"):
            continue
        result = getattr(block, "content", None)
        items = getattr(result, "content", None)  # e.g. CodeExecutionResultBlock.content: List[CodeExecutionOutputBlock]
        if not items:
            continue
        for item in items:
            fid = getattr(item, "file_id", None)
            if fid:
                ids.append(fid)
    return ids


def _resolve_files(client, file_ids: list) -> list:
    """Turn raw file_ids into {id, filename, mimeType, sizeBytes} the frontend
    can render (an inline image for a chart, a download link otherwise) via
    GET /api/files/{id}. A metadata lookup failure doesn't drop the file
    entirely — the id is still usable — it just won't have a nice filename.

    An "img-" prefixed id is one of our own OpenAI-generated images (see
    _GENERATED_IMAGES above) and never came from Anthropic's Files API, so
    those are resolved straight out of the in-process store instead of
    calling client.files.retrieve_metadata (which would just 404 on an id
    Anthropic has never heard of)."""
    files, seen = [], set()
    for fid in file_ids:
        if fid in seen:
            continue
        seen.add(fid)
        if fid.startswith("img-"):
            rec = _GENERATED_IMAGES.get(fid)
            if rec:
                files.append({"id": fid, "filename": rec["filename"], "mimeType": rec["mime_type"], "sizeBytes": rec["size_bytes"]})
            else:
                files.append({"id": fid, "filename": "generated-image.png", "mimeType": "image/png", "sizeBytes": None})
            continue
        try:
            meta = client.files.retrieve_metadata(fid)
            files.append({"id": fid, "filename": meta.filename, "mimeType": meta.mime_type, "sizeBytes": meta.size_bytes})
        except Exception:
            files.append({"id": fid, "filename": "output", "mimeType": "application/octet-stream", "sizeBytes": None})
    return files


def run_agent_turn(persona: Persona, decisions: list, goals: list, user_message: str, history=None, company_id: str = ""):
    # Rate/cap CHECKS deliberately live in the route handlers below, not
    # here — an HTTPException raised from inside this function would get
    # swallowed by the route's `except Exception: _log_and_raise_502(...)`
    # and turned into a misleading 502. This function only RECORDS the
    # actual cost incurred (see the `finally` block), regardless of which
    # caller invoked it.
    client = Anthropic()
    impls = _make_tool_impls(persona, decisions, goals, company_id)
    messages = list(history or [])
    messages.append({"role": "user", "content": user_message})
    tool_call_log = []
    file_ids = []
    all_tools = TOOL_SCHEMAS + SERVER_TOOLS
    turn_cost_usd = 0.0

    try:
        for _ in range(MAX_TOOL_ROUNDS):
            response = client.messages.create(
                model=MODEL, max_tokens=8192, system=_system_prompt(persona, decisions),
                tools=all_tools, tool_choice={"type": "auto"}, messages=messages,
            )
            turn_cost_usd += _usage_cost_usd(getattr(response, "usage", None))
            file_ids.extend(_extract_file_ids(response.content))

            if response.stop_reason != "tool_use":
                final_text = "".join(block.text for block in response.content if block.type == "text")
                return {"answer": final_text, "tool_calls": tool_call_log, "files": _resolve_files(client, file_ids)}

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                # web_search / code_execution are Anthropic SERVER tools: the API
                # executes them itself and the result is already inline in this
                # same response (the paired *_tool_result block above) — only our
                # own tools (search_decisions, get_decision, ...) need a
                # client-supplied tool_result fed back in the next round.
                if block.name not in impls:
                    continue
                fn = impls[block.name]
                result = fn(block.input)
                tool_call_log.append({"name": block.name, "input": block.input, "result": result})
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)})
                # generate_image's file_id lives inside its own JSON tool result,
                # not in a *_tool_result content block the way a server tool's
                # output does — _extract_file_ids can't see it, so it's pulled
                # out here instead.
                if block.name == "generate_image" and isinstance(result, dict) and result.get("file_id"):
                    file_ids.append(result["file_id"])
                    turn_cost_usd += IMAGE_COST_USD.get(OPENAI_IMAGE_QUALITY, 0.05)

            if not tool_results:
                # Every tool_use block in this round was a server tool already
                # resolved inline (stop_reason=="tool_use" should not normally
                # happen in that case, but this keeps the loop from breaking if
                # it ever does) — just let Claude continue from here.
                continue
            messages.append({"role": "user", "content": tool_results})

        return {
            "answer": "I wasn't able to settle on an answer within the tool-call budget for this turn — try a narrower question.",
            "tool_calls": tool_call_log,
            "files": _resolve_files(client, file_ids),
        }
    finally:
        # Recorded even on an exception mid-turn (a partial multi-round turn
        # that then fails shouldn't be "free") and even on the tool-budget-
        # exceeded fallback above, not just the success path.
        _record_cost(company_id, turn_cost_usd)


def _status_label_for(block_type: str, tool_name: str) -> str:
    """A short, human status line for whatever tool just started — this is
    what makes the UI show real, specific progress ("Searching the web…")
    instead of a generic spinner while a turn is in flight. block_type
    distinguishes an Anthropic server tool (web_search/web_fetch/
    code_execution — the API executes these itself) from one of our own
    company-memory tools (search_decisions, get_decision, ...)."""
    if block_type == "server_tool_use":
        return {
            "web_search": "Searching the web…",
            "web_fetch": "Reading a page…",
            "code_execution": "Running code…",
            "bash_code_execution": "Running code…",
            "text_editor_code_execution": "Working with a file…",
        }.get(tool_name, "Working…")
    return {
        "search_decisions": "Searching Decision Memory…",
        "get_decision": "Pulling up a decision record…",
        "search_goals": "Searching company goals…",
        "get_goal_chain": "Tracing goal alignment…",
        "get_my_context": "Checking your role and context…",
        "search_documents": "Searching company documents…",
        "get_document": "Reading a document…",
        "generate_image": "Generating an image…",
    }.get(tool_name, "Thinking…")


def run_agent_turn_stream(persona: Persona, decisions: list, goals: list, user_message: str, history=None, company_id: str = ""):
    """Same agentic tool-use loop as run_agent_turn, but a generator that
    yields small dicts as things actually happen, instead of computing the
    whole answer and handing it back in one lump: {"event": "status", ...}
    the instant a tool call starts (so the UI can show real, specific
    progress instead of a frozen spinner), {"event": "text", "text": ...}
    for each text chunk AS Claude generates it (so the UI can type the
    answer out, the way Claude's own apps do, instead of the text appearing
    all at once when the whole response finally lands), and one final
    {"event": "done", ...} carrying the same tool_calls/files metadata
    run_agent_turn returns. The FastAPI route below turns these into
    Server-Sent Events.

    Added 2026-08-29 directly in response to the founder wanting Ask Cognex
    to look and feel like using Claude itself, not just have the same
    underlying capability — the non-streaming run_agent_turn() above is
    left in place rather than replaced, since it's simpler for anything
    that doesn't need incremental rendering (and is what test_offline.py
    and any future non-streaming caller can keep using unchanged)."""
    # Rate/cap CHECKS live in the route handler (ask_stream), before this
    # generator is ever constructed — see run_agent_turn's comment on why;
    # the same reasoning applies here, plus a streaming response can't raise
    # a normal HTTPException mid-stream anyway (the 200 + SSE headers are
    # already committed). This function only RECORDS the cost incurred.
    client = Anthropic()
    impls = _make_tool_impls(persona, decisions, goals, company_id)
    messages = list(history or [])
    messages.append({"role": "user", "content": user_message})
    tool_call_log = []
    file_ids = []
    all_tools = TOOL_SCHEMAS + SERVER_TOOLS
    turn_cost_usd = 0.0

    try:
        for _ in range(MAX_TOOL_ROUNDS):
            with client.messages.stream(
                model=MODEL, max_tokens=8192, system=_system_prompt(persona, decisions),
                tools=all_tools, tool_choice={"type": "auto"}, messages=messages,
            ) as stream:
                for event in stream:
                    if event.type == "content_block_start":
                        block = event.content_block
                        block_type = getattr(block, "type", None)
                        if block_type in ("tool_use", "server_tool_use"):
                            yield {"event": "status", "label": _status_label_for(block_type, block.name)}
                    elif event.type == "text":
                        yield {"event": "text", "text": event.text}
                response = stream.get_final_message()

            turn_cost_usd += _usage_cost_usd(getattr(response, "usage", None))
            file_ids.extend(_extract_file_ids(response.content))

            if response.stop_reason != "tool_use":
                yield {"event": "done", "tool_calls": tool_call_log, "files": _resolve_files(client, file_ids)}
                return

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                if block.name not in impls:
                    continue
                fn = impls[block.name]
                result = fn(block.input)
                tool_call_log.append({"name": block.name, "input": block.input, "result": result})
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)})
                if block.name == "generate_image" and isinstance(result, dict) and result.get("file_id"):
                    file_ids.append(result["file_id"])
                    turn_cost_usd += IMAGE_COST_USD.get(OPENAI_IMAGE_QUALITY, 0.05)

            if not tool_results:
                continue
            messages.append({"role": "user", "content": tool_results})

        yield {
            "event": "done",
            "answer_override": "I wasn't able to settle on an answer within the tool-call budget for this turn — try a narrower question.",
            "tool_calls": tool_call_log,
            "files": _resolve_files(client, file_ids),
        }
    finally:
        # Same reasoning as run_agent_turn's finally: recorded on normal
        # completion, on the tool-budget-exceeded fallback above, AND if an
        # exception propagates out of this generator mid-stream — a partial
        # turn that then fails still spent real tokens.
        _record_cost(company_id, turn_cost_usd)


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------
def _log_and_raise_502(label: str, e: Exception):
    """Print the real upstream exception server-side (visible via `railway
    logs` / the Railway dashboard) before turning it into a client-facing
    502. Previously the except blocks below only put the error in the HTTP
    response body — invisible in server logs, which made a genuine upstream
    failure (an unsupported/misconfigured tool, an invalid or unauthorized
    API key, a real Anthropic-side error) indistinguishable from "we can't
    tell what broke" from the outside. Added 2026-08-29 after exactly that
    happened investigating why Ask Cognex was silently falling back to its
    offline demo mode in production — see that build-log entry."""
    import traceback
    print(f"[{label}] Model call failed: {e!r}", flush=True)
    traceback.print_exc()
    raise HTTPException(status_code=502, detail=f"Model call failed: {e}")


router = APIRouter(prefix="/api")


class PersonaIn(BaseModel):
    id: str
    name: str
    title: str
    level: int
    dept: str = ""


class DecisionIn(BaseModel):
    id: str
    title: str
    decided: str = ""
    owner: str = ""
    visibility: int = 1
    derivedLevel: int = 1
    why: list = []
    alternatives: list = []
    assumptions: list = []
    risks: list = []
    review: str = ""
    result: str = ""
    tags: list = []
    derived: str = ""
    contributor: Optional[str] = None
    domain: str = "other"


class GoalIn(BaseModel):
    id: str
    title: str
    kicker: str = ""
    depth: int = 0
    parent: Optional[str] = None
    dept: Optional[str] = None
    owner: Optional[str] = None


class AskRequest(BaseModel):
    persona: PersonaIn
    decisions: list[DecisionIn] = []
    goals: list[GoalIn] = []
    message: str
    history: list = []
    # Added 2026-08-31 to back the per-company rate limit + cost cap (see
    # _check_rate_limit/_check_cost_cap above) — nothing in this request
    # previously identified WHICH company a turn belonged to, so there was
    # no key to cap or rate-limit by. Defaults to "" (rather than being
    # required) so an old cached frontend that hasn't picked up this field
    # yet degrades to "not rate-limited/capped" instead of a hard 422 —
    # deliberately fails open here, the same trade-off _check_rate_limit and
    # _check_cost_cap both already make for an empty company_id.
    company_id: str = ""


def _resolve_ask_identity(req: "AskRequest", request: Request, token: Optional[str]):
    """Resolves the actual company_id and persona an /ask (or /ask/stream)
    call is allowed to use, from the AUTHENTICATED session — not from
    whatever the client's request body claims. Added 2026-09-03 after the
    full-app-audit-2026-09-02 critical/high findings: previously nothing
    stopped a request from claiming any persona.level (including 6, CEO
    only) and any company_id, with zero credential of any kind — which both
    bypassed clearance filtering entirely (permissions.py's level check is
    only as trustworthy as the level it's given) and let an attacker point
    the per-company rate limit/cost cap at a victim company just by naming
    it.

    A company session resolves BOTH company_id and persona from the session
    itself, ignoring whatever the client sent for either — this is the
    normal login path, and the one this fix actually closes.

    A platform (staff) session — "Enter as support" — has no company_id or
    persona identity of its own; staff is deliberately impersonating
    whichever admin persona the frontend already picked when previewing a
    company's account (see renderPlatform's "Enter as support" in
    static/index.html), so that one case still trusts the client's
    company_id/persona, matching the existing support-preview flow exactly
    as it worked before this pass.

    Decisions/goals are still always taken from the request body either
    way — this module's whole design (see the file's own docstring) is
    that the client supplies its current company snapshot per request;
    nothing server-side stores one centrally for /ask to look up instead."""
    session = require_session(token_from_request(request, token))
    if session["kind"] == "platform":
        return req.company_id, req.persona
    company_id = session.get("company_id") or req.company_id
    with get_session() as db:
        row = db.get(DBPersona, (company_id, session.get("persona_id")))
    if not row:
        raise HTTPException(status_code=401, detail="Your session's account could not be found — try signing in again.")
    return company_id, PersonaIn(id=row.id, name=row.name, title=row.title, level=row.level, dept=row.dept or "")


@router.get("/health")
def health():
    return {"ok": True, "api_key_configured": bool(os.environ.get("ANTHROPIC_API_KEY"))}


@router.get("/files/{file_id}")
def get_generated_file(file_id: str, request: Request, token: Optional[str] = None):
    """Streams back a file code_execution generated (a chart image, a CSV,
    etc.) by proxying Anthropic's Files API through our own server API key —
    the frontend never talks to Anthropic directly.

    Auth, tightened 2026-09-03 (full-app-audit-2026-09-02 critical finding:
    this route had NO company scoping at all — any caller who had or
    guessed a "doc-" id could download a different company's uploaded
    documents straight out of the library, with zero credential). Every
    branch now requires SOME valid, signed-in session before serving
    anything at all. The "doc-" branch (a real company document-library
    upload — the one that can hold genuinely confidential files) goes
    further and requires that session be scoped to the SAME company the
    document belongs to, exactly like every other company-data route.

    The "img-" and raw-Anthropic-file-id branches below still aren't scoped
    to a specific company — neither the in-process image store nor
    Anthropic's Files API records which company generated a given id, so
    there's nothing to check it against yet. Requiring a valid session at
    all (rather than none) closes the fully-anonymous version of this gap;
    threading company_id through image/code_execution file storage too is a
    further hardening step, not done in this pass.

    An "img-" prefixed id is an OpenAI-generated image served straight out of
    the in-process _GENERATED_IMAGES store (see the image-generation block
    near the top of this file) rather than Anthropic's Files API — those ids
    never existed on Anthropic's side at all."""
    require_any_session(token_from_request(request, token))
    if file_id.startswith("img-"):
        rec = _GENERATED_IMAGES.get(file_id)
        if not rec:
            raise HTTPException(status_code=404, detail=f"Generated image '{file_id}' was not found — it may have expired (only the most recent {_GENERATED_IMAGES_MAX} images are kept) or the server may have restarted since it was generated.")
        return Response(
            content=rec["bytes"],
            media_type=rec["mime_type"] or "application/octet-stream",
            headers={"Content-Disposition": f'inline; filename="{rec["filename"] or file_id}"'},
        )
    if file_id.startswith("doc-"):
        # A company document library upload (added 2026-08-31, see
        # persistence.py's upload_document — Document ids are minted with
        # this exact "doc-" prefix). Unlike img- above, this one real DB
        # read is a deliberate, narrow departure from this module's
        # otherwise-stateless "everything comes from the request body"
        # design — see the search_documents/get_document tool comment below
        # for the same reasoning.
        with get_session() as db:
            d = db.get(Document, file_id)
        if not d:
            raise HTTPException(status_code=404, detail=f"Document '{file_id}' was not found — it may have been deleted.")
        require_session(token_from_request(request, token), company_id=d.company_id)
        return Response(
            content=d.content_bytes,
            media_type=d.mime_type or "application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{d.filename or file_id}"'},
        )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set on the server.")
    client = Anthropic()
    try:
        meta = client.files.retrieve_metadata(file_id)
        binary = client.files.download(file_id)
        content = binary.read()
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not fetch file '{file_id}': {e}")
    return Response(
        content=content,
        media_type=meta.mime_type or "application/octet-stream",
        headers={"Content-Disposition": f'inline; filename="{meta.filename or file_id}"'},
    )


@router.post("/ask")
def ask(req: AskRequest, request: Request, token: Optional[str] = None):
    # Auth checked FIRST, before even the API-key/503 check below -- an
    # unauthenticated caller shouldn't learn anything about the server's
    # configuration state, and this is the one place a real 401 can reach
    # the client for this route (see the try/except's own reasoning below).
    req.company_id, req.persona = _resolve_ask_identity(req, request, token)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set on the server.")
    # Deliberately BEFORE the try/except below: an HTTPException raised here
    # (429 rate-limited, or over the monthly cost cap) must reach the client
    # as-is. Inside the try/except it would get caught by `except Exception`
    # and rewritten into a misleading 502 "Model call failed".
    _check_rate_limit(req.company_id)
    _check_cost_cap(req.company_id)
    persona = Persona(**req.persona.model_dump())
    decisions = [Decision(**d.model_dump()) for d in req.decisions]
    goals = [Goal(**g.model_dump()) for g in req.goals]
    try:
        return run_agent_turn(persona, decisions, goals, req.message, req.history, company_id=req.company_id)
    except Exception as e:
        _log_and_raise_502("ask", e)


@router.post("/ask/stream")
def ask_stream(req: AskRequest, request: Request, token: Optional[str] = None):
    """Same request/response contract as POST /ask, but streamed as
    Server-Sent Events instead of one blocking JSON response — added
    2026-08-29 so Ask Cognex can type its answer out live and show real,
    specific progress ("Searching the web…") while a tool is running,
    instead of a frozen composer and then the whole answer appearing at
    once. This is what the frontend actually calls now; POST /ask is kept
    unchanged as the simpler non-streaming path (used by test_offline.py's
    reference-backend tests, and available to any future caller that just
    wants a plain JSON response).

    A plain HTTPException can't be raised partway through an SSE response
    (the 200 and the text/event-stream headers are already committed by the
    time an error could happen), so failures are instead sent as a final
    `{"event": "error", ...}` frame — see submitQuestionStreaming's error
    handling in the frontend, which renders this exactly like the existing
    offline-fallback path a non-streaming failure already produces."""
    # Auth checked FIRST, same reasoning as POST /ask -- synchronously,
    # BEFORE the API-key check and BEFORE the StreamingResponse is ever
    # constructed. This is the one place a real 401/429 can still reach the
    # client for this route, since once the SSE response starts, only
    # {"event": "error"} frames are possible (see the docstring above).
    req.company_id, req.persona = _resolve_ask_identity(req, request, token)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set on the server.")
    _check_rate_limit(req.company_id)
    _check_cost_cap(req.company_id)
    persona = Persona(**req.persona.model_dump())
    decisions = [Decision(**d.model_dump()) for d in req.decisions]
    goals = [Goal(**g.model_dump()) for g in req.goals]

    def sse():
        try:
            for event in run_agent_turn_stream(persona, decisions, goals, req.message, req.history, company_id=req.company_id):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            import traceback
            print(f"[ask/stream] Model call failed: {e!r}", flush=True)
            traceback.print_exc()
            yield f"data: {json.dumps({'event': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------------------
# Complete the Story — question generation. Added 2026-08-29: this step had
# NEVER actually called Claude — every persona always got the same
# hardcoded client-side template ("{title} — current priorities", three
# generic questions), regardless of whether they had any real individual
# goal or contributed decision to reference. That's why the CEO — who has no
# individual goal in the seed data — was asked the identical generic
# questions as everyone else, and why the questions never named anything
# specific. This mirrors the offboarding question-generation pattern
# (generate_handoff_questions above), grounded in the SAME kind of context
# (an individual goal, recent contributed decisions), with one addition: the
# model is explicitly told to say there's nothing to ask about rather than
# invent a plausible-sounding generic question when there's no real signal —
# directly per the founder's own framing: "if there's nothing to complete
# there shouldn't be any questions, otherwise the questions need to be
# specific."
#
# Extended 2026-09-02, direct founder feedback: "Complete the story doesnt
# have anything right now but I have had quite a few convos with the model
# and it should definitely have identified some parts of my convos that
# require more clarification." The gap: this only ever looked at formal
# signal — an assigned individual goal, or a Decision Memory entry with this
# person listed as `contributor` — never at the actual content of their Ask
# Cognex conversations (see ChatThread in models.py, added the same week).
# Someone who mostly just talks to Cognex rather than filing goals/decisions
# under their own name legitimately had zero signal before, no matter how
# many real conversations they'd had. `recent_threads` below is a light
# summary of that persona's own recent threads (title + their own messages,
# not Cognex's answers — the ambiguity worth asking about lives in what THEY
# said, not what the model already answered), and the model is now
# explicitly told to look there for anything left vague or unresolved, not
# just at goals/decisions.
GENERATE_STORY_QUESTIONS_SCHEMA = {
    "name": "submit_story_questions",
    "description": "Submit whether there is real work to ask this person about, and if so, specific questions grounded in it.",
    "input_schema": {
        "type": "object",
        "properties": {
            "has_work_to_capture": {
                "type": "boolean",
                "description": (
                    "True if the goal/decision context OR the recent conversation excerpts given "
                    "actually give something concrete and current to ask about — including a specific "
                    "point left vague or unresolved in something they said in a recent conversation, "
                    "even if there's no formal tracked goal or decision behind it. False only if there "
                    "is genuinely nothing specific anywhere — no individual goal, no recent contributed "
                    "decisions, and nothing in the conversation excerpts worth following up on. Do not "
                    "invent a plausible-sounding 'current priorities' question when there's nothing "
                    "real to reference; many senior roles (CEO, CFO) legitimately have no single "
                    "tracked individual goal, and that's a normal, expected false here when there's "
                    "also no conversation signal, not a gap to paper over."
                ),
            },
            "work_item_label": {
                "type": "string",
                "description": (
                    "A short, specific label for what's being captured — the actual goal or decision "
                    "title if that's the source, or a short name for the specific conversation topic "
                    "being followed up on if it came from the conversation excerpts instead (e.g. the "
                    "thread's own title). Never a role-based placeholder like '<title> — current "
                    "priorities'. Empty string if has_work_to_capture is false."
                ),
            },
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string", "enum": ["who", "why", "risk"]},
                        "prompt": {
                            "type": "string",
                            "description": "A specific question naming the actual goal, decision, or conversation topic by name and, where it applies, pointing at the specific thing that was left unclear — never a generic templated question like 'why does this matter for the team right now'.",
                        },
                    },
                    "required": ["field", "prompt"],
                },
                "description": "2-3 specific questions if has_work_to_capture is true; empty array otherwise.",
            },
        },
        "required": ["has_work_to_capture", "work_item_label", "questions"],
    },
}


def generate_story_questions(
    persona: dict, individual_goal: Optional[dict], recent_decisions: list, recent_threads: Optional[list] = None
) -> dict:
    client = Anthropic()
    recent_threads = recent_threads or []
    context = f"{persona.get('name', '')}, {persona.get('title', '')}, {persona.get('dept', '')} department.\n"
    if individual_goal:
        context += f"Their individual goal: {individual_goal.get('title', '')}"
        if individual_goal.get("kicker"):
            context += f" ({individual_goal['kicker']})"
        context += "\n"
    if recent_decisions:
        context += "Decisions they're a contributor on: " + "; ".join(d.get("title", "") for d in recent_decisions) + "\n"
    if recent_threads:
        context += "\nExcerpts from their own recent Ask Cognex conversations (their own messages only — look here for anything left vague, unresolved, or worth a follow-up):\n"
        for t in recent_threads:
            snippet = " / ".join(m[:280] for m in (t.get("messages") or []) if m)
            if snippet:
                context += f'- Thread "{t.get("title", "")}": {snippet}\n'
    if not individual_goal and not recent_decisions and not recent_threads:
        context += "No individual goal is assigned to them, no decisions list them as a contributor, and they have no recent conversation history either.\n"

    response = client.messages.create(
        model=MODEL,
        max_tokens=768,
        system=(
            "You decide whether there is real, specific work to ask this person about, based on the "
            "goal/decision context AND the recent conversation excerpts given. Prefer grounding "
            "questions in a genuine individual goal or contributed decision when one exists. When "
            "there isn't one but recent conversation excerpts are given, read them for anything the "
            "person raised without fully resolving — a plan mentioned but not detailed, a decision "
            "implied but not confirmed, a risk or open question they didn't answer themselves — and "
            "ask about THAT specifically, naming the actual topic. If neither the goal/decision "
            "context nor the conversation excerpts give anything concrete, set has_work_to_capture to "
            "false rather than asking generic 'what are your current priorities'-style questions — "
            "that produces meaningless answers and wastes the person's time."
        ),
        tools=[GENERATE_STORY_QUESTIONS_SCHEMA],
        tool_choice={"type": "tool", "name": "submit_story_questions"},
        messages=[{"role": "user", "content": context}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_story_questions":
            return block.input
    raise RuntimeError("Claude did not return the expected submit_story_questions tool call.")


class RecentThreadIn(BaseModel):
    title: str = ""
    messages: list[str] = []


class StoryQuestionsRequest(BaseModel):
    persona: PersonaIn
    individual_goal: Optional[GoalIn] = None
    recent_decisions: list[DecisionIn] = []
    recent_threads: list[RecentThreadIn] = []


@router.post("/complete-story/questions")
def complete_story_questions(req: StoryQuestionsRequest, request: Request, token: Optional[str] = None):
    # Added 2026-09-03 (full-app-audit-2026-09-02 finding): this endpoint has
    # no company_id to scope against -- everything it needs comes straight
    # off the request body (see this module's docstring on that design) --
    # but it still shouldn't be reachable with zero credential at all. Same
    # reasoning on every other route below with no company_id field.
    require_any_session(token_from_request(request, token))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set on the server.")
    try:
        return generate_story_questions(
            req.persona.model_dump(),
            req.individual_goal.model_dump() if req.individual_goal else None,
            [d.model_dump() for d in req.recent_decisions],
            [t.model_dump() for t in req.recent_threads],
        )
    except Exception as e:
        _log_and_raise_502("complete-story/questions", e)


# ---------------------------------------------------------------------------
# Memory consolidation — the Brain Board. Originally added 2026-08-30 so
# Decision Memory would read as a small, evolving set of topic nodes instead
# of one record per save; extended 2026-08-31 into a real department-hub
# tree per direct founder feedback ("I want core business departments...
# stranded into smaller parts over time as the brain of the company grows...
# consolidate or expand smartly"). Every save path (Ask Cognex's "Save to
# Decision Memory", Complete the Story, the Slack candidate-confirm flow,
# Vantage's "mark plan done") now calls consolidate_memory FIRST: given the
# board's existing nodes (lightweight: id/title/tags/domain/parent/child
# count/current summary — full why/alternatives/risks content is
# deliberately NOT sent, to keep this call
# fast and cheap) and the new content just captured, Claude assigns a broad
# knowledge domain and picks one of four actions:
#   - "new_trunk"  — a genuinely new top-level topic under a domain that has
#                    no good existing home for it yet.
#   - "new_branch" — a specific, distinct sub-topic under an existing trunk
#                    (or branch) — the domain's tree gains a new limb, the
#                    "expand" the founder asked for.
#   - "merge"      — really the same topic as an existing node; fold in and
#                    rewrite its summary (the "consolidate" the founder
#                    asked for, same behavior this had before 2026-08-31).
#   - "split"      — merging into the target would blur together genuinely
#                    distinct sub-topics that each deserve their own node.
#                    A second, more expensive call (plan_memory_split)
#                    only fires on this rare path, given that ONE node's
#                    full history, to actually partition it — see below for
#                    why this is a separate call rather than one bigger one.
# This is still a real judgment call, not a keyword-overlap heuristic — "we
# sell design services to startups" and "we're a boutique design and
# engineering studio" are the same topic despite sharing almost no words
# (see the 2026-08-29 memory-grounding entry, the same class of gap on the
# search side of this feature). Deliberately scoped to depth-2 splitting
# only (v1): a node that already has children of its own can't be split
# again in this pass (downgraded to "merge" instead, see the guard block
# below) — recursively re-splitting an already-split node means deciding
# which grandchild subtree absorbs which surviving fact, real complexity
# not asked for here.
#
# Departments used to be a fixed six-value enum baked into this module
# (product/gtm/ops/pricing/hiring/other) — direct founder feedback
# (2026-09-01) was that this is backwards: "there are set departments but
# they haven't been discussed, the company brain should smartly grow using
# AI and gain new departments as required." So `domain` is now just a
# free-text string Claude assigns, same as `title`/`summary` — nothing in
# this file constrains it to a fixed list. To keep the board legible
# instead of fragmenting into near-duplicate departments ("Marketing" next
# to "Go-to-market" next to "GTM"), every call is shown the company's
# CURRENT distinct departments (extracted from the candidate list, see
# `_normalize_domain` below) and is instructed to reuse one whenever it
# reasonably fits, copying its exact spelling — a new department name is
# meant to be rare, not the default. DEFAULT_DOMAIN is only ever a
# fallback when a real classification genuinely isn't available (the
# offline/no-API-key path in the frontend, and a defensive default here).
DEFAULT_DOMAIN = "Other"


def _normalize_domain(raw: str, existing_domains: list) -> str:
    """Claude picks the department name freely now (see the module comment
    above), which makes accidental fragmentation a real risk: "Product" vs
    "product" vs "Products" would otherwise render as three separate ring
    sectors for what a person means as one department. Reuse the EXISTING
    department's exact spelling whenever the new one matches
    case-insensitively (a cheap, deterministic backstop underneath the
    prompt's own "copy the exact spelling" instruction, not a replacement
    for it), and only fall back to title-casing a fresh all-lowercase name
    (Claude's own mixed-case names and short acronyms like "R&D" or "GTM"
    are left exactly as written rather than mangled)."""
    text = " ".join((raw or "").split()).strip()[:40] or DEFAULT_DOMAIN
    for existing in existing_domains:
        if existing.lower() == text.lower():
            return existing
    return text.title() if text.islower() else text


CONSOLIDATE_MEMORY_SCHEMA = {
    "name": "submit_memory_consolidation",
    "description": (
        "Assign a broad department/knowledge area to newly captured information and decide how it fits "
        "onto the company's existing Decision Memory tree: fold into an existing node, hang a new sharper "
        "branch off one, start a new top-level topic, or split an existing node apart."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "domain": {
                "type": "string",
                "description": (
                    "The broad department/knowledge area this belongs to — think of the handful of areas "
                    "a new hire would need briefing on (Product, Go-to-market, Operations, Pricing, Hiring "
                    "& Team, Finance, Legal, Partnerships, ...). REUSE one of the company's EXISTING "
                    "departments listed in the context below whenever it reasonably fits, copying its "
                    "exact spelling and casing — do not invent 'Marketing' when 'Go-to-market' already "
                    "exists, or 'product' when 'Product' already exists. Only introduce a genuinely new "
                    "department name when none of the existing ones are a sensible home for this topic; "
                    "keep any new name short (1-3 words) and broad, like a real org-chart department, not "
                    "a narrow topic label. For 'merge' or 'split', this SHOULD match the target node's own "
                    "domain — it is not used to move an already-placed node to a different domain."
                ),
            },
            "action": {
                "type": "string",
                "enum": ["new_trunk", "new_branch", "merge", "split"],
                "description": (
                    "'merge' — really the same topic as an existing node (prefer this whenever there's "
                    "real overlap, INCLUDING two saves that are worded differently but are about the same "
                    "underlying subject — e.g. 'Instagram influencer marketing plan' and 'Instagram "
                    "influencer outreach strategy' are the same topic and must merge, not sit as two "
                    "similar-sounding nodes). 'new_branch' — a specific, genuinely distinct sub-topic that "
                    "belongs UNDER an existing trunk/branch (e.g. a specific outreach tactic under an "
                    "existing broad 'customer acquisition' node) — use this instead of 'merge' when "
                    "folding it in would blur together things someone would want to see separately, and "
                    "instead of 'new_trunk' when a sensible parent already exists. 'new_trunk' — a "
                    "genuinely new top-level topic with no existing node in its domain it belongs under. "
                    "'split' — an EXISTING node (named as target_id) has accumulated enough genuinely "
                    "distinct sub-topics that it should be broken apart; only choose this for a node whose "
                    "child_count is 0 in the candidate list (it has no sub-topics yet)."
                ),
            },
            "target_id": {
                "type": "string",
                "description": (
                    "The exact id, copied from the candidate list: the node to merge into (merge), the "
                    "parent trunk/branch to attach under (new_branch), or the node to split apart "
                    "(split). Empty string for new_trunk."
                ),
            },
            "title": {
                "type": "string",
                "description": (
                    "A short, general topic title describing the durable SUBJECT this is about — e.g. "
                    "'Instagram influencer marketing strategy' — never a literal copy of the source "
                    "message's own subject line, chat title, or the exact question someone asked. Two "
                    "saves about the same underlying subject, worded differently, must end up under the "
                    "SAME title, not two near-duplicate ones. Ignored when action is 'split' (see "
                    "plan_memory_split, which titles the resulting nodes instead)."
                ),
            },
            "summary": {
                "type": "string",
                "description": (
                    "The DERIVED summary for this topic — NOT a private record. This is what a viewer "
                    "BELOW the source content's own clearance level sees instead of the raw why/"
                    "alternatives/risks (see decisionView's source-vs-derived split in the frontend); by "
                    "default a new save is visible company-wide regardless of how sensitive the "
                    "underlying content is, so treat every summary you write as something the most "
                    "junior person at the company will read. Describe the ORGANIZATIONAL CONSEQUENCE — "
                    "what this means or what changes for other people's work — one coherent paragraph, "
                    "written fresh to incorporate the new information, not the old summary with the new "
                    "fact bolted on. Do NOT restate confidential financial figures or specific dollar "
                    "amounts, named third parties (people, companies, candidates under consideration), "
                    "or the detailed private reasoning behind a sensitive call — that detail stays in the "
                    "source record this summary is standing in for, not repeated here. When genuinely "
                    "unsure whether a detail is safe to include, leave it out; a vaguer summary is always "
                    "safer than a leaked one. Ignored when action is 'split'."
                ),
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3-8 short lowercase topic tags. Ignored when action is 'split'.",
            },
        },
        "required": ["domain", "action", "target_id", "title", "summary", "tags"],
    },
}


def consolidate_memory(candidates: list, new_content: dict) -> dict:
    client = Anthropic()
    # Existing departments are read directly off the candidates rather than
    # from any fixed list (see the module comment above) — deduped
    # case-insensitively but keeping the first-seen spelling, since that's
    # the exact string every board-open call already needs to reuse
    # verbatim to avoid fragmenting "Product" from "product".
    domain_counts: dict = {}
    existing_domains: list = []
    for c in candidates:
        d = (c.get("domain") or DEFAULT_DOMAIN).strip() or DEFAULT_DOMAIN
        domain_counts[d] = domain_counts.get(d, 0) + 1
        if d.lower() not in {x.lower() for x in existing_domains}:
            existing_domains.append(d)
    if candidates:
        cand_lines = [
            f'- id="{c.get("id","")}" | title="{c.get("title","")}" | domain={c.get("domain") or DEFAULT_DOMAIN} | '
            f'parent={c.get("parent_id") or "(none — this is a trunk)"} | child_count={c.get("child_count", 0)} | '
            f'tags={c.get("tags", [])} | current summary: {c.get("derived","") or "(none yet)"}'
            for c in candidates
        ]
        candidates_block = "\n".join(cand_lines)
        domains_block = "\n".join(
            f'- "{d}" ({domain_counts[d]} topic{"s" if domain_counts[d] != 1 else ""})' for d in existing_domains
        )
    else:
        candidates_block = "(the board is empty — this will be the first node)"
        domains_block = "(none yet — this will be the company brain's first department)"

    context = (
        f"The board currently has {len(candidates)} node(s) across these existing departments:\n{domains_block}\n\n"
        f"Existing Decision Memory nodes (the company's current brain):\n{candidates_block}\n\n"
        "New information just captured:\n"
        f"Source: {new_content.get('source_kind', '')} (by {new_content.get('source_persona', '')} on {new_content.get('date', '')})\n"
        f"Title/prompt: {new_content.get('title', '')}\n"
        f"Content: {new_content.get('body', '')}\n"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=768,
        system=(
            "You maintain a company's collective knowledge board as a small TREE of broad department "
            "areas — the departments themselves are NOT a fixed list; they grow organically as the "
            "company's real knowledge grows, the same way a real org chart gains a new department only "
            "when the work genuinely warrants one, not one per topic. Each department can grow sharper "
            "sub-topic branches over time as real knowledge in that area deepens — NOT a flat pile of "
            "narrow one-off notes, and not one node per question someone happened to ask, and not a new "
            "department per node either.\n\n"
            "Reuse an existing department (see the domain field's own instructions) far more often than "
            "you invent one — a healthy board has a handful of departments, not a dozen near-duplicates. "
            "Default to 'merge' whenever the new information is really about the same topic as an "
            "existing node — a specific instance or detail WITHIN a broader area an existing node "
            "already covers, folded into its summary rather than kept separate just because the exact "
            "wording or the exact question wasn't identical to what was covered before; a topic's TITLE "
            "should read as a durable subject, not a transcript of whatever was asked (see the title "
            "field's own instructions). Reach for 'new_branch' when the new information is a genuinely "
            "distinct sub-topic that deserves its own node but clearly belongs under an existing trunk — "
            "this is how a department's knowledge grows sharper branches over time instead of one node "
            "trying to cover everything on its own. Only use 'new_trunk' when nothing in that department "
            "is a sensible parent at all. Use 'split' only when a listed node's own child_count is 0 and "
            "merging the new information into it would blur together things that genuinely deserve "
            "separate nodes — this should be rare; most saves should merge or branch, not split. The "
            "board should stay small and legible: the more nodes already exist in a department, the "
            "harder you should lean toward merge/branch over creating something new.\n\n"
            "CRITICAL — the `summary` field is a REDACTION step, not a recap: it becomes the ONLY "
            "version of this topic that a lower-clearance viewer ever sees (this board defaults new "
            "saves to company-wide visibility regardless of how sensitive the source content is), while "
            "the confidential detail behind it — financial figures, named people or companies, the "
            "private reasoning — stays only in the source record, which this summary must NOT restate. "
            "Write what changed or what it means for people's work, never the confidential substance "
            "behind it. Treat this as true for every save, not just ones that look obviously sensitive — "
            "the safest default is less detail, not more."
        ),
        tools=[CONSOLIDATE_MEMORY_SCHEMA],
        tool_choice={"type": "tool", "name": "submit_memory_consolidation"},
        messages=[{"role": "user", "content": context}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_memory_consolidation":
            result = dict(block.input)
            candidate_ids = {c.get("id") for c in candidates}
            candidate_by_id = {c.get("id"): c for c in candidates}
            action = result.get("action")
            # Defense-in-depth against a hallucinated id: never let a
            # "merge"/"new_branch"/"split" target an id that isn't actually
            # one of the candidates given silently corrupt a different node
            # or 404 downstream — downgrade to "new_trunk" instead, which
            # is always safe (worst case one extra node, never data loss).
            if action in ("merge", "new_branch", "split") and result.get("target_id") not in candidate_ids:
                action = "new_trunk"
                result["target_id"] = ""
            # A "split" target that already has children is out of scope
            # for v1 (see module comment above) — downgrade to "merge"
            # instead of silently no-op'ing or erroring, so the save still
            # lands somewhere sensible even when the model picks split on
            # a node it shouldn't have.
            elif action == "split" and candidate_by_id.get(result.get("target_id"), {}).get("child_count", 0) > 0:
                action = "merge"
            result["action"] = action
            # Departments are free text now (no more fixed enum) -- collapse
            # accidental case/whitespace variants of an existing department
            # onto its one true spelling before this ever reaches the board.
            result["domain"] = _normalize_domain(result.get("domain", ""), existing_domains)
            return result
    raise RuntimeError("Claude did not return the expected submit_memory_consolidation tool call.")


# ---------------------------------------------------------------------------
# "Tidy the board with AI" — added 2026-09-02, direct founder feedback on the
# live board: "the node names are all my exact prompt, not the topic/sub-
# topic... There is one category right now and it is already named 'other'."
# consolidate_memory (above) already writes clean titles/departments for
# every NEW save — but that's forward-only, and doesn't touch a topic that
# was captured back when this ran offline (no ANTHROPIC_API_KEY configured,
# so the frontend's own createAsDrafted fallback stored the raw prompt text
# verbatim as the title, see static/index.html) or before this classifier
# existed at all. This is the retroactive counterpart: given the board's
# current TRUNK topics (branches/splits already got a real title from
# whatever consolidation created them, so this is scoped to trunks only —
# see the /board/tidy caller in static/index.html), rewrite each one's title
# to a short, clean topic name and reclassify its department, reusing the
# _normalize_domain backstop so this can't fragment "Product" into
# "Product"/"product" the same way consolidate_memory already guards
# against that on the forward path.
# ---------------------------------------------------------------------------
TIDY_BOARD_SCHEMA = {
    "name": "submit_tidy_board",
    "description": "Submit a cleaned-up title and department for each topic given.",
    "input_schema": {
        "type": "object",
        "properties": {
            "topics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Must exactly match one of the topic ids given."},
                        "title": {
                            "type": "string",
                            "description": (
                                "A short, clean, general topic title (a handful of words, like a section "
                                "heading) describing the durable SUBJECT this topic is about — never a "
                                "literal copy of the source text, a chat message, or the exact question "
                                "someone asked. If the current title already reads as a clean topic name, "
                                "it's fine to return it unchanged."
                            ),
                        },
                        "domain": {
                            "type": "string",
                            "description": (
                                "The department this topic belongs under. Reuse one of the existing "
                                "departments listed in the prompt, copying its exact spelling, whenever it "
                                "reasonably fits. Only invent a new short (1-3 word) department name when "
                                "none of the existing ones fit — and prefer splitting a vague catch-all "
                                "department like 'Other' into real departments over leaving things in it."
                            ),
                        },
                    },
                    "required": ["id", "title", "domain"],
                },
            }
        },
        "required": ["topics"],
    },
}


def tidy_board_topics(topics: list) -> list:
    client = Anthropic()
    existing_domains: list = []
    for t in topics:
        d = (t.get("domain") or DEFAULT_DOMAIN).strip() or DEFAULT_DOMAIN
        if d.lower() not in {x.lower() for x in existing_domains}:
            existing_domains.append(d)
    domains_block = ", ".join(f'"{d}"' for d in existing_domains) or "(none yet)"
    topics_text = "\n\n".join(
        f'- id="{t["id"]}" | current title="{t.get("title","")}" | current department={t.get("domain") or DEFAULT_DOMAIN}\n'
        f'  detail: {(t.get("why") or [""])[0][:300]}'
        for t in topics
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=(
            "You are cleaning up a company's knowledge board. Each topic below currently has a messy "
            "title — very often a literal verbatim copy of whatever chat message or question created "
            "it, not a real topic name. For EACH topic given, write a short, clean, general topic title "
            "(a handful of words, like a section heading — never a restatement of a question or a "
            "literal prompt) and assign it to a department, reusing an exact existing spelling whenever "
            "it genuinely fits. Departments already in use on this board: "
            f"{domains_block}. Two topics that are really about the same underlying subject should get "
            "the same or a very similar title. You must return exactly one result per topic id given, "
            "in any order."
        ),
        tools=[TIDY_BOARD_SCHEMA],
        tool_choice={"type": "tool", "name": "submit_tidy_board"},
        messages=[{"role": "user", "content": f"Topics:\n\n{topics_text}"}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_tidy_board":
            results = block.input.get("topics", [])
            for r in results:
                normalized = _normalize_domain(r.get("domain", ""), existing_domains)
                r["domain"] = normalized
                if normalized.lower() not in {x.lower() for x in existing_domains}:
                    existing_domains.append(normalized)
            return results
    raise RuntimeError("Claude did not return the expected submit_tidy_board tool call.")


class TidyTopicIn(BaseModel):
    id: str
    title: str
    domain: str = ""
    why: list[str] = []


class TidyBoardRequest(BaseModel):
    topics: list[TidyTopicIn] = []


@router.post("/board/tidy")
def board_tidy(req: TidyBoardRequest, request: Request, token: Optional[str] = None):
    require_any_session(token_from_request(request, token))
    if not req.topics:
        return {"topics": []}
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set on the server.")
    try:
        return {"topics": tidy_board_topics([t.model_dump() for t in req.topics])}
    except Exception as e:
        _log_and_raise_502("board/tidy", e)


SPLIT_MEMORY_SCHEMA = {
    "name": "submit_memory_split",
    "description": (
        "Partition one overloaded Decision Memory node's full history into 2-4 sharper, genuinely "
        "distinct sub-nodes, plus a short umbrella description for the original node."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "trunk_title": {"type": "string", "description": "The original node's title going forward — usually unchanged or only slightly broadened."},
            "trunk_summary": {
                "type": "string",
                "description": (
                    "A SHORT umbrella description of this broader topic now that its detail lives in the "
                    "children below — not a repeat of any one child's content. This becomes the DERIVED "
                    "summary a lower-clearance viewer sees instead of the raw record (this board defaults "
                    "to company-wide visibility regardless of source sensitivity) — describe what the "
                    "topic area covers, never confidential figures, named third parties, or private "
                    "reasoning from the underlying history."
                ),
            },
            "trunk_tags": {"type": "array", "items": {"type": "string"}, "description": "3-6 broad tags for the umbrella topic."},
            "split_children": {
                "type": "array",
                "minItems": 2,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "A short, sharp title for this sub-topic."},
                        "summary": {"type": "string", "description": "The DERIVED summary for this sub-topic — the ONLY version a lower-clearance viewer sees (company-wide by default). Describe the organizational consequence of this sub-topic in one coherent paragraph; never restate confidential financial figures, named third parties, or the detailed private reasoning behind it — that detail stays in the source why[] lines this summary stands in for."},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "3-8 short lowercase tags."},
                        "why_indices": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "0-based indices into the target node's existing why[] history that belong to this sub-topic. Every index across all children must appear exactly once — nothing may be dropped or duplicated.",
                        },
                    },
                    "required": ["title", "summary", "tags", "why_indices"],
                },
                "description": "2-4 sharper sub-nodes to split the target's accumulated history into.",
            },
            "new_content_child_index": {
                "type": "integer",
                "description": "0-based index into split_children: which new sub-node the brand-new information just captured belongs to.",
            },
        },
        "required": ["trunk_title", "trunk_summary", "trunk_tags", "split_children", "new_content_child_index"],
    },
}


def plan_memory_split(target: dict, new_content: dict) -> dict:
    """The second, more expensive call in the split path — only invoked
    when consolidate_memory() above has already decided a specific node
    (given here in FULL, including its complete why[] history) has
    genuinely outgrown itself. Kept as a separate call on purpose: the
    first call runs on EVERY save and is deliberately cheap (lightweight
    candidate summaries only, no full history for the whole board);
    sending every node's full why[] on every save just in case it might
    need splitting would defeat that. This call only ever looks at ONE
    node's full content, and only fires on the rare path where splitting
    was actually chosen."""
    client = Anthropic()
    why = target.get("why") or []
    why_block = "\n".join(f"[{i}] {line}" for i, line in enumerate(why)) or "(no prior history — only the new content below)"
    context = (
        f"Node to split: \"{target.get('title','')}\" (domain: {target.get('domain','other')})\n"
        f"Current tags: {target.get('tags', [])}\n"
        f"Current summary: {target.get('derived','') or '(none)'}\n\n"
        f"Full history (each line is one prior save, indexed):\n{why_block}\n\n"
        "New information just captured that triggered this split:\n"
        f"Source: {new_content.get('source_kind', '')} (by {new_content.get('source_persona', '')} on {new_content.get('date', '')})\n"
        f"Content: {new_content.get('body', '')}\n"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=(
            "Split this one overloaded Decision Memory node into 2-4 sharper, genuinely distinct "
            "sub-nodes. Every existing why[] line must end up assigned to exactly one child — read each "
            "line and group it by which specific sub-topic it's really about. trunk_summary should be "
            "short and umbrella-level now that detail lives in the children, not a copy of any one "
            "child's content.\n\n"
            "CRITICAL — every summary field here (trunk_summary and each child's summary) is a "
            "REDACTION step, not a recap: it becomes the ONLY version of that topic a lower-clearance "
            "viewer ever sees, while confidential figures, named parties, and private reasoning stay "
            "only in the source why[] lines, which these summaries must not restate. Describe "
            "organizational consequence, never confidential substance — when unsure, leave it out."
        ),
        tools=[SPLIT_MEMORY_SCHEMA],
        tool_choice={"type": "tool", "name": "submit_memory_split"},
        messages=[{"role": "user", "content": context}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_memory_split":
            result = dict(block.input)
            children = result.get("split_children") or []
            n = len(why)
            # Defense-in-depth: the model's why_indices partition must
            # cover every existing history line exactly once. Rather than
            # trust it blindly (a dropped line is silent data loss; a
            # duplicated one shows the same fact twice), repair any gap or
            # overlap deterministically instead of failing the whole split.
            seen = set()
            for child in children:
                fixed = []
                for idx in child.get("why_indices", []):
                    if isinstance(idx, int) and 0 <= idx < n and idx not in seen:
                        fixed.append(idx)
                        seen.add(idx)
                child["why_indices"] = fixed
            missing = [i for i in range(n) if i not in seen]
            if missing and children:
                # Any line the model failed to place lands on the largest
                # child rather than being silently dropped.
                largest = max(children, key=lambda c: len(c["why_indices"]))
                largest["why_indices"].extend(missing)
            idx = result.get("new_content_child_index", 0)
            if not isinstance(idx, int) or not (0 <= idx < len(children)):
                idx = 0
            result["new_content_child_index"] = idx
            result["split_children"] = children
            return result
    raise RuntimeError("Claude did not return the expected submit_memory_split tool call.")


class MemoryCandidateIn(BaseModel):
    id: str
    title: str
    tags: list[str] = []
    derived: str = ""
    domain: str = "other"
    parent_id: Optional[str] = None
    child_count: int = 0


class NewMemoryContentIn(BaseModel):
    title: str = ""
    body: str = ""
    source_kind: str = ""
    source_persona: str = ""
    date: str = ""


class ConsolidateMemoryRequest(BaseModel):
    candidates: list[MemoryCandidateIn] = []
    new_content: NewMemoryContentIn


@router.post("/memory/consolidate")
def memory_consolidate(req: ConsolidateMemoryRequest, request: Request, token: Optional[str] = None):
    require_any_session(token_from_request(request, token))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set on the server.")
    try:
        return consolidate_memory([c.model_dump() for c in req.candidates], req.new_content.model_dump())
    except Exception as e:
        _log_and_raise_502("memory/consolidate", e)


class SplitTargetIn(BaseModel):
    id: str
    title: str
    tags: list[str] = []
    derived: str = ""
    domain: str = "other"
    why: list = []


class SplitMemoryRequest(BaseModel):
    target: SplitTargetIn
    new_content: NewMemoryContentIn


@router.post("/memory/split_plan")
def memory_split_plan(req: SplitMemoryRequest, request: Request, token: Optional[str] = None):
    # The second call in the split path (see plan_memory_split's own
    # docstring for why this is a separate endpoint from /memory/consolidate
    # rather than one bigger call) — the frontend only calls this after
    # /memory/consolidate has already returned action="split", and sends
    # the target node's FULL why[] history (already held client-side in
    # company.decisions, never re-fetched) since that's what a real split
    # decision needs to read.
    require_any_session(token_from_request(request, token))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set on the server.")
    try:
        return plan_memory_split(req.target.model_dump(), req.new_content.model_dump())
    except Exception as e:
        _log_and_raise_502("memory/split_plan", e)


class StoryExtractRequest(BaseModel):
    work_item: str
    qa_pairs: list[dict] = []


@router.post("/complete-story/extract")
def complete_story_extract(req: StoryExtractRequest, request: Request, token: Optional[str] = None):
    require_any_session(token_from_request(request, token))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set on the server.")
    from extract import draft_decision_from_story
    try:
        return draft_decision_from_story(req.work_item, req.qa_pairs)
    except Exception as e:
        _log_and_raise_502("complete-story/extract", e)


# ---------------------------------------------------------------------------
# Offboarding / handoff — "Complete the Story" run in reverse at departure.
# Same shape as extract.py: Claude asks targeted questions, then later
# extracts structure from free-text answers via a forced tool call. Grounded
# in the departing persona's OWN goals and contributed decisions (not a fixed
# checklist), and a third call matches the resulting items to the best-fit
# teammate by department and current goal load.
# ---------------------------------------------------------------------------
GENERATE_QUESTIONS_SCHEMA = {
    "name": "submit_handoff_questions",
    "description": "Submit 3-5 targeted handoff questions for a departing employee.",
    "input_schema": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "3-5 short, specific questions that would surface unfinished work, undocumented "
                    "context, and risks a successor would need. Reference their actual goal/decision "
                    "context by name where relevant rather than asking generically."
                ),
            }
        },
        "required": ["questions"],
    },
}


def generate_handoff_questions(persona: dict, goals: list, decisions: list) -> list:
    client = Anthropic()
    context = f"Departing employee: {persona['name']}, {persona['title']}, {persona.get('dept', '')} department.\n"
    if goals:
        context += "Their individual goal(s): " + "; ".join(g["title"] for g in goals) + "\n"
    if decisions:
        context += "Decisions they contributed to company memory: " + "; ".join(d["title"] for d in decisions) + "\n"
    if not goals and not decisions:
        context += "No individual goals or contributed decisions are on file for them — ask generally.\n"

    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=(
            "You generate a short, targeted exit-interview-style question set for an employee who is "
            "leaving their role, so their successor doesn't lose context. Ground questions in the "
            "specifics given rather than a generic checklist."
        ),
        tools=[GENERATE_QUESTIONS_SCHEMA],
        tool_choice={"type": "tool", "name": "submit_handoff_questions"},
        messages=[{"role": "user", "content": context}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_handoff_questions":
            return block.input.get("questions", [])
    raise RuntimeError("Claude did not return the expected submit_handoff_questions tool call.")


EXTRACT_HANDOFF_SCHEMA = {
    "name": "submit_handoff_report",
    "description": "Submit the structured handoff report extracted from the departing employee's answers.",
    "input_schema": {
        "type": "object",
        "properties": {
            "responsibilities": {"type": "array", "items": {"type": "string"}, "description": "Ongoing responsibilities/ownership that need a new owner."},
            "incomplete_tasks": {"type": "array", "items": {"type": "string"}, "description": "Specific unfinished tasks or work in progress."},
            "key_context": {"type": "array", "items": {"type": "string"}, "description": "Undocumented context, relationships, or institutional knowledge a successor would otherwise lose."},
            "risks": {"type": "array", "items": {"type": "string"}, "description": "Risks or open issues flagged in the answers."},
            "needs_clarification": {"type": "array", "items": {"type": "string"}, "description": "Any question that wasn't actually answered usefully."},
        },
        "required": ["responsibilities", "incomplete_tasks", "key_context", "risks", "needs_clarification"],
    },
}


def extract_handoff_report(persona: dict, qa_pairs: list) -> dict:
    client = Anthropic()
    transcript = "\n".join(f"Q: {p.get('question','')}\nA: {p.get('answer','')}" for p in qa_pairs)
    response = client.messages.create(
        model=MODEL,
        max_tokens=768,
        system=(
            f"You turn {persona.get('name', 'an employee')}'s exit-interview answers into a clean structured "
            "handoff report for their successor and manager. Stay strictly grounded in what was actually said "
            "— do not invent detail. Flag an unusable answer via needs_clarification rather than papering over it."
        ),
        tools=[EXTRACT_HANDOFF_SCHEMA],
        tool_choice={"type": "tool", "name": "submit_handoff_report"},
        messages=[{"role": "user", "content": transcript}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_handoff_report":
            return block.input
    raise RuntimeError("Claude did not return the expected submit_handoff_report tool call.")


SUGGEST_DELEGATES_SCHEMA = {
    "name": "submit_delegate_suggestions",
    "description": "Submit a suggested assignee for each handoff item, based on fit and current workload.",
    "input_schema": {
        "type": "object",
        "properties": {
            "suggestions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string", "description": "The exact responsibility/task/goal text this suggestion is for."},
                        "candidate_id": {"type": "string", "description": "id of the suggested candidate — must be one of the ids given."},
                        "rationale": {"type": "string", "description": "One short sentence on why this person fits."},
                    },
                    "required": ["item", "candidate_id", "rationale"],
                },
            }
        },
        "required": ["suggestions"],
    },
}


def suggest_delegates(items: list, candidates: list) -> list:
    client = Anthropic()
    cand_text = "\n".join(
        f"- id={c['id']}: {c['name']}, {c['title']}, {c.get('dept', '')} dept, currently owns {c.get('workload', 0)} goal(s)"
        for c in candidates
    )
    items_text = "\n".join(f"- {i}" for i in items)
    response = client.messages.create(
        model=MODEL,
        max_tokens=768,
        system=(
            "You match each handoff item to the best-fit candidate from the list, weighing department fit "
            "and current workload (fewer existing goals means more capacity). Use ONLY the candidate ids given, "
            "and return one suggestion per item."
        ),
        tools=[SUGGEST_DELEGATES_SCHEMA],
        tool_choice={"type": "tool", "name": "submit_delegate_suggestions"},
        messages=[{"role": "user", "content": f"Candidates:\n{cand_text}\n\nItems needing an owner:\n{items_text}"}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_delegate_suggestions":
            return block.input.get("suggestions", [])
    raise RuntimeError("Claude did not return the expected submit_delegate_suggestions tool call.")


class OffboardingQuestionsRequest(BaseModel):
    persona: PersonaIn
    goals: list[GoalIn] = []
    decisions: list[DecisionIn] = []


@router.post("/offboarding/questions")
def offboarding_questions(req: OffboardingQuestionsRequest, request: Request, token: Optional[str] = None):
    require_any_session(token_from_request(request, token))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set on the server.")
    try:
        questions = generate_handoff_questions(
            req.persona.model_dump(), [g.model_dump() for g in req.goals], [d.model_dump() for d in req.decisions]
        )
        return {"questions": questions}
    except Exception as e:
        _log_and_raise_502("offboarding/questions", e)


class OffboardingExtractRequest(BaseModel):
    persona: PersonaIn
    qa_pairs: list[dict] = []


@router.post("/offboarding/extract")
def offboarding_extract(req: OffboardingExtractRequest, request: Request, token: Optional[str] = None):
    require_any_session(token_from_request(request, token))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set on the server.")
    try:
        return extract_handoff_report(req.persona.model_dump(), req.qa_pairs)
    except Exception as e:
        _log_and_raise_502("offboarding/extract", e)


# ---------------------------------------------------------------------------
# The Vantage Room — the strategist / gap-finder (renamed from "Vantage"
# 2026-09-02; internal identifiers below keep the shorter name). Detection of
# WHAT counts as a gap (overdue review, orphaned goal, dead-end goal,
# overlapping decisions) is deliberately deterministic plain-JS logic on the
# frontend (see detectVantageCandidates() in static/index.html) — no model
# call is needed to know a review date has passed or a goal has no children,
# and keeping detection rule-based means it's auditable and doesn't
# hallucinate gaps that aren't there. Claude's job here is narrower and
# lower-risk: take each already-detected candidate and rewrite it in the
# voice of a Company Strategy Head — a gap worth attacking now, or an idea
# for expansion it points to — with a concrete suggested next step, for a
# specific viewer's clearance. Still one candidate in, one polished gap out,
# never inventing new gaps outside the candidates it was given. The frontend
# then walks freshly surfaced gaps one at a time (see the vantageReviewQueue
# handling around runVantageScan) rather than dumping the whole list at
# once, direct founder feedback: "it should give random thoughts and the
# user can choose to start a chat based on that thought... or choose to skip
# it and get the next gap."
# ---------------------------------------------------------------------------
POLISH_GAPS_SCHEMA = {
    "name": "submit_vantage_gaps",
    "description": "Submit a polished write-up for each gap candidate given.",
    "input_schema": {
        "type": "object",
        "properties": {
            "gaps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string", "description": "Must exactly match one of the candidate ids given."},
                        "title": {"type": "string", "description": "A short, clear headline for this gap."},
                        "description": {"type": "string", "description": "1-3 sentences explaining the gap in plain language, grounded strictly in the candidate's summary — do not invent facts not present in it."},
                        "severity": {"type": "string", "enum": ["high", "medium", "low"], "description": "Your own judgment of urgency; may differ from the candidate's suggested severity if warranted."},
                        "recommended_next_step": {"type": "string", "description": "One concrete, actionable next step someone could actually take."},
                    },
                    "required": ["candidate_id", "title", "description", "severity", "recommended_next_step"],
                },
            }
        },
        "required": ["gaps"],
    },
}


def polish_vantage_gaps(persona: dict, candidates: list) -> list:
    client = Anthropic()
    cand_text = "\n\n".join(
        f"- candidate_id={c['id']} | type={c['type']} | suggested severity={c.get('severity','medium')}\n"
        f"  {c['summary']}"
        for c in candidates
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=(
            f"You are the Company Strategy Head inside The Vantage Room, Cognex's forward-looking "
            f"strategy space. You're writing for {persona.get('name','the viewer')} "
            f"({persona.get('title','')}, clearance level {persona.get('level', 1)} of 6). You are "
            "given a list of gap candidates already detected by deterministic rules (overdue "
            "decision reviews, unassigned or dead-end goals, decisions worth cross-checking) — your "
            "job is to reframe each one as a genuine strategic thought: either a gap worth attacking "
            "now, or an idea for expansion it points to. Stay strictly grounded in the summary given "
            "for each candidate; do not invent additional facts, and do not add gaps beyond the "
            "candidates given. Be direct and specific, like a sharp strategy lead thinking out loud, "
            "not vague corporate-speak."
        ),
        tools=[POLISH_GAPS_SCHEMA],
        tool_choice={"type": "tool", "name": "submit_vantage_gaps"},
        messages=[{"role": "user", "content": f"Gap candidates:\n\n{cand_text}"}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_vantage_gaps":
            return block.input.get("gaps", [])
    raise RuntimeError("Claude did not return the expected submit_vantage_gaps tool call.")


class GapCandidateIn(BaseModel):
    id: str
    type: str
    title: str
    severity: str = "medium"
    summary: str
    related_goal_ids: list[str] = []
    related_decision_ids: list[str] = []


class VantageScanRequest(BaseModel):
    persona: PersonaIn
    candidates: list[GapCandidateIn] = []


@router.post("/vantage/scan")
def vantage_scan(req: VantageScanRequest, request: Request, token: Optional[str] = None):
    require_any_session(token_from_request(request, token))
    if not req.candidates:
        return {"gaps": []}
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set on the server.")
    try:
        gaps = polish_vantage_gaps(req.persona.model_dump(), [c.model_dump() for c in req.candidates])
        return {"gaps": gaps}
    except Exception as e:
        _log_and_raise_502("vantage/scan", e)


class DelegateCandidateIn(BaseModel):
    id: str
    name: str
    title: str
    dept: str = ""
    workload: int = 0


class SuggestDelegatesRequest(BaseModel):
    items: list[str] = []
    candidates: list[DelegateCandidateIn] = []


@router.post("/offboarding/suggest-delegates")
def offboarding_suggest_delegates(req: SuggestDelegatesRequest, request: Request, token: Optional[str] = None):
    require_any_session(token_from_request(request, token))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set on the server.")
    try:
        suggestions = suggest_delegates(req.items, [c.model_dump() for c in req.candidates])
        return {"suggestions": suggestions}
    except Exception as e:
        _log_and_raise_502("offboarding/suggest-delegates", e)
