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
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel

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
    GENERATE_IMAGE_SCHEMA,
]


def _make_tool_impls(persona: Persona, decisions: list, goals: list):
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

    return {
        "search_decisions": lambda tool_input: search_decisions(tool_input.get("query", "")),
        "get_decision": lambda tool_input: get_decision(tool_input.get("decision_id", "")),
        "search_goals": lambda tool_input: search_goals(tool_input.get("query", "")),
        "get_goal_chain": lambda tool_input: get_goal_chain(tool_input.get("goal_id", "")),
        "get_my_context": lambda tool_input: get_my_context(),
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
        line = f'- {d.id}: "{d.title}"'
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


def run_agent_turn(persona: Persona, decisions: list, goals: list, user_message: str, history=None):
    client = Anthropic()
    impls = _make_tool_impls(persona, decisions, goals)
    messages = list(history or [])
    messages.append({"role": "user", "content": user_message})
    tool_call_log = []
    file_ids = []
    all_tools = TOOL_SCHEMAS + SERVER_TOOLS

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model=MODEL, max_tokens=8192, system=_system_prompt(persona, decisions),
            tools=all_tools, tool_choice={"type": "auto"}, messages=messages,
        )
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
        "generate_image": "Generating an image…",
    }.get(tool_name, "Thinking…")


def run_agent_turn_stream(persona: Persona, decisions: list, goals: list, user_message: str, history=None):
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
    client = Anthropic()
    impls = _make_tool_impls(persona, decisions, goals)
    messages = list(history or [])
    messages.append({"role": "user", "content": user_message})
    tool_call_log = []
    file_ids = []
    all_tools = TOOL_SCHEMAS + SERVER_TOOLS

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

        if not tool_results:
            continue
        messages.append({"role": "user", "content": tool_results})

    yield {
        "event": "done",
        "answer_override": "I wasn't able to settle on an answer within the tool-call budget for this turn — try a narrower question.",
        "tool_calls": tool_call_log,
        "files": _resolve_files(client, file_ids),
    }


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


@router.get("/health")
def health():
    return {"ok": True, "api_key_configured": bool(os.environ.get("ANTHROPIC_API_KEY"))}


@router.get("/files/{file_id}")
def get_generated_file(file_id: str):
    """Streams back a file code_execution generated (a chart image, a CSV,
    etc.) by proxying Anthropic's Files API through our own server API key —
    the frontend never talks to Anthropic directly. NOTE on scope: like the
    rest of this app, this isn't access-controlled per company/persona (any
    caller who has a file_id can fetch it) — that's consistent with this
    app's documented "no real auth yet" state (see the persistence build-log
    entry) and these ids are opaque, short-lived, and only ever handed to the
    same session that asked the question that generated them. Real per-user
    file access control is part of the same future "real authentication"
    phase already tracked for the rest of the app.

    An "img-" prefixed id is an OpenAI-generated image served straight out of
    the in-process _GENERATED_IMAGES store (see the image-generation block
    near the top of this file) rather than Anthropic's Files API — those ids
    never existed on Anthropic's side at all."""
    if file_id.startswith("img-"):
        rec = _GENERATED_IMAGES.get(file_id)
        if not rec:
            raise HTTPException(status_code=404, detail=f"Generated image '{file_id}' was not found — it may have expired (only the most recent {_GENERATED_IMAGES_MAX} images are kept) or the server may have restarted since it was generated.")
        return Response(
            content=rec["bytes"],
            media_type=rec["mime_type"] or "application/octet-stream",
            headers={"Content-Disposition": f'inline; filename="{rec["filename"] or file_id}"'},
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
def ask(req: AskRequest):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set on the server.")
    persona = Persona(**req.persona.model_dump())
    decisions = [Decision(**d.model_dump()) for d in req.decisions]
    goals = [Goal(**g.model_dump()) for g in req.goals]
    try:
        return run_agent_turn(persona, decisions, goals, req.message, req.history)
    except Exception as e:
        _log_and_raise_502("ask", e)


@router.post("/ask/stream")
def ask_stream(req: AskRequest):
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
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set on the server.")
    persona = Persona(**req.persona.model_dump())
    decisions = [Decision(**d.model_dump()) for d in req.decisions]
    goals = [Goal(**g.model_dump()) for g in req.goals]

    def sse():
        try:
            for event in run_agent_turn_stream(persona, decisions, goals, req.message, req.history):
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
GENERATE_STORY_QUESTIONS_SCHEMA = {
    "name": "submit_story_questions",
    "description": "Submit whether there is real work to ask this person about, and if so, specific questions grounded in it.",
    "input_schema": {
        "type": "object",
        "properties": {
            "has_work_to_capture": {
                "type": "boolean",
                "description": (
                    "True ONLY if the goal/decision context given actually gives something concrete "
                    "and current to ask about. False if there is genuinely nothing specific — no "
                    "individual goal and no recent contributed decisions. Do not invent a plausible-"
                    "sounding 'current priorities' question when there's nothing real to reference; "
                    "many senior roles (CEO, CFO) legitimately have no single tracked individual goal, "
                    "and that's a normal, expected false here, not a gap to paper over."
                ),
            },
            "work_item_label": {
                "type": "string",
                "description": (
                    "A short, specific label for what's being captured, naming the actual goal or "
                    "decision title given (e.g. the literal goal title, not a role-based placeholder "
                    "like '<title> — current priorities'). Empty string if has_work_to_capture is false."
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
                            "description": "A specific question that names the actual goal or decision by title — never a generic templated question like 'why does this matter for the team right now'.",
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


def generate_story_questions(persona: dict, individual_goal: Optional[dict], recent_decisions: list) -> dict:
    client = Anthropic()
    context = f"{persona.get('name', '')}, {persona.get('title', '')}, {persona.get('dept', '')} department.\n"
    if individual_goal:
        context += f"Their individual goal: {individual_goal.get('title', '')}"
        if individual_goal.get("kicker"):
            context += f" ({individual_goal['kicker']})"
        context += "\n"
    if recent_decisions:
        context += "Decisions they're a contributor on: " + "; ".join(d.get("title", "") for d in recent_decisions) + "\n"
    if not individual_goal and not recent_decisions:
        context += "No individual goal is assigned to them, and no decisions list them as a contributor.\n"

    response = client.messages.create(
        model=MODEL,
        max_tokens=768,
        system=(
            "You decide whether there is real, specific work to ask this person about, based ONLY on "
            "the goal/decision context given. If they have no individual goal and no recent "
            "contributed decisions, set has_work_to_capture to false rather than asking generic "
            "'what are your current priorities'-style questions — that produces meaningless answers "
            "and wastes the person's time. If there IS real context, write 2-3 short, specific "
            "questions that name the actual goal or decision by title, not generic filler."
        ),
        tools=[GENERATE_STORY_QUESTIONS_SCHEMA],
        tool_choice={"type": "tool", "name": "submit_story_questions"},
        messages=[{"role": "user", "content": context}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_story_questions":
            return block.input
    raise RuntimeError("Claude did not return the expected submit_story_questions tool call.")


class StoryQuestionsRequest(BaseModel):
    persona: PersonaIn
    individual_goal: Optional[GoalIn] = None
    recent_decisions: list[DecisionIn] = []


@router.post("/complete-story/questions")
def complete_story_questions(req: StoryQuestionsRequest):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set on the server.")
    try:
        return generate_story_questions(
            req.persona.model_dump(),
            req.individual_goal.model_dump() if req.individual_goal else None,
            [d.model_dump() for d in req.recent_decisions],
        )
    except Exception as e:
        _log_and_raise_502("complete-story/questions", e)


# ---------------------------------------------------------------------------
# Memory consolidation — the Brain Board, added 2026-08-30. The founder's
# explicit call (asked directly rather than guessed): Decision Memory should
# stop growing one record per save and instead read as a small, evolving set
# of topic nodes around a central "company brain" — a real collective
# summary, not a chat transcript. Every save path (Ask Cognex's "Save to
# Decision Memory", Complete the Story, the Slack candidate-confirm flow,
# Vantage's "mark plan done") now calls this endpoint FIRST: given the
# board's existing nodes (lightweight: id/title/tags/current summary — full
# why/alternatives/risks content is deliberately NOT sent, to keep this call
# fast and cheap) and the new content just captured, Claude decides whether
# it's really the same topic as an existing node (merge, rewriting that
# node's summary to reflect everything now known together) or genuinely new
# (a new node). This is a real judgment call, not a keyword-overlap
# heuristic — "we sell design services to startups" and "we're a boutique
# design and engineering studio" are the same topic despite sharing almost
# no words, and a keyword match would have missed that (see the 2026-08-29
# memory-grounding entry, the same class of gap on the search side of this
# feature).
CONSOLIDATE_MEMORY_SCHEMA = {
    "name": "submit_memory_consolidation",
    "description": (
        "Decide whether newly captured information belongs on an existing Decision Memory node "
        "(the same topic) or needs a new node, and produce the resulting title/summary/tags."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["merge", "new_node"],
                "description": (
                    "'merge' if the new information is really about the same topic as one of the "
                    "existing nodes listed — prefer this whenever there's real topical overlap, since "
                    "the whole point is fewer, richer nodes rather than one per save. 'new_node' only "
                    "when it genuinely doesn't fit any existing node's topic."
                ),
            },
            "merge_into_id": {
                "type": "string",
                "description": "The exact id of the existing node to merge into, copied from the candidate list. Required when action is 'merge'; empty string otherwise.",
            },
            "title": {
                "type": "string",
                "description": "A short, clear topic title (e.g. 'What we sell', 'Pricing strategy', 'Q4 product launch'). For a merge, refine the existing title only if the new content genuinely broadens the topic; otherwise keep it close to what's already there.",
            },
            "summary": {
                "type": "string",
                "description": (
                    "One coherent paragraph describing everything currently known about this topic, "
                    "written fresh to incorporate the new information — not the old summary with the "
                    "new fact bolted on, and not just the new fact alone. This is what someone sees "
                    "at a glance on the board."
                ),
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3-8 short lowercase topic tags for this node (merged with, not replacing, any relevant existing tags for a merge).",
            },
        },
        "required": ["action", "merge_into_id", "title", "summary", "tags"],
    },
}


def consolidate_memory(candidates: list, new_content: dict) -> dict:
    client = Anthropic()
    if candidates:
        cand_lines = [
            f'- id="{c.get("id","")}" | title="{c.get("title","")}" | tags={c.get("tags", [])} | current summary: {c.get("derived","") or "(none yet)"}'
            for c in candidates
        ]
        candidates_block = "\n".join(cand_lines)
    else:
        candidates_block = "(the board is empty — this will be the first node)"

    context = (
        f"The board currently has {len(candidates)} node(s).\n\n"
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
            "You maintain a company's collective knowledge board -- a small set of BROAD topic nodes "
            "(think of the handful of areas a new hire would need briefing on: the product/what the "
            "company sells, go-to-market and customer acquisition, operations and supply chain, pricing "
            "and unit economics, hiring and team, etc.) -- not a growing pile of narrow one-off notes, "
            "and not one node per question someone happened to ask. Reported directly by the founder "
            "(2026-08-31): earlier behavior left the board looking like a sprawling web of narrow, "
            "single-fact nodes instead of a small set of nodes that actually show what the company "
            "collectively knows -- your job is to prevent that.\n\n"
            "Decide whether the new information belongs on one of the existing nodes listed (merge into "
            "it, rewriting its summary to reflect everything now known about that broader area together) "
            "or is genuinely a new area of knowledge with no real home yet (create a new node). Default "
            "to merging: if the new information is a specific instance, example, or detail WITHIN a "
            "broader area an existing node already covers (e.g. a specific outreach tactic belongs under "
            "an existing 'customer acquisition' or 'marketing' node; a specific vendor risk belongs under "
            "an existing 'operations'/'supply chain' node), merge into it and fold the new specifics into "
            "the summary -- do not create a narrow new node just because the exact wording or sub-topic "
            "wasn't covered before. Only create a new node when the information is about a genuinely "
            "different area of the business that none of the existing nodes are really about at all. The "
            "more nodes already exist, the harder you should lean toward merging rather than adding "
            "another -- a board with a dozen or more nodes has almost certainly fragmented too far, and "
            "you should actively look for the best-fit existing node to fold new information into rather "
            "than defaulting to a fresh one. The summary you write must read as one coherent paragraph "
            "describing the current state of knowledge on that broader topic, not the newest fact bolted "
            "onto old text -- it should genuinely teach someone what the company now knows in this area, "
            "not just log that something happened."
        ),
        tools=[CONSOLIDATE_MEMORY_SCHEMA],
        tool_choice={"type": "tool", "name": "submit_memory_consolidation"},
        messages=[{"role": "user", "content": context}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_memory_consolidation":
            result = dict(block.input)
            # Defense-in-depth against a hallucinated id: never let a
            # "merge" into an id that isn't actually one of the candidates
            # given silently corrupt a different node or 404 downstream --
            # downgrade to a new node instead, which is always safe.
            candidate_ids = {c.get("id") for c in candidates}
            if result.get("action") == "merge" and result.get("merge_into_id") not in candidate_ids:
                result["action"] = "new_node"
                result["merge_into_id"] = ""
            return result
    raise RuntimeError("Claude did not return the expected submit_memory_consolidation tool call.")


class MemoryCandidateIn(BaseModel):
    id: str
    title: str
    tags: list[str] = []
    derived: str = ""


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
def memory_consolidate(req: ConsolidateMemoryRequest):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set on the server.")
    try:
        return consolidate_memory([c.model_dump() for c in req.candidates], req.new_content.model_dump())
    except Exception as e:
        _log_and_raise_502("memory/consolidate", e)


class StoryExtractRequest(BaseModel):
    work_item: str
    qa_pairs: list[dict] = []


@router.post("/complete-story/extract")
def complete_story_extract(req: StoryExtractRequest):
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
def offboarding_questions(req: OffboardingQuestionsRequest):
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
def offboarding_extract(req: OffboardingExtractRequest):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set on the server.")
    try:
        return extract_handoff_report(req.persona.model_dump(), req.qa_pairs)
    except Exception as e:
        _log_and_raise_502("offboarding/extract", e)


# ---------------------------------------------------------------------------
# Vantage — the strategist / gap-finder. Detection of WHAT counts as a gap
# (overdue review, orphaned goal, dead-end goal, overlapping decisions) is
# deliberately deterministic plain-JS logic on the frontend (see
# detectVantageCandidates() in static/index.html) — no model call is needed
# to know a review date has passed or a goal has no children, and keeping
# detection rule-based means it's auditable and doesn't hallucinate gaps that
# aren't there. Claude's job here is narrower and lower-risk: take each
# already-detected candidate and turn it into a clear, well-prioritized
# write-up with a concrete suggested next step, for a specific viewer's
# clearance — one candidate in, one polished gap out, never inventing new
# gaps outside the candidates it was given.
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
            f"You are Vantage, Cognex's forward-looking strategist. You're writing for "
            f"{persona.get('name','the viewer')} ({persona.get('title','')}, clearance level "
            f"{persona.get('level', 1)} of 6). You are given a list of gap candidates already "
            "detected by deterministic rules (overdue decision reviews, unassigned or dead-end "
            "goals, decisions worth cross-checking) — your job is ONLY to write each one up "
            "clearly and suggest one concrete next step. Stay strictly grounded in the summary "
            "given for each candidate; do not invent additional facts, and do not add gaps beyond "
            "the candidates given. Be direct and specific, not vague corporate-speak."
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
def vantage_scan(req: VantageScanRequest):
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
def offboarding_suggest_delegates(req: SuggestDelegatesRequest):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set on the server.")
    try:
        suggestions = suggest_delegates(req.items, [c.model_dump() for c in req.candidates])
        return {"suggestions": suggestions}
    except Exception as e:
        _log_and_raise_502("offboarding/suggest-delegates", e)
