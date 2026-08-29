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

from anthropic import Anthropic
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
MAX_TOOL_ROUNDS = 8

# ---------------------------------------------------------------------------
# Real web search + real code execution — added 2026-08-29 so Ask Cognex can
# actually search the web and actually run numbers/build charts, instead of
# only answering from the model's own knowledge and prose estimates. These
# are Anthropic *server* tools: unlike search_decisions/get_decision/etc.
# below (which we implement and execute ourselves), Anthropic's API executes
# these itself and returns the result inline in the same response — see the
# tool_use handling in run_agent_turn for why that changes the loop slightly.
#
# max_uses=5 bounds worst-case web search cost per turn ($10/1,000 searches,
# per Anthropic's pricing docs, plus normal token cost for retrieved pages).
# Paired with a web_search version >= 20260209, Anthropic does not separately
# bill code_execution time at all (only standard token costs apply) per their
# current docs — worth re-checking if that changes, but as of this writing
# adding real computation/charts here has no meaningful extra infra cost on
# top of the web search cost already budgeted.
SERVER_TOOLS = [
    {"type": "web_search_20260318", "name": "web_search", "max_uses": 5},
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
                    score += 1
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
    }


def _system_prompt(persona: Persona) -> str:
    return f"""You are Cognex, the AI assistant {persona.name} ({persona.title}, {persona.dept}
department, clearance level {persona.level} of 6) uses for day-to-day company work — research,
writing, coding, documentation, analysis, debugging, brainstorming, and anything else they'd
otherwise reach for a general-purpose AI assistant to do. Answer those requests fully and
directly, the same way you would in any other context: complete, well-structured answers, real
working code with explanation when asked to write code, thorough research or documentation when
asked for it. Do not artificially shorten an answer that deserves depth — match length and
structure to what the question actually needs, not to a fixed target.

You have two real capabilities beyond your own training, and you should reach for them whenever
they would make an answer better rather than defaulting to a prose guess:
- Web search: use it for anything current, time-sensitive, or better answered with a real source
  (prices, news, a competitor's current product, a fact you're not fully certain of) — don't
  guess at something searchable. Do NOT put confidential company details (decision content,
  financial figures, names of people or deals) into a search query — search generically; if a
  question genuinely needs company-internal information, use your company-memory tools below
  instead of the web for that part.
- Code execution (a real Python sandbox): use it whenever a question involves actual computation,
  data analysis, or a chart — run the numbers for real instead of estimating them in prose, and
  generate an actual chart image when a visualization would help instead of describing one in
  words. If the person gives you data (pasted numbers, a table), work with it directly in code.

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
    entirely — the id is still usable — it just won't have a nice filename."""
    files, seen = [], set()
    for fid in file_ids:
        if fid in seen:
            continue
        seen.add(fid)
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
            model=MODEL, max_tokens=8192, system=_system_prompt(persona),
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


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------
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
    phase already tracked for the rest of the app."""
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
    except Exception as e:  # surface upstream Claude/API errors legibly rather than a bare 500
        raise HTTPException(status_code=502, detail=f"Model call failed: {e}")


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
        raise HTTPException(status_code=502, detail=f"Model call failed: {e}")


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
        raise HTTPException(status_code=502, detail=f"Model call failed: {e}")


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
        raise HTTPException(status_code=502, detail=f"Model call failed: {e}")


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
        raise HTTPException(status_code=502, detail=f"Model call failed: {e}")


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
        raise HTTPException(status_code=502, detail=f"Model call failed: {e}")
