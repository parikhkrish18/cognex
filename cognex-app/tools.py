"""
Tool implementations Claude can call during an Ask Cognex turn, plus their
JSON Schema definitions for the Anthropic tool-use API.

Design point: every function here takes `persona` as its FIRST argument, bound
by the server from the authenticated request — never from Claude's tool-call
JSON. Claude's tool call only ever supplies the *query*. This is what makes
permission enforcement real rather than advisory: even if a compromised or
adversarial prompt convinced Claude to ask for "the CEO's confidential
decisions," the tool functions don't accept an identity argument from Claude
to spoof — they use whichever persona the server bound for this HTTP request.

In production, swap this file's simple in-memory filtering for real calls to
your vector index (semantic search over decision/goal free text, e.g. with
Voyage AI embeddings as Anthropic recommends) and graph store (multi-hop
traversal for get_related). The function signatures and the permission
discipline stay the same either way.
"""

import re
from data import DECISIONS, GOALS, Persona, goal_chain
from permissions import serialize_decision_for, accessible


def _tokenize(s: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9\s]", " ", s.lower()).split())


def search_decisions(persona: Persona, query: str) -> list[dict]:
    """Semantic-ish keyword search here; swap for a real vector search in production."""
    q_words = _tokenize(query)
    scored = []
    for d in DECISIONS:
        score = 0
        for tag in d.tags:
            tag_words = _tokenize(tag)
            if tag_words and tag_words.issubset(q_words):
                score += 4 if len(tag_words) >= 2 else 2
        for w in _tokenize(d.title):
            if len(w) > 3 and w in q_words:
                score += 1
        if score > 0:
            scored.append((score, d))
    scored.sort(key=lambda pair: -pair[0])
    # Permission filtering happens HERE, after ranking, before anything is returned.
    return [serialize_decision_for(persona, d) for _, d in scored[:5]]


def get_decision(persona: Persona, decision_id: str) -> dict:
    d = next((x for x in DECISIONS if x.id == decision_id), None)
    if not d:
        return {"error": f"No decision with id '{decision_id}'."}
    return serialize_decision_for(persona, d)


def search_goals(persona: Persona, query: str) -> list[dict]:
    q_words = _tokenize(query)
    results = []
    for g in GOALS:
        title_words = _tokenize(g.title)
        matches = any(len(w) > 3 and w in q_words for w in title_words)
        matches = matches or (g.dept and g.dept.lower() in q_words)
        if matches:
            results.append(g)
    # All goals in this dataset are company-wide visible, so no filtering step
    # is needed here — but in production this is where a goal-level ACL check
    # would go, identical in shape to serialize_decision_for.
    return [
        {"id": g.id, "kicker": g.kicker, "title": g.title, "dept": g.dept, "depth": g.depth}
        for g in results
    ]


def get_goal_chain(persona: Persona, goal_id: str) -> list[dict]:
    """Walk a goal up to the company root — e.g. an individual's work up to company strategy."""
    from data import goal_by_id
    g = goal_by_id(goal_id)
    if not g:
        return [{"error": f"No goal with id '{goal_id}'."}]
    return [{"id": x.id, "kicker": x.kicker, "title": x.title} for x in goal_chain(g)]


def get_my_context(persona: Persona, _unused: str = "") -> dict:
    """Grounds Claude on who is actually asking, without requiring it to guess from prose."""
    my_goal = next((g for g in GOALS if g.owner == persona.id), None)
    return {
        "name": persona.name,
        "title": persona.title,
        "department": persona.dept,
        "clearance_level": persona.level,
        "individual_goal": (
            {"id": my_goal.id, "title": my_goal.title} if my_goal else None
        ),
    }


# ---------------------------------------------------------------------------
# Tool schemas for the Anthropic Messages API `tools` parameter.
# Note there is no "persona" or "user_id" parameter on any of these — the
# identity the tool executes against is bound server-side, not supplied by
# the model. See agent.py's TOOL_IMPLEMENTATIONS dispatch.
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
            "properties": {
                "query": {"type": "string", "description": "Free-text search query, e.g. 'Germany market entry'."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_decision",
        "description": "Fetch a single decision record by id, filtered to the current viewer's clearance.",
        "input_schema": {
            "type": "object",
            "properties": {"decision_id": {"type": "string"}},
            "required": ["decision_id"],
        },
    },
    {
        "name": "search_goals",
        "description": "Search the company/department/team/individual goal hierarchy for records relevant to a query.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "get_goal_chain",
        "description": (
            "Given a goal id, return the full chain from that goal up to the company-level goal it "
            "ladders up to. Use this to answer 'why am I working on X' or 'how does this connect to "
            "company strategy' questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"goal_id": {"type": "string"}},
            "required": ["goal_id"],
        },
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

TOOL_IMPLEMENTATIONS = {
    "search_decisions": search_decisions,
    "get_decision": get_decision,
    "search_goals": search_goals,
    "get_goal_chain": get_goal_chain,
    "get_my_context": get_my_context,
}
