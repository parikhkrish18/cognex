# Cognex reference backend

A minimal, real implementation of how the Cognex prototype's rule-based
engine becomes an actual Claude integration — same data model, same
permission logic, but Claude does the reasoning via tool use instead of a
regex intent list.

This is a **reference implementation**, not a production service: in-memory
data (resets on restart), no real auth, no vector/graph database. Its job is
to get the *architecture* right so you can swap in real infrastructure piece
by piece without re-deriving the design.

## What's here

| File | Role |
|---|---|
| `data.py` | The memory model — personas, decisions, goals. Same records as the front-end prototype, ported to Python. |
| `permissions.py` | The enforcement point. Strips restricted fields from a decision *before* it can be serialized into anything sent to Claude. |
| `tools.py` | The tools Claude can call (`search_decisions`, `get_decision`, `search_goals`, `get_goal_chain`, `get_my_context`) — each one filters through `permissions.py`. |
| `agent.py` | The Ask Cognex orchestration loop: calls Claude with tools, executes whatever Claude asks for, feeds results back, repeats until Claude answers. |
| `extract.py` | Complete the Story's structured extraction — turns messy free-text answers into a clean Decision Memory draft via a forced tool call. |
| `derived_context.py` | The privileged, offline process that drafts a decision's *derived* (lower-clearance-safe) summary once, at approval time — deliberately separate from the live query path. |
| `server.py` | FastAPI app exposing all of the above over HTTP. |
| `test_offline.py` | Tests that don't call the network — verify permission enforcement and wiring. **These are the ones I actually ran.** |
| `demo_cli.py` | A CLI that runs the flagship "same question, different clearance" scenario through a real Claude call, once you have an API key. |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # required for anything that calls Claude
```

Then either:

```bash
python3 test_offline.py          # no API key needed — permission logic + wiring
python3 demo_cli.py              # needs API key — a real live run of the flagship scenario
uvicorn server:app --reload --port 8787   # run it as a service
```

With the server running:

```bash
curl "http://localhost:8787/decisions?persona_id=intern"
curl -X POST http://localhost:8787/ask \
  -H "content-type: application/json" \
  -d '{"persona_id": "ceo", "message": "Why did we decide not to enter Germany?"}'
```

## What's actually been verified vs. what hasn't

I ran `test_offline.py` in this environment and all 12 tests pass — that
covers permission enforcement (the part that matters most: restricted fields
are provably absent from the serialized dict, not just hidden in a UI),
correct tool dispatch, goal-chain traversal, and the FastAPI endpoints
end-to-end through the HTTP layer.

I did **not** have an `ANTHROPIC_API_KEY` available in the environment I
built this in, so `agent.py`, `extract.py`, and `derived_context.py` — the
three files that actually call Claude — are written against the installed
`anthropic` SDK's real method signature (checked directly via
`inspect.signature`), but have not been exercised against a live API call.
Run `demo_cli.py` with your own key before trusting the tool-use loop in
front of anyone; the shape should be right, but "should be right" and
"I watched it work" are different claims and I want to be honest about which
one this is.

## The architectural decisions this code makes, and why

**Permission filtering happens in Python, before any Claude call — not as a
system-prompt instruction.** `permissions.py`'s `serialize_decision_for`
doesn't return a redacted-looking field; it doesn't create the key at all
when the viewer isn't cleared. `test_intern_never_receives_raw_acquisition_fields`
checks this by asserting the raw sentences aren't even present in the
JSON-serialized tool result. Claude never has an opportunity to leak what it
was never given, regardless of how the question is phrased.

**Identity is server-bound, never model-supplied.** Every tool schema in
`tools.py` deliberately has no `persona` or `user_id` parameter (see
`test_tool_schemas_are_well_formed`). `agent.py` injects the authenticated
persona into every tool call from the server side. Claude can ask
`search_decisions` for anything it wants; it cannot ask to be someone else.

**Derived-context summaries are authored once, offline, by a privileged
process — never generated live in response to a lower-clearance query.**
See the docstring in `derived_context.py`. This is the piece most tempting to
get wrong (it's simpler to just ask Claude to summarize-down on the spot),
and getting it wrong means raw confidential content enters a live request
triggered by exactly the person you're trying to keep it from.

**Claude decides which tools to call, rather than a hard-coded intent
list.** The front-end prototype's `INTENTS` regex array only handles the
questions it was written for. `agent.py` gives Claude the same five tools for
any question and lets it figure out what to retrieve — this is what makes
"ask anything" actually true instead of "ask one of these six things."

**The agent loop is stateless per call.** `run_agent_turn` doesn't hold
memory between requests — the memory is `data.py`/the real data store behind
it. This is the actual distinction between Cognex and a chatbot with a long
context window: the organization's memory persists in a database that
outlives any single conversation, and Claude is re-grounded from it on every
turn rather than carrying it around.

## Where the real infrastructure goes when you're past the reference stage

- `data.py`'s dicts → an identity provider (SCIM-synced from Okta/Azure AD)
  for `PERSONAS`, and a real graph store (Neo4j, or Postgres with an edges
  table) for `DECISIONS`/`GOALS`.
- `tools.py`'s keyword search → a vector index (Anthropic recommends Voyage
  AI embeddings) over the free-text fields, likely hybrid with the keyword/tag
  matching that's already there.
- `server.py`'s `persona_id` parameter → real auth middleware resolving
  identity from a verified session before any handler runs.
- Prompt caching (`cache_control` on the system block in `agent.py`) once
  you have real request volume — the system prompt and tool schemas are
  identical on every call for a given deployment.
- Structured logging of every tool call `agent.py` makes, per request — you
  want an audit trail of what memory was actually retrieved for whom, both
  for debugging and because "what did the AI show this person" is a real
  question your security team will ask eventually.
