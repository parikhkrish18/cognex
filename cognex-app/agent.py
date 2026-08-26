"""
The Ask Cognex orchestration loop — this is the "integrate Claude" part.

Flow per turn:
  1. Take the authenticated persona (bound server-side — see server.py) and the
     user's message.
  2. Call Claude with the tool schemas from tools.py and tool_choice="auto".
     Claude decides which tools to call, in what order, possibly more than once,
     based on the question — this replaces the hard-coded regex INTENTS list
     from the front-end prototype with something that generalizes to arbitrary
     questions.
  3. Whenever Claude requests a tool call, execute the corresponding Python
     function with the SERVER-BOUND persona (never a persona Claude supplies)
     and feed the (already permission-filtered) result back as a tool_result.
  4. Repeat until Claude stops requesting tools and returns a final text answer.
  5. Return the answer text plus a log of every tool call made, so the caller
     can render "retrieved from company memory" citations exactly like the
     front-end prototype does.

Context management notes (see README for the full explanation):
  - Conversation history is passed back in full for now; past a few dozen turns
    you'd summarize older turns rather than let the transcript grow unbounded.
  - System prompt + tool schemas are identical on every call for a given
    deployment, which makes them a good candidate for Anthropic's prompt
    caching (cache_control on the system block) once you're doing real request
    volume — cuts cost and latency on everything after the first call.
  - Nothing about "memory" lives in this loop. The loop is stateless per call;
    the memory is the data layer in data.py/tools.py. That's the architectural
    point: Claude reasons over what's retrieved for it, it doesn't carry the
    company's memory around in a context window between sessions.
"""

import json
import os
from anthropic import Anthropic
from data import Persona
from tools import TOOL_SCHEMAS, TOOL_IMPLEMENTATIONS

# Model IDs change over time — check https://docs.claude.com/en/docs/about-claude/models
# for the current recommended model before deploying this for real.
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
MAX_TOOL_ROUNDS = 6  # backstop against a runaway tool-call loop


def _system_prompt(persona: Persona) -> str:
    return f"""You are Cognex, an organizational memory assistant. You answer questions by
retrieving from the company's Decision Memory and Goal graph using your tools — you have
no knowledge of company specifics beyond what your tools return in this conversation.

You are currently answering {persona.name} ({persona.title}, {persona.dept} department,
clearance level {persona.level} of 6). Every tool call you make is automatically filtered
to what this person is allowed to see — you cannot retrieve more than that, and you should
not imply you know more than what a tool result actually contained. If a tool result says a
record is restricted or only a derived summary is available, say so plainly rather than
guessing at what the full record might contain.

Always ground your answer in at least one tool call before answering a question about company
decisions, goals, or priorities — do not answer from assumption. Keep answers concise (2-4
sentences) and cite what you retrieved by name so the person can see where the answer came from."""


def _run_tool(persona: Persona, name: str, tool_input: dict):
    fn = TOOL_IMPLEMENTATIONS.get(name)
    if not fn:
        return {"error": f"Unknown tool '{name}'"}
    # `persona` is injected here, server-side — Claude's tool_input never
    # contains an identity field for any of these tools (see tools.py schemas).
    if name in ("search_decisions", "search_goals"):
        return fn(persona, tool_input.get("query", ""))
    if name == "get_decision":
        return fn(persona, tool_input.get("decision_id", ""))
    if name == "get_goal_chain":
        return fn(persona, tool_input.get("goal_id", ""))
    if name == "get_my_context":
        return fn(persona)
    return {"error": f"No dispatch defined for tool '{name}'"}


def run_agent_turn(persona: Persona, user_message: str, history: list[dict] | None = None):
    """
    Returns {"answer": str, "tool_calls": [{"name": str, "input": dict, "result": Any}, ...]}
    `history` is a list of prior {"role": "user"|"assistant", "content": ...} turns in the
    Anthropic Messages format (already interleaved with any tool_use/tool_result blocks
    from earlier turns, if you're persisting full turns — simplest is to only persist the
    final text of each past turn and re-ground with fresh tool calls each time, which is
    what this reference implementation does).
    """
    client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    messages = list(history or [])
    messages.append({"role": "user", "content": user_message})

    tool_call_log = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=_system_prompt(persona),
            tools=TOOL_SCHEMAS,
            tool_choice={"type": "auto"},
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return {"answer": final_text, "tool_calls": tool_call_log}

        # Claude wants to call one or more tools. Append its turn, execute each
        # tool call server-side, and append the results as a user turn of
        # tool_result blocks before looping back.
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = _run_tool(persona, block.name, block.input)
            tool_call_log.append({"name": block.name, "input": block.input, "result": result})
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })
        messages.append({"role": "user", "content": tool_results})

    return {
        "answer": "I wasn't able to settle on an answer within the tool-call budget for this turn — try a narrower question.",
        "tool_calls": tool_call_log,
    }
