"""
Derived-context authoring — a PRIVILEGED, OFFLINE process, deliberately kept
separate from agent.py.

This is the piece that implements brief section 6 ("context rather than raw
access") correctly. The wrong way to build this: when a low-clearance user
asks about a confidential decision, ask Claude to look at the raw decision
and generate a safe-for-them summary on the spot. That means the raw content
enters a live request in response to a lower-clearance user's action, which
is exactly the exposure you're trying to prevent — and it means the summary's
wording depends on live model behavior, which is inconsistent and can be
prompt-engineered around by a creative question.

The right way: generate the derived summary ONCE, at decision-approval time,
as a privileged internal step. A human (typically the decision owner) reviews
and can edit it before it's stored as its own object with its OWN, separately-
set visibility level. Every subsequent query at any clearance level just reads
that pre-approved object — nothing is generated live, and nothing raw is ever
in a lower-clearance request.

This module is meant to be run by an internal tool or admin workflow, invoked
by someone with clearance to see the raw decision — never exposed as an
end-user-facing endpoint.
"""

import os
from anthropic import Anthropic
from data import Decision

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")

DRAFT_SCHEMA = {
    "name": "submit_derived_summary",
    "description": "Submit a draft derived-context summary for a restricted decision.",
    "input_schema": {
        "type": "object",
        "properties": {
            "derived_summary": {
                "type": "string",
                "description": (
                    "1-2 sentences describing the ORGANIZATIONAL CONSEQUENCE of this decision — "
                    "what changes for other people's work as a result — without restating the "
                    "confidential reasoning, financial detail, or names of parties involved."
                ),
            },
            "rationale_for_reviewer": {
                "type": "string",
                "description": "One sentence to the human reviewer explaining what was deliberately left out and why.",
            },
        },
        "required": ["derived_summary", "rationale_for_reviewer"],
    },
}


def draft_derived_summary(decision: Decision) -> dict:
    """
    Called by a privileged internal workflow (e.g. triggered when a decision's
    owner marks it 'approved') with the FULL raw decision. Returns a draft for
    a human to review — this function does not write anything to the data
    store itself. The human-approved final text is what gets saved as
    `Decision.derived`, with its own visibility level set by policy (often
    company-wide, sometimes department-only), independent of the raw record's
    visibility.
    """
    client = Anthropic()

    raw = (
        f"Title: {decision.title}\n"
        f"Why:\n" + "\n".join(f"- {w}" for w in decision.why) + "\n"
        f"Alternatives considered:\n" + "\n".join(f"- {a}" for a in decision.alternatives) + "\n"
        f"Result: {decision.result}"
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=(
            "You draft the DERIVED, lower-clearance-safe version of a confidential company "
            "decision. Your output will be shown to a human reviewer before anyone else ever "
            "sees it — draft conservatively. Describe only the downstream consequence for other "
            "people's work; never restate confidential financial figures, named third parties, "
            "or the detailed reasoning behind the decision."
        ),
        tools=[DRAFT_SCHEMA],
        tool_choice={"type": "tool", "name": "submit_derived_summary"},
        messages=[{"role": "user", "content": raw}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_derived_summary":
            return block.input

    raise RuntimeError("Claude did not return the expected submit_derived_summary tool call.")
