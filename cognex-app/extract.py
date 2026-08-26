"""
Complete the Story — structured extraction.

The front-end prototype maps each free-text answer straight into a field
(who -> contributor note, why -> why bullet, risk -> risk bullet) with no
real extraction happening. That's fine for a click-through demo, but the
actual value Claude adds here is turning messy, informal, multi-sentence
answers into clean structured Decision Memory fields — and flagging when an
answer doesn't actually contain what was asked for, so a human can be
prompted to clarify rather than silently writing a low-quality memory record.

This uses tool_choice to FORCE a single structured tool call (the "submit"
tool) rather than free text — a reliable way to get JSON-shaped output from
Claude without parsing prose, and without a partial/malformed response.
"""

import os
from anthropic import Anthropic

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")

SUBMIT_SCHEMA = {
    "name": "submit_decision_draft",
    "description": "Submit the structured Decision Memory draft extracted from the employee's answers.",
    "input_schema": {
        "type": "object",
        "properties": {
            "why": {
                "type": "array",
                "items": {"type": "string"},
                "description": "1-2 concise bullet(s) capturing WHY this work is happening, in third person, suitable for someone else reading it cold.",
            },
            "risks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "1-2 concise bullet(s) capturing the risk(s) the employee flagged.",
            },
            "contributor_note": {
                "type": "string",
                "description": "A short third-person sentence naming who is involved / who owns this, based on the 'who' answer.",
            },
            "suggested_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3-6 lowercase keyword tags for future retrieval, drawn from the work item and answers.",
            },
            "needs_clarification": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Names of any fields (why / risks / contributor_note) where the employee's answer "
                    "didn't actually address the question — e.g. they answered 'not sure' or something "
                    "unrelated. Empty array if all answers were usable."
                ),
            },
        },
        "required": ["why", "risks", "contributor_note", "suggested_tags", "needs_clarification"],
    },
}


def draft_decision_from_story(work_item: str, qa_pairs: list[dict]) -> dict:
    """
    qa_pairs: [{"question": "...", "field": "who"|"why"|"risk", "answer": "..."}, ...]
    Returns the structured draft dict (matching SUBMIT_SCHEMA's input_schema), ready
    to show the employee for confirmation before it's written into Decision Memory.
    Nothing here writes to the data store directly — extraction and persistence are
    kept separate so a human always confirms before company memory changes.
    """
    client = Anthropic()

    transcript = "\n".join(f"Q: {p['question']}\nA: {p['answer']}" for p in qa_pairs)

    response = client.messages.create(
        model=MODEL,
        max_tokens=768,
        system=(
            "You turn an employee's quick answers about their current work into a clean, "
            "structured draft for the company's Decision Memory. Stay strictly grounded in what "
            "they actually said — do not invent detail, names, or numbers that weren't given. "
            "If an answer is evasive or doesn't address the question, say so via needs_clarification "
            "rather than papering over it."
        ),
        tools=[SUBMIT_SCHEMA],
        tool_choice={"type": "tool", "name": "submit_decision_draft"},
        messages=[{
            "role": "user",
            "content": f"Work item: {work_item}\n\n{transcript}\n\nExtract the structured draft.",
        }],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_decision_draft":
            return block.input

    # tool_choice forces a matching call, so reaching here means something upstream
    # changed shape — fail loudly rather than returning a silently empty draft.
    raise RuntimeError("Claude did not return the expected submit_decision_draft tool call.")
