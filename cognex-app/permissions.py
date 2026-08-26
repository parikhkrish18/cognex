"""
Permission enforcement — the single most important file in this reference build.

The rule this module exists to enforce: a restricted field is never SERIALIZED
into anything that reaches Claude for a user without clearance. Not masked, not
redacted-by-instruction — simply absent from the Python dict before it's ever
turned into a tool_result. Claude cannot leak what it was never given.

Contrast this with the tempting-but-wrong approach of handing Claude the full
decision and a system-prompt instruction like "don't reveal this to non-execs."
That's a policy Claude has to *choose* to follow every time, under adversarial
prompting, forever. Filtering in this module happens in plain Python before any
network call to Claude — there's no instruction to work around because the data
was never in the request.
"""

from data import Decision, Persona, LEVEL_LABEL


def accessible(persona: Persona, min_level: int) -> bool:
    return persona.level >= min_level


def decision_view(persona: Persona, decision: Decision) -> str:
    """Returns 'source', 'derived', or 'none' — same three states as the prototype."""
    if accessible(persona, decision.visibility):
        return "source"
    if accessible(persona, decision.derived_level):
        return "derived"
    return "none"


def serialize_decision_for(persona: Persona, decision: Decision) -> dict:
    """
    The enforcement point. Builds the dict that will be JSON-encoded into a
    tool_result and sent to Claude. What's NOT accessible is not a field with
    a null or redacted value — it's simply not a key in the returned dict.
    """
    mode = decision_view(persona, decision)
    base = {
        "id": decision.id,
        "title": decision.title,
        "decided": decision.decided,
        "owner": decision.owner,
        "clearance_required": LEVEL_LABEL[decision.visibility],
        "access": mode,
    }

    if mode == "source":
        base.update({
            "why": decision.why,
            "alternatives": decision.alternatives,
            "assumptions": decision.assumptions,
            "risks": decision.risks,
            "review": decision.review,
            "result": decision.result,
        })
        if decision.contributor:
            base["reported_by_note"] = decision.contributor
    elif mode == "derived":
        base["derived_summary"] = decision.derived
        base["note"] = (
            "The underlying decision record is restricted above this viewer's clearance. "
            "Only the pre-approved derived summary is available — do not imply you have "
            "seen the raw why/alternatives/risks, because you have not been given them."
        )
    else:
        base["note"] = "This record and its derived context both require higher clearance than this viewer has."

    return base
