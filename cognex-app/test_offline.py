"""
Tests that don't touch the network — no ANTHROPIC_API_KEY needed. These verify
the part that matters most to get right: permission enforcement actually
strips restricted fields rather than just hiding them in a UI layer, and the
tool layer + FastAPI app are wired correctly.

Run with:  python3 test_offline.py
(Plain asserts + a runner, so it works without pytest installed. Use pytest
if you have it: `pytest test_offline.py -v`.)
"""

import json
from data import PERSONAS, DECISIONS
from permissions import serialize_decision_for, decision_view
from tools import search_decisions, search_goals, get_goal_chain, get_my_context, TOOL_SCHEMAS


def test_ceo_sees_raw_acquisition():
    ceo = PERSONAS["ceo"]
    d = next(x for x in DECISIONS if x.id == "d-acquire-northwind")
    out = serialize_decision_for(ceo, d)
    assert out["access"] == "source"
    assert "why" in out and len(out["why"]) > 0
    assert "risks" in out


def test_intern_never_receives_raw_acquisition_fields():
    intern = PERSONAS["intern"]
    d = next(x for x in DECISIONS if x.id == "d-acquire-northwind")
    out = serialize_decision_for(intern, d)
    assert out["access"] == "derived"
    # The core guarantee: these keys must not exist at all, not be present-but-empty.
    for forbidden_key in ("why", "alternatives", "assumptions", "risks", "result"):
        assert forbidden_key not in out, f"leaked '{forbidden_key}' to a viewer without clearance"
    assert "derived_summary" in out
    # Belt-and-suspenders: even JSON-serialized, no raw sentence should appear.
    serialized = json.dumps(out)
    assert "18-month" not in serialized
    assert "Northwind's Gulf-region warehouse" not in serialized


def test_employee_below_germany_derived_level_gets_none():
    # Germany's derived_level is "manager" (3); employee is level 2.
    employee = PERSONAS["employee"]
    d = next(x for x in DECISIONS if x.id == "d-no-germany")
    out = serialize_decision_for(employee, d)
    assert out["access"] == "none"
    assert "derived_summary" not in out
    assert "why" not in out


def test_manager_gets_germany_derived_not_source():
    manager = PERSONAS["manager"]
    d = next(x for x in DECISIONS if x.id == "d-no-germany")
    assert decision_view(manager, d) == "derived"


def test_search_decisions_permission_filtered_for_intern():
    intern = PERSONAS["intern"]
    results = search_decisions(intern, "why do we manufacture in india instead of vietnam")
    assert len(results) >= 1
    india = next(r for r in results if r["id"] == "d-india-vietnam")
    # india-vietnam derived_level is "employee" (2); intern is level 1 -> "none"
    assert india["access"] == "none"
    assert "why" not in india


def test_search_decisions_finds_germany_for_query():
    ceo = PERSONAS["ceo"]
    results = search_decisions(ceo, "why did we decide not to enter germany")
    assert any(r["id"] == "d-no-germany" for r in results)


def test_search_goals_finds_uae():
    # Note: search_goals only matches title words longer than 3 characters (same
    # threshold as the front-end prototype, to keep the two implementations in
    # lockstep) — so short tokens like "uae" alone won't match on their own.
    # "2027" is in the goal's title and is long enough to match.
    results = search_goals(PERSONAS["ceo"], "what's our top goal for 2027")
    assert any(g["id"] == "g-uae" for g in results)


def test_goal_chain_walks_to_root():
    chain = get_goal_chain(PERSONAS["employee"], "g-jordan")
    assert chain[0]["id"] == "g-uae"
    assert chain[-1]["id"] == "g-jordan"


def test_get_my_context_returns_individual_goal_for_employee():
    ctx = get_my_context(PERSONAS["employee"])
    assert ctx["individual_goal"]["id"] == "g-jordan"


def test_get_my_context_none_for_ceo():
    ctx = get_my_context(PERSONAS["ceo"])
    assert ctx["individual_goal"] is None


def test_tool_schemas_are_well_formed():
    names = [t["name"] for t in TOOL_SCHEMAS]
    assert len(names) == len(set(names)), "duplicate tool names"
    for t in TOOL_SCHEMAS:
        assert "description" in t and len(t["description"]) > 10
        assert t["input_schema"]["type"] == "object"
        # no tool schema should accept a caller-supplied identity/persona field
        assert "persona" not in t["input_schema"]["properties"]
        assert "user_id" not in t["input_schema"]["properties"]


def test_fastapi_app_boots_and_filters_via_http():
    from fastapi.testclient import TestClient
    from server import app

    client = TestClient(app)

    r = client.get("/personas")
    assert r.status_code == 200
    assert any(p["id"] == "ceo" for p in r.json())

    r = client.get("/decisions", params={"persona_id": "intern"})
    assert r.status_code == 200
    body = {d["id"]: d for d in r.json()}
    assert body["d-acquire-northwind"]["access"] == "derived"
    assert "why" not in body["d-acquire-northwind"]

    r = client.get("/decisions", params={"persona_id": "ceo"})
    body = {d["id"]: d for d in r.json()}
    assert body["d-acquire-northwind"]["access"] == "source"
    assert "why" in body["d-acquire-northwind"]

    # unknown persona -> 404, not a silent empty result
    r = client.get("/decisions", params={"persona_id": "nope"})
    assert r.status_code == 404


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok   {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERR  {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
