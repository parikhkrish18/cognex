"""
Small CLI to sanity-check the live Claude integration once you have
ANTHROPIC_API_KEY set. Runs the flagship demo from the front-end prototype
through the real agent loop: the same acquisition question, asked as the CEO
and then as the intern, so you can see the permission boundary hold up against
an actual model rather than the rule-based engine.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 demo_cli.py
"""

import os
import sys
from data import PERSONAS
from agent import run_agent_turn


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set in this environment.")
        print("Run:  export ANTHROPIC_API_KEY=sk-ant-...   then re-run this script.")
        sys.exit(1)

    question = "Are we planning any acquisitions right now, and why?"

    for persona_id in ("ceo", "intern"):
        persona = PERSONAS[persona_id]
        print(f"\n=== Asking as {persona.name} ({persona.title}, clearance {persona.level}) ===")
        print(f"Q: {question}")
        result = run_agent_turn(persona, question)
        print(f"A: {result['answer']}")
        print("\nTool calls made:")
        for call in result["tool_calls"]:
            r = call["result"]
            if isinstance(r, dict):
                detail = f"access={r.get('access', 'n/a')}"
            elif isinstance(r, list):
                detail = f"{len(r)} result(s): " + ", ".join(
                    f"{item.get('id', item.get('title', '?'))}[{item.get('access', '?')}]"
                    for item in r if isinstance(item, dict)
                )
            else:
                detail = str(r)
            print(f"  - {call['name']}({call['input']}) -> {detail}")


if __name__ == "__main__":
    main()
