r"""
Stage 8 (FINALE): confidence-based routing + human-in-the-loop.

The full Stage 7 pipeline runs (fetch -> 2 perspectives -> synthesize ->
fan-out one verifier per contested claim -> aggregate). THEN a conditional
edge inspects the confidence of every verdict and decides who finishes the job:

  aggregate --(every claim >= threshold)-------------> finalize_report -> END
            --(any claim <  threshold)--> human_review --> finalize_report -> END
                                             (interrupt: pauses for a person)

Only the LOW-confidence claims escalate to a human; confident ones finalize
automatically. This is the payoff of the whole series -- it reuses:
  - Stage 5 perspective subgraph
  - Stage 6 verifier agent loop
  - Stage 7 Send fan-out + reducer
  - Stage 3 interrupt() / resume  (which is why a checkpointer is needed)

Run it in TWO steps (like Stage 3), because interrupt() pauses the graph:

  1) Start (runs the whole pipeline, then pauses IF a review is needed):
       .\.venv\Scripts\python.exe stage8_finale.py

  2) Resume with your review decision (same thread, a new launch):
       .\.venv\Scripts\python.exe stage8_finale.py --resume "approved"
     or override a verdict in plain text:
       .\.venv\Scripts\python.exe stage8_finale.py --resume "Claim 2 is actually Refuted."

  To start over from scratch, delete the db file:  checkpoints_stage8.sqlite
"""

import operator
import sys
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command          # from Stage 3

# Reuse the ENTIRE Stage 7 pipeline (which itself reuses Stages 5 and 6).
# We only bolt a new tail (the confidence gate) onto the end.
from stage7_fanout import (
    fetch_articles,
    perspective_a_node,
    perspective_b_node,
    synthesize,
    dispatch_verification,
    verify_claim,
    aggregate,
)

DB_FILE = "checkpoints_stage8.sqlite"
THREAD_ID = "verify-1"

# Claims scoring at or above this are trusted and finalized automatically;
# anything below escalates to a human.
#   - raise toward 1.0  -> more claims look "shaky" -> forces the human path
#   - lower toward 0.0  -> everything auto-finalizes (never pauses)
CONFIDENCE_THRESHOLD = 0.75


# ===========================================================================
# STATE  — Stage 7's state PLUS the human's decision
# ===========================================================================
class State(TypedDict):
    topic: str
    articles: list[str]
    perspective_a: str
    perspective_b: str
    contested_claims: list[str]
    verifications: Annotated[list[dict], operator.add]   # reducer (Stage 7)
    report: str
    human_decision: str                                  # NEW IN STAGE 8


# ===========================================================================
# NEW NODES + the confidence router
# ===========================================================================
def _low_confidence(verifications: list[dict]) -> list[dict]:
    """The claims that fell below our trust threshold."""
    return [v for v in verifications if v["confidence"] < CONFIDENCE_THRESHOLD]


def route_by_confidence(state: State) -> str:
    """Conditional edge: any shaky claim -> human_review; otherwise finalize."""
    shaky = _low_confidence(state.get("verifications", []))
    if shaky:
        print(f"[route] {len(shaky)} claim(s) below {CONFIDENCE_THRESHOLD} -> human review")
        return "human_review"
    print(f"[route] all claims >= {CONFIDENCE_THRESHOLD} -> auto-finalize")
    return "finalize_report"


def human_review(state: State) -> dict:
    """PAUSE here and show the human ONLY the low-confidence claims.

    interrupt() freezes the graph and hands this payload back to the caller.
    On resume, this whole function re-runs and interrupt() RETURNS the human's
    answer instead of pausing (same behaviour you learned in Stage 3).
    """
    shaky = _low_confidence(state.get("verifications", []))
    decision = interrupt({
        "message": (
            f"{len(shaky)} claim(s) scored below the {CONFIDENCE_THRESHOLD} "
            f"confidence threshold and need your review."
        ),
        "low_confidence_claims": [
            {"claim": v["claim"], "verdict": v["verdict"],
             "confidence": v["confidence"], "reasoning": v["reasoning"]}
            for v in shaky
        ],
    })
    # Everything below runs ONLY after you resume:
    print(f"[human_review] resumed with decision: {decision!r}")
    return {"human_decision": decision}


def finalize_report(state: State) -> dict:
    """Build the final report, noting any human decision that came in."""
    verifications = state.get("verifications", [])
    decision = state.get("human_decision", "")
    lines = []
    for v in verifications:
        flag = "" if v["confidence"] >= CONFIDENCE_THRESHOLD else "   (reviewed)"
        lines.append(f"- [{v['verdict']} | confidence {v['confidence']:.2f}] {v['claim']}{flag}")
    report = f"=== FINAL report: {state['topic']!r} ===\n" + "\n".join(lines)
    if decision:
        report += f"\n\nHUMAN REVIEW NOTE: {decision}"
    print("[finalize_report] final report ready")
    return {"report": report}


# ===========================================================================
# BUILD  — Stage 7 wiring, plus the confidence gate on the end
# ===========================================================================
builder = StateGraph(State)
builder.add_node("fetch_articles", fetch_articles)
builder.add_node("perspective_a", perspective_a_node)
builder.add_node("perspective_b", perspective_b_node)
builder.add_node("synthesize", synthesize)
builder.add_node("verify_claim", verify_claim)
builder.add_node("aggregate", aggregate)
builder.add_node("human_review", human_review)
builder.add_node("finalize_report", finalize_report)

builder.add_edge(START, "fetch_articles")
builder.add_edge("fetch_articles", "perspective_a")
builder.add_edge("fetch_articles", "perspective_b")
builder.add_edge("perspective_a", "synthesize")
builder.add_edge("perspective_b", "synthesize")
builder.add_conditional_edges("synthesize", dispatch_verification, ["verify_claim", "aggregate"])
builder.add_edge("verify_claim", "aggregate")

# NEW: the confidence gate replaces Stage 7's "aggregate -> END".
builder.add_conditional_edges(
    "aggregate",
    route_by_confidence,
    {"human_review": "human_review", "finalize_report": "finalize_report"},
)
builder.add_edge("human_review", "finalize_report")
builder.add_edge("finalize_report", END)

# For Studio: compile WITHOUT a checkpointer (the dev server supplies one).
graph = builder.compile()


# ===========================================================================
# RUN: start OR resume  (needs a checkpointer, because of interrupt())
# ===========================================================================
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Add it to your .env file.")
        raise SystemExit(1)

    with SqliteSaver.from_conn_string(DB_FILE) as checkpointer:
        app = builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": THREAD_ID}}

        resuming = len(sys.argv) > 1 and sys.argv[1] == "--resume"

        if resuming:
            answer = sys.argv[2] if len(sys.argv) > 2 else "approved"
            print(f"=== RESUMING thread {THREAD_ID!r} with: {answer!r} ===")
            result = app.invoke(Command(resume=answer), config)
            print("\n" + result["report"])
        else:
            print(f"=== STARTING thread {THREAD_ID!r} ===")
            result = app.invoke({"topic": "the economy"}, config)

            if "__interrupt__" in result:
                payload = result["__interrupt__"][0].value
                print("\n*** GRAPH PAUSED for human review ***")
                print(payload["message"])
                for i, c in enumerate(payload["low_confidence_claims"], 1):
                    print(f"  {i}. [{c['verdict']} | conf {c['confidence']:.2f}] {c['claim']}")
                    print(f"     reason: {c['reasoning']}")
                print("\nThe paused state is SAVED on disk. Resume with:")
                print('    .\\.venv\\Scripts\\python.exe stage8_finale.py --resume "approved"')
            else:
                print("\n(all claims were confident -- no human review needed)")
                print(result["report"])
