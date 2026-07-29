r"""
Stage 9: Make the human review REAL — structured overrides.

Stage 8 paused for a human, but "approved" was just a note; the agent's verdicts
never changed. Here the human's decision is STRUCTURED DATA that actually
rewrites the verdicts:

  - resume with 'approved'            -> accept every shaky verdict as-is
  - resume with {"1": "Refuted"}      -> override claim #1's verdict to Refuted
  - resume with {"1": "Refuted", "3": "Mixed"} -> override several at once

This is the mirror image of Stage 6: there we forced the LLM to answer in a fixed
shape so code could act on it; here we do the same to the HUMAN.

The graph shape is identical to Stage 8 (same confidence gate). Only what happens
INSIDE human_review and finalize_report changes.

Run it in TWO steps (interrupt still pauses the graph):

  1) Start:
       .\.venv\Scripts\python.exe stage9_human_override.py

  2) Resume — approve everything as-is:
       .\.venv\Scripts\python.exe stage9_human_override.py --resume "approved"
     or override specific claims by their number (quote-free INDEX=VERDICT form —
     best on the Windows command line, no JSON quoting to get mangled):
       .\.venv\Scripts\python.exe stage9_human_override.py --resume "1=Refuted"
       .\.venv\Scripts\python.exe stage9_human_override.py --resume "1=Refuted,3=Mixed"

  To start over, delete the db file:  checkpoints_stage9.sqlite
"""

import json
import operator
import sys
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command

# Reuse the whole pipeline (Stages 5-7) ...
from stage7_fanout import (
    fetch_articles,
    perspective_a_node,
    perspective_b_node,
    synthesize,
    dispatch_verification,
    verify_claim,
    aggregate,
)
# ... and the confidence gate we built in Stage 8.
from stage8_finale import CONFIDENCE_THRESHOLD, route_by_confidence

DB_FILE = "checkpoints_stage9.sqlite"
THREAD_ID = "verify-1"


# ===========================================================================
# STATE  — Stage 8's fields, plus the human's structured decision + the merged result
# ===========================================================================
class State(TypedDict):
    topic: str
    articles: list[str]
    perspective_a: str
    perspective_b: str
    contested_claims: list[str]
    verifications: Annotated[list[dict], operator.add]   # reducer: writes APPEND here
    overrides: dict                                       # NEW: {claim_index: new_verdict}
    reviewed: bool                                        # NEW: did a human look at it?
    final_verifications: list[dict]                       # NEW: post-review list (NO reducer)
    report: str


# ===========================================================================
# Turn whatever the human sent into a clean {int: str} override map
# ===========================================================================
def _parse_review(raw) -> dict:
    """Turn the human's answer into a map of claim-index -> corrected verdict.

    Accepts several forms so it works everywhere:
      - a dict, e.g. {"1": "Refuted"}          (Studio / Command(resume=...))
      - a JSON string, e.g. '{"1": "Refuted"}' (Studio's resume box)
      - PLAIN PAIRS, e.g. '1=Refuted' or '1=Refuted,3=Mixed'
            ^-- use THIS on the Windows command line: no quotes/braces to mangle
      - a word like 'approved' (or empty)      -> {} (approve all as-is)
    """
    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in raw.items()}

    s = str(raw).strip()
    if not s or s.lower() in ("approve", "approved", "ok", "yes"):
        return {}

    if s.startswith("{"):                       # JSON object (works in Studio)
        return {int(k): str(v) for k, v in json.loads(s).items()}

    # Plain "index=verdict[,index=verdict]" form — quote-free, PowerShell-safe.
    out = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        sep = "=" if "=" in pair else (":" if ":" in pair else None)
        if sep is None:
            raise ValueError(f"can't parse {pair!r}; use INDEX=VERDICT, e.g. 1=Refuted")
        k, v = pair.split(sep, 1)
        out[int(k.strip())] = v.strip()
    return out


# ===========================================================================
# NODES that changed
# ===========================================================================
def human_review(state: State) -> dict:
    """Pause and show the shaky claims WITH their index numbers, so the human
    can name which ones to override."""
    verifications = state.get("verifications", [])
    shaky = [(i, v) for i, v in enumerate(verifications)
             if v["confidence"] < CONFIDENCE_THRESHOLD]

    raw = interrupt({
        "message": (
            f"{len(shaky)} claim(s) scored below {CONFIDENCE_THRESHOLD}. "
            "Resume with 'approved' to accept all as-is, OR name overrides as "
            "INDEX=VERDICT, e.g. '1=Refuted' or '1=Refuted,3=Mixed'. "
            '(In Studio you can also use JSON: {"1": "Refuted"}.)'
        ),
        "low_confidence_claims": [
            {"index": i, "claim": v["claim"], "verdict": v["verdict"],
             "confidence": v["confidence"], "reasoning": v["reasoning"]}
            for i, v in shaky
        ],
    })
    # runs only AFTER resume:
    overrides = _parse_review(raw)
    print(f"[human_review] resumed; {len(overrides)} override(s): {overrides or 'none'}")
    return {"overrides": overrides, "reviewed": True}


def finalize_report(state: State) -> dict:
    """Apply the human's decision to each verdict, then build the report.

    IMPORTANT: we write the corrected list to a FRESH key (final_verifications),
    NOT back into `verifications` — that field has the operator.add reducer, so
    writing to it would APPEND (duplicate the list) instead of replacing it.
    """
    verifications = state.get("verifications", [])
    overrides = state.get("overrides", {})
    reviewed = state.get("reviewed", False)

    final = []
    for i, v in enumerate(verifications):
        entry = dict(v)                                   # copy so we don't mutate the original
        if i in overrides:
            entry["verdict"] = overrides[i]               # <-- the human's verdict WINS
            entry["review"] = "corrected by human"
        elif reviewed and v["confidence"] < CONFIDENCE_THRESHOLD:
            entry["review"] = "approved by human"
        else:
            entry["review"] = ""                          # auto-finalized (was confident)
        final.append(entry)

    lines = []
    for entry in final:
        tag = f"   ({entry['review']})" if entry["review"] else ""
        lines.append(
            f"- [{entry['verdict']} | confidence {entry['confidence']:.2f}] {entry['claim']}{tag}"
        )
    report = f"=== FINAL report: {state['topic']!r} ===\n" + "\n".join(lines)
    print(f"[finalize_report] applied {len(overrides)} override(s)")
    return {"report": report, "final_verifications": final}


# ===========================================================================
# BUILD  — identical wiring to Stage 8; only human_review/finalize_report differ
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
builder.add_conditional_edges(
    "aggregate",
    route_by_confidence,                                  # reused from Stage 8
    {"human_review": "human_review", "finalize_report": "finalize_report"},
)
builder.add_edge("human_review", "finalize_report")
builder.add_edge("finalize_report", END)

# For Studio: compile WITHOUT a checkpointer (the dev server supplies one).
graph = builder.compile()


# ===========================================================================
# RUN: start OR resume
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
                for c in payload["low_confidence_claims"]:
                    print(f"  #{c['index']}  [{c['verdict']} | conf {c['confidence']:.2f}] {c['claim']}")
                    print(f"        reason: {c['reasoning']}")
                print("\nResume with e.g.:")
                print('    .\\.venv\\Scripts\\python.exe stage9_human_override.py --resume "approved"')
                print('    .\\.venv\\Scripts\\python.exe stage9_human_override.py --resume "1=Refuted"')
            else:
                print("\n(all claims were confident -- no human review needed)")
                print(result["report"])
