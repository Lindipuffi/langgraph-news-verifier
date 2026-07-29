r"""
Stage 7: Dynamic fan-out with the Send API (map-reduce).

synthesize finds however many contested claims exist, then we spawn ONE verifier
per claim IN PARALLEL (the Send "map"), and gather all results back into one list
via a REDUCER (the "reduce").

  fetch -> perspective_a ┐
                         ├-> synthesize -> [Send per claim] -> verify_claim xN ┐
        -> perspective_b ┘                                                     ├-> aggregate -> END
                                                                              ┘

Reuses the Stage 5 perspective subgraph and the Stage 6 verifier subgraph.

Run it:
  .\.venv\Scripts\python.exe stage7_fanout.py
"""

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send                       # NEW IN STAGE 7

# Reuse what we already built:
from stage5_subgraphs import perspective_subgraph, LENS_A, LENS_B
from stage6_verifier import verifier

MODEL = "claude-haiku-4-5-20251001"


# ===========================================================================
# STATE  — note the REDUCER on `verifications`
# ===========================================================================
class State(TypedDict):
    topic: str
    articles: list[str]
    perspective_a: str
    perspective_b: str
    contested_claims: list[str]
    # Annotated[..., operator.add] = "when multiple nodes write here, CONCATENATE
    # the lists instead of overwriting." This is what lets N parallel verifiers
    # each contribute one result to the same list.
    verifications: Annotated[list[dict], operator.add]
    report: str


# ===========================================================================
# LLM for extracting contested claims (lazy, injectable for tests)
# ===========================================================================
class ContestedClaims(BaseModel):
    claims: list[str] = Field(description="specific, checkable factual claims the two views disagree on")


_claims_llm = None


def get_claims_llm():
    global _claims_llm
    if _claims_llm is None:
        from langchain_anthropic import ChatAnthropic
        _claims_llm = ChatAnthropic(model=MODEL, temperature=0).with_structured_output(ContestedClaims)
    return _claims_llm


# ===========================================================================
# NODES
# ===========================================================================
def fetch_articles(state: State) -> dict:
    topic = state["topic"]
    print(f"[fetch_articles] fetching articles about: {topic!r}")
    return {"articles": [
        f"Outlet A: {topic} — GDP grew 3% last quarter, unemployment at record lows.",
        f"Outlet B: {topic} — inflation still high, analysts warn the growth may not last.",
        f"Outlet C: {topic} — consumer spending up, but household debt is rising fast.",
    ]}


def perspective_a_node(state: State) -> dict:
    print("[perspective_a] running subgraph (optimistic lens)")
    result = perspective_subgraph.invoke({"articles": state["articles"], "lens": LENS_A})
    return {"perspective_a": result["view"]}


def perspective_b_node(state: State) -> dict:
    print("[perspective_b] running subgraph (skeptical lens)")
    result = perspective_subgraph.invoke({"articles": state["articles"], "lens": LENS_B})
    return {"perspective_b": result["view"]}


def synthesize(state: State) -> dict:
    """Extract the specific claims the two perspectives DISAGREE on."""
    print("[synthesize] extracting contested claims")
    prompt = (
        f"Two analysts reviewed the same news articles.\n\n"
        f"OPTIMIST SAID:\n{state['perspective_a']}\n\n"
        f"SKEPTIC SAID:\n{state['perspective_b']}\n\n"
        f"List the 2-4 specific, checkable FACTUAL claims they most disagree about. "
        f"Each should be a single verifiable statement."
    )
    result = get_claims_llm().invoke(prompt)
    print(f"[synthesize] found {len(result.claims)} contested claim(s)")
    return {"contested_claims": result.claims}


# --- the MAP step: one Send (=> one verify_claim instance) per claim -------
def dispatch_verification(state: State):
    claims = state["contested_claims"]
    if not claims:
        return "aggregate"        # nothing to verify -> skip straight to aggregate
    print(f"[dispatch] fanning out {len(claims)} verifier(s) in parallel")
    # Each Send launches ONE verify_claim, carrying just that claim as its state.
    return [Send("verify_claim", {"claim": c}) for c in claims]


def verify_claim(state: dict) -> dict:
    """Runs for EACH claim in parallel. Its input is one Send's payload."""
    claim = state["claim"]
    print(f"[verify_claim] verifying: {claim!r}")
    result = verifier.invoke({"claim": claim, "evidence": [], "search_count": 0})
    # Return a ONE-item list; the reducer concatenates all of these together.
    return {"verifications": [{
        "claim": claim,
        "verdict": result["verdict"],
        "confidence": result["confidence"],
        "reasoning": result["reasoning"],
    }]}


def aggregate(state: State) -> dict:
    """The REDUCE step: all verifier results are now gathered in one list."""
    verifications = state.get("verifications", [])
    print(f"[aggregate] joining {len(verifications)} verification result(s)")
    lines = [
        f"- [{v['verdict']} | confidence {v['confidence']:.2f}] {v['claim']}"
        for v in verifications
    ]
    report = f"=== Verification report: {state['topic']!r} ===\n" + "\n".join(lines)
    return {"report": report}


# ===========================================================================
# BUILD
# ===========================================================================
builder = StateGraph(State)
builder.add_node("fetch_articles", fetch_articles)
builder.add_node("perspective_a", perspective_a_node)
builder.add_node("perspective_b", perspective_b_node)
builder.add_node("synthesize", synthesize)
builder.add_node("verify_claim", verify_claim)
builder.add_node("aggregate", aggregate)

builder.add_edge(START, "fetch_articles")
builder.add_edge("fetch_articles", "perspective_a")
builder.add_edge("fetch_articles", "perspective_b")
builder.add_edge("perspective_a", "synthesize")
builder.add_edge("perspective_b", "synthesize")

# The Send fan-out. The list is the set of nodes this edge can target
# (needed so LangGraph/Studio can draw the graph).
builder.add_conditional_edges("synthesize", dispatch_verification, ["verify_claim", "aggregate"])

builder.add_edge("verify_claim", "aggregate")   # fan-in: aggregate waits for ALL verifiers
builder.add_edge("aggregate", END)

graph = builder.compile()


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Add it to your .env file.")
        raise SystemExit(1)

    result = graph.invoke({"topic": "the economy"})
    print("\n" + result["report"])
