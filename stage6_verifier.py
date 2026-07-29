r"""
Stage 6: A tool-calling agent LOOP (the ReAct pattern), for ONE hardcoded claim.

The agent (an LLM) repeatedly decides: SEARCH for more evidence, or CONCLUDE.
  START -> agent --(search)--> tools --> agent --(search)--> ...
                 \--(conclude)--------------------------------> END
                 \--(out of retries)--> conclude --> END
A retry counter caps the number of searches so the loop can't run forever.

The search tool is a STUB (canned results) for now. (TODO: swap for a real
search API later — see get_search_result / SEARCH TOOL section.)

Run it:
  .\.venv\Scripts\python.exe stage6_verifier.py
"""

from typing import Literal, TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

MODEL = "claude-haiku-4-5-20251001"
MAX_SEARCHES = 3   # the retry limit — the loop's safety belt


# ===========================================================================
# SEARCH TOOL  (stub for now)
# ===========================================================================
# TODO(real-data): replace this canned function with a real search API call
# (e.g. Tavily). Everything else in this file stays the same.
def get_search_result(query: str) -> str:
    """Fake search: returns canned 'evidence' regardless of the query."""
    q = query.lower()
    if "gdp" in q or "grow" in q or "3%" in q:
        return ("Bureau of Economic Analysis: real GDP rose 3.0% (annualized) last "
                "quarter, revised from an initial 2.8% estimate.")
    if "unemployment" in q or "jobs" in q:
        return ("Bureau of Labor Statistics: unemployment held at 3.9%, near record lows.")
    if "inflation" in q or "price" in q:
        return ("Consumer Price Index rose 3.2% year-over-year, still above the 2% target.")
    return ("Multiple outlets report broadly similar economic figures, but with "
            "differing emphasis; no single authoritative number stands out for this query.")


# ===========================================================================
# STATE + the agent's structured DECISION
# ===========================================================================
class VerifierState(TypedDict):
    claim: str            # input: the one claim to verify
    evidence: list[str]   # accumulated search results (grows each loop)
    search_count: int     # how many searches we've done (the retry counter)
    next_action: str      # "search" or "conclude" (set by the agent)
    pending_query: str    # the query to search, if next_action == "search"
    verdict: str          # output
    confidence: float     # output: 0.0 - 1.0
    reasoning: str         # output: why


class Decision(BaseModel):
    """The shape we force the LLM to answer in, each time it reasons."""
    action: Literal["search", "conclude"] = Field(description="search for more, or conclude now")
    query: str = Field(default="", description="the search query, if action=search")
    verdict: str = Field(default="", description="short judgement, if action=conclude")
    confidence: float = Field(default=0.0, description="0.0-1.0, if action=conclude")
    reasoning: str = Field(default="", description="brief reason for this decision")


# The LLM, forced to answer as a Decision. Created lazily so the file imports
# without an API key (and so tests can swap it out).
_decider = None


def get_decider():
    global _decider
    if _decider is None:
        from langchain_anthropic import ChatAnthropic
        _decider = ChatAnthropic(model=MODEL, temperature=0).with_structured_output(Decision)
    return _decider


# ===========================================================================
# NODES
# ===========================================================================
def agent(state: VerifierState) -> dict:
    """REASON: look at the claim + evidence so far, decide search vs conclude."""
    evidence = state.get("evidence", [])
    count = state.get("search_count", 0)
    evidence_text = "\n".join(f"  {i+1}. {e}" for i, e in enumerate(evidence)) or "  (none yet)"

    prompt = (
        f"You are fact-checking ONE claim. Decide your next action.\n\n"
        f"CLAIM: {state['claim']}\n\n"
        f"EVIDENCE SO FAR ({count} of {MAX_SEARCHES} searches used):\n{evidence_text}\n\n"
        f"If you lack enough reliable evidence, choose action='search' with a focused `query`.\n"
        f"If you have enough (or are out of searches), choose action='conclude' with a "
        f"`verdict` (e.g. Supported / Refuted / Mixed) and a `confidence` from 0.0 to 1.0.\n"
        f"Always give brief `reasoning`."
    )
    decision = get_decider().invoke(prompt)
    print(f"[agent] decision={decision.action!r} reasoning={decision.reasoning!r}")

    if decision.action == "search":
        return {"next_action": "search", "pending_query": decision.query,
                "reasoning": decision.reasoning}
    return {"next_action": "conclude", "verdict": decision.verdict,
            "confidence": decision.confidence, "reasoning": decision.reasoning}


def tools(state: VerifierState) -> dict:
    """ACT + OBSERVE: run the search, add the result to the evidence, count it."""
    query = state["pending_query"]
    result = get_search_result(query)
    print(f"[tools] searched {query!r} -> {result[:60]}...")
    return {
        "evidence": state.get("evidence", []) + [f"Q: {query} | A: {result}"],
        "search_count": state.get("search_count", 0) + 1,
    }


def conclude(state: VerifierState) -> dict:
    """Forced conclusion when the agent ran out of searches."""
    evidence = state.get("evidence", [])
    evidence_text = "\n".join(f"  {i+1}. {e}" for i, e in enumerate(evidence)) or "  (none)"
    prompt = (
        f"You have used all {MAX_SEARCHES} searches and MUST conclude now.\n\n"
        f"CLAIM: {state['claim']}\n\nEVIDENCE:\n{evidence_text}\n\n"
        f"Give action='conclude' with a `verdict`, an honest `confidence` (low if the "
        f"evidence is weak), and brief `reasoning`."
    )
    decision = get_decider().invoke(prompt)
    print(f"[conclude] forced conclusion (verdict={decision.verdict!r})")
    return {"verdict": decision.verdict, "confidence": decision.confidence,
            "reasoning": decision.reasoning}


# ===========================================================================
# ROUTING  (the conditional edge that makes the LOOP)
# ===========================================================================
def route_after_agent(state: VerifierState) -> str:
    if state.get("next_action") == "conclude":
        return "done"                       # agent concluded on its own -> END
    if state.get("search_count", 0) >= MAX_SEARCHES:
        return "conclude"                   # wanted to search, but out of retries
    return "tools"                          # go search, then loop back to agent


# ===========================================================================
# BUILD
# ===========================================================================
builder = StateGraph(VerifierState)
builder.add_node("agent", agent)
builder.add_node("tools", tools)
builder.add_node("conclude", conclude)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {"tools": "tools", "conclude": "conclude", "done": END},
)
builder.add_edge("tools", "agent")          # <-- the loop: tools goes BACK to agent
builder.add_edge("conclude", END)

verifier = builder.compile()


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Add it to your .env file.")
        raise SystemExit(1)

    CLAIM = "The economy grew 3% last quarter, with unemployment at record lows."
    print(f"Verifying claim: {CLAIM!r}\n")

    result = verifier.invoke({"claim": CLAIM, "evidence": [], "search_count": 0})

    print("\n=== RESULT ===")
    print("verdict:   ", result["verdict"])
    print("confidence:", result["confidence"])
    print("reasoning: ", result["reasoning"])
    print(f"\n(searched {result['search_count']} time(s))")
    for e in result["evidence"]:
        print("  -", e)
