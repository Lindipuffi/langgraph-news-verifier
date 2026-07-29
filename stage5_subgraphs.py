r"""
Stage 5: Real subgraphs + the first real LLM call.

Each "perspective" is now a real SUBGRAPH (a small graph used as a step inside
the main graph). We build ONE perspective subgraph and REUSE it twice with a
different "lens" (angle), which is a big reason subgraphs exist.

  fetch_articles ->  perspective_a (subgraph)  \
                 ->  perspective_b (subgraph)  /-> synthesize -> END
       (the two perspectives run in parallel, then join at synthesize)

Run it:
  1) Put your Anthropic API key in the .env file:  ANTHROPIC_API_KEY=sk-ant-...
  2) .\.venv\Scripts\python.exe stage5_subgraphs.py
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# The LLM. Created lazily (only when first needed) so the file can be imported
# and its graph structure inspected without an API key present.
# ---------------------------------------------------------------------------
_llm = None


def get_llm():
    global _llm
    if _llm is None:
        from langchain_anthropic import ChatAnthropic
        # Haiku = small, fast, cheap. Fine for a learning project. Swap the
        # model string for a bigger one (e.g. claude-sonnet-5) any time.
        _llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
    return _llm


# ===========================================================================
# THE PERSPECTIVE SUBGRAPH  (built once, reused twice)
# ===========================================================================
# A subgraph has its OWN state schema, separate from the parent graph's.
class PerspectiveState(TypedDict):
    articles: list[str]   # input: the articles to look at
    lens: str             # input: the angle/instruction for this perspective
    article_text: str     # internal: articles formatted into one block
    view: str             # output: this perspective's take


def gather(state: PerspectiveState) -> dict:
    """Plain (no LLM) prep step: format the articles into one text block."""
    block = "\n".join(f"- {a}" for a in state["articles"])
    return {"article_text": block}


def analyze(state: PerspectiveState) -> dict:
    """LLM step: read the articles through this perspective's lens."""
    prompt = (
        f"You are a news analyst with this specific angle: {state['lens']}.\n"
        f"Here are today's article summaries:\n{state['article_text']}\n\n"
        f"In 2-3 sentences, give your take on what the articles say, "
        f"emphasizing your angle. Be concrete."
    )
    reply = get_llm().invoke(prompt)   # <-- the actual call to Claude
    return {"view": reply.content.strip()}


def build_perspective_subgraph():
    """Build and compile ONE perspective subgraph. We'll reuse it for A and B."""
    sub = StateGraph(PerspectiveState)
    sub.add_node("gather", gather)
    sub.add_node("analyze", analyze)
    sub.add_edge(START, "gather")
    sub.add_edge("gather", "analyze")
    sub.add_edge("analyze", END)
    return sub.compile()


perspective_subgraph = build_perspective_subgraph()   # built ONCE


# ===========================================================================
# THE PARENT GRAPH
# ===========================================================================
class State(TypedDict):
    topic: str
    articles: list[str]
    perspective_a: str   # filled by the A wrapper
    perspective_b: str   # filled by the B wrapper
    report: str


# The two different "lenses" — the ONLY difference between A and B.
LENS_A = "an optimistic economist who focuses on positive indicators and growth"
LENS_B = "a cautious skeptic who focuses on risks, downsides, and what's unproven"


def fetch_articles(state: State) -> dict:
    topic = state["topic"]
    print(f"[fetch_articles] fetching articles about: {topic!r}")
    return {"articles": [
        f"Outlet A: {topic} — GDP grew 3% last quarter, unemployment at record lows.",
        f"Outlet B: {topic} — inflation still high, analysts warn the growth may not last.",
        f"Outlet C: {topic} — consumer spending up, but household debt is rising fast.",
    ]}


def perspective_a_node(state: State) -> dict:
    """Wrapper node: run the perspective subgraph with lens A, store in its own field."""
    print("[perspective_a] running subgraph (optimistic lens)")
    result = perspective_subgraph.invoke({"articles": state["articles"], "lens": LENS_A})
    return {"perspective_a": result["view"]}


def perspective_b_node(state: State) -> dict:
    """Wrapper node: SAME subgraph, lens B, different field. Runs in parallel with A."""
    print("[perspective_b] running subgraph (skeptical lens)")
    result = perspective_subgraph.invoke({"articles": state["articles"], "lens": LENS_B})
    return {"perspective_b": result["view"]}


def synthesize(state: State) -> dict:
    """Plain join step. (Extracting the actual CONTESTED claims comes in Stage 7.)"""
    print("[synthesize] combining both perspectives")
    report = (
        f"=== Report on {state['topic']!r} ===\n\n"
        f"OPTIMISTIC VIEW:\n{state['perspective_a']}\n\n"
        f"SKEPTICAL VIEW:\n{state['perspective_b']}\n\n"
        f"(Next stages will extract the specific contested claims and verify them.)"
    )
    return {"report": report}


builder = StateGraph(State)
builder.add_node("fetch_articles", fetch_articles)
builder.add_node("perspective_a", perspective_a_node)
builder.add_node("perspective_b", perspective_b_node)
builder.add_node("synthesize", synthesize)

builder.add_edge(START, "fetch_articles")
# Fan-out: fetch_articles leads to BOTH perspectives (they run in parallel).
builder.add_edge("fetch_articles", "perspective_a")
builder.add_edge("fetch_articles", "perspective_b")
# Fan-in: synthesize waits for BOTH perspectives to finish before running.
builder.add_edge("perspective_a", "synthesize")
builder.add_edge("perspective_b", "synthesize")
builder.add_edge("synthesize", END)

graph = builder.compile()


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()   # read ANTHROPIC_API_KEY (and others) from the .env file
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set.")
        print("Add a line to your .env file:  ANTHROPIC_API_KEY=sk-ant-...")
        raise SystemExit(1)

    result = graph.invoke({"topic": "the economy"})
    print("\n" + result["report"])
