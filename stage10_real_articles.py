r"""
Stage 10: GO LIVE (part 1) — real articles from Tavily.

Everything you built still runs unchanged. The ONLY difference from Stage 9 is
that fetch_articles now searches the real web (via the Tavily API) instead of
returning three hardcoded fake outlets. This is the payoff of the clean stub
boundary: the rest of the pipeline never cared WHERE the articles came from, so
swapping the source touches exactly one node.

  fetch_articles (REAL now) -> perspective_a/b -> synthesize -> verify (fan-out)
      -> aggregate -> confidence gate -> human_review / finalize_report

Setup:
  1) Get a free Tavily key at https://tavily.com and add it to .env:
       TAVILY_API_KEY=tvly-...
  2) (already there) ANTHROPIC_API_KEY=sk-ant-...

Run it in TWO steps (interrupt still pauses for human review):

  1) Start:
       .\.venv\Scripts\python.exe stage10_real_articles.py
  2) Resume — approve all, or override by index (quote-free form):
       .\.venv\Scripts\python.exe stage10_real_articles.py --resume "approved"
       .\.venv\Scripts\python.exe stage10_real_articles.py --resume "0=Refuted,2=Mixed"

  To start over, delete the db file:  checkpoints_stage10.sqlite
"""

import os
import sys

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

# Reuse the whole pipeline (Stages 5-7) ...
from stage7_fanout import (
    perspective_a_node,
    perspective_b_node,
    synthesize,
    dispatch_verification,
    verify_claim,
    aggregate,
)
# ... the confidence gate (Stage 8) ...
from stage8_finale import CONFIDENCE_THRESHOLD, route_by_confidence
# ... and the real human-override review (Stage 9). State comes from there too.
from stage9_human_override import State, human_review, finalize_report

DB_FILE = "checkpoints_stage10.sqlite"
THREAD_ID = "verify-1"

# How many real articles to pull per run. More = richer perspectives but slower
# and more tokens. 4 is a good starting point.
NUM_ARTICLES = 4


# ===========================================================================
# THE ONE NEW THING: a real fetch_articles backed by Tavily
# ===========================================================================
_tavily = None


def get_tavily():
    """Lazy Tavily client, so this file imports without a key (and tests can fake it)."""
    global _tavily
    if _tavily is None:
        from tavily import TavilyClient
        _tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return _tavily


def fetch_articles(state: State) -> dict:
    """Search the real web for recent news about the topic, return clean text."""
    topic = state["topic"]
    print(f"[fetch_articles] searching Tavily for real news about: {topic!r}")
    response = get_tavily().search(
        query=topic,
        topic="news",              # bias toward recent news articles
        search_depth="advanced",   # better-quality extracted content
        max_results=NUM_ARTICLES,
    )
    results = response.get("results", [])
    articles = [
        f"{r.get('title', 'Untitled')} - {r.get('content', '').strip()} "
        f"(source: {r.get('url', '')})"
        for r in results
    ]
    if not articles:
        articles = [f"No recent news found for {topic!r}."]
    print(f"[fetch_articles] got {len(articles)} real article(s)")
    return {"articles": articles}


# ===========================================================================
# BUILD  — identical wiring to Stage 9; only fetch_articles is different
# ===========================================================================
builder = StateGraph(State)
builder.add_node("fetch_articles", fetch_articles)          # <-- the real one
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
    route_by_confidence,
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
    from dotenv import load_dotenv

    # Real news text contains em dashes, smart quotes, accents, etc. Force UTF-8
    # output so printing them can't crash on the Windows console.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    load_dotenv()
    missing = [k for k in ("ANTHROPIC_API_KEY", "TAVILY_API_KEY") if not os.getenv(k)]
    if missing:
        print(f"ERROR: missing key(s) in .env: {', '.join(missing)}")
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
            topic = sys.argv[1] if len(sys.argv) > 1 else "the economy"
            print(f"=== STARTING thread {THREAD_ID!r} on topic {topic!r} ===")
            result = app.invoke({"topic": topic}, config)

            if "__interrupt__" in result:
                payload = result["__interrupt__"][0].value
                print("\n*** GRAPH PAUSED for human review ***")
                print(payload["message"])
                for c in payload["low_confidence_claims"]:
                    print(f"  #{c['index']}  [{c['verdict']} | conf {c['confidence']:.2f}] {c['claim']}")
                    print(f"        reason: {c['reasoning']}")
                print("\nResume with e.g.:")
                print('    .\\.venv\\Scripts\\python.exe stage10_real_articles.py --resume "approved"')
                print('    .\\.venv\\Scripts\\python.exe stage10_real_articles.py --resume "0=Refuted"')
            else:
                print("\n(all claims were confident -- no human review needed)")
                print(result["report"])
