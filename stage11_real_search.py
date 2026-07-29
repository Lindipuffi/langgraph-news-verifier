r"""
Stage 11: GO LIVE (part 2) — real search for the verifier. THE PIPELINE IS NOW
FULLY LIVE, end to end, on real data.

Stage 10 made ARTICLE fetching real. But when a verifier agent went to CHECK a
claim, it still hit the canned get_search_result stub from Stage 6. Stage 11
points that search tool at the real web (Tavily) too.

The graph wiring is IDENTICAL to Stage 10 — we reuse its `builder` unchanged.
The ONLY new thing is one line: we replace stage6's get_search_result with a
real Tavily-backed search. Because the verifier's `tools` node looks that name
up in the stage6 module at call time, this single swap upgrades every verifier
instance at once. (Same dependency-injection idea as the lazy _llm/_decider
clients — here applied to swap out a whole tool.)

Setup: same as Stage 10 (needs ANTHROPIC_API_KEY and TAVILY_API_KEY in .env).

Run it in TWO steps:
  1) Start (optionally pass a topic):
       .\.venv\Scripts\python.exe stage11_real_search.py
       .\.venv\Scripts\python.exe stage11_real_search.py "UK housing market 2026"
  2) Resume — approve all, or override by index (quote-free form):
       .\.venv\Scripts\python.exe stage11_real_search.py --resume "approved"
       .\.venv\Scripts\python.exe stage11_real_search.py --resume "0=Refuted,2=Mixed"

  To start over, delete the db file:  checkpoints_stage11.sqlite
"""

import os
import sys
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

import stage6_verifier                                  # we will swap its search tool
from stage10_real_articles import get_tavily, builder   # reuse the Tavily client + graph

DB_FILE = "checkpoints_stage11.sqlite"
THREAD_FILE = ".stage11_thread"     # remembers the last run's thread id, so --resume can find it


# ===========================================================================
# THE ONE NEW THING: a real search tool for the verifier
# ===========================================================================
def tavily_search(query: str) -> str:
    """Real web search the verifier uses to gather evidence for a claim.

    Signature matches the Stage 6 stub exactly (query -> evidence string), which
    is why we can drop it straight in.
    """
    print(f"    [tavily_search] {query!r}")
    resp = get_tavily().search(
        query=query,
        search_depth="advanced",
        max_results=3,
        include_answer=True,        # Tavily's synthesized answer = concise evidence
    )
    parts = []
    if resp.get("answer"):
        parts.append(f"Summary: {resp['answer']}")
    for r in resp.get("results", []):
        parts.append(f"{r.get('title', '')}: {r.get('content', '').strip()} "
                     f"({r.get('url', '')})")
    return " || ".join(parts) if parts else "No results found."


# --- the swap: one line turns the whole verifier live -----------------------
# The verifier's `tools` node calls stage6_verifier.get_search_result at runtime,
# so reassigning the name here upgrades every verifier instance.
stage6_verifier.get_search_result = tavily_search

# Reuse Stage 10's graph EXACTLY. Fetch was already real; now search is too.
graph = builder.compile()


# ===========================================================================
# RUN: start OR resume
# ===========================================================================
if __name__ == "__main__":
    from dotenv import load_dotenv

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv()
    missing = [k for k in ("ANTHROPIC_API_KEY", "TAVILY_API_KEY") if not os.getenv(k)]
    if missing:
        print(f"ERROR: missing key(s) in .env: {', '.join(missing)}")
        raise SystemExit(1)

    with SqliteSaver.from_conn_string(DB_FILE) as checkpointer:
        app = builder.compile(checkpointer=checkpointer)

        resuming = len(sys.argv) > 1 and sys.argv[1] == "--resume"

        if resuming:
            # Resume the most recent start — its thread id was saved on start.
            if os.path.exists(THREAD_FILE):
                with open(THREAD_FILE) as f:
                    thread_id = f.read().strip()
            else:
                thread_id = "verify-1"
            answer = sys.argv[2] if len(sys.argv) > 2 else "approved"
            config = {"configurable": {"thread_id": thread_id}}
            print(f"=== RESUMING thread {thread_id!r} with: {answer!r} ===")
            result = app.invoke(Command(resume=answer), config)
            print("\n" + result["report"])
        else:
            # Each fresh start gets its OWN thread id, so separate topics can
            # never bleed together. (The `verifications` reducer would otherwise
            # ACCUMULATE claims from earlier runs on a shared thread.) We save the
            # id so a following --resume knows which run to continue.
            thread_id = f"verify-{uuid.uuid4().hex[:8]}"
            with open(THREAD_FILE, "w") as f:
                f.write(thread_id)
            topic = sys.argv[1] if len(sys.argv) > 1 else "the economy"
            config = {"configurable": {"thread_id": thread_id}}
            print(f"=== STARTING thread {thread_id!r} on topic {topic!r} ===")
            result = app.invoke({"topic": topic}, config)

            if "__interrupt__" in result:
                payload = result["__interrupt__"][0].value
                print("\n*** GRAPH PAUSED for human review ***")
                print(payload["message"])
                for c in payload["low_confidence_claims"]:
                    print(f"  #{c['index']}  [{c['verdict']} | conf {c['confidence']:.2f}] {c['claim']}")
                    print(f"        reason: {c['reasoning']}")
                print("\nResume with e.g.:")
                print('    .\\.venv\\Scripts\\python.exe stage11_real_search.py --resume "approved"')
                print('    .\\.venv\\Scripts\\python.exe stage11_real_search.py --resume "0=Refuted"')
            else:
                print("\n(all claims were confident -- no human review needed)")
                print(result["report"])
