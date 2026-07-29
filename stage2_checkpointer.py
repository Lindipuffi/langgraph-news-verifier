"""
Stage 2: Add a checkpointer (persistence).

Same graph as Stage 1, with ONE new idea: a checkpointer that saves a snapshot
of the state after every step into a SQLite file. Because state is now saved,
we can read the full step-by-step history back out.

Everything marked  # NEW IN STAGE 2  is what changed vs. stage1_basics.py.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver   # NEW IN STAGE 2

DB_FILE = "checkpoints.sqlite"   # NEW IN STAGE 2: the database file on disk


# --- State + nodes: identical in spirit to Stage 1 -------------------------
class State(TypedDict):
    topic: str
    articles: list[str]
    report: str


def fetch_articles(state: State) -> dict:
    topic = state["topic"]
    print(f"[fetch_articles] fetching articles about: {topic!r}")
    fake_articles = [
        f"Outlet A reports on {topic}: the economy grew 3%.",
        f"Outlet B reports on {topic}: the economy shrank 1%.",
    ]
    return {"articles": fake_articles}


def analyze(state: State) -> dict:
    n = len(state["articles"])
    print(f"[analyze] analyzing {n} articles")
    return {"report": f"Report on {state['topic']!r}: analyzed {n} articles. (placeholder)"}


def insufficient(state: State) -> dict:
    print("[insufficient] not enough articles to analyze")
    return {"report": f"Could not verify {state['topic']!r}: too few articles found."}


def route_after_fetch(state: State) -> str:
    return "analyze" if len(state["articles"]) >= 2 else "insufficient"


# --- Build the graph blueprint (same as Stage 1) ---------------------------
builder = StateGraph(State)
builder.add_node("fetch_articles", fetch_articles)
builder.add_node("analyze", analyze)
builder.add_node("insufficient", insufficient)
builder.add_edge(START, "fetch_articles")
builder.add_conditional_edges(
    "fetch_articles",
    route_after_fetch,
    {"analyze": "analyze", "insufficient": "insufficient"},
)
builder.add_edge("analyze", END)
builder.add_edge("insufficient", END)


if __name__ == "__main__":
    # NEW IN STAGE 2 -------------------------------------------------------
    # SqliteSaver is used as a context manager (the `with` block). It opens
    # the database file, and closes it cleanly when the block ends.
    with SqliteSaver.from_conn_string(DB_FILE) as checkpointer:

        # Compile WITH the checkpointer attached. This one argument is the
        # whole difference: now every step gets saved automatically.
        graph = builder.compile(checkpointer=checkpointer)

        # Because a checkpointer is attached, we MUST say which thread this
        # run belongs to. thread_id is any string we choose.
        config = {"configurable": {"thread_id": "run-1"}}

        print("=== running the graph (thread_id='run-1') ===")
        final_state = graph.invoke({"topic": "the economy"}, config)

        print("\n=== FINAL STATE (via return value) ===")
        for key, value in final_state.items():
            print(f"{key}: {value}")

        # get_state(): the LATEST checkpoint for this thread.
        print("\n=== get_state() -> latest checkpoint ===")
        snapshot = graph.get_state(config)
        print("values:", snapshot.values)
        # .next tells you which node would run next. Empty () means "finished".
        print("next node(s):", snapshot.next)

        # get_state_history(): EVERY checkpoint for this thread, newest first.
        # This is the breadcrumb trail — one entry per step the graph took.
        print("\n=== get_state_history() -> the full breadcrumb trail ===")
        for i, snap in enumerate(graph.get_state_history(config)):
            next_node = snap.next if snap.next else "(done)"
            print(f"  checkpoint #{i}: next={next_node}  articles={len(snap.values.get('articles', []))}  report={snap.values.get('report', '(none yet)')!r}")

    print(f"\nDone. The state above now lives in the file: {DB_FILE}")
    print("Run  stage2_inspect.py  next to read it back WITHOUT running the graph.")
