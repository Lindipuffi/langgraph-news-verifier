r"""
Stage 3: Human-in-the-loop with interrupt().

The graph now PAUSES to ask a human to approve (or correct) the draft report
before finalizing. Pausing works because a checkpointer saves the state at the
interrupt, so the program can exit and resume later from the same spot.

Run it in TWO steps:

  1) Start the run (it will pause and exit):
       .\.venv\Scripts\python.exe stage3_interrupt.py

  2) Resume with your answer (same thread, separate program launch):
       .\.venv\Scripts\python.exe stage3_interrupt.py --resume "approve"
     or supply a correction instead of approving:
       .\.venv\Scripts\python.exe stage3_interrupt.py --resume "Growth was about 3 percent."

  To start over from scratch, delete the db file:  checkpoints_stage3.sqlite
"""

import sys
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command   # NEW IN STAGE 3

DB_FILE = "checkpoints_stage3.sqlite"
THREAD_ID = "review-1"


# --- State -----------------------------------------------------------------
class State(TypedDict):
    topic: str
    articles: list[str]
    report: str
    human_decision: str   # NEW IN STAGE 3: what the human told us


# --- Nodes -----------------------------------------------------------------
def fetch_articles(state: State) -> dict:
    topic = state["topic"]
    print(f"[fetch_articles] fetching articles about: {topic!r}")
    return {"articles": [
        f"Outlet A reports on {topic}: the economy grew 3%.",
        f"Outlet B reports on {topic}: the economy shrank 1%.",
    ]}


def analyze(state: State) -> dict:
    n = len(state["articles"])
    print(f"[analyze] analyzing {n} articles")
    return {"report": f"DRAFT: outlets disagree on {state['topic']!r} (grew 3% vs shrank 1%)."}


def insufficient(state: State) -> dict:
    print("[insufficient] not enough articles to analyze")
    return {"report": f"Could not verify {state['topic']!r}: too few articles found."}


def human_review(state: State) -> dict:
    # interrupt() PAUSES the graph here and hands this payload back to the
    # program. On the FIRST pass, execution stops at this line. On RESUME, the
    # whole function re-runs and interrupt() RETURNS the human's answer instead.
    decision = interrupt({
        "question": "Approve this draft report, or type a correction?",
        "draft_report": state["report"],
    })
    # Everything below runs ONLY after you resume:
    print(f"[human_review] resumed with human decision: {decision!r}")
    return {"human_decision": decision}


def finalize(state: State) -> dict:
    decision = state["human_decision"]
    if decision.strip().lower() == "approve":
        final = state["report"].replace("DRAFT: ", "")
        print("[finalize] human approved the draft")
    else:
        final = f"(human-corrected) {decision}"
        print("[finalize] human supplied a correction")
    return {"report": final}


def route_after_fetch(state: State) -> str:
    return "analyze" if len(state["articles"]) >= 2 else "insufficient"


# --- Build the graph -------------------------------------------------------
builder = StateGraph(State)
builder.add_node("fetch_articles", fetch_articles)
builder.add_node("analyze", analyze)
builder.add_node("insufficient", insufficient)
builder.add_node("human_review", human_review)
builder.add_node("finalize", finalize)

builder.add_edge(START, "fetch_articles")
builder.add_conditional_edges(
    "fetch_articles",
    route_after_fetch,
    {"analyze": "analyze", "insufficient": "insufficient"},
)
builder.add_edge("analyze", "human_review")   # analyze -> pause for a human
builder.add_edge("human_review", "finalize")
builder.add_edge("finalize", END)
builder.add_edge("insufficient", END)


# --- Run: start OR resume, depending on command-line argument --------------
if __name__ == "__main__":
    with SqliteSaver.from_conn_string(DB_FILE) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": THREAD_ID}}

        resuming = len(sys.argv) > 1 and sys.argv[1] == "--resume"

        if resuming:
            answer = sys.argv[2] if len(sys.argv) > 2 else "approve"
            print(f"=== RESUMING thread {THREAD_ID!r} with answer: {answer!r} ===")
            result = graph.invoke(Command(resume=answer), config)
            print("\n=== FINISHED ===")
            print("final report:", result["report"])
        else:
            print(f"=== STARTING thread {THREAD_ID!r} ===")
            result = graph.invoke({"topic": "the economy"}, config)

            if "__interrupt__" in result:
                payload = result["__interrupt__"][0].value
                print("\n*** GRAPH PAUSED (interrupt) ***")
                print("It is waiting for a human. It sent this payload:")
                print(f'    question:     {payload["question"]}')
                print(f'    draft_report: {payload["draft_report"]}')
                print("\nThe program is exiting now, but the paused state is SAVED on disk.")
                print("Resume it with either of these:")
                print('    .\\.venv\\Scripts\\python.exe stage3_interrupt.py --resume "approve"')
                print('    .\\.venv\\Scripts\\python.exe stage3_interrupt.py --resume "Growth was about 3 percent."')
            else:
                print("(no interrupt happened) final report:", result["report"])
