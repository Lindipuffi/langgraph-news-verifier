"""
Stage 1: A basic StateGraph.

Goal: see the three core pieces of LangGraph with nothing else in the way —
  1. a STATE schema  (the shared memory that flows through the graph)
  2. NODES           (plain functions that read state and return a patch)
  3. EDGES           (wiring, including ONE conditional edge that decides where to go)

No LLM, no persistence, no loops. Just the skeleton.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# 1. STATE SCHEMA
# ---------------------------------------------------------------------------
# This TypedDict declares every field that can live in our shared state.
# Nodes will read from these fields and return dicts that patch them.
# By DEFAULT, when a node returns a field, it OVERWRITES the old value.
# (We'll change that default with a "reducer" later, when nodes run in parallel.)
class State(TypedDict):
    topic: str            # what the user wants verified news about (input)
    articles: list[str]   # the articles we "fetched"
    report: str           # the final human-readable output


# ---------------------------------------------------------------------------
# 2. NODES  (each is just: state in -> partial-state-update out)
# ---------------------------------------------------------------------------
def fetch_articles(state: State) -> dict:
    """Pretend to fetch articles for the topic. (Real fetching comes later.)"""
    topic = state["topic"]
    print(f"[fetch_articles] fetching articles about: {topic!r}")

    # Fake data for now. Try changing this list to 0 or 1 items to see the
    # conditional edge route differently.
    fake_articles = [
        f"Outlet A reports on {topic}: the economy grew 3%.",
        f"Outlet B reports on {topic}: the economy shrank 1%.",
    ]

    # Return ONLY the field we're changing. LangGraph merges it into state.
    return {"articles": fake_articles}


def analyze(state: State) -> dict:
    """We had enough articles — produce a (placeholder) report."""
    n = len(state["articles"])
    print(f"[analyze] analyzing {n} articles")
    report = f"Report on {state['topic']!r}: analyzed {n} articles. (placeholder)"
    return {"report": report}


def insufficient(state: State) -> dict:
    """Not enough articles — bail out with an explanatory report."""
    print("[insufficient] not enough articles to analyze")
    report = f"Could not verify {state['topic']!r}: too few articles found."
    return {"report": report}


# ---------------------------------------------------------------------------
# 3. ROUTING FUNCTION  (the brain of the conditional edge)
# ---------------------------------------------------------------------------
# A conditional edge's router looks at state and RETURNS THE NAME of the next
# node as a string. It does not change state; it only chooses a direction.
def route_after_fetch(state: State) -> str:
    if len(state["articles"]) >= 2:
        return "analyze"
    return "insufficient"


# ---------------------------------------------------------------------------
# 4. BUILD THE GRAPH
# ---------------------------------------------------------------------------
builder = StateGraph(State)

# Register the nodes. First arg = node name (a string), second = the function.
builder.add_node("fetch_articles", fetch_articles)
builder.add_node("analyze", analyze)
builder.add_node("insufficient", insufficient)

# Entry point: from the special START node, always go to fetch_articles.
builder.add_edge(START, "fetch_articles")

# THE conditional edge: after fetch_articles, run route_after_fetch to decide.
# The third arg maps each possible return value of the router to a real node.
builder.add_conditional_edges(
    "fetch_articles",
    route_after_fetch,
    {
        "analyze": "analyze",
        "insufficient": "insufficient",
    },
)

# Both terminal nodes lead to END.
builder.add_edge("analyze", END)
builder.add_edge("insufficient", END)

# compile() turns the blueprint into a runnable graph.
graph = builder.compile()


# ---------------------------------------------------------------------------
# 5. RUN IT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # invoke() runs the graph start-to-finish and returns the FINAL state.
    # The dict we pass in is the initial state.
    final_state = graph.invoke({"topic": "the economy"})

    print("\n--- FINAL STATE ---")
    for key, value in final_state.items():
        print(f"{key}: {value}")
