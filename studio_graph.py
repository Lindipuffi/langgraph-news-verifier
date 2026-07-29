"""
Stage 4: the graph, exposed for LangGraph/LangSmith Studio.

Studio's local server needs a module-level variable that IS a compiled graph.
langgraph.json points here:  "news_verifier": "./studio_graph.py:graph"

We reuse the Stage 3 graph (with the interrupt) so we can watch the
human-in-the-loop pause happen visually. NOTE: we compile WITHOUT our own
checkpointer here — the Studio dev server supplies persistence itself.
"""

from stage3_interrupt import builder

# Compile with no checkpointer; the dev server injects one.
graph = builder.compile()
