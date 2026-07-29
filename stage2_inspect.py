"""
Stage 2 (part 2): Read saved checkpoints back WITHOUT running the graph.

This is a SEPARATE program. It never calls invoke(). It just opens the same
database file stage2_checkpointer.py wrote to and reads the history. If this
prints anything, it proves the state truly persisted to disk and outlived the
program that created it.

We attach the checkpointer to the SAME graph blueprint. We need the blueprint
because get_state_history() is a method on a compiled graph — but note we do
NOT invoke it, so no nodes run.
"""

from langgraph.checkpoint.sqlite import SqliteSaver
from stage2_checkpointer import builder, DB_FILE   # reuse the graph blueprint

if __name__ == "__main__":
    with SqliteSaver.from_conn_string(DB_FILE) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "run-1"}}

        print(f"Reading saved checkpoints for thread 'run-1' from {DB_FILE}")
        print("(this program never ran the graph)\n")

        history = list(graph.get_state_history(config))
        if not history:
            print("No checkpoints found. Run stage2_checkpointer.py first.")
        else:
            print(f"Found {len(history)} saved checkpoints, newest first:\n")
            for i, snap in enumerate(history):
                next_node = snap.next if snap.next else "(done)"
                print(f"  #{i}: next={next_node}")
                print(f"      topic={snap.values.get('topic')!r}")
                print(f"      report={snap.values.get('report', '(none yet)')!r}")
