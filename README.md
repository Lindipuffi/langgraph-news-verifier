# LangGraph News Verifier

A staged, hands-on learning project that builds up a **news-claim verification agent** with [LangGraph](https://langchain-ai.github.io/langgraph/), one concept at a time. Each `stageN_*.py` file introduces a single new capability on top of the previous one, so the repo doubles as a walkthrough of LangGraph's core features.

## What it does

The end goal is an agent that takes a news claim, searches the web for supporting and contradicting evidence, and returns a reasoned verdict — with checkpointing, human-in-the-loop overrides, and parallel verification along the way.

## The stages

| File | Concept introduced |
|------|--------------------|
| `stage1_basics.py` | Minimal `StateGraph`, nodes, and edges |
| `stage2_checkpointer.py` / `stage2_inspect.py` | Persisting state with a checkpointer; inspecting saved state |
| `stage3_interrupt.py` | Interrupts / pausing a graph |
| `stage5_subgraphs.py` | Composing graphs with subgraphs |
| `stage6_verifier.py` | The claim-verification subgraph |
| `stage7_fanout.py` | Fan-out / parallel node execution |
| `stage8_finale.py` | Bringing the pieces together |
| `stage9_human_override.py` | Human-in-the-loop overrides |
| `stage10_real_articles.py` | Fetching and reasoning over real articles |
| `stage11_real_search.py` | Live web search via Tavily |

`studio_graph.py` + `langgraph.json` expose a graph for use with [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt   # or install langgraph, langchain-anthropic, tavily-python
```

Create a `.env` file with your keys (this file is gitignored):

```
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
LANGSMITH_API_KEY=lsv2_...      # optional, for tracing
```

Then run any stage, e.g.:

```bash
python stage11_real_search.py
```

## Notes

- `checkpoints*.sqlite` files (local run state) are gitignored.
- `Learning Documents/` contains the accompanying training material (guide, slides, `TRAINING DOCUMENT.pdf`).
