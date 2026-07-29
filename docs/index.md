---
title: LangGraph News Verifier
description: Build a news-verification agent with LangGraph, one stage at a time.
---

Learn [LangGraph](https://langchain-ai.github.io/langgraph/) by building a news-claim verification agent one concept at a time — from "what is a graph" all the way to a live, web-searching fact-checker.

## [📘 Read the full training guide →](TRAINING-GUIDE.html)

**11 stages, written for people with no agent-framework background.** Each stage adds a single capability on top of the last, with the idea, the code, how to run it, and the pitfalls.

- 📄 [Download as PDF](TRAINING-GUIDE.pdf)
- 📊 [Slides](LangGraph-Project.pptx)
- 💻 [Source code on GitHub](https://github.com/Lindipuffi/langgraph-news-verifier)

## What it does

Give the agent a news claim. It searches the web for supporting and contradicting evidence, weighs it, and returns a reasoned verdict — with persistent memory, human-in-the-loop overrides, and parallel verification along the way.

## The stages

1. **A graph that makes a decision** — nodes, edges, state
2. **Memory that survives** — checkpointers
3. **Pausing for a human** — interrupts
4. **Seeing the graph** — LangGraph Studio
5. **Two readings, one reusable machine** — subgraphs
6. **The agent loop** — the claim-verification subgraph
7. **Many agents at once** — fan-out / parallel execution
8. **Escalate only what is uncertain** — conditional routing
9. **Giving the human real authority** — overrides
10. **Real articles** — reasoning over fetched text
11. **Real evidence, fully live** — web search via Tavily

[**Start reading →**](TRAINING-GUIDE.html)
