# Building a News-Verification Agent with LangGraph

**A stage-by-stage training guide, from first principles to a live system**

| | |
|---|---|
| **Document version** | 2.0 |
| **Last updated** | 28 July 2026 |
| **Audience** | Newcomers to agent frameworks. No prior knowledge of graphs, agents, or LLM APIs assumed. |
| **Prerequisites** | Ability to read basic Python (functions, dictionaries, classes) and run commands in a terminal. |
| **Software covered** | Python 3.14.5 · LangGraph 1.2.8 · LangChain Core 1.4.8 · Claude (Anthropic API) · Tavily Search API |
| **Estimated time** | 8–12 hours to build along; 2–3 hours to read the conceptual half only. |

---

## Contents

**Part I — Orientation**
- [0. How to use this guide](#0-how-to-use-this-guide)
- [1. What you are building](#1-what-you-are-building)
- [2. Core vocabulary](#2-core-vocabulary)
- [3. The stage map](#3-the-stage-map)
- [4. Environment and setup](#4-environment-and-setup)

**Part II — The eleven stages**
- [Stage 1 — A graph that makes a decision](#stage-1--a-graph-that-makes-a-decision)
- [Stage 2 — Memory that survives](#stage-2--memory-that-survives)
- [Stage 3 — Pausing for a human](#stage-3--pausing-for-a-human)
- [Stage 4 — Seeing the graph](#stage-4--seeing-the-graph)
- [Stage 5 — Two readings, one reusable machine](#stage-5--two-readings-one-reusable-machine)
- [Stage 6 — The agent loop](#stage-6--the-agent-loop)
- [Stage 7 — Many agents at once](#stage-7--many-agents-at-once)
- [Stage 8 — Escalate only what is uncertain](#stage-8--escalate-only-what-is-uncertain)
- [Stage 9 — Giving the human real authority](#stage-9--giving-the-human-real-authority)
- [Stage 10 — Real articles](#stage-10--real-articles)
- [Stage 11 — Real evidence, fully live](#stage-11--real-evidence-fully-live)

**Part III — Operating the system**
- [12. Reading the output](#12-reading-the-output)
- [13. Cost, limits, and safe operation](#13-cost-limits-and-safe-operation)

**Part IV — Synthesis**
- [14. Concept index](#14-concept-index)
- [15. Principles worth carrying elsewhere](#15-principles-worth-carrying-elsewhere)
- [16. Where to take it next](#16-where-to-take-it-next)

**Appendices**
- [A. Glossary](#appendix-a--glossary)
- [B. Command reference](#appendix-b--command-reference)
- [C. Troubleshooting index](#appendix-c--troubleshooting-index)
- [D. File and dependency map](#appendix-d--file-and-dependency-map)
- [E. Answer key](#appendix-e--answer-key)

---

# Part I — Orientation

## 0. How to use this guide

### What this document teaches

A working AI system: an agent that reads real news about a topic, works out which factual claims are disputed, fact-checks each one against the live web, and asks a human to review only the conclusions it is not confident about.

It is a teaching artefact, not a reference manual. The system was built in **eleven stages**, each introducing exactly one new idea and building on the last, and the guide follows that order deliberately. Concepts are defined the first time they appear and are not assumed before then.

### Two reading paths

Every stage is presented in two halves:

| Half | Contains | Read it if |
|---|---|---|
| **Part 1 — The idea** | What the stage accomplishes and why, in plain language. No code. | You want a conceptual tour, or you want to understand the design before seeing the implementation. |
| **Part 2 — How it works** | The mechanics: code, exact syntax, structures, and the traps. | You are building along, or you want a working understanding. |

Two supported paths through the document:

- **Conceptual read.** Chapters 0–3, then Part 1 of each stage, then Part III and IV. Roughly two hours, no software required.
- **Build-along.** Everything, in order, running each stage's code before moving on. Do not skip ahead: Stage 7 is incomprehensible without Stages 5 and 6, and Stage 9 depends on a subtlety introduced in Stage 7.

### Each stage follows the same template

1. **At a glance** — the concept added, what it builds on, which file, what gets installed.
2. **Learning objectives** — what you should be able to do afterwards.
3. **Part 1 — The idea.**
4. **Part 2 — How it works.**
5. **Run it** — the exact command and what correct output looks like.
6. **Pitfalls** — the specific ways this stage goes wrong.
7. **Check your understanding** — three questions. Answers in [Appendix E](#appendix-e--answer-key).

### Conventions

| Convention | Meaning |
|---|---|
| `code font` | A filename, command, function, or literal value. |
| **Bold on first use** | A term being defined. All definitions are collected in [Appendix A](#appendix-a--glossary). |
| ▸ **Concept** | A callout marking a transferable idea, not a LangGraph detail. |
| ⚠ **Pitfall** | A failure mode that has actually occurred in this project. |
| ✎ **Try it** | An optional experiment that makes the concept concrete. |

Commands are written for **Windows PowerShell**, since that is where this project was built. On macOS or Linux, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`. Where the shell itself causes a problem — and it does once, in Stage 9 — the guide says so explicitly.

---

## 1. What you are building

The finished pipeline, end to end:

```
                     TOPIC  ("rare earth minerals")
                       │
              ┌────────▼────────┐
              │  fetch articles │   real news search (Tavily)
              └────────┬────────┘
                       │  the same articles, sent to both readers
          ┌────────────┴────────────┐
          ▼                         ▼
  ┌───────────────┐         ┌───────────────┐
  │  optimistic   │         │   skeptical   │   two lenses, one LLM
  │    reading    │         │    reading    │
  └───────┬───────┘         └───────┬───────┘
          └────────────┬────────────┘
                       ▼
              ┌─────────────────┐
              │    synthesize   │   what do the two readings disagree about?
              └────────┬────────┘
                       │  N contested claims — N is unknown until runtime
      ┌────────────────┼────────────────┐
      ▼                ▼                ▼
 ┌─────────┐      ┌─────────┐      ┌─────────┐
 │ verify  │      │ verify  │      │ verify  │   one independent agent per claim,
 │ claim 1 │      │ claim 2 │      │ claim 3 │   each searching the real web
 └────┬────┘      └────┬────┘      └────┬────┘
      └────────────────┼────────────────┘
                       ▼
              ┌─────────────────┐
              │    aggregate    │   collect every verdict + confidence
              └────────┬────────┘
                       ▼
              ╱ any claim below ╲
              ╲   0.75 confidence?╱
                 │            │
                no            yes
                 │            │
                 │            ▼
                 │    ┌───────────────┐
                 │    │ human review  │  ← graph pauses, possibly for days
                 │    └───────┬───────┘
                 ▼            ▼
              ┌─────────────────┐
              │ finalize report │   verdicts + provenance annotations
              └─────────────────┘
```

Every stage in Part II adds one piece of that picture. Stage 1 builds the smallest possible version of a single box and a single fork; Stage 11 completes the diagram.

### A worked example of the output

```
=== FINAL report: 'rare earth minerals' ===
- [Refuted | confidence 0.92] Japan's deep-sea deposit has confirmed extractable reserves
- [Supported | confidence 0.95] China controls 70-90% of global refining capacity
- [Mixed | confidence 0.62] Western supply projects will close the gap by 2030   (corrected by human)
- [Mixed | confidence 0.55] Prices have stabilised since the export controls      (approved by human)
```

Three things to notice, because they are the whole design in miniature: each claim carries a **verdict**, each carries the agent's **confidence** in that verdict, and the low-confidence ones carry an annotation recording that a **human** saw them. Chapter 12 explains how to read this properly.

---

## 2. Core vocabulary

Seven words do most of the work in this project. Everything else is built from them.

**LLM (Large Language Model).** A text-prediction engine — Claude, in this project — that you send text to and get text back from. On its own it can only produce words. It cannot search the web, remember previous conversations, or take actions. Everything else in this guide exists to give an LLM the ability to *do* things.

**Agent.** An LLM placed in a loop with tools and a goal. Rather than answering once, it repeats a cycle: *look at what I know → decide what to do next → do it → look again*. The difference between "an LLM" and "an agent" is the loop and the ability to act. An agent decides **when it is finished**.

**Graph.** A map of the steps a program can take and the paths between them. Not a chart with axes — a network of boxes and arrows, like a flowchart. Ordinary code runs top to bottom; a graph runs *box to box*, and the path can branch, loop back, or split into parallel routes depending on what happens. Providing this structure is what LangGraph is for.

**Node.** One box in that map: one unit of work. "Fetch the articles" is a node. "Decide what to do next" is a node. In code, a node is just a function.

**Edge.** One arrow between boxes: what happens next. A plain edge always leads to the same place. A **conditional edge** decides at runtime, based on what has happened so far — that is where a graph stops being a straight line.

**State.** The shared notebook that travels through the graph. Every node reads from it and writes back to it. When the articles are fetched, they go into the state; the analysis step reads them from there. State is how nodes communicate — they never call each other directly.

**Tool.** An ordinary function an agent is allowed to call: a web search, a database lookup, a calculator. Tools are how an LLM reaches outside its own text-prediction ability and touches the real world.

> ▸ **Concept — why a graph rather than a script?**
> A script encodes control flow in nested `if` statements and loops scattered through the code, where it can only be understood by reading all of it. A graph makes control flow a **declared structure**: branches, cycles, and parallelism become objects you can draw, inspect, and test in isolation. That is the single largest reason to accept the extra ceremony LangGraph asks for.

---

## 3. The stage map

Read this table before starting. It is the whole curriculum on one page: each row is one stage, and the middle column is the one idea that stage exists to teach.

| # | Stage | Concept introduced | Why it matters |
|---|---|---|---|
| **1** | A graph that makes a decision | `State`, nodes, edges, **conditional edges** | Control flow becomes a map with junctions instead of a straight line. |
| **2** | Memory that survives | **Checkpointer**, **thread_id** | State is saved after every step, so a run can be inspected, stopped, and resumed. |
| **3** | Pausing for a human | `interrupt()` and `Command(resume=…)` | The graph can stop mid-run, hand a question to a person, and wait indefinitely. |
| **4** | Seeing the graph | **LangGraph Studio** | Structure and live runs become visible; debugging turns from guesswork into observation. |
| **5** | Two readings, one reusable machine | **Subgraphs**; prompt **lenses**; parallel writes to distinct keys | Complex graphs are composed from small self-contained ones, and disagreement is manufactured on purpose. |
| **6** | The agent loop | **Structured output** (Pydantic); think–act–observe cycle; **retry cap** | An LLM that decides for itself when it has enough evidence — with a hard stop that does not depend on its judgement. |
| **7** | Many agents at once | **`Send`** fan-out; **reducers** (map-reduce) | Work splits across an unknown number of parallel workers and merges back into one list. |
| **8** | Escalate only what is uncertain | **Confidence-threshold routing** | Human attention is spent only where the system doubts itself. |
| **9** | Giving the human real authority | **Structured human input**; the **reducer trap** | The reviewer's answer becomes data that rewrites verdicts, not a comment appended to a report. |
| **10** | Real articles | Replacing a **stub** at a clean boundary; **API keys** | Live news enters the system by swapping exactly one node. |
| **11** | Real evidence, fully live | **Dependency injection**; thread isolation | The verifier's search becomes real in one line — and one persistence bug is revealed and fixed. |

### How the stages relate

Stages 1–4 are **mechanics on fake data**: no LLM, no cost, nothing real. Stages 5–9 build the **analysis machinery** with a real model but invented articles and invented search results. Stages 10–11 **go live**, one data source at a time.

That sequencing is a deliberate teaching choice worth stealing:

> ▸ **Concept — build the machinery on fake data first.**
> Real data is slow, costs money, and changes between runs. A fan-out is far easier to debug when the inputs are predictable and free. Stubs also force you to define the boundary — "given a topic, return a list of article texts" — which is precisely what makes swapping in the real source a two-file change in Stage 10 rather than a rewrite.

### One file per stage, and they stack

Each stage is a **single self-contained Python file** that can be run on its own. Nothing is hidden in a framework or spread across a folder tree: opening `stage6_verifier.py` shows the entire agent loop, top to bottom.

| Stage | File | What it is |
|---|---|---|
| 1 | `stage1_basics.py` | A graph with one decision point |
| 2 | `stage2_checkpointer.py` | Saving progress to disk |
| — | `stage2_inspect.py` | Helper that prints the saved snapshots |
| 3 | `stage3_interrupt.py` | Pausing for a human |
| 4 | `langgraph.json` + `studio_graph.py` | Visual-tool setup — no stage file of its own |
| 5 | `stage5_subgraphs.py` | Two perspectives from one reusable subgraph |
| 6 | `stage6_verifier.py` | The agent loop that fact-checks one claim |
| 7 | `stage7_fanout.py` | One agent per claim, in parallel |
| 8 | `stage8_finale.py` | Confidence threshold + human review |
| 9 | `stage9_human_override.py` | Human corrections that change verdicts |
| 10 | `stage10_real_articles.py` | Real news articles |
| 11 | `stage11_real_search.py` | Real evidence — the finished system |

Stage 4 is the exception: it adds no logic, only visibility, so it contributes `langgraph.json` (a registry of which graphs the visual tool should offer) and `studio_graph.py` (a thin wrapper that hands one of them over).

**Which file to actually run.** From Stage 5 onward each file **imports** from its predecessors rather than copying their code. Stage 7 imports the perspective subgraph from Stage 5 and the verifier from Stage 6; Stage 8 imports Stage 7's pipeline; Stage 11 reuses Stage 10's graph outright. Therefore:

- **`stage11_real_search.py` is the finished product.** Run that one to use the system.
- Every earlier file remains runnable and is worth running while learning that stage, because each demonstrates its idea in isolation with less noise around it.
- **Deleting an earlier file breaks the later ones.** They are not superseded drafts; they are components still in use. [Appendix D](#appendix-d--file-and-dependency-map) shows the full dependency graph.

---

## 4. Environment and setup

Complete this chapter before Stage 1. It takes about fifteen minutes.

### 4.1 Versions

| Component | Version used | Notes |
|---|---|---|
| Python | 3.14.5 | Any recent 3.11+ should work. |
| `langgraph` | 1.2.8 | The framework itself. |
| `langchain-core` | 1.4.8 | Pulled in as a dependency. |
| `langgraph-checkpoint-sqlite` | 3.1.0 | Added at Stage 2. |
| `langgraph-cli[inmem]` | 0.4.31 | Added at Stage 4 (the Studio dev server). |
| `langchain-anthropic` | 1.4.8 | Added at Stage 5 (first real LLM call). |
| `tavily-python` | 0.7.26 | Added at Stage 10 (first real web search). |

Packages are installed **when their stage needs them**, not all at once. Each stage states what it adds.

### 4.2 Project layout

```
news-verifier/
├── .venv/                      virtual environment (never committed)
├── .env                        API keys (never committed)
├── .gitignore                  excludes .env, .venv, *.sqlite, __pycache__, .langgraph_api
├── langgraph.json              registry of graphs for LangGraph Studio
├── studio_graph.py             thin wrapper exposing one graph to Studio
├── stage1_basics.py            … through stage11_real_search.py
├── checkpoints_stage2.sqlite   saved state, created on first run of each stage
└── sac_workaround/             Windows-specific, see 4.5
```

### 4.3 Creating the environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install langgraph
```

Everything in this guide invokes the interpreter **inside the virtual environment explicitly** (`.\.venv\Scripts\python.exe …`) rather than relying on the environment being "activated". This is more typing and considerably fewer mysteries.

### 4.4 API keys

Two keys are needed, and neither before the stage that introduces it:

| Key | Needed from | Where to get it | Cost |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Stage 5 | console.anthropic.com | Pay-as-you-go; this project uses the cheapest model. |
| `TAVILY_API_KEY` | Stage 10 | tavily.com | Free tier ≈ 1,000 searches/month. |

Both live in a file named `.env` in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
```

> ⚠ **Pitfall — never commit `.env`.** Anyone holding the key can spend against your account. Create `.gitignore` at the same time you create `.env`, with at minimum:
> ```
> .env
> .venv/
> *.sqlite
> __pycache__/
> .langgraph_api/
> ```

The files load the key at run time via `python-dotenv`, so the key is read from `.env` rather than typed into any source file.

### 4.5 Windows-specific: Smart App Control

This applies only to Windows 11 machines where **Smart App Control (SAC)** is enforced. Symptom:

```
ImportError: DLL load failed while importing _uuid_utils:
An Application Control policy has blocked this file.
```

SAC blocks unsigned compiled extension modules. Popular Rust/C wheels (`pydantic-core`, `orjson`, `charset-normalizer`) carry enough reputation to load; less common ones do not.

> ⚠ **Do not turn Smart App Control off to fix this.** Disabling SAC is a one-way door — re-enabling it requires reinstalling Windows. Shim the offending package instead.

In this project only `uuid_utils` was blocked. The fix was a pure-Python shim overwriting `.venv/Lib/site-packages/uuid_utils/__init__.py`, with the source and a re-apply script kept in `sac_workaround/`:

```powershell
.\.venv\Scripts\python.exe sac_workaround\apply_sac_workaround.py
```

Re-run that after any `pip install` that touches `uuid_utils`. The general rule when a native import fails: read the error text to confirm it is a policy block, identify the exact package, and prefer a pure-Python replacement over weakening the machine's security posture.

### 4.6 A note for people new to terminals

Two habits prevent the two most common accidents:

- **Click into the terminal panel before typing or pasting.** In an editor like VS Code, keyboard focus stays wherever you last clicked. A command pasted while focus is in the editor lands *in the source file* — which in this project once corrupted line 1 of `stage2_checkpointer.py` and produced a baffling syntax error.
- **Run commands one line at a time**, and read the output before running the next one.

### 4.7 Setup checklist

Before Stage 1 you should have:

- [ ] A folder opened in your editor (not "no folder opened") containing `.venv/`
- [ ] `.\.venv\Scripts\python.exe --version` printing 3.11 or newer
- [ ] `.\.venv\Scripts\python.exe -c "import langgraph; print(langgraph.__version__)"` printing a version
- [ ] A `.gitignore` file
- [ ] No API keys yet — you will not need one until Stage 5

---

# Part II — The eleven stages

## Stage 1 — A graph that makes a decision

| | |
|---|---|
| **Concept introduced** | `State`, nodes, edges, and the **conditional edge** |
| **Builds on** | Nothing |
| **File** | `stage1_basics.py` |
| **Installs** | `langgraph` |
| **Real data?** | None. No LLM, no network, no cost. |

**Learning objectives.** After this stage you can: declare a state schema; write a node as a plain function; wire nodes with plain and conditional edges; compile and invoke a graph; and explain why a node returns a partial dictionary.

### Part 1 — The idea

The starting point is the smallest thing that still deserves to be called a graph: a few steps, and one point where the program **chooses** between two paths instead of marching straight ahead.

Picture a tiny assembly line. Each station on that line is a **node** — one step, one job. The first node collects some articles. Then there is a fork: if enough articles arrived, send the work to the "analyze" node; if not, send it to the "we couldn't do this" node. Same starting point, two possible endings, and the choice is made from what actually turned up rather than decided in advance.

That fork is a **conditional edge**, and it is the entire point of this stage. An ordinary program is a recipe: do this, then this, then this. A graph is a **map with junctions**. As the system grows, junctions are what let it react — to a shaky answer, to a missing document, to a human saying no. Everything in the later stages is more elaborate junctions.

The second idea is the shared notebook, whose real name is **state**. Nodes do not hand work to each other directly. Each writes its results into the state, the state travels along the line, and the next node reads whatever it needs. Adding a node later therefore does not require rewiring its neighbours — it just reads the state and writes back to it.

At this stage everything is self-contained: the topic is written directly into the code and the "articles" are two invented sentences. Nothing is fetched, nothing is real, no AI model is involved. That is deliberate. The purpose is to see the shape of a graph clearly, with nothing else moving.

### Part 2 — How it works

Three primitives, and nothing else.

**State** is declared as a `TypedDict` — a dictionary whose keys and value types are written down in advance:

```python
class State(TypedDict):
    topic: str
    articles: list[str]
    report: str
```

This is the notebook's table of contents. It documents what *may* exist in the state and gives editors and type-checkers something to verify against.

**Nodes** are plain functions. Each takes the whole state and returns a dictionary of *only the fields it wants to change*:

```python
def fetch_articles(state: State) -> dict:
    topic = state["topic"]
    return {"articles": [
        f"Outlet A reports on {topic}: the economy grew 3%.",
        f"Outlet B reports on {topic}: the economy shrank 1%.",
    ]}
```

Returning `{"articles": [...]}` does not erase `topic`. LangGraph **merges** each returned dictionary into the running state. This partial-update convention is what keeps nodes independent of each other.

Notice that the two placeholder sentences deliberately **disagree** — one says the economy grew, the other that it shrank. Fake data chosen to exercise the logic under test is a useful habit: the whole point of the later stages is handling disagreement, so the placeholder data contains some from the very beginning.

**Edges** are the wiring, declared on a builder:

```python
builder.add_edge(START, "fetch_articles")
builder.add_conditional_edges(
    "fetch_articles",
    route_after_fetch,                                       # returns a label
    {"analyze": "analyze", "insufficient": "insufficient"},  # label -> node
)
```

`START` and `END` are the graph's entry and exit points. A plain `add_edge` is unconditional. `add_conditional_edges` takes a **router function** — an ordinary function that reads the state and returns a string label — plus a mapping from those labels to real nodes:

```python
def route_after_fetch(state: State) -> str:
    return "analyze" if len(state["articles"]) >= 2 else "insufficient"
```

The router decides nothing about *how* to do the work; it decides only *where the work goes next*.

> ▸ **Concept — separate routing from working.** A node does a job. A router chooses a path. Keeping them apart means each can be tested on its own, and the graph's control flow can be read without reading any business logic. Every branch in this project, up to the confidence gate in Stage 8, is this same pattern.

Finally the graph is **compiled** — turned from a description into something runnable — and invoked with a starting state:

```python
graph = builder.compile()
result = graph.invoke({"topic": "the economy"})
```

The dictionary passed to `invoke` is the **starting state**: the contents of the notebook before any node has run. Here it carries one field, and its value is hardcoded in the file — there is no command-line argument yet. Stage 10 eventually supplies the topic at run time, and everything in between simply reads `state["topic"]` without knowing or caring where the value came from.

Notice also which fields are *absent* from that starting dictionary. `articles` and `report` are declared in `State` but not supplied; they do not exist yet and do not need to, because nodes fill them in as the run proceeds. **A state schema describes what may be present over the life of a run, not what must be present at the start.**

`invoke` runs the graph to completion and returns the final state, so `result["report"]` holds the finished output. Everything in the remaining ten stages is this same pattern with more interesting nodes and more interesting junctions.

### Run it

```powershell
.\.venv\Scripts\python.exe stage1_basics.py
```

Expected: the graph runs `fetch_articles`, the router sees two articles, takes the `analyze` branch, and prints a report built from them. The `insufficient` branch is not reached.

> ✎ **Try it.** Edit `fetch_articles` to return only one article and run again. The router now returns `"insufficient"` and the other ending prints. You have just watched a conditional edge do the only thing it does — and confirmed that the branch you were not testing actually works, which is a habit worth keeping.

### Pitfalls

- **Returning the whole state from a node.** Return only what changed. Returning everything works, but obscures which node owns which field and makes parallel execution (Stage 5) dangerous.
- **Expecting a node to be called.** Nodes are never called directly by other nodes. If a node did not run, the answer is always in the edges, not in the node.

### Check your understanding

1. Why does `fetch_articles` return `{"articles": [...]}` rather than the full state dictionary?
2. `State` declares three fields, but `invoke` is given only one. Why is that not an error?
3. What would you change to make the graph choose the `insufficient` path?

---

## Stage 2 — Memory that survives

| | |
|---|---|
| **Concept introduced** | **Checkpointers** and **threads** |
| **Builds on** | Stage 1 |
| **Files** | `stage2_checkpointer.py`, `stage2_inspect.py` |
| **Installs** | `langgraph-checkpoint-sqlite` |
| **Real data?** | None. |

**Learning objectives.** After this stage you can: attach a checkpointer at compile time; pass a `thread_id` in a config; read a run's saved history back; and explain what the `next` field means.

### Part 1 — The idea

The Stage 1 graph forgets everything the moment it finishes. That is fine for a calculation and useless for anything that needs to pause, be inspected, or continue tomorrow.

Stage 2 adds a **saved record**. After every node runs, the system writes down the complete contents of the state — like a photographer taking a picture of the shared notebook after each station on the line. These snapshots are **checkpoints**, the component that takes them is a **checkpointer**, and they go into a real file on disk that outlives the program.

The second idea is the **thread**. A thread is one job with its own separate history, identified by a piece of text the program chooses, called a `thread_id`. If two topics are being verified, each gets its own thread and neither can see the other's state. This is exactly how a chat application works: every conversation is a separate thread, so replies land in the right place and one conversation never contaminates another.

Two consequences follow, and both matter later:

- Because work is saved after every step, it can be **stopped and resumed** — even after the program has fully exited. That is what makes Stage 3 possible.
- Because a thread *remembers*, running new work on an old thread **continues that old job** rather than starting fresh. Useful when continuity is wanted; a genuine hazard when it is not. Stage 11 shows exactly how it bites.

### Part 2 — How it works

A **checkpointer** is an object that persists state. This project uses `SqliteSaver`, which writes to a local SQLite database file:

```python
with SqliteSaver.from_conn_string("checkpoints_stage2.sqlite") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
```

Persistence is attached at **compile time**, not declared in the graph. The same builder can be compiled with a checkpointer, without one, or with a different backend entirely — a separation Stages 8–11 rely on heavily.

Once a graph has a checkpointer, every invocation must say which thread it belongs to, via a second argument: a **config** dictionary.

```python
config = {"configurable": {"thread_id": "review-1"}}
graph.invoke({"topic": "the economy"}, config)
```

`thread_id` is just a string chosen by the caller. Same string → same saved history, and the graph picks up where that thread left off. New string → a clean slate.

Both the database filename and the thread id are hardcoded constants at the top of the file:

```python
DB_FILE = "checkpoints_stage2.sqlite"
THREAD_ID = "review-1"
```

Fixed values are fine while learning — every run reuses one thread on purpose, so its accumulated history can be inspected. In a real application both would be dynamic, with the thread id derived from a user, a document, or a request. **Stage 11 demonstrates precisely what goes wrong when a fixed thread id survives into a system that has grown up around it.**

The inspection script reads the saved checkpoints back:

```python
for snapshot in graph.get_state_history(config):
    print(snapshot.next, snapshot.values)
```

Each snapshot carries the state values at that moment and a `next` field naming the node the graph was *about to run*.

> ▸ **Concept — `next` is how a pause is recognised.** A completed run has an empty `next`. A run that stopped in the middle has a non-empty one, naming where it would resume. Every pause-and-resume mechanism in Stages 3 and 8–11 rests on that single field.

### Run it

```powershell
.\.venv\Scripts\python.exe stage2_checkpointer.py
.\.venv\Scripts\python.exe stage2_inspect.py
```

Expected: the first command behaves like Stage 1 and additionally creates `checkpoints_stage2.sqlite`. The second prints a list of snapshots, newest first — one per node that ran, each showing the state as it stood at that moment.

> ✎ **Try it.** Run `stage2_checkpointer.py` a second time and inspect again. The history is longer: the same thread accumulated another run. Now change `THREAD_ID` to `"review-2"` and inspect — a clean, short history. That contrast is the entire mental model of a thread.

### Pitfalls

- **Forgetting the config.** Invoking a checkpointed graph without a `thread_id` raises an error. Persistence and identity arrive together.
- **Assuming a fresh start.** Reusing a thread id resumes a job. If a run behaves as though it remembers something it should not, check the thread id first.

### Check your understanding

1. Why is the checkpointer supplied at `compile()` rather than declared inside the graph?
2. Two runs use `thread_id="review-1"`. What does the second one see?
3. What does a non-empty `next` field on the most recent snapshot tell you?

---

## Stage 3 — Pausing for a human

| | |
|---|---|
| **Concept introduced** | `interrupt()` and `Command(resume=…)` — **human-in-the-loop** |
| **Builds on** | Stage 2 (a pause is a saved position, not a waiting process) |
| **File** | `stage3_interrupt.py` |
| **Installs** | Nothing new |
| **Real data?** | None. |

**Learning objectives.** After this stage you can: pause a graph from inside a node; detect a pause from the caller; resume with a value; and explain why code above `interrupt()` runs twice.

### Part 1 — The idea

Some decisions should not be made by software alone. Stage 3 gives the graph the ability to **stop in the middle of its work, hand the question to a person, and wait** — for a second or for a week. The pause is called an **interrupt**, and it happens inside an ordinary node: one station on the line whose job is to stop the line and ask.

The general pattern has a name worth knowing: **human-in-the-loop**. It describes any system that deliberately routes a decision through a person rather than acting entirely on its own — not as a failure or a fallback, but as a designed step. It is the standard answer to a system that is useful but not trustworthy enough to be left unsupervised, which describes most AI systems doing consequential work. Stages 8 and 9 develop it considerably: the finished system pauses *only* when it is unsure, and lets the person overrule what it concluded.

The experience is deliberately dramatic. The program runs, reaches the point where a human is needed, prints the question, and **exits completely**. The terminal returns. Nothing is running. Later — after a coffee break, or a reboot — the same command is run again with an answer attached, and the work continues from exactly where it stopped.

This works only because of Stage 2. The pause is not a program sitting idle in memory; it is a **saved position on disk**. The graph's entire state was checkpointed at the moment it stopped, so resuming means loading that checkpoint and carrying on. Persistence is what makes a pause survivable.

One quirk is worth internalising early, because it produces genuinely surprising bugs: when the work resumes, the paused node **starts over from its beginning**, not from the line where it stopped. Anything that node did before pausing happens a second time.

### Part 2 — How it works

The pause is a single function call inside a node:

```python
def human_review(state: State) -> dict:
    decision = interrupt({
        "question": "Approve this draft report, or type a correction?",
        "draft_report": state["report"],
    })
    # everything below runs ONLY after resuming
    return {"human_decision": decision}
```

`interrupt()` does two things: it stops the entire graph, and it sends its argument — any JSON-serialisable payload — back to whoever called `invoke`. That payload is the question being put to the human, together with whatever context they need in order to answer.

The caller detects the pause by looking for a special key in the result:

```python
result = graph.invoke({"topic": "the economy"}, config)
if "__interrupt__" in result:
    payload = result["__interrupt__"][0].value      # the dict passed to interrupt()
```

Resuming happens in a separate process run, with a `Command`:

```python
graph.invoke(Command(resume="approve"), config)
```

Two details matter. First, the config must carry the **same `thread_id`** — that is what identifies which paused job to continue. Second, the value inside `Command(resume=…)` becomes the **return value of the `interrupt()` call**. That is the channel through which a human's answer enters the graph, and Stage 9 exploits it heavily.

In practice the answer arrives as a command-line argument, and the file reads it to decide whether this launch is a fresh start or a continuation:

```python
resuming = len(sys.argv) > 1 and sys.argv[1] == "--resume"
if resuming:
    answer = sys.argv[2] if len(sys.argv) > 2 else "approve"
    result = graph.invoke(Command(resume=answer), config)
else:
    result = graph.invoke({"topic": "the economy"}, config)
```

The answer itself is free text at this stage. Typing `approve` accepts the draft; anything else is treated as a correction and replaces the report wholesale:

```python
if decision.strip().lower() == "approve":
    final = state["report"].replace("DRAFT: ", "")
else:
    final = f"(human-corrected) {decision}"
```

That is a crude way to handle a human's answer — a single string, interpreted by one `if`. It is enough to demonstrate the mechanism, and Stage 9 replaces it with something that can change specific results.

#### The re-run boundary

On resume, LangGraph does not restart the node from the middle. It re-executes the node function **from the top**, and this time `interrupt()` returns the supplied value instead of pausing. Any code above the `interrupt()` call therefore runs **twice**.

> ⚠ **Pitfall — side effects above `interrupt()`.** Keep everything above the call cheap and repeatable: reading state, formatting a question. Put anything that costs money, sends email, or writes to a database *below* the interrupt, or make it safe to repeat. A node that charges a credit card before pausing will charge it again on resume.

### Run it

Two launches, deliberately:

```powershell
.\.venv\Scripts\python.exe stage3_interrupt.py
.\.venv\Scripts\python.exe stage3_interrupt.py --resume "approve"
```

Expected: the first command prints the draft report and the review question, then exits to the prompt with the run unfinished and `checkpoints_stage3.sqlite` on disk. The second prints the finished report with the draft marker removed.

> ✎ **Try it.** Resume with a correction instead: `--resume "The growth figure is disputed."` The report is replaced by your text. Then run the start command again — because the thread already has a completed history, you will see Stage 2's accumulation behaviour first-hand. Delete the `.sqlite` file for a clean slate.

### Pitfalls

| Symptom | Cause |
|---|---|
| Resume does nothing, or starts a new run | Different `thread_id` between the two launches. |
| Work above the interrupt happened twice | Expected. See the re-run boundary above. |
| `interrupt()` errors or never persists | The graph was compiled without a checkpointer. Interrupts require persistence. |

### Check your understanding

1. Why can the program exit completely and still resume later?
2. What exactly does `Command(resume="approve")` deliver, and where does it arrive?
3. Which line of a `human_review` node is the safe place to put an expensive API call?

---

## Stage 4 — Seeing the graph

| | |
|---|---|
| **Concept introduced** | Visual inspection with **LangGraph Studio** |
| **Builds on** | Stages 1–3 (it visualises them) |
| **Files** | `langgraph.json`, `studio_graph.py` |
| **Installs** | `langgraph-cli[inmem]` |
| **Real data?** | None. |

**Learning objectives.** After this stage you can: register a graph in `langgraph.json`; start the dev server; run a graph from the browser; answer an interrupt in the UI; and recognise the four things that look broken but are not.

### Part 1 — The idea

Up to this point the graph exists only as text. Stage 4 makes it **visible**: a diagram in a browser where each box is a node and each arrow an edge, lighting up as work flows through it.

This is not decoration. A graph's whole advantage is that its structure is explicit, and a picture of that structure is the fastest way to confirm that what was built matches what was intended. Loops are obvious as loops. A branch that can never be reached is obvious as a dead end. Stepping through a run and inspecting the notebook at each step turns debugging from guesswork into observation.

The tool is **LangGraph Studio**. A small server runs locally on your machine, holding the actual code; the interface is a web page that connects to it. The code never leaves the computer.

Studio is also where the human-in-the-loop work of Stages 8–11 is most pleasant to drive: when a run reaches an `interrupt()`, the interface offers a box to type the answer into, so the two-launch command-line dance is unnecessary.

### Part 2 — How it works

A registry file tells the server which graphs exist and where to find them:

```json
{
  "dependencies": ["."],
  "graphs": {
    "news_verifier": "./studio_graph.py:graph",
    "stage7_fanout": "./stage7_fanout.py:graph"
  },
  "env": ".env"
}
```

Each entry maps a display name to `file.py:variable`, and the variable must be a **compiled** graph. Every later stage adds one line here, which is why any stage can be opened in the visual tool.

The graph exposed to Studio is compiled **without** a checkpointer:

```python
graph = builder.compile()      # the dev server supplies its own persistence
```

This is the Stage 2 separation paying off: the same builder is compiled bare for the server and with `SqliteSaver` for direct command-line runs. From Stage 8 onward, files do exactly that — both, in the same file.

### Run it

```powershell
.\.venv\Scripts\langgraph.exe dev --no-reload
```

> ⚠ **Pitfall — `--no-reload` is required.** Without it, the file-watcher restarts the server repeatedly and the browser session is dropped mid-run. The server must keep running the whole time the interface is in use; closing that terminal ends the session.

Then, in the browser interface:

1. Choose a graph from the **dropdown at the top** — the names come from the registry above.
2. In the input panel, supply the same starting state that `invoke` would take, typed as JSON:
   ```json
   {"topic": "the economy"}
   ```
3. Press **Submit**. This is the visual equivalent of `graph.invoke({"topic": "the economy"})` — same entry point, driven from a browser.
4. Watch the left-hand diagram highlight nodes as they execute. Click any step in the right-hand run timeline to inspect the state at that moment.
5. If the run pauses at an `interrupt()`, a **resume box** appears. Type the answer there and submit; no second launch is needed.

Studio is hosted at `smith.langchain.com/studio` and connects to your local server. A free LangSmith browser login is required.

### Pitfalls: four things that look broken but are not

| What you see | Explanation |
|---|---|
| **The Trace tab is empty.** | Tracing is a separate cloud service and needs `LANGSMITH_API_KEY` in `.env`. Local visualisation works fine without it. |
| **Threads vanish after a server restart.** | The dev server keeps its threads **in memory**. This does not touch the `.sqlite` files written by direct command-line runs — those are a different store entirely. |
| **The "Memory" panel is empty.** | That panel shows the long-term **Store**, a separate feature this project does not use. Live state for a run is under the interaction panel: click a step in the timeline. |
| **A nested graph's output is labelled oddly.** | The run timeline labels a subgraph's output by the **parent field it lands in** (`perspective_a`), not by the internal node that produced it (`analyze`). The internal nodes are visible in the left-hand diagram. Nested graphs *do* render, as expandable boxes. |

### Check your understanding

1. Why is the graph in `studio_graph.py` compiled without a checkpointer?
2. You restart the dev server and your threads are gone, but `checkpoints_stage3.sqlite` still exists. Explain.
3. Where do you look in the UI to see the state partway through a run?

---

## Stage 5 — Two readings, one reusable machine

| | |
|---|---|
| **Concept introduced** | **Subgraphs**; prompt **lenses**; parallel writes to distinct keys |
| **Builds on** | Stages 1–4 |
| **File** | `stage5_subgraphs.py` |
| **Installs** | `langchain-anthropic` |
| **Real data?** | **Real LLM calls begin here.** Articles are still invented. Requires `ANTHROPIC_API_KEY`. |

**Learning objectives.** After this stage you can: build a subgraph with its own state and invoke it from a parent node; explain why one subgraph serves two roles; run two nodes in parallel safely; and describe why the LLM client is created lazily.

### Part 1 — The idea

This stage introduces the project's central editorial trick and its main structural tool at the same time.

#### The editorial trick

Rather than producing one neutral summary of the news, the system reads the *same* articles **twice, through two different mindsets**: once as an optimistic economist looking for growth and good indicators, and once as a cautious skeptic looking for risk and what is unproven.

A "mindset" here is nothing mystical. It is one sentence of instruction handed to the language model along with the articles, telling it what to pay attention to. That instruction is called a **lens** in this project, and it is the only difference between the two readings. The full block of text sent to the model — lens plus articles plus request — is a **prompt**.

Both readers see **all** the articles. The articles are not divided between them. The optimist and the skeptic receive the identical stack; only their instructions differ. A set of news articles typically contains both encouraging and worrying material, so the optimist naturally foregrounds the growth figures while the skeptic foregrounds the warnings — from the same source text.

Why deliberately manufacture two biased readings instead of one balanced one? Because the goal is fact-checking, and the most useful thing to fact-check is **whatever two reasonable readings disagree about**. A neutral summary hides those tensions; two opposed summaries expose them. The disagreements become the checkable claims that drive the rest of the system.

> ▸ **Concept — disagreement as a search strategy.** This generalises well beyond news. When you need to find the weak points in a document, a plan, or a codebase, generating two opposed readings and diffing them surfaces contested material far more reliably than asking for "the important points."

#### The structural tool

The two readers are not two pieces of code. There is *one* reader, built once, used twice with a different lens each time. A small self-contained machine, reusable wherever it is needed — this is a **subgraph**: a complete graph, with its own nodes and its own state, that acts as a single step inside a bigger graph. The reader has two internal steps of its own: tidy the articles into one block of text, then send them to the model.

This is also the stage where the system stops being a simulation. Real requests go to a real language model for the first time — specifically **Claude Haiku**, chosen because it is fast and inexpensive, which matters when a single run will eventually make dozens of calls.

The articles themselves are still invented. Making *those* real is a separate upgrade and waits until Stage 10: there is no benefit to paying for live news while the analysis machinery is still being built.

### Part 2 — How it works

A subgraph is an ordinary graph with **its own state schema**, compiled once and then invoked from inside a node of the parent:

```python
class PerspectiveState(TypedDict):
    articles: list[str]     # input
    lens: str               # input: the angle to read from
    article_text: str       # internal
    view: str               # output

def build_perspective_subgraph():
    sub = StateGraph(PerspectiveState)
    sub.add_node("gather", gather)      # formats the articles (no LLM)
    sub.add_node("analyze", analyze)    # calls the LLM through the lens
    sub.add_edge(START, "gather")
    sub.add_edge("gather", "analyze")
    sub.add_edge("analyze", END)
    return sub.compile()

perspective_subgraph = build_perspective_subgraph()     # built ONCE
```

The subgraph's state is separate from the parent's. It receives only what it is given and returns only what it produces — a clean interface rather than a shared pool of variables. Note the four fields split into three roles: two inputs, one purely internal scratch field, one output. Writing that down is a useful discipline; it makes the contract explicit.

The "perspective" is nothing more than an instruction string:

```python
LENS_A = "an optimistic economist who focuses on positive indicators and growth"
LENS_B = "a cautious skeptic who focuses on risks, downsides, and what's unproven"
```

interpolated into the prompt inside `analyze`:

```python
prompt = (
    f"You are a news analyst with this specific angle: {state['lens']}.\n"
    f"Here are today's article summaries:\n{state['article_text']}\n\n"
    f"In 2-3 sentences, give your take..., emphasizing your angle."
)
reply = get_llm().invoke(prompt)
return {"view": reply.content.strip()}
```

Two thin wrapper nodes in the parent invoke the same compiled subgraph with different lenses and — critically — write to **different state keys**:

```python
def perspective_a_node(state: State) -> dict:
    result = perspective_subgraph.invoke({"articles": state["articles"], "lens": LENS_A})
    return {"perspective_a": result["view"]}

def perspective_b_node(state: State) -> dict:
    result = perspective_subgraph.invoke({"articles": state["articles"], "lens": LENS_B})
    return {"perspective_b": result["view"]}
```

Both pass the same `state["articles"]`. Nothing is split.

#### Parallel execution without conflict

Both perspective nodes are wired directly from `fetch_articles`, so LangGraph runs them **in parallel**; `synthesize` is wired from both, so it waits for both to finish:

```python
builder.add_edge("fetch_articles", "perspective_a")   # fan-out
builder.add_edge("fetch_articles", "perspective_b")
builder.add_edge("perspective_a", "synthesize")       # fan-in
builder.add_edge("perspective_b", "synthesize")
```

Because the two nodes write to *different* keys, their parallel writes cannot collide and no special merge rule is required. Stage 7 introduces the opposite case — many nodes writing to the *same* key — and the mechanism needed to handle it.

#### The lazy client

The LLM is created on first use rather than at import:

```python
_llm = None

def get_llm():
    global _llm
    if _llm is None:
        from langchain_anthropic import ChatAnthropic
        _llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
    return _llm
```

Three benefits: the file can be imported and its structure inspected **without an API key**; nothing is constructed unless it is actually needed; and tests can replace `_llm` with a stand-in.

> ▸ **Concept — the lazy accessor is the seam.** That last benefit is the foundation of every test in this project, and in Stage 11 it becomes the mechanism for upgrading the entire live system in one line. When you find yourself writing `get_x()` instead of a module-level `x`, you are creating a place where the implementation can later be swapped from outside.

`temperature=0` requests the most deterministic output the model can give — appropriate for analysis, where reproducibility matters more than variety.

#### What this looks like in practice

The placeholder articles at this stage are three sentences, deliberately mixing good and bad news:

```
Outlet A: the economy — GDP grew 3% last quarter, unemployment at record lows.
Outlet B: the economy — inflation still high, analysts warn the growth may not last.
Outlet C: the economy — consumer spending up, but household debt is rising fast.
```

Both readers receive all three. The optimist's summary foregrounds the growth figure, the record-low unemployment, and the rising spending; the skeptic's foregrounds the persistent inflation, the warning that growth may not last, and the household debt. Neither invents anything — each weights the same three sentences differently, because each was told to.

That is what makes the next stage possible: two summaries built from identical source material that nonetheless disagree about what the situation is.

### Run it

First, put your key in `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
```

```powershell
.\.venv\Scripts\python.exe stage5_subgraphs.py
```

Expected: two short paragraphs, visibly different in emphasis, both traceable to the same three placeholder sentences. This is the first stage that costs money — a few tenths of a cent.

**In Studio**, add the graph to `langgraph.json`:

```json
"stage5_perspectives": "./stage5_subgraphs.py:graph"
```

then switch to it via the top dropdown. The `perspective_a` and `perspective_b` nodes render as **expandable boxes containing `gather → analyze`** — this is what a nested subgraph looks like.

> ✎ **Try it.** Change `LENS_B` to something unrelated to economics ("a sports commentator who relates everything to team dynamics") and rerun. Watching how far the summary moves on one sentence of instruction is the fastest possible lesson in what a prompt actually does.

### Pitfalls

- **Both perspective nodes writing the same key** would silently lose one result. They write `perspective_a` and `perspective_b` precisely to avoid this. The general fix, when you *do* need a shared key, is Stage 7's reducer.
- **Creating the LLM at module level** breaks imports for anyone without a key — including your own tests and Studio's registry loading.

### Check your understanding

1. Both perspective nodes call the *same* compiled subgraph object. Why does that not cause the two results to interfere?
2. Why do the two parallel nodes need no reducer, when Stage 7's parallel nodes do?
3. Name two things `get_llm()` makes possible that a module-level `llm = ChatAnthropic(...)` would not.

---

## Stage 6 — The agent loop

| | |
|---|---|
| **Concept introduced** | **Structured output** (Pydantic); the think–act–observe **agent loop**; the **retry cap** |
| **Builds on** | Stage 5 |
| **File** | `stage6_verifier.py` |
| **Installs** | Nothing new |
| **Real data?** | Real LLM reasoning; **stub** search results. |

**Learning objectives.** After this stage you can: define a Pydantic decision schema; force a model to answer in that shape; build a cycle in a graph; cap that cycle; and explain why the cap counts searches rather than reasoning steps.

### Part 1 — The idea

This is the heart of the project, and the stage where "a program that calls an LLM" becomes "an agent."

Consider how a person fact-checks a single claim. They read it. They decide they do not know enough. They search. They read the result. They decide whether that settled it — and if not, they search again, differently. At some point they judge that they have enough and write a conclusion. Crucially, *nobody tells them how many searches to do.* They decide that themselves, as they go.

That is exactly what this stage builds. A verifier is given **one claim** and repeats a cycle:

> **Think** — given the claim and everything gathered so far, is there enough to judge it?
> **Act** — if not, search for something specific.
> **Observe** — add the result to the pile.
> …and think again.

The loop ends one of two ways: the agent decides it has enough and delivers a verdict, or it hits a **hard limit** on how many searches it is allowed — a **retry cap**, set here to three.

> ▸ **Concept — every loop needs a stop that does not depend on the looper.** An agent that keeps deciding "not enough yet" will loop forever, burning money and time. The cap is not a fallback for when things go wrong; it is the only actual guarantee that the loop terminates. This one turns out to bound the project's bill as well (Stage 11).

The search itself is not real yet. At this stage it is a stand-in function returning pre-written text — a **stub**. The agent genuinely decides *whether* and *what* to search; it simply receives invented results. This keeps the loop free to run hundreds of times while it is being built and understood. Real search arrives in Stage 11.

#### The second idea: making the agent's answer machine-readable

For the loop to work, the surrounding program must act on the agent's decision — send it to the search step, or to the conclusion step. But an LLM naturally replies in prose: *"I think I should probably look up the GDP figures first, since the numbers here seem vague."* A program cannot reliably act on that. Detecting the word "search" in a paragraph breaks the moment the model writes "searching", or "I won't search".

The solution is to stop accepting paragraphs. Instead, the agent is handed a **form with labelled boxes** and required to answer by filling it in:

> **Action:** `search` *or* `conclude` — one of exactly these two, nothing else
> **Query:** (if searching) the exact text to search for
> **Verdict:** (if concluding) the judgement
> **Confidence:** (if concluding) a number from 0 to 1
> **Reasoning:** a brief explanation

Now the program simply reads the Action box. No interpretation, no guessing.

That form has a name: a **Pydantic model** — a declaration, written in Python, of exactly which fields an answer must contain and what type each must be. Requiring the model to answer in that shape is called **structured output**. The combination — *an agent looping while filling in a decision form each time around* — is what this guide calls the **structured decision loop**.

A note on approach: LangChain offers higher-level helpers for tool-calling agents (`bind_tools`, `ToolNode`). This project deliberately builds the loop by hand, because a hand-built loop shows every moving part. Once the mechanism is understood, the helpers are a convenience rather than a mystery.

### Part 2 — How it works

Two constants govern the whole stage:

```python
MODEL = "claude-haiku-4-5-20251001"
MAX_SEARCHES = 3        # the retry limit — the loop's safety belt
```

#### The stub tool

The search the agent calls is a plain function returning canned text, keyed off a few words in the query:

```python
def get_search_result(query: str) -> str:
    q = query.lower()
    if "gdp" in q or "grow" in q:
        return "Bureau of Economic Analysis: real GDP rose 3.0% (annualized) last quarter..."
    if "unemployment" in q or "jobs" in q:
        return "Bureau of Labor Statistics: unemployment held at 3.9%, near record lows."
    ...
    return "Multiple outlets report broadly similar figures, but with differing emphasis..."
```

Note the shape of its signature: **one string in, one string out**. That contract is what makes the function replaceable, and in Stage 11 a real web search with the identical signature drops straight into its place.

#### The form: a Pydantic model

A **Pydantic model** is a Python class declaring the exact shape of a piece of data — its fields, their types, and their rules:

```python
class Decision(BaseModel):
    action: Literal["search", "conclude"] = Field(description="search for more, or conclude now")
    query: str        = Field(default="", description="the search query, if action=search")
    verdict: str      = Field(default="", description="short judgement, if action=conclude")
    confidence: float = Field(default=0.0, description="0.0-1.0, if action=conclude")
    reasoning: str    = Field(default="", description="brief reason for this decision")
```

Reading it piece by piece:

| Element | Meaning |
|---|---|
| `BaseModel` | Inheriting from it marks this class as a Pydantic model — that is what gives it validation. |
| `action: …` | The name and type of one field. |
| `str`, `float` | The field's type: text, decimal number. |
| `Literal["search", "conclude"]` | The value must be **exactly one of these two strings**. Nothing else is accepted. |
| `Field(default=…)` | The value used when the field is not filled in. |
| `Field(description=…)` | A plain-English note about the field. **This text is sent to the model** so it knows what belongs there — it is not a comment for human readers. |

`Literal` is the load-bearing piece. It guarantees `action` can never arrive as `"searching"`, `"maybe"`, or a sentence, which is what makes the routing logic safe rather than hopeful.

Pydantic **validates**: text where a number belongs, or a value outside the allowed set, is rejected rather than silently passed along. The result is either a correct object or a loud error — never quietly wrong data flowing into the graph.

With a validated object in hand, reading the decision is a dot away:

```python
decision.action        # "search"  — guaranteed one of two values
decision.query         # "GDP growth last quarter"
decision.confidence    # 0.85
```

#### Forcing the model to fill in the form

One method connects the model to the schema:

```python
_decider = ChatAnthropic(model=MODEL, temperature=0).with_structured_output(Decision)
```

`with_structured_output(Decision)` changes the contract: instead of prose, the model must return something fitting the `Decision` shape. So this call returns a populated `Decision` **object**:

```python
decision = get_decider().invoke(prompt)
```

Concretely, a first time around the loop might produce:

```python
Decision(action="search", query="US GDP growth last quarter official figure",
         verdict="", confidence=0.0, reasoning="No evidence gathered yet.")
```

and a later one, after evidence has accumulated:

```python
Decision(action="conclude", query="", verdict="Supported", confidence=0.9,
         reasoning="Official BEA figures match the claimed 3% growth.")
```

The unused fields fall back to the declared defaults, which is why `query` is empty in the second and `verdict` empty in the first. One object, two very different meanings — distinguished entirely by `action`, which the router reads.

#### The loop itself

Three nodes and a router. The `agent` node reasons:

```python
def agent(state: VerifierState) -> dict:
    prompt = (
        f"You are fact-checking ONE claim. Decide your next action.\n\n"
        f"CLAIM: {state['claim']}\n\n"
        f"EVIDENCE SO FAR ({count} of {MAX_SEARCHES} searches used):\n{evidence_text}\n\n"
        f"If you lack enough reliable evidence, choose action='search' with a focused `query`.\n"
        f"If you have enough (or are out of searches), choose action='conclude' with a "
        f"`verdict` ... and a `confidence` from 0.0 to 1.0."
    )
    decision = get_decider().invoke(prompt)
    if decision.action == "search":
        return {"next_action": "search", "pending_query": decision.query, ...}
    return {"next_action": "conclude", "verdict": decision.verdict, ...}
```

Note that the prompt tells the agent how many searches it has used. Making a budget visible to the thing spending it is what allows it to conclude gracefully rather than being cut off.

The `tools` node acts and observes — running the search, appending the result, incrementing the counter:

```python
return {
    "evidence": state.get("evidence", []) + [f"Q: {query} | A: {result}"],
    "search_count": state.get("search_count", 0) + 1,
}
```

The router turns those decisions into movement, and this is where the loop is actually created:

```python
def route_after_agent(state: VerifierState) -> str:
    if state.get("next_action") == "conclude":
        return "done"                      # agent finished on its own -> END
    if state.get("search_count", 0) >= MAX_SEARCHES:
        return "conclude"                  # wants more, but out of retries
    return "tools"                         # search, then loop back
```

with the wiring:

```python
builder.add_conditional_edges("agent", route_after_agent,
                              {"tools": "tools", "conclude": "conclude", "done": END})
builder.add_edge("tools", "agent")         # <-- the cycle
```

`tools → agent` is the loop. This is something a straight-line program cannot express and a graph handles naturally: the same node runs repeatedly, each time with more evidence in the state.

#### What the cap actually counts

`MAX_SEARCHES = 3` limits **searches**, not thinking steps. Two different endings follow:

1. The agent chooses `conclude` on its own — possibly after 0, 1, or 2 searches. The cap never applies.
2. The agent keeps choosing `search` and is forced into the dedicated `conclude` node once the count reaches 3. That node prompts the model to deliver a final verdict with an honest — probably lower — confidence.

A useful counting detail: the `agent` node runs **one more time than the number of searches**. A claim that uses all three looks like:

```
agent → tools → agent → tools → agent → tools → [cap reached] → conclude
```

Three searches, four reasoning steps.

### Run it

The file's `__main__` block feeds in a single hand-written claim so the loop can be exercised in isolation:

```python
CLAIM = "The economy grew 3% last quarter, with unemployment at record lows."
result = verifier.invoke({"claim": CLAIM, "evidence": [], "search_count": 0})
```

```powershell
.\.venv\Scripts\python.exe stage6_verifier.py
```

Expected: a trace of the agent's decisions — one or more searches with the queries it chose, then a verdict with a confidence score. Note the starting state: the claim, an **empty** evidence list, and a search count of **zero**. Those three fields are the agent's entire working memory, and they are reset for every claim.

That literal `CLAIM` is a **test harness only**. When this verifier is used by the pipeline in Stage 7, claims arrive from the pipeline and the hardcoded one is never read.

Be precise about what is and is not invented at this point: the claim is hardcoded, the search results are canned, but the agent's reasoning, its choice of query, and its final verdict are all genuinely produced by the model.

**In Studio**, register it as `"stage6_verifier"` — this is the first graph in the project that visibly contains a cycle.

> ✎ **Try it.** Set `MAX_SEARCHES = 1` and rerun. Watch the agent get cut off and produce a lower-confidence verdict. Then set it to `6` and watch it stop on its own before reaching the cap. The gap between "what the agent wants" and "what it is allowed" is the whole design space of agent safety limits.

### Pitfalls

- **Parsing prose instead of structure.** Any code that inspects the model's free text for keywords is a bug waiting for a rephrasing. Use `Literal` and read the field.
- **Forgetting to reset `evidence` and `search_count`** when reusing a verifier for a second claim. Stage 7 does this correctly, per instance.
- **A loop with no cap.** Even a well-behaved model will occasionally decide it needs one more search, indefinitely.

### Check your understanding

1. What specifically does `Literal["search", "conclude"]` prevent?
2. `MAX_SEARCHES = 3`. How many times does the `agent` node run in a worst-case verification, and why?
3. Why is `Field(description=...)` more than documentation?

---

## Stage 7 — Many agents at once

| | |
|---|---|
| **Concept introduced** | **`Send`** dynamic fan-out; **reducers**; map-reduce |
| **Builds on** | Stage 5 (perspectives) and Stage 6 (the verifier, used as a subgraph) |
| **File** | `stage7_fanout.py` |
| **Installs** | Nothing new |
| **Real data?** | Real LLM throughout; stub search; invented articles. |

**Learning objectives.** After this stage you can: extract a list from a model with structured output; fan out to an unknown number of parallel workers with `Send`; merge their results with a reducer; and explain why each worker returns a one-item list.

### Part 1 — The idea

Stage 6 verifies one claim. Real articles produce several. This stage handles *however many there turn out to be*.

Two things happen here.

**First, the disputed claims are identified.** The optimistic and skeptical readings from Stage 5 are put side by side and the model is asked: *what specific, checkable facts do these two disagree about?* The output is a list of short factual statements — the points of genuine contention.

These claims are **not written into the program**. They are produced fresh on every run from whatever the articles actually said, which means the number of them is unknown until the moment they arrive. It might be two; it might be five.

**Second, one independent agent is launched per claim, all at once.** Not one agent working through a list — a separate verifier for each claim, running side by side, each with its own evidence pile and its own budget of searches. When they all finish, their verdicts are collected into a single list.

Splitting-then-gathering has a standard name: **map-reduce**. The "map" is launching one worker per item; the "reduce" is combining their results. In LangGraph the map is performed by **`Send`** and the combining rule is called a **reducer**. Both appear in Part 2; the shape to hold in mind is a fan-out followed by a fan-in.

#### Why one agent per claim?

| Reason | Explanation |
|---|---|
| **Focus** | An agent working on a single claim searches for exactly that claim. One agent juggling five produces vaguer searches and blurrier conclusions. |
| **Speed** | Five agents working simultaneously finish in roughly the time of the slowest, not the sum of all five. |
| **Fair budgets** | Each claim gets its own three searches. In a shared loop, one difficult claim could consume the entire budget and starve the rest. |
| **Isolation** | Evidence about claim 1 cannot contaminate the verdict on claim 4. |
| **Clean routing afterwards** | Each claim ends with its own verdict and its own confidence — which is exactly what makes Stage 8's routing possible. |

In fairness: batching all claims into a single request is a legitimate alternative and costs less. The trade is quality and independence against cost, and for fact-checking, independence is worth paying for.

#### On the number of claims

The instruction asks for "the 2-4 claims they most disagree about." In practice the result is often exactly 4 — not because 4 is hardcoded, but because a stated range acts as an anchor and language models tend to fill to the top of it, and because two opposed readings of a news story usually do disagree at least four times. Changing that one phrase in the prompt changes the typical count; the machinery downstream handles any number.

#### On confidence

Each verdict arrives with a confidence score, and it is worth being precise about what that number is: the **model's own stated certainty about its verdict**, not a calibrated statistical probability. A low score means the agent considers its own conclusion shaky, regardless of which verdict it reached. It is a self-report — and it is the signal Stage 8 acts on.

> ▸ **Concept — latency, throughput, concurrency.** Three related terms, easy to confuse. **Latency** is how long one request takes end to end. **Throughput** is how many requests finish per unit of time. **Concurrency** is how many things are in flight at once. The fan-out improves *latency* for a single run by increasing *concurrency*. Separately: **training time** is when a model is built, once, at enormous cost; **inference time** is every time the finished model is used — what this project pays for. Running an agent in a loop deliberately spends more inference-time compute to get a better answer.

### Part 2 — How it works

#### Extracting the claims

A second Pydantic model, and a second structured-output call:

```python
class ContestedClaims(BaseModel):
    claims: list[str] = Field(description="specific, checkable factual claims the two views disagree on")

def synthesize(state: State) -> dict:
    prompt = (
        f"Two analysts reviewed the same news articles.\n\n"
        f"OPTIMIST SAID:\n{state['perspective_a']}\n\n"
        f"SKEPTIC SAID:\n{state['perspective_b']}\n\n"
        f"List the 2-4 specific, checkable FACTUAL claims they most disagree about. "
        f"Each should be a single verifiable statement."
    )
    result = get_claims_llm().invoke(prompt)
    return {"contested_claims": result.claims}
```

Same technique as Stage 6, different shape: here the structure requested is a list of strings.

#### The map step: `Send`

A conditional edge can return a **list of `Send` objects** instead of a destination label. Each `Send` launches one copy of a node with its own private input:

```python
def dispatch_verification(state: State):
    claims = state["contested_claims"]
    if not claims:
        return "aggregate"                                    # nothing to verify
    return [Send("verify_claim", {"claim": c}) for c in claims]
```

Three claims produce three `Send`s, hence three parallel `verify_claim` instances. This is **dynamic fan-out**: the parallelism is decided at runtime from data, not written into the graph. The guard clause handles the empty case by routing straight to the reduce step.

Each spawned node runs the entire Stage 6 verifier as a subgraph:

```python
def verify_claim(state: dict) -> dict:
    result = verifier.invoke({"claim": state["claim"], "evidence": [], "search_count": 0})
    return {"verifications": [{
        "claim": state["claim"],
        "verdict": result["verdict"],
        "confidence": result["confidence"],
        "reasoning": result["reasoning"],
    }]}
```

Note the fresh `evidence: []` and `search_count: 0` per instance — that is the independent budget and the isolation described above. Note also that each returns a **one-item list**; the next section explains why.

The conditional edge declares its possible destinations so the graph can still be drawn:

```python
builder.add_conditional_edges("synthesize", dispatch_verification, ["verify_claim", "aggregate"])
builder.add_edge("verify_claim", "aggregate")     # fan-in: waits for ALL instances
```

#### The reduce step: a reducer

Now many nodes write to the *same* state key — the exact collision Stage 5 avoided by using separate keys. The default behaviour of a state field is **last write wins**, which would discard all but one verdict. A **reducer** replaces that rule:

```python
verifications: Annotated[list[dict], operator.add]
```

`Annotated[type, reducer]` attaches a merge function to a field. `operator.add` on two lists concatenates them, so parallel writes **accumulate** instead of overwriting. This is why each verifier returns a one-item list: three one-item lists reduce to one three-item list.

> ⚠ **Pitfall — a reducer applies in both directions.** Anything written to a reduced field is **appended, never substituted**. There is no way to "overwrite" a reduced field by writing to it; it merges. Stage 9 shows the trap this sets, and Stage 11 shows the persistence bug it enables.

#### A run, concretely

For the topic "the economy", a typical run produces four contested claims and therefore four parallel verifiers, each returning one record:

```python
[
  {"claim": "GDP grew 3% last quarter",        "verdict": "Supported", "confidence": 0.95, ...},
  {"claim": "Unemployment is at record lows",  "verdict": "Mixed",     "confidence": 0.62, ...},
  {"claim": "Inflation remains above target",  "verdict": "Supported", "confidence": 0.88, ...},
  {"claim": "Household debt threatens growth", "verdict": "Mixed",     "confidence": 0.55, ...},
]
```

Four one-item lists, reduced into one four-item list. The `aggregate` node then formats them:

```
=== Verification report: 'the economy' ===
- [Supported | confidence 0.95] GDP grew 3% last quarter
- [Mixed | confidence 0.62] Unemployment is at record lows
...
```

Those confidence numbers are the raw material for the next stage: two of these four fall below a trust threshold, and that is exactly what Stage 8 acts on.

### Run it

```powershell
.\.venv\Scripts\python.exe stage7_fanout.py
```

Expected: the two perspectives, then a list of contested claims, then several verifications running and reporting, then a combined report. The number of claims will vary between runs — that is the point.

**In Studio**, register `"stage7_fanout"`. The fan-out is the most striking thing in the project to watch: `verify_claim` appears once in the diagram but lights up N times in the run timeline.

> ✎ **Try it.** Change `2-4` in the synthesize prompt to `1-2` and rerun. Fewer claims, fewer verifiers, a faster and cheaper run. The one phrase in one prompt is the volume knob for the entire downstream system.

### Pitfalls

- **Forgetting the reducer.** Without `Annotated[list[dict], operator.add]`, four parallel verifiers produce one surviving result and no error message.
- **Returning a bare dict instead of a one-item list** from `verify_claim`. `operator.add` concatenates lists; give it something else and it will fail or misbehave.
- **Assuming a fixed number of claims** anywhere downstream. Everything after `synthesize` must work for N = 0, 1, or 9. Hence the empty-claims guard.

### Check your understanding

1. `dispatch_verification` can return either a string or a list. What does each mean?
2. Why does `verify_claim` return `{"verifications": [one_record]}` rather than `{"verifications": one_record}`?
3. What would happen, exactly, if `verifications` had no reducer?

---

## Stage 8 — Escalate only what is uncertain

| | |
|---|---|
| **Concept introduced** | **Confidence-threshold routing**; selective escalation |
| **Builds on** | Stage 3 (`interrupt()`) and Stage 7 (the verdict list) |
| **File** | `stage8_finale.py` |
| **Installs** | Nothing new |
| **Real data?** | As Stage 7. |

**Learning objectives.** After this stage you can: route on aggregated results; show a human only what needs attention; compile one builder two ways in one file; and deliberately force either branch for testing.

### Part 1 — The idea

At this point every disputed claim has a verdict and a confidence score. The question becomes: *who signs off?*

Sending every result to a human defeats the purpose — the system would just be generating homework. Sending none defeats the safeguard. The answer is a **threshold**: a single number, set in the code, above which results are finalised automatically and below which they are put in front of a person.

The system draws that line at **0.75**. Once all verdicts are in, a junction — a conditional edge, the same device introduced in Stage 1 — asks one question of the batch: *is there anything here below the line?* If not, the report is finalised and the run ends with nobody disturbed. If there is, the graph **stops** at an `interrupt()`, the pause mechanism from Stage 3, and presents **only the low-confidence claims** for review. There is no reason to make someone re-examine conclusions the agent was sure about.

The threshold is a dial, not a law. Raising it toward 1.0 makes the system more cautious and escalates more often; lowering it toward 0 lets everything through automatically. Where to set it is a judgement about how costly a mistake is versus how much human attention is available.

Be precise about what approval means at this stage: the human is **signing off on the agent's low-confidence verdicts** — agreeing they may stand in the final report — and that sign-off is recorded. The verdicts themselves are not altered. Approval here is a gate and an audit trail, not an edit. Stage 9 changes that.

> ▸ **Concept — escalate by uncertainty, not by policy.** "A human reviews everything" and "a human reviews nothing" are both easy to implement and both wrong. Routing on the system's own confidence spends scarce human judgement exactly where it can change the outcome. The design question is then not *whether* to involve a person but *at what threshold*.

### Part 2 — How it works

This stage reuses Stage 7's node functions wholesale and adds a new tail. The new state field is `human_decision`.

The gate is an ordinary conditional edge whose router inspects the accumulated results:

```python
CONFIDENCE_THRESHOLD = 0.75

def _low_confidence(verifications: list[dict]) -> list[dict]:
    return [v for v in verifications if v["confidence"] < CONFIDENCE_THRESHOLD]

def route_by_confidence(state: State) -> str:
    return "human_review" if _low_confidence(state.get("verifications", [])) else "finalize_report"
```

The decision is made **once for the batch**, not per claim: by the time `aggregate` has run, every verdict is in one list, so a single routing decision covers them all.

```python
builder.add_conditional_edges(
    "aggregate",
    route_by_confidence,
    {"human_review": "human_review", "finalize_report": "finalize_report"},
)
builder.add_edge("human_review", "finalize_report")
builder.add_edge("finalize_report", END)
```

Both branches converge on `finalize_report`, so there is exactly one place where the report is built.

The review node filters before pausing, so the payload contains only what needs attention:

```python
def human_review(state: State) -> dict:
    shaky = _low_confidence(state.get("verifications", []))
    decision = interrupt({
        "message": f"{len(shaky)} claim(s) scored below {CONFIDENCE_THRESHOLD}...",
        "low_confidence_claims": [
            {"claim": v["claim"], "verdict": v["verdict"],
             "confidence": v["confidence"], "reasoning": v["reasoning"]}
            for v in shaky
        ],
    })
    return {"human_decision": decision}
```

#### Two compilations, one builder

Because `interrupt()` requires persistence, this stage compiles the graph twice — bare for the visual tool, and with `SqliteSaver` for direct runs:

```python
graph = builder.compile()                                # module level, for Studio

if __name__ == "__main__":
    with SqliteSaver.from_conn_string(DB_FILE) as checkpointer:
        app = builder.compile(checkpointer=checkpointer)  # for the command line
        config = {"configurable": {"thread_id": THREAD_ID}}
```

The command-line flow is the Stage 3 two-launch pattern: start, detect `__interrupt__`, exit; then resume in a second launch with `Command(resume=…)`. In Studio none of this is needed — the dev server provides persistence and offers a resume box.

### Run it

```powershell
.\.venv\Scripts\python.exe stage8_finale.py
.\.venv\Scripts\python.exe stage8_finale.py --resume "approved"
```

What the pause looks like:

```
*** GRAPH PAUSED for human review ***
2 claim(s) scored below the 0.75 confidence threshold and need your review.
  1. [Mixed | conf 0.62] Unemployment is at record lows
     reason: Sources disagree on whether the current rate is a record.
  2. [Mixed | conf 0.55] Household debt threatens growth
     reason: Found commentary but no authoritative figures.

The paused state is SAVED on disk. Resume with:
    python stage8_finale.py --resume "approved"
```

Only two of the four claims appear. The two the agent was confident about are not shown, because nothing is being asked about them.

#### Forcing either branch

Which path a run takes depends on numbers the model produces, which makes testing awkward. The threshold constant doubles as the control:

| Setting | Effect |
|---|---|
| `CONFIDENCE_THRESHOLD = 0.99` | Almost everything counts as shaky → guarantees the human-review path. |
| `CONFIDENCE_THRESHOLD = 0.0` | Nothing counts as shaky → guarantees the automatic path. |

> ▸ **Concept — a branch that only triggers by luck is a branch that never gets tested.** Building in a deliberate way to force each path is worth the small ugliness of a tunable constant.

Delete `checkpoints_stage8.sqlite` to start a genuinely fresh run.

### The limit of this design

In `finalize_report`, the human's input touches only the closing note:

```python
report = f"=== FINAL report ...\n" + "\n".join(lines)
if decision:
    report += f"\n\nHUMAN REVIEW NOTE: {decision}"
```

Every verdict and confidence in the body is printed unchanged from the agent's output. Typing "approved" and typing "this is completely wrong" produce identical reports apart from that trailing line. The pause is real and the sign-off is recorded — but the human cannot yet *change* anything. That gap is the subject of the next stage.

### Check your understanding

1. Why is the routing decision made once for the batch rather than once per claim?
2. Why does `human_review` filter the claims before calling `interrupt()`?
3. You want to demonstrate the automatic path on a topic that usually escalates. What is the one-character-ish change?

---

## Stage 9 — Giving the human real authority

| | |
|---|---|
| **Concept introduced** | **Structured human input**; provenance annotations; the **reducer trap** |
| **Builds on** | Stages 7 and 8 |
| **File** | `stage9_human_override.py` |
| **Installs** | Nothing new |
| **Real data?** | As Stage 8. |

**Learning objectives.** After this stage you can: parse a human's answer into data; apply partial overrides by index; record provenance for every result; and explain why the corrected list must go into a new state key.

### Part 1 — The idea

A review that cannot change the outcome is a formality. Stage 9 turns the human's response from a comment into an **instruction that rewrites the results**.

The interaction is designed around a simple principle: **only name the exceptions.** The reviewer does not issue a decision for every claim. They approve the batch as a whole, or they name the specific claims they disagree with and what the verdict should be instead — a short list of corrections called an **override**. Anything not mentioned is accepted as the agent had it.

Claims are referred to by **number**, and the numbers shown at the pause are the ones to use. They count from **zero**, as positions in a list do throughout programming, so the first claim is `0` and the second is `1`. Naming a number that does not exist changes nothing rather than raising an error — forgiving, but it does mean a mistyped number silently leaves that claim approved.

That asymmetry matters. If an agent gets nine claims right and one wrong, the reviewer should type one correction, not ten confirmations. Silence is consent, and the work of reviewing scales with the number of **errors**, not the number of claims.

The final report then distinguishes three states: claims the agent was confident about (finalised automatically), claims a human looked at and let stand (**approved by human**), and claims a human overruled (**corrected by human**). The report records not just the conclusions but *how each conclusion was reached*, which is what makes it trustworthy to a reader who was not there.

> ▸ **Concept — constrain any answer that drives an action.** Stage 6 constrained the *model's* answer into a fixed shape so a program could act on it. This stage does the same to the *human's* answer. Free text is expressive but ambiguous; a structured response can be applied automatically. Whenever an answer must drive an action, constrain its shape.

### Part 2 — How it works

#### Structured human input

The resume value is parsed into a map of claim index → corrected verdict. Several input forms are accepted, because the same graph is driven from different places:

```python
def _parse_review(raw) -> dict:
    if isinstance(raw, dict):                      # a real dict (Studio / Command)
        return {int(k): str(v) for k, v in raw.items()}
    s = str(raw).strip()
    if not s or s.lower() in ("approve", "approved", "ok", "yes"):
        return {}                                  # approve everything as-is
    if s.startswith("{"):                          # a JSON string
        return {int(k): str(v) for k, v in json.loads(s).items()}
    out = {}                                       # plain "1=Refuted,3=Mixed"
    for pair in s.split(","):
        ...
    return out
```

An empty map means "no overrides". A populated map names only the claims to change.

The review node exposes the **index** of each low-confidence claim so it can be referred to:

```python
shaky = [(i, v) for i, v in enumerate(verifications)
         if v["confidence"] < CONFIDENCE_THRESHOLD]
```

These indices are **zero-based** — a claim's position in the full results list, not its position among the shaky ones. An index that matches nothing is simply ignored, leaving that claim approved.

#### Applying the decision

```python
final = []
for i, v in enumerate(verifications):
    entry = dict(v)                                     # copy; don't mutate the original
    if i in overrides:
        entry["verdict"] = overrides[i]                 # the human's verdict wins
        entry["review"] = "corrected by human"
    elif reviewed and v["confidence"] < CONFIDENCE_THRESHOLD:
        entry["review"] = "approved by human"
    else:
        entry["review"] = ""                            # auto-finalised
    final.append(entry)
```

Three provenance states, exactly as described above, carried into the report.

#### The reducer trap

The corrected list is written to a **new** state key:

```python
final_verifications: list[dict]      # NO reducer on this field
```

```python
return {"report": report, "final_verifications": final}
```

The reason is important. `verifications` carries `operator.add`. Writing the corrected list back into it would **append** the corrected records to the originals — producing eight entries where there were four — rather than replacing them. A reducer cannot be overwritten by writing to it; it merges.

> ⚠ **Pitfall — the general rule.** A reduced field is an **accumulator**. To transform its contents, write the result to a separate, un-reduced field. Reducers are for collecting; ordinary fields are for conclusions.

### Run it

Given four claims where numbers 1 and 3 fell below the threshold:

```powershell
.\.venv\Scripts\python.exe stage9_human_override.py
.\.venv\Scripts\python.exe stage9_human_override.py --resume "1=Refuted"
```

produces:

```
=== FINAL report: 'the economy' ===
- [Supported | confidence 0.95] GDP grew 3% last quarter
- [Refuted | confidence 0.62] Unemployment is at record lows        (corrected by human)
- [Supported | confidence 0.88] Inflation remains above target
- [Mixed | confidence 0.55] Household debt threatens growth         (approved by human)
```

Three things to read out of that:

- Claim 1's verdict has genuinely **changed** — the agent said Mixed, the report says Refuted.
- Claim 3, also low-confidence but not mentioned, was accepted and is marked as such.
- Claims 0 and 2 carry no annotation at all, because they never went to review.

Note that the **confidence numbers are left untouched** even where the verdict was overruled. They record what the agent thought, which remains true and worth knowing; the annotation is what records that a human disagreed. Overwriting the score would destroy the evidence that the review was needed in the first place.

To correct several claims at once, separate them with commas:

```powershell
.\.venv\Scripts\python.exe stage9_human_override.py --resume "1=Refuted,3=Mixed"
```

Accepted resume forms:

| Form | Meaning | Where it comes from |
|---|---|---|
| `approved` / `approve` / `ok` / `yes` / empty | Accept every shaky verdict as-is | Command line, Studio |
| `1=Refuted,3=Mixed` | Override claims 1 and 3; approve the rest | Command line (quote-safe) |
| `{"1": "Refuted"}` | Same, as JSON | Studio's resume box |
| A real dict | Same, as a Python object | `Command(resume={...})` in code |

### Pitfalls

> ⚠ **Windows PowerShell strips double quotes** from arguments passed to native executables, so a JSON argument such as `{"1": "Refuted"}` arrives as `{1: Refuted}` and fails to parse with a `JSONDecodeError`. This is exactly why the parser also accepts the quote-free `INDEX=VERDICT` form. Use `1=Refuted` from PowerShell and save JSON for Studio.

The wider point is a design one: **an interface should match how its input actually arrives.** A JSON box in a browser and a person typing into a Windows shell are different environments, and a tolerant parser can serve both without forcing either to adapt.

Other traps:

- **Off-by-one indices.** They are zero-based and count positions in the *full* list, not among the shaky subset.
- **A mistyped index silently approves.** No error is raised. Read the final report to confirm your correction landed.

### Check your understanding

1. Why write the corrected results to `final_verifications` instead of back into `verifications`?
2. A reviewer resumes with `"2=Refuted"` but claim 2 was above the threshold and never shown. What happens?
3. Why does the report keep the original confidence score on a claim whose verdict a human overruled?

---

## Stage 10 — Real articles

| | |
|---|---|
| **Concept introduced** | Replacing a **stub** at a clean boundary; **APIs** and **API keys** |
| **Builds on** | Stages 7–9, all imported unchanged |
| **File** | `stage10_real_articles.py` |
| **Installs** | `tavily-python` |
| **Real data?** | **Real news articles.** Verifier search still stubbed. Requires `TAVILY_API_KEY`. |

**Learning objectives.** After this stage you can: swap an implementation behind an unchanged signature; keep credentials out of source; handle empty results and messy real-world text; and pass a topic in at run time.

### Part 1 — The idea

Everything so far has run on three invented news snippets written directly into the code. That was deliberate: mechanics first, real data second.

This stage connects the system to the actual news. A topic is submitted, a search service returns several genuine recent articles about it, and those become the raw material — read by the optimist and the skeptic, mined for disputed claims, and fact-checked.

The striking thing is how little it disturbs. **One node is replaced** — `fetch_articles`, the very first station on the line. It has always had the same job: *given a topic, return a list of article texts.* Everything downstream cares only about that list, never about where it came from. So the fake source is unplugged, a real one is plugged in, and the rest of the system does not notice.

> ▸ **Concept — stubs are where a system is designed to be replaceable.** When the seam between "what this does" and "how it does it" is drawn cleanly on day one, swapping the implementation later is a contained change instead of a rewrite. A stub with an honest signature is not merely a placeholder; it is a commitment to an interface.

Reaching a service over the internet requires an **API** — a defined way for one program to ask another for something — and access requires an **API key**, a private password identifying the account being billed. Keys live in a separate `.env` file, never written into the code and never shared, because anyone holding the key can spend against the account.

The service used here is **Tavily**, a search API built for AI systems rather than for human readers. Given a query it searches the live web and returns a handful of results whose article text has already been stripped of navigation, adverts, and markup, along with each source's title and URL. It also offers a news mode that favours recent journalism over general web pages, which suits a news verifier exactly.

That cleaning step is the reason a purpose-built service beats simply downloading web pages. A raw page is mostly clutter — menus, cookie banners, related-article rails — and feeding all of that to a language model wastes both its attention and the money charged per word processed.

Tavily was also chosen for a reason that pays off one stage later: the same service and the same single key can supply the *evidence searches* the fact-checking agents need. One signup covers both halves of going live. The free allowance is around a thousand searches a month — comfortably more than this project consumes, though no longer zero. **From this stage onward, running the system costs something.**

### Part 2 — How it works

The entire change is one node:

```python
def fetch_articles(state: State) -> dict:
    response = get_tavily().search(
        query=state["topic"],
        topic="news",              # bias toward recent news
        search_depth="advanced",   # better-quality extracted content
        max_results=NUM_ARTICLES,
    )
    articles = [
        f"{r.get('title', 'Untitled')} - {r.get('content', '').strip()} "
        f"(source: {r.get('url', '')})"
        for r in response.get("results", [])
    ]
    if not articles:
        articles = [f"No recent news found for {state['topic']!r}."]
    return {"articles": articles}
```

The signature is unchanged — state in, `{"articles": [...]}` out — so the graph wiring is identical to the previous stage. Everything else is imported and reused: the pipeline nodes from Stage 7, the confidence gate from Stage 8, the state and review nodes from Stage 9.

The client follows the same lazy pattern as the LLM:

```python
_tavily = None

def get_tavily():
    global _tavily
    if _tavily is None:
        from tavily import TavilyClient
        _tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return _tavily
```

Importable without a key, replaceable in tests, constructed only when needed.

The empty-result fallback keeps the pipeline honest: rather than crashing or silently producing an empty analysis, it passes along an explicit "nothing found" article.

Each result is flattened into one string carrying the headline, the extracted text, and — importantly — the source URL:

```
Fed holds rates steady - Policymakers left the benchmark rate unchanged, citing
persistent services inflation... (source: https://example.com/fed-holds)
```

Keeping the URL inside the article text means **provenance survives every later step**: it travels into the perspectives, and a claim can in principle be traced back to the story it came from.

Two settings control the volume:

```python
NUM_ARTICLES = 4                 # how many articles to pull per run
search_depth = "advanced"        # better-quality extracted text than the default
```

More articles give the two readers richer material and produce more contested claims — and since every claim becomes its own verifier, more articles ultimately mean more searches and more cost.

The topic is no longer hardcoded:

```python
topic = sys.argv[1] if len(sys.argv) > 1 else "the economy"
```

This is the small change that turns the project from a fixed demonstration into something that can be pointed at any subject.

#### Working with real text

Real headlines contain em dashes, curly quotes, and accented characters. Printing those to a Windows console can raise an encoding error and abort a run mid-flight, so output is forced to UTF-8 at start-up:

```python
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
```

> ▸ **Concept — real input is messier than test input.** The failure this prevents has nothing to do with the logic being tested, which is exactly what makes it so annoying to debug. Going live means budgeting for a category of problem — encoding, rate limits, empty results, malformed records — that never appears while the data is invented.

### Run it

Add the key to `.env`:

```
TAVILY_API_KEY=tvly-...
```

```powershell
.\.venv\Scripts\python.exe stage10_real_articles.py "UK housing market"
.\.venv\Scripts\python.exe stage10_real_articles.py --resume "approved"
```

Expected: four genuine recent articles about the topic, followed by the familiar pipeline. The perspectives now discuss real events; the contested claims are real disputes; the verdicts are still based on canned evidence, which Stage 11 fixes.

> ⚠ **Pitfall — Stages 8–10 use a fixed `THREAD_ID`.** Running two different topics without deleting the stage's `.sqlite` file will accumulate results across runs, because of the interaction described fully in Stage 11. Delete `checkpoints_stage10.sqlite` between topics.

### Check your understanding

1. Why did replacing the data source require no change to any edge in the graph?
2. Why is the source URL embedded inside the article string rather than dropped?
3. What does `if not articles:` protect against, and why is a fallback article better than an exception?

---

## Stage 11 — Real evidence, fully live

| | |
|---|---|
| **Concept introduced** | **Dependency injection**; one run, one thread |
| **Builds on** | Everything. Reuses Stage 10's builder unchanged. |
| **File** | `stage11_real_search.py` |
| **Installs** | Nothing new |
| **Real data?** | **Fully live**: real articles, real evidence searches, real verdicts. |

**Learning objectives.** After this stage you can: swap a dependency at run time by rebinding a module-level name; explain why that works; shape a tool's output for its consumer; and avoid the persistence bug that silently merges two runs.

### Part 1 — The idea

One piece of fiction remains. The articles are real and the claims are real, but when a verifier goes looking for **evidence** it still receives canned text written into the code. The agents are investigating real claims with imaginary sources.

This stage connects the verifiers' search to the live web. From here the system is real end to end: real news in, real research on each disputed claim, real verdicts out.

What is notable is the size of the change. **One line.** Not a rewrite of the verifier, not a change to any wiring — a single statement that replaces the fake search function with a real one, everywhere it is used, all at once.

This is possible because of a habit maintained since Stage 5: components look up their dependencies *when they run*, rather than being permanently bound to them when they are built. A verifier does not contain a search tool; it *reaches for whichever search tool is in place at the moment it needs one*. Change what is in place, and every verifier — present and future — uses the new one.

The name for this arrangement is **dependency injection**: a component says what it needs, and something outside it decides what fills that need. It is the same principle that has let every test in this project substitute a fake language model for the real one.

This is also the most expensive stage to run, for a reason that traces straight back to Stage 6: every claim gets its own agent, every agent may search up to three times, and each of those searches is now a real billable request.

> ▸ **Concept — safety limits pay for themselves twice.** The retry cap existed to prevent an infinite loop. It turns out to be the thing that bounds the bill. Correctness limits and cost limits are frequently the same limit.

### Part 2 — How it works

The real tool, with a signature identical to the stub it replaces:

```python
def tavily_search(query: str) -> str:
    resp = get_tavily().search(
        query=query,
        search_depth="advanced",
        max_results=3,
        include_answer=True,       # a synthesized answer makes concise evidence
    )
    parts = []
    if resp.get("answer"):
        parts.append(f"Summary: {resp['answer']}")
    for r in resp.get("results", []):
        parts.append(f"{r.get('title', '')}: {r.get('content', '').strip()} ({r.get('url', '')})")
    return " || ".join(parts) if parts else "No results found."
```

And the swap:

```python
import stage6_verifier
stage6_verifier.get_search_result = tavily_search
```

That works because the verifier's `tools` node calls `get_search_result(query)` — a **module-level name resolved at call time**, not a reference captured when the graph was built. Rebinding the name in the module changes what every future call resolves to. No graph is rebuilt; the Stage 10 builder is reused exactly as it stands.

> ⚠ **Note the blast radius.** Because the two stages share one verifier module, importing `stage11_real_search` also upgrades Stage 10's verifier. Harmless here, and a good illustration of the trade-off: global rebinding is powerful and indiscriminate. In a larger system you would pass the tool in as a parameter instead.

The evidence a verifier now receives is a synthesised answer followed by supporting extracts, rather than Stage 6's canned sentence:

```
Summary: Real GDP increased at an annual rate of 3.0% in the quarter, per BEA. ||
Bureau of Economic Analysis: The third estimate confirms 3.0% growth... (https://bea.gov/...) ||
Reuters: Economists had expected 2.8%, making the revision modestly positive... (https://reuters.com/...)
```

`include_answer=True` is what produces that leading summary, and it matters more than it looks: an agent given three raw web extracts must spend a reasoning step working out what they collectively mean, whereas a pre-synthesised answer lets it move directly to judging the claim. **Shaping a tool's output for its consumer — here, a language model — is part of designing the tool.**

#### One run, one thread

There is a failure mode here that combines two earlier stages, and it is worth understanding because it is quiet rather than loud.

A `thread_id` identifies a persistent job (Stage 2). The `verifications` field carries a reducer that appends (Stage 7). Run two different topics under the *same* thread id and the second run's verdicts are appended to the first run's — producing a final report containing claims about a topic that was never asked about. **Nothing errors. The output is simply wrong.**

This is not hypothetical; it happened during this project's first live session. Two topics were run in succession and the second report contained claims from the first.

The fix is to give every independent run its own identity:

```python
thread_id = f"verify-{uuid.uuid4().hex[:8]}"
with open(THREAD_FILE, "w") as f:
    f.write(thread_id)                       # so a later --resume can find this run
```

The generated id is written to a small file (`.stage11_thread`) so that a subsequent `--resume` knows which paused run to continue.

> ▸ **Concept — one independent job, one fresh `thread_id`.** Reuse a thread only when continuity is the intention, such as resuming a paused run, where continuing the *same* job is precisely the point. Two independent jobs sharing a thread is not a smaller version of the same thing; it is a different thing that happens to run.

Only Stage 11 received this fix. Stages 8–10 still use a fixed `THREAD_ID` and will show the same accumulation if run on multiple topics without deleting their `.sqlite` files — which is itself instructive, because you can reproduce the bug on demand.

### Run it

```powershell
.\.venv\Scripts\python.exe stage11_real_search.py "rare earth minerals"
.\.venv\Scripts\python.exe stage11_real_search.py --resume "0=Refuted"
```

Expected: four real articles, two perspectives on them, a handful of genuinely contested claims, and one verifier per claim doing real web research with real citations in its evidence. If any verdict lands below 0.75, the run pauses for review as in Stages 8–9.

This is the finished system.

> ✎ **Try it.** Run the same topic twice, an hour apart, on a fast-moving news story. The claims and verdicts will differ. That is not a defect: the system reads what is published now, and a fact-checker that gives identical answers regardless of the news is not reading the news.

### Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| The report contains claims from a previous, different topic | Fixed thread id + accumulating reducer | Fresh `thread_id` per run (Stage 11 does this); or delete the `.sqlite` |
| `--resume` says there is nothing to resume | The thread id file was deleted, or the run completed | Start a fresh run |
| Runs are slow | Real searches, running up to `MAX_SEARCHES` per claim | Lower `MAX_SEARCHES` or `NUM_ARTICLES` |
| `KeyError: 'TAVILY_API_KEY'` | `.env` not loaded or key missing | Check `.env` is in the project root and `python-dotenv` loads it in `__main__` |

### Check your understanding

1. Why does rebinding `stage6_verifier.get_search_result` affect verifiers that were compiled before the rebinding happened?
2. Explain the two-stage interaction that causes one run's claims to appear in another run's report.
3. Why does `include_answer=True` improve the *agent's* behaviour, not just the readability of the evidence?

---

# Part III — Operating the system

## 12. Reading the output

The final report lists one line per contested claim:

```
- [Refuted | confidence 0.92] Japan's deep-sea deposit has confirmed data ...
- [Supported | confidence 0.95] China controls 70-90% of global refining capacity
- [Mixed | confidence 0.75] Billionaires advocate higher taxes while paying little   (approved by human)
```

### The verdict

| Verdict | Meaning |
|---|---|
| **Supported** | The evidence backs the claim — it appears to be **true**. |
| **Refuted** | The evidence contradicts the claim — it appears to be **false**. |
| **Mixed** | The evidence points both ways — partly true, or true only with qualification. |

Two qualifications matter when reading these.

**A verdict applies to the claim exactly as worded.** "Refuted" does not mean the underlying subject is fictional; it means that particular sentence did not hold up. Claims are frequently refuted for being *too strong* — a real trend stated with more certainty, or a tighter deadline, than the evidence supports.

**A verdict is a judgement, not a fact.** It is a conclusion drawn by a language model from a handful of search results. The accompanying confidence is the model's *own* sense of certainty, not a calibrated probability: 0.92 means "I feel quite sure", not "92% likely to be correct". This is precisely why the system routes low-confidence conclusions to a person and allows those conclusions to be overruled. **The agent does the research; the human remains the authority.**

### The annotation

| Annotation | Meaning |
|---|---|
| *(none)* | The agent was confident; no review was needed. |
| *approved by human* | A person examined a low-confidence verdict and let it stand. |
| *corrected by human* | A person overruled the agent's verdict. |

---

## 13. Cost, limits, and safe operation

### What a single live run spends

| Resource | Per run | Driver |
|---|---|---|
| Article searches | 1 | `NUM_ARTICLES` controls breadth, not count |
| LLM calls — perspectives | 2 | one per lens |
| LLM calls — claim extraction | 1 | |
| LLM calls — verifier reasoning | up to 4 per claim | `MAX_SEARCHES + 1` |
| Evidence searches | up to 3 per claim | `MAX_SEARCHES` |
| LLM calls — final report | 0–1 | |

For a typical four-claim run: roughly **10–15 search requests** and **several dozen model calls**. Comfortably inside Tavily's free monthly allowance for occasional use, and a few cents of Claude Haiku.

### The three levers

| Lever | Where | Effect |
|---|---|---|
| `MAX_SEARCHES` | `stage6_verifier.py` | The most direct control on both cost and latency. |
| The `2-4` phrase | `synthesize` prompt, `stage7_fanout.py` | Controls how many verifiers are launched at all. |
| `NUM_ARTICLES` | `stage10_real_articles.py` | Richer input, more claims, more cost. |

### Operating checklist

- [ ] One topic per run; let Stage 11 generate the `thread_id`.
- [ ] Delete a stage's `.sqlite` file if you are reusing Stages 8–10 across topics.
- [ ] Keep `.env` out of version control.
- [ ] Treat every verdict as a research lead, not a finding, and read the reasoning before acting on it.

---

# Part IV — Synthesis

## 14. Concept index

| Stage | Concept introduced | Reused by |
|---|---|---|
| 1 | State, nodes, conditional edges | everything |
| 2 | Checkpointers and threads | 3, 8–11 |
| 3 | `interrupt()` and resume | 8–11 |
| 4 | Visual inspection | all |
| 5 | Subgraphs; parallel writes to distinct keys; lazy clients | 7–11 |
| 6 | Structured output; the agent loop; retry caps | 7–11 |
| 7 | `Send` fan-out; reducers | 8–11 |
| 8 | Threshold routing; selective escalation | 9–11 |
| 9 | Structured human input; the reducer trap | 10, 11 |
| 10 | Replacing a stub at a clean boundary | 11 |
| 11 | Dependency injection; thread isolation | — |

## 15. Principles worth carrying elsewhere

**A graph makes control flow inspectable.** Branches, loops, and parallelism become structure rather than nested conditionals — visible in a diagram, testable in isolation.

**Constrain any answer that drives an action.** Whether the answer comes from a model or a person, prose invites parsing and parsing invites bugs. A declared schema turns interpretation into a field lookup.

**Every loop needs a stop that does not depend on the looper.** An agent judging its own completion will sometimes never finish. An external cap is not a fallback; it is the guarantee.

**Persistence is what makes pausing possible.** Saving state after every step is the difference between a program that must run to completion and one that can wait for a human indefinitely.

**Reducers accumulate — respect that in both directions.** Use one when parallel results must merge; write transformed results elsewhere; and never assume a fresh start on a thread that already has history.

**Escalate by uncertainty, not by policy.** Reviewing everything wastes attention and reviewing nothing forfeits the safeguard. Routing on the system's own confidence spends human judgement where it changes the outcome.

**Design the seams first.** Stubs with honest signatures made going live a two-file change. The boundary drawn on day one determined how much work day eleven required.

**Build the machinery on fake data.** Predictable, free, instant inputs are what make a complex control flow debuggable. Reality can wait until the shape is right.

## 16. Where to take it next

Natural extensions, roughly in order of effort:

| Extension | What it teaches |
|---|---|
| Cite sources in the final report | Threading provenance through every layer — the URLs are already in the state |
| Replace the hand-built loop with `bind_tools` + `ToolNode` | What the framework's higher-level abstraction buys, now that you know what it hides |
| Give the verifier a second tool (e.g. a calculator or a date checker) | Multi-tool routing, and why `Literal` needs a third value |
| Batch claims into one verifier and compare quality | Measuring the cost/quality trade-off asserted in Stage 7 |
| Swap `SqliteSaver` for a Postgres checkpointer | What "compile time, not graph time" persistence was for |
| Calibrate confidence against a labelled set | The difference between self-reported confidence and a real probability |
| Deploy the graph behind an HTTP API | Dynamic thread ids per request — the production version of Stage 11's fix |

---

# Appendix A — Glossary

| Term | Definition |
|---|---|
| **Agent** | An LLM in a loop with tools and a goal, deciding for itself when it is finished. |
| **API** | A defined way for one program to ask another for something. |
| **API key** | A private credential identifying the account being billed for API use. Kept in `.env`, never in source. |
| **Checkpoint** | A saved snapshot of the complete state, taken after each node runs. |
| **Checkpointer** | The component that writes checkpoints. This project uses `SqliteSaver`. |
| **Compile** | Turning a graph description (a builder) into something runnable. Persistence is attached here. |
| **Conditional edge** | An edge whose destination is decided at run time by a router function. |
| **Confidence** | The model's own stated certainty about its verdict; a self-report, not a calibrated probability. |
| **Config** | The dictionary carrying run identity, notably `{"configurable": {"thread_id": …}}`. |
| **Dependency injection** | A component declares what it needs by name; something outside decides what fills it. |
| **Edge** | An arrow between nodes: what happens next. |
| **Fan-out / fan-in** | Splitting work across parallel branches, then waiting for all of them to converge. |
| **Graph** | A network of nodes and edges describing the paths a program can take. |
| **Human-in-the-loop** | A design in which a person is a deliberate step in the process, not a fallback. |
| **Inference time** | Every use of a finished model — what this project pays for. Contrast **training time**. |
| **`interrupt()`** | The call that stops a graph mid-node and returns a payload to the caller. |
| **Latency / throughput / concurrency** | Time for one request / requests finished per unit time / things in flight at once. |
| **Lens** | This project's term for the one-sentence instruction that gives a perspective its angle. |
| **LLM** | Large Language Model: a text-in, text-out prediction engine. |
| **Map-reduce** | Launching one worker per item, then combining their results. |
| **Node** | One box in the graph: one unit of work, written as a plain function. |
| **Override** | A human's instruction to replace a specific claim's verdict, given by index. |
| **Prompt** | The complete block of text sent to a model. |
| **Provenance** | The record of how a conclusion was reached — here, the review annotations. |
| **Pydantic model** | A Python class declaring the exact fields, types, and rules a piece of data must satisfy. |
| **Reducer** | A merge function attached to a state field, replacing "last write wins". |
| **Retry cap** | A hard limit on loop iterations that does not depend on the agent's judgement. |
| **Router function** | A function that reads state and returns a label naming the next destination. |
| **`Send`** | The object that launches one copy of a node with its own private input; the "map" of map-reduce. |
| **State** | The shared notebook passed between nodes; the only channel of communication. |
| **Structured output** | Requiring a model to answer in a declared schema rather than prose. |
| **Stub** | A stand-in implementation with a real signature and fake behaviour. |
| **Subgraph** | A complete graph, with its own state, used as a single step inside a larger graph. |
| **Thread / `thread_id`** | One persistent job and the string that identifies it. |
| **Tool** | An ordinary function an agent may call to affect or observe the outside world. |
| **Verdict** | The agent's judgement of a claim: Supported, Refuted, or Mixed. |

---

# Appendix B — Command reference

All commands are run from the project root. On macOS/Linux replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

| Stage | Command |
|---|---|
| Setup | `python -m venv .venv` |
| Setup | `.\.venv\Scripts\python.exe -m pip install langgraph` |
| 1 | `.\.venv\Scripts\python.exe stage1_basics.py` |
| 2 | `.\.venv\Scripts\python.exe stage2_checkpointer.py` |
| 2 | `.\.venv\Scripts\python.exe stage2_inspect.py` |
| 3 | `.\.venv\Scripts\python.exe stage3_interrupt.py` |
| 3 | `.\.venv\Scripts\python.exe stage3_interrupt.py --resume "approve"` |
| 4 | `.\.venv\Scripts\langgraph.exe dev --no-reload` |
| 5 | `.\.venv\Scripts\python.exe stage5_subgraphs.py` |
| 6 | `.\.venv\Scripts\python.exe stage6_verifier.py` |
| 7 | `.\.venv\Scripts\python.exe stage7_fanout.py` |
| 8 | `.\.venv\Scripts\python.exe stage8_finale.py` |
| 8 | `.\.venv\Scripts\python.exe stage8_finale.py --resume "approved"` |
| 9 | `.\.venv\Scripts\python.exe stage9_human_override.py --resume "1=Refuted,3=Mixed"` |
| 10 | `.\.venv\Scripts\python.exe stage10_real_articles.py "UK housing market"` |
| 11 | `.\.venv\Scripts\python.exe stage11_real_search.py "rare earth minerals"` |
| 11 | `.\.venv\Scripts\python.exe stage11_real_search.py --resume "0=Refuted"` |
| Windows | `.\.venv\Scripts\python.exe sac_workaround\apply_sac_workaround.py` |

---

# Appendix C — Troubleshooting index

| Symptom | Likely cause | Fix | Stage |
|---|---|---|---|
| `ImportError: DLL load failed … Application Control policy has blocked this file` | Windows Smart App Control blocking an unsigned native wheel | Apply the pure-Python shim; **do not disable SAC** | §4.5 |
| A pasted command ends up inside a source file | Editor had keyboard focus, not the terminal | Click into the terminal panel first | §4.6 |
| Invoking a checkpointed graph raises an error about configuration | No `thread_id` supplied | Pass `{"configurable": {"thread_id": …}}` | 2 |
| A run "remembers" something it should not | Thread id reused | New `thread_id`, or delete the `.sqlite` | 2, 11 |
| Code above `interrupt()` runs twice | Expected: the node re-executes from the top on resume | Move side effects below the interrupt | 3 |
| `--resume` starts a new run instead of continuing | Different `thread_id` between launches | Use the same thread id | 3 |
| Studio drops the session repeatedly | The file-watcher is reloading the server | Start with `--no-reload` | 4 |
| Studio's Trace tab is blank | No `LANGSMITH_API_KEY`; tracing is a cloud service | Optional — local visualisation works without it | 4 |
| Studio threads vanish after restarting the server | The dev server stores threads in memory | Expected; unrelated to the `.sqlite` files | 4 |
| Studio's "Memory" panel is empty | That panel shows the long-term Store, unused here | Click a timeline step to see run state | 4 |
| A subgraph's output is labelled by the wrong node | The timeline labels by the parent field it lands in | Read the left-hand diagram for internal steps | 4, 5 |
| Only one parallel result survives | Missing reducer on the shared key | `Annotated[list[dict], operator.add]` | 7 |
| Corrected results appear *in addition to* the originals | Writing to a reduced field appends | Write to a separate un-reduced key | 9 |
| `JSONDecodeError` on a JSON `--resume` argument in PowerShell | PowerShell strips double quotes from native-exe arguments | Use the `1=Refuted` form | 9 |
| `UnicodeEncodeError` while printing real articles | Windows console encoding | `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` | 10 |
| A report contains claims about a different topic | Fixed thread id + accumulating reducer | One fresh `thread_id` per independent run | 11 |

---

# Appendix D — File and dependency map

```
stage1_basics.py            (standalone)
stage2_checkpointer.py      (standalone)  ←── stage2_inspect.py
stage3_interrupt.py         (standalone)  ←── studio_graph.py ←── langgraph.json
stage5_subgraphs.py         (standalone, first real LLM)
stage6_verifier.py          (standalone, the agent loop)
        │
        ▼
stage7_fanout.py            imports: stage5 (perspective subgraph), stage6 (verifier)
        │
        ▼
stage8_finale.py            imports: stage7 (all pipeline nodes)
        │
        ▼
stage9_human_override.py    imports: stage7 (pipeline), stage8 (threshold + gate)
        │
        ▼
stage10_real_articles.py    imports: stage7, stage8, stage9 — replaces fetch_articles
        │
        ▼
stage11_real_search.py      imports: stage10 (builder), rebinds stage6.get_search_result
                            ← THE FINISHED SYSTEM
```

Generated artefacts (safe to delete; excluded from version control): `checkpoints_stage*.sqlite`, `.stage11_thread`, `__pycache__/`, `.langgraph_api/`.

---

# Appendix E — Answer key

**Stage 1.** (1) Nodes return partial updates, which LangGraph merges into the running state; this keeps each node independent of fields it does not own. (2) A state schema lists what *may* exist over the run, not what must exist at the start; `articles` and `report` are filled in by nodes. (3) Make `fetch_articles` return fewer than two articles, so `route_after_fetch` returns `"insufficient"`.

**Stage 2.** (1) So the same builder can be compiled with persistence, without it (for Studio), or against a different backend — persistence is a deployment concern, not a graph concern. (2) It continues the first run's saved history rather than starting fresh. (3) That the run stopped partway through, and names the node it would resume at.

**Stage 3.** (1) Because the pause is a checkpoint on disk, not a process in memory; resuming means loading that checkpoint. (2) It delivers the value as the return value of the `interrupt()` call inside the node, on the same `thread_id`. (3) Below the `interrupt()` call — anything above it runs a second time on resume.

**Stage 4.** (1) The dev server supplies its own persistence; compiling with a checkpointer as well would conflict. (2) Studio's dev server keeps threads in memory, while direct command-line runs write to SQLite — two separate stores. (3) The interaction panel: click a step in the run timeline.

**Stage 5.** (1) Each `.invoke()` call gets its own private `PerspectiveState`; the compiled object holds no per-run data. (2) Because they write to different keys (`perspective_a`, `perspective_b`), so there is no collision to resolve. (3) Importing the module without an API key, and substituting a fake model in tests. (Also: not constructing a client that may never be used.)

**Stage 6.** (1) It prevents the model from returning anything other than exactly `"search"` or `"conclude"` — so the router never has to interpret prose. (2) Four times: three searches, each preceded by a reasoning step, plus the step that requests the fourth search and is refused by the cap. (3) The description text is sent to the model as part of the schema, so it functions as an instruction, not a comment.

**Stage 7.** (1) A string is a single destination label (used for the empty-claims guard); a list of `Send` objects fans out to one node instance per item. (2) Because `operator.add` concatenates lists — N one-item lists reduce to one N-item list. (3) The default "last write wins" would apply and only one verdict would survive, with no error raised.

**Stage 8.** (1) By the time `aggregate` has run, all verdicts are in a single list; one decision covers the batch and produces one pause rather than N. (2) So the person is shown only what is actually in question — reviewing confident verdicts is wasted attention. (3) Set `CONFIDENCE_THRESHOLD = 0.0`.

**Stage 9.** (1) `verifications` carries `operator.add`, so writing to it appends rather than replaces, duplicating every record. (2) Nothing — an index that matches no shaky claim is ignored, and no error is raised. (3) The score records what the agent thought, which stays true; the annotation records the disagreement. Overwriting it would erase the evidence that review was warranted.

**Stage 10.** (1) Because the node's signature never changed: state in, `{"articles": [...]}` out. Downstream nodes depend on the contract, not the implementation. (2) So provenance travels with the text through the perspectives and claims, making a claim traceable to its source. (3) An empty search result; a fallback article keeps the pipeline running and makes the "nothing found" state visible in the output rather than crashing.

**Stage 11.** (1) Because the `tools` node looks up `get_search_result` in the module *at call time*; rebinding the name changes what every subsequent call resolves to. (2) A fixed `thread_id` (Stage 2) means the second run continues the first run's saved state, and the `operator.add` reducer on `verifications` (Stage 7) appends the new verdicts to the old ones instead of replacing them. (3) It saves the agent a reasoning step: a pre-synthesised summary can be judged directly, whereas three raw extracts must first be interpreted.

---

*End of guide.*
