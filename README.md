<p align="center">
  <img src="images/logo.png" alt="Open Auto Researcher" width="420">
</p>

# Open Auto Researcher

**A scaffold for building domain-specific, self-improving AI research systems.**

Open Auto Researcher is a framework for the next chapter of AI-assisted research: systems that read the literature, propose novel designs, test them against a real benchmark, and keep the ideas that actually work. It runs an evolutionary loop where an LLM rewrites its own solution generation after generation, guided by scientific papers and grounded in measured results.

This is not an attempt to remove humans from science. It is the opposite. The framework is built so that human intuition lives exactly where it matters most, and the machine handles the tireless search around it.

---

## The vision

Most of research is search. You read the field, form a hunch about what might work, build it, measure it, and repeat. The reading, building, and measuring are slow. The hunch is the valuable part.

Open Auto Researcher automates the slow loop and keeps the human at the wheel for the decisions that need taste. You pick the papers worth reading. You define what "better" means. You write the agent that proposes ideas. Then you let the system run hundreds of iterations overnight, exploring a design space far larger than any person could hand-explore, and you wake up to an archive of evaluated, ranked, fully reproducible candidates.

The result is a flywheel: papers become inspiration, inspiration becomes code, code becomes a score, and the best scores become the parents of the next generation.

---

## Humans stay in the loop

The framework deliberately exposes three seams where human judgement is required. Everything else is mechanical. You do not tune the orchestrator. You define three things, and the system does the rest.

1. **The papers.** You choose the scientific literature the proposer is allowed to draw from. This is your prior over the design space. A different reading list produces a different researcher.
2. **The evaluator.** You decide, in code, what counts as success. The evaluator is the objective function. If it is wrong, the system will optimise the wrong thing, so this is where your understanding of the problem matters most.
3. **The improvement agent.** You write the agent that turns a parent solution plus its failures plus the papers into a new candidate. This is where you encode your research instincts: what to explore, what constraints to enforce, what a good idea looks like.

These three are intentionally pluggable. To build a researcher for a new domain you implement these interfaces and drop in your reading list. Nothing in the core loop changes.

---

## How the research process is decomposed

The orchestrator in `backend/run_self_improvement.py` ties everything together. It is run-agnostic: it reads a single `RUN_NAME` from `backend/config.py` and dynamically loads the matching evaluator and improvement agent. To stand up a new research domain you create three sibling folders named after your run:

```
backend/
  scientific_papers/<run_name>/      # the reading list (PDFs + processed summaries)
  improvement_agents/<run_name>/     # the proposer that writes new candidates
  evaluators/<run_name>/             # the objective function
  baselines/<run_name>/              # the seed solution the loop starts from
```

The loop itself is a Darwin Godel Machine style search. It keeps an **archive** of every evaluated candidate (a `Node`), selects a parent by trading off accuracy against novelty, asks the improvement agent for a child, evaluates the child, and adds it back to the archive. Parent selection (`backend/utils/archive.py`) weights each node by `sigmoid(lambda * (accuracy - alpha0))` for exploitation and `1 / (1 + num_children)` for exploration, so strong-but-unexplored solutions get picked most.

The four moving parts, in the order you define them:

### 1. Choosing the scientific papers

Research starts with reading. You drop PDFs into your run's `scientific_papers/<run_name>/` folder, and `backend/scientific_papers/ingest_papers.py` extracts the text and asks an LLM to distil each one into a structured summary: a title, an overview, and a list of concrete, actionable insights.

The system prompt that steers this distillation is yours to edit. It is where you tell the reader what kind of ideas to look for:

```python
SYSTEM_PROMPT = """\
You summarise research papers on conversational and agentic memory systems.

Given the full text of a paper, produce:
  - title
  - summary
  - key_insights: concrete, actionable ideas relevant to building
    memory systems that ingest long conversations and answer questions
...
"""
```

Each paper becomes one JSON file under `processed/`. These summaries are later fed verbatim to the improvement agent as inspiration. Swap the reading list and you change what the researcher knows.

### 2. The improvement agent

The improvement agent is the proposer. Given the current best solution, its measured failures, and the paper summaries, it returns a brand new candidate solution. This is the creative core, and it implements one method:

```python
class ImprovementAgent(ABC):
    @abstractmethod
    async def propose_improvement(
        self,
        llm: OpenRouterLLM,
        parent_code: str,
        parent_accuracy: float,
        metadata: dict,
    ) -> tuple[str, str, str]:
        """Propose an improved pipeline given the parent source and eval metadata.

        Returns:
            (new_code, reasoning, novelty), all strings.
        """
```

The contract is simple: in goes the parent's source, its accuracy, and metadata (including sampled wrong answers); out comes the full source of a new candidate, plus a short explanation of the reasoning and what makes it novel. The orchestrator writes the new code to disk, evaluates it, and records the reasoning alongside the score so every step is auditable.

Inside the agent you control the prompt entirely. In the included case study, the prompt loads the processed paper summaries, formats the parent's wrong-answer samples for diagnosis, and enforces hard constraints on the design (for example, that every candidate must combine a parametric memory and a text-based external memory). This is where your research methodology becomes executable.

### 3. The evaluator

The evaluator is your objective function expressed as code. It is the single source of truth for whether a candidate is good. It implements one method:

```python
class Evaluator(ABC):
    @abstractmethod
    async def evaluate(self, pipeline_path: str, output_dir: str) -> tuple[float, dict]:
        """Load and run the pipeline at pipeline_path.

        Returns:
            (accuracy, metadata) where accuracy is a float in [0, 1] and
            metadata is a JSON-serializable dict with run-specific details
            (e.g. correct, total, wrong_samples, early_stopped).

        Raise on hard failure (syntax error, missing class, crash).
        """
```

The evaluator dynamically loads a candidate's source, runs it against your benchmark, and returns a scalar accuracy plus a metadata dict. That metadata is not just for logging: the `wrong_samples` it collects are fed straight back into the improvement agent, closing the loop between failure and the next idea. The evaluator also owns practical concerns like sampling, parallelism, and early-stopping weak candidates so compute is not wasted.

Because the evaluator raises on hard failures, broken candidates are skipped cleanly and the search keeps moving.

### 4. The agentic memory (the thing being evolved)

In any run, the artefact under evolution is a "baseline": a concrete solution to the domain problem. For the case study below the baseline is an agentic memory system. Every candidate must implement two async methods:

```python
class Baseline(ABC):
    name: str = "baseline"

    @abstractmethod
    async def ingest_conversation(
        self, conv_idx: int, entry: dict, output_dir: str,
    ) -> tuple[bool, str, int]:
        """Store one conversation in the memory backend.
        Returns (success, user_id, items_stored)."""

    @abstractmethod
    async def process_question(self, question: str, conv_idx: int) -> str:
        """Retrieve from memory and return a short answer string."""
```

`ingest_conversation` writes durable memory to disk; `process_question` runs an agentic, multi-step retrieval loop and returns an answer. The seed implementation in `backend/baselines/agentic_mem/` is a plain-text memory store with a `read_memory` tool: the agent repeatedly reads slices of the conversation transcript until it can answer. From this humble seed, the improvement agent evolves richer architectures (graph memory, embeddings, hybrid retrieval, parametric scorers) generation by generation.

---

## Case study: agentic memory on the LoCoMo benchmark

The framework ships with a complete, working research domain so you can see all four pieces in action.

The question: **how should an AI agent remember long conversations?** The benchmark is [LoCoMo](https://github.com/snap-research/locomo), a dataset of very long, multi-session dialogues with question-answer pairs spanning single-hop recall, multi-hop reasoning, temporal questions, and more. A subset lives at `backend/evaluators/agentic_mem/locomo10.json`.

The four seams, filled in:

- **Papers:** nine landmark memory papers sit in `backend/scientific_papers/agentic_mem/`, including MemGPT, Mem0, HippoRAG, A-Mem, Generative Agents, Titans, and General Agentic Memory. Their processed summaries feed the proposer.
- **Improvement agent:** `AgenticMemImprovementAgent` prompts the proposer to design memory architectures inspired by human cognition, diagnoses the parent's wrong answers, and requires every candidate to combine parametric and text-based memory.
- **Evaluator:** `AgenticMemEvaluator` samples conversations and stratified questions from LoCoMo, runs each candidate, scores answers with an LLM judge, and reports accuracy plus a sample of wrong answers for the next round.
- **Baseline:** the seed is a plain-text store with an agentic `read_memory` loop, ready to be outgrown.

Run it overnight and the archive fills with memory designs that no one wrote by hand, each one measured, ranked, and reproducible.

---

## Getting started

### Requirements

- Python 3.12
- An [OpenRouter](https://openrouter.ai) API key

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your key in `backend/.env`:

```
OPENROUTER_API_KEY=your-key-here
```

### Run the case study

```bash
cd backend

# 1. Distil the reading list into summaries (once)
python scientific_papers/ingest_papers.py

# 2. Run the self-improvement loop
python run_self_improvement.py
```

Everything for a run is written to `backend/runs/<timestamp>/`:

```
runs/<timestamp>/
  archive.json              # every node and its score
  run_<timestamp>.log
  nodes/
    root/                   # the seed baseline
    gen1_c1/
      pipeline.py           # the candidate's source
      metadata.json         # accuracy, reasoning, novelty, lineage
      diff.patch            # what changed vs. its parent
      eval_output/          # raw eval results
    ...
```

When the loop finishes it prints the best pipeline found and its accuracy. The run resumes cleanly: point `RESUME_RUN_DIR` in `backend/config.py` at an existing folder to continue where you left off.

### Tuning the search

All knobs live in `backend/config.py`:

| Setting | Meaning |
| --- | --- |
| `RUN_NAME` | Which domain to run (selects the matching evaluator, agent, baseline, papers) |
| `K_ITERATIONS` | How many evolution rounds to run |
| `CHILDREN_PER_PARENT` | Candidates proposed per selected parent |
| `RUN_TIMEOUT` | Per-evaluation timeout in seconds |
| `PROPOSER_MODEL` | LLM that writes new candidates |
| `ANSWERER_MODEL` | LLM the baseline uses at query time |
| `JUDGE_MODEL` | LLM that grades answers |
| `DGM_LAMBDA`, `DGM_ALPHA0`, `DGM_MIN_PARENT_ACCURACY` | Parent-selection trade-off between exploitation and exploration |

---

## Building your own researcher

To point the framework at a new problem:

1. Pick a `RUN_NAME`, say `protein_folding`.
2. Create `backend/scientific_papers/protein_folding/` and add the PDFs you want the proposer to learn from.
3. Implement `backend/baselines/protein_folding/` with a seed `Baseline` subclass: the starting solution.
4. Implement `backend/evaluators/protein_folding/evaluator.py` with an `Evaluator` subclass: your objective function.
5. Implement `backend/improvement_agents/protein_folding/improvement_agent.py` with an `ImprovementAgent` subclass: your proposer and its prompt.
6. Set `RUN_NAME = "protein_folding"` in `backend/config.py` and run.

The orchestrator, archive, storage, and resumption logic all stay exactly the same. You bring the taste; the framework brings the search.

---

## Project layout

```
backend/
  run_self_improvement.py        # the orchestrator (run-agnostic)
  config.py                      # all user-editable settings
  baselines/                     # seed solutions + Baseline interface
  evaluators/                    # objective functions + Evaluator interface
  improvement_agents/            # proposers + ImprovementAgent interface
  scientific_papers/             # reading lists + paper ingestion
  utils/
    archive.py                   # DGM-style archive and parent selection
    storage.py                   # run/node persistence on disk
    llm_utils.py                 # OpenRouter client
    locomo_utils.py              # dataset helpers
  runs/                          # generated run artefacts
```
