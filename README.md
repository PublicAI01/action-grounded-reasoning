# Actions Without Reasons: Filling the Reasoning Gap in Real Coding-Agent Trajectories

Code and data for measuring how much reasoning real coding-agent trajectories are missing,
and for synthesizing reasoning to fill that gap without teaching the model to reason
backward from an answer it has already seen.

| | |
|---|---|
| 📄 **Paper** | Actions Without Reasons (NeurIPS 2026 Workshop) — *link to be added* |
| 🤗 **Dataset** | https://huggingface.co/datasets/publicai-dev/Trajector-2.5B |
| 📋 **Full collection (by application)** | *link to be added* |

---

## The dataset

**Trajector-2.5B** is a corpus of **26,999 real coding-agent sessions (2.50B tokens)**,
contributed by developers who authorized their sessions for publication, redacted, and
released under ODC-BY. Each session records a complete interaction: the system prompt, the
tool definitions, and the alternating sequence of user messages, assistant messages, tool
calls, and tool results, spanning six model generations.

What defines the corpus is what it lacks. Across **1.65M assistant turns** it carries only
**15.3 non-empty reasoning blocks per 100 turns**, and **34% of sessions contain no reasoning
at all**. The actions are recorded; the reasons are not. A model trained on such data learns
to imitate what was done rather than work out what to do, and closing that gap is what this
repository is about.

The public subset is one third of a collection of **88,129 consented sessions (8.1B tokens)**.
Redaction of the remaining **5.6B tokens** is in progress. Access to the full collection is by
application: *link to be added*.

> The three paragraphs above are the canonical description of the dataset. They are reproduced
> verbatim in the paper and on the dataset card; change them in one place and change them
> everywhere.

## What this code does

Reasoning cannot be recovered — the original chains never left the provider's servers — so it
has to be synthesized. Synthesis here is necessarily backward: the action is already known,
and the rationale is written to explain it. That creates the problem the method exists to
solve.

A writer sees the context and the *kind* of the next action, never its arguments. Candidates
then pass two mechanical filters. An **entity-leak gate** rejects any rationale naming an
object that appears in the action but never in the context, which is foreknowledge by
construction. A **sufficiency ranker** then picks among candidates for the same position by
how much each raises a frozen student's log-probability of the true action.

The ranking is group-wise and has no absolute threshold, and that is not an implementation
detail. Selecting rationales by whether they lead to the observed action prefers rationales
written with that action in view, by roughly thirty to one — the obvious quality filter is a
selection pressure in the wrong direction. Comparing only candidates written under identical
constraints is what avoids it.

## Repository layout

| Directory | Contents |
|---|---|
| `synthesis/` | Candidate generation. Writer prompting under coarse conditioning, the deliberately-leaking hindsight control used to make the bias measurable, and position selection. |
| `scoring/` | The method and its evaluation. Entity-leak gate, sufficiency scoring, group-wise ranking, ruler validation, and the harnesses for held-out thinking perplexity and action log-probability. |
| `paper_tools/` | Fact-checking gates for the paper: every citation must resolve, every reported number must recompute from a source file, and every figure must derive from data rather than hardcoded values. |
| `figures/` | Figure generation. Each figure emits a provenance record alongside its PDF. |

## Usage

Every script carries a docstring giving its inputs and outputs. They expect a corpus laid out
as session-level JSONL and are written to be read and adapted rather than run end to end;
there is no orchestrator.

Generation requires an OpenAI-compatible endpoint. Scoring requires a local model served
through vLLM or transformers. Neither is bundled.

```bash
pip install -r requirements.txt
```

## Citation

```bibtex
@inproceedings{wang2026actions,
  title     = {Actions Without Reasons: Filling the Reasoning Gap in Real Coding-Agent Trajectories},
  author    = {Wang, Qin},
  booktitle = {NeurIPS 2026 Workshop: Transitioning from Pre-Training to Post-Training},
  year      = {2026}
}
```

## License

Code is MIT. The corpus is released separately under ODC-BY.
