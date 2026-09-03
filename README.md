# Action-grounded reasoning synthesis for coding-agent trajectories

Code for synthesizing the reasoning that recorded coding-agent trajectories do not contain,
and for the experiments that measure whether it helps.

The corpus is on Hugging Face: **[link to be added]**.

## What is here

| Directory | Contents |
|---|---|
| `synthesis/` | Candidate generation. Writer prompting under coarse conditioning (context plus the *kind* of action, never its arguments), the hindsight control used to measure filter bias, and position selection. |
| `scoring/` | The method and its evaluation. Entity-leak gate, sufficiency scoring, group-wise ranking, ruler validation, and the harnesses for held-out thinking perplexity and action log-probability. |
| `paper_tools/` | Fact-checking gates for the paper: every citation must resolve, every reported number must recompute from a source file, and every figure must derive from data rather than hardcoded values. |
| `figures/` | Figure generation. Each figure emits a provenance record alongside its PDF. |

## Two things worth reading first

**The obvious filter selects for the wrong thing.** Keeping rationales that raise the
probability of the observed action prefers rationales written with that action already in
view, by roughly thirty to one. `scoring/entity_leak_check.py` detects the literal form of
that leakage. `scoring/score_candidates.py` and `synthesis/rank_positions.py` then rank
survivors group-wise rather than against an absolute threshold, which is what keeps the bias
from turning into a selection criterion.

**Conditioning is deliberately coarse.** The writer sees the context and the kind of the next
action but never its arguments. Withholding the action entirely leaves the writer guessing
among many continuations; supplying it in full invites transcription. `synthesis/gen_deepseek.py`
carries the prompt, and `synthesis/gen_hindsight.py` is its deliberately-leaking counterpart,
which exists to make the leakage measurable rather than to produce release data.

## Running the scripts

Each script has a docstring giving its inputs and outputs. They expect a corpus laid out as
session-level JSONL and are meant to be read and adapted rather than run as a pipeline; there
is no orchestrator.

Generation requires an OpenAI-compatible endpoint. Scoring requires a local model served
through vLLM or transformers. Neither is bundled.

## License

MIT.
