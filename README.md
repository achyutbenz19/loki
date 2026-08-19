# Loki

The LLM pipeline, implemented from scratch. 

Started Aug 14, 2026, as my one-week team learning experiment. See HYPOTHESIS.md for the goal and gate, FRICTION.md for the learning-companion research log.

## Curriculum (verified Aug 14)

Spine — Karpathy, Neural Networks: Zero to Hero (karpathy.ai/zero-to-hero.html, ~13.5h):

1. micrograd — backprop from scratch (2h25m) → `01-micrograd/`
2. makemore parts 1–5 — language modeling fundamentals (~8h) → `02-makemore/`
3. Let's build GPT: from scratch, in code, spelled out (1h56m) → `03-gpt/`
4. Let's build the GPT Tokenizer (2h13m) → `04-tokenizer/`

Verifier — Stanford CS336 `assignment1-basics` (github.com/stanford-cs336/assignment1-basics):
cloned at `~/Projects/cs336-basics` (separate repo), implementations go in its `cs336_basics/`, wired through `tests/adapters.py`. `uv run pytest` is the ground truth — 48 tests. Plan, deltas from Karpathy, and the running score: `05-cs336/PLAN.md`.

Stretch (week 2) — Let's reproduce GPT-2 (124M) (separate Karpathy video), then SFT toward a chat model.

## Rules

- Watch a segment → close the video → reimplement from memory → struggle → only then peek.
- Claude explains concepts; Claude does not write my code.
- Every stall goes in FRICTION.md before it gets fixed.
