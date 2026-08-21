# CS336 assignment 1 — the gate

HYPOTHESIS.md says the gate is "the CS336 tests pass." This is the plan for that.

## Where it lives

Inside loki, so my implementations live in my own repo with history:

```
05-cs336/
  PLAN.md                         <- this file: plan, deltas, running score
  assignment1-basics/             <- the assignment, vendored (upstream .git dropped)
    tests/adapters.py             <- the 21 functions I fill in (the wiring)
    cs336_basics/                 <- MY implementations go here (installed package)
    data/                         <- TinyStories 2.1 GB (gitignored)
    .venv/                        <- uv env (gitignored)
```

Upstream: github.com/stanford-cs336/assignment1-basics @ a158843 (Aug 19, 2026). Vendored rather than submoduled — one-shot assignment, and I want my code in my repo.

Run: `cd 05-cs336/assignment1-basics && uv sync && uv run pytest`

## Status (set up Aug 19)

- env solved: uv, python 3.13, torch 2.11, einops, regex, tiktoken. `uv sync` done
- **48 tests collected, 48 red** (all `NotImplementedError`) — baseline measured
- data: TinyStoriesV2 **valid** (21 MB) kept. The 2.1 GB train file was deleted — §7 experiments are deferred, and it's one command to get back:
  `curl -L -o data/TinyStoriesV2-GPT4-train.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt`
  OpenWebText (4.3 GB) never downloaded — only §7.4, which is cut
- the repo ships its own CLAUDE.md agent policy: explain, review, debug — never write student code. Same as my rule in HYPOTHESIS.md. No conflict.

## The 48 tests, by file

| file | tests | adapters | effort |
|---|---|---|---|
| test_tokenizer.py | 25 | `get_tokenizer` | the long pole |
| test_model.py | 13 | linear, embedding, rmsnorm, silu, swiglu, rope, sdpa, mha, mha+rope, block, lm | medium (GPT video covers ~half) |
| test_nn_utils.py | 3 | softmax, cross_entropy, gradient_clipping | quick |
| test_train_bpe.py | 3 | `run_train_bpe` | hard (speed test: 1.5s on corpus.en) |
| test_optimizer.py | 2 | `get_adamw_cls`, `run_get_lr_cosine_schedule` | quick |
| test_data.py | 1 | `run_get_batch` | quick |
| test_serialization.py | 1 | `run_save_checkpoint`, `run_load_checkpoint` | quick |

Tokenizer alone is 28/48 = 58% of the suite.

## Deltas: Karpathy's GPT vs what CS336 wants

This is the important table. The video gets me most of the way, but the spec is modern-Llama-style, not GPT-2-style. Every row is a thing I have to write that the video does NOT give me:

| component | Karpathy video | CS336 spec |
|---|---|---|
| normalization | LayerNorm | **RMSNorm** (no mean subtraction, no bias) |
| positions | learned position embedding table | **RoPE** (rotary, applied to q/k inside attention) |
| FFN | ReLU, 4*d_model | **SwiGLU**: w2(SiLU(w1 x) * w3 x), d_ff = 8/3 d_model |
| Linear | bias=True | **no bias anywhere** |
| softmax / CE | F.softmax, F.cross_entropy | **mine**, numerically stable (subtract max) |
| optimizer | torch.optim.AdamW | **my own AdamW** subclassing torch.optim.Optimizer |
| lr schedule | none | **cosine with linear warmup** |
| grad clipping | none | **required**, by global l2 norm |
| data loading | random offsets into a tensor | **np.memmap** over a uint16 token file |
| checkpointing | none | **save/load** model+optimizer+iteration |
| tokenizer | char-level | **byte-level BPE**, GPT-2 regex pretokenization, special tokens |

Shared/reused as-is: embeddings, scaled dot-product attention, causal masking, multi-head split, pre-norm residual blocks, the training loop skeleton.

## Order of attack (rewritten Thu night)

**Correction that reshaped this plan:** I had it ordered by test count, which is optimizing the VERIFIER instead of the goal — Goodhart, on the project whose whole theme is reward hacking. HYPOTHESIS says the goal is "implement the LLM pipeline FROM SCRATCH... and I can explain every component because I wrote it." The suite is the gate, not the objective.

And "from scratch" means writing the components myself in pytorch — not avoiding pytorch. gpt.py is from scratch at the ARCHITECTURE level (Head, MultiHeadAttention, Block, residual stream) but leans on nn.Linear, nn.Embedding, nn.LayerNorm, F.softmax, F.cross_entropy, torch.optim.AdamW. This assignment is the same thing one level deeper: write those too.

So the real target is my transformer rebuilt as a MODERN model with every primitive mine.

### What's actually new vs gpt.py

| gpt.py (2019-style) | here (2026-style) |
|---|---|
| LayerNorm | **RMSNorm** — no mean subtraction, no bias. The centering does nothing; dropping it is cheaper and just as good |
| ReLU FFN, 2 matrices, 4x | **SwiGLU** — 3 matrices, `w2(SiLU(w1 x) * w3 x)`, d_ff 8/3x. A learned gate, not a fixed threshold |
| learned position table | **RoPE** — rotate q/k by angle ∝ position, inside attention. Dot products then depend on RELATIVE offset, and context can be extended after training (impossible with a table) |
| nn.Linear / nn.Embedding | mine, with the spec'd init |
| F.softmax / F.cross_entropy | mine, numerically stable (subtract the max; logsumexp) |
| torch.optim.AdamW | mine — moment buffers, bias correction, decoupled decay |
| no schedule, no clipping | cosine w/ warmup + global-L2 grad clipping |
| tensor in RAM | np.memmap loader + checkpointing |

### Order (highest learning first, cheapest tiebreak)

1. **softmax + cross_entropy** (~1h, 2 tests). The content is numerical stability: exp(1000) is inf, so subtract the max first — mathematically a no-op, numerically the difference between working and NaN. I've used F.cross_entropy for six days without looking inside.
2. **AdamW + cosine schedule + grad clipping** (~2h, 3 tests). Opening the optimizer I've been driving blind.
3. **RMSNorm → SiLU → SwiGLU** (~1.5h, 3 tests). Small, and the contrast against my LayerNorm/ReLU is the lesson.
4. **Linear + Embedding** (~30m, 2 tests). Mechanical; I know both.
5. **RoPE** (~2-4h, 2 tests). The one worth the day — in every frontier model, explained well nowhere. PARK RISK: both of my hardest walls this week were new MATH (derivative definition, tensor backprop), not new code. If it stalls past ~45 min of real struggle, take the rest and come back.
6. **sdpa → mha → block → full lm** (~2h, 5 tests). Assembly; I've written all of it once already.
7. **get_batch (memmap) + checkpointing** (~1h, 2 tests). Also the pieces a TinyStories run would need.

### Tokenizer wiring — only if there's time

`04-tokenizer/bpe.py` is done (20/20 on my own tests). Wiring it to `run_train_bpe` / `get_tokenizer` is 28 tests but it's CONNECTING work, not building work — the four mismatches are: they pass a file path not a string; merges as `list[tuple[bytes,bytes]]` not `{(int,int):int}`; vocab_size INCLUDES specials (500 with 1 special = 243 merges); the tokenizer is constructed from given vocab+merges rather than trained. Plus `encode_iterable` (lazy generator, memory-tested) and specials on by default.
High test count, low learning. Do it last.

### Deferred on purpose
- if I ever do the speed test or train a tokenizer on a multi-GB corpus, the handout's §2.5 approach is: find `<|endoftext|>` boundaries in the file, hand each chunk to a separate process, merge the pre-token counts. Upstream shipped a `find_chunk_boundaries` helper for this (deleted — unrunnable snippet, and writing it myself is the point).
- `test_train_bpe_speed` (1.5s clock) — profiling grind
- TinyStories end-to-end — plumbing, not from-scratch work
- exercise step 3 / matches_tiktoken

## Status (Fri Aug 21, afternoon)

Goal unchanged: every component from scratch, mine, explained because I wrote it. The TIMELINE moved, not the goalpost — assignment 1 continues past the sprint week, with the loop restored to watch-then-build: CS336 lecture 3 (architectures: RMSNorm/SwiGLU/RoPE) -> implement from memory -> tests as judge. Lecture 2 (pytorch + resource accounting) slots in alongside. Spec-to-code without a demonstration was the broken mode, not the material.

Done so far: nn_utils complete (softmax, cross_entropy, gradient_clipping), AdamW passing. 4/48 — understood after the fact, not independently produced (asterisked in FRICTION; the lecture-first redo clears it) — the schedule function and everything after resumes lecture-first.

## Realistic target
**15-25 / 48**, weighted toward §3-5. The full assignment is 14-22h against my own measured rate (~4-6h building per hour of video, six days of logged data). A partial green with a clear account of what the rest needs is the deliverable — same finding as the val-loss curve: the number matters less than knowing why it is what it is.

## Gate

`uv run pytest` green, and I can explain every component because I wrote it. Track the number here after each session:

| date | passing | note |
|---|---|---|
| Aug 19 | 0/48 | baseline, env verified |
| Aug 20 | 0/48 | tokenizer built but not yet wired — 20/20 green on my own `04-tokenizer/test_bpe.py` |
| Aug 21 | **3/48** | nn_utils done: softmax, cross_entropy, gradient_clipping. All 18 non-tokenizer adapters wired |
