# CS336 assignment 1 — the gate

HYPOTHESIS.md says the gate is "the CS336 tests pass." This is the plan for that.

## Where it lives

Cloned separately (per README), NOT inside loki:

```
~/Projects/cs336-basics/          <- the assignment repo, its own git repo
  tests/adapters.py               <- the 21 functions I fill in (the wiring)
  cs336_basics/                   <- MY implementations go here (installed package)
  data/                           <- TinyStories (gitignored)
~/Projects/loki/05-cs336/         <- this plan + results. no code.
```

Run: `cd ~/Projects/cs336-basics && uv run pytest`

## Status (set up Aug 19)

- env solved: uv, python 3.13, torch 2.11, einops, regex, tiktoken. `uv sync` done
- **48 tests collected, 48 red** (all `NotImplementedError`) — baseline measured
- data downloading: TinyStoriesV2 train 2.07 GB + valid 0.02 GB. OpenWebText (4.3 GB gz) skipped — only needed for §7.4, which is cut
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

## Order of attack

Dependency-free — every adapter is testable alone (tests pass in reference weights). So order by value/effort, not by dependency.

**1. Quick wins (~2h, 7 tests):** softmax → cross_entropy → gradient_clipping → AdamW → cosine schedule → get_batch → checkpointing.
Why first: builds the `uv run pytest` loop, gets green on the board, and AdamW-from-scratch is the thing I just learned conceptually today.

**2. Model (~3-4h, 13 tests):** linear → embedding → silu → rmsnorm → swiglu → sdpa → mha → rope → mha+rope → block → full lm.
Why second: fresh off the GPT video. The three new pieces are RMSNorm, SwiGLU, RoPE — everything else I've written.

**3. Tokenizer (~4-6h, 28 tests):** train_bpe (correctness) → train_bpe (speed, 1.5s) → Tokenizer class (encode/decode/encode_iterable/special tokens).
Why last: biggest chunk, needs the tokenizer video first, and it's the one where the speed test bites.

**4. TinyStories run:** encode the corpus with my tokenizer to a uint16 .npy → memmap → train. Overnight, single GPU/MPS.

## Schedule (demo Fri Aug 21)

- **Tue (today)** — finish GPT video, train, consolidate gpt.py. CS336 cloned + red baseline ✅
- **Wed** — AM: quick wins (7 tests). PM: tokenizer video + train_bpe. EOD target: **~20/48**
- **Thu** — AM: Tokenizer class → 28 done. PM: model adapters. Launch TinyStories encode + training run. EOD target: **48/48**
- **Fri** — TinyStories samples in, demo + FRICTION readout
- **Sat** — buffer: whatever Thu didn't close, optional part-4 re-entry

Cut from the original plan (honest, not goalpost-moving): GPT-2 124M on 8xH100 (cost + setup risk), SFT chat model (that's WEEK2.md), OpenWebText §7.4, the leaderboard.

## Gate

`uv run pytest` green, and I can explain every component because I wrote it. Track the number here after each session:

| date | passing | note |
|---|---|---|
| Aug 19 | 0/48 | baseline, env verified |
