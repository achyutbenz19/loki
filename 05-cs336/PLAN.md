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

## Order of attack (rewritten Thu night, after 04 landed)

Where I actually am: `04-tokenizer/bpe.py` has **BasicTokenizer and RegexTokenizer working** — train / encode / decode, gpt4 split pattern, verified against independently computed merges, 19/20 local tests green. Karpathy exercise steps 1-2 done. Missing: special tokens (step 4).

Every adapter is testable alone (tests supply reference weights), so this is ordered by value, not dependency.

**1. Special tokens (~45m, unblocks everything else).** Both adapter signatures require `special_tokens`, so nothing in the tokenizer suite can be wired without it. `register_special_tokens` + an `allowed_special` path in encode that splits them out BEFORE the regex, so they're never merged into.
Tests it lights up: `test_train_bpe_special_tokens`, `test_roundtrip_unicode_string_with_special_tokens`, `test_overlapping_special_tokens`, `test_encode_special_token_trailing_newlines`, `test_encode_special_token_double_newline_non_whitespace`. Let the tests define the edge cases — don't invent semantics.

**2. Wire `run_train_bpe` + `get_tokenizer` (~30m).** Thin shims over my classes. One conversion to get right: minbpe stores `merges = {(int,int): int}`, CS336 wants `list[tuple[bytes, bytes]]`. Same information, different shape. Do NOT write a second implementation.
Expect a big block of `test_tokenizer.py` to go green here.

**3. The `matches_tiktoken` tests (~2h).** Karpathy exercise step 3: recover_merges from `enc._mergeable_ranks` + the GPT-4 byte permutation. He explicitly permits copying `recover_merges`. This is where most of the remaining 25 live.

**4. Quick wins (~3h, 7 tests):** softmax → cross_entropy → gradient_clipping → AdamW → cosine schedule → get_batch → checkpointing. All concepts I own; AdamW-from-scratch is the fiddliest.

**5. Model (~4-6h, 13 tests):** linear → embedding → silu → rmsnorm → swiglu → sdpa → mha → **rope** → mha+rope → block → full lm.
RoPE is the park risk — new math, and both of my hardest walls this week (derivative definition, tensor backprop) were new math specifically, not new code. If it stalls, park it like part 4 and take the other 8.

**Explicitly deferred:** `test_train_bpe_speed` (the 1.5s clock — a profiling grind, teaches less per hour than RoPE) and the TinyStories run (needs a finished tokenizer + hours of wall clock).

## Schedule (revised Thu night)

- **Thu** ✅ tokenizer video + BasicTokenizer + RegexTokenizer, 19/20 local
- **Fri** — CS336 only, in the order above. Realistic: **15-25 / 48**, with the tokenizer block being most of it
- **Sat** — buffer + demo. Whatever Friday didn't close, plus the FRICTION readout

Honest note: 48/48 is not happening by Saturday — the full assignment is 14-22h against my own measured rate. A partial green with a clear account of what the rest needs is the deliverable, and it matches the week's other finding: the number matters less than knowing why it is what it is.

## Gate

`uv run pytest` green, and I can explain every component because I wrote it. Track the number here after each session:

| date | passing | note |
|---|---|---|
| Aug 19 | 0/48 | baseline, env verified |
