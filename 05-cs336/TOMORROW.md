# Friday — CS336 only

State: `04-tokenizer/bpe.py` has BasicTokenizer + RegexTokenizer working, 20/20 on my own tests.
Gate: 0/48 (nothing wired yet). Everything below is about connecting what I built to what they test.

---

## 1. Read the two contracts first (10 min, do not skip)

```python
run_train_bpe(input_path, vocab_size, special_tokens) -> (dict[int,bytes], list[tuple[bytes,bytes]])
get_tokenizer(vocab, merges, special_tokens=None)     -> object with .encode() .decode() .encode_iterable()
```

Four mismatches with what I have. These are the actual work:

| mine | theirs |
|---|---|
| `train(text, ...)` — takes a string | takes a **file path** — read it in the shim |
| `merges = {(int,int): int}` | `list[tuple[bytes,bytes]]`, creation order |
| vocab_size = 256 + merges | vocab_size **includes special tokens** — 500 with 1 special = 243 merges, not 244 |
| tokenizer is *trained* | tokenizer is *constructed from given vocab+merges* — needs a load path, not train |

Plus two things I have not written at all:
- **`encode_iterable(iterable)`** — takes an iterable of strings (an open file), yields ids **lazily**. Two memory tests run it over a 5MB file; it must not materialise the whole thing.
- **specials on by default** — their tests call plain `tokenizer.encode(text)` and still expect `<|endoftext|>` to survive. My default is `allowed_special="none"`. In the adapter's tokenizer, registered specials are always active.

## 2. Order

1. **merges conversion helpers** — `{(int,int):int}` <-> `list[tuple[bytes,bytes]]`, both directions, using vocab to resolve ids to bytes. 15 min. Everything else depends on it.
2. **`run_train_bpe`** — read file, strip/split on special_tokens so no merge can contain them (`test_train_bpe_special_tokens` asserts no vocab entry contains `b"<|"`), train to `vocab_size - len(special_tokens)`, add the specials, convert merges, return. → 2 tests (speed test deferred).
3. **`get_tokenizer`** — build a RegexTokenizer from *given* state instead of training. Needs `merges` back in my `{(int,int):int}` form + `self.vocab` set directly. → most of `test_tokenizer.py`.
4. **`encode_iterable`** — generator: for each string from the iterable, yield from encode. Careful: a special token could straddle two chunks of a file read, so don't blindly encode line-by-line without thinking about boundaries.
5. **Run the gate. Record the number in PLAN.md.**

## 3. After the tokenizer block

6. **Quick wins (7 tests, ~3h):** softmax -> cross_entropy -> gradient_clipping -> AdamW -> cosine schedule -> get_batch -> checkpointing. All own concepts; own-AdamW subclassing `torch.optim.Optimizer` is the fiddly one.
7. **Model (13 tests, ~4-6h):** linear, embedding, silu, rmsnorm, swiglu, sdpa, mha, **rope**, mha+rope, block, full lm. RoPE = park risk (new math; my two hardest walls this week were both new math). If it stalls past 45 min, take the other 8 and move on.

## Deferred, on purpose
- `test_train_bpe_speed` (1.5s clock) — profiling grind, lowest learning per hour
- TinyStories run — needs a finished tokenizer + hours of wall clock
- Exercise step 3 / `matches_tiktoken` — only if the rest lands early

## Realistic target
**15-25 / 48.** Not 48. The full assignment is 14-22h against my own measured rate (~4-6h building per hour of video, six days of data). A partial green with a clear account of what the rest needs is the deliverable — same finding as the val-loss curve: the number matters less than knowing why it is what it is.
