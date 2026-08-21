# Assignment 1 — non-code deliverables checklist

The handout is code problems + WRITTEN problems + EXPERIMENT problems. Tests only cover
the code. This is the ledger of everything else, so nothing is silently dropped.

## Written answers (do in this file or a notebook)
- [ ] unicode1 (1pt) — understanding unicode
- [ ] unicode2 (3pt) — unicode encodings, why UTF-8 over UTF-16/32
- [ ] transformer_accounting (5pt) — parameter/FLOPs table for given configs
- [ ] adamw_accounting (2pt) — memory + FLOPs of training with AdamW
- [ ] learning_rate_tuning (1pt) — the SGD toy, lr 1/10/100/1000

## Tokenizer experiments (need the trained tokenizers)
- [ ] train_bpe_tinystories (2pt) — train 10K vocab on TinyStories; report longest token, does it make sense
- [ ] train_bpe_expts_owt (2pt) — 32K on OpenWebText (needs the OWT download — deferred)
- [ ] tokenizer_experiments (4pt) — compression ratios TS vs OWT, throughput estimate, encode both datasets to uint16 .npy

## Training experiments (each needs GPU time; TinyStories 17M-param scale)
- [ ] experiment_log (3pt) — the running log of every experiment + loss curves (FRICTION.md style, but for runs)
- [ ] learning_rate (3pt) — lr sweep incl. one divergent run
- [ ] batch_size_experiment (1pt) — batch 1 → as large as memory allows
- [ ] generate (1pt) — decode from the trained model, comment on fluency
- [ ] layer_norm_ablation (1pt) — remove RMSNorm, watch it hurt
- [ ] pre_norm_ablation (1pt) — post-norm variant
- [ ] no_pos_emb (1pt) — NoPE vs RoPE
- [ ] swiglu_ablation (1pt) — SwiGLU vs SiLU FFN
- [ ] main_experiment (2pt) — OWT run (deferred with OWT)
- [ ] leaderboard (6pt) — best val loss in 90 min H100 (optional, skipped)

## Deliverables that are code but have NO tests (easy to forget)
- [ ] decoding (3pt) — cs336_basics/generation.py: temperature + top-p sampling
- [ ] training_together (4pt) — cs336_basics/train.py: the full training script
