# Trained models (Lambda A100, Aug 20 2026)

## ckpt_A_baseline.pt — the one that survived

My GPT, trained on tiny shakespeare (1,115,394 chars, 65-char vocab, 90/10 split).

| | |
|---|---|
| params | 10.79M |
| n_embd / n_head / n_layer | 384 / 6 / 6 |
| block_size / batch_size | 256 / 64 |
| lr / dropout / steps | 3e-4 / 0.2 / 5000 |
| **best val** | **1.4878 @ step 4000** (karpathy: 1.48) |
| saved at | step 5000 — train 1.0536, val 1.5058 |
| hardware | 1x A100 40GB SXM4, ~10 min |

Note the checkpoint is the step-5000 state, which is slightly PAST the optimum (see runs/curve_A_baseline.csv — val bottoms at 4000 then climbs). No early stopping in the loop; that's a real thing to add.

Load it: `uv run python 03-gpt/sample.py "ROMEO:" 400`
Config must match the table above or load_state_dict throws.

## lost with the instance

- **ckpt_B_longer.pt** (10,000 steps, val 1.6686) — never fetched before terminating. Only its curve and log survived, which is all the overfitting finding needs; the weights themselves were the worse model anyway.
- **C_wider / D_deeper / E_context / F_dropout** — killed at step 1000 to save GPU time, never reached a save. Their step-1000 numbers are in FRICTION.md.

Lesson: fetch artifacts before terminating, not after deciding you want them.
