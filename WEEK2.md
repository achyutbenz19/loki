# Week 2 (Aug 22–28) — Build My Own nanochat

Precondition: week 1 done — my base model exists, CS336 tests green. Week 1 ends where Zero to Hero ends (base model); week 2 ends where nanochat ends (deployed chat model with RL). All my own code; nanochat modules are per-day answer keys.

## Schedule
- D1–2: SFT + conversation format — special tokens, loss-masking to assistant turns, chat data. → 05-chat/
- D3: inference engine — KV cache, sampling, streaming. → 06-engine/
- D4: evals harness — ARC, GSM8K, MMLU, HumanEval wiring. → 07-evals/
- D5–6: RL stage — GRPO on GSM8K, R1-style. Watch for reward hacking WITH FULL VISIBILITY this time. → 08-rl/
- D7: web UI + full end-to-end speedrun of MY pipeline. Demo #2.

## Guides (no video exists for this part — these are the map)
1. nanochat repo (~/Projects/nanochat) — readable, modular, per-stage answer key
2. Karpathy's linked write-ups in the repo: intro discussion (Oct 2025), miniseries guide (Jan 2026), "Beating GPT-2 for <<$100: the nanochat journey" (Feb 2026) — the why behind every design choice
3. "Counting r in strawberry" guide — add-a-capability workflow, final-day exercise
4. RLHF Book (rlhfbook.com) + DeepSeek R1 paper — read as an implementer, for the RL stage
5. CS336 later assignments — optional test suites if I want graded verification beyond assignment 1

## Rule carryover
Same verifier rule, same 25-min struggle rule, same FRICTION.md. Week 2 only starts if week 1's gate actually passed — no rolling forward on vibes.
