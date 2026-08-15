# Loki — Hypothesis (Aug 14–21, 2026)

Goal: implement the LLM pipeline FROM SCRATCH — my own tokenizer (BPE), my own transformer, my own training loop, written from memory after each Zero to Hero video.

By Friday Aug 21 — EVERYTHING, one week, full-time sprint: my implementations passing the CS336 assignment-1 test suite, my model trained on TinyStories (single GPU), GPT-2 124M reproduced (8xH100 overnight), SFT'd into a chat model, demoed to the team.

**Schedule:** D1 micrograd · D2 makemore 1-2 · D3 makemore 3-4 (the ninja day) · D4 build GPT + CS336 wiring · D5 tokenizer + overnight TinyStories run · D6 CS336 all green + overnight GPT-2 run · D7 SFT + demo prep · Aug 21 demo.

**Gate:** the CS336 tests pass, and I can explain every component because I wrote it — not because I watched it.

**Anti-goalpost rule:** the schedule is itself a hackable reward — "on pace" via typing along with the video passes the milestone and learns nothing. Guard: 25-minute struggle rule before any peek; every peek logged in FRICTION.md. A slipped day with real understanding beats a met day without it.

**Verifier rule:** self-assessed understanding is a leaky reward function. The test suite is ground truth. Watch → close video → reimplement from memory → struggle → only then peek. Claude explains concepts, never writes my code — anything else is reward hacking my own experiment.

**Secondary output:** FRICTION.md — every stall, logged, as user research for the AI learning companion idea.

**Stretch (week 2 if fire holds):** reproduce GPT-2 124M, then SFT it into a chat model.
