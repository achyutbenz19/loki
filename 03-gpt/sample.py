"""Load a trained checkpoint and generate. Inference only — no training.

  uv run python 03-gpt/sample.py                      # 500 chars, unprompted
  uv run python 03-gpt/sample.py "ROMEO:" 1000        # prompted, 1000 chars
"""
import sys
from pathlib import Path

import torch

from gpt import BigramLanguageModel, DEVICE, encode, decode

CKPT = Path(__file__).parent / "ckpt_A_baseline.pt"

prompt = sys.argv[1] if len(sys.argv) > 1 else None
n_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 500

model = BigramLanguageModel().to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()  # turns dropout off — sampling with it on is noticeably worse

if prompt:
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
else:
    idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)  # index 0 = newline

with torch.no_grad():
    out = model.generate(idx, n_tokens)[0].tolist()

print(decode(out))
