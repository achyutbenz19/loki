from enum import StrEnum
from pathlib import Path

import regex as re

VOCAB_SIZE = 276

class Pattern(StrEnum):
    """Split patterns, copied from tiktoken. A StrEnum member IS a str, so it can be
    handed straight to re.compile() — Pattern.GPT4 works anywhere the raw string does."""

    GPT2 = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    GPT4 = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def get_pattern(name="gpt4"):
    """'gpt4' / 'gpt2' (case-insensitive) -> the pattern string."""
    return Pattern[name.upper()]


# kept as module-level names too, so `from bpe import GPT4_SPLIT_PATTERN` still works
GPT2_SPLIT_PATTERN = Pattern.GPT2
GPT4_SPLIT_PATTERN = Pattern.GPT4

def get_stats(ids):
    stats = {}
    for i in zip(ids, ids[1:]):
        stats[i] = stats.get(i, 0) + 1
    return stats

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

class BasicTokenizer():
    def __init__(self, pattern=None) -> None:
        self.merges = {}
        self.vocab = {}
        self.pattern = pattern

    def train(self, text, vocab_size):
        ids = text.encode("utf-8")
        merge_size = vocab_size - 256

        for i in range(merge_size):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx

        for i in range(256):
            self.vocab[i] = bytes([i])
        
        for i, pair in enumerate(self.merges.items()):
            tuple, y = pair # ((101, 32), 256)
            self.vocab[y] = self.vocab[tuple[0]] + self.vocab[tuple[1]]

        return self.vocab, self.merges
    
    def encode(self, text):
        ids = text.encode("utf-8")
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break 
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def decode(self, ids):
        tokens = [self.vocab[id] for id in ids]
        text = b"".join(tokens)
        return text.decode("utf-8", errors="replace")

def report(tok, text, torture, vocab_size, label):
    tok.train(text, vocab_size)
    enc = tok.encode(text)
    raw = len(text.encode("utf-8"))
    print(f"\n--- {label} (vocab {vocab_size}) ---")
    print(f"vocab {len(tok.vocab)} | merges {len(tok.merges)}")
    print(f"{raw} bytes -> {len(enc)} ids | compression {raw/len(enc):.2f}X")
    print("roundtrip corpus :", tok.decode(enc) == text)
    print("roundtrip torture:", tok.decode(tok.encode(torture)) == torture)
    print("decode([128])    :", repr(tok.decode([128])))
    learned = [tok.vocab[i] for i in range(256, min(276, vocab_size))]
    print("first 20 learned :", learned)
    return tok


if __name__ == "__main__":
    # quick manual run: uv run python 04-tokenizer/bpe.py
    D = Path(__file__).parent / "data"
    text = (D / "taylorswift.txt").read_text()
    torture = (D / "unicode_torture.txt").read_text()
    print(f"{len(text)} chars, {len(text.encode('utf-8'))} utf-8 bytes")

    basic = report(BasicTokenizer(), text, torture, 276, "BasicTokenizer")

    if "RegexTokenizer" in dir():
        rgx = report(RegexTokenizer(), text, torture, 276, "RegexTokenizer (gpt4 split)")

        # the point of the split pattern: no token may span two categories
        def crossers(tok):
            out = []
            for i in range(256, len(tok.vocab)):
                t = tok.vocab[i].decode("utf-8", errors="replace")
                kinds = {"alpha" if c.isalpha() else "digit" if c.isdigit()
                         else "space" if c.isspace() else "punct" for c in t}
                kinds.discard("punct")
                if len(kinds) > 1:
                    out.append(tok.vocab[i])
            return out

        print("\n--- category crossers (should be [] for regex) ---")
        print("basic:", crossers(basic))
        print("regex:", crossers(rgx))
    else:
        print("\n(RegexTokenizer not written yet)")
