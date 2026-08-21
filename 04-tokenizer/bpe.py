import regex as re
from pathlib import Path

VOCAB_SIZE = 276

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

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
    def __init__(self) -> None:
        self.merges = {}
        self.vocab = {}

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


if __name__ == "__main__":
    # quick manual run: uv run python 04-tokenizer/bpe.py
    D = Path(__file__).parent / "data"
    text = (D / "taylorswift.txt").read_text()
    torture = (D / "unicode_torture.txt").read_text()

    bt = BasicTokenizer()
    bt.train(text, 276)

    print(f"{len(text)} chars, {len(text.encode('utf-8'))} utf-8 bytes")
    print(f"vocab {len(bt.vocab)} | merges {len(bt.merges)}")
    print("learned:", [bt.vocab[i] for i in range(256, 276)])

    enc = bt.encode(text)
    print(f"encoded {len(enc)} ids | compression {len(text.encode('utf-8'))/len(enc):.2f}X")
    print("roundtrip corpus :", bt.decode(enc) == text)
    print("roundtrip torture:", bt.decode(bt.encode(torture)) == torture)
    print("decode([128])    :", repr(bt.decode([128])))
