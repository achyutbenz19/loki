"""Local contract tests — instant feedback while building. Not the gate.
The gate is 05-cs336/assignment1-basics (48 tests); this is the loop you run every 2 min.

    uv run pytest 04-tokenizer/test_bpe.py -x -q

Expects bpe.py to expose BasicTokenizer and (later) RegexTokenizer, each with
train(text, vocab_size) / encode(text) -> list[int] / decode(ids) -> str.
Everything here is a CONTRACT, not an implementation hint.
"""
from pathlib import Path

import pytest

import bpe

D = Path(__file__).parent / "data"
TORTURE = (D / "unicode_torture.txt").read_text()
CORPUS = (D / "taylorswift.txt").read_text()[:20000]

basic = pytest.mark.skipif(not hasattr(bpe, "BasicTokenizer"), reason="BasicTokenizer not written yet")
rgx = pytest.mark.skipif(not hasattr(bpe, "RegexTokenizer"), reason="RegexTokenizer not written yet")


def trained(cls, vocab_size=300, text=CORPUS):
    t = cls()
    t.train(text, vocab_size)
    return t


# ---------------------------------------------------------------- basic
@basic
@pytest.mark.parametrize("s", ["", "a", "hello world", TORTURE, "안녕하세요 👋", "   ", "\n\n\t"])
def test_roundtrip(s):
    t = trained(bpe.BasicTokenizer)
    assert t.decode(t.encode(s)) == s, "decode(encode(s)) must be identity"


@basic
def test_vocab_size_is_honoured():
    t = trained(bpe.BasicTokenizer, vocab_size=300)
    assert len(t.vocab) == 300, "256 base bytes + (vocab_size - 256) merges"
    assert len(t.merges) == 300 - 256


@basic
def test_base_bytes_present():
    t = trained(bpe.BasicTokenizer)
    for i in range(256):
        assert t.vocab[i] == bytes([i]), f"id {i} must map to the raw byte {i}"


@basic
def test_ids_in_range():
    t = trained(bpe.BasicTokenizer, vocab_size=300)
    assert all(0 <= i < 300 for i in t.encode(TORTURE))


@basic
def test_encode_is_deterministic():
    t = trained(bpe.BasicTokenizer)
    assert t.encode(CORPUS[:500]) == t.encode(CORPUS[:500])


@basic
def test_training_is_deterministic():
    a, b = trained(bpe.BasicTokenizer), trained(bpe.BasicTokenizer)
    assert a.merges == b.merges, "no randomness anywhere in BPE"


@basic
def test_it_actually_compresses():
    t = trained(bpe.BasicTokenizer, vocab_size=512)
    raw = len(CORPUS.encode("utf-8"))
    assert len(t.encode(CORPUS)) < raw * 0.75, "512 tokens should beat raw bytes by a lot"


@basic
def test_decode_survives_garbage():
    """An arbitrary id sequence can produce invalid utf-8 — must not raise."""
    t = trained(bpe.BasicTokenizer)
    t.decode([128])          # lone continuation byte
    t.decode([200, 201, 202])


# ---------------------------------------------------------------- regex
@rgx
@pytest.mark.parametrize("s", ["", "a", TORTURE, "hello world!!!? (안녕하세요!) lol123 😉"])
def test_regex_roundtrip(s):
    t = trained(bpe.RegexTokenizer)
    assert t.decode(t.encode(s)) == s


@rgx
def test_no_learned_token_spans_a_chunk_boundary():
    """The real invariant: every learned token must itself be exactly one chunk
    under the split pattern. If the pattern would break it in two, BPE was never
    allowed to build it — so `b'the '` is illegal while `b' the'` and `b"'s"` are
    both fine (the gpt4 pattern defines them as single chunks)."""
    import regex as re
    t = trained(bpe.RegexTokenizer, vocab_size=512)
    pat = bpe.Pattern.GPT4
    for i in range(256, len(t.vocab)):
        s = t.vocab[i].decode("utf-8", errors="replace")
        assert re.findall(pat, s) == [s], f"token {i}={s!r} spans a chunk boundary"


@rgx
def test_special_tokens_roundtrip():
    t = trained(bpe.RegexTokenizer)
    if not hasattr(t, "register_special_tokens"):
        pytest.skip("special tokens not implemented yet")
    t.register_special_tokens({"<|endoftext|>": 100257})
    s = "hello<|endoftext|>world"
    assert t.decode(t.encode(s, allowed_special="all")) == s
    assert 100257 in t.encode(s, allowed_special="all")
