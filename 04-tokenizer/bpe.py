# %%
# Day 6 (Thu Aug 20) — BPE tokenizer: train, encode, decode, from memory.
# (.py like minbpe; CS336 adapters import this)

import regex as re  # NOT the stdlib `re` — need \p{L} unicode property classes

# The only paste in this file: the split patterns, copied from tiktoken the way
# karpathy does on camera. Nobody derives these.
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# differences worth noticing between the two:
#   (?i:...)  gpt4 matches 'S 'T 'LL contractions case-insensitively, gpt2 doesn't
#   \p{N}{1,3}  gpt4 caps number runs at 3 digits -> 1234 splits as 123|4
#   ++ and ?+  possessive quantifiers, no backtracking
