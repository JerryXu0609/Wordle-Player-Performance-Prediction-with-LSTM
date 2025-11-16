# -*- coding: utf-8 -*-
"""
Character-level encoding and optional Word2Vec training
"""
import numpy as np

def build_char_vocab(words):
    chars = sorted({ch for w in words.astype(str) for ch in w})
    special = ["<pad>", "<unk>"]
    idx2char = special + chars
    char2idx = {c:i for i,c in enumerate(idx2char)}
    return char2idx, idx2char

def words_to_char_indices(words, char2idx, max_len=5):
    seqs = []
    for w in words.astype(str):
        seq = [char2idx.get(ch, char2idx["<unk>"]) for ch in w.lower()]
        seq = seq[:max_len] + [char2idx["<pad>"]] * max(0, max_len - len(seq))
        seqs.append(seq)
    return np.array(seqs, dtype=np.int32)