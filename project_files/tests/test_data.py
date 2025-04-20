import pytest
import numpy as np
from src.data import one_hot_encode, reverse_complement, generate_negative_samples

def test_one_hot_encode():
    seq = "ACGTN"
    expected = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
        [0,0,0,0]
    ])
    result = one_hot_encode(seq)
    assert np.array_equal(result, expected)

def test_reverse_complement():
    assert reverse_complement("ATGCN") == "NGCAT"

def test_generate_negative_samples():
    positives = ["ATGC", "CGTA"]
    negatives = generate_negative_samples(positives)
    assert len(negatives) == len(positives)
    for original, neg in zip(positives, negatives):
        assert sorted(original) == sorted(neg)
