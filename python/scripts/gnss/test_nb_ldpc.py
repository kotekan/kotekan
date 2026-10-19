#!/usr/bin/env python3
"""nb_ldpc.decode: a codeword that converges on the last iteration is returned as decoded,
and the per-matrix edge cache never serves another matrix's edges.
"""
import os
import sys
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import beidou_bcnav2 as B  # noqa: E402
import nb_ldpc as NB  # noqa: E402


def _codeword():
    """A valid B-CNAV2 codeword (n*6 bits) and its info bits (m*6)."""
    NB.init_gf()
    rng = np.random.RandomState(1)
    info = rng.randint(0, NB.Q_GF, B.N_CHECK).astype(np.uint8)
    cw = NB.encode_systematic(B.H_BCNV2_IDX, B.H_BCNV2_ELE, B.N_CHECK, B.N_VAR, info)
    return cw, cw[: B.N_CHECK * NB.N_GF]


def _decode(syms):
    return NB.decode(B.H_BCNV2_IDX, B.H_BCNV2_ELE, B.N_CHECK, B.N_VAR, syms)


class TestLastIteration(unittest.TestCase):
    def test_converged_on_last_update_is_decoded(self):
        cw, info = _codeword()
        bad = cw.copy()
        bad[7] ^= 1  # one bit error: one update repairs it
        with mock.patch.object(NB, "MAX_ITER", 1):
            bits, nerr = _decode(bad)
        self.assertEqual(nerr, 1)
        self.assertTrue(np.array_equal(bits, info))

    def test_clean_codeword_unchanged(self):
        cw, info = _codeword()
        bits, nerr = _decode(cw)
        self.assertEqual(nerr, 0)
        self.assertTrue(np.array_equal(bits, info))


class TestEdgeCache(unittest.TestCase):
    def test_same_index_matrix_other_elements(self):
        idx = ((0, 1), (1, 2))
        self.assertEqual(NB._edges(idx, ((1, 2), (3, 4)), 2, 3)[2], [1, 2, 3, 4])
        self.assertEqual(NB._edges(idx, ((5, 6), (7, 8)), 2, 3)[2], [5, 6, 7, 8])

    def test_freed_matrices_never_alias(self):
        # Matrices built at run time and dropped after use: CPython often hands the next
        # same-sized tuple the freed one's address, which an id-keyed cache must not trust.
        for k in range(20):
            rows = [[(k + j) % 3, 3 + (k + j) % 2] for j in range(2)]
            idx = tuple(tuple(r) for r in rows)
            ele = tuple(tuple(k + 1 + c for c in range(2)) for _ in range(2))
            ie, je, he = NB._edges(idx, ele, 2, 5)[:3]
            self.assertEqual(je, [c for r in rows for c in r])
            self.assertEqual(he, [k + 1, k + 2, k + 1, k + 2])
            del idx, ele


if __name__ == "__main__":
    unittest.main()
