import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "python", "scripts", "gnss")
)
import gps_nav_decode as nd  # noqa: E402


# ---- LNAV parity / word coding ----
def test_parity_roundtrip_and_chain():
    rng = np.random.default_rng(0)
    srcs = [list(rng.integers(0, 2, 24)) for _ in range(50)]
    # encode a chain, tracking the running previous-bits
    words, e29, e30 = [], 0, 0
    for src in srcs:
        word = nd.parity_encode(src, e29, e30)
        e29, e30 = int(word[28]), int(word[29])
        words.append(word)
    # decode the chain, tracking the running previous-bits, recover each source
    c29, c30 = 0, 0
    for src, word in zip(srcs, words):
        rec, ok = nd.parity_check(word, c29, c30)
        c29, c30 = int(word[28]), int(word[29])
        assert ok and list(rec) == src


def test_corrupted_word_fails_parity():
    word = nd.parity_encode([1, 0, 1, 1] + [0] * 20, 0, 0)
    word[5] ^= 1  # flip a data bit
    _, ok = nd.parity_check(word, 0, 0)
    assert not ok


def test_trim_word_resets_the_chain():
    """Words 2 and 10 are trimmed so their D29=D30=0: the next word starts a clean chain."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        d29, d30 = (int(x) for x in rng.integers(0, 2, 2))
        src = nd.trim_word(rng.integers(0, 2, 24), d29, d30)
        word = nd.parity_encode(src, d29, d30)
        assert int(word[28]) == 0 and int(word[29]) == 0
        assert nd.parity_check(word, d29, d30)[1]
