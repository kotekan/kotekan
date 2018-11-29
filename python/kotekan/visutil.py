"""Miscellaneous utilities.
"""
# Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import (ascii, bytes, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      str, super, zip)


class prod_ctype(object):
    """Store a product type."""

    def __init__(self, ipa, ipb):
        # Index of input A
        self.input_a = ipa
        # Index of input B
        self.input_b = ipb


def cmap(i, j, n):
    """Map row column to triangular index."""
    return (n * (n + 1) // 2) - ((n - i) * (n - i + 1) // 2) + (j - i)


def icmap(k, n):
    """Map from an upper triangular index to row column."""
    ii = 0
    for ii in range(n):
        if cmap(ii, n - 1, n) >= k:
            break

    j = k - cmap(ii, ii, n) + ii
    return prod_ctype(ii, j)


def ts_to_double(ts):
    """Calculate a double precision UNIX time from a timespec."""
    return ts.tv + 1e-9 * ts.tv_nsec


