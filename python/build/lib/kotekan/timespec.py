# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import ctypes

import numpy as np


class time_spec(ctypes.Structure):
    """Struct repr of a timespec type."""

    _fields_ = [("tv", ctypes.c_int64), ("tv_nsec", ctypes.c_uint64)]

    @classmethod
    def from_float(cls, v):
        """Create a time_spec from a float.

        Parameters
        ----------------
        v : float
            The interval in seconds.

        Returns
        -------
        ts : time_spec
        """
        ts = cls()
        ts.tv = np.floor(v).astype(np.int64)
        ts.tv_nsec = ((v % 1.0) * 1e9).astype(np.int64)
        return ts

    def to_float(self):
        """
        Create a float from a time_spec.

        Returns
        -------
        float
        """
        return self.tv + self.tv_nsec / 1e9


class timeval(ctypes.Structure):
    """Struct repr of a timeval type."""

    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]
