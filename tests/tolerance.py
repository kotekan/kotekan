"""Shared floating-point tolerances for kotekan tests.

These constants and helpers are for *numerical* comparisons (rounding error
from arithmetic on floats). They are not intended for physically-motivated
tolerances (e.g. nanoseconds of time, degrees of rotation), which should
remain local to the test that defines them.
"""

import numpy as np

# np.finfo(...).resolution is 10**ceil(log10(eps)); for float32 ~1e-6, float64 ~1e-15.
FP32_RTOL = float(np.finfo(np.float32).resolution)
FP64_RTOL = float(np.finfo(np.float64).resolution)


def float_allclose(a, b, atol=0.0):
    """Compare two float (arrays) using a tolerance derived from dtype precision.

    Uses np.finfo(...).resolution for the wider of the two inputs as rtol.
    """
    res_a = np.finfo(np.array(a).dtype).resolution
    res_b = np.finfo(np.array(b).dtype).resolution
    rtol = max(res_a, res_b)
    return np.allclose(a, b, rtol=rtol, atol=atol)
