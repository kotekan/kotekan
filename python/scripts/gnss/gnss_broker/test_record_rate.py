"""The record-stream rate's sigma is the full-arc rate error, not the half-arc one.

    python3 -m gnss_broker.test_record_rate

fleet._fit_record_rate returns (rate_hz, rate_sigma_hz), with the sigma from the disagreement
of the two halves of the arc. Over many noisy tones the sigma's rms must match the rms error of
rate_hz itself; a sigma read off the half arcs overstates it about twofold.

@author Keith Vanderlinde
"""

import sys

import numpy as np

from gnss_broker.fleet import _fit_record_rate

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def test_sigma_matches_error():
    print("_fit_record_rate: rate_sigma_hz against the rate's own error")
    # 200 tones on a gap-free lattice of 256 records at 100 records/s, unit amplitude, complex
    # noise of rms 2 per record: the coherent SNR (~8) clears the noise peaks of either half.
    rng = np.random.default_rng(1)
    n, rate = 256, 100.0
    hops = list(range(n))
    t = np.arange(n) / rate
    errs, sigs = [], []
    for _ in range(200):
        f0 = rng.uniform(-10.0, 10.0)
        noise = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        z = np.exp(2j * np.pi * f0 * t) + 2.0 * noise / np.sqrt(2.0)
        f, s = _fit_record_rate(hops, z, rate)
        errs.append(f - f0)
        sigs.append(s)
    rms_err = float(np.sqrt(np.mean(np.square(errs))))
    rms_sig = float(np.sqrt(np.mean(np.square(sigs))))
    ratio = rms_sig / rms_err
    check(
        0.6 <= ratio <= 1.5,
        "rms sigma %.4f Hz against rms error %.4f Hz: ratio %.2f, inside [0.6, 1.5]"
        % (rms_sig, rms_err, ratio),
    )


def main():
    test_sigma_matches_error()
    print("\n%s (%d check(s) failed)" % ("FAIL" if _fails else "PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
