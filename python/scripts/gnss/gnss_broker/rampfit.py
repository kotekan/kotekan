"""Per-satellite ramp estimation: the least-squares slope of a standing trim series.

WHAT THIS IS FOR. #93's shadow asks whether a satellite's standing C++ trim is DRIFTING, and
at what rate -- that drift is the code-side view of the same non-dispersive error the carrier
loop sees, and the whole joint-estimator argument rests on comparing the two. It is
measurement code, and measurement code fails differently from control code: a bug here does
not crash, it produces a WRONG VERDICT, quietly, in a log line that looks exactly like a right
one. That is the worst failure mode this project has, and it is the reason this file exists
separately with tests instead of living inline in the readback loop.

⚠️ A DISCONTINUITY IS NOT A RATE. The trim gets re-anchored, wiped, and handed over (#92).
Fitting a line across one of those events measures the JUMP divided by the window, which can
exceed the real drift by orders of magnitude and carries the jump's sign. Any step over
`step_reset_chips` therefore RESTARTS the window rather than joining it. (The #92 deltas are
subtracted before they get here -- see `TrimHandover.corrected` -- so only unexplained steps
reset the fit.)

⚠️ A RELEASED TRIM'S SLOPE IS THE LEAK, NOT THE SKY. When a PRN is disarmed the C++ loop stops
integrating and the standing trim decays by its leak term. That is a beautifully clean ramp
which has nothing to do with the satellite, so a disarmed PRN's history is DROPPED, not paused.

⚠️ A SHORT WINDOW IS NOT A MEASUREMENT. Both a minimum point count and a minimum SPAN are
required: four samples spread over four seconds is a slope estimate with essentially infinite
variance, and it will happily print +0.4 chips/s. `min_span_s` is the guard that actually
binds, because the readback cadence can burst.

RELATED DISCIPLINE, learned the hard way elsewhere in this project: "the unwrap manufactured
the ramp" (#72) -- when a series is preprocessed before fitting, verify the preprocessing
cannot CREATE the trend you are about to measure. Here the preprocessing is the #92 handover
subtraction, which is why `TrimHandover.corrected` is tested for exactly that.

@author Keith Vanderlinde
"""


class RampTracker:
    """Windowed per-key linear fits of a value against time.

    Keys are whatever the caller uses (a PRN, or a (chain, prn) pair); nothing here is
    GNSS-specific beyond the trap notes above.
    """

    def __init__(self, window_s=600.0, step_reset_chips=0.3, min_points=4, min_span_s=120.0):
        self.window_s = float(window_s)
        self.step_reset_chips = float(step_reset_chips)
        self.min_points = int(min_points)
        self.min_span_s = float(min_span_s)
        self.hist = {}
        # How many times a discontinuity restarted a window. A high count against a quiet sky
        # means something is stepping the trim, and the slopes being printed are fragments.
        self.resets = 0

    def update(self, key, t, value):
        """Add a sample, resetting the window on a discontinuity and trimming to the window."""
        h = self.hist.setdefault(key, [])
        if h and abs(value - h[-1][1]) > self.step_reset_chips:
            del h[:]
            self.resets += 1
        h.append((t, value))
        while h and t - h[0][0] > self.window_s:
            h.pop(0)

    def drop(self, key):
        """Forget this key entirely -- use on release, never a pause (see the leak note)."""
        self.hist.pop(key, None)

    def retain(self, keys):
        """Drop every key not in `keys`, so a vanished satellite cannot leave a stale series
        that silently resumes hours later against a different regime."""
        for k in [k for k in self.hist if k not in keys]:
            self.hist.pop(k, None)

    def fit(self, key):
        """(slope, mean_value, span_s, n) for this key, or None if it does not yet qualify.

        Slope is per unit of `t`. The mean value is returned alongside deliberately: for a
        trim in equilibrium the leak drag is leak_per_s * trim, so keeping the mean makes that
        term computable offline instead of needing another instrument.
        """
        h = self.hist.get(key)
        if not h or len(h) < self.min_points:
            return None
        span = h[-1][0] - h[0][0]
        if span < self.min_span_s:
            return None
        n = len(h)
        tm = sum(x[0] for x in h) / n
        ym = sum(x[1] for x in h) / n
        sxx = sum((x[0] - tm) ** 2 for x in h)
        if sxx <= 0.0:
            return None
        slope = sum((x[0] - tm) * (x[1] - ym) for x in h) / sxx
        return slope, ym, span, n
