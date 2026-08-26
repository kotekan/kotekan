"""The receiver clock-frequency bias, and how much to trust it.

WHAT IT IS. The receiver's local oscillator sits at some offset from GPS time, and that offset
appears as a common Doppler shift on every satellite. Solving it is what lets a search window
be narrow: the hint the detectors receive is `predicted Doppler + this bias`.

WHY IT IS AN OBJECT. Five names travelled together through `main()`'s namespace -- the solved
value, its EMA, the calibrated variant, the time it was last genuinely measured, and whether
that makes it stale. They are one concept and they have one invariant, and keeping them apart
meant the invariant lived nowhere.

⚠️ A SOLVED BIAS NOBODY HAS MEASURED FOR MINUTES IS A LIABILITY, NOT A CONSTANT. This is the
invariant, and it is counter-intuitive enough to have cost real sky time: if the value latched
away from truth -- as it did mid-walk during the 2026-07-20 GPSDO unlock -- then the NARROW
search windows it centres are exactly what prevent the measurements that would correct it. The
bias becomes self-sustaining precisely because it is confident.

So staleness must WIDEN the search rather than narrow it, while the value itself is still used
for seeding (continuity there beats freshness). Those two consumers want opposite things from
the same number, which is why `stale` is carried alongside the value instead of being
recomputed by whoever happens to need it.

@author Keith Vanderlinde
"""


class ClockBias(object):
    """The receiver clock-frequency bias and its freshness."""

    __slots__ = ("value", "ema", "cal", "meas_t", "stale", "available",
                 "code_ema", "code_cal")

    def __init__(self, value=0.0, ema=None, cal=None, meas_t=0.0):
        # The bias currently used for seeding and hint centring, in Hz.
        self.value = value
        # Smoothed estimate; None until enough satellites have been solved together.
        self.ema = ema
        # The calibrated variant, when a calibration source is armed.
        self.cal = cal
        # Wall time of the last genuine MULTI-SATELLITE measurement. Not the last time the
        # value was read or copied -- only a real solve refreshes this, because that is the
        # only event that says the number still describes the oscillator.
        self.meas_t = meas_t
        # Set by the staleness check each cycle; consumers read it rather than re-deriving.
        self.stale = False
        # Whether a bias is available at all this cycle (solved or held).
        self.available = False
        # ---- THE CODE-RATE CLOCK, (l-a) -------------------------------------------------
        # The same oscillator seen on the CODE side: a dimensionless rate offset (~2.6 ppm on
        # the airspy prototype, 0.02-0.10 on the CHORD GPSDO). Kept here because it is the
        # same physical clock as `value`, measured a different way.
        #
        # ⚠️ IT IS PER BAND, NOT PER CHAIN. A detector-less chain cannot measure it and must
        # borrow the band sibling's -- without that, every dead-reckon seed ships
        # code_phase_rate = 0.0 and the prompt walks out of the correlation window in 3-11
        # minutes. That was E5a's rise-peak-fall envelope with the disc railed and E >> L.
        self.code_ema = None
        self.code_cal = None

    def check_stale(self, t0, max_age_s):
        """Update and return `stale`: is the last real measurement older than max_age_s?

        A zero or negative `max_age_s` disables the check -- the value is then trusted
        indefinitely, which is a deliberate configuration choice and not an oversight.
        """
        self.stale = bool(max_age_s > 0.0 and self.ema is not None
                          and t0 - self.meas_t > max_age_s)
        return self.stale

    def age_s(self, t0):
        """Seconds since the last genuine measurement -- what the alarm line reports."""
        return t0 - self.meas_t
