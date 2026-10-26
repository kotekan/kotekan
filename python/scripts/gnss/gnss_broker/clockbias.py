"""The receiver clock-frequency bias, and how much to trust it.

The receiver's local oscillator runs at some frequency offset, which appears as a common Doppler
shift on every satellite. Solving it is what lets a search window be narrow: the hint the
detectors receive is `predicted Doppler + this bias`. The solved value, its EMA, the drift-alarm
calibration, the time of the last measurement and the staleness flag share one invariant, so
they live on one object.

The invariant: a solved bias that nobody has measured for minutes is a liability, not a
constant. If the value latched away from truth (a GPSDO unlock walking the EMA), the narrow
search windows it centres are exactly what prevent the measurements that would correct it, so
the bias sustains itself because it is confident. Staleness must therefore widen the search,
while the held value still centres the wide hints and, under --seed-bias-source ema, the seeds
(continuity there beats freshness). `stale` is carried beside the value rather than recomputed
by each consumer.

The seeds can opt out of the solve (--seed-bias-source zero). The solve is a median of a handful
of search detections: it wanders by Hz, and on the first solve and after every stale gap it
snaps to one fresh median that can sit 10-20 Hz off. With the carrier loop open, whatever the
seeds ride is each satellite's residual carrier. On a receiver whose LO is locked to the site
reference the true offset is ~0, so seeding the solve commands its noise and nothing else.
'zero' keeps the solve for centring and widening the search and seeds 0 Hz.

@author Keith Vanderlinde
"""


import statistics


class ClockBias(object):
    """The receiver clock-frequency bias and its freshness."""

    __slots__ = (
        "value",
        "ema",
        "cal",
        "meas_t",
        "stale",
        "available",
        "seed",
        "code_ema",
        "code_hold",
        "code_repin_n",
    )

    def __init__(self, value=0.0, ema=None, cal=None, meas_t=0.0):
        # The bias used for hint centring, in Hz.
        self.value = value
        # The bias the seed consumers read, in Hz (update_seed). The hint EMA wanders by Hz with
        # the median of grid-quantized detections from a few satellites; the search window
        # tolerates that, but a seed commands it into its replica's carrier and, through
        # k = f_chip/f_carrier, its code rate, walking the chain off peak. Under 'zero' it is
        # 0.0 for the life of the process: nothing but update_seed writes it, and the broker
        # constructs this with value 0.0.
        self.seed = value
        # Smoothed estimate; None until a cycle has bias_min_sats fresh detections above
        # bias_min_snr.
        self.ema = ema
        # The drift alarm's reference, in Hz: set only by a stale-rescue re-solve on at least
        # clock_bias_cal_min_sats satellites. None until then, and the alarm is silent while None.
        self.cal = cal
        # Wall time of the last genuine multi-satellite measurement. Only a real solve refreshes
        # it, because that is the only event that says the number still describes the oscillator.
        self.meas_t = meas_t
        # Set by the staleness check each cycle; consumers read it rather than re-deriving.
        self.stale = False
        # Whether a bias is available at all this cycle (solved or held).
        self.available = False
        # The same oscillator seen on the code side, (l-a): a fractional frequency offset
        # (dimensionless, logged in ppm), seeded to every unheld satellite as its code-phase rate.
        # It fills only from this chain's own cp-fit pool, but it is a property of the receiver
        # clock, so while it is None the dead-reckon stage borrows another in-process chain's
        # through the Receiver (same band first, then any band). Without it every dead-reckon
        # seed ships code_phase_rate 0 and the prompt walks out of the correlation window in
        # minutes.
        self.code_ema = None
        # The boresight transit the code-rate clock is held through ((sep_deg, (sys, prn)), or
        # None), and the run of consecutive cycles whose refused samples agree on another rate
        # (update_code_rate).
        self.code_hold = None
        self.code_repin_n = 0

    def update_code_rate(
        self, samples, rejected, near, min_sats, max_frac, alpha, tol_frac, repin_cycles
    ):
        """One cycle of the code-rate clock (l-a): the median of this cycle's samples, EMA'd.

        samples: mature fits' (l-a) that passed the cp-rate tolerance; rejected: mature fits'
        (l-a) that the tolerance refused (seeding's sample gate); near: the transit sky's
        nearest-to-boresight (sep_deg, key) when inside the freeze cone, else None.

        Held, not updated, through a boresight transit (and not bootstrapped inside one): the
        search reports noise code phases for every satellite then, so the fits it feeds are
        slopes through noise, and this rate is what every chain's seeds ride. Updated only from min_sats samples and a median
        inside max_frac. Re-pinned to the refused samples' median when, for repin_cycles cycles
        in a row, there are too few accepted samples and at least min_sats refused ones agree
        to within tol_frac: the tolerance is measured from this very EMA, so an EMA that is
        itself wrong would otherwise refuse every honest fit for ever.

        Returns (action, raw): action is 'hold' (transit), 'update', 'repin', 'refused' (median
        beyond max_frac) or 'few'; raw the median the action used, or None."""
        if near is not None:
            self.code_hold = near
            self.code_repin_n = 0
            return "hold", None
        self.code_hold = None
        if len(samples) >= min_sats:
            self.code_repin_n = 0
            raw = statistics.median(samples)
            if abs(raw) >= max_frac:
                return "refused", raw
            self.code_ema = (
                raw
                if self.code_ema is None
                else self.code_ema + alpha * (raw - self.code_ema)
            )
            return "update", raw
        if (
            self.code_ema is not None
            and len(rejected) >= min_sats
            and max(rejected) - min(rejected) <= tol_frac
        ):
            self.code_repin_n += 1
            if self.code_repin_n >= repin_cycles:
                raw = statistics.median(rejected)
                self.code_ema = raw
                self.code_repin_n = 0
                return "repin", raw
        else:
            self.code_repin_n = 0
        return "few", None

    def update_seed(self, source):
        """Advance the seed-side bias after this cycle's multi-satellite solve.

        'ema' keeps the seed equal to the hint EMA. 'zero' ignores the solve entirely: no
        crawl, and above all no snap, which would command one few-satellite median into every
        seed at once.
        """
        if source == "zero":
            self.seed = 0.0
        else:
            self.seed = self.ema if self.ema is not None else self.value
        return self.seed
