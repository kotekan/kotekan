"""TIME-BASE DETECTOR: is the epoch the nodes serve the epoch the sky is on?

THE FAULT THIS NAMES. Every node latches the F-engine's sample-0 epoch from chive at startup.
When the F-engine re-bases while chive still serves the old value, the nodes come back on a
stale epoch, AGREE with each other, and every consumer -- this broker, the observables writers
-- takes their word: t = frame0 + hop lands hours in the past. Nothing then complains, because
tracking is blind: the L5 search acquires by scanning, the trackers follow the sky, the data
keeps flowing -- on the wrong day, with the geometry evaluated there. Twice in one week the
tell was found offline, in the archive, a day later.

THE TELL IS ALREADY COMPUTED EVERY CYCLE. The blind search reports each satellite's measured
Doppler; the almanac predicts it from BRDC at the epoch the broker believes. Their difference is
normally ONE number for all satellites (the receiver clock frequency bias, tens of Hz) plus a
few Hz of scatter. A wrong epoch instead puts every satellite off by ITS OWN Doppler rate times
the epoch error -- a spread across satellites, not an offset, growing with |dt| at ~0.3-0.5 Hz
per second of error -- and one shift dt explains the whole spread. That is a fingerprint no
clock bias, ephemeris error or front-end fault shares.

WHAT IT DOES. Per cycle, with >= min_sats fresh detections: rms of the residuals about their
median; above spread_hz for `persist` consecutive cycles it raises the verdict, fits dt as the
least-squares slope of (residual - median) against (rate - median rate) and says how much of
the spread that one number explains. The verdict is MODULE-LEVEL, shared by every chain in the
process: an epoch is a property of the telescope, not a chain, and only the chains with a blind
search can measure it while every chain must carry the flag. It clears after `clear_after`
consecutive quiet cycles. Consumers (publish.py -> gnss_observables.py) withhold geometry while
it stands; the log line repeats no faster than an operator can act on it.
"""
import math


class Verdict(object):
    __slots__ = ("suspect", "dt_s", "explained", "spread_hz", "n", "chain", "since_t")

    def __init__(self):
        self.clear()

    def clear(self):
        self.suspect = False
        self.dt_s = None
        self.explained = None
        self.spread_hz = None
        self.n = 0
        self.chain = None
        self.since_t = None


VERDICT = Verdict()


def fit_epoch_shift(resid, rate):
    """(dt_s, pre_rms, post_rms): the shift that best explains the spread of `resid` (Hz) given
    each satellite's Doppler rate (Hz/s); both lists are per satellite, same order. The clock
    bias is the median and is removed first -- it is common, a shift is not."""
    n = len(resid)
    med_r = sorted(resid)[n // 2]
    med_a = sorted(rate)[n // 2]
    dev = [r - med_r for r in resid]
    x = [a - med_a for a in rate]
    pre = math.sqrt(sum(d * d for d in dev) / n)
    sxx = sum(v * v for v in x)
    if sxx <= 0.0:
        return None, pre, pre
    dt = sum(v * d for v, d in zip(x, dev)) / sxx
    post = math.sqrt(sum((d - dt * v) ** 2 for v, d in zip(x, dev)) / n)
    return dt, pre, post


class TimeBaseDetector(object):
    def __init__(self, min_sats=4, spread_hz=100.0, persist=3, clear_after=2, fit_frac=0.3):
        self.min_sats = int(min_sats)
        self.spread_hz = float(spread_hz)     # ~0.4 Hz/s x 250 s: below this is noise/bias
        self.persist = int(persist)
        self.clear_after = int(clear_after)
        self.fit_frac = float(fit_frac)       # post-fit rms below this fraction of the spread
        self.bad = 0                          # = "one epoch shift explains it"
        self.quiet = 0
        self.verdict = VERDICT

    def note(self, t, pairs, chain=None):
        """pairs: {prn: (resid_hz, rate_hz_per_s)} for this cycle's fresh detections.
        Returns a message when the verdict changes, else None. Too few satellites is not
        evidence either way and changes nothing."""
        if len(pairs) < self.min_sats:
            return None
        resid = [p[0] for p in pairs.values()]
        rate = [p[1] for p in pairs.values()]
        dt, pre, post = fit_epoch_shift(resid, rate)
        v = self.verdict
        if pre < self.spread_hz:
            self.bad = 0
            self.quiet += 1
            if v.suspect and self.quiet >= self.clear_after:
                msg = ("TIME BASE CLEAR (%s): %d satellites within %.0f Hz rms of the model "
                       "after %.0f s; geometry resumes" % (chain, len(pairs), pre, t - v.since_t))
                v.clear()
                return msg
            return None
        self.quiet = 0
        self.bad += 1
        explained = dt is not None and post < self.fit_frac * pre
        if self.bad < self.persist:
            return None
        first = not v.suspect
        v.suspect = True
        v.chain = chain
        v.n = len(pairs)
        v.spread_hz = pre
        v.dt_s = dt if explained else None
        v.explained = explained
        if v.since_t is None:
            v.since_t = t
        if not first:
            return None
        if explained:
            return ("*** TIME BASE SUSPECT (%s): %d satellites disagree with the model by %.0f Hz "
                    "rms after the clock bias is removed, and ONE epoch shift of %+.0f s (%+.2f h) "
                    "explains %.0f%% of it. The nodes are almost certainly serving a STALE "
                    "sample-0 epoch: refresh chive, then restart the nodes (the broker and the "
                    "writers restart themselves). Geometry is WITHHELD from every published row "
                    "until this clears."
                    % (chain, len(pairs), pre, dt, dt / 3600.0, 100.0 * (1.0 - post / pre)))
        return ("*** TIME BASE SUSPECT (%s): %d satellites disagree with the model by %.0f Hz rms "
                "after the clock bias is removed, and an epoch shift does NOT explain it "
                "(post-fit %.0f Hz): the sky and the model disagree for another reason -- "
                "ephemeris, front end, or a per-satellite fault. Geometry is WITHHELD from every "
                "published row until this clears." % (chain, len(pairs), pre, post))


DETECTOR = TimeBaseDetector()


def observe(ctx, log_rl):
    """Feed this cycle's fresh, strong detections of one chain to the shared detector."""
    pairs = {}
    for p, b in ctx.best.items():
        q = ctx.pred.get(p)
        if q is None or b[0] < ctx.args.bias_min_snr:
            continue
        if ctx.t0 - ctx.det_fresh.get(p, (None, 0.0))[1] >= ctx.args.bias_det_fresh_s:
            continue
        pairs[p] = (b[1] - q[0], q[1])
    msg = DETECTOR.note(ctx.t0, pairs, chain=getattr(ctx, "chain_id", None))
    if msg:
        log_rl("timebase", msg, every_s=60.0)
