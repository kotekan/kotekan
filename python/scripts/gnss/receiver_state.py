"""Receiver shared-state export (S2 / Mechanism B, observer phase).

WRITE-ONLY. This module changes no estimate and no seed; it publishes what each broker
already computes internally so the quantities can be compared across chains for the first
time. `score, don't steer` -- the fusion that consumes this comes later, after a soak.

WHY THIS EXISTS
---------------
Eight brokers each independently estimate the same four physical quantities, and four of
those estimators measure the SAME per-dongle clock error by different routes:

  * `clock_bias_ema`  -- carrier side, from search Doppler minus prediction   (persisted)
  * `car_trim` median -- carrier side, from the shared carrier loop's NCO     (NOT persisted)
  * `code_bias_ema`   -- code side, the l-a slope fit                         (persisted)
  * `dr_state[drift]` -- code side, receiver-clock drift                      (NOT persisted)

Nothing has ever reconciled carrier-side against code-side, and nothing anywhere carries a
variance -- the richest quality signal in the system is a satellite COUNT, used for
thresholding, never for weighting. A covariance-weighted fuser cannot be written until the
inputs measure their own scatter, so that is what this exports.

THE ONE SUBTLETY: RAW vs SMOOTHED
---------------------------------
`clock_bias_ema` is already sibling-fused, so the agreement between two chains' persisted
.hz files is partly MANUFACTURED -- they read each other. Using that spread as a quality
metric would measure the fusion, not the estimator. So every quantity here exports the
chain's own PRE-fusion value (`raw`) and its scatter (`mad`) alongside the smoothed value
actually in use. The raw column is the honest one; the fuser will be built on it.

SCOPE KEY
---------
`dongle` names what physically shares an LO -- one airspy per band, so every chain on a
band is measuring ONE number. It is the correct fusion scope, and it is why the existing
sibling files are per-band. Do not fuse across dongles: the measured per-band offsets
(-151 / -15 / +31 Hz) are frac-N synthesis constants, not a common reference error.

FORMAT
------
One JSON object per chain, atomically replaced at `flush_s` cadence. Current state only --
history is the scorer's job, because 8 brokers appending at 1 Hz is ~200 MB/day in a cache
directory that must survive reboots. This file is also the intended transport for the
eventual fused state, so it is a schema, not scaffolding.

Absent is expressed as `null`, never as 0. (The live path's habit of `bias or 0.0` is
exactly the confusion this is meant to end.)
"""

import json
import math
import os
import statistics

SCHEMA = 1

# MAD -> sigma for a Gaussian, and sigma -> standard error of a MEDIAN (the pi/2 factor).
# Kept here beside the schema so every consumer converts identically -- two consumers with
# two conventions would silently disagree about how much to trust the same number.
MAD_TO_SIGMA = 1.4826
SE_MEDIAN = math.sqrt(math.pi / 2.0)


def se_of_median(mad, n):
    """Standard error of a median estimated from n samples with the given MAD.

    This is the weight a covariance-weighted fuser needs, and the reason `mad` and `n` are
    exported rather than a pre-baked sigma: the conversion is a convention, and baking it
    into the published record would make a later change of convention silently rewrite
    history.
    """
    if mad is None or n is None:
        return None
    try:
        n = float(n)
        if n < 2:
            return None
        return SE_MEDIAN * MAD_TO_SIGMA * float(mad) / math.sqrt(n)
    except Exception:
        return None


def mad(vals, center=None):
    """Median absolute deviation. Returns None when undefined (<2 samples).

    Not scaled to sigma here: the consumer decides. For a Gaussian, sigma ~ 1.4826*MAD and
    the standard error of the median ~ 1.253*sigma/sqrt(n) -- export the primitive, derive
    downstream, so a change of convention does not silently rewrite history.
    """
    try:
        v = [float(x) for x in vals]
        if len(v) < 2:
            return None
        c = statistics.median(v) if center is None else float(center)
        return statistics.median([abs(x - c) for x in v])
    except Exception:
        return None


class StateWriter:
    """Accumulates observations during a broker cycle, publishes one JSON object.

    Every method is failure-tolerant by construction: this is a diagnostic riding in the
    control loop of a live receiver, and it must never be able to take the node down. A
    broken observer that silently writes nothing is a bad day; one that raises into the
    seed loop is an outage.
    """

    def __init__(self, path, chain, dongle, carrier_hz, log=None, flush_s=1.0):
        self.path = path
        self.chain = chain
        self.dongle = dongle
        self.carrier_hz = float(carrier_hz) if carrier_hz else None
        self._log = log or (lambda m: None)
        self.flush_s = float(flush_s)
        self._last = 0.0
        self._obs = {}
        self._fails = 0

    # -- accumulate -------------------------------------------------------------
    def observe(self, group, **fields):
        """Record one group's fields. Unknown/None fields are kept as null on purpose."""
        try:
            self._obs.setdefault(group, {}).update(fields)
        except Exception:
            pass

    # -- publish ----------------------------------------------------------------
    def flush(self, t_now, force=False):
        """Atomically replace the state file. Rate-limited; never raises."""
        if not self.path:
            return False
        try:
            if not force and t_now - self._last < self.flush_s:
                return False
            self._last = t_now
            rec = {
                "schema": SCHEMA,
                "t": round(float(t_now), 3),
                "chain": self.chain,
                "dongle": self.dongle,
                "carrier_hz": self.carrier_hz,
            }
            rec.update(self._obs)
            self._obs = {}
            d = os.path.dirname(self.path)
            if d:
                try:
                    os.makedirs(d, exist_ok=True)
                except Exception:
                    pass
            # tmp+rename: a reader must never see a half-written object. Same-directory
            # tmp so the rename stays on one filesystem (rename is only atomic within one).
            tmp = "%s.tmp.%d" % (self.path, os.getpid())
            with open(tmp, "w") as f:
                json.dump(rec, f, separators=(",", ":"))
                f.write("\n")
            os.replace(tmp, self.path)
            return True
        except Exception as e:
            self._fails += 1
            # Rate-limited by powers of two: a permission/full-disk fault says so once,
            # then stays quiet instead of drowning the broker log it shares.
            if self._fails & (self._fails - 1) == 0:
                self._log("state export failed (%d): %s" % (self._fails, e))
            self._obs = {}
            return False


def sources_from(rec):
    """Every independent LO measurement a chain's record carries, as FRACTIONAL ppm.

    ★ WHY PPM IS THE FUSED FRAME. The carrier estimator reports Hz at its own band and the
    code estimator reports l-a in ppm, but both measure ONE fractional clock error -- which
    is why their sum in ppm vanishes (measured 0.00022-0.00080 ppm on all 8 chains). ppm is
    the frame where they are literally the same number; each chain converts back through its
    own carrier frequency. Fusing in Hz would silently mean three different quantities on a
    three-band node.

    ★ ONLY `raw_*` IS READ, NEVER the smoothed `hz`/`ppm`. The smoothed values already
    incorporate the siblings via --clock-bias-siblings, so fusing them would feed the
    estimate back on itself: the result would tighten with every pass and its covariance
    would be fiction. Raw values are independent per chain, which is what makes the
    inverse-variance weights mean anything.

    The two RESIDUAL monitors (car_trim, dr drift) are deliberately NOT sources -- measured
    at 0.000-0.113 of the primary, they are what is left after the seed already carries the
    estimate, so counting them would be double-counting the same information.
    """
    out = []
    fc = rec.get("carrier_hz")
    c = rec.get("carrier") or {}
    if fc and c.get("raw_hz") is not None:
        se = se_of_median(c.get("mad_hz"), c.get("n"))
        if se:
            out.append({"chain": rec.get("chain"), "family": "carrier",
                        "ppm": float(c["raw_hz"]) / float(fc) * 1e6,
                        "se_ppm": se / float(fc) * 1e6, "n": c.get("n")})
    d = rec.get("code") or {}
    if d.get("raw_ppm") is not None:
        se = se_of_median(d.get("mad_ppm"), d.get("n"))
        if se:
            # OPPOSITE SIGN CONVENTION: l-a is +0.0956 ppm where the carrier reads -0.0958.
            out.append({"chain": rec.get("chain"), "family": "code",
                        "ppm": -float(d["raw_ppm"]), "se_ppm": se, "n": d.get("n")})
    return out


def fuse_dongle(records, floor_ppm=0.0, reject_sigma=0.0):
    """Fuse every chain's independent measurements into ONE fractional LO estimate.

    `records` are the state records for a single dongle (one airspy, one LO, one physical
    number). Returns None when nothing usable is present -- absent, not zero.

    COVARIANCE FLOOR (`floor_ppm`): no source may claim a standard error below this. Without
    it a chain that momentarily reports a tiny MAD -- a handful of satellites that happen to
    agree -- takes essentially all the weight and captures the estimate. Measured evidence
    says no floor is needed today (pairwise |z| = 0.2 across chains on both multi-chain
    dongles, i.e. the spread is pure noise), but the floor costs nothing and the failure it
    prevents is the one that hurts every chain at once.

    REJECTION (`reject_sigma`, 0 = off): drop sources further than N sigma from the
    unrejected fit, then refit once. Reported alongside the unrejected answer so the effect
    of rejecting is always visible rather than silent.
    """
    src = []
    for r in records:
        src.extend(sources_from(r))
    if not src:
        return None

    def _fit(items):
        num = den = 0.0
        for s in items:
            se = max(float(s["se_ppm"]), float(floor_ppm)) if floor_ppm else float(s["se_ppm"])
            if se <= 0:
                continue
            w = 1.0 / (se * se)
            num += float(s["ppm"]) * w
            den += w
        if den <= 0:
            return None, None
        return num / den, math.sqrt(1.0 / den)

    ppm_all, se_all = _fit(src)
    if ppm_all is None:
        return None

    def _se(s):
        return (max(float(s["se_ppm"]), float(floor_ppm)) if floor_ppm
                else float(s["se_ppm"]))

    for s in src:
        s["rejected"] = False

    ppm, se_ = ppm_all, se_all
    n_rej = 0
    all_out = False
    if reject_sigma and len(src) > 2:
        # ★★ REJECT AGAINST A ROBUST CENTRE, NEVER THE MEAN THE OUTLIER POISONED.
        # Judging residuals against the inverse-variance mean is the classic breakdown:
        # with one bad source among three, the mean is dragged far enough that the GOOD
        # sources also exceed the threshold, the survivor list comes back EMPTY, and a
        # naive `if keep:` then rejects nothing and publishes the contaminated estimate
        # while reporting n_rejected = 0 -- "everything looks fine". That is precisely the
        # "one poisoned input hits every chain" coupling risk, occurring inside the fuser
        # meant to contain it. The median is uncontaminated by a minority of bad sources.
        centre = statistics.median([float(s["ppm"]) for s in src])
        keep = [s for s in src if abs(float(s["ppm"]) - centre) / _se(s) <= reject_sigma]
        if not keep:
            # Every source disagrees with the robust centre: pathological, and NOT
            # something to paper over by silently keeping everything. Publish the
            # unrejected fit but say so loudly.
            all_out = True
        elif len(keep) < len(src):
            p2, s2 = _fit(keep)
            if p2 is not None:
                kept = {id(s) for s in keep}
                for s in src:
                    s["rejected"] = id(s) not in kept
                n_rej = len(src) - len(keep)
                ppm, se_ = p2, s2

    # per-source residual against the PUBLISHED estimate -- THE health monitor
    # (mitigation #1), reported for every source whether or not rejection is enabled.
    for s in src:
        s["resid_ppm"] = float(s["ppm"]) - ppm
        s["resid_sigma"] = (abs(s["resid_ppm"]) / _se(s)) if _se(s) > 0 else None

    return {
        "lo_ppm": ppm,
        "se_ppm": se_,
        "lo_ppm_norej": ppm_all,
        "n_src": len(src),
        "n_rejected": n_rej,
        "all_outliers": all_out,
        "n_carrier": sum(1 for s in src if s["family"] == "carrier" and not s["rejected"]),
        "n_code": sum(1 for s in src if s["family"] == "code" and not s["rejected"]),
        "chains": sorted({s["chain"] for s in src if s["chain"] and not s["rejected"]}),
        "worst_sigma": max([s["resid_sigma"] for s in src
                            if s["resid_sigma"] is not None and not s["rejected"]] or [None]),
        "sources": src,
    }


def read_state(path, max_age_s=None, t_now=None):
    """Read one chain's state. Returns None if absent, malformed, or stale.

    Staleness is a REFUSAL, not a fallback -- an old LO estimate is a different epoch's
    number, and the one estimator in this repo that gets this right (gnss_tec's `med_at`)
    returns None rather than serving an unsupported value.
    """
    try:
        with open(path) as f:
            rec = json.load(f)
        if not isinstance(rec, dict) or rec.get("schema") != SCHEMA:
            return None
        if max_age_s is not None and t_now is not None:
            if t_now - float(rec.get("t", 0.0)) > max_age_s:
                return None
        return rec
    except Exception:
        return None


def read_dongle(dirpath, dongle, max_age_s=None, t_now=None, exclude=None):
    """All fresh state records for one dongle (== one LO). The fusion scope."""
    out = []
    try:
        names = sorted(os.listdir(dirpath))
    except Exception:
        return out
    for n in names:
        if not n.endswith(".json"):
            continue
        rec = read_state(os.path.join(dirpath, n), max_age_s=max_age_s, t_now=t_now)
        if rec is None or rec.get("dongle") != dongle:
            continue
        if exclude is not None and rec.get("chain") == exclude:
            continue
        out.append(rec)
    return out
