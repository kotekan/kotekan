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
import os
import statistics

SCHEMA = 1


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
