"""On-node nav-decode health export (viewer surface + the fallback's status hook).

WRITE-ONLY, and the exact sibling of receiver_state.py: every broker that decodes a nav
message publishes, per (signal, PRN), what it already computes on the 60 s health cadence --
whether the satellite is SYNCED, whether a full ephemeris was EXTRACTED, the BRDC-agreement
residual (`dpos_m`), and how FRESH the decode is. One JSON object per broker chain, atomically
replaced; the viewer (and, later, the decoded-eph BRDC fallback) reads the directory and merges.

Why a file, not an endpoint: the broker has no HTTP server (it only POSTs seeds to kotekan),
and the viewer already runs Python on the same host. A tmp+rename file is the same transport
receiver_state.py proved, so a reader never sees a half-written object and the broker can never
be taken down by a diagnostic riding in its control loop.

Current state only -- history is the logger's job. Absent is `null`, never 0.
"""

import json
import os

SCHEMA = 1

# Display order + the frequency band each decoded signal sits on, so the viewer groups them
# without re-deriving it. Keyed by the `signal` string the broker observes under.
SIGNAL_BAND = {
    "GPS_L1_LNAV": ("G", "L1"), "GPS_L2C_CNAV": ("G", "L2"), "GPS_L5_CNAV": ("G", "L5"),
    "GAL_E1B_INAV": ("E", "L1"), "GAL_E5AI_FNAV": ("E", "L5"),
    "BDS_B1C_BCNAV1": ("C", "L1"), "BDS_B2A_BCNAV2": ("C", "L5"),
}


class DecodeHealthWriter:
    """Accumulates per-(signal, PRN) decode health during a broker cycle, publishes one JSON.

    Failure-tolerant by construction (this rides the live seed loop): a broken writer that
    silently publishes nothing is acceptable; one that raises into the broker is an outage.
    """

    def __init__(self, path, chain, sys=None, log=None, flush_s=5.0, stale_sat_s=300.0,
                 dead_decode_s=1800.0):
        self.path = path
        self.chain = chain
        self.sys = sys
        self._log = log or (lambda m: None)
        self.flush_s = float(flush_s)
        self.stale_sat_s = float(stale_sat_s)
        self.dead_decode_s = float(dead_decode_s)
        self._last = 0.0
        # PERSISTENT current state: observe() runs on the 60 s health cadence but flush() runs
        # every broker cycle (~0.2 s), so _sats must survive flushes -- else the file is empty
        # 99% of the time. A sat not re-observed for stale_sat_s is dropped (set / decoder gone).
        self._sats = {}      # (signal, prn) -> field dict (current state)
        self._seen_t = {}    # (signal, prn) -> wall-clock t of last observe (for aging out)
        self._count = {}     # (signal, prn) -> last seen monotonic decode count
        self._decode_t = {}  # (signal, prn) -> wall-clock t when count last INCREASED
        self._fails = 0

    def observe(self, signal, prn, t_now, count=None, **fields):
        """Record one satellite's decode state for `signal`.

        `count` is any monotonic per-satellite decode counter (decoded subframes / pages /
        messages). The writer times when it last INCREASED and reports the age as `last_s` --
        that is the honest "last decode" metric (a synced-but-stalled sat reads old, not fresh).
        """
        try:
            key = (str(signal), int(prn))
            rec = self._sats.setdefault(key, {})
            rec.update(fields)
            self._seen_t[key] = float(t_now)
            if count is not None:
                prev = self._count.get(key)
                if prev is None or count > prev:
                    self._decode_t[key] = float(t_now)
                self._count[key] = count
        except Exception:
            pass

    def flush(self, t_now, force=False):
        """Atomically replace the file. Rate-limited; never raises."""
        if not self.path:
            return False
        try:
            if not force and t_now - self._last < self.flush_s:
                return False
            self._last = t_now
            # Age out a satellite when the decoder DROPS it (not re-observed within stale_sat_s)
            # OR when its decode has been frozen far longer (dead_decode_s): a set / false-lock
            # sat lingers in the decoder's PRN dict and keeps being observed with a static count,
            # so observe-based aging alone never clears it -- it would sit forever at last_s ~=
            # uptime and clutter the list. dead_decode_s is generous so a slow-but-alive weak sat
            # survives between its infrequent decodes.
            for key in list(self._seen_t):
                seen = self._seen_t.get(key, 0.0)
                dec = self._decode_t.get(key, seen)
                if (t_now - seen > self.stale_sat_s) or (t_now - dec > self.dead_decode_s):
                    for d in (self._sats, self._seen_t, self._count, self._decode_t):
                        d.pop(key, None)
            sats = []
            for (sig, prn), f in sorted(self._sats.items()):
                band = SIGNAL_BAND.get(sig, (self.sys, "?"))
                row = {"signal": sig, "sys": band[0], "band": band[1], "prn": prn}
                dt = self._decode_t.get((sig, prn))
                row["last_s"] = round(t_now - dt, 1) if dt is not None else None
                row.update(f)
                sats.append(row)
            rec = {"schema": SCHEMA, "t": round(float(t_now), 3),
                   "chain": self.chain, "sys": self.sys, "sats": sats}
            d = os.path.dirname(self.path)
            if d:
                try:
                    os.makedirs(d, exist_ok=True)
                except Exception:
                    pass
            tmp = "%s.tmp.%d" % (self.path, os.getpid())
            with open(tmp, "w") as fp:
                json.dump(rec, fp, separators=(",", ":"))
                fp.write("\n")
            os.replace(tmp, self.path)
            return True
        except Exception as e:
            self._fails += 1
            if self._fails & (self._fails - 1) == 0:  # powers of two only
                self._log("decode-health export failed (%d): %s" % (self._fails, e))
            self._sats = {}
            return False


def _read_one(path, max_age_s, t_now):
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


def read_all(dirpath, max_age_s=120.0, t_now=None):
    """Merge every fresh broker file under `dirpath` into a per-signal view for the viewer:

        {"signals": {sig: {"sys","band","n_sync","n_eph","worst_dpos_m","median_dpos_m",
                           "oldest_decode_s","sats":[row,...]}},
         "chains": [{"chain","t_age_s"}...]}

    A stale file (broker died / no fresh publish) is DROPPED, not served -- the viewer then
    shows the signal going dark, which is the honest thing on a decode outage.
    """
    import statistics
    out_sig, chains = {}, []
    try:
        names = sorted(os.listdir(dirpath))
    except Exception:
        return {"signals": {}, "chains": []}
    for n in names:
        if not n.endswith(".json"):
            continue
        rec = _read_one(os.path.join(dirpath, n), max_age_s, t_now)
        if rec is None:
            continue
        if t_now is not None:
            chains.append({"chain": rec.get("chain"),
                           "t_age_s": round(t_now - float(rec.get("t", 0.0)), 1)})
        for s in rec.get("sats", []):
            sig = s.get("signal")
            if sig is None:
                continue
            g = out_sig.setdefault(sig, {"sys": s.get("sys"), "band": s.get("band"),
                                         "sats": []})
            g["sats"].append(s)
    for sig, g in out_sig.items():
        rows = g["sats"]
        rows.sort(key=lambda r: r.get("prn") or 0)
        dpos = [r["dpos_m"] for r in rows
                if r.get("dpos_m") is not None]
        last = [r["last_s"] for r in rows if r.get("last_s") is not None]
        g["n_sats"] = len(rows)
        g["n_sync"] = sum(1 for r in rows if r.get("synced"))
        g["n_eph"] = sum(1 for r in rows if r.get("eph"))
        g["worst_dpos_m"] = max(dpos) if dpos else None
        g["median_dpos_m"] = statistics.median(dpos) if dpos else None
        # newest = the LIVENESS number (a set/false-lock straggler must not make an active chain
        # look dead); oldest kept for the straggler count.
        g["newest_decode_s"] = min(last) if last else None
        g["oldest_decode_s"] = max(last) if last else None
        g["n_stale"] = sum(1 for x in last if x > 300)
    return {"signals": out_sig, "chains": chains}
