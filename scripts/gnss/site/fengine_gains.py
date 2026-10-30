#!/usr/bin/env python3
"""Per-channel F-engine gain requests for the GNSS lobes: bandpass-shaped and time-variable.

The F-engine requantises every channel to 4+4 bits, each component rounded and clipped to +-7.
A channel whose power rises (a satellite near the beam, a GuoWang burst, the GNSS ensemble's
glow) rails; one cut too deep loses sensitivity to quantisation. This holds every managed
channel's output at a target rms by asking the F-engine for a per-channel voltage multiplier on
its equalised quiet-sky table (the baseline), from the measured output power, and cuts further
while a channel is bright.

    fengine_gains.py run    [--endpoints URL ...] --out DIR [--port N] [--push URL]
    fengine_gains.py once   (the same arguments; one pass, state kept in DIR, for cron)
    fengine_gains.py replay --rail FILE --out PREFIX [--since HH:MM] [--until HH:MM]
                            --baseline-table NAME --applied-table NAME
    fengine_gains.py quant  (the quantiser model: output power, rail and efficiency per rms)

Measurement: each GPU's voltage tap, GET <node>/gnss<g>_srch_tap/rf_stats (the rail recorder's
source): mean |x|^2 per monitored channel over all inputs, in LSB^2 per complex sample, with the
frame's fpga_seq and its age. Only the taps' channels are measured (the chain channels in the
lobes). A managed channel within --interp-max-gap channels of measured ones takes their request
interpolated in dB; one further away keeps its gain (--default-gain at start).

The multiplier in force when a measurement was taken is the last request whose apply_at_fpga_seq
is at or before the measurement's seq (an acknowledgement moves it to the seq actually used).
Before any request is applied it is --applied-table / --baseline-table, two tables of the
F-engine's archive, or 1. Replay holds it at that ratio throughout: the night was recorded
under that table, and the requests it makes are never applied.

Request (GET /fengine_gains, DIR/current.json; every request is appended to DIR/requests.jsonl):
  schema, request_id (increasing), created_unix, apply_at_fpga_seq (the newest measured frame's
  seq at publication + --lead-s, rounded up to --align samples), apply_at_unix (with a time0),
  n_freq 8192, default_gain 1.0, freq_id [...] and gain [...] (voltage multipliers on the
  baseline, for the channels that differ from default_gain), basis {...}.
POST /fengine_gains/applied {"request_id": N, "applied_fpga_seq": S} acknowledges one; acks are
appended to DIR/applied.jsonl. The interface is a draft for the F-engine side to settle.
"""
import argparse
import datetime
import json
import math
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_S = 5.12e-6  # one F-engine sample, the unit of fpga_seq
N_FREQ = 8192  # gain-table channels; the index is the freq_id
CHAN_MHZ = 0.1953125
SCHEMA = "chord-fengine-gain-request/0"
GAIN_ARCHIVE = "/mnt/cs00/data/chord_pathfinder_f_engine_data/data/digital_gains"
RAIL = 7  # the quantiser saturates at +-7 per component

# Managed ranges (freq_id, inclusive): the GNSS lobes, 1166.4-1294.1 MHz.
RANGES = [(5972, 6626)]
BANDS = [
    ("L5/E5a/B2a", 5972, 6076),
    ("E5 gap", 6077, 6127),
    ("E5b/B2b/L3", 6128, 6232),
    ("L2", 6233, 6338),
    ("GLO L2", 6339, 6441),
    ("B3I", 6442, 6521),
    ("E6", 6522, 6573),
    ("E6 upper", 6574, 6626),
]


# ---- the 4+4-bit quantiser: each component of a complex Gaussian with rms sigma, rounded to the
#      nearest integer and clipped to +-RAIL ----


def _phi(z):
    return 0.5 * np.vectorize(math.erfc)(-np.asarray(z, dtype=float) / math.sqrt(2.0))


def level_probs(sigma):
    """P(q = k), k = -RAIL..RAIL, for one component; sigma an array, result (len(sigma), 2 RAIL+1)."""
    s = np.atleast_1d(np.asarray(sigma, dtype=float))[:, None]
    e = np.concatenate(([-np.inf], np.arange(-RAIL + 0.5, RAIL, 1.0), [np.inf]))
    c = _phi(e[None, :] / s)
    return c[:, 1:] - c[:, :-1]


def quant_power(sigma):
    """Mean |x|^2 per complex output sample (LSB^2): what the tap's `power` measures."""
    k = np.arange(-RAIL, RAIL + 1, dtype=float)
    return 2.0 * (level_probs(sigma) * k ** 2).sum(1)


def quant_rail(sigma):
    """Fraction of nibbles at +RAIL: what the tap's `clip_hi` measures."""
    return 1.0 - _phi((RAIL - 0.5) / np.atleast_1d(np.asarray(sigma, dtype=float)))


def quant_efficiency(sigma):
    """Correlation SNR kept by the quantiser, eta^2 with eta = E[x q] / sqrt(E[x^2] E[q^2])."""
    s = np.atleast_1d(np.asarray(sigma, dtype=float))[:, None]
    k = np.arange(-RAIL, RAIL + 1, dtype=float)
    # E[x; a < x < b] = s / sqrt(2 pi) (exp(-a^2 / 2 s^2) - exp(-b^2 / 2 s^2)); the end bins are open
    d = np.exp(-0.5 * np.arange(-RAIL + 0.5, RAIL, 1.0)[None, :] ** 2 / s ** 2)
    z = np.zeros((s.shape[0], 1))
    exq = (
        k * s / math.sqrt(2.0 * math.pi) * (np.hstack([z, d]) - np.hstack([d, z]))
    ).sum(1)
    eq2 = (level_probs(s[:, 0]) * k ** 2).sum(1)
    return exq ** 2 / (s[:, 0] ** 2 * eq2)


_SG = np.geomspace(0.02, 40.0, 3000)
_PG = None
_EG = None


def loss_db(sigma):
    """Quantisation loss in dB at rms sigma (interpolated from quant_efficiency)."""
    global _EG
    if _EG is None:
        _EG = -10.0 * np.log10(quant_efficiency(_SG))
    return np.interp(np.asarray(sigma, dtype=float), _SG, _EG)


def sigma_from_power(p):
    """Invert quant_power: per-component rms in LSB from a measured mean |x|^2 (LSB^2)."""
    global _PG
    if _PG is None:
        _PG = quant_power(_SG)
    return np.interp(np.asarray(p, dtype=float), _PG, _SG)


# ---- gain tables ----


def table_profile(name):
    """Median |coefficient| over the live inputs per channel, of one archived table (a directory
    name in GAIN_ARCHIVE, or a path to its .h5)."""
    import glob

    import h5py

    p = name
    if not os.path.exists(p):
        p = os.path.join(GAIN_ARCHIVE, name)
    if os.path.isdir(p):
        fs = sorted(glob.glob(os.path.join(p, "*.h5")))
        if not fs:
            raise FileNotFoundError("no .h5 in %s" % p)
        p = fs[-1]
    with h5py.File(p, "r") as h:
        g = np.abs(h["gain_coeff"][-1])
    live = g.max(0) > 0
    return np.median(g[:, live], axis=1)


def table_multiplier(applied, baseline):
    """The multiplier `applied` puts on `baseline`, per channel (1 where the baseline is 0)."""
    a, b = table_profile(applied), table_profile(baseline)
    m = np.ones(N_FREQ)
    ok = b > 0
    m[ok] = a[ok] / b[ok]
    return m


# ---- measurement ----


def tap_record(reply, t_poll):
    """One rf_stats reply -> a record, or None when it cannot be used."""
    f, pw = reply.get("freq_ids") or [], reply.get("power") or []
    seq, age = reply.get("fpga_seq", -1), reply.get("age_s", -1.0)
    if (
        not reply.get("enabled")
        or not f
        or len(f) != len(pw)
        or seq is None
        or seq < 0
        or age < 0
    ):
        return None
    clip = reply.get("clip_hi") or [float("nan")] * len(f)
    return {
        "fids": np.asarray(f, int),
        "power": np.asarray(pw, float),
        "clip": np.asarray(clip, float),
        "seq": int(seq),
        "t_frame": t_poll - float(age),
    }


def _median(vals):
    v = [x for x in vals if np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def merge(records):
    """Several taps' records -> one snapshot: per channel the median over the taps that measure it;
    seq the oldest frame's (a gain change after it touches no measurement), seq_ref the newest
    frame's seq with the unix time it had."""
    recs = [r for r in records if r is not None]
    if not recs:
        return None
    by = {}
    for r in recs:
        for f, p, c in zip(r["fids"], r["power"], r["clip"]):
            by.setdefault(int(f), []).append((p, c))
    fids = np.array(sorted(by))
    newest = recs[int(np.argmax([r["t_frame"] for r in recs]))]
    return {
        "fids": fids,
        "power": np.array([_median([x[0] for x in by[f]]) for f in fids]),
        "clip": np.array([_median([x[1] for x in by[f]]) for f in fids]),
        "seq": min(r["seq"] for r in recs),
        "seq_ref": newest["seq"],
        "t_ref": newest["t_frame"],
        "t": newest["t_frame"],
        "n_taps": len(recs),
    }


def poll_taps(urls, timeout=0.8):
    from concurrent.futures import ThreadPoolExecutor

    def get(u):
        try:
            with urllib.request.urlopen(
                u.rstrip("/") + "/rf_stats", timeout=timeout
            ) as r:
                return tap_record(json.loads(r.read().decode()), time.time())
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=min(16, max(1, len(urls)))) as ex:
        return merge(list(ex.map(get, urls)))


def fleet_taps():
    sys.path.insert(0, HERE)
    import rail_watch

    return rail_watch.fleet_taps()


def read_time0(base_url, timeout=2.0):
    """F-engine sample 0 (unix ns) from a node, or None."""
    try:
        with urllib.request.urlopen(
            base_url.rstrip("/") + "/telescope/time0_ns", timeout=timeout
        ) as r:
            v = json.loads(r.read().decode()).get("time0_ns")
        return int(v) if v else None
    except Exception:
        return None


# ---- the controller ----


class Policy:
    """What the requests aim at. Defaults: hold the managed channels at 1.4 LSB rms per component
    (the quiet continuum sits at ~1.8 under the baseline), never above the baseline, never below
    -20 dB; cut once an excess has lasted two passes, raise in large steps once it has been gone
    for 30 s (a channel left cut after a burst loses sensitivity to quantisation)."""

    def __init__(self, **kw):
        self.ranges = RANGES
        self.target = 1.4  # rms per component, LSB
        self.gain_min = 0.1
        self.gain_max = 1.0
        self.step_db = 0.5
        self.deadband_db = 1.0
        self.urgent_db = (
            3.0  # this far over target goes out at once, past min_interval_s
        )
        self.min_interval_s = 10.0
        self.attack_persist_s = (
            2.0  # a channel wants less gain this long before it is cut
        )
        self.release_hold_s = 30.0  # ... more gain this long before it is raised
        self.release_step_db = 6.0
        self.interp_max_gap = 24
        self.default_gain = 1.0
        for k, v in kw.items():
            if not hasattr(self, k):
                raise TypeError("unknown policy field %s" % k)
            setattr(self, k, v)

    def managed(self):
        m = np.zeros(N_FREQ, bool)
        for lo, hi in self.ranges:
            m[lo : hi + 1] = True
        return m


def _q(db, step):
    """Round a gain in dB DOWN to the step grid (a request never asks for more than wanted)."""
    return np.floor(np.asarray(db) / step + 1e-9) * step


class Controller:
    def __init__(self, policy, in_force=None):
        self.p = policy
        self.man = policy.managed()
        start = (
            np.ones(N_FREQ) if in_force is None else np.asarray(in_force, float).copy()
        )
        # the gains as they stand: the multiplier in force, or the default before any is known
        self.gain = start.copy()
        if in_force is None:
            self.gain[self.man] = policy.default_gain
        # (apply seq, multiplier) in apply order; the first is what was in force before any request
        self.history = [(-1, start)]
        self.override = None  # replay: the multiplier in force, fixed
        self.up_since = np.full(N_FREQ, np.nan)
        self.down_since = np.full(N_FREQ, np.nan)
        self.last_request_t = -np.inf
        self.request_id = 0
        self.last_want = None

    def in_force(self, seq):
        if self.override is not None:
            return self.override
        m = self.history[0][1]
        for s, g in self.history:
            if s <= seq:
                m = g
        return m

    def wanted(self, snap):
        """The gain each managed channel should have (dB, unquantised), NaN where unknown."""
        p = self.p
        fids = snap["fids"]
        keep = self.man[fids]
        fids = fids[keep]
        want = np.full(N_FREQ, np.nan)
        if fids.size == 0:
            return want, 0
        sig = sigma_from_power(snap["power"][keep])
        sig0 = sig / self.in_force(snap["seq"])[fids]
        w = 20.0 * np.log10(
            np.clip(p.target / np.maximum(sig0, 1e-6), p.gain_min, p.gain_max)
        )
        idx = np.arange(N_FREQ)
        for lo, hi in p.ranges:
            inr = (fids >= lo) & (fids <= hi)
            if not inr.any():
                continue
            fr, wr = fids[inr], w[inr]
            ch = idx[lo : hi + 1]
            want[lo : hi + 1] = np.interp(ch, fr, wr)
            # nearest measured channel on each side; too far from both -> unknown
            j = np.searchsorted(fr, ch)
            dl = np.where(j > 0, ch - fr[np.maximum(j - 1, 0)], np.inf)
            dr = np.where(j < fr.size, fr[np.minimum(j, fr.size - 1)] - ch, np.inf)
            want[lo : hi + 1][np.minimum(dl, dr) > p.interp_max_gap] = np.nan
        return want, int(fids.size)

    def update(self, snap, now):
        """One measurement. Returns the new gain array when a request is due, else None."""
        p = self.p
        want_db, n = self.wanted(snap)
        self.last_want = want_db
        cur_db = 20.0 * np.log10(self.gain)
        known = np.isfinite(want_db)
        wq = np.where(known, _q(np.where(known, want_db, 0.0), p.step_db), np.nan)
        # the deadband is judged on the wanted level; the request is that level on the step grid
        wd = np.where(known, want_db, 0.0)
        new_db = cur_db.copy()
        down = known & (wd < cur_db - p.deadband_db)
        self.down_since[~down] = np.nan
        self.down_since[down & np.isnan(self.down_since)] = now
        down &= now - self.down_since >= p.attack_persist_s - 1e-9
        new_db[down] = wq[down]
        up = known & (wd > cur_db + p.deadband_db)
        self.up_since[~up] = np.nan
        self.up_since[up & np.isnan(self.up_since)] = now
        ripe = up & (now - self.up_since >= p.release_hold_s)
        new_db[ripe] = np.minimum(wq[ripe], cur_db[ripe] + p.release_step_db)
        changed = np.abs(new_db - cur_db) > 1e-6
        if not changed.any():
            return None
        urgent = down & (wd < cur_db - p.urgent_db)
        if now - self.last_request_t < p.min_interval_s and not urgent.any():
            return None  # deferred: the hold timers keep running
        self.up_since[ripe] = now
        self.gain = 10.0 ** (new_db / 20.0)
        self.last_request_t = now
        self.request_id += 1
        return self.gain.copy()

    def applied(self, request_seq, gain):
        """A request goes into force at request_seq (its apply seq, or the seq an ack reports)."""
        self.history.append((int(request_seq), np.asarray(gain, float).copy()))
        self.history.sort(key=lambda x: x[0])
        if len(self.history) > 64:
            self.history = self.history[-64:]


def apply_seq(snap, now, lead_s, align):
    """The seq the request takes effect at: the newest measured frame's seq, moved to `now`,
    plus lead_s, rounded up to a multiple of align."""
    s = snap["seq_ref"] + (now - snap["t_ref"]) / SAMPLE_S + lead_s / SAMPLE_S
    return int(math.ceil(s / align) * align)


def request(gain, rid, seq, now, policy, snap, time0_ns=None, note=None):
    """The request document. Only the channels whose gain differs from 1 are listed."""
    g = np.round(gain, 4)
    sel = np.nonzero(np.abs(g - 1.0) > 1e-6)[0]
    doc = {
        "schema": SCHEMA,
        "request_id": rid,
        "created_unix": round(now, 3),
        "apply_at_fpga_seq": int(seq),
        "n_freq": N_FREQ,
        "default_gain": 1.0,
        "freq_id": [int(i) for i in sel],
        "gain": [float(x) for x in g[sel]],
        "basis": {
            "gain_is": "voltage multiplier on the F-engine's equalised quiet-sky table",
            "target_rms_lsb_per_component": policy.target,
            "measured_fpga_seq": int(snap["seq"]),
            "measured_channels": int(np.sum(policy.managed()[snap["fids"]])),
            "taps": int(snap.get("n_taps", 0)),
        },
    }
    if time0_ns:
        doc["apply_at_unix"] = round(time0_ns * 1e-9 + seq * SAMPLE_S, 6)
    if note:
        doc["basis"]["note"] = note
    return doc


# ---- outlets ----


def write_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, separators=(",", ":"))
    os.replace(tmp, path)


def append_jsonl(path, obj):
    with open(path, "a") as fh:
        fh.write(json.dumps(obj, separators=(",", ":")) + "\n")


class Outlet:
    """Holds the current request; serves it, takes acks, pushes it."""

    def __init__(self, out_dir, ctrl, push_url=None):
        self.dir, self.ctrl, self.push_url = out_dir, ctrl, push_url
        self.lock = threading.Lock()
        self.current = None
        self.last_measure_t = None
        self.push_fail_t = 0.0
        p = os.path.join(out_dir, "current.json") if out_dir else None
        if p and os.path.exists(p):
            with open(p) as fh:
                self.current = json.load(fh)

    def publish(self, doc):
        with self.lock:
            self.current = doc
        if self.dir:
            write_atomic(os.path.join(self.dir, "current.json"), doc)
            append_jsonl(os.path.join(self.dir, "requests.jsonl"), doc)
        if self.push_url:
            try:
                req = urllib.request.Request(
                    self.push_url,
                    data=json.dumps(doc).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=2.0).read()
            except Exception as e:
                if time.time() - self.push_fail_t > 60:
                    print(
                        "fengine_gains: push to %s failed: %s" % (self.push_url, e),
                        file=sys.stderr,
                        flush=True,
                    )
                    self.push_fail_t = time.time()

    def ack(self, body):
        rid, seq = body.get("request_id"), body.get("applied_fpga_seq")
        if not isinstance(rid, int) or not isinstance(seq, int) or seq < 0:
            raise ValueError(
                "request_id and applied_fpga_seq must be non-negative integers"
            )
        with self.lock:
            cur = self.current
            if cur is None or cur.get("request_id") != rid:
                raise ValueError("request %r is not the current one" % rid)
            g = np.ones(N_FREQ)
            g[np.asarray(cur["freq_id"], int)] = cur["gain"]
            self.ctrl.history = [
                h for h in self.ctrl.history if h[0] != cur["apply_at_fpga_seq"]
            ]
            self.ctrl.applied(seq, g)
        rec = {"t": round(time.time(), 3), "request_id": rid, "applied_fpga_seq": seq}
        if self.dir:
            append_jsonl(os.path.join(self.dir, "applied.jsonl"), rec)
        return rec

    def serve(self, port, host="0.0.0.0"):
        outlet = self

        class H(BaseHTTPRequestHandler):
            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.path.rstrip("/") == "/fengine_gains":
                    with outlet.lock:
                        cur = outlet.current
                    return (
                        self._send(200, cur)
                        if cur
                        else self._send(404, {"error": "no request yet"})
                    )
                if self.path.rstrip("/") == "/healthz":
                    with outlet.lock:
                        cur = outlet.current
                    return self._send(
                        200,
                        {
                            "request_id": cur and cur.get("request_id"),
                            "last_measure_unix": outlet.last_measure_t,
                        },
                    )
                self._send(404, {"error": "unknown path"})

            def do_POST(self):
                if self.path.rstrip("/") != "/fengine_gains/applied":
                    return self._send(404, {"error": "unknown path"})
                try:
                    n = int(self.headers.get("Content-Length", "0"))
                    body = json.loads(self.rfile.read(n).decode() or "{}")
                    self._send(200, outlet.ack(body))
                except (ValueError, json.JSONDecodeError) as e:
                    self._send(400, {"error": str(e)})

            def log_message(self, *a):
                pass

        srv = ThreadingHTTPServer((host, port), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv


# ---- state for cron (once) ----


def save_state(path, ctrl):
    st = {
        "request_id": ctrl.request_id,
        "last_request_t": ctrl.last_request_t,
        "gain": {
            str(i): float(ctrl.gain[i])
            for i in np.nonzero(np.abs(ctrl.gain - 1) > 1e-9)[0]
        },
        "up_since": {
            str(i): float(ctrl.up_since[i])
            for i in np.nonzero(np.isfinite(ctrl.up_since))[0]
        },
        "down_since": {
            str(i): float(ctrl.down_since[i])
            for i in np.nonzero(np.isfinite(ctrl.down_since))[0]
        },
        "history": [
            [s, {str(i): float(g[i]) for i in np.nonzero(np.abs(g - 1) > 1e-9)[0]}]
            for s, g in ctrl.history[-8:]
        ],
    }
    write_atomic(path, st)


def load_state(path, ctrl):
    if not os.path.exists(path):
        return
    with open(path) as fh:
        st = json.load(fh)

    def arr(d, fill):
        a = np.full(N_FREQ, fill)
        for k, v in d.items():
            a[int(k)] = v
        return a

    ctrl.request_id = int(st.get("request_id", 0))
    ctrl.last_request_t = float(st.get("last_request_t", -np.inf))
    ctrl.gain = arr(st.get("gain", {}), 1.0)
    ctrl.up_since = arr(st.get("up_since", {}), np.nan)
    ctrl.down_since = arr(st.get("down_since", {}), np.nan)
    if st.get("history"):
        ctrl.history = [(int(s), arr(d, 1.0)) for s, d in st["history"]]


# ---- modes ----


def make_controller(args):
    pol = Policy(
        target=args.target,
        gain_min=10 ** (args.min_db / 20.0),
        step_db=args.step_db,
        deadband_db=args.deadband_db,
        urgent_db=args.urgent_db,
        min_interval_s=args.min_interval_s,
        attack_persist_s=args.attack_persist_s,
        release_hold_s=args.release_hold_s,
        release_step_db=args.release_step_db,
        interp_max_gap=args.interp_max_gap,
        default_gain=args.default_gain,
    )
    in_force = None
    if args.baseline_table:
        in_force = table_multiplier(
            args.applied_table or latest_table(), args.baseline_table
        )
    return Controller(pol, in_force)


def latest_table():
    names = sorted(d for d in os.listdir(GAIN_ARCHIVE) if d.endswith("digitalgain"))
    return names[-1]


def one_pass(ctrl, outlet, urls, args, time0_ns):
    snap = poll_taps(urls, timeout=args.timeout)
    now = time.time()
    if snap is None or now - snap["t_ref"] > args.max_age_s:
        return None
    outlet.last_measure_t = round(snap["t_ref"], 3)
    g = ctrl.update(snap, now)
    if g is None:
        return None
    seq = apply_seq(snap, now, args.lead_s, args.align)
    doc = request(g, ctrl.request_id, seq, now, ctrl.p, snap, time0_ns)
    ctrl.applied(seq, g)
    outlet.publish(doc)
    return doc


def run(args, loop):
    urls = args.endpoints or fleet_taps()
    os.makedirs(args.out, exist_ok=True)
    ctrl = make_controller(args)
    state = os.path.join(args.out, "state.json")
    load_state(state, ctrl)
    outlet = Outlet(args.out, ctrl, args.push)
    if args.port is not None:
        outlet.serve(args.port)
    base = urls[0].split("/gnss")[0]
    time0 = read_time0(args.time0_url or base)
    while True:
        t0 = time.time()
        doc = one_pass(ctrl, outlet, urls, args, time0)
        if doc:
            print(
                "fengine_gains: request %d apply_at_fpga_seq %d, %d channels below 1, min %.3f"
                % (
                    doc["request_id"],
                    doc["apply_at_fpga_seq"],
                    len(doc["gain"]),
                    min(doc["gain"]) if doc["gain"] else 1.0,
                ),
                flush=True,
            )
        save_state(state, ctrl)
        if not loop:
            return 0
        time.sleep(max(0.0, args.period - (time.time() - t0)))


def rail_snapshots(path, t0=None, t1=None):
    """The rail recorder's file as per-second snapshots (rf_stats as recorded: `pw` = power,
    `hi` = clip_hi, `f` = freq_ids on the first line per instance and on change)."""
    fl, cur, cur_s = {}, {}, None
    with open(path) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            inst, t = d.get("inst"), d.get("t")
            if inst is None or t is None:
                continue
            if "f" in d:
                fl[inst] = d["f"]
            if "pw" not in d or inst not in fl or len(fl[inst]) != len(d["pw"]):
                continue
            if t0 is not None and t < t0:
                continue
            if t1 is not None and t > t1:
                break
            s = int(t)
            if cur_s is not None and s != cur_s and cur:
                yield merge(list(cur.values()))
                cur = {}
            cur_s = s
            cur[inst] = tap_record(
                {
                    "enabled": True,
                    "freq_ids": fl[inst],
                    "power": d["pw"],
                    "clip_hi": d.get("hi"),
                    "fpga_seq": d.get("seq", -1),
                    "age_s": d.get("age", 0.0),
                },
                t,
            )
    if cur:
        yield merge(list(cur.values()))


def replay(args):
    ctrl = make_controller(args)
    ctrl.override = ctrl.history[0][1]
    day = None
    with open(args.rail) as fh:
        for line in fh:
            try:
                day = json.loads(line)["t"] // 86400 * 86400
                break
            except (ValueError, KeyError):
                continue
    lat = args.lead_s + 1.0  # the request's lead plus one poll
    queue, in_force = [], ctrl.override.copy()
    T, F, SM, SC, G, CL, REQ = [], None, [], [], [], [], []
    for snap in rail_snapshots(
        args.rail, _hhmm(day, args.since), _hhmm(day, args.until)
    ):
        if snap is None:
            continue
        now = snap["t_ref"]
        while queue and queue[0][0] <= now:
            in_force = queue.pop(0)[1]
        if F is None:
            F = snap["fids"]
        g = ctrl.update(snap, now)
        if g is not None:
            queue.append((now + lat, g))
            REQ.append((now, int(np.sum(np.abs(g - 1) > 1e-6))))
        at = np.searchsorted(snap["fids"], F)
        have = (at < snap["fids"].size) & (
            snap["fids"][np.minimum(at, snap["fids"].size - 1)] == F
        )
        sig = np.full(F.size, np.nan)
        clip = np.full(F.size, np.nan)
        sig[have] = sigma_from_power(snap["power"][at[have]])
        clip[have] = snap["clip"][at[have]]
        T.append(now)
        SM.append(sig)
        G.append(in_force[F])
        CL.append(clip)
    T, SM, G, CL = map(np.array, (T, SM, G, CL))
    # under the requests: the unattenuated rms (as recorded / the table's multiplier) x the request in force
    SC = SM / ctrl.override[F][None, :] * G
    np.savez(
        args.out + ".npz",
        t=T,
        fids=F,
        sigma_meas=SM,
        sigma_ctrl=SC,
        gain=G,
        clip_meas=CL,
        in_force_table=ctrl.override[F],
        requests=np.array(REQ),
    )
    hours = (T[-1] - T[0]) / 3600.0 if len(T) > 1 else 0.0
    lines = [
        "replay %s, %s .. %s UTC, %.1f h, %d channels measured, %d requests (%.1f per hour), target %.2f LSB"
        % (
            os.path.basename(args.rail),
            _hms(T[0]),
            _hms(T[-1]),
            hours,
            len(F),
            len(REQ),
            len(REQ) / max(hours, 1e-9),
            ctrl.p.target,
        ),
        "sig = rms per component (LSB), max over the band's measured channels; s>1e-3 = seconds any channel"
        " rails above 1e-3 per nibble; rail = mean rail per nibble; loss = mean quantisation loss (dB)",
        "%-12s %5s | %-30s | %-30s"
        % (
            "band",
            "chans",
            "as recorded: sig p50/p99, s>1e-3",
            "requested: sig p50/p99, s>1e-3",
        )
        + " | rail rec / req   | loss rec / req",
    ]
    thr = 2.1  # sigma at which quant_rail reaches ~1e-3
    for name, lo, hi in BANDS:
        c = (F >= lo) & (F <= hi)
        if not c.any():
            continue
        a, b = _rowmax(SM[:, c]), _rowmax(SC[:, c])
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        sm, sc = SM[:, c][np.isfinite(SM[:, c])], SC[:, c][np.isfinite(SC[:, c])]
        lines.append(
            "%-12s %5d | %5.2f %5.2f %8d        | %5.2f %5.2f %8d        | %.1e / %.1e | %.2f / %.2f"
            % (
                name,
                c.sum(),
                np.median(a),
                np.percentile(a, 99),
                int((a > thr).sum()),
                np.median(b),
                np.percentile(b, 99),
                int((b > thr).sum()),
                np.mean(quant_rail(sm)),
                np.mean(quant_rail(sc)),
                np.mean(loss_db(sm)),
                np.mean(loss_db(sc)),
            )
        )
    txt = "\n".join(lines)
    print(txt)
    with open(args.out + ".txt", "w") as fh:
        fh.write(txt + "\n")
    if args.plot:
        plot_replay(args.out + ".png", T, F, SM, SC, G, ctrl.p.target)
    return 0


def _rowmax(a):
    """Max over each row ignoring NaN; NaN for an all-NaN row."""
    out = np.full(a.shape[0], np.nan)
    ok = np.isfinite(a).any(1)
    out[ok] = np.nanmax(a[ok], 1)
    return out


def _hhmm(day, s):
    """HH:MM on the file's UTC day, as unix seconds; None when not given."""
    if not s:
        return None
    return day + int(s[:2]) * 3600 + int(s[3:5]) * 60


def _hms(t):
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime(
        "%H:%M:%S"
    )


def plot_replay(path, T, F, SM, SC, G, target):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = (T - T[0] // 86400 * 86400) / 3600.0
    rows = [(n, lo, hi) for n, lo, hi in BANDS if ((F >= lo) & (F <= hi)).any()]
    fig, ax = plt.subplots(
        len(rows), 2, figsize=(16, 2.4 * len(rows)), sharex=True, squeeze=False
    )
    for r, (name, lo, hi) in enumerate(rows):
        c = (F >= lo) & (F <= hi)
        a = ax[r, 0]
        a.plot(
            h,
            _rowmax(SM[:, c]),
            lw=0.6,
            color="#b03a2e",
            label="as recorded (max over the band's taps)",
        )
        a.plot(
            h, _rowmax(SC[:, c]), lw=0.6, color="#1f5f8b", label="under the requests"
        )
        a.axhline(target, color="#555555", lw=0.6, ls="--")
        a.axhline(2.1, color="#999999", lw=0.6, ls=":")
        a.set_ylim(0, 8)
        a.set_ylabel("rms / component [LSB]")
        a.set_title(name, loc="left", fontsize=9)
        if r == 0:
            a.legend(fontsize=7, loc="upper right")
        b = ax[r, 1]
        b.plot(
            h,
            20 * np.log10(G[:, c].min(1)),
            lw=0.6,
            color="#1f5f8b",
            label="min over the band",
        )
        b.plot(
            h,
            20 * np.log10(G[:, c].max(1)),
            lw=0.6,
            color="#76a5c9",
            label="max over the band",
        )
        b.set_ylim(-21, 1)
        b.set_ylabel("requested gain [dB]")
        if r == 0:
            b.legend(fontsize=7, loc="lower right")
    for a in ax[-1]:
        a.set_xlabel("UTC [h]")
    fig.suptitle(
        "fengine_gains replay: output rms per component, as recorded vs under the requests "
        "(dashed = target, dotted = 1e-3 rail)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def quant_table(args):
    s = np.array([0.5, 0.7, 0.86, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0])
    print("rms/comp  mean|x|^2  rail/nibble  efficiency")
    for si, p, r, e in zip(s, quant_power(s), quant_rail(s), quant_efficiency(s)):
        print(
            "%6.2f   %8.3f   %10.2e   %8.4f (%.2f dB)"
            % (si, p, r, e, -10 * math.log10(e))
        )
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["run", "once", "replay", "quant"])
    ap.add_argument(
        "--endpoints",
        nargs="*",
        help="tap URLs (default: both taps on every fleet host)",
    )
    ap.add_argument(
        "--out", help="run/once: state and request directory; replay: output prefix"
    )
    ap.add_argument("--port", type=int, help="serve GET /fengine_gains on this port")
    ap.add_argument("--push", help="POST every request to this URL")
    ap.add_argument(
        "--time0-url",
        help="node REST base serving /telescope/time0_ns (default: the first tap's)",
    )
    ap.add_argument("--period", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=0.8)
    ap.add_argument(
        "--max-age-s",
        type=float,
        default=5.0,
        help="ignore a measurement older than this",
    )
    ap.add_argument(
        "--lead-s",
        type=float,
        default=1.0,
        help="apply this far after the newest frame",
    )
    ap.add_argument(
        "--align",
        type=int,
        default=8192,
        help="apply seq is a multiple of this (one N2 frame)",
    )
    ap.add_argument(
        "--target", type=float, default=1.4, help="output rms per component, LSB"
    )
    ap.add_argument("--min-db", type=float, default=-20.0)
    ap.add_argument("--step-db", type=float, default=0.5)
    ap.add_argument("--deadband-db", type=float, default=1.0)
    ap.add_argument("--urgent-db", type=float, default=3.0)
    ap.add_argument("--min-interval-s", type=float, default=10.0)
    ap.add_argument("--attack-persist-s", type=float, default=2.0)
    ap.add_argument("--release-hold-s", type=float, default=30.0)
    ap.add_argument("--release-step-db", type=float, default=6.0)
    ap.add_argument("--interp-max-gap", type=int, default=24)
    ap.add_argument("--default-gain", type=float, default=1.0)
    ap.add_argument(
        "--baseline-table",
        help="archived table the requests multiply (e.g. 20261023T210749Z_...)",
    )
    ap.add_argument(
        "--applied-table", help="archived table in force now (default: the newest)"
    )
    ap.add_argument(
        "--rail", help="replay: a rail recorder file (rf_chan_YYYYMMDD.jsonl)"
    )
    ap.add_argument("--since")
    ap.add_argument("--until")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)
    if args.mode == "quant":
        return quant_table(args)
    if not args.out:
        ap.error("--out is required")
    if args.mode == "replay":
        if not (args.rail and args.baseline_table and args.applied_table):
            ap.error("replay needs --rail, --baseline-table and --applied-table")
        return replay(args)
    return run(args, loop=args.mode == "run")


if __name__ == "__main__":
    sys.exit(main())
