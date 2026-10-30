#!/usr/bin/env python3
"""fengine_gains.py's own check: the quantiser model, the controller (cut at once, raise after a
hold, deadband, rate limit, interpolation, the multiplier in force), the request document, the
HTTP outlet and a replay of a small synthetic rail file.

    python3 scripts/gnss/site/test_fengine_gains.py

Nothing reaches a node or the F-engine: the server binds to localhost on a free port and the
files go to a throwaway directory.
"""
import json
import os
import shutil
import sys
import tempfile
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fengine_gains as fg  # noqa: E402

_fails = []


def check(ok, what):
    if not ok:
        _fails.append(what)
        print("  FAIL", what)


def snap(fids, sigma, seq=1000, t=0.0):
    """A merged snapshot whose channels sit at the given rms per component."""
    fids = np.asarray(fids, int)
    return {
        "fids": fids,
        "power": fg.quant_power(np.broadcast_to(sigma, fids.shape).astype(float)),
        "clip": np.zeros(fids.size),
        "seq": seq,
        "seq_ref": seq,
        "t_ref": t,
        "t": t,
        "n_taps": 1,
    }


def test_quantiser_model():
    s = np.array([0.4, 0.86, 1.4, 2.0, 3.0, 4.0])
    back = fg.sigma_from_power(fg.quant_power(s))
    check(
        np.allclose(back, s, rtol=2e-3),
        "sigma_from_power inverts quant_power (%s)" % back,
    )
    # unclipped and well above an LSB, rounding adds 1/12 per component
    check(
        abs(fg.quant_power(2.0)[0] - 2 * (4 + 1 / 12.0)) < 0.02,
        "quant_power(2) ~ 2 (4 + 1/12)",
    )
    check(abs(fg.quant_rail(2.0)[0] - 5.77e-4) < 2e-5, "rail at 2 LSB = 1 - Phi(3.25)")
    e = fg.quant_efficiency(np.array([0.86, 3.0]))
    check(
        abs(-10 * np.log10(e[0]) - 0.46) < 0.02 and e[1] > 0.98,
        "4-bit efficiency (%s)" % e,
    )


def test_cut_at_once_raise_after_hold():
    pol = fg.Policy(
        ranges=[(100, 120)],
        target=1.4,
        min_interval_s=10.0,
        release_hold_s=30.0,
        release_step_db=1.0,
        attack_persist_s=2.0,
    )
    c = fg.Controller(pol)
    check(
        c.update(snap([100, 110, 120], 1.4 * 4.0), now=-2.0) is None,
        "one pass over target is not yet a cut",
    )
    g = c.update(snap([100, 110, 120], 1.4 * 4.0), now=0.0)  # 12 dB over target for 2 s
    check(
        g is not None and abs(20 * np.log10(g[110]) + 12.0) < 0.51,
        "a 12 dB excess is cut at once (%s)" % (None if g is None else g[110]),
    )
    check(
        g is not None and g[99] == 1.0 and g[121] == 1.0,
        "outside the ranges the gain stays 1",
    )
    c.applied(1000, g)
    # the excess goes away: nothing until the hold has run, then one release step
    quiet = snap(
        [100, 110, 120], 1.4 * g[110], seq=2000
    )  # unattenuated level now = target
    check(c.update(quiet, now=40.0) is None, "no raise before the hold")
    g2 = c.update(quiet, now=71.0)
    check(
        g2 is not None and abs(20 * np.log10(g2[110] / g[110]) - 1.0) < 1e-6,
        "after the hold the gain rises one release step (%s)"
        % (g2 is not None and 20 * np.log10(g2[110] / g[110])),
    )


def test_deadband_and_rate_limit():
    pol = fg.Policy(
        ranges=[(100, 120)],
        target=1.4,
        min_interval_s=10.0,
        deadband_db=1.0,
        urgent_db=3.0,
        attack_persist_s=0.0,
    )
    c = fg.Controller(pol, in_force=np.ones(fg.N_FREQ))
    check(
        c.update(snap([100, 120], 1.4 * 10 ** (0.8 / 20)), now=0.0) is None,
        "0.8 dB over target: no request",
    )
    check(
        c.update(snap([100, 120], 1.4 * 10 ** (2.0 / 20)), now=1.0) is not None,
        "2 dB over: a request",
    )
    c.applied(1000, c.gain)
    # 2 dB more at t = 3 s: inside min_interval and not urgent -> deferred
    s2 = snap([100, 120], 1.4 * 10 ** (4.0 / 20) * c.gain[100], seq=2000)
    check(
        c.update(s2, now=3.0) is None, "a non-urgent change inside min_interval_s waits"
    )
    s3 = snap([100, 120], 1.4 * 10 ** (8.0 / 20) * c.gain[100], seq=2000)
    check(c.update(s3, now=4.0) is not None, "an urgent one does not")


def test_interpolation_and_gaps():
    pol = fg.Policy(
        ranges=[(100, 200)],
        target=1.4,
        interp_max_gap=10,
        step_db=0.5,
        attack_persist_s=0.0,
    )
    c = fg.Controller(pol, in_force=np.ones(fg.N_FREQ))
    s = snap([100, 110], np.array([1.4 * 2.0, 1.4 * 4.0]))  # want -6 and -12 dB
    w, n = c.wanted(s)
    check(
        n == 2 and abs(w[105] + 9.03) < 0.05,
        "interpolated in dB between measured channels (%s)" % w[105],
    )
    check(
        np.isfinite(w[120]) and not np.isfinite(w[121]),
        "known within interp_max_gap, unknown beyond",
    )
    g = c.update(s, now=0.0)
    check(g[150] == 1.0, "a managed channel far from any measurement keeps its gain")


def test_in_force_history():
    pol = fg.Policy(ranges=[(100, 120)], target=1.4)
    c = fg.Controller(pol, in_force=np.ones(fg.N_FREQ))
    half = np.ones(fg.N_FREQ)
    half[100:121] = 0.5
    c.applied(5000, half)
    # measured at 2.8 LSB: before seq 5000 that is the unattenuated level, after it half of it
    w_before, _ = c.wanted(snap([100, 120], 2.8, seq=4999))
    w_after, _ = c.wanted(snap([100, 120], 2.8, seq=5001))
    check(
        abs(w_before[110] + 6.02) < 0.05 and abs(w_after[110] + 12.04) < 0.05,
        "the multiplier in force is the request applied at or before the seq (%s %s)"
        % (w_before[110], w_after[110]),
    )


def test_request_document_and_apply_seq():
    pol = fg.Policy(ranges=[(100, 120)])
    g = np.ones(fg.N_FREQ)
    g[[100, 105]] = [0.5, 0.25]
    s = snap([100], 1.0, seq=10_000_000, t=100.0)
    seq = fg.apply_seq(s, now=100.3, lead_s=1.0, align=8192)
    check(
        seq % 8192 == 0
        and seq * fg.SAMPLE_S >= 10_000_000 * fg.SAMPLE_S + 1.3 - 1e-9
        and seq * fg.SAMPLE_S < 10_000_000 * fg.SAMPLE_S + 1.3 + 8192 * fg.SAMPLE_S,
        "apply seq is lead_s after the newest frame, moved to now, on the alignment (%d)"
        % seq,
    )
    d = fg.request(g, 7, seq, 100.3, pol, s, time0_ns=1_700_000_000_000_000_000)
    check(
        d["schema"] == fg.SCHEMA and d["request_id"] == 7 and d["n_freq"] == 8192,
        "schema, id, n_freq",
    )
    check(
        d["freq_id"] == [100, 105] and d["gain"] == [0.5, 0.25],
        "only the channels off 1, in order",
    )
    check(
        abs(d["apply_at_unix"] - (1.7e9 + seq * fg.SAMPLE_S)) < 1e-3,
        "apply_at_unix from time0",
    )
    json.dumps(d)


def test_outlet_http_and_ack():
    d = tempfile.mkdtemp()
    try:
        pol = fg.Policy(ranges=[(100, 120)])
        c = fg.Controller(pol, in_force=np.ones(fg.N_FREQ))
        out = fg.Outlet(d, c)
        srv = out.serve(0, host="127.0.0.1")
        port = srv.server_address[1]
        base = "http://127.0.0.1:%d" % port
        try:
            urllib.request.urlopen(base + "/fengine_gains", timeout=2)
            check(False, "404 before any request")
        except urllib.error.HTTPError as e:
            check(e.code == 404, "404 before any request")
        g = np.ones(fg.N_FREQ)
        g[110] = 0.5
        doc = fg.request(g, 1, 81920, 0.0, pol, snap([110], 1.0))
        c.applied(81920, g)
        out.publish(doc)
        got = json.loads(
            urllib.request.urlopen(base + "/fengine_gains", timeout=2).read()
        )
        check(got == doc, "GET serves the current request")
        check(
            json.load(open(os.path.join(d, "current.json"))) == doc,
            "current.json written",
        )
        body = json.dumps({"request_id": 1, "applied_fpga_seq": 90112}).encode()
        r = urllib.request.urlopen(
            urllib.request.Request(
                base + "/fengine_gains/applied", data=body, method="POST"
            ),
            timeout=2,
        )
        check(r.status == 200, "ack accepted")
        check(
            any(s == 90112 for s, _ in c.history)
            and not any(s == 81920 for s, _ in c.history),
            "the ack moves the request to the seq actually used",
        )
        bad = json.dumps({"request_id": 9, "applied_fpga_seq": 1}).encode()
        try:
            urllib.request.urlopen(
                urllib.request.Request(
                    base + "/fengine_gains/applied", data=bad, method="POST"
                ),
                timeout=2,
            )
            check(False, "an ack for another request is refused")
        except urllib.error.HTTPError as e:
            check(e.code == 400, "an ack for another request is refused")
        srv.shutdown()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_state_round_trip():
    pol = fg.Policy(ranges=[(100, 120)], attack_persist_s=0.0)
    c = fg.Controller(pol, in_force=np.ones(fg.N_FREQ))
    c.update(snap([100, 120], 5.6), now=0.0)
    c.applied(1000, c.gain)
    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, "state.json")
        fg.save_state(p, c)
        c2 = fg.Controller(pol)
        fg.load_state(p, c2)
        check(
            np.allclose(c2.gain, c.gain) and c2.request_id == c.request_id,
            "gain and request id survive a restart",
        )
        check(
            np.allclose(c2.in_force(2000), c.in_force(2000)),
            "the multiplier in force survives a restart",
        )
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_replay_synthetic():
    """Two taps, a 60-s bright event on one band: the replay cuts it to the target and lets go."""
    d = tempfile.mkdtemp()
    try:
        rail = os.path.join(d, "rf_chan_20261029.jsonl")
        t0 = 1793260800.0  # 08:00 UTC
        with open(rail, "w") as fh:
            for k in range(700):
                t = t0 + k
                hot = 4.0 if 100 <= k < 160 else 1.0
                for inst, f in (("cx27.0", [6020, 6030]), ("cx27.1", [6490, 6500])):
                    s = (
                        np.array([1.8 * hot, 1.8 * hot])
                        if inst == "cx27.0"
                        else np.array([1.8, 1.8])
                    )
                    rec = {
                        "t": t,
                        "inst": inst,
                        "seq": int(1e9 + k / fg.SAMPLE_S),
                        "age": 0.2,
                        "pw": [float(x) for x in fg.quant_power(s)],
                        "hi": [0.0, 0.0],
                    }
                    if k == 0:
                        rec["f"] = f
                    fh.write(json.dumps(rec) + "\n")
        base = os.path.join(d, "base.h5")
        _fake_table(base, 1.0)
        prefix = os.path.join(d, "replay")
        rc = fg.main(
            [
                "replay",
                "--rail",
                rail,
                "--out",
                prefix,
                "--baseline-table",
                base,
                "--applied-table",
                base,
            ]
        )
        check(rc == 0, "replay runs")
        z = np.load(prefix + ".npz")
        sc, t = z["sigma_ctrl"], z["t"] - t0
        f = list(z["fids"])
        i = f.index(6020)
        during = (t > 112) & (t < 155)
        check(
            np.nanmax(sc[during, i]) < 1.6
            and np.nanmax(z["sigma_meas"][during, i]) > 7.0,
            "the bright minute is held at the target under the requests (%.2f)"
            % np.nanmax(sc[during, i]),
        )
        after = t > 680
        check(
            np.nanmin(z["gain"][after, i]) > 0.7,
            "and the gain comes back after it (%.2f)" % np.nanmin(z["gain"][after, i]),
        )
        check(
            len(z["requests"]) < 30,
            "a handful of requests, not one per second (%d)" % len(z["requests"]),
        )
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_once_against_a_fake_tap():
    """`once` polls a tap, writes the request and the state; the apply seq is ahead of the frame."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    seq0 = 5_000_000_000

    class Tap(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.endswith("/rf_stats"):
                body = {
                    "enabled": True,
                    "freq_ids": [6020, 6030, 6490],
                    "fpga_seq": seq0,
                    "age_s": 0.3,
                    "power": [
                        float(x) for x in fg.quant_power(np.array([5.6, 5.6, 1.4]))
                    ],
                    "clip_hi": [0.01, 0.01, 0.0],
                }
            elif self.path.endswith("/telescope/time0_ns"):
                body = {"time0_ns": 1_793_207_549_000_003_890}
            else:
                self.send_response(404)
                self.end_headers()
                return
            b = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Tap)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    d = tempfile.mkdtemp()
    try:
        url = "http://127.0.0.1:%d/gnss0_srch_tap" % srv.server_address[1]
        rc = fg.main(
            ["once", "--endpoints", url, "--out", d, "--attack-persist-s", "0"]
        )
        check(rc == 0, "once exits 0")
        doc = json.load(open(os.path.join(d, "current.json")))
        g = dict(zip(doc["freq_id"], doc["gain"]))
        check(
            abs(20 * np.log10(g[6020]) + 12.0) < 0.6
            and abs(20 * np.log10(g[6025]) + 12.0) < 0.6,
            "the 12 dB excess is requested away, between the taps too (%s)"
            % g.get(6020),
        )
        check(6490 not in g or g[6490] > 0.9, "a channel at target is left alone")
        check(
            doc["apply_at_fpga_seq"] % 8192 == 0
            and 1.0
            <= (doc["apply_at_fpga_seq"] - seq0) * fg.SAMPLE_S
            < 1.0 + 0.3 + 1.0,
            "apply seq ~lead_s + age after the frame (%d)"
            % (doc["apply_at_fpga_seq"] - seq0),
        )
        check("apply_at_unix" in doc, "time0 read from the tap's node")
        check(
            os.path.exists(os.path.join(d, "state.json")), "state kept for the next run"
        )
    finally:
        srv.shutdown()
        shutil.rmtree(d, ignore_errors=True)


def _fake_table(path, level):
    import h5py

    with h5py.File(path, "w") as h:
        h["gain_coeff"] = np.full((1, fg.N_FREQ, 4), level * 4.75e8, np.complex64)
        h["gain_exp"] = np.zeros((1, 4), np.int32)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        try:
            t()
        except Exception as e:
            check(False, "%s raised %s: %s" % (t.__name__, type(e).__name__, e))
    if _fails:
        print("\n%d FAILED" % len([f for f in _fails if f]))
        return 1
    print("\nall passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
