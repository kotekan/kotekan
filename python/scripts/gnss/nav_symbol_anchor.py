#!/usr/bin/env python3
"""Anchors for the nav-symbol feed (GnssNavSymbolSend): each symbol stream's satellite time and
sign, from the decoded messages.

    python3 nav_symbol_anchor.py [--rest-url http://localhost:12048] [--every-s 2]

GnssNavSymbols decides each symbol up to one sign per stream and counts it in its stream, on a
receive-time grid (symbol k starts at t0 + k * sym_dt). The decoders' time marks
(nav_decode --timemarks-file) give the receive time of a message's first symbol and the time
count it carries, so a mark falls on one symbol of a stream and gives that symbol's satellite
time. There the stream holds the message's sync pattern (F/NAV), its preamble (B-CNAV2), its
preamble and PRN (B-CNAV3) or, for CNAV, which has none, the message re-encoded; the correlation
gives the stream's sign and confirms the alignment. B-CNAV3's marks come from the combiner's
emits rather than the stream's own grid, so its frame start is looked for within SLACK symbols. Every --every-s this posts each stream's anchor {sig, prn, stream_id,
k, sv, sign} to the feed's set_anchors, where it turns the stream's symbols into satellite-time
indices with their true signs; every later mark is checked against it, and one that disagrees
replaces it (logged).

The index (CHORD's key): floor(t / T_sym), t the satellite clock's transmit time in seconds
since 1980-01-06 on the system's own scale (GPST, GST = GPST, BDT = GPST - 14 s).
"""

import argparse
import datetime
import json
import os
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import beidou_bcnav2  # noqa: E402
import beidou_bcnav3  # noqa: E402
import galileo_fnav  # noqa: E402
import galileo_inav  # noqa: E402
import gnss_ephemeris as ge  # noqa: E402
import gnss_skytime  # noqa: E402

# symbol stage -> (sys, NSYM sig code, symbol period s, timemarks file, the marks' signal),
# as navsym_send's inputs (config/bladerf_gnss.yaml)
SOURCES = {
    "l2cm_symbols": ("G", 1, 0.02, "timemarks_l2c.json", "GPS_L2C_CNAV"),
    "l5i_symbols": ("G", 3, 0.01, "timemarks_misc.json", "GPS_L5_CNAV"),
    "e5ai_symbols": ("E", 4, 0.02, "timemarks_fnav.json", "GAL_E5AI_FNAV"),
    "b2ad_symbols": ("C", 6, 0.005, "timemarks_bcnav2.json", "BDS_B2A_BCNAV2"),
    "b2bi_symbols": ("C", 2, 0.001, "timemarks_bcnav3.json", "BDS_B2B_BCNAV3"),
}
# symbols either side of a mark searched for its message start, for marks not on the stream's grid
SLACK = {"BDS_B2B_BCNAV3": 3}
# symbols from one sync pattern to the next: a mark older than the status window is carried
# forward by whole frames (pages) into it, its satellite-time index with it
PERIOD = {"GAL_E5AI_FNAV": 500, "BDS_B2A_BCNAV2": 600, "BDS_B2B_BCNAV3": 1000}
MIN_CORR = 0.8  # correlation per symbol a mark's symbols must reach
REPOST_S = 60.0  # every anchor is posted again this often
FORGET_S = 600.0  # an anchor whose stream has not been seen this long is forgotten


def log(msg):
    print("[navsym-anchor %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def sv_index(mark, sysc, t_sym):
    """The satellite-time symbol index of the first symbol of a mark's message, or None if the
    message does not start on a symbol: its time count (gnss_skytime.tx_sow, GPST seconds of
    week), in the week nearest its receive time, on the system's scale."""
    sow = gnss_skytime.tx_sow(mark)
    t_rx = ge.gpst_of_utc(
        datetime.datetime.fromtimestamp(mark["t_host"], tz=datetime.timezone.utc)
    )
    t = round((t_rx - sow) / 604800.0) * 604800.0 + sow
    if sysc == "C":
        t += ge.BDT_GPST
    q = t / t_sym
    return int(round(q)) if abs(q - round(q)) < 1e-3 else None


def expected(mark, signal):
    """(offset, +-1 symbols): what the stream holds from a mark's message start: the sync
    pattern (F/NAV), the preamble (B-CNAV2), or the CNAV message re-encoded from its 7th bit (the
    encoder's first six outputs hang on the message before), on air without the G2 inversion.
    None for a CNAV mark without its bits."""
    if signal == "GAL_E5AI_FNAV":
        return 0, 1 - 2 * np.array(galileo_fnav.SYNC)
    if signal == "BDS_B2A_BCNAV2":
        return 0, 1 - 2 * np.array(beidou_bcnav2.PREAMBLE)
    if signal == "BDS_B2B_BCNAV3":
        return 0, 1 - 2 * beidou_bcnav3.preamble_for(int(mark["prn"])).astype(int)
    if "bits" not in mark:
        return None
    v = int(mark["bits"], 16)
    bits = [(v >> (299 - i)) & 1 for i in range(300)]
    sym = galileo_inav.conv_encode(bits, flush=False)[12:].astype(int)
    sym[1::2] ^= 1
    return 12, 1 - 2 * sym


def mark_sign(mark, signal, buf, k_first, t0, sym_dt):
    """(k, sign, shift) of a mark on a stream whose symbols buf start at stream index k_first, or
    None: the stream symbol its message starts at, within SLACK of where the mark falls and, for
    a mark older than buf, carried forward by shift (whole PERIODs) to the first sync in buf; and
    the stream's sign from the correlation there with what the message holds (MIN_CORR).
    """
    k_mark = int(round((mark["t_host"] - t0) / sym_dt))
    exp = expected(mark, signal)
    if exp is None:
        return None
    off, pat = exp
    slack = SLACK.get(signal, 0)
    shift = 0
    if signal in PERIOD and k_mark - slack < k_first:
        p = PERIOD[signal]
        shift = -((k_mark - slack - k_first) // p) * p
    best = None
    for k in range(k_mark + shift - slack, k_mark + shift + slack + 1):
        i = k - k_first + off
        if k < k_first or i + len(pat) > len(buf):
            continue
        c = float(np.dot(buf[i : i + len(pat)], pat)) / len(pat)
        if abs(c) >= MIN_CORR and (best is None or abs(c) > abs(best[1])):
            best = (k, c)
    return (best[0], 1 if best[1] > 0 else -1, shift) if best else None


def anchor_of(row, marks, signal, sysc, t_sym):
    """The anchor (k, sv, sign) the newest usable mark gives a GnssNavSymbols status row's
    stream, or None."""
    obs = row["nav_obs"]
    buf = np.array([s for _, s in obs["bits"]], dtype=np.int8)
    k_first = int(row["n_sym"]) - len(buf)
    t0 = float(row["t0"])
    sym_dt = float(obs["br"]) * float(obs["rec_dt"])
    for m in sorted(marks, key=lambda m: -m["t_host"]):
        if m["t_host"] < t0:
            break
        got = mark_sign(m, signal, buf, k_first, t0, sym_dt)
        if got is None:
            continue
        sv = sv_index(m, sysc, t_sym)
        if sv is not None:
            return got[0], sv + got[2], got[1]
    return None


def get_json(url, body=None):
    req = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5.0) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


def run(a):
    anchors = {}  # (sig, prn, stream_id) -> (k, sv, sign)
    seen = {}  # (sig, prn, stream_id) -> when its stream was last in a status
    t_post = 0.0
    marks_cache = {}
    while True:
        t = time.time()
        post = []
        for ep, (sysc, sig, t_sym, fname, signal) in SOURCES.items():
            path = os.path.join(a.marks_dir, fname)
            try:
                mt = os.stat(path).st_mtime_ns
                if marks_cache.get(path, (None,))[0] != mt:
                    with open(path) as fh:
                        marks_cache[path] = (mt, json.load(fh).get("marks", []))
                rows = get_json("%s/%s/get_status" % (a.rest_url, ep))
            except (OSError, ValueError) as e:
                log("%s: %s" % (ep, e))
                continue
            marks = [m for m in marks_cache[path][1] if m["signal"] == signal]
            for r in rows:
                if not r.get("nav_obs") or "stream_id" not in r:
                    continue
                prn, sid = int(r["prn"]), int(r["stream_id"])
                seen[(sig, prn, sid)] = t
                got = anchor_of(
                    r, [m for m in marks if m["prn"] == prn], signal, sysc, t_sym
                )
                if got is None:
                    continue
                key = (sig, prn, sid)
                old = anchors.get(key)
                if (
                    old is not None
                    and got[1] - old[1] == got[0] - old[0]
                    and got[2] == old[2]
                ):
                    continue
                if old is not None:
                    log(
                        "%s %s%02d stream %d: RE-ANCHORED, a mark puts symbol %d at %d sign %+d, "
                        "the anchor at %d sign %+d"
                        % (
                            ep,
                            sysc,
                            prn,
                            sid,
                            got[0],
                            got[1],
                            got[2],
                            old[1] + got[0] - old[0],
                            old[2],
                        )
                    )
                else:
                    log(
                        "%s %s%02d stream %d: anchored, symbol %d = satellite-time index %d, "
                        "sign %+d" % (ep, sysc, prn, sid, got[0], got[1], got[2])
                    )
                anchors[key] = got
                post.append(key)
        for key in [k for k in anchors if t - seen.get(k, t) > FORGET_S]:
            del anchors[key]
        if time.time() - t_post >= REPOST_S:
            t_post = time.time()
            post = list(anchors)
        if post:
            body = [
                {"sig": s, "prn": p, "stream_id": i, "k": k, "sv": sv, "sign": sg}
                for (s, p, i) in post
                for (k, sv, sg) in [anchors[(s, p, i)]]
            ]
            try:
                get_json("%s/%s/set_anchors" % (a.rest_url, a.sender), body)
            except (OSError, ValueError) as e:
                log("set_anchors failed (%s)" % e)
        time.sleep(max(0.0, a.every_s - (time.time() - t)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rest-url", default="http://localhost:12048")
    ap.add_argument(
        "--sender", default="navsym_send", help="the GnssNavSymbolSend stage"
    )
    ap.add_argument(
        "--marks-dir",
        default="/home/lwlab/gnss/bladerf/navdec",
        help="where nav_decode writes its --timemarks-file",
    )
    ap.add_argument("--every-s", type=float, default=2.0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
