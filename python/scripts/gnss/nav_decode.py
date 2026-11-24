#!/usr/bin/env python3
"""Nav-message decode in its own process: the receiver's symbols in, decoded ephemerides out.

The decoders (cnav_predictor, fnav_predictor, inav_predictor, bcnav2_predictor, bcnav3_predictor)
used to run inside the broker, whose chains share one interpreter: their CPU starved the broker's
cycle and every dead-reckoned chain with it. Here each --source names a signal and the endpoint
that serves its symbol decisions (a GnssNavSymbols stage, or for a data-only chain its combiner's
bit_export). Each reply holds a window of the newest decisions, which the decoders rescan whole, so
a source is polled again only once --window-frac of the window it returned has passed (and at
least --every seconds): the windows still overlap, as the decoders need to chain each window's
polarity to the last, but each decision is read a few times rather than at every poll. Every
--report-every seconds the job logs each satellite's decode state and its BRDC cross-check, as the
broker did, and writes:

  --eph-out             the decoded ephemerides, one JSON document replaced atomically, which
                        the broker reads with --decoded-eph-file for its decoded-ephemeris
                        fallback
  --decode-health-file  the viewer's decode-health panel (decode_health.py)

and, with --progress-file, each second: per satellite its decoded-message count and the times
it grew over the last two minutes, so a viewer can show messages arriving; with --payload-file,
every 2 s: per satellite its current ephemeris, decode state and the newest decode of each
message type, so a viewer can show the message contents.

and a line of CPU per decoder (ingest, and fetch plus JSON), so the load is visible. One process
is one interpreter: to spread decoders over cores, run several instances with disjoint sources
and their own output files.

    nav_decode.py --source GPS_L5_CNAV=http://localhost:12048/l5i_symbols
                  [--source SIGNAL=URL ...] --eph-out PATH [--decode-health-file PATH]
                  [--every 1] [--window-frac 0.75] [--report-every 60]
"""

import argparse
import collections
import importlib
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bds_d1_predictor  # noqa: E402
import decode_health  # noqa: E402
import decoded_eph  # noqa: E402
import gnss_ephemeris as ge  # noqa: E402
import glonass_nav_predictor  # noqa: E402
import nav_fields  # noqa: E402
import sbas_l5_predictor  # noqa: E402
from gnss_broker import sky  # noqa: E402

# signal -> (constellation, decoder module, class, BRDC cross-check, the ephemeris's toe field).
# The toe field travels with each exported ephemeris: the broker places it in GPS time with
# decoded_eph.toe_gpst_near.
DECODERS = {
    "GPS_L5_CNAV": (
        "G",
        "cnav_predictor",
        "CnavPredictor",
        sky._cnav_brdc_xcheck,
        "toe",
    ),
    "GPS_L2C_CNAV": (
        "G",
        "cnav_predictor",
        "CnavPredictor",
        sky._cnav_brdc_xcheck,
        "toe",
    ),
    "GAL_E5AI_FNAV": (
        "E",
        "fnav_predictor",
        "FnavPredictor",
        sky._fnav_brdc_xcheck,
        "t0e",
    ),
    "GAL_E5BI_INAV": (
        "E",
        "inav_predictor",
        "InavPredictor",
        sky._inav_brdc_xcheck,
        "t0e",
    ),
    "GAL_E1B_INAV": (
        "E",
        "inav_predictor",
        "InavPredictor",
        sky._inav_brdc_xcheck,
        "t0e",
    ),
    "BDS_B2A_BCNAV2": (
        "C",
        "bcnav2_predictor",
        "Bcnav2Predictor",
        sky._bcnav2_brdc_xcheck,
        "t_oe",
    ),
    "BDS_B2B_BCNAV3": (
        "C",
        "bcnav3_predictor",
        "Bcnav3Predictor",
        sky._bcnav3_brdc_xcheck,
        "t_oe",
    ),
    "BDS_B3I_D1": (
        "C",
        "bds_d1_predictor",
        "D1Predictor",
        bds_d1_predictor.brdc_xcheck,
        "toe_sow",
    ),
    "SBAS_L5_DFMC": (
        "S",
        "sbas_l5_predictor",
        "SbasL5Predictor",
        sbas_l5_predictor.brdc_xcheck,
        "toe_sow",
    ),
    "GLO_L2OF": (
        "R",
        "glonass_nav_predictor",
        "GloNavPredictor",
        glonass_nav_predictor.brdc_xcheck,
        "toe_gpst",
    ),
}

BRDC_REFRESH_S = 1800.0
# A decoder keeps the last ephemeris it decoded for a satellite long after the satellite set;
# one whose reference time is further than this from now is not reported, checked or exported.
EPH_FRESH_S = 4 * 3600.0
PROGRESS_EVERY_S = 1.0
PAYLOAD_EVERY_S = 2.0
PROGRESS_KEEP_S = 120.0


def log(msg):
    print("[nav_decode %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def get_rows(url, timeout=5.0):
    """{prn: row} of a get_status reply, the rows that carry a PRN."""
    with urllib.request.urlopen(url, timeout=timeout) as r:
        d = json.load(r)
    rows = d if isinstance(d, list) else d.get("prns", [])
    return {int(x["prn"]): x for x in rows if isinstance(x, dict) and x.get("prn")}


def decoded_count(h):
    """The decoded-message count in a decoder's health(): words, messages, frames or pages."""
    return h.get("words") or h.get("decoded") or h.get("frames") or h.get("pages") or 0


def note_progress(src, prn, now):
    """Record when a satellite's decoded-message count grows (a restart that lowers it is not
    an event)."""
    h = src["dec"].health(prn)
    n = decoded_count(h) if h else 0
    p = src["prog"].setdefault(prn, {"n": n, "ev": collections.deque()})
    if n > p["n"]:
        p["ev"].append((round(now, 2), n - p["n"]))
    p["n"] = n
    while p["ev"] and now - p["ev"][0][0] > PROGRESS_KEEP_S:
        p["ev"].popleft()


def message_types(signal, st):
    """{message type: [count or None, newest decode time or None, title, fields]} from a
    decoder's per-PRN state: I/NAV words, F/NAV pages, B-CNAV2/3 message types, D1 subframes
    (subframe 5's time-offset pages on their own), GLONASS strings, SBAS message types (counts),
    CNAV message types (counts); fields as nav_fields.parse reads the newest of each."""
    out = {}

    def put(key, kind, num, n, t, bits, frame=None):
        title, rows = nav_fields.parse(
            signal, kind, num, bits if bits is not None else [], frame
        )
        out[key] = [n, t, title, rows]

    if signal in ("GAL_E5BI_INAV", "GAL_E1B_INAV"):
        for k, v in st.words.items():
            put("word %d" % k, "word", k, None, v[2], v[1])
    elif signal == "GAL_E5AI_FNAV":
        for k, v in st.pages.items():
            put("page %d" % k, "page", k, None, v[2], v[1])
    elif signal in ("BDS_B2A_BCNAV2", "BDS_B2B_BCNAV3"):
        for k, v in st.frames.items():
            put("type %d" % k, "type", k, None, v[2], v[0])
    elif signal == "BDS_B3I_D1":
        newest = {}
        for (fra, _), (bits, t) in st.subframes.items():
            key = "subframe %d" % fra
            if fra == 5 and nav_fields._read(bits, 43, 7, "u") in (9, 10):
                key += " page %d" % nav_fields._read(bits, 43, 7, "u")
            if key not in newest or t > newest[key][1]:
                newest[key] = (fra, t, bits)
        for key, (fra, t, bits) in newest.items():
            put(key, "subframe", fra, None, t, bits)
    elif signal == "GLO_L2OF":
        newest = {}
        for ts, (m, b, t) in st.strings.items():
            if m not in newest or t > newest[m][1]:
                newest[m] = (ts, t, b)
        for m, (ts, t, b) in newest.items():
            put("string %d" % m, "string", m, None, t, b, int(ts % 150 // 30) + 1)
    elif signal == "SBAS_L5_DFMC":
        for k, n in st.types.items():
            put("type %d" % k, "type", k, n, None, None)
    elif signal in ("GPS_L5_CNAV", "GPS_L2C_CNAV"):
        count, newest = {}, {}
        for slot, bits in st.decoded.items():
            k = nav_fields._read(bits, 14, 6, "u")
            count[k] = count.get(k, 0) + 1
            if k not in newest or slot > newest[k][0]:
                newest[k] = (slot, bits)
        for k, (_, bits) in newest.items():
            put("type %d" % k, "type", k, count[k], None, bits)
    return out


def write_timemarks(srcs, path, now):
    """Each CRC-checked CNAV message as a time mark: the host-labelled UTC of its first
    symbol's edge and the TOW count it carries, to measure the capture's time anchor against.
    The edge is the satellite's newest, to a record (1 ms), and moves ~3 us/s."""
    marks = []
    for s in srcs:
        if s["signal"] not in ("GPS_L5_CNAV", "GPS_L2C_CNAV"):
            continue
        for prn, st in sorted(s["dec"]._p.items()):
            if not st.sym_s:
                continue
            for slot, bits in sorted(st.decoded.items()):
                marks.append(
                    {
                        "signal": s["signal"],
                        "prn": prn,
                        "t_host": round(slot * st.sym_s + st.edge_s, 5),
                        "tow": nav_fields._read(bits, 20, 17, "u"),
                        "type": nav_fields._read(bits, 14, 6, "u"),
                    }
                )
    write_atomic({"schema": 1, "t": round(now, 2), "marks": marks}, path)


def write_payload(srcs, path, now):
    doc = {"schema": 1, "t": round(now, 2), "signals": {}}
    for s in srcs:
        sysc, _, _, _, toef = DECODERS[s["signal"]]
        dec = s["dec"]
        prns = {}
        for prn, st in sorted(dec._p.items()):
            try:
                h = dec.health(prn)
                if not h:
                    continue
                eph = dec.ephemeris(prn)
                age = eph_age_s(sysc, eph, toef, now) if eph is not None else None
                if age is not None and abs(age) > EPH_FRESH_S:
                    eph, age = None, None
                prns[str(prn)] = {
                    "health": h,
                    "eph": eph,
                    "eph_age_s": None if age is None else round(age),
                    "msgs": message_types(s["signal"], st),
                }
            except Exception as e:
                log("%s PRN %d: payload failed (%s)" % (s["signal"], prn, e))
        doc["signals"][s["signal"]] = {"sys": sysc, "prns": prns}
    write_atomic(doc, path)


def write_progress(srcs, path, now):
    doc = {"schema": 1, "t": round(now, 2), "signals": {}}
    for s in srcs:
        doc["signals"][s["signal"]] = {
            "sys": DECODERS[s["signal"]][0],
            "prns": {
                str(prn): {"n": p["n"], "ev": list(p["ev"])}
                for prn, p in sorted(s["prog"].items())
            },
        }
    write_atomic(doc, path)


def window_s(obs):
    """The span of time a nav_obs window covers, s; 0 if it does not say."""
    try:
        return len(obs["bits"]) * int(obs["br"]) * float(obs["rec_dt"])
    except (KeyError, TypeError, ValueError):
        return 0.0


def jsonable(o):
    """json.dump's fallback for the decoders' numpy scalars and containers."""
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError("not JSON serializable: %r" % type(o))


def write_atomic(doc, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, default=jsonable)
    os.replace(tmp, path)


def eph_age_s(sysc, eph, toef, t_unix):
    """Seconds from the ephemeris's reference time to t_unix, or None when it carries none."""
    toe = eph.get(toef)
    if toe is None:
        return None
    now = ge.gpst_of_utc(t_unix)
    if toef != "toe_gpst":
        toe = decoded_eph.toe_gpst_near(sysc, toe, now)
    return now - toe


def make_decoder(signal):
    _, mod, cls, _, _ = DECODERS[signal]
    c = getattr(importlib.import_module(mod), cls)
    if cls != "CnavPredictor":
        return c()
    # CnavPredictor lets one PRN decode per _min_gap across all PRNs, sized for the broker's 1 s
    # loop. Here a poll ingests every PRN at once, a window apart, so the gate would let one
    # PRN decode per poll; its round-robin still shares the decodes out.
    d = c(log=log)
    d._min_gap = 0.0
    return d


def load_brdc(store, block):
    """Re-parse BRDC into the cross-check store ({"mod", "eph"}, as the broker's). A failure keeps
    the set in hand."""
    try:
        store["eph"] = ge.parse_rinex_nav(ge.fetch_brdc(block=block))
        store["t"] = time.time()
        log("BRDC loaded for the cross-check (%d sats)" % len(store["eph"]))
    except Exception as e:
        store["t"] = time.time()
        log("BRDC unavailable for the cross-check (%s)" % e)


def report(srcs, store, dhw, eph_out):
    """Log each satellite's decode state and BRDC cross-check, feed the decode-health export, and
    write the decoded ephemerides."""
    out = []
    now = time.time()
    brdc = store if store.get("eph") else None
    for s in srcs:
        sysc, _, _, xcheck, toef = DECODERS[s["signal"]]
        dec = s["dec"]
        for prn in sorted(dec._p):
            try:
                h = dec.health(prn)
                if not h:
                    continue
                eph = dec.ephemeris(prn)
                age = eph_age_s(sysc, eph, toef, now) if eph is not None else None
                if age is not None and abs(age) > EPH_FRESH_S:
                    eph = None
                xc = (
                    xcheck(brdc, sysc, prn, eph, log)
                    if (eph is not None and brdc)
                    else ""
                )
                log(
                    "%s PRN %d: %s, eph %s%s"
                    % (
                        s["signal"],
                        prn,
                        ", ".join(
                            "%s %s" % kv for kv in sorted(h.items()) if kv[0] != "have"
                        ),
                        "YES" if eph is not None else "no",
                        xc,
                    )
                )
                cnt = decoded_count(h)
                syn = h.get("synced")
                if syn is None:
                    syn = bool(cnt) or eph is not None
                if dhw is not None:
                    dhw.observe(
                        s["signal"],
                        prn,
                        now,
                        count=cnt or 0,
                        synced=bool(syn),
                        eph=eph is not None,
                        dpos_m=sky._dh_dpos(xc),
                    )
                if eph is not None and toef in eph:
                    out.append(
                        {
                            "signal": s["signal"],
                            "sys": sysc,
                            "prn": prn,
                            "toe_field": toef,
                            "eph": eph,
                            "dpos_m": sky._dh_dpos(xc),
                        }
                    )
            except Exception as e:
                log("%s PRN %d: report failed (%s)" % (s["signal"], prn, e))
    if dhw is not None:
        dhw.flush(now, force=True)
    if eph_out:
        write_atomic({"schema": 1, "t_unix": now, "eph": out}, eph_out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--source",
        action="append",
        required=True,
        help="SIGNAL=URL: a decoder (%s) and the endpoint serving its symbols"
        % ", ".join(sorted(DECODERS)),
    )
    ap.add_argument(
        "--eph-out", help="the decoded ephemerides (the broker's --decoded-eph-file)"
    )
    ap.add_argument(
        "--decode-health-file", help="decode-health export (decode_health.py)"
    )
    ap.add_argument(
        "--progress-file",
        help="per satellite, the decoded-message count and when it grew (rewritten each second)",
    )
    ap.add_argument(
        "--payload-file",
        help="per satellite, the current ephemeris, decode state and newest message types "
        "(rewritten every 2 s)",
    )
    ap.add_argument(
        "--timemarks-file",
        help="each decoded CNAV message's host-labelled start and TOW (rewritten every 2 s)",
    )
    ap.add_argument("--every", type=float, default=1.0, help="shortest poll period, s")
    ap.add_argument(
        "--window-frac",
        type=float,
        default=0.75,
        help="poll a source again after this fraction of the span its last reply covered",
    )
    ap.add_argument("--report-every", type=float, default=60.0, help="report period, s")
    args = ap.parse_args()

    srcs = []
    for spec in args.source:
        signal, _, url = spec.partition("=")
        if signal not in DECODERS or not url:
            ap.error(
                "--source %s: expected SIGNAL=URL with SIGNAL one of %s"
                % (spec, sorted(DECODERS))
            )
        srcs.append(
            {
                "signal": signal,
                "url": url.rstrip("/") + "/get_status",
                "dec": make_decoder(signal),
                "cpu": 0.0,
                "io": 0.0,
                "fail": 0,
                "next_t": 0.0,
                "prog": {},
            }
        )
    dhw = None
    if args.decode_health_file:
        dhw = decode_health.DecodeHealthWriter(
            args.decode_health_file,
            chain=os.path.basename(args.decode_health_file).rsplit(".", 1)[0],
            log=log,
        )
    store = {"mod": ge, "eph": None, "t": 0.0}
    load_brdc(store, block=True)
    log("decoding %s" % ", ".join("%s <- %s" % (s["signal"], s["url"]) for s in srcs))

    t_report = time.time()
    t_cpu = time.time()
    t_prog = t_pay = 0.0
    while True:
        for s in srcs:
            if time.time() < s["next_t"]:
                continue
            s["next_t"] = time.time() + args.every
            t0 = time.process_time()
            try:
                rows = get_rows(s["url"])
            except Exception as e:
                s["fail"] += 1
                if s["fail"] == 1 or s["fail"] % 60 == 0:
                    log("%s: %s unreadable (%s)" % (s["signal"], s["url"], e))
                continue
            t1 = time.process_time()
            span = 0.0
            for prn, r in rows.items():
                obs = r.get("nav_obs")
                if obs:
                    s["dec"].ingest(prn, obs)
                    span = max(span, window_s(obs))
                    if args.progress_file:
                        note_progress(s, prn, time.time())
            s["next_t"] = max(s["next_t"], time.time() + args.window_frac * span)
            s["io"] += t1 - t0
            s["cpu"] += time.process_time() - t1
        if (args.payload_file or args.timemarks_file) and (
            time.time() - t_pay >= PAYLOAD_EVERY_S
        ):
            t_pay = time.time()
            if args.payload_file:
                write_payload(srcs, args.payload_file, t_pay)
            if args.timemarks_file:
                write_timemarks(srcs, args.timemarks_file, t_pay)
        if args.progress_file and time.time() - t_prog >= PROGRESS_EVERY_S:
            t_prog = time.time()
            write_progress(srcs, args.progress_file, t_prog)
        if time.time() - store["t"] > BRDC_REFRESH_S:
            load_brdc(store, block=False)
        if time.time() - t_report >= args.report_every:
            t_report = time.time()
            report(srcs, store, dhw, args.eph_out)
            span = time.time() - t_cpu
            log(
                "cpu over %.0f s: %s"
                % (
                    span,
                    "; ".join(
                        "%s %.0f ms/s ingest, %.0f ms/s fetch"
                        % (s["signal"], 1e3 * s["cpu"] / span, 1e3 * s["io"] / span)
                        for s in srcs
                    ),
                )
            )
            for s in srcs:
                s["cpu"] = s["io"] = 0.0
            t_cpu = time.time()
        t_next = min([s["next_t"] for s in srcs] + [t_report + args.report_every])
        time.sleep(min(args.every, max(0.0, t_next - time.time())))


if __name__ == "__main__":
    main()
