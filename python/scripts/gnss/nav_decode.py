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
import beidou_bcnav2  # noqa: E402
import decode_health  # noqa: E402
import decoded_eph  # noqa: E402
import eph_cache  # noqa: E402
import galileo_fnav  # noqa: E402
import gnss_ephemeris as ge  # noqa: E402
import glonass_nav_predictor  # noqa: E402
import gnss_skytime  # noqa: E402
import nav_almanac  # noqa: E402
import nav_fields  # noqa: E402
import sbas_l5_predictor  # noqa: E402
from gnss_broker import sky  # noqa: E402

# signal -> (constellation, decoder module, class, BRDC cross-check, the ephemeris's toe field).
# The toe field travels with each exported ephemeris: the broker places it in GPS time with
# decoded_eph.toe_gpst_near.
DECODERS = {
    # GuoWang's low-rate message: the sender's own orbit and clock; no BRDC to check it against.
    "GW_1202_NAV": ("W", "gw_nav_predictor", "GwNavPredictor", None, "toe"),
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
    # E6-B's HAS pages: corrections to the broadcast orbits and clocks, no ephemeris.
    "GAL_E6B_CNAV": ("E", "has_predictor", "HasPredictor", None, "t0e"),
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


def message_types(signal, st, dec=None, prn=None):
    """{message type: [count or None, newest decode time or None, title, fields]} from a
    decoder's per-PRN state: I/NAV words, F/NAV pages, B-CNAV2/3 message types, D1 subframes
    (subframe 5's time-offset pages on their own), GLONASS strings, SBAS message types (counts,
    and the newest MT32 of each satellite slot), CNAV message types (counts), the newest C/NAV
    page and the HAS corrections for the satellite (from dec); fields as nav_fields.parse reads
    the newest of each."""
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
            t, bits = st.newest.get(k, (None, None))
            put("type %d" % k, "type", k, n, t, bits)
        for slot, (t, bits) in sorted(st.mt32.items()):
            put("type 32 slot %d" % slot, "type", 32, None, t, bits)
    elif signal == "GAL_E6B_CNAV":
        if st.page is not None:
            put("page", "page", 0, st.n_pages, st.page[1], st.page[0])
        for kind, (t, d) in sorted(dec.newest.items()) if dec is not None else ():
            rows = nav_fields.has_rows(d, "E%02d" % prn, kind)
            if rows:
                title = "HAS %s (message %d, TOH %d s)" % (kind, d["mid"], d["toh"])
                out["HAS " + kind] = [None, t, title, rows]
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


def timemarks(srcs):
    """Each CRC-checked message as a time mark, to measure the capture's time anchor against
    (gnss_skytime.py): the host-labelled UTC of its first symbol's edge, to a record (1 ms), and
    the time count it carries, as broadcast: a CNAV TOW count (6 s), an F/NAV page's TOW (s), a
    B-CNAV2 frame's SOW count (3 s), a B-CNAV3 frame's BDT SOW (s), a GuoWang subframe's BDT SOW
    (s), a GLONASS string's start as GPS seconds of week (strings 1-4, once its satellite's
    strings are dated: glonass_nav_predictor's utc_off). With it the week number
    where the message has one (a CNAV type 10, F/NAV pages 1-3, a B-CNAV2 type 10, GuoWang page
    1, every GLONASS mark). Each message's edge is placed when it decodes:
    edges move ~3 us/s against the slot grid, and decoders keep messages for minutes. A CNAV
    mark carries the message too ("bits", hex), which nav_symbol_feed re-encodes to find the
    symbol stream's sign."""
    marks = []
    for s in srcs:
        sig = s["signal"]
        for prn, st in sorted(s["dec"]._p.items()):
            if sig in ("GPS_L5_CNAV", "GPS_L2C_CNAV") and st.sym_s:
                for slot, bits in sorted(st.decoded.items()):
                    mark = {
                        "signal": sig,
                        "prn": prn,
                        "t_host": round(st.t_host[slot], 5),
                        "tow": nav_fields._read(bits, 20, 17, "u"),
                        "type": nav_fields._read(bits, 14, 6, "u"),
                        "bits": "%075x" % int("".join(str(int(b)) for b in bits), 2),
                    }
                    if mark["type"] == 10:
                        mark["wn"] = nav_fields._read(bits, 38, 13, "u")
                    marks.append(mark)
            elif sig == "GAL_E5AI_FNAV":
                for pt, (_, bits, _, t_host) in sorted(st.pages.items()):
                    if pt not in galileo_fnav.FNAV_TIME:
                        continue
                    wn_at, tow_at = galileo_fnav.FNAV_TIME[pt]
                    mark = {
                        "signal": sig,
                        "prn": prn,
                        "t_host": round(t_host, 5),
                        "tow": nav_fields._read(bits, tow_at, 20, "u"),
                        "type": pt,
                    }
                    if wn_at is not None:
                        mark["wn"] = nav_fields._read(bits, wn_at, 12, "u")
                    marks.append(mark)
            elif sig == "BDS_B2A_BCNAV2":
                for mt, (bits, sow, _, _, t_host) in sorted(st.frames.items()):
                    mark = {
                        "signal": sig,
                        "prn": prn,
                        "t_host": round(t_host, 5),
                        "tow": sow,
                        "type": mt,
                    }
                    if mt == 10:
                        mark["wn"] = nav_fields._read(bits, *beidou_bcnav2.WN_10, "u")
                    marks.append(mark)
            elif sig == "BDS_B2B_BCNAV3":
                for sow, (t_host, mt) in sorted(st.marks.items()):
                    marks.append(
                        {
                            "signal": sig,
                            "prn": prn,
                            "t_host": round(t_host, 5),
                            "tow": sow,
                            "type": mt,
                        }
                    )
            elif sig == "GW_1202_NAV":
                for sow, (page, t_host, wn) in sorted(st.marks.items()):
                    mark = {
                        "signal": sig,
                        "prn": prn,
                        "t_host": round(t_host, 5),
                        "tow": sow,
                        "type": page,
                    }
                    if wn is not None:
                        mark["wn"] = wn
                    marks.append(mark)
            elif sig == "GLO_L2OF" and st.utc_off is not None:
                for ts, (m, _, _) in sorted(st.strings.items()):
                    if m > 4 or ts not in st.edges:
                        continue
                    g = round(ge.gpst_of_utc(ts + st.utc_off))
                    marks.append(
                        {
                            "signal": sig,
                            "prn": prn,
                            "t_host": round(st.edges[ts], 5),
                            "tow": g % 604800,
                            "wn": g // 604800,
                            "type": m,
                        }
                    )
    return marks


def write_timemarks(srcs, path, now):
    write_atomic({"schema": 1, "t": round(now, 2), "marks": timemarks(srcs)}, path)


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
                    "msgs": message_types(s["signal"], st, dec, prn),
                }
            except Exception as e:
                log("%s PRN %d: payload failed (%s)" % (s["signal"], prn, e))
        doc["signals"][s["signal"]] = {"sys": sysc, "prns": prns}
        if hasattr(dec, "corrections"):
            doc["signals"][s["signal"]]["has"] = dec.corrections()
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


def report(srcs, store, dhw, eph_out, cache=None, alm_cache=None):
    """Log each satellite's decode state and BRDC cross-check, feed the decode-health export, and
    write the decoded ephemerides: those within EPH_FRESH_S of the host clock to eph_out, and
    every one to the cache (eph_cache.Writer), placed in the week the sky's own time marks give
    (gnss_skytime.week_ref) where they give one, else the host clock's. The almanacs read since
    the last report go to alm_cache (nav_almanac.py)."""
    out, cached = [], []
    now = time.time()
    brdc = store if store.get("eph") else None
    g_sky = gnss_skytime.week_ref(timemarks(srcs)) if cache is not None else None
    g_ref = g_sky if g_sky is not None else ge.gpst_of_utc(now)
    for s in srcs:
        sysc, _, _, xcheck, toef = DECODERS[s["signal"]]
        dec = s["dec"]
        for prn in sorted(dec._p):
            try:
                h = dec.health(prn)
                if not h:
                    continue
                eph = dec.ephemeris(prn)
                if cache is not None and eph is not None and toef in eph:
                    cached.append(
                        {
                            "sys": sysc,
                            "prn": prn,
                            "signal": s["signal"],
                            "toe_field": toef,
                            "eph": eph,
                            "toe_gpst": (
                                eph[toef]
                                if toef == "toe_gpst"  # GLONASS's: continuous already
                                else decoded_eph.toe_gpst_near(sysc, eph[toef], g_ref)
                            ),
                            "week_from": "sky" if g_sky is not None else "host",
                        }
                    )
                age = eph_age_s(sysc, eph, toef, now) if eph is not None else None
                if age is not None and abs(age) > EPH_FRESH_S:
                    eph = None
                xc = (
                    xcheck(brdc, sysc, prn, eph, log)
                    if (eph is not None and brdc and xcheck)
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
    if cache is not None and cached:
        try:
            cache.put(cached, now)
        except (OSError, ValueError) as e:
            log("ephemeris cache %s not written (%s)" % (cache.path, e))
    if alm_cache is not None:
        alm = [e for s in srcs if s["alm"] is not None for e in s["alm"].take(g_ref)]
        try:
            new = alm_cache.put(alm, now) if alm else 0
            if new:
                log(
                    "almanacs: %d satellites read, %d new issues cached"
                    % (len({(e["sys"], e["prn"]) for e in alm}), new)
                )
        except (OSError, ValueError) as e:
            log("almanac cache %s not written (%s)" % (alm_cache.path, e))


def main():
    global EPH_FRESH_S
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
        "--eph-cache",
        nargs="?",
        const=eph_cache.DEFAULT_DIR,
        help="also keep every decoded issue in this ephemeris cache directory (eph_cache.py; "
        "alone, %s), the broker's decoded orbit feed, and the almanacs read (nav_almanac.py) "
        "in a file of their own there" % eph_cache.DEFAULT_DIR,
    )
    ap.add_argument(
        "--eph-cache-name",
        help="this decoder's file in the cache (default: its signals); one process per name",
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
        help="each decoded CNAV, F/NAV or B-CNAV2 message's host-labelled start and time "
        "count (rewritten every 2 s)",
    )
    ap.add_argument("--every", type=float, default=1.0, help="shortest poll period, s")
    ap.add_argument(
        "--window-frac",
        type=float,
        default=0.75,
        help="poll a source again after this fraction of the span its last reply covered",
    )
    ap.add_argument("--report-every", type=float, default=60.0, help="report period, s")
    ap.add_argument(
        "--eph-fresh-s",
        type=float,
        default=EPH_FRESH_S,
        help="export no ephemeris whose reference is further than this from the host clock; "
        "0 exports every one (gnss_bootstrap.py's, for a host clock that is wrong)",
    )
    ap.add_argument(
        "--no-brdc",
        action="store_true",
        help="no BRDC cross-check (and no fetch for it)",
    )
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
                "alm": (
                    nav_almanac.Collector(signal)
                    if signal in nav_almanac.SIGNALS
                    else None
                ),
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
    cache = alm_cache = None
    if args.eph_cache:
        name = args.eph_cache_name or "+".join(sorted({s["signal"] for s in srcs}))
        cache = eph_cache.Writer(name, args.eph_cache)
        log("ephemeris cache %s (%d issues held)" % (cache.path, len(cache.entries)))
        if any(s["alm"] is not None for s in srcs):
            alm_cache = eph_cache.Writer(name + ".alm", args.eph_cache)
    EPH_FRESH_S = args.eph_fresh_s if args.eph_fresh_s > 0 else float("inf")
    store = {"mod": ge, "eph": None, "t": 0.0 if not args.no_brdc else float("inf")}
    if not args.no_brdc:
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
            if s["alm"] is not None:
                try:
                    s["alm"].scan(s["dec"])
                except Exception as e:
                    log("%s: almanac read failed (%s)" % (s["signal"], e))
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
            report(srcs, store, dhw, args.eph_out, cache, alm_cache)
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
