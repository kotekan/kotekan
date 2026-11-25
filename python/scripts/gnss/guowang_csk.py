#!/usr/bin/env python3
"""GuoWang 1202.025 MHz high-rate (CSK) messages: framing, LDPC, CRC and the relayed GNSS
ephemerides.

Each UTC second carries 500 six-bit symbols, the CSK shift of each 2 ms frame, which
GnssCskDespread publishes: slots 0-3 the sync word 56 36 55 40, then three systematic LDPC
codewords over GF(64) (x^6 + x + 1), the shift being the GF symbol: slots 4-203 and 204-403
with BeiDou B1C subframe 2's (200,100) code, 404-499 with B2a's (96,48). Their info symbols hold
1488 bits, a 1464-bit message and a CRC-24Q over it. The layout is the site's
(docs/chord/guowang/PAYLOAD.md):

  0-11      shell flag (bit 2) and sender ID (bits 4-11); BDT second of week 16/20; week 36/13
  49-538    record 1: one GNSS satellite's orbit and clock
  539-1028  record 2
  1029-1308 one GuoWang almanac entry (not parsed here)

A record (offset/width): system 0/3 (0 BeiDou, 1 GPS, 2 Galileo), PRN 3/6, toe = BDT second
of week / 20 12/15, M0 29/35, e 64/28 (2^-33), sqrtA 92/34 (2^-21), i0 126/35, Omega0 176/35,
omega 211/35 (angles in semicircles x 2^-34; Omega0 takes the -omega_e toe term at the BDT
second of week), Cus 246/20 and Cuc 266/20 (2^-33 rad), af0 316/30 (2^-36 s, on one time scale
some 415 ns from the broadcast clocks), af1 350/12 (2^-44). A plain Kepler orbit with the two
harmonics and no secular terms: the site found it within 1 m (BeiDou) and 12 m (GPS, Galileo)
of the broadcast at the epoch.

    guowang_csk.py selftest
    guowang_csk.py find FILE                  a satellite's CSK code from a GnssCskDespread capture
    guowang_csk.py capture FILE G2            decode a capture with that CSK code
    guowang_csk.py live --url http://localhost:12048/gw_csk --log-dir DIR [--relay-out FILE]
"""

import argparse
import collections
import json
import math
import os
import sys
import time
import urllib.request

import numpy as np
from scipy.special import i0e

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gnss_ephemeris as ge  # noqa: E402
import nb_ldpc  # noqa: E402
from beidou_bcnav1 import H_SF2_ELE, H_SF2_IDX  # noqa: E402
from beidou_bcnav2 import H_BCNV2_ELE, H_BCNV2_IDX  # noqa: E402
from galileo_inav import crc24q  # noqa: E402

SYNC = (56, 36, 55, 40)
SLOTS = 500
# (first slot, info symbols, codeword symbols, H)
CODEWORDS = (
    (4, 100, 200, H_SF2_IDX, H_SF2_ELE),
    (204, 100, 200, H_SF2_IDX, H_SF2_ELE),
    (404, 48, 96, H_BCNV2_IDX, H_BCNV2_ELE),
)
MSG_BITS = 1464
RECORDS = (49, 539)
SYSTEM = {0: "C", 1: "G", 2: "E"}
MAX_PRN = {"C": 63, "G": 32, "E": 36}
BDT_WEEK0 = 1356  # GPS week of BDT week 0
SEMI = math.pi * 2.0**-34
# The fewest symbols of a second worth an LDPC attempt: belief propagation recovers a quarter
# of the symbols missing at 12 dB.
MIN_SLOTS = 300
# A code whose last FAIL_RUN seconds all failed to decode is tried one second in FAIL_EVERY until
# one decodes: a failing second costs belief propagation its full 50 iterations (a decoding one
# a median of 1), and on 11-25 three codes that never decoded took half of this process.
FAIL_RUN = 5
FAIL_EVERY = 8


def log(msg):
    print("[guowang_csk %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def gw1202_codes():
    """Every G2' phase's 2046-chip code as +-1, [2047][2046], as guowang1202Code.cpp builds them:
    B1I's G1 from 01010101010, 1206 chips in, XOR G2' (1 + x + x^3 + x^8 + x^9 + x^10 + x^11
    from all ones) at the phase."""
    s = [1] * 11
    for k in range(11, 2047):
        s.append(s[k - 1] ^ s[k - 3] ^ s[k - 8] ^ s[k - 9] ^ s[k - 10] ^ s[k - 11])
    g2 = np.array(s, dtype=np.uint8)
    r = [i % 2 for i in range(11)]
    g1 = []
    for k in range(1206 + 2046):
        if k >= 1206:
            g1.append(r[0])
        r = r[1:] + [r[0] ^ r[1] ^ r[2] ^ r[3] ^ r[4] ^ r[10]]
    idx = (np.arange(2047)[:, None] + np.arange(2046)[None, :]) % 2047
    return 1 - 2 * (np.array(g1, dtype=np.uint8)[None, :] ^ g2[idx]).astype(np.int8)


def read_capture(path):
    """A GnssCskDespread capture: per frame its id, UTC, Doppler and 2046 chip outputs."""
    rec = np.dtype([("id", "<i8"), ("utc", "<f8"), ("dop", "<f8"), ("z", "<c8", 2046)])
    return np.fromfile(path, dtype=rec)


def find(path):
    """The CSK code of a capture's satellite: every G2' phase correlated at every lag over 40
    frames; the right one peaks in the 64 shifts of each (lags -63..0)."""
    d = read_capture(path)
    z = d["z"][:: max(1, len(d) // 40)][:40].astype(np.complex128)
    zc = np.conj(np.fft.fft(np.conj(z), axis=1))  # ifft(zc * C)_k = sum_j z_j c_{j+k}
    res = []
    for g2, c in enumerate(gw1202_codes()):
        p = np.abs(np.fft.ifft(zc * np.fft.fft(c)[None, :], axis=1)) ** 2
        lag = p.argmax(axis=1)
        inside = np.mean((lag == 0) | (lag >= 2046 - 63))
        res.append((float(np.median(p.max(axis=1) / p.mean(axis=1))), inside, g2))
    res.sort(reverse=True)
    for score, inside, g2 in res[:3]:
        print(
            "G2' %4d: peak %.0f x the mean, %.0f%% of frames peak in the 64 shifts"
            % (g2, score, 100 * inside)
        )
    return res[0]


def bits_u(b, o, n):
    v = 0
    for x in b[o : o + n]:
        v = (v << 1) | int(x)
    return v


def bits_s(b, o, n):
    v = bits_u(b, o, n)
    return v - (1 << n) if v >> (n - 1) else v


def loglik_from_powers(p):
    """Each slot's 64 shift powers over the noise -> the non-coherent log-likelihood of each
    value, log I0(2 sqrt(S P)), S the signal power: the second's median of the strongest shift's
    excess over the noise. A slot of zeros is an erasure (a flat row)."""
    p = np.maximum(np.asarray(p, dtype=float), 0.0)
    pmax = p.max(axis=1)
    have = pmax > 0.0
    if not have.any():
        return np.zeros_like(p)
    s = max(float(np.median(pmax[have])) - 1.0, 1.0)
    x = 2.0 * np.sqrt(s * p)
    ll = np.log(i0e(x)) + x
    ll[~have] = 0.0
    return ll


def powers_from_top(top):
    """[k1, db1, ..., k4, db4] (GnssCskDespread's four strongest, dB over the rest) -> 64 powers,
    the other 60 at the noise."""
    p = np.ones(64)
    for i in range(0, len(top), 2):
        if top[i + 1] is None:  # -inf dB (a frame with no power) reads as JSON null
            continue
        p[int(top[i])] = 10.0 ** (float(top[i + 1]) / 10.0)
    return p


def decode_second(ll):
    """500 x 64 symbol log-likelihoods (slot 0 the sync word's first) -> (1488 info bits,
    symbols corrected) when the three codewords decode (nb_ldpc.decode_bp) and the CRC holds,
    else (None, -1)."""
    info, nerr = [], 0
    for first, m, n, idx, ele in CODEWORDS:
        bits, ne = nb_ldpc.decode_bp(idx, ele, m, n, ll[first : first + n])
        if ne < 0:
            return None, -1
        info.append(bits)
        nerr += ne
    b = np.concatenate(info)
    if crc24q(b[:MSG_BITS]) != bits_u(b, MSG_BITS, 24):
        return None, -1
    return b, nerr


def parse(b):
    """A CRC-valid message -> {sender, shell, sow, week, records}. Each record is an ephemeris
    in gnss_ephemeris form (sat_pos_clk), toe on GPS time."""
    week, sow = bits_u(b, 36, 13), bits_u(b, 16, 20)
    msg = {
        "sender": bits_u(b, 4, 8),
        "shell": int(b[2]),
        "sow": sow,
        "week": week,
        "records": [],
    }
    for o in RECORDS:
        sysc = SYSTEM.get(bits_u(b, o, 3))
        prn = bits_u(b, o + 3, 6)
        sqrta = bits_u(b, o + 92, 34) * 2.0**-21
        toe = 20.0 * bits_u(b, o + 12, 15)
        dt = (toe - sow + 302400) % 604800 - 302400
        # The senders outside the almanac fill their records (0x65 repeated, or zeros): only a
        # GNSS orbit with an epoch near the message's is a record.
        if (
            sysc is None
            or not 0 < prn <= MAX_PRN[sysc]
            or not 5000 < sqrta < 6600
            or abs(dt) > 3600
        ):
            continue
        wk = week + (1 if toe - sow < -302400 else -1 if toe - sow > 302400 else 0)
        toe_gpst = (
            (wk + BDT_WEEK0) * 604800.0 + toe - ge.BDT_GPST
        )  # BDT_GPST = BDT - GPST
        msg["records"].append(
            {
                "sys": sysc,
                "prn": prn,
                "sqrta": sqrta,
                "ecc": bits_u(b, o + 64, 28) * 2.0**-33,
                "m0": bits_s(b, o + 29, 35) * SEMI,
                "i0": bits_s(b, o + 126, 35) * SEMI,
                "omega0": bits_s(b, o + 176, 35) * SEMI,
                "omega": bits_s(b, o + 211, 35) * SEMI,
                "cus": bits_s(b, o + 246, 20) * 2.0**-33,
                "cuc": bits_s(b, o + 266, 20) * 2.0**-33,
                "crs": 0.0,
                "crc": 0.0,
                "cis": 0.0,
                "cic": 0.0,
                "dn": 0.0,
                "idot": 0.0,
                "omegadot": 0.0,
                "toe_sow": toe,
                "toe_gpst": toe_gpst,
                "toc_gpst": toe_gpst,
                "af0": bits_s(b, o + 316, 30) * 2.0**-36,
                "af1": bits_s(b, o + 350, 12) * 2.0**-44,
                "af2": 0.0,
            }
        )
    return msg


# The relay's clocks sit on one time scale this far below the broadcast ones (the site's
# five-day archive and the live check here agree to a few ns; BeiDou scatters ~13 ns with its
# group delays), s. Added back where a relayed clock seeds code phase.
CLOCK_OFFSET = {"G": 412e-9, "E": 416e-9, "C": 433e-9}


def sv_position_relay(eph, t_sow, wrap=True):
    """A relayed orbit at t_sow, the satellite system's seconds of week (BDT for BeiDou, as the
    broker's decoded-ephemeris fallback feeds them): decoded_eph's propagator for GW_CSK_RELAY.
    wrap=False: t_sow counts on from the start of toe's week.
    """
    tk = t_sow - eph["toe"]
    if wrap:
        tk = (tk + 302400) % 604800 - 302400
    return ge.sat_pos_clk(eph, eph["toe_gpst"] + tk)[0]


def fallback_eph(rec):
    """A relayed record in the form decoded_eph reads (nav_decode's --eph-out entries): toe in
    the system's own seconds of week, the clock offset added back, and the Kepler and clock
    names its relativistic term looks up."""
    toe = (
        rec["toe_sow"] if rec["sys"] == "C" else (rec["toe_sow"] - ge.BDT_GPST) % 604800
    )
    e = dict(
        rec, toe=toe, toc=toe, M0=rec["m0"], dn=0.0, sqrtA=rec["sqrta"], e=rec["ecc"]
    )
    e["af0"] = rec["af0"] + CLOCK_OFFSET[rec["sys"]]
    return {
        "signal": "GW_CSK_RELAY",
        "sys": rec["sys"],
        "prn": rec["prn"],
        "toe_field": "toe",
        "eph": e,
    }


def xcheck(brdc, rec):
    """The relayed orbit against the broadcast one at the relay's epoch: (dpos m, dclk ns)."""
    recs = brdc.get((rec["sys"], rec["prn"])) if brdc else None
    be = ge.best_eph(recs, rec["toe_gpst"]) if recs else None
    if be is None:
        return None, None
    a, _, ca = ge.sat_pos_clk(be, rec["toe_gpst"])
    b, _, cb = ge.sat_pos_clk(rec, rec["toe_gpst"])
    return math.dist(a, b), (cb - ca) * 1e9


def describe(msg, brdc):
    out = []
    for r in msg["records"]:
        dp, dc = xcheck(brdc, r)
        out.append(
            "%s%02d toe %d%s"
            % (
                r["sys"],
                r["prn"],
                r["toe_sow"],
                "" if dp is None else " (%.1f m, clock %+.0f ns vs BRDC)" % (dp, dc),
            )
        )
    return "sender %d, BDT %d/%d: %s" % (
        msg["sender"],
        msg["week"],
        msg["sow"],
        "; ".join(out) or "no records",
    )


def selftest():
    """Random messages, encoded, at 9 dB a symbol (about 15 % of symbols wrong) with 15 % of
    them missing: each must come back."""
    rng = np.random.RandomState(3)
    ok = 0
    for trial in range(3):
        msg = rng.randint(0, 2, MSG_BITS)
        crc = crc24q(msg)
        bits = np.concatenate([msg, [(crc >> (23 - j)) & 1 for j in range(24)]])
        syms = list(SYNC)
        pos = 0
        for first, m, n, idx, ele in CODEWORDS:
            info = nb_ldpc.bin2gf(bits[pos : pos + 6 * m])
            pos += 6 * m
            syms += list(
                nb_ldpc.bin2gf(nb_ldpc.encode_systematic(idx, ele, m, n, info))
            )
        p = rng.exponential(1.0, (SLOTS, 64))
        z = (rng.randn(SLOTS) + 1j * rng.randn(SLOTS)) / math.sqrt(2) + 10**0.45
        p[np.arange(SLOTS), syms] = np.abs(z) ** 2
        p[rng.rand(SLOTS) < 0.15] = 0.0
        b, nerr = decode_second(loglik_from_powers(p))
        good = b is not None and np.array_equal(b[:MSG_BITS], msg)
        ok += good
        print(
            "trial %d: %s (%d symbol errors corrected)"
            % (trial, "PASS" if good else "FAIL", nerr)
        )
    return ok == 3


def capture(path, g2):
    """Decode every whole second of a GnssCskDespread capture (all 64 shifts per frame)."""
    n = 2046
    d = read_capture(path)
    c = gw1202_codes()[g2].astype(float)
    z = d["z"].astype(np.complex128)
    corr = np.fft.ifft(
        np.conj(np.fft.fft(np.conj(z), axis=1)) * np.fft.fft(c)[None, :], axis=1
    )
    pw = np.abs(corr[:, (-np.arange(64)) % n]) ** 2
    pw /= np.median(pw, axis=1, keepdims=True) / math.log(
        2
    )  # exponential noise: median ln 2
    by_id = {int(i): p for i, p in zip(d["id"], pw)}
    k = {i: int(np.argmax(p)) for i, p in by_id.items()}
    starts = [i for i in by_id if all(k.get(i + j) == SYNC[j] for j in range(4))]
    brdc = ge.parse_rinex_nav(ge.fetch_brdc(block=True))
    for s in sorted(starts):
        p = np.zeros((SLOTS, 64))
        for j in range(SLOTS):
            if s + j in by_id:
                p[j] = by_id[s + j]
        have = int((p.max(axis=1) > 0).sum())
        if have < MIN_SLOTS:
            print("second at %d: %d of 500 slots, skipped" % (s, have))
            continue
        b, nerr = decode_second(loglik_from_powers(p))
        if b is None:
            print("second at %d: no decode" % s)
            continue
        print(
            "second at %d (%d symbol errors): %s" % (s, nerr, describe(parse(b), brdc))
        )


def get_json(url, timeout=3.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def write_atomic(doc, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh)
    os.replace(tmp, path)


def live(args):
    """Poll GnssCskDespread, frame each satellite's symbols on the sync word, decode each whole
    second once, log the messages and keep the newest relayed ephemeris per GNSS satellite.
    """
    import decode_health

    url = args.url.rstrip("/") + "/get_status"
    store = {"eph": None, "t": 0.0}
    syms = collections.defaultdict(dict)  # code -> frame id -> [k1, db1, ..., k4, db4]
    phase = {}  # code -> second-start frame id mod 500
    tried = collections.defaultdict(set)  # code -> second starts attempted
    relay = {}  # (sys, prn) -> newest relayed record and its check
    fallback = {}  # (sys, prn) -> its decoded_eph entry
    stats = collections.Counter()
    failing = collections.Counter()  # code -> seconds failed in a row
    prog = collections.defaultdict(lambda: {"n": 0, "ev": collections.deque()})
    dhw = (
        decode_health.DecodeHealthWriter(
            args.decode_health_file, chain="gw_csk", log=log
        )
        if args.decode_health_file
        else None
    )
    t_report = t_prog = 0.0
    while True:
        now = time.time()
        if now - store["t"] > 1800:
            try:
                store["eph"] = ge.parse_rinex_nav(
                    ge.fetch_brdc(block=store["eph"] is None)
                )
            except Exception as e:
                log("BRDC unavailable for the cross-check (%s)" % e)
            store["t"] = now
        try:
            rows = get_json(url)
        except Exception as e:
            log("%s unreadable (%s)" % (url, e))
            time.sleep(5)
            continue
        for r in rows:
            h = syms[r["prn"]]
            for row in r["sym"]:
                h[int(row[0])] = row[1:]
        for code, h in syms.items():
            if not h:
                continue
            newest = max(h)
            for i in [i for i in h if i < newest - 3 * SLOTS]:
                del h[i]
            k = {i: v[0] for i, v in h.items()}
            starts = [i for i in h if all(k.get(i + j) == SYNC[j] for j in range(4))]
            if starts:
                phase[code] = max(starts) % SLOTS
            if code not in phase:
                continue
            first = min(h) + (phase[code] - min(h)) % SLOTS
            for s in range(first, newest - SLOTS + 2, SLOTS):
                if s in tried[code]:
                    continue
                tried[code].add(s)
                p = np.zeros((SLOTS, 64))
                for j in range(SLOTS):
                    if s + j in h:
                        p[j] = powers_from_top(h[s + j])
                if (p.max(axis=1) > 0).sum() < MIN_SLOTS:
                    stats["short"] += 1
                    continue
                if failing[code] >= FAIL_RUN and (s // SLOTS) % FAIL_EVERY:
                    stats["skipped"] += 1
                    continue
                b, nerr = decode_second(loglik_from_powers(p))
                if b is None:
                    stats["fail"] += 1
                    failing[code] += 1
                    continue
                failing[code] = 0
                msg = parse(b)
                stats["ok"] += 1
                pg = prog[code]
                pg["n"] += 1
                pg["ev"].append((round(now, 2), 1))
                out = []
                for rec in msg["records"]:
                    dp, dc = xcheck(store["eph"], rec)
                    key = (rec["sys"], rec["prn"])
                    relay[key] = {
                        "sys": rec["sys"],
                        "prn": rec["prn"],
                        "toe_unix": rec["toe_gpst"] + now - ge.gpst_of_utc(now),
                        "af0": rec["af0"],
                        "dpos_m": None if dp is None else round(dp, 2),
                        "dclk_ns": None if dc is None else round(dc, 1),
                        "code": code,
                        "sender": msg["sender"],
                        "t_unix": now,
                    }
                    out.append(dict(relay[key], eph=rec))
                    fallback[key] = dict(fallback_eph(rec), dpos_m=relay[key]["dpos_m"])
                if args.log_dir:
                    path = os.path.join(
                        args.log_dir,
                        "gw_csk_%s.jsonl" % time.strftime("%Y%m%d", time.gmtime()),
                    )
                    with open(path, "a") as fh:
                        fh.write(
                            json.dumps(
                                {
                                    "t_unix": now,
                                    "code": code,
                                    "sender": msg["sender"],
                                    "week": msg["week"],
                                    "sow": msg["sow"],
                                    "nerr": nerr,
                                    "records": out,
                                }
                            )
                            + "\n"
                        )
                if args.verbose:
                    log(
                        "code %d (%d symbols corrected): %s"
                        % (code, nerr, describe(msg, store["eph"]))
                    )
            while len(tried[code]) > 20:
                tried[code].remove(min(tried[code]))
        if args.relay_out:
            write_atomic(
                {"schema": 1, "t_unix": now, "relay": list(relay.values())},
                args.relay_out,
            )
        if args.eph_out:
            write_atomic(
                {"schema": 1, "t_unix": now, "eph": list(fallback.values())},
                args.eph_out,
            )
        if args.progress_file and now - t_prog >= 1.0:
            t_prog = now
            for pg in prog.values():
                while pg["ev"] and now - pg["ev"][0][0] > 120:
                    pg["ev"].popleft()
            write_atomic(
                {
                    "schema": 1,
                    "t": round(now, 2),
                    "signals": {
                        "GW_CSK": {
                            "sys": "W",
                            "prns": {
                                str(c): {"n": pg["n"], "ev": list(pg["ev"])}
                                for c, pg in prog.items()
                            },
                        }
                    },
                },
                args.progress_file,
            )
        if now - t_report >= args.report_every:
            t_report = now
            fresh = [v for v in relay.values() if now - v["t_unix"] < 600]
            dps = sorted(v["dpos_m"] for v in fresh if v["dpos_m"] is not None)
            log(
                "messages: %d decoded, %d failed, %d short, %d skipped (codes failing: %s); "
                "%d GNSS satellites relayed in 10 min%s"
                % (
                    stats["ok"],
                    stats["fail"],
                    stats["short"],
                    stats["skipped"],
                    ",".join(str(c) for c in sorted(failing) if failing[c] >= FAIL_RUN)
                    or "none",
                    len(fresh),
                    ", median %.1f m from BRDC" % dps[len(dps) // 2] if dps else "",
                )
            )
            if dhw is not None:
                for code, pg in prog.items():
                    dhw.observe(
                        "GW_CSK",
                        code,
                        now,
                        count=pg["n"],
                        synced=code in phase,
                        eph=False,
                    )
                dhw.flush(now, force=True)
        time.sleep(args.every)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    f = sub.add_parser("find")
    f.add_argument("file")
    c = sub.add_parser("capture")
    c.add_argument("file")
    c.add_argument("g2", type=int, help="the satellite's CSK G2' phase")
    lv = sub.add_parser("live")
    lv.add_argument("--url", default="http://localhost:12048/gw_csk")
    lv.add_argument(
        "--log-dir", help="append each message to gw_csk_YYYYMMDD.jsonl here"
    )
    lv.add_argument(
        "--relay-out", help="the newest relayed ephemeris per GNSS satellite (JSON)"
    )
    lv.add_argument(
        "--eph-out",
        help="the same as decoded ephemerides (nav_decode's --eph-out form) for the broker's "
        "--decoded-eph-file, where they stand behind every other decoded source",
    )
    lv.add_argument(
        "--decode-health-file", help="decode-health export (decode_health.py)"
    )
    lv.add_argument(
        "--progress-file", help="messages decoded per sender, as nav_decode's"
    )
    lv.add_argument("--every", type=float, default=0.4, help="poll period, s")
    lv.add_argument("--report-every", type=float, default=60.0)
    lv.add_argument("--verbose", action="store_true", help="log every message")
    args = ap.parse_args()
    if args.cmd == "selftest":
        raise SystemExit(0 if selftest() else 1)
    if args.cmd == "find":
        find(args.file)
    if args.cmd == "capture":
        capture(args.file, args.g2)
    if args.cmd == "live":
        live(args)


if __name__ == "__main__":
    main()
