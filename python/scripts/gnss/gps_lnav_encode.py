#!/usr/bin/env python3
"""Broadcast ephemeris -> GPS LNAV subframe 1/2/3 data words (IS-GPS-200).

WHY: a weak satellite cannot decode its own nav message, so the peel has no bit signs for it
and `nobits` keeps it unpeeled -- which, since the 2026-07-25 floor fix, is the ONLY thing
holding the peel off most of the GPS constellation. But subframes 1-3 ARE the ephemeris, and
we already hold the ephemeris from BRDC for every satellite in the sky, decoded or not. So the
bits can be CONSTRUCTED instead of received.

Measured first, built second (diag/navbit_reuse.py, 13 min of live bits, 6 PRNs):
  * sf1-3 are STATIC per satellite -- over 12.5 min only the HOW word changes (the TOW
    counter), so one construction serves ~2 h.
  * The bits RINEX does NOT carry -- sf1 words 4,5,6 (RESERVED) and the TLM word -- are
    100% BIT-IDENTICAL ACROSS SATELLITES, and are real content, not zeros. So they are
    borrowed from any strong satellite rather than invented. This is what makes the whole
    approach work; it was the risk that looked fatal on paper.
  * Coverage: 900 of 1500 bits per 30 s frame = 60%, as a CONTIGUOUS 18 s block.
sf4/5 are NOT constructible this way (cross-satellite identity measured at only 47-91%) --
see docs/buglist.md, parked low priority.

Word indices here are 0-based over the 10 words of a subframe: 0 = TLM, 1 = HOW, 2..9 = the
eight data words (ICD "word 3".."word 10"). This module builds 2..9; TLM is borrowed and HOW
is a pure function of TOW (make_how).

Self-test: python3 gps_lnav_encode.py --check <navobs.jsonl>
    encodes from BRDC and diffs against subframes DECODED FROM AIR on strong satellites.
    The ephemeris words must match EXACTLY; anything else is a packing bug, not a rounding one.
"""
import argparse
import math
import sys

import numpy as np

SEMI = 1.0 / math.pi  # radians -> semicircles

# URA: RINEX stores the user range accuracy in METRES, the subframe carries a 4-bit INDEX.
# Nominal values, IS-GPS-200 20.3.3.3.1.3 (index 15 = "no accuracy prediction").
URA_M = [2.0, 2.8, 4.0, 5.7, 8.0, 11.3, 16.0, 32.0,
         64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0]


def ura_index(metres):
    """Metres -> URA index: the smallest index whose nominal bound covers it (measured on air
    2026-07-25: RINEX 2.0 m <-> index 0 on every satellite in view)."""
    for i, m in enumerate(URA_M):
        if metres <= m * 1.001:
            return i
    return 15


def eph_ambiguous(recs, sel, tol_s=120.0):
    """Is `sel` indistinguishable from a RIVAL record we could equally have chosen?

    Measured live 2026-07-26. Our BRDC is a MERGE of the daily file and an hourly one, and the
    hourly can carry a second record for the same epoch: G15 had toe 15:59:44 (IODE 41) AND
    16:00:00 (IODE 102), 16 s apart, both genuine (different af0, different delta-n).
    select_eph takes the newest and got IODE 102; the satellite was broadcasting IODE 41,
    decoded straight off the air from sf2 word 3. Hold-one-out on that satellite read 65.56%
    while every IODE-matched satellite read 93.7-100.00% -- a perfect split, and exactly the
    "confidently wrong bits" failure select_eph's docstring warns about.

    A 16 s toe difference is NOTHING for geometry (same orbit, range/Doppler identical to
    nanoseconds) -- which is why this never showed up in tracking. It is everything for BIT
    CONSTRUCTION, where iode/toe/af0 are literally encoded into the words.

    We cannot tell which rival is on air without decoding, and a satellite that needs
    constructed bits is by definition one we cannot decode. So the honest answer is to refuse:
    ambiguity -> no bits -> the tracker reads `nobits` and does not subtract. Fails SAFE, the
    same discipline as the combiner's stale-anchor gate.
    """
    si = sel.get("iode")
    for r in recs:
        if r is sel:
            continue
        if abs(r["toe_gpst"] - sel["toe_gpst"]) <= tol_s and r.get("iode") != si:
            return True
    return False


def select_eph(recs, t_gps, fit_h=4.0):
    """The ephemeris record the satellite is BROADCASTING at GPS time `t_gps`.

    ⚠️ THIS IS NOT "the newest record", and getting it wrong silently produces confidently
    WRONG bits -- the known-mask cannot catch it, because the bits are known, just not the ones
    on the air. Measured 2026-07-25: the same verification run scored 100.0000% and then
    74.1038% with no code change at all, purely because BRDC refreshed underneath it.

    GPS places toe at the CENTRE of the fit interval, so the broadcast record's toe is up to
    fit/2 in the FUTURE. Hence: newest record whose window has already opened. Measured against
    air on 6 satellites, 113,820 bits:
        newest overall          74.1038%   (what recs[-1] does -- wrong)
        min |t - toe|           69.4544%   (nearest toe -- also wrong)
        newest, toe-1h <= t     69.4544%
        newest, toe-2h <= t    100.0000%   <-- this
    """
    lead = 0.5 * fit_h * 3600.0
    started = [r for r in recs if r["toe_gpst"] - lead <= t_gps]
    return max(started, key=lambda r: r["toe_gpst"]) if started else recs[0]


def _u(val, nbits):
    """Unsigned integer -> nbits MSB-first."""
    v = int(round(val)) & ((1 << nbits) - 1)
    return [(v >> (nbits - 1 - i)) & 1 for i in range(nbits)]


def _s(val, nbits):
    """Two's-complement signed integer -> nbits MSB-first."""
    return _u(int(round(val)) & ((1 << nbits) - 1), nbits)


def _scaled(val, power, nbits, signed=True):
    """Physical value -> its LNAV integer at 2**power LSB."""
    q = val / (2.0 ** power)
    return _s(q, nbits) if signed else _u(q, nbits)


def _w(*chunks):
    """Concatenate bit lists into one 24-bit data word."""
    out = []
    for c in chunks:
        out.extend(c)
    assert len(out) == 24, f"word is {len(out)} bits, not 24"
    return np.array(out, dtype=np.int8)


def encode_sf1(e, wn=None):
    """Subframe 1 data words 3..10 (indices 2..9). Reserved words are left ZERO -- the caller
    must overwrite indices 3,4,5 and the top 16 bits of 6 with borrowed bits (see module note);
    they are not derivable from RINEX and are not invented here."""
    week = int(e["week"]) if wn is None else int(wn)
    iodc = int(round(e["iodc"]))
    toc_sow = float(e["toc_gpst"]) - float(e["week"]) * 604800.0
    return [
        _w(_u(week % 1024, 10), _u(e.get("l2_codes", 0), 2), _u(ura_index(e.get("accuracy", 0.0)), 4),
           _u(e.get("health", 0), 6), _u(iodc >> 8, 2)),                      # idx 2
        _w(_u(e.get("l2p_flag", 0), 1), _u(0, 23)),                           # idx 3 reserved
        _w(_u(0, 24)),                                                        # idx 4 reserved
        _w(_u(0, 24)),                                                        # idx 5 reserved
        _w(_u(0, 16), _scaled(e.get("tgd", 0.0), -31, 8)),                    # idx 6 resv+TGD
        _w(_u(iodc & 0xFF, 8), _scaled(toc_sow, 4, 16, signed=False)),        # idx 7
        _w(_scaled(e["af2"], -55, 8), _scaled(e["af1"], -43, 16)),            # idx 8
        _w(_scaled(e["af0"], -31, 22), _u(0, 2)),                             # idx 9 (+solve)
    ]


def encode_sf2(e):
    """Subframe 2 data words 3..10 (indices 2..9). Fully determined by the ephemeris."""
    m0 = _scaled(e["m0"] * SEMI, -31, 32)
    ecc = _scaled(e["ecc"], -33, 32, signed=False)
    sqa = _scaled(e["sqrta"], -19, 32, signed=False)
    return [
        _w(_u(e["iode"], 8), _scaled(e["crs"], -5, 16)),                      # idx 2
        _w(_scaled(e["dn"] * SEMI, -43, 16), m0[:8]),                         # idx 3
        _w(m0[8:]),                                                           # idx 4
        _w(_scaled(e["cuc"], -29, 16), ecc[:8]),                              # idx 5
        _w(ecc[8:]),                                                          # idx 6
        _w(_scaled(e["cus"], -29, 16), sqa[:8]),                              # idx 7
        _w(sqa[8:]),                                                          # idx 8
        _w(_scaled(e["toe_sow"], 4, 16, signed=False), _u(0, 1),
           _u(0, 5), _u(0, 2)),                                              # idx 9 fit/AODO
    ]


def encode_sf3(e):
    """Subframe 3 data words 3..10 (indices 2..9). Fully determined by the ephemeris."""
    om0 = _scaled(e["omega0"] * SEMI, -31, 32)
    i0 = _scaled(e["i0"] * SEMI, -31, 32)
    omg = _scaled(e["omega"] * SEMI, -31, 32)
    return [
        _w(_scaled(e["cic"], -29, 16), om0[:8]),                              # idx 2
        _w(om0[8:]),                                                          # idx 3
        _w(_scaled(e["cis"], -29, 16), i0[:8]),                               # idx 4
        _w(i0[8:]),                                                           # idx 5
        _w(_scaled(e["crc"], -5, 16), omg[:8]),                               # idx 6
        _w(omg[8:]),                                                          # idx 7
        _w(_scaled(e["omegadot"] * SEMI, -43, 24)),                           # idx 8
        _w(_u(e["iode"], 8), _scaled(e["idot"] * SEMI, -43, 14), _u(0, 2)),   # idx 9
    ]


def make_how(tow_next, sfid, alert=0, as_flag=1):
    """HOW (index 1): 17-bit TOW count of the NEXT subframe, alert, A-S, subframe id, 2 solve
    bits. The solve bits are left 0 here; solve_tail fixes them where the ICD requires
    D29=D30=0 (words 2 and 10).

    A-S DEFAULTS TO 1: anti-spoofing is on across the constellation, and a live HOW reads
    `0xbc3227` = TOW 96356 | alert 0 | A-S 1 | sfid 1. Defaulting it to 0 cost ~7 bits per
    subframe (the flag plus the parity it feeds) and held the end-to-end match at ~97%."""
    return _w(_u(tow_next, 17), _u(alert, 1), _u(as_flag, 1), _u(sfid, 3), _u(0, 2))


def solve_tail(data24, D29s, D30s):
    """Set the last two data bits so the encoded word's parity bits 29,30 are ZERO, as the ICD
    requires for words 2 and 10. Brute force over the 4 combinations -- it is exact and this is
    not a hot path."""
    from gps_nav_decode import parity_encode
    for a in (0, 1):
        for b in (0, 1):
            d = np.array(data24, dtype=np.int8).copy()
            d[22], d[23] = a, b
            w = parity_encode(d, D29s, D30s)
            if int(w[28]) == 0 and int(w[29]) == 0:
                return d
    return np.array(data24, dtype=np.int8)


GPS_EPOCH_UNIX = 315964782.0  # unix -> GPS seconds (includes the 18 s leap offset); the obs
                              # rows carry both t and t_gps and their difference IS this.


def assemble_subframe(tow_next, sfid, words29, tlm=None, known29=None):
    """One subframe -> (bits300, known300).

    `words29` is the eight data words (indices 2..9); `tlm` the borrowed TLM word or None.
    `known29`/the tlm-None case mark words we could NOT construct; their bits come back
    known=0 so the consumer treats them as UNKNOWN rather than subtracting a guess.

    Parity is chained word to word, and a word's parity bits are only known when every data
    bit of that word AND the incoming chain bits are known. Each subframe starts from
    D29s=D30s=0: the solve bits at the end of words 2 and 10 force exactly that, which is why
    a subframe can be built standalone even when the preceding one is unknown.
    """
    from gps_nav_decode import parity_encode
    words = [tlm if tlm is not None else np.zeros(24, np.int8),
             make_how(tow_next, sfid)] + list(words29)
    kn = [tlm is not None, True] + list(known29 if known29 is not None else [True] * 8)
    bits, known = [], []
    D29s = D30s = 0
    chain_ok = True
    for i, (d, k) in enumerate(zip(words, kn)):
        if k and chain_ok and i in (1, 9):
            d = solve_tail(d, D29s, D30s)      # ICD: words 2 and 10 end with D29=D30=0
        w = parity_encode(d, D29s, D30s)
        bits.extend(int(x) for x in w)
        # data bits are known iff the word is; parity bits also need the incoming chain.
        known.extend([1 if k else 0] * 24 + [1 if (k and chain_ok) else 0] * 6)
        D29s, D30s = int(w[28]), int(w[29])
        chain_ok = chain_ok and k
        if i in (1, 9) and k and chain_ok:
            D29s = D30s = 0                    # solved to zero by construction
    return np.array(bits, np.int8), np.array(known, np.int8)


def constructed_bits(e, t_utc0, dur_s, borrow=None):
    """LNAV bits for [t_utc0, t_utc0+dur_s) built from ephemeris `e`.

    Returns (utc0_aligned, bit_s, bits) with bits in the +-1/0 convention the tracker's
    `nav_bits` seed uses: +-1 = known, 0 = unknown. sf4/5 come back all-unknown; so do the
    sf1 reserved words unless `borrow` supplies them.

    GLOBAL POLARITY DOES NOT MATTER: the peel de-bits its gain EMA with these same predicted
    signs and re-applies them at subtraction time, so a uniformly inverted stream peels
    identically. Only SELF-CONSISTENCY matters, which construction guarantees.

    `borrow` = {"tlm": word24, "sf1_resv": [w3, w4, w5, w6hi16]} from any decoded satellite --
    those bits are constellation-common (measured 100%), not satellite-specific.
    """
    bit_s = 0.02
    t_gps0 = t_utc0 - GPS_EPOCH_UNIX
    sf_i0 = int(math.floor(t_gps0 / 6.0))          # index of the subframe containing t_utc0
    n_sf = int(math.ceil((t_gps0 + dur_s) / 6.0)) - sf_i0
    utc0 = sf_i0 * 6.0 + GPS_EPOCH_UNIX
    out = []
    for k in range(max(n_sf, 1)):
        sf_i = sf_i0 + k
        sfid = (sf_i % 5) + 1
        tow_next = (sf_i + 1) % 100800          # TOW count of the NEXT subframe, 17 bits
        if sfid in ENCODERS and e is not None:
            words = ENCODERS[sfid](e)
            known = [True] * 8
            tlm = None
            if sfid == 1:
                if borrow and borrow.get("sf1_resv") is not None:
                    r = borrow["sf1_resv"]
                    words[1], words[2], words[3] = r[0], r[1], r[2]      # idx 3,4,5
                    words[4] = _w(list(r[3][:16]), list(words[4][16:]))  # idx 6: resv | TGD
                else:
                    known[1] = known[2] = known[3] = known[4] = False
            if sfid == 2:
                known[7] = False                # idx 9 carries fit + AODO, neither in RINEX
            if borrow:
                tlm = borrow.get("tlm")
            b, kn = assemble_subframe(tow_next, sfid, words, tlm=tlm, known29=known)
        else:
            b, kn = np.zeros(300, np.int8), np.zeros(300, np.int8)   # sf4/5: unknown
        # transmitted 0/1 -> +-1, with unknown as 0
        out.extend(int(np.where(kn[i], 1 - 2 * int(b[i]), 0)) for i in range(300))
    return utc0, bit_s, out


ENCODERS = {1: encode_sf1, 2: encode_sf2, 3: encode_sf3}
# Words that are PURE ephemeris and must match the air bit-for-bit. Excluded deliberately:
#   sf1 idx 3,4,5      RESERVED, and idx 6 top 16 bits -- borrowed (measured 100% cross-sat)
#   sf2 idx 9          toe is ours, but fit/AODO are not in RINEX (AODO is transmit-only)
# Word idx 9 of every subframe ends in 2 SOLVE bits (chosen so parity D29=D30=0), which depend
# on the parity chain and so are set at frame-assembly time, not here -- compare 22 bits only.
EPH_WORDS = {1: [2, 7, 8, 9], 2: [2, 3, 4, 5, 6, 7, 8], 3: [2, 3, 4, 5, 6, 7, 8, 9]}
PARTIAL = {(1, 9): 22, (3, 9): 22}   # word -> compare only the first N bits
# Words carrying non-RINEX content that must be BORROWED from a strong satellite, with the
# measured cross-satellite identity that justifies borrowing each one.
BORROW = {(1, 0): "TLM 100%", (1, 3): "reserved 100%", (1, 4): "reserved 100%",
          (1, 5): "reserved 100%", (1, 6): "reserved(hi16)+TGD 9%"}


def _check(navobs_path):
    """Encode from BRDC, diff against subframes DECODED FROM AIR."""
    sys.path.insert(0, "/home/lwlab/airspy_gps/kotekan/python/scripts/gnss/diag")
    import gnss_ephemeris as ge
    from navbit_reuse import stitch, subframes

    eph = ge.parse_rinex_nav(ge.fetch_brdc())
    per = {p: v for p, v in ((p, subframes(m)) for p, m in stitch(navobs_path).items()) if v}
    print(f"decoded {len(per)} PRNs from air; BRDC has "
          f"{len([k for k in eph if k[0] == 'G'])} GPS sats\n")
    print(f"{'PRN':>4s} {'sf':>3s} {'word':>5s} {'air':>8s} {'brdc':>8s}  {'bits':>4s}")
    tot = {1: [0, 0], 2: [0, 0], 3: [0, 0]}
    for prn, sfs in sorted(per.items()):
        recs = eph.get(("G", prn))
        if not recs:
            continue
        seen = set()
        for tow, sfid, data in sfs:
            if sfid not in ENCODERS or sfid in seen:
                continue
            seen.add(sfid)
            e = select_eph(recs, (tow - 1) * 6.0 + int(recs[-1]["week"]) * 604800.0)
            enc = ENCODERS[sfid](e)
            for wi in EPH_WORDS[sfid]:
                n = PARTIAL.get((sfid, wi), 24)
                air = data[wi][:n]
                brd = enc[wi - 2][:n]
                nb = int((air ^ brd).sum())
                tot[sfid][1] += 1
                if nb == 0:
                    tot[sfid][0] += 1
                else:
                    h = lambda w: hex(int("".join(str(int(b)) for b in w), 2))[2:].zfill(6)
                    print(f"G{prn:02d} {sfid:3d} {wi:5d} {h(air):>8s} {h(brd):>8s}  {nb:4d}")
    print()
    for sfid in (1, 2, 3):
        ok, n = tot[sfid]
        print(f"  sf{sfid}: {ok}/{n} ephemeris words match exactly"
              + ("  ✅" if n and ok == n else "  ❌"))


def _verify(navobs_path):
    """END-TO-END: constructed stream vs the AIR stream, split by whether the AIR subframe
    passed parity.

    This split is the whole point. On parity-CLEAN subframes the construction must be perfect
    (measured 2026-07-25: 113,820 of 113,820 bits, 6 satellites, zero errors). On parity-FAILED
    subframes it disagrees -- because the AIR is wrong there, not the construction. G24 had 28
    of 61 subframes fail parity and the construction supplies correct bits for all of them,
    which is precisely the point of building this: on a weak satellite, constructed bits beat
    received ones.

    Global polarity is allowed to flip (irrelevant to the peel -- see constructed_bits).
    Alignment here comes from the air's own frame_sync; a weak satellite gets it from the
    tracking solution instead (code phase + BRDC range), NOT from decoding.
    """
    sys.path.insert(0, "/home/lwlab/airspy_gps/kotekan/python/scripts/gnss/diag")
    import gnss_ephemeris as ge
    from gps_nav_decode import frame_sync, decode_subframe
    from navbit_reuse import stitch, subframes

    eph = ge.parse_rinex_nav(ge.fetch_brdc())
    merged = stitch(navobs_path)
    borrow = None
    for prn, m in sorted(merged.items()):
        for _tow, sfid, data in subframes(m):
            if sfid == 1:
                borrow = {"tlm": data[0],
                          "sf1_resv": [data[3], data[4], data[5], data[6][:16]]}
                break
        if borrow:
            print(f"borrowing TLM + sf1 reserved words from G{prn:02d}\n")
            break
    if borrow is None:
        raise SystemExit("no satellite decoded an sf1 -- nothing to borrow from")

    print(f"{'PRN':>4} | {'AIR PARITY-CLEAN':>23} | {'AIR PARITY-FAILED':>22}")
    print(f"{'':4} | {'n':>4} {'bits':>8} {'match':>9} | {'n':>4} {'bits':>7} {'match':>8}")
    grand = [0, 0]
    for prn, m in sorted(merged.items()):
        recs = eph.get(("G", prn))
        if not recs:
            continue
        slots = sorted(m)
        runs, cur = [], [slots[0]]
        for a, b in zip(slots, slots[1:]):
            if b == a + 1:
                cur.append(b)
            else:
                runs.append(cur); cur = [b]
        runs.append(cur)
        run = max(runs, key=len)
        if len(run) < 620:
            continue
        ba = np.array([(1 - m[s]) // 2 for s in run], dtype=np.int8)
        acc = {True: [0, 0, 0], False: [0, 0, 0]}
        for i, tow, sfid in frame_sync(ba):
            if sfid not in ENCODERS or i + 300 > len(ba):
                continue
            # the record actually on the air at this subframe's transmit time
            e = select_eph(recs, (tow - 1) * 6.0 + int(recs[-1]["week"]) * 604800.0)
            D29s = int(ba[i - 2]) if i >= 2 else 0
            D30s = int(ba[i - 1]) if i >= 1 else 0
            _, ok, _, _, _, _ = decode_subframe(ba, i, D29s, D30s)
            words = ENCODERS[sfid](e)
            known = [True] * 8
            if sfid == 1:
                r = borrow["sf1_resv"]
                words[1], words[2], words[3] = r[0], r[1], r[2]
                words[4] = _w(list(r[3][:16]), list(words[4][16:]))
            if sfid == 2:
                known[7] = False        # fit + AODO: not in RINEX
            b, kn = assemble_subframe(tow, sfid, words, tlm=borrow["tlm"], known29=known)
            air = ba[i:i + 300]
            idx = [j for j in range(300) if kn[j]]
            ag = sum(1 for j in idx if int(air[j]) == int(b[j]))
            d = acc[bool(ok)]
            d[0] += 1; d[1] += len(idx); d[2] += max(ag, len(idx) - ag)
        if not (acc[True][1] or acc[False][1]):
            continue
        grand[0] += acc[True][1]; grand[1] += acc[True][2]
        f = lambda d: f"{d[0]:4d} {d[1]:7d} {100 * d[2] / max(d[1], 1):8.3f}%"
        print(f"G{prn:02d} | {f(acc[True])} | {f(acc[False])}")
    print(f"\nPARITY-CLEAN TOTAL: {grand[1]}/{grand[0]} = "
          f"{100 * grand[1] / max(grand[0], 1):.4f}%"
          + ("  \u2705 exact" if grand[0] and grand[1] == grand[0] else "  \u274c"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", metavar="NAVOBS_JSONL",
                    help="diff BRDC-encoded words against air-decoded subframes")
    ap.add_argument("--verify", metavar="NAVOBS_JSONL",
                    help="END-TO-END: constructed bit stream vs the AIR bit stream")
    a = ap.parse_args()
    if a.check:
        _check(a.check)
    elif a.verify:
        _verify(a.verify)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
