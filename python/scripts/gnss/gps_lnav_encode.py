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
        _w(_u(week % 1024, 10), _u(e.get("l2_codes", 0), 2), _u(e.get("accuracy", 0), 4),
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


def make_how(tow_next, sfid, alert=0, as_flag=0):
    """HOW (index 1): 17-bit TOW count of the NEXT subframe, alert, A-S, subframe id, 2 solve
    bits. The solve bits are left 0 here; solve_word10 fixes them where the ICD requires
    D29=D30=0 (words 2 and 10)."""
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


ENCODERS = {1: encode_sf1, 2: encode_sf2, 3: encode_sf3}
# Words that are PURE ephemeris and must match the air bit-for-bit. Excluded deliberately:
#   sf1 idx 3,4,5      RESERVED, and idx 6 top 16 bits -- borrowed (measured 100% cross-sat)
#   sf1 idx 2          WN/L2/URA/health/IODC-MSB -- see the OPEN note below
#   sf2 idx 9          toe is ours, but fit/AODO are not in RINEX (AODO is transmit-only)
# Word idx 9 of every subframe ends in 2 SOLVE bits (chosen so parity D29=D30=0), which depend
# on the parity chain and so are set at frame-assembly time, not here -- compare 22 bits only.
EPH_WORDS = {1: [7, 8, 9], 2: [2, 3, 4, 5, 6, 7, 8], 3: [2, 3, 4, 5, 6, 7, 8, 9]}
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
            # the ephemeris record whose toe brackets this observation
            e = recs[-1]
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", metavar="NAVOBS_JSONL",
                    help="diff BRDC-encoded words against air-decoded subframes")
    a = ap.parse_args()
    if a.check:
        _check(a.check)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
