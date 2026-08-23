#!/usr/bin/env python3
"""MGEX differential code biases (CAS Bias-SINEX) -- the per-satellite half of A0b.

WHY THIS EXISTS. The broadcast group delay (`gnss_ephemeris.group_delay_s`) is what the
satellite says about its own inter-signal delay, and for two of our five signals it says
nothing at all: BeiDou B2a's TGD_B2ap lives in B-CNAV2 and RINEX 3 does not carry it. The
IGS/MGEX analysis centres measure the same physical quantity from a global network and
publish it daily, covering every signal we track.

⚠️⚠️ THIS PRODUCT CANNOT SUPPLY A CONSTELLATION-COMMON TERM, AND THAT IS BY CONSTRUCTION.
Its own header says so: "A zero-mean constellation condition is applied to separate
satellite and receiver biases on a daily basis." Verified on the 2026-08-230 file -- the
mean of the derived B3I-minus-B2a differential over 30 BeiDou satellites is EXACTLY 0.000
ns. So this fixes the PER-SATELLITE spread and can never explain a constellation offset.
Measured for our tracked BeiDou birds: -0.006..+0.121 chips, i.e. roughly a tenth of the
+0.74..+1.3 chip common trim BeiDou actually shows. Do not expect it to move that.

THE DATUM, per constellation, is whatever combination the BROADCAST clock is fitted to --
that is the thing our model already assumes, so the correction is (datum - our signal):
    GPS     LNAV  -> L1P/L2P ionosphere-free   (C1W/C2W)
    Galileo F/NAV -> E1/E5a ionosphere-free    (C1C/C5Q);  I/NAV -> E1/E5b (C1C/C7Q)
    BeiDou  D1/D2 -> B3I                       (C6I)
Sign convention: the returned value is ADDED to the broadcast satellite clock, matching
`group_delay_s`, so a signal that leaves the satellite LATE than the datum gets a negative
correction and the model's predicted code phase moves later with it.

Latency: the rapid product runs ~5 days behind, and code biases move on week-to-month
timescales, so a stale file is fine -- but `max_age_days` refuses a genuinely ancient one
rather than silently applying last month's numbers.
"""
import gzip
import math
import os
import time
import urllib.request
from datetime import datetime, timezone, timedelta

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "kotekan_gps")

F_L1 = F_E1 = 1575.42e6
F_L2 = 1227.60e6
F_L5 = F_E5A = 1176.45e6
F_E5B = 1207.14e6
GAMMA_L1L2 = (F_L1 / F_L2) ** 2
GAMMA_E1E5A = (F_E1 / F_E5A) ** 2
GAMMA_E1E5B = (F_E1 / F_E5B) ** 2


def _token():
    t = os.environ.get("EARTHDATA_TOKEN")
    if t and t.strip():
        return t.strip()
    try:
        with open(os.path.join(CACHE, ".earthdata_token")) as f:
            return f.read().strip() or None
    except Exception:
        return None


def fetch_dcb(when=None, cache_dir=CACHE, max_back_days=14):
    """Newest available CAS daily bias file, cached locally. None if unreachable.

    Walks BACKWARDS from `when`: the rapid product lands ~5 days late, so today's name is
    always a 404 and that is normal, not an error worth logging loudly.
    """
    tok = _token()
    if not tok:
        return None
    when = when or datetime.now(timezone.utc)
    os.makedirs(cache_dir, exist_ok=True)
    for back in range(1, max_back_days + 1):
        d = when - timedelta(days=back)
        doy = d.timetuple().tm_yday
        # Two generations of the CAS naming; try the current one first.
        for name in ("CAS0OPSRAP_%04d%03d0000_01D_01D_DCB.BIA.gz" % (d.year, doy),
                     "CAS0MGXRAP_%04d%03d0000_01D_01D_DCB.BSX.gz" % (d.year, doy)):
            local = os.path.join(cache_dir, name)
            if os.path.exists(local) and os.path.getsize(local) > 1000:
                return local
            url = ("https://cddis.nasa.gov/archive/gnss/products/bias/%04d/%s"
                   % (d.year, name))
            try:
                req = urllib.request.Request(url, headers={"Authorization": "Bearer " + tok})
                with urllib.request.urlopen(req, timeout=45) as r:
                    raw = r.read()
                if len(raw) < 1000:
                    continue
                tmp = local + ".tmp"
                with open(tmp, "wb") as f:
                    f.write(raw)
                os.replace(tmp, local)          # atomic: shared cache, see _atomic_write_bytes
                return local
            except Exception:
                continue
    return None


def parse_dcb(path):
    """Bias-SINEX -> {(sysc, prn): {(obs1, obs2): seconds}} for SATELLITE rows only.

    Station rows carry a 4-char site in the STATION field and are skipped: they are the
    receiver half of the same separation and mean nothing for us.
    """
    out = {}
    if not path:
        return out
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", errors="replace") as f:
        for ln in f:
            if not ln.startswith(" DSB "):
                continue
            fld = ln.split()
            # ' DSB  C217 C01           C2I  C6I  <start> <end> ns  <val> <sd>'
            # A station row has a site code where OBS1 sits, so OBS1 would start with a digit
            # or the field count shifts; require the two obs codes to look like obs codes.
            if len(fld) < 10:
                continue
            prn, o1, o2 = fld[2], fld[3], fld[4]
            if len(prn) != 3 or prn[0] not in "GECR" or not prn[1:].isdigit():
                continue
            if len(o1) != 3 or len(o2) != 3 or not o1[0] == "C" or not o2[0] == "C":
                continue
            try:
                val = float(fld[-2])
            except ValueError:
                continue
            if fld[-3] != "ns":
                continue
            out.setdefault((prn[0], int(prn[1:])), {})[(o1, o2)] = val * 1e-9
    return out


def _dsb(tab, a, b):
    """DSB(a,b) = bias_a - bias_b, resolved through the stored direction or its reverse."""
    if (a, b) in tab:
        return tab[(a, b)]
    if (b, a) in tab:
        return -tab[(b, a)]
    return None


def signal_bias_s(dcb, sysc, prn, signal, gal_inav=False):
    """Seconds to ADD to the broadcast clock for `signal`, from measured DCBs; None if the
    needed pairs are absent (caller falls back to the broadcast TGD/BGD).

    Every case is (datum - our signal), with the datum being the combination the broadcast
    clock is fitted to. The ionosphere-free algebra used twice below:
        b_X - b_IF(1,2) = -DSB(1,X) - DSB(1,2)/(gamma_12 - 1)
    """
    tab = (dcb or {}).get((sysc, prn))
    if not tab:
        return None
    sig = str(signal or "").lower()
    if sysc == "G" and "l5" in sig:
        # datum = L1P/L2P IF (C1W/C2W). Reach C5X via C1C, the pivot CAS publishes.
        d_1c_5x = _dsb(tab, "C1C", "C5X") or _dsb(tab, "C1C", "C5Q")
        d_1c_1w = _dsb(tab, "C1C", "C1W")
        d_1w_2w = _dsb(tab, "C1W", "C2W")
        if d_1c_5x is None or d_1c_1w is None or d_1w_2w is None:
            return None
        d_1w_5 = d_1c_5x - d_1c_1w                      # b_C1W - b_C5X
        return d_1w_5 + d_1w_2w / (GAMMA_L1L2 - 1.0)
    if sysc == "E":
        # The datum depends on the RECORD TYPE, which the caller knows and we do not:
        # F/NAV clocks are E1/E5a iono-free, I/NAV clocks are E1/E5b. With both DSBs in
        # hand every combination is reachable, so unlike the broadcast path there is no
        # approximation here -- and a record-type flip stays continuous.
        #   b_IF(1,x) = (g_x*b_1 - b_x)/(g_x - 1),  d1x = b_1 - b_x
        #   same-band :  datum - b_x  =  g_x * d1x / (g_x - 1)
        #   cross-band:  datum - b_y  =  d1x / (g_x - 1) + d1y
        d15 = _dsb(tab, "C1C", "C5Q")
        if d15 is None:
            d15 = _dsb(tab, "C1X", "C5X")
        d17 = _dsb(tab, "C1C", "C7Q")
        if d17 is None:
            d17 = _dsb(tab, "C1X", "C7X")
        if "e5a" in sig:
            if not gal_inav:
                return None if d15 is None else GAMMA_E1E5A * d15 / (GAMMA_E1E5A - 1.0)
            if d15 is None or d17 is None:
                return None
            return d17 / (GAMMA_E1E5B - 1.0) + d15
        if "e5b" in sig:
            if gal_inav:
                return None if d17 is None else GAMMA_E1E5B * d17 / (GAMMA_E1E5B - 1.0)
            if d15 is None or d17 is None:
                return None
            return d15 / (GAMMA_E1E5A - 1.0) + d17
        return None
    if sysc == "C":
        # datum = B3I (C6I). B2a is C5P (BDS-3 pilot) / C5X; B2b is C7D / C7Z.
        if "b2a" in sig:
            for one, five in (("C1P", "C5P"), ("C1X", "C5X"), ("C1D", "C5D")):
                d15 = _dsb(tab, one, five)
                d16 = _dsb(tab, one, "C6I")
                if d15 is not None and d16 is not None:
                    return d15 - d16                    # = b_C6I - b_C5P
            return None
        if "b2b" in sig:
            for one, seven in (("C1P", "C7D"), ("C1X", "C7D"), ("C1X", "C7Z")):
                d17 = _dsb(tab, one, seven)
                d16 = _dsb(tab, one, "C6I")
                if d17 is not None and d16 is not None:
                    return d17 - d16
            return None
        return None
    return None


if __name__ == "__main__":
    import statistics as st
    p = fetch_dcb()
    print("DCB:", p)
    tab = parse_dcb(p)
    print("satellites: %d (%s)" % (len(tab), "".join(sorted({k[0] for k in tab}))))
    for sysc, sig in (("G", "gps_l5"), ("E", "gal_e5a"), ("E", "gal_e5b"),
                      ("C", "bds_b2a"), ("C", "bds_b2b")):
        v = [(k[1], signal_bias_s(tab, sysc, k[1], sig)) for k in sorted(tab) if k[0] == sysc]
        v = [(p_, x) for p_, x in v if x is not None]
        if not v:
            print("%-8s none" % sig)
            continue
        ns = [x * 1e9 for _, x in v]
        print("%-8s n=%2d  median %+7.3f ns (%+.3f chips)  mean %+7.3f  range %+7.2f..%+7.2f"
              % (sig, len(ns), st.median(ns), st.median(ns) * 1e-9 * 1.023e7,
                 st.mean(ns), min(ns), max(ns)))
