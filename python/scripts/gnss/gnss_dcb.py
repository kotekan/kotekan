#!/usr/bin/env python3
"""Fetch + parse satellite Differential Code Biases (DCBs) for the TEC pipeline.

    python3 gnss_dcb.py [--date YYYY-DDD] [--out /tmp/dcb.json] [--pairs G:C1C-C5Q,G:C1C-C2L]

WHY: even a perfect dual-band code measurement contains each satellite's inter-band transmit
hardware delay (a few ns = several TECU, different per SV) -- constant, and therefore
indistinguishable from TEC until subtracted. These are published daily: the CAS (Chinese Academy
of Sciences) multi-GNSS rapid DCB product in Bias-SINEX, mirrored ANONYMOUSLY at IGS IGN
(igs.ign.fr) -- no Earthdata login, unlike CDDIS. The RECEIVER DCB is deliberately NOT here: ours
spans three separate dongles + the constant inter-band anchor offset, and is estimated jointly
with TEC from the sky (see gnss_tec.py).

Output json: {"date": .., "source": .., "dcb": {"G01": {"C1C-C5Q": {"ns": .., "tecu_l1l5": ..},
...}}}. TECU conversion for a pair (f_a, f_b): TEC = DCB_seconds * c / (K*(1/f_a^2 - 1/f_b^2)).
Our band pairs: GPS C1C-C5Q (L1 C/A vs L5Q), C1C-C2L (vs L2C-CM); GAL C1C-C5Q (E1C vs E5a-Q);
BDS C1P-C5P (B1Cp vs B2ap). All present in the CAS product.
"""
import argparse
import datetime
import gzip
import io
import json
import re
import urllib.request

C = 299792458.0
K = 40.308e16
FREQ = {("G", "1"): 1575.42e6, ("G", "2"): 1227.60e6, ("G", "5"): 1176.45e6,
        ("E", "1"): 1575.42e6, ("E", "5"): 1176.45e6,
        ("C", "1"): 1575.42e6, ("C", "5"): 1176.45e6}
DEFAULT_PAIRS = ["G:C1C-C5Q", "G:C1C-C2L", "E:C1C-C5Q", "C:C1P-C5P"]


def fetch(date):
    """CAS rapid BSX for the date (falls back day by day: rapid lags ~1-2 days)."""
    for back in range(6):
        d = date - datetime.timedelta(days=back)
        yyyy, ddd = d.strftime("%Y"), d.strftime("%j")
        name = f"CAS0MGXRAP_{yyyy}{ddd}0000_01D_01D_DCB.BSX.gz"
        for base in (f"https://igs.ign.fr/pub/igs/products/mgex/dcb/{yyyy}/",
                     f"ftp://igs.ign.fr/pub/igs/products/mgex/dcb/{yyyy}/"):
            url = base + name
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    raw = r.read()
                print(f"fetched {name} ({len(raw)} bytes) from {base}")
                return gzip.decompress(raw).decode("ascii", "replace"), name
            except Exception as e:
                print(f"  {url}: {e}")
    raise SystemExit("no DCB product reachable (network? try --date further back)")


def parse(text, pairs):
    """DSB lines: ' DSB  G063 G01 ... C1C  C5Q  ...  <value ns>  <sigma>'. Keep sat rows only
    (PRN field set); stations have a 9-char site name there instead."""
    want = {(p.split(":")[0], *p.split(":")[1].split("-")) for p in pairs}
    out = {}
    for line in text.splitlines():
        if not line.startswith(" DSB"):
            continue
        f = line.split()
        # DSB <svn> <prn> [<station>] <obs1> <obs2> <t0> <t1> <unit> <value> <sigma>
        if len(f) < 10 or not re.fullmatch(r"[GECR]\d\d", f[2]):
            continue
        prn, o1, o2 = f[2], f[3], f[4]
        if re.fullmatch(r"[A-Z0-9]{9}", o1):  # station row: site name sits where obs1 would be
            continue
        if (prn[0], o1, o2) not in want:
            continue
        try:
            ns = float(f[8])
        except ValueError:
            continue
        fa = FREQ.get((prn[0], o1[1]))
        fb = FREQ.get((prn[0], o2[1]))
        tecu = None
        if fa and fb:
            tecu = ns * 1e-9 * C / (K * (1.0 / fa**2 - 1.0 / fb**2))
        out.setdefault(prn, {})[f"{o1}-{o2}"] = {"ns": ns,
                                                 "tecu": round(tecu, 3) if tecu else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYY-DDD (default: today UTC)")
    ap.add_argument("--pairs", default=",".join(DEFAULT_PAIRS))
    ap.add_argument("--out", default="/tmp/gnss_dcb.json")
    args = ap.parse_args()
    date = (datetime.datetime.strptime(args.date, "%Y-%j").date() if args.date
            else datetime.datetime.now(datetime.timezone.utc).date())
    text, name = fetch(date)
    dcb = parse(text, args.pairs.split(","))
    json.dump({"source": name, "pairs": args.pairs, "dcb": dcb}, open(args.out, "w"), indent=1)
    n = sum(len(v) for v in dcb.values())
    print(f"wrote {args.out}: {len(dcb)} satellites, {n} pair entries")
    for prn in sorted(dcb)[:6]:
        print(" ", prn, dcb[prn])


if __name__ == "__main__":
    main()
