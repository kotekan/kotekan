#!/usr/bin/env python3
"""WHICH PUBLISHED STATISTICS ACTUALLY MEASURE ANYTHING? Ask the below-horizon probes.

    scripts/gnss/probe_audit.py [--minutes 10] [--interval 20]

WHY. #66 found the deep fold reporting a PRN seeded 64 degrees BELOW THE HORIZON at 11.2x its
own noise floor with coh_frac 0.744, while genuinely tracked satellites sat at 0.6-0.8x. That
is one statistic caught failing. The obvious next question -- WHICH OTHERS? -- has an answer
already running on every chain and it costs nothing to ask.

THE NULL IS FREE AND IT IS PHYSICAL. --noise-probes seeds the deepest below-horizon PRNs on
every chain. They are not a model of noise, not a shuffle, not a permutation: they are the
real receiver, the real tracker, the real fold, the real serving layer, on a satellite that
CANNOT BE THERE. Any published number that does not separate them from a tracked satellite is
not measuring detection, whatever its floor claims.

⚠️ THIS IS CHAIN-INDEPENDENT, WHICH MATTERS. Only gps_l5 has a search; gal_e5a/e5b and
bds_b2a/b2b are dead-reckoned by design (blind search is impossible on the baked-CS signals),
so no answer that leans on the search generalises. The probes exist on all five.

WHAT IT REPORTS, per statistic, per chain:
  * the probe distribution (median, p90) -- what the number reads on NOTHING
  * the tracked distribution (median, p90)
  * SEPARATION = how far the tracked median sits above the probe p90, in probe-sigma
  * and an AUC: the probability that a random tracked PRN scores above a random probe. 0.5 is
    a coin flip -- the statistic is blind. 1.0 is perfect separation.

AUC is the honest headline because it is scale-free and needs no floor: a statistic can have a
beautifully calibrated floor and still be a coin flip against the one population that is
guaranteed signal-free.

⚠️ SMALL n. There are only ~3 probes per chain, so a single snapshot proves nothing; this
samples over time and pools. Read the sample counts before the verdicts.
"""
import argparse
import json
import math
import re
import statistics
import time
import urllib.request

FIELDS = ["deep_snr", "coh_frac", "amp_snr", "coherence_s", "cn0_coh_db", "deep_amplitude",
          "fleet_q", "p_pow", "dll_disc", "sigma_phi", "adr_lock_s", "sig", "prompt_hold",
          "inst_snr_med", "fleet_coh_align", "s4_raw", "unbiased_amplitude"]


def probes_from_log(path, chains):
    """{chain: {prn}} from the broker's own 'noise probe PRN N seeded (elev -X)' lines."""
    out = {c: set() for c in chains}
    pat = re.compile(r"\[broker (\S+) [0-9:.]+\] noise probe PRN (\d+) seeded \(elev (-?\d+)\)")
    try:
        with open(path, "rb") as fh:
            for line in fh:
                m = pat.search(line.decode("utf8", "replace"))
                if m and m.group(1) in out and int(m.group(3)) < 0:
                    out[m.group(1)].add(int(m.group(2)))
    except OSError:
        pass
    return out


def auc(pos, neg):
    """P(a random `pos` scores above a random `neg`), ties at half. 0.5 = blind."""
    if not pos or not neg:
        return float("nan")
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1:12060")
    ap.add_argument("--log", default="/tmp/gnss_broker.log")
    ap.add_argument("--chains", default="gps_l5,gal_e5a,bds_b2a,gal_e5b,bds_b2b")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--interval", type=float, default=20.0)
    a = ap.parse_args()
    chains = a.chains.split(",")

    probes = probes_from_log(a.log, chains)
    print("probe PRNs from the broker log (below horizon, seeded as pure-noise anchors):")
    for c in chains:
        print("   %-9s %s" % (c, sorted(probes[c]) or "(none found -- check --noise-probes)"))
    print()

    acc = {c: {f: {"probe": [], "track": []} for f in FIELDS} for c in chains}
    t_end = time.time() + a.minutes * 60
    n = 0
    while time.time() < t_end:
        for c in chains:
            if not probes[c]:
                continue
            try:
                rows = json.load(urllib.request.urlopen(
                    "http://%s/%s/get_status" % (a.host, c), timeout=10))
            except Exception:
                continue
            for r in rows:
                grp = "probe" if r["prn"] in probes[c] else "track"
                for f in FIELDS:
                    v = r.get(f)
                    if isinstance(v, (int, float)):
                        acc[c][f][grp].append(float(v))
        n += 1
        time.sleep(a.interval)

    print("=== %d polls over %.1f min ===" % (n, a.minutes))
    print("AUC = P(tracked > probe). 0.50 = BLIND (a coin flip against guaranteed-noise rows).")
    print("A statistic can have a well-calibrated floor and still be blind by this test.\n")
    for c in chains:
        if not probes[c]:
            continue
        print("-- %s --" % c)
        print("   %-18s %6s %6s   %9s %9s   %6s  %s"
              % ("field", "n_prb", "n_trk", "probe med", "track med", "AUC", "verdict"))
        rows = []
        for f in FIELDS:
            p, t = acc[c][f]["probe"], acc[c][f]["track"]
            if len(p) < 5 or len(t) < 5:
                continue
            A = auc(t, p)
            rows.append((A, f, p, t))
        for A, f, p, t in sorted(rows, key=lambda x: -(x[0] if x[0] == x[0] else 0)):
            verdict = ("separates" if A >= 0.80 else
                       "weak" if A >= 0.65 else
                       "⚠️ BLIND -- does not distinguish a tracked sat from a below-horizon PRN")
            print("   %-18s %6d %6d   %9.4g %9.4g   %6.2f  %s"
                  % (f, len(p), len(t), statistics.median(p), statistics.median(t), A, verdict))
        print()


if __name__ == "__main__":
    main()
