#!/usr/bin/env python3
"""Per-chain tracking-health mask from the observables writers' own flags.

Every obs record carries the tracker's verdict on itself: `prompt_lock`, `prompt_rayleigh`,
`fleet_present`. A chain whose records are ~95% Rayleigh for twelve hours is not tracking,
whatever level its despread happens to report -- and the beam cube, which carries none of
these flags, will average that noise into a map. This turns the flags into a per-chain,
per-bin mask (JSON) and a list of episodes, so a map build can veto them and a monitor can
page on them.

    gnss_chain_health.py <obs jsonl...> [--bin 300] [--min-present 0.3] [--min-run 900]
                         [--el 15] [--out mask.json] [--tail-bytes N] [--recent S]
                         [--alert-file PATH]

An EPISODE is >= --min-run seconds of consecutive bins with present-fraction < --min-present.
Exit status 1 if any episode is found (so a cron can alert on it).

--tail-bytes N   read only the last N bytes of each file (the live check every 15 min must
                 not re-read a day's several GB over NFS; ~40 MB is about an hour of records).
--recent S       count only episodes that end within the last S seconds (a live monitor cares
                 about now, not about a morning that is already in the day's mask).
--alert-file P   write the episode lines to P when there are any, remove P when there are
                 none -- the file's existence is the flag.

Consumers: gnss_beam_cube.py build --health-mask <mask.json> vetoes the masked bins;
scripts/gnss/chain_health_cron.sh runs the live check.
"""
import argparse, json, sys, collections, os, time
from datetime import datetime, timezone

def iso(t): return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M")

def load_mask(path):
    """mask.json -> {chain: (bin_s, set of bin-start times that are NOT ok)}. Missing file
    -> {} (no veto), so a build never fails for want of a mask."""
    if not path or not os.path.exists(path):
        return {}
    with open(path) as fh:
        m = json.load(fh)
    bin_s = float(m.get("bin_s", 300.0))
    return {c: (bin_s, {int(r["t"]) for r in rows if not r["ok"]}) for c, rows in m.get("chains", {}).items()}

def unhealthy(mask, chain, t):
    """Boolean array: which of the times t (unix s) fall in a masked bin for this chain."""
    import numpy as np
    if chain not in mask:
        return np.zeros(len(t), bool)
    bin_s, bad = mask[chain]
    if not bad:
        return np.zeros(len(t), bool)
    b = (np.asarray(t) // bin_s * bin_s).astype(np.int64)
    return np.isin(b, np.fromiter(bad, np.int64))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+"); ap.add_argument("--bin", type=float, default=300.0)
    ap.add_argument("--min-present", type=float, default=0.3); ap.add_argument("--min-run", type=float, default=900.0)
    ap.add_argument("--el", type=float, default=15.0); ap.add_argument("--out")
    ap.add_argument("--tail-bytes", type=int, default=0); ap.add_argument("--recent", type=float, default=0.0)
    ap.add_argument("--alert-file")
    a = ap.parse_args()
    mask = {}; bad = 0; alerts = []
    for f in a.files:
        acc = collections.defaultdict(lambda: [0, 0, 0, 0]); chain = None
        with open(f) as fh:
            if a.tail_bytes and os.path.getsize(f) > a.tail_bytes:
                fh.seek(os.path.getsize(f) - a.tail_bytes); fh.readline()   # drop the partial line
            for line in fh:
                try: d = json.loads(line)
                except Exception: continue
                chain = chain or d.get("band")
                if d.get("el") is None or d["el"] < a.el: continue
                b = int(d["t"] // a.bin) * int(a.bin); c = acc[b]
                c[0] += 1; c[1] += bool(d.get("fleet_present")); c[2] += bool(d.get("prompt_lock")); c[3] += bool(d.get("prompt_rayleigh"))
        if not acc: continue
        bins = sorted(acc); rows = []
        for b in bins:
            n, pr, lk, ry = acc[b]
            rows.append(dict(t=b, n=n, present=pr / n, lock=lk / n, rayleigh=ry / n, ok=(n < 20) or (pr / n >= a.min_present)))
        mask[chain] = rows
        # episodes: runs of not-ok bins
        run = []
        for r in rows + [dict(ok=True, t=rows[-1]["t"] + a.bin)]:
            if not r["ok"]: run.append(r)
            elif run:
                span = run[-1]["t"] + a.bin - run[0]["t"]
                if span >= a.min_run and (not a.recent or time.time() - (run[-1]["t"] + a.bin) < a.recent):
                    bad += 1
                    msg = ("EPISODE %-8s %s -> %s (%.1f h): present %.0f%%, lock %.0f%%, rayleigh %.0f%% over %d records"
                           % (chain, iso(run[0]["t"]), iso(run[-1]["t"] + a.bin), span / 3600,
                              100 * sum(x["present"] * x["n"] for x in run) / sum(x["n"] for x in run),
                              100 * sum(x["lock"] * x["n"] for x in run) / sum(x["n"] for x in run),
                              100 * sum(x["rayleigh"] * x["n"] for x in run) / sum(x["n"] for x in run), sum(x["n"] for x in run)))
                    print(msg); alerts.append(msg)
                run = []
        okf = sum(r["n"] for r in rows if r["ok"]) / max(sum(r["n"] for r in rows), 1)
        print("%-8s %s: %d bins, %.0f%% of records in healthy bins" % (chain, os.path.basename(f), len(rows), 100 * okf))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        tmp = a.out + ".tmp"
        with open(tmp, "w") as fh: json.dump(dict(bin_s=a.bin, min_present=a.min_present, el_min=a.el, chains=mask,
                                                  written=iso(time.time()), files=[os.path.basename(f) for f in a.files]), fh)
        os.replace(tmp, a.out)          # a reader never sees a half-written mask
        print("wrote", a.out)
    if a.alert_file:
        if alerts:
            with open(a.alert_file, "w") as fh: fh.write("\n".join([iso(time.time()) + " UTC"] + alerts) + "\n")
        elif os.path.exists(a.alert_file):
            os.remove(a.alert_file)
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
