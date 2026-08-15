#!/usr/bin/env python3
"""PAIRED DUTY RECORDER for the fleet code loop (task #51 F4 step 4).

Samples the C++ controller's own /get_dll -- one HTTP call covering ALL FIVE CHAINS at once --
and writes a jsonl line per sample. The armed chain and its four controls are therefore
measured in the SAME sample, against the SAME sky, from the SAME statistic. That is what
"paired" has to mean here: gps_l5 and gal_e5a differ in constellation, band and satellite set,
so a level difference between them says nothing; only their behaviour THROUGH TIME, sampled
together, is comparable.

WHY THIS AND NOT THE 30 s POLLER IT REPLACES. The 2026-08-15 overnight run sampled q every
30 s and concluded almost nothing, because the on-peak episodes turned out to be SHORTER than
30 s -- PRN 30 showed 57 on-peak samples in 48 separate runs, i.e. essentially every episode
isolated. A sampler coarser than the phenomenon measures the phenomenon's existence and not
its duration. Default here is 5 s, and `n_updates` is recorded so the loop's own step count is
available rather than inferred from the sample cadence.

⚠️ JUDGE ON q. Never on deep_snr, cn0_coh or coh_frac: #66 measured a PRN 64 degrees BELOW THE
HORIZON reading 11.2x its deep floor, and probe_audit put almost every published statistic at
AUC ~0.5 against those probes. q = 2P/(E+L) is exactly 1.0 with no peak and ~3.6 locked at
0.5-chip spacing, and it comes from the same fold the loop actuates on.

    ./fleet_trim_watch.py --out /tmp/ftwatch.jsonl          # run until killed
    ./fleet_trim_watch.py --out ... --hours 8               # or bounded

Analyse with --report on the file: per-chain on-peak duty and RUN LENGTHS, which is the
statistic that distinguishes "pulls in and loses it" from "holds".
"""
import argparse
import json
import os
import sys
import time
import urllib.request


def get(url, timeout=5.0):
    with urllib.request.urlopen(url, timeout=timeout) as h:
        return json.loads(h.read().decode())


def record(a):
    t_end = time.time() + a.hours * 3600 if a.hours else None
    n = 0
    with open(a.out, "a") as fh:
        while t_end is None or time.time() < t_end:
            t0 = time.time()
            try:
                dll = get(a.url.rstrip("/") + "/get_dll")
                st = get(a.url.rstrip("/") + "/get_stats")
                row = {"t": t0, "chains": {}}
                for chain, prns in dll.items():
                    row["chains"][chain] = {
                        p: {"q": r["q"], "disc": r["disc"], "n_src": r["n_src"],
                            "upd": r["n_updates"]} for p, r in prns.items()}
                # The armed set travels WITH the sample. Without it a later reader cannot tell
                # "the loop was not commanding this PRN" from "the loop was commanding it and
                # it did not help" -- which is the entire question.
                row["armed"] = {c: v.get("armed", []) for c, v in (st.get("policy") or {}).items()}
                row["post_ok"] = st.get("post_ok", 0)
                row["post_fail"] = st.get("post_fail", 0)
                row["close_hz"] = st.get("close_hz_measured", 0.0)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                n += 1
            except Exception as e:
                fh.write(json.dumps({"t": t0, "err": str(e)}) + "\n")
                fh.flush()
            dt = a.interval - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
    return n


def report(a):
    rows = []
    for line in open(a.out):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "chains" in r:
            rows.append(r)
    if len(rows) < 10:
        raise SystemExit("only %d usable samples in %s" % (len(rows), a.out))
    span = (rows[-1]["t"] - rows[0]["t"]) / 3600.0
    print("%d samples over %.2f h (%.1f s cadence)" % (len(rows), span,
                                                       (rows[-1]["t"] - rows[0]["t"]) / max(1, len(rows) - 1)))
    print("post_ok %d  post_fail %d  close_hz %.2f\n"
          % (rows[-1].get("post_ok", 0), rows[-1].get("post_fail", 0), rows[-1].get("close_hz", 0)))

    chains = sorted({c for r in rows for c in r["chains"]})
    print("%-9s %6s %7s %8s %8s   %s" % ("chain", "N_prn", "dutyQ2", "med_run", "max_run",
                                         "armed(mean)"))
    for c in chains:
        # per PRN: the sample indices where it was on peak
        series = {}
        for i, r in enumerate(rows):
            for p, v in r["chains"].get(c, {}).items():
                series.setdefault(p, {})[i] = v["q"]
        on_tot = tot = 0
        runs = []
        for p, s in series.items():
            idx = sorted(s)
            on = [i for i in idx if s[i] > 2.0]
            on_tot += len(on)
            tot += len(idx)
            if not on:
                continue
            a0 = p0 = on[0]
            for i in on[1:]:
                if i == p0 + 1:
                    p0 = i
                else:
                    runs.append(p0 - a0 + 1)
                    a0 = p0 = i
            runs.append(p0 - a0 + 1)
        runs.sort()
        armed = [len(r.get("armed", {}).get(c, [])) for r in rows]
        print("%-9s %6d %6.2f%% %7s %8s   %.2f"
              % (c, len(series), 100.0 * on_tot / max(1, tot),
                 ("%.1f s" % (runs[len(runs) // 2] * a.interval)) if runs else "-",
                 ("%.1f s" % (runs[-1] * a.interval)) if runs else "-",
                 sum(armed) / max(1, len(armed))))
    print("\n⚠️ RUN LENGTH is the statistic, not duty. #51's thesis is that the loop pulls in "
          "and cannot HOLD;\n   a fix shows up as longer arcs, and duty alone cannot "
          "distinguish that from a brighter sky.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:12051/fleet_trim")
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--hours", type=float, default=0.0, help="0 = until killed")
    ap.add_argument("--report", action="store_true", help="analyse --out instead of recording")
    a = ap.parse_args()
    return report(a) if a.report else (record(a) and 0)


if __name__ == "__main__":
    sys.exit(main())
