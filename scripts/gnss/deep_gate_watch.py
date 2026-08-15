#!/usr/bin/env python3
"""#49: does the deep gate actually pull satellites IN? Armed PRNs vs in-chain controls.

    scripts/gnss/deep_gate_watch.py --gate 4,9,27 [--chain gps_l5] [--hours 6]
    scripts/gnss/deep_gate_watch.py --report /tmp/deep_gate_gps_l5.jsonl   # analyse only

Referenced by config/gnss_chains_chord.yaml since the gal_e5a arm and never actually written;
written 2026-08-15 when the gate was armed on gps_l5.

⚠️ WHY THIS IS NOT A BEFORE/AFTER. Arming the gate needs a broker restart, and a restart
re-derives every seed: measured within one minute of the gps_l5 arm, the armed PRN 4 went
dll_disc +0.755 -> -0.043 AND the untouched controls moved just as far (PRN 11 +0.775 ->
+0.094, PRN 26 +0.567 -> -0.004). Any before/after across that restart would have credited the
gate with the restart's own effect. The sky also churns faster than the change (docs: deep_snr
max swung 52-197 in four minutes).

SO THE DESIGN IS PAIRED IN TIME AND WITHIN ONE CHAIN. Armed and control PRNs are sampled in
the SAME poll, on the same sky, the same clock and the same cycle, for hours. What is compared
is a DUTY -- the fraction of samples a PRN sits at large |disc| -- not any single reading.

⚠️ AND THE COMPARISON IS CONDITIONED ON BEING DETECTED. A control PRN that the fold never sees
is not evidence the gate does nothing; it is a satellite the gate would never have admitted
either. Only samples where deep_snr >= margin*deep_floor count for the primary statistic, and
the unconditioned numbers are printed beside them so the conditioning is visible rather than
hidden.

THE CLAIM BEING TESTED, from #49: an armed PRN spends LESS time stuck at large |disc| and its
excursions are SHORTER. NOT that it never drifts, and NOT anything about C/N0 -- cn0_coh
cannot see code error at all (#47).
"""
import argparse
import json
import os
import statistics
import time
import urllib.request

STUCK = 0.6      # |disc| above this = "stuck off-peak", the bar #49's gal_e5a note uses


def sample(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def collect(args):
    url = "http://%s/%s/get_status" % (args.host, args.chain)
    path = args.out or "/tmp/deep_gate_%s.jsonl" % args.chain
    gate = {int(x) for x in args.gate.replace(",", " ").split()} if args.gate else set()
    t_end = time.time() + args.hours * 3600.0
    n = 0
    print("watching %s, armed %s, every %.0f s -> %s"
          % (args.chain, sorted(gate) or "(none)", args.interval, path))
    with open(path, "a") as fh:
        while time.time() < t_end:
            try:
                rows = sample(url)
            except Exception as e:                       # a poll failure must not end the watch
                print("poll failed: %s" % e)
                time.sleep(args.interval)
                continue
            rec = {"t": time.time(), "gate": sorted(gate), "rows": [
                {"prn": r["prn"], "q": r.get("fleet_q"), "present": r.get("fleet_present"),
                 "disc": r.get("dll_disc"), "deep": r.get("deep_snr"),
                 "floor": r.get("deep_floor")} for r in rows]}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()                                    # crash-safe: the analysis is the file
            n += 1
            if n % 10 == 0:
                print("  %d samples" % n)
            time.sleep(args.interval)
    print("done, %d samples -> %s" % (n, path))
    return path


def report(path, margin=3.0):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    if not recs:
        raise SystemExit("no samples in %s" % path)
    gate = set(recs[-1].get("gate") or [])
    per = {}
    for rec in recs:
        for r in rec["rows"]:
            d = per.setdefault(r["prn"], {"disc": [], "q": [], "pres": [], "det": []})
            if r["disc"] is None:
                continue
            det = bool(r["floor"]) and (r["deep"] or 0) >= margin * r["floor"]
            d["det"].append(det)
            d["pres"].append(bool(r["present"]))
            if det:                      # conditioned: only where the gate COULD have admitted
                d["disc"].append(abs(r["disc"]))
                d["q"].append(r["q"] or 0.0)
    span = (recs[-1]["t"] - recs[0]["t"]) / 60.0
    print("%d samples over %.1f min, armed %s\n" % (len(recs), span, sorted(gate) or "(none)"))
    print("conditioned on deep_snr >= %.1fx floor -- 'n_det' is how many samples qualified" % margin)
    print("PRN  arm   n_det  det%%   present%%  med|disc|  stuck%%(>%.1f)  med q" % STUCK)
    agg = {"GATE": {"stuck": [], "disc": [], "pres": []},
           "ctl": {"stuck": [], "disc": [], "pres": []}}
    for prn in sorted(per):
        d = per[prn]
        if not d["det"]:
            continue
        arm = "GATE" if prn in gate else "ctl"
        detpc = 100.0 * sum(d["det"]) / len(d["det"])
        prespc = 100.0 * sum(d["pres"]) / len(d["pres"])
        if d["disc"]:
            md = statistics.median(d["disc"])
            stuck = 100.0 * sum(1 for x in d["disc"] if x > STUCK) / len(d["disc"])
            mq = statistics.median(d["q"])
            agg[arm]["stuck"].append(stuck)
            agg[arm]["disc"].append(md)
            agg[arm]["pres"].append(prespc)
        else:
            md = stuck = mq = float("nan")
        print("%-4d %-5s %5d  %5.1f  %8.1f  %9.3f  %12.1f  %5.2f"
              % (prn, arm, len(d["disc"]), detpc, prespc, md, stuck, mq))
    print()
    for arm in ("GATE", "ctl"):
        a = agg[arm]
        if not a["stuck"]:
            print("%-5s no detected samples" % arm)
            continue
        print("%-5s n_prn %2d   median stuck%% %5.1f   median |disc| %.3f   median present%% %5.1f"
              % (arm, len(a["stuck"]), statistics.median(a["stuck"]),
                 statistics.median(a["disc"]), statistics.median(a["pres"])))
    if agg["GATE"]["stuck"] and agg["ctl"]["stuck"]:
        g, c = statistics.median(agg["GATE"]["stuck"]), statistics.median(agg["ctl"]["stuck"])
        print("\nVERDICT: armed PRNs stuck %.1f%% of detected samples vs %.1f%% for controls -- %s"
              % (g, c, "the gate is pulling them in" if g < c - 5
                 else "no difference yet" if abs(g - c) <= 5 else "⚠️ armed are WORSE"))
        print("⚠️ n_prn is small and these are medians over PRNs; read it with the per-PRN rows,")
        print("   and remember a satellite the fold never detects contributes nothing either way.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1:12060")
    ap.add_argument("--chain", default="gps_l5")
    ap.add_argument("--gate", default="", help="armed PRNs, e.g. 4,9,27")
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--margin", type=float, default=3.0)
    ap.add_argument("--report", default="", help="analyse an existing jsonl and exit")
    a = ap.parse_args()
    if a.report:
        report(a.report, a.margin)
        return
    report(collect(a), a.margin)


if __name__ == "__main__":
    main()
