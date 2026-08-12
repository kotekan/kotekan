#!/usr/bin/env python3
"""Task #49 A/B: does the deep presence gate shorten off-peak excursions?

WHY A SAMPLER AND NOT A SNAPSHOT. The fault ALTERNATES between a satellite's two bands. On
2026-08-12 the two Galileo bands swapped roles inside 20 minutes -- e5a went disc -0.878 /
L-P 4.98 / trim frozen -> +0.146 / 0.27 / trimmed, while e5b went the other way. Either
snapshot, taken alone, "confirms" whichever story it is held up against. So poll continuously
and compare DUTY CYCLE and EXCURSION LENGTH across many alternations.

The pair is the same satellite on two bands: same sky, same geometry, same receiver clock,
one gate changed. That is the tightest control available -- and it is a control in TIME, which
is what [[chord-the-sky-moves-faster-than-you]] says to use here.

    deep_gate_watch.py --prn 33 --armed gal_e5a --control gal_e5b --out /tmp/deep_gate_33.jsonl
    deep_gate_watch.py --report /tmp/deep_gate_33.jsonl
"""
import argparse
import json
import statistics as st
import time
import urllib.request

STUCK = 0.6            # |disc| above this = off the peak (measured: healthy sits well below)


def poll(url, chain, prn):
    with urllib.request.urlopen("%s/get_status?chain=%s" % (url, chain), timeout=8) as r:
        rows = json.load(r)
    if isinstance(rows, dict):
        rows = rows.get("rows") or []
    for x in rows:
        if x.get("prn") == prn:
            p = x.get("p_pow") or 0.0
            return {"disc": x.get("dll_disc"), "trim": x.get("dll_trim"),
                    "present": x.get("fleet_present"), "gate": x.get("present_gate"),
                    "deep_snr": x.get("deep_snr"), "deep_floor": x.get("deep_floor"),
                    # #50: the far-regime error signal and its shuffled-null significance,
                    # plus q -- which is what says whether the DISCRIMINATOR had anything to
                    # follow. Without q in the record, a flat disc cannot be told apart from
                    # "on the peak" and the A/B is unreadable (#47's lesson, one layer down).
                    "spec_tau": x.get("spec_tau_chips"),
                    "spec_ratio": x.get("spec_peak_ratio"), "q": x.get("fleet_q"),
                    "lp": (x.get("l_pow") or 0.0) / p if p else None,
                    "ep": (x.get("e_pow") or 0.0) / p if p else None}
    return None


def report(path):
    rec = [json.loads(l) for l in open(path) if l.startswith("{")]
    if not rec:
        print("no samples")
        return 1
    arms = sorted({r["arm"] for r in rec})
    print("%d samples over %.1f min\n" % (len(rec), (rec[-1]["t"] - rec[0]["t"]) / 60.0))
    print("%-10s %-9s %6s %8s %9s %8s %9s %8s" %
          ("chain", "role", "n", "stuck%", "med|disc|", "med L/P", "trimmed%", "longest"))
    for arm in arms:
        v = [r for r in rec if r["arm"] == arm]
        d = [abs(r["disc"]) for r in v if r.get("disc") is not None]
        lp = [r["lp"] for r in v if r.get("lp") is not None]
        pres = [r for r in v if r.get("present")]
        # longest unbroken stuck run, in samples -> minutes
        run = best = 0
        for r in v:
            run = run + 1 if (r.get("disc") is not None and abs(r["disc"]) > STUCK) else 0
            best = max(best, run)
        dt = (v[-1]["t"] - v[0]["t"]) / max(1, len(v) - 1)
        print("%-10s %-9s %6d %7.1f%% %9.3f %8.2f %7.1f%% %6.1f min" %
              (v[0]["chain"], arm, len(v),
               100.0 * sum(1 for x in d if x > STUCK) / max(1, len(d)),
               st.median(d) if d else float("nan"),
               st.median(lp) if lp else float("nan"),
               100.0 * len(pres) / len(v), best * dt / 60.0))
    print("\nJUDGE: the ARMED band should show a lower stuck%%, a lower median |disc| and a"
          "\nSHORTER longest excursion. The claim is faster pull-in, not immunity -- and a"
          "\ndifference smaller than the alternation period is not yet evidence.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://cf06:12060")
    ap.add_argument("--prn", type=int, default=33)
    ap.add_argument("--armed", default="gal_e5a")
    ap.add_argument("--control", default="gal_e5b")
    ap.add_argument("--interval", type=float, default=20.0)
    ap.add_argument("--minutes", type=float, default=180.0)
    ap.add_argument("--out")
    ap.add_argument("--report")
    a = ap.parse_args()
    if a.report:
        return report(a.report)
    end = time.time() + a.minutes * 60.0
    with open(a.out, "a") as f:
        while time.time() < end:
            t = time.time()
            for arm, chain in (("armed", a.armed), ("control", a.control)):
                try:
                    r = poll(a.url, chain, a.prn)
                except Exception:
                    r = None
                if r:
                    f.write(json.dumps(dict(t=t, arm=arm, chain=chain, prn=a.prn, **r)) + "\n")
            f.flush()
            time.sleep(max(1.0, a.interval - (time.time() - t)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
