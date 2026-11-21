#!/usr/bin/env python3
"""Log every tracked satellite's sky position and received power, for a beam map of the
receiver's antenna (beam_map.py).

Each line is one satellite at one time: the broker's azimuth and elevation (get_sky), its fleet
row's prompt power p_pow and the probes' median p_med (the noise reference the presence gate
uses), q, presence, amp_snr and cn0_coh_db, and for GuoWang the symbol SNR from gw_symbols and,
from the broker's GuoWang table (--gw-table), the satellite's name: GuoWang reuses each code on
several satellites, so the code alone does not say which one was measured.

    beam_log.py --out beam.jsonl [--every 5] [--broker http://localhost:12060]
                [--symbols http://localhost:12048/gw_symbols] [--gw-table gw_ephem.json]
"""

import argparse
import json
import os
import time
import urllib.request

# The broker's chains by get_sky's constellation letter.
CONST = {"gps_l5": "G", "gal_e5a": "E", "bds_b2a": "C", "gw_1202": "W"}


def get(url):
    return json.load(urllib.request.urlopen(url, timeout=5))


def gw_names(path, cache):
    """{prn: [(t0_gpst, t1_gpst, name)]} from the GuoWang table, reread when it changes."""
    try:
        m = os.stat(path).st_mtime_ns
        if cache.get("mtime") != m:
            with open(path) as fh:
                recs = json.load(fh)["records"]
            names = {}
            for r in recs:
                names.setdefault(r["prn"], []).append(
                    (r["t0_gpst"], r["t1_gpst"], r.get("name"))
                )
            cache.update(mtime=m, names=names)
    except (OSError, ValueError, KeyError) as e:
        print("GuoWang table unreadable: %s" % e, flush=True)
    return cache.get("names", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="jsonl, appended")
    ap.add_argument("--every", type=float, default=5.0, help="seconds between samples")
    ap.add_argument("--broker", default="http://localhost:12060")
    ap.add_argument("--symbols", default="http://localhost:12048/gw_symbols")
    ap.add_argument(
        "--gw-table", help="the broker's GuoWang table, for satellite names"
    )
    a = ap.parse_args()
    cache = {}
    with open(a.out, "a") as out:
        while True:
            t = time.time()
            try:
                rows = get(a.broker + "/get_status")
                sky = {(s["const"], s["prn"]): s for s in get(a.broker + "/get_sky")}
            except Exception as e:
                print("broker unreadable: %s" % e, flush=True)
                time.sleep(a.every)
                continue
            try:
                sym = {r["prn"]: r["snr_db"] for r in get(a.symbols + "/get_status")}
            except Exception:
                sym = {}
            names = gw_names(a.gw_table, cache) if a.gw_table else {}
            t_gpst = t - 315964800 + 18  # GPS = UTC + 18 s
            # p_med is one per chain; a row above it gives it as p_pow - unbiased_amplitude^2.
            p_med = {}
            for r in rows:
                u = r.get("unbiased_amplitude") or 0.0
                if u > 0:
                    p_med[r.get("chain")] = r["p_pow"] - u * u
            for r in rows:
                c = CONST.get(r.get("chain"))
                s = sky.get((c, r["prn"]))
                if s is None:
                    continue
                rec = dict(
                    t=round(t, 1),
                    const=c,
                    prn=r["prn"],
                    az=s["az"],
                    el=s["el"],
                    present=r.get("fleet_present"),
                    q=r.get("fleet_q"),
                    p_pow=r.get("p_pow"),
                    p_med=p_med.get(r.get("chain")),
                    amp_snr=r.get("amp_snr"),
                    cn0_coh_db=r.get("cn0_coh_db"),
                )
                if c == "W":
                    rec["sym_snr_db"] = sym.get(r["prn"])
                    for t0, t1, name in names.get(r["prn"], []):
                        if t0 <= t_gpst <= t1:
                            rec["name"] = name
                out.write(json.dumps(rec) + "\n")
            out.flush()
            time.sleep(max(0.0, a.every - (time.time() - t)))


if __name__ == "__main__":
    main()
