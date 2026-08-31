#!/bin/bash
# Per-kernel GPU time breakdown for one tracker node, from kotekan's own /gpu_profile
# endpoints (utilization = fraction of the 41.94 ms frame). Usage: gpu_breakdown.sh [host]
# The A/B instrument for the fp16 (item 3) + centered-window (item 6) arms: run before and
# after the restart, compare the cudaGnssInject rows -- synthesis is 73-89% of that command.
H=${1:-$(hostname -s)}
python3 - "$H" << 'PY'
import json, sys, urllib.request
host = sys.argv[1]
eps = json.load(urllib.request.urlopen(f"http://{host}:12048/endpoints", timeout=5))
profs = sorted(e for e in eps.get("GET", []) if e.startswith("/gpu_profile/"))
tot_inj = tot_corr = tot_all = 0.0
print(f"{'instance':28s} {'inject ms':>10s} {'corrDual ms':>12s} {'copy ms':>9s} {'util %':>7s}")
for e in profs:
    try:
        p = json.load(urllib.request.urlopen(f"http://{host}:12048{e}", timeout=5))
    except Exception as ex:
        print(f"{e:28s}  UNREADABLE: {ex}"); continue
    inj = sum(k["time"] for k in p["kernel"] if "Inject" in k["name"]) * 1e3
    cor = sum(k["time"] for k in p["kernel"] if "Correlator" in k["name"]) * 1e3
    cp = (p.get("copy_in_total_time", 0) + p.get("copy_out_total_time", 0)) * 1e3
    ut = 100.0 * (p.get("kernel_utilization", 0) + p.get("copy_out_utilization", 0)
                  + p.get("copy_in_utilization", 0))
    tot_inj += inj; tot_corr += cor; tot_all += inj + cor + cp
    print(f"{e.split('/')[-1]:28s} {inj:10.3f} {cor:12.3f} {cp:9.3f} {ut:7.2f}")
print(f"{'TOTAL (all instances)':28s} {tot_inj:10.3f} {tot_corr:12.3f}"
      f"  | all {tot_all:.3f} ms per 41.94 ms frame = {100*tot_all/41.94:.1f}% of one GPU-equivalent")
PY
