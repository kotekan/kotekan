#!/bin/bash
# Bring the GNSS aggregator up on the host it is run from (cf06 since 2026-08-04).
# Exists because the equivalent inline ssh one-liner kept losing its redirect and its kill
# to quoting -- a stalled aggregator survived SIGTERM while a new one silently failed to
# bind 12050, which looks exactly like "the move broke the search".
set -u
K=/home/kvand/gnss/kotekan
# BIN: PICK BY HOST. The aggregator normally runs on cf06, which has no DPDK, so the default
# has to be build_nodpdk/ -- it used to default to build/ (the cx node build), and on cf06 that
# does not start. The failure is quiet in the worst way: pkill kills the old aggregator, the new
# one dies on the DPDK symbols, and every node then logs "Connection refused" to 11040, which
# reads as a node problem rather than a launcher problem. Override with GNSS_BIN.
H=$(hostname -s)
case "$H" in
cx*) DEF=$K/build/kotekan/kotekan ;;        # node host: the DPDK build
*)   DEF=$K/build_nodpdk/kotekan/kotekan ;; # cf06 and anything else: DPDK-free
esac
BIN=${GNSS_BIN:-$DEF}
if [ ! -x "$BIN" ]; then
    echo "FAILED: no aggregator binary at $BIN (host $H). Build it, or set GNSS_BIN." >&2
    exit 1
fi
# CONFIG: PREFER THE GPU SEARCH WHERE THERE IS A GPU.
#
# This default used to be the CPU config unconditionally, with the CUDA one sitting beside it
# under a different name -- so taking the default on a GPU host silently bought the 46x-slower
# search. That is exactly what happened on 2026-08-07: cf06 has two L40S, both sat at 0% while
# GnssChannelizedSearch ran its CPU path 100%-blocked and discarding 90% of frames, and the
# aggregator produced ZERO detections for the best part of an hour. The 29-hour instance it
# replaced had the same defect, so nobody had noticed; three separate restarts that day all
# took the default and all reproduced it.
#
# So: pick the _cuda config when this host actually has a GPU and that config exists, and SAY
# which one was chosen and why. An explicit first argument still wins -- passing the CPU config
# on a GPU host is a legitimate control run, it just should not be what you get by accident.
DEF_CFG=$K/config/generated/chord_gnss_agg6.yaml
CUDA_CFG=$K/config/generated/chord_gnss_agg6_cuda.yaml
WHY="CPU search (default)"
if [ -f "$CUDA_CFG" ] && nvidia-smi -L >/dev/null 2>&1; then
    DEF_CFG=$CUDA_CFG
    WHY="GPU search ($(nvidia-smi -L 2>/dev/null | wc -l) GPU(s) present)"
fi
CFG=${1:-$DEF_CFG}
[ -n "${1:-}" ] && WHY="explicitly requested"
LOG=${2:-/tmp/gnss_agg.log}
echo "aggregator config: $(basename "$CFG") -- $WHY"
# The search only uses the GPU if the CONFIG asks it to; the filename is not the contract.
# Check the key itself, so a hand-picked or regenerated config cannot quietly disagree with
# the name it is stored under.
if nvidia-smi -L >/dev/null 2>&1 && ! grep -q 'use_cuda_acquire: *true' "$CFG"; then
    echo "  NOTE: this host has a GPU but $(basename "$CFG") does not set use_cuda_acquire --" >&2
    echo "        the search will run on the CPU (measured ~46x slower, and it starves)." >&2
fi
pkill -9 -f "kotekan --config.*agg" 2>/dev/null || true
# WAIT FOR THE PORTS, NOT A FIXED SLEEP. 2026-08-12: a 5 s sleep was not enough -- the new
# instance found :11040 still held by the dying one, hit "Address already in use", and shut
# ITSELF down as a FatalError. The launcher then reported success (a process existed: the one
# that was busy exiting), so the fleet sat with no aggregator at all while the log said
# "aggregator up". Poll instead, and say so if the port never frees.
for _i in $(seq 1 30); do
    ss -ltn 2>/dev/null | grep -qE ":(11040|12050)\b" || break
    sleep 1
done
if ss -ltn 2>/dev/null | grep -qE ":(11040|12050)\b"; then
    echo "FAILED: :11040/:12050 still bound after 30 s -- something else holds them" >&2
    ss -ltnp 2>/dev/null | grep -E ":(11040|12050)\b" >&2
    exit 1
fi
mkdir -p /tmp/gnss
nohup setsid env GNSS_SEARCH_PROFILE=1 "$BIN" \
    --config "$CFG" --bind-address 0.0.0.0:12050 > "$LOG" 2>&1 < /dev/null &
disown
sleep 8
pgrep -f "kotekan --config.*[a]gg" > /dev/null \
    || { echo "FAILED to start -- check $LOG" >&2; exit 1; }
if grep -qi "FatalError" "$LOG" 2>/dev/null; then
    echo "FAILED: the stage graph raised a FatalError and is shutting down:" >&2
    grep -i "FatalError" "$LOG" | head -3 >&2
    exit 1
fi
echo "aggregator up on $H using $BIN (log $LOG)"

# ---- POST-START EPOCH ASSERTION (2026-08-12) ---------------------------------------------
# THE DEPLOY GATE THIS LAUNCHER DID NOT HAVE. /home/kvand is NFS with TWO build trees --
# build/ (cx nodes, DPDK) and build_nodpdk/ (here) -- so "I reverted the source" and "the
# binary running on cf06 is reverted" are different statements. On 2026-08-12 they diverged:
# this host ran a stale binary whose detection_phase referenced a hop's FIRST sample instead
# of its last, every GPS seed commanded a code phase 52.37 chips off the peak, gps_l5 went
# from 6 strong satellites to 1, and nothing anywhere said so. It took an hour to find and
# seconds to fix.
#
# The invariant is exact and cheap: the payload's two phase fields sit ONE HOP apart by
# construction -- cp0 back-references to sample 0 at the window's first sample, cp_at_ref is
# referenced at its last (ChannelizedReplicaBank::window_advance_chips). That difference is
# chip_rate/hops_per_sec = 52.3776 chips at CHORD, Doppler-scaled, and it is a property of
# the CODE, not of the sky -- so any healthy build reproduces it on the first detection.
sleep 12
python3 - "$LOG" <<'EPOCHCHK' || true
import json, sys, urllib.request
HPS, CHIP, CARR, SGN, L = 195312.5, 10.23e6, 1176.45e6, 1.0, 10230.0
try:
    with urllib.request.urlopen("http://localhost:12050/gps_search/get_detections",
                                timeout=5) as r:
        dets = json.load(r)
except Exception as e:
    print("  epoch check: SKIPPED (no detections yet: %s)" % e)
    raise SystemExit
rows = []
for d in dets:
    car = d.get("code_phase_at_ref_chips", -1.0)
    if car is None or car < 0 or d.get("snr", 0) < 50:
        continue
    t = d["ref_hop"] / HPS
    recon = (d["code_phase_chips"] + t * CHIP * (1.0 + SGN * d["doppler_hz"] / CARR)) % L
    rows.append((d["prn"], ((car % L - recon + L / 2) % L) - L / 2))
if not rows:
    print("  epoch check: SKIPPED (no strong detections yet)")
    raise SystemExit
bad = [(p, v) for p, v in rows if abs(v - 52.3776) > 1.0]
if bad:
    print("  *** EPOCH CHECK FAILED on %d/%d detections: cp_at_ref - reconstruction is %s,"
          % (len(bad), len(rows), ", ".join("%+.2f" % v for _, v in bad[:4])))
    print("  *** expected +52.38 (one hop). A ~0 offset means THIS HOST'S BINARY IS STALE:")
    print("  ***   cd /home/kvand/gnss/kotekan/build_nodpdk && ninja kotekan   (then restart)")
    print("  *** Do not leave it running: every search-fed seed is ~52 chips off the peak.")
else:
    print("  epoch check OK: cp_at_ref - reconstruction = %+.2f chips over %d detection(s)"
          % (sum(v for _, v in rows) / len(rows), len(rows)))
EPOCHCHK
