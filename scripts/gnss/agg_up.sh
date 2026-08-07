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
sleep 5
mkdir -p /tmp/gnss
nohup setsid env GNSS_SEARCH_PROFILE=1 "$BIN" \
    --config "$CFG" --bind-address 0.0.0.0:12050 > "$LOG" 2>&1 < /dev/null &
disown
sleep 8
pgrep -f "kotekan --config.*agg" > /dev/null \
    && echo "aggregator up on $H using $BIN (log $LOG)" \
    || { echo "FAILED to start -- check $LOG" >&2; exit 1; }
