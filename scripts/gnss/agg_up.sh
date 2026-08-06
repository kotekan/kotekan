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
CFG=${1:-$K/config/generated/chord_gnss_agg6.yaml}
LOG=${2:-/tmp/gnss_agg.log}
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
