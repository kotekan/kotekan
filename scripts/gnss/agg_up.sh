#!/bin/bash
# Bring the GNSS aggregator up on the host it is run from (cf06 since 2026-08-04).
# Exists because the equivalent inline ssh one-liner kept losing its redirect and its kill
# to quoting -- a stalled aggregator survived SIGTERM while a new one silently failed to
# bind 12050, which looks exactly like "the move broke the search".
set -u
K=/home/kvand/gnss/kotekan
# BIN: the DPDK-free build for a non-node host (cf06); build/ on a cx node.
CFG=${1:-$K/config/generated/chord_gnss_agg6.yaml}
LOG=${2:-/tmp/gnss_agg.log}
pkill -9 -f "kotekan --config.*agg" 2>/dev/null || true
sleep 5
mkdir -p /tmp/gnss
nohup setsid env GNSS_SEARCH_PROFILE=1 "${GNSS_BIN:-$K/build/kotekan/kotekan}" \
    --config "$CFG" --bind-address 0.0.0.0:12050 > "$LOG" 2>&1 < /dev/null &
disown
sleep 8
pgrep -f "kotekan --config.*agg" > /dev/null && echo "aggregator up (log $LOG)" || echo "FAILED to start"
