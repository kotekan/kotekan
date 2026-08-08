#!/bin/bash
# Start the ONE CHORD GNSS viewer -- every constellation, one instance (task #27 M6).
#
#   scripts/gnss/viewer_up.sh            # http://cf06:8080
#
# WHAT THIS REPLACES: two instances, one per broker port (8080 -> 12060 GPS, 8081 -> 12061
# E5a), because a publisher served one chain per port. The broker now publishes every chain
# on ONE port and filters on a path segment, so one viewer polls /gps_l5/get_status and
# /gal_e5a/get_status against the same origin.
#
# THE CHAIN TABLE COMES FROM THE BROKER, not from a flag and not from a static table.
# livebeam_server calls discover_broker_chains() against --kotekan-rest-port; the broker
# answers /get_chains with what it is actually running, including the constellation, the
# record length and whether a search feeds it. So adding B2a to config/gnss_chains_chord.yaml
# is the ONLY step -- the viewer picks it up on its next start with nothing edited here.
#
# ⚠️ DO NOT pass --gps-constellations / --gps-search-stage / --gps-combiner-stage. They feed
# the static AIRSPY table (gal_search, gal_combiner, airspy_in), which is what a viewer
# pointed at a broker port used to fall back to -- polling stage names that have never
# existed on CHORD and surfacing it as a CORS error three layers from the cause. They are
# still right for the airspy prototype and for pointing at a kotekan NODE, where /get_chains
# does not answer and the old path runs unchanged.
#
# usage:  viewer_up.sh [extra livebeam_server args...]
set -u
K=/home/kvand/gnss/kotekan
PY=/home/kvand/gnss/venv/bin/python
LOG=${GNSS_VIEWER_LOG:-/tmp/gnss_viewer_unified.log}
PORT=${GNSS_VIEWER_PORT:-8080}
REST=${GNSS_BROKER_PORT:-12060}

cd "$K/python/scripts/js_viewer" || exit 1

# The bracket trick, same as broker_restart.sh: `pkill -f livebeam_server` would match this
# script's own command line and kill the shell running it, so the rest never executes.
pkill -f "[l]ivebeam_server" 2>/dev/null || true
sleep 2

nohup setsid $PY -u livebeam_server.py \
    --no-power-stream \
    --http-port "$PORT" --ws-port 8539 \
    --kotekan-rest-port "$REST" \
    --lat 49.32075144444 --lon -119.62081125 --alt 545 \
    --band l5 \
    "$@" > "$LOG" 2>&1 < /dev/null &
disown
sleep 6

if ss -ltn 2>/dev/null | grep -q ":$PORT"; then
    echo "viewer up on :$PORT (log $LOG)"
    grep -a "broker chain discovery" "$LOG" | tail -1
    echo "chains: $(curl -s -m3 "http://localhost:$PORT/wsport" \
        | $PY -c 'import json,sys; print(", ".join(c["name"] for c in json.load(sys.stdin)["chains"]))' 2>/dev/null)"
else
    echo "FAILED to start -- last lines of $LOG:"
    tail -15 "$LOG"
    exit 1
fi
