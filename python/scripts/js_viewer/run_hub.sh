#!/usr/bin/env bash
#
# run_hub.sh -- launch the multi-source viewer hub (Option A).
#
# Starts one livebeam_server per source listed in sources.json, each in a
# respawn loop so that --exit-on-disconnect + this loop = it re-accepts when
# its kotekan reconnects. No sudo/systemd needed (livebeam is unprivileged).
#
#   ./run_hub.sh [start|stop|status|landing]     (default: start)
#     start    -- (re)start all source instances + the landing server
#     landing  -- (re)start ONLY the static landing server (leaves streams up)
#
# A small static landing server also runs (default :8090) serving chooser.html
# as /, so people hit  http://<this-host>/  (port 80 redirects to it) instead of
#   http://<this-host>:<any http_port>/chooser.html
#
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY="${VIEWER_PYTHON:-$HERE/../venv-viewer/bin/python}"
SRC="$HERE/sources.json"
LOGDIR="$HERE/../hublogs"; mkdir -p "$LOGDIR"
# Static landing server: serves chooser.html as / on this high port. Port 80
# reaches it via a firewalld forward-port (REDIRECT) -- see ARO_OPERATIONS.md.
LANDING_PORT="${LANDING_PORT:-8090}"

# name http ws kotekan sumfreq dtype  (one line per source)
read_sources() {
  "$PY" - "$SRC" <<'PY'
import json, sys
for s in json.load(open(sys.argv[1]))["sources"]:
    print(s["name"], s["http_port"], s["ws_port"], s["kotekan_port"],
          s.get("sum_freq", 1), s.get("power_dtype", "uint32"),
          s.get("band_map", "-"))
PY
}

# Start the static landing server (chooser at /) in its own respawn loop.
# Independent of the source instances, so it can be (re)started on its own
# without disturbing live streams. Its bash -c cmdline contains landing_server.py,
# so `pkill -f '[l]anding_server.py'` (stop/restart) kills the wrapper too.
start_landing() {
  local DETACH; DETACH=$(command -v setsid >/dev/null 2>&1 && echo setsid || echo nohup)
  pkill -9 -f '[l]anding_server.py' 2>/dev/null; sleep 1
  local llog="$LOGDIR/landing.log"
  local LCMD="\"$PY\" \"$HERE/landing_server.py\" --port $LANDING_PORT --root \"$HERE\""
  $DETACH bash -c "while true; do $LCMD >> \"$llog\" 2>&1; \
    echo \"[hub] landing exited; restarting in 3s\" >> \"$llog\"; sleep 3; done" \
    < /dev/null > /dev/null 2>&1 &
  disown 2>/dev/null || true
  echo "  started landing (http $LANDING_PORT, / -> chooser.html) -> $llog"
}

cmd="${1:-start}"
case "$cmd" in
  stop)
    # SIGKILL defeats the respawn loop (SIGTERM lets it relaunch); the [l]
    # bracket keeps the pattern from matching the shell running this pkill.
    for i in 1 2 3; do
      pkill -9 -f '[l]ivebeam_server.py' 2>/dev/null
      pkill -9 -f '[l]anding_server.py'  2>/dev/null
      sleep 1
    done
    echo "hub stopped."; exit 0;;
  status)
    read_sources | while read -r name hp wp kp sf dt bm; do
      pgrep -f "[s]ource-name $name" >/dev/null && s=RUNNING || s=down
      printf "  %-10s http:%-5s ws:%-5s kotekan:%-5s sum-freq:%-3s [%s]\n" \
             "$name" "$hp" "$wp" "$kp" "$sf" "$s"
    done
    pgrep -f "[l]anding_server.py" >/dev/null && ls=RUNNING || ls=down
    printf "  %-10s http:%-5s (serves chooser at /) [%s]\n" "landing" "$LANDING_PORT" "$ls"
    exit 0;;
  landing)
    # (Re)start ONLY the static landing server, leaving live source instances be.
    command -v "$PY" >/dev/null 2>&1 || [ -x "$PY" ] || { echo "no python at $PY (set VIEWER_PYTHON)"; exit 1; }
    start_landing
    echo "landing up. chooser: http://<this-host>:$LANDING_PORT/"
    exit 0;;
esac

# start
# Accept either a full path (venv python) or a PATH-resolvable command name
# (e.g. VIEWER_PYTHON=python3 on a box with system deps, like the fearless Pi).
command -v "$PY" >/dev/null 2>&1 || [ -x "$PY" ] || { echo "no python at $PY (set VIEWER_PYTHON to a python path or command)"; exit 1; }
# Detach so instances survive this script (and the ssh session) exiting. setsid
# (Linux) fully detaches; nohup is the portable fallback (e.g. macOS testing).
DETACH=$(command -v setsid >/dev/null 2>&1 && echo setsid || echo nohup)
pkill -9 -f '[l]ivebeam_server.py' 2>/dev/null; sleep 1
# Process substitution (not a pipe) so the loop runs in this shell and the
# backgrounded jobs are ours to disown, rather than dying with a subshell.
while read -r name hp wp kp sf dt bm; do
  log="$LOGDIR/$name.log"
  CMD="\"$PY\" \"$HERE/livebeam_server.py\" --source-name $name --power-dtype $dt \
--sum-freq $sf --kotekan-port $kp --http-port $hp --ws-port $wp --exit-on-disconnect"
  # Optional per-source band-map (multi-sub-band de-interleave); "-" = none.
  [ "$bm" != "-" ] && CMD="$CMD --band-map \"$HERE/$bm\""
  $DETACH bash -c "while true; do $CMD >> \"$log\" 2>&1; \
    echo \"[hub] $name livebeam exited; re-accepting in 3s\" >> \"$log\"; sleep 3; done" \
    < /dev/null > /dev/null 2>&1 &
  disown 2>/dev/null || true
  echo "  started $name (http $hp / ws $wp / kotekan $kp / sum-freq $sf) -> $log"
done < <(read_sources)

# Landing server: static chooser at /, independent of any source being up.
start_landing

echo "hub up ($DETACH)."
echo "  chooser: http://<this-host>:$LANDING_PORT/   (and http://<this-host>/ once the port-80 redirect is in place)"
