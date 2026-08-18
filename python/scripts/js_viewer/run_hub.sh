#!/usr/bin/env bash
#
# run_hub.sh -- launch the multi-source viewer hub (Option A).
#
# Starts one livebeam_server per source listed in sources.json, each in a
# respawn loop so that --exit-on-disconnect + this loop = it re-accepts when
# its kotekan reconnects. No sudo/systemd needed (livebeam is unprivileged).
#
#   ./run_hub.sh [start|stop|status]     (default: start)
#
# The receiver box runs all instances; open the chooser at
#   http://<this-host>:<any http_port>/chooser.html
#
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY="${VIEWER_PYTHON:-$HERE/../venv-viewer/bin/python}"
SRC="$HERE/sources.json"
LOGDIR="$HERE/../hublogs"; mkdir -p "$LOGDIR"

# name http ws kotekan sumfreq dtype  (one line per source)
read_sources() {
  "$PY" - "$SRC" <<'PY'
import json, sys
for s in json.load(open(sys.argv[1]))["sources"]:
    print(s["name"], s["http_port"], s["ws_port"], s["kotekan_port"],
          s.get("sum_freq", 1), s.get("power_dtype", "uint32"))
PY
}

cmd="${1:-start}"
case "$cmd" in
  stop)
    # SIGKILL defeats the respawn loop (SIGTERM lets it relaunch); the [l]
    # bracket keeps the pattern from matching the shell running this pkill.
    for i in 1 2 3; do pkill -9 -f '[l]ivebeam_server.py' 2>/dev/null; sleep 1; done
    echo "hub stopped."; exit 0;;
  status)
    read_sources | while read -r name hp wp kp sf dt; do
      pgrep -f "[s]ource-name $name" >/dev/null && s=RUNNING || s=down
      printf "  %-10s http:%-5s ws:%-5s kotekan:%-5s sum-freq:%-3s [%s]\n" \
             "$name" "$hp" "$wp" "$kp" "$sf" "$s"
    done; exit 0;;
esac

# start
[ -x "$PY" ] || { echo "no venv python at $PY (set VIEWER_PYTHON)"; exit 1; }
# Detach so instances survive this script (and the ssh session) exiting. setsid
# (Linux) fully detaches; nohup is the portable fallback (e.g. macOS testing).
DETACH=$(command -v setsid >/dev/null 2>&1 && echo setsid || echo nohup)
pkill -9 -f '[l]ivebeam_server.py' 2>/dev/null; sleep 1
# Process substitution (not a pipe) so the loop runs in this shell and the
# backgrounded jobs are ours to disown, rather than dying with a subshell.
while read -r name hp wp kp sf dt; do
  log="$LOGDIR/$name.log"
  CMD="\"$PY\" \"$HERE/livebeam_server.py\" --source-name $name --power-dtype $dt \
--sum-freq $sf --kotekan-port $kp --http-port $hp --ws-port $wp --exit-on-disconnect"
  $DETACH bash -c "while true; do $CMD >> \"$log\" 2>&1; \
    echo \"[hub] $name livebeam exited; re-accepting in 3s\" >> \"$log\"; sleep 3; done" \
    < /dev/null > /dev/null 2>&1 &
  disown 2>/dev/null || true
  echo "  started $name (http $hp / ws $wp / kotekan $kp / sum-freq $sf) -> $log"
done < <(read_sources)
echo "hub up ($DETACH). Chooser: http://<this-host>:$(read_sources | head -1 | awk '{print $2}')/chooser.html"
