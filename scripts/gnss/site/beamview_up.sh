#!/bin/bash
# Serve the beam-cube viewer on cf06: the static page that draws the healpix beam maps
# exported by gnss_beam_cube.py (python/scripts/gnss/beam_cube_viewer/).
#
#   ssh cf06 '/home/kvand/gnss/kotekan/scripts/gnss/site/beamview_up.sh'     # http://cf06:8877  (877 is privileged)
#
# It serves $WEB/index.json + cube_<day>.json/.bin (written by `gnss_beam_cube.py export`) and
# symlinks to the page sources in the repo, so editing beamcube.js in the tree is live on
# reload. Nothing here is computed: a plain http.server, because the browser blocks the
# cross-origin read of index.json from a file:// page and that failure looks identical to a
# missing file.
#
# The maps are built by beamcube_daily.sh, which the compactor loop runs once a day, not by
# this script; `beamcube_daily.sh YYYYMMDD` builds one day by hand (BEAMCUBE_FORCE=1 rebuilds
# an existing master). Days in one index must share units and pointing; the page skips (and
# names) any that do not.
#
# This server has no boot unit: start it by hand after a cf06 reboot.
set -u
WEB=${BEAMVIEW_WEB:-/home/kvand/gnss/fixtures/beamcube/web}
PORT=${BEAMVIEW_PORT:-8877}
LOG=${BEAMVIEW_LOG:-/tmp/gnss_beamview.log}
SRC=/home/kvand/gnss/kotekan/python/scripts/gnss/beam_cube_viewer

HOST=${BEAMVIEW_HOST_NAME:-cf06}
if [ "$(hostname -s)" != "$HOST" ]; then
    echo "REFUSING: the beam-cube viewer is served from $HOST, this is $(hostname -s)." >&2
    echo "Use:  ssh $HOST '$0'" >&2
    exit 1
fi
[ -f "$WEB/index.json" ] || { echo "REFUSING: $WEB/index.json missing -- run gnss_beam_cube.py export first" >&2; exit 1; }
ln -sf "$SRC/index.html" "$WEB/index.html"
ln -sf "$SRC/beamcube.js" "$WEB/beamcube.js"

# The bracket keeps the pattern from matching a command line that carries the pattern itself
# (a `bash -c` running this line); a bare pattern would kill that shell.
pkill -f "[h]ttp.server $PORT" 2>/dev/null || true
sleep 1
[ -f "$LOG" ] && mv "$LOG" "$LOG.$(date -u +%Y%m%d_%H%M%S)"
cd "$WEB" || exit 1
nohup setsid python3 -m http.server "$PORT" --bind 0.0.0.0 >> "$LOG" 2>&1 &
sleep 2
if ss -ltn | grep -q ":$PORT "; then
    echo "beam-cube viewer up: http://$HOST:$PORT/  serving $WEB ($(python3 -c "import json;d=json.load(open('$WEB/index.json'));print(' '.join(x['day'] for x in d['days']))"))"
else
    echo "FAILED to bind :$PORT; see $LOG" >&2; tail -n 20 "$LOG" >&2; exit 1
fi
