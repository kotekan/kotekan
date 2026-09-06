#!/bin/bash
# Serve the BEAM-CUBE VIEWER on cf06 -- the static page that draws the healpix beam maps
# exported by gnss_beam_cube.py (python/scripts/gnss/beam_cube_viewer/).
#
#   ssh cf06 '/home/kvand/gnss/kotekan/scripts/gnss/beamview_up.sh'     # http://cf06:8877  (877 is privileged)
#
# WHAT IT SERVES: $WEB/index.json + cube_<day>.json/.bin (written by `gnss_beam_cube.py
# export`) and symlinks to the page sources in the repo, so editing beamcube.js in the tree
# is live on reload. Nothing here is computed: a plain http.server, because the browser
# blocks the cross-origin read of index.json from a file:// page and that failure looks
# identical to a missing file.
#
# THE MAP IS REBUILT BY HAND (P5), not by this script:
#   gnss_beam_cube.py build --source l0 --archive .../rung12/p0_dec40p73 --days YYYYMMDD
#   gnss_beam_cube.py export fixtures/beamcube/cube_YYYYMMDD_nside64.npz ... --nside 32
# Days in one index MUST share units and pointing; the page skips (and names) any that do not.
#
# Part of the cf06 stack (agg, gather, broker, cubearch, cubecompact, obs, viewer, beamview):
# *_up.sh scripts a person runs after a reboot, checked by looking for the process. P6 turns
# them into boot units together.
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

# The bracket trick -- a bare pkill -f pattern matches this script's own command line.
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
