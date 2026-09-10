#!/bin/bash
# Start the beam-cube COMPACTOR loop on cf06 -- raw rawFileWrite bundles -> L0 HDF5 -> rungs.
#
# WHAT IT DOES, FOREVER, every CUBE_COMPACT_PERIOD seconds (default 300):
#   1. gnss_cube_compact.py compact   folds every CLOSED raw file not yet in the manifest into
#      <ROOT>/l0/<pointing>/<sender>/<YYYYMMDD>.h5 (idempotent; ~17x faster than real time).
#   2. once per UTC day, after 00:20 UTC: gnss_cube_compact.py rung --n 12 and --n 60 for
#      YESTERDAY's L0 files (a rung is a full rebuild of the day, so it waits for the day to close).
#
# WHY A LOOP AND NOT A CRON: the cf06 stack (agg, gather, broker, cubearch, obs, viewer) is
# started by *_up.sh scripts that a person runs after a reboot, and it is checked by looking for
# the process. A cron entry would be the one piece that is not visible that way. P6 turns all of
# these into boot units together.
#
# ⚠️ THE EPOCH (--utc0) IS ONLY FOR v2 FRAMES. Frames recorded after the v3 cube cycle carry
# their own utc0 and the flag is ignored for them. For the 2026-09-05 v2 archive the value is
# time0 = 1788541059.000002870 (frame0_nano 1169225859000002870, GPS-week-rollover corrected),
# and the compactor GATES it against each raw file's mtime (|last window - mtime| < 60 s;
# measured +1.2..+2.6 s), so a wrong epoch REFUSES the file rather than dating it wrong. If the
# F-engine restarts before the v3 cycle, the v2 files after the restart need a new CUBE_UTC0 --
# the refusals in the log are how you find out.
#
# ⚠️ RUNS ON cf06 ONLY. The raw tree is on /mnt/cs00 and the nodes are production trackers.
#
# usage:  ssh cf06 '/home/kvand/gnss/kotekan/scripts/gnss/cubecompact_up.sh'
#         CUBE_UTC0=<s> CUBE_ROOT=/mnt/cs00/data/kvand/gnss_cube_test ... (a test run)
set -u
K=/home/kvand/gnss/kotekan
PY=${GNSS_PY:-/home/kvand/gnss/venv/bin/python}     # needs h5py: venv, NOT venv-ft
ROOT=${CUBE_ROOT:-/mnt/cs00/data/kvand/gnss_cube}
RAW=${CUBE_RAW:-$ROOT/raw}
UTC0=${CUBE_UTC0:-1788541059.000002870}
PERIOD=${CUBE_COMPACT_PERIOD:-300}
LOG=${CUBE_COMPACT_LOG:-/tmp/gnss_cubecompact.log}
TOOL=$K/python/scripts/gnss/gnss_cube_compact.py

HOST=${CUBE_HOST_NAME:-cf06}
if [ "$(hostname -s)" != "$HOST" ]; then
    echo "REFUSING: the compactor runs on $HOST (raw tree $RAW), this is $(hostname -s)." >&2
    echo "Use:  ssh $HOST '$0'" >&2
    exit 1
fi
if ! "$PY" -c "import h5py, numpy" 2>/dev/null; then
    echo "REFUSING: $PY has no h5py (venv-ft does not; use /home/kvand/gnss/venv/bin/python)" >&2
    exit 1
fi
[ -d "$RAW" ] || { echo "REFUSING: raw tree $RAW does not exist" >&2; exit 1; }

# The bracket trick -- a bare pkill -f pattern matches this script's own command line.
pkill -f "[c]ubecompact_loop.sh" 2>/dev/null || true

# ⚠️⚠️ WAIT FOR THE CHILD, NOT JUST THE LOOP. pkill takes down the bash loop and leaves any
# in-flight `gnss_cube_compact.py` ORPHANED -- a pass runs ~25 s and the rung/publish pass runs
# minutes. Starting the new loop on a 1 s sleep would put two compactors on the same 89 L0
# files, which is a corrupting write, not a slow one. Wait for the old child to finish (it is
# append-per-raw-file and exits cleanly), and only give up after 15 min.
for _i in $(seq 1 180); do
    pgrep -f "[g]nss_cube_compact.py" >/dev/null || break
    [ "$_i" = 1 ] && echo "waiting for the in-flight compact pass to finish..."
    sleep 5
done
if pgrep -f "[g]nss_cube_compact.py" >/dev/null; then
    echo "REFUSING: a gnss_cube_compact.py is still running after 15 min. Two compactors on one" >&2
    echo "L0 tree corrupt it -- find out what that pass is doing before starting another." >&2
    exit 1
fi
# Same for a publish left holding the lock: it reads the archive and writes the viewer dir.
for _i in $(seq 1 120); do
    pgrep -f "[b]eamcube_daily.sh" >/dev/null || break
    sleep 5
done
mkdir -p "$ROOT/l0" "$(dirname "$LOG")"
[ -f "$LOG" ] && mv "$LOG" "$LOG.$(date -u +%Y%m%d_%H%M%S)"

nohup setsid "$K/scripts/gnss/cubecompact_loop.sh" "$PY" "$TOOL" "$RAW" "$ROOT" "$UTC0" "$PERIOD" \
    >> "$LOG" 2>&1 &
sleep 3
if pgrep -f "[c]ubecompact_loop.sh" >/dev/null; then
    echo "cube compactor loop up: raw $RAW -> $ROOT/l0 every $PERIOD s, rungs after 00:20 UTC; log $LOG"
    tail -n 5 "$LOG"
else
    echo "FAILED to start; see $LOG" >&2
    tail -n 20 "$LOG" >&2
    exit 1
fi
