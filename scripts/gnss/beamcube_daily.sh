#!/bin/bash
# Build one day's beam-cube master and re-export the whole viewer directory.
#
#   beamcube_daily.sh              # yesterday UTC (what the compactor loop calls)
#   beamcube_daily.sh 20260907     # a named day
#   BEAMCUBE_FORCE=1 beamcube_daily.sh 20260907    # rebuild a master that already exists
#
# WHY THIS EXISTS: until 2026-09-10 these two commands were hand-run, and the failure mode is
# invisible -- the viewer keeps serving its last export, which is indistinguishable from a
# healthy viewer. Four days of data sat unpublished that way. The compactor loop already knows
# the exact moment a day's rungs are complete, so it is the right place to call this from.
#
# ⚠️ THE EXPORT TAKES EVERY MASTER, ALWAYS. `export` rewrites index.json from its arguments
# alone, so passing only the new day would drop every other day off the page. The glob below is
# the whole point of having a script: it cannot be got wrong by hand.
set -u
K=/home/kvand/gnss/kotekan
PY=${GNSS_PY:-/home/kvand/gnss/venv/bin/python}     # needs h5py + healpy: venv, NOT venv-ft
TOOL=$K/python/scripts/gnss/gnss_beam_cube.py
BEAM=${BEAMCUBE_DIR:-/home/kvand/gnss/fixtures/beamcube}
ROOT=${CUBE_ROOT:-/mnt/cs00/data/kvand/gnss_cube}
NSIDE=${BEAMCUBE_NSIDE:-64}
WEB_NSIDE=${BEAMCUBE_WEB_NSIDE:-32}
DAY=${1:-$(date -u -d yesterday +%Y%m%d)}
LOCK=${BEAMCUBE_LOCK:-/tmp/gnss_beamcube_daily.lock}

say() { echo "beamcube_daily $(date -u +%FT%TZ) $*"; }

# One at a time. A hand-run during the nightly pass would have two builders writing one master.
exec 9>"$LOCK" || exit 1
if ! flock -n 9; then
    say "another run holds $LOCK; nothing to do"
    exit 0
fi

# ⚠️ ONE MASTER IS ONE POINTING, and the pointing is a DECLARED epoch table that can change
# under us. Rather than bake p0_dec40p73 in, find which pointing actually holds this day and
# refuse if more than one does -- summing two pointings is the error this guards.
mapfile -t POINTINGS < <(
    for p in "$ROOT"/rung12/*/; do
        compgen -G "$p"'*/'"$DAY"'.h5' >/dev/null 2>&1 && basename "$p"
    done)
if [ "${#POINTINGS[@]}" -eq 0 ]; then
    say "no rung12 for $DAY under $ROOT/rung12 -- the compactor cuts rungs after 00:20 UTC for"
    say "the day BEFORE; today exists at L0 only (and reading it live needs a cp snapshot)."
    exit 2
elif [ "${#POINTINGS[@]}" -gt 1 ]; then
    say "REFUSING: $DAY spans ${#POINTINGS[@]} pointings (${POINTINGS[*]}); one master is one"
    say "pointing. Build them separately with an explicit --archive."
    exit 3
fi
ARCHIVE=$ROOT/rung12/${POINTINGS[0]}

MASTER=$BEAM/cube_${DAY}_nside${NSIDE}.npz
if [ -s "$MASTER" ] && [ -z "${BEAMCUBE_FORCE:-}" ]; then
    say "master exists, skipping build: $MASTER (BEAMCUBE_FORCE=1 to rebuild)"
else
    say "build $DAY nside $NSIDE from ${POINTINGS[0]}"
    # Build into a staging dir: a half-written master that the export then reads is worse than
    # no master, and the build writes its .npz incrementally.
    STAGE=$BEAM/.staging_$DAY.$$
    mkdir -p "$STAGE" || exit 1
    trap 'rm -rf "$STAGE"' EXIT
    if ! nice -n 19 "$PY" -u "$TOOL" build --source l0 --archive "$ARCHIVE" \
            --days "$DAY" --nside "$NSIDE" --outdir "$STAGE"; then
        say "BUILD FAILED for $DAY -- viewer left as it was"
        exit 4
    fi
    if [ ! -s "$STAGE/cube_${DAY}_nside${NSIDE}.npz" ]; then
        say "BUILD produced no master for $DAY -- viewer left as it was"
        exit 4
    fi
    mv -f "$STAGE/cube_${DAY}_nside${NSIDE}.npz" "$MASTER"
    say "master $(du -h "$MASTER" | cut -f1) -> $MASTER"
fi

shopt -s nullglob
MASTERS=("$BEAM"/cube_*_nside${NSIDE}.npz)
shopt -u nullglob
if [ "${#MASTERS[@]}" -eq 0 ]; then
    say "no masters to export"
    exit 5
fi
say "export ${#MASTERS[@]} day(s) -> $BEAM/web at nside $WEB_NSIDE"
if ! nice -n 19 "$PY" -u "$TOOL" export "${MASTERS[@]}" --outdir "$BEAM/web" --nside "$WEB_NSIDE"; then
    say "EXPORT FAILED -- index.json may name days whose .bin files did not update"
    exit 6
fi
say "done: $(grep -o '"day": "[0-9]*"' "$BEAM/web/index.json" | grep -o '[0-9]*' | tr '\n' ' ')"
