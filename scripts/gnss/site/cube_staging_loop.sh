#!/bin/bash
# Beam-cube local staging (cf06): the archiver writes its raw files to local disk (STAGE)
# instead of cs00, whose NFS writes over cf06's one 1 GbE link fell behind the fleet's
# inflow (bufferRecv dropped ~17 % of the frames, 2026-10-30). Each pass, in one process
# so l0/manifest.json keeps one writer:
#   1. fold every closed STAGE file into L0 on cs00;
#   2. ship each STAGE file the manifest records to <ROOT>/raw_zst/<name>.zst (zstd -1,
#      ~4.8x; tested before the local copy is deleted). The newest file by mtime (still
#      being written) and the highest-numbered one stay: continue_numbering resumes past
#      the highest number in STAGE, and an emptied STAGE would restart at 0000000 under
#      names the manifest already holds;
#   3. fold up to BACKLOG files of the cs00 raw backlog (0 = none).
# The archiver's file stem must be one no other raw directory has used
# (gen_chord_gnss_config.py --cube-archive-file-name): the manifest keys by basename.
# Stop = `systemctl --user stop`: TERM is taken between steps, after the running fold.
#   cube_staging_loop.sh PY TOOL STAGE ROOT [BACKLOG] [PAUSE_S]
set -u
PY=$1; TOOL=$2; STAGE=$3; ROOT=$4; BACKLOG=${5:-100}; PAUSE=${6:-30}
ZST=$ROOT/raw_zst
LOW_GB=300
stopping=0
trap 'stopping=1' TERM
mkdir -p "$ZST"
log() { echo "== $(date -u +%FT%TZ) $*"; }

# The STAGE files the manifest records, less the newest by mtime and the highest-numbered.
shippable() {
    "$PY" - "$STAGE" "$ROOT/l0/manifest.json" <<'EOF'
import glob, json, os, sys
stage, manifest = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(stage, "*.raw")))
if len(files) < 2:
    sys.exit(0)
keep = {max(files, key=os.path.getmtime), files[-1]}
done = json.load(open(manifest)) if os.path.exists(manifest) else {}
for f in files:
    if f not in keep and os.path.basename(f) in done:
        print(f)
EOF
}

ship() {
    local n=0 f b out
    for f in $(shippable); do
        b=$(basename "$f")
        out=$ZST/$b.zst
        if [ -e "$out" ]; then
            log "SHIP REFUSED $b: $out exists (a renumbered archiver?) -- local copy kept"
            continue
        fi
        if zstd -1 -q -T2 -o "$out.part" "$f" && zstd -t -q "$out.part" && mv "$out.part" "$out"; then
            rm -f "$f"
            n=$((n + 1))
        else
            log "SHIP FAILED $b -- local copy kept"
            rm -f "$out.part"
        fi
        [ "$stopping" = 0 ] || break
    done
    log "shipped $n file(s); $(ls "$STAGE"/*.raw 2>/dev/null | wc -l) in $STAGE, $(df -BG --output=avail "$STAGE" | tail -1 | tr -d ' G') GB free"
}

while [ "$stopping" = 0 ]; do
    if [ "$(ls "$STAGE"/*.raw 2>/dev/null | wc -l)" -ge 2 ]; then
        log "compact $STAGE"
        "$PY" -u "$TOOL" compact --raw "$STAGE" --out "$ROOT" --newest-first --readers 4 2>&1
    fi
    [ "$stopping" = 0 ] || break
    ship
    free_gb=$(df -BG --output=avail "$STAGE" | tail -1 | tr -d ' G')
    if [ "$free_gb" -lt "$LOW_GB" ]; then
        log "LOW DISK: $free_gb GB free in $STAGE (unfolded files are not shipped)"
    fi
    [ "$stopping" = 0 ] || break
    if [ "$BACKLOG" -gt 0 ]; then
        log "compact backlog $ROOT/raw (up to $BACKLOG)"
        "$PY" -u "$TOOL" compact --raw "$ROOT/raw" --out "$ROOT" --newest-first \
            --max-files "$BACKLOG" --readers 2 2>&1
    fi
    [ "$stopping" = 0 ] || break
    sleep "$PAUSE" & wait $! || true
done
log "staging loop stopped on SIGTERM after a complete step"
