#!/bin/bash
# Beam-cube staging: the archiver writes its raw files to STAGE (cf06's local disk, whose
# NFS writes over one 1 GbE link fell behind the fleet's inflow, 2026-10-30; or a directory
# on cs00 from the gnss VM) and this loop folds them into L0 and compresses them. Each pass,
# with one process writing l0/manifest.json:
#   1. fold every closed STAGE file into L0 on cs00 (READERS read ahead, a manifest save
#      every BATCH files; memory grows with both), while
#   2. a second process ships each STAGE file the manifest already records to
#      <ROOT>/raw_zst/<name>.zst (zstd -1, ~4.4x; tested before the raw copy is deleted).
#      The two touch disjoint files, and in sequence they take about as long as a file takes
#      to arrive. The newest file by mtime (still being written) and the highest-numbered one
#      stay: continue_numbering resumes past the highest number in STAGE, and an emptied
#      STAGE would restart at 0000000 under names the manifest already holds;
#   3. fold up to BACKLOG files of the cs00 raw backlog (0 = none).
# The archiver's file stem must be one no other raw directory has used
# (gen_chord_gnss_config.py --cube-archive-file-name): the manifest keys by basename.
# Stop = `systemctl --user stop`: TERM is taken between steps, after the running fold.
#   cube_staging_loop.sh PY TOOL STAGE ROOT [BACKLOG] [PAUSE_S] [READERS] [BATCH]
set -u
PY=$1; TOOL=$2; STAGE=$3; ROOT=$4; BACKLOG=${5:-100}; PAUSE=${6:-30}; READERS=${7:-4}
BATCH=${8:-12}
ZST=$ROOT/raw_zst
LOW_GB=300
stopping=0
shipper=
trap 'stopping=1; [ -z "$shipper" ] || kill -TERM "$shipper" 2>/dev/null' TERM
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
    (trap 'stopping=1' TERM; ship) &
    shipper=$!
    if [ "$(ls "$STAGE"/*.raw 2>/dev/null | wc -l)" -ge 2 ]; then
        log "compact $STAGE"
        "$PY" -u "$TOOL" compact --raw "$STAGE" --out "$ROOT" --newest-first \
            --readers "$READERS" --batch "$BATCH" 2>&1
    fi
    # A TERM interrupts the first wait; the second lets the shipper finish its file.
    wait "$shipper"
    wait "$shipper" 2>/dev/null
    shipper=
    [ "$stopping" = 0 ] || break
    free_gb=$(df -BG --output=avail "$STAGE" | tail -1 | tr -d ' G')
    if [ "$free_gb" -lt "$LOW_GB" ]; then
        log "LOW DISK: $free_gb GB free in $STAGE (unfolded files are not shipped)"
    fi
    [ "$stopping" = 0 ] || break
    if [ "$BACKLOG" -gt 0 ]; then
        log "compact backlog $ROOT/raw (up to $BACKLOG)"
        "$PY" -u "$TOOL" compact --raw "$ROOT/raw" --out "$ROOT" --newest-first \
            --max-files "$BACKLOG" --readers 2 --batch "$BATCH" 2>&1
    fi
    [ "$stopping" = 0 ] || break
    sleep "$PAUSE" & wait $! || true
done
log "staging loop stopped on SIGTERM after a complete step"
