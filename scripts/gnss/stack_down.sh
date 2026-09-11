#!/bin/bash
# Stop the cf06 GNSS stack, in dependency order, archiving its logs first.
#
# WHY THIS EXISTS: there were six `*_up.sh` scripts and NO way down, so every teardown was
# hand-assembled from `ps` output at the moment it was needed -- which is how a component gets
# missed. Two were missed the first time this was written by hand: the static viewer
# (`python3 -m http.server 8877`, which matches none of the obvious patterns) and the
# cube-compactor loop. They are both in the table below now.
#
# ⚠️ /tmp IS CLEARED ON BOOT (`D /tmp` in /usr/lib/tmpfiles.d/tmp.conf) and cf06 reboots weekly
# at ~03:03. A stack log that is not archived before a teardown is gone at the next boot, not at
# the next restart -- which is why this archives BEFORE it stops, and why that is not optional
# for the live logs.
#
# ORDER IS CONSUMERS BEFORE PRODUCERS. Stopping the gather first leaves the broker and the
# aggregator logging connection failures into the logs we are trying to archive, and the obs
# writers writing rows against a broker that is going away. Down is the reverse of up.
#
# usage:  stack_down.sh              # archive live logs, then stop everything
#         stack_down.sh --no-archive # stop only
#         stack_down.sh --list       # show what WOULD be stopped, change nothing
set -u
LOGDIR=/home/kvand/gnss/logs
STALE_H=6            # logs untouched this long are history, not this run -- skipped
ARCHIVE=1; LIST=0
for a in "$@"; do
    case "$a" in
        --no-archive) ARCHIVE=0 ;;
        --list)       LIST=1 ;;
        *) echo "unknown option: $a" >&2; exit 2 ;;
    esac
done

# name | pattern (bracket trick: must not match this script's own command line)
COMPONENTS="
obs-writers|[g]nss_observables.py
viewer-livebeam|[l]ivebeam_server.py
viewer-static|[h]ttp.server 8877
cube-compactor|[c]ubecompact_loop.sh
broker|[b]roker_multi.py
aggregator|[k]otekan .*chord_gnss_agg
gather|[k]otekan .*chord_gnss_gather
cube-archiver|[k]otekan .*chord_gnss_cubearch
"

if [ "$LIST" = 1 ]; then
    printf '%-16s %s\n' COMPONENT PIDS
    while IFS='|' read -r name pat; do
        [ -z "$name" ] && continue
        printf '%-16s %s\n' "$name" "$(pgrep -f "$pat" | tr '\n' ' ')"
    done <<< "$COMPONENTS"
    exit 0
fi

if [ "$ARCHIVE" = 1 ]; then
    mkdir -p "$LOGDIR"
    T=$(date -u +%Y%m%d_%H%M%S)
    echo "archiving live stack logs to $LOGDIR (skipping any untouched for ${STALE_H}h)..."
    pids=""
    for f in /tmp/gnss_*.log; do
        [ -s "$f" ] || continue
        if [ -n "$(find "$f" -mmin +$((STALE_H*60)) 2>/dev/null)" ]; then
            echo "  skip  $(basename "$f") ($(du -h "$f" | cut -f1), stale)"
            continue
        fi
        echo "  keep  $(basename "$f") ($(du -h "$f" | cut -f1))"
        # -1: this runs while someone is waiting to go home; the ratio hardly differs on text.
        nice -n 10 gzip -1 -c "$f" > "$LOGDIR/$(basename "$f" .log)_$T.log.gz" &
        pids="$pids $!"
    done
    for p in $pids; do wait "$p"; done
    echo "archive done."
fi

rc=0
while IFS='|' read -r name pat; do
    [ -z "$name" ] && continue
    found=$(pgrep -f "$pat" | tr '\n' ' ')
    if [ -z "$found" ]; then
        printf '%-16s not running\n' "$name"
        continue
    fi
    # TERM first, always. These write files (records, cubes, jsonl rows); -9 on the archiver or
    # a writer truncates whatever was mid-write, and every one of them exits cleanly on TERM.
    pkill -TERM -f "$pat" 2>/dev/null
    for _ in $(seq 1 20); do
        sleep 0.5
        [ -z "$(pgrep -f "$pat")" ] && break
    done
    left=$(pgrep -f "$pat" | tr '\n' ' ')
    if [ -n "$left" ]; then
        echo "  $name did not exit on TERM after 10 s (pids $left) -- sending KILL" >&2
        pkill -KILL -f "$pat" 2>/dev/null
        sleep 1
        [ -n "$(pgrep -f "$pat")" ] && { echo "  $name STILL RUNNING" >&2; rc=1; }
        printf '%-16s killed (was %s)\n' "$name" "$found"
    else
        printf '%-16s stopped (was %s)\n' "$name" "$found"
    fi
done <<< "$COMPONENTS"

echo
echo "remaining stack processes: $(pgrep -fc '[g]nss_observables|[l]ivebeam_server|[b]roker_multi|[c]ubecompact_loop|[k]otekan .*chord_gnss' 2>/dev/null || echo 0)"
exit $rc
