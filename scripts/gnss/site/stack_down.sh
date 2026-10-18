#!/bin/bash
# Stop the cf06 beam-cube leg (static viewer :8877, compactor loop, archiver), in dependency
# order, archiving its logs first.
#
# The live stack is not stopped here: the broker, gather, aggregator, obs writers, rail recorder
# and live viewer run under systemd on the gnss VM (gnss-stack.target); stop them there
# (docs/CHORD_GNSS_RUNBOOK.md). This covers only the cube leg on cf06, and its log archiving
# covers only cf06's own /tmp.
#
# The cube leg has three `*_up.sh` scripts (beamview_up.sh, cubecompact_up.sh, cubearch_up.sh)
# and this is their way down. The table below names every process, including the static viewer
# (`python3 -m http.server 8877`), which matches none of the obvious patterns.
#
# /tmp is cleared on boot (`D /tmp` in /usr/lib/tmpfiles.d/tmp.conf) and cf06 takes unattended
# reboots near 03:03 (about weekly), so a log not archived before a teardown is gone at the next
# boot. Hence the archive runs before the stop.
#
# Consumers stop before producers: the viewer serves what the compactor publishes, and the
# compactor folds what the archiver writes. Down is the reverse of up.
#
# usage:  stack_down.sh              # archive live logs, then stop everything
#         stack_down.sh --no-archive # stop only
#         stack_down.sh --list       # show what would be stopped, change nothing
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
viewer-static|[h]ttp.server 8877
cube-compactor|[c]ubecompact_loop.sh
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
    echo "archiving this host's /tmp/gnss_*.log to $LOGDIR (skipping any untouched for ${STALE_H}h)..."
    pids=""
    for f in /tmp/gnss_*.log; do
        [ -s "$f" ] || continue
        if [ -n "$(find "$f" -mmin +$((STALE_H*60)) 2>/dev/null)" ]; then
            echo "  skip  $(basename "$f") ($(du -h "$f" | cut -f1), stale)"
            continue
        fi
        echo "  keep  $(basename "$f") ($(du -h "$f" | cut -f1))"
        # -1: fast, and the ratio hardly differs on text.
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
    # TERM first: these write files, and KILL truncates whatever was mid-write. The compactor
    # loop exits on TERM only after any in-flight pass, which can outlast the 10 s wait; KILL
    # then stops the loop alone and the pass's own process runs on to the end.
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
        [ -n "$(pgrep -f "$pat")" ] && { echo "  $name still running after KILL" >&2; rc=1; }
        printf '%-16s killed (was %s)\n' "$name" "$found"
    else
        printf '%-16s stopped (was %s)\n' "$name" "$found"
    fi
done <<< "$COMPONENTS"

echo
echo "remaining cube-leg processes: $(pgrep -fc '[h]ttp.server 8877|[c]ubecompact_loop|[k]otekan .*chord_gnss_cubearch' 2>/dev/null || echo 0)"
exit $rc
