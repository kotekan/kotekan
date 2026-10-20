#!/bin/bash
# Bring the GNSS beam-cube archiver up on the host it is run from (cf06 in practice).
#
#   scripts/gnss/site/cubearch_up.sh [config] [log]
#   scripts/gnss/site/cubearch_up.sh --check [config]   # the pre-flight only (the unit's ExecStartPre)
#
# The far side of the beam-cube push leg: one bufferRecv on :11070 for the whole fleet's
# completed ~1 s (subband x element) windows, and a rawFileWrite that lands them on cs00. This
# archive is what the per-element, per-frequency beam maps are built from. It is a separate
# process from the gather and the search aggregator so that restarting either cannot punch a
# hole in it.
#
# Start the archiver before the nodes. There are ~90 senders (one per chain instance, ~15 per
# node), each logging a WARN per connection attempt every 30 s (reconnect_time), and windows
# sent while the archiver is down are lost: the senders drop rather than block, so the archive
# can never stall a tracker.
#
# A reboot stops an archiver started by hand. cf06 takes unattended kernel reboots at ~03:03,
# and a gap that starts then is this, not the instrument. On cf06 the archiver is
# gnss-cubearch.service (gnss-cube.target, enabled there; scripts/gnss/site/systemd), whose
# ExecStartPre is this script's --check; restart it with
# `systemctl --user restart gnss-cubearch.service`. Launch by hand only with that unit stopped,
# for a test config too: the kill -9 below matches any kotekan on a *cubearch* config, the
# unit's included, systemd reads that as the unit's archiver failing, and the unit restarts it
# 30 s later beside the one started here.
#
# Run the DPDK-free binary (NFS holds two build trees), poll for the port instead of sleeping,
# and check the log for a FatalError rather than trusting that a process exists.
set -u
CHECK=0
[ "${1:-}" = "--check" ] && { CHECK=1; shift; }
K=/home/kvand/gnss/kotekan
KOTEKAN_SRC=${KOTEKAN_SRC:-$K}   # the kotekan tree, for its build
H=$(hostname -s)
BIN=${GNSS_BIN:-$KOTEKAN_SRC/build_nodpdk/kotekan/kotekan}   # cf06 has no DPDK
if [ ! -x "$BIN" ]; then
    echo "FAILED: no kotekan binary at $BIN (host $H). Build it, or set GNSS_BIN." >&2
    exit 1
fi

CFG=${1:-$K/config/generated/chord_gnss_cubearch.yaml}
LOG=${2:-/tmp/gnss_cubearch.log}
if [ ! -f "$CFG" ]; then
    echo "FAILED: no config at $CFG." >&2
    echo "  regenerate: python3 $K/config/gen_chord_gnss_config.py \\" >&2
    echo "      --base $K/config/base/live_config_20260831.json --node cx19 \\" >&2
    echo "      --cube-archive-instance --rest-port 12052 \\" >&2
    echo "      --out $CFG" >&2
    exit 1
fi

read -r PORT DIR FRAME <<EOF
$(python3 - "$CFG" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
print(c["cube_recv"]["listen_port"], c["cube_write"]["base_dir"], c["cube_buf"]["frame_size"])
PY
)
EOF
echo "cube archiver: $(basename "$CFG") -- recv :$PORT, frame $FRAME B, writing $DIR"

# The archive directory must exist and be writable before the stage opens a file: rawFileWrite
# takes base_dir from the config and exits on a missing path, and on cs00 that is an NFS mount
# that can be absent without the host noticing.
if ! mkdir -p "$DIR" 2>/dev/null || ! touch "$DIR/.wtest" 2>/dev/null; then
    echo "FAILED: cannot write $DIR (mounted? owned by you?)." >&2
    exit 1
fi
rm -f "$DIR/.wtest"
AVAIL=$(df -BG --output=avail "$DIR" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$AVAIL" ] && [ "$AVAIL" -lt 500 ]; then
    echo "  warning: only ${AVAIL} GB free on $DIR -- this leg writes ~780 GB/day raw." >&2
fi

# The fleet's cube frame_size must match this buffer's. bufferRecv closes a connection whose
# frame_size differs, so a mismatch corrupts nothing but delivers nothing, from a listener that
# is up, on an open port, with senders that keep reconnecting. Check it before anything writes.
MISMATCH=0
for f in "$K"/config/generated/chord_gnss_cx*_multi.yaml; do
    [ -f "$f" ] || continue
    NF=$(grep -m1 -oP '^\s+frame_size:\s*\K[0-9]+' <(grep -A6 "cube_buf:" "$f") 2>/dev/null)
    if [ -n "$NF" ] && [ "$NF" != "$FRAME" ]; then
        echo "  $(basename "$f") sends ${NF} B, this archiver holds ${FRAME} B" >&2
        MISMATCH=1
    fi
done
if [ "$MISMATCH" = 1 ]; then
    echo "FAILED: sender/receiver frame_size disagree. bufferRecv would accept every" >&2
    echo "connection and deliver nothing. Regenerate both from one tree: the archiver with config/gen_chord_gnss_config.py --cube-archive-instance, the nodes with" >&2
    echo "  python3 $K/scripts/gnss/gen_fleet.py $K/config/gnss_fleet_chord.yaml" >&2
    exit 1
fi
if [ "$CHECK" = 1 ]; then
    echo "cube archiver pre-flight OK on $H: $(basename "$CFG"), recv :$PORT, frame $FRAME B, $DIR writable"
    exit 0
fi

pkill -9 -f "[k]otekan -c .*cubearch" 2>/dev/null || true
for _i in $(seq 1 30); do
    ss -ltn 2>/dev/null | grep -qE ":${PORT}\b" || break
    sleep 1
done

nohup setsid "$BIN" -c "$CFG" > "$LOG" 2>&1 < /dev/null &
disown
for _i in $(seq 1 30); do
    ss -ltn 2>/dev/null | grep -qE ":${PORT}\b" && break
    sleep 1
done
if grep -qa "FatalError\|FATAL" "$LOG"; then
    echo "FAILED -- fatal in $LOG:" >&2
    grep -a -m3 "FatalError\|FATAL" "$LOG" >&2
    exit 1
fi
if ! ss -ltn 2>/dev/null | grep -qE ":${PORT}\b"; then
    echo "FAILED: not listening on $PORT after 30 s -- last lines of $LOG:" >&2
    tail -15 "$LOG" >&2
    exit 1
fi
echo "cube archiver up on $H using $BIN (log $LOG)"
echo "  read it back:  python3 $K/python/scripts/gnss/gnss_cube_read.py ls $DIR"
