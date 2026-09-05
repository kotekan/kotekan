#!/bin/bash
# Bring the GNSS BEAM-CUBE ARCHIVER up on the host it is run from (cf06 in practice).
#
#   scripts/gnss/cubearch_up.sh [config] [log]
#
# This is the far side of the beam-cube push leg: one bufferRecv on :11070 for the whole fleet's
# completed ~1 s (subband x element) windows, and a rawFileWrite that lands them on cs00. It is
# the ARCHIVE -- the thing per-element/per-frequency beam maps are built from -- and it is a
# SEPARATE process from both the gather and the search aggregator on purpose: a gather restart
# wipes every standing trim and the aggregator is bounced routinely, and neither should be able
# to punch a hole in a record.
#
# ⚠️ ORDER MATTERS: ARCHIVER FIRST, THEN THE NODES. There are NINETY senders (15 assemblers x 6
# nodes), each logging a WARN per connection attempt at reconnect_time 30. A fleet restarted
# into a missing archiver buries its own logs, and the windows sent meanwhile are gone -- the
# senders drop rather than block, which is what keeps an archive from ever stalling a tracker.
#
# ⚠️ NOTHING RESTARTS THIS ON A REBOOT. cf06 takes an unattended kernel reboot at 03:03 whenever
# one lands (2026-08-12, 08-19, 09-05) and every process here is a manual nohup, so the archive
# simply stops with the fleet looking perfectly healthy. Until a boot unit exists, a gap that
# starts at 03:0x is this, not the instrument.
#
# Modelled on gather_up.sh, and for the same hard-won reasons: pick the binary by host (NFS
# holds two build trees), poll for the port instead of sleeping, and check the log for a
# FatalError rather than trusting that a process exists.
set -u
K=/home/kvand/gnss/kotekan
H=$(hostname -s)
case "$H" in
cx*) DEF=$K/build/kotekan/kotekan ;;        # node host: the DPDK build
*)   DEF=$K/build_nodpdk/kotekan/kotekan ;; # cf06 and anything else: DPDK-free
esac
BIN=${GNSS_BIN:-$DEF}
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

# ⚠️ THE ARCHIVE DIRECTORY MUST EXIST AND BE WRITABLE BEFORE THE STAGE OPENS A FILE.
# rawFileWrite takes base_dir straight from the config and dies on a missing path, and on cs00
# that is an NFS mount which can be absent without the host noticing.
if ! mkdir -p "$DIR" 2>/dev/null || ! touch "$DIR/.wtest" 2>/dev/null; then
    echo "FAILED: cannot write $DIR (mounted? owned by you?)." >&2
    exit 1
fi
rm -f "$DIR/.wtest"
AVAIL=$(df -BG --output=avail "$DIR" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$AVAIL" ] && [ "$AVAIL" -lt 500 ]; then
    echo "  ⚠️ only ${AVAIL} GB free on $DIR -- this leg writes ~780 GB/day RAW." >&2
fi

# ⚠️ THE FLEET'S FRAME SIZE MUST MATCH THIS BUFFER, AND A MISMATCH IS SILENT WHERE IT MATTERS.
# bufferRecv compares the sender's frame_size against its own and CLOSES the connection -- so a
# disagreement never corrupts data, it just delivers none, from a listener that is up, on a port
# that is open, with senders that keep reconnecting. It cost exactly one afternoon here: the
# frame grew by 8 bytes (the maxima that make the archive self-describing) and the node configs
# had not been regenerated. Check it BEFORE anything writes a file.
MISMATCH=0
for f in "$K"/config/generated/chord_gnss_cx*_multi.yaml; do
    [ -f "$f" ] || continue
    NF=$(grep -m1 -oP '^\s+frame_size:\s*\K[0-9]+' <(grep -A6 "cube_buf:" "$f") 2>/dev/null)
    if [ -n "$NF" ] && [ "$NF" != "$FRAME" ]; then
        echo "  ⚠️ $(basename "$f") sends ${NF} B, this archiver holds ${FRAME} B" >&2
        MISMATCH=1
    fi
done
if [ "$MISMATCH" = 1 ]; then
    echo "FAILED: sender/receiver frame_size disagree. bufferRecv would accept every" >&2
    echo "connection and deliver NOTHING. Regenerate both from the same tree:" >&2
    echo "  python3 $K/scripts/gnss/gen_fleet.py $K/config/gnss_fleet_chord.yaml" >&2
    exit 1
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
