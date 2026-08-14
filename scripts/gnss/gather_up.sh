#!/bin/bash
# Bring the GNSS TELEMETRY GATHER up on the host it is run from (task #59; cf06 in practice).
#
#   scripts/gnss/gather_up.sh [config] [log]
#
# The gather is the broker-side half of the frame-synced tracker transport: one bufferRecv for
# the whole fleet on :11060, and GnssTelemGather handing the frames to the PYTHON broker on
# 127.0.0.1:11061. It collates nothing -- every frame already carries an absolute window index,
# so grouping is the broker's exact integer match.
#
# ⚠️ ORDER MATTERS: GATHER FIRST, THEN THE NODES. Every sender logs one WARN per connection
# attempt and there are 60 of them (12 instances x 5 chains). The generated node configs set
# reconnect_time 30 to keep that survivable, but a fleet restarted into a missing gather still
# buries its own logs for as long as it takes to notice.
#
# Modelled on agg_up.sh, and for the same hard-won reasons: pick the binary by host (NFS holds
# two build trees), poll for the port instead of sleeping, and check the log for a FatalError
# rather than trusting that a process exists.
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

CFG=${1:-$K/config/generated/chord_gnss_gather.yaml}
LOG=${2:-/tmp/gnss_gather.log}
if [ ! -f "$CFG" ]; then
    echo "FAILED: no config at $CFG." >&2
    echo "  regenerate: python3 $K/config/gen_chord_gnss_config.py \\" >&2
    echo "      --base $K/config/base/live_config_20260730.json --node cx19 \\" >&2
    echo "      --gather-instance --out $CFG" >&2
    exit 1
fi

# Ports come from the CONFIG, not from this script: they are the wire contract with 60 senders
# and a second copy here is a second thing to drift.
RECV=$(python3 - "$CFG" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
print(c["telem_recv"]["listen_port"])
PY
)
SERVE=$(python3 - "$CFG" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
print(c["telem_gather"]["serve_port"])
PY
)
REST=$(python3 - "$CFG" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
print(c["rest_server"]["port"])
PY
)
echo "gather config: $(basename "$CFG") -- recv :$RECV, serve :$SERVE, rest :$REST"

pkill -9 -f "kotekan --config.*gather" 2>/dev/null || true
for _i in $(seq 1 30); do
    ss -ltn 2>/dev/null | grep -qE ":($RECV|$SERVE|$REST)\b" || break
    sleep 1
done
if ss -ltn 2>/dev/null | grep -qE ":($RECV|$SERVE|$REST)\b"; then
    echo "FAILED: :$RECV/:$SERVE/:$REST still bound after 30 s -- something else holds them" >&2
    ss -ltnp 2>/dev/null | grep -E ":($RECV|$SERVE|$REST)\b" >&2
    exit 1
fi

nohup setsid "$BIN" --config "$CFG" --bind-address "0.0.0.0:$REST" \
    > "$LOG" 2>&1 < /dev/null &
disown
sleep 5
pgrep -f "kotekan --config.*[g]ather" > /dev/null \
    || { echo "FAILED to start -- check $LOG" >&2; exit 1; }
if grep -qi "FatalError" "$LOG" 2>/dev/null; then
    echo "FAILED: the stage graph raised a FatalError and is shutting down:" >&2
    grep -i "FatalError" "$LOG" | head -3 >&2
    exit 1
fi
echo "gather up on $H using $BIN (log $LOG)"

# ---- POST-START HEALTH ------------------------------------------------------------------
# WHAT TO LOOK AT, and why this and not a frame rate: `spread` is max-min of the senders' most
# recent window index. A rate that looks right can still be every instance sitting on a
# DIFFERENT window, which is the entire defect class this transport exists to remove -- and it
# is invisible in any throughput number.
sleep 8
python3 - "$REST" <<'HEALTH' || true
import json, sys, urllib.request
try:
    with urllib.request.urlopen("http://localhost:%s/telem_gather/get_stats" % sys.argv[1],
                                timeout=5) as r:
        st = json.load(r)
except Exception as e:
    print("  health: SKIPPED (%s)" % e)
    raise SystemExit
s = st.get("senders") or []
if not s:
    print("  health: NO SENDERS yet. The gather is up and listening; the nodes are not")
    print("          sending. They need a config carrying --telem-host and a RESTART.")
    raise SystemExit
by_chain = {}
for row in s:
    chain = row["key"].split("/", 1)[0]
    by_chain.setdefault(chain, []).append(row)
print("  %d senders, %d bad frames, %d client drops" %
      (len(s), st.get("bad_frames", 0), st.get("client_drops", 0)))
for chain in sorted(by_chain):
    rows = by_chain[chain]
    wins = [r["last_win"] for r in rows]
    gaps = sum(r["gaps"] for r in rows)
    stale = [r["key"] for r in rows if r["age_s"] > 5.0]
    flag = "" if max(wins) - min(wins) <= 1 else "   *** MISALIGNED ***"
    print("  %-9s %2d inst  win %d..%d (spread %d)  gaps %d%s%s"
          % (chain, len(rows), min(wins), max(wins), max(wins) - min(wins), gaps,
             ("  stale: " + ",".join(stale)) if stale else "", flag))
HEALTH
