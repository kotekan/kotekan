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
# LISTENING sockets only, deliberately. bufferRecv now sets SO_REUSEADDR unconditionally, so a
# fresh listener may rebind over the ~60 s of TIME_WAIT left by 60 sender connections; waiting
# for those to drain would add half a minute to every restart for no reason. Before that fix
# this loop broke immediately (TIME_WAIT is not LISTEN) and the new instance then died on
# EADDRINUSE -- which is how the first #59 gather restart failed.
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
print("  %d senders, %d bad frames, %d client drops (stale after %.0f s)" %
      (len(s), st.get("bad_frames", 0), st.get("client_drops", 0),
       st.get("stale_after_s", 5.0)))
# THE VERDICT IS OVER LIVE SENDERS ONLY -- the gather computes it, so this script and the
# broker cannot drift apart on what "spread" means. A stopped instance keeps its last window
# forever; folded in, one death reads as a four-digit alignment alarm (measured: 984 while the
# nine live instances sat at 1). The stale ones are named instead, because that is the half
# you can act on.
for chain, c in sorted((st.get("chains") or {}).items()):
    stale = c.get("stale") or []
    if not c.get("live"):
        print("  %-9s *** ALL SENDERS STALE: %s" % (chain, ",".join(stale)))
        continue
    flag = "" if c["spread"] <= 1 else "   *** MISALIGNED ***"
    print("  %-9s %2d live  win %d..%d (spread %d)  gaps %d%s%s"
          % (chain, c["live"], c["win_min"], c["win_max"], c["spread"], c["gaps"],
             ("  STALE: " + ",".join(stale)) if stale else "", flag))
    if not c.get("all_slots_present"):
        print("            (some records missing from a frame -- check the node's record rate)")
HEALTH
