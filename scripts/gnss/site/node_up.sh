#!/bin/bash
# Bring a CHORD GNSS node's kotekan up (or down, or check it).
#
# The node service is a transient systemd unit: `systemd-run --unit=gnss-node` creates it in
# /run/systemd/transient/, and stopping it deletes it. After `systemctl stop gnss-node`,
# `systemctl start gnss-node` fails with "Unit not found"; only the full systemd-run line below
# brings it back.
#
# The exec path must be absolute: systemd-run does not resolve a relative one against
# --working-directory, and the unit starts and immediately dies.
#
# usage:  node_up.sh <node> [up|down|restart|status]      default: up
#         node_up.sh cx51
#         node_up.sh cx51 status
#         for n in cx19 cx27 cx42 cx43 cx44 cx51; do scripts/gnss/site/node_up.sh $n restart; done
#
# `up` and `restart` both stop the unit and start it inside one ssh session. Use either rather
# than `down` then `up`: sudo's credential cache is per-TTY and `ssh -t` allocates a fresh one
# each call, so separate down and up loops prompt for the password twice per node.
#
#   env:  GNSS_BIN  binary            (default $KOTEKAN_SRC/build/kotekan/kotekan)
#         KOTEKAN_SRC  kotekan tree   (default: found as kotekan_src.sh says)
#         GNSS_CFG  config            (default: the config the fleet manifest owns for <node>)
#         GNSS_LOG  log file          (default /tmp/gnss_node.log)
#         GNSS_ENV  extra environment for kotekan, "VAR=value [VAR=value ...]" (this start only)
#         GNSS_EOP_MIN_H       hours of baked EOP table required ahead of now (default 12)
#         GNSS_SKIP_PREFLIGHT  1 skips the EOP-table, F-engine controller and interface checks
#         GNSS_SKIP_IFACE      1 skips the interface check only
#
# Needs an interactive sudo on the target, so it uses `ssh -t` and prompts per node (`status`
# needs neither).
set -u
# shellcheck source=kotekan_src.sh
. "$(dirname "${BASH_SOURCE[0]}")/kotekan_src.sh"
# shellcheck source=iface_preflight.sh
. "$(dirname "${BASH_SOURCE[0]}")/iface_preflight.sh"
K=$GNSS_CHECKOUT
KOTEKAN_SRC=$(gnss_kotekan_src "$K")   # the kotekan tree: build/ and lib/stages/gnss
export KOTEKAN_SRC                      # for gen_fleet.py, which renders kotekan's templates
BIN=${GNSS_BIN:-$KOTEKAN_SRC/build/kotekan/kotekan}
N=${1:?usage: node_up.sh <node> [up|down|restart|status]}
ACT=${2:-up}
# GNSS_CFG / GNSS_LOG run a node on a one-off config (profiling, a test) without editing
# anything:
#
#   GNSS_CFG=$K/config/generated/<one-off>.yaml GNSS_LOG=/tmp/gnss_prof.log \
#       scripts/gnss/site/node_up.sh cx19
#
# Give a profiling run its own log: it writes an INFO block per GPU frame per GPU, which buries
# /tmp/gnss_node.log.
# The default is the config the fleet manifest owns for this node (gen_fleet.py --print-path):
# its name carries the manifest's suffix, so it cannot be derived from the node name.
if [ -z "${GNSS_CFG:-}" ] && [ -f "$K/config/gnss_fleet_chord.yaml" ]; then
    GNSS_CFG=$(python3 "$K/scripts/gnss/gen_fleet.py" "$K/config/gnss_fleet_chord.yaml" \
                       --node "$N" --print-path 2>/dev/null || true)
fi
# No fallback to a bare chord_gnss_<node>.yaml: such a file can start cleanly with the wrong
# chains and come up green. Empty is fine here: only the start path needs a config, and
# `down`/`status` must work for a node the manifest does not list. preflight() refuses it.
CFG=${GNSS_CFG:-}
LOG=${GNSS_LOG:-/tmp/gnss_node.log}
# GNSS_ENV becomes systemd-run --setenv flags on the transient unit, so it lasts for this start
# (and the unit's own on-failure restarts) and a plain node_up drops it. Entries are restricted to
# a quoting-safe character set because they travel through ssh and sudo inside REMOTE_UP.
SETENV=""
for kv in ${GNSS_ENV:-}; do
    case "$kv" in
        [A-Za-z_]*=*) case "$kv" in *[!A-Za-z0-9_.,:/=+-]*) kv="" ;; esac ;;
        *) kv="" ;;
    esac
    if [ -z "$kv" ]; then
        echo "FAILED: GNSS_ENV entries must be VAR=value using only letters, digits and _.,:/=+-" >&2
        exit 1
    fi
    SETENV="$SETENV --setenv=$kv"
done

# --- shared by `up` and `restart` -------------------------------------------------------------
# One preflight and one systemd-run line, so the two actions cannot start a node differently.
preflight() {
    # Fail loudly on a missing config or binary. The systemd-run below is guarded by
    # `test -r '$CFG'`; a failing test creates no unit, `systemctl status` then says
    # "could not be found", and the remote side can report only "config unreadable". Checking
    # here names the bad path. These are local checks of the NFS-shared tree the node reads.
    if [ -z "$CFG" ]; then
        echo "FAILED: no config for $N -- the fleet manifest does not name one." >&2
        echo "  config/gnss_fleet_chord.yaml is the authority; run it directly to see why:" >&2
        echo "    python3 $K/scripts/gnss/gen_fleet.py $K/config/gnss_fleet_chord.yaml \\" >&2
        echo "            --node $N --print-path" >&2
        echo "  Either $N is not in the manifest, or gen_fleet.py failed." >&2
        echo "  To run a one-off config deliberately, set GNSS_CFG=<absolute path>." >&2
        exit 1
    fi
    if [ ! -r "$CFG" ]; then
        echo "FAILED: config not readable: $CFG" >&2
        echo "  (a relative or elided path? GNSS_CFG must be absolute)" >&2
        exit 1
    fi
    if [ ! -x "$BIN" ]; then
        echo "FAILED: binary not executable: $BIN" >&2
        exit 1
    fi
    if [ "${GNSS_SKIP_PREFLIGHT:-0}" != 1 ]; then
        # The baked EOP table rolls. With fatal_eop_out_of_range set, a node started from a
        # config whose table ends near now exits on "Requesting EOP later than in table". A
        # live /earth_rotation_data push does not survive a restart; only the baked table does.
        # Refuse below GNSS_EOP_MIN_H hours of headroom (default 12).
        _eop_h=$(python3 - "$CFG" <<'PYEOP'
import re, sys, time
ts = [int(x) for x in re.findall(r"t_inst_ns:\s*(\d+)", open(sys.argv[1]).read())]
print("%.1f" % ((max(ts) / 1e9 - time.time()) / 3600) if ts else "nan")
PYEOP
)
        _min_h=${GNSS_EOP_MIN_H:-12}
        if [ "$_eop_h" = nan ] || [ "$(python3 -c "print(int(float('$_eop_h') < $_min_h))")" = 1 ]; then
            echo "FAILED: the baked EOP table in $CFG has ${_eop_h} h of headroom (< ${_min_h} h)." >&2
            echo "  Regenerate the fleet's configs and gate them, then start:" >&2
            echo "    python3 $K/scripts/gnss/gen_fleet.py $K/config/gnss_fleet_chord.yaml" >&2
            echo "    python3 $K/scripts/gnss/gen_fleet.py $K/config/gnss_fleet_chord.yaml --check" >&2
            echo "  and push the live table once the nodes answer:  $K/scripts/gnss/site/eop_push.sh" >&2
            echo "  (GNSS_SKIP_PREFLIGHT=1 overrides, deliberately.)" >&2
            exit 1
        fi
        # The F-engine controller must answer first: the one this config names
        # (fpga_controller), crs-control since the 10-17 wiring. ConfigTracker fetches its
        # /config and /get-frame0-time at startup and fails fatally once its retries run out;
        # CHORDTelescope reads frame0 from /get-frame0-time once, at startup. A node started
        # while the controller is down dies; one started while it still publishes the epoch
        # from before an F-engine re-base latches the old axis and timestamps the sky in the
        # past. It can take many seconds to answer, hence the 20 s timeout.
        _fc=$(python3 -c 'import sys, yaml
L = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
c = (yaml.load(open(sys.argv[1]), Loader=L) or {}).get("fpga_controller") or {}
print("%s:%s%s" % (c["host"], c["port"], c.get("timing_endpoint", "/get-frame0-time")))' \
            "$CFG" 2>/dev/null) || _fc="crs-control:54321/get-frame0-time"
        if ! curl -sf -m 20 "http://$_fc" 2>/dev/null | grep -q frame0_nano; then
            echo "FAILED: $_fc does not answer with frame0_nano." >&2
            echo "  The F-engine controller is down or restarting. A node started now either dies" >&2
            echo "  (ConfigTracker) or latches the wrong epoch (CHORDTelescope). Wait until" >&2
            echo "    curl -s http://$_fc" >&2
            echo "  returns a start_ctime you believe, then retry. (GNSS_SKIP_PREFLIGHT=1 overrides.)" >&2
            exit 1
        fi
        # The binary's own interface document against the broker's golden and the frame sizes
        # this config was generated for (iface_preflight.sh). It runs on the node, where the
        # binary's libraries are; only a disagreement refuses, anything unchecked warns.
        if ! gnss_iface_preflight node "$CFG" ssh -o BatchMode=yes -o ConnectTimeout=10 "$N" \
                "timeout 20 '$BIN' --interfaces-json"; then
            echo "FAILED: $BIN on $N disagrees with the broker's interface golden or with the" >&2
            echo "  frame sizes $CFG was generated for (above). Build the binary and generate" >&2
            echo "  the config from one tree. (GNSS_SKIP_IFACE=1 overrides.)" >&2
            exit 1
        fi
    fi
    # Warn when the binary is older than the library it links: `ninja lib/stages/all` builds
    # the static library without relinking kotekan, so a clean build can leave an old
    # executable on disk. Only a warning: mtimes over NFS are provenance, not a correctness
    # oracle, and refusing a restart on one would do more harm than the stale binary.
    _LIB=$(dirname "$BIN")/../lib/stages/libkotekan_stages.a
    if [ -f "$_LIB" ] && [ "$_LIB" -nt "$BIN" ]; then
        echo "warning: $BIN is older than $(basename "$_LIB") -- it was not relinked." >&2
        echo "    binary  $(date -r "$BIN" '+%F %T')" >&2
        echo "    library $(date -r "$_LIB" '+%F %T')" >&2
        echo "    'ninja lib/stages/all' does not relink kotekan. Run 'ninja kotekan'." >&2
        echo "    Starting anyway -- but this node will run the stale binary." >&2
    fi
    # Also warn when the binary is older than the newest GNSS source, which the check above
    # cannot see: /home/kvand is NFS-shared and there are two build trees (`build/`, the
    # nodes' DPDK build launched here, and `build_nodpdk/`, cf06's), so a change built in one
    # leaves the other's binary and library consistently old.
    _NEWEST_SRC=$(find "$KOTEKAN_SRC/lib/stages/gnss" -name '*.cpp' -o -name '*.hpp' \
                  2>/dev/null | xargs -r ls -t 2>/dev/null | head -1)
    if [ -n "$_NEWEST_SRC" ] && [ "$_NEWEST_SRC" -nt "$BIN" ]; then
        echo "warning: $BIN is older than $(basename "$_NEWEST_SRC") -- this build tree is out of date." >&2
        echo "    binary  $(date -r "$BIN" '+%F %T')" >&2
        echo "    source  $(date -r "$_NEWEST_SRC" '+%F %T')  ($_NEWEST_SRC)" >&2
        echo "    Two build trees exist (build/ = nodes, build_nodpdk/ = cf06); rebuild this one:" >&2
        echo "    cd $(dirname "$BIN")/.. && ninja kotekan" >&2
        echo "    Starting anyway -- but this node will run the stale binary." >&2
    fi
    # Provenance: if this config is the one the fleet manifest owns for the node, warn when it
    # no longer matches; a node started from a config nobody can regenerate has no explanation
    # when it misbehaves. Never a refusal: a hand-rolled config (profiling, a one-off) is
    # legitimate, and is not judged against the fleet config (the --print-path comparison).
    # Exit 1 from `gen_fleet.py --check` is read as a difference; any other non-zero exit as the
    # check itself failing (gen_fleet exits 2 when the generator cannot run), which says nothing
    # about the config.
    local MAN="$K/config/gnss_fleet_chord.yaml" OWNED RC OUT
    if [ -f "$MAN" ]; then
        OWNED=$(python3 "$K/scripts/gnss/gen_fleet.py" "$MAN" --node "$N" --print-path 2>/dev/null)
        if [ -n "$OWNED" ] && [ "$OWNED" = "$CFG" ]; then
            OUT=$(python3 "$K/scripts/gnss/gen_fleet.py" "$MAN" --node "$N" --check 2>&1)
            RC=$?
            if [ "$RC" -eq 1 ]; then
                echo "warning: $CFG no longer matches config/gnss_fleet_chord.yaml." >&2
                echo "$OUT" | sed 's/^/    /' >&2
                echo "    Starting it anyway (regenerate: scripts/gnss/gen_fleet.py $MAN)" >&2
            elif [ "$RC" -ne 0 ]; then
                echo "note: could not verify $CFG against the manifest (checker exited $RC)." >&2
                echo "      This says nothing about the config -- the check itself failed:" >&2
                echo "$OUT" | tail -3 | sed 's/^/      /' >&2
            fi
        fi
    fi
    [ -n "$SETENV" ] && echo "note: starting with$SETENV" >&2
    true
}

# The mkdir makes /tmp/gnss, the configs' record_dir, on node-local disk (not the NFS home).
# A stopped or failed transient unit can stay loaded, and systemd-run then refuses the name
# ("Unit gnss-node.service was already loaded or has a fragment file"); reset-failed clears it
# and is harmless when there is nothing to clear, hence the `|| true`. An unreadable config ends
# the session (exit 2); a failing systemd-run is left to the retry loop below, whose next attempt
# stops and resets the unit first.
# The unit restarts itself on failure. An F-engine reconfigure (FPGAMonitor exits when the
# controller no longer matches the config and frame0 the node started with) or a crash ends a
# healthy node; Restart=on-failure brings it back after RestartSec, re-reading the F-engine controller on the way
# up, so it converges on a new epoch by itself. While the controller is down it keeps failing and
# retrying; StartLimitIntervalSec=0 means systemd never gives up. A clean exit (/kill,
# `systemctl stop`) stays down.
# The log is appended, not truncated, so the fatal that caused a restart survives it; each
# node_up start moves the previous file to $LOG.1 as root first. That also keeps a log left by
# another user from blocking the root unit's append (fs.protected_regular: status=209/STDOUT).
# LimitCORE=infinity keeps a crash's cause: the log holds at most glibc's one line. systemd's
# default soft core limit is 0, under which the kernel writes nothing; LimitCORE= sets soft and
# hard. The node's kernel.core_pattern decides where a core goes: a file pattern writes it
# there; apport's stock pipe keeps it in /var/lib/apport/coredump (5 per uid) but discards it if
# the binary was replaced since the process started, which a live deploy does. A core is tens
# of GB, mostly the hugepage voltage rings (the default coredump_filter includes private
# hugepages), and a crash loop writes one per relaunch.
# needrestart must never restart the node: Ubuntu's daily unattended-upgrade (06:00-07:00Z)
# runs it, it restarts every service that maps an upgraded library (kotekan maps libfreetype),
# and the restart drops every REST-set state. The override below exempts this unit, so an
# upgraded library is picked up at the next planned restart. Written only if needrestart is
# installed and the line is not already there.
NR_CONF=/etc/needrestart/conf.d/gnss-node.conf
NR_LINE='$nrconf{override_rc}{qr(^gnss-node\.service$)} = 0;'
REMOTE_UP="mkdir -p /tmp/gnss && sudo systemctl reset-failed gnss-node 2>/dev/null || true; \
      { [ -d $(dirname $NR_CONF) ] && ! grep -qxF '$NR_LINE' $NR_CONF 2>/dev/null \
        && echo '$NR_LINE' | sudo tee $NR_CONF >/dev/null; true; }; \
      { [ -f '$LOG' ] && sudo mv -f '$LOG' '$LOG.1'; true; }; \
      test -r '$CFG' || { echo \"node_up: config unreadable\"; exit 2; }; \
      sudo systemd-run --unit=gnss-node \
        --working-directory=$K \
        --property=Restart=on-failure --property=RestartSec=20 \
        --property=StartLimitIntervalSec=0 \
        --property=LimitCORE=infinity$SETENV \
        --property=StandardOutput=append:$LOG \
        --property=StandardError=inherit \
        '$BIN' --config '$CFG' --bind-address 0.0.0.0:12048"

# ---------------------------------------------------------------------------------------
# Start, verify, retry: whether a start succeeds depends on the F-engine controller
# (fpga_controller in the config). ConfigTracker's fetch from it is fatal once its retries run
# out, and its single-threaded REST server can stall for many seconds while it reads the boards,
# so some nodes of a fleet restart fail while others start. The failure looks per-node but is a
# timeout.
#
# The retry runs inside the one ssh session: sudo's credential cache is per-TTY, so retrying
# from here would cost a password per attempt.
#
# Wait on the REST port, not on the unit: `systemctl is-active` reports active as soon as
# systemd-run forks, long before kotekan binds 12048. The port answering is the first evidence
# the process got past ConfigTracker. The wait ends early only when the unit leaves
# active/activating; a unit waiting out RestartSec after a failure reads "activating", so
# that wait continues while systemd restarts it.
REMOTE_START_VERIFY="for attempt in 1 2 3; do \
      sudo systemctl stop gnss-node 2>/dev/null || true; sleep 1; \
      $REMOTE_UP || { echo \"node_up: systemd-run failed on attempt \$attempt -- retrying\"; continue; }; \
      ok=0; \
      for i in \$(seq 1 100); do \
        if curl -sf -m 2 -o /dev/null http://localhost:12048/telescope/time0_ns 2>/dev/null; \
          then ok=1; break; fi; \
        systemctl is-active gnss-node 2>/dev/null | grep -qE '^activ' || break; \
        sleep 2; \
      done; \
      if [ \$ok = 1 ]; then echo \"node_up: up on attempt \$attempt\"; exit 0; fi; \
      echo \"node_up: attempt \$attempt did not reach :12048 -- retrying\"; \
    done; \
    echo \"node_up: FAILED after 3 attempts; last 15 log lines:\"; tail -15 '$LOG'; exit 1"
# ---------------------------------------------------------------------------------------

case "$ACT" in
  status)
    # `systemctl is-active` exits non-zero for a stopped unit, so a bare `|| echo unreachable`
    # would report every reachable-but-stopped node as unreachable. Capture instead of chaining.
    printf "%-6s " "$N"
    S=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$N" systemctl is-active gnss-node 2>/dev/null)
    echo "${S:-unreachable}"
    ;;
  down)
    ssh -t "$N" "sudo systemctl stop gnss-node"
    echo "note: the unit is transient -- it is now gone. Use '$0 $N' to bring it back."
    ;;
  up)
    preflight
    ssh -t "$N" "$REMOTE_START_VERIFY"
    ;;
  restart)
    # Same as `up`: the stop and the systemd-run share one ssh session and one sudo prompt.
    # The stop inside REMOTE_START_VERIFY is `|| true` because the transient unit does not
    # exist on a node that is already down. Preflight runs before the stop, so a typo in
    # GNSS_CFG cannot take a healthy node down and then refuse to bring it back.
    preflight
    ssh -t "$N" "$REMOTE_START_VERIFY"
    ;;
  *) echo "unknown action '$ACT' (up|down|restart|status)"; exit 2 ;;
esac
