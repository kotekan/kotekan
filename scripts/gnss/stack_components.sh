#!/bin/bash
# THE SINGLE DEFINITION OF WHAT EACH GNSS COMPONENT IS. Sourced, never executed.
#
# WHY THIS EXISTS. The launch details lived in two places at once: the `*_up.sh` scripts and,
# after the 2026-09-17 move to the gnss VM, the systemd units. That is not a style problem. The
# units were written by reproducing each script's COMMAND LINE faithfully -- and the broker's
# `export OPENBLAS_NUM_THREADS=1` is not on its command line, so it was silently dropped and the
# broker spent two thirds of its cycles in OpenBLAS's busy-wait until `perf` found it (#134).
# A definition that carries the argv and the environment TOGETHER makes that failure structural
# rather than a thing to remember.
#
# ⚠️ WHY THE UNITS DO NOT SIMPLY CALL `*_up.sh`, WHICH IS THE OBVIOUS IDEA AND IS WRONG.
# Those scripts are DAEMONISING launchers: every one of them does `nohup setsid ... &` plus
# `disown`, and most also `pkill` a previous instance first. Under systemd that is three
# separate faults --
#   * `Type=simple` expects ExecStart to BE the process; a script that backgrounds and exits
#     looks like a service that died on startup;
#   * `setsid` deliberately detaches the child from the unit's cgroup, so systemd loses
#     supervision entirely -- no Restart=, no MemoryMax, no clean stop;
#   * the internal `pkill` races systemd's own restart logic.
# So the direction is inverted: the shared definition is the truth, a FOREGROUND runner
# (`run_component.sh`) execs it, and BOTH the units and the `*_up.sh` scripts go through that.
# The scripts keep what they are genuinely good at and systemd does not do: host guards,
# preflight conditions, log rotation, and post-start health checks.
#
#     stack_components.sh          <- what to run, and with what environment  (HERE)
#            |
#            +-- run_component.sh  <- execs it in the FOREGROUND
#                    |      |
#                    |      +-- systemd ExecStart=          (the live path)
#                    +-- *_up.sh   <- daemonises it, adds guards/rotation/health checks
#
#     stack_contract_gate.sh       <- proves a RUNNING process still matches this file
#
# Usage:
#     . stack_components.sh
#     gnss_component broker          # -> sets GNSS_EXEC[], GNSS_CWD; exports the env
#     "${GNSS_EXEC[@]}"
#
# @author Keith Vanderlinde
set -u

GNSS_K=${GNSS_K:-/home/kvand/gnss/kotekan}
GNSS_ROOT=${GNSS_ROOT:-/home/kvand/gnss}
GNSS_VENV=${GNSS_VENV:-$GNSS_ROOT/venv}            # 3.12, everything except the broker
GNSS_VENV_FT=${GNSS_VENV_FT:-$GNSS_ROOT/venv-ft}   # 3.14t free-threaded, THE BROKER ONLY

# The kotekan binary. cx* nodes run the DPDK build; everywhere else the DPDK-free one.
case "$(hostname -s)" in
cx*) GNSS_BIN_DEFAULT=$GNSS_K/build/kotekan/kotekan ;;
*)   GNSS_BIN_DEFAULT=$GNSS_K/build_nodpdk/kotekan/kotekan ;;
esac
GNSS_BIN=${GNSS_BIN:-$GNSS_BIN_DEFAULT}

GNSS_BROKER_URL=${GNSS_BROKER_URL:-http://127.0.0.1:12060}
GNSS_FRAME0_URL=${GNSS_FRAME0_URL:-http://cx43:12048}
GNSS_OBS_OUT=${GNSS_OBS_OUT:-$GNSS_ROOT/fixtures/obs}
GNSS_SITE_LL=${GNSS_SITE_LL:-"--lat 49.32001414 --lon -119.62262691 --alt 545"}

# ── the eight chains and their RF constants ────────────────────────────────────────────────
# One copy. obs_up.sh had the only one, and obs_unit.sh used to PARSE it back out of that file
# with sed, which was right on the day it was written and one edit from being wrong.
#   chain  sys carrier_hz   chip_rate_hz  code_len  comb_mult
GNSS_CHAIN_RF="
gps_l5  G 1176450000 10230000 10230 1
gal_e5a E 1176450000 10230000 10230 1
bds_b2a C 1176450000 10230000 10230 1
gal_e5b E 1207140000 10230000 10230 1
bds_b2b C 1207140000 10230000 10230 1
bds_b3i C 1268520000 10230000 10230 1
gal_e6  E 1278750000  5115000  5115 1
gps_l2c G 1227600000   511500 10230 2
"

gnss_chains() { echo "$GNSS_CHAIN_RF" | awk 'NF {print $1}'; }

gnss_chain_rf() {          # gnss_chain_rf <chain> -> "chain sys carrier chip code comb"
    echo "$GNSS_CHAIN_RF" | awk -v c="$1" '$1 == c {print; found=1} END {exit !found}'
}

gnss_components() { echo "broker gather aggregator viewer obs"; }

# ── the definitions ────────────────────────────────────────────────────────────────────────
gnss_component() {
    local name=${1:?gnss_component <broker|gather|aggregator|viewer|obs>}
    GNSS_CWD=""
    case "$name" in

    broker)
        # ⚠️ FREE-THREADED 3.14t. Under the GIL the gather drops its consumer every 15 s, every
        # chain reports all instances stale, and only L5 keeps working -- which makes the
        # failure look survivable. It is not; it cost two hours of dead chains.
        export GNSS_PY="${GNSS_PY:-$GNSS_VENV_FT/bin/python}"
        # ⚠️ ONE BLAS THREAD (#134). numpy's OpenBLAS pool defaults to one BUSY-SPINNING worker
        # per core and the joint filter's ~57x57 matmuls wake all of them: ~53 cores of spin on
        # cf06, which starved the telemetry reader; 4.3 cores of 6 on the VM, two thirds of them
        # in blas_thread_server against 0.52% in the actual dgemm. At these sizes one thread is
        # also simply faster. THIS IS PART OF THE COMMAND, NOT AN OPTIMISATION.
        export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
        GNSS_EXEC=("$GNSS_PY" -u "$GNSS_K/scripts/gnss/broker_multi.py"
                   "${GNSS_CHAINS_YAML:-$GNSS_K/config/gnss_chains_chord.yaml}")
        ;;

    gather)
        GNSS_EXEC=("$GNSS_BIN" --config
                   "${GNSS_GATHER_CFG:-$GNSS_K/config/generated/chord_gnss_gather.yaml}"
                   --bind-address "0.0.0.0:${GNSS_GATHER_REST:-12051}")
        ;;

    aggregator)
        # Needs CUDA: the search runs on the GPU (use_cuda_acquire). Not a -DUSE_CUDA=OFF tree.
        # ⚠️ GNSS_SEARCH_PROFILE: agg_up.sh set this unconditionally, so the cf06 instance ran
        # with the pass/consumer profile ON for its whole life -- 113,997 [consumer] lines in its
        # last log. The first systemd unit dropped it silently (the #134 shape again, found while
        # consolidating). It is OFF here because it is a diagnostic that costs log volume, and
        # that is now a DECISION rather than an accident. KV, 2026-09-17: it dates to a period
        # when the search had lagged badly and was being sped up, and was simply never turned
        # off -- so OFF is the correct steady state. Set GNSS_SEARCH_PROFILE=1 to bring it back
        # for an investigation; agg_up.sh still does.
        [ -n "${GNSS_SEARCH_PROFILE:-}" ] && export GNSS_SEARCH_PROFILE
        GNSS_EXEC=("$GNSS_BIN" --config
                   "${GNSS_AGG_CFG:-$GNSS_K/config/generated/chord_gnss_agg6_cuda.yaml}"
                   --bind-address "0.0.0.0:${GNSS_AGG_REST:-12050}")
        ;;

    viewer)
        # ⚠️ The cwd is load-bearing: livebeam_server.py resolves its static assets relative to
        # it. Reproduce the argument list exactly rather than tidying it.
        GNSS_CWD="$GNSS_K/python/scripts/js_viewer"
        GNSS_EXEC=("$GNSS_VENV/bin/python" -u livebeam_server.py
                   --no-power-stream --http-port "${GNSS_VIEWER_PORT:-8080}" --ws-port 8539
                   --kotekan-rest-port "${GNSS_BROKER_PORT:-12060}"
                   $GNSS_SITE_LL --band l5 --unified
                   --pvt-obs-globs "$GNSS_OBS_OUT/[gb]*_2026*.jsonl")
        ;;

    obs)
        # One writer per chain; $GNSS_CHAIN names it.
        local row
        row=$(gnss_chain_rf "${GNSS_CHAIN:?set GNSS_CHAIN for the obs component}") || {
            echo "stack_components: no RF row for chain '${GNSS_CHAIN}'" >&2; return 1; }
        # shellcheck disable=SC2086  # deliberate word split of the table row
        set -- $row
        GNSS_EXEC=("$GNSS_VENV/bin/python" -u "$GNSS_K/python/scripts/gnss/gnss_observables.py"
                   --url "$GNSS_BROKER_URL" --combiner "$1" --search "$1" --airspy "$1"
                   --sys "$2" --band "$1"
                   --carrier-hz "$3" --chip-rate-hz "$4" --code-length "$5" --comb-mult "$6"
                   $GNSS_SITE_LL --frame0-url "$GNSS_FRAME0_URL"
                   --out "$GNSS_OBS_OUT/$1_%Y%m%d.jsonl")
        ;;

    *)
        echo "stack_components: unknown component '$name' (have: $(gnss_components))" >&2
        return 1 ;;
    esac
}
