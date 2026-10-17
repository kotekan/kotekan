#!/bin/bash
# The single definition of what each GNSS component runs. Sourced, never executed.
#
# Each definition carries the argv and the environment together. A unit written from a
# component's command line alone drops every variable that is not on it, such as the broker's
# OPENBLAS_NUM_THREADS=1.
#
# The units exec a foreground runner, not a daemonising launcher. A launcher that does
# `nohup setsid ... &` plus `disown`, and `pkill`s a previous instance first, breaks systemd:
#   * `Type=simple` expects ExecStart to be the process; a script that backgrounds and exits
#     looks like a service that died on startup, and its cgroup is then stopped with it;
#   * the backgrounded child is not the unit's main PID, so Restart= never sees it die;
#   * the internal `pkill` races systemd's own restart logic.
#
#     stack_components.sh          <- what to run, and with what environment  (here)
#            |
#            +-- run_component.sh  <- execs it in the foreground
#                           |
#                           +-- systemd ExecStart=          (the live path)
#
#     stack_contract_gate.sh       <- checks a running process still matches this file
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
GNSS_VENV_FT=${GNSS_VENV_FT:-$GNSS_ROOT/venv-ft}   # 3.14t free-threaded, the broker only

# The kotekan binary: the DPDK-free build. These components run on the gnss VM and cf06, never
# on a cx node (the nodes run the DPDK build through node_up.sh).
GNSS_BIN=${GNSS_BIN:-$GNSS_K/build_nodpdk/kotekan/kotekan}

GNSS_BROKER_URL=${GNSS_BROKER_URL:-http://127.0.0.1:12060}
GNSS_FRAME0_URL=${GNSS_FRAME0_URL:-http://cx43:12048}
GNSS_OBS_OUT=${GNSS_OBS_OUT:-$GNSS_ROOT/fixtures/obs}
GNSS_SITE_LL=${GNSS_SITE_LL:-"--lat 49.32001414 --lon -119.62262691 --alt 545"}

# ── the nine chains and their RF constants ─────────────────────────────────────────────────
# Not free parameters: carrier, chip rate and code length must match the SignalDef the broker
# runs, which it logs at startup ("signal gps_l5: SignalDef(...)"); if a chain's signal
# changes, take the numbers from that line. L2C's chips are CM chips at 0.5115 Mcps (the
# CM/CL multiplex), 586 m each, not 29 m.
#   chain  sys carrier_hz   chip_rate_hz  code_len
GNSS_CHAIN_RF="
gps_l5  G 1176450000 10230000 10230
gal_e5a E 1176450000 10230000 10230
bds_b2a C 1176450000 10230000 10230
gal_e5b E 1207140000 10230000 10230
bds_b2b C 1207140000 10230000 10230
bds_b3i C 1268520000 10230000 10230
gal_e6  E 1278750000  5115000  5115
gps_l2c G 1227600000   511500 10230
glo_l3oc R 1202025000 10230000 10230
"

gnss_chains() { echo "$GNSS_CHAIN_RF" | awk 'NF {print $1}'; }

gnss_chain_rf() {          # gnss_chain_rf <chain> -> "chain sys carrier chip code"
    echo "$GNSS_CHAIN_RF" | awk -v c="$1" '$1 == c {print; found=1} END {exit !found}'
}

gnss_components() { echo "broker gather aggregator viewer obs rail"; }   # the VM's stack
gnss_cube_components() { echo "cubearch cubecompact"; }                      # cf06's cube leg

# ── the definitions ────────────────────────────────────────────────────────────────────────
gnss_component() {
    local name=${1:?gnss_component <broker|gather|aggregator|viewer|obs|rail|cubearch|cubecompact>}
    GNSS_CWD=""
    case "$name" in

    broker)
        # Free-threaded 3.14t. Under the GIL the gather keeps dropping the broker's connection
        # and every chain reports all instances stale; L5 alone keeps working, which makes the
        # failure look survivable.
        export GNSS_PY="${GNSS_PY:-$GNSS_VENV_FT/bin/python}"
        # One BLAS thread. numpy's OpenBLAS pool starts a busy-spinning worker per core, and
        # the joint filter's small matmuls wake all of them; the spin takes most of the CPU
        # and starves the telemetry reader. At these sizes one thread is also faster. Part of
        # the command, not an optimisation.
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
        # The search runs on the GPU (use_cuda_acquire), so GNSS_BIN must be a CUDA build, not
        # a -DUSE_CUDA=OFF tree. The config is generated and untracked. Regenerate it with
        #   config/gen_chord_gnss_config.py --base config/base/live_config_20260730.json \
        #       --node cx19 --aggregator-instance cx19 cx27 cx42 cx43 cx44 cx51 \
        #       --rest-port 12050 --cuda-acquire --aggregator-cores 0,1,2,3,4,5 \
        #       --out config/generated/chord_gnss_agg6_cuda.yaml
        GNSS_EXEC=("$GNSS_BIN" --config
                   "${GNSS_AGG_CFG:-$GNSS_K/config/generated/chord_gnss_agg6_cuda.yaml}"
                   --bind-address "0.0.0.0:12050")
        ;;

    viewer)
        # The cwd is load-bearing: the script is named by a relative path, and it serves its
        # static files from its own directory.
        GNSS_CWD="$GNSS_K/python/scripts/js_viewer"
        GNSS_EXEC=("$GNSS_VENV/bin/python" -u livebeam_server.py
                   --no-power-stream --http-port "${GNSS_VIEWER_PORT:-8080}" --ws-port 8539
                   --kotekan-rest-port "${GNSS_BROKER_PORT:-12060}"
                   $GNSS_SITE_LL --unified
                   --pvt-obs-globs "$GNSS_OBS_OUT/[gb]*_20[0-9][0-9][0-9][0-9][0-9][0-9].jsonl")
        ;;

    obs)
        # One writer per chain; $GNSS_CHAIN names it.
        local row
        row=$(gnss_chain_rf "${GNSS_CHAIN:?set GNSS_CHAIN for the obs component}") || {
            echo "stack_components: no RF row for chain '${GNSS_CHAIN}'" >&2; return 1; }
        # shellcheck disable=SC2086  # deliberate word split of the table row
        set -- $row
        GNSS_EXEC=("$GNSS_VENV/bin/python" -u "$GNSS_K/python/scripts/gnss/gnss_observables.py"
                   --url "$GNSS_BROKER_URL" --combiner "$1" --search "$1"
                   --sys "$2" --band "$1"
                   --carrier-hz "$3" --chip-rate-hz "$4" --code-length "$5"
                   $GNSS_SITE_LL --frame0-url "$GNSS_FRAME0_URL"
                   --out "$GNSS_OBS_OUT/$1_%Y%m%d.jsonl")
        ;;

    rail)
        # Per-channel 4+4b rail fraction and band power from every GPU's voltage tap, every pass
        # (1 s on the nodes). Standalone: stdlib only, reads the nodes' rf_stats and nothing else,
        # so it can never touch the control path.
        GNSS_EXEC=(/usr/bin/python3 -u "$GNSS_K/scripts/gnss/site/rail_watch.py" record
                   --out "$GNSS_OBS_OUT/rf_chan_%Y%m%d.jsonl")
        ;;

    cubearch)
        # cf06 only: the far side of the beam-cube push leg, one bufferRecv on :11070 for the
        # whole fleet's windows and a rawFileWrite onto /mnt/cs00. Start it before the nodes
        # (cubearch_up.sh says why). The pre-flight (config present, archive dir writable,
        # sender/receiver frame_size agree) is `cubearch_up.sh --check`, which
        # gnss-cubearch.service runs as ExecStartPre.
        GNSS_EXEC=("$GNSS_BIN" -c
                   "${GNSS_CUBEARCH_CFG:-$GNSS_K/config/generated/chord_gnss_cubearch.yaml}")
        ;;
    cubecompact)
        # cf06 only: the compactor loop (raw -> L0 -> rungs -> the daily publish), with the
        # same five arguments cubecompact_up.sh passes. h5py lives in venv, not venv-ft.
        local root=${CUBE_ROOT:-/mnt/cs00/data/kvand/gnss_cube}
        GNSS_EXEC=("$GNSS_K/scripts/gnss/site/cubecompact_loop.sh" "$GNSS_VENV/bin/python"
                   "$GNSS_K/python/scripts/gnss/gnss_cube_compact.py"
                   "${CUBE_RAW:-$root/raw}" "$root" "${CUBE_COMPACT_PERIOD:-300}")
        ;;
    *)
        echo "stack_components: unknown component '$name' (have: $(gnss_components))" >&2
        return 1 ;;
    esac
}
