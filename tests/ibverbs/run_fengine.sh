#!/bin/bash
#
# Runs the F-engine lab trio: the D-engine simulator (crs_dengine_sim.yaml) on the sender
# host, the F-engine (spark_fengine.yaml) on the engine host, and the wire-mode spectrum
# checker (fengine_check.yaml) back on the sender host, then prints the checker's
# verdict and the engine's aligner and GPU counters.
#
# usage: run_fengine.sh <engine_host> <engine_kotekan> <engine_config> \
#                       <sender_host> <sender_kotekan> <sim_config> <check_config> [seconds]
#
# A host of "localhost" runs that side here; anything else runs it over ssh (user@host
# works). The kotekan binaries need CAP_NET_RAW on both hosts, the engine host a CUDA build
# and the huge pages its config asks for. The checker exits by itself once its num_windows
# windows per ADC came back clean; [seconds] (default 60) only bounds the wait. Logs go to
# /tmp/kotekan-fengine.log and /tmp/kotekan-f{sim,check}.log on the respective hosts.
#
set -eu

ENG_HOST=$1; ENG_BIN=$2; ENG_CFG=$3
SND_HOST=$4; SND_BIN=$5; SIM_CFG=$6; CHK_CFG=$7
RUN_SECONDS=${8:-60}
ENG_PORT=12048
SIM_PORT=12049
CHK_PORT=12050

run() { # host command...
    local host=$1
    shift
    if [ "$host" = localhost ]; then "$@"; else ssh "$host" "$@"; fi
}

run_shell() { # host "shell command line"
    if [ "$1" = localhost ]; then sh -c "$2"; else ssh "$1" "$2"; fi
}

start() { # host binary port config log; prints the pid
    local host=$1 bin=$2 port=$3 cfg=$4 log=$5
    if [ "$host" = localhost ]; then
        nohup "$bin" -b "0.0.0.0:$port" -c "$cfg" > "$log" 2>&1 < /dev/null &
        echo $!
    else
        ssh "$host" "nohup $bin -b 0.0.0.0:$port -c $cfg > $log 2>&1 < /dev/null & echo \$!"
    fi
}

ENG_PID=$(start "$ENG_HOST" "$ENG_BIN" $ENG_PORT "$ENG_CFG" /tmp/kotekan-fengine.log)
trap 'run "$ENG_HOST" kill -INT $ENG_PID 2>/dev/null || true' EXIT
# The engine allocates and registers its buffers and warms the GPU before it can post.
sleep 20
CHK_PID=$(start "$SND_HOST" "$SND_BIN" $CHK_PORT "$CHK_CFG" /tmp/kotekan-fcheck.log)
SIM_PID=$(start "$SND_HOST" "$SND_BIN" $SIM_PORT "$SIM_CFG" /tmp/kotekan-fsim.log)
trap 'run "$SND_HOST" kill -INT $SIM_PID $CHK_PID 2>/dev/null || true
      run "$ENG_HOST" kill -INT $ENG_PID 2>/dev/null || true' EXIT
echo "running for up to $RUN_SECONDS s"

VERDICT=""
for _ in $(seq $(( (RUN_SECONDS + 4) / 5 ))); do
    sleep 5
    VERDICT=$(run_shell "$SND_HOST" "grep -hE 'TEST_PASSED|TEST_FAILED' /tmp/kotekan-fcheck.log 2>/dev/null" || true)
    [ -n "$VERDICT" ] && break
done

echo "== engine $ENG_HOST"
run_shell "$ENG_HOST" "curl -s localhost:$ENG_PORT/metrics" \
    | grep -E '^kotekan_(crs_window_align|cuda_spark_fengine|ibv_udp_recv_out_of_buffer)' || true
echo "== checker $SND_HOST"
run_shell "$SND_HOST" "grep -E 'Checked|TEST_' /tmp/kotekan-fcheck.log" || true
run "$SND_HOST" kill -INT "$SIM_PID" "$CHK_PID" 2>/dev/null || true
run "$ENG_HOST" kill -INT "$ENG_PID" 2>/dev/null || true
trap - EXIT

case "$VERDICT" in
    *TEST_PASSED*) echo "PASS" ;;
    *) echo "FAIL (no TEST_PASSED within ${RUN_SECONDS}s)"; exit 1 ;;
esac
