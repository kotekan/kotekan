#!/bin/bash
#
# Runs ibvUdpRecv -> udpPacketCheck on one host and udpPacketGen -> ibvUdpSend on another for a
# while, then prints the receive rate, the receiver's CPU usage, and what was sent, received and
# lost, from the two instances' REST endpoints.
#
# usage: run_line_rate.sh <recv_host> <recv_kotekan> <recv_config> \
#                         <send_host> <send_kotekan> <send_config> [seconds]
#
# A host of "localhost" runs that side here; anything else runs it over ssh (user@host works).
# The kotekan binaries need CAP_NET_RAW on both hosts. Logs go to /tmp/kotekan-ibv-{recv,send}.log
# on each host.
#
# The sender's packet count includes the frames still queued in its NIC when it was stopped, so
# it is a little larger than the receiver's; the checker's lost count and the NIC drop count are
# the exact loss figures.
#
set -eu

RECV_HOST=$1; RECV_BIN=$2; RECV_CFG=$3
SEND_HOST=$4; SEND_BIN=$5; SEND_CFG=$6
RUN_SECONDS=${7:-20}
RECV_PORT=12048
SEND_PORT=12049

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
        nohup "$bin" -b "127.0.0.1:$port" -c "$cfg" > "$log" 2>&1 < /dev/null &
        echo $!
    else
        ssh "$host" "nohup $bin -b 127.0.0.1:$port -c $cfg > $log 2>&1 < /dev/null & echo \$!"
    fi
}

metrics() { # host port
    run "$1" curl -s "localhost:$2/metrics" | grep -E '^kotekan_(ibv_udp|udp_packet)' | sed 's/{[^}]*}//'
}

value() { # metrics name; 0 when the counter was never incremented
    local v
    v=$(echo "$1" | awk -v m="$2" '$1 == m {print $2; exit}')
    echo "${v:-0}"
}

RECV_PID=$(start "$RECV_HOST" "$RECV_BIN" $RECV_PORT "$RECV_CFG" /tmp/kotekan-ibv-recv.log)
trap 'run "$RECV_HOST" kill -INT $RECV_PID 2>/dev/null || true' EXIT
sleep 3
SEND_PID=$(start "$SEND_HOST" "$SEND_BIN" $SEND_PORT "$SEND_CFG" /tmp/kotekan-ibv-send.log)
trap 'run "$SEND_HOST" kill -INT $SEND_PID 2>/dev/null || true; run "$RECV_HOST" kill -INT $RECV_PID 2>/dev/null || true' EXIT
echo "sending for $RUN_SECONDS s"

# The receive rate over the middle of the run.
sleep 3
M1=$(metrics "$RECV_HOST" $RECV_PORT); T1=$(date +%s.%N)
sleep $((RUN_SECONDS > 6 ? RUN_SECONDS - 6 : 1))
M2=$(metrics "$RECV_HOST" $RECV_PORT); T2=$(date +%s.%N)
RECV_CPU=$(run "$RECV_HOST" top -b -H -n 1 -p "$RECV_PID" | awk 'NR > 7 && $9 > 0')
SEND_CPU=$(run "$SEND_HOST" top -b -H -n 1 -p "$SEND_PID" | awk 'NR > 7 && $9 > 0')

# Read the sender's counters and stop it in one round trip, so the two are close in time.
SEND=$(run_shell "$SEND_HOST" "curl -s localhost:$SEND_PORT/metrics; kill -INT $SEND_PID" \
       | grep -E '^kotekan_(ibv_udp|udp_packet)' | sed 's/{[^}]*}//')
sleep 2
RECV=$(metrics "$RECV_HOST" $RECV_PORT)
run "$RECV_HOST" kill -INT "$RECV_PID"
trap - EXIT

echo "== sender $SEND_HOST"
echo "$SEND"
echo "== receiver $RECV_HOST"
echo "$RECV"
echo "== sender threads using CPU (top -H, %CPU is the 9th column)"
echo "$SEND_CPU"
echo "== receiver threads using CPU"
echo "$RECV_CPU"
awk -v p1="$(value "$M1" kotekan_ibv_udp_recv_packets_total)" \
    -v p2="$(value "$M2" kotekan_ibv_udp_recv_packets_total)" \
    -v b1="$(value "$M1" kotekan_ibv_udp_recv_bytes_total)" \
    -v b2="$(value "$M2" kotekan_ibv_udp_recv_bytes_total)" -v t1="$T1" -v t2="$T2" \
    'BEGIN { dt = t2 - t1; pps = (p2 - p1) / dt;
             printf "== receive rate over %.1f s: %.3f Mpps, %.2f Gb/s of packets, %.2f Gb/s on the wire\n",
                    dt, pps / 1e6, (b2 - b1) * 8 / dt / 1e9, ((b2 - b1) + (p2 - p1) * 24) * 8 / dt / 1e9 }'
echo "sent $(value "$SEND" kotekan_ibv_udp_send_packets_total)," \
     "received $(value "$RECV" kotekan_ibv_udp_recv_packets_total)," \
     "lost $(value "$RECV" kotekan_udp_packet_check_lost_packets_total)," \
     "bad $(value "$RECV" kotekan_udp_packet_check_bad_packets_total)," \
     "NIC drops $(value "$RECV" kotekan_ibv_udp_recv_out_of_buffer)"
