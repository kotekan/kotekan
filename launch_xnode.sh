#!/bin/bash
# Launch the 3-instance LOCAL cross-node split-band GNSS pipeline.
#   front (REST 12048)  = rawFileRead -> fftwEngine -> split -> send channels,
#                         recv records -> combiner -> write
#   worker A (REST 12049) = recv subA channels -> search+track -> send records
#   worker B (REST 12050) = recv subB channels -> search+track -> send records
# Channel + record streams cross localhost TCP via bufferSend/bufferRecv, carrying
# chordMetadata fpga_seq (the cross-node absolute-sample reference under test).
#
# Usage:  ./launch_xnode.sh            # generate configs + launch all 3
#         ./launch_xnode.sh stop       # kill all instances
set -e
cd "$(dirname "$0")"
KO=./build_mac/kotekan/kotekan

if [ "$1" = "stop" ]; then
    pkill -f "dist_node_(front|a|b)\.yaml" 2>/dev/null && echo "stopped" || echo "nothing running"
    exit 0
fi

python3 config/gen_xnode_configs.py
rm -rf /tmp/gpsxnode && mkdir -p /tmp/gpsxnode

# Workers first: they listen for the channel streams (front connects to them).
# Their record-senders reconnect until the front's recv ports come up.
$KO -c config/dist_node_a.yaml -b 0.0.0.0:12049 > /tmp/xnode_a.log 2>&1 &
echo "worker A (REST 12049, listen 14001, send 14003) pid $!"
$KO -c config/dist_node_b.yaml -b 0.0.0.0:12050 > /tmp/xnode_b.log 2>&1 &
echo "worker B (REST 12050, listen 14002, send 14004) pid $!"
sleep 2

$KO -c config/dist_node_front.yaml -b 0.0.0.0:12048 > /tmp/xnode_front.log 2>&1 &
echo "front    (REST 12048, send 14001/14002, recv 14003/14004) pid $!"
echo "logs: /tmp/xnode_{front,a,b}.log   combined records: /tmp/gpsxnode/   stop: ./launch_xnode.sh stop"
