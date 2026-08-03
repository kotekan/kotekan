#!/bin/bash
# Start the CHORD L5 broker with the FLEET-COMBINED DLL (docs/CHORD_GNSS_SHARED_DLL.md).
#
# WHY THIS EXISTS: --dll-combiners is a 14-endpoint list whose shape is NOT uniform -- cx19 and
# cx51 were regenerated with --combine-gpus and expose ONE merged 14-channel combiner
# (gnss0_combine), while the other six still predate that change and expose TWO 7-channel ones
# (gnss0_combine and gnss1_combine). Typing that by hand once is fine; typing it right every
# time is not. Regenerate all eight configs when production is serving /config on 12048 again
# and this asymmetry goes away (STATE 8.16.4) -- then every node takes the {0..1} form.
#
# The fleet combine costs nothing if a node is missing: an unreachable instance is skipped with
# a log line, and a PRN with fewer than --dll-min-instances agreeing instances falls back to the
# single --combiner discriminator.
#
# usage:  broker_up.sh [extra broker args...]
#         broker_up.sh --dll-gain 0          # fleet DLL polled but not applied (a control run)
set -u
K=/home/kvand/gnss/kotekan
PY=/home/kvand/gnss/venv/bin/python
cd "$K"

# One merged combiner (--combine-gpus nodes)
MERGED="http://cx19:12049/gnss0_combine,http://cx51:12049/gnss0_combine"
# Two per-GPU combiners (the rest)
SPLIT=""
for n in cx27 cx42 cx43 cx44 cx47 cx52; do
    SPLIT="$SPLIT,http://$n:12049/gnss{0..1}_combine"
done

exec $PY -u python/scripts/gnss/gps_distributed_broker.py \
    --rest-url http://localhost:12049 \
    --detectors http://localhost:12050/gps_search \
    --trackers "http://cx19:12049/gnss{0..1}_track,http://cx27:12049/gnss{0..1}_track,\
http://cx42:12049/gnss{0..1}_track,http://cx43:12049/gnss{0..1}_track,\
http://cx44:12049/gnss{0..1}_track,http://cx47:12049/gnss{0..1}_track,\
http://cx51:12049/gnss{0..1}_track,http://cx52:12049/gnss{0..1}_track,\
http://127.0.0.1:12099/sink_track" \
    --combiner gnss0_combine \
    --dll-combiners "${MERGED}${SPLIT}" \
    --almanac --almanac-source brdc --dead-reckon --narrow-search \
    --time0-endpoint telescope/time0_ns --dr-clock-chips 0.0 \
    --constellation G --carrier-hz 1176.45e6 --chip-rate-hz 10.23e6 \
    --code-length 10230 --hops-per-sec 195312.5 \
    --cl-assist --long-code-segments 20 --long-code-epoch-s 0.02 \
    --seed-doppler det --acquire-snr 30 \
    --dll-gain 0.25 --carrier-gain 0.0 \
    --code-bias-alpha 0.05 --code-bias-min-sats 2 \
    --lat 49.32075144444 --lon -119.62081125 --alt 545 --mask-deg 0 --interval 2 \
    --search-margin-wide-hz 200 --search-margin-hz 100 \
    --fit-gap-s 900 --fit-min-snr 0 --bias-min-snr 60 \
    --nh-period-offset -4 \
    "$@"
