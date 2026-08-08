#!/bin/bash
# Start the CHORD L5 broker with the FLEET-COMBINED DLL (docs/CHORD_GNSS_SHARED_DLL.md).
#
# WHY THIS EXISTS: --dll-combiners is a multi-endpoint list whose shape is NOT uniform -- cx19
# was regenerated with --combine-gpus and exposes ONE merged 14-channel combiner
# (gnss0_combine), while the others predate that change and expose TWO 7-channel ones
# (gnss0_combine and gnss1_combine). Typing that by hand once is fine; typing it right every
# time is not. Regenerate all configs when production is serving /config on 12048 again
# and this asymmetry goes away (STATE 8.16.4) -- then every node takes the {0..1} form.
#
# SIX NODES since 2026-08-04: cx47 and cx52 went back to the other developers (the
# BEST pair to give away -- worst sidelobe 0.233 at 19.65 chips vs 0.253 at 6.58 for
# cx51+cx52; see the ranking of all 28 pairs in git history). That is a
# CONFIGURATION change, not a degradation to absorb quietly -- the aggregator must be
# regenerated to match (chord_gnss_agg6.yaml, 79 channels over 12 feeds), because
# GnssChanAlignMerge waits on every feed it was built with and the missing ones never arrive.
# The comb survives it: cx47/cx52 held mod-8 residues 3 and 1, and the remaining six still
# contain ADJACENT residues (cx42 at 5, cx43 at 6), so g = 1 and there is no modular lag
# ambiguity. Cost is 106 -> 79 channels (-1.3 dB).
#
# DROPPING A NODE RENUMBERS FEED SLOTS. Ports are assigned node-major over the aggregator's
# list, so removing cx47 moved cx51 from slot 6 to slot 5: its config had been sending to
# 11052/11053 while the new aggregator listens on 11050/11051. Regenerate the NODE configs
# too (--search-port-base per node) and verify every server_port against the aggregator's
# feed list -- the 2026-08-03 cx51->cx27 slot error is what happens when you do not.
#
# The fleet combine costs nothing if a node is missing: an unreachable instance is skipped with
# a log line, and a PRN with fewer than --dll-min-instances agreeing instances falls back to the
# single --combiner discriminator.
#
# --seed-doppler auto since 2026-08-04 (was `det`). The seed feeds the tracker's f_ref fence,
# and `det` hands it the SEARCH's per-pass Doppler -- a 16 ms estimate, against the tracker's
# own 1.049 s deep window. That is a 65x shorter coherent span, and it shows: the detection
# Doppler steps a median 6.00 Hz between passes with 32% of steps exceeding the 9.54 Hz fence,
# versus 3.00 Hz and 4% for the model+clock-bias prediction. Re-pinning a 1.05 s reference to a
# 16 ms measurement throws the integration away. Measured effect: reference jumps in
# deep_rate_hz fell 18% -> 7% of steps and their interval went 10.1 s -> 20.1 s.
# `det` remains right for ACQUISITION (no track yet, or the model is what is being checked);
# it is wrong for a satellite already locked. The standing residual (~24 Hz) is NOT changed by
# this -- that is a separate, still-open defect (STATE 8.20.3).
#
# --signal gps_l5 REPLACES the six constants that used to be typed here by hand (constellation,
# carrier, chip rate, code length, long-code segments+epoch, overlay length). They are now
# derived from lib/stages/gnss/gnssSignal.hpp -- the same descriptors the trackers' replica bank
# is built from -- so the broker and the thing it is seeding cannot disagree.
#
# This was never cosmetic. --nh-overlay-len is GPS L5 Q5's NH20 and the broker's DEFAULT is
# 1800 (B1C): getting it wrong does not error, the alignment is computed mod the wrong length
# and the hint narrows the search to an effectively RANDOM alignment (observed: offset 1015,
# where the only legal values are 0..19). Same class as the E5a long-code defect of 2026-08-08.
# To re-check the mapping, put the old flags back alongside --signal: a disagreement is now a
# hard error naming both numbers (that is how this substitution was verified).
#
# And do NOT put comments between the continued lines below: a comment on a backslash-joined
# line terminates the command, silently dropping every argument after it.
#
# RUNS ON cf06 since 2026-08-04 (with the aggregator): --rest-url must name a real NODE, so
# it points at cx19 rather than localhost -- cf06 runs no gnss node of its own. --detectors
# stays on localhost because the aggregator moved here too.
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
for n in cx27 cx42 cx43 cx44; do
    SPLIT="$SPLIT,http://$n:12049/gnss{0..1}_combine"
done

# cx19's gnss{0..1}_inject entries are the PATH-B injector endpoints (cudaGnssInject) -- the
# SAME seed payload as _track, consumed by the N^2 synthetic-lane injector. They exist only
# when cx19 runs config/generated/chord_gnss_cx19_n2dual.yaml; on the normal config the broker
# logs "set_seeds ... failed" for them each cycle, which is benign (the per-endpoint try/except
# posts everything else regardless).
exec $PY -u python/scripts/gnss/gps_distributed_broker.py \
    --rest-url http://cx19:12049 \
    --detectors http://localhost:12050/gps_search \
    --trackers "http://cx19:12049/gnss{0..1}_track,http://cx27:12049/gnss{0..1}_track,\
http://cx42:12049/gnss{0..1}_track,http://cx43:12049/gnss{0..1}_track,\
http://cx44:12049/gnss{0..1}_track,http://cx51:12049/gnss{0..1}_track,\
http://cx19:12049/gnss{0..1}_inject,\
http://127.0.0.1:12099/sink_track" \
    --combiner gnss0_combine \
    --dll-combiners "${MERGED}${SPLIT}" \
    --publish-port 12060 \
    --carrier-from-code \
    --nh-hint --nh-hint-span 2 \
    --almanac --almanac-source brdc --dead-reckon --narrow-search \
    --time0-endpoint telescope/time0_ns --dr-clock-chips 0.0 \
    --state-file /tmp/gnss_state/gps_l5.json --state-dongle l5 \
    --signal gps_l5 --hops-per-sec 195312.5 \
    --cl-assist \
    --seed-doppler auto --acquire-snr 30 \
    --dll-gain 0.25 --carrier-gain 0.0 \
    --code-bias-alpha 0.05 --code-bias-min-sats 2 \
    --lat 49.32075144444 --lon -119.62081125 --alt 545 --mask-deg 0 --interval 2 \
    --search-margin-wide-hz 200 --search-margin-hz 100 \
    --fit-gap-s 3600 --fit-min-snr 0 --bias-min-snr 60 \
    --nh-period-offset 0 \
    "$@"
