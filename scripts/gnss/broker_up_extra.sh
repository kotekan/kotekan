#!/bin/bash
# Start an EXTRA-CONSTELLATION broker for one of the chains a node's config was generated
# with (gen_chord_gnss_config.py --extra-signal). One broker per constellation, alongside
# the GPS L5 one from broker_up.sh -- see docs/CHORD_MULTIBAND.md.
#
# usage:  broker_up_extra.sh e5a [extra broker args...]
#         broker_up_extra.sh b2a --dll-gain 0        # DLL polled but not applied (control run)
#
# THE BIG DIFFERENCE FROM broker_up.sh: there are NO DETECTIONS on these chains, by design.
# E5a-Q and B2a-P carry PER-PRN secondaries, which the replica bank's single-sequence overlay
# slot cannot hold, so `nh_search` on them is a silent no-op and blind multi-period acquisition
# does not exist (CHORD_MULTIBAND.md section 5). The generator therefore emits no search feed
# for an extra chain, and this broker runs with --detectors EMPTY: purely model-primary, every
# visible satellite seeded from BRDC + the receiver clock. That is not a workaround -- the
# broker documents it as "the CHORD configuration", because the station position is known and
# the F-engine is GPS-disciplined.
#
# THE CLOCK COMES FROM THE GPS BROKER, AND IS NOW ADOPTED RATHER THAN PASTED. With no detections
# this broker cannot solve its own receiver clock -- `offs` is always empty, so the EMA never
# runs. It used to hold whatever --dr-clock-chips was primed with FOREVER, read out of the GPS
# broker's log by a human, wrong after any F-engine restart and silent about it.
#
# --dr-clock-adopt below reads it live from the band sibling's receiver-state file every cycle
# (receiver_state.py, the export that already existed with nothing consuming it). The transfer
# is EXACT: E5a/B2a sit on L5's 1176.45 MHz carrier, so the cable delay, F-engine pipeline and
# PFB group delay are the same hardware in the same band -- which is exactly what --state-dongle
# l5 asserts. It does NOT transfer across a retune (E5b at 1207 MHz has a different group
# delay), so that dongle key is load-bearing, not decoration.
#
# THE GPS BROKER MUST PUBLISH for this to work: give it --state-file /tmp/gnss_state/gps_l5.json
# --state-dongle l5. Without it this chain logs "no fresh sibling" and holds --dr-clock-chips,
# i.e. degrades to exactly the old behaviour rather than to something worse.
#
# Inter-constellation time is already handled: gnss_ephemeris.py applies BDT = GPST - 14 s to
# both toc and toe, and Galileo is GPS-aligned. Do not add an offset here.
#
# Do NOT put comments between the continued lines below -- a comment on a backslash-joined
# line terminates the command and silently drops every argument after it.
set -u
K=/home/kvand/gnss/kotekan
PY=/home/kvand/gnss/venv/bin/python
cd "$K"

CHAIN="${1:-}"; shift || true
case "$CHAIN" in
    e5a) SIGNAL=gal_e5a; PORT=12061 ;;
    # BDS-3 only: B2a does not exist on the BDS-2 birds C1-C18 (they broadcast B1I at
    # 1561 MHz, which is not even in the science band). The broker defaults --dr-min-prn to
    # 19 for C, so this is belt-and-braces, stated rather than relied upon.
    b2a) SIGNAL=bds_b2a; PORT=12062 ;;
    *)   echo "usage: $0 <e5a|b2a> [extra broker args...]" >&2; exit 2 ;;
esac

# Same node/GPU topology as broker_up.sh, with the chain tag inserted. Regenerate the node
# configs with the MATCHING --extra-signal before running this: these endpoints exist only if
# the chain was generated (a missing one logs "set_seeds ... failed" every cycle and is
# otherwise benign, so a typo here degrades quietly -- check the endpoints once by hand).
# PATH B ENDPOINTS. Extra signals are path-B chains as of 2026-08-08 (gen_chord_gnss_config.py
# gives --extra-signal its own cudaGnssInject + cudaCorrelatorDual rather than a path-A tracker),
# so the seed sink is `_inject` and the combiner is `_n2combine`. A path-A tracker chain for an
# extra signal no longer exists, and pointing at `_track`/`_combine` would POST seeds into
# nothing every cycle -- which logs and is otherwise silent, exactly the quiet degradation this
# file warns about below.
#
# Path B is per GPU with no --combine-gpus collapse (measured worth 0.997 and it costs
# instances, gnss_gpu_search.md 11.15), so EVERY node contributes both GPUs -- there is no
# merged/split split to mirror here as there is for the GPS chain.
CMB=""
TRK=""
for n in cx19 cx27 cx42 cx43 cx44 cx51; do
    CMB="${CMB:+$CMB,}http://$n:12049/gnss{0..1}_${CHAIN}_n2combine"
    TRK="${TRK:+$TRK,}http://$n:12049/gnss{0..1}_${CHAIN}_inject"
done

# --signal REPLACES the nine constants that used to be typed here (constellation, carrier,
# chip rate, code length, long-code segments+epoch, overlay length, and B2a's BDS-3-only
# --dr-min-prn 19). They now come from lib/stages/gnss/gnssSignal.hpp, which is where the
# tracker's own replica descriptors live: gal_e5a resolves to GAL_E5A_Q / GAL_E5A_Q_CS and
# derives the CS100 secondary from the pair rather than trusting a human to retype 100 twice.
#
# This file's own warning is why: "getting these wrong does not error -- the overlay period is
# computed mod the wrong length and the seed lands in an effectively random one of the 100
# periods". That is precisely the shape of the 2026-08-08 E5a defect.
exec $PY -u python/scripts/gnss/gps_distributed_broker.py \
    --rest-url http://cx19:12049 \
    --trackers "$TRK" \
    --combiner "gnss0_${CHAIN}_n2combine" \
    --dll-combiners "$CMB" \
    --n2-combiners "$CMB" \
    --publish-port $PORT \
    --carrier-from-code \
    --almanac --almanac-source brdc --dead-reckon \
    --time0-endpoint telescope/time0_ns --dr-clock-chips 0.0 \
    --dr-clock-adopt --state-read-dir /tmp/gnss_state --state-dongle l5 \
    --signal $SIGNAL --hops-per-sec 195312.5 \
    --cl-assist \
    --seed-doppler auto \
    --dll-gain 0.25 --carrier-gain 0.0 \
    --code-bias-alpha 0.05 --code-bias-min-sats 2 \
    --lat 49.32075144444 --lon -119.62081125 --alt 545 --mask-deg 0 --interval 2 \
    --fit-gap-s 3600 --fit-min-snr 0 --bias-min-snr 60 \
    --nh-period-offset 0 \
    "$@"
