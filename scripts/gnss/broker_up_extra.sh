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
# WHICH MEANS THE CLOCK COMES FROM THE GPS BROKER. With no detections this broker cannot solve
# its own receiver clock: it uses whatever --dr-clock-chips is primed with, forever. The number
# to prime it with is the GPS broker's converged clock, and it TRANSFERS EXACTLY -- E5a/B2a sit
# on L5's 1176.45 MHz carrier, so the cable delay, the F-engine pipeline and the PFB group
# delay are the same hardware in the same band. It does NOT transfer across a retune (E5b), and
# it does not survive an F-engine restart. Read it from the GPS broker's log ("clock" in the
# integrity line) and pass --dr-clock-chips; the 0.0 default below is a COLD START, good to
# whatever the instrumental constant happens to be, and 0.4 chips is the DLL capture range.
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
    e5a) SYS=E; PORT=12061; PRNARG="" ;;
    # BDS-3 only: B2a does not exist on the BDS-2 birds C1-C18 (they broadcast B1I at
    # 1561 MHz, which is not even in the science band). The broker defaults --dr-min-prn to
    # 19 for C, so this is belt-and-braces, stated rather than relied upon.
    b2a) SYS=C; PORT=12062; PRNARG="--dr-min-prn 19" ;;
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

# --long-code-{segments,epoch-s} 100 / 0.1 and --nh-overlay-len 100 are the CS100 secondary
# baked into the tracker's 1023000-chip code (GAL_E5A_Q_CS / BDS_B2A_P_CS). Getting these
# wrong does not error: the overlay period is computed mod the wrong length and the seed lands
# in an effectively random one of the 100 periods.
exec $PY -u python/scripts/gnss/gps_distributed_broker.py \
    --rest-url http://cx19:12049 \
    --trackers "$TRK" \
    --combiner "gnss0_${CHAIN}_n2combine" \
    --dll-combiners "$CMB" \
    --n2-combiners "$CMB" \
    --publish-port $PORT \
    --carrier-from-code \
    --nh-overlay-len 100 \
    --almanac --almanac-source brdc --dead-reckon \
    --time0-endpoint telescope/time0_ns --dr-clock-chips 0.0 \
    --constellation $SYS --dr-constellation $SYS $PRNARG \
    --carrier-hz 1176.45e6 --chip-rate-hz 10.23e6 \
    --code-length 10230 --hops-per-sec 195312.5 \
    --cl-assist --long-code-segments 100 --long-code-epoch-s 0.1 \
    --seed-doppler auto \
    --dll-gain 0.25 --carrier-gain 0.0 \
    --code-bias-alpha 0.05 --code-bias-min-sats 2 \
    --lat 49.32075144444 --lon -119.62081125 --alt 545 --mask-deg 0 --interval 2 \
    --fit-gap-s 3600 --fit-min-snr 0 --bias-min-snr 60 \
    --nh-period-offset 0 \
    "$@"
