#!/usr/bin/env python3
"""Generate a CHORD GNSS node config: the production config plus a GNSS branch.

A CHORD node cannot run a second kotekan instance (DPDK owns the NICs exclusively), so the
GNSS chain lives in the same process as the ingest. The generator takes production's config
as a base and injects into it rather than restating the pipeline, so upstream changes to
config/chord/pathfinder.j2 are picked up and the ingest stages are preserved by construction.

    # the base is production's own template, rendered stock exactly as kotekan renders it
    python3 config/gen_chord_gnss_config.py --base config/chord/pathfinder.j2 --node cx19 \\
        --telem-host HOST --cube-host HOST --out config/generated/chord_gnss_cx19.yaml

The fleet's full flag set lives in config/gnss_fleet_chord.yaml; scripts/gnss/gen_fleet.py
regenerates every node config from it, and its --check verifies them without writing.
--aggregator-instance, --gather-instance and --cube-archive-instance emit the off-node search,
telemetry-gather and beam-cube-archiver instances instead of a node config.

The production science pipeline (correlator, RFI, eigen) runs unchanged beside the GNSS
branch. Its bufferSend legs are dropped unless --n2-send, which keeps the N2 legs and the
bad-feed-mask legs and stamps their endpoint (default 10.222.0.51); --n2-send-legs picks which
N2 legs.

    --rest-port N   A node config defaults to runtime.rest_port from the node table
                    (12048): our kotekan replaces production's on these nodes, and
                    Earth-orientation and bad-input updates arrive as REST POSTs to that
                    port. The off-node instances default to their own ports.
"""

import argparse
import copy
import datetime
import glob
import hashlib
import json
import os
import shlex
import sys
import textwrap
import time

import yaml

CONF = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CONF)

from chord_band_plan import (
    covering_channels,  # noqa: E402
    node_channels,
    signal_table,
)
from gnss_record_layout import (
    cube_frame_bytes,  # noqa: E402
    record_stride,
    telem_frame_bytes,
    telem_max_chan,
    chan_floats,
    prnctl_bytes,
)
import kotekan_src  # noqa: E402  (python/scripts/gnss, on sys.path through chord_band_plan)
import gnss_arraymap  # noqa: E402
from gnss_fleet import broker_chain_name, signal_tag  # noqa: E402
from gnss_broker import iface  # noqa: E402

NODE_FILE = os.path.join(CONF, "chord_gnss_node.yaml")
# The broker's chains file, read only by check_telem_windows.
CHAINS_FILE = os.path.join(CONF, "gnss_chains_chord.yaml")

# ── Fixed settings ───────────────────────────────────────────────────────────────────────────
# Several are wire or alignment contracts that every sender and receiver must agree on: change
# one, then regenerate every node config and the gather and archiver configs together.

# 10.49 ms at CHORD's 5.12 us hop; divides the 8192-hop frame 4 ways and is shorter than the
# 20 ms period of the NH20-tiled codes, so a record straddles at most one code-period boundary
# (where the P_HEAD row splits it).
HOPS_PER_RECORD = 2048
# PRN slots per chain on the synth axis: a chain's lane range is SYNTH_SLOTS * lanes_per_slot.
SYNTH_SLOTS = 32
# The one n2k_dual instantiation (external/n2k_dual: 128 antenna + 384 synthetic stations at
# 384 channels): the shared synth axis is this wide whatever the chains use of it.
DUAL_SYNTH_LANES = 384
# Chains sharing a channel are placed on disjoint lane ranges at this step: whole tile rows
# (16 lanes), the narrower chain width (two lanes per slot), so a 64-lane and a 128-lane chain
# tile the axis without gaps a tile row could not address.
LANE_STEP = 64


def lanes_per_slot_for(args):
    """Synth lanes per PRN slot a chain packs (gnssGpuChain.hpp LANES_FULL / LANES_PROMPT).

    With --epl-from-prompt every chain packs two (P and P_HEAD): its E and L rows are the
    prompt rotated per channel on the host from the rotation the kernel carries in PrnCtl::rot.
    No chain is excepted for holding one or two channels per instance (L2C, L2OC): the code
    loop closes on the lobe-coherent discriminator, every instance's channels rotated onto one
    phase reference and summed across the fleet before squaring (gnss_broker/combdll.py,
    gnssFleetDll.hpp), so the rotation acts on the pooled lobe, where a simulation at the real
    channel geometry puts the rotated discriminator's slope within 0.3% of the true one with
    no zero-crossing bias at 0.4 chips of 511.5 kcps (fixtures/gpu_budget/epl_rotation_sim.py).
    A one-channel instance's own E and L are then the prompt times a phase, which only its
    status page and the polled fallback see; such an instance has no usable discriminator of
    its own with either method.
    """
    return 2 if args.epl_from_prompt else 4


# Frames per GNSS chain buffer.
BUFFER_DEPTH = 4
# The primary chain's PRN slots (each extra chain carries its own list).
PRIMARY_PRNS = list(range(1, 33))
# PRN slots an extra search (--aggregator-extra-search) scans when the spec names none, by
# the signal name's constellation prefix: the broadcast PRN ranges, not a lane budget.
EXTRA_SEARCH_PRNS = {
    "GPS": list(range(1, 33)),
    "GAL": list(range(1, 37)),
    "BDS": list(range(19, 64)),
    "GLO": list(range(1, 25)),
}
# Combiner integration length, records.
INTEGRATION_LENGTH = 100
# /get_spectrum accumulation window, in records. Windows are quantised on the F-engine sample
# clock (index = wstart // (records x hops_per_record x fft_length)), so every instance assigns
# a record to the same window without coordination. 100 records = 1.0486 s, the combiner's
# coherent span (INTEGRATION_LENGTH).
SPECTRUM_WINDOW_RECORDS = 100
# Completed spectrum windows kept per instance (~8.4 s), far longer than the record lag between
# instances, so a laggard can still be asked for a window its peers already returned.
SPECTRUM_RING_DEPTH = 8
# RF-monitor pass: sample every Nth hop (as the element_power sampler does). 256 of 8192 hops
# is 32768 samples per channel across the 128 inputs, plenty for a clip fraction.
RF_STATS_HOP_STRIDE = 32
# Element, relative to the start of the live range, that the acquisition search runs on.
# Single-antenna by design: a healthy, high-gain feed.
SEARCH_ELEMENT = 0

# ElemCal integration time constant (s). The per-element self-cal is warm after ~3 tau of
# continuous lock, and only then combines the array instead of the reference element alone.
# 2 s (warm in ~6 s; the stage default is 5) is short enough to rewarm between re-anchors, at
# the cost of noisier gains.
ELEM_SUM_TAU_S = 2.0
# The shared model's consensus EMA time constant (s), and the per-satellite inter-pol
# coefficient EMA time constant (s).
ELEM_SUM_SHARED_TAU_S = 300.0
ELEM_SUM_POL_TAU_S = 3.0

# Seconds the injector holds a /set_trim value with no refresh before zeroing it. A trim left
# standing by a dead controller is a fixed code offset that the broker's own DLL then fights.
TRIM_TTL_S = 4.0

# The injector's per-hop PFB chip gather, truncated to this many chips centred on the ~210-chip
# PFB span. The prototype peaks mid-span, so 80 centred chips lose ~0.05 dB, where a one-sided
# cap cuts through the peak; a centred window reproduces the full-span code phase down to 60
# chips, so 80 leaves margin (scripts/gnss/e2e.cpp --max-chips --chips-centered). Synthesis
# cost is linear in the chip count: 2.6x less than the full span. The stage accepts only this
# window, with fp16 Phi tables; all three keys are still emitted so an older binary, whose
# defaults are the full span and fp32, runs the same despread.
DESPREAD_MAX_CHIPS = 80

# Telemetry wire contract. One gather port serves the whole fleet: every frame carries its own
# chain and instance tags. 4 records per frame = one 8192-hop correlator frame, 23.84 frames/s,
# batched on the absolute window index wstart / (records x hops x fft_len), so every instance
# batches the same records without coordination. The records per frame set the frame size:
# changing it means regenerating the gather too and restarting both ends.
TELEM_PORT = 11060
TELEM_RECORDS_PER_FRAME = 4
# The gather's local stream to the Python broker.
GATHER_SERVE_HOST = "127.0.0.1"
GATHER_SERVE_PORT = 11061

# GnssFleetTrim (the gather's fast code loop). The trim store lets a restart skip the pull-in;
# a store older than the max age (s) is refused, because a trim corrects the broker's model and
# after a long outage that model is not the one about to be republished. The /get_taps depth
# must match the broker's telem-windows (32, config/gnss_chains_chord.yaml), not the loop's own
# --fleet-trim-windows, which is far too short for the policy cycle's statistics; the
# combiners' record_export is the same depth in records, and check_telem_windows refuses to
# generate when the chains file disagrees. Instances required before a discriminator forms.
FLEET_TRIM_STATE_FILE = "/tmp/gnss_fleet_trim.json"
FLEET_TRIM_STATE_MAX_AGE_S = 300.0
FLEET_TRIM_TAPS_WINDOWS = 32
FLEET_TRIM_MIN_INSTANCES = 2

# Beam-cube wire contract. One archiver port serves the whole fleet: every frame carries its own
# <host>/<stage> address, gpu, absolute window index and wstart. 96 records per window = 24
# correlator frames = 1.00663 s, a whole number of records and of frames, so no record is split
# (no exact 1.000 s window exists at the 195312.5 Hz hop rate). One bin per covering channel, at
# the full 195 kHz resolution. The ring holds 8 windows open (~8 s for out-of-order records).
# CUBE_MAX_PRN x CUBE_MAX_BINS size every sender's frame and the archiver's receive buffer, and
# bufferRecv closes any connection whose frame_size disagrees, so chains with fewer PRN slots or
# covering channels pad with zero rows (the offline converter strips them).
CUBE_PORT = 11070
CUBE_WINDOW_RECORDS = 96
CUBE_RING_DEPTH = 8
CUBE_MAX_PRN = 32
CUBE_MAX_BINS = 8
# Where the archiver writes: bulk storage, not /tmp (cf06's /tmp does not survive a reboot) and
# not NFS home (far smaller). 900 frames per file is ~10 s of a ~90-sender fleet (~91 MB), which
# keeps the file count manageable; rawFileRead derives the frame count from the file size.
CUBE_ARCHIVE_DIR = "/mnt/cs00/data/kvand/gnss_cube/raw"
CUBE_ARCHIVE_FRAMES_PER_FILE = 900

# The aggregator's acquisition search. Threads for the aggregate half (parallel over Doppler
# bins x coarse lags; the affinity set is widened to match, up to the pool). Fine-lag
# decimation: the fine-lag lobe is fft_len / (comb span in bins) samples wide, ~156 at CHORD, so
# a per-sample axis oversamples it; the step must stay well under the lobe width. Detection
# threshold: at 1 window the pure-noise ceiling is a Gamma(1) tail (~19), which the stage logs
# every pass; keep it in step with the broker's acquire-snr.
ACQUIRE_THREADS = 16
ACQUIRE_FINE_STEP = 128
ACQUIRE_SNR = 30.0
# Windows stacked per acquisition attempt. Must stay 1 at CHORD: a search window is 3125 hops =
# 16 L5 primary-code periods, not a whole number of NH20 periods, so each successive window's
# overlay alignment shifts by 4 of the 20 chips, while the stage stacks every window at one
# alignment. Stacking would put the signal in different nh bins and noise in all of them; it
# needs window w routed to bin (a + 4w) mod 20. The stage default is 64, so this must be emitted.
ACQUIRE_WINDOWS = 1

# GnssN2Project: rank cap, covariance EMA time constant (s), the trigger (the first spread
# component's share of the off-diagonal energy, its eigenvalue in units of the mean live auto,
# and a participation ratio 1/sum|q|^4 of at least this many inputs; a lone correlated pair
# scores 2), and frames between solves.
N2_PROJECT_K_MAX = 3
N2_PROJECT_TAU_S = 0.5
N2_PROJECT_FRAC_MIN = 0.4
N2_PROJECT_LAMBDA_MIN = 3.0
N2_PROJECT_PR_MIN = 6.0
N2_PROJECT_SOLVE_EVERY = 4

# Per-GPU chain vars, filled by build_n2dual_branch; merge_n2dual_gpu reads the sink core.
GPU_VARS = {}


def gpu_of_channel(cfg, node, freq_id):
    """Which of the node's two GPUs (0/1) holds this freq_id, and its index in that comb."""
    off16 = cfg["nodes"][node]["gpu_offsets_mod16"]
    for gpu, off in enumerate(off16):
        if freq_id % 16 == off:
            sb = cfg["science_band"]
            first = next(
                f
                for f in range(sb["min_freq_id"], sb["max_freq_id"] + 1)
                if f % 16 == off
            )
            return gpu, (freq_id - first) // 16
    raise SystemExit(f"freq_id {freq_id} is not on {node} (offsets {off16})")


def gpu_pairs_for(cfg, node, gpu, chans):
    """(freq_id, comb index) of the channels in `chans` that live on this GPU, in comb order."""
    out = []
    for fid in chans:
        g, idx = gpu_of_channel(cfg, node, fid)
        if g == gpu:
            out.append((fid, idx))
    return sorted(out)


def dll_spacing_chips(name):
    """Early/Late tap offset from prompt, in the signal's own (component) chips.

    The offset is set by the width of the correlation the loop closes on, not by the chip: E
    and L must straddle that correlation's mainlobe, inside its first null, or the
    discriminator reverses sign off-peak and the loop has stable false locks.

    Every chain but GPS L2C despreads a multi-freq_id comb inside one instance, so its
    correlation is chip-narrow and 0.5 chip sits on the slope. L2C's covering set is six
    freq_ids, fewer than the comb stride, so each instance despreads one 195 kHz channel, whose
    |sinc|^2 nulls at 2.6 CM chips. But the fleet loop sums the senders' complex taps
    coherently, and on L2C the senders are the fleet's contiguous L2C channels, so the loop sees
    an N-channel Dirichlet whose first null is 1/(NB): 0.52 CM chips for the six-node fleet's
    five (6284-6288), 0.44 for the eight-node fleet's six (6283 on cx47). The offset must sit
    inside that combined width, where the discriminator is monotonic (a q far higher than ~6
    means the taps have landed in a null, not that the lock improved). 0.33 puts the six-channel
    taps at the level 0.4 gave five (|R| 0.28 of the peak), and five at 0.46: it holds whether
    or not cx47 reports. At 0.4 six channels put the taps at 0.09, beside the null.

    GLONASS L2OC is BOC(1,1) under a data/pilot TDM: its autocorrelation in 2.046 Mcps chips
    crosses zero at +-0.67 and bottoms at -0.5 at +-1, and its spectrum has a null at the
    carrier. Each instance holds one or two of the lobe's 22 channels; six nodes hold 17 of
    them (~60% of the power), eight all 22 (at 0.4: q 3.8, the sign kept to +-0.82 chip, so
    0.4 serves both). On the 17-channel sum, at 0.4 spacing, q on peak is 4.4 (a
    BPSK chain's is 4 at 0.5, so the fixed 2.2 lock bars read alike), the S-curve is linear to
    +-0.23 chip and keeps its sign to +-0.91, outside which the BOC side peaks (+-1.05 chips)
    are stable false locks. Its slope is 1.8x what the trim law assumes (dll_tau scales a
    triangle's), so the chain runs the fast loop at fleet-trim-bandwidth 1.4 rather than 2.5:
    the same closed loop as the other chains. 0.33 would match the law but cap q at 2.5, under
    the lock bars; 0.5 would put the taps by the zero (q 15, slope 4.2x). Re-derive if the
    node set changes.

    Must match the broker chain's dll-spacing (config/gnss_chains_chord.yaml): the C++ fleet
    loop takes its law's spacing from the arming payload while the taps come from here, so
    cycle the nodes before the broker and the taps lead the law.
    """
    return {"GPS_L2C_CM": 0.33, "GLO_L2OC_P_OC": 0.4}.get(name, 0.5)


# Record slot 2's modulus per tracker code (GnssGpuRecordAssemble record_cp_mod_chips): the
# primary code length, for a baked code long enough that a float32 phase over it loses
# precision. GLO_L2OC_P_OC is 2,046,000 chips (a 0.125-chip ulp); its primary is 40920. The
# broker wraps slot 2 mod the primary anyway (gnss_broker/trkresid.py), so nothing is lost.
RECORD_CP_MOD_CHIPS = {"GLO_L2OC_P_OC": 40920}


# PRN rows per chain on the telemetry wire; every other chain gets _TELEM_ROWS_DEFAULT. See
# telem_rows(). The gather's buffer follows the widest chain (telem_rows_max), so changing a
# row count means regenerating the gather config too.
_TELEM_ROWS = {
    "gal_e5a": 20,
    "gal_e5b": 20,
    "gal_e6": 20,
    "bds_b2a": 24,
    "bds_b2b": 24,
    "bds_b3i": 24,
    "gps_l2c": 18,
}
_TELEM_ROWS_DEFAULT = 14


def telem_rows(chain_name):
    """PRN rows this chain's senders put on the wire.

    A wire capacity, not a PRN list: GnssTelemPack compacts rows onto the PRNs actually
    despread in the window, so this is how many satellites a sender may carry at once. Over it,
    the packer ships the lowest-numbered and drops the rest, and downstream a dropped PRN is
    indistinguishable from one that set.

    It is per chain because the constellations do not seed alike: one number for the fleet
    either truncates the busiest chain or pads every other, and the padding rides a 1 GbE link
    the whole fleet shares. Size each chain to its own live count with a few rows of headroom,
    and watch the packer's overflow WARN, which names the first PRN it dropped.
    """
    return _TELEM_ROWS.get(chain_name, _TELEM_ROWS_DEFAULT)


def telem_rows_max():
    """The most rows any chain can put on the wire: what the gather's buffer must hold.

    Read off the same table rather than off the chains in this run: the gather serves the whole
    fleet, including chains a later node config adds, and bufferRecv closes the connection of a
    sender whose frame exceeds the buffer, so that sender delivers nothing at all.
    """
    return max([_TELEM_ROWS_DEFAULT] + list(_TELEM_ROWS.values()))


def vis_capture_chain(args, chain, sig):
    """True when this chain feeds the (N+M)^2 visibility capture (--vis-capture)."""
    if not getattr(args, "vis_capture", None):
        return False
    name = broker_chain_name(chain["signal"] if chain else sig["primary"])
    return name in [c.strip() for c in args.vis_capture.split(",") if c.strip()]


def n2_tiles_per_chan(n_live, num_synth):
    """Gathered tiles per comb channel in a chain's own tiles frame: the mixed block (synth rows
    x live antenna columns). GnssN2RecordAssemble reads nothing else (its normalization uses the
    true pre-quantization energy from the ctl block); the visibility capture's AA/BB tiles ride
    in a separate per-GPU gather (merge_n2dual_gpu).
    One contract, three sites: this, cudaCorrelatorDual's build_tile_selection and
    GnssN2RecordAssemble's tile count. If they disagree, the assembler's frame-size check
    refuses to start."""
    na16, nt16 = 128 // 16, (128 + num_synth) // 16
    nlive16 = (n_live + 15) // 16
    return (nt16 - na16) * nlive16


def rf_monitor_channels(cfg, node, gpu, args, primary_freq_ids, primary_local):
    """Every channel any chain on this GPU covers, as local frame indices: the RF watch list.

    The union, not the primary's set. The chains span several bands, and what separates "an
    external source is up" from "the sky or the gain moved" is whether the other bands moved
    with it, which a single band cannot show.

    Two axes that look alike: a chain's `chans` are global freq_ids, while the tap indexes its
    frame by local comb position (0..num_local_freq). gpu_pairs_for() converts, and also drops
    channels that live on the other GPU, which a raw set union would keep. Never merge channel
    lists without checking the axis.
    """
    idx = set(primary_local)
    for ch in parse_extra_signals(cfg, args, node, list(primary_freq_ids)):
        idx.update(i for _, i in gpu_pairs_for(cfg, node, gpu, ch["chans"]))
    return sorted(idx)


def parse_extra_signals(cfg, args, node, primary_chans):
    """--extra-signal NAME:PRNS -> chain dicts, with their covering channels resolved.

    The descriptor comes from signal_table() (parsed out of gnssSignal.hpp), so the carrier and
    chip rate that set the covering channels are the same constants the stage builds its
    replica from. A chain whose covering set equals the primary's shares the voltage tap.
    """
    if not args.extra_signal:
        return []
    table = signal_table()
    tags = set()
    chains = []
    for spec in args.extra_signal:
        name, _, prn_s = spec.partition(":")
        if name not in table:
            raise SystemExit(
                f"--extra-signal {name}: unknown signal (not in gnssSignal.hpp)"
            )
        if not prn_s.strip():
            raise SystemExit(
                f"--extra-signal {name}: PRNs are required, e.g. {name}:1,2,3. Constellations "
                "have different PRN ranges and the list fills the chain's PRN slots "
                f"({SYNTH_SLOTS} slots per chain), so it is not something to default silently."
            )
        prns = [int(p) for p in prn_s.split(",") if p.strip()]
        s = table[name]
        tag = signal_tag(name)
        if tag in tags:
            raise SystemExit(
                f"--extra-signal {name}: tag '{tag}' collides with another chain; "
                "every buffer and endpoint is named from it"
            )
        tags.add(tag)
        chans = covering_channels(
            node_channels(cfg, node),
            s["carrier_hz"],
            s["chip_rate_hz"],
            float(cfg["signals"]["max_doppler_hz"]),
            modulation=s["modulation"],
            boc_m=s["boc_m"],
            boc_n=s["boc_n"],
        )
        if not chans:
            # A narrowband signal can legitimately miss a node. A wideband chain should not (L5
            # spans 106 freq_ids, so a node holding none is a band-plan error), but L2C's
            # covering set is six freq_ids, fewer than the 8-id comb stride, so some nodes own
            # none of them. That is not an error: the fleet loop combines over the instances
            # that report, and a PRN an instance never despreads is skipped by the combine. Any
            # chain is skipped with a note on stderr rather than an error; gen_fleet relays it
            # with the node's name.
            sys.stderr.write(
                f"  note: {name} has no covering channel on {node} "
                f"(carrier {s['carrier_hz']/1e6:.2f} MHz, mainlobe "
                f"{2 * s['chip_rate_hz'] / 1e6:.3f} MHz) -- chain SKIPPED on this node\n"
            )
            continue
        chains.append(
            {
                "signal": name,
                "tracker": name,
                "prns": prns,
                "tag": tag,
                "ord": len(chains)
                + 1,  # 0 is the primary chain; drives the core rotation
                "carrier_hz": s["carrier_hz"],
                "chip_rate_hz": s["chip_rate_hz"],
                "chans": chans,
                "shares_tap": chans == primary_chans,
            }
        )

    # Sideband partners must carry the same PRNs. Galileo E5 is one AltBOC(15,10) transmission
    # centred at 1191.795 MHz; E5a (1176.45) and E5b (1207.14) are its lower and upper
    # sidebands, generated coherently by one modulator on one spacecraft, and BeiDou B2
    # (ACE-BOC(15,10), same centre) has the same shape. A PRN present on one sideband is present
    # on the other, so differing lists are a configuration error. They would also leave
    # tau_band (per-band instrumental delay) inseparable from b_sat (per-satellite bias), which
    # needs one ray observed on two carriers, and a sideband seeded with satellites its partner
    # is not tracking reads noise, indistinguishable from a broken chain.
    SIDEBAND_PARTNERS = [("GAL_E5A_Q", "GAL_E5B_Q"), ("BDS_B2A_P", "BDS_B2B_I")]

    def _base(n):
        return n[:-3] if n.endswith("_CS") else n

    by_base = {}
    for c in chains:
        by_base.setdefault(_base(c["signal"]), c)
    for lo, hi in SIDEBAND_PARTNERS:
        a, b = by_base.get(lo), by_base.get(hi)
        if not a or not b:
            continue  # only one sideband configured -- nothing to compare
        if sorted(a["prns"]) != sorted(b["prns"]):
            only_a = sorted(set(a["prns"]) - set(b["prns"]))
            only_b = sorted(set(b["prns"]) - set(a["prns"]))
            raise SystemExit(
                "--extra-signal %s / %s are the two sidebands of one transmission from the "
                "same satellite (Galileo E5 AltBOC, BeiDou B2 ACE-BOC), so their PRN lists "
                "must be identical.\n"
                "  only on %s: %s\n  only on %s: %s\n"
                "A PRN seeded on one sideband and not the other cannot contribute to tau_band, "
                "and a sideband seeded with satellites its partner is not tracking cannot be "
                "told apart from a broken chain."
                % (a["signal"], b["signal"], a["signal"], only_a, b["signal"], only_b)
            )
    return chains


def build_search_leg(cfg, node, gpu, chan_idx, args, freq_ids=None):
    """The search leg for one GPU's covering channels: a one-element voltage tap shipped to the
    aggregator's acquisition search, which detects GPS and so solves the receiver clock every
    chain on the node is reckoned from.

    Also returns the record stride (which the caller prints) and the live element count.
    """
    arr = cfg["array"]
    n_elem = live_element_count(arr)
    elem0 = live_element_ranges(arr)[0][0]
    n_chan = len(chan_idx)
    cores = gnss_cores(cfg["runtime"], gpu)
    pre = f"gnss{gpu}_"

    def core(k):
        return cores[k % len(cores)]

    record_floats = record_stride(n_elem)
    blocks = {
        # A second tap on the same voltage buffer, taking one element: the acquisition search
        # is single-antenna by design, so this is ~57 kB/frame (1.4 MB/s) for 7 channels rather
        # than 1.8 MB/frame for all 32 live elements. Shipped as 4+4b bytes and unpacked on the
        # far side, 8x less on the wire than complex floats.
        f"{pre}srch_buf": {
            "kotekan_buffer": "standard",
            "metadata_pool": "gnss_pool",
            "num_frames": BUFFER_DEPTH,
            "frame_size": f"samples_per_data_set * {n_chan} * 1",
        },
        f"{pre}srch_tap": {
            "kotekan_stage": "GnssChordVoltageTap",
            "in_buf": f"host_voltage_buffer_{gpu}",
            "out_buf": f"{pre}srch_buf",
            "chan_ids": chan_idx,
            "n_elements": 1,
            "element_offset": elem0 + SEARCH_ELEMENT,
            "frame_chan_stride": "num_local_freq",
            "frame_elem_stride": "num_elements",
            "fft_length": cfg["fengine"]["fft_length"],
            "cpu_affinity": [core(gpu + 1)],
            # The RF monitor rides the search tap rather than adding a stage. The tap already
            # drops rather than blocks (so the GNSS branch never back-pressures the F-engine
            # ingest), already holds the whole frame, and already decodes 4+4b for
            # element_power; a second consumer on host_voltage_buffer would be one more stage
            # that must release frames promptly to keep the science pipeline moving.
            #
            # The channel set is the union across chains (rf_monitor_channels): a band-selective
            # source is only diagnosable against a band that is quiet in the same sample, and
            # this tap's own chan_ids are the primary's covering set alone.
            "band_power_chans": rf_monitor_channels(
                cfg, node, gpu, args, freq_ids, chan_idx
            ),
            "band_power_period_s": args.rf_stats_period_s,
            "band_power_hop_stride": RF_STATS_HOP_STRIDE,
        },
        f"{pre}srch_send": {
            "kotekan_stage": "bufferSend",
            "buf": f"{pre}srch_buf",
            "server_ip": args.search_host,
            "server_port": args.search_port_base + gpu,
            # The search must never back-pressure the ingest: acquisition is a bootstrap
            # convenience, the science chain is not.
            "drop_frames": True,
            **send_pacing(
                cfg,
                cfg.get("samples_per_data_set", 8192),
                int(cfg.get("samples_per_data_set", 8192)) * n_chan,
            ),
            "reconnect_time": 5,
            # Bounds the sender's exposure to a wedged receiver. bufferSend's default is 20 s:
            # an aggregator that accepts the connection and stops reading parks this stage in
            # send() that long per frame while the buffer fills and drops. 2 s is ~48 frames.
            "send_timeout": 2,
            # Pinned on both ends: bufferSend and bufferRecv default use_config_tracker to
            # whether the instance has a /config_tracker block, which the node inherits from the
            # production base and the search instance lacks. Left to default, the two disagree
            # on header length and frame_size is read from the wrong field ("Frame size does not
            # match between server: 57344 and client: 12").
            "use_config_tracker": False,
            "cpu_affinity": [core(gpu + 3)],
        },
    }
    return blocks, record_floats, n_elem


def gnss_chain_vars(cfg, node, gpu, chan_idx, freq_ids, args, spds, chain=None):
    """The variable data for one chain on one GPU: channel and PRN lists, the CPU-core
    rotation, and frame sizes. build_n2dual_branch reads the core rotation and the ctl/epl
    frame sizes from here, and merge_n2dual_gpu the sink core (via GPU_VARS). Everything here
    must be a property of the chain, not a detail of how a block is assembled.
    """
    sig, rt, arr = cfg["signals"], cfg["runtime"], cfg["array"]
    cores = gnss_cores(rt, gpu)  # this GPU's own NUMA node (see gnss_cores)
    nc = len(cores)
    tag = chain["tag"] if chain else ""
    n_chan = len(chan_idx)
    prns = chain["prns"] if chain else PRIMARY_PRNS
    n_prn = len(prns)
    ordv = chain["ord"] if chain else 0
    _aslots = [
        int(x)
        for x in str(getattr(args, "assembler_core_slots", "2") or "2").split(",")
    ]
    n_live = live_element_count(arr)
    track_signal = chain["tracker"] if chain else sig.get("tracker", sig["primary"])
    lanes_per_slot = lanes_per_slot_for(args)
    num_synth = lanes_per_slot * SYNTH_SLOTS

    # --- frame sizes (ctl/epl: gnssGpuChain.hpp frame_bytes, ROWS_PLAIN = 4) ------------------
    n_rec_per_frame = max(1, int(spds) // HOPS_PER_RECORD)
    tiles_frame_bytes = (
        n_rec_per_frame * n_chan * n2_tiles_per_chan(n_live, num_synth) * 512 * 4
    )
    prnctl = prnctl_bytes()  # sizeof(PrnCtl), parsed from gnssGpuChain.hpp
    epl_bytes = (
        48
        + 8 * 16
        + prnctl * 16 * n_prn
        + 16 * 4 * n_prn * 16 * n_chan * n_live
        + 8 * 4 * n_prn * 16 * n_chan
    )
    ctl_bytes = 48 + 8 * 16 + prnctl * 16 * n_prn + 8 * 4 * n_prn * 16 * n_chan
    record_floats = record_stride(n_live)  # parsed from gnssRecord.hpp

    # --- the CPU-core rotation ----------------------------------------------------------
    # dual, tiles and combine step 3*ordv through the pool so chains spread across cores; the
    # other stages are per GPU only.
    return {
        "tag": tag,
        # The broker-facing chain label (gps_l5, gal_e5a, ...), from the signal via
        # broker_chain_name.
        "chain": broker_chain_name(chain["signal"] if chain else sig["primary"]),
        # The tracked signal descriptor (GPS_L5_Q_NH, GAL_E5A_Q_CS, ...): the label names the
        # chain, this names the code being despread.
        "signal": chain["tracker"] if chain else sig.get("tracker", sig["primary"]),
        "channel_ids": list(freq_ids),
        "local_channels": list(chan_idx),
        "prns": list(prns),
        "f_offset_hz": float(chain["carrier_hz"])
        if chain
        else float(sig["carrier_hz"]),
        "cores": {
            # Distinct per chain within this GPU's pool: see dual_core().
            "dual": dual_core(rt, gpu, ordv),
            "tiles": cores[(gpu + 9 + 3 * ordv) % nc],
            # One slot for every chain by default (slot 2): at 96 elements the ten assemblers
            # of a GPU fill that core (cx51 10-20: ~99%, 5-9% each) while the other slots
            # idle at 10-15%, so --assembler-core-slots rotates them like the combiners.
            "assemble": cores[(gpu + _aslots[ordv % len(_aslots)]) % nc],
            # The combiner is the heaviest host stage (its cost scales with PRN count), so it
            # rotates per chain like dual and tiles: stacked on one core, the chains' combiners
            # together approach a full core, and any backlog walks back through the shared
            # voltage ring into the F-engine ingest (and the science N2 with it).
            "combine": cores[(gpu + 4 + 3 * ordv) % nc],
            "telem": cores[(gpu + 5) % nc],
            "send": cores[(gpu + 6) % nc],
            "sink": cores[(gpu + 7) % nc],
        },
        "lanes": {"per_slot": lanes_per_slot, "num_synth": num_synth},
        "sizes": {
            "tiles": tiles_frame_bytes,
            "ctl": ctl_bytes,
            "epl": epl_bytes,
            # rec carries the per-channel comb rows (n_prn * n_chan * chan_floats()) that the
            # assembler appends after the PRN records for /get_spectrum and the fleet delay fit;
            # cmb, the combiner's output, does not.
            "rec": (
                f"{n_prn} * {record_floats} * sizeof_float32"
                f" + {n_prn} * {n_chan} * {chan_floats()} * sizeof_float32"
            ),
            "cmb": f"{n_prn} * {record_floats} * sizeof_float32",
            "telem": telem_frame_bytes(
                TELEM_RECORDS_PER_FRAME,
                telem_rows(broker_chain_name(sig["primary"])),
                n_chan,
            ),
        },
    }


def gnss_cores(rt, gpu):
    """The core pool for this GPU's stages: runtime.cpu_affinity_gpu<N>, cores on the GPU's own
    NUMA node (GPU0 is NUMA node 0, GPU1 node 1; `nvidia-smi topo -m`). One pool for both GPUs
    would drive one of them across the interconnect and put both GPUs' cudaProcess stages on
    the same cores; two cudaProcess stages must never share a logical cpu.
    """
    return rt["cpu_affinity_gpu%d" % gpu]


def load_summary(out, cfg, gpus):
    """The three busiest cores among the config's gnss* stages, and the size of the pool they
    are drawn from: the union of these GPUs' runtime.cpu_affinity_gpu<N>."""
    load = {}
    for k, v in out.items():
        if k.startswith("gnss") and isinstance(v, dict) and "cpu_affinity" in v:
            for c in v["cpu_affinity"]:
                load.setdefault(c, []).append(v.get("kotekan_stage", "?"))
    busiest = sorted(load.items(), key=lambda kv: -len(kv[1]))[:3]
    pool = {c for g in gpus for c in gnss_cores(cfg["runtime"], g)}
    return (
        ", ".join(f"core {c}: {len(v)} stages" for c, v in busiest)
        + f"  (pool of {len(pool)})"
    )


def dual_core(rt, gpu, ordv):
    """The cudaProcess core for chain `ordv` on `gpu`: stride 3 through this GPU's pool, so with
    a pool length coprime to 3 (10 in the node table) up to that many chains land on distinct
    logical cpus. merge_n2dual_gpu folds every chain on a GPU into one cudaProcess on the
    primary chain's core (ordv 0), so only that core is emitted.
    """
    pool = gnss_cores(rt, gpu)
    return pool[(3 * ordv) % len(pool)]


def cube_assembler_keys(args, cfg, gpu, pre):
    """Beam-cube keys for the record assembler: the (subband x element) axis, uncollapsed,
    pushed to the archiver.

    Everything else GnssGpuRecordAssemble emits has collapsed one of the two axes (element
    blocks are summed over channels, the comb over elements), and a beam map needs both.

    The window length is in F-engine samples, the same clock as wstart, and is derived rather
    than chosen, as spectrum_window_samples is: the window index is floor(wstart / win_samples)
    in integer arithmetic, so instances given different values would split windows differently
    and lose the cross-instance alignment. records x hops_per_record x fft_length puts every
    boundary on a record boundary, so no record is split and the record count per window is
    constant. It is set explicitly although it equals the stage default, because it must match
    across instances.
    """
    return {
        "beam_cube": True,
        "beam_cube_window_samples": (
            CUBE_WINDOW_RECORDS * HOPS_PER_RECORD * int(cfg["fengine"]["fft_length"])
        ),
        "beam_cube_ring_depth": CUBE_RING_DEPTH,
        # Uniform across the fleet, never per chain: see CUBE_MAX_PRN.
        "beam_cube_max_prn": CUBE_MAX_PRN,
        "beam_cube_max_bins": CUBE_MAX_BINS,
        "beam_cube_gpu": gpu,
        # beam_cube_chain is left to the stage, which builds "<short hostname>/<stage>" itself;
        # generated here, a config copied to another node would keep the old name and mislabel
        # the archive at the source.
        "cube_buf": f"{pre}cube_buf",
    }


def live_element_ranges(arr):
    """[(lo, hi), ...] inclusive element ranges that carry data, from the array config.

    The live set is not contiguous: the production dpdk config's crs_board_remap groups
    polarizations, so the four dish CRS boards land at output slots 0, 1, 8, 9 and the live
    elements are {0..15, 64..79}: `live_element_ranges: [[0, 15], [64, 79]]`.

    The assembler's shared model splits the element axis at n/2 into pol 0 and pol 1, so the
    ranges must be exactly one per pol, the same dishes in both: [lo, hi] and [lo + 64, hi + 64].
    Anything else would put one pol's elements in the other's half with nothing to tell.
    """
    r = [(int(a), int(b)) for a, b in arr["live_element_ranges"]]
    if (
        len(r) != 2
        or not (0 <= r[0][0] <= r[0][1] < 64)
        or r[1] != (r[0][0] + 64, r[0][1] + 64)
    ):
        raise SystemExit(
            f"live_element_ranges: {r} must be one range per pol over the same dishes, "
            "[lo, hi] then [lo + 64, hi + 64] -- the shared model splits the elements at n/2"
        )
    return r


def live_element_count(arr):
    return sum(hi - lo + 1 for lo, hi in live_element_ranges(arr))


def live_tile_columns(arr, tile=16):
    """The 16-element tile columns the live ranges cover, in element order.

    Each range must be tile-aligned at both ends: a half-covered column would put dead inputs
    inside a gathered tile with nothing to mark them (a silent -3 dB), so it is an error, not a
    rounding.
    """
    cols = []
    for lo, hi in live_element_ranges(arr):
        if lo % tile or (hi + 1) % tile:
            raise SystemExit(
                f"live_element_ranges: [{lo}, {hi}] is not {tile}-aligned; "
                "the mixed-tile gather cannot express a partial column"
            )
        cols += list(range(lo // tile, (hi + 1) // tile))
    if len(cols) != len(set(cols)):
        raise SystemExit(f"live_element_ranges: overlapping tile columns {cols}")
    return cols


def elem_positions_from_arraymap(arr, n_elem):
    """Measured element positions: flat [n_elem][3] ENU metres from the array-epoch table
    (config/chord_array_epochs.json via gnss_arraymap) for the epoch valid now, in the
    assembler's element order, which is the epoch's cube_order and must equal the config's live
    element ranges. Returns (positions, epoch).

    The epoch table records what the array actually was, by date, and refuses rather than
    extrapolates. A position the table does not know is a refusal too: a steer built from a
    guessed position is a wrong steer, not a weaker one.
    """
    ep = gnss_arraymap.at(time.time())
    live = [e for lo, hi in live_element_ranges(arr) for e in range(lo, hi + 1)]
    if list(ep.cube_order) != live:
        raise SystemExit(
            "array-epoch positions: epoch %s cube_order %s != the config's "
            "live elements %s -- the assembler's element axis would be mislabelled"
            % (ep.name, ep.cube_order, live)
        )
    if len(live) != n_elem:
        raise SystemExit(
            "array-epoch positions: %d live elements, assembler has %d"
            % (len(live), n_elem)
        )
    out = []
    for cube_i, e in ep.cube_elements():
        if len(e.enu_m) != 3:
            raise SystemExit(
                "array-epoch positions: element %d (%s) has no enu_m in "
                "epoch %s" % (e.index, e.label(), ep.name)
            )
        out += [float(v) for v in e.enu_m]
    return out, ep


def n2_project_live_ids(args):
    """The freq_ids GnssN2Project writes: --n2-project-live-freq-ids, 'exempt' meaning the
    --rfi-excision-exempt-freq-ids ranges. Empty when neither is given."""
    spec = args.n2_project_live_freq_ids
    if spec == "exempt":
        spec = args.rfi_excision_exempt_freq_ids
    ids = set()
    for part in str(spec or "").split(","):
        if part.strip():
            a, _, b = part.strip().partition("-")
            ids.update(range(int(a), int(b or a) + 1))
    return ids


def lane_transfer_keys(cfg, node, gpu, tag, freq_ids, args):
    """--lane-transfer TAG:LO-HI (MHz, repeatable): the chain tagged TAG also publishes its
    lane directions on this GPU's channels centred in [LO, HI] that GnssN2Project writes, less
    its own (#190 tier 1.5: the same satellites' other signals, e.g. the Galileo E6 PRS lobe
    beside E6-B/C). Needs --elem-dir-publish all; empty otherwise."""
    if not args.lane_transfer or args.elem_dir_publish != "all" or not tag:
        return {}
    live = n2_project_live_ids(args)
    off = cfg["nodes"][node]["gpu_offsets_mod16"][gpu]
    own = set(freq_ids)
    out = set()
    for spec in args.lane_transfer:
        t, _, band = spec.partition(":")
        lo, _, hi = band.partition("-")
        if t.strip("_") != tag.strip("_"):
            continue
        for fid, centre_hz, _width in node_channels(cfg, node):
            if (
                float(lo) <= centre_hz / 1e6 <= float(hi)
                and fid % 16 == off
                and fid in live
                and fid not in own
            ):
                out.add(fid)
    return {"lane_transfer_freq_ids": sorted(out)} if out else {}


def elem_proj_keys(args):
    """Assembler keys for the bright-satellite projection (gnssProjSubspace.hpp); nothing unless
    --elem-proj-mode is shadow or live, so a fleet without it is byte-identical. The mode is
    also a live switch (POST /set_elem_proj); this key makes it survive a restart."""
    mode = getattr(args, "elem_proj_mode", "off") or "off"
    out = {}
    # #190: the lane directions GnssN2Project reads (source lanes); off emits nothing. It
    # does not need the bright-satellite projection on (the rows are then raw).
    if getattr(args, "elem_dir_publish", "off") == "all":
        out["elem_dir_publish"] = "all"
    if mode not in ("shadow", "live"):
        return out
    out["elem_proj_mode"] = mode
    if getattr(args, "elem_proj_deg", None) is not None:
        out["elem_proj_deg"] = args.elem_proj_deg
    return out


def elem_shared_keys(args):
    """Assembler keys for the shared element model; nothing unless --elem-sum-shared, so
    a fleet without it is byte-identical."""
    if not getattr(args, "elem_sum_shared", False):
        return {}
    out = {
        "elem_sum_shared": True,
        "elem_sum_adapt": False,
        "elem_sum_shared_tau_s": ELEM_SUM_SHARED_TAU_S,
        "elem_sum_pol_tau_s": ELEM_SUM_POL_TAU_S,
    }
    if getattr(args, "elem_freeze_deg", None) is not None:
        out["elem_sum_shared_freeze_deg"] = args.elem_freeze_deg
    return out


def elem_shared_ref_keys(args, band, steer, n_elem, bad=()):
    """One band's entry of the fleet reference snapshot (python/scripts/gnss/elem_shared_ref.py
    snapshot) for the shared model's phase pin; nothing unless --elem-shared-ref-mode is live
    and --elem-sum-shared is on, so a fleet without it is byte-identical. The snapshot is of the
    steered instrument, so it is refused unless it was taken against the epoch these positions
    came from. The flagged inputs' entries go in as 0: the pin's offset is scale-free
    (gnssSharedPin.hpp), so a zero entry weighs nothing."""
    mode = getattr(args, "elem_shared_ref_mode", "off") or "off"
    if mode == "off" or not getattr(args, "elem_sum_shared", False):
        return {}
    if not args.elem_shared_ref:
        raise SystemExit(
            "--elem-shared-ref-mode %s needs --elem-shared-ref FILE" % mode
        )
    snap = json.load(open(args.elem_shared_ref))
    epoch = steer.get("elem_positions_epoch")
    if snap.get("epoch") != epoch:
        raise SystemExit(
            "--elem-shared-ref %s was taken against epoch %s, these positions are %s "
            "-- take a new snapshot (elem_shared_ref.py snapshot)"
            % (args.elem_shared_ref, snap.get("epoch"), epoch)
        )
    entry = snap["bands"].get(band)
    if entry is None:
        print(
            "note: %s has no '%s' entry: that band keeps its own pin"
            % (args.elem_shared_ref, band),
            file=sys.stderr,
        )
        return {}
    if len(entry["ref"]) != n_elem:
        raise SystemExit(
            "--elem-shared-ref %s: band %s has %d elements, the assembler has %d"
            % (args.elem_shared_ref, band, len(entry["ref"]), n_elem)
        )
    return {
        "elem_sum_shared_ref": [
            0.0 if e in bad else round(float(x), 6)
            for e, pair in enumerate(entry["ref"])
            for x in pair
        ],
        "elem_sum_shared_ref_mode": mode,
    }


def elem_steer_keys(args, arr, n_elem):
    """The assembler keys that arm steering for one band: the measured positions of the array
    epoch valid now (refined by --elem-positions-file), the measured sign, and the key the
    positions came from, so a config can be matched to the array it was built for."""
    pos, ep = elem_positions_from_arraymap(arr, n_elem)
    key = ep.key()
    if args.elem_positions_file:
        # A refinement on top of the epoch, solved from steered per-element phases. A file
        # with `pointing` holds one position per dish, solved on that pointing: it holds for
        # any epoch on it, and is keyed by the epoch's ref_key (pointing and wiring), which a
        # health change does not move. A file with `epoch_key` is in cube order against one
        # epoch key, and is valid only for that epoch.
        ref = json.load(open(args.elem_positions_file))
        if "pointing" in ref:
            if ref["pointing"] != ep.pointing.name:
                raise SystemExit(
                    "--elem-positions-file %s was solved on pointing %s, the epoch valid "
                    "now (%s) is on %s -- refusing"
                    % (
                        args.elem_positions_file,
                        ref["pointing"],
                        ep.name,
                        ep.pointing.name,
                    )
                )
            by_dish = ref["positions_by_dish"]
            missing = sorted(
                {e.dish for _, e in ep.cube_elements() if str(e.dish) not in by_dish}
            )
            if missing:
                raise SystemExit(
                    "--elem-positions-file %s has no position for dish(es) %s -- refusing"
                    % (args.elem_positions_file, missing)
                )
            pos = [
                float(v) for _, e in ep.cube_elements() for v in by_dish[str(e.dish)]
            ]
            key = ep.ref_key()
        else:
            live = [e for lo, hi in live_element_ranges(arr) for e in range(lo, hi + 1)]
            if ref.get("epoch_key") != key:
                raise SystemExit(
                    "--elem-positions-file %s was solved against epoch %s, the epoch "
                    "valid now is %s -- refusing"
                    % (args.elem_positions_file, ref.get("epoch_key"), key)
                )
            if list(ref.get("cube_order", [])) != live:
                raise SystemExit(
                    "--elem-positions-file cube_order != the config's live elements"
                )
            pos = [float(v) for row in ref["positions_enu"] for v in row]
        if len(pos) != 3 * n_elem:
            raise SystemExit(
                "--elem-positions-file: %d positions, assembler has %d elements"
                % (len(pos) // 3, n_elem)
            )
        key = "%s+%s" % (key, os.path.basename(args.elem_positions_file))
    return {
        "elem_positions_enu": pos,
        "elem_steer_sign": args.elem_steer_sign,
        "elem_positions_epoch": key,
    }


# Every push leg (telemetry, cube, search tap) is paced at this multiple of its mean rate; see
# send_pacing. Pacing is the remedy that needs no hardware; a faster link into the receiver
# would make it unnecessary.
SEND_PACING_FACTOR = 2.0


def send_pacing(cfg, spds, frame_bytes, frames_per_fengine_frame=1.0):
    """bufferSend `max_pacing_rate_mbps` for a leg that emits `frame_bytes` per F-engine frame
    (or that fraction of one): SEND_PACING_FACTOR x its mean rate.

    Every sender in the fleet emits on the same F-engine frame clock, so unpaced they all burst
    into the receiver's 1 GbE link in the same millisecond, and the switch drops what its egress
    queue cannot hold; each loss is a ~200 ms TCP retransmit stall, and a telemetry frame that
    stalls reaches the gather after its window has closed. At 2x, a frame spreads over half its
    period and the aggregate at the link is the mean, which the link carries. The cap is per
    connection, so a leg that fell behind catches up at (factor - 1) x its mean rate, which is
    why this is a factor and not a rate.
    """
    frame_s = (
        float(spds)
        * float(cfg["fengine"]["fft_length"])
        / (float(cfg["fengine"]["sampling_rate_MHz"]) * 1e6)
    )
    mean_mbps = frame_bytes * frames_per_fengine_frame * 8.0 / frame_s / 1e6
    return {"max_pacing_rate_mbps": round(mean_mbps * SEND_PACING_FACTOR, 3)}


def build_n2dual_branch(cfg, node, gpu, chan_idx, freq_ids, args, spds, chain=None):
    """One GNSS chain (path B) on one GPU: its cudaGnssInject and cudaCorrelatorDual commands
    (folded with the GPU's other chains into one cudaProcess by merge_n2dual_gpu) and its host
    stages: tiles assembler, record assembler, combiner, and the telemetry and beam-cube legs.

    One chain per signal. `chain` is None for the primary and a parse_extra_signals() entry
    otherwise; every buffer, GPU array and endpoint is tagged from it. The per-chain GPU array
    names must be tagged: get_gpu_memory_array is scoped per device, so two chains on one GPU
    would share an untagged array (silently if the sizes match, as a fatal allocation error if
    they do not).

    Consumes the GPU voltage ring that production's run_send_voltage feeds. The extended
    triangle stays on the GPU (production's run_n2k computes the science N^2); the gathered
    GNSS tiles come to the host for the assemblers, and the combined records end in a
    dropAllFrames.
    """
    sig = cfg["signals"]
    arr = cfg["array"]
    tag = chain["tag"] if chain else ""
    pre = f"gnss{gpu}{tag}_"
    n_chan = len(chan_idx)
    prns = chain["prns"] if chain else PRIMARY_PRNS
    n_prn = len(prns)
    # The replica's sky carrier, from the chain, never the primary: a wrong f_offset builds
    # every replica in the wrong part of the band and every mixed tile reads noise.
    carrier_hz = float(chain["carrier_hz"]) if chain else float(sig["carrier_hz"])
    track_signal = chain["tracker"] if chain else sig.get("tracker", sig["primary"])
    n_live = live_element_count(arr)

    # Tile count per channel: see n2_tiles_per_chan (one contract, three sites). The lane
    # range is the chain's slots times its lanes per slot (lanes_per_slot_for).
    lanes_per_slot = lanes_per_slot_for(args)
    num_synth = lanes_per_slot * SYNTH_SLOTS
    # A frame is samples_per_data_set hops (spds, from the base config, which production owns),
    # split into records of HOPS_PER_RECORD hops.
    n_rec_per_frame = max(1, int(spds) // HOPS_PER_RECORD)
    # The tiles frame's time axis is records per frame: the dual correlator integrates one
    # record (sub_integration_ntime below).
    tiles_frame_bytes = (
        n_rec_per_frame * n_chan * n2_tiles_per_chan(n_live, num_synth) * 512 * 4
    )

    if lanes_per_slot * n_prn > num_synth:
        raise SystemExit(
            f"chain tag '{tag}': {lanes_per_slot}*{n_prn} lanes for its PRNs exceed its "
            f"{num_synth} synthetic lanes ({lanes_per_slot} lanes per PRN slot, "
            f"{SYNTH_SLOTS} slots per chain). Shrink this chain's PRN list."
        )

    ring = f"host_voltage_ringbuffer_{gpu}"
    tiles_buf = f"{pre}n2tiles_buf"
    ctl_buf = f"{pre}n2ctl_buf"
    epl_buf = f"{pre}n2epl_buf"
    n2rec_buf = f"{pre}n2rec_buf"
    # The ctl and epl frame sizes and the core rotation come from gnss_chain_vars, which reads
    # sizeof(PrnCtl) from gnssGpuChain.hpp rather than restating it. The epl frame is
    # gnss_gpu::frame_bytes(n_prn, n_chan, ROWS_PLAIN=4, ...), the layout GnssN2RecordAssemble
    # emits and GnssGpuRecordAssemble reads.
    V = gnss_chain_vars(cfg, node, gpu, chan_idx, freq_ids, args, spds, chain)
    GPU_VARS.setdefault(gpu, []).append(V)
    epl_bytes, ctl_bytes = V["sizes"]["epl"], V["sizes"]["ctl"]
    # Parsed from gnssRecord.hpp, never restated.
    record_floats = record_stride(n_live)
    steer = elem_steer_keys(args, arr, n_live)

    blocks = {
        tiles_buf: {
            "kotekan_buffer": "standard",
            "metadata_pool": "gnss_pool",
            "num_frames": BUFFER_DEPTH,
            "frame_size": tiles_frame_bytes,
        },
        # The chain's injector and correlator, as a process block that merge_n2dual_gpu folds
        # into the GPU's one gnss<g>_n2dual (it takes the primary chain's stage, gpu_id,
        # cpu_affinity and in_buffers, and every chain's two commands).
        f"{pre}n2dual": {
            "kotekan_stage": "cudaProcess",
            "gpu_id": gpu,
            # Two cudaProcess stages must never share a core (the light host stages may): see
            # dual_core().
            "cpu_affinity": [V["cores"]["dual"]],
            "in_buffers": {"host_voltage_ringbuffer": ring},
            "commands": [
                # Order matters: the injector's pack runs on the same stream before the
                # correlator's kernel reads the synth array, and that is all the
                # synchronization there is (the synth array has no ring semantics).
                {
                    "name": "cudaGnssInject",
                    # fp16 Phi tables and the centred chip window: see DESPREAD_MAX_CHIPS. The
                    # stage refuses any other setting.
                    "phi_fp16": True,
                    "despread_max_chips": DESPREAD_MAX_CHIPS,
                    "despread_chips_centered": True,
                    "voltage_name": "voltage",
                    # The F-engine output is conjugated relative to the decode, and the N^2
                    # kernel's antenna input is production's, so the pack conjugates the replica
                    # instead; without it every mixed tile reads noise. The stage refuses false.
                    "conjugate": True,
                    "num_synth": num_synth,
                    "lanes_per_slot": lanes_per_slot,
                    # Local frame indices of the comb (the tap's chan_ids), never freq_ids.
                    "gnss_local_channels": chan_idx,
                    # The state class, cudaGnssChordTrackState: seeds, trims and slot
                    # membership, on this chain's own endpoints.
                    "prns": prns,
                    "n_channels": n_chan,
                    "n_elements": 1,  # unused by the injector; the state class requires it
                    "channel_ids": freq_ids,  # global bins, local order (replica sky freq)
                    "signal": track_signal,
                    # E/L tap offset in the signal's chips, set by the channel geometry: see
                    # dll_spacing_chips. Must match the broker chain's dll-spacing.
                    "dll_spacing": dll_spacing_chips(track_signal),
                    # The E/L lanes as rotations of the prompt, one chip walk per hop
                    # (lanes_per_slot_for).
                    **({"epl_from_prompt": True} if args.epl_from_prompt else {}),
                    # Replicas from code-phase tables, fused with the pack (--psi-synth).
                    **({"psi_synth": True} if args.psi_synth else {}),
                    "f_offset_hz": carrier_hz,
                    "hops_per_record": HOPS_PER_RECORD,
                    "fft_length": cfg["fengine"]["fft_length"],
                    "sample_rate": float(cfg["fengine"]["sampling_rate_MHz"]) * 1e6,
                    # Per-chain GPU arrays (see the docstring); merge_n2dual_gpu replaces the
                    # synth name with the GPU's shared synth array.
                    "gnss_synth_name": f"{pre}synth",
                    "gnss_ctl_name": f"{pre}n2ctl",
                    "seed_endpoint": f"/{pre}inject/set_seeds",
                    # Live PRN membership (docs/CHORD_LIVE_PRN_RECONFIG.md). GET is a
                    # diagnostic ("which satellite is slot 7?"); POST is the broker's actuator.
                    "get_prns_endpoint": f"/{pre}inject/get_prns",
                    "set_prns_endpoint": f"/{pre}inject/set_prns",
                    # No effect: the stage reads it only so a usage report does not list it.
                    "trim_endpoint": f"/{pre}inject/get_trim",
                    # GnssFleetTrim's actuator, per instance: every injector in this process
                    # shares one REST server, so a shared path would apply one chain's trims to
                    # another chain's PRNs.
                    "set_trim_endpoint": f"/{pre}inject/set_trim",
                    "trim_ttl_s": TRIM_TTL_S,
                    # A pinned F-engine epoch (the node file's frame0_utc, or --frame0-nano)
                    # overrides the telescope's GPS time0 as the record epoch. Records are
                    # stamped frame0_utc + wstart/rate, and GnssGpuRecordAssemble stops on a
                    # frame without one.
                    **(
                        {"frame0_utc": float(cfg["fengine"]["frame0_utc"])}
                        if cfg["fengine"].get("frame0_utc")
                        else {}
                    ),
                },
                {
                    "name": "cudaCorrelatorDual",
                    # One visibility per record, not per frame, so each tile slice is one
                    # record the injector synthesised. Over a frame-length window the
                    # per-record carrier-rate phase ramp would sit inside the coherent sum, out
                    # of reach of the combiner's rate search; the records also land on the
                    # hops_per_record grid the fleet DLL groups by.
                    "sub_integration_ntime": HOPS_PER_RECORD,
                    "voltage_name": "voltage",
                    "rfi_RFImask_name": "rfi_RFImask",  # ignored; older binaries require it
                    "gnss_synth_name": f"{pre}synth",
                    # The kernel gets a constant all-good mask, which matches production's own
                    # mask wherever first-stage excision is off or exempt (the fleet's
                    # --rfi-excision-exempt-freq-ids covers the GNSS lobes).
                    "rfi_all_pass": True,
                    # The launch covers only the comb channels and only the mixed and synthetic
                    # blocks (AA only for a capture gather); production's run_n2k computes the
                    # science N^2. The stage refuses false here, as for rfi_all_pass and
                    # gnss_synth_compact; older binaries default all three to false.
                    "gnss_freq_map": True,
                    # Ignored by the stage. Older binaries allocate a GPU array <name>_buffer
                    # sized from their own station count, so the name must differ from
                    # production cudaCorrelator's: a size mismatch is fatal at startup.
                    "n2k_correlation_name": f"{pre}n2corr",
                    "gnss_tiles_name": f"{pre}tiles",
                    "num_synth": num_synth,
                    "num_live_elements": n_live,
                    # Which columns, not just how many: the live set is not contiguous (see
                    # live_element_ranges).
                    "live_element_tiles": live_tile_columns(arr),
                    "gnss_local_channels": chan_idx,
                },
            ],
        },
        ctl_buf: {
            "kotekan_buffer": "standard",
            "metadata_pool": "gnss_pool",
            "num_frames": BUFFER_DEPTH,
            "frame_size": ctl_bytes,
        },
        epl_buf: {
            "kotekan_buffer": "standard",
            "metadata_pool": "gnss_pool",
            "num_frames": BUFFER_DEPTH,
            "frame_size": epl_bytes,
        },
        n2rec_buf: {
            "kotekan_buffer": "standard",
            "metadata_pool": "gnss_pool",
            "num_frames": BUFFER_DEPTH,
            # PRN records plus the unsummed comb (chan_export, gnssRecord.hpp), appended after
            # the records so a consumer that reads only the records indexes them unchanged.
            # The same expression as gnss_chain_vars' rec.
            "frame_size": (
                f"{n_prn} * {record_floats} * sizeof_float32"
                f" + {n_prn} * {n_chan} * {chan_floats()} * sizeof_float32"
            ),
            # The newest record frame stays readable over /buffer_frame without a rawFileWrite.
            "peek_hold": True,
        },
        # Tiles + control block -> the epl frame GnssGpuRecordAssemble reads.
        f"{pre}n2assemble_tiles": {
            "kotekan_stage": "GnssN2RecordAssemble",
            "ctl_buf": ctl_buf,
            "tiles_buf": tiles_buf,
            "out_buf": epl_buf,
            "num_elements": 128,
            "num_synth": num_synth,
            "lanes_per_slot": lanes_per_slot,
            "n_live_elements": n_live,
            "n_gnss_channels": n_chan,
            "n_prn": n_prn,
            "hops_per_record": HOPS_PER_RECORD,
            "cpu_affinity": [V["cores"]["tiles"]],
        },
        # epl frames -> records, plus the per-channel spectrum and the beam cube.
        f"{pre}n2assemble": {
            "kotekan_stage": "GnssGpuRecordAssemble",
            "in_buf": epl_buf,
            "out_buf": n2rec_buf,
            "prns": prns,
            "n_elements": n_live,
            **(
                {"record_cp_mod_chips": float(RECORD_CP_MOD_CHIPS[track_signal])}
                if track_signal in RECORD_CP_MOD_CHIPS
                else {}
            ),
            # The global freq_ids the injector despreads, in local order. They label each
            # channel's sky frequency for /get_spectrum, the comb export and steering: a delay
            # is a phase ramp across frequency, which the cross-channel sum destroys.
            "channel_ids": freq_ids,
            # In F-engine samples and identical on every instance, so every instance assigns a
            # record to the same window (see SPECTRUM_WINDOW_RECORDS). Derived from the record
            # geometry, like the beam-cube window, so window boundaries land on record
            # boundaries and the record count per window is constant.
            "spectrum_window_samples": (
                SPECTRUM_WINDOW_RECORDS
                * HOPS_PER_RECORD
                * int(cfg["fengine"]["fft_length"])
            ),
            "spectrum_ring_depth": SPECTRUM_RING_DEPTH,
            # The unsummed comb, per record, appended after the PRN records (gnssRecord.hpp);
            # the telemetry leg carries it to the broker. n2rec_buf is sized for it.
            "chan_export": True,
            "reference_element": args.reference_element,
            # Element steering, every chain, from the array-epoch table's positions. Their
            # presence (elem_positions_enu) arms steering in GnssGpuRecordAssemble; the
            # geometry then arrives from the broker's post-sat-geometry feed. The sign is a
            # measured convention.
            **steer,
            # Once a PRN's cal is warm (see ELEM_SUM_TAU_S), the record header carries the
            # self-calibrated weighted mean over all elements (the reference element's phase,
            # single-element scale) instead of the bare reference element: ~sqrt(N_healthy)
            # more per-record SNR for every header consumer. The cal comes from the satellite
            # itself; dead elements are gated out and a dead reference element fails over.
            "elem_sum": True,
            "elem_sum_tau_s": ELEM_SUM_TAU_S,
            # Channels weighted by 1/rms of their own measured power, not by the F-engine
            # digital gain, with the power averaged over the frame inputs this stage sums (the
            # live elements, the set live_element_tiles gathers).
            "chan_noise_norm": True,
            "chan_noise_norm_elems": [
                e
                for lo, hi in live_element_ranges(arr)
                for e in range(lo, hi + 1)
                if e not in bad_input_set(cfg, args)
            ],
            **elem_shared_keys(args),
            **elem_shared_ref_keys(
                args,
                (tag or signal_tag(sig["primary"])).strip("_"),
                steer,
                n_live,
                bad_input_set(cfg, args),
            ),
            **elem_proj_keys(args),
            **lane_transfer_keys(cfg, node, gpu, tag, freq_ids, args),
            **cube_assembler_keys(args, cfg, gpu, pre),
            "sample_rate": float(cfg["fengine"]["sampling_rate_MHz"]) * 1e6,
            "cpu_affinity": [V["cores"]["assemble"]],
        },
        # The chain's combiner: integrates the records into amplitudes and loop observables and
        # serves them over REST (/<pre>n2combine/get_status, /get_records).
        f"{pre}n2cmb_buf": {
            "kotekan_buffer": "standard",
            "metadata_pool": "gnss_pool",
            "num_frames": BUFFER_DEPTH,
            "frame_size": f"{n_prn} * {record_floats} * sizeof_float32",
        },
        f"{pre}n2combine": {
            "kotekan_stage": "GnssCoherentCombiner",
            # One record stream per combiner (the stage accepts exactly one): the broker
            # combines across GPUs and nodes, and merging the two GPUs here would only reduce
            # the instance count its fleet combine works with.
            "in_bufs": [n2rec_buf],
            "out_buf": f"{pre}n2cmb_buf",
            "n_prn": n_prn,
            "n_elements": n_live,
            # EMA time constant and deep window, records
            "integration_length": INTEGRATION_LENGTH,
            "integration_mode": "rolling",
            "deep_coherent": True,
            "deep_rate_search": True,
            "deep_rate_min_q": 10.0,
            # The sky-phase-corrected prompt (record slots 24/25: half the elements derotated
            # by the other half's phase, ElemCal::combine_split) is scored as one more deep
            # candidate beside the plain sum, under a floor that pays for it, so a cold or bad
            # cal loses on merit. deep_coherent, deep_rate_search, deep_rate_min_q and sky_deep
            # are set to the only values the stage accepts.
            "sky_deep": True,
            "fft_len": cfg["fengine"]["fft_length"],
            # The broker's telem-windows (config/gnss_chains_chord.yaml) in records, so the
            # polled and gathered feeds integrate the same length; check_telem_windows holds it
            # to FLEET_TRIM_TAPS_WINDOWS.
            "record_export": FLEET_TRIM_TAPS_WINDOWS * TELEM_RECORDS_PER_FRAME,
            "cpu_affinity": [V["cores"]["combine"]],
        },
        f"{pre}n2sink": {"kotekan_stage": "dropAllFrames", "in_buf": f"{pre}n2cmb_buf"},
    }

    # -- The telemetry leg to the gather ------------------------------------------------------
    #
    # A second consumer of n2rec_buf, beside the combiner. GnssTelemPack cuts each record to its
    # gnssRecord.hpp header and comb block and batches TELEM_RECORDS_PER_FRAME of them on the
    # absolute window index; a bufferSend ships each frame to the gather instance. The address
    # (chain, instance, absolute window) travels with the data on the F-engine's clock, so the
    # broker never infers which instance or window a frame describes.
    #
    # The frame size is the sender's own: its chain's PRN rows and the comb columns this
    # instance despreads. One bufferRecv still serves the fleet (the gather's buffer is sized to
    # the widest sender, with allow_short_frames), and an unfilled row or column would be bytes
    # on a link the whole fleet shares.
    chain_name = broker_chain_name(chain["signal"] if chain else sig["primary"])
    # May be fewer than n_prn: the wire rows are compacted onto the PRNs despread (see
    # telem_rows).
    telem_max_prn = telem_rows(chain_name)
    telem_bytes = telem_frame_bytes(TELEM_RECORDS_PER_FRAME, telem_max_prn, n_chan)
    blocks.update(
        {
            f"{pre}telem_buf": {
                "kotekan_buffer": "standard",
                "metadata_pool": "gnss_pool",
                # Deep enough that a momentary stall in the send leg costs latency rather than
                # records; at 23.84 frames/s this is ~1.3 s of slack.
                "num_frames": 32,
                "frame_size": telem_bytes,
            },
            f"{pre}telem_pack": {
                "kotekan_stage": "GnssTelemPack",
                "in_buf": n2rec_buf,
                "out_buf": f"{pre}telem_buf",
                "chain": chain_name,
                # The sender's identity on the far side, with chain: node.gpu, the granularity
                # of the combiner and every per-instance endpoint.
                "inst": f"{node}.{gpu}",
                "n_prn": n_prn,
                "n_elements": n_live,
                "max_prn": telem_max_prn,
                "records_per_frame": TELEM_RECORDS_PER_FRAME,
                "hops_per_record": HOPS_PER_RECORD,
                "fft_len": cfg["fengine"]["fft_length"],
                "n_chan": n_chan,
                # The comb and its column labels: the same channel_ids the assembler and the
                # injector were given. The broker reads the freq_ids off the wire, so no copy on
                # its side can drift out of step with the node.
                "chan_export": True,
                "channel_ids": list(freq_ids),
                "cpu_affinity": [V["cores"]["telem"]],
            },
            f"{pre}telem_send": {
                "kotekan_stage": "bufferSend",
                "buf": f"{pre}telem_buf",
                "server_ip": args.telem_host,
                "server_port": TELEM_PORT,
                # Telemetry must never back-pressure the chain: a gather that is down costs
                # dropped frames and nothing else.
                "drop_frames": True,
                # one wire frame per TELEM_RECORDS_PER_FRAME records
                **send_pacing(
                    cfg,
                    spds,
                    telem_bytes,
                    n_rec_per_frame / float(TELEM_RECORDS_PER_FRAME),
                ),
                # 30 s, not bufferSend's 5 s: while the gather is down every sender logs a WARN
                # per attempt, and across the fleet the default buries the real problem in
                # "Connection refused". Bring the gather up first (gnss-gather.service), then
                # the nodes.
                "reconnect_time": 30,
                # As on the search leg: a gather that accepts the connection and stops reading
                # costs a dropped frame, not a 20 s stall in send() on telem_buf's only consumer.
                "send_timeout": 2,
                # Pinned on both ends: see srch_send in build_search_leg.
                "use_config_tracker": False,
                "cpu_affinity": [V["cores"]["send"]],
            },
        }
    )
    # -- The beam-cube push leg (--cube-host) -------------------------------------------------
    # One completed ~1 s window per frame, pushed as soon as a later window opens rather than
    # polled: the assembler's ring holds CUBE_RING_DEPTH windows, so a poller that stalled
    # longer would lose windows with no trace in the archive. Same transport as the telemetry
    # leg.
    #
    # Backpressure drops, never blocks, so the archive can never stall the chain. The assembler
    # counts the windows it drops and every frame carries the cumulative count, so a gap in the
    # window index without a rise in that count was lost downstream (in this bufferSend or on
    # the far side).
    cube_bytes = cube_frame_bytes(CUBE_MAX_PRN, CUBE_MAX_BINS, n_live)
    if n_chan > CUBE_MAX_BINS:
        raise SystemExit(
            f"chain {pre.strip('_')}: {n_chan} covering channels but "
            f"CUBE_MAX_BINS {CUBE_MAX_BINS}; the frame cannot hold "
            f"them. Raise it (every sender and the archiver change together)."
        )
    if n_prn > CUBE_MAX_PRN:
        raise SystemExit(
            f"chain {pre.strip('_')}: {n_prn} PRN slots but CUBE_MAX_PRN "
            f"{CUBE_MAX_PRN}; raise it (every sender and the archiver change together)."
        )
    blocks.update(
        {
            f"{pre}cube_buf": {
                "kotekan_buffer": "standard",
                "metadata_pool": "gnss_pool",
                # ~1 frame/s per sender: 16 frames (1.6 MB) absorb ~16 s of a stalled far side.
                # Past that the assembler drops the window and counts it.
                "num_frames": 16,
                "frame_size": cube_bytes,
            },
            f"{pre}cube_send": {
                "kotekan_stage": "bufferSend",
                "buf": f"{pre}cube_buf",
                "server_ip": args.cube_host,
                "server_port": CUBE_PORT,
                "drop_frames": True,
                "reconnect_time": 30,
                "send_timeout": 2,
                **send_pacing(
                    cfg, spds, cube_bytes, n_rec_per_frame / float(CUBE_WINDOW_RECORDS),
                ),
                # Pinned to match the archiver: see srch_send in build_search_leg.
                "use_config_tracker": False,
                "cpu_affinity": [V["cores"]["send"]],
            },
        }
    )
    return blocks


def merge_n2dual_gpu(blocks, cfg, args, node, gpu, chains, spds):
    """One cudaProcess per GPU for every path-B chain on it.

    build_n2dual_branch builds each chain's injector and correlator commands; here they are
    folded into one process. Every chain's injector writes its lanes into one compact synth
    array [T][comb][num_synth] at its own lane offset, one cudaCorrelatorDual correlates over
    the union of the combs, and one gather per chain copies out that chain's tiles frame, so
    the host stages are those of a chain on its own. With --vis-capture, one more gather per
    GPU takes the live N^2 once, every captured chain's mixed rows and, with --vis-capture-bb,
    the replica block across those chains, gated in step with their ctl blocks.

    Lanes: chains that share a channel get disjoint lane ranges, each the chain's own width
    (its injector's num_synth: SYNTH_SLOTS slots times two or four lanes per slot); chains on
    disjoint channels reuse a range, since their replicas never meet in a product. Ranges
    start on LANE_STEP boundaries, whole tile rows, so a chain's slot arithmetic from its
    range start and its tile rows are those of a chain on its own. The axis is the n2k_dual
    instantiation's width (DUAL_SYNTH_LANES) whatever the chains use of it; unused lanes stay
    0x88 and cost the correlator nothing but the launch they share.
    """
    sig = cfg["signals"]
    rt = cfg["runtime"]
    pres = []
    for c in chains:
        ch = c["chain"]
        tag = ch["tag"] if ch else ""
        inj0 = next(
            cmd
            for cmd in blocks[f"gnss{gpu}{tag}_n2dual"]["commands"]
            if cmd["name"] == "cudaGnssInject"
        )
        pres.append(
            dict(
                pre=f"gnss{gpu}{tag}_",
                tag=tag,
                chain=ch,
                name=broker_chain_name(ch["signal"] if ch else sig["primary"]),
                chan_idx=[int(x) for x in c["chan_idx"]],
                chan=set(int(x) for x in c["chan_idx"]),
                lanes=int(inj0["num_synth"]),
                viscap=vis_capture_chain(args, ch, sig),
            )
        )
    union = sorted(set().union(*[p["chan"] for p in pres]))
    for p in pres:
        base = 0
        while any(
            q["chan"] & p["chan"]
            and q["base"] < base + p["lanes"]
            and base < q["base"] + q["lanes"]
            for q in pres
            if "base" in q
        ):
            base += LANE_STEP
        p["base"] = base
    used = max(p["base"] + p["lanes"] for p in pres)
    if used > DUAL_SYNTH_LANES:
        raise SystemExit(
            f"gnss{gpu}: the GPU's chains need {used} synth lanes, the dual correlator "
            f"has {DUAL_SYNTH_LANES} (external/n2k_dual's one instantiation). Fewer chains "
            f"sharing a channel on this GPU, fewer PRN slots, or a wider instantiation."
        )
    num_synth = DUAL_SYNTH_LANES

    procs = {p["pre"]: blocks.pop(f"{p['pre']}n2dual") for p in pres}
    primary = procs[f"gnss{gpu}_"]
    merged = {k: v for k, v in primary.items() if k != "commands"}
    # Private CUDA streams: cuda_stream_base is a pipeline index, and 1 moves this process's
    # copy-in, copy-out and kernel streams to 3, 4, 5, leaving 0, 1, 2 to production's pipelines,
    # so the GNSS chains never queue against production's per-stream locks (one thread blocked in
    # the CUDA driver there would stall the GPU's pipelines and back up to the NIC).
    # prepareStreams only grows the device's pool, so asking for 6 holds whatever the
    # construction order.
    merged["cuda_stream_base"] = 1
    merged["num_cuda_streams"] = 6
    out_buffers = {}
    injects, outputs = [], []
    for p in pres:
        inj = dict(
            next(
                c for c in procs[p["pre"]]["commands"] if c["name"] == "cudaGnssInject"
            )
        )
        inj["gnss_synth_name"] = f"gnss{gpu}_synth"
        inj["synth_lane_base"] = p["base"]
        inj["synth_lane_pitch"] = num_synth
        inj["gnss_synth_channels"] = union
        injects.append(inj)
        out_buffers[f"host_{p['pre']}tiles"] = f"{p['pre']}n2tiles_buf"
        out_buffers[f"host_{p['pre']}n2ctl"] = f"{p['pre']}n2ctl_buf"
        outputs += [
            {
                "name": "cudaOutputData",
                "gpu_mem": f"{p['pre']}tiles_buffer",
                "out_buf": f"host_{p['pre']}tiles",
            },
            # The injector's control block (FrameHdr, window starts, PrnCtl and the replica
            # energies): the epl layout without corr, which GnssN2RecordAssemble fills from
            # the tiles (docs/gnss_gpu_search.md 11.11). No _buffer suffix: cudaGnssInject
            # allocates it with get_gpu_memory_array under its own name, unlike the tiles,
            # which are an NDArrayBuffer addressed as <name>_buffer.
            {
                "name": "cudaOutputData",
                "gpu_mem": f"{p['pre']}n2ctl",
                "out_buf": f"host_{p['pre']}n2ctl",
            },
        ]

    corr = dict(
        next(c for c in primary["commands"] if c["name"] == "cudaCorrelatorDual")
    )
    corr["gnss_synth_name"] = f"gnss{gpu}_synth"
    corr["num_synth"] = num_synth
    corr["gnss_local_channels"] = union
    corr["gnss_synth_compact"] = True
    names = []
    for p in pres:
        names.append(p["name"])
        corr[f"gather_{p['name']}_tiles_name"] = f"{p['pre']}tiles"
        corr[f"gather_{p['name']}_channels"] = p["chan_idx"]
        corr[f"gather_{p['name']}_lane_base"] = p["base"]
        corr[f"gather_{p['name']}_lanes"] = p["lanes"]

    # The capture: one gather over the captured chains' channels and lane rows, with the live
    # N^2 and, with --vis-capture-bb, the replica block across those chains; one gate holds
    # that frame in step with every captured chain's ctl block (its slot -> PRN map).
    cap_blocks = {}
    captured = [p for p in pres if p["viscap"]]
    if captured:
        cap_chans = sorted(set().union(*[p["chan"] for p in captured]))
        base = min(p["base"] for p in captured)
        cap_lanes = max(p["base"] + p["lanes"] for p in captured) - base
        bb = bool(args.vis_capture_bb)
        names.append("cap")
        corr.update(
            {
                "gather_cap_tiles_name": f"gnss{gpu}_captiles",
                "gather_cap_channels": cap_chans,
                "gather_cap_lane_base": base,
                "gather_cap_lanes": cap_lanes,
                "gather_cap_aa": True,
                "gather_cap_bb": bb,
            }
        )
        # The gathered tile count mirrors cudaCorrelatorDual::build_tile_selection for this
        # gather: mixed rows x live columns, the AA triangle, the BB triangle over the rows.
        nlive16 = (live_element_count(cfg["array"]) + 15) // 16
        nrows = cap_lanes // 16
        n_tile = (
            nrows * nlive16
            + nlive16 * (nlive16 + 1) // 2
            + (nrows * (nrows + 1) // 2 if bb else 0)
        )
        n_rec = max(1, int(spds) // HOPS_PER_RECORD)
        cap_bytes = n_rec * len(cap_chans) * n_tile * 512 * 4
        cap_buf = f"gnss{gpu}_captiles_buf"
        out_buffers["host_gnss_captiles"] = cap_buf
        outputs.append(
            {
                "name": "cudaOutputData",
                "gpu_mem": f"gnss{gpu}_captiles_buffer",
                "out_buf": "host_gnss_captiles",
            }
        )
        sink_core = GPU_VARS[gpu][0]["cores"]["sink"]
        vc_dir = (args.record_dir or rt["record_dir"]) + "/viscap"
        vc_tiles = f"gnss{gpu}_viscap_tiles_buf"
        vc_ctls = [f"{p['pre']}viscap_ctl_buf" for p in captured]
        cap_blocks[cap_buf] = {
            "kotekan_buffer": "standard",
            "metadata_pool": "gnss_pool",
            "num_frames": BUFFER_DEPTH,
            "frame_size": cap_bytes,
        }
        cap_blocks[vc_tiles] = {
            "kotekan_buffer": "standard",
            "metadata_pool": "gnss_pool",
            "num_frames": args.vis_capture_depth,
            "frame_size": cap_bytes,
        }
        for p, vb in zip(captured, vc_ctls):
            cap_blocks[vb] = {
                "kotekan_buffer": "standard",
                "metadata_pool": "gnss_pool",
                "num_frames": args.vis_capture_depth,
                "frame_size": blocks[f"{p['pre']}n2ctl_buf"]["frame_size"],
            }
        cap_blocks[f"gnss{gpu}_viscap_gate"] = {
            "kotekan_stage": "FrameWindowGate",
            "in_bufs": [cap_buf] + [f"{p['pre']}n2ctl_buf" for p in captured],
            "out_bufs": [vc_tiles] + vc_ctls,
            "clock_buf": 1,
            "clock_source": "frame",
            "clock_offset": 16,
            "cpu_affinity": [sink_core],
        }
        sink = {
            "kotekan_stage": "rawFileWrite",
            "base_dir": vc_dir,
            "file_ext": "raw",
            "prefix_hostname": False,
            "num_frames_per_file": args.vis_capture_frames_per_file,
            "allow_ndarray": True,
            "continue_numbering": True,
            "create_base_dir": True,
            "cpu_affinity": [sink_core],
        }
        cap_blocks[f"gnss{gpu}_viscap_tiles_sink"] = dict(
            sink, in_buf=vc_tiles, file_name=f"{node}_gnss{gpu}_capvistiles"
        )
        for p, vb in zip(captured, vc_ctls):
            cap_blocks[f"{p['pre']}viscap_ctl_sink"] = dict(
                sink, in_buf=vb, file_name=f"{node}_gnss{gpu}{p['tag']}_visctl"
            )

    corr["gnss_gathers"] = names
    merged["in_buffers"] = primary["in_buffers"]
    merged["out_buffers"] = out_buffers
    merged["commands"] = injects + [corr, {"name": "cudaSyncOutput"}] + outputs
    blocks[f"gnss{gpu}_n2dual"] = merged
    blocks.update(cap_blocks)
    print(
        f"  gpu{gpu}: {len(pres)} chains in one dual-correlator pass, {len(union)} channels, "
        f"{num_synth} synth lanes"
        + (f", capture over {len(captured)} chain(s)" if captured else ""),
        file=sys.stderr,
    )


def _comb_g(chan_ids, fft_len):
    """gcd of the covering channels' frequency differences with fft_len.

    The acquire's fine-lag axis (surface_dims, gnssChannelizedAcquire.cpp) stores
    s_stored = fft_len/g columns, its period, which sets how far the refine has to search.
    g == 1 means the axis spans a whole hop and the coarse phase is unambiguous.
    """
    from math import gcd

    if not chan_ids:
        return fft_len
    g = 0
    for c in chan_ids[1:]:
        g = gcd(g, abs(c - chan_ids[0]))
    return gcd(g, fft_len) if g else fft_len


def search_stage(cfg, args, in_buf, chan_ids, core, signal=None, prns=None):
    """The aggregator's GnssChannelizedSearch block.

    `signal` and `prns` default to the primary and its slots; an extra search
    (--aggregator-extra-search) names another signal on the same comb, so every other key,
    the carrier included, stays the primary's."""
    sig = cfg["signals"]
    fe = cfg["fengine"]
    n_chan = len(chan_ids)
    return {
        "kotekan_stage": "GnssChannelizedSearch",
        "in_buf": in_buf,
        "signal": signal or sig["primary"],
        "prns": list(prns) if prns else PRIMARY_PRNS,
        "sample_rate": float(fe["sampling_rate_MHz"]) * 1e6,
        # The sky carrier, not an IF: covering_bins() places the signal at f_offset, so 0.0
        # puts it at DC, where no covering channel overlaps this subband.
        "f_offset": float(sig["carrier_hz"]),
        "spectrum_length": fe["num_bins"],
        "num_taps": fe["pfb_taps"],
        "pfb_window": fe["pfb_window"],
        # The sparse comb, addressed explicitly by channel_ids rather than zero-filled to a
        # contiguous span. channel_offset only labels the stage's log lines.
        "channel_offset": chan_ids[0],
        "n_channels": n_chan,
        "channel_ids": list(chan_ids),
        # No doppler_min/max: the instance's /clock_profile sets the Doppler extent.
        # 62.5 Hz = 1/T_coh for the 16 ms (3125-hop) coherent window, so the worst miss is
        # 1/(2 T_coh), about what a coherent window tolerates. It is also the transform's bin
        # spacing Fs/(Mp*fft_len): on that grid a Doppler wipe is an exact cyclic shift of one
        # forward FFT, and channelized_peak's sub-grid refine, which inverts an amplitude ratio
        # in bins, needs that alignment (on a half-bin grid it saturates and biases toward the
        # neighbouring cell).
        "doppler_step": 62.5,
        # 30 s, not the stage's 8 s: the broker re-sends hints once per cycle, and the TTL must
        # outlast its longest cycle. With require_hint an expired hint skips the PRN before its
        # SNR is computed, so a short TTL silences the search with nothing in its output to say
        # why.
        "hint_ttl_s": 30.0,
        "hold_snapshots": 5,
        "doppler_margin_hz": float(sig["max_doppler_hz"]),
        # Search only PRNs the broker has hinted, and only near the hinted Doppler: the broker
        # predicts every visible satellite's Doppler from BRDC, so a blind grid over all 32
        # PRNs is work not worth doing. The stage refuses false; older binaries default to
        # false, so this stays emitted.
        "require_hint": True,
        "use_cuda_acquire": bool(args.cuda_acquire),
        "acquire_windows": ACQUIRE_WINDOWS,
        "acquire_snr": ACQUIRE_SNR,
        # Alignments scanned either side of the broker's ephemeris-predicted one (its
        # --nh-hint). The acquire builds a full surface per alignment, so 20 of them are ~92%
        # of a pass; a span of 2 scans 5, enough to cover the predicted alignment's observed
        # spread. A PRN without a hint scans all 20.
        "nh_hint_span": 2,
        "acquire_fine_step": ACQUIRE_FINE_STEP,
        # The L5 Q5 pilot carries the NH20 overlay, one +-1 chip per 1 ms code period, and the
        # 16 ms coherent window spans 16 of them, so an overlay-blind replica sums 16 chips of
        # a near-balanced sequence: 12.7 dB rms loss over the 20 alignments, a null at some.
        # Searching the 20 alignments recovers the full coherent gain.
        "nh_search": True,
        # Post-detection code-phase refine, one exact despread per step, run per detection.
        # Step at half the covering set's resolution (fft_len/n_chan samples): only the
        # covering channels enter the despread, so a finer step buys no precision, and the
        # stage default of 1 sample would cost 2*refine_span+1 despreads per detection.
        "refine_step": max(1, int(2 * fe["num_bins"] / max(1, n_chan) / 2)),
        # Refine integration length, hops: two L5 code periods (2 x 195.3125). The refine only
        # localises a peak that already cleared detection, at one replica build per trial
        # phase, and two periods pick the same trial phase as the full 16-period record because
        # the step quantisation dominates (scripts/gnss/e2e.cpp --refine-hops measures it).
        "refine_hops": 391,
        # Refine span, samples either side, from the comb rather than the node count. When
        # g == 1 (see _comb_g) the coarse phase is unambiguous and +-512 samples covers its
        # error. Otherwise the fine axis stores only fft_len/g columns, the phase is right only
        # modulo that (the grating-lobe ambiguity), and the span must cover a full hop.
        "refine_span": 512
        if _comb_g(chan_ids, fe["num_bins"] * 2) == 1
        else fe["num_bins"] * 2,
        "cpu_affinity": [core],
        # Pick each surface's peak on the mainlobe pair-sum, and vote the NH label across the
        # pass's alignments. The stage refuses false; older binaries default both to false.
        "acq_pairsum_select": True,
        "nh_label_consensus": True,
    }


def aggregator_node(ent, default_base):
    """An --aggregator-instance entry, NODE or NODE:BASE -> (node, base)."""
    node, sep, base = str(ent).partition(":")
    if not sep:
        return node, default_base
    try:
        return node, int(base)
    except ValueError:
        raise SystemExit(
            f"--aggregator-instance {ent}: expected NODE or NODE:PORT_BASE (e.g. cx19:11040)"
        )


def build_aggregator_instance(cfg, nodes, args, port, spds=8192):
    """One acquisition search over the union of several nodes' combs. One node GPU's feed holds
    ~7 of the ~106 L5 covering channels (-11.8 dB), which no amount of local integration
    recovers; the union of the feeds is denser by construction. The feeds arrive over
    bufferSend/bufferRecv, one per node GPU; GnssChanAlignMerge aligns them on the F-engine's
    global sample counter (equal sample_seq is simultaneity, so the union search is coherent
    across nodes), and one GnssChannelizedSearch runs on the merged comb.

    Port layout: `nodes` holds NODE or NODE:BASE entries; node k's GPU g feed listens on
    BASE + g, or search_port_base + 2k + g for a bare NODE. Each node's config sends to its own
    --search-port-base + gpu. gen_fleet.py passes NODE:BASE from the fleet list
    (config/gnss_fleet.py: 11040 + 2 x the node's slot), so a node that leaves the fleet moves
    no other node's port. A GPU with no covering channel has no feed and leaves its port
    unused rather than shifting the later nodes'.

    `spds` is the base's samples_per_data_set, which sizes the senders' frames; bufferRecv
    closes a feed whose frame size differs.
    """
    sig = cfg["signals"]
    rt = cfg["runtime"]
    # runtime.cpu_affinity (config/chord_gnss_node.yaml) is a CHORD node's core map; on a host
    # without those cores pass --aggregator-cores: a wrong core is an EINVAL per stage.
    cores = (
        [int(c) for c in args.aggregator_cores.split(",") if c.strip() != ""]
        if getattr(args, "aggregator_cores", None)
        else rt["cpu_affinity"]
    )
    if not cores:
        raise SystemExit("--aggregator-cores parsed to an empty list")
    out = {
        "type": "config",
        "log_level": "info",
        "samples_per_data_set": spds,
        "sizeof_float32": 4,
        "cpu_affinity": cores,
        "rest_server": {"port": port, "cpu_affinity": cores, "enable_cors": True},
        "clock_profile": {"name": cfg["clock"]["profile"]},
        "gnss_pool": {
            "kotekan_metadata_pool": "GnssChanMetadata",
            "num_metadata_objects": 30 * BUFFER_DEPTH,
        },
        "telescope": {
            "name": "ICETelescope",
            "num_polarizations": 1,
            "num_dishes": 1,
            "query_gps": False,
            "require_gps": False,
        },
    }

    # One feed per (node, gpu), node-major, each with its listen port. The merge concatenates
    # columns in feed order, so union_ids is built in that order; the search takes it as
    # channel_ids and never assumes sortedness.
    feeds = []
    for k, ent in enumerate(nodes):
        node, base = aggregator_node(ent, args.search_port_base + 2 * k)
        nchans = covering_channels(
            node_channels(cfg, node),
            float(sig["carrier_hz"]),
            float(sig["chip_rate_hz"]),
            float(sig["max_doppler_hz"]),
        )
        if not nchans:
            raise SystemExit(f"{node} holds no covering channels for {sig['primary']}")
        pg = {}
        for fid in nchans:
            gpu, idx = gpu_of_channel(cfg, node, fid)
            pg.setdefault(gpu, []).append(fid)
        for gpu in sorted(pg):
            feeds.append((node, gpu, pg[gpu], base + gpu))

    union_ids = []
    in_buf_names = []
    in_channels = []
    for i, (node, gpu, fids, listen_port) in enumerate(feeds):
        n_chan = len(fids)
        pre = f"agg{i}_"
        union_ids.extend(fids)
        in_buf_names.append(f"{pre}f_buf")
        in_channels.append(n_chan)
        out.update(
            {
                f"{pre}in_buf": {
                    "kotekan_buffer": "standard",
                    "metadata_pool": "gnss_pool",
                    "num_frames": BUFFER_DEPTH * 2,
                    "frame_size": f"samples_per_data_set * {n_chan} * 1",
                },
                f"{pre}f_buf": {
                    "kotekan_buffer": "standard",
                    "metadata_pool": "gnss_pool",
                    "num_frames": BUFFER_DEPTH * 2,
                    "frame_size": f"samples_per_data_set * {n_chan} * 8",  # cfloat32
                },
                f"{pre}recv": {
                    "kotekan_stage": "bufferRecv",
                    "buf": f"{pre}in_buf",
                    "listen_port": listen_port,
                    "num_threads": 1,
                    "drop_frames": True,
                    # Must match the sender: see srch_send in build_search_leg.
                    "use_config_tracker": False,
                    "cpu_affinity": [cores[i % len(cores)]],
                },
                f"{pre}deq": {
                    "kotekan_stage": "GnssChordDequantize",
                    "in_buf": f"{pre}in_buf",
                    "out_buf": f"{pre}f_buf",
                    "n_channels": n_chan,
                    "n_elements": 1,
                    "element": 0,
                    # The F-engine's channelized output is conjugated relative to the nominal
                    # decode (indistinguishable from a nibble swap: the two differ by a constant
                    # phase, which cancels in every x_i x_j* product); without this the despread
                    # matches no satellite. The stage refuses false. See
                    # GnssChordDequantize.cpp.
                    "conjugate": True,
                    "cpu_affinity": [cores[(i + 2) % len(cores)]],
                },
            }
        )

    n_union = len(union_ids)
    out.update(
        {
            "agg_merged_buf": {
                "kotekan_buffer": "standard",
                "metadata_pool": "gnss_pool",
                "num_frames": BUFFER_DEPTH * 2,
                "frame_size": f"samples_per_data_set * {n_union} * 8",
            },
            "agg_merge": {
                "kotekan_stage": "GnssChanAlignMerge",
                "in_bufs": in_buf_names,
                "out_buf": "agg_merged_buf",
                "in_channels": in_channels,
                # Absent-feed timeout, stated although it is the stage default. A feed that
                # delivers nothing this long is absent: its channels are zeroed, never reused
                # (stale samples against the wrong epoch would manufacture detections), the
                # merge continues on the rest, and the feed rejoins by itself. The stage refuses
                # <= 0, which would let a config listing more feeds than running nodes stall the
                # merge.
                "input_timeout_s": 5.0,
                "cpu_affinity": [cores[3 % len(cores)]],
            },
            # Named gps_search: the broker's detectors URL (config/gnss_chains_chord.yaml) points
            # here, and the broker polls its /get_detections and posts its Doppler hints.
            "gps_search": search_stage(
                cfg, args, "agg_merged_buf", union_ids, cores[4 % len(cores)]
            ),
        }
    )
    # The union surface grows as channels x stored lags, so the search runs ACQUIRE_THREADS
    # threads pinned to the pool's last max(ACQUIRE_THREADS, 6) cores; threads beyond the cores
    # in that set contend for them and buy nothing.
    nth = max(1, int(ACQUIRE_THREADS))
    out["gps_search"]["acquire_threads"] = nth
    out["gps_search"]["cpu_affinity"] = list(cores[-max(nth, 6) :])
    # --aggregator-extra-search: more searches on the same merged comb, one stage each, named
    # <tag>_search (signal_tag) so a broker chain's `detectors:` can point at it. The signal
    # must share the primary's carrier: the comb is the primary's covering set, and a signal
    # elsewhere in the band would be searched on channels that do not carry it. The #191
    # step 1 experiment is GAL_E5A_Q beside GPS_L5_Q, one frequency, identical channels.
    table = signal_table() if args.aggregator_extra_search else {}
    for spec in args.aggregator_extra_search or []:
        name, _, prn_s = spec.partition(":")
        if name not in table:
            raise SystemExit(
                f"--aggregator-extra-search {name}: unknown signal (not in gnssSignal.hpp)"
            )
        if (
            abs(float(table[name]["carrier_hz"]) - float(cfg["signals"]["carrier_hz"]))
            > 1.0
        ):
            raise SystemExit(
                f"--aggregator-extra-search {name}: its carrier ({table[name]['carrier_hz']} Hz)"
                f" is not the primary's ({cfg['signals']['carrier_hz']} Hz); the merged comb"
                " covers the primary's band only"
            )
        prns = (
            [int(p) for p in prn_s.split(",") if p.strip()]
            if prn_s.strip()
            else EXTRA_SEARCH_PRNS.get(name[:3])
        )
        if not prns:
            raise SystemExit(
                f"--aggregator-extra-search {name}: give the PRNs, e.g. {name}:1,2,3"
            )
        key = signal_tag(name).lstrip("_") + "_search"
        if key in out:
            raise SystemExit(
                f"--aggregator-extra-search {name}: stage '{key}' already exists"
            )
        out[key] = search_stage(
            cfg,
            args,
            "agg_merged_buf",
            union_ids,
            cores[4 % len(cores)],
            signal=name,
            prns=prns,
        )
        out[key]["acquire_threads"] = nth
        out[key]["cpu_affinity"] = list(cores[-max(nth, 6) :])
    return out, feeds, union_ids


def build_cube_archive_instance(cfg, args, port):
    """The beam-cube archiver: the whole fleet's completed cube windows onto bulk storage.

    One bufferRecv accepts every sender (~90 assemblers) on a single port, and one rawFileWrite
    lands the frames, bundled, under CUBE_ARCHIVE_DIR.

    Its own instance, separate from the gather and the search aggregator, so that restarting
    either never leaves holes in the record. Moving it is --cube-host on the nodes; its own cores
    are runtime.cpu_affinity, which has no override flag.

    Raw frames, not a typed file format: each frame is self-describing (version, absolute
    window index, wstart, the sender's <host>/<stage> address, gpu, each bin's freq_ids, each
    slot's PRN and a cumulative dropped-window count all travel in the payload,
    gnssRecord.hpp), so the bytes on disk carry their own provenance, and converting them to a
    science format is an offline job that can be redone. Short chains' zero padding (see
    CUBE_MAX_PRN) is the converter's to strip, not the transport's.

    No frame is split across files: rawFileWrite bundles a fixed count, so a reader derives the
    frame count from the file size and can start anywhere.
    """
    rt = cfg["runtime"]
    cores = rt["cpu_affinity"]
    n_elem = live_element_count(cfg["array"])
    frame_bytes = cube_frame_bytes(CUBE_MAX_PRN, CUBE_MAX_BINS, n_elem)
    out = {
        "type": "config",
        "log_level": "info",
        # bufferRecv deserializes the sender's metadata into a pool object of the same type, so
        # the pool must exist here even though nothing in this instance reads it.
        "gnss_pool": {
            "kotekan_metadata_pool": "GnssChanMetadata",
            "num_metadata_objects": 4096,
        },
        "cpu_affinity": cores,
        "rest_server": {"port": port, "cpu_affinity": cores, "enable_cors": True},
        # kotekan constructs a Telescope whatever the stage graph contains; minimal valid block,
        # same as the gather and aggregator instances. query_gps false: no F-engine to ask here.
        "telescope": {
            "name": "ICETelescope",
            "num_polarizations": 1,
            "num_dishes": 1,
            "query_gps": False,
            "require_gps": False,
        },
        "cube_buf": {
            "kotekan_buffer": "standard",
            "metadata_pool": "gnss_pool",
            # ~90 senders x ~1 frame/s: 512 frames (52 MB) absorb ~5 s of a writer stalled
            # behind an NFS commit without pushing back on a sender.
            "num_frames": 512,
            "frame_size": frame_bytes,
        },
        "cube_recv": {
            "kotekan_stage": "bufferRecv",
            "buf": "cube_buf",
            "listen_port": CUBE_PORT,
            # One thread per ~15 senders; the work is socket reads, not compute.
            "num_threads": 6,
            "drop_frames": True,
            # Must match the senders: see srch_send in build_search_leg.
            "use_config_tracker": False,
            "cpu_affinity": list(cores[:6]),
        },
        "cube_write": {
            "kotekan_stage": "rawFileWrite",
            "in_buf": "cube_buf",
            "base_dir": CUBE_ARCHIVE_DIR,
            "file_name": "gnss_cube",
            "file_ext": "raw",
            # The archiver's own hostname in every filename carries no information; the name
            # that would (which sender wrote a frame) is in the frame, since one file holds
            # frames from every sender.
            "prefix_hostname": False,
            "num_frames_per_file": CUBE_ARCHIVE_FRAMES_PER_FILE,
            # The archive outlives the process. rawFileWrite numbers from 0 at every start and
            # opens without O_TRUNC, so a restart would overwrite the oldest files; resume past
            # the highest existing number instead.
            "continue_numbering": True,
            "cpu_affinity": [cores[6 % len(cores)]],
        },
    }
    return out


def build_gather_instance(cfg, args, port):
    """The gather instance: the whole fleet's record telemetry into one local stream.

    One bufferRecv accepts every sender on a single port, and GnssTelemGather writes each frame,
    length-prefixed and verbatim, to the Python broker over a local socket: identity and
    delivery, no collation. GnssFleetTrim rides beside it on the same buffer and does collate,
    because the fast code loop needs a fleet-wide view at frame rate and this is the only
    process that has one. The policy (who is armed, every gate, the clock, the ephemeris) stays
    in the broker; only the discriminator, the integrator and the post run here. See
    docs/CHORD_FAST_TRIM.md.

    Grouping instances needs no coordination: every frame carries the absolute window index
    wstart/(records_per_frame * hops_per_record * fft_length), computed by each sender from the
    same three integers, so it is an exact integer match rather than an inference from arrival
    order or from a UTC stamp each instance derives on its own.

    One listener, one buffer, senders of different shapes. A frame's identity is in its payload
    (chain and instance tags), not in which socket it arrived on, so there is no port map to
    wire up wrong. Each sender ships its chain's PRN rows and its own comb columns, so the buffer
    is sized to the widest frame the fleet can produce and bufferRecv accepts anything up to it;
    the frames are self-describing and every consumer validates the shape against the length it
    was given.

    Standalone, like the search aggregator, so that the aggregator's frequent restarts do not
    take it down. Moving it is --telem-host on the nodes, plus --gather-cores if the new host
    lacks the default cores.
    """
    rt = cfg["runtime"]
    # runtime.cpu_affinity (config/chord_gnss_node.yaml) is a CHORD node's core map; on a host
    # without those cores pass --gather-cores: the pin is a required key, so a wrong core is an
    # EINVAL per stage.
    cores = (
        [int(c) for c in args.gather_cores.split(",") if c.strip() != ""]
        if getattr(args, "gather_cores", None)
        else rt["cpu_affinity"]
    )
    if not cores:
        raise SystemExit("--gather-cores parsed to an empty list")
    # The widest sender on each axis independently (telem_rows_max rows, TELEM_MAX_CHAN
    # columns), not the widest chain's product: rows and columns are set by different things,
    # and bufferRecv closes the connection of a sender whose frame exceeds the buffer, which
    # then reads downstream as a dead node.
    frame_bytes = telem_frame_bytes(
        TELEM_RECORDS_PER_FRAME, telem_rows_max(), telem_max_chan(),
    )
    out = {
        "type": "config",
        "log_level": "info",
        # bufferRecv deserializes the sender's metadata into a pool object of the same type, so
        # the pool must exist here even though nothing in this instance reads sample_seq.
        "gnss_pool": {
            "kotekan_metadata_pool": "GnssChanMetadata",
            "num_metadata_objects": 4096,
        },
        "cpu_affinity": cores,
        "rest_server": {"port": port, "cpu_affinity": cores, "enable_cors": True},
        # kotekan constructs a Telescope whatever the stage graph contains, so a minimal valid
        # block is required even though nothing here has a sky model -- same as the search
        # instance. query_gps false: this host has no F-engine to ask.
        "telescope": {
            "name": "ICETelescope",
            "num_polarizations": 1,
            "num_dishes": 1,
            "query_gps": False,
            "require_gps": False,
        },
        "telem_buf": {
            "kotekan_buffer": "standard",
            "metadata_pool": "gnss_pool",
            # Past this depth bufferRecv drops, which is correct (telemetry must never
            # back-pressure a tracker), but a drop here is every sender's frame at once, a whole
            # window missing fleet-wide, and the broker's fleet ADR ends every arc on every
            # chain when it sees one. 8192 frames is ~4 s at ~90 senders x 23.84 frames/s
            # (~330 MB at the widest frame), enough to ride out a briefly starved consumer.
            "num_frames": 8192,
            "frame_size": frame_bytes,
        },
        "telem_recv": {
            "kotekan_stage": "bufferRecv",
            "buf": "telem_buf",
            "listen_port": TELEM_PORT,
            # The work is socket reads of frames up to ~40 kB at a modest rate, not compute.
            "num_threads": 4,
            "drop_frames": True,
            # Senders ship their own shapes into this one buffer; see the docstring. Every
            # consumer of telem_buf strides by the frame's header, never by this buffer's size.
            "allow_short_frames": True,
            # Must match the senders: see srch_send in build_search_leg.
            "use_config_tracker": False,
            "cpu_affinity": list(cores[:4]),
        },
        "telem_gather": {
            "kotekan_stage": "GnssTelemGather",
            "in_buf": "telem_buf",
            "serve_host": GATHER_SERVE_HOST,
            "serve_port": GATHER_SERVE_PORT,
            # A sender silent this long is stale: logged once, left out of the window-spread
            # verdict (where its frozen window index would swamp the live senders' spread), and
            # live again on its next frame. Nothing here blocks on a sender.
            "stale_after_s": 5.0,
            "cpu_affinity": [cores[4 % len(cores)]],
        },
    }
    # The fast code loop: GnssFleetTrim forms the fleet discriminator, integrates it and posts
    # the trims (/set_trim, every window); the broker keeps the policy. See
    # docs/CHORD_FAST_TRIM.md. It is a second consumer of telem_buf rather than a second parse
    # of the served stream, so it sees exactly the frames the broker sees.
    #
    # A second consumer shares the buffer's fate: if it stalls, telem_buf fills and bufferRecv
    # drops frames for the broker too. Its work per frame is bounded, and get_stats serves
    # fold_us_per_frame to check it against the 42 ms frame period.
    out["fleet_trim"] = {
        "kotekan_stage": "GnssFleetTrim",
        "in_buf": "telem_buf",
        "n_win": args.fleet_trim_windows,
        # The broker's depth, not the loop's (see FLEET_TRIM_TAPS_WINDOWS): the loop's
        # discriminator averages the newest n_win windows, while /get_taps serves the policy
        # cycle, which averages telem-windows for its gates' SNR. scripts/gnss/fleetdll_gate.py's
        # split-ring leg checks the deeper taps and an unmoved loop.
        "taps_win": FLEET_TRIM_TAPS_WINDOWS,
        # The trim store. The integrator is the only state here that cannot be rebuilt from
        # the stream; without it a restart leaves every tracker off-peak by its standing trim.
        # On local disk, not NFS, since it is saved every 2 s; a host reboot clears /tmp,
        # which is correct, as the store is only meaningful across a short outage.
        "trim_state_file": FLEET_TRIM_STATE_FILE,
        "trim_state_max_age_s": FLEET_TRIM_STATE_MAX_AGE_S,
        "min_instances": FLEET_TRIM_MIN_INSTANCES,
        # Poster threads. The targets are not here: the broker sends them with /set_policy,
        # because it owns the tracker endpoint list and a copy in this file would miss a node
        # added to the fleet.
        "post_threads": 4,
        "cpu_affinity": [cores[5 % len(cores)]],
    }
    return out


def render_stock_template(path):
    """A kotekan .j2 config rendered as kotekan renders it (kotekan/kotekan.cpp): jinja2 with a
    FileSystemLoader on the template's own directory and select_autoescape(), then
    yaml.safe_load. No template variables, so this is the stock render.

    A render rather than a config captured from a running node: a capture goes stale silently as
    production changes (boards, dish table, RFI settings), while a render of production's own
    template keeps the stock half stock by construction. scripts/gnss/stock_parity.py checks it
    against the render and, with --live, against a running stock node.
    """
    import jinja2

    d, f = os.path.split(os.path.abspath(path))
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(d), autoescape=jinja2.select_autoescape()
    )
    return yaml.safe_load(env.get_template(f).render({}))


def choco_bad_inputs(timeout=20.0):
    """bffs's current bad-input list, or None. Returns (block, source), the block in the shape a
    node's /updatable_config/bad_inputs holds.

    The translation (bffs's feed labels -> input indices, start_time = its update time plus
    choco's sync_delay) lives in config/bffs_bad_inputs.py, shared with
    scripts/gnss/site/bad_inputs_cron.py, so a node started from this config and a running node
    the cron keeps current hold the block a stock node does (stock_parity.py --live checks).
    """
    import bffs_bad_inputs

    try:
        body, src = bffs_bad_inputs.fetch(
            bffs_bad_inputs.STATE, bffs_bad_inputs.CONF, timeout
        )
        return dict(body, kotekan_update_endpoint="json"), src
    except Exception as e:
        print(
            "  bad inputs bffs source unavailable (%s: %s) -- asking the stock nodes"
            % (type(e).__name__, e),
            file=sys.stderr,
        )
        return None


def node_names(cfg):
    """The hosts in the node table. nodes: also holds bringup_order, a list, which is not one."""
    return [k for k, v in cfg["nodes"].items() if isinstance(v, dict)]


def node_rest_port(cfg, args):
    """The port to ask our running nodes on: --rest-port for a node config, which gives the node
    that port; runtime.rest_port for an off-node instance, whose --rest-port is its own."""
    instance = (
        args.aggregator_instance or args.gather_instance or args.cube_archive_instance
    )
    if args.rest_port is not None and not instance:
        return args.rest_port
    return cfg["runtime"]["rest_port"]


def bad_input_block(cfg, args):
    """bffs's bad-input block, (block, source) or None, fetched once per run and kept on args:
    the node's updatable_config and the GNSS branch (bad_input_set) must read one list.
    --bad-inputs-from FILE takes the block from a file instead (gen_fleet.py --check passes the
    block the config on disk carries, so the lists derived from it regenerate identically)."""
    if not hasattr(args, "_bad_inputs"):
        path = getattr(args, "bad_inputs_from", None)
        if path:
            with open(path) as f:
                blk = yaml.safe_load(f)
            if not isinstance(blk, dict) or "bad_inputs" not in blk:
                raise SystemExit(
                    f"--bad-inputs-from {path}: not a bad-input block (no bad_inputs key)"
                )
            bi = (blk, path)
        else:
            bi = choco_bad_inputs()
            if bi is None:
                bi = live_bad_inputs(cfg)
        args._bad_inputs = bi
    return args._bad_inputs


#: bffs flagging more than this fraction of the inputs is no data, not bad feeds: the
#: F-engine streaming zeros (10-25 01:46Z: 127 of 128 flagged) or a dead link.
BAD_INPUTS_OUTAGE_FRACTION = 0.5


def bad_input_set(cfg, args):
    """The inputs out of the GNSS branch's static element lists, as a set: bffs's flagged inputs
    (none when no source answered) and the inputs out for hardware reasons
    (config/gnss_bad_inputs.json, read once per run). Both index the flat [pol][dish] input
    axis, the same axis as the GNSS element axis (cube_order), so no translation. The element
    sum needs no list (ElemCal gates a dead input on its own coherence), but the stations the
    projection nulls on, the inputs the channel noise is averaged over and the fleet
    reference's entries are static lists, and a flagged input there sits at full weight."""
    bi = bad_input_block(cfg, args)
    bad = set(int(i) for i in bi[0]["bad_inputs"]) if bi else set()
    n = live_element_count(cfg["array"])
    if len(bad) > BAD_INPUTS_OUTAGE_FRACTION * n:
        # Applied, it would empty the stations, the channel-noise inputs and the shared ref
        # for as long as the config is run, long after the data came back.
        raise SystemExit(
            f"the bad-input list ({bi[0].get('update_id')}, from {bi[1]}) flags {len(bad)} of "
            f"{n} inputs: an outage (no data), not bad feeds. The GNSS branch's element lists "
            "would be left empty, so nothing is generated. Regenerate when the data is back "
            "and bffs has unflagged the feeds, or pass --bad-inputs-from with a good block."
        )
    if not hasattr(args, "_manual_bad_inputs"):
        args._manual_bad_inputs = gnss_arraymap.manual_bad_inputs()
    return bad | args._manual_bad_inputs


def project_stations(spec, bad=()):
    """--n2-project-stations ("0-6,8-127") as a sorted list, minus the flagged inputs."""
    st = []
    for part in str(spec).split(","):
        a, _, b = part.strip().partition("-")
        st.extend(range(int(a), int(b or a) + 1))
    return sorted(set(st) - set(bad))


_STOCK_NOTED = []


def stock_reference_hosts(cfg):
    """The stock nodes the generator asks for the EOP table (besides ours) and the bad-input
    block when choco cannot be read: the site's hosts outside the GNSS fleet
    (config/gnss_fleet.py), which run production's kotekan and are kept current by choco. With
    none left, the EOP table comes from choco and our nodes and the bad inputs from bffs only;
    that is said once, not skipped silently."""
    import gnss_fleet

    try:
        refs = gnss_fleet.Fleet(gnss_fleet.MANIFEST, cfg).stock
    except (OSError, gnss_fleet.FleetError) as e:
        refs = []
        why = f"the fleet list does not load ({e})"
    else:
        why = "every site host is in the GNSS fleet"
    if not refs and not _STOCK_NOTED:
        _STOCK_NOTED.append(True)
        print(
            f"  WARNING no stock reference node: {why}. The EOP table comes from choco and "
            "our nodes only, and the bad inputs from bffs only.",
            file=sys.stderr,
        )
    return refs


def live_bad_inputs(cfg, timeout=5.0):
    """The bad-input block a running stock node (stock_reference_hosts) holds, as
    (block, host), or None. The fallback when bffs's state cannot be read: choco keeps those
    nodes current. A block still at the template's initial_flags is skipped.
    """
    import urllib.request

    # Stock nodes run production's kotekan, on production's port whatever --rest-port says.
    port = cfg["runtime"]["rest_port"]
    for host in stock_reference_hosts(cfg):
        try:
            with urllib.request.urlopen(
                f"http://{host}:{port}/config", timeout=timeout
            ) as r:
                blk = json.loads(r.read().decode())["updatable_config"]["bad_inputs"]
        except Exception:
            continue
        if isinstance(blk, dict) and blk.get("update_id") != "initial_flags":
            return blk, host
    return None


def choco_eop_table(timeout=10.0):
    """The observatory's current EOP table, or None. Returns (table, source, last_ns), the shape
    live_eop_table returns.

    Read over ssh: choco's /var/lib/choco is not mounted on the generator hosts. A failure is
    not fatal (the caller falls back to the fleet) but is reported, since the last fallback is
    the base's own table: empty in the stock template, stale in a captured base.
    """
    import subprocess

    path = "choco:/var/lib/choco/eop/state.json"
    host, _, remote = path.partition(":")
    try:
        raw = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", host, "cat " + remote],
            capture_output=True,
            timeout=timeout,
        ).stdout.decode()
        tbl = json.loads(raw).get("earth_orientation_parameter_table") or []
        bare = [
            {
                "t_inst_ns": int(e["t_inst_ns"]),
                "delta_UT1_inst": float(e["delta_UT1_inst"]),
                "xp_as": float(e["xp_as"]),
                "yp_as": float(e["yp_as"]),
            }
            for e in tbl
        ]
        if not bare:
            return None
        return bare, path, max(e["t_inst_ns"] for e in bare)
    except Exception as e:
        print(
            "  eop       choco source unavailable (%s: %s) -- falling back to the fleet"
            % (type(e).__name__, e),
            file=sys.stderr,
        )
        return None


def live_eop_table(cfg, args, timeout=3.0):
    """The freshest Earth-orientation table any reachable kotekan serves, or None.

    Returns (table, source_host, last_ns), with `table` the BareEOP list receive_eop_updates()
    accepts: t_inst_ns, delta_UT1_inst, xp_as and yp_as only, since ERA_deg and t_ut1_ns are
    derived by BareEOP::to_EOP() and are not BareEOP fields.

    A running node holds only the table it was last given, so this cannot recover from a table
    that is stale fleet-wide; the caller reports how far the table reaches.
    """
    import urllib.request

    # Our nodes and the stock reference nodes; the freshest table wins. After a long outage of
    # our own, the stock nodes are the ones still being refreshed. They run production's
    # kotekan, so they are asked on production's port.
    refs = stock_reference_hosts(cfg)
    hosts = node_names(cfg) + [h for h in refs if h not in cfg["nodes"]]
    best = None
    for host in hosts:
        port = (
            cfg["runtime"]["rest_port"] if host in refs else node_rest_port(cfg, args)
        )
        try:
            url = f"http://{host}:{port}/telescope/eop_table"
            with urllib.request.urlopen(url, timeout=timeout) as r:
                tbl = json.loads(r.read().decode()).get("eop_table") or []
        except Exception:
            continue
        if not tbl:
            continue
        bare = [
            {
                "t_inst_ns": int(e["t_inst_ns"]),
                "delta_UT1_inst": float(e["delta_UT1_inst"]),
                "xp_as": float(e["xp_as"]),
                "yp_as": float(e["yp_as"]),
            }
            for e in tbl
        ]
        bare.sort(key=lambda e: e["t_inst_ns"])
        last = bare[-1]["t_inst_ns"]
        if best is None or last > best[2]:
            best = (bare, host, last)
    return best


def live_frame0_utc(cfg, args, timeout=3.0):
    """The F-engine's frame 0 (UTC seconds) as the running fleet reports it, or None.

    Every kotekan serves its frame 0 at telescope/time0_ns, read once at startup, so this shows
    what the F-engine was doing when each node started, not whether it is running now. Nodes
    started either side of an F-engine restart disagree; then no value is current, so this
    warns and returns None rather than pick one.
    """
    import urllib.request

    # The nodes' REST port, which carries the telescope endpoints; production's on the stock
    # reference nodes.
    refs = stock_reference_hosts(cfg)
    seen = {}
    for node in node_names(cfg):
        port = (
            cfg["runtime"]["rest_port"] if node in refs else node_rest_port(cfg, args)
        )
        try:
            url = f"http://{node}:{port}/telescope/time0_ns"
            with urllib.request.urlopen(url, timeout=timeout) as r:
                ns = int(json.load(r)["time0_ns"])
            seen.setdefault(ns, []).append(node)
        except Exception:
            continue
    if not seen:
        return None
    if len(seen) > 1:
        # Warnings go to stderr: without --out, stdout is the config (gen_fleet reads it from
        # there, to write and to --check), so a line printed there would land inside it.
        print(
            "  WARNING   running nodes disagree about frame 0 -- some started either side of "
            "an F-engine restart, so none of them is authoritative:",
            file=sys.stderr,
        )
        for ns, nodes in sorted(seen.items()):
            print(
                "              %.9f  %s" % (ns * 1e-9, ", ".join(sorted(nodes))),
                file=sys.stderr,
            )
        print(
            "            Re-read it from chive, or restart the stragglers, before trusting "
            "any record stamp.",
            file=sys.stderr,
        )
        return None
    return next(iter(seen)) * 1e-9


def check_frame0(cfg, args, live_t0):
    """Check the record epoch this config pins against the frame 0 the running fleet serves.

    cudaGnssChordTrackState takes frame0_utc (the UTC of the F-engine's absolute sample 0;
    records are stamped frame0_utc + wstart/rate) from Telescope::to_time_ns(0) unless the config
    carries one, so by default none is emitted and nothing can go stale. A pinned value
    (--frame0-nano, else the node file's fengine.frame0_utc) goes stale silently at the next
    F-engine restart, since every cross-record estimator works in differences. `live_t0` is
    live_frame0_utc's answer (None when no single value came back): a pin that disagrees with
    it is an error, and a node-file pin it cannot confirm is a warning.
    """
    if args.frame0_nano:
        have = args.frame0_nano * 1e-9
    elif cfg["fengine"].get("frame0_utc"):
        have = float(cfg["fengine"]["frame0_utc"])
    else:
        print(
            "  epoch     frame0_utc not emitted -- the stages take it from the telescope's "
            "live GPS time0 (week-rollover corrected). Nothing to keep in step.",
            file=sys.stderr,
        )
        return
    if live_t0 is None:
        if not args.frame0_nano:
            # stderr: without --out, stdout is the config (see live_frame0_utc).
            print(
                "  WARNING   the running nodes gave no single telescope/time0_ns (none "
                f"answered, or they disagree) -- frame0_utc {have:.6f} is unverified. If the "
                "F-engine has restarted since it was written, every record stamp is wrong.",
                file=sys.stderr,
            )
        return
    if abs(have - live_t0) <= 1e-3:
        return
    gap = (
        f"{have:.9f} but the running fleet serves {live_t0:.9f} "
        f"({abs(have - live_t0) / 86400.0:.2f} days apart).\n"
    )
    if args.frame0_nano:
        raise SystemExit(
            f"--frame0-nano gives frame0_utc {gap}"
            "Re-read telescope/time0_ns from a node started since the last F-engine restart "
            "(curl -s http://cx27:12048/telescope/time0_ns). Records would carry the wrong "
            "epoch and nothing downstream would say so."
        )
    raise SystemExit(
        f"frame0_utc in {NODE_FILE} is {gap}"
        f"The F-engine has been restarted since that value was written. Records would "
        f"carry the wrong epoch and nothing downstream would say so.\n"
        f"Fix it:  sed -i 's/^    frame0_utc: .*/    frame0_utc: {live_t0:.9f}/' "
        f"{NODE_FILE}\n"
        f"Better: delete frame0_utc from {NODE_FILE}; the stages then take the epoch "
        f"from the telescope's live GPS time0, and no config value can go stale."
    )


def _rel(p):
    """A path inside the repo or the kotekan tree relative to it, so a recorded recipe holds on
    any checkout and reads the same whether or not the two are one tree. The deeper root wins:
    a kotekan submodule sits inside the repo."""
    ap_ = os.path.abspath(p)
    roots = [os.path.dirname(CONF)]
    if kotekan_src.SRC:
        roots.append(kotekan_src.SRC)
    for root in sorted(roots, key=len, reverse=True):
        if ap_.startswith(root + os.sep):
            return os.path.relpath(ap_, root)
    return p


def _kotekan_input(p):
    """An input path as given or, relative and not there from the cwd, the same path in the
    kotekan tree if it is there: _rel records a kotekan file relative to that tree
    (config/chord/pathfinder.j2), so a recipe pastes back in a checkout without one."""
    if os.path.isabs(p) or os.path.exists(p) or not kotekan_src.SRC:
        return p
    k = kotekan_src.src(p)
    return k if os.path.exists(k) else p


def recipe_lines(argv):
    """The header comment lines that regenerate a config: the generator and its `argv`, less
    --out, wrapped without breaking a token so the recipe pastes back whole.

    --out is left out: it says where the file was written, not how it was built, and gen_fleet
    regenerates to stdout and compares byte for byte, so a recorded --out would never match.
    Add --out when pasting the recipe, or let gen_fleet write the file. --bad-inputs-from is
    left out for the same reason: it names a temporary file of live data (gen_fleet --check
    passes the block the config already carries), and the block itself is in the config.
    """
    sink_flags = ("--out", "--bad-inputs-from")
    keep = []
    i = 0
    while i < len(argv):
        if argv[i] in sink_flags:
            i += 2
            continue
        if any(argv[i].startswith(f + "=") for f in sink_flags):
            i += 1
            continue
        keep.append(argv[i])
        i += 1
    return [
        "#   " + ln
        for ln in textwrap.wrap(
            "config/gen_chord_gnss_config.py "
            + " ".join(shlex.quote(_rel(a)) for a in keep),
            width=92,
            subsequent_indent="    ",
            break_on_hyphens=False,
            # A long --extra-signal PRN list must stay whole: the recipe is meant to be pasted.
            break_long_words=False,
        )
    ]


def check_telem_windows(path=CHAINS_FILE):
    """Refuse to generate when the broker's telem-windows (`common:` or any chain's override in
    the chains file) differs from FLEET_TRIM_TAPS_WINDOWS. The broker averages telem-windows
    gathered windows, the gather's /get_taps serves taps_win and the combiners export
    record_export records; a mismatch integrates different lengths on each path with no error.
    """
    with open(path) as fh:
        chains = yaml.safe_load(fh)
    have = {"common": (chains.get("common") or {}).get("telem-windows")}
    for name, spec in (chains.get("chains") or {}).items():
        if isinstance(spec, dict) and "telem-windows" in spec:
            have[name] = spec["telem-windows"]
    bad = {k: v for k, v in have.items() if v != FLEET_TRIM_TAPS_WINDOWS}
    if bad:
        raise SystemExit(
            "telem-windows in %s (%s) must equal FLEET_TRIM_TAPS_WINDOWS (%d, the gather's "
            "taps_win) and the combiners' record_export (%d records = %d windows of %d) in "
            "config/gen_chord_gnss_config.py: change all three together."
            % (
                path,
                ", ".join("%s: %s" % kv for kv in bad.items()),
                FLEET_TRIM_TAPS_WINDOWS,
                FLEET_TRIM_TAPS_WINDOWS * TELEM_RECORDS_PER_FRAME,
                FLEET_TRIM_TAPS_WINDOWS,
                TELEM_RECORDS_PER_FRAME,
            )
        )


def iface_stamp():
    """The header line naming the interface golden this config's frames were sized from, with
    the digest of each sizing section (gnss_broker/iface.py stamp_line). The deploy preflight
    refuses a binary whose own sizing sections differ from it."""
    return iface.stamp_line(iface.golden())


def check_tree_golden():
    """Warn when the kotekan tree here carries an interface golden whose signals or sizing
    sections differ from the vendored one, which the frames are sized from: a binary built from
    that tree would not match these configs until the vendored golden is synced from it."""
    if not kotekan_src.SRC:
        return
    path = kotekan_src.src(iface.KOTEKAN_GOLDEN)
    if not os.path.exists(path):
        return
    g = iface.golden()
    try:
        with open(path) as fh:
            have = iface.section_digests(json.load(fh))
        moved = [
            s
            for s in ("signals",) + iface.SIZING
            if have.get(s) != g.section_digests[s]
        ]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        moved = ["(unreadable: %s)" % e]
    if moved:
        print(
            f"WARNING: {path} differs from the vendored interface golden (pin kotekan "
            f"{g.commit[:12]}) in {', '.join(moved)}. The frames are sized from the vendored "
            "one, so a binary built from this tree will not match these configs. Sync it, "
            f"from python/scripts/gnss: python3 -m gnss_broker.iface sync --from "
            f"{kotekan_src.SRC}",
            file=sys.stderr,
        )


def check_descriptors(cfg, args):
    """Refuse a descriptor the interface golden does not define, naming it: a binary built
    from the kotekan the golden pins would not resolve it, and the node would stop at stage
    construction."""
    g = iface.golden()
    names = [
        ("signals.primary", cfg["signals"]["primary"]),
        ("signals.tracker", cfg["signals"].get("tracker")),
    ]
    names += [("--extra-signal", s.partition(":")[0]) for s in args.extra_signal or []]
    names += [
        ("--aggregator-extra-search", s.partition(":")[0])
        for s in args.aggregator_extra_search or []
    ]
    for where, name in names:
        if name is not None and name not in g.sections["signals"]:
            raise SystemExit(
                f"{where} {name}: not a descriptor in the interface golden "
                f"(python/scripts/gnss/gnss_broker/iface_golden.json, pin kotekan "
                f"{g.commit[:12]}). Sync the golden from a kotekan that defines it, or fix "
                "the name."
            )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--base",
        required=True,
        help="the production config the GNSS branch is injected into: a .j2 template, "
        "rendered as kotekan renders it, with no template variables (the fleet uses "
        "config/chord/pathfinder.j2), or a .json or yaml config",
    )
    ap.add_argument("--node", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--rest-port",
        type=int,
        default=None,
        help="REST port of the generated instance. Default: runtime.rest_port from "
        "config/chord_gnss_node.yaml for a node config, runtime.aggregator_rest_port for "
        "--aggregator-instance, runtime.aggregator_rest_port + 1 for --gather-instance and "
        "+ 2 for --cube-archive-instance (those two refuse runtime.rest_port and "
        "runtime.aggregator_rest_port). The generator queries our running nodes (EOP table, "
        "frame 0) on this port for a node config and on runtime.rest_port for an off-node "
        "instance; the stock reference nodes always on runtime.rest_port.",
    )
    ap.add_argument(
        "--n2-send-legs",
        choices=("both", "full", "subset"),
        default="both",
        help="which --n2-send N^2 legs to keep (default %(default)s): full = "
        "buffer_send_n2_full (n2_buffer, the accumulated visibilities); subset = "
        "buffer_send_n2_subset (n2_eigen_buffer, the eigen products); both, as on a "
        "stock node. A leg not selected is deleted from the config, not left "
        "unaddressed: a bufferSend consumes its buffer whether or not its socket "
        "connects. n2_buffer has another consumer (n2_subset), so the full leg changes "
        "nothing upstream. The subset leg is n2_eigen_buffer's only consumer, so it is "
        "what lets eigencalc (EigenN2Iter) run; a frame eigencalc cannot decompose (an "
        "autocorrelation with an imaginary part, say) goes out with zero-filled "
        "eigenpairs and is counted in kotekan_eigenN2iter_num_failed_eigencalc. The "
        "bad-feed mask legs follow --n2-send, whichever legs this keeps.",
    )
    ap.add_argument(
        "--rf-stats-period-s",
        type=float,
        default=10.0,
        help="seconds between RF-monitor passes on each GPU's search tap "
        "(band_power_period_s, default %(default)s): clip fraction and band power per "
        "channel over every chain's covering channels on that GPU, served at "
        "<tap>/rf_stats. A pass reads a decimated slice of the frame in place rather "
        "than copying the frame, so a short period is cheap.",
    )
    ap.add_argument(
        "--n2-send",
        action="store_true",
        help="ship the N^2 visibilities to the standard downstream receiver, as a stock "
        "node does. Keeps the base's buffer_send_n2_* legs (n2_buffer to "
        "--n2-send-port-full, n2_eigen_buffer to --n2-send-port-subset, both GPUs; "
        "--n2-send-legs picks which) and its bad-feed mask legs (--n2-send-port-mask), "
        "and stamps --n2-send-ip and the ports into each leg rather than inheriting the "
        "base's, so the destination can be read in the generated file. Without this "
        "flag every bufferSend leg of the base is dropped. drop_frames defaults to true, "
        "so a slow or absent receiver loses frames instead of backpressuring the GPU "
        "chain. The default endpoint is the receiver the stock nodes send to, so the "
        "frames land in shared science storage beside theirs: turn it on deliberately.",
    )
    ap.add_argument(
        "--n2-send-ip",
        default="10.222.0.51",
        help="destination host of the --n2-send legs, N^2 and bad-feed mask (default: "
        "the standard CHORD N^2 receiver, as in config/chord/pathfinder.j2)",
    )
    ap.add_argument(
        "--n2-send-port-full",
        type=int,
        default=11027,
        help="port for the full n2_buffer leg (default 11027, stock)",
    )
    ap.add_argument(
        "--n2-send-port-subset",
        type=int,
        default=11025,
        help="port for the n2_eigen_buffer subset leg (default 11025, stock)",
    )
    ap.add_argument(
        "--n2-send-port-mask",
        type=int,
        default=11029,
        help="port for the bad-feed mask legs that ride with the N^2 frames "
        "(default 11029, stock)",
    )
    ap.add_argument(
        "--vis-capture",
        nargs="?",
        const="gps_l5",
        default=None,
        metavar="CHAINS",
        help="build the (N+M)^2 visibility capture for these broker chains (comma "
        "list; bare flag = gps_l5). Each GPU's correlator adds one capture gather "
        "(the live antennas' N^2 plus the captured chains' mixed rows), and a "
        "REST-armed FrameWindowGate passes it, with those chains' ctl frames, to "
        "rawFileWrites under <record_dir>/viscap (--record-dir). Nothing is written "
        "until scripts/gnss/viscap.py arms a window. See docs/CHORD_VIS_CAPTURE.md.",
    )
    ap.add_argument(
        "--vis-capture-bb",
        action="store_true",
        dest="vis_capture_bb",
        help="with --vis-capture: the capture gather also takes the BB (replica x "
        "replica) block across the captured chains, completing the (N+M)^2 "
        "triangle.",
    )
    ap.add_argument(
        "--vis-capture-frames-per-file",
        type=int,
        default=24,
        metavar="N",
        dest="vis_capture_frames_per_file",
        help="frames per --vis-capture file (default 24, ~1 s at 41.94 ms frames)",
    )
    ap.add_argument(
        "--vis-capture-depth",
        type=int,
        default=8,
        metavar="N",
        dest="vis_capture_depth",
        help="frames of slack between the gate and each writer (default 8)",
    )
    ap.add_argument(
        "--frame0-nano",
        type=int,
        default=None,
        metavar="NS",
        help="start without chive: take the F-engine's frame-0 time (int64 ns) from this "
        "value instead of querying chive:54321, use it as the records' frame0_utc too, "
        "and drop config_tracker, which fetches from the same service. A running node "
        "serves the value: curl -s http://cx27:12048/telescope/time0_ns. require_gps "
        "stays true, so a missing value still fails at startup rather than falling back "
        "to the host clock. The value holds only while the F-engine keeps counting from "
        "the same frame 0: an F-engine restart or firmware or mode change starts a new "
        "one, and each node reads it once at startup, so a running node shows what the "
        "F-engine was doing when that node started. Re-read it after any F-engine "
        "restart. A stale value fails silently: records carry the wrong epoch, the "
        "seeds predict the wrong code phase, and every PRN despreads noise while "
        "looking like a clock or geometry fault.",
    )
    ap.add_argument(
        "--extra-signal",
        action="append",
        default=[],
        metavar="NAME:PRNS",
        help="add a tracker chain for another signal (NAME from gnssSignal.hpp), e.g. "
        "--extra-signal GAL_E5A_Q_CS:1,2,3. Repeatable. PRNs are required "
        "(constellations number their PRNs differently, and a chain has "
        f"{SYNTH_SLOTS} PRN slots). A chain whose covering "
        "channels match the primary's shares its voltage tap (E5a and B2a are on "
        "L5's carrier); a different carrier gets its own tap. Buffer and endpoint "
        "names carry a tag derived from the signal name. Sideband partners given "
        "together (E5a and E5b, B2a and B2b) must list the same PRNs. See "
        "docs/CHORD_MULTIBAND.md.",
    )
    ap.add_argument(
        "--elem-positions-file",
        default="",
        help="element positions for steering, in place of the array epoch's "
        "(config/chord_array_epochs.json via gnss_arraymap): a JSON with pointing and "
        "positions_by_dish, refused unless its pointing is the epoch's valid at "
        "generation and it places every dish of the epoch; or, the older form, with "
        "epoch_key, cube_order and positions_enu, refused unless its epoch_key is the "
        "epoch valid at generation and its cube_order is the config's live elements",
    )
    ap.add_argument(
        "--elem-steer-sign",
        type=float,
        default=1.0,
        help="sign (+1 or -1) of the element steering phase. It depends on the "
        "F-engine's FFT and conjugation conventions, so it is calibrated from data "
        "(see gnssElemSteer.hpp)",
    )
    ap.add_argument(
        "--reference-element",
        type=int,
        default=0,
        help="the reference element, as an index into the live elements: its correlation "
        "fills the record header until a PRN's element-sum cal is warm, and after that it "
        "is the phase anchor of the calibrated element sum. The header is what the "
        "broker's DLL and carrier loops track. Also a live switch "
        "(POST <assemble>/set_reference_element)",
    )
    ap.add_argument(
        "--elem-sum-shared",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="shared instrument model in the assembler: one per-element gain per "
        "polarisation learned as a slow consensus of every satellite's cal, "
        "plus one inter-pol coefficient per satellite; per-PRN weights are "
        "held to it (elem_sum_adapt false), so a boresight transit cannot "
        "capture the weak satellites' element sums. Emitted only when on.",
    )
    ap.add_argument(
        "--elem-proj-mode",
        default="off",
        choices=("off", "shadow", "live"),
        help="bright-satellite projection in the assembler (gnssProjSubspace.hpp): "
        "shadow = projected shadow learner + capture diagnostics only; live = "
        "the rows are projected in place. Emitted only when not off; the "
        "assembler also takes it live over POST /set_elem_proj.",
    )
    ap.add_argument(
        "--rfi-excision-exempt-freq-ids",
        type=str,
        default=None,
        help="Comma-separated absolute freq_id ranges (lo-hi) exempt from first-stage "
        "RFI excision, injected into every cudaRFISKtilde command as "
        "rfi_first_stage_excision_exempt_freq_ids (our key; declared in "
        "stock_parity.py). The GNSS lobes: the stock SK flagger excises them "
        "most of the time and their N2 feeds the satellite projection.",
    )
    ap.add_argument(
        "--epl-from-prompt",
        action="store_true",
        help="cudaGnssChordTrack epl_from_prompt: synthesise the prompt replica only and "
        "take the E/L rows as its per-channel rotations (one chip walk per hop instead of "
        "three), and pack two synth lanes per PRN slot (P, P_HEAD) instead of four, "
        "GnssN2RecordAssemble rotating P into the E and L rows (lanes_per_slot), on every "
        "chain.",
    )
    ap.add_argument(
        "--psi-synth",
        action="store_true",
        help="cudaGnssChordTrack psi_synth on every chain: replica synthesis from per-PRN "
        "code-phase tables over the full PFB span, fused with the N^2 pack (one launch per "
        "record instead of gnss_waveform + gnss_pack44; lib/cuda/cudaGnssPsiSynth.hpp). "
        "Requires --epl-from-prompt. The replica carries +1.1%% more energy than the "
        "80-chip gather, so every absolute SNR gate steps at the cutover.",
    )
    ap.add_argument(
        "--n2-project",
        choices=["off", "live"],
        default="off",
        help="GnssN2Project on the science N2: live = per frame and channel, project the "
        "dominant spatially coherent components (the transiting satellites) out of the "
        "--n2-project-stations block, and N2Accumulate reads the projected copy "
        "(gnss_n2_proj_buffer{,_1}). Needs --n2-project-live-freq-ids.",
    )
    ap.add_argument(
        "--bad-inputs-from",
        metavar="FILE",
        default=None,
        help="take the bad-input block (a node's updatable_config/bad_inputs, YAML or JSON) "
        "from FILE instead of bffs or the stock nodes. gen_fleet.py --check passes the block "
        "the config on disk carries, so the GNSS lists derived from it compare exactly.",
    )
    ap.add_argument(
        "--n2-project-stations",
        type=str,
        default="0-15,56-79,120-127",
        help="live correlator inputs (stations) the projection works on, comma-separated "
        "ranges (lo-hi) or single indices (default %(default)s)",
    )
    ap.add_argument(
        "--n2-project-source",
        choices=["eigen", "lanes", "lanes+residual"],
        default="eigen",
        help="GnssN2Project's direction source (#190): eigen = the N2 block's own top-k "
        "subspace (default); lanes = the tracked satellites' directions the chains publish "
        "(each PRN's averaged prompt row, gnssLaneDirection.hpp; needs --elem-dir-publish all); "
        "lanes+residual = lanes first, then the eigen path on what is left",
    )
    ap.add_argument(
        "--assembler-core-slots",
        type=str,
        default="2",
        help="pool slots (comma list, relative to the GPU like every chain core) the "
        "chains' GnssGpuRecordAssemble stages rotate through by chain order; the default "
        "keeps them all on slot 2. The slot is taken modulo the pool after adding the GPU "
        "index, like every chain core, and pool[0] is each GPU's cudaProcess core, so a slot "
        "must stay clear of 0, 5 (telemetry + GnssN2Project), 6 (the senders) and 7 (the "
        "sink) on BOTH GPUs: '2,1,3,4,8' does (9 is pool[0] on GPU 1) and leaves at most "
        "two assemblers per core",
    )
    ap.add_argument(
        "--n2-project-core-index",
        type=int,
        default=-1,
        help="GnssN2Project's core as an index into the GPU's runtime pool; -1 (default) = "
        "pool[(gpu + 5) %% len], the telemetry packers' core. With lane directions the stage "
        "does ~4 k n^2 operations per channel-frame, so a canary moves it to a slot with "
        "light stages (cx51: 9 = cores 43 / 63)",
    )
    ap.add_argument(
        "--n2-project-cpus",
        type=str,
        default="",
        help="GnssN2Project's logical cpu per GPU, 'gpu0,gpu1', outside the runtime pools; "
        "overrides --n2-project-core-index. The nodes leave whole physical cores idle on each "
        "NUMA node (15/47 on node 0, 19/51 on node 1), and the lanes on 127 stations want one",
    )
    ap.add_argument(
        "--lane-transfer",
        action="append",
        default=[],
        help="TAG:LO-HI (MHz), repeatable: the chain tagged TAG (e6, l2c, e5a, ...) also "
        "publishes its lane directions on the channels centred in [LO, HI] that GnssN2Project "
        "writes, turned through each satellite's geometric delay (#190 tier 1.5: the same "
        "satellites' other signals, e.g. e6:1263.4-1273.6 for the Galileo E6 PRS lobe)",
    )
    ap.add_argument(
        "--n2-project-eigen-channels",
        choices=["all", "live"],
        default="all",
        help="channels GnssN2Project's eigen path runs on: all (default; every channel gets "
        "the gauges) or live (only --n2-project-live-freq-ids, which it writes). The cost is "
        "per channel and ~n^2 in the stations, so live is what pays for a larger station set",
    )
    ap.add_argument(
        "--n2-project-lanes-max",
        type=int,
        default=8,
        help="lane directions projected per channel at most (each costs 1/n of the sky and "
        "~4 n^2 operations per channel-frame; default %(default)s)",
    )
    ap.add_argument(
        "--elem-dir-publish",
        choices=["off", "all"],
        default="off",
        help="GnssGpuRecordAssemble: publish every running non-probe slot's averaged prompt-row "
        "direction per channel on the ProjBoard for GnssN2Project (#190); default off",
    )
    ap.add_argument(
        "--n2-project-live-freq-ids",
        type=str,
        default=None,
        help="the absolute freq_id ranges (lo-hi,...) live mode writes; "
        "'exempt' = the --rfi-excision-exempt-freq-ids ranges. Required with "
        "--n2-project live.",
    )
    ap.add_argument(
        "--elem-proj-deg",
        type=float,
        default=None,
        help="projection window: own-row sources and the probe-stack gate inside this "
        "many degrees of boresight (assembler default 4). Emitted only when given "
        "and --elem-proj-mode is not off.",
    )
    ap.add_argument(
        "--elem-freeze-deg",
        type=float,
        default=None,
        help="shared-model transit freeze: no learning while the broker's nearest-to-"
        "boresight separation is inside this many degrees (assembler default 6). "
        "Emitted only when given, with --elem-sum-shared. The assembler also refuses to "
        "form a first model inside a wider cone (8 deg) and re-seeds a half the fleet "
        "reference stops describing while its sibling pol still does (after 600 s, at "
        "sibling similarity 0.8); those are fixed in the stage, not config keys.",
    )
    ap.add_argument(
        "--elem-shared-ref",
        default="",
        help="the fleet reference snapshot for the shared model's phase pin "
        "(python/scripts/gnss/elem_shared_ref.py snapshot), one vector per band. "
        "Refused unless taken against the epoch of the element positions",
    )
    ap.add_argument(
        "--elem-shared-ref-mode",
        default="off",
        choices=("off", "live"),
        help="off: no reference keys emitted; live: each assembler pins its shared "
        "model to --elem-shared-ref, slewed when warm. Needs --elem-sum-shared. Also "
        "a live switch (POST <assemble>/set_elem_sum_shared_ref)",
    )
    ap.add_argument(
        "--search-host",
        default="127.0.0.1",
        help="host of the search aggregator (--aggregator-instance) that each GPU's "
        "search tap sends to, on --search-port-base + gpu. The default points a node "
        "at itself. Moving the aggregator means changing this and restarting the "
        "nodes.",
    )
    ap.add_argument(
        "--aggregator-instance",
        nargs="+",
        metavar="NODE[:BASE]",
        default=None,
        help="emit one standalone search instance over the union of these nodes'\
 combs (e.g. --aggregator-instance cx19:11040). NODE:BASE's GPU g feed listens on\
 BASE + g; a bare node k (from 0 in this list) on search_port_base + 2k + g: the port that\
 node's config sends to. gen_fleet.py passes the fleet's (config/gnss_fleet.py). No DPDK\
 or hugepages, and no GPU unless --cuda-acquire; runs as an ordinary user.",
    )
    # -- The frame-synced tracker -> broker telemetry transport -------------------------------
    ap.add_argument(
        "--telem-host",
        default=None,
        help="host of the gather instance (--gather-instance) every node sends its per-record "
        "telemetry to (bufferSend -> bufferRecv). Required for a node config. The "
        "gather serves the broker on 127.0.0.1, so it runs on the broker's host.",
    )
    # -- The beam cube: the continuous per-(subband x element) recording ----------------------
    ap.add_argument(
        "--cube-host",
        default=None,
        help="host of the beam-cube archiver (--cube-archive-instance) every node pushes its "
        "completed windows to (bufferSend -> bufferRecv). Required for a node config. "
        "The archiver is its own instance, separate from the telemetry gather and the "
        "search aggregator, so that restarting either does not interrupt the record.",
    )
    ap.add_argument(
        "--cube-archive-instance",
        action="store_true",
        help="emit the standalone beam-cube archiver instance: one bufferRecv for "
        "the whole fleet's cube frames and a rawFileWrite that lands them on "
        "bulk storage. No DPDK, no GPU, no hugepages. Separate from the "
        "gather and the aggregator, so that restarting either does not leave a "
        "hole in the record.",
    )
    ap.add_argument(
        "--gather-instance",
        action="store_true",
        help="emit the standalone gather instance: one bufferRecv for the whole fleet's "
        "telemetry, GnssTelemGather, which hands the frames to the Python broker over a "
        "local socket, and GnssFleetTrim, the fast code loop (docs/CHORD_FAST_TRIM.md). "
        "No DPDK, no GPU, no hugepages; runs as an ordinary user on the broker's host. "
        "GnssTelemGather collates nothing: the frames carry an absolute window index, "
        "so grouping is an exact integer match the broker does itself.",
    )
    ap.add_argument(
        "--aggregator-cores",
        help="logical CPUs the aggregator instance may use, comma-separated "
        "(--aggregator-instance). Default: runtime.cpu_affinity from "
        "config/chord_gnss_node.yaml, a CHORD node's core map. As with --gather-cores, "
        "kotekan's Stage reads cpu_affinity with a required get, so a core the host "
        "lacks fails the pin with EINVAL on each stage. A single-socket host wants all "
        "of its cpus listed.",
    )
    ap.add_argument(
        "--gather-cores",
        help="logical CPUs the gather instance may use, comma-separated "
        "(--gather-instance). Default: runtime.cpu_affinity from "
        "config/chord_gnss_node.yaml, a CHORD node's core map, which another host may "
        "lack. kotekan's Stage reads cpu_affinity with a required get, so the key "
        "cannot simply be dropped; a core that does not exist fails the pin with "
        "EINVAL (logged as ERROR lines) and the stage runs unpinned. The list is "
        "used as the built-in one is (the first four for bufferRecv, then [4] and [5] "
        "modulo its length for GnssTelemGather and GnssFleetTrim), so a single-socket "
        "host wants all of its cpus listed, e.g. 0,1,2,3,4,5.",
    )
    ap.add_argument(
        "--fleet-trim-windows",
        type=int,
        default=2,
        help="windows GnssFleetTrim averages into one discriminator (default "
        "%(default)s). A window is 4 records, so 2 windows = 8 records = 84 ms. The "
        "broker's fleet-trim-bandwidth (config/gnss_chains_chord.yaml) is stable only "
        "while bandwidth x the loop's round trip, which this average is part of, stays "
        "under ~0.5: its 2.5 assumes 2 windows, and a longer average needs a lower "
        "bandwidth.",
    )
    ap.add_argument(
        "--record-dir",
        default=None,
        help="base directory of the --vis-capture writers, which write under "
        "<DIR>/viscap and create it if needed. Default: runtime.record_dir from "
        "config/chord_gnss_node.yaml.",
    )
    ap.add_argument(
        "--search-port-base",
        type=int,
        default=11040,
        help="search-feed port base (default %(default)s). A node's GPU 0 search tap "
        "sends to --search-host on this port and GPU 1 on base+1; --aggregator-instance "
        "listens for node k's GPU g (k from 0) on base+2k+g, so the node bases must be "
        "2 apart, in the order --aggregator-instance lists the nodes.",
    )
    ap.add_argument(
        "--aggregator-extra-search",
        action="append",
        metavar="SIGNAL[:PRNS]",
        default=None,
        help="with --aggregator-instance, add a GnssChannelizedSearch for SIGNAL beside the "
        "primary's, on the same merged comb and cores. SIGNAL is a gnssSignal.hpp "
        "descriptor name whose carrier is the primary's (e.g. GAL_E5A_Q beside GPS L5). "
        "Repeatable. PRNS is a comma list; without it the search scans the constellation's "
        "broadcast PRN range by name prefix (GPS 1-32, GAL 1-36, BDS 19-63, GLO 1-24), and "
        "any other prefix needs PRNS. The stage is named <tag>_search (GAL_E5A_Q -> "
        "e5a_search), the endpoint a broker chain's --detectors names; like every search "
        "it scans only the PRNs a broker hints (that chain's --narrow-search).",
    )
    ap.add_argument(
        "--cuda-acquire",
        action="store_true",
        help="run the aggregator's acquisition surface on the GPU "
        "(--aggregator-instance; docs/gnss_gpu_search.md). The surface never leaves "
        "the device. The engine needs a Doppler grid on the transform's bin lattice "
        "(the generated doppler_step, 62.5 Hz, is one) and declines any other, which "
        "then runs on the CPU with a WARN. Requires a CUDA build; falls back to the CPU "
        "with a WARN otherwise.",
    )
    args = ap.parse_args()
    if args.psi_synth and not args.epl_from_prompt:
        raise SystemExit(
            "--psi-synth requires --epl-from-prompt: the fused synthesis packs two lanes per "
            "slot (P, P_HEAD) with the E/L rows rotated from the prompt"
        )
    args.base = _kotekan_input(args.base)

    with open(NODE_FILE) as fh:
        cfg = yaml.safe_load(fh)
    if args.node not in node_names(cfg):
        raise SystemExit(f"unknown node {args.node}")
    check_telem_windows()
    check_descriptors(cfg, args)
    check_tree_golden()

    # Record epoch (see check_frame0). Only a pinned epoch probes the running fleet.
    pinned = bool(cfg["fengine"].get("frame0_utc")) or bool(args.frame0_nano)
    check_frame0(cfg, args, live_frame0_utc(cfg, args) if pinned else None)

    # --frame0-nano sets the record epoch as well as the telescope's startup GPS time (below):
    # cudaGnssChordTrackState prefers a configured frame0_utc over the telescope's, so setting
    # only the telescope's would leave a node-file value stamping the records.
    if args.frame0_nano:
        cfg["fengine"]["frame0_utc"] = args.frame0_nano * 1e-9

    if args.base.endswith(".j2"):
        base = render_stock_template(args.base)
    else:
        with open(args.base) as fh:
            base = json.load(fh) if args.base.endswith(".json") else yaml.safe_load(fh)
    out = copy.deepcopy(base)

    # --frame0-nano: start without chive's timing service. CHORDTelescope needs a GPS time at
    # startup, which makes chive:54321 a startup dependency of every node (running nodes do not
    # need it). Every running node serves the array-wide frame 0 at telescope/time0_ns:
    #     curl -s http://cx27:12048/telescope/time0_ns
    # Nodes read time0 once at startup, so that value shows only what the F-engine was doing
    # when that node started; it holds while the F-engine keeps counting from the same frame 0.
    # After an F-engine restart re-read it from chive or from a node started since: a stale
    # epoch fails silently, with wrong record stamps and wrong predicted code phases.
    # require_gps stays true: set_gps_time_params_from_config enables GPS time only from a
    # gps_time.frame0_nano, so a missing value still fails at startup rather than falling back
    # to the host clock. config_tracker is dropped too: it fetches from the same service
    # (fpga_controller, chive:54321) and a failed fetch is fatal at startup.
    if args.frame0_nano:
        out.setdefault("telescope", {})["query_gps"] = False
        out["gps_time"] = {"frame0_nano": int(args.frame0_nano)}
        out.pop("config_tracker", None)
        for _k, _v in out.items():
            if isinstance(_v, dict) and "use_config_tracker" in _v:
                _v["use_config_tracker"] = False

    # Widen ConfigTracker's startup fetch budget. Production's template (config/chord/common.j2)
    # sets 2 retries x 10 s, and a failed fetch from chive is fatal at startup. chive's
    # controller (fpga_master) is a single-threaded server that can take more than 10 s to
    # answer while it reads the boards' registers, so that budget fails node starts at random.
    # 5 x 30 s bounds the wait for a chive that is really down at 150 s; a slow start is
    # recoverable, a fleet that half-starts is not. Startup only.
    if "config_tracker" in out and isinstance(out["config_tracker"], dict):
        out["config_tracker"]["upstream_fetch_retries"] = 5
        out["config_tracker"]["upstream_fetch_timeout_seconds"] = 30

    sig = cfg["signals"]
    chans = covering_channels(
        node_channels(cfg, args.node),
        float(sig["carrier_hz"]),
        float(sig["chip_rate_hz"]),
        float(sig["max_doppler_hz"]),
    )
    if not chans:
        raise SystemExit(f"{args.node} holds no covering channels for {sig['primary']}")

    extra_chains = parse_extra_signals(cfg, args, args.node, chans)

    # Split the covering set by which GPU's comb holds each channel, and convert global freq_id
    # to that comb's buffer index -- the tap indexes into the frame, not the sky.
    per_gpu = {}
    for fid in chans:
        gpu, idx = gpu_of_channel(cfg, args.node, fid)
        per_gpu.setdefault(gpu, []).append((fid, idx))

    if args.aggregator_instance:
        # The aggregator has its own port (runtime.aggregator_rest_port), independent of the
        # node port.
        port = (
            args.rest_port
            if args.rest_port is not None
            else cfg["runtime"]["aggregator_rest_port"]
        )
        out, feeds, union_ids = build_aggregator_instance(
            cfg,
            args.aggregator_instance,
            args,
            port,
            out.get("samples_per_data_set", 8192),
        )
        text = (
            "# GENERATED by config/gen_chord_gnss_config.py --aggregator-instance -- DO NOT "
            "HAND-EDIT.\n" + iface_stamp() + "\n"
            f"# nodes {' '.join(args.aggregator_instance)}  signal {sig['primary']}  "
            f"rest port {port}\n"
            f"# union comb: {len(union_ids)} channels {min(union_ids)}..{max(union_ids)}\n"
            "# The off-node acquisition search: one search of the signal above, plus one\n"
            "# per --aggregator-extra-search, over the union of these nodes' combs, aligned\n"
            "# on the F-engine's global sample counter. A NODE:BASE above has its GPU g feed\n"
            "# on BASE + g, the port that node's config sends to.\n"
            "# Regenerate with scripts/gnss/gen_fleet.py config/gnss_fleet_chord.yaml, which\n"
            "# takes the nodes and their ports from the fleet list and the flags from the\n"
            "# manifest's aggregator: block.\n"
            + yaml.safe_dump(out, default_flow_style=False, sort_keys=True)
        )
        if args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            open(args.out, "w").write(text)
            print(f"wrote {args.out}")
        else:
            sys.stdout.write(text)
        print(f"  AGGREGATOR instance over {args.aggregator_instance}", file=sys.stderr)
        for i, (node, gpu, fids, listen_port) in enumerate(feeds):
            print(
                f"  feed {i}: {node} gpu{gpu} {len(fids)} ch  <- port {listen_port}",
                file=sys.stderr,
            )
        print(f"  union    {len(union_ids)} channels", file=sys.stderr)
        print(f"  rest     {port}", file=sys.stderr)
        return

    # The gather and the archiver default to the ports the site runs them on (stack_components.sh
    # binds the gather at 12051, cubearch_up.sh passes 12052), clear of the aggregator's.
    refused = {
        cfg["runtime"]["rest_port"]: "the nodes' REST port",
        cfg["runtime"]["aggregator_rest_port"]: "the aggregator's",
    }

    if args.cube_archive_instance:
        port = (
            args.rest_port
            if args.rest_port is not None
            else cfg["runtime"]["aggregator_rest_port"] + 2
        )
        if port in refused:
            raise SystemExit(
                f"refusing port {port} ({refused[port]}; an off-node instance takes its own)"
            )
        out = build_cube_archive_instance(cfg, args, port)
        n_elem = live_element_count(cfg["array"])
        nbytes = cube_frame_bytes(CUBE_MAX_PRN, CUBE_MAX_BINS, n_elem)
        per_file = nbytes * CUBE_ARCHIVE_FRAMES_PER_FILE
        text = (
            "# GENERATED by config/gen_chord_gnss_config.py --cube-archive-instance -- DO "
            "NOT HAND-EDIT.\n" + iface_stamp() + "\n"
            "# The beam-cube archiver: every sender's completed ~1 s (subband x element)\n"
            f"# windows, landed raw under {CUBE_ARCHIVE_DIR}.\n"
            f"# wire frame {nbytes} B = {CUBE_MAX_PRN} PRN x {CUBE_MAX_BINS} bin "
            f"x {n_elem} elem; listen {CUBE_PORT}; rest {port}\n"
            f"# {CUBE_ARCHIVE_FRAMES_PER_FILE} frames/file = {per_file / 1e6:.1f} MB\n"
            "# The generator's CUBE_MAX_PRN and CUBE_MAX_BINS and the live element count set\n"
            "# frame_size here and on every sender, and bufferRecv closes any connection whose\n"
            "# frame size differs: no data rather than bad data, and nothing says which end is\n"
            "# wrong. When any of them changes, regenerate this file and every node config\n"
            "# together and restart both ends (scripts/gnss/site/cubearch_up.sh has the\n"
            "# command for this file).\n"
            + yaml.safe_dump(out, default_flow_style=False, sort_keys=True)
        )
        if args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            open(args.out, "w").write(text)
            print(f"wrote {args.out}")
        else:
            sys.stdout.write(text)
        print("  BEAM-CUBE ARCHIVER instance", file=sys.stderr)
        print(f"  listens  {CUBE_PORT} (all senders)", file=sys.stderr)
        print(f"  writes   {CUBE_ARCHIVE_DIR}", file=sys.stderr)
        print(f"  frame    {nbytes} B", file=sys.stderr)
        print(f"  rest     {port}", file=sys.stderr)
        return

    if args.gather_instance:
        port = (
            args.rest_port
            if args.rest_port is not None
            else cfg["runtime"]["aggregator_rest_port"] + 1
        )
        if port in refused:
            raise SystemExit(
                f"refusing port {port} ({refused[port]}; an off-node instance takes its own)"
            )
        out = build_gather_instance(cfg, args, port)
        nbytes = telem_frame_bytes(
            TELEM_RECORDS_PER_FRAME, telem_rows_max(), telem_max_chan(),
        )
        # The recipe, as the node configs record it, so the file says how to regenerate it.
        text = (
            "# GENERATED by config/gen_chord_gnss_config.py --gather-instance -- DO NOT "
            "HAND-EDIT.\n" + iface_stamp() + "\n"
            "# REGENERATE WITH (every flag but --out; add it):\n"
            + "\n".join(recipe_lines(sys.argv[1:]))
            + "\n"
            f"# The fleet's per-record telemetry, gathered on one port and served\n"
            f"# to the Python broker on {GATHER_SERVE_HOST}:{GATHER_SERVE_PORT}.\n"
            f"# buffer frame {nbytes} B = {TELEM_RECORDS_PER_FRAME} records x "
            f"{telem_rows_max()} PRN rows x {telem_max_chan()} comb columns"
            f" -- the widest sender, not what any one ships; listen {TELEM_PORT}; "
            f"rest {port}\n"
            "# Senders ship their own shapes (their chain's PRN rows, their instance's comb\n"
            "# columns), and this buffer only has to hold the widest: bufferRecv closes any\n"
            "# connection whose frame exceeds it. Raising TELEM_RECORDS_PER_FRAME, a chain's\n"
            "# rows (_TELEM_ROWS) or gnss::TELEM_MAX_CHAN means regenerating this file with\n"
            "# the node configs and restarting the gather before the nodes.\n"
            + yaml.safe_dump(out, default_flow_style=False, sort_keys=True)
        )
        if args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            open(args.out, "w").write(text)
            print(f"wrote {args.out}")
        else:
            sys.stdout.write(text)
        print(f"  GATHER instance", file=sys.stderr)
        print(f"  listens  {TELEM_PORT} (all senders)", file=sys.stderr)
        print(
            f"  serves   {GATHER_SERVE_HOST}:{GATHER_SERVE_PORT} to the broker",
            file=sys.stderr,
        )
        print(f"  frame    {nbytes} B", file=sys.stderr)
        print(f"  rest     {port}", file=sys.stderr)
        return

    # A node config always ships its telemetry to the gather and its beam cube to the archiver.
    for _flag in ("telem_host", "cube_host"):
        if not getattr(args, _flag):
            raise SystemExit(
                "--%s is required for a node config" % _flag.replace("_", "-")
            )

    # The REST port defaults to production's (see --rest-port in the module docstring). The
    # updatable paths this config subscribes to there are a stock node's: /earth_rotation_data,
    # /rfi_second_stage_enabled, /rfi_second_stage_thresholds and /updatable_config/bad_inputs.
    port = args.rest_port if args.rest_port is not None else cfg["runtime"]["rest_port"]
    # CORS: the live viewer is served from its own HTTP port and calls kotekan's REST
    # cross-origin, so without this the browser blocks every request and the panels stay empty.
    # A wildcard origin is acceptable on the private site network.
    out["rest_server"] = {
        "port": port,
        "cpu_affinity": cfg["runtime"]["cpu_affinity"],
        "enable_cors": True,
    }

    # --- Earth orientation: the current table, not the base's ---------------------------------
    # The table is a rolling window of a few days, and the stock telescope block sets
    # fatal_eop_out_of_range, so a node asked for a time past its last entry shuts down. Take it
    # from choco:/var/lib/choco/eop/state.json, where the observatory refreshes it, falling back
    # to the freshest running node, which holds only what it was last given (nothing when the
    # fleet is down). scripts/gnss/site/eop_cron.sh keeps running nodes current.
    _eop = choco_eop_table()
    if _eop is None:
        _eop = live_eop_table(cfg, args)
    if _eop is not None:
        _tbl, _src, _last = _eop
        out.setdefault("earth_rotation_data", {})[
            "earth_orientation_parameter_table"
        ] = _tbl
        _cov = (_last / 1e9) - time.time()
        print(
            "  eop       %d entries from %s, ends %s (%+.1f days from now)"
            % (
                len(_tbl),
                _src,
                datetime.datetime.utcfromtimestamp(_last / 1e9).strftime("%Y-%m-%d"),
                _cov / 86400.0,
            ),
            file=sys.stderr,
        )
        if _cov < 0:
            print(
                "  WARNING   the EOP table above already ends in the past. The "
                "stock telescope block sets fatal_eop_out_of_range, so a node asked for a time "
                "past the table's last entry shuts down, ours and production's alike -- this "
                "is not a GNSS problem and regenerating will not fix it. Someone must refresh "
                "the fleet's EOP.",
                file=sys.stderr,
            )
    else:
        print(
            "  WARNING   neither choco nor any node served an EOP table -- keeping the base's "
            "earth rotation table, which is empty in the stock template and frozen at the "
            "capture date in a captured base. A node with fatal_eop_out_of_range set (the "
            "stock telescope block sets it) shuts down on such a table; without it, records "
            "carry a wrong dUT1 and any collator that checks frame metadata will reject them.",
            file=sys.stderr,
        )

    # --- Bad inputs: bffs's current list, the other value stock gets by REST ------------------
    # The template ships an empty list. choco relays bffs's list to the stock nodes but not to
    # ours (they are in maintenance mode), and scripts/gnss/site/bad_inputs_cron.py pushes it
    # only to running nodes, so a node starts from whatever its config carries. A wrong list is
    # wrong data in the bad-feed mask stream and the N^2 fold, so take it from bffs, as the EOP
    # table above. The same list leaves the GNSS branch's static element lists (bad_input_set),
    # with config/gnss_bad_inputs.json's on top; that mask stays out of this block.
    _bi = bad_input_block(cfg, args)
    if _bi is not None:
        _blk, _src = _bi
        out.setdefault("updatable_config", {})["bad_inputs"] = _blk
        print(
            "  bad inputs %d flagged, %s from %s; with config/gnss_bad_inputs.json, out of "
            "the GNSS stations, channel-noise inputs and shared ref: %s"
            % (
                len(_blk["bad_inputs"]),
                _blk["update_id"],
                _src,
                sorted(bad_input_set(cfg, args)),
            ),
            file=sys.stderr,
        )
    else:
        print(
            "  WARNING   no source served the bad-input list -- keeping the base's (the stock "
            "template's flags no feed), so every input reads good until "
            "scripts/gnss/site/bad_inputs_cron.py pushes bffs's list to the running node. "
            "gen_fleet refuses to write a config in this state.",
            file=sys.stderr,
        )

    # --- Production's bufferSend legs: dropped unless --n2-send keeps them --------------------
    dropped = []
    n2_send_kept = []
    for key in list(out.keys()):
        if args.n2_send and key.startswith("buffer_send_n2"):
            # A leg not selected is deleted, not just left unaddressed: a bufferSend consumes
            # its buffer whether or not it connects, and a consumer on n2_eigen_buffer lets
            # eigencalc run (see --n2-send-legs).
            want = (
                ("subset" in key)
                if args.n2_send_legs == "subset"
                else ("subset" not in key)
                if args.n2_send_legs == "full"
                else True
            )
            if not want:
                del out[key]
                dropped.append(key)
                continue
            # The endpoint is stamped from the flags rather than inherited from the base, so
            # the destination can be read in the generated file.
            leg = out[key]
            if isinstance(leg, dict):
                leg["server_ip"] = args.n2_send_ip
                leg["server_port"] = (
                    args.n2_send_port_subset
                    if "subset" in key
                    else args.n2_send_port_full
                )
                leg.setdefault("drop_frames", True)
                # recv1's bufferRecv requires the frame-descriptor handshake on every N^2 port
                # (use_frame_desc: true); a sender without it is read as a descriptor and kills
                # the receiver. Stamped like the endpoint, since a captured base may predate it.
                # bufferSend reads reconnect_time; an older base's retry_time is read by nothing.
                leg["use_frame_desc"] = True
                leg["reconnect_time"] = int(
                    leg.pop("retry_time", None) or leg.get("reconnect_time") or 10
                )
                n2_send_kept.append(
                    "%s(%s -> %s:%d)"
                    % (key, leg.get("buf"), leg["server_ip"], leg["server_port"])
                )
            continue
        if args.n2_send and key.startswith("buffer_send_bad_feed_mask"):
            # The mask that goes with the N^2 frames (recv1 records it as /bad_feed_mask), so it
            # follows the N^2 legs.
            leg = out[key]
            if isinstance(leg, dict):
                leg["server_ip"] = args.n2_send_ip
                leg["server_port"] = args.n2_send_port_mask
                n2_send_kept.append(
                    "%s(%s -> %s:%d)"
                    % (key, leg.get("buf"), leg["server_ip"], leg["server_port"])
                )
            continue
        if key.startswith("buffer_send"):
            del out[key]
            dropped.append(key)
            if key.startswith("buffer_send_bad_feed_mask"):
                # Its feeder goes too: bufferCopy into a buffer nobody drains blocks once the
                # buffer fills, and then stops draining host_bf_mask_buffer -- which the GPU
                # RFI stages and N2Accumulate also read.
                for _k in (
                    "copy_bad_feed_mask",
                    "host_bad_feed_mask_send_buffer_0",
                    "host_bad_feed_mask_send_buffer_1",
                ):
                    if out.pop(_k, None) is not None:
                        dropped.append(_k)

    # Channels exempt from first-stage RFI excision (our cudaRFISKtilde key, off upstream;
    # declared in stock_parity.py). The stock SK flagger excises the GNSS main lobes most of the
    # time, and their N2 feeds the satellite projection, so the correlator keeps every sample
    # there; the SK statistics are still computed. Each GPU resolves the absolute ids against
    # its own channels at its first frame.
    if args.rfi_excision_exempt_freq_ids:
        _ids = set()
        for _part in str(args.rfi_excision_exempt_freq_ids).split(","):
            _a, _, _b = _part.strip().partition("-")
            _lo, _hi = int(_a), int(_b or _a)
            if _hi < _lo:
                raise SystemExit(f"--rfi-excision-exempt-freq-ids: bad range {_part!r}")
            _ids.update(range(_lo, _hi + 1))
        _ids = sorted(_ids)
        _n = 0
        for _st in out.values():
            if not isinstance(_st, dict):
                continue
            for _leg in _st.values():
                if not isinstance(_leg, dict):
                    continue
                for _cmd in _leg.get("commands") or []:
                    if isinstance(_cmd, dict) and _cmd.get("name") == "cudaRFISKtilde":
                        _cmd["rfi_first_stage_excision_exempt_freq_ids"] = list(_ids)
                        _n += 1
        if _n == 0:
            raise SystemExit(
                "--rfi-excision-exempt-freq-ids: the base has no cudaRFISKtilde command"
            )
        sys.stderr.write(
            f"  exempted {len(_ids)} freq_ids from first-stage RFI excision "
            f"in {_n} cudaRFISKtilde commands\n"
        )

    # GnssN2Project (lib/stages/gnss/GnssN2Project.cpp): project the dominant coherent
    # components (the transiting satellites) out of the science N2 per frame, before
    # N2Accumulate. One stage per GPU under gnss_n2_project, a gnss* key to the parity check;
    # its output copy replaces N2Accumulate's input, which stock_parity.py declares.
    if args.n2_project != "off":
        _st = project_stations(args.n2_project_stations, bad_input_set(cfg, args))
        _pool0 = gnss_cores(cfg["runtime"], 0)
        _pool1 = gnss_cores(cfg["runtime"], 1)
        out["gnss_n2_project"] = {}
        for _gpu, _pool in ((0, _pool0), (1, _pool1)):
            _corr = "host_correlation_buffer_%d" % _gpu
            _suf = "" if _gpu == 0 else "_1"
            if _corr not in out:
                raise SystemExit("--n2-project: %s is not in the base" % _corr)
            _stage = {
                "kotekan_stage": "GnssN2Project",
                # The telemetry packer's core (see gnss_chain_vars) by default: a light host
                # stage on this GPU's own NUMA pool; --n2-project-core-index picks another slot.
                "cpu_affinity": [
                    int(args.n2_project_cpus.split(",")[_gpu])
                    if args.n2_project_cpus
                    else (
                        _pool[(_gpu + 5) % len(_pool)]
                        if args.n2_project_core_index < 0
                        else _pool[args.n2_project_core_index % len(_pool)]
                    )
                ],
                "in_buf": _corr,
                # num_elements / num_local_freq / samples_per_data_set / sub_integration_ntime
                # are read from the globals (a local `num_elements: num_elements` is a
                # self-reference that recurses the expression evaluator to a stack overflow).
                "mode": args.n2_project,
                "stations": list(_st),
                "k_max": int(N2_PROJECT_K_MAX),
                "tau_s": float(N2_PROJECT_TAU_S),
                "frac_first_min": float(N2_PROJECT_FRAC_MIN),
                "lambda_min_rel": float(N2_PROJECT_LAMBDA_MIN),
                "pr_min": float(N2_PROJECT_PR_MIN),
                "solve_every": int(N2_PROJECT_SOLVE_EVERY),
                "metric_period_s": 1.0,
            }
            if args.n2_project_eigen_channels != "all":
                _stage["eigen_channels"] = args.n2_project_eigen_channels
            if args.n2_project_source != "eigen":
                # #190: the chains' directions. The board group is the assemblers' unique_name
                # up to the first '_' ("/gnss<gpu>"); tracker_stations is the chains' element
                # axis, station of element e = live_element_tiles[e >> 4] * 16 + (e & 15).
                _tiles = live_tile_columns(cfg["array"])
                _stage["source"] = args.n2_project_source
                _stage["board_group"] = "/gnss%d" % _gpu
                _stage["tracker_stations"] = [
                    int(_t) * 16 + _i for _t in _tiles for _i in range(16)
                ]
                _stage["k_lanes_max"] = int(args.n2_project_lanes_max)
                _missing = sorted(
                    set(_stage["tracker_stations"])
                    - set(_st)
                    - bad_input_set(cfg, args)
                )
                if _missing:
                    sys.stderr.write(
                        "  WARNING: %d tracker elements are outside --n2-project-stations "
                        "(e.g. %s): a lane direction cannot null them\n"
                        % (len(_missing), _missing[:6])
                    )
            _lids = n2_project_live_ids(args)
            if not _lids:
                raise SystemExit(
                    "--n2-project live needs --n2-project-live-freq-ids (or 'exempt' with "
                    "--rfi-excision-exempt-freq-ids): GnssN2Project requires live_freq_ids"
                )
            _stage["live_freq_ids"] = sorted(_lids)
            _buf = copy.deepcopy(out[_corr])  # no YAML aliases
            out["gnss_n2_proj_buffer" + _suf] = _buf
            _stage["out_buf"] = "gnss_n2_proj_buffer" + _suf
            out["n2_accumulate"]["accum_%d" % _gpu]["in_buf"] = (
                "gnss_n2_proj_buffer" + _suf
            )
            out["gnss_n2_project"]["proj_%d" % _gpu] = _stage
        sys.stderr.write(
            "  GnssN2Project %s on both GPUs (%d stations, k_max %d, source %s)\n"
            % (args.n2_project, len(_st), int(N2_PROJECT_K_MAX), args.n2_project_source)
        )

    # --- metadata pool for the GNSS chain -----------------------------------------------------
    out["gnss_pool"] = {
        "kotekan_metadata_pool": "GnssChanMetadata",
        "num_metadata_objects": 30 * BUFFER_DEPTH,
    }

    record_floats = None
    for gpu, pairs in sorted(per_gpu.items()):
        blocks, record_floats, n_elem = build_search_leg(
            cfg,
            args.node,
            gpu,
            [i for _, i in pairs],
            args,
            freq_ids=[f for f, _ in pairs],
        )
        blocks.update(
            build_n2dual_branch(
                cfg,
                args.node,
                gpu,
                [i for _, i in pairs],
                [f for f, _ in pairs],
                args,
                out.get("samples_per_data_set", 8192),
            )
        )
        n2dual_chains = [dict(chain=None, chan_idx=[i for _, i in pairs])]
        # Extra signal chains on this GPU, each under its own tag. A chain on the primary's
        # channels shares its voltage tap, so it adds GPU and CPU load but no ingest.
        for ch in extra_chains:
            ch_pairs = (
                [(f, i) for f, i in pairs]
                if ch["shares_tap"]
                else [
                    (f, i) for f, i in gpu_pairs_for(cfg, args.node, gpu, ch["chans"])
                ]
            )
            if not ch_pairs:
                print(
                    f"  NOTE {ch['signal']}: no covering channels on gpu{gpu}, chain skipped",
                    file=sys.stderr,
                )
                continue
            blocks.update(
                build_n2dual_branch(
                    cfg,
                    args.node,
                    gpu,
                    [i for _, i in ch_pairs],
                    [f for f, _ in ch_pairs],
                    args,
                    out.get("samples_per_data_set", 8192),
                    chain=ch,
                )
            )
            n2dual_chains.append(dict(chain=ch, chan_idx=[i for _, i in ch_pairs]))
        merge_n2dual_gpu(
            blocks,
            cfg,
            args,
            args.node,
            gpu,
            n2dual_chains,
            out.get("samples_per_data_set", 8192),
        )
        out.update(blocks)

    # The header is the provenance, complete enough to regenerate from: the argv and a hash of
    # the base. The generator and config/chord_gnss_node.yaml are inputs too, deliberately not
    # recorded as a git SHA: gen_fleet --check regenerates and compares the output, so it fails
    # exactly when either changes the result, and the regeneration is then reviewed as a diff.
    # A .j2 base includes the templates beside it, so all of them are hashed.
    _base_files = [args.base]
    if args.base.endswith(".j2"):
        _base_files = sorted(
            glob.glob(os.path.join(os.path.dirname(args.base), "*.j2"))
        )
    try:
        _h = hashlib.sha256()
        for _f in _base_files:
            with open(_f, "rb") as _bf:
                _h.update(_bf.read())
        base_sha = _h.hexdigest()[:16]
    except Exception:
        base_sha = "unreadable"

    base_disp = _rel(args.base)
    base_hashed = (
        os.path.join(os.path.dirname(base_disp), "*.j2")
        if args.base.endswith(".j2")
        else base_disp
    )
    hdr = (
        [
            "# GENERATED by config/gen_chord_gnss_config.py -- DO NOT HAND-EDIT.",
            iface_stamp(),
            f"# node {args.node}  base {os.path.basename(args.base)}  signal {sig['primary']}",
            f"# covering channels {len(chans)}: freq_id {chans[0]}..{chans[-1]}",
            f"# rest port {port} (production's by default: our kotekan runs in place of theirs)",
            f"# outputs disabled: {sorted(dropped)[:4]}{' ...' if len(dropped) > 4 else ''}",
            f"# base sha256 {base_sha} ({base_hashed})",
            "# REGENERATE WITH (every flag but --out; add it, or let gen_fleet write the file):",
        ]
        + recipe_lines(sys.argv[1:])
        + [
            "# Prefer `scripts/gnss/gen_fleet.py config/gnss_fleet_chord.yaml`, which holds the",
            "# whole fleet's flags in one versioned file and can --check them against these.",
            "# Edit config/chord_gnss_node.yaml and regenerate instead of hand-editing here.",
        ]
    )
    text = (
        "\n".join(hdr)
        + "\n"
        + yaml.safe_dump(out, default_flow_style=False, sort_keys=True)
    )
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)

    print(f"  node          {args.node}", file=sys.stderr)
    print(
        f"  covering ch   {len(chans)} -> per GPU "
        + ", ".join(f"gpu{g}:{len(v)}" for g, v in sorted(per_gpu.items())),
        file=sys.stderr,
    )
    print(
        f"  record_floats {record_floats} (RECORD_FLOATS + n_elem*ELEM_FLOATS)",
        file=sys.stderr,
    )
    print(f"  rest port     {port}", file=sys.stderr)
    print(f"  dropped       {len(dropped)} production blocks", file=sys.stderr)
    if n2_send_kept:
        # The one flag that puts our frames into shared science storage, so the destination
        # is printed at generation time.
        print("  N2 send armed " + "; ".join(n2_send_kept), file=sys.stderr)

    # Extra chains share each GPU's fixed core pool (runtime.cpu_affinity_gpu<N>), so a
    # multi-chain node oversubscribes it; the per-chain rotation only keeps different chains'
    # copies of the same heavy stage off one core. Print the busiest cores and the taps so the
    # cost shows at generation time.
    if extra_chains:
        for ch in extra_chains:
            n_here = sum(
                1 for g in per_gpu if gpu_pairs_for(cfg, args.node, g, ch["chans"])
            )
            print(
                f"  + chain       {ch['signal']} tag '{ch['tag'].strip('_')}' "
                f"{len(ch['prns'])} PRNs, {len(ch['chans'])} ch on {n_here} gpu(s), "
                f"tap {'SHARED with the primary' if ch['shares_tap'] else 'OWN (new carrier)'}",
                file=sys.stderr,
            )
        print("  cpu load      " + load_summary(out, cfg, per_gpu), file=sys.stderr)
        print(
            "  NB extra chains are dead-reckon seeded: no search feed is emitted for them "
            "(docs/CHORD_MULTIBAND.md section 5)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
