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

NODE_FILE = os.path.join(CONF, "chord_gnss_node.yaml")

# ── Fixed settings ───────────────────────────────────────────────────────────────────────────
# Several are wire or alignment contracts that every sender and receiver must agree on: change
# one, then regenerate every node config and the gather and archiver configs together.

# 10.49 ms at CHORD's 5.12 us hop; divides the 8192-hop frame 4 ways and is shorter than the
# 20 ms period of the NH20-tiled codes, so a record straddles at most one code-period boundary
# (where the P_HEAD row splits it).
HOPS_PER_RECORD = 2048
# Frames per GNSS chain buffer.
BUFFER_DEPTH = 4
# The primary chain's PRN slots (each extra chain carries its own list).
PRIMARY_PRNS = list(range(1, 33))
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
# --fleet-trim-windows, which is far too short for the policy cycle's statistics. Instances
# required before a discriminator forms.
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


def signal_tag(name):
    """Short buffer/endpoint suffix for an extra signal chain, e.g. GAL_E5A_Q_CS -> _e5a.

    Must be distinct across the chains on one node: every buffer, stage and REST endpoint of
    the chain is named from it, and kotekan keys its per-consumer bookkeeping by stage name.
    """
    parts = name.split("_")
    body = parts[1] if len(parts) > 1 else parts[0]
    return "_" + body.lower()


def dll_spacing_chips(name):
    """Early/Late tap offset from prompt, in the signal's own (component) chips.

    The offset is set by the width of the correlation the loop closes on, not by the chip: E
    and L must straddle that correlation's mainlobe, inside its first null, or the
    discriminator reverses sign off-peak and the loop has stable false locks.

    Every chain but GPS L2C despreads a multi-freq_id comb inside one instance, so its
    correlation is chip-narrow and 0.5 chip sits on the slope. L2C's covering set is six
    freq_ids, fewer than the comb stride, so each instance despreads one 195 kHz channel, whose
    |sinc|^2 nulls at 2.6 CM chips. But the fleet loop sums the senders' complex taps
    coherently, and on L2C the senders are five contiguous channels, so the loop sees a
    5-channel Dirichlet whose first null is 1/(5B) = 0.52 CM chips. The offset must sit inside
    that combined width, where the discriminator is monotonic (q on peak is then ~6; a q far
    higher means the taps have landed in a null, not that the lock improved).

    Must match the broker chain's dll-spacing (config/gnss_chains_chord.yaml): the C++ fleet
    loop takes its law's spacing from the arming payload while the taps come from here, so
    cycle the nodes before the broker and the taps lead the law.
    """
    return {"GPS_L2C_CM": 0.4}.get(name, 0.5)


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


def broker_chain_name(name):
    """gnssSignal.hpp name -> the broker's chain key. GPS_L5_Q -> gps_l5, GAL_E5A_Q_CS -> gal_e5a.

    config/gnss_chains_chord.yaml keys its chains the same way (gps_l5, gal_e5a, bds_b2a, ...),
    so the key is derived rather than carried in a table. It goes on the telemetry wire
    (gnssTelem.hpp) as a name, not an integer id: an id would need identical tables in the
    generator, the C++ and the broker, and a drifted one would attribute one constellation's
    data to another with nothing to say so.
    """
    return "_".join(name.split("_")[:2]).lower()


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
                "have different PRN ranges and the count IS the path-B lane budget "
                "(4 lanes/PRN of 128), so it is not something to default silently."
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
            # chain is skipped with a note on stderr rather than an error; gen_fleet prints a
            # generator's stderr only when it fails, so a fleet regeneration hides the note.
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
                "--extra-signal %s / %s are the two SIDEBANDS of one AltBOC transmission from "
                "the same satellite, so their PRN lists must be identical.\n"
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
    n_live = live_element_count(arr)
    num_synth = 128

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
            "assemble": cores[(gpu + 2) % nc],
            # The combiner is the heaviest host stage (its cost scales with PRN count), so it
            # rotates per chain like dual and tiles: stacked on one core, the chains' combiners
            # together approach a full core, and any backlog walks back through the shared
            # voltage ring into the F-engine ingest (and the science N2 with it).
            "combine": cores[(gpu + 4 + 3 * ordv) % nc],
            "telem": cores[(gpu + 5) % nc],
            "send": cores[(gpu + 6) % nc],
            "sink": cores[(gpu + 7) % nc],
        },
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
    """
    return [(int(a), int(b)) for a, b in arr["live_element_ranges"]]


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
    element ranges. Returns (positions, epoch_key).

    The epoch table records what the array actually was, by date, and refuses rather than
    extrapolates. A position the table does not know is a refusal too: a steer built from a
    guessed position is a wrong steer, not a weaker one.
    """
    sys.path.insert(0, os.path.join(CONF, "..", "python", "scripts", "gnss"))
    import gnss_arraymap

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
    return out, ep.key()


def elem_proj_keys(args):
    """Assembler keys for the bright-satellite projection (gnssProjSubspace.hpp); nothing unless
    --elem-proj-mode is shadow or live, so a fleet without it is byte-identical. The mode is
    also a live switch (POST /set_elem_proj); this key makes it survive a restart."""
    mode = getattr(args, "elem_proj_mode", "off") or "off"
    if mode not in ("shadow", "live"):
        return {}
    out = {"elem_proj_mode": mode}
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


def elem_shared_ref_keys(args, band, steer, n_elem):
    """One band's entry of the fleet reference snapshot (python/scripts/gnss/elem_shared_ref.py
    snapshot) for the shared model's phase pin; nothing unless --elem-shared-ref-mode is live
    and --elem-sum-shared is on, so a fleet without it is byte-identical. The snapshot is of the
    steered instrument, so it is refused unless it was taken against the epoch these positions
    came from."""
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
            round(float(x), 6) for pair in entry["ref"] for x in pair
        ],
        "elem_sum_shared_ref_mode": mode,
    }


def elem_steer_keys(args, arr, n_elem):
    """The assembler keys that arm steering for one band: the measured positions of the array
    epoch valid now (refined by --elem-positions-file), the measured sign, and the epoch key the
    positions came from, so a config can be matched to the array it was built for."""
    pos, key = elem_positions_from_arraymap(arr, n_elem)
    if args.elem_positions_file:
        # A refinement on top of the epoch, solved from steered per-element phases and written
        # in cube order against a named epoch key. It is valid only for the epoch it was solved
        # against (a re-cabling makes a different array), so the key must match the epoch valid
        # now and the order must be the epoch's cube_order.
        ref = json.load(open(args.elem_positions_file))
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

    # Tile count per channel: see n2_tiles_per_chan (one contract, three sites).
    num_synth = 128
    # A frame is samples_per_data_set hops (spds, from the base config, which production owns),
    # split into records of HOPS_PER_RECORD hops.
    n_rec_per_frame = max(1, int(spds) // HOPS_PER_RECORD)
    # The tiles frame's time axis is records per frame: the dual correlator integrates one
    # record (sub_integration_ntime below).
    tiles_frame_bytes = (
        n_rec_per_frame * n_chan * n2_tiles_per_chan(n_live, num_synth) * 512 * 4
    )

    if 4 * n_prn > num_synth:
        raise SystemExit(
            f"--n2-dual{tag}: 4*{n_prn} PRNs exceeds {num_synth} synthetic "
            f"lanes (4 lanes per PRN slot). The lane budget is PER CHAIN here "
            f"(each signal has its own synth ring), so this one signal is too "
            f"wide; across signals the binding limit is GPU memory, 1.61 GB per "
            f"ring. Shrink this chain's PRN list, or raise NSB to 256."
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
                e for lo, hi in live_element_ranges(arr) for e in range(lo, hi + 1)
            ],
            **elem_shared_keys(args),
            **elem_shared_ref_keys(
                args, (tag or signal_tag(sig["primary"])).strip("_"), steer, n_live
            ),
            **elem_proj_keys(args),
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
            "record_export": 128,
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
            f"{CUBE_MAX_PRN}; raise it on EVERY sender and the archiver."
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

    Lanes: chains that share a channel get disjoint 128-lane ranges; chains on disjoint
    channels reuse a range, since their replicas never meet in a product. The ranges are whole
    multiples of 128, so a chain's slot arithmetic (4 lanes per slot from its range start) and
    its 8 tile rows are those of a chain on its own.
    """
    sig = cfg["signals"]
    rt = cfg["runtime"]
    lanes = 128
    pres = []
    for c in chains:
        ch = c["chain"]
        tag = ch["tag"] if ch else ""
        pres.append(
            dict(
                pre=f"gnss{gpu}{tag}_",
                tag=tag,
                chain=ch,
                name=broker_chain_name(ch["signal"] if ch else sig["primary"]),
                chan_idx=[int(x) for x in c["chan_idx"]],
                chan=set(int(x) for x in c["chan_idx"]),
                viscap=vis_capture_chain(args, ch, sig),
            )
        )
    union = sorted(set().union(*[p["chan"] for p in pres]))
    for p in pres:
        base = 0
        while any(
            q["chan"] & p["chan"]
            and q["base"] < base + lanes
            and base < q["base"] + lanes
            for q in pres
            if "base" in q
        ):
            base += lanes
        p["base"] = base
    num_synth = max(p["base"] for p in pres) + lanes

    procs = {p["pre"]: blocks.pop(f"{p['pre']}n2dual") for p in pres}
    primary = procs[f"gnss{gpu}_"]
    merged = {k: v for k, v in primary.items() if k != "commands"}
    # Private CUDA streams: cuda_stream_base moves this process's copy-in, copy-out and kernel
    # streams to 3, 4, 5, leaving 0, 1, 2 to production's pipelines, so the GNSS chains never
    # queue against production's per-stream locks (one thread blocked in the CUDA driver there
    # would stall the GPU's pipelines and back up to the NIC). prepareStreams only grows the
    # device's pool, so asking for 6 holds whatever the construction order.
    merged["cuda_stream_base"] = 3
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
        corr[f"gather_{p['name']}_lanes"] = lanes

    # The capture: one gather over the captured chains' channels and lane rows, with the live
    # N^2 and, with --vis-capture-bb, the replica block across those chains; one gate holds
    # that frame in step with every captured chain's ctl block (its slot -> PRN map).
    cap_blocks = {}
    captured = [p for p in pres if p["viscap"]]
    if captured:
        cap_chans = sorted(set().union(*[p["chan"] for p in captured]))
        base = min(p["base"] for p in captured)
        cap_lanes = max(p["base"] for p in captured) + lanes - base
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
        f"  gpu{gpu}: {len(pres)} path-B chains in one pass, {len(union)} channels, "
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


def search_stage(cfg, args, in_buf, chan_ids, core):
    """The aggregator's GnssChannelizedSearch block."""
    sig = cfg["signals"]
    fe = cfg["fengine"]
    n_chan = len(chan_ids)
    return {
        "kotekan_stage": "GnssChannelizedSearch",
        "in_buf": in_buf,
        "signal": sig["primary"],
        "prns": PRIMARY_PRNS,
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


def build_aggregator_instance(cfg, nodes, args, port):
    """One acquisition search over the union of several nodes' combs. One node GPU's feed holds
    ~7 of the ~106 L5 covering channels (-11.8 dB), which no amount of local integration
    recovers; the union of the feeds is denser by construction. The feeds arrive over
    bufferSend/bufferRecv, one per node GPU; GnssChanAlignMerge aligns them on the F-engine's
    global sample counter (equal sample_seq is simultaneity, so the union search is coherent
    across nodes), and one GnssChannelizedSearch runs on the merged comb.

    Port layout: feed i (node-major, GPU-minor) listens on search_port_base + i. Each node's
    config sends to its own --search-port-base + gpu, so the nodes' bases must be 2 apart in
    the order given here (config/gnss_fleet_chord.yaml).
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
        "samples_per_data_set": 8192,
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

    # One feed per (node, gpu), node-major to match the port assignment. The merge
    # concatenates columns in feed order, so union_ids is built in that order; the search
    # takes it as channel_ids and never assumes sortedness.
    feeds = []
    for node in nodes:
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
            feeds.append((node, gpu, pg[gpu]))

    union_ids = []
    in_buf_names = []
    in_channels = []
    for i, (node, gpu, fids) in enumerate(feeds):
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
                    "listen_port": args.search_port_base + i,
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
                    # decode (indistinguishable from a nibble swap, and invisible to the
                    # |.|^2-only X-engine); without this the despread matches no satellite. The
                    # stage refuses false. See GnssChordDequantize.cpp.
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


def live_bad_inputs(cfg, args, timeout=5.0):
    """The bad-input block a running stock node (runtime.eop_reference_nodes) holds, as
    (block, host), or None. The fallback when bffs's state cannot be read: choco keeps those
    nodes current. A block still at the template's initial_flags is skipped.
    """
    import urllib.request

    port = args.rest_port if args.rest_port is not None else cfg["runtime"]["rest_port"]
    for host in cfg["runtime"].get("eop_reference_nodes", []):
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

    port = args.rest_port if args.rest_port is not None else cfg["runtime"]["rest_port"]
    # Our nodes and the stock reference nodes; the freshest table wins. After a long outage of
    # our own, the stock nodes are the ones still being refreshed.
    hosts = list(cfg["nodes"]) + [
        h
        for h in cfg["runtime"].get("eop_reference_nodes", [])
        if h not in cfg["nodes"]
    ]
    best = None
    for host in hosts:
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

    # The nodes' REST port, which carries the telescope endpoints.
    port = args.rest_port if args.rest_port is not None else cfg["runtime"]["rest_port"]
    seen = {}
    for node in cfg["nodes"]:
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


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--base",
        required=True,
        help="production config: a .j2 template, rendered STOCK exactly as kotekan "
        "renders it (config/chord/pathfinder.j2 -- what the manifest uses), or "
        "a JSON/yaml config",
    )
    ap.add_argument("--node", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--rest-port",
        type=int,
        default=None,
        help="default: runtime.rest_port from the node table (NOT 12048)",
    )
    ap.add_argument(
        "--n2-send-legs",
        choices=("both", "full", "subset"),
        default="both",
        help="WHICH --n2-send legs to build (default both). Legs not selected are "
        "DELETED from the config, not merely left unaddressed -- a bufferSend "
        "is a buffer CONSUMER whether or not its socket connects, and that is "
        "the whole point of this flag.\n"
        "\n"
        "⚠️ THE TWO LEGS CARRY DIFFERENT RISK, learned the hard way on "
        "2026-08-18 when arming both core-dumped all six nodes in a minute:\n"
        "  full   = n2_buffer, the ACCUMULATED VISIBILITIES. n2_buffer already "
        "has a consumer (n2_subset), so it is flowing today and a second "
        "consumer changes nothing upstream. This is the genuine last-hop-only "
        "change, and it is the N^2 data.\n"
        "  subset = n2_eigen_buffer, the EIGEN products. That buffer has NO "
        "consumer, so eigencalc (EigenN2Iter) has been blocked at "
        "wait_for_empty_frame for the life of this instrument. Arming this leg "
        "makes it run for the first time -- and it throws immediately: "
        "'Invalid assignment to diagonal matrix element' from blaze's "
        "HermitianMatrix, i.e. a visibility DIAGONAL (an autocorrelation) "
        "arrived with a non-zero imaginary part. Stock data, stock stages, our "
        "config merely uncovered it. Keep this off until it is fixed upstream.",
    )
    ap.add_argument(
        "--rf-stats-period-s",
        type=float,
        default=10.0,
        help="Seconds between RF-monitor passes (default 10). The pass reads the "
        "frame IN PLACE -- it does not copy it -- so this bounds a ~1 ms cost, "
        "not a 402 MB memcpy.",
    )
    ap.add_argument(
        "--n2-send",
        action="store_true",
        help="SHIP THE N^2 VISIBILITIES to the standard downstream consumer, i.e. "
        "run stock-equivalent on the science pipeline. Keeps the four "
        "buffer_send_n2_* legs from the captured base (n2_eigen_buffer -> "
        "--n2-send-port-subset, n2_buffer -> --n2-send-port-full, both GPUs) "
        "and STAMPS the endpoint explicitly rather than inheriting whatever "
        "the capture happened to hold -- a silently inherited destination is "
        "not a destination you can grep for before a deploy.\n"
        "\n"
        "MEASURED 2026-08-18: our nodes already COMPUTE these products. The "
        "run_n2k command list is byte-identical to config/chord/pathfinder.j2's, "
        "host_correlation_buffer has the identical shape/dtype, and "
        "n2_accumulate carries identical parameters (238 subints/bin, "
        "bin_in_ERA over 8640, CHORDBeamformer, EvenOddPosDef, no fringestop) "
        "-- the products just had no consumer and were dropped on the floor. "
        "This flag is the last hop, not a new pipeline.\n"
        "\n"
        "COST: 7.7 MB/s per node (full 128-element FullUpperTri + the 32-element "
        "eigen subset), 0.37 Gbps fleet-wide. drop_frames stays true, so a slow "
        "or absent receiver degrades instead of backpressuring the GPU chain.\n"
        "\n"
        "⚠️ IT WRITES INTO SHARED SCIENCE STORAGE -- the default "
        "endpoint is the same host and ports choco's production nodes use, so "
        "our frames land beside theirs. Flip it on deliberately.",
    )
    ap.add_argument(
        "--n2-send-ip",
        default="10.222.0.51",
        help="destination host for --n2-send (default: the standard CHORD N^2 "
        "receiver, matching config/chord/pathfinder.j2)",
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
        help="arm the (N+M)^2 visibility capture on these broker chains (comma "
        "list; bare flag = gps_l5). Each GPU's correlator adds one capture gather "
        "(the live antennas' N^2 plus the captured chains' mixed rows), and a "
        "REST-gated FrameWindowGate feeds it with those chains' ctl frames to "
        "rawFileWrites under <record_dir>/viscap. Nothing is written until "
        "scripts/gnss/viscap.py arms a window.",
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
        help="start WITHOUT chive: supply the F-engine frame-0 time (int64 ns) "
        "instead of querying chive:54321, and drop config_tracker (same "
        "service). Get the value from a node that is CURRENTLY PROCESSING "
        "DATA: curl -s http://cx27:12048/telescope/time0_ns . require_gps "
        "stays TRUE, so a missing/bad value still FATALs rather than silently "
        "falling back to host wall clock. "
        "*** VALID ONLY WHILE THE F-ENGINE KEEPS COUNTING. *** frame 0 is "
        "re-established by an F-engine restart or firmware/mode change, and "
        "every node CACHES it at startup -- so a value read from a running "
        "node proves only what the F-engine was doing when THAT node started, "
        "not what it is doing now. Re-read it after any F-engine restart. "
        "Using a stale one fails SILENTLY: records carry the wrong epoch, the "
        "seeds predict the wrong code phase, and every PRN despreads noise "
        "while looking like a clock or geometry bug.",
    )
    ap.add_argument(
        "--extra-signal",
        action="append",
        default=[],
        metavar="NAME:PRNS",
        help="add a SECOND tracker chain for another signal, e.g. "
        "--extra-signal GAL_E5A_Q_CS:1,2,3. Repeatable. PRNs are REQUIRED "
        "(constellations have different PRN ranges, and the count is the "
        "path-B lane budget -- 4 lanes per PRN). A chain whose covering "
        "channels match the primary's SHARES its voltage tap (E5a/B2a are on "
        "L5's carrier, so this is the free case); a different carrier gets its "
        "own tap. Buffer/endpoint names are suffixed with a tag derived from "
        "the signal name. See docs/CHORD_MULTIBAND.md.",
    )
    ap.add_argument(
        "--elem-positions-file",
        default="",
        help="#102: a refined-positions JSON (cube order, epoch_key, positions_enu) "
        "applied on top of the array-epoch positions (config/chord_array_epochs.json "
        "via gnss_arraymap); refused unless its epoch_key is the epoch valid now",
    )
    ap.add_argument(
        "--elem-steer-sign",
        type=float,
        default=1.0,
        help="#102 steering phase sign -- a MEASURED convention (see "
        "gnssElemSteer.hpp); calibrated live: e5a vs its band-sibling",
    )
    ap.add_argument(
        "--reference-element",
        type=int,
        default=0,
        help="antenna whose correlation fills the record HEADER -- the broker's DLL "
        "and carrier loop reference, and the phase anchor of the calibrated "
        "element sum",
    )
    ap.add_argument(
        "--elem-sum-shared",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="SHARED instrument model in the assembler: one per-element gain per "
        "polarisation learned as a slow consensus of every satellite's cal, "
        "plus one inter-pol coefficient per satellite; per-PRN weights are "
        "held to it (elem_sum_adapt false), so a boresight transit cannot "
        "capture the weak satellites' element sums. Emitted only when on.",
    )
    ap.add_argument(
        "--elem-proj-mode",
        default="off",
        choices=("off", "shadow", "live"),
        help="bright-satellite projection in the assembler (PROJECTION_PLAN.md): "
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
        "--n2-project",
        choices=["off", "live"],
        default="off",
        help="GnssN2Project on the science N2 (phase 3a): live = solve each "
        "channel's dominant subspace per frame and N2Accumulate reads the projected "
        "copy (gnss_n2_proj_buffer{,_1}).",
    )
    ap.add_argument(
        "--n2-project-stations",
        type=str,
        default="0-15,56-79,120-127",
        help="live correlator station ranges the projection works on",
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
        "FORM a first model inside a wider cone (8 deg) and re-seeds a half the fleet "
        "reference stops describing while its sibling pol still does (after 600 s, at "
        "sibling similarity 0.8); those are fixed in the stage, not config keys (#170).",
    )
    ap.add_argument(
        "--elem-shared-ref",
        default="",
        help="#154: the fleet reference snapshot (elem_shared_ref.py snapshot), one "
        "vector per band, for the shared model's phase pin",
    )
    ap.add_argument(
        "--elem-shared-ref-mode",
        default="off",
        choices=("off", "live"),
        help="off: no reference keys (byte-identical); live: each assembler pins "
        "to the reference, slewed when warm. Also a live switch (POST "
        "<assemble>/set_elem_sum_shared_ref)",
    )
    ap.add_argument(
        "--search-host",
        default="127.0.0.1",
        help="where the search instance listens. Localhost today; changing this is "
        "the ONLY edit needed to move the search to another machine.",
    )
    ap.add_argument(
        "--aggregator-instance",
        nargs="+",
        metavar="NODE",
        default=None,
        help="emit ONE standalone search instance over the UNION of these nodes'\
 combs (e.g. --aggregator-instance cx19 cx27). Feeds listen on search_port_base + i in\
 node-major, GPU-minor order -- the same ports the node configs already target. No\
 DPDK/GPU/hugepages; runs as an ordinary user.",
    )
    # -- The frame-synced tracker -> broker telemetry transport -------------------------------
    ap.add_argument(
        "--telem-host",
        default=None,
        help="TASK #59: the gather instance every node sends its per-record "
        "telemetry to (bufferSend -> bufferRecv). Required for a node config. "
        "The gather is normally the broker host.",
    )
    # -- The beam cube: the continuous per-(subband x element) recording ----------------------
    ap.add_argument(
        "--cube-host",
        default=None,
        help="the beam-cube archiver every node pushes its completed windows to "
        "(bufferSend -> bufferRecv). Required for a node config. Normally cf06 "
        "(10.222.3.6), which also runs the search aggregator and the telemetry "
        "gather -- but the archiver is a SEPARATE instance from both, because "
        "a restart of either must never interrupt the record.",
    )
    ap.add_argument(
        "--cube-archive-instance",
        action="store_true",
        help="emit the standalone beam-cube ARCHIVER instance: one bufferRecv for "
        "the whole fleet's cube frames and a rawFileWrite that lands them on "
        "bulk storage. No DPDK, no GPU, no hugepages. SEPARATE from the "
        "gather and the aggregator on purpose -- a gather restart wipes every "
        "standing trim and the aggregator is restarted often, and neither "
        "should be able to punch a hole in a record.",
    )
    ap.add_argument(
        "--gather-instance",
        action="store_true",
        help="TASK #59: emit the standalone GATHER instance -- one bufferRecv for "
        "the whole fleet's telemetry plus GnssTelemGather, which hands the "
        "frames to the PYTHON broker over a local socket. No DPDK, no GPU, no "
        "hugepages; runs as an ordinary user beside the search aggregator on "
        "cf06. It collates NOTHING: the frames carry an absolute window index, "
        "so grouping is an exact integer match the broker does itself.",
    )
    ap.add_argument(
        "--aggregator-cores",
        help="logical CPUs the aggregator instance may use, comma-separated "
        "(--aggregator-instance). Same trap as --gather-cores: the default "
        "comes from the base config's runtime.cpu_affinity, which is the "
        "SEARCH HOST's core map, and kotekan's Stage reads cpu_affinity with a "
        "required get -- so a core that does not exist is an EINVAL per stage, "
        "not a no-op. A single-socket host wants all of its cpus listed.",
    )
    ap.add_argument(
        "--gather-cores",
        help="logical CPUs the gather instance may use, comma-separated "
        "(--gather-instance). Default: the base config's runtime.cpu_affinity, "
        "which is the SEARCH HOST's core map -- wrong the moment the gather "
        "runs anywhere else. kotekan's Stage reads cpu_affinity with a REQUIRED "
        "get, so the key cannot simply be dropped; a core that does not exist "
        "fails the pin with EINVAL, six ERROR lines per start, and falls back "
        "to unpinned. The list is consumed exactly as the built-in one is -- "
        "first four for the buffer, then [4] and [5] modulo its length -- so a "
        "single-socket host wants all of its cpus listed, e.g. 0,1,2,3,4,5.",
    )
    ap.add_argument(
        "--fleet-trim-windows",
        type=int,
        default=4,
        help="windows GnssFleetTrim averages into one discriminator. 4 windows = "
        "16 records = 168 ms.",
    )
    ap.add_argument(
        "--record-dir",
        default=None,
        help="override runtime.record_dir from the node file. Needed on any node "
        "without /data: rawFileWrite takes base_dir from the config, so pointing "
        "the LOGS elsewhere does not move the RECORDS, and the stage fails on a "
        "directory that is not there.",
    )
    ap.add_argument(
        "--search-port-base",
        type=int,
        default=11040,
        help="bufferRecv port for GPU 0; GPU 1 uses base+1",
    )
    ap.add_argument(
        "--cuda-acquire",
        action="store_true",
        help="run the acquisition surface on the GPU (docs/gnss_gpu_search.md). "
        "Blind dims measured 8.3 ms against 1.15 s of CPU, the surface never "
        "leaves the device, and the peak agrees with the CPU path to rel 6e-8. "
        "ALSO sets doppler_step to the bin-aligned 62.5 Hz -- the engine "
        "declines a half-bin grid and falls back to the CPU. Requires a CUDA "
        "build; falls back with a WARN otherwise.",
    )
    args = ap.parse_args()

    with open(NODE_FILE) as fh:
        cfg = yaml.safe_load(fh)
    if args.node not in cfg["nodes"]:
        raise SystemExit(f"unknown node {args.node}")

    # Record epoch. cudaGnssChordTrackState takes frame0_utc (the UTC of the F-engine's absolute
    # sample 0; records are stamped frame0_utc + wstart/rate) from Telescope::to_time_ns(0)
    # unless the config carries one, so by default none is emitted and nothing can go stale. A
    # pinned value (the node file's fengine.frame0_utc, or --frame0-nano) goes stale silently at
    # the next F-engine restart, since every cross-record estimator works in differences; only
    # then is the running fleet probed, and a node-file value that disagrees with it is an error.
    pinned = bool(cfg["fengine"].get("frame0_utc")) or bool(args.frame0_nano)
    live_t0 = live_frame0_utc(cfg, args) if pinned else None
    if not pinned:
        print(
            "  epoch     frame0_utc NOT emitted -- the stages take it from the telescope's "
            "live GPS time0 (week-rollover corrected). Nothing to keep in step.",
            file=sys.stderr,
        )
    if live_t0 is not None:
        have = float(cfg["fengine"].get("frame0_utc", 0.0))
        if abs(have - live_t0) > 1e-3:
            raise SystemExit(
                f"frame0_utc in {NODE_FILE} is {have:.9f} but the running fleet serves "
                f"{live_t0:.9f} ({abs(have - live_t0) / 86400.0:.2f} days apart).\n"
                f"The F-engine has been restarted since that value was written. Records would "
                f"carry the wrong epoch and nothing downstream would say so.\n"
                f"Fix it:  sed -i 's/^  frame0_utc: .*/  frame0_utc: {live_t0:.9f}/' "
                f"{NODE_FILE}\n"
                f"(or pass --frame0-nano to override both this and the telescope block.)\n"
                f"BETTER: DELETE frame0_utc from {NODE_FILE} entirely -- the stages now "
                f"read the live telescope epoch and this whole class of staleness goes away."
            )
    elif not args.frame0_nano:
        # stderr: without --out, stdout is the config (see live_frame0_utc).
        print(
            "  WARNING   no running node answered telescope/time0_ns -- frame0_utc "
            f"{float(cfg['fengine'].get('frame0_utc', 0.0)):.6f} is UNVERIFIED. If the "
            "F-engine has restarted since it was written, every record stamp is wrong.",
            file=sys.stderr,
        )

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

    # Name the quantity each RFI ring copy produces. cudaCopyFromRingbuffer builds its output
    # buffer's ndarray descriptor from the ring's slot-0 metadata, which on a fresh pipeline can
    # be a pool object recycled from another quantity, still carrying its dimension names.
    # Publishing that conflicts with the buffer's descriptor, ensure_frame_desc() fails fatally
    # and the node shuts down. A size check cannot catch it (dim[0] is forced from frame_size),
    # so with expect_quantity_name set the copy defers until the metadata names this quantity.
    # Names from the producing kernels' array declarations (cudaRFISKtilde, cudaRFISKbar).
    _RING_QUANTITY = {
        "rfi_skbar_buffer": "SKbar",
        "rfi_sktilde_buffer": "SKtilde",
        "rfi_RFImask_buffer": "RFImask",
    }
    _stamped = 0
    for _st in out.values():
        if not isinstance(_st, dict):
            continue
        for _leg in _st.values():
            if not isinstance(_leg, dict):
                continue
            for _cmd in _leg.get("commands") or []:
                if (
                    isinstance(_cmd, dict)
                    and _cmd.get("name") == "cudaCopyFromRingbuffer"
                    and _cmd.get("gpu_mem_input") in _RING_QUANTITY
                    and "expect_quantity_name" not in _cmd
                ):
                    _cmd["expect_quantity_name"] = _RING_QUANTITY[_cmd["gpu_mem_input"]]
                    _stamped += 1
    # stderr: without --out, stdout is the config.
    if _stamped:
        sys.stderr.write(
            f"  hardened {_stamped} ring copies with expect_quantity_name\n"
        )

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
            cfg, args.aggregator_instance, args, port
        )
        text = (
            "# GENERATED by config/gen_chord_gnss_config.py --aggregator-instance -- DO NOT "
            "HAND-EDIT.\n"
            f"# nodes {' '.join(args.aggregator_instance)}  signal {sig['primary']}  "
            f"rest port {port}\n"
            f"# union comb: {len(union_ids)} channels {min(union_ids)}..{max(union_ids)}\n"
            "# ONE search over the gathered union, aligned on the global sample counter.\n"
            + yaml.safe_dump(out, default_flow_style=False, sort_keys=True)
        )
        if args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            open(args.out, "w").write(text)
            print(f"wrote {args.out}")
        else:
            sys.stdout.write(text)
        print(f"  AGGREGATOR instance over {args.aggregator_instance}", file=sys.stderr)
        for i, (node, gpu, fids) in enumerate(feeds):
            print(
                f"  feed {i}: {node} gpu{gpu} {len(fids)} ch  <- port "
                f"{args.search_port_base + i}",
                file=sys.stderr,
            )
        print(f"  union    {len(union_ids)} channels", file=sys.stderr)
        print(f"  rest     {port}", file=sys.stderr)
        return

    if args.cube_archive_instance:
        port = (
            args.rest_port
            if args.rest_port is not None
            else cfg["runtime"]["rest_port"] + 3
        )
        if port == 12048:
            raise SystemExit("refusing port 12048 (choco owns it)")
        out = build_cube_archive_instance(cfg, args, port)
        n_elem = live_element_count(cfg["array"])
        nbytes = cube_frame_bytes(CUBE_MAX_PRN, CUBE_MAX_BINS, n_elem)
        per_file = nbytes * CUBE_ARCHIVE_FRAMES_PER_FILE
        text = (
            "# GENERATED by config/gen_chord_gnss_config.py --cube-archive-instance -- DO "
            "NOT HAND-EDIT.\n"
            "# THE BEAM-CUBE ARCHIVER: every sender's completed ~1 s (subband x element)\n"
            f"# windows, landed raw under {CUBE_ARCHIVE_DIR}.\n"
            f"# wire frame {nbytes} B = {CUBE_MAX_PRN} PRN x {CUBE_MAX_BINS} bin "
            f"x {n_elem} elem; listen {CUBE_PORT}; rest {port}\n"
            f"# {CUBE_ARCHIVE_FRAMES_PER_FILE} frames/file = {per_file / 1e6:.1f} MB\n"
            "# ⚠️ cube_max_prn, cube_max_bins and the live element count must match EVERY\n"
            "#    sender: together they set frame_size, and bufferRecv closes any\n"
            "#    connection that disagrees -- it delivers no data rather than bad data,\n"
            "#    which is safe but says nothing about which end is wrong.\n"
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
            else cfg["runtime"]["rest_port"] + 2
        )
        if port == 12048:
            raise SystemExit("refusing port 12048 (choco owns it)")
        out = build_gather_instance(cfg, args, port)
        nbytes = telem_frame_bytes(
            TELEM_RECORDS_PER_FRAME, telem_rows_max(), telem_max_chan(),
        )
        _cmd = "config/gen_chord_gnss_config.py " + " ".join(
            shlex.quote(a) for a in sys.argv[1:]
        )
        _wrapped = "\n".join(
            "#       " + _cmd[i : i + 92] for i in range(0, len(_cmd), 92)
        )
        # Record the exact arguments, as the node configs do, so the file says how to
        # regenerate it.
        text = (
            "# GENERATED by config/gen_chord_gnss_config.py --gather-instance -- DO NOT "
            "HAND-EDIT.\n"
            "# REGENERATE WITH:\n" + _wrapped + "\n"
            f"# TASK #59: the fleet's per-record telemetry, gathered on ONE port and handed\n"
            f"# to the PYTHON broker on {GATHER_SERVE_HOST}:{GATHER_SERVE_PORT}.\n"
            f"# buffer frame {nbytes} B = {TELEM_RECORDS_PER_FRAME} records x "
            f"{telem_rows_max()} PRN rows x {telem_max_chan()} comb columns"
            f" -- THE WIDEST SENDER, not what any one ships; listen {TELEM_PORT}; "
            f"rest {port}\n"
            "# ⚠️ SENDERS SHIP THEIR OWN SHAPES -- their chain's PRN rows, their instance's\n"
            "#    comb columns -- and this buffer only has to HOLD the widest. bufferRecv\n"
            "#    closes any connection whose frame EXCEEDS its buffer, so raising a\n"
            "#    chain's rows means regenerating this file in the same commit.\n"
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
                "  WARNING   the FRESHEST table any node serves already ENDS IN THE PAST. "
                "Every node is extrapolating, ours and production's alike -- this is not a "
                "GNSS problem and regenerating will not fix it. Someone must refresh the "
                "fleet's EOP.",
                file=sys.stderr,
            )
    else:
        print(
            "  WARNING   no node served telescope/eop_table -- keeping the BASE's earth "
            "rotation table, which is frozen at the capture date and is almost certainly "
            "stale. Records will carry a wrong dUT1 and any collator that checks frame "
            "metadata will reject them.",
            file=sys.stderr,
        )

    # --- Bad inputs: bffs's current list, the other value stock gets by REST ------------------
    # The template ships an empty list. choco relays bffs's list to the stock nodes but not to
    # ours (they are in maintenance mode), and scripts/gnss/site/bad_inputs_cron.py pushes it
    # only to running nodes, so a node starts from whatever its config carries. A wrong list is
    # wrong data in the bad-feed mask stream and the N^2 fold, so take it from bffs, as the EOP
    # table above.
    _bi = choco_bad_inputs()
    if _bi is None:
        _bi = live_bad_inputs(cfg, args)
    if _bi is not None:
        _blk, _src = _bi
        out.setdefault("updatable_config", {})["bad_inputs"] = _blk
        print(
            "  bad inputs %d flagged, %s from %s"
            % (len(_blk["bad_inputs"]), _blk["update_id"], _src),
            file=sys.stderr,
        )
    else:
        print(
            "  WARNING   no source served the bad-input list -- keeping the template's, so "
            "every input reads GOOD until bffs next CHANGES the list. gen_fleet refuses to "
            "write a config in this state.",
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
        _st = []
        for _part in str(args.n2_project_stations).split(","):
            _a, _, _b = _part.strip().partition("-")
            _st.extend(range(int(_a), int(_b or _a) + 1))
        _st = sorted(set(_st))
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
                # The telemetry packer's core (see gnss_chain_vars): a light host stage on this
                # GPU's own NUMA pool.
                "cpu_affinity": [_pool[(_gpu + 5) % len(_pool)]],
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
            _lf = args.n2_project_live_freq_ids
            if _lf == "exempt":
                _lf = args.rfi_excision_exempt_freq_ids
            if not _lf:
                raise SystemExit(
                    "--n2-project live needs --n2-project-live-freq-ids (or 'exempt' with "
                    "--rfi-excision-exempt-freq-ids): GnssN2Project requires live_freq_ids"
                )
            _lids = set()
            for _part in str(_lf).split(","):
                _a, _, _b = _part.strip().partition("-")
                _lids.update(range(int(_a), int(_b or _a) + 1))
            _stage["live_freq_ids"] = sorted(_lids)
            _buf = copy.deepcopy(out[_corr])  # no YAML aliases
            out["gnss_n2_proj_buffer" + _suf] = _buf
            _stage["out_buf"] = "gnss_n2_proj_buffer" + _suf
            out["n2_accumulate"]["accum_%d" % _gpu]["in_buf"] = (
                "gnss_n2_proj_buffer" + _suf
            )
            out["gnss_n2_project"]["proj_%d" % _gpu] = _stage
        sys.stderr.write(
            "  GnssN2Project %s on both GPUs (%d stations, k_max %d)\n"
            % (args.n2_project, len(_st), int(N2_PROJECT_K_MAX))
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

    # Paths inside the repo are recorded relative to it, so the recipe holds on any checkout.
    def _rel(p):
        ap_ = os.path.abspath(p)
        return os.path.relpath(ap_, K_ROOT) if ap_.startswith(K_ROOT + os.sep) else p

    K_ROOT = os.path.dirname(CONF)
    # --out is left out of the recipe: it says where the file was written, not how it was
    # built, and gen_fleet regenerates to stdout and compares byte for byte, so a recorded --out
    # would never match. Add --out when pasting the recipe, or let gen_fleet write the file.
    _SINK_FLAGS = ("--out",)
    _argv = list(sys.argv[1:])
    _keep = []
    _i = 0
    while _i < len(_argv):
        if _argv[_i] in _SINK_FLAGS:
            _i += 2
            continue
        if any(_argv[_i].startswith(f + "=") for f in _SINK_FLAGS):
            _i += 1
            continue
        _keep.append(_argv[_i])
        _i += 1
    argv_line = " ".join(shlex.quote(_rel(a)) for a in _keep)
    base_disp = _rel(args.base)
    base_hashed = (
        os.path.join(os.path.dirname(base_disp), "*.j2")
        if args.base.endswith(".j2")
        else base_disp
    )
    hdr = (
        [
            "# GENERATED by config/gen_chord_gnss_config.py -- DO NOT HAND-EDIT.",
            f"# node {args.node}  base {os.path.basename(args.base)}  signal {sig['primary']}",
            f"# covering channels {len(chans)}: freq_id {chans[0]}..{chans[-1]}",
            f"# rest port {port} (our nodes replace production's kotekan on these six)",
            f"# outputs disabled: {sorted(dropped)[:4]}{' ...' if len(dropped) > 4 else ''}",
            f"# base sha256 {base_sha} ({base_hashed})",
            "# REGENERATE WITH (this is the whole recipe -- flags included):",
        ]
        + [
            "#   " + ln
            for ln in textwrap.wrap(
                "config/gen_chord_gnss_config.py " + argv_line,
                width=92,
                subsequent_indent="    ",
                break_on_hyphens=False,
                # A long --extra-signal PRN list must stay whole: the recipe is meant to be pasted.
                break_long_words=False,
            )
        ]
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
    print(f"  record_floats {record_floats} (26 header + n_elem*12)", file=sys.stderr)
    print(f"  rest port     {port}", file=sys.stderr)
    print(f"  dropped       {len(dropped)} production blocks", file=sys.stderr)
    if n2_send_kept:
        # The one flag that puts our frames into shared science storage, so the destination
        # is printed at generation time.
        print("  N2 SEND ARMED  " + "; ".join(n2_send_kept), file=sys.stderr)

    # Extra chains share each GPU's fixed core pool (runtime.cpu_affinity_gpu<N>), so a
    # multi-chain node oversubscribes it; the per-chain rotation only keeps different chains'
    # copies of the same heavy stage off one core. Print the busiest cores and the taps so the
    # cost shows at generation time.
    if extra_chains:
        load = {}
        for k, v in out.items():
            if k.startswith("gnss") and isinstance(v, dict) and "cpu_affinity" in v:
                for c in v["cpu_affinity"]:
                    load.setdefault(c, []).append(v.get("kotekan_stage", "?"))
        busiest = sorted(load.items(), key=lambda kv: -len(kv[1]))[:3]
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
        print(
            "  cpu load      "
            + ", ".join(f"core {c}: {len(v)} stages" for c, v in busiest)
            + f"  (pool of {len(cfg['runtime']['cpu_affinity'])})",
            file=sys.stderr,
        )
        print(
            "  NB extra chains are DEAD-RECKON seeded: no search feed is emitted for them "
            "(docs/CHORD_MULTIBAND.md section 5)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
