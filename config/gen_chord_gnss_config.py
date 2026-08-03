#!/usr/bin/env python3
"""Generate a CHORD GNSS node config: the production config PLUS a GNSS branch.

A CHORD node cannot run a second kotekan instance -- DPDK owns the NICs
exclusively -- so the GNSS chain has to live in the SAME process as the ingest.
This generator therefore takes the production config as a BASE and injects into
it, rather than restating the pipeline. Upstream changes to chord_pathfinder.j2
are picked up for free, and the ingest stages we depend on are preserved by
construction rather than by copy.

    # capture the base from a running node (or a rendered .j2)
    curl -s http://cx19:12048/config -o base.json

    python3 config/gen_chord_gnss_config.py --base base.json --node cx19 \
        --out config/generated/chord_gnss_cx19.yaml

Safety switches, both ON by default because the intended use is a shared node:

    --rest-port 12049   Do NOT collide with production's 12048. choco talks to
                        12048; an instance answering there can be redirected or
                        reconfigured out from under us.
    --disable-outputs   Drop the bufferSend legs so nothing is injected into the
                        downstream N2 consumer at 10.222.0.51. Our records go to
                        local disk only.

Use --keep-n2 to retain the science pipeline (correlator, RFI, eigen) alongside
the GNSS branch. Off by default: the GNSS branch only needs ingest, and running
the N2 chain as well competes for the same GPUs for no benefit while debugging.
"""

import argparse
import copy
import json
import os
import sys

import yaml

CONF = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CONF)

from chord_band_plan import all_band_channels, covering_channels, node_channels  # noqa: E402

DEFAULT_NODE_FILE = os.path.join(CONF, "chord_gnss_node.yaml")

# Stages that exist ONLY to feed the N2 science products. Dropped unless --keep-n2.
#
# ⚠️ GET THIS LIST WRONG AND THE PIPELINE SILENTLY STALLS. A kotekan stage blocks until its
# inputs arrive, so dropping a stage that some SURVIVING stage consumes from does not raise an
# error -- the consumer simply waits forever and the whole ingest wedges with no log line.
# Two mistakes, both found on the first live run (2026-07-29):
#
#   * process_packet_mask was in this list. It is INGEST, not science: it turns DPDK's
#     per-packet receipt bitmaps into host_pl_mask_exp_buffer, which TransposeBasebandArray
#     REQUIRES. Dropped it, and both transposes blocked on their first frame -- DPDK happily
#     filled network_input_buffer to 21/24 while host_voltage_buffer stayed empty.
#
#   * run_send_voltage was NOT in this list and needed to be. It uploads the full 402 MB frame
#     to the GPU for the N2 correlator; with run_n2k dropped its output ringbuffer has no
#     consumer, so it fills, blocks, and stops releasing host_voltage_buffer frames -- stalling
#     the transposes from the other side. Our tap reads host_voltage_buffer directly and does
#     not want that upload anyway, so dropping it also saves 9.6 GB/s of PCIe.
#
# The rule: drop only what nothing surviving CONSUMES from. --dry-run now checks this
# (buffers left with consumers but no producer) instead of leaving it to a live stall.
N2_STAGE_PREFIXES = (
    "run_n2k", "n2_accumulate", "eigencalc", "n2_subset", "hex_dump",
    "buffer_send_n2", "compute_RFI_frame_mask", "count_rfi_mask", "rfi_sk_metrics",
    "run_rfi_", "run_recv_rfi_", "run_pl_", "count_PL", "PL_mask_compactor",
    "run_send_pl_mask", "set_bf_mask", "run_send_voltage",
)


def gpu_of_channel(cfg, node, freq_id):
    """Which of the node's two GPUs (0/1) holds this freq_id, and its index in that comb."""
    off16 = cfg["nodes"][node]["gpu_offsets_mod16"]
    for gpu, off in enumerate(off16):
        if freq_id % 16 == off:
            sb = cfg["science_band"]
            first = next(f for f in range(sb["min_freq_id"], sb["max_freq_id"] + 1)
                         if f % 16 == off)
            return gpu, (freq_id - first) // 16
    raise SystemExit(f"freq_id {freq_id} is not on {node} (offsets {off16})")


def build_gnss_branch(cfg, node, gpu, chan_idx, args, freq_ids=None):
    """The GNSS stages+buffers to inject, for one GPU's covering channels."""
    sig = cfg["signals"]
    rt = cfg["runtime"]
    arr = cfg["array"]
    n_elem = arr["live_elements"][1] - arr["live_elements"][0] + 1
    elem0 = arr["live_elements"][0]
    n_chan = len(chan_idx)
    cores = rt["cpu_affinity"]
    pre = f"gnss{gpu}_"

    tap_out = f"{pre}volt_buf"
    rec_buf = f"{pre}rec_buf"
    cmb_buf = f"{pre}cmb_buf"

    # Record frames carry the element axis: RECORD_FLOATS + n_elem*ELEM_FLOATS per PRN
    # (gnssRecord.hpp record_stride()). Kept in one place; the stages assert against it.
    record_floats = 24 + n_elem * 12
    n_prn = len(args.prns)

    blocks = {
        tap_out: {
            "kotekan_buffer": "standard",
            "metadata_pool": "gnss_pool",
            "num_frames": args.buffer_depth,
            # [hop][chan][elem], one byte per complex sample
            "frame_size": f"samples_per_data_set * {n_chan} * {n_elem}",
        },
        rec_buf: {
            "kotekan_buffer": "standard",
            "metadata_pool": "gnss_pool",
            "num_frames": args.buffer_depth,
            "frame_size": f"{n_prn} * {record_floats} * sizeof_float32",
        },
        cmb_buf: {
            "kotekan_buffer": "standard",
            "metadata_pool": "gnss_pool",
            "num_frames": args.buffer_depth,
            "frame_size": f"{n_prn} * {record_floats} * sizeof_float32",
        },
        f"{pre}tap": {
            "kotekan_stage": "GnssChordVoltageTap",
            "in_buf": f"host_voltage_buffer_{gpu}",
            "out_buf": tap_out,
            "chan_ids": chan_idx,
            "n_elements": n_elem,
            "element_offset": elem0,
            "frame_chan_stride": "num_local_freq",
            "frame_elem_stride": "num_elements",
            # NB: do NOT emit `samples_per_data_set: samples_per_data_set` here. Config
            # expressions resolve identifiers by walking UP from the current path, so a key
            # whose value names ITSELF resolves to itself and recurses until the stack dies --
            # and it dies inside Config's std::regex tokenizer, so the backtrace is 200 frames
            # of regex internals with nothing pointing at the config. The stage's lookup already
            # walks up to the top-level value, so simply omitting the key is both correct and
            # safe.
            "fft_length": cfg["fengine"]["fft_length"],
            "cpu_affinity": [cores[gpu % len(cores)]],
        },
        f"{pre}epl_buf": {
            "kotekan_buffer": "standard",
            "metadata_pool": "gnss_pool",
            "num_frames": args.buffer_depth,
            # gnss_gpu::frame_bytes(n_prn, n_chan, ROWS_PLAIN=4, n_elem): header + winstart +
            # PrnCtl + corr[jobs][chan][elem] + energy[jobs][chan]. MAX_REC = 16.
            "frame_size": (48 + 8 * 16 + 64 * 16 * n_prn
                           + 16 * 4 * n_prn * 16 * n_chan * n_elem
                           + 8 * 4 * n_prn * 16 * n_chan),
        },
        f"{pre}gpu": {
            "kotekan_stage": "cudaProcess",
            "gpu_id": gpu,
            "cpu_affinity": [cores[(gpu + 6) % len(cores)]],
            "in_buffers": {"gnss_volt_in": tap_out},
            "out_buffers": {"gnss_epl_out": f"{pre}epl_buf"},
            "commands": [
                {"name": "cudaInputData", "in_buf": "gnss_volt_in",
                 "gpu_mem": f"{pre}voltage"},
                {"name": "cudaGnssChordTrack",
                 # F-engine conjugation, measured on sky 2026-07-30 (see GnssChordDequantize).
                 "conjugate": True,
                 "gpu_mem_input": f"{pre}voltage",
                 "gpu_mem_output": f"{pre}epl",
                 # NH-baked tracker code (2026-07-31): multi-period records despread the bare
                 # primary to ~zero (NH20 partial sums cancel); see chord_gnss_node.yaml.
                 "signal": sig.get("tracker", sig["primary"]),
                 # GLOBAL bins of this GPU's covering comb, in the tap's local order. The
                 # replica for a channel must be built at ITS OWN sky frequency, and CHORD's
                 # comb is stride-16, so a contiguous chan_offset cannot describe it -- passing
                 # 0 put every replica at DC and nothing ever locked (2026-07-31).
                 "channel_ids": list(freq_ids if freq_ids is not None else chan_idx),
                 "prns": args.prns,
                 "n_channels": n_chan,
                 "n_elements": n_elem,
                 "elem_stride": n_elem,
                 "frame_chan_stride": n_chan,
                 # THE REPLICA'S CARRIER, and on CHORD it is the SKY frequency, not an IF.
                 # The airspy node downconverts, so its f_offset is the post-mixer IF (a few
                 # MHz). CHORD does not downconvert at all: the RFSoC samples 0-1600 MHz
                 # directly and L5 sits at bin 6023 of 8192. Leaving this at the 0.0 default
                 # generates the replica at DC, so covering_bins lands on bins -51..52 while
                 # the data is at 5971..6076 -- zero overlap, and every correlation is noise.
                 "f_offset_hz": float(sig["carrier_hz"]),
                 "hops_per_record": args.hops_per_record,
                 "fft_length": cfg["fengine"]["fft_length"],
                 "sample_rate": float(cfg["fengine"]["sampling_rate_MHz"]) * 1e6,
                 "seed_endpoint": f"/{pre}track/set_seeds",
                 # In-tracker DLL code trim (ported 2026-07-31): per-frame closure is the only
                 # loop fast enough for the clock chain's +-1 chip / ~20 s breathing. The
                 # broker's own DLL (3c) sees disc ~ 0 once this holds and stays quiet.
                 "code_trim": True,
                 "trim_endpoint": f"/{pre}track/get_trim",
                 # GPS-disciplined UTC of absolute sample 0 -- without it the assembler
                 # stamps records with HOST wall clock (see cudaGnssChordTrack.cpp).
                 "frame0_utc": float(cfg["fengine"].get("frame0_utc", 0.0))},
                {"name": "cudaSyncOutput"},
                {"name": "cudaOutputData", "gpu_mem": f"{pre}epl",
                 "out_buf": "gnss_epl_out"},
            ],
        },
        f"{pre}assemble": {
            "kotekan_stage": "GnssGpuRecordAssemble",
            "in_buf": f"{pre}epl_buf",
            "out_buf": rec_buf,
            "prns": args.prns,
            "n_elements": n_elem,
            "reference_element": args.reference_element,
            "sample_rate": float(cfg["fengine"]["sampling_rate_MHz"]) * 1e6,
            "cpu_affinity": [cores[(gpu + 8) % len(cores)]],
        },
        # SEARCH FEED. A SECOND tap on the same voltage buffer, taking ONE element -- the
        # acquisition search is single-antenna by design, so this is ~57 kB/frame (1.4 MB/s)
        # rather than the 1.8 MB/frame the tracking tap moves. Shipped as 4+4b bytes and
        # unpacked on the far side, which is 8x cheaper on the wire than sending floats.
        f"{pre}srch_buf": {
            "kotekan_buffer": "standard",
            "metadata_pool": "gnss_pool",
            "num_frames": args.buffer_depth,
            "frame_size": f"samples_per_data_set * {n_chan} * 1",
        },
        f"{pre}srch_tap": {
            "kotekan_stage": "GnssChordVoltageTap",
            "in_buf": f"host_voltage_buffer_{gpu}",
            "out_buf": f"{pre}srch_buf",
            "chan_ids": chan_idx,
            "n_elements": 1,
            "element_offset": elem0 + args.search_element,
            "frame_chan_stride": "num_local_freq",
            "frame_elem_stride": "num_elements",
            "fft_length": cfg["fengine"]["fft_length"],
            "cpu_affinity": [cores[(gpu + 1) % len(cores)]],
        },
        f"{pre}srch_send": {
            "kotekan_stage": "bufferSend",
            "buf": f"{pre}srch_buf",
            "server_ip": args.search_host,
            "server_port": args.search_port_base + gpu,
            # The search must NEVER back-pressure the ingest: acquisition is a bootstrap
            # convenience, the science chain is not.
            "drop_frames": True,
            "retry_time": 5.0,
            # PIN THE WIRE FORMAT EXPLICITLY on both ends. bufferSend/bufferRecv default
            # use_config_tracker to whether the INSTANCE has a /config_tracker block, and the two
            # instances differ: the node inherits one from the production base config, the search
            # instance has none. The sender then writes a 3-field header and the receiver reads a
            # 2-field one, the stream shifts, and frame_size is read out of the neighbouring
            # field -- surfacing as "Frame size does not match between server: 57344 and client:
            # 12", where 12 is sizeof(GnssChanMetadata). Nothing about that message points at
            # config_tracker, so pin it rather than inherit it.
            "use_config_tracker": False,
            "cpu_affinity": [cores[(gpu + 3) % len(cores)]],
        },
        f"{pre}combine": {
            "kotekan_stage": "GnssCoherentCombiner",
            "in_bufs": [rec_buf],
            "out_buf": cmb_buf,
            "n_prn": n_prn,
            "n_elements": n_elem,
            "integration_length": args.integration_length,
            "integration_mode": "rolling",
            # DEEP COHERENT INTEGRATION, no wipe. The tracker despreads GPS_L5_Q_NH -- the
            # 204600-chip code with NH20 baked in -- so each record's amplitude already has the
            # overlay removed, chip by chip, and the deep integration is a straight sum. It
            # cannot use the wipe rungs: overlay_apply advances the overlay one chip per RECORD,
            # an identity that holds on airspy (record = one primary period) and fails here
            # (2048 hops = 10.4857 periods). Without this the combiner has NO route to
            # coherence_s at all and integrates one 10.5 ms record at a time.
            "deep_coherent": True,
            "cpu_affinity": [cores[(gpu + 2) % len(cores)]],
        },
        f"{pre}record": {
            "kotekan_stage": "rawFileWrite",
            "in_buf": cmb_buf,
            "base_dir": args.record_dir or rt["record_dir"],
            "file_name": f"{node}_gnss{gpu}_cmb",
            "file_ext": "raw",
            "cpu_affinity": [cores[(gpu + 4) % len(cores)]],
        },
    }
    return blocks, record_floats, n_elem


def _comb_g(chan_ids, fft_len):
    """gcd of the covering channels' frequency differences with fft_len.

    This is exactly aggregate_accumulate's `g`: it sets s_stored = fft_len/g, the period of the
    fine-lag axis, and hence how far the refine has to search. g == 1 means the axis spans a
    whole hop and the coarse phase is unambiguous.
    """
    from math import gcd
    if not chan_ids:
        return fft_len
    g = 0
    for c in chan_ids[1:]:
        g = gcd(g, abs(c - chan_ids[0]))
    return gcd(g, fft_len) if g else fft_len


def search_stage(cfg, args, in_buf, chan_ids, core):
    """The GnssChannelizedSearch block, shared by the per-node and aggregator instances so
    their search configuration CANNOT drift apart -- every hard-won constant below applies
    identically to both."""
    sig = cfg["signals"]
    fe = cfg["fengine"]
    n_chan = len(chan_ids)
    return {
        "kotekan_stage": "GnssChannelizedSearch",
        "in_buf": in_buf,
        "signal": sig["primary"],
        "prns": args.prns,
        "sample_rate": float(fe["sampling_rate_MHz"]) * 1e6,
        # Sky carrier, not an IF -- see the note on the tracker's f_offset_hz.
        # covering_bins() sets local.carrier_hz = f_offset, so 0.0 makes the search
        # look for the signal at DC and report "carrier not in this subband".
        "f_offset": float(sig["carrier_hz"]),
        "spectrum_length": fe["num_bins"],
        "num_taps": fe["pfb_taps"],
        "pfb_window": fe["pfb_window"],
        # The SPARSE comb, addressed explicitly. Correct and 15x cheaper than zero-filling
        # to a contiguous span (see the dequantize note).
        "channel_offset": chan_ids[0],
        "n_channels": n_chan,
        "channel_ids": list(chan_ids),
        # Explicit bounds are a FALLBACK: with /clock_profile present the stage derives
        # the extent itself. Kept so the config is readable without knowing that.
        "doppler_min": -float(sig["max_doppler_hz"]),
        "doppler_max": float(sig["max_doppler_hz"]),
        # 31.25 Hz = 1/(2*T_coh) for the 16 ms (3125-hop) coherent window. A coherent
        # window of length T cannot tolerate a Doppler error much past 1/(2T): at the
        # old 250 Hz step a half-bin miss is 125 Hz, which rotates the phase TWO FULL
        # CYCLES across 16 ms and lands the correlation in a sinc null. The Doppler grid
        # resolution and the coherent window length are ONE parameter, not two.
        "doppler_step": 31.25,
        # 30 s, matching the live airspy L5 chain -- NOT the 8 s default. The broker
        # refreshes hints every 10 s, so an 8 s TTL guarantees a window each cycle where
        # every hint is stale; with require_hint that means every PRN is skipped before
        # its SNR is even computed, and the stage goes silent for a reason nothing in its
        # output explains. Two timers that must agree, and the defaults do not.
        "hint_ttl_s": 30.0,
        "hold_snapshots": 5,
        "code_doppler_sign": 1.0,
        "doppler_margin_hz": float(sig["max_doppler_hz"]),
        # Only search PRNs the broker has HINTED, and only near the hinted Doppler. The
        # broker knows every visible satellite's Doppler to ~Hz from BRDC, so a blind
        # grid over all 32 PRNs is work we can simply not do.
        "require_hint": True,
        "acquire_windows": args.acquire_windows,
        "acquire_snr": args.acquire_snr,
        "acquire_fine_step": args.acquire_fine_step,
        "prns_per_pass": args.prns_per_pass,
        # L5 Q5 is a dataless PILOT, which does not mean unmodulated: it carries the
        # NH20 secondary overlay, one +-1 chip per 1 ms code period. The coherent window
        # is 16 ms = SIXTEEN code periods, so an overlay-blind replica sums 16 chips of a
        # near-balanced sequence: measured -12.7 dB rms over the 20 alignments, and
        # EXACTLY ZERO for three of them. Consecutive windows step the alignment by
        # 16 mod 20, so a snapshot only visits {0,4,8,12,16} -- one of which is a null.
        # Searching the 20 alignments recovers full coherent gain; 12.7 dB won
        # COHERENTLY would need ~350x more incoherent windows to match.
        "nh_search": True,
        # Post-detection code-phase refine: an exact despread per step, over +-1 hop.
        # The stage default step is 1 SAMPLE, right for the airspy bank (fft_len 20 ->
        # 41 despreads) and catastrophic here (fft_len 16384 -> 32769) -- and it would
        # only bite on the FIRST DETECTION, looking like a hang precisely when the thing
        # finally worked. Also pointless precision: only the covering channels enter the
        # despread, so it cannot resolve better than ~fft_len/n_chan samples. Step at 1/8.
        # Step at HALF the covering set's intrinsic resolution (fft_len/n_chan samples), not an
        # eighth: once satellites actually detect, the refine runs PER DETECTION and its cost is
        # step-count x 0.7 s -- at /8 with 27 channels that was 437 steps ~ 5 min per detected
        # satellite, which with ~8 detecting sats pushed a "fast" 8-window pass to 30+ min and
        # made every published Doppler stale before the broker could solve on it. /2 keeps the
        # cp error well inside the comb ambiguity the model resolves anyway.
        "refine_step": max(1, int(2 * fe["num_bins"] / max(1, n_chan) / 2)),
        # SPAN: derived from the comb, not guessed. channelized_peak's coarse phase used to be
        # wrong by up to a whole hop (the fine lag was added with the wrong sign, 2026-08-03),
        # so the refine had to scan +-fft_len to find the peak at all -- 426 evaluations, and
        # the single biggest item in a pass at 88 s. Corrected, its chosen offset falls from
        # +50.27 chips to -0.44 and a +-512 sample window is ample: measured 88.50 -> 4.74 s
        # with the answer unchanged (period exact, 0.107 chips, and 0/8 wrong lobe or period at
        # noise 20 AND 40).
        #
        # But only when g == 1. When the covering channels share a factor g with fft_len the
        # fine axis stores just fft_len/g columns, so the recovered phase is right only modulo
        # that -- the grating-lobe ambiguity, 13.09 chips at g=4 -- and the span must still
        # cover it. 8 nodes give g=1; fewer can too (any two nodes with adjacent comb offsets),
        # so compute it rather than assume the node count.
        "refine_span": 512 if _comb_g(chan_ids, fe["num_bins"] * 2) == 1
                       else fe["num_bins"] * 2,
        # The aggregate is parallel over Doppler bins and is the whole cost of an aggregator
        # pass (27 channels x 4096 lags x nd bins ~ 10 s/window on one core). The per-node
        # instances keep 1 thread; the aggregator overrides after construction.
        "acquire_threads": 1,
        "cpu_affinity": [core],
    }




def build_aggregator_instance(cfg, nodes, args, port):
    """ONE search over the union of several nodes' combs -- the sensitivity the per-node
    searches cannot reach. A node sees 7 of the ~106 covering channels (-11.8 dB) and no
    amount of local integration recovers that; the union of N nodes is denser by
    construction (all eight mod-8 offsets -> the full contiguous cover). The feeds arrive
    exactly as for per-node search instances (bufferSend/bufferRecv, one per node GPU);
    GnssChanAlignMerge aligns them on the F-engine's GLOBAL sample counter -- equality of
    sample_seq IS simultaneity, so the union search is coherent across nodes -- and one
    GnssChannelizedSearch runs on the merged comb.

    Port layout: feed i (node-major, GPU-minor) listens on search_port_base + i, which is
    exactly where the node configs already send (cx19 -> 11040/11041 local, cx27 ->
    11042/11043 at --search-host cx19).
    """
    sig = cfg["signals"]
    rt = cfg["runtime"]
    cores = rt["cpu_affinity"]
    out = {
        "type": "config",
        "log_level": "info",
        "samples_per_data_set": 8192,
        "sizeof_float32": 4,
        "cpu_affinity": cores,
        "rest_server": {"port": port, "cpu_affinity": cores, "enable_cors": True},
        "clock_profile": {"name": cfg["clock"]["profile"]},
        "gnss_pool": {"kotekan_metadata_pool": "GnssChanMetadata",
                      "num_metadata_objects": 30 * args.buffer_depth},
        "telescope": {"name": "ICETelescope", "num_polarizations": 1, "num_dishes": 1,
                      "query_gps": False, "require_gps": False},
    }

    # One feed per (node, gpu), in node-major order matching the port assignment. The merge
    # concatenates columns in feed order, so union_ids is BUILT in that order -- feed order
    # and channel_ids order are the same fact stated twice, and the search never assumes
    # sortedness.
    feeds = []
    for node in nodes:
        nchans = covering_channels(node_channels(cfg, node), float(sig["carrier_hz"]),
                                   float(sig["chip_rate_hz"]), float(sig["max_doppler_hz"]))
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
        out.update({
            f"{pre}in_buf": {
                "kotekan_buffer": "standard", "metadata_pool": "gnss_pool",
                "num_frames": args.buffer_depth * 2,
                "frame_size": f"samples_per_data_set * {n_chan} * 1",
            },
            f"{pre}f_buf": {
                "kotekan_buffer": "standard", "metadata_pool": "gnss_pool",
                "num_frames": args.buffer_depth * 2,
                "frame_size": f"samples_per_data_set * {n_chan} * 8",  # cfloat32
            },
            f"{pre}recv": {
                "kotekan_stage": "bufferRecv",
                "buf": f"{pre}in_buf",
                "listen_port": args.search_port_base + i,
                "num_threads": 1,
                "drop_frames": True,
                # Must match the sender exactly -- see the note on the node's srch_send leg.
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
                # Measured on sky 2026-07-30: the F-engine channelized output is CONJUGATED
                # relative to the nominal decode (or equivalently nibble-swapped; the two are
                # indistinguishable and both invisible to the |.|^2-only X-engine). Without
                # this the despread is blind to every satellite while the aggregate GNSS glow
                # sits at +2.4 dB in the very same bins. See GnssChordDequantize.cpp.
                "conjugate": True,
                "cpu_affinity": [cores[(i + 2) % len(cores)]],
            },
        })

    n_union = len(union_ids)
    out.update({
        "agg_merged_buf": {
            "kotekan_buffer": "standard", "metadata_pool": "gnss_pool",
            "num_frames": args.buffer_depth * 2,
            "frame_size": f"samples_per_data_set * {n_union} * 8",
        },
        "agg_merge": {
            "kotekan_stage": "GnssChanAlignMerge",
            "in_bufs": in_buf_names,
            "out_buf": "agg_merged_buf",
            "in_channels": in_channels,
            "cpu_affinity": [cores[3 % len(cores)]],
        },
        # Named gps_search -- the CANONICAL spelling the browser viewer's /wsport chains and
        # the airspy tooling poll (/gps_search/get_detections). Renaming our invented
        # "agg_search" is cheaper and more durable than teaching every client an alias.
        "gps_search": search_stage(cfg, args, "agg_merged_buf", union_ids,
                                   cores[4 % len(cores)]),
    })
    # The union surface is ~16x a single node's (4x channels x 4x stored lags); give the
    # search worker the spare cores and thread the aggregate across them.
    # The union surface grows as (channels x stored lags): 2 nodes -> 27ch/4096, 8 nodes ->
    # 106ch/16384, i.e. ~16x. Give the search worker the spare cores and thread across them;
    # the affinity set must be at least as wide as the thread count or the extra threads
    # contend on the same cores and buy nothing.
    nth = max(1, int(args.acquire_threads))
    out["gps_search"]["acquire_threads"] = nth
    out["gps_search"]["cpu_affinity"] = list(cores[-max(nth, 6):])
    return out, feeds, union_ids


def build_search_instance(cfg, node, per_gpu, args, port):
    """The SEARCH instance: a standalone kotekan that only acquires.

    Separate process, not a branch of the node config, for two reasons. It must not share the
    node instance's fate -- acquisition is a bootstrap aid and the science chain must not stop
    when it is restarted -- and keeping it standalone means moving it to its own machine is a
    change of --search-host, nothing else. It takes no DPDK, no GPU and no hugepages, so it
    also runs as an ordinary user.
    """
    sig = cfg["signals"]
    rt = cfg["runtime"]
    fe = cfg["fengine"]
    cores = rt["cpu_affinity"]
    out = {
        "type": "config",
        "log_level": "info",
        "samples_per_data_set": 8192,
        "sizeof_float32": 4,
        "cpu_affinity": cores,
        "rest_server": {"port": port, "cpu_affinity": cores, "enable_cors": True},
        # CLOCK PROFILE, read by GnssChannelizedSearch from /clock_profile: when present the
        # carrier search extent is DERIVED from the clock's frequency accuracy plus the band's
        # max sky Doppler, instead of a guessed doppler_min/max. gpsdo = 0.06 ppm, which at
        # 1176.45 MHz is ~71 Hz of clock uncertainty against +-5 kHz of geometry -- so the
        # extent is set by the orbit, as it should be. CHORD's F-engine is GNSS-disciplined
        # (arXiv:2607.01625 s2), so this is the honest preset, not an optimistic one.
        "clock_profile": {"name": cfg["clock"]["profile"]},
        "gnss_pool": {"kotekan_metadata_pool": "GnssChanMetadata",
                      "num_metadata_objects": 30 * args.buffer_depth},
        # The search stage builds a Telescope-free replica bank, but kotekan constructs a
        # Telescope regardless, so it needs a minimal valid block.
        "telescope": {"name": "ICETelescope", "num_polarizations": 1, "num_dishes": 1,
                      "query_gps": False, "require_gps": False},
    }
    # Contiguous span covering the signal's whole main lobe on the PFB grid. Derived from the
    # SAME covering_channels() the band plan and the tap use -- deliberately not re-derived from
    # frequencies, because that criterion (centre inside the band) differs from covering's
    # (passband OVERLAP) and drops the edge channels the tap actually delivers. Both GPUs share
    # one span so the two searches are directly comparable.
    full_cover = covering_channels(all_band_channels(cfg), float(sig["carrier_hz"]),
                                   float(sig["chip_rate_hz"]), float(sig["max_doppler_hz"]))
    span_lo, span_hi = full_cover[0], full_cover[-1]
    span_n = span_hi - span_lo + 1

    for gpu, pairs in sorted(per_gpu.items()):
        n_chan = len(pairs)
        pre = f"srch{gpu}_"
        # Global channel index of this subband's first channel: the search reports code phase
        # and Doppler against the ABSOLUTE frequency grid, so it has to know where it sits.
        chan0 = pairs[0][0]
        out.update({
            f"{pre}in_buf": {
                "kotekan_buffer": "standard", "metadata_pool": "gnss_pool",
                "num_frames": args.buffer_depth * 2,
                "frame_size": f"samples_per_data_set * {n_chan} * 1",
            },
            f"{pre}f_buf": {
                "kotekan_buffer": "standard", "metadata_pool": "gnss_pool",
                "num_frames": args.buffer_depth * 2,
                "frame_size": f"samples_per_data_set * {n_chan} * 8",  # cfloat32
            },
            f"{pre}recv": {
                "kotekan_stage": "bufferRecv",
                "buf": f"{pre}in_buf",
                "listen_port": args.search_port_base + gpu,
                "num_threads": 1,
                "drop_frames": True,
                # Must match the sender exactly -- see the note on the node's srch_send leg.
                "use_config_tracker": False,
                "cpu_affinity": [cores[gpu % len(cores)]],
            },
            f"{pre}deq": {
                "kotekan_stage": "GnssChordDequantize",
                "in_buf": f"{pre}in_buf",
                "out_buf": f"{pre}f_buf",
                "n_channels": n_chan,
                "n_elements": 1,
                "element": 0,
                "conjugate": True,  # F-engine conjugation, measured on sky -- see the aggregator note
                # NO zero-fill. channelized_accumulate FFTs along the HOP axis WITHIN each
                # channel and sums the per-channel surfaces -- it is "the distributable half of
                # the search", built for channels scattered across nodes, so a sparse comb is
                # fine by construction. Widening 7 channels into the 106-channel span made the
                # search transform 93% zeros for no gain: 15x the work, and it never finished a
                # pass. (The 640 ns code-phase ambiguity the comb DOES cause is a property of
                # the measurement, not a defect in the algorithm, and the BRDC model resolves it
                # with >=16x margin.)
                "cpu_affinity": [cores[(gpu + 2) % len(cores)]],
            },
            f"{pre}search": search_stage(cfg, args, f"{pre}f_buf",
                                          [fid for fid, _ in pairs],
                                          cores[(gpu + 4) % len(cores)]),
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="production config (JSON from /config, or yaml)")
    ap.add_argument("--node", required=True)
    ap.add_argument("--node-file", default=DEFAULT_NODE_FILE)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rest-port", type=int, default=None,
                    help="default: runtime.rest_port from the node table (NOT 12048)")
    ap.add_argument("--disable-outputs", dest="disable_outputs", action="store_true", default=True)
    ap.add_argument("--enable-outputs", dest="disable_outputs", action="store_false",
                    help="DANGEROUS on a shared node: re-enables the bufferSend legs to the "
                         "downstream N2 consumer")
    ap.add_argument("--no-ingest", action="store_true",
                    help="drop dpdk + the transposes, leaving host_voltage_buffer unfed. The "
                         "GNSS branch still CONSTRUCTS, so --dry-run validates it without "
                         "needing the root privileges DPDK's hugepages require. Not runnable.")
    ap.add_argument("--keep-n2", action="store_true",
                    help="retain the science pipeline alongside the GNSS branch")
    ap.add_argument("--prns", type=int, nargs="*", default=list(range(1, 33)))
    ap.add_argument("--integration-length", type=int, default=100)
    ap.add_argument("--hops-per-record", type=int, default=2048,
                    help="10.49 ms at CHORD's 5.12 us hop; divides the 8192-hop frame 4 ways and "
                         "stays under the 20 ms NH20 period, so a record straddles at most one "
                         "overlay transition (which P_HEAD handles)")
    ap.add_argument("--reference-element", type=int, default=0,
                    help="antenna whose correlation fills the record HEADER -- the broker's DLL "
                         "and carrier loop reference")
    ap.add_argument("--buffer-depth", type=int, default=4)
    ap.add_argument("--search-element", type=int, default=0,
                    help="element (relative to the live range) the acquisition search runs on. "
                         "Single-antenna by design -- pick a healthy, high-gain feed.")
    ap.add_argument("--search-host", default="127.0.0.1",
                    help="where the search instance listens. Localhost today; changing this is "
                         "the ONLY edit needed to move the search to another machine.")
    ap.add_argument("--aggregator-instance", nargs="+", metavar="NODE", default=None,
                    help="emit ONE standalone search instance over the UNION of these nodes'\
 combs (e.g. --aggregator-instance cx19 cx27). Feeds listen on search_port_base + i in\
 node-major, GPU-minor order -- the same ports the node configs already target. Implies\
 everything --search-instance implies (no DPDK/GPU/hugepages, ordinary user).")
    ap.add_argument("--record-dir", default=None,
                    help="override runtime.record_dir from the node file. Needed on any node "
                         "without /data: rawFileWrite takes base_dir from the config, so pointing "
                         "the LOGS elsewhere does not move the RECORDS, and the stage fails on a "
                         "directory that is not there.")
    ap.add_argument("--search-port-base", type=int, default=11040,
                    help="bufferRecv port for GPU 0; GPU 1 uses base+1")
    ap.add_argument("--acquire-threads", type=int, default=6,
                    help="threads for the aggregate half of the acquire (parallel over Doppler "
                         "bins x coarse lags). Aggregator only; per-node instances keep 1.")
    ap.add_argument("--acquire-fine-step", type=int, default=1,
                    help="fine-lag decimation in the acquire surface. The fine axis resolves a "
                         "lobe sph/(comb span in bins) wide -- ~156 samples at CHORD regardless "
                         "of channel COUNT (the span sets it, not the density) -- so storing it "
                         "per sample is ~156x oversampled and the surface is the whole cost of a "
                         "pass. Step must stay well under the lobe width or the peak is missed.")
    ap.add_argument("--prns-per-pass", type=int, default=0,
                    help="how many ELIGIBLE PRNs to search per snapshot, round-robin. 0 = all "
                         "(airspy's behaviour). A detection's ref_hop is the SNAPSHOT's start "
                         "hop, so with one snapshot per pass the last PRN searched carries an "
                         "epoch as old as the pass; seed error is Doppler error x that age at "
                         "0.0087 chips/Hz/s. Bounds the epoch at emit by (pass time)/(this).")
    ap.add_argument("--acquire-windows", type=int, default=32,
                    help="windows stacked per acquisition attempt. CHOOSE BY INTEGRATION TIME, "
                         "not by copying the airspy count: a window is repl_period_hops long, "
                         "which is 1 ms there (1000 hops) but 16 ms here (3125 hops), because "
                         "CHORD's 1 ms code period is not commensurate with its 5.12 us hop. So "
                         "the same COUNT buys 16x the integration and costs 16x more. 32 windows "
                         "= 512 ms, ~5x the airspy node's 100 ms -- deliberate compensation for "
                         "holding only 7 of 106 channels (-11.8 dB).")
    ap.add_argument("--acquire-snr", type=float, default=12.0)
    ap.add_argument("--search-instance", action="store_true",
                    help="emit the SEARCH instance config instead of the node config")
    args = ap.parse_args()

    with open(args.node_file) as fh:
        cfg = yaml.safe_load(fh)
    if args.node not in cfg["nodes"]:
        raise SystemExit(f"unknown node {args.node}")

    with open(args.base) as fh:
        base = json.load(fh) if args.base.endswith(".json") else yaml.safe_load(fh)
    out = copy.deepcopy(base)

    sig = cfg["signals"]
    chans = covering_channels(node_channels(cfg, args.node), float(sig["carrier_hz"]),
                              float(sig["chip_rate_hz"]), float(sig["max_doppler_hz"]))
    if not chans:
        raise SystemExit(f"{args.node} holds no covering channels for {sig['primary']}")

    # Split the covering set by which GPU's comb holds each channel, and convert global freq_id
    # to that comb's buffer index -- the tap indexes into the frame, not the sky.
    per_gpu = {}
    for fid in chans:
        gpu, idx = gpu_of_channel(cfg, args.node, fid)
        per_gpu.setdefault(gpu, []).append((fid, idx))

    if args.aggregator_instance:
        port = args.rest_port if args.rest_port is not None else cfg["runtime"]["rest_port"] + 1
        if port == 12048:
            raise SystemExit("refusing port 12048 (choco owns it)")
        out, feeds, union_ids = build_aggregator_instance(cfg, args.aggregator_instance, args, port)
        text = ("# GENERATED by config/gen_chord_gnss_config.py --aggregator-instance -- DO NOT "
                "HAND-EDIT.\n"
                f"# nodes {' '.join(args.aggregator_instance)}  signal {sig['primary']}  "
                f"rest port {port}\n"
                f"# union comb: {len(union_ids)} channels {min(union_ids)}..{max(union_ids)}\n"
                "# ONE search over the gathered union, aligned on the global sample counter.\n"
                + yaml.safe_dump(out, default_flow_style=False, sort_keys=True))
        if args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            open(args.out, "w").write(text)
            print(f"wrote {args.out}")
        else:
            sys.stdout.write(text)
        print(f"  AGGREGATOR instance over {args.aggregator_instance}", file=sys.stderr)
        for i, (node, gpu, fids) in enumerate(feeds):
            print(f"  feed {i}: {node} gpu{gpu} {len(fids)} ch  <- port "
                  f"{args.search_port_base + i}", file=sys.stderr)
        print(f"  union    {len(union_ids)} channels", file=sys.stderr)
        print(f"  rest     {port}", file=sys.stderr)
        return

    if args.search_instance:
        port = args.rest_port if args.rest_port is not None else cfg["runtime"]["rest_port"] + 1
        out = build_search_instance(cfg, args.node, per_gpu, args, port)
        text = ("# GENERATED by config/gen_chord_gnss_config.py --search-instance -- DO NOT "
                "HAND-EDIT.\n"
                f"# node {args.node}  signal {sig['primary']}  rest port {port}\n"
                "# Standalone acquisition instance: no DPDK, no GPU, no hugepages -- runs as an\n"
                "# ordinary user. Fed by bufferSend from the node instance.\n"
                + yaml.safe_dump(out, default_flow_style=False, sort_keys=True))
        if args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            open(args.out, "w").write(text)
            print(f"wrote {args.out}")
        else:
            sys.stdout.write(text)
        print(f"  SEARCH instance for {args.node}", file=sys.stderr)
        print(f"  listens  {args.search_port_base}..{args.search_port_base + len(per_gpu) - 1}",
              file=sys.stderr)
        print(f"  rest     {port}", file=sys.stderr)
        return

    # --- safety: never answer on production's port -------------------------------------------
    port = args.rest_port if args.rest_port is not None else cfg["runtime"]["rest_port"]
    if port == 12048:
        raise SystemExit("refusing to generate a config on port 12048: choco owns that port and "
                         "would redirect or reconfigure this instance")
    # CORS: the live viewer is served on its own HTTP port and fetches kotekan's REST
    # cross-origin, so without this the browser silently blocks every status poll and the
    # panels just sit empty. Wildcard is fine here -- this instance is bound on a private
    # site network and serves nothing but diagnostics.
    out["rest_server"] = {"port": port, "cpu_affinity": cfg["runtime"]["cpu_affinity"],
                          "enable_cors": True}

    # --- safety: don't inject into the downstream science consumer ----------------------------
    dropped = []
    for key in list(out.keys()):
        if args.disable_outputs and key.startswith("buffer_send"):
            del out[key]
            dropped.append(key)
        elif not args.keep_n2 and key.startswith(N2_STAGE_PREFIXES):
            del out[key]
            dropped.append(key)

    if args.no_ingest:
        for key in list(out.keys()):
            if key == "dpdk" or key.startswith("transpose_voltage"):
                del out[key]
                dropped.append(key)

    # --- metadata pool for the GNSS chain -----------------------------------------------------
    out["gnss_pool"] = {"kotekan_metadata_pool": "GnssChanMetadata",
                        "num_metadata_objects": 30 * args.buffer_depth}

    record_floats = None
    for gpu, pairs in sorted(per_gpu.items()):
        blocks, record_floats, n_elem = build_gnss_branch(
            cfg, args.node, gpu, [i for _, i in pairs], args,
            freq_ids=[f for f, _ in pairs])
        out.update(blocks)

    hdr = [
        "# GENERATED by config/gen_chord_gnss_config.py -- DO NOT HAND-EDIT.",
        f"# node {args.node}  base {os.path.basename(args.base)}  signal {sig['primary']}",
        f"# covering channels {len(chans)}: freq_id {chans[0]}..{chans[-1]}",
        f"# rest port {port} (production owns 12048)",
        f"# outputs disabled: {sorted(dropped)[:4]}{' ...' if len(dropped) > 4 else ''}",
        "# Edit config/chord_gnss_node.yaml and regenerate instead.",
    ]
    text = "\n".join(hdr) + "\n" + yaml.safe_dump(out, default_flow_style=False, sort_keys=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)

    print(f"  node          {args.node}", file=sys.stderr)
    print(f"  covering ch   {len(chans)} -> per GPU " +
          ", ".join(f"gpu{g}:{len(v)}" for g, v in sorted(per_gpu.items())), file=sys.stderr)
    print(f"  record_floats {record_floats} (24 header + n_elem*12)", file=sys.stderr)
    print(f"  rest port     {port}", file=sys.stderr)
    print(f"  dropped       {len(dropped)} production blocks", file=sys.stderr)


if __name__ == "__main__":
    main()
