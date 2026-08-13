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
import hashlib
import json
import os
import shlex
import sys
import textwrap

import yaml

CONF = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CONF)

from chord_band_plan import (all_band_channels, covering_channels,  # noqa: E402
                             node_channels, signal_table)
from gnss_record_layout import record_stride  # noqa: E402

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

# --n2-primary REPLACES run_n2k and nothing else. cudaCorrelatorDual computes the identical N^2
# prefix and ships it on the identical leg, so run_n2k must go or two stages produce one buffer.
#
# WHICH CONSUMERS SURVIVE IS --keep-n2's JOB, not this flag's, and the two compose. That split is
# forced by the captured base rather than chosen: N2Accumulate does not consume correlations
# alone -- each accum_N also takes counts, PL counts, RFI counts and the RFI frame mask -- and
# that chain is HEADLESS here. PL_mask_compactor -> host_pl_mask_buffer -> count_PL ->
# host_pl_counts_buffer -> n2_accumulate, but PL_mask_compactor's own input
# host_pl_mask_exp_buffer has NO PRODUCER ANYWHERE IN THE BASE (verified 2026-08-08); in the real
# deployment it arrives over the network from a stage the capture does not include.
#
# So on a dev node the science consumers CANNOT be fed, whatever this flag does:
#   --n2-primary            -> the prefix is produced and exported to host_correlation_buffer,
#                              with no consumers. This is the configuration for VALIDATING it
#                              against cudaCorrelator, which is what has to happen first anyway.
#   --n2-primary --keep-n2  -> the full chain, for a deployment where the PL/RFI feeds exist.
#
# BUT PREFER --n2-dual --keep-n2 (no --n2-primary) FOR PRODUCTION. n2k_dual's freq_map and
# block_class_mask are BOTH global to a launch, so "AA on every frequency + MIXED|BB on the comb"
# is not expressible as one launch -- full mode computes the extended (N+M)^2 on ALL 384 local
# frequencies, and on 377 of them the M half is entirely zeros. Measured cost of that waste:
# 3.698 ms/frame against 0.076 ms for the mapped launch.
#
# TWO launches express it exactly, and the first one already exists: production's cudaCorrelator
# (bare N^2, NS=128, every frequency) plus cudaCorrelatorDual in FREQ-MAP mode (MIXED|BB, 7
# channels). They do not collide -- the mapped launch never writes the standard output, because
# the AA block it would copy is not computed. Cost is bare N^2 + 0.076 ms instead of 3.698 ms.


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

    Names must be distinct across the chains on one node -- every buffer, stage and REST
    endpoint of the chain is derived from it, and kotekan's per-consumer bookkeeping is
    keyed by stage name (a collision is the `gnss_ring` trap in cudaGnssTrack.cpp:382,
    which cost a whole BDS chain silently degrading on the first tri-constellation night).
    """
    parts = name.split("_")
    body = parts[1] if len(parts) > 1 else parts[0]
    return "_" + body.lower()


def parse_extra_signals(cfg, args, node, primary_chans):
    """--extra-signal NAME:PRNS -> chain dicts, with their covering channels resolved.

    The descriptor comes from signal_table() (parsed out of gnssSignal.hpp) so the carrier
    and chip rate that set the covering channels are the SAME constants the stage builds its
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
            raise SystemExit(f"--extra-signal {name}: unknown signal (not in gnssSignal.hpp)")
        if not prn_s.strip():
            raise SystemExit(
                f"--extra-signal {name}: PRNs are required, e.g. {name}:1,2,3. Constellations "
                "have different PRN ranges and the count IS the path-B lane budget "
                "(4 lanes/PRN of 128), so it is not something to default silently.")
        prns = [int(p) for p in prn_s.split(",") if p.strip()]
        s = table[name]
        tag = signal_tag(name)
        if tag in tags:
            raise SystemExit(f"--extra-signal {name}: tag '{tag}' collides with another chain; "
                             "every buffer and endpoint is named from it")
        tags.add(tag)
        chans = covering_channels(node_channels(cfg, node), s["carrier_hz"], s["chip_rate_hz"],
                                  float(cfg["signals"]["max_doppler_hz"]),
                                  modulation=s["modulation"], boc_m=s["boc_m"], boc_n=s["boc_n"])
        if not chans:
            raise SystemExit(f"--extra-signal {name}: {node} holds no covering channels "
                             f"(carrier {s['carrier_hz']/1e6:.2f} MHz -- outside the science "
                             "band, or this node's comb misses it entirely)")
        chains.append({
            "signal": name, "tracker": name, "prns": prns, "tag": tag,
            "ord": len(chains) + 1,  # 0 is the primary chain; drives the core rotation
            "carrier_hz": s["carrier_hz"], "chip_rate_hz": s["chip_rate_hz"],
            "chans": chans, "shares_tap": chans == primary_chans,
        })

    # -- SIDEBAND PARTNERS MUST CARRY THE SAME PRNs, and this is an identity, not a policy --
    #
    # Galileo E5 is ONE AltBOC(15,10) transmission centred at 1191.795 MHz; E5a (1176.45) and
    # E5b (1207.14) are its lower and upper sidebands, generated coherently by one modulator on
    # one spacecraft. BeiDou B2 is the same shape (ACE-BOC(15,10), same centre). So there is no
    # such thing as a satellite that broadcasts E5a but not E5b: a PRN present on one sideband
    # is present on the other BY CONSTRUCTION.
    #
    # WHY IT IS CHECKED RATHER THAN TRUSTED. E5b shipped with 8 PRNs chosen to "spread across
    # orbital slots so something is always up" -- a sensible rule for an independent chain, and
    # the wrong rule for a sideband. It overlapped E5a's live detections in exactly ONE PRN, and
    # that one read deep 3.2 on E5a (noise), while E5a's real detections (PRN 26 at 255.8, PRN
    # 29 at 72.4) were not seeded on E5b at all. The chain could not have produced a positive
    # result, so "E5b reads noise" was indistinguishable from "E5b is broken" for a full day.
    #
    # It also destroys the measurement the second band exists for: tau_band (per-BAND
    # instrumental delay) is separable from b_sat (per-SATELLITE bias) only by observing ONE ray
    # on TWO carriers. Disjoint seed lists make that unobservable however correct the code is.
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
                % (a["signal"], b["signal"], a["signal"], only_a, b["signal"], only_b))
    return chains


def build_gnss_branch(cfg, node, gpu, chan_idx, args, freq_ids=None, chain=None):
    """The GNSS stages+buffers to inject, for one GPU's covering channels.

    @param chain  None = the PRIMARY chain (prefix `gnss{gpu}_`, byte-identical to what the
                  single-signal generator has always emitted). Otherwise a dict
                  {tracker, search, prns, tag, shares_tap} describing an EXTRA signal chain:
                  its blocks are prefixed `gnss{gpu}{tag}_`, and when `shares_tap` it
                  consumes the primary chain's voltage tap instead of opening a second one
                  (kotekan buffers are per-consumer, so both chains see every frame).
    """
    sig = cfg["signals"]
    rt = cfg["runtime"]
    arr = cfg["array"]
    n_elem = arr["live_elements"][1] - arr["live_elements"][0] + 1
    elem0 = arr["live_elements"][0]
    n_chan = len(chan_idx)
    cores = rt["cpu_affinity"]
    tag = chain["tag"] if chain else ""
    pre = f"gnss{gpu}{tag}_"

    # PER-CHAIN CORE ROTATION. Every affinity below is cores[(gpu + k) % len], a function of
    # the STAGE and the GPU only, so without a rotation every chain stacks its cudaProcess on
    # one core, its combiner on another, and so on. The rotation spreads them: 3 per chain,
    # coprime to the 10-core pool, so N chains walk N distinct cores per stage.
    #
    # HONEST STATUS OF THE JUSTIFICATION: this is hygiene, NOT a known-harmful bug being
    # fixed. A cudaProcess thread mostly blocks on GPU completion rather than burning CPU, so
    # co-scheduling several is probably fine. An earlier collision (gnss1_n2dual on core 60
    # with gnss0_gpu) was fixed on 2026-08-06 and INVESTIGATED OUT as the cause of the
    # degradation it was found near -- the real cause was elsewhere. Do not cite it as
    # evidence that stacking hurts; it is evidence that it can look guilty. The rotation costs
    # nothing and makes the assignment deterministic, which is the whole claim.
    #
    # It does NOT create free cores -- a multi-chain node oversubscribes a 10-core pool, and
    # main() prints the resulting load so it is a stated cost rather than a surprise.
    rot = 3 * (chain["ord"] if chain else 0)

    def core(k):
        return cores[(k + rot) % len(cores)]

    # The tap output is SIGNAL-INDEPENDENT -- [hop][chan][elem] 4+4b voltages, no replica
    # anywhere in it -- so a chain on the same covering channels reads the primary's tap
    # rather than opening a second one. That is the whole reason E5a/B2a cost no extra
    # ingest, PCIe or CPU: same carrier, same channels, same bytes.
    shares_tap = bool(chain and chain.get("shares_tap"))
    tap_out = f"gnss{gpu}_volt_buf" if shares_tap else f"{pre}volt_buf"
    rec_buf = f"{pre}rec_buf"
    cmb_buf = f"{pre}cmb_buf"

    # Which signal this chain tracks, which PRNs, and at WHAT SKY CARRIER. The carrier is
    # the replica's f_offset and is the single easiest thing to get wrong here (a wrong one
    # builds every replica in the wrong part of the band and every correlation is noise --
    # see the f_offset note below), so it comes from the chain, never from the primary.
    track_signal = chain["tracker"] if chain else sig.get("tracker", sig["primary"])
    prns = chain["prns"] if chain else args.prns
    carrier_hz = chain["carrier_hz"] if chain else float(sig["carrier_hz"])

    # Record frames carry the element axis: RECORD_FLOATS + n_elem*ELEM_FLOATS per PRN.
    # READ FROM gnssRecord.hpp, not restated here: this used to be the literal `26 + n_elem*12`
    # under a comment claiming it was "kept in one place", which is exactly the drift that broke
    # the airspy configs when RECORD_FLOATS went 24 -> 26 (34 stages FATAL at construction).
    record_floats = record_stride(n_elem)
    n_prn = len(prns)

    blocks = {}
    if not shares_tap:
        blocks[tap_out] = {
            "kotekan_buffer": "standard",
            "metadata_pool": "gnss_pool",
            "num_frames": args.buffer_depth,
            # [hop][chan][elem], one byte per complex sample
            "frame_size": f"samples_per_data_set * {n_chan} * {n_elem}",
            # peek_hold: keep the newest full frame until the next lands, so
            # /buffer/<name>/frame can dump the ACTUAL 4+4b bytes the tracker despreads.
            # Costs one frame slot. Added 2026-08-06 while chasing a node that despread
            # noise for every PRN with correct records and correct replicas -- the one
            # thing there was no way to inspect was the data itself.
            "peek_hold": True,
        }

    blocks.update({
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
    })
    if not shares_tap:
        blocks[f"{pre}tap"] = {
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
            "cpu_affinity": [core(gpu)],
        }

    blocks.update({
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
            "cpu_affinity": [core(gpu + 6)],
            "in_buffers": {"gnss_volt_in": tap_out},
            "out_buffers": {"gnss_epl_out": f"{pre}epl_buf"},
            "commands": [
                {"name": "cudaInputData", "in_buf": "gnss_volt_in",
                 "gpu_mem": f"{pre}voltage"},
                {"name": "cudaGnssChordTrack",
                 # TASK #52 A/B ARM -- ⚠️ TEMPORARY, remove with task #55. Emitted on BOTH
                 # producers because cudaGnssChordTrackState (which owns the despread, and
                 # therefore the arm) is constructed with the COMMAND's unique_name -- so path A
                 # and path B each need their own copy or one silently keeps the default.
                 "carrier_phase_from_ref": (gpu == 0 if args.carrier_phase_from_ref == "ab"
                                            else args.carrier_phase_from_ref == "1"),
                 # F-engine conjugation, measured on sky 2026-07-30 (see GnssChordDequantize).
                 "conjugate": True,
                 "gpu_mem_input": f"{pre}voltage",
                 "gpu_mem_output": f"{pre}epl",
                 # OVERLAY-BAKED tracker code (2026-07-31): multi-period records despread the
                 # bare primary to ~zero (the NH20/CS100 partial sums cancel by design); see
                 # chord_gnss_node.yaml and docs/CHORD_MULTIBAND.md section 4.
                 "signal": track_signal,
                 # GLOBAL bins of this GPU's covering comb, in the tap's local order. The
                 # replica for a channel must be built at ITS OWN sky frequency, and CHORD's
                 # comb is stride-16, so a contiguous chan_offset cannot describe it -- passing
                 # 0 put every replica at DC and nothing ever locked (2026-07-31).
                 "channel_ids": list(freq_ids if freq_ids is not None else chan_idx),
                 "prns": prns,
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
                 "f_offset_hz": float(carrier_hz),
                 "hops_per_record": args.hops_per_record,
                 "fft_length": cfg["fengine"]["fft_length"],
                 "sample_rate": float(cfg["fengine"]["sampling_rate_MHz"]) * 1e6,
                 "seed_endpoint": f"/{pre}track/set_seeds",
                 # In-tracker DLL code trim (ported 2026-07-31): per-frame closure is the only
                 # loop fast enough for the clock chain's +-1 chip / ~20 s breathing. The
                 # broker's own DLL (3c) sees disc ~ 0 once this holds and stays quiet.
                 "code_trim": True,
                 # LOCAL trim gain, 0 by default -- the fleet DLL owns the code loop.
                 #
                 # E, P and L are measured relative to the code phase THIS instance despread at.
                 # With every instance running its own trim they drift to different delays, and
                 # summing E/L taken at different delays SMEARS the discriminator instead of
                 # sharpening it -- the broker's 11.8 dB of combined bandwidth would be thrown
                 # away silently, with nothing in the logs to say so. So there is exactly one
                 # code loop, in the broker (--dll-combiners), and this one is off.
                 #
                 # Set --local-trim-gain 0.15 to restore the in-tracker loop (the pre-2026-08-03
                 # behaviour) for a single-node bench, or as the control in a before/after.
                 "trim_gain": args.local_trim_gain,
                 "trim_endpoint": f"/{pre}track/get_trim",
                 # NO reseed_hold_* HERE, AND NO f_ref FENCE AT ALL. This stage is
                 # cudaGnssChordTrack, NOT cudaGnssTrack: it has no f_ref, no fll_reacq_hz and
                 # no re-anchor logic. It uses the seed Doppler DIRECTLY as the replica carrier,
                 # refreshed every window (`ss.doppler_hz = sd.doppler_hz`), with the NCO
                 # carrying only the trim (`c.f_nco = sd.ctrim_hz`). So on CHORD the SEED IS THE
                 # REFERENCE, and a jump in the seeded Doppler is a reference jump directly --
                 # there is no fence to hold it and nothing to tune. That is why the fix that
                 # worked was --seed-doppler auto (smooth model+bias seed) and why the
                 # cudaGnssTrack lock-hold, which the config used to set here, did NOTHING:
                 # those keys were being written into a stage that ignores them (2026-08-04).
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
            "prns": prns,
            "n_elements": n_elem,
            # PER-CHANNEL SPECTRUM EXPORT (task #32, docs/CHORD_JOINT_TRACKING.md P1): the
            # same GLOBAL-bin list the despread command runs, so /get_spectrum can label each
            # channel with its sky frequency. A delay is a phase ramp across frequency; this
            # stage's cross-channel sum is the one combine the broker can never undo, and
            # this key is what lets it see the spectrum BEFORE the sum -- as sufficient
            # statistics (per-PRN-per-channel sums), never per-element streams.
            "channel_ids": list(freq_ids if freq_ids is not None else chan_idx),
            "reference_element": args.reference_element,
            # SELF-CALIBRATED ELEMENT SUM (gnssElemCal.hpp, STATE 8.21.5). The header
            # correlation slots carry the calibrated weighted mean over all elements instead of
            # the bare reference element: same reference-anchored phase convention, same "one
            # element" scale, per-record SNR up ~sqrt(N_healthy) ~ 5x -- the array gain the
            # DLL/carrier loops and the combiner's phase tracker all inherit for free. The cal
            # is bootstrap MRC from the satellite itself; until warm (~3 tau) the output is
            # byte-identical to reference-element-only.
            "elem_sum": args.elem_sum,
            # PER-CHANNEL PROMPT DUMP (--chan-dump-prn). Emitted ONLY when enabled: writing the
            # keys unconditionally changed every production node config by three lines for a
            # feature that was off, which is exactly the drift that makes "is the deployed
            # config current?" unanswerable. The cross-channel sum inside this stage is the one
            # combine step the broker can never undo, so whether it is lossless is a question
            # only the per-channel phases can answer -- and they are what the sum hides.
            **({"chan_dump_prn": args.chan_dump_prn,
                "chan_dump_decim": args.chan_dump_decim,
            # ONE FILE PER CHAIN. Both GPUs' assemblers default to the same path, and they
            # interleave: 1.2% of lines came out torn, and worse, BOTH chains label their
            # channels 0..6 locally, so a shared file cannot be demultiplexed at all -- grouping
            # by utc silently mixes two different combs. Found the hard way 2026-08-07.
                "chan_dump_path": f"/tmp/gnss_chan_phase_{node}_{gpu}a{tag}.txt"}
               if args.chan_dump_prn >= 0 else {}),
            "sample_rate": float(cfg["fengine"]["sampling_rate_MHz"]) * 1e6,
            "cpu_affinity": [core(gpu + 8)],
        },
    })

    # SEARCH FEED -- PRIMARY CHAIN ONLY. Extra chains are dead-reckon seeded: the signals we
    # add first (E5a/B2a) carry PER-PRN secondaries, which the replica bank's single-sequence
    # overlay slot cannot hold, so `nh_search` on them is a silent no-op and blind multi-period
    # acquisition does not exist (docs/CHORD_MULTIBAND.md section 5). Emitting a search feed
    # for them would ship bytes to an acquire that cannot use them. The primary chain's search
    # measures the instrumental delay and the clock, which is what the extras are reckoned from.
    if not chain:
        blocks.update({
        # A SECOND tap on the same voltage buffer, taking ONE element -- the
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
            "cpu_affinity": [core(gpu + 1)],
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
            "cpu_affinity": [core(gpu + 3)],
        },
        })

    blocks.update({
        f"{pre}combine": {
            "kotekan_stage": "GnssCoherentCombiner",
            # --combine-gpus: GPU 0's combiner takes BOTH streams (the other GPU's rec_buf name
            # is deterministic, so no cross-branch plumbing is needed) and GPU 1's is dropped
            # below. Order is GPU-major so the subband layout is stable across regenerations.
            "in_bufs": ([rec_buf, f"gnss{1 - gpu}{tag}_rec_buf"] if (args.combine_gpus and gpu == 0)
                        else [rec_buf]),
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
            # PHASE-RATE SEARCH before that sum (2026-08-04). Measured on sky: the per-record
            # prompts carry a linear phase ramp beyond the record-rate Nyquist, so the straight
            # sum recovered 2-5% of the power while the incoherent moments said +10..15 dB per
            # record -- every ladder rung read the Rayleigh value. Searching the rate and
            # derotating recovers 65-80%. Gated on peak/median of the search's own spectrum
            # (17.9-22.0 on signal, 2.8-6.1 on noise), not a closed-form floor, which mispredicts
            # it badly because oversampled bins are not independent.
            "deep_rate_search": True,
            "deep_rate_min_q": 10.0,
            # COMMON-PHASE TRACKER before the deep sum (STATE 8.21.5): the batch form of the
            # carrier loop the airspy closed at 1 kHz. Removes the per-satellite ~0.9 rad
            # propagation wander that capped every deep at ~11-14 sigma; leave-one-out per
            # record => fail-closed on noise, and the half-width candidates compete with the
            # straight sum under the same estimator (the floor pays the selection). Exports
            # coh_frac -- the chopping-independent coherence measure -- beside deep_snr.
            "phase_track": args.phase_track,
            "sky_deep": args.sky_deep,
            # Hops per record frame: turns the frame metadata's sample_seq into the absolute HOP
            # index, which is what the seed (ref_hop), the replica generators and the search all
            # speak. Published as pow_hop so the broker can group EVERY node's E/L powers by an
            # exact integer match and close ONE code loop at full L5 bandwidth -- a single
            # instance sees 6.7% of the lobe (docs/CHORD_GNSS_SHARED_DLL.md).
            "fft_len": cfg["fengine"]["fft_length"],
            # PER-RECORD EXPORT for the fleet's coherent combine (/get_records). One instance
            # sees 7 of 106 channels -- 8.8 dB below the fleet -- and per-record SNR is what caps
            # the coherent span. Summing the per-record complex prompts across instances recovers
            # it; the window means cannot, the phase noise has already eaten those. Sized to the
            # deep window (integration_length) so a consumer always has the same records the
            # local ladder ran on, with a little margin for poll jitter.
            "record_export": args.integration_length + 28,
            # PER-RECORD PHASE DUMP, off unless asked for. /get_records gives the complex prompt
            # but NOT the commanded phase increment or the per-record code phase, and those are
            # what an investigation into the 0.7 rad deep-fold floor (STATE 8.20.9-8.20.15) needs:
            # the floor is phase-only, per-satellite and shared across nodes, and every mechanism
            # guessed at from the harness so far has been wrong, because the harness synthesizes
            # the sky with the same model it despreads against. This dumps the live per-record
            # trajectory instead of inferring it. One line per record per listed PRN -- keep the
            # list short.
            "phase_dump_prns": args.phase_dump_prns,
            "phase_dump_path": f"/tmp/gnss_phase_dump_{node}_{gpu}{tag}.txt",
            "cpu_affinity": [core(gpu + 2)],
        },
        f"{pre}record": {
            "kotekan_stage": "rawFileWrite",
            "in_buf": cmb_buf,
            "base_dir": args.record_dir or rt["record_dir"],
            "file_name": f"{node}_gnss{gpu}{tag}_cmb",
            "file_ext": "raw",
            "cpu_affinity": [core(gpu + 4)],
        },
    })
    # -- --no-path-a: drop the path-A TRACKER, keep the search leg (task #26) -------------
    # Path A costs 17.712 ms/frame against path B's 2.512 (the numbers in the caller's note),
    # so on a node running both it is ~7/10 of the GNSS budget -- and once the broker seeds
    # gnss*_inject instead of gnss*_track, every one of those milliseconds buys nothing.
    #
    # ⚠️ WHAT MUST SURVIVE, AND WHY IT IS NOT OBVIOUS: `srch_tap`/`srch_send` are built in
    # THIS function but are NOT part of path A. They tap `host_voltage_buffer_{gpu}`
    # directly and ship to the aggregator's search, which is what detects GPS, which is what
    # solves the receiver clock that every other chain consumes in-process. Deleting this
    # whole branch would therefore take down all five chains, not one. Path B is unaffected
    # either way: it reads the voltage RINGBUFFER, never path A's tap.
    #
    # Kept available deliberately (KV): path A is the reference every path-B number in
    # section 11 was judged against, so single-signal debugging still wants it. This is a
    # flag, not a deletion.
    if getattr(args, "no_path_a", False) and not chain:
        for _k in (tap_out, f"{pre}tap", f"{pre}epl_buf", rec_buf,
                   f"{pre}gpu", f"{pre}assemble", f"{pre}combine", f"{pre}record"):
            blocks.pop(_k, None)

    return blocks, record_floats, n_elem


def build_n2dual_branch(cfg, node, gpu, chan_idx, freq_ids, args, spds, chain=None):
    """GNSS path B (--n2-dual): one GPU's cudaGnssInject + cudaCorrelatorDual process.

    ONE COMPLETE CHAIN PER SIGNAL, exactly as build_gnss_branch does for path A. `chain` is
    None for the primary and a parse_extra_signals() entry otherwise; every buffer, GPU array
    and endpoint is tagged from it.

    WHY A FULL PARALLEL CHAIN RATHER THAN SHARED LANES. E5a sits on GPS L5's carrier, so the
    two signals cover the SAME channels, and one correlator could in principle serve both if
    the injectors wrote disjoint lane ranges of one synth ring. That needs a lane offset in
    cudaGnssInject and a merged control block -- two injectors cannot both author the FrameHdr
    and PrnCtl the assembler reads. A full chain per signal needs no CUDA change at all,
    because every GPU array name is already config-settable, and it costs one extra mapped
    correlator launch: 0.077 ms/frame measured. Against path A's 17.712 ms per extra signal
    that is not a tradeoff worth agonising over, and the shared-lane version stays available
    later if the lane budget gets tight.

    THE GPU ARRAY NAMES MUST BE TAGGED. get_gpu_memory_array is scoped per DEVICE, so the two
    GPUs' chains never collided on the untagged defaults ("gnss_synth", "gnss_n2ctl",
    "gnss_tiles") -- but two SIGNALS on one GPU would, and the failure mode is the one that
    cost a restart on 2026-08-08: allocation fails, the unit comes up active, and no data
    moves.

    Consumes the GPU voltage ring that run_send_voltage feeds (which --n2-dual keeps).
    The N^2 prefix of the extended triangle stays on-GPU in this mode (no science consumer
    in dev); the gathered GNSS tiles come to the host and go to a sink (dropAllFrames, or
    rawFileWrite with --n2-dump for short captures).
    """
    sig = cfg["signals"]
    rt = cfg["runtime"]
    arr = cfg["array"]
    cores = rt["cpu_affinity"]
    tag = chain["tag"] if chain else ""
    pre = f"gnss{gpu}{tag}_"
    n_chan = len(chan_idx)
    prns = chain["prns"] if chain else args.prns
    n_prn = len(prns)
    # The replica's sky carrier. From the CHAIN, never the primary -- a wrong f_offset builds
    # every replica in the wrong part of the band and every mixed tile reads noise.
    carrier_hz = float(chain["carrier_hz"]) if chain else float(sig["carrier_hz"])
    track_signal = chain["tracker"] if chain else sig.get("tracker", sig["primary"])
    ordv = chain["ord"] if chain else 0
    n_live = arr["live_elements"][1] - arr["live_elements"][0] + 1

    # Tile count per channel: mixed (synth rows x live-antenna col tiles) + the synthetic
    # triangle. MUST match cudaCorrelatorDual's build_tile_selection (and the consumer's
    # recomputation): num_elements 128, num_synth 128.
    num_synth = 128
    # frames are samples_per_data_set hops; the tracker splits them into records. Taken from
    # the BASE config (production owns it), passed in rather than guessed.
    n_rec_per_frame = max(1, int(spds) // args.hops_per_record)
    na16, nt16 = 128 // 16, (128 + num_synth) // 16
    mixed = (nt16 - na16) * ((n_live + 15) // 16)
    bb = (nt16 - na16) * (nt16 - na16 + 1) // 2
    # nt_outer = records per frame now that the dual correlator integrates one record
    tiles_frame_bytes = n_rec_per_frame * n_chan * (mixed + bb) * 512 * 4

    if 4 * n_prn > num_synth:
        raise SystemExit(f"--n2-dual{tag}: 4*{n_prn} PRNs exceeds {num_synth} synthetic "
                         f"lanes (4 lanes per PRN slot). The lane budget is PER CHAIN here "
                         f"(each signal has its own synth ring), so this one signal is too "
                         f"wide; across signals the binding limit is GPU memory, 1.61 GB per "
                         f"ring. Shrink this chain's PRN list, or raise NSB to 256.")

    ring = "host_voltage_ringbuffer" + ("" if gpu == 0 else f"_{gpu}")
    tiles_buf = f"{pre}n2tiles_buf"
    ctl_buf = f"{pre}n2ctl_buf"
    epl_buf = f"{pre}n2epl_buf"
    n2rec_buf = f"{pre}n2rec_buf"
    # epl frame: gnss_gpu::frame_bytes(n_prn, n_chan, ROWS_PLAIN=4, n_elem) -- the SAME
    # expression the tracker's epl_buf uses, since the consumer emits that exact layout.
    epl_bytes = (48 + 8 * 16 + 64 * 16 * n_prn
                 + 16 * 4 * n_prn * 16 * n_chan * n_live
                 + 8 * 4 * n_prn * 16 * n_chan)
    # control block = the same minus the corr array.
    ctl_bytes = (48 + 8 * 16 + 64 * 16 * n_prn + 8 * 4 * n_prn * 16 * n_chan)
    record_floats = 26 + n_live * 12

    blocks = {
        tiles_buf: {
            "kotekan_buffer": "standard",
            "metadata_pool": "gnss_pool",
            "num_frames": args.buffer_depth,
            "frame_size": tiles_frame_bytes,
        },
        f"{pre}n2dual": {
            "kotekan_stage": "cudaProcess",
            "gpu_id": gpu,
            # NEVER share a core with a TRACKER cudaProcess. The tracker uses
            # cores[(gpu+6)%n] (= 60, 61 on the deployed pool), and the naive
            # cores[(gpu+5)%n] put gnss1_n2dual on 60 -- the same core as gnss0_gpu, i.e. two
            # heavy GPU-submission loops (one of them driving a 3.1 ms N^2 plus synthesis) on
            # one core. The light stages in this config do share cores, and always have; two
            # cudaProcess stages must not. Explicit, not arithmetic, so it stays checkable.
            "cpu_affinity": [cores[(5 - 2 * gpu + 3 * ordv) % len(cores)]],
            "in_buffers": {"host_voltage_ringbuffer": ring},
            # --n2-primary: the N^2 PREFIX leaves the GPU on the standard leg, so
            # host_correlation_buffer{,_1} -> N2Accumulate -> the science pipeline is fed by
            # cudaCorrelatorDual instead of by the (dropped) run_n2k. The prefix is bit-identical
            # to cudaCorrelator's by construction -- triangular tile packing orders by ihi, so
            # tiles 0..35 of the extended triangle ARE the pure-128 output (n2dualtest gate [3]).
            "out_buffers": ({"host_gnss_tiles": tiles_buf, "host_gnss_n2ctl": ctl_buf,
                             "host_correlation": "host_correlation_buffer"
                                                 + ("" if gpu == 0 else "_1")}
                            if args.n2_primary else
                            {"host_gnss_tiles": tiles_buf, "host_gnss_n2ctl": ctl_buf}),
            "commands": [
                # ORDER MATTERS: the injector's pack lands on the same stream before the
                # correlator's kernel reads the synth array -- that ordering is the entire
                # synchronization story (no ring semantics on gnss_synth).
                {"name": "cudaGnssInject",
                 # TASK #52 A/B ARM -- ⚠️ TEMPORARY, remove with task #55. Emitted on BOTH
                 # producers because cudaGnssChordTrackState (which owns the despread, and
                 # therefore the arm) is constructed with the COMMAND's unique_name -- so path A
                 # and path B each need their own copy or one silently keeps the default.
                 "carrier_phase_from_ref": (gpu == 0 if args.carrier_phase_from_ref == "ab"
                                            else args.carrier_phase_from_ref == "1"),
                 "voltage_name": "voltage",
                 # MUST match the tracker's `conjugate`. The N^2 kernel has no conj_data flag
                 # and its antenna input is production's, so the F-engine conjugation is
                 # absorbed by conjugating the REPLICA in the pack. Without it every mixed
                 # tile reads noise -- measured on sky 2026-08-06, and it is the same
                 # conjugation lesson as 2026-07-30.
                 "conjugate": True,
                 "num_synth": num_synth,
                 # LOCAL frame indices of the comb (the tap's chan_ids) -- never freq_ids.
                 "gnss_local_channels": chan_idx,
                 # cudaGnssInjectState (= the tracker's state class): same seed contract,
                 # same propagation, own endpoints. The broker POSTs the same payload here.
                 "prns": prns,
                 "n_channels": n_chan,
                 "n_elements": 1,  # unused by the injector; the state class requires it
                 "channel_ids": freq_ids,  # GLOBAL bins, local order (replica sky freq)
                 "signal": track_signal,
                 "f_offset_hz": carrier_hz,
                 "hops_per_record": args.hops_per_record,
                 "fft_length": cfg["fengine"]["fft_length"],
                 "sample_rate": float(cfg["fengine"]["sampling_rate_MHz"]) * 1e6,
                 # Per-chain GPU arrays. Untagged they are one namespace per DEVICE, so a
                 # second signal on the same GPU silently fights the first for the allocation.
                 "gnss_synth_name": f"{pre}synth",
                 "gnss_ctl_name": f"{pre}n2ctl",
                 "seed_endpoint": f"/{pre}inject/set_seeds",
                 "trim_endpoint": f"/{pre}inject/get_trim",
                 # THE SAME F-ENGINE ANCHOR THE TRACKER GETS, AND NOT OPTIONAL. This one
                 # missing key was the whole of the 11.14.1 coh_frac defect. Without it
                 # cudaGnssInject writes utc0 = 0, GnssGpuRecordAssemble falls back to HOST
                 # WALL CLOCK, and every path-B record is stamped at the instant it was
                 # assembled -- so the four sub-records of a frame land microseconds apart
                 # instead of 10.49 ms apart. The combiner's rate search works in UTC, takes
                 # dt = the MINIMUM consecutive spacing, and therefore built its integer grid
                 # on those microseconds: the records were scrambled across the transform and
                 # q fell from ~22 to 4.5-11.4, straight through the q >= 10 gate. Measured
                 # 2026-08-07: utc - hop*5.12us was constant to 0.0 s on path A and wandered
                 # 45 ms (4.3 record periods) on path B.
                 #
                 # NB the amplitude estimators never noticed -- amp_snr was 71-83% of path A
                 # throughout -- because they are per-record and use no time base at all. Only
                 # the cross-record estimators (rate search, coherence_s, carrier fit) read
                 # UTC, which is exactly the set that failed.
                 "frame0_utc": float(cfg["fengine"].get("frame0_utc", 0.0))},
                {"name": "cudaCorrelatorDual",
                 # ONE VISIBILITY PER RECORD, not per frame. Measured 2026-08-06: with the
                 # frame-length window path B's amplitude was 0.55 of path A's, because the
                 # per-record phase ramp (deep_rate_hz, up to 40 Hz) sits INSIDE the coherent
                 # sum where the combiner's rate search cannot reach it -- the one satellite
                 # with zero rate scored 0.75 and coh 0.88. Matching the tracker's record also
                 # puts path B on the fleet DLL's exact pow_hop grid. NB at full CHORD (N=1024)
                 # the vis matrix cannot be dumped this often and the window must lengthen
                 # again; the fix there is to fold the rate into the SYNTHESIZED replica's
                 # carrier, which the injector is already positioned to do.
                 "sub_integration_ntime": args.hops_per_record,
                 "voltage_name": "voltage",
                 "rfi_RFImask_name": "rfi_RFImask",  # unused with rfi_all_pass, key required
                 "gnss_synth_name": f"{pre}synth",
                 # Production runs rfi_first_stage_excision_enabled: false -- its correlator
                 # sees an all-good mask too, so this is behavior-identical, not approximate.
                 "rfi_all_pass": True,
                 # FREQ MAP: compute the mixed + synthetic blocks over ONLY the comb (7 of 384).
                 # Measured 2.90x -> 1.21x stock N^2 (scripts/gnss/n2timing), bitwise-verified
                 # against the full launch (n2dualtest gate [6]), and 0.076 ms/frame in situ.
                 #
                 # THE AA BLOCK IS NOT COMPUTED IN THIS MODE, so the standard N^2 pass-through
                 # is unavailable (cudaCorrelatorDual.cpp: "SPLIT 1 ... SKIPPED in freq-map
                 # mode"). That is fine for a dev config that drops run_n2k, and it is exactly
                 # what blocks path B from BEING the science pipeline -- there is no N^2 to
                 # hand downstream. --n2-full-freq turns it off: the full triangle over every
                 # frequency, AA included, so the bit-identical prefix copy can feed
                 # cudaOutputData -> host_correlation_buffer -> N2Accumulate.
                 #
                 # Cost is the whole question. n2timing measured the marginal OFFLINE at
                 # +2.00 ms/frame full versus +0.22 ms with the map; in situ the mapped launch
                 # came in at 0.076 ms, 3x better than its own projection, so the full number
                 # needs measuring rather than extrapolating. vis_len scales with
                 # dp.n_freq_out automatically (7 -> 384, ~428 MB per frame slot), so no buffer
                 # config changes with the mode.
                 "gnss_freq_map": not (args.n2_full_freq or args.n2_primary),
                 # GPU MEMORY NAME, and it must not collide with production's. cudaCorrelator
                 # and cudaCorrelatorDual both allocate an array called <name>_buffer, sized
                 # from THEIR OWN station count: 28,311,552 B at NS=128 against 113,246,208 B at
                 # NS=256. Coexisting under one name is a hard startup failure --
                 # "get_gpu_memory_array failed: requested name correlation_buffer size 28311552
                 # ... but existing memory is size 113246208" -- and the node comes up with the
                 # units active and NO data moving, which reads exactly like a stall.
                 #
                 # Only --n2-primary wants production's name, because there run_n2k is dropped
                 # and the standard cudaOutputData leg has to find this array. Coexisting, the
                 # dual's copy is never even written (the N^2 pass-through is skipped in
                 # freq-map mode), so a private name costs nothing.
                 "n2k_correlation_name": ("correlation" if args.n2_primary
                                          else f"{pre}n2corr"),
                 "gnss_tiles_name": f"{pre}tiles",
                 "num_synth": num_synth,
                 "num_live_elements": n_live,
                 "gnss_local_channels": chan_idx},
                {"name": "cudaSyncOutput"},
                {"name": "cudaOutputData",
                 "gpu_mem": f"{pre}tiles_buffer",
                 "out_buf": "host_gnss_tiles"},
                # M5: the injector's control block (FrameHdr + winstart + PrnCtl + the true
                # replica energies) -- the epl layout minus corr, which the consumer fills
                # from the tiles. See docs/gnss_gpu_search.md 11.11.
                {"name": "cudaOutputData",
                 # NO _buffer SUFFIX. cudaGnssInject allocates this with
                 # get_gpu_memory_array(_mem_ctl) directly, so the gpu_mem name IS the array
                 # name -- unlike the tiles and correlation arrays, which go through
                 # NDArrayBuffer and are addressed as <name>_buffer. The two conventions sit
                 # three lines apart in this block and do not match.
                 "gpu_mem": f"{pre}n2ctl",
                 "out_buf": "host_gnss_n2ctl"},
            ] + ([
                # Byte-for-byte the third command of production's run_n2k gpu_N block, so the
                # science consumer sees the identical buffer contract it does today.
                {"name": "cudaOutputData",
                 "gpu_mem": "correlation_buffer",
                 "out_buf": "host_correlation"},
            ] if args.n2_primary else []),
        },
        ctl_buf: {
            "kotekan_buffer": "standard",
            "metadata_pool": "gnss_pool",
            "num_frames": args.buffer_depth,
            "frame_size": ctl_bytes,
        },
        epl_buf: {
            "kotekan_buffer": "standard",
            "metadata_pool": "gnss_pool",
            "num_frames": args.buffer_depth,
            "frame_size": epl_bytes,
        },
        n2rec_buf: {
            "kotekan_buffer": "standard",
            "metadata_pool": "gnss_pool",
            "num_frames": args.buffer_depth,
            "frame_size": f"{n_prn} * {record_floats} * sizeof_float32",
            # dump a path-B record frame over REST without a rawFileWrite (see the tap buffers)
            "peek_hold": True,
        },
        # M5 CONSUMER: tiles + control -> the epl layout the SHIPPED assembler eats.
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
            "hops_per_record": args.hops_per_record,
            "cpu_affinity": [cores[(gpu + 9 + 3 * ordv) % len(cores)]],
        },
        # ... and then the EXISTING record assembler, unchanged.
        f"{pre}n2assemble": {
            "kotekan_stage": "GnssGpuRecordAssemble",
            "in_buf": epl_buf,
            "out_buf": n2rec_buf,
            "prns": prns,
            "n_elements": n_live,
            # Task #32: sky-frequency labels for /get_spectrum -- see the path-A assemble
            # block's comment; same list the inject command despreads.
            "channel_ids": freq_ids,
            # TASK #53: ADDRESSABLE, HOP-QUANTISED SPECTRUM WINDOWS. The window index is
            # floor(wstart / spectrum_window_samples) on the F-engine sample clock, so every
            # instance assigns a record to the same window without talking to any other.
            #
            # ⚠️ THIS VALUE MUST BE IDENTICAL ON EVERY INSTANCE. It is computed here, from the
            # SAME record geometry every stage in this file already uses, precisely so it
            # cannot be hand-set per node and drift -- a mismatch would leave the instances
            # unaligned again while every reply still looked well-formed. Derived, not chosen:
            # records_per_window x hops_per_record x fft_length, so window boundaries land on
            # record boundaries and each window holds a constant record count.
            "spectrum_window_samples": (args.spectrum_window_records
                                        * args.hops_per_record
                                        * int(cfg["fengine"]["fft_length"])),
            "spectrum_ring_depth": args.spectrum_ring_depth,
            "reference_element": args.reference_element,
            "elem_sum": args.elem_sum,
            # PER-CHANNEL PROMPT DUMP (--chan-dump-prn). Emitted ONLY when enabled: writing the
            # keys unconditionally changed every production node config by three lines for a
            # feature that was off, which is exactly the drift that makes "is the deployed
            # config current?" unanswerable. The cross-channel sum inside this stage is the one
            # combine step the broker can never undo, so whether it is lossless is a question
            # only the per-channel phases can answer -- and they are what the sum hides.
            **({"chan_dump_prn": args.chan_dump_prn,
                "chan_dump_decim": args.chan_dump_decim,
            # ONE FILE PER CHAIN. Both GPUs' assemblers default to the same path, and they
            # interleave: 1.2% of lines came out torn, and worse, BOTH chains label their
            # channels 0..6 locally, so a shared file cannot be demultiplexed at all -- grouping
            # by utc silently mixes two different combs. Found the hard way 2026-08-07.
                "chan_dump_path": f"/tmp/gnss_chan_phase_{node}_{gpu}b.txt"}
               if args.chan_dump_prn >= 0 else {}),
            "sample_rate": float(cfg["fengine"]["sampling_rate_MHz"]) * 1e6,
            "cpu_affinity": [cores[(gpu + 2) % len(cores)]],
        },
        # PATH B'S OWN COMBINER. Gives the path-B record stream a REST endpoint
        # (/gnssN_n2combine/get_status, /get_records) so path A and path B can be compared
        # LIVE on the same node, same sky, same PRNs -- which is the M5 validation and stays
        # useful afterwards. integration_length is scaled by n_rec because path B emits ONE
        # record per frame where the tracker emits n_rec, so this keeps the same WALL-CLOCK
        # deep window rather than the same record count.
        f"{pre}n2cmb_buf": {
            "kotekan_buffer": "standard",
            "metadata_pool": "gnss_pool",
            "num_frames": args.buffer_depth,
            "frame_size": f"{n_prn} * {record_floats} * sizeof_float32",
        },
        f"{pre}n2combine": {
            "kotekan_stage": "GnssCoherentCombiner",
            # ONE GPU PER PATH-B COMBINER -- and NOT because the merge does not work. It does
            # (the earlier stall was the wall-clock record stamp, 11.14.3, now fixed). It is
            # because the merge buys NOTHING once the broker combines across nodes.
            #
            # MEASURED ON SKY 2026-08-07, identical records from cx19/cx42/cx43 at one moment:
            # fleet_coherent over 6 instances x 7 channels against 3 x 14 -- the same channels,
            # energy-weighted exactly as a merged combiner would sum them, and the two GPUs are
            # phase-coherent (|corr| 0.86-0.99 at phase 0.00 +- 0.17 rad) so the offline merge is
            # faithful -- gives a median fleet deep_snr ratio of 0.997 over 7 PRNs. The WEAK
            # PRNs did slightly better SPLIT (PRN 3: 5.2 vs 4.0; PRN 7: 3.4 vs 2.5), because more
            # instances give the leave-one-out reference and the one-way S/R split more to work
            # with.
            #
            # The GPU boundary is an artefact of which comb landed on which card. Once
            # fleet_coherent exists it carries no information, and collapsing it only reduces the
            # instance count -- the one axis that does matter at the margins. Same argument
            # applies to --combine-gpus on the tracker side; see its help.
            "in_bufs": [n2rec_buf],
            "out_buf": f"{pre}n2cmb_buf",
            "n_prn": n_prn,
            "n_elements": n_live,
            # same record cadence as the tracker now, so the same deep window.
            "integration_length": args.integration_length,
            "integration_mode": "rolling",
            "deep_coherent": True,
            "deep_rate_search": True,
            "deep_rate_min_q": 10.0,
            "phase_track": args.phase_track,
            "sky_deep": args.sky_deep,
            "fft_len": cfg["fengine"]["fft_length"],
            "record_export": 128,
            "phase_dump_prns": [],
            "phase_dump_path": f"/tmp/gnss_n2phase_{node}_{gpu}.txt",
            "cpu_affinity": [cores[(gpu + 4) % len(cores)]],
        },
        f"{pre}n2sink": (
            {"kotekan_stage": "rawFileWrite",
             "in_buf": f"{pre}n2cmb_buf",
             "base_dir": args.record_dir or rt["record_dir"],
             "file_name": f"{node}_gnss{gpu}_n2rec",
             "file_ext": "raw",
             # The tiles buffer carries an NDArray descriptor (set dynamically by the GPU
             # path); we write raw bytes and the reader supplies the layout, which is fixed
             # by construction (see cudaCorrelatorDual.hpp's tile-list note).
             "allow_ndarray": True,
             "cpu_affinity": [cores[(gpu + 7) % len(cores)]]}
            if args.n2_dump else
            {"kotekan_stage": "dropAllFrames",
             "in_buf": f"{pre}n2cmb_buf"}),
    }
    return blocks


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
        # ...but on the GPU path it must ALSO be a whole number of the transform's own bin
        # spacing Fs/(Mp*fft_len) = 62.5 Hz, because that is what makes a Doppler wipe an exact
        # cyclic shift of one forward FFT instead of a fresh transform per trial.
        # 31.25 is exactly HALF a bin, so it misses that by the maximum possible amount, and
        # GnssCudaAcquire will decline the grid and fall back to the CPU (with a WARN saying so).
        # 62.5 is bin-aligned, halves the grid, and costs nothing measurable: channelized_peak's
        # parabolic vertex fit already recovers the Doppler to ~step/20.
        "doppler_step": 62.5 if args.cuda_acquire else 31.25,
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
        # Hinted-only is a COST measure, not a correctness one: the broker knows every visible
        # satellite's Doppler to ~Hz from BRDC, so a blind grid over all 32 PRNs was work we
        # simply could not afford. On the GPU it costs ~8 ms per (PRN, alignment), so
        # reacquisition comes back -- bounded to blind_prns_per_pass rotating slots rather than
        # every below-horizon satellite at once.
        "require_hint": True if not args.cuda_acquire else args.require_hint,
        "blind_prns_per_pass": args.blind_prns_per_pass,
        "use_cuda_acquire": bool(args.cuda_acquire),
        "acquire_windows": args.acquire_windows,
        "acquire_snr": args.acquire_snr,
        # Alignments scanned either side of the broker's ephemeris-predicted one (--nh-hint).
        # The acquire builds a FULL surface per alignment, so 20 of them are ~92% of a pass;
        # 5 of 20 is 4x. Span 2 because the measured offset spread was +-2 on 2026-08-04 --
        # which is itself an open anomaly, nh being deterministic given ephemeris and time.
        # Harmless without a hint: no hint scans all 20, exactly as before.
        "nh_hint_span": 2,
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
        # Refine integration length, hops. The refine LOCALISES a peak that already cleared
        # detection, and costs one full n_chan x n_hops replica build per trial phase -- 82% of
        # all search compute at CHORD dimensions (measured 2026-08-04). Two code periods give a
        # BIT-IDENTICAL answer to the full 16-period record, noiseless and at 20x noise, because
        # the refine only has to pick the best of 14 discrete trial phases and the 0.246-chip
        # step quantisation dominates. 391 ~= 2 x 195.3125.
        "refine_hops": 391,
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


def live_frame0_utc(cfg, args, timeout=3.0):
    """The F-engine epoch as the RUNNING FLEET reports it, or None if nobody answers.

    Every kotekan serves its cached frame 0 at telescope/time0_ns. Nodes CACHE it at startup, so
    this reports what the F-engine was doing when each node started -- which is exactly the
    question here ("is the checked-in value still the live one?") and NOT a liveness check on the
    F-engine. Nodes that started either side of an F-engine restart therefore disagree, and that
    disagreement is itself the answer: no single value is current, so return None and let the
    caller warn rather than silently pick one.
    """
    import urllib.request

    # The node's OWN rest port -- 12049. (The +1 seen elsewhere in this file belongs to the
    # second GPU chain's endpoints, not to the telescope block, which only one server carries.)
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
        # ⚠️ STDERR, NEVER STDOUT. Without --out the generated CONFIG is stdout -- that is
        # how gen_fleet's --check compares byte-for-byte without touching the tree -- so a
        # warning printed here lands INSIDE the config text. This is a LIVE-FLEET probe
        # inside what every consumer treats as a pure function of (base, flags): during a
        # rolling restart the probe sees a node either down or freshly up with no anchor,
        # emits these lines, and --check reports "does not match the manifest" about six
        # byte-identical configs. That exact false alarm fired on cx27/cx42/cx51 during the
        # 2026-08-09 B2a rollout (misdiagnosed then as a checker transient) and AGAIN,
        # legibly this time, during the #32 rollout -- same lesson as the ephemeris pin: an
        # undeclared input does not announce itself.
        print("  WARNING   running nodes disagree about frame 0 -- some started either side of "
              "an F-engine restart, so none of them is authoritative:", file=sys.stderr)
        for ns, nodes in sorted(seen.items()):
            print("              %.9f  %s" % (ns * 1e-9, ", ".join(sorted(nodes))),
                  file=sys.stderr)
        print("            Re-read it from chive, or restart the stragglers, before trusting "
              "any record stamp.", file=sys.stderr)
        return None
    return next(iter(seen)) * 1e-9


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
    ap.add_argument("--n2-dual", action="store_true",
                    help="GNSS path B (docs/gnss_gpu_search.md 11): keep run_send_voltage (the "
                         "GPU voltage ring) and add a run-per-GPU cudaProcess with "
                         "cudaGnssInject + cudaCorrelatorDual -- the two-input N^2 whose mixed "
                         "tiles ARE the despread. Seeds: the broker POSTs the SAME payload to "
                         "/gnss<N>_inject/set_seeds (add the endpoints to --dll-combiners' "
                         "tracker list in broker_up.sh). RFI mask is all-pass, which is "
                         "behavior-identical to production (first-stage excision is off there).")
    ap.add_argument("--n2-debug", action="store_true",
                    help="with --n2-dual: INFO logging + log_profiling on both cudaProcess "
                         "stages + log_kernel_split on the tracker, for per-command attribution. "
                         "Verbose (per-frame lines) -- give it its own GNSS_LOG. A GENERATOR "
                         "FLAG rather than a hand-built file: the hand-built one went stale and "
                         "was profiled hours after the config it described had moved on.")
    ap.add_argument("--n2-dump", action="store_true",
                    help="with --n2-dual: rawFileWrite the gathered tiles to /tmp/gnss "
                         "(~17 MB/s -- short captures only, 1 GB/min) instead of dropAllFrames")
    ap.add_argument("--n2-full-freq", action="store_true",
                    help="cudaCorrelatorDual computes the FULL triangle over every frequency "
                         "(AA included) instead of the mixed+synthetic blocks over the GNSS comb "
                         "only. Required for path B to be the SCIENCE pipeline: the N^2 "
                         "pass-through is skipped in freq-map mode because the AA block is never "
                         "computed there. Costs more -- how much is the open question this flag "
                         "exists to answer (offline: +2.00 ms full vs +0.22 ms mapped; in situ "
                         "the mapped launch runs 0.076 ms).")
    ap.add_argument("--n2-primary", action="store_true",
                    help="PATH B AS THE SCIENCE PIPELINE. Implies --n2-full-freq (the N^2 "
                         "pass-through needs the AA block, which freq-map mode never computes), "
                         "keeps host_correlation_buffer{,_1} and the N^2 consumers "
                         "(n2_accumulate, eigencalc, n2_subset, buffer_send_n2), and adds the "
                         "standard cudaOutputData leg so cudaCorrelatorDual feeds them in place "
                         "of the dropped run_n2k. The prefix is bit-identical to cudaCorrelator's "
                         "(n2dualtest gate [3]) -- but VERIFY IT LIVE with injection on vs off "
                         "before trusting the science output, not just via the unit gate.\n"
                         "\n"
                         "Measured on cx19 2026-08-08, 200 frames: cudaCorrelatorDual runs "
                         "3.698 ms/frame full versus 0.076 ms mapped, and cudaGnssInject 2.443 "
                         "ms, against cudaGnssChordTrack's 17.771 ms for the path it replaces.")
    ap.add_argument("--no-path-a", action="store_true",
                    help="Do not instantiate the path-A tracker for the PRIMARY chain "
                         "(gnss*_gpu / _assemble / _combine / _record / _tap). Requires the "
                         "broker to seed gnss*_inject and read gnss*_n2combine instead -- see "
                         "config/gnss_chains_chord.yaml, where gps_l5 was switched on "
                         "2026-08-09. The SEARCH leg (srch_tap/srch_send) is built in the same "
                         "function and is KEPT: it feeds the aggregator's detector, which "
                         "solves the clock for every chain. Path A measures 17.712 ms/frame "
                         "vs path B's 2.512, so this is ~7/10 of the GNSS frame budget back.")
    ap.add_argument("--keep-n2", action="store_true",
                    help="retain the science pipeline alongside the GNSS branch")
    ap.add_argument("--frame0-nano", type=int, default=None, metavar="NS",
                    help="start WITHOUT chive: supply the F-engine frame-0 time (int64 ns) "
                         "instead of querying chive:54321, and drop config_tracker (same "
                         "service). Get the value from a node that is CURRENTLY PROCESSING "
                         "DATA: curl -s http://cx27:12049/telescope/time0_ns . require_gps "
                         "stays TRUE, so a missing/bad value still FATALs rather than silently "
                         "falling back to host wall clock. "
                         "*** VALID ONLY WHILE THE F-ENGINE KEEPS COUNTING. *** frame 0 is "
                         "re-established by an F-engine restart or firmware/mode change, and "
                         "every node CACHES it at startup -- so a value read from a running "
                         "node proves only what the F-engine was doing when THAT node started, "
                         "not what it is doing now. Re-read it after any F-engine restart. "
                         "Using a stale one fails SILENTLY: records carry the wrong epoch, the "
                         "seeds predict the wrong code phase, and every PRN despreads noise "
                         "while looking like a clock or geometry bug.")
    ap.add_argument("--prns", type=int, nargs="*", default=list(range(1, 33)))
    ap.add_argument("--extra-signal", action="append", default=[], metavar="NAME:PRNS",
                    help="add a SECOND tracker chain for another signal, e.g. "
                         "--extra-signal GAL_E5A_Q_CS:1,2,3. Repeatable. PRNs are REQUIRED "
                         "(constellations have different PRN ranges, and the count is the "
                         "path-B lane budget -- 4 lanes per PRN). A chain whose covering "
                         "channels match the primary's SHARES its voltage tap (E5a/B2a are on "
                         "L5's carrier, so this is the free case); a different carrier gets its "
                         "own tap. Buffer/endpoint names are suffixed with a tag derived from "
                         "the signal name. See docs/CHORD_MULTIBAND.md.")
    ap.add_argument("--integration-length", type=int, default=100)
    ap.add_argument("--hops-per-record", type=int, default=2048,
                    help="10.49 ms at CHORD's 5.12 us hop; divides the 8192-hop frame 4 ways and "
                         "stays under the 20 ms NH20 period, so a record straddles at most one "
                         "overlay transition (which P_HEAD handles)")
    ap.add_argument("--spectrum-window-records", type=int, default=100,
                    help="task #53: /get_spectrum accumulation window, in RECORDS. Windows are "
                         "quantised on the F-engine sample clock (index = wstart // "
                         "(records x hops_per_record x fft_length)), so every instance assigns "
                         "a record to the same window with no negotiation -- which is what the "
                         "old reset-on-read behaviour made impossible, since the window was "
                         "defined by when each GET happened to arrive. 100 records = 1.0486 s, "
                         "matching the combiner's own coherent span so the two agree.")
    ap.add_argument("--spectrum-ring-depth", type=int, default=8,
                    help="task #53: completed spectrum windows kept per instance. Must exceed "
                         "the per-node record lag spread (measured ~0.15 s = ~4 records, task "
                         "#46) so a laggard can still be asked for the window its peers already "
                         "returned; 8 leaves headroom without meaningful memory cost.")
    ap.add_argument("--reference-element", type=int, default=0,
                    help="antenna whose correlation fills the record HEADER -- the broker's DLL "
                         "and carrier loop reference (and, with --elem-sum, the phase anchor of "
                         "the calibrated sum)")
    ap.add_argument("--chan-dump-prn", type=int, default=-1, metavar="PRN",
                    help="DIAGNOSTIC: dump the per-channel PROMPT correlation for this PRN "
                         "(-1 = off) to chan_dump_path, one line per covering channel per "
                         "dumped record: 'utc ch corr_re corr_im energy', raw and "
                         "pre-NCO-rotation so the CROSS-CHANNEL RELATIVE PHASES are the "
                         "observable.\n"
                         "\n"
                         "WHY IT MATTERS BEYOND ITS ORIGINAL USE (the 2026-07-21 ADR-wander "
                         "hunt, where a narrow 5-channel set wandered 5-6x worse than the full "
                         "10): the cross-channel sum in GnssGpuRecordAssemble is the one "
                         "combining step downstream can NEVER undo. Instances, GPUs and nodes "
                         "all stay separable to the broker; channels do not. So if a phase "
                         "varies across FREQUENCY -- a residual code delay tau shows up as a "
                         "ramp 2*pi*df*tau, and 0.1 chip over a 7-channel stride-16 comb "
                         "(~18.75 MHz) is already 1.15 rad -- that coherence is lost inside "
                         "this sum and no amount of fleet combining recovers it. This dump is "
                         "how you find out whether it is happening before rebuilding the record "
                         "format to carry channels.\n"
                         "\n"
                         "~60 KB/s at chan_dump_decim 10; raise the decimation for a long run.")
    ap.add_argument("--chan-dump-decim", type=int, default=10, metavar="N",
                    help="with --chan-dump-prn: dump every Nth record of that PRN.")
    ap.add_argument("--sky-deep", action=argparse.BooleanOptionalAction, default=False,
                    help="combiner: score the SKY-PHASE-CORRECTED prompt (record slots 24/25, "
                         "gnssElemCal's leave-one-out element derotation) as a deep-fold "
                         "candidate rung. Gated off on 2026-08-05 because the split-aperture "
                         "estimator appeared to beat a genie; RESOLVED 2026-08-10: that was a "
                         "comparator error (the split's honest bound is the HALF-aperture "
                         "genie, and against it the split reads 0.84-0.95 in every seed, "
                         "anomaly case included, with the null fail-closed at 4.5-7.8 vs "
                         "27-63 on signal). ON SKY the slots hold phase-coherence 0.92-0.96 "
                         "over 120 s of sparse samples where the raw prompt sits at the "
                         "random-walk floor, amplitude-neutral (|sky|/|raw| ~ 1), weak sats "
                         "failing soft -- measured 2026-08-10 with elem_sum live. The rung "
                         "competes under the same measured floor as every other candidate, "
                         "so a cold or bad cal loses on merit rather than corrupting the "
                         "fold (docs 11.32).")
    ap.add_argument("--elem-sum", action=argparse.BooleanOptionalAction, default=True,
                    help="record HEADER = self-calibrated weighted mean over ALL elements "
                         "(bootstrap MRC, reference-anchored phase, 'one element' scale) instead "
                         "of the bare reference element. ~sqrt(N_healthy) ~ 5x per-record SNR "
                         "for every header consumer: the broker's DLL/carrier loops, the "
                         "combiner's deep fold and its phase tracker. The per-element cal comes "
                         "from the satellite itself (validated: 5.8x of the 7.95x MRC bound on "
                         "synthetic, dead elements auto-gated, dead reference fails over). "
                         "STATE 8.21.5 item 1 -- the phase-floor fix's SNR half.")
    ap.add_argument("--carrier-phase-from-ref", choices=("1", "0", "ab"), default="1",
                    help="TASK #52 A/B ARM -- ⚠️ TEMPORARY, remove with task #55. 1 (default) = "
                         "the carrier phase comes from DespreadJob::ang0 at the window's "
                         "reference sample, so the ~1.18 GHz carrier's rounding never "
                         "multiplies the absolute sample index. 0 = the pre-86349ac4d "
                         "wc*n_abs expression. 'ab' = GPU 0 gets the fix and GPU 1 the old "
                         "code ON EVERY NODE, which is the tightest pairing available: same "
                         "node, same sky, same seeds, same poll. A before/after across two "
                         "restarts CANNOT resolve this -- measured 2026-08-13, deep_snr max "
                         "swung 52-197 inside four minutes and the seeded PRN count moved "
                         "12 -> 5 on geometry alone.")
    ap.add_argument("--phase-track", action=argparse.BooleanOptionalAction, default=False,
                    help="combiner: leave-one-out common-phase tracker before the deep coherent "
                         "sum -- the batch form of the carrier loop the airspy chain closed, "
                         "removing the per-satellite ~0.9 rad propagation wander that capped "
                         "every deep_snr at ~11-14 regardless of brightness (STATE 8.21). "
                         "Fail-closed on noise (self-excluded estimates cannot align it; "
                         "validated on synthetic). Exports coh_frac (the chopping-independent "
                         "coherence measure) and pt_hw beside deep_snr. DEFAULT OFF since\n"
                         "2026-08-05: measured on sky it costs 14.5 -> 13.2 (the wander is ~white\n"
                         "in time, so temporal neighbours carry no information) and it raises the\n"
                         "local deep floor from 2.18 to 3.15 for every satellite. The working fix\n"
                         "is the broker's fleet_coherent (STATE 8.21.6).")
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
    ap.add_argument("--combine-gpus", action="store_true",
                    help="ONE GnssCoherentCombiner over BOTH GPUs' tracker record streams "
                         "instead of one per GPU. The stage is documented for exactly this -- "
                         "'Per-subband tracker record streams', combined coherently, 'one loop "
                         "at full-band SNR instead of N noise-driven per-channel FLLs' -- and "
                         "the two GPUs hold INTERLEAVED stride-16 combs (GPU0 5972,5988,...; "
                         "GPU1 5980,5996,...), so together they are the node's full stride-8 "
                         "comb: 14 channels instead of 7, +3 dB per record. Per-record SNR is "
                         "what currently blocks coherent integration -- carrier_resid_hz is a "
                         "phase fit over the buffered records, so incoherent records give a "
                         "noise fit and the carrier loop has nothing to lock to (STATE 8.12). "
                         "GPU 1's combiner and record stage are dropped; the merged pair is "
                         "written by gnss0_record.\n"
                         "\n"
                         "NOT A SENSITIVITY LEVER ANY MORE, and the rationale above is the "
                         "PRE-fleet_coherent one. Once the broker combines across instances the "
                         "GPU boundary carries no information -- it only records which comb "
                         "landed on which card. Measured on sky 2026-08-07 (path B, identical "
                         "records from cx19/cx42/cx43): 6 instances x 7 channels against 3 x 14, "
                         "median fleet deep_snr ratio 0.997, and the WEAK PRNs did BETTER split "
                         "(5.2 vs 4.0) because more instances give the leave-one-out reference "
                         "and the one-way S/R split more to work with. The synthetic version of "
                         "the same test (scripts/gnss/fleetcoh_partition.py) is flat to 2% from "
                         "2x6 down to 12x1.\n"
                         "\n"
                         "So use it for what it still does -- HALVE THE ENDPOINT COUNT the "
                         "broker polls, and give a node one published deep_snr instead of two -- "
                         "and not for SNR. If the fleet is small, prefer leaving it OFF: "
                         "instance count is the one axis the combine is mildly sensitive to, "
                         "and collapsing spends it for nothing.")
    ap.add_argument("--dish-coelev-deg", type=float, default=-8.59,
                    help="dish pointing co-elevation (degrees from vertical, negative = south), "
                         "OVERRIDING whatever the production base carries. Bases have been "
                         "captured with -27.3, which is 18.7 deg wrong and puts boresight at "
                         "dec +22.0 instead of the true dec +40.73 (elevation 81.41, Cyg A's "
                         "declination, matching its measured 81.5 deg transit). Nothing in the "
                         "GNSS chain reads it -- it feeds CHORDTelescope's dish geometry -- but "
                         "every satellite-pass prediction is computed from it, so a stale value "
                         "silently sends you looking for the wrong satellite on the wrong night.")
    ap.add_argument("--local-trim-gain", type=float, default=0.0,
                    help="cudaGnssChordTrack's IN-TRACKER code-trim gain. 0 (the default since "
                         "2026-08-03) hands the code loop to the broker's fleet DLL "
                         "(--dll-combiners, docs/CHORD_GNSS_SHARED_DLL.md), which sums Early/"
                         "Late POWERS across every instance and so closes at the full 20.46 MHz "
                         "L5 bandwidth instead of the 6.7%% one instance can see. Two loops "
                         "cannot coexist: E/L are measured relative to the phase each instance "
                         "despread at, so independent local trims drift apart and the fleet sum "
                         "smears rather than sharpens. 0.15 restores the stage default for a "
                         "single-node bench or an A/B control.")
    ap.add_argument("--record-dir", default=None,
                    help="override runtime.record_dir from the node file. Needed on any node "
                         "without /data: rawFileWrite takes base_dir from the config, so pointing "
                         "the LOGS elsewhere does not move the RECORDS, and the stage fails on a "
                         "directory that is not there.")
    ap.add_argument("--phase-dump-prns", type=int, nargs="*", default=[],
                    help="PRNs whose per-record despread trajectory the combiner should dump\n"
                         "(arg A, E/L powers, commanded phase increment, code phase) to\n"
                         "/tmp/gnss_phase_dump_<node>_<gpu>.txt. Empty = off. Diagnostic for\n"
                         "the deep-fold phase floor; one line per record per PRN, so keep it\n"
                         "to one or two bright satellites.")
    ap.add_argument("--search-port-base", type=int, default=11040,
                    help="bufferRecv port for GPU 0; GPU 1 uses base+1")
    ap.add_argument("--acquire-threads", type=int, default=16,
                    help="threads for the aggregate half of the acquire (parallel over Doppler "
                         "bins x coarse lags). Aggregator only; per-node instances keep 1. The "
                         "affinity set is widened to match, so this also sizes the core list.")
    ap.add_argument("--acquire-fine-step", type=int, default=128,
                    help="fine-lag decimation in the acquire surface. The fine axis resolves a "
                         "lobe sph/(comb span in bins) wide -- ~156 samples at CHORD regardless "
                         "of channel COUNT (the span sets it, not the density) -- so storing it "
                         "per sample is ~156x oversampled and the surface is the whole cost of a "
                         "pass. Step must stay well under the lobe width or the peak is missed.")
    ap.add_argument("--prns-per-pass", type=int, default=1,
                    help="how many ELIGIBLE PRNs to search per snapshot, round-robin. 0 = all "
                         "(airspy's behaviour). A detection's ref_hop is the SNAPSHOT's start "
                         "hop, so with one snapshot per pass the last PRN searched carries an "
                         "epoch as old as the pass; seed error is Doppler error x that age at "
                         "0.0087 chips/Hz/s. Bounds the epoch at emit by (pass time)/(this).")
    ap.add_argument("--cuda-acquire", action="store_true",
                    help="run the acquisition surface on the GPU (docs/gnss_gpu_search.md). "
                         "Blind dims measured 8.3 ms against 1.15 s of CPU, the surface never "
                         "leaves the device, and the peak agrees with the CPU path to rel 6e-8. "
                         "ALSO sets doppler_step to the bin-aligned 62.5 Hz -- the engine "
                         "declines a half-bin grid and falls back to the CPU. Requires a CUDA "
                         "build; falls back with a WARN otherwise.")
    ap.add_argument("--require-hint", action="store_true", default=True,
                    help="search only PRNs the broker has hinted (see the config comment). "
                         "Only consulted with --cuda-acquire; without it, hinted-only is forced "
                         "because a blind pass is minutes.")
    ap.add_argument("--no-require-hint", dest="require_hint", action="store_false",
                    help="also blind-search unhinted PRNs, bounded by --blind-prns-per-pass.")
    ap.add_argument("--blind-prns-per-pass", type=int, default=0,
                    help="how many UNHINTED PRNs may be blind-scanned per pass, rotating. An "
                         "unhinted PRN costs a FULL Doppler grid, ~30x a hinted one, so with "
                         "--no-require-hint and a sky of below-horizon sats the pass would be "
                         "dominated by satellites that cannot be found. 0 = the old behaviour.")
    ap.add_argument("--acquire-windows", type=int, default=1,
                    help="windows stacked per acquisition attempt. MUST STAY 1 at CHORD until "
                         "the overlay bookkeeping lands (STATE 5r.1). A window is 3125 hops = 16 "
                         "PRIMARY periods, so the code phase is stationary window to window -- "
                         "but 16 is not a whole number of NH20 periods, so each window's overlay "
                         "advances +4 mod 20 and lands in a DIFFERENT nh bin. Stacking therefore "
                         "puts the signal in one bin and noise in all of them: measured on "
                         "noiseless synthetic, 2 windows gave EXACTLY HALF the snr of 1, every "
                         "injection; live, PRN 23 read 342 with one window against 56 with "
                         "eight, and the reported nh was scrambled. This is the same 16-vs-20 "
                         "geometry as the record-length defects, so it does not go away by "
                         "itself. The fix is to route window w into bin (a + 4w) mod 20 -- pure "
                         "bookkeeping -- after which this can rise again and buy real "
                         "integration. Raising it WITHOUT that costs sensitivity and lies about "
                         "nh. It is also the dominant cost term: 32 here is 32x the pass time.")
    ap.add_argument("--acquire-snr", type=float, default=30.0,
                    help="detection threshold. Tied to --acquire-windows: at 1 window the "
                         "pure-noise ceiling is a Gamma(1) tail (~19) rather than Gamma(8) "
                         "(~4.5), so a threshold carried over from a multi-window config lets "
                         "pure noise through as detections. The stage logs its own computed "
                         "ceiling every pass and flags a threshold below it -- read that line "
                         "after any change here. Keep in step with the broker's --acquire-snr.")
    ap.add_argument("--search-instance", action="store_true",
                    help="emit the SEARCH instance config instead of the node config")
    args = ap.parse_args()

    with open(args.node_file) as fh:
        cfg = yaml.safe_load(fh)
    if args.node not in cfg["nodes"]:
        raise SystemExit(f"unknown node {args.node}")

    # THE F-ENGINE EPOCH IS A LIVE VALUE, AND A STALE ONE FAILS SILENTLY. frame0_utc is the UTC
    # of the F-engine's absolute sample 0; every record is stamped frame0_utc + wstart/rate, and
    # an F-engine restart re-establishes frame 0. chord_gnss_node.yaml carries a checked-in
    # value, so it goes stale the moment the F-engine comes back -- and nothing downstream
    # complains, because every cross-record estimator uses DIFFERENCES and a uniformly-wrong
    # epoch is uniform. Found 2026-08-07 while chasing the path-B q gap: the file still held
    # 1784941003.000002861 from a bring-up 13.78 DAYS earlier while every running node served
    # 1786131189.000002870. Path A had been stamping every record 13.78 days in the past for
    # as long as that was true.
    #
    # So: ask a node that is actually running, and refuse to emit a config that disagrees with
    # it. Advisory when nothing answers (the F-engine and chive have both been down this week
    # and the generator must still work), hard error when something does -- a mismatch is
    # always a real staleness, never noise.
    live_t0 = live_frame0_utc(cfg, args)
    if live_t0 is not None:
        have = float(cfg["fengine"].get("frame0_utc", 0.0))
        if abs(have - live_t0) > 1e-3:
            raise SystemExit(
                f"frame0_utc in {args.node_file} is {have:.9f} but the running fleet serves "
                f"{live_t0:.9f} ({abs(have - live_t0) / 86400.0:.2f} days apart).\n"
                f"The F-engine has been restarted since that value was written. Records would "
                f"carry the wrong epoch and nothing downstream would say so.\n"
                f"Fix it:  sed -i 's/^  frame0_utc: .*/  frame0_utc: {live_t0:.9f}/' "
                f"{args.node_file}\n"
                f"(or pass --frame0-nano to override both this and the telescope block.)")
    elif not args.frame0_nano:
        # stderr for the same reason as the frame0-disagreement warning above: stdout IS the
        # config when --out is absent, and this fires whenever the fleet is mid-restart.
        print("  WARNING   no running node answered telescope/time0_ns -- frame0_utc "
              f"{float(cfg['fengine'].get('frame0_utc', 0.0)):.6f} is UNVERIFIED. If the "
              "F-engine has restarted since it was written, every record stamp is wrong.",
              file=sys.stderr)

    # --frame0-nano overrides the RECORD EPOCH too, not just the telescope's startup GPS time.
    # Those are two different consumers of one number and supplying only the first leaves the
    # assembler stamping records from the stale yaml -- the exact silent failure the flag exists
    # to avoid.
    if args.frame0_nano:
        cfg["fengine"]["frame0_utc"] = args.frame0_nano * 1e-9

    with open(args.base) as fh:
        base = json.load(fh) if args.base.endswith(".json") else yaml.safe_load(fh)
    out = copy.deepcopy(base)

    # THE PRODUCTION BASE CARRIES A STALE POINTING. Captured bases have read
    # telescope.dish_coelev_deg = -27.3, which puts boresight at dec +22.0 / elevation 62.7.
    # The dishes are actually 8.59 deg SOUTH of zenith -- dec +40.73, elevation 81.41 -- which
    # is Cyg A's declination and matches its measured 81.5 deg transit. The 18.7 deg
    # disagreement is not academic: it moves every predicted satellite pass. Taking the field at
    # face value on 2026-08-03 produced a completely wrong transit list ("PRN 3 at 0.55 deg in
    # 2 h"; the truth was PRN 19 at 0.40 deg, 10 h later, a different satellite on a different
    # night). Override rather than inherit, and say so in the emitted config.
    if isinstance(out.get("telescope"), dict) and "dish_coelev_deg" in out["telescope"]:
        out["telescope"]["dish_coelev_deg"] = args.dish_coelev_deg

    # --frame0-nano: START WITHOUT chive's timing service.
    #
    # kotekan requires a GPS time at STARTUP (CHORDTelescope), which makes chive:54321 a hard
    # dependency of every config -- so when the F-engine team pulled that service on 2026-08-06
    # cx19 could not start at all, while the five nodes that were ALREADY RUNNING carried on
    # tracking normally (it is a startup-only dependency). That asymmetry is the whole problem:
    # the instrument is fine, the data is fine, and the dev node is locked out.
    #
    # The value is recoverable WITHOUT chive, because every running node serves the same
    # array-wide frame 0 at telescope/time0_ns. Read it from a live node and pass it here:
    #     curl -s http://cx27:12049/telescope/time0_ns
    #
    # WHAT THAT VALUE DOES AND DOES NOT PROVE (learned the embarrassing way, 2026-08-07).
    # Nodes CACHE time0 at startup. Reading the same value from three running nodes shows they
    # all started against the same F-engine frame 0 -- it is NOT evidence the F-engine is
    # running now, and it was wrongly cited as such here. On that morning all five surviving
    # nodes were serving FROZEN status (deep_snr, adr_lock_s and record counts bit-identical
    # across a 25 s poll, and identical between nodes to 12 significant figures) because the
    # F-engine was still in its dev mode with the links up and no data flowing. ALWAYS poll a
    # counter twice before calling a fleet healthy.
    #
    # So this flag is valid only while the F-engine keeps counting from the same frame 0. A
    # restart or mode/firmware change re-establishes it, every cached copy goes stale at once
    # (including the other nodes'), and a stale epoch fails SILENTLY -- wrong record stamps,
    # wrong predicted code phase, every PRN despreading noise while it reads as a clock bug.
    # After any F-engine restart: re-read time0 from chive, or from a node started SINCE.
    #
    # NOTE WHAT THIS DOES NOT DO: `require_gps` stays TRUE. The FATAL fires when NO time was
    # found, and set_gps_time_params_from_config sets gps_enabled from this block -- so a
    # missing or malformed value still dies loudly instead of silently falling back to host
    # wall clock (which would stamp every record ~13 days wrong and look like a clock bug).
    # We are supplying the answer from a verified second source, not disabling the check.
    #
    # config_tracker is dropped for the same outage: it GETs 10.222.0.54:54321/config, the same
    # service, and its failure is fatal at construction.
    if args.frame0_nano:
        out.setdefault("telescope", {})["query_gps"] = False
        out["gps_time"] = {"frame0_nano": int(args.frame0_nano)}
        out.pop("config_tracker", None)
        for _k, _v in out.items():
            if isinstance(_v, dict) and "use_config_tracker" in _v:
                _v["use_config_tracker"] = False

    # WIDEN THE STARTUP FETCH BUDGET (2026-08-08). chord_pathfinder.j2 -- production's
    # template, not ours to change -- ships upstream_fetch_retries 2 / timeout 10 s, and
    # ConfigTracker's failure is FATAL at construction. Against the real service that is far
    # too tight.
    #
    # MEASURED, 20 sequential GETs of chive:54321/config at ~1/s, right after the F-engine
    # came back: median 6 ms, p90 6.5 s, MAX 16.2 s. Bimodal, because fpga_master is a
    # single-threaded asyncio server whose per-second metrics gather now does real register
    # reads over 16 live boards (while the boards were dark it failed instantly and the
    # server looked fast). A request landing in a gather window waits it out.
    #
    # Consequence: a fleet restart was a coin flip. On 2026-08-08 five of six nodes died with
    # "failed to GET FPGA config after 2 retries" while cx42, started BETWEEN two of the
    # failures, came up fine -- so it reads like a node-specific fault rather than a timeout,
    # which is exactly how it wasted an hour.
    #
    # 5 x 30 s bounds the worst case at 150 s before a genuinely-down chive is declared, vs
    # ~20 s before. That is the right trade: a slow start is recoverable, a fleet that
    # half-starts is not. Startup only -- nothing here runs in steady state.
    if "config_tracker" in out and isinstance(out["config_tracker"], dict):
        out["config_tracker"]["upstream_fetch_retries"] = 5
        out["config_tracker"]["upstream_fetch_timeout_seconds"] = 30

    sig = cfg["signals"]
    chans = covering_channels(node_channels(cfg, args.node), float(sig["carrier_hz"]),
                              float(sig["chip_rate_hz"]), float(sig["max_doppler_hz"]))
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
        elif args.n2_primary and key.startswith("run_n2k"):
            # Replaced, in both --keep-n2 modes: cudaCorrelatorDual produces the same prefix on
            # the same leg, and leaving run_n2k in place would give host_correlation_buffer two
            # producers.
            del out[key]
            dropped.append(key)
        elif not args.keep_n2 and key.startswith(N2_STAGE_PREFIXES):
            # --n2-dual: run_send_voltage survives (the dual correlator consumes the GPU
            # voltage ring it feeds); run_n2k itself is still dropped -- cudaCorrelatorDual
            # replaces it and keeps the N^2 prefix on-GPU.
            if args.n2_dual and key.startswith("run_send_voltage"):
                continue
            del out[key]
            dropped.append(key)
        elif (args.n2_dual and not args.n2_primary and not args.keep_n2
              and key.startswith("host_correlation_buffer")):
            # Nothing produces or consumes it in this mode (no cudaOutputData for the N^2
            # prefix in dev); leaving it would just allocate dead host memory.
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
        if args.n2_dual:
            blocks.update(build_n2dual_branch(cfg, args.node, gpu, [i for _, i in pairs],
                                              [f for f, _ in pairs], args,
                                              out.get("samples_per_data_set", 8192)))
        # EXTRA SIGNAL CHAINS on this GPU. Each is a full tracker branch (GPU process,
        # assembler, combiner, writer) under its own tag; a chain on the primary's channels
        # shares the voltage tap, so the marginal cost is GPU + CPU, not ingest.
        for ch in extra_chains:
            ch_pairs = ([(f, i) for f, i in pairs] if ch["shares_tap"]
                        else [(f, i) for f, i in gpu_pairs_for(cfg, args.node, gpu, ch["chans"])])
            if not ch_pairs:
                print(f"  NOTE {ch['signal']}: no covering channels on gpu{gpu}, chain skipped",
                      file=sys.stderr)
                continue
            if args.n2_dual:
                # PATH B FOR EXTRA SIGNALS, and path A only for the primary. Building both for
                # every constellation is what runs the node out of frame: a path-A tracker
                # chain measures 17.712 ms/frame against path B's 2.512 (inject 2.435 +
                # mapped correlator 0.077), and cx19 already sits at ~51% of the 41.94 ms
                # frame with one of each. Two constellations on path A would be ~93%.
                #
                # The primary keeps its path-A chain deliberately: it is the reference every
                # path-B number in section 11 has been judged against, and losing it would
                # mean validating each new signal against nothing.
                blocks.update(build_n2dual_branch(cfg, args.node, gpu, [i for _, i in ch_pairs],
                                                  [f for f, _ in ch_pairs], args,
                                                  out.get("samples_per_data_set", 8192),
                                                  chain=ch))
            else:
                xb, _, _ = build_gnss_branch(cfg, args.node, gpu, [i for _, i in ch_pairs], args,
                                             freq_ids=[f for f, _ in ch_pairs], chain=ch)
                blocks.update(xb)

        if args.combine_gpus and gpu != 0:
            # GPU 0's combiner consumed this GPU's rec_buf too, so its own combiner would be a
            # duplicate reader of the same stream -- two stages competing for one buffer, each
            # seeing half the frames. Drop it and its record stage; the merged output is written
            # by gnss0_record. The BUFFER stays: gnss1_rec_buf is still produced, just consumed
            # by the other branch.
            # NB the path-B combiner is deliberately NOT collapsed alongside this one -- see its
            # in_bufs note: measured on sky, the collapse is worth 0.997 and costs instances.
            for k in (f"gnss{gpu}_combine", f"gnss{gpu}_record", f"gnss{gpu}_cmb_buf"):
                blocks.pop(k, None)
        out.update(blocks)

    if getattr(args, "n2_debug", False):
        out["log_level"] = "INFO"
        # run_n2k too, when it survives: it is the ONLY way to get bare N^2's per-frame time,
        # and without it the coexistence comparison has a hole exactly where the interesting
        # number is. Costs one INFO block per GPU frame, same as the GNSS chains already emit.
        if "run_n2k" in out:
            for g in ("gpu_0", "gpu_1"):
                if g in out["run_n2k"]:
                    out["run_n2k"][g]["log_level"] = "INFO"
        for g in (0, 1):
            for blk in (f"gnss{g}_gpu", f"gnss{g}_n2dual"):
                if blk in out:
                    out[blk]["log_profiling"] = True
            for cm in out.get(f"gnss{g}_gpu", {}).get("commands", []):
                if cm.get("name") == "cudaGnssChordTrack":
                    cm["log_kernel_split"] = True

    # ⚠️ THE HEADER IS THE PROVENANCE, AND IT MUST BE COMPLETE ENOUGH TO REGENERATE FROM.
    #
    # It used to say only node/base-basename/signal. That is a description, not a recipe: the
    # ~48 flags below it are what actually shape the file, and none of them were recorded
    # anywhere -- not here, not in a script, not in a commit message. So the six configs the
    # fleet ran all through 2026-08-08 were, in practice, unreproducible; regenerating one
    # meant guessing which flags had been typed days earlier, and the honest answer to "can
    # we regenerate this?" was no. (Recovered 2026-08-09 by diffing candidate outputs against
    # the committed files until all six came back byte-identical -- an hour that a single
    # comment line would have saved.)
    #
    # ARGV + THE BASE'S HASH, because those are the only two inputs. Deliberately NOT the
    # generator's git SHA: the check that matters is "do these inputs still produce this
    # file", which compares OUTPUT, so it goes red exactly when the generator's behaviour
    # moves -- and then a regeneration is what you want, reviewed as a diff.
    try:
        with open(args.base, "rb") as _bf:
            base_sha = hashlib.sha256(_bf.read()).hexdigest()[:16]
    except Exception:
        base_sha = "unreadable"

    # Paths inside the repo are recorded RELATIVE to it. An absolute /home/kvand/... in a
    # committed file is provenance that stops being true on any other checkout, and the whole
    # point of this header is that someone else can act on it.
    def _rel(p):
        ap_ = os.path.abspath(p)
        return os.path.relpath(ap_, K_ROOT) if ap_.startswith(K_ROOT + os.sep) else p
    K_ROOT = os.path.dirname(CONF)
    # --out IS DROPPED FROM THE RECIPE, deliberately. It records WHERE this file was
    # written, which is not part of how it was BUILT -- and including it made the manifest
    # gate structurally incapable of passing: gen_fleet writes with --out and checks by
    # regenerating to stdout WITHOUT it, so every committed file differed from its own
    # regeneration by exactly these two header lines. That is how the fleet came to warn
    # "does not match the manifest" on every single node restart (2026-08-09) -- a gate that
    # always fires teaches everyone to ignore it, which is worse than no gate at all. The
    # recipe stays paste-able: add --out yourself, or let gen_fleet do it.
    _argv = list(sys.argv[1:])
    _keep = []
    _i = 0
    while _i < len(_argv):
        if _argv[_i] == "--out":
            _i += 2
            continue
        if _argv[_i].startswith("--out="):
            _i += 1
            continue
        _keep.append(_argv[_i])
        _i += 1
    argv_line = " ".join(shlex.quote(_rel(a)) for a in _keep)
    base_disp = _rel(args.base)
    hdr = [
        "# GENERATED by config/gen_chord_gnss_config.py -- DO NOT HAND-EDIT.",
        f"# node {args.node}  base {os.path.basename(args.base)}  signal {sig['primary']}",
        f"# covering channels {len(chans)}: freq_id {chans[0]}..{chans[-1]}",
        f"# rest port {port} (production owns 12048)",
        f"# outputs disabled: {sorted(dropped)[:4]}{' ...' if len(dropped) > 4 else ''}",
        f"# base sha256 {base_sha} ({base_disp})",
        "# REGENERATE WITH (this is the whole recipe -- flags included):",
    ] + ["#   " + ln for ln in textwrap.wrap(
        "config/gen_chord_gnss_config.py " + argv_line, width=92,
        subsequent_indent="    ", break_on_hyphens=False,
        # A 200-char --extra-signal PRN list must survive intact: a recipe chopped mid-token
        # is not a recipe, and this line exists to be pasted.
        break_long_words=False)] + [
        "# Prefer `scripts/gnss/gen_fleet.py config/gnss_fleet_chord.yaml`, which holds the",
        "# whole fleet's flags in one versioned file and can --check them against these.",
        "# Edit config/chord_gnss_node.yaml and regenerate instead of hand-editing here.",
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
    print(f"  record_floats {record_floats} (26 header + n_elem*12)", file=sys.stderr)
    print(f"  rest port     {port}", file=sys.stderr)
    print(f"  dropped       {len(dropped)} production blocks", file=sys.stderr)

    # EXTRA CHAINS: state what they cost. A 10-core pool does not grow, so a multi-chain node
    # oversubscribes it; the rotation only guarantees that no two chains put the SAME heavy
    # stage on one core. Print the busiest cores and the shared taps so the cost is read off
    # the generator rather than discovered from a slow node.
    if extra_chains:
        load = {}
        for k, v in out.items():
            if k.startswith("gnss") and isinstance(v, dict) and "cpu_affinity" in v:
                for c in v["cpu_affinity"]:
                    load.setdefault(c, []).append(v.get("kotekan_stage", "?"))
        busiest = sorted(load.items(), key=lambda kv: -len(kv[1]))[:3]
        for ch in extra_chains:
            n_here = sum(1 for g in per_gpu if gpu_pairs_for(cfg, args.node, g, ch["chans"]))
            print(f"  + chain       {ch['signal']} tag '{ch['tag'].strip('_')}' "
                  f"{len(ch['prns'])} PRNs, {len(ch['chans'])} ch on {n_here} gpu(s), "
                  f"tap {'SHARED with the primary' if ch['shares_tap'] else 'OWN (new carrier)'}",
                  file=sys.stderr)
        print("  cpu load      " + ", ".join(f"core {c}: {len(v)} stages" for c, v in busiest)
              + f"  (pool of {len(cfg['runtime']['cpu_affinity'])})", file=sys.stderr)
        print("  NB extra chains are DEAD-RECKON seeded: no search feed is emitted for them "
              "(docs/CHORD_MULTIBAND.md section 5)", file=sys.stderr)


if __name__ == "__main__":
    main()
