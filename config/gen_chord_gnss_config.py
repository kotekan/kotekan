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

from chord_band_plan import covering_channels, node_channels  # noqa: E402

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


def build_gnss_branch(cfg, node, gpu, chan_idx, args):
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
                 "gpu_mem_input": f"{pre}voltage",
                 "gpu_mem_output": f"{pre}epl",
                 "signal": sig["primary"],
                 "prns": args.prns,
                 "n_channels": n_chan,
                 "n_elements": n_elem,
                 "elem_stride": n_elem,
                 "frame_chan_stride": n_chan,
                 "hops_per_record": args.hops_per_record,
                 "fft_length": cfg["fengine"]["fft_length"],
                 "sample_rate": float(cfg["fengine"]["sampling_rate_MHz"]) * 1e6,
                 "seed_endpoint": f"/{pre}track/set_seeds"},
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
            "cpu_affinity": [cores[(gpu + 2) % len(cores)]],
        },
        f"{pre}record": {
            "kotekan_stage": "rawFileWrite",
            "in_buf": cmb_buf,
            "base_dir": rt["record_dir"],
            "file_name": f"{node}_gnss{gpu}_cmb",
            "file_ext": "raw",
            "cpu_affinity": [cores[(gpu + 4) % len(cores)]],
        },
    }
    return blocks, record_floats, n_elem


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
        "gnss_pool": {"kotekan_metadata_pool": "GnssChanMetadata",
                      "num_metadata_objects": 30 * args.buffer_depth},
        # The search stage builds a Telescope-free replica bank, but kotekan constructs a
        # Telescope regardless, so it needs a minimal valid block.
        "telescope": {"name": "ICETelescope", "num_polarizations": 1, "num_dishes": 1,
                      "query_gps": False, "require_gps": False},
    }
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
                "cpu_affinity": [cores[(gpu + 2) % len(cores)]],
            },
            f"{pre}search": {
                "kotekan_stage": "GnssChannelizedSearch",
                "in_buf": f"{pre}f_buf",
                "signal": sig["primary"],
                "prns": args.prns,
                "sample_rate": float(fe["sampling_rate_MHz"]) * 1e6,
                "f_offset": 0.0,
                "spectrum_length": fe["num_bins"],
                "num_taps": fe["pfb_taps"],
                "pfb_window": fe["pfb_window"],
                "channel_offset": chan0,   # unused when channel_ids is given; kept for logs
                "n_channels": n_chan,
                # The comb is SPARSE -- these are every 16th bin, not a contiguous block --
                # so the search needs the explicit global indices. With only channel_offset it
                # would correlate against replicas for the wrong bins and report "no satellite".
                "channel_ids": [fid for fid, _ in pairs],
                "doppler_min": -float(sig["max_doppler_hz"]),
                "doppler_max": float(sig["max_doppler_hz"]),
                "doppler_step": 250.0,
                "acquire_windows": args.acquire_windows,
                "acquire_snr": args.acquire_snr,
                "cpu_affinity": [cores[(gpu + 4) % len(cores)]],
            },
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
    ap.add_argument("--search-port-base", type=int, default=11040,
                    help="bufferRecv port for GPU 0; GPU 1 uses base+1")
    ap.add_argument("--acquire-windows", type=int, default=64,
                    help="incoherent 1 ms windows stacked per acquisition attempt")
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
            cfg, args.node, gpu, [i for _, i in pairs], args)
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
