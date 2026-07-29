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

# Stages that exist only to feed the N2 science products. Dropped unless --keep-n2.
N2_STAGE_PREFIXES = (
    "run_n2k", "n2_accumulate", "eigencalc", "n2_subset", "hex_dump",
    "buffer_send_n2", "compute_RFI_frame_mask", "count_rfi_mask", "rfi_sk_metrics",
    "run_rfi_", "run_recv_rfi_", "run_pl_", "count_PL", "PL_mask_compactor",
    "run_send_pl_mask", "set_bf_mask", "process_packet_mask",
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
    ap.add_argument("--buffer-depth", type=int, default=4)
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

    # --- safety: never answer on production's port -------------------------------------------
    port = args.rest_port if args.rest_port is not None else cfg["runtime"]["rest_port"]
    if port == 12048:
        raise SystemExit("refusing to generate a config on port 12048: choco owns that port and "
                         "would redirect or reconfigure this instance")
    out["rest_server"] = {"port": port, "cpu_affinity": cfg["runtime"]["cpu_affinity"]}

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
