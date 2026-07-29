#!/usr/bin/env python3
"""gen_band_config.py -- emit the three live band configs from config/gnss_node.yaml
(audit 2026-07-19 rec C: the band yamls become GENERATED artifacts so per-chain parameters
can never silently diverge again).

    python3 config/gen_band_config.py                    -> config/generated/live_*.yaml
    python3 config/gen_band_config.py --out-dir config   -> overwrite the deployed files
    python3 config/gen_band_config.py --check            -> generate + STRUCTURAL diff vs the
                                                            hand files in config/ (yaml.safe_load
                                                            deep-compare; exit 1 on any diff)

The generator sits UPSTREAM of gen_3band_config.py: the emitted files keep every shape that
script (and run_live.sh / run_band.sh) greps for -- column-0 stage names, flow-style
`clock_profile:` / `bds_combiner:` one-liners, double-quoted seed_endpoint / base_dir /
CORS urls, 4-space block indentation, single-quoted viewer exec lines.

Kotekan arithmetic expressions ("samples_per_data_set * sizeof_float") are emitted VERBATIM
as plain scalars -- they are strings to yaml.safe_load and are evaluated downstream by
kotekan, never here.

Stdlib + PyYAML only.
"""
import argparse
import math
import os
import re
import sys

import yaml

CONF = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- emission layer
class Raw(str):
    """A scalar emitted verbatim (expressions, pre-quoted strings, hex serials)."""


def dq(s):
    return Raw('"%s"' % s)


NEED_QUOTE = re.compile(r"""[{}\[\],:#'"]|\$\{""")


def fmt(v):
    """Format a scalar/list/dict for YAML emission (flow style for containers)."""
    if isinstance(v, Raw):
        return str(v)
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join("%s: %s" % (k, fmt(x)) for k, x in v.items()) + " }"
    if isinstance(v, str):
        if v == "":
            return '""'
        if NEED_QUOTE.search(v):
            return "'" + v.replace("'", "''") + "'"
        return v
    raise TypeError("cannot emit %r" % (v,))


class E:
    """One emitted entry: key, value, comment block above, inline comment after."""

    def __init__(self, key, val, c=None, i=None):
        self.key, self.val, self.c, self.i = key, val, c, i


SPACER = E(None, None)


class Block(list):
    """Block-style mapping (list of E), 4-space indent per level."""


class Cmds(list):
    """cudaProcess commands: list of Block, emitted as '- key: val' items."""


def _comment_lines(text, pad):
    return [(pad + "# " + ln).rstrip() for ln in text.rstrip("\n").split("\n")]


def emit(entries, out, indent=0):
    pad = " " * indent
    for e in entries:
        if e.key is None:
            out.append("")
            continue
        if e.c:
            out.extend(_comment_lines(e.c, pad))
        if isinstance(e.val, Block):
            out.append("%s%s:" % (pad, e.key))
            emit(e.val, out, indent + 4)
        elif isinstance(e.val, Cmds):
            out.append("%s%s:" % (pad, e.key))
            ipad = " " * (indent + 4)
            for cmd in e.val:
                first = True
                for ce in cmd:
                    if ce.c:
                        out.extend(_comment_lines(ce.c, ipad + ("" if first else "  ")))
                    line = ipad + ("- " if first else "  ") + "%s: %s" % (ce.key, fmt(ce.val))
                    if ce.i:
                        line += "    # " + ce.i
                    out.append(line)
                    first = False
        else:
            line = "%s%s: %s" % (pad, e.key, fmt(e.val))
            if e.i:
                line += "  # " + e.i
            out.append(line)


# --------------------------------------------------------------------------- band assembly
def prn_list(n):
    return list(range(1, n + 1))


def chain_prefix(chain):
    return "" if chain["constellation"] == "gps" else chain["constellation"] + "_"


def np_var(chain):
    """The n_prn variable name this chain's frame-size expressions reference."""
    return chain_prefix(chain) + "n_prn"


def std_buf(pool, frame_size, depth="buffer_depth"):
    return {
        "kotekan_buffer": "standard",
        "metadata_pool": pool,
        "num_frames": Raw(depth),
        "frame_size": Raw(frame_size),
    }


def valve_depth(band, sh):
    """Frame count for the buffers DOWNSTREAM of the Valve, sized in TIME not frames.

    ``buffer_depth`` is a frame COUNT, and the bands do not agree on what a frame is:
    L1/L2C run 9.984 ms frames, L5 runs 2.496 ms. Sixteen of each therefore bought L1
    320 ms of elasticity and L5 only 80 -- and L5 is the band with the least GPU slack
    to begin with. That asymmetry is visible in the drop counters: a GPU stall that L1
    and L2C ride out makes L5 throw frames away (measured 2026-07-27 while an offline
    despread bench shared the device).

    A buffer converts VARIANCE into LATENCY; it cannot fix a mean deficit. L5's tracker
    phase is ~0.61 ms of a 2.496 ms budget, so the drops are pure variance -- other
    bands' bursts, the search stages, anything else on the GPU -- and elasticity is
    exactly the right medicine. The cost is latency, and it is irrelevant here: every
    consumer downstream is anchored on absolute sample time, and the freshness gates
    that do exist (hint_ttl_s, --bias-det-fresh-s) are 30 s.
    """
    frame_ms = 1e3 * float(band["samples_per_data_set"]) / float(band["sample_rate"])
    want = int(math.ceil(float(sh["valve_slack_ms"]) / frame_ms))
    return max(int(sh["buffer_depth"]), want)


def search_stage(sh, chain):
    """The GnssChannelizedSearch stage for one chain (primary -> flow, else block)."""
    prim = chain["constellation"] == "gps"
    head = [
        ("kotekan_stage", "GnssChannelizedSearch"),
        ("in_buf", "chan_buf"),
        ("channel_offset", sh["channel_offset"]),
        ("n_channels", sh["n_channels"]),
    ]
    if not prim:
        head += [
            ("signal", chain["signal"]),
            ("n_prn", chain["n_prn"]),
            ("prns", prn_list(chain["n_prn"])),
        ]
    params = list(chain["search"]["params"].items())
    if prim:
        return dict(head + params)
    return Block([E(k, v) for k, v in head + params])


def tracker_cmd(sh, chain, stream, per_chain_streams, first):
    """One cudaGnssTrack command Block."""
    prefix = chain_prefix(chain)
    gm_out = (prefix + "epl") if prefix else "gnss_epl"
    trk = chain["tracker"]
    cmd = Block()
    cmd.append(E("name", "cudaGnssTrack"))
    if chain.get("tracker_signal_explicit", True):
        cmd.append(E("signal", chain["signal"]))
    cmd.append(E("max_anchor_age_s", sh["max_anchor_age_s"],
                 c=sh["notes"]["re_pin"] if first else None))
    cmd.append(E("fll_reacq_hz", trk["fll_reacq_hz"],
                 c=trk.get("note"), i=trk.get("note_inline")))
    if "max_cover_bins" in trk:  # diagnostic channel-width trim (see cudaGnssTrack.hpp)
        cmd.append(E("max_cover_bins", trk["max_cover_bins"], i=trk.get("max_cover_note")))
    cmd.append(E("cuda_stream", stream))
    cmd.append(E("gpu_mem_input", "gnss_chan_frame"))
    cmd.append(E("gpu_mem_output", gm_out))
    cmd.append(E("quantized_input", sh["quantized_input"], i=sh["notes"]["quantized_inline"]))
    cmd.append(E("in_frame_len", Raw(sh["expr"]["in_frame_len"])))
    cmd.append(E("n_channels", sh["n_channels"]))
    cmd.append(E("channel_offset", sh["channel_offset"]))
    if prefix:
        cmd.append(E("n_prn", chain["n_prn"]))
        cmd.append(E("prns", prn_list(chain["n_prn"])))
    cmd.append(E("capture_utc0", sh["capture_utc0"]))
    cmd.append(E("fll_gain", sh["fll_gain"]))
    cmd.append(E("carrier_shared", sh["carrier_shared"]))
    cmd.append(E("seed_endpoint", dq(chain["seed_endpoint"])))
    # Generic per-chain tracker extras (tracker: params: {...}) -- the peel knobs arrived this
    # way (2026-07-24) and the next chain-specific cudaGnssTrack key should too, instead of
    # growing another special case above.
    for k, v in trk.get("params", {}).items():
        cmd.append(E(k, v, i=trk.get("params_note_inline") if k == "peel" else None))

    out = Block([E("name", "cudaOutputData")])
    if per_chain_streams:
        out.append(E("cuda_stream", stream))
    out.append(E("gpu_mem", gm_out))
    out.append(E("out_buf", (prefix or "") + "epl_buf"))
    out.append(E("in_buf", ""))
    return cmd, out


def combiner_stage(chain, prefix):
    p = {"kotekan_stage": "GnssCoherentCombiner"}
    params = dict(chain["combiner"]["params"])
    for k in ("nh_assist", "nh_min_refs"):  # keep on the flow line's front (run_live greps
        if k in params:                     # "bds_combiner:.*nh_assist" on ONE line)
            p[k] = params.pop(k)
    p["in_bufs"] = [prefix + "rec_buf"]
    p["out_buf"] = prefix + "out_buf"
    for k, v in params.items():
        if k == "prns" and v == "RANGE":
            v = prn_list(chain["n_prn"])
        elif k == "phase_dump_path":
            v = dq(v)
        p[k] = v
    return p


def record_stage(band, chain, prefix):
    sh_files = band["_shared"]
    return {
        "kotekan_stage": "rawFileWrite",
        "in_buf": prefix + "out_buf",
        "base_dir": dq(band["record_dir"]),
        "file_name": dq(chain["record_name"]),
        "file_ext": dq("raw"),
        "prefix_hostname": False,
        "num_frames_per_file": sh_files["num_frames_per_file"],
        "exit_after_n_files": sh_files["exit_after_n_files"],
    }


def build_band(node, band_id):
    sh = node["shared"]
    band = dict(node["bands"][band_id])
    band["_shared"] = sh
    chains = band["chains"]
    prim = chains[0]
    prim_search = prim["search"]["params"]

    doc = Block()
    add = lambda k, v, c=None, i=None: doc.append(E(k, v, c, i))

    add("type", "config")
    add("log_level", sh["log_level"])
    add("instrument_name", sh["instrument_name"])
    add("telescope", dict(sh["telescope"]))
    doc.append(SPACER)
    cors = [dq("http://%s:%d" % (h, band["http_port"])) for h in sh["cors_hosts"]]
    add("rest_server", Block([E("cors_allow_origins", cors)]), c=sh["notes"]["cors"])
    doc.append(SPACER)
    add("clock_profile", {"name": sh["clock_profile"]["name"]}, c=sh["notes"]["clock_profile"])
    doc.append(SPACER)
    add("samples_per_data_set", band["samples_per_data_set"], c=band.get("frame_note"))
    add("bytes_per_sample", sh["bytes_per_sample"])
    add("buffer_depth", sh["buffer_depth"])
    vdepth = valve_depth(band, sh)
    add("valve_depth", vdepth, i="post-Valve elasticity in FRAMES = ceil(%d ms / %.3f ms "
                                 "frame); buffer_depth is a frame count and the bands "
                                 "disagree on what a frame is, so it is sized in time"
                                 % (sh["valve_slack_ms"],
                                    1e3 * float(band["samples_per_data_set"])
                                    / float(band["sample_rate"])))
    add("sizeof_float", sh["sizeof_float"])
    add("cpu_affinity", sh["cpu_affinity"], i=sh["cpu_affinity_note"])
    doc.append(SPACER)
    add("spectrum_length", sh["spectrum_length"], i=band.get("spectrum_note"))
    add("num_taps", sh["num_taps"])
    add("sample_rate", Raw(band["sample_rate"]))
    add("f_offset", Raw(band["f_offset"]), i=band.get("f_offset_note"))
    add("signal", prim["signal"])
    add("code_doppler_sign", sh["code_doppler_sign"])
    add("n_prn", prim["n_prn"])
    add("record_floats", sh["record_floats"])
    add("doppler_margin_hz", sh["doppler_margin_hz"])
    add("acquire_snr", prim_search["acquire_snr"])
    add("acquire_windows", prim_search["acquire_windows"])
    add("doppler_min", band["doppler_min"], c=band.get("doppler_note"))
    add("doppler_max", band["doppler_max"])
    add("doppler_step", band["doppler_step"])
    add("prns", prn_list(prim["n_prn"]), c=band.get("prns_note"))
    doc.append(SPACER)

    add("gnss_pool", {"kotekan_metadata_pool": "GnssChanMetadata",
                      "num_metadata_objects": Raw(sh["expr"]["metadata_objects"])})
    add("input_buf", std_buf("none", sh["expr"]["input_frame"]))
    add("chan_buf", std_buf("gnss_pool", sh["expr"]["chan_frame"]))
    # The two buffers the Valve protects: everything between it and the GPU. These are
    # what absorb a transient GPU stall, so they are the ones sized in time (valve_depth).
    add("chan_buf2", std_buf("gnss_pool", sh["expr"]["chan_frame"], depth="valve_depth"))
    add("q_buf", std_buf("gnss_pool", sh["expr"]["q_frame"], depth="valve_depth"),
        c=sh["notes"]["q_buf"])
    add("rec_buf", std_buf("none", sh["expr"]["rec_frame"].format(np="n_prn")))
    add("out_buf", std_buf("none", sh["expr"]["rec_frame"].format(np="n_prn")))
    add("epl_buf",
        std_buf("none", prim.get("epl_frame", sh["expr"]["epl_frame"]).format(np="n_prn")),
        c=sh["notes"]["epl_buf"])
    doc.append(SPACER)

    asp = band["airspy"]
    ab = Block()
    ab.append(E("cpu_affinity", asp["cpu_affinity"]))
    ab.append(E("kotekan_stage", "airspyInput"))
    ab.append(E("out_buf", "input_buf"))
    ab.append(E("serial", Raw(asp["serial"]), c=sh["notes"]["serial_pin"],
                i=asp.get("serial_note")))
    ab.append(E("freq", asp["freq"]))
    ab.append(E("sample_bw", asp["sample_bw"], i=asp.get("sample_bw_note")))
    for g in ("gain_lna", "gain_mix", "gain_if"):
        ab.append(E(g, asp["gain"], c=asp.get("gain_note") if g == "gain_lna" else None))
    ab.append(E("biast_power", False, c=sh["notes"]["biast"]))
    ab.append(E("dither_disable", True, c=sh["notes"]["dither"]))
    add("airspy_in", ab)
    add("fengine", {"kotekan_stage": "fftwEngine", "in_buf": "input_buf",
                    "out_buf": "chan_buf", "input_type": "real", "pfb_window": "hamming"})
    doc.append(SPACER)
    add("gps_search", search_stage(sh, prim), c=prim["search"].get("note"))
    add("valve", {"kotekan_stage": "Valve", "in_buf": "chan_buf", "out_buf": "chan_buf2"})
    add("quant", {"kotekan_stage": "GnssQuantize44", "in_buf": "chan_buf2", "out_buf": "q_buf",
                  "target_lsb": sh["target_lsb"], "measure_frames": sh["measure_frames"]},
        i=sh["notes"]["quant_inline"])
    doc.append(SPACER)

    gt = Block()
    gt.append(E("kotekan_stage", "cudaProcess"))
    gt.append(E("gpu_id", 0))
    gt.append(E("num_cuda_streams", band["num_cuda_streams"]))
    gt.append(E("frame_arrival_period", 0.0))
    gt.append(E("in_buffers", {"q_buf": "q_buf"}))
    outbufs = {}
    for c in chains:
        name = chain_prefix(c) + "epl_buf"
        outbufs[name] = name
    gt.append(E("out_buffers", outbufs))
    cmds = Cmds()
    cmds.append(Block([E("name", "cudaInputData",
                         c="ONE host->device copy of the channelized 4+4b frame"),
                       E("in_buf", "q_buf"),
                       E("gpu_mem", "gnss_chan_frame")]))
    per_chain = band["per_chain_streams"]
    for idx, c in enumerate(chains):
        stream = idx + 1 if per_chain else 1
        trk, out = tracker_cmd(sh, c, stream, per_chain, first=(idx == 0))
        cmds.append(trk)
        cmds.append(out)
    gt.append(E("commands", cmds))
    add("gputrack", gt, c=band.get("gputrack_note"))
    add("assemble", {"kotekan_stage": "GnssGpuRecordAssemble", "in_buf": "epl_buf",
                     "out_buf": "rec_buf", "sample_rate": Raw(band["sample_rate"]),
                     **{k: (dq(v) if k == "chan_dump_path" else v)
                        for k, v in prim.get("assemble", {}).items()}})
    add("gps_combiner", combiner_stage(prim, ""), c=prim["combiner"].get("note"))
    add("record", record_stage(band, prim, ""))
    doc.append(SPACER)

    for c in chains[1:]:
        prefix = chain_prefix(c)
        npv = np_var(c)
        add(npv, c["n_prn"], c=c.get("section_note"))
        # rec_buf_depth: per-chain override of the combiner's INPUT elasticity, in frames.
        # buffer_depth (16) is sized for a combiner whose emit is short compared with a frame.
        # A combiner whose deep block takes far longer than one frame period stalls its input
        # while it runs, and because gputrack is SHARED across the band, that backpressure
        # reaches every chain on it -- one slow combiner drops the whole band's frames.
        # MEASURED 2026-07-29: the L5-I composed wipe emits every ~1 s of data and takes 785 ms
        # to do it, at a duty of 0.671 -- i.e. it KEEPS UP on average and only needs to ride out
        # the burst. 16 frames is ~40 ms at L5. These frames are n_prn*record_floats*4 bytes
        # (3 KB at n_prn 32), so depth here is essentially free.
        add(prefix + "rec_buf", std_buf("none", sh["expr"]["rec_frame"].format(np=npv),
                                        depth=c.get("rec_buf_depth", "buffer_depth")))
        add(prefix + "out_buf", std_buf("none", sh["expr"]["rec_frame"].format(np=npv)))
        add(prefix + "epl_buf",
            std_buf("none", c.get("epl_frame", sh["expr"]["epl_frame"]).format(np=npv)))
        # A chain without a `search:` block has NO search stage -- the L2C CL pilot is
        # DERIVED from its CM sibling (broker --cl-tracker), never acquired. Everything
        # else (tracker command, assemble, combiner, record) is emitted as usual.
        if "search" in c:
            add(prefix + "search", search_stage(sh, c), c=c["search"].get("note"))
        add(prefix + "assemble",
            {"kotekan_stage": "GnssGpuRecordAssemble", "in_buf": prefix + "epl_buf",
             "out_buf": prefix + "rec_buf", "sample_rate": Raw(band["sample_rate"]),
             "prns": prn_list(c["n_prn"]),
             **{k: (dq(v) if k == "chan_dump_path" else v)
                for k, v in c.get("assemble", {}).items()}})
        add(prefix + "combiner", combiner_stage(c, prefix), c=c["combiner"].get("note"))
        add(prefix + "record", record_stage(band, c, prefix))
        doc.append(SPACER)

    add("spawn_pyviewer",
        {"kotekan_stage": "SpawnProcess", "in_buf": "out_buf",
         "exec": sh["viewer_exec_prefix"] + " " + band["viewer_args"]})

    lines = ["---",
             "# AUTO-GENERATED by config/gen_band_config.py from config/gnss_node.yaml -- DO NOT",
             "# EDIT (hand-editing a generated live_*.yaml is a bug; edit the node table and",
             "# regenerate; `gen_band_config.py --check` must pass before deploy)."]
    lines.extend(_comment_lines(band["header"], ""))
    emit(doc, lines)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- structural diff
def deep_diff(hand, gen, path="", diffs=None):
    """Every path where the parsed hand file and the parsed generated file differ.
    Strict on scalar types too (50 vs 50.0 is a difference: kotekan may care)."""
    if diffs is None:
        diffs = []
    if type(hand) is not type(gen):
        diffs.append("%s: type %s (%r) vs generated %s (%r)"
                     % (path or "<root>", type(hand).__name__, hand,
                        type(gen).__name__, gen))
    elif isinstance(hand, dict):
        for k in sorted(set(hand) | set(gen), key=str):
            p = "%s.%s" % (path, k) if path else str(k)
            if k not in gen:
                diffs.append("%s: MISSING from generated (hand: %r)" % (p, hand[k]))
            elif k not in hand:
                diffs.append("%s: EXTRA in generated: %r" % (p, gen[k]))
            else:
                deep_diff(hand[k], gen[k], p, diffs)
    elif isinstance(hand, list):
        if len(hand) != len(gen):
            diffs.append("%s: list length %d vs generated %d" % (path, len(hand), len(gen)))
        for i, (a, b) in enumerate(zip(hand, gen)):
            deep_diff(a, b, "%s[%d]" % (path, i), diffs)
    elif hand != gen:
        diffs.append("%s: %r vs generated %r" % (path, hand, gen))
    return diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--node", default=os.path.join(CONF, "gnss_node.yaml"))
    ap.add_argument("--out-dir", default=os.path.join(CONF, "generated"))
    ap.add_argument("--check", action="store_true",
                    help="generate + structural-diff against the hand files in config/")
    args = ap.parse_args()

    with open(args.node) as f:
        node = yaml.safe_load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    failed = False
    for band_id in node["bands"]:
        out_file = node["bands"][band_id]["out_file"]
        text = build_band(node, band_id)
        out_path = os.path.join(args.out_dir, out_file)
        with open(out_path, "w") as f:
            f.write(text)
        gen = yaml.safe_load(text)  # always re-parse: emitted text must be valid YAML
        keys = re.findall(r"(?m)^([A-Za-z_][A-Za-z0-9_]*):", text)
        assert len(keys) == len(set(keys)), "duplicate top-level keys in %s" % out_file
        print("wrote %s  (%d top-level keys)" % (out_path, len(gen)))
        if args.check:
            hand_path = os.path.join(CONF, out_file)
            with open(hand_path) as f:
                hand = yaml.safe_load(f)
            diffs = deep_diff(hand, gen)
            if diffs:
                failed = True
                print("  STRUCTURAL DIFF vs %s: %d difference(s)" % (hand_path, len(diffs)))
                for d in diffs:
                    print("    " + d)
            else:
                print("  structural diff vs %s: EMPTY (identical parsed content)" % hand_path)
    if args.check:
        print("CHECK %s" % ("FAILED" if failed else "PASSED: all 3 files structurally identical"))
        return 1 if failed else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
