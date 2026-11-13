#!/usr/bin/env python3
"""Per-row equivalence harness for a GNSS GPU half (cudaGnssInject + cudaCorrelatorDual).

    scripts/gnss/gpu_rows_equiv.py extract config/generated/chord_gnss_<node>_multi.yaml X.json
    scripts/gnss/gpu_rows_equiv.py config X.json OUT.yaml [--chains gps_l5 gal_e5a | all]
    scripts/gnss/gpu_rows_equiv.py run KOTEKAN CONFIG.yaml OUTDIR [--warm 40] [--frames 8]
    scripts/gnss/gpu_rows_equiv.py compare A/digests.json B/digests.json [--only ROW]

extract writes a kotekan config fragment: the node config's root scalars, its gnss<g>_n2dual
block, and the host buffers that block names with their metadata pools, verbatim.
tests/boost/test_gnss_tile_selection reads it through GNSS_ROWS_CONFIG=X.json.

config writes the harness config from an extract (config/ci-tests/gpu_batch/
run_gnss_rows_equiv.yaml is its output for the default chains): test voltage into the ring, one
cudaProcess holding the chains' cudaGnssInject commands and the cudaCorrelatorDual with their
gathers, verbatim except that the comb is renumbered to local channels 0..n-1 so the voltage
carries only the comb (--node-channels keeps the node's channels and num_local_freq), copy-outs
of each chain's tiles and control block and of the synth array, and a rawFileWrite per output.
Frames are paced at the hop rate (5.12 us).

run drives a config. It posts each chain's PRN map with its first two slots swapped (a refused map
is recorded in prn_map_refused) and steps --pre-frames frames, so the swap applies before any seed
and the generator has filled its random frames; posts seeds and trims; steps --warm warm-up frames
and --frames digest frames; and digests each row (chain) of each frame: its synth rectangle (its
channels x lanes [synth_lane_base, +num_synth)), its tiles, and its control block up to its last
energy row (the rows beyond n_jobs are never written and hold whatever the GPU memory held). The
seeds are re-posted every 20 s (their TTL is 60 s) and trim_ttl_s is raised to 1e6, so neither
expires. The injects run with psi_sync_build, so the Psi tables a record uses do not depend on the
builder's timing; --async-build runs the builder as a node does (the Psi coverage and cost
baselines), and then only the warm frames of seeds with a constant Doppler reproduce. A digest
frame must be warm, or the run exits 1: every slot that runs at all runs in every record of it and
of the frame before, the data form of cudaGnssInject's psi_dark_hop0 == -1 (a state field, not on
REST; the log lines it gates are reported too). --lever ROW moves one seed of that row. --dop-rate,
--leo-slots/--leo-rate and --reseed-at/--reseed-jump make the Psi coverage scenario; every run
reports its dark slots per frame, over the slots that run at all (or, with --live-from, that ran
in a control run). --scale-r0 adds the check that a slot running in a record after record 0
dequantises, by GnssN2RecordAssemble's rule (scale_r0 restates it, so it changes with it), to the
replica energy the pack reduced from its float synthesis (P lane), both for a slot dark at record
0 and for one whose position among the running slots moved since record 0; the run exits 2 while
either fails. digests.json records the run's arguments and the half's cpu_affinity.

compare reports which (row, kind) digests differ between two runs over their digest frames
(--all-frames: every digested frame). It fails unless both runs have the same rows and frames,
every digest frame is warm, and neither run's digests depend on the builder's timing (an
--async-build run whose seeds ramp or reseed, or any --async-build run under --all-frames); --only
ROW requires exactly that row's digests to differ.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import struct
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request

import numpy as np
import yaml

MAX_REC = 16  # gnss_gpu::MAX_REC
HDR = struct.Struct("<iiiiqdiiq")  # gnss_gpu::FrameHdr, 48 bytes
OFF_PRNCTL = HDR.size + 8 * MAX_REC  # gnss_gpu::off_prnctl
# run, reanchored, prn and job0: the head of gnss_gpu::PrnCtl
PRNCTL = struct.Struct("<BBHi")
PRNCTL_SIZE = 112
ROW_P = 1  # gnss_gpu::ROW_P
VOLTAGE_STAGE = "gen_data_voltage"
HALF_STAGE_RE = re.compile(r"^gnss(\d+)_n2dual$")


# ---- extract ----


def extract(config_path, gpu=0):
    """The root scalars, the gnss<gpu>_n2dual block and its buffers and pools of the node config
    at @p config_path."""
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.CSafeLoader)
    key = f"gnss{gpu}_n2dual"
    if key not in cfg:
        raise KeyError(f"{config_path}: no {key}")
    out = {k: v for k, v in cfg.items() if not isinstance(v, dict)}
    out[key] = cfg[key]
    bufs = list(cfg[key]["in_buffers"].values()) + list(
        cfg[key]["out_buffers"].values()
    )
    for b in bufs:
        out[b] = cfg[b]
        out[cfg[b]["metadata_pool"]] = cfg[cfg[b]["metadata_pool"]]
    out["source"] = os.path.basename(config_path)
    return out


# ---- config ----

DEFAULT_CHAINS = ("gps_l5", "gal_e5a", "glo_l2oc")


def half_key(cfg):
    keys = [k for k in cfg if HALF_STAGE_RE.match(k)]
    if len(keys) != 1:
        raise ValueError(f"expected one gnss<g>_n2dual block, found {keys}")
    return keys[0]


def chain_of(inject, half):
    """The chain name of an inject command: its gnss_ctl_name without 'gnss<g>_' and '_n2ctl'."""
    pre = half.replace("_n2dual", "_")
    name = inject["gnss_ctl_name"]
    if not (name.startswith(pre) and name.endswith("_n2ctl")):
        raise ValueError(f"inject gnss_ctl_name {name} is not {pre}<chain>_n2ctl")
    return name[len(pre) : -len("_n2ctl")]


def make_config(x, chains, n_frames=8, node_channels=False):
    """The harness config (a dict) and its comb for @p chains (or 'all') of the extract @p x.
    The comb is renumbered to local channels 0..n-1 unless @p node_channels."""
    half = half_key(x)
    g = HALF_STAGE_RE.match(half).group(1)
    blk = x[half]
    cmds = blk["commands"]
    injects = [c for c in cmds if c["name"] == "cudaGnssInject"]
    corr = [c for c in cmds if c["name"] == "cudaCorrelatorDual"]
    if len(corr) != 1:
        raise ValueError(f"{half}: expected one cudaCorrelatorDual")
    corr = copy.deepcopy(corr[0])
    have = [chain_of(c, half) for c in injects]
    if chains == ["all"]:
        chains = have
    for ch in chains:
        if ch not in have or ch not in corr["gnss_gathers"]:
            raise ValueError(f"{half} has no chain {ch} (chains: {have})")
    rows = [copy.deepcopy(c) for c in injects if chain_of(c, half) in chains]

    # The comb restricted to the chains' channels.
    gathers = [n for n in corr["gnss_gathers"] if n in chains]
    used = set()
    for r in rows:
        used.update(r["gnss_local_channels"])
    for n in gathers:
        used.update(corr.get(f"gather_{n}_channels", corr["gnss_local_channels"]))
    comb = [c for c in corr["gnss_local_channels"] if c in used]
    remap = {c: (i if not node_channels else c) for i, c in enumerate(comb)}
    n_local = x["num_local_freq"] if node_channels else len(comb)

    for k in [k for k in corr if k.startswith("gather_")]:
        if not any(k.startswith(f"gather_{n}_") for n in gathers):
            del corr[k]
    corr["gnss_gathers"] = gathers
    corr["gnss_local_channels"] = [remap[c] for c in comb]
    for n in gathers:
        k = f"gather_{n}_channels"
        if k in corr:
            corr[k] = [remap[c] for c in corr[k]]
    for r in rows:
        r["gnss_local_channels"] = [remap[c] for c in r["gnss_local_channels"]]
        r["gnss_synth_channels"] = [remap[c] for c in comb]

    # Outputs: each chain's tiles then control block, in inject order, then the synth array.
    host_of = {}
    for c in cmds:
        if c["name"] == "cudaOutputData":
            host_of[c["gpu_mem"]] = (c["out_buf"], blk["out_buffers"][c["out_buf"]])
    cfg = {"type": "config", "log_level": "INFO"}
    for k in ("buffer_depth", "samples_per_data_set", "num_times", "num_polarizations"):
        cfg[k] = x[k]
    cfg["num_dishes"] = x["num_dishes"]
    cfg["num_elements"] = x["num_elements"]
    cfg["num_local_freq"] = n_local
    cfg["frame_arrival_period"] = x["frame_arrival_period"]
    cfg["cpu_affinity"] = [2, 3, 4, 5, 8, 9, 10, 11]
    cfg["telescope"] = {"sampling_rate": 3200.0, "fft_length": 16384, "nyquist_zone": 1}
    ring_in = blk["in_buffers"]["host_voltage_ringbuffer"]
    pools = {x[ring_in]["metadata_pool"]}
    bufs = {}
    outputs = []
    for r in rows:
        ch = chain_of(r, half)
        tiles = host_of[corr[f"gather_{ch}_tiles_name"] + "_buffer"]
        ctl = host_of[r["gnss_ctl_name"]]
        for gpu_mem, (internal, host) in (
            (corr[f"gather_{ch}_tiles_name"] + "_buffer", tiles),
            (r["gnss_ctl_name"], ctl),
        ):
            outputs.append((gpu_mem, internal, host))
            bufs[host] = dict(x[host])
            pools.add(x[host]["metadata_pool"])
    synth_host = f"gnss{g}_synth_buf"
    synth_len = f"samples_per_data_set * {len(comb)} * {corr.get('num_synth', 128)}"
    outputs.append((corr["gnss_synth_name"], f"host_gnss{g}_synth", synth_host))
    for p in sorted(pools):
        cfg[p] = x[p]
    cfg["host_voltage_buffer"] = {
        "kotekan_buffer": "ndarray",
        "num_frames": "buffer_depth",
        "value_type": "int4x2_swapped_withoffset",
        "extents": [
            "samples_per_data_set",
            "num_local_freq",
            "num_polarizations",
            "num_dishes",
        ],
        "metadata_pool": x[ring_in]["metadata_pool"],
    }
    cfg[ring_in] = x[ring_in]
    cfg.update(bufs)
    cfg[synth_host] = {
        "kotekan_buffer": "standard",
        "metadata_pool": x[ring_in]["metadata_pool"],
        "num_frames": "buffer_depth",
        "frame_size": synth_len,
    }
    spd = x["samples_per_data_set"]
    cfg[VOLTAGE_STAGE] = {
        "kotekan_stage": "testDataGen",
        "type": "random_signed_offset",
        "value": 120,
        "reuse_random": True,
        "out_buf": "host_voltage_buffer",
        "name": "E",
        "array_shape": [spd, n_local, 2, 64],
        "dim_name": ["T", "F", "P", "D"],
        "dim_scaling": [1, 1, 1, 1],
        "num_frames": n_frames,
    }
    cfg["run_send_voltage"] = {
        "gpu_0": {
            "kotekan_stage": "cudaProcess",
            "gpu_id": blk["gpu_id"],
            "in_buffers": {"host_voltage_buffer_in": "host_voltage_buffer"},
            "out_buffers": {"host_voltage_ringbuffer_out": ring_in},
            "commands": [
                {
                    "name": "cudaCopyToRingbuffer",
                    "input_size": "samples_per_data_set * num_local_freq * num_elements",
                    "ring_buffer_size": "buffer_depth * input_size",
                    "in_buf": "host_voltage_buffer_in",
                    "signal_buf": "host_voltage_ringbuffer_out",
                    "gpu_mem_output": "voltage_buffer",
                }
            ],
        }
    }
    half_cfg = {
        k: blk[k]
        for k in ("kotekan_stage", "gpu_id", "cuda_stream_base", "num_cuda_streams")
    }
    half_cfg["in_buffers"] = dict(blk["in_buffers"])
    half_cfg["out_buffers"] = {internal: host for _, internal, host in outputs}
    half_cfg["commands"] = (
        rows
        + [corr, {"name": "cudaSyncOutput"}]
        + [
            {"name": "cudaOutputData", "gpu_mem": gpu_mem, "out_buf": internal}
            for gpu_mem, internal, _ in outputs
        ]
    )
    cfg[half] = half_cfg
    for i, (_, _, host) in enumerate(outputs):
        sink = {
            "kotekan_stage": "rawFileWrite",
            "in_buf": host,
            "base_dir": "/tmp/kotekan_gnss_rows_equiv",
            "file_name": host,
            "file_ext": "raw",
            "prefix_hostname": False,
            "num_frames_per_file": 1,
            "allow_ndarray": True,
            "create_base_dir": True,
        }
        if i == len(outputs) - 1:
            sink["exit_after_n_files"] = n_frames
        cfg[host + "_sink"] = sink
    return cfg, comb


CONFIG_TEXT = (
    "Written by scripts/gnss/gpu_rows_equiv.py config from a node config's {half} block: {n} rows "
    "({chains}), their cudaGnssInject commands and cudaCorrelatorDual gathers verbatim, {chan}. "
    "The half runs on the root's cores, not on the node's cpu_affinity {affinity} (run "
    "--half-cores pins it). Each row's tiles and control block and the synth array are copied "
    "out and written to /tmp by rawFileWrite, one frame per file. With no seeds every slot is "
    "dark, so this standalone run checks only that the half builds and runs; gpu_rows_equiv.py "
    "run seeds it and digests each row."
)
RENUMBERED = (
    "except that the comb ({comb}) is renumbered to local channels 0..{nm1}, so the voltage "
    "carries only the comb"
)
NODE_CHANNELS = "on the node's local channels (num_local_freq {n_local})"


class _Dumper(yaml.SafeDumper):
    """Block mappings, flow lists of scalars."""


def _list(dumper, v):
    flow = not any(isinstance(e, (dict, list)) for e in v)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", v, flow_style=flow)


_Dumper.add_representer(list, _list)


def write_config(x, chains, out, n_frames, node_channels):
    cfg, comb = make_config(x, chains, n_frames, node_channels)
    half = half_key(cfg)
    rows = [c for c in cfg[half]["commands"] if c["name"] == "cudaGnssInject"]
    chan = (NODE_CHANNELS if node_channels else RENUMBERED).format(
        comb=" ".join(str(c) for c in comb),
        nm1=len(comb) - 1,
        n_local=cfg["num_local_freq"],
    )
    text = CONFIG_TEXT.format(
        half=half,
        chains=", ".join(chain_of(r, half) for r in rows),
        n=len(rows),
        chan=chan,
        affinity=x[half].get("cpu_affinity", "(none)"),
    )
    rule = "#" * 42
    head = "\n".join(
        ["---", rule, "#", f"# {os.path.basename(out)}", "#"]
        + textwrap.wrap(text, 98, initial_indent="# ", subsequent_indent="# ")
        + ["#", rule, ""]
    )
    body = yaml.dump(
        cfg, Dumper=_Dumper, sort_keys=False, default_flow_style=False, width=100
    )
    with open(out, "w") as f:
        f.write(head + body)


# ---- run ----


def load_config(path):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.CSafeLoader)


class Row:
    """One chain of the half, as the run config describes it."""

    def __init__(self, cmd, half, corr, out_host, sinks):
        self.cmd = cmd
        self.name = chain_of(cmd, half)
        self.n_prn = len(cmd["prns"])
        self.n_chan = cmd["n_channels"]
        self.lane_base = cmd["synth_lane_base"]
        self.num_synth = cmd["num_synth"]
        self.lps = cmd.get("lanes_per_slot", 4)
        self.hpr = cmd["hops_per_record"]
        synth = cmd["gnss_synth_channels"]
        self.chan_idx = [synth.index(c) for c in cmd["gnss_local_channels"]]
        self.tiles_sink = sinks[
            out_host[corr[f"gather_{self.name}_tiles_name"] + "_buffer"]
        ]
        self.ctl_sink = sinks[out_host[cmd["gnss_ctl_name"]]]


def rows_of(cfg):
    half = half_key(cfg)
    cmds = cfg[half]["commands"]
    corr = [c for c in cmds if c["name"] == "cudaCorrelatorDual"][0]
    out_host = {
        c["gpu_mem"]: cfg[half]["out_buffers"][c["out_buf"]]
        for c in cmds
        if c["name"] == "cudaOutputData"
    }
    sinks = {
        v["in_buf"]: v["file_name"]
        for v in cfg.values()
        if isinstance(v, dict) and v.get("kotekan_stage") == "rawFileWrite"
    }
    rows = [
        Row(c, half, corr, out_host, sinks)
        for c in cmds
        if c["name"] == "cudaGnssInject"
    ]
    synth_sink = sinks[out_host[corr["gnss_synth_name"]]]
    return half, corr, rows, synth_sink


def rest(port, path, body=None, timeout=10):
    url = f"http://127.0.0.1:{port}{path}"
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, method="GET" if body is None else "POST"
    )
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        txt = r.read()
    return json.loads(txt) if txt else None


def seed_for(base_seed, row, prn):
    """A deterministic seed for (row, PRN)."""
    rng = random.Random(f"{base_seed}:{row}:{prn}")
    return {
        "prn": prn,
        "doppler_hz": rng.uniform(-4000.0, 4000.0),
        "code_phase_chips": rng.uniform(0.0, 1000.0),
        "ref_hop": 0,
    }


def frame_file(base, sink, i):
    return os.path.join(base, f"{sink}_{i:07d}.raw")


def read_frame(base, sink, i):
    with open(frame_file(base, sink, i), "rb") as f:
        data = f.read()
    (msize,) = struct.unpack_from("<I", data, 0)
    return data[4 + msize :]


def frames_written(base, sink, frame_size, n=0):
    """Frames of @p sink complete on disk (one per file), counting from @p n known complete."""
    while True:
        p = frame_file(base, sink, n)
        try:
            st = os.stat(p)
        except FileNotFoundError:
            return n
        with open(p, "rb") as f:
            hdr = f.read(4)
        if len(hdr) < 4 or st.st_size < 4 + struct.unpack("<I", hdr)[0] + frame_size:
            return n
        n += 1


def parse_ctl(buf):
    """The header fields, PrnCtl run and job0 per (record, slot), and the energy rows of a
    control block."""
    n_rec, n_prn, n_chan, n_jobs, seq0 = HDR.unpack_from(buf, 0)[:5]
    run = np.zeros((n_rec, n_prn), dtype=np.uint8)
    job0 = np.zeros((n_rec, n_prn), dtype=np.int32)
    for r in range(n_rec):
        for p in range(n_prn):
            o = OFF_PRNCTL + (r * n_prn + p) * PRNCTL_SIZE
            run[r, p], _, _, job0[r, p] = PRNCTL.unpack_from(buf, o)
    off_energy = OFF_PRNCTL + PRNCTL_SIZE * MAX_REC * n_prn
    energy = np.frombuffer(buf, dtype="<f8", count=n_jobs * n_chan, offset=off_energy)
    return {
        "n_rec": n_rec,
        "n_jobs": n_jobs,
        "seq0": seq0,
        "run": run,
        "job0": job0,
        "energy": energy.reshape(n_jobs, n_chan),
        "defined": off_energy + 8 * n_jobs * n_chan,
    }


def digest(b):
    return hashlib.blake2b(b, digest_size=12).hexdigest()


def scale_r0(row, ctl, synth_rect):
    """[scale-r0] of one frame: (kind, slot, record, channel, E_deq/E_true - 1) for every running
    (record >= 1, slot) and covering channel. E_deq is the P lane's packed energy undone by
    GnssN2RecordAssemble's scale, from record 0's PrnCtl::job0 rows; E_true is the record's own
    energy row, which the pack reduced from its float synthesis. kind "dark_r0": the slot is dark
    at record 0 (the [scale-r0] gate); "shifted": it runs at record 0 but its position among the
    running slots, which the pack's energy_scale row follows, differs at this record; "control":
    the same position."""
    out = []
    lane = 1 if row.lps == 4 else 0  # gnss_gpu::lane_of_row(lps, ROW_P)
    run = ctl["run"]
    for r in range(1, ctl["n_rec"]):
        for p in range(row.n_prn):
            if not run[r, p]:
                continue
            if not run[0, p]:
                kind = "dark_r0"
            elif run[0, :p].sum() != run[r, :p].sum():
                kind = "shifted"
            else:
                kind = "control"
            q = synth_rect[r * row.hpr : (r + 1) * row.hpr, :, row.lps * p + lane]
            re_ = (q >> 4).astype(np.int32) - 8
            im_ = (q & 15).astype(np.int32) - 8
            eq = (re_ * re_ + im_ * im_).sum(axis=0)
            for c in range(row.n_chan):
                e_true = ctl["energy"][ctl["job0"][r, p] + ROW_P, c]
                if e_true <= 0:
                    continue
                e0 = ctl["energy"][ctl["job0"][0, p] + ROW_P, c]
                s = 7.0 / (3.0 * math.sqrt(e0 / row.hpr)) if e0 > 0 else 0.0
                e_deq = eq[c] / (s * s) if s > 0 else 0.0
                out.append((kind, p, r, c, e_deq / e_true - 1.0))
    return out


def run(a):
    cfg = load_config(a.config)
    half, corr, rows, synth_sink = rows_of(cfg)
    spd = cfg["samples_per_data_set"]
    os.makedirs(a.outdir, exist_ok=True)
    raw = os.path.join(a.outdir, "raw")
    shutil.rmtree(raw, ignore_errors=True)
    os.makedirs(raw)

    n_pre = a.pre_frames
    n_total = n_pre + a.warm + a.frames
    gen = cfg[VOLTAGE_STAGE]
    gen["rest_mode"] = "step"
    gen["num_frames"] = n_total + 1000
    for k, v in cfg.items():
        if isinstance(v, dict) and v.get("kotekan_stage") == "rawFileWrite":
            v["base_dir"] = raw
            v.pop("exit_after_n_files", None)
    if a.no_synth:
        stage = [
            k
            for k, v in cfg.items()
            if isinstance(v, dict) and v.get("file_name") == synth_sink
        ][0]
        drop = {stage, cfg[stage]["in_buf"]}
        cmds = cfg[half]["commands"]
        internal = [
            k for k, v in cfg[half]["out_buffers"].items() if v == cfg[stage]["in_buf"]
        ]
        cfg[half]["commands"] = [
            c
            for c in cmds
            if not (c["name"] == "cudaOutputData" and c["out_buf"] in internal)
        ]
        for k in internal:
            del cfg[half]["out_buffers"][k]
        for k in drop:
            del cfg[k]
        synth_sink = None
    if a.half_cores:
        cfg[half]["cpu_affinity"] = a.half_cores
    for r in rows:
        r.cmd["trim_ttl_s"] = 1.0e6
        if not a.async_build:
            r.cmd["psi_sync_build"] = True
        if a.psi_refresh_hz:
            r.cmd["psi_refresh_hz"] = a.psi_refresh_hz
    run_cfg = os.path.join(a.outdir, "run.yaml")
    with open(run_cfg, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    log = open(os.path.join(a.outdir, "kotekan.log"), "w")
    cmd = (a.launcher.split() if a.launcher else []) + [
        a.kotekan,
        "-c",
        run_cfg,
        "-b",
        f"127.0.0.1:{a.port}",
    ]
    t_start = time.time()
    proc = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    port = a.port
    timing = {}
    try:
        t0 = time.time()
        while True:
            if proc.poll() is not None:
                raise SystemExit(
                    f"kotekan exited {proc.returncode} before REST came up"
                )
            try:
                if rest(port, "/status").get("running"):
                    break
            except OSError:
                pass
            if time.time() - t0 > a.startup_timeout:
                raise SystemExit("kotekan REST did not come up")
            time.sleep(0.5)
        timing["startup_s"] = time.time() - t0
        ep = {r.name: r.cmd for r in rows}
        # The PRN maps, first two slots swapped, applied at the first frame. A refused map is
        # recorded and the row keeps its own.
        refused = {}
        for r in rows:
            m = rest(port, ep[r.name]["get_prns_endpoint"])["prns"]
            m[0], m[1] = m[1], m[0]
            try:
                rest(port, ep[r.name]["set_prns_endpoint"], {"prns": m})
            except urllib.error.HTTPError as e:
                refused[r.name] = f"{e.code}: {e.read().decode(errors='replace')}"
        step = f"/{VOLTAGE_STAGE}/generate_test_data"
        rest(port, step, {"num_frames": n_pre})
        maps = {}
        t0 = time.time()
        while len(maps) < len(rows):
            for r in rows:
                g = rest(port, ep[r.name]["get_prns_endpoint"])
                if not g["pending"] and g["last_hop"] == (n_pre - 1) * spd:
                    maps[r.name] = g["prns"]
            if time.time() - t0 > a.frame_timeout:
                raise SystemExit("the PRN maps did not apply")
            time.sleep(0.2)

        def seeds_of(frame):
            out = {}
            for r in rows:
                ss = []
                for p, prn in enumerate(maps[r.name]):
                    s = seed_for(a.seed, r.name, prn)
                    if a.lever == r.name and p == 3:
                        s["code_phase_chips"] += 0.25
                    rate = a.dop_rate * (1 if prn % 2 else -1)
                    if p < a.leo_slots:
                        rate = a.leo_rate * (1 if prn % 2 else -1)
                    if rate:
                        s["doppler_rate_hz_s"] = rate
                    if a.reseed_at is not None and frame >= a.reseed_at:
                        s["doppler_hz"] += a.reseed_jump
                    ss.append(s)
                out[r.name] = ss
            return out

        def post_seeds(frame):
            for r, ss in zip(rows, [seeds_of(frame)[r.name] for r in rows]):
                rest(port, ep[r.name]["seed_endpoint"], ss)

        post_seeds(0)
        for r in rows:
            rest(
                port,
                ep[r.name]["set_trim_endpoint"],
                [{"prn": prn, "trim_chips": 0.05} for prn in maps[r.name]],
            )
        t_seed = time.time()
        sizes = {}
        for k, v in cfg.items():
            if isinstance(v, dict) and v.get("kotekan_stage") == "rawFileWrite":
                sizes[v["file_name"]] = cfg[v["in_buf"]]["frame_size"]
        n_comb = len(corr["gnss_local_channels"])
        if synth_sink:
            sizes[synth_sink] = spd * n_comb * corr.get("num_synth", 128)

        seen = {k: 0 for k in sizes}

        def wait_frames(n):
            t0 = time.time()
            while True:
                for k, sz in sizes.items():
                    seen[k] = frames_written(raw, k, sz, seen[k])
                done = min(seen.values())
                if done >= n:
                    return
                if proc.poll() is not None:
                    raise SystemExit(
                        f"kotekan exited {proc.returncode} at frame {done}"
                    )
                if time.time() - t0 > a.frame_timeout + n * 0.2:
                    raise SystemExit(f"only {done} of {n} frames written")
                if time.time() - post_seeds.last > 20:
                    post_seeds(max(0, done - n_pre))
                    post_seeds.last = time.time()
                time.sleep(0.1)

        post_seeds.last = t_seed
        t0 = time.time()
        if a.reseed_at is not None:
            rest(port, step, {"num_frames": a.reseed_at})
            wait_frames(n_pre + a.reseed_at)
            post_seeds(a.reseed_at)
            post_seeds.last = time.time()
            rest(port, step, {"num_frames": n_total - n_pre - a.reseed_at})
        else:
            rest(port, step, {"num_frames": n_total - n_pre})
        wait_frames(n_total)
        timing["frames_s"] = time.time() - t0
        if a.profile_out:
            prof = rest(port, f"/gpu_profile/{half}")
            with open(a.profile_out, "w") as f:
                json.dump(prof, f, indent=1)
    finally:
        try:
            rest(port, "/kill", timeout=5)
        except OSError:
            pass
        try:
            proc.wait(timeout=600 if a.launcher else 60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        log.close()
    timing["total_s"] = time.time() - t_start

    # Digest and coverage, frame by frame.
    frames = []
    ever = {r.name: np.zeros(r.n_prn, dtype=bool) for r in rows}
    ctls = []
    for i in range(n_pre, n_total):
        cs = {}
        for r in rows:
            buf = read_frame(raw, r.ctl_sink, i)
            c = parse_ctl(buf)
            c["digest"] = digest(buf[: c["defined"]])
            cs[r.name] = c
            ever[r.name] |= c["run"].any(axis=0)
        ctls.append(cs)
    dark_struct = {k: [int(p) for p in np.where(~v)[0]] for k, v in ever.items()}
    live = ever
    if a.live_from:
        with open(a.live_from) as f:
            ref = json.load(f)["structurally_dark"]
        live = {r.name: np.ones(r.n_prn, dtype=bool) for r in rows}
        for r in rows:
            live[r.name][ref[r.name]] = False
    covered = []
    for cs in ctls:
        covered.append(all(cs[r.name]["run"][:, live[r.name]].all() for r in rows))
    r0 = []
    for j, i in enumerate(range(n_pre, n_total)):
        cs = ctls[j]
        seq0 = {cs[r.name]["seq0"] for r in rows}
        if len(seq0) != 1:
            raise SystemExit(
                f"frame {i}: the rows' control blocks disagree on seq0 {seq0}"
            )
        rec = {
            "index": i,
            "hop0": seq0.pop() // rows[0].cmd["fft_length"],
            "warm": bool(covered[j] and j > 0 and covered[j - 1]),
            "window": i >= n_pre + a.warm,
            "rows": {},
        }
        digest_frame = i >= n_pre + a.warm or a.digest_all
        synth = None
        if synth_sink and (digest_frame or a.scale_r0):
            sb = read_frame(raw, synth_sink, i)
            synth = np.frombuffer(sb, dtype=np.uint8).reshape(spd, n_comb, -1)
        for r in rows:
            c = cs[r.name]
            d = {
                "dark": int((c["run"][:, live[r.name]] == 0).sum()),
                "dark0": int((c["run"][0, live[r.name]] == 0).sum()),
            }
            rect = None
            if synth is not None:
                rect = synth[:, r.chan_idx, r.lane_base : r.lane_base + r.num_synth]
            if digest_frame:
                d["ctl"] = c["digest"]
                d["tiles"] = digest(read_frame(raw, r.tiles_sink, i))
                if rect is not None:
                    d["synth"] = digest(np.ascontiguousarray(rect).tobytes())
            if a.scale_r0 and rect is not None:
                r0 += [(r.name, i) + t for t in scale_r0(r, c, rect)]
            rec["rows"][r.name] = d
        frames.append(rec)

    out = {
        "config": os.path.abspath(a.config),
        "args": vars(a),
        "half_cpu_affinity": cfg[half].get("cpu_affinity", cfg["cpu_affinity"]),
        "psi_sync_build": not a.async_build,
        "seeds_ramp": bool(
            a.dop_rate or (a.leo_slots and a.leo_rate) or a.reseed_at is not None
        ),
        "rows": [r.name for r in rows],
        "maps": maps,
        "prn_map_refused": refused,
        "warm": a.warm,
        "frames_digested": a.frames,
        "structurally_dark": dark_struct,
        "timing": timing,
        "psi_log": psi_log(os.path.join(a.outdir, "kotekan.log")),
        "frames": frames,
    }
    bad = [f["index"] for f in frames if f["window"] and not f["warm"]]
    out["cold_digest_frames"] = bad
    out["coverage"] = coverage(frames, a.reseed_at)
    timing["frame_ms"] = 1e3 * timing["frames_s"] / (n_total - n_pre)
    if a.scale_r0:
        out["scale_r0"] = summarize_r0(r0, a.r0_tol)
    with open(os.path.join(a.outdir, "digests.json"), "w") as f:
        json.dump(out, f, indent=1)
    if not a.keep:
        shutil.rmtree(raw, ignore_errors=True)
    print(
        json.dumps(
            {
                "outdir": a.outdir,
                "timing": timing,
                "cold_digest_frames": bad,
                "first_warm": next((f["index"] for f in frames if f["warm"]), None),
                "coverage": {k: v for k, v in out["coverage"].items() if k != "dark"},
                "scale_r0": out.get("scale_r0", {}).get("verdict"),
            }
        )
    )
    if bad:
        print(f"digest frames not warm: {bad}", file=sys.stderr)
        return 1
    return 2 if a.scale_r0 and out["scale_r0"]["verdict"] != "pass" else 0


def coverage(frames, reseed_at):
    """Dark slot-records (a slot that runs at all, dark in a record) per frame, and per phase
    (from the seeds, and from a reseed): the total, the worst frame and the frames until none
    is dark."""
    dark = [sum(d["dark"] for d in f["rows"].values()) for f in frames]
    phases = [("start", 0, reseed_at if reseed_at is not None else len(dark))]
    if reseed_at is not None:
        phases.append(("reseed", reseed_at, len(dark)))
    out = {"dark": dark}
    for name, lo, hi in phases:
        v = dark[lo:hi]
        clear = next((i for i, x in enumerate(v) if x == 0), None)
        after = v[clear:] if clear is not None else []
        out[name] = {
            "frames": len(v),
            "dark_total": sum(v),
            "dark_max": max(v, default=0),
            "frames_to_clear": clear,
            "dark_mean_after_clear": sum(after) / len(after) if after else None,
            "dark_mean_last_half": (
                sum(v[len(v) // 2 :]) / len(v[len(v) // 2 :]) if v else None
            ),
        }
    return out


def psi_log(path):
    """cudaGnssInject's coverage log lines (the psi_dark_hop0 transitions)."""
    keep = []
    with open(path, errors="replace") as f:
        for line in f:
            if "Psi tables pending" in line or "Psi coverage complete" in line:
                keep.append(line.strip())
    return keep


def summarize_r0(r0, tol):
    """Per kind: cases, failures beyond @p tol, the worst error and examples. The verdict is
    the gate's: every dark_r0 and shifted case within tol, with controls, all within tol, to show
    the tolerance holds."""
    out = {"tolerance": tol}
    for kind in ("dark_r0", "shifted", "control"):
        v = [t for t in r0 if t[2] == kind]
        bad = sorted((t for t in v if abs(t[-1]) > tol), key=lambda t: -abs(t[-1]))
        out[kind] = {
            "n": len(v),
            "n_failing": len(bad),
            "worst_error": max((abs(t[-1]) for t in v), default=None),
            "error_percentiles_1_50_99": (
                [float(x) for x in np.percentile([t[-1] for t in v], [1, 50, 99])]
                if v
                else None
            ),
            "by_frame": {
                f: [
                    float(x)
                    for x in np.percentile(
                        [t[-1] for t in v if t[1] == f], [0, 50, 100]
                    )
                ]
                for f in sorted({t[1] for t in v})
            },
            "examples": [
                dict(
                    zip(
                        ("row", "frame", "slot", "record", "chan", "err"), t[:2] + t[3:]
                    )
                )
                for t in bad[:6]
            ],
        }
    if not out["control"]["n"] or out["control"]["n_failing"]:
        out["verdict"] = "controls fail"
    elif not out["dark_r0"]["n"] + out["shifted"]["n"]:
        out["verdict"] = "no cases"
    elif out["dark_r0"]["n_failing"] or out["shifted"]["n_failing"]:
        out["verdict"] = "fail"
    else:
        out["verdict"] = "pass"
    return out


# ---- compare ----


def digested(run, all_frames):
    """A run's digest frames by hop0, or with @p all_frames every frame it digested."""
    return {
        f["hop0"]: f
        for f in run["frames"]
        if f["window"] or all_frames and "ctl" in next(iter(f["rows"].values()))
    }


def compare(a):
    A = json.load(open(a.a))
    B = json.load(open(a.b))
    fa = digested(A, a.all_frames)
    fb = digested(B, a.all_frames)
    refused = []
    if A["rows"] != B["rows"]:
        refused.append(f"the runs' rows differ: {A['rows']} and {B['rows']}")
    if set(fa) != set(fb):
        refused.append(f"frames in one run only: hop0 {sorted(set(fa) ^ set(fb))}")
    for path, R, f in ((a.a, A, fa), (a.b, B, fb)):
        cold = [x["index"] for x in f.values() if not x["warm"]]
        if cold and not a.all_frames:
            refused.append(f"{path}: digest frames not warm: {cold}")
        if "psi_sync_build" not in R:
            refused.append(f"{path}: no psi_sync_build record (an older run)")
        elif not R["psi_sync_build"] and (a.all_frames or R["seeds_ramp"]):
            refused.append(
                f"{path}: --async-build with "
                + ("warm-up frames" if a.all_frames else "ramped or reseeded seeds")
                + ": its digests depend on the Psi builder's timing"
            )
    if a.only and a.only not in A["rows"]:
        refused.append(f"no row {a.only}")
    hops = sorted(set(fa) & set(fb))
    diff = {}
    n = 0
    for h in hops:
        for row, da in fa[h]["rows"].items():
            db = fb[h]["rows"].get(row, {})
            for kind in ("synth", "tiles", "ctl"):
                if kind not in da:
                    continue
                n += 1
                if da.get(kind) != db.get(kind):
                    diff.setdefault(row, {}).setdefault(kind, []).append(h)
    rows = sorted(fa[hops[0]]["rows"]) if hops else []
    res = {
        "frames_compared": len(hops),
        "digests_compared": n,
        "differing": {r: {k: len(v) for k, v in d.items()} for r, d in diff.items()},
        "identical_rows": [r for r in rows if r not in diff],
        "refused": refused,
    }
    ok = not diff and not refused
    if a.only and not refused and hops:
        kinds = [
            k for k in ("synth", "tiles", "ctl") if k in fa[hops[0]]["rows"][a.only]
        ]
        ok = set(diff) == {a.only} and all(
            len(diff[a.only].get(k, [])) == len(hops) for k in kinds
        )
    res["verdict"] = "pass" if ok and hops else "fail"
    print(json.dumps(res, indent=1))
    return 0 if res["verdict"] == "pass" else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser(
        "extract", help="write a node config's gnss<g>_n2dual block as JSON"
    )
    p.add_argument("config", help="generated node config (config/generated/*.yaml)")
    p.add_argument("out", help="JSON to write")
    p.add_argument("--gpu", type=int, default=0)
    p = sub.add_parser("config", help="write the harness config from an extract")
    p.add_argument("extract", help="extract JSON")
    p.add_argument("out", help="yaml to write")
    p.add_argument("--chains", nargs="+", default=list(DEFAULT_CHAINS), help="or 'all'")
    p.add_argument("--frames", type=int, default=8, help="frames of the standalone run")
    p.add_argument(
        "--node-channels",
        action="store_true",
        help="keep the node's local channels and num_local_freq instead of renumbering the "
        "comb (n2k_dual is instantiated for a few num_local_freq values: 8, 384, 464)",
    )
    p = sub.add_parser("run", help="run a harness config and digest each row")
    p.add_argument("kotekan", help="kotekan binary")
    p.add_argument(
        "config", help="harness config (run_gnss_rows_equiv.yaml or `config` output)"
    )
    p.add_argument("outdir")
    p.add_argument(
        "--pre-frames",
        type=int,
        default=4,
        help="frames before the seeds: the PRN swap applies at the first, and the generator "
        "fills its buffer_depth random frames (later frames reuse them)",
    )
    p.add_argument(
        "--warm", type=int, default=40, help="warm-up frames after the seeds"
    )
    p.add_argument(
        "--frames", type=int, default=8, help="frames digested after the warm-up"
    )
    p.add_argument(
        "--digest-all", action="store_true", help="digest the warm-up frames too"
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--lever", help="row whose slot 3 code phase moves by 0.25 chips")
    p.add_argument(
        "--dop-rate", type=float, default=0.0, help="Doppler rate of every slot, Hz/s"
    )
    p.add_argument(
        "--leo-slots", type=int, default=0, help="slots per row on --leo-rate"
    )
    p.add_argument("--leo-rate", type=float, default=300.0, help="Hz/s")
    p.add_argument(
        "--reseed-at", type=int, help="frame after the seeds of a reseed burst"
    )
    p.add_argument(
        "--reseed-jump", type=float, default=500.0, help="its Doppler jump, Hz"
    )
    p.add_argument(
        "--live-from",
        help="digests.json of a control run: the slots that never ran there (FDMA slots with no "
        "lobe here) are the ones left out of the dark counts, instead of this run's own",
    )
    p.add_argument("--psi-refresh-hz", type=float)
    p.add_argument(
        "--async-build",
        action="store_true",
        help="no psi_sync_build: the Psi builder runs as on a node, so digests of ramped or "
        "reseeded seeds, and of warm-up frames, depend on its timing",
    )
    p.add_argument("--scale-r0", action="store_true", help="run the [scale-r0] check")
    p.add_argument("--r0-tol", type=float, default=0.1)
    p.add_argument("--no-synth", action="store_true", help="no synth copy-out")
    p.add_argument("--half-cores", type=int, nargs="+", help="cpu_affinity of the half")
    p.add_argument("--launcher", help="command prefix, e.g. an nsys profile line")
    p.add_argument("--profile-out", help="write /gpu_profile at the end of the frames")
    p.add_argument("--port", type=int, default=12048)
    p.add_argument("--startup-timeout", type=float, default=300.0)
    p.add_argument("--frame-timeout", type=float, default=120.0)
    p.add_argument("--keep", action="store_true", help="keep the raw frames")
    p = sub.add_parser("compare", help="compare two runs' digests")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--only", help="require exactly this row's digests to differ")
    p.add_argument(
        "--all-frames",
        action="store_true",
        help="the warm-up frames too (--digest-all)",
    )
    a = ap.parse_args()
    if a.cmd == "extract":
        r = extract(a.config, a.gpu)
        with open(a.out, "w") as f:
            json.dump(r, f, indent=1, sort_keys=True)
        print(f"{a.out}: gnss{a.gpu}_n2dual from {r['source']}", file=sys.stderr)
    elif a.cmd == "config":
        with open(a.extract) as f:
            write_config(json.load(f), a.chains, a.out, a.frames, a.node_channels)
    elif a.cmd == "run":
        sys.exit(run(a))
    elif a.cmd == "compare":
        sys.exit(compare(a))


if __name__ == "__main__":
    main()
