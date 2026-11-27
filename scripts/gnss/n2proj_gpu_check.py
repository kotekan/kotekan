#!/usr/bin/env python3
"""The N^2 lane projection on the GPU against the CPU path (docs/CHORD_GPU_PROJECTION.md).

    scripts/gnss/n2proj_gpu_check.py --kotekan <build>/kotekan/kotekan
    scripts/gnss/n2proj_gpu_check.py --files-only        # the files of an earlier run

Runs config/ci-tests/gpu_batch/run_n2accum_gnssproject_gpu.yaml (its writers told to keep
going, so this script reads the gauges once every frame is written, then stops kotekan), and
checks:
  - both paths wrote every frame, in the same order, and each record's seq is its frame's;
  - the GPU's record of what it applied names an earlier source frame (lag >= 1), or none;
  - where the GPU applied the same basis as the CPU path (the lanes are constant once on the
    board), every word of the projected frames is within a count (the GPU projects in FP32, the
    CPU path in FP64), with the fraction that differs;
  - where the GPU applied none, its frame is the raw N^2; outside the live channels both paths
    leave the raw N^2;
  - the live blocks: the raw half is the raw N^2, the projected half the GPU's frame, the k in
    the header the record's;
  - the gauges, per freq_id: lane_k and auto_raw_db equal on both stages, auto_proj_db within
    0.001 dB; null_db is reported (the GPU stage starts its EMA a frame later); the command's lag
    and dropped frames are reported.
Exit 0 when every check passes.
"""
import argparse
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CONFIG = os.path.join(
    ROOT, "config", "ci-tests", "gpu_batch", "run_n2accum_gnssproject_gpu.yaml"
)
WORDS = 36 * 16 * 16 * 2  # int32 per channel of an n2k frame

fails = []


def check(ok, what):
    print("  [%s] %s" % ("ok" if ok else "FAIL", what))
    if not ok:
        fails.append(what)


def read_raw(path, frame_size):
    """The frames of a rawFileWrite file (each after its metadata size and metadata)."""
    out = []
    with open(path, "rb") as fh:
        b = fh.read()
    off = 0
    while off + 4 <= len(b):
        msz = int(np.frombuffer(b, "<u4", 1, off)[0])
        off += 4 + msz
        if off + frame_size > len(b):
            break
        out.append(b[off : off + frame_size])
        off += frame_size
    return out


def record(b):
    """(seq, src_seq, k[slots], slot bytes[slots]) of a basis record."""
    (seq,) = np.frombuffer(b, "<i8", 1, 0)
    ver, n, nint, slots, k_slot, slot_bytes, magic = np.frombuffer(b, "<i4", 7, 8)
    (src,) = np.frombuffer(b, "<i8", 1, 40)
    assert ver == 1 and magic == 0x53414250, "not a basis record"
    head = 64 + 4 * n
    sl = [b[head + s * slot_bytes : head + (s + 1) * slot_bytes] for s in range(slots)]
    k = [int(np.frombuffer(x, "<i4", 1, 4)[0]) for x in sl]
    return (
        int(seq),
        int(src),
        k,
        sl,
        [int(np.frombuffer(x, "<i4", 1, 0)[0]) for x in sl],
    )


def metrics(port):
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:%d/metrics" % port, timeout=2
        ) as r:
            return r.read().decode()
    except Exception:
        return None


def gauge(text, name, stage):
    """{freq_id or None: value} of kotekan_gnss_n2project_<name> for one stage."""
    out = {}
    for m in re.finditer(
        r"^kotekan_gnss_n2project_%s\{([^}]*)\} (\S+)" % name, text, re.M
    ):
        lab = m.group(1)
        if 'stage_name="%s"' % stage not in lab:
            continue
        f = re.search(r'freq_id="(\d+)"', lab)
        out[int(f.group(1)) if f else None] = float(m.group(2))
    return out


def run(kotekan, cfg, port, outdir):
    """Run the config with its writers kept going; return the last /metrics once every
    1:1 stream is written and both stages have taken every frame they will see."""
    c = yaml.safe_load(open(cfg))
    n = int(c["frames_to_generate"])
    for k, v in c["writers"].items():
        if isinstance(v, dict):
            v["exit_after_n_files"] = 0
    c["writers"]["base_dir"] = outdir
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(c, fh, sort_keys=False)
        tmp = fh.name
    log = open(os.path.join(outdir, "kotekan.log"), "w")
    p = subprocess.Popen(
        [kotekan, "-b", "127.0.0.1:%d" % port, "--config", tmp], stdout=log, stderr=log
    )
    text, t_end = None, time.time() + 120
    try:
        while time.time() < t_end and p.poll() is None:
            time.sleep(0.5)
            t = metrics(port)
            if not t:
                continue
            text = t
            done = gauge(t, "frames_total", "/gnss_n2_project/proj_cpu").get(None, 0)
            if done >= n:
                # the GPU stage sees every frame but those dropped; give it a moment
                time.sleep(2.0)
                text = metrics(port) or text
                break
    finally:
        if p.poll() is None:
            p.send_signal(signal.SIGINT)
            try:
                p.wait(30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        os.unlink(tmp)
    return text, n


def check_files(outdir, cfg, n_want):
    c = yaml.safe_load(open(cfg))
    fids = list(c["gen_data_voltage"]["manual_freq_ids"])
    nf = len(fids)
    fsz = nf * WORDS * 4
    rec_sz = int(c["gpu_basis_buffer"]["frame_size"])
    live_sz = int(c["n2live_buffer"]["frame_size"])

    def f(name, size):
        return read_raw(os.path.join(outdir, name + "_0000000.raw"), size)

    gpu_f, gpu_r = f("gpu_frames", fsz), f("gpu_records", rec_sz)
    cpu_f, cpu_r = f("cpu_frames", fsz), f("cpu_records", rec_sz)
    raw_f, live_f = f("raw_frames", fsz), f("live_frames", live_sz)
    n = min(len(gpu_f), len(gpu_r), len(cpu_f), len(cpu_r), len(raw_f))
    check(
        n >= n_want,
        "every frame written: %d of %d (gpu %d+%d, cpu %d+%d, raw %d, live %d)"
        % (
            n,
            n_want,
            len(gpu_f),
            len(gpu_r),
            len(cpu_f),
            len(cpu_r),
            len(raw_f),
            len(live_f),
        ),
    )
    G = [np.frombuffer(x, "<i4").reshape(nf, WORDS) for x in gpu_f[:n]]
    C = [np.frombuffer(x, "<i4").reshape(nf, WORDS) for x in cpu_f[:n]]
    R = [np.frombuffer(x, "<i4").reshape(nf, WORDS) for x in raw_f[:n]]
    gr = [record(x) for x in gpu_r[:n]]
    cr = [record(x) for x in cpu_r[:n]]
    spd = int(c["samples_per_data_set"])
    seqs = [r[0] for r in cr]
    check(
        [r[0] for r in gr] == seqs
        and all(b - a == spd for a, b in zip(seqs, seqs[1:])),
        "both paths' records follow the frames, one per %d samples" % spd,
    )
    check(
        all(r[1] == r[0] for r in cr),
        "the CPU path's record: built from the frame it applies to",
    )
    lags = [(r[0] - r[1]) // spd if r[1] >= 0 else -1 for r in gr]
    check(
        all(l == -1 or l >= 1 for l in lags) and all(r[1] < r[0] for r in gr),
        "the GPU's record: the basis built from an earlier frame (lags %s)"
        % sorted(set(lags)),
    )
    slot_ch = [fids.index(x) for x in gr[0][4]]
    other = [ch for ch in range(nf) if ch not in slot_ch]
    same = close = none = none_ok = 0
    n_diff = n_words = max_d = 0
    for i in range(n):
        if gr[i][3] == cr[i][3]:
            same += 1
            d = np.abs(G[i].astype(np.int64) - C[i])
            close += d.max() <= 1
            n_diff += np.count_nonzero(d[slot_ch])
            n_words += d[slot_ch].size
            max_d = max(max_d, int(d.max()))
        if max(gr[i][2]) == 0:
            none += 1
            none_ok += np.array_equal(G[i], R[i])
    check(
        same >= n // 2 and close == same,
        "%d of %d frames had the same basis on both paths; within a count on %d (%d of %d "
        "live-channel words differ, %.3f%%, max |d| %d)"
        % (same, n, close, n_diff, n_words, 100.0 * n_diff / max(n_words, 1), max_d),
    )
    check(
        none_ok == none,
        "%d frames with no basis applied: the raw N^2 on %d" % (none, none_ok),
    )
    check(
        all(np.array_equal(G[i][other], R[i][other]) for i in range(n))
        and all(np.array_equal(C[i][other], R[i][other]) for i in range(n)),
        "the %d channels outside the live list are the raw N^2 on both paths"
        % len(other),
    )
    by_seq = {r[0]: i for i, r in enumerate(gr)}
    good = checked = 0
    for b in live_f:
        seq, src = np.frombuffer(b, "<i8", 2, 0)
        ver, S, words, magic = np.frombuffer(b, "<i4", 4, 16)
        k = list(np.frombuffer(b, "<i4", S, 64))
        head = 64 + (4 * S + 63) // 64 * 64
        d = np.frombuffer(b, "<i4", 2 * S * words, head).reshape(2, S, words)
        i = by_seq.get(int(seq))
        if i is None or ver != 1 or magic != 0x4556494C:
            continue
        checked += 1
        good += (
            np.array_equal(d[0], R[i][slot_ch])
            and np.array_equal(d[1], G[i][slot_ch])
            and k == gr[i][2]
            and int(src) == gr[i][1]
        )
    check(
        checked > 0 and good == checked,
        "the live blocks: raw, projected and k as the frames and the record (%d of %d frames)"
        % (good, checked),
    )
    return slot_ch, [fids[ch] for ch in slot_ch]


def check_gauges(text, live_fids, n_want):
    if not text:
        check(False, "the gauges were read")
        return
    gpu, cpu = "/gnss_n2_project/proj_gpu", "/gnss_n2_project/proj_cpu"
    fr_g = gauge(text, "frames_total", gpu).get(None, 0)
    fr_c = gauge(text, "frames_total", cpu).get(None, 0)
    ms_g = gauge(text, "frame_ms", gpu).get(None, 0)
    ms_c = gauge(text, "frame_ms", cpu).get(None, 0)
    lag = re.search(
        r"^kotekan_gnss_n2project_gpu_basis_lag_frames\{[^}]*\} (\S+)", text, re.M
    )
    drop = re.search(
        r"^kotekan_gnss_n2project_gpu_live_dropped_total\{[^}]*\} (\S+)", text, re.M
    )
    print(
        "  frames: GPU stage %d, CPU stage %d; frame_ms GPU stage %.2f, CPU stage %.2f; the "
        "command's lag %s, live frames dropped %s"
        % (fr_g, fr_c, ms_g, ms_c, lag and lag.group(1), drop and drop.group(1))
    )
    check(
        lag is not None and float(lag.group(1)) >= 1,
        "the command's lag gauge is 1 or more",
    )
    check(drop is not None, "the command's dropped-frame counter is served")
    lk = [gauge(text, "lane_k", s) for s in (gpu, cpu)]
    check(
        all(lk[0].get(f) == lk[1].get(f) and lk[0].get(f, 0) > 0 for f in live_fids),
        "lane_k equal on both stages: %s" % [lk[0].get(f) for f in live_fids],
    )
    if fr_g < n_want or fr_c < n_want:
        print("  (a stage saw fewer frames: the auto gauges are not compared)")
        return
    ar = [gauge(text, "auto_raw_db", s) for s in (gpu, cpu)]
    ap = [gauge(text, "auto_proj_db", s) for s in (gpu, cpu)]
    nd = [gauge(text, "null_db", s) for s in (gpu, cpu)]
    check(
        all(ar[0][f] == ar[1][f] for f in live_fids),
        "auto_raw_db equal on both stages",
    )
    d_ap = max(abs(ap[0][f] - ap[1][f]) for f in live_fids)
    check(d_ap < 1e-3, "auto_proj_db within 0.001 dB (%.2g)" % d_ap)
    print(
        "  null_db GPU / CPU stage: %s"
        % ", ".join(
            "%d %.2f/%.2f" % (f, nd[0].get(f, 0), nd[1].get(f, 0))
            for f in live_fids
            if f in nd[1]
        )
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--kotekan", default=os.path.join(ROOT, "build", "kotekan", "kotekan")
    )
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--port", type=int, default=12148)
    ap.add_argument("--dir", default="/tmp/n2proj_gpu_ci")
    ap.add_argument("--files-only", action="store_true")
    a = ap.parse_args()
    text = None
    n = int(yaml.safe_load(open(a.config))["frames_to_generate"])
    if not a.files_only:
        os.makedirs(a.dir, exist_ok=True)
        for x in os.listdir(a.dir):
            if x.endswith(".raw"):
                os.unlink(os.path.join(a.dir, x))
        print("running %s" % a.config)
        text, n = run(a.kotekan, a.config, a.port, a.dir)
    print("files in %s" % a.dir)
    _, live_fids = check_files(a.dir, a.config, n)
    if not a.files_only:
        print("gauges")
        check_gauges(text, live_fids, n)
    print("\n%d FAILED" % len(fails) if fails else "\nall passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
