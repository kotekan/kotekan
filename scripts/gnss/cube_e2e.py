#!/usr/bin/env python3
"""END-TO-END GATE for the beam-cube recording leg (#110) -- offline, ~15 s, no sky, no GPU.

Drives the SHIPPED CODE, not a model of it: real GnssGpuRecordAssemble with the cube ARMED,
real bufferSend, real bufferRecv, real rawFileWrite, then the real reader
(gnss_cube_read.iter_frames). Only the GPU despread output is synthetic -- rawFileRead replays
gnss_gpu::FrameHdr/PrnCtl/corr/energy frames this script writes.

    python3 scripts/gnss/cube_e2e.py [--binary PATH] [--keep] [--expect-crash]

WHY THIS EXISTS. The leg shipped 2026-09-05 behind three checks -- config validation, a
synthetic-sender wire loopback, a frame-size comparison -- and every one of them tested code
DOWNSTREAM of the bug. The assembler never ran. On sky, every node segfaulted 45-75 s after
start: emit_cube_window marked frames full with no metadata object and bufferSend dereferenced
the null (#110). A gate for a producer must RUN THE PRODUCER; this one does, and with
--expect-crash it demonstrates the failure on the pre-fix binary (the mechanism, not just the
fix: verify-the-mechanism-not-just-the-fix).

WHAT IT CHECKS, each against a specific way the leg can fail silently:

  [0] THE PROCESS SURVIVES ITS FIRST EMIT. The #110 shape: connections up, zero frames, node
      dead. kotekan must still be alive after the whole replay has been sent.

  [1] EVERY WINDOW ARRIVES, IN ORDER, WITH NO DROPS. Window indices consecutive, sender
      dropped_windows 0, the last window (never closed -- no later one opens) is NOT emitted.

  [2] SILENCE IS SILENCE. A slot the producer never ran must carry prn = 0 and all-zero rows --
      never a row of zeros labelled with a PRN, which downstream reads as "no power here".
      A PRN that starts LATE appears only from its first window.

  [3] THE NUMBERS. Each (PRN, bin, element) cell is written with its own address as a value
      (a mark), and the replica energy varies per channel, so the 1/E_c normalisation, the
      per-record accumulation, the element stride and the pad-stripping in the reader are all
      checked by VALUE: coh = n_rec * mark (rot == 1 with f_nco == 0), incoh = n_rec * mark^2,
      w = n_rec, energy = n_rec * E_c, freq_id lo/hi = channel_ids, phi0 == 0,
      n_reanchor == the one fresh acquisition planted.

  [4] THE FRAME DESCRIBES ITSELF. Header maxima (32 PRN x 8 bins) exceed the actual extents
      (4 x 7), so the reader must recover the layout from the frame alone; chain, gpu,
      window_samples, w0/w1 all round-trip.
"""
import argparse
import math
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

K = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(K, "python", "scripts", "gnss"))
sys.path.insert(0, os.path.join(K, "config"))

import gnss_cube_read as cr  # noqa: E402
from gnss_record_layout import record_stride, cube_frame_bytes  # noqa: E402

# Ports deliberately far from anything live (11040-11070, 12048-12052 are in use on the site,
# telem_e2e.py holds 11890/11891/12890). A gate that collides with production is not a gate.
PORT_RECV = 11893
PORT_REST = 12893

# -- the synthetic chain ---------------------------------------------------------------------
HOPS_PER_RECORD = 2048
FFT_LEN = 16384
SAMPLE_RATE = 3.2e9
REC_SAMPLES = HOPS_PER_RECORD * FFT_LEN
REC_PER_FRAME = 4            # records per GPU frame, as on CHORD
REC_PER_WIN = 8              # records per cube window (production is 96; the arithmetic is
                             # identical and this keeps the gate to seconds)
N_WINDOWS = 6                # windows of data replayed; N_WINDOWS-1 are provably complete
N_PRN = 4
PRNS = [3, 7, 19, 25]
DEAD_PRN = 19                # never runs: must be SILENCE on the wire
LATE_PRN = 25                # runs from window LATE_FROM on
LATE_FROM = 3
REANCHOR_AT = (2, 1)         # (window, record) where PRN 3 sees a FRESH acquisition (reanchored 1)
N_CHAN = 7
CHAN_IDS = [5972, 5988, 6004, 6020, 6036, 6052, 6068]
N_ELEM = 4
ROWS_SPEC = 4                # E, P, L, P_HEAD
MAX_REC = 16                 # gnss_gpu::MAX_REC
CUBE_MAX_PRN = 32            # frame maxima > actual extents on purpose: exercises the pad
CUBE_MAX_BINS = 8
CUBE_GPU = 1
CHAIN = "cube_e2e/gate"

# gnss_gpu layout (gnssGpuChain.hpp). Restated here ONLY as sizes; the assembler is the reader.
HDR_FMT = "<iiiiqdiiq"       # n_rec n_prn n_chan n_jobs seq0 utc0 n_rows_spec _pad0 _pad1
assert struct.calcsize(HDR_FMT) == 48
PRNCTL_FMT = "<BBHiffddQddddd"  # run reanchored prn job0 fcar_report n_owned cp_seed f_nco
                                # chan_mask ctrim_hz ang0 phi_ddop fcar dcyc
assert struct.calcsize(PRNCTL_FMT) == 80
OFF_WINSTART = 48
OFF_PRNCTL = OFF_WINSTART + 8 * MAX_REC
OFF_CORR = OFF_PRNCTL + 80 * MAX_REC * N_PRN
MAX_JOBS = ROWS_SPEC * N_PRN * MAX_REC
OFF_ENERGY = OFF_CORR + 16 * MAX_JOBS * N_CHAN * N_ELEM
GPU_FRAME_BYTES = OFF_ENERGY + 8 * MAX_JOBS * N_CHAN


def mark(p, ch, el):
    """The cell's own address as a value: exact in float32 and small enough that
    REC_PER_WIN * mark^2 < 2^24 stays exact too (max 8 * 399^2 = 1.27e6)."""
    return float(100 * (p + 1) + 10 * ch + el + 1)


def energy_of(ch):
    return float(ch + 1)  # per-channel replica energy: 1/E_c is exercised, not identity


def prn_runs(p, w):
    prn = PRNS[p]
    if prn == DEAD_PRN:
        return False
    if prn == LATE_PRN:
        return w >= LATE_FROM
    return True


def write_gpu_file(dirpath):
    """One rawFileRead-format file: [u32 meta_size] then [meta][frame]... (rawFileRead reads the
    metadata size once at the head). The assembler ignores in_buf metadata, but the file format
    needs it present."""
    meta_size = 12  # GnssChanMetadata: int64 sample_seq + uint32 n_chan_scale(0)
    path = os.path.join(dirpath, "gpu_0000000.raw")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<I", meta_size))
        rec_global = 0
        n_frames = N_WINDOWS * REC_PER_WIN // REC_PER_FRAME
        for fi in range(n_frames):
            frame = bytearray(GPU_FRAME_BYTES)
            corr = np.zeros((MAX_JOBS, N_CHAN, N_ELEM, 2), dtype="<f8")
            energy = np.zeros((MAX_JOBS, N_CHAN), dtype="<f8")
            job = 0
            for r in range(REC_PER_FRAME):
                wstart = rec_global * REC_SAMPLES
                w = rec_global // REC_PER_WIN
                rr = rec_global % REC_PER_WIN
                struct.pack_into("<q", frame, OFF_WINSTART + 8 * r, wstart)
                for p in range(N_PRN):
                    run = prn_runs(p, w)
                    job0 = job if run else -1
                    rean = 1 if (run and PRNS[p] == 3 and (w, rr) == REANCHOR_AT) else 0
                    struct.pack_into(PRNCTL_FMT, frame, OFF_PRNCTL + 80 * (r * N_PRN + p),
                                     1 if run else 0, rean, PRNS[p], job0,
                                     0.0, float(N_CHAN), 0.0, 0.0,        # fcar_report n_owned cp_seed f_nco
                                     (1 << N_CHAN) - 1,                    # chan_mask: all channels
                                     0.0, 0.0, 0.0, 0.0, 0.0)              # ctrim ang0 phi_ddop fcar dcyc
                    if not run:
                        continue
                    for t in range(ROWS_SPEC):
                        for ch in range(N_CHAN):
                            ec = energy_of(ch)
                            energy[job0 + t, ch] = ec
                            for el in range(N_ELEM):
                                # PROMPT row carries the mark; the others carry a decoy that
                                # must never reach the cube (it reads row job0+1 only).
                                v = mark(p, ch, el) if t == 1 else -777.0
                                corr[job0 + t, ch, el, 0] = v * ec
                                corr[job0 + t, ch, el, 1] = 0.0
                    job += ROWS_SPEC
                rec_global += 1
            struct.pack_into(HDR_FMT, frame, 0, REC_PER_FRAME, N_PRN, N_CHAN, job,
                             (rec_global - REC_PER_FRAME) * REC_SAMPLES, 1.7e9, ROWS_SPEC, 0, 0)
            frame[OFF_CORR:OFF_CORR + corr.nbytes] = corr.tobytes()
            frame[OFF_ENERGY:OFF_ENERGY + energy.nbytes] = energy.tobytes()
            fh.write(struct.pack("<qI", (rec_global - REC_PER_FRAME) * REC_SAMPLES, 0))
            fh.write(frame)
    return path


def write_config(dirpath, out_dir):
    """One kotekan process, both ends over localhost -- the same node mimic telem_e2e.py uses:
    rawFileRead -> GnssGpuRecordAssemble(cube armed) -> bufferSend -> bufferRecv -> rawFileWrite."""
    cube_fb = cube_frame_bytes(CUBE_MAX_PRN, CUBE_MAX_BINS, N_ELEM)
    rec_fb = N_PRN * record_stride(N_ELEM) * 4
    lines = [
        "type: config",
        "log_level: info",
        "cpu_affinity: [0, 1, 2, 3]",
        "gnss_pool: {kotekan_metadata_pool: GnssChanMetadata, num_metadata_objects: 512}",
        "telescope: {name: ICETelescope, num_polarizations: 1, num_dishes: 1,"
        " query_gps: false, require_gps: false}",
        "rest_server: {port: %d}" % PORT_REST,
        # -- the node side --
        "epl_buf: {kotekan_buffer: standard, metadata_pool: gnss_pool, num_frames: 8,"
        " frame_size: %d}" % GPU_FRAME_BYTES,
        "rec_buf: {kotekan_buffer: standard, metadata_pool: gnss_pool, num_frames: 8,"
        " frame_size: %d}" % rec_fb,
        "cube_buf: {kotekan_buffer: standard, metadata_pool: gnss_pool, num_frames: 16,"
        " frame_size: %d}" % cube_fb,
        "gpu_read: {kotekan_stage: rawFileRead, buf: epl_buf, base_dir: %s, file_name: gpu,"
        " file_ext: raw, prefix_hostname: false, end_interrupt: false}" % dirpath,
        "assemble: {kotekan_stage: GnssGpuRecordAssemble, in_buf: epl_buf, out_buf: rec_buf,"
        " prns: [%s], sample_rate: %r, n_elements: %d, reference_element: 0, elem_sum: false,"
        " channel_ids: [%s], spectrum_window_samples: %d, spectrum_ring_depth: 4,"
        " beam_cube: true, beam_cube_bin_width: 0, beam_cube_window_samples: %d,"
        " beam_cube_ring_depth: 4, beam_cube_max_prn: %d, beam_cube_max_bins: %d,"
        " beam_cube_gpu: %d, beam_cube_chain: %s, cube_buf: cube_buf}"
        % (", ".join(map(str, PRNS)), SAMPLE_RATE, N_ELEM, ", ".join(map(str, CHAN_IDS)),
           REC_PER_WIN * REC_SAMPLES, REC_PER_WIN * REC_SAMPLES, CUBE_MAX_PRN, CUBE_MAX_BINS,
           CUBE_GPU, CHAIN),
        "rec_sink: {kotekan_stage: dropAllFrames, in_buf: rec_buf}",
        "cube_send: {kotekan_stage: bufferSend, buf: cube_buf, server_ip: 127.0.0.1,"
        " server_port: %d, drop_frames: false, use_config_tracker: false}" % PORT_RECV,
        # -- the archiver side (chord_gnss_cubearch.yaml's shape) --
        "arch_buf: {kotekan_buffer: standard, metadata_pool: gnss_pool, num_frames: 64,"
        " frame_size: %d}" % cube_fb,
        "cube_recv: {kotekan_stage: bufferRecv, buf: arch_buf, listen_port: %d, num_threads: 2,"
        " drop_frames: false, use_config_tracker: false}" % PORT_RECV,
        "cube_write: {kotekan_stage: rawFileWrite, in_buf: arch_buf, base_dir: %s,"
        " file_name: gnss_cube, file_ext: raw, prefix_hostname: false, num_frames_per_file: 2,"
        " allow_ndarray: true}" % out_dir,
    ]
    path = os.path.join(dirpath, "cube_e2e.yaml")
    open(path, "w").write("\n".join(lines) + "\n")
    return path


def kotekan_binary():
    host = socket.gethostname().split(".")[0]
    cand = ([os.path.join(K, "build", "kotekan", "kotekan")] if host.startswith("cx")
            else [os.path.join(K, "build_nodpdk", "kotekan", "kotekan")])
    cand.append(os.path.join(K, "build_nodpdk", "kotekan", "kotekan"))
    cand.append(os.path.join(K, "build", "kotekan", "kotekan"))
    for c in cand:
        if os.path.exists(c):
            return c
    sys.exit("no kotekan binary found under %s" % K)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", help="kotekan binary (default: the tree for this host)")
    ap.add_argument("--keep", action="store_true", help="keep the scratch dir and the log")
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--expect-crash", action="store_true",
                    help="PASS iff kotekan dies with SIGSEGV before any window is archived "
                         "(the #110 mechanism, run against a pre-fix binary)")
    a = ap.parse_args()

    d = tempfile.mkdtemp(prefix="cube_e2e-")
    out_dir = os.path.join(d, "arch")
    os.makedirs(out_dir)
    write_gpu_file(d)
    cfg = write_config(d, out_dir)
    log = os.path.join(d, "kotekan.log")
    binary = a.binary or kotekan_binary()
    print("kotekan  %s" % binary)
    print("scratch  %s" % d)

    want_windows = N_WINDOWS - 1
    # ⚠️ --bind-address IS NOT OPTIONAL: kotekan's REST default is 0.0.0.0:12048, PRODUCTION's
    # port on a node.
    proc = subprocess.Popen([binary, "--config", cfg, "--bind-address", "127.0.0.1:%d" % PORT_REST],
                            stdout=open(log, "wb"), stderr=subprocess.STDOUT)
    fails = []
    try:
        t0 = time.time()
        frames = []
        while time.time() - t0 < a.timeout:
            if proc.poll() is not None:
                break
            frames = read_archive(out_dir)
            if len(frames) >= want_windows:
                break
            time.sleep(0.25)
        time.sleep(1.0)  # the tail
        frames = read_archive(out_dir)
        rc = proc.poll()

        # -- [0] the process survives its first emit -------------------------------------------
        if a.expect_crash:
            crashed = rc is not None and rc < 0 and (-rc) in (11,)  # SIGSEGV
            print("exit     %s  windows archived %d" % (rc, len(frames)))
            if not crashed:
                fails.append("expected SIGSEGV (the #110 mechanism) and got rc=%s with %d windows"
                             % (rc, len(frames)))
            elif frames:
                fails.append("crashed, but %d windows were archived first -- not the #110 shape "
                             "(zero frames)" % len(frames))
            return finish(fails, d, log, a.keep, proc, crash_mode=True)
        if rc is not None:
            fails.append("kotekan EXITED rc=%s before the replay finished (SIGSEGV = %s) -- see %s"
                         % (rc, rc == -11, log))
        print("alive    %s  windows archived %d (wanted %d)" % (rc is None, len(frames), want_windows))

        # -- [1] every window, in order, no drops ---------------------------------------------
        idxs = [h["idx"] for h, _ in frames]
        if idxs != list(range(want_windows)):
            fails.append("window indices %s != %s" % (idxs, list(range(want_windows))))
        for h, _ in frames:
            if h["dropped"] != 0:
                fails.append("window %d reports %d sender drops" % (h["idx"], h["dropped"]))

        # -- [2]/[3]/[4] per window ------------------------------------------------------------
        for h, arr in frames:
            w = h["idx"]
            tag = "w%d" % w
            # [4] self-description
            exp = dict(version=2, n_prn=N_PRN, n_bin=N_CHAN, n_elem=N_ELEM, max_prn=CUBE_MAX_PRN,
                       max_bins=CUBE_MAX_BINS, gpu=CUBE_GPU, bin_width=1, chain=CHAIN,
                       win_samples=float(REC_PER_WIN * REC_SAMPLES), sample_rate=SAMPLE_RATE,
                       wstart0=w * REC_PER_WIN * REC_SAMPLES,
                       wstart1=(w * REC_PER_WIN + REC_PER_WIN - 1) * REC_SAMPLES)
            for k, v in exp.items():
                if h[k] != v:
                    fails.append("%s header %s = %r, expected %r" % (tag, k, h[k], v))
            if list(arr["freq_id_lo"]) != CHAN_IDS or list(arr["freq_id_hi"]) != CHAN_IDS:
                fails.append("%s freq_id lo/hi %s/%s != channel_ids"
                             % (tag, list(arr["freq_id_lo"]), list(arr["freq_id_hi"])))
            for p in range(N_PRN):
                runs = prn_runs(p, w)
                if not runs:
                    # [2] silence
                    if arr["prn"][p] != 0 or arr["n_rec"][p] != 0:
                        fails.append("%s slot %d: PRN %d never ran but the frame says prn=%d "
                                     "n_rec=%d" % (tag, p, PRNS[p], arr["prn"][p], arr["n_rec"][p]))
                    for key in ("w", "energy", "coh_re", "coh_im", "incoh"):
                        if np.any(arr[key][p] != 0):
                            fails.append("%s slot %d (silent): %s is not all zero" % (tag, p, key))
                    continue
                # [3] the numbers
                if arr["prn"][p] != PRNS[p]:
                    fails.append("%s slot %d prn %d != %d" % (tag, p, arr["prn"][p], PRNS[p]))
                if arr["n_rec"][p] != REC_PER_WIN:
                    fails.append("%s PRN %d n_rec %d != %d" % (tag, PRNS[p], arr["n_rec"][p], REC_PER_WIN))
                want_nre = 1 if (PRNS[p] == 3 and w == REANCHOR_AT[0]) else 0
                if arr["n_reanchor"][p] != want_nre:
                    fails.append("%s PRN %d n_reanchor %d != %d" % (tag, PRNS[p], arr["n_reanchor"][p], want_nre))
                if arr["phi0"][p] != 0.0:
                    fails.append("%s PRN %d phi0 %r != 0 (f_nco was 0)" % (tag, PRNS[p], arr["phi0"][p]))
                for ch in range(N_CHAN):
                    ec = energy_of(ch)
                    if arr["w"][p, ch] != REC_PER_WIN:
                        fails.append("%s PRN %d ch %d w %r != %d" % (tag, PRNS[p], ch, arr["w"][p, ch], REC_PER_WIN))
                    if not math.isclose(arr["energy"][p, ch], REC_PER_WIN * ec, rel_tol=1e-6):
                        fails.append("%s PRN %d ch %d energy %r != %r" % (tag, PRNS[p], ch, arr["energy"][p, ch], REC_PER_WIN * ec))
                    for el in range(N_ELEM):
                        m = mark(p, ch, el)
                        got = (arr["coh_re"][p, ch, el], arr["coh_im"][p, ch, el], arr["incoh"][p, ch, el])
                        want = (REC_PER_WIN * m, 0.0, REC_PER_WIN * m * m)
                        if not all(math.isclose(g, x, rel_tol=1e-6, abs_tol=1e-9) for g, x in zip(got, want)):
                            fails.append("%s PRN %d ch %d el %d (coh_re, coh_im, incoh) %s != %s -- a "
                                         "stride, normalisation or pad error" % (tag, PRNS[p], ch, el, got, want))
                        if got[0] == -777.0 * REC_PER_WIN:
                            fails.append("%s: a NON-PROMPT row reached the cube" % tag)
        return finish(fails, d, log, a.keep, proc)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()


def read_archive(out_dir):
    frames = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.endswith(".raw"):
            try:
                frames.extend(cr.iter_frames(os.path.join(out_dir, fn)))
            except SystemExit as e:
                frames.append(({"idx": "CORRUPT: %s" % e, "dropped": 0}, None))
    frames = [f for f in frames if f[1] is not None]
    frames.sort(key=lambda f: f[0]["idx"])
    return frames


def finish(fails, d, log, keep, proc, crash_mode=False):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
    if fails:
        print("\nFAIL (%d):" % len(fails))
        for f in fails[:40]:
            print("  - " + f)
        if len(fails) > 40:
            print("  ... %d more" % (len(fails) - 40))
        print("log: %s" % log)
        return 1
    if crash_mode:
        print("\nPASS -- the pre-fix binary SEGFAULTS at its first emit with nothing archived: the "
              "#110 mechanism reproduces offline")
    else:
        print("\nPASS -- assembler, bufferSend/Recv, rawFileWrite and the reader agree end to end")
    if not keep:
        shutil.rmtree(d, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
