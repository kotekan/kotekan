#!/usr/bin/env python3
"""Decode a --vis-capture pair (vistiles + visctl rawFileWrite files) into visibilities.

    viscap_read.py /tmp/gnss/viscap --node cx43 --gpu 0 --config config/generated/chord_gnss_cx43_multi.yaml
    viscap_read.py ... --to-h5 cx43_gnss0.h5

FILE FRAMING (rawFileWrite): per frame  [uint32 metadata_size][metadata bytes][frame bytes],
`num_frames_per_file` frames per file, files numbered _NNNNNNN in seq order. Nothing in the
tiles file says its shape -- the ctl file does (FrameHdr.n_rec/n_chan/n_prn) plus the live
element count, which is why the two are read together.

CTL FRAME (lib/stages/gnss/gnssGpuChain.hpp): FrameHdr(48) | winstart int64[16] |
PrnCtl(80)[16 rec][n_prn] | energy f64[4*n_prn*16 jobs][n_chan]. seq0 is the absolute
F-engine ADC sample (3.2 GS/s; one hop = 16384 samples, one frame = 8192 hops = 134217728
samples) of the frame's first hop; utc0 the UTC of sample 0; winstart[r] the absolute sample of
sub-integration r (2048 hops = 33554432 samples apart). energy[(job0+t)*n_chan + c] is the TRUE (pre-quantization)
replica energy of lane row t of that PRN slot over one record.

TILES FRAME: int32 [n_rec][n_chan][n_tile][16][16][2] (re, im), n_tile per channel =
  mixed  : 8 synth rows (ihi 8..15) x nlive16 live columns -> tile (ihi-8)*nlive16 + slot
  AA     : lower triangle over live columns, tile k1*(k1+1)/2 + k2 (k1 >= k2)
Synth lane L = 4*slot + row (row 0 E, 1 P, 2 L, 3 P_HEAD) sits at gi = 128 + L: ihi = gi>>4,
ilo = gi&15. Element e (0..n_live-1, in live-column slot order) is at jhi = e>>4, jlo = e&15.
V_mixed[lane, e] = sum synth * conj(antenna)  (cudaCorrelatorDual: synth on the row side).
V_aa[i, j]       = sum E_i * conj(E_j), i >= j.
The replica lanes were quantized to 4 bits with s = 7 / (3 * sqrt(energy_rec0 / hops_per_record))
frozen at record 0 of the frame; divide V_mixed by s for the tracker's absolute units.

Absolute channel ids and the live element ids are not in the frames: --config pulls
channel_ids from the instance's cudaGnssInject block and array.live_element_ranges from
config/chord_gnss_node.yaml, so every axis written to the h5 is ABSOLUTE (freq_id, element).
"""
import argparse
import glob
import os
import struct
import sys

import numpy as np

HDR = np.dtype([("n_rec", "<i4"), ("n_prn", "<i4"), ("n_chan", "<i4"), ("n_jobs", "<i4"),
                ("seq0", "<i8"), ("utc0", "<f8"), ("n_rows_spec", "<i4"), ("_pad0", "<i4"),
                ("_pad1", "<i8")])
PRNCTL = np.dtype([("run", "u1"), ("reanchored", "u1"), ("prn", "<u2"), ("job0", "<i4"),
                   ("fcar_report", "<f4"), ("n_owned", "<f4"), ("cp_seed", "<f8"),
                   ("f_nco", "<f8"), ("chan_mask", "<u8"), ("ctrim_hz", "<f8"),
                   ("ang0", "<f8"), ("phi_ddop", "<f8"), ("fcar", "<f8"), ("dcyc", "<f8")])
assert HDR.itemsize == 48 and PRNCTL.itemsize == 80
MAX_REC = 16
ROWS = 4


def ctl_frame_bytes(n_prn, n_chan):
    return 48 + 8 * MAX_REC + 80 * MAX_REC * n_prn + 8 * ROWS * n_prn * MAX_REC * n_chan


def raw_frames(paths, frame_bytes=None):
    """Yield (metadata_bytes, frame_bytes) across a numbered rawFileWrite series."""
    for p in paths:
        with open(p, "rb") as f:
            data = f.read()
        off = 0
        while off + 4 <= len(data):
            (ms,) = struct.unpack_from("<I", data, off)
            off += 4
            meta = data[off:off + ms]
            off += ms
            if frame_bytes is None:
                # first ctl frame: size from its own header
                h = np.frombuffer(data, HDR, 1, off)[0]
                frame_bytes = ctl_frame_bytes(int(h["n_prn"]), int(h["n_chan"]))
            if off + frame_bytes > len(data):
                print("  %s: truncated frame at %d, stopping" % (p, off), file=sys.stderr)
                return
            yield meta, data[off:off + frame_bytes]
            off += frame_bytes


def decode_ctl(buf):
    h = np.frombuffer(buf, HDR, 1)[0]
    n_prn, n_chan = int(h["n_prn"]), int(h["n_chan"])
    win = np.frombuffer(buf, "<i8", MAX_REC, 48)
    ctl = np.frombuffer(buf, PRNCTL, MAX_REC * n_prn, 48 + 8 * MAX_REC).reshape(MAX_REC, n_prn)
    energy = np.frombuffer(buf, "<f8", ROWS * n_prn * MAX_REC * n_chan,
                           48 + 8 * MAX_REC + 80 * MAX_REC * n_prn).reshape(-1, n_chan)
    return h, win, ctl, energy


def series(d, node, gpu, tag, kind):
    pat = os.path.join(d, "*%s_gnss%d%s_%s_[0-9]*.raw" % (node, gpu, tag, kind))
    paths = sorted(glob.glob(pat))
    if not paths:
        sys.exit("no files match %s" % pat)
    return paths


def config_axes(config_path, gpu, tag, node_yaml):
    import yaml
    cfg = yaml.safe_load(open(config_path))
    dual = cfg["gnss%d%s_n2dual" % (gpu, tag)]
    inj = next(c for c in dual["commands"] if c.get("name") == "cudaGnssInject")
    corr = next(c for c in dual["commands"] if c.get("name") == "cudaCorrelatorDual")
    hops = int(inj["hops_per_record"])
    ncfg = yaml.safe_load(open(node_yaml))
    elems = []
    for lo, hi in ncfg["array"]["live_element_ranges"]:
        elems += list(range(lo, hi + 1))
    if len(elems) != int(corr["num_live_elements"]):
        sys.exit("live_element_ranges gives %d elements, config says %d" % (
            len(elems), corr["num_live_elements"]))
    return [int(c) for c in inj["channel_ids"]], elems, hops, bool(corr.get("gnss_gather_aa"))


def tile_counts(n_live, has_aa):
    """(n_tile, n_mixed, n_aa, nlive16) for an instance's live-column count."""
    nlive16 = (n_live + 15) // 16
    n_mixed = 8 * nlive16
    n_aa = nlive16 * (nlive16 + 1) // 2 if has_aa else 0
    return n_mixed + n_aa, n_mixed, n_aa, nlive16


def read_pair(ctl_paths, tile_paths, n_tile, n_chan_expect=None, max_frames=0):
    """Read matching ctl/tiles frames -> list of (hdr, winstart, ctl, energy, tiles int32)."""
    tile_it = None
    frames = []
    for i, (_, cbuf) in enumerate(raw_frames(ctl_paths)):
        h, win, ctl, energy = decode_ctl(cbuf)
        n_rec, n_chan = int(h["n_rec"]), int(h["n_chan"])
        if n_chan_expect is not None and n_chan != n_chan_expect:
            sys.exit("ctl says %d channels, config says %d" % (n_chan, n_chan_expect))
        if tile_it is None:
            tile_it = raw_frames(tile_paths, n_rec * n_chan * n_tile * 16 * 16 * 2 * 4)
        try:
            _, tbuf = next(tile_it)
        except StopIteration:
            print("tiles series ended before ctl series (frame %d)" % i, file=sys.stderr)
            break
        tiles = np.frombuffer(tbuf, "<i4").reshape(n_rec, n_chan, n_tile, 16, 16, 2)
        frames.append((h, win[:n_rec].copy(), ctl[:n_rec].copy(), energy, tiles))
        if max_frames and len(frames) >= max_frames:
            break
    if not frames:
        sys.exit("no frames")
    return frames


def decode(frames, n_live, hops, has_aa):
    """Frames -> dict of record-indexed arrays (R = frames x n_rec):
    winstart[R] i8, prn/run/fcar_report/f_nco/fcar/cp_seed[R, n_prn], energy/scale[R, n_prn, 4, n_chan],
    vis_mixed[R, n_chan, n_prn, 4, n_live] c8 (raw tile units; divide by scale),
    vis_aa[R, n_chan, n_live, n_live] c8 (Hermitian-completed) or None."""
    _, n_mixed, n_aa, nlive16 = tile_counts(n_live, has_aa)
    h0 = frames[0][0]
    n_prn, n_chan = int(h0["n_prn"]), int(h0["n_chan"])
    # mixed: lane L=4p+row at gi=128+L -> tile (ihi-8)*nlive16 + jhi, cell [ilo, jlo]
    p_ = np.arange(n_prn)[:, None, None]
    row_ = np.arange(ROWS)[None, :, None]
    e_ = np.arange(n_live)[None, None, :]
    gi = 128 + 4 * p_ + row_
    k_m = ((gi >> 4) - 8) * nlive16 + (e_ >> 4)
    ilo_m, jlo_m = np.broadcast_to(gi & 15, k_m.shape), np.broadcast_to(e_ & 15, k_m.shape)
    # AA: tile n_mixed + k1(k1+1)/2 + k2 holds rows 16k1.., cols 16k2.., k1 >= k2
    i_, j_ = np.tril_indices(n_live)
    k1, k2 = i_ >> 4, j_ >> 4
    k_a = n_mixed + k1 * (k1 + 1) // 2 + k2
    ilo_a, jlo_a = i_ & 15, j_ & 15

    out = {k: [] for k in ("winstart", "prn", "run", "fcar_report", "f_nco", "fcar", "cp_seed",
                           "energy", "scale", "vis_mixed", "vis_aa")}
    for h, wstart, ctl, energy, tiles in frames:
        nr = int(h["n_rec"])
        t = tiles[..., 0].astype(np.float32) + 1j * tiles[..., 1].astype(np.float32)
        out["winstart"].append(wstart[:nr])
        for k in ("prn", "run", "fcar_report", "f_nco", "fcar", "cp_seed"):
            out[k].append(ctl[k])
        e_arr = np.zeros((nr, n_prn, ROWS, n_chan))
        s_arr = np.zeros_like(e_arr)
        for p in range(n_prn):
            for row in range(ROWS):
                j0 = int(ctl[0, p]["job0"]) + row          # quantizer scale frozen at rec 0
                for r in range(nr):
                    if not ctl[r, p]["run"]:
                        continue
                    e_arr[r, p, row] = energy[int(ctl[r, p]["job0"]) + row]
                    rms = np.sqrt(np.maximum(energy[j0], 0) / hops)
                    s_arr[r, p, row] = np.where(rms > 0, 7.0 / (3.0 * rms), 0.0)
        out["energy"].append(e_arr)
        out["scale"].append(s_arr)
        out["vis_mixed"].append(t[:, :, k_m, ilo_m, jlo_m])
        if n_aa:
            aa = np.zeros((nr, n_chan, n_live, n_live), np.complex64)
            v = t[:, :, k_a, ilo_a, jlo_a]
            aa[:, :, i_, j_] = v
            aa[:, :, j_, i_] = np.conj(v)
            out["vis_aa"].append(aa)
    res = {k: (np.concatenate(v) if v else None) for k, v in out.items()}
    return res


def load(d, node, gpu, config, node_yaml, tag="", files=None, max_frames=0):
    """One call for notebooks: (axes dict, decoded dict). `files` = slice into the numbered
    file list (each file holds a few frames), e.g. slice(120, 140)."""
    freq_ids, elems, hops, has_aa = config_axes(config, gpu, tag, node_yaml)
    n_tile = tile_counts(len(elems), has_aa)[0]
    ctl_paths = series(d, node, gpu, tag, "visctl")
    tile_paths = series(d, node, gpu, tag, "vistiles")
    if files is not None:
        ctl_paths, tile_paths = ctl_paths[files], tile_paths[files]
    frames = read_pair(ctl_paths, tile_paths, n_tile, len(freq_ids), max_frames)
    axes = dict(freq_id=np.array(freq_ids, np.int32), element_id=np.array(elems, np.int32),
                hops_per_record=hops, has_aa=has_aa, utc0=float(frames[0][0]["utc0"]),
                sample_rate_hz=3.2e9)
    return axes, decode(frames, len(elems), hops, has_aa)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--node", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--tag", default="", help="chain tag, e.g. _e5a (default: primary)")
    ap.add_argument("--config", required=True, help="the generated node config (channel_ids)")
    ap.add_argument("--node-yaml", default=os.path.join(os.path.dirname(os.path.abspath(
        __file__)), "..", "..", "config", "chord_gnss_node.yaml"))
    ap.add_argument("--to-h5", metavar="OUT.h5")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    freq_ids, elems, hops, has_aa = config_axes(a.config, a.gpu, a.tag, a.node_yaml)
    n_live = len(elems)
    n_tile, n_mixed, n_aa, _ = tile_counts(n_live, has_aa)
    if not has_aa:
        print("NOTE: this instance was not built with gnss_gather_aa -- no AA (N^2) block",
              file=sys.stderr)
    frames = read_pair(series(a.dir, a.node, a.gpu, a.tag, "visctl"),
                       series(a.dir, a.node, a.gpu, a.tag, "vistiles"),
                       n_tile, len(freq_ids), a.max_frames)

    h0, hN = frames[0][0], frames[-1][0]
    seq_step = int(frames[1][0]["seq0"] - h0["seq0"]) if len(frames) > 1 else 0
    seqs = np.array([int(f[0]["seq0"]) for f in frames])
    gaps = np.diff(seqs) // seq_step - 1 if seq_step else np.zeros(0, int)
    print("%d frames, seq0 %d .. %d (step %d), utc0 %.6f; %d frames missing inside the span"
          % (len(frames), h0["seq0"], hN["seq0"], seq_step, h0["utc0"], int(gaps.sum())))
    print("n_rec %d, n_chan %d (freq_ids %s), n_prn %d, n_live %d, tiles/chan %d (%d mixed + %d AA)"
          % (h0["n_rec"], h0["n_chan"], freq_ids, h0["n_prn"], n_live, n_tile, n_mixed, n_aa))
    live = [(p, int(c["prn"])) for p, c in enumerate(frames[0][2][0]) if c["run"]]
    print("record 0 live slots (slot, PRN): %s" % live)

    if not a.to_h5:
        return

    import h5py
    n_rec = int(h0["n_rec"])
    d = decode(frames, n_live, hops, has_aa)
    R = len(d["winstart"])
    with h5py.File(a.to_h5, "w") as f:
        f.attrs.update(node=a.node, gpu=a.gpu, tag=a.tag, hops_per_record=hops,
                       n_rec_per_frame=n_rec, seq_step_per_frame=seq_step,
                       utc0_sample0=float(h0["utc0"]), sample_rate_hz=3.2e9,
                       orientation_mixed="V[lane, e] = sum synth_lane * conj(antenna_e)",
                       orientation_aa="V[i, j] = sum E_i * conj(E_j)",
                       note="divide vis_mixed by scale[rec, slot, row, chan] for absolute units")
        f["freq_id"] = np.array(freq_ids, np.int32)
        f["element_id"] = np.array(elems, np.int32)
        for k in ("winstart", "prn", "run", "fcar_report", "f_nco", "fcar", "cp_seed", "energy",
                  "scale"):
            f[k] = d[k]
        f.create_dataset("vis_mixed", data=d["vis_mixed"], compression="lzf",
                         chunks=(n_rec,) + d["vis_mixed"].shape[1:])
        if d["vis_aa"] is not None:
            f.create_dataset("vis_aa", data=d["vis_aa"], compression="lzf",
                             chunks=(n_rec,) + d["vis_aa"].shape[1:])
    print("wrote %s: %d records" % (a.to_h5, R))


if __name__ == "__main__":
    main()
