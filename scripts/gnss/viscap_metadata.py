#!/usr/bin/env python3
"""Write the metadata bundle a vis-capture needs to be interpreted without this repo.

    viscap_metadata.py /home/kvand/gnss/fixtures/viscap_20260907

Produces <dir>/metadata/:
  timing.json     sample clock (3.2 GS/s), hop/record/frame geometry, utc0, the captured
                  seq windows (from the ctl files of one instance) and their UTC.
  elements.csv    element_id -> dish index, polarization, dish label, grid indices, ENU
                  position (m) along the grid axes (grid_x_axis ~ East, grid_y_axis ~ North,
                  see station.json) -- for the 32 live elements. Station id convention:
                  dish = e % num_dishes, pol = e // num_dishes.
  channels.csv    (node, gpu) -> the ABSOLUTE freq_ids that instance correlated, in MHz.
  station.json    the broker's receiver coordinates (#99-solved phase centre) and the
                  node configs' array origin (origin_itrs_*), with the difference stated.
  sat_azel.csv    az/el/range/range-rate for every satellite in the broadcast ephemeris,
                  every --step s across each window, at the broker's coordinates, via the
                  broker's own gnss_ephemeris.predict_all.
  brdc/           the RINEX navigation files used.

Everything here is derived from the configs snapshotted in <dir>/configs/ (written at
capture time) and the broker's cached BRDC; rerunning it regenerates the bundle.
"""
import argparse
import csv
import glob
import json
import os
import shutil
import struct
import sys
from datetime import datetime, timezone

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python", "scripts", "gnss"))
import gnss_ephemeris as eph  # noqa: E402

FS_HZ = 3.2e9
FFT = 16384                     # samples per hop
HOPS_PER_RECORD = 2048
HOPS_PER_FRAME = 8192
HDR = np.dtype([("n_rec", "<i4"), ("n_prn", "<i4"), ("n_chan", "<i4"), ("n_jobs", "<i4"),
                ("seq0", "<i8"), ("utc0", "<f8")])


def ctl_seq0s(paths):
    """seq0 of every frame in a visctl series (header only; skips the metadata blob)."""
    out, utc0 = [], None
    for p in paths:
        d = open(p, "rb").read()
        off, fb = 0, None
        while off + 4 <= len(d):
            (ms,) = struct.unpack_from("<I", d, off)
            off += 4 + ms
            h = np.frombuffer(d, HDR, 1, off)[0]
            if fb is None:
                n_prn, n_chan = int(h["n_prn"]), int(h["n_chan"])
                fb = 48 + 128 + 80 * 16 * n_prn + 8 * 4 * n_prn * 16 * n_chan
            out.append(int(h["seq0"]))
            utc0 = float(h["utc0"])
            off += fb
    return np.array(out), utc0


def windows(seqs, step):
    """Contiguous runs of frames -> [(first_seq, end_seq_exclusive, n_frames)]."""
    cuts = np.where(np.diff(seqs) != step)[0] + 1
    runs = np.split(seqs, cuts)
    return [(int(r[0]), int(r[-1] + step), len(r)) for r in runs]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--chains", default=os.path.join(HERE, "..", "..", "config",
                                                     "gnss_chains_chord.yaml"))
    ap.add_argument("--node-yaml", default=None,
                    help="default <dir>/configs/chord_gnss_node.yaml")
    ap.add_argument("--brdc-cache", default=os.path.expanduser("~/.cache/kotekan_gps"))
    ap.add_argument("--step", type=float, default=10.0, help="sat_azel cadence, s")
    a = ap.parse_args()
    out = os.path.join(a.dir, "metadata")
    os.makedirs(out, exist_ok=True)
    cfg_dir = os.path.join(a.dir, "configs")
    node_yaml = a.node_yaml or os.path.join(cfg_dir, "chord_gnss_node.yaml")

    # ---- timing, from one instance's ctl series
    ctl = sorted(glob.glob(os.path.join(a.dir, "*", "*_gnss0_visctl_*.raw")))
    if not ctl:
        sys.exit("no *_gnss0_visctl_*.raw under %s/<node>/" % a.dir)
    node0 = os.path.basename(os.path.dirname(ctl[0]))
    ctl = [p for p in ctl if os.path.dirname(p).endswith(node0)]
    seqs, utc0 = ctl_seq0s(ctl)
    step = HOPS_PER_FRAME * FFT
    wins = windows(seqs, step)
    dt = FFT / FS_HZ
    timing = {
        "sample_rate_hz": FS_HZ, "samples_per_hop": FFT, "hop_s": dt,
        "hops_per_record": HOPS_PER_RECORD, "hops_per_frame": HOPS_PER_FRAME,
        "record_s": HOPS_PER_RECORD * dt, "frame_s": HOPS_PER_FRAME * dt,
        "seq_units": "F-engine ADC samples since time 0 (seq0, winstart, start_seq, end_seq)",
        "seq_per_record": HOPS_PER_RECORD * FFT, "seq_per_frame": step,
        "utc0_unix": utc0, "utc0_iso": datetime.fromtimestamp(utc0, tz=timezone.utc).isoformat(),
        "utc_of_seq": "utc0_unix + seq / sample_rate_hz",
        "windows": [{"start_seq": s, "end_seq": e, "n_frames": n,
                     "duration_s": n * HOPS_PER_FRAME * dt,
                     "start_utc": datetime.fromtimestamp(utc0 + s / FS_HZ, tz=timezone.utc).isoformat(),
                     "end_utc": datetime.fromtimestamp(utc0 + e / FS_HZ, tz=timezone.utc).isoformat()}
                    for s, e, n in wins],
        "source_series": node0 + "_gnss0_visctl",
    }
    json.dump(timing, open(os.path.join(out, "timing.json"), "w"), indent=1)
    print("windows:", [(w["n_frames"], w["start_utc"]) for w in timing["windows"]])

    # ---- elements: live ids -> dish / pol / grid / ENU
    ncfg = yaml.safe_load(open(node_yaml))
    live = []
    for lo, hi in ncfg["array"]["live_element_ranges"]:
        live += list(range(lo, hi + 1))
    gen = sorted(glob.glob(os.path.join(cfg_dir, "chord_gnss_cx*_multi.yaml")))
    if not gen:
        sys.exit("no generated configs under %s" % cfg_dir)
    cfg0 = yaml.safe_load(open(gen[0]))
    tel = cfg0["telescope"]
    n_dish = int(cfg0["num_dishes"])
    dishes = {int(d["dish_idx"]): d for d in tel["dish_inputs"]}
    dx, dy = float(tel["dish_separation_x_m"]), float(tel["dish_separation_y_m"])
    with open(os.path.join(out, "elements.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["element_id", "dish_idx", "pol", "label", "grid_x_idx", "grid_y_idx",
                    "grid_x_m", "grid_y_m"])
        for e in live:
            di, pol = e % n_dish, e // n_dish
            d = dishes.get(di)
            if d is None:
                w.writerow([e, di, pol, "", "", "", "", ""])
                continue
            w.writerow([e, di, pol, d.get("label", ""), d["grid_x_idx"], d["grid_y_idx"],
                        "%.4f" % (d["grid_x_idx"] * dx), "%.4f" % (d["grid_y_idx"] * dy)])

    # ---- channels per instance
    with open(os.path.join(out, "channels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node", "gpu", "chan_index", "freq_id", "freq_MHz"])
        for g in gen:
            node = os.path.basename(g).split("_")[2]
            cfg = yaml.safe_load(open(g))
            for gpu in (0, 1):
                dual = cfg.get("gnss%d_n2dual" % gpu)
                if not dual:
                    continue
                inj = next(c for c in dual["commands"] if c.get("name") == "cudaGnssInject")
                for i, fid in enumerate(inj["channel_ids"]):
                    w.writerow([node, gpu, i, int(fid), "%.7f" % (int(fid) * 0.1953125)])

    # ---- station
    ch = yaml.safe_load(open(a.chains))
    st = ch["common"]
    lat, lon, alt = float(st["lat"]), float(st["lon"]), float(st["alt"])
    station = {
        "receiver_lat_deg": lat, "receiver_lon_deg": lon, "receiver_alt_m": alt,
        "receiver_note": "broker point approximation of the live dishes' phase centre "
                         "(config/gnss_chains_chord.yaml); sat_azel.csv is computed here",
        "array_origin_lat_deg": float(tel["origin_itrs_lat_deg"]),
        "array_origin_lon_deg": float(tel["origin_itrs_lon_deg"]),
        "array_origin_note": "frame the dish grid offsets in elements.csv are defined against "
                             "(node telescope block); ~155 m from the receiver point (+132 E, "
                             "+82 N of the dishes)",
        "dish_separation_x_m": dx, "dish_separation_y_m": dy,
        "dish_coelev_deg": tel.get("dish_coelev_deg"),
        "grid_x_axis": tel.get("grid_x_axis"), "grid_y_axis": tel.get("grid_y_axis"),
    }
    json.dump(station, open(os.path.join(out, "station.json"), "w"), indent=1)

    # ---- ephemeris + az/el
    t_first = utc0 + wins[0][0] / FS_HZ
    doy = datetime.fromtimestamp(t_first, tz=timezone.utc).timetuple().tm_yday
    yr = datetime.fromtimestamp(t_first, tz=timezone.utc).year
    navs = sorted(glob.glob(os.path.join(a.brdc_cache, "BRDC00WRD_?_%d%03d0000_01D_MN.rnx.gz" % (yr, doy))))
    if not navs:
        sys.exit("no BRDC for %d/%03d in %s" % (yr, doy, a.brdc_cache))
    os.makedirs(os.path.join(out, "brdc"), exist_ok=True)
    for n in navs:
        shutil.copy2(n, os.path.join(out, "brdc", os.path.basename(n)))
    E = eph.parse_rinex_nav(navs)
    with open(os.path.join(out, "sat_azel.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utc_unix", "utc_iso", "seq", "sys", "prn", "az_deg", "el_deg", "range_m",
                    "range_rate_mps", "toe_age_s"])
        for s, e_, n in wins:
            dseq = int(round(a.step * FS_HZ))
            seq = s
            while seq < e_ + 1:
                t = utc0 + seq / FS_HZ
                pred = eph.predict_all(E, lat, lon, alt, t, mask_deg=-90.0, max_age=6 * 3600)
                iso = datetime.fromtimestamp(t, tz=timezone.utc).isoformat()
                for (sy, prn), v in sorted(pred.items()):
                    w.writerow([("%.3f" % t), iso, seq, sy, prn, "%.3f" % v["az"], "%.3f" % v["el"],
                                "%.1f" % v["range_m"], "%.3f" % v["range_rate_mps"],
                                "%.0f" % v.get("toe_age_s", -1)])
                seq += dseq
    print("wrote", out, "with", len(navs), "BRDC file(s), %d live elements, %d configs" % (
        len(live), len(gen)))


if __name__ == "__main__":
    main()
