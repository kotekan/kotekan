#!/usr/bin/env python3
"""Write the metadata bundle a vis-capture needs to be interpreted without this repo.

    viscap_metadata.py CAPTURE_DIR

CAPTURE_DIR holds <node>/ subdirectories with the visctl series and configs/ with the node
configs (chord_gnss_cx*_multi.yaml, chord_gnss_node.yaml) snapshotted at capture time.

Produces <dir>/metadata/:
  timing.json     sample clock (3.2 GS/s), hop/record/frame geometry, utc0, the captured
                  seq windows (from the first node's gnss0 visctl series) and their UTC.
  elements.csv    element_id -> dish index, polarization, dish label, grid indices, position
                  (m) along the grid axes (grid_x_axis ~ East, grid_y_axis ~ North, see
                  station.json) -- for every element in array.live_element_ranges. Station id
                  convention: dish = e % num_dishes, pol = e // num_dishes. Plus the array
                  epoch covering the capture start and the element's beam-cube index in it
                  (gnss_arraymap; both empty when no epoch is recorded for that time).
  channels.csv    (node, gpu, chain) -> chan_index, absolute freq_id and freq_MHz: every
                  cudaGnssInject's channels in its own order (its visctl series' channel
                  axis) and, as chain 'cap', the capture gather's (the capvistiles channel
                  axis: the union of the captured chains' channels).
  slots.csv       (node, gpu, chain) -> the PRN in each correlator slot, its signal and the
                  chain's synth lane base, from every cudaGnssInject in the node configs.
  gps_l5_capability.csv  GPS PRN -> block, and whether that block carries L5 (IGS SINEX).
  station.json    the broker's receiver coordinates (the dishes' effective phase centre,
                  common lat/lon/alt in --chains), the node configs' array origin
                  (origin_itrs_*) with the difference stated, and the boresight sat_azel.csv
                  measures sep_deg from.
  sat_azel.csv    az/el/range/range-rate, separation from boresight and the ephemeris health
                  word for every satellite with a broadcast record within 6 h (unhealthy ones
                  still radiate, so the broker's health gate is not applied), every --step s
                  across each window, at the broker's coordinates.
  brdc/           the RINEX navigation files used.

The timing, elements, channels and slots come from the capture itself (visctl headers and the
snapshotted configs); the station from --chains (the repo's chains file by default), the GPS
blocks from the current IGS SINEX cache, and the ephemeris from the broker's cached BRDC.
Rerunning regenerates the bundle.
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import shutil
import struct
import sys
from datetime import datetime, timezone

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python", "scripts", "gnss"))
import gnss_arraymap as amap  # noqa: E402
import gnss_ephemeris as eph  # noqa: E402
from gnss_broker import prnmap  # noqa: E402

BORE_AZ_EL = (178.4, 81.60)  # boresight (deg), from the input-averaged beam fit
FS_HZ = 3.2e9
FFT = 16384  # samples per hop
HOPS_PER_RECORD = 2048
HOPS_PER_FRAME = 8192
HDR = np.dtype(
    [
        ("n_rec", "<i4"),
        ("n_prn", "<i4"),
        ("n_chan", "<i4"),
        ("n_jobs", "<i4"),
        ("seq0", "<i8"),
        ("utc0", "<f8"),
    ]
)


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


def _unit(az, el):
    a, e = math.radians(az), math.radians(el)
    return (math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e))


def sep_deg(az, el):
    """Angle from boresight, deg."""
    d = sum(x * y for x, y in zip(_unit(az, el), _unit(*BORE_AZ_EL)))
    return math.degrees(math.acos(max(-1.0, min(1.0, d))))


def predict_all_health_agnostic(E, lat, lon, alt, t_utc, max_age):
    """eph.predict_all without its health gate (which drops every non-BeiDou record with
    health != 0) and without its elevation mask: {(sys, prn): dict}, each satellite's health
    word added from the record predict_all chose (best_eph's nearest toe)."""
    pred = eph.predict_all(
        E, lat, lon, alt, t_utc, mask_deg=-90.0, max_age=max_age, health_gate=False
    )
    t = eph.gpst_of_utc(t_utc)
    for key, v in pred.items():
        v["health"] = eph.best_eph(E[key], t, max_age).get("health", 0)
    return pred


def channel_rows(node, cfg):
    """channels.csv rows [node, gpu, chain, chan_index, freq_id, freq_MHz] of one node config:
    every cudaGnssInject's channels in its own order (the channel axis of its visctl series),
    then, where the correlator has the capture gather, chain 'cap' with the capvistiles channel
    axis (gather_cap_channels: local channel indices, the union of the captured chains', each
    named by the freq_id an injector pairs it with in gnss_local_channels)."""
    rows = []
    for name, blk in cfg.items():
        m = re.fullmatch(r"gnss(\d)(?:_\w+)?_n2dual", name)
        if not m or not isinstance(blk, dict):
            continue
        gpu = int(m.group(1))
        fid_of, cap = {}, []
        for c in blk.get("commands", []):
            if c.get("name") == "cudaGnssInject":
                chain = "_".join(c["signal"].split("_")[:2]).lower()
                fids = [int(f) for f in c["channel_ids"]]
                rows += [[node, gpu, chain, i, f] for i, f in enumerate(fids)]
                fid_of.update(zip(c.get("gnss_local_channels", []), fids))
            elif c.get("name") == "cudaCorrelatorDual" and "cap" in c.get(
                "gnss_gathers", []
            ):
                # cudaCorrelatorDual's default for a gather's channels is the whole comb
                cap = c.get("gather_cap_channels", c.get("gnss_local_channels", []))
        rows += [[node, gpu, "cap", i, fid_of[ch]] for i, ch in enumerate(cap)]
    return [r + ["%.7f" % (r[4] * 0.1953125)] for r in rows]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dir")
    ap.add_argument(
        "--chains",
        default=os.path.join(HERE, "..", "..", "config", "gnss_chains_chord.yaml"),
    )
    ap.add_argument(
        "--node-yaml",
        default=None,
        help="node manifest whose array.live_element_ranges selects the elements in "
        "elements.csv (default: <dir>/configs/chord_gnss_node.yaml)",
    )
    ap.add_argument("--brdc-cache", default=os.path.expanduser("~/.cache/kotekan_gps"))
    ap.add_argument(
        "--step",
        type=float,
        default=10.0,
        help="sat_azel.csv time step, seconds (default %(default)s)",
    )
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
        "sample_rate_hz": FS_HZ,
        "samples_per_hop": FFT,
        "hop_s": dt,
        "hops_per_record": HOPS_PER_RECORD,
        "hops_per_frame": HOPS_PER_FRAME,
        "record_s": HOPS_PER_RECORD * dt,
        "frame_s": HOPS_PER_FRAME * dt,
        "seq_units": "F-engine ADC samples since time 0 (seq0, winstart, start_seq, end_seq)",
        "seq_per_record": HOPS_PER_RECORD * FFT,
        "seq_per_frame": step,
        "utc0_unix": utc0,
        "utc0_iso": datetime.fromtimestamp(utc0, tz=timezone.utc).isoformat(),
        "utc_of_seq": "utc0_unix + seq / sample_rate_hz",
        "windows": [
            {
                "start_seq": s,
                "end_seq": e,
                "n_frames": n,
                "duration_s": n * HOPS_PER_FRAME * dt,
                "start_utc": datetime.fromtimestamp(
                    utc0 + s / FS_HZ, tz=timezone.utc
                ).isoformat(),
                "end_utc": datetime.fromtimestamp(
                    utc0 + e / FS_HZ, tz=timezone.utc
                ).isoformat(),
            }
            for s, e, n in wins
        ],
        "source_series": node0 + "_gnss0_visctl",
    }
    json.dump(timing, open(os.path.join(out, "timing.json"), "w"), indent=1)
    print("windows:", [(w["n_frames"], w["start_utc"]) for w in timing["windows"]])

    # ---- elements: live ids -> dish / pol / grid / ENU, plus the array epoch at capture start
    t_first = utc0 + wins[0][0] / FS_HZ
    try:
        ep = amap.at(t_first)
        epoch, cube = ep.name, {e.index: i for i, e in ep.cube_elements()}
    except amap.Refused as ex:
        print(
            "NOTE: elements.csv array_epoch/cube_index left empty:", ex, file=sys.stderr
        )
        epoch, cube = "", {}
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
        w.writerow(
            [
                "element_id",
                "dish_idx",
                "pol",
                "label",
                "grid_x_idx",
                "grid_y_idx",
                "grid_x_m",
                "grid_y_m",
                "array_epoch",
                "cube_index",
            ]
        )
        for e in live:
            di, pol = e % n_dish, e // n_dish
            d = dishes.get(di)
            if d is None:
                w.writerow([e, di, pol, "", "", "", "", "", epoch, cube.get(e, "")])
                continue
            w.writerow(
                [
                    e,
                    di,
                    pol,
                    d.get("label", ""),
                    d["grid_x_idx"],
                    d["grid_y_idx"],
                    "%.4f" % (d["grid_x_idx"] * dx),
                    "%.4f" % (d["grid_y_idx"] * dy),
                    epoch,
                    cube.get(e, ""),
                ]
            )

    # ---- channels and PRN slots per instance. One cudaGnssInject per chain: in its own
    #      gnss<g>_<tag>_n2dual block, or all of a GPU's chains in gnss<g>_n2dual (merged).
    with open(os.path.join(out, "channels.csv"), "w", newline="") as fc, open(
        os.path.join(out, "slots.csv"), "w", newline=""
    ) as fs:
        wc, ws = csv.writer(fc), csv.writer(fs)
        wc.writerow(["node", "gpu", "chain", "chan_index", "freq_id", "freq_MHz"])
        ws.writerow(["node", "gpu", "chain", "slot", "prn", "signal", "lane_base"])
        for g in gen:
            node = os.path.basename(g).split("_")[2]
            cfg = yaml.safe_load(open(g))
            wc.writerows(channel_rows(node, cfg))
            for name, blk in cfg.items():
                m = re.fullmatch(r"gnss(\d)(?:_\w+)?_n2dual", name)
                if not m or not isinstance(blk, dict):
                    continue
                for c in blk.get("commands", []):
                    if c.get("name") != "cudaGnssInject":
                        continue
                    chain = "_".join(
                        c["signal"].split("_")[:2]
                    ).lower()  # GAL_E5A_Q_CS -> gal_e5a
                    for slot, prn in enumerate(c["prns"]):
                        ws.writerow(
                            [
                                node,
                                int(m.group(1)),
                                chain,
                                slot,
                                int(prn),
                                c["signal"],
                                int(c.get("synth_lane_base", 0)),
                            ]
                        )

    # ---- GPS L5 capability per PRN, from the IGS registry the broker filters on
    blocks = prnmap.gps_prn_blocks()
    if not blocks:
        print(
            "NOTE: GPS block registry unavailable (%s): gps_l5_capability.csv is header-only"
            % prnmap._IGS_SNX,
            file=sys.stderr,
        )
    with open(os.path.join(out, "gps_l5_capability.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prn", "block", "l5"])
        for prn, blk in sorted(blocks.items()):
            w.writerow([prn, blk, int(prnmap.block_carries("GPS_L5", blk))])

    # ---- station
    ch = yaml.safe_load(open(a.chains))
    st = ch["common"]
    lat, lon, alt = float(st["lat"]), float(st["lon"]), float(st["alt"])
    station = {
        "receiver_lat_deg": lat,
        "receiver_lon_deg": lon,
        "receiver_alt_m": alt,
        "receiver_note": "broker point approximation of the live dishes' phase centre "
        "(config/gnss_chains_chord.yaml); sat_azel.csv is computed here",
        "array_origin_lat_deg": float(tel["origin_itrs_lat_deg"]),
        "array_origin_lon_deg": float(tel["origin_itrs_lon_deg"]),
        "array_origin_note": "frame the dish grid offsets in elements.csv are defined against "
        "(node telescope block); a different point from receiver_lat_deg/"
        "receiver_lon_deg",
        "dish_separation_x_m": dx,
        "dish_separation_y_m": dy,
        "dish_coelev_deg": tel.get("dish_coelev_deg"),
        "grid_x_axis": tel.get("grid_x_axis"),
        "grid_y_axis": tel.get("grid_y_axis"),
        "boresight_az_el_deg": list(BORE_AZ_EL),
        "boresight_note": "from the input-averaged beam fit; sat_azel.csv sep_deg is from it",
    }
    json.dump(station, open(os.path.join(out, "station.json"), "w"), indent=1)

    # ---- ephemeris + az/el
    doy = datetime.fromtimestamp(t_first, tz=timezone.utc).timetuple().tm_yday
    yr = datetime.fromtimestamp(t_first, tz=timezone.utc).year
    navs = sorted(
        glob.glob(
            os.path.join(
                a.brdc_cache, "BRDC00WRD_?_%d%03d0000_01D_MN.rnx.gz" % (yr, doy)
            )
        )
    )
    if not navs:
        sys.exit("no BRDC for %d/%03d in %s" % (yr, doy, a.brdc_cache))
    os.makedirs(os.path.join(out, "brdc"), exist_ok=True)
    for n in navs:
        shutil.copy2(n, os.path.join(out, "brdc", os.path.basename(n)))
    E = eph.parse_rinex_nav(navs)
    with open(os.path.join(out, "sat_azel.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "utc_unix",
                "utc_iso",
                "seq",
                "sys",
                "prn",
                "az_deg",
                "el_deg",
                "range_m",
                "range_rate_mps",
                "toe_age_s",
                "sep_deg",
                "health",
            ]
        )
        for s, e_, n in wins:
            dseq = int(round(a.step * FS_HZ))
            seq = s
            while seq < e_ + 1:
                t = utc0 + seq / FS_HZ
                pred = predict_all_health_agnostic(
                    E, lat, lon, alt, t, max_age=6 * 3600
                )
                iso = datetime.fromtimestamp(t, tz=timezone.utc).isoformat()
                for (sy, prn), v in sorted(pred.items()):
                    w.writerow(
                        [
                            ("%.3f" % t),
                            iso,
                            seq,
                            sy,
                            prn,
                            "%.3f" % v["az"],
                            "%.3f" % v["el"],
                            "%.1f" % v["range_m"],
                            "%.3f" % v["range_rate_mps"],
                            "%.0f" % v["toe_age_s"],
                            "%.3f" % sep_deg(v["az"], v["el"]),
                            "%g" % v["health"],
                        ]
                    )
                seq += dseq
    print(
        "wrote",
        out,
        "with",
        len(navs),
        "BRDC file(s), %d live elements, %d configs" % (len(live), len(gen)),
    )


if __name__ == "__main__":
    main()
