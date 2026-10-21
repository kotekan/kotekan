#!/usr/bin/env python3
"""The interface golden's mirrors outside the broker package, each against the golden or the
broker's SignalDef: the generator's layout and band plan (config/gnss_record_layout.py,
config/chord_band_plan.py), and the tables tools keep for themselves (stack_components.sh's
GNSS_CHAIN_RF, gnss_tec_chord's FREQ, gnss_observables' --sys, gnss_cube_read's header sizes,
viscap_read's structs, telem_flap's metadata struct).

    python3 python/scripts/gnss/test_iface_mirrors.py
"""
import importlib.util
import os
import re
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "config"))

import chord_band_plan  # noqa: E402
import gnss_cube_read  # noqa: E402
import gnss_record_layout  # noqa: E402
import gnss_tec_chord  # noqa: E402
from gnss_broker import iface, signals  # noqa: E402

G = iface.golden()
S = G.sections
# The numpy dtype of each document field type, for viscap_read's structs.
_NP = {
    "u8": "u1",
    "u16": "<u2",
    "u32": "<u4",
    "u64": "<u8",
    "i32": "<i4",
    "i64": "<i8",
    "f32": "<f4",
    "f64": "<f8",
}


def load(path):
    """A module of scripts/gnss by path, so its names cannot shadow this directory's."""
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestGeneratorLayout(unittest.TestCase):
    def test_constants(self):
        L = gnss_record_layout
        self.assertEqual(L.record_floats(), S["record.row"]["RECORD_FLOATS"])
        self.assertEqual(L.elem_floats(), S["record.elem"]["ELEM_FLOATS"])
        self.assertEqual(L.chan_floats(), S["record.chan"]["CHAN_FLOATS"])
        self.assertEqual(L.telem_header_bytes(), S["telem"]["TELEM_HEADER_BYTES"])
        self.assertEqual(L.telem_max_chan(), S["telem"]["TELEM_MAX_CHAN"])
        self.assertEqual(L.prnctl_bytes(), S["gpu_chain"]["PrnCtl"]["size"])
        self.assertEqual(L.cube_chain_chars(), S["cube"]["CUBE_CHAIN_CHARS"])
        self.assertEqual(L.cube_header_bytes(), S["cube"]["CUBE_HEADER_BYTES"])

    def test_frame_sizes_are_the_goldens_samples(self):
        # telem_frame_bytes and cube_frame_bytes restate C++ expressions; the golden samples
        # the C++ ones.
        L = gnss_record_layout
        for s in S["telem"]["frame_bytes"]:
            self.assertEqual(
                L.telem_frame_bytes(s["n_rec"], s["n_prn"], s["n_cols"]), s["bytes"], s
            )
        for s in S["telem"]["row_offset"]:
            self.assertEqual(
                (s["r"] * s["n_prn"] + s["p"]) * L.telem_row_floats(s["n_cols"]),
                s["floats"],
                s,
            )
        for s in S["cube"]["frame_bytes"]:
            self.assertEqual(
                L.cube_frame_bytes(s["max_prn"], s["max_bins"], s["n_elem"]),
                s["bytes"],
                s,
            )


class TestBandPlan(unittest.TestCase):
    def test_signal_table_is_the_goldens(self):
        t = chord_band_plan.signal_table()
        self.assertEqual(sorted(t), sorted(S["signals"]))
        for name, row in t.items():
            d = S["signals"][name]
            self.assertEqual(row["carrier_hz"], float(d["carrier_hz"]), name)
            self.assertEqual(row["modulation"], d["mod"], name)

    def test_mirror_band_edges_are_the_cpps_except_time_multiplexed(self):
        # gnssBandPlan.cpp gives a time-multiplexed signal (L2C CM/CL) twice the chip rate as
        # its half-bandwidth; the mirror, and so the generator's covering sets, use the chip
        # rate. Pinned here so a change on either side is seen.
        for name, row in chord_band_plan.signal_table().items():
            kw = {
                "modulation": row["modulation"],
                "boc_m": row["boc_m"],
                "boc_n": row["boc_n"],
            }
            half = chord_band_plan.half_bandwidth_hz(row["chip_rate_hz"], **kw)
            lo, hi = chord_band_plan.occupied_band(
                row["carrier_hz"], row["chip_rate_hz"], 0.0, **kw
            )
            doc = chord_band_plan.band_edges(name)
            if S["signals"][name]["time_multiplexed"]:
                self.assertEqual(half, row["chip_rate_hz"], name)
                self.assertEqual(doc[0], 2 * row["chip_rate_hz"], name)
            else:
                self.assertEqual((half, lo, hi), doc, name)


class TestToolTables(unittest.TestCase):
    def test_stack_rf_table_is_the_signaldef(self):
        # In SignalDef units: L2C's chips are CM chips at 0.5115 Mcps, and L2OC's code_len is
        # the 40920-chip record_cp_mod_chips.
        with open(os.path.join(ROOT, "scripts/gnss/site/stack_components.sh")) as fh:
            m = re.search(r'^GNSS_CHAIN_RF="\n(.*?)^"', fh.read(), re.M | re.S)
        rows = [ln.split() for ln in m.group(1).splitlines() if ln.strip()]
        self.assertGreater(len(rows), 5)
        for chain, sysc, carrier, chip, code in rows:
            s = signals.get(chain)
            self.assertEqual(
                (sysc, float(carrier), float(chip), int(code)),
                (s.constellation, s.carrier_hz, s.chip_rate_hz, s.code_length),
                chain,
            )

    def test_tec_carriers_are_the_signaldef(self):
        for chain, hz in gnss_tec_chord.FREQ.items():
            self.assertEqual(hz, signals.get(chain).carrier_hz, chain)

    def test_observables_sys_choices_are_the_goldens_constellations(self):
        with open(os.path.join(HERE, "gnss_observables.py")) as fh:
            m = re.search(r'"--sys",.*?choices=\(([^)]*)\)', fh.read(), re.S)
        choices = set(re.findall(r'"(\w)"', m.group(1)))
        want = set(signals._SYS[n.split("_")[0]] for n in S["signals"])
        self.assertEqual(choices, want)

    def test_cube_reader_sizes_the_current_version(self):
        c = S["cube"]
        R = gnss_cube_read
        self.assertEqual(max(R.CUBE_FIXED_BYTES), c["CUBE_VERSION"])
        self.assertEqual(R.CUBE_FIXED_BYTES[c["CUBE_VERSION"]], c["CUBE_CHAIN_OFFSET"])
        self.assertEqual(R.CUBE_CHAIN_CHARS, c["CUBE_CHAIN_CHARS"])
        self.assertEqual(R.CUBE_HEADER_BYTES[c["CUBE_VERSION"]], c["CUBE_HEADER_BYTES"])
        for s in c["frame_bytes"]:
            self.assertEqual(
                R.frame_bytes(s["max_prn"], s["max_bins"], s["n_elem"]), s["bytes"], s
            )

    def test_viscap_structs_are_the_gpu_chains(self):
        v = load("scripts/gnss/viscap_read.py")
        gc = S["gpu_chain"]
        for dt, table in ((v.HDR, gc["FrameHdr"]), (v.PRNCTL, gc["PrnCtl"])):
            self.assertEqual(dt.itemsize, table["size"])
            self.assertEqual(list(dt.names), [f[0] for f in table["fields"]])
            for name, off, size, ftype in table["fields"]:
                base, n = re.match(r"^(\w+?)(?:\[(\d+)\])?$", ftype).groups()
                want = np.dtype((_NP[base], (int(n),)) if n else _NP[base])
                self.assertEqual(dt.fields[name][:2], (want, off), name)
                self.assertEqual(want.itemsize, size, name)
        self.assertEqual(v.MAX_REC, gc["MAX_REC"])
        self.assertEqual(v.ROWS, gc["ROWS_PLAIN"])

    def test_telem_flap_metadata_is_the_goldens(self):
        f = load("scripts/gnss/telem_flap.py")
        m = S["metadata"]["GnssChanMetadata"]
        self.assertEqual(f._META.format, iface.struct_format(m))
        ser = m["serialized"]
        self.assertEqual(
            f._META.pack(ser["sample_seq"], 0), bytes.fromhex(ser["bytes"])
        )


if __name__ == "__main__":
    unittest.main()
