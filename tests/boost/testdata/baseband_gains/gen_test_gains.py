#!/usr/bin/env python3
"""Generate a test digital gains HDF5 file in the new format.

Produces test_gains.h5 alongside this script, with the same structure as
a real DigitalGainArchive file from pychfpga.
"""

import os
import numpy as np
import h5py

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(SCRIPT_DIR, "test_gains.h5")

N_UPDATE_TIMES = 1
N_FREQ = 16  # small for testing
N_INPUT = 4  # small for testing

with h5py.File(OUTPUT, "w") as f:
    # Root attributes
    f.attrs["acquisition_name"] = "20250101T000000Z_test_digitalgain"
    f.attrs["archive_version"] = "3.2.0"
    f.attrs["collection_server"] = "test-server"
    f.attrs["instrument_name"] = "test_instrument"
    f.attrs["notes"] = ""
    f.attrs["system_user"] = "test"
    f.attrs["type"] = "DigitalGainArchive"
    f.attrs["version"] = "0.5"

    # gain_coeff: (update_time, freq, input) complex64
    rng = np.random.default_rng(42)
    gain_coeff = (rng.standard_normal((N_UPDATE_TIMES, N_FREQ, N_INPUT))
                  + 1j * rng.standard_normal((N_UPDATE_TIMES, N_FREQ, N_INPUT))).astype(np.complex64)
    f.create_dataset("gain_coeff", data=gain_coeff, chunks=(1, min(N_FREQ, 512), min(N_INPUT, 4)))

    # gain_exp: (update_time, input) int32
    gain_exp = np.full((N_UPDATE_TIMES, N_INPUT), -1, dtype=np.int32)
    f.create_dataset("gain_exp", data=gain_exp, chunks=(1, N_INPUT))

    # compute_time: (update_time, input) float64
    compute_time = np.full((N_UPDATE_TIMES, N_INPUT), 1.7e9, dtype=np.float64)
    f.create_dataset("compute_time", data=compute_time, chunks=(1, N_INPUT))

    # update_id: (update_time,) vlen string
    dt_vlen = h5py.special_dtype(vlen=str)
    update_id = f.create_dataset("update_id", shape=(N_UPDATE_TIMES,), dtype=dt_vlen,
                                 chunks=(N_UPDATE_TIMES,))
    update_id[0] = "digitalgain_20250101T000000.000000Z"

    # index_map group
    idx = f.create_group("index_map")

    # index_map/freq: compound (centre: f8, width: f8)
    freq_dtype = np.dtype([("centre", "<f8"), ("width", "<f8")])
    freq_data = np.array(
        [(i * 0.1953125, 0.1953125) for i in range(N_FREQ)], dtype=freq_dtype
    )
    idx.create_dataset("freq", data=freq_data)

    # index_map/input: compound (chan_id: u2, correlator_input: S32)
    input_dtype = np.dtype([("chan_id", "<u2"), ("correlator_input", "S32")])
    input_data = np.array(
        [(i, f"test_input{i:06d}".encode()) for i in range(N_INPUT)], dtype=input_dtype
    )
    idx.create_dataset("input", data=input_data)

    # index_map/update_time: (update_time,) float64
    ut = idx.create_dataset("update_time", shape=(N_UPDATE_TIMES,), dtype=np.float64,
                            chunks=(N_UPDATE_TIMES,))
    ut[0] = 1.7e9

    # Set axis attributes on datasets
    f["gain_coeff"].attrs["axis"] = ["update_time", "freq", "input"]
    f["gain_exp"].attrs["axis"] = ["update_time", "input"]
    f["compute_time"].attrs["axis"] = ["update_time", "input"]
    f["update_id"].attrs["axis"] = ["update_time"]

print(f"Generated {OUTPUT}")
