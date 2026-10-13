import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "python", "scripts", "gnss")
)
import gps_beamtrack as gb  # noqa: E402


def _write_frames(path, frames):
    """frames: list of (list-of-9-float-tuples, list-of-utc-doubles)."""
    with open(path, "wb") as f:
        for recs, utcs in frames:
            f.write(struct.pack("<I", 0))  # metadata length
            for r, u in zip(recs, utcs):
                f.write(np.array(r, dtype="<f4").tobytes())
                f.write(struct.pack("<d", u))


def test_read_records_roundtrip(tmp_path):
    p = str(tmp_path / "gps_0000000.raw")
    recs = [
        (5, 100.0, 12.3, 4.5, 0.1, 0.2, 30.0, 1.0, 0.7),
        (12, -200.0, 55.0, 3.0, 0.0, 0.0, 8.0, 0.0, 0.0),
    ]
    utcs = [1.7e9 + 0.25, 1.7e9 + 0.25]
    _write_frames(p, [(recs, utcs)])

    assert gb._infer_n_prn(p) == 2
    arr = gb.read_records([p])
    assert len(arr) == 2
    assert list(arr["prn"].astype(int)) == [5, 12]
    assert arr["doppler_hz"][0] == pytest.approx(100.0)
    assert arr["snr"][1] == pytest.approx(8.0)
    assert arr["nav_sign"][0] == pytest.approx(1.0)
    # float64 UTC survives the trailing-two-slots packing
    assert arr["utc"][0] == pytest.approx(1.7e9 + 0.25, abs=1e-3)


def test_read_records_multiframe(tmp_path):
    p = str(tmp_path / "gps_0000000.raw")
    f0 = ([(5, 0, 0, 1, 0, 0, 20, 1, 0)], [1.0])
    f1 = ([(5, 0, 0, 1, 0, 0, 22, 1, 0)], [2.0])
    _write_frames(p, [f0, f1])
    arr = gb.read_records([p], n_prn=1)  # multi-frame needs explicit n_prn
    assert len(arr) == 2
    assert list(arr["utc"]) == [1.0, 2.0]
