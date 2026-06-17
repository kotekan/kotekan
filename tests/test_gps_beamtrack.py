import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python", "scripts"))
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


def test_attach_altaz_with_local_tle(tmp_path):
    """If skyfield is available, alt/az is computed for PRNs present in the TLE
    set and NaN for those absent. Uses a local TLE file (no network)."""
    pytest.importorskip("skyfield")
    # A real archived GPS TLE (PRN 11). Value correctness isn't asserted -- only
    # that the pipeline runs and fills finite alt/az for the matched PRN.
    tle = (
        "GPS BIIR-3  (PRN 11)\n"
        "1 25933U 99055A   20001.50000000  .00000000  00000-0  00000+0 0  9990\n"
        "2 25933  51.8000  10.0000 0150000  60.0000 300.0000  2.00560000  10000\n"
    )
    tle_path = str(tmp_path / "gps.tle")
    open(tle_path, "w").write(tle)

    recs = np.array(
        [(11, 0, 0, 1, 0, 0, 30, 1, 0, 1.70e9), (7, 0, 0, 1, 0, 0, 30, 1, 0, 1.70e9)],
        dtype=[(f, "<f4") for f in gb.FIELDS] + [("utc", "<f8")],
    )
    alt, az = gb.attach_altaz(recs, lat=43.66, lon=-79.40, alt_m=100.0, tle_source=tle_path)
    assert np.isfinite(alt[0]) and -90.0 <= alt[0] <= 90.0
    assert np.isnan(alt[1])  # PRN 7 absent from the TLE set
