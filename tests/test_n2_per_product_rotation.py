"""Check fringe rotation against an analytic zenith direction.

Covers CHIME beamformer/cylinder orders with zero polar motion, constant DUT1,
complete EOP coverage, and frame-aligned bins. For delta=ERA_target-ERA_sample,
the local direction is E=cos(lat)*sin(delta) and
N=sin(lat)*cos(lat)*(1-cos(delta)).
"""

from copy import deepcopy

import numpy as np
import pytest
from test_n2_accumulate import chime_tel
from test_n2_per_product_support import (
    BINS,
    FREQ,
    SUBS,
    E,
    F,
    PerProductDump,
    S,
    fixture,
    reorder_reference,
    station_order,
)

from kotekan import runner

# IERS ERA rate in turns per UT1 day, used for the change in ERA.
SIDEREAL_RATE = 1.00273781191135448
LIGHT_SPEED = 299792458.0
BOOT_NS = 1767287750500000000


def independent_expectation(streams, period, spf, scale, input_order="CHIMEBeamformer"):
    rows, cols = np.triu_indices(E)
    tile = (cols // 16) * (cols // 16 + 1) // 2 + rows // 16
    hi, lo = cols // 8, rows // 8
    count_tile = (hi // 8) * (hi // 8 + 1) // 2 + lo // 8
    stations = np.array(station_order(input_order))
    east_m = (stations[:, 1] - 1.5) * 22.0
    north_m = (stations[:, 2] - 7.5) * 0.3048
    latitude = np.deg2rad(50.0)
    integration_s = S * scale * period * 2.56e-6
    phase_span = 0.0
    result = {}
    for b in range(BINS):
        for f in range(F):
            raw, counts, admitted = [], [], []
            for t in range(SUBS):
                frame, sub = divmod(b * SUBS + t, spf)
                matrix = streams["in_buf"][frame].data[sub, f]
                x = matrix[tile, cols % 16, rows % 16, 0].astype(np.complex128)
                x -= 1j * matrix[tile, cols % 16, rows % 16, 1]
                count_matrix = streams["in_counts_buf"][frame].data[sub, f]
                n = count_matrix[count_tile, hi % 8, lo % 8].astype(np.uint64)
                # Midpoint of the bin minus midpoint of this subintegration.
                dt = (SUBS / 2 - t - 0.5) * integration_s
                delta = 2 * np.pi * SIDEREAL_RATE * dt / 86400.0
                pointing_east = np.cos(latitude) * np.sin(delta)
                pointing_north = (
                    np.sin(latitude) * np.cos(latitude) * (1 - np.cos(delta))
                )
                freq_hz = (800.0 - FREQ[f] * 800.0 / 2048.0) * 1e6
                angle = (
                    -2
                    * np.pi
                    * freq_hz
                    / LIGHT_SPEED
                    * (east_m * pointing_east + north_m * pointing_north)
                )
                # Round input phases to single precision before forming baseline products.
                phase = np.exp(1j * angle).astype(np.complex64).astype(np.complex128)
                baseline_phase = phase[rows] * phase[cols].conjugate()
                phase_span = max(
                    phase_span, float(np.max(np.abs(np.angle(baseline_phase))))
                )
                raw.append(np.where(n > 0, x * baseline_phase, 0))
                counts.append(n)
                admitted.append(
                    bool(streams["in_rfiframemask_buf"][frame].data[sub, f])
                )
            n = np.zeros(len(rows), dtype=np.uint64)
            k = np.zeros(len(rows), dtype=np.uint64)
            total = np.zeros(len(rows), dtype=np.complex128)
            q = np.zeros(len(rows))
            for t in range(0, SUBS, 2):
                if not (admitted[t] and admitted[t + 1]):
                    continue
                a, z = counts[t], counts[t + 1]
                n += a + z
                total += raw[t] + raw[t + 1]
                good = (a > 0) & (z > 0)
                k += good
                diff = raw[t][good] / a[good] - raw[t + 1][good] / z[good]
                q[good] += (
                    a[good].astype(float)
                    * z[good]
                    / (a[good] + z[good])
                    * np.abs(diff) ** 2
                )
            mean = np.divide(total, n, out=np.zeros_like(total), where=n > 0)
            weight = np.divide(
                n.astype(float) * k, q, out=np.zeros_like(q), where=(q > 0) & (k > 0)
            )
            result[b + 1, int(FREQ[f])] = {
                "mean": mean,
                "weight": weight,
                "count": n * period,
                "k": k,
                "q": q,
            }
    return result, phase_span


@pytest.mark.parametrize("period,spf", [(1, 2), (1, 1)])
@pytest.mark.parametrize(
    "input_order,output_order",
    [
        ("CHIMEBeamformer", "CHIMEBeamformer"),
        ("CHIMEBeamformer", "CHIMECylinder"),
        ("CHIMECylinder", "CHIMEBeamformer"),
    ],
)
def test_nontrivial_rotation_with_heterogeneous_support(
    tmp_path, period, spf, input_order, output_order
):
    scale = 512
    streams, unrotated = fixture(period, spf, scale)
    expected, phase_span = independent_expectation(
        streams, period, spf, scale, input_order
    )
    if input_order != output_order:
        expected = {
            key: reorder_reference(ref, input_order, output_order)
            for key, ref in expected.items()
        }
        unrotated = [
            reorder_reference(ref, input_order, output_order) for ref in unrotated
        ]
    # The fixture must produce a nonzero rotation.
    assert phase_span > 0.01
    config = {
        "buffer_depth": 3,
        "samples_per_data_set": spf * S * scale,
        "sub_integration_ntime": S * scale,
        "num_local_freq": F,
        "num_elements": E,
        "num_polarizations": 2,
        "num_dishes": E // 2,
        "num_ev": 0,
        "telescope": deepcopy(chime_tel),
        "gps_time": {"frame0_nano": BOOT_NS},
        "eop": {
            "kotekan_update_endpoint": "json",
            "earth_orientation_parameter_table": [
                {
                    "t_inst_ns": BOOT_NS + offset,
                    "delta_UT1_inst": 0.0,
                    "xp_as": 0.0,
                    "yp_as": 0.0,
                }
                for offset in [-3600_000_000_000, 3600_000_000_000]
            ],
        },
    }
    config["telescope"].update(
        frame0_nano=BOOT_NS,
        eop_updatable_config="/eop",
        require_eop=True,
        fatal_eop_out_of_range=True,
        feed_sep_EW=22.0,
        feed_sep_NS=0.3048,
        num_cylinders=4,
    )
    readers = {
        key: runner.ReadChordBuffer(str(tmp_path), frames)
        for key, frames in streams.items()
    }
    for reader in readers.values():
        reader.write()
    output = PerProductDump(
        str(tmp_path), exit_after_n_files=BINS * F, num_elements=E, num_ev=0, num_freq=F
    )
    stage = runner.KotekanStageTester(
        "N2Accumulate",
        {
            "num_freq_per_n2k_frame": F,
            "packet_loss_is_scalar": False,
            "bin_in_ERA": False,
            "num_subintegrations_per_bin": SUBS,
            "variance_mode": "EvenOddPosDef",
            "do_fringestop": True,
            "input_order": input_order,
            "output_order": output_order,
        },
        readers,
        output,
        config,
    )
    stage.run()
    actual = {
        (int(frame.metadata.abs_time_idx), int(frame.metadata.freq_id)): frame
        for frame in output.load()
    }
    assert len(actual) == BINS * F
    max_rotation_effect = 0.0
    for original in unrotated:
        key = original["bin"], original["freq"]
        ref, got = expected[key], actual[key]
        np.testing.assert_array_equal(ref["count"], original["count"])
        np.testing.assert_array_equal(got.valid_fpga_ticks, original["count"])
        np.testing.assert_allclose(got.vis, ref["mean"], rtol=3e-6, atol=2e-7)
        np.testing.assert_allclose(got.weight, ref["weight"], rtol=2e-5, atol=1e-7)
        assert got.metadata.fpga_start_tick == original["seq"]
        assert got.metadata.frame_length_fpga_ticks == original["length"]
        np.testing.assert_array_equal(got.weight[ref["k"] == 0], 0)
        effect = float(np.max(np.abs(got.vis - original["mean"])))
        max_rotation_effect = max(max_rotation_effect, effect)
    assert max_rotation_effect > 0.001
