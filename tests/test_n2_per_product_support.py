"""Test per-product accumulation with masks applied to individual samples.

Fixtures use 128 inputs, 16 packet groups, and two frequencies. They cross
count and correlation tile boundaries and include disjoint masks with equal
counts. Samples are repeated to exercise large count values.
"""

from copy import deepcopy
from pathlib import Path

import kotekan.telescope as tel
import numpy as np
import pytest
from test_n2_accumulate import chime_tel, make_zeroed_chord_buffer

from kotekan import n2buffer, runner

E, F, S, SUBS, BINS = 128, 2, 128, 8, 3
FREQ = np.array([202, 614], dtype=np.int32)


class PerProductDump(runner.DumpN2Buffer):
    def __init__(self, *args, mode="per_product_v1", **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.buffer_block[self.name]["support_mode"] = mode

    def load(self):
        return n2buffer.N2Buffer.load_files(
            f"{self.output_dir}/*{self.name}*.dump",
            num_elements=self.num_elements,
            num_prod=self.num_prod,
            num_ev=self.num_ev,
            support_mode=self.mode,
        )


def fixture(period, spf, scale):
    nsub = S * scale
    frames = BINS * SUBS // spf
    start = SUBS * nsub * period

    def buf(name, dtype, typename, tail=(), dims=(), scales=(), extra=None):
        return make_zeroed_chord_buffer(
            name,
            dtype,
            typename,
            (spf, F) + tail,
            ("Tc", "F") + dims,
            (nsub, 1) + scales,
            start,
            spf * nsub * period,
            frames,
            freq_ids=FREQ,
            time_downsampling=nsub * period,
            extra_meta=extra,
        )

    streams = {
        "in_buf": buf(
            "n2k_correlation",
            np.int32,
            "int32",
            (36, 16, 16, 2),
            ("DPhi", "DPlo1", "DPlo2", "C"),
            (16, 1, 1, 1),
        ),
        "in_counts_buf": buf(
            "n2k_counts",
            np.int32,
            "int32",
            (3, 8, 8),
            ("D8Phi", "D8Plo1", "D8Plo2"),
            (64, 8, 8),
        ),
        "in_rficounts_buf": buf("RFImask_counts", np.int32, "int32"),
        "in_plcounts_buf": buf("pl_lost_counts_scalar", np.int32, "int32"),
        "in_rfiframemask_buf": buf(
            "RFIFrameMask",
            np.uint8,
            "uint8",
            extra={
                "rfi_frame_excision_enabled": True,
                "rfi_frame_excision_thresholds": np.array(
                    [[3.0, 0.1]], dtype=np.float32
                ),
            },
        ),
    }
    rows, cols = np.triu_indices(E)
    block = (cols // 16) * (cols // 16 + 1) // 2 + rows // 16
    bi, bj = cols % 16, rows % 16
    # Build sample masks and voltages before forming correlation matrices.
    sample, group = np.indices((S, E // 8))
    sample_x, element = np.indices((S, E))
    truth = []
    for b in range(BINS):
        for f in range(F):
            raw, counts, admitted = [], [], []
            for t in range(SUBS):
                present = (sample + 3 * group + t + f) % (5 + group % 4) > 0
                # Equal counts do not imply overlapping samples.
                present[:, 0] = sample[:, 0] < S // 2
                present[:, 1] = sample[:, 1] >= S // 2
                present[:, 2] = False
                # One-sided pairs affect the mean, but not the variance estimate.
                present[:, 3] = t % 2 == 0
                if b == 1:
                    present[:] = False
                keep = (np.arange(S) + 2 * t + f) % 13 != 0
                valid = present & keep[:, None]
                x = ((sample_x + element + t + f) % 5 - 2).astype(np.complex128)
                x += 1j * ((2 * sample_x + 3 * element + t) % 5 - 2)
                x *= np.repeat(valid, 8, axis=1)
                upper = x.T @ x.conjugate()
                joint = valid.astype(np.int64).T @ valid.astype(np.int64)
                c = joint[rows // 8, cols // 8] * scale
                v = upper[rows, cols] * scale
                admit = int(not (b == 2 and t in (3 + f,)))
                raw.append(v)
                counts.append(c)
                admitted.append(admit)
                frame, sub = divmod(b * SUBS + t, spf)
                co = streams["in_buf"][frame].data[sub, f]
                co[block, bi, bj, 0] = v.real
                co[block, bi, bj, 1] = -v.imag
                # Nonzero payload with zero counts must be ignored.
                co[block[c == 0], bi[c == 0], bj[c == 0], :] = 123456
                cc = streams["in_counts_buf"][frame].data[sub, f]
                cc[:] = -123  # Poison unused upper-triangle entries.
                tile = 0
                for hi in range(2):
                    for lo in range(hi + 1):
                        for ii in range(8):
                            for jj in range(8):
                                i, j = hi * 8 + ii, lo * 8 + jj
                                if j <= i:
                                    cc[tile, ii, jj] = joint[i, j] * scale
                        tile += 1
                streams["in_rfiframemask_buf"][frame].data[sub, f] = admit
                streams["in_rficounts_buf"][frame].data[sub, f] = (~keep).sum() * scale
                # A single group's loss count cannot describe every product.
                streams["in_plcounts_buf"][frame].data[sub, f] = (
                    ~present[:, 0]
                ).sum() * scale
            raw, counts, admitted = np.array(raw), np.array(counts), np.array(admitted)
            n = np.zeros(len(rows), dtype=np.uint64)
            k = np.zeros(len(rows), dtype=np.uint64)
            total = np.zeros(len(rows), dtype=np.complex128)
            q = np.zeros(len(rows))
            for t in range(0, SUBS, 2):
                if not admitted[t : t + 2].all():
                    continue
                n0, n1 = counts[t], counts[t + 1]
                n += (n0 + n1).astype(np.uint64)
                total += raw[t] + raw[t + 1]
                supported = (n0 > 0) & (n1 > 0)
                k += supported.astype(np.uint64)
                diff = (
                    raw[t, supported] / n0[supported]
                    - raw[t + 1, supported] / n1[supported]
                )
                q[supported] += (
                    n0[supported].astype(float)
                    * n1[supported]
                    / (n0[supported] + n1[supported])
                ) * np.abs(diff) ** 2
            mean = np.divide(total, n, out=np.zeros_like(total), where=n > 0)
            weight = np.divide(
                n.astype(float) * k, q, out=np.zeros_like(q), where=(q > 0) & (k > 0)
            )
            truth.append(
                {
                    "bin": b + 1,
                    "freq": int(FREQ[f]),
                    "mean": mean,
                    "weight": weight,
                    "count": n * period,
                    "k": k,
                    "seq": start + b * SUBS * nsub * period,
                    "length": SUBS * nsub * period,
                }
            )
    return streams, truth


def station_order(order, cylinders=4):
    dishes_per_cylinder = E // 2 // cylinders
    if order == "CHIMEBeamformer":
        return [
            (p, c, d)
            for p in range(2)
            for c in range(cylinders)
            for d in range(dishes_per_cylinder)
        ]
    if order == "CHIMECylinder":
        return [
            (p, c, d)
            for c in range(cylinders)
            for p in range(2)
            for d in range(dishes_per_cylinder)
        ]
    raise ValueError(order)


def reorder_reference(ref, input_order, output_order, cylinders=4):
    inputs = station_order(input_order, cylinders)
    permutation = [
        inputs.index(station) for station in station_order(output_order, cylinders)
    ]
    rows, cols = np.triu_indices(E)
    result = dict(ref)
    for key in ("mean", "weight", "count", "k"):
        dense = np.zeros((E, E), dtype=ref[key].dtype)
        dense[rows, cols] = ref[key]
        dense[cols, rows] = ref[key].conjugate()
        result[key] = dense[np.ix_(permutation, permutation)][rows, cols]
    return result


def run(
    tmp_path,
    period=1,
    spf=2,
    scale=1,
    fault=None,
    input_order="CHIMEBeamformer",
    output_order="CHIMEBeamformer",
    cylinders=1,
):
    streams, truth = fixture(period, spf, scale)
    if input_order != output_order:
        truth = [
            reorder_reference(ref, input_order, output_order, cylinders)
            for ref in truth
        ]
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
    }
    boot = tel.get_unix_time_ns("2026-01-01T17:15:50.5", "utc")
    config["telescope"]["frame0_nano"] = boot
    config["telescope"]["num_cylinders"] = cylinders
    config["gps_time"] = {"frame0_nano": boot}
    mode = "scalar" if fault == "legacy-output" else "per_product_v1"
    if fault == "mask-value":
        streams["in_rfiframemask_buf"][0].data[0, 0] = 2
    if fault == "policy-change":
        streams["in_rfiframemask_buf"][2].metadata[
            "rfi_frame_excision_thresholds"
        ] = np.array([[4.0, 0.1]], dtype=np.float32)
    if fault == "policy-nonfinite":
        streams["in_rfiframemask_buf"][0].metadata[
            "rfi_frame_excision_thresholds"
        ] = np.array([[np.nan, 0.1]], dtype=np.float32)
    readers = {
        key: runner.ReadChordBuffer(str(tmp_path), frames)
        for key, frames in streams.items()
    }
    for reader in readers.values():
        reader.write()
    output = PerProductDump(
        str(tmp_path),
        exit_after_n_files=len(truth),
        num_elements=E,
        num_ev=0,
        num_freq=F,
        mode=mode,
    )
    stage = runner.KotekanStageTester(
        "N2Accumulate",
        {
            "num_freq_per_n2k_frame": F,
            "packet_loss_is_scalar": fault == "scalar-producer",
            "bin_in_ERA": False,
            "num_subintegrations_per_bin": SUBS,
            "variance_mode": "CHIMEv1"
            if fault == "legacy-variance"
            else "EvenOddPosDef",
            "do_fringestop": False,
            "input_order": input_order,
            "output_order": output_order,
        },
        readers,
        output,
        config,
        # A policy change mid-run is tolerated with a warning, not refused.
        expect_failure=fault is not None and fault != "policy-change",
    )
    stage.run()
    if fault:
        return stage
    actual = {
        (int(x.metadata.abs_time_idx), int(x.metadata.freq_id)): x
        for x in output.load()
    }
    assert len(actual) == len(truth)
    for ref in truth:
        got = actual[(ref["bin"], ref["freq"])]
        np.testing.assert_array_equal(got.valid_fpga_ticks, ref["count"])
        np.testing.assert_allclose(got.vis, ref["mean"], rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(got.weight, ref["weight"], rtol=3e-6, atol=1e-7)
        assert got.metadata.fpga_start_tick == ref["seq"]
        assert got.metadata.frame_length_fpga_ticks == ref["length"]
        for name in (
            "n_valid_fpga_ticks",
            "n_pl_fpga_ticks",
            "n_rfi_fpga_ticks",
            "n_rfi_only_fpga_ticks",
        ):
            assert getattr(got.metadata, name) == 0
        assert np.all(np.isfinite(got.vis)) and np.all(np.isfinite(got.weight))
        # Require an explicit per-product layout when loading.
        dump = next(Path(tmp_path).glob(f"*{output.name}*.dump"))
        with pytest.raises(RuntimeError, match="expected size"):
            n2buffer.N2Buffer.from_file(
                dump, num_elements=E, num_prod=E * (E + 1) // 2, num_ev=0
            )
    return actual, truth


@pytest.mark.parametrize("period,spf,scale", [(1, 2, 1), (1, 1, 1), (1, 2, 512)])
def test_heterogeneous_support_and_means(tmp_path, period, spf, scale):
    run(tmp_path, period, spf, scale)


@pytest.mark.parametrize(
    "input_order,output_order",
    [("CHIMEBeamformer", "CHIMECylinder"), ("CHIMECylinder", "CHIMEBeamformer")],
)
def test_heterogeneous_support_reordered(tmp_path, input_order, output_order):
    inputs = station_order(input_order)
    permutation = np.array([inputs.index(s) for s in station_order(output_order)])
    rows, cols = np.triu_indices(E)
    reversed_pairs = permutation[rows] > permutation[cols]
    _, truth = run(
        tmp_path,
        period=1,
        spf=1,
        input_order=input_order,
        output_order=output_order,
        cylinders=4,
    )
    # Ensure the fixture exercises complex conjugation.
    assert max(np.max(np.abs(ref["mean"][reversed_pairs].imag)) for ref in truth) > 0.01


@pytest.mark.parametrize(
    "fault,diagnostic",
    [
        ("legacy-output", "support_mode must match"),
        ("scalar-producer", "support_mode must match"),
        ("legacy-variance", "requires EvenOddPosDef"),
        ("mask-value", "frame mask must be binary"),
        ("policy-nonfinite", "RFI policy must be finite"),
    ],
)
def test_unsupported_contract_refused(tmp_path, fault, diagnostic):
    got = run(tmp_path, fault=fault)
    assert got.return_code != 0
    assert diagnostic in got.output


def test_policy_change_warns_and_continues(tmp_path):
    # The scalar path picks up a new RFI policy per output frame; per-product mode
    # does the same, and says so once instead of stopping the run.
    got = run(tmp_path, fault="policy-change")
    assert got.return_code == 0, got.output[-4000:]
    assert "dataset or RFI policy changed mid-run" in got.output
