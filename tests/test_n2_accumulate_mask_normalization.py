"""Test EvenOddPosDef means, counts and weights with deterministic correlator sums.

Counts and first-stage masks are the same across inputs, as required by
N2Accumulate.
"""

from copy import deepcopy

import kotekan.telescope as tel
import numpy as np
import pytest
from test_n2_accumulate import chime_tel, make_zeroed_chord_buffer

from kotekan import runner

_CASES = (
    "all-pairs-supported",
    "either-member-frame-rejected",
    "single-member-zero",
    "all-counts-zero",
    "no-complete-pair-with-positive-mean",
    "zero-pair-variance",
    "unequal-counts",
    "mixed-missing-and-rejected",
)
_NUM_ELEMENTS = 64  # Smallest array that fills both count and correlation blocks.
_NUM_FREQ = 3
_NUM_BINS = len(_CASES)
_SUBS_PER_BIN = 8


def _case_samples(case, scale):
    counts = np.full(_SUBS_PER_BIN, 16, dtype=np.int64)
    admit = np.ones(_SUBS_PER_BIN, dtype=np.uint8)
    if case == "either-member-frame-rejected":
        # Reject pair 1 via its odd member and pair 2 via its even member.
        admit[[3, 4]] = 0
    elif case == "single-member-zero":
        counts[:] = [16, 0, 8, 4, 0, 12, 16, 16]
    elif case == "all-counts-zero":
        counts[:] = 0
    elif case == "no-complete-pair-with-positive-mean":
        counts[:] = [16, 0, 0, 8, 4, 0, 0, 12]
    elif case == "unequal-counts":
        counts[:] = [3, 8, 5, 13, 7, 16, 2, 11]
    elif case == "mixed-missing-and-rejected":
        counts[:] = [0, 0, 4, 0, 8, 12, 16, 16]
        admit[7] = 0
    counts *= scale
    # Integer means keep the int32 correlator sums exact.
    # The constant case gives Q=0 with samples in every pair.
    means = np.array([1, 3, 2, 1, 3, 2, 1, 2], dtype=np.float64)
    if case == "zero-pair-variance":
        means[:] = 2
    return counts, admit, means


def _reference(counts, admit, raw):
    """Calculate expected values before arranging sums in correlator blocks.

    raw[t, product] contains unnormalized sums. Accepted pairs contribute to
    the mean; both frames need samples to contribute to the variance estimate.
    Python integers avoid count-product overflow.
    """
    total = 0
    usable = 0
    summed = np.zeros(raw.shape[1], dtype=np.complex128)
    q = np.zeros(raw.shape[1], dtype=np.float64)
    for even in range(0, len(counts), 2):
        odd = even + 1
        if not (admit[even] and admit[odd]):
            continue
        n0, n1 = int(counts[even]), int(counts[odd])
        total += n0 + n1
        summed += raw[even] + raw[odd]
        if n0 > 0 and n1 > 0:
            usable += 1
            difference = raw[even] / n0 - raw[odd] / n1
            q += (n0 * n1 / (n0 + n1)) * np.abs(difference) ** 2
    vis = summed.conjugate() / total if total else summed * 0
    weight = np.zeros_like(q)
    if total and usable:
        np.divide(total * usable, q, out=weight, where=q > 0)
    return vis, weight, total, usable, q


def _run_accumulation(
    tmpdir_factory, scale=1, period=1, subintegrations_per_frame=2,
):
    subintegration = 16 * scale
    num_frames = _NUM_BINS * _SUBS_PER_BIN // subintegrations_per_frame
    first_seq = _SUBS_PER_BIN * subintegration * period
    frequencies = np.array([202, 614, 800], dtype=np.int32)
    telescope = deepcopy(chime_tel)
    boot_time = tel.get_unix_time_ns("2026-01-01T17:15:50.5", "utc")
    telescope["frame0_nano"] = boot_time
    config = {
        "buffer_depth": 3,
        "samples_per_data_set": subintegrations_per_frame * subintegration,
        "sub_integration_ntime": subintegration,
        "num_local_freq": _NUM_FREQ,
        "num_elements": _NUM_ELEMENTS,
        "num_polarizations": 2,
        "num_dishes": _NUM_ELEMENTS // 2,
        "num_ev": 0,
        "telescope": telescope,
        "gps_time": {"frame0_nano": boot_time},
    }

    def buffers(
        name, dtype, typename, tail_shape, tail_names, tail_scalings, extra_meta=None
    ):
        return make_zeroed_chord_buffer(
            name,
            dtype,
            typename,
            (subintegrations_per_frame, _NUM_FREQ) + tail_shape,
            ("Tc", "F") + tail_names,
            (subintegration, 1) + tail_scalings,
            first_seq,
            subintegrations_per_frame * subintegration * period,
            num_frames,
            freq_ids=frequencies,
            time_downsampling=subintegration * period,
            extra_meta=extra_meta,
        )

    # 64 inputs => 10 lower triangular 16x16 correlation blocks and one
    # 8x8 count block, with each count entry representing eight inputs.
    corr = buffers(
        "n2k_correlation",
        np.int32,
        "int32",
        (10, 16, 16, 2),
        ("DPhi", "DPlo1", "DPlo2", "C"),
        (16, 1, 1, 1),
    )
    counts = buffers(
        "n2k_counts",
        np.int32,
        "int32",
        (1, 8, 8),
        ("D8Phi", "D8Plo1", "D8Plo2"),
        (64, 8, 8),
    )
    rfi = buffers("RFImask_counts", np.int32, "int32", (), (), ())
    pl = buffers("pl_lost_counts_scalar", np.int32, "int32", (), (), ())
    mask = buffers(
        "RFIFrameMask",
        np.uint8,
        "uint8",
        (),
        (),
        (),
        {
            "rfi_frame_excision_enabled": True,
            "rfi_frame_excision_thresholds": np.array([[3.0, 0.1]], dtype=np.float32),
        },
    )

    # Build the reference in upper-triangular order, then store the input sums
    # in the correlator's lower-triangular blocks.
    rows, cols = np.triu_indices(_NUM_ELEMENTS)
    phase = (1 + (rows + 2 * cols) % 3).astype(np.complex128)
    phase += 1j * np.where(rows == cols, 0, (2 * rows + cols) % 3 - 1)
    block_row, block_col = cols // 16, rows // 16
    block = block_row * (block_row + 1) // 2 + block_col
    block_i, block_j = cols % 16, rows % 16
    expected = {}
    for bin_idx in range(_NUM_BINS):
        for f in range(_NUM_FREQ):
            # Vary cases by frequency and bin to check that counts stay separate
            # and the accumulator resets, including after k=0.
            case = _CASES[(bin_idx + 3 * f) % len(_CASES)]
            n, admitted, means = _case_samples(case, scale)
            raw = n[:, None] * means[:, None] * phase[None, :]
            reference = _reference(n, admitted, raw)
            pair_admit = admitted[::2] & admitted[1::2]
            # Exercise merged first-stage RFI and packet-loss counts. A
            # second-stage frame rejection attributes the whole pair to RFI.
            first_stage_rfi = np.zeros(_SUBS_PER_BIN, dtype=np.int64)
            if case in ("single-member-zero", "mixed-missing-and-rejected"):
                first_stage_rfi[n == 0] = subintegration
            lost = subintegration - n - first_stage_rfi
            packet_loss = int(np.sum(lost))
            rfi_per_pair = first_stage_rfi.reshape(-1, 2).sum(axis=1)
            rfi_ticks = int(
                np.sum(np.where(pair_admit, rfi_per_pair, 2 * subintegration))
            )
            expected[(bin_idx + 1, int(frequencies[f]))] = {
                "case": case,
                "reference": reference,
                "packet_loss": packet_loss * period,
                "rfi_ticks": rfi_ticks * period,
                "valid_ticks": reference[2] * period,
                "first_seq": first_seq
                + bin_idx * _SUBS_PER_BIN * subintegration * period,
                "ticks": _SUBS_PER_BIN * subintegration * period,
                "telescope": telescope,
            }
            for t in range(_SUBS_PER_BIN):
                frame, sub = divmod(
                    bin_idx * _SUBS_PER_BIN + t, subintegrations_per_frame
                )
                corr[frame].data[sub, f, block, block_i, block_j, 0] = raw[t].real
                corr[frame].data[sub, f, block, block_i, block_j, 1] = raw[t].imag
                counts[frame].data[sub, f] = n[t]
                pl[frame].data[sub, f] = lost[t]
                rfi[frame].data[sub, f] = first_stage_rfi[t]
                mask[frame].data[sub, f] = admitted[t]

    work = str(tmpdir_factory.mktemp("n2-mask-normalization"))
    inputs = {}
    for name, data in (
        ("in_buf", corr),
        ("in_counts_buf", counts),
        ("in_rficounts_buf", rfi),
        ("in_plcounts_buf", pl),
        ("in_rfiframemask_buf", mask),
    ):
        inputs[name] = runner.ReadChordBuffer(work, data)
        inputs[name].write()
    output = runner.DumpN2Buffer(
        work,
        exit_after_n_files=len(expected),
        num_elements=_NUM_ELEMENTS,
        num_ev=0,
        num_freq=_NUM_FREQ,
    )
    stage = runner.KotekanStageTester(
        "N2Accumulate",
        {
            "num_freq_per_n2k_frame": _NUM_FREQ,
            "packet_loss_is_scalar": True,
            "bin_in_ERA": False,
            "num_subintegrations_per_bin": _SUBS_PER_BIN,
            "variance_mode": "EvenOddPosDef",
            "do_fringestop": False,
            "input_order": "CHIMEBeamformer",
        },
        inputs,
        output,
        config,
    )
    stage.run()
    actual = output.load()
    assert len(actual) == len(expected)
    keyed = {(int(v.metadata.abs_time_idx), int(v.metadata.freq_id)): v for v in actual}
    assert set(keyed) == set(expected)
    return keyed, expected


@pytest.fixture(
    scope="module",
    params=[(1, 1, 2), (4096, 1, 2), (1, 1, 1)],
    ids=["counts-16-period-1", "counts-65536-period-1", "cross-frame-pairs",],
)
def masked_accumulation(request, tmpdir_factory):
    return _run_accumulation(tmpdir_factory, *request.param)


@pytest.mark.parametrize("case", _CASES)
def test_masked_mean_counts_and_precision(masked_accumulation, case):
    actual, expected = masked_accumulation
    for key, record in expected.items():
        if record["case"] != case:
            continue
        frame = actual[key]
        vis, weight, total, usable, q = record["reference"]
        context = f"{case}, bin={key[0]}, freq={key[1]}, N={total}, k={usable}"
        assert frame.metadata.n_valid_fpga_ticks == record["valid_ticks"], context
        assert frame.metadata.n_pl_fpga_ticks == record["packet_loss"], context
        assert frame.metadata.n_rfi_fpga_ticks == record["rfi_ticks"], context
        assert frame.metadata.n_rfi_only_fpga_ticks == (
            record["ticks"] - record["valid_ticks"] - record["packet_loss"]
        ), context
        assert frame.metadata.fpga_start_tick == record["first_seq"], context
        assert frame.metadata.frame_length_fpga_ticks == record["ticks"], context
        assert frame.metadata.frame_start_time_ns == tel.get_t_inst_ns(
            record["first_seq"], record["telescope"]
        ), context
        expected_center_ns = tel.get_t_inst_ns(
            record["first_seq"] + record["ticks"] // 2, record["telescope"]
        )
        assert abs(frame.metadata.time_center_eop.t_inst_ns - expected_center_ns) <= 5
        assert np.all(np.isfinite(frame.vis)), context
        assert np.all(np.isfinite(frame.weight)), context
        assert np.all(frame.weight >= 0), context
        np.testing.assert_allclose(
            frame.vis, vis, rtol=2e-6, atol=1e-7, err_msg=context
        )
        np.testing.assert_allclose(
            frame.weight, weight, rtol=2e-5, atol=1e-7, err_msg=context
        )
        if case == "no-complete-pair-with-positive-mean":
            assert total > 0 and usable == 0
            assert np.all(np.abs(frame.vis) > 0)
            assert not np.any(frame.weight)
        if case == "zero-pair-variance":
            assert total > 0 and usable == 4 and not np.any(q)
            assert not np.any(frame.weight)
