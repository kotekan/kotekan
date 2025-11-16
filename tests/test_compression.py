# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import pytest
import numpy as np

from kotekan import runner


diag_global_params = {
    "num_elements": 16,
    "num_ev": 2,
    "total_frames": 128,
    "cadence": 2048.0,
    "mode": "phase_ij",
    "freq_ids": [0, 250],
    "buffer_depth": 5,
    "dataset_manager": {"use_dataset_broker": False},
}

diag_stage_params = {"stack_type": "diagonal"}

smallchime_global_params = {
    "num_elements": 128,
    "num_ev": 1,
    "total_frames": 2,
    "wait": True,
    "cadence": 1.0,
    "mode": "chime",
    "freq_ids": [0, 250, 500],
    "buffer_depth": 100,
    "dataset_manager": {"use_dataset_broker": False},
}

chime_stage_params = {"stack_type": "chime_in_cyl"}


def float_allclose(a, b):
    """Compare two float (arrays).

    This comparison uses a tolerance related to the precision of the datatypes
    to account for rounding errors in arithmetic.
    """

    res_a = np.finfo(np.array(a).dtype).resolution
    res_b = np.finfo(np.array(b).dtype).resolution

    tol = max(res_a, res_b)
    return np.allclose(a, b, rtol=tol, atol=0)


def _chime_feed_props(chan_id):
    """Mirror chimeFeed::from_input from the C++ code."""
    return {
        "cyl": chan_id // 512,
        "pol": ((chan_id // 256) + 1) % 2,
        "feed": chan_id % 256,
    }


# The following function was written by codex, and I have
# not verified the math in it. -JM
def _chime_stack_expectations(num_elements):
    """Replicate stack_chime_in_cyl + baselineCompression for FakeVis(chime)."""

    # Build the feed properties
    feeds = [_chime_feed_props(ii) for ii in range(num_elements)]

    # All products i<=j as in FakeVis
    prods = []
    feed_diffs = []
    conj_flags = []
    vis_values = []

    for i in range(num_elements):
        for j in range(i, num_elements):
            fi, fj = feeds[i], feeds[j]

            # Canonical ordering and conjugation flag (matches calculate_chime_vis)
            conj = False
            if (
                (fi["cyl"] > fj["cyl"])
                or (fi["cyl"] == fj["cyl"] and fi["feed"] > fj["feed"])
                or (
                    fi["cyl"] == fj["cyl"]
                    and fi["feed"] == fj["feed"]
                    and fi["pol"] > fj["pol"]
                )
            ):
                fi, fj = fj, fi
                conj = True

            feed_diff = (
                fi["pol"],
                fj["pol"],
                fi["cyl"],
                fj["cyl"],
                fj["feed"] - fi["feed"],
            )

            feed_diffs.append(feed_diff)
            conj_flags.append(conj)

            # FakeVis(chime) pattern value for the original (i, j) ordering
            vi, vj = feeds[i], feeds[j]
            vis = complex(vj["cyl"] - vi["cyl"], vj["feed"] - vi["feed"])
            vis_values.append(vis)
            prods.append((i, j))

    # Recreate stack_chime_in_cyl ordering
    sort_ind = sorted(range(len(feed_diffs)), key=lambda idx: feed_diffs[idx])
    stack_map = [None] * len(feed_diffs)
    cur_stack = 0
    cur_fd = feed_diffs[sort_ind[0]] if sort_ind else None
    for idx in sort_ind:
        if feed_diffs[idx] != cur_fd:
            cur_fd = feed_diffs[idx]
            cur_stack += 1
        stack_map[idx] = cur_stack

    num_stacks = 0 if not stack_map else max(stack_map) + 1
    out_vis = np.zeros(num_stacks, dtype=np.complex128)
    out_weight = np.zeros(num_stacks, dtype=np.float64)

    # BaselineCompression accumulation
    for idx, stack in enumerate(stack_map):
        vis = vis_values[idx]
        if conj_flags[idx]:
            vis = np.conjugate(vis)
        out_vis[stack] += vis
        out_weight[stack] += 1.0

    # Normalise to get the averaged visibilities; weights remain as counts
    nonzero = out_weight != 0
    out_vis[nonzero] /= out_weight[nonzero]

    return out_vis, out_weight


@pytest.fixture(scope="module")
def diagonal_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("diagonal")

    fakevis_buffer = runner.FakeVisBuffer(
        freq_ids=diag_global_params["freq_ids"],
        num_frames=diag_global_params["total_frames"],
    )

    dump_buffer = runner.DumpVisBuffer(str(tmpdir))

    test = runner.KotekanStageTester(
        "baselineCompression",
        diag_stage_params,
        fakevis_buffer,
        dump_buffer,
        diag_global_params,
    )

    test.run()

    yield dump_buffer.load()


@pytest.fixture(scope="module")
def smallchime_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("chime")

    fakevis_buffer = runner.FakeVisBuffer(
        freq_ids=smallchime_global_params["freq_ids"],
        num_frames=smallchime_global_params["total_frames"],
        wait=True,
    )

    dump_buffer = runner.DumpVisBuffer(str(tmpdir))

    test = runner.KotekanStageTester(
        "baselineCompression",
        chime_stage_params,
        fakevis_buffer,
        dump_buffer,
        smallchime_global_params,
    )

    test.run()

    yield dump_buffer.load()


def test_metadata(diagonal_data):

    freq_ids = np.array([frame.metadata.freq_id for frame in diagonal_data])
    fpga_seqs = np.array([frame.metadata.fpga_seq for frame in diagonal_data])
    dset_ids = np.array([frame.metadata.dataset_id for frame in diagonal_data])
    nprod = np.array([frame.metadata.num_prod for frame in diagonal_data])

    assert (freq_ids.reshape((-1, 2)) == np.array([[0, 250]])).all()
    assert (
        (fpga_seqs.reshape((-1, 2)) / 800e6)
        == (np.arange(diag_global_params["total_frames"]))[:, np.newaxis]
    ).all()
    assert (nprod == diag_global_params["num_elements"]).all()


def test_chime(smallchime_data):

    expected_vis, expected_weight = _chime_stack_expectations(
        smallchime_global_params["num_elements"]
    )

    for frame in smallchime_data:
        assert frame.vis.shape[0] == expected_vis.shape[0]
        assert float_allclose(frame.vis, expected_vis)
        assert float_allclose(frame.weight, expected_weight)
