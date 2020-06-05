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
    "wait": False,
    "dataset_manager": {"use_dataset_broker": False},
}

diag_stage_params = {"stack_type": "diagonal"}

chime_global_params = {
    "num_elements": 2048,
    "num_ev": 2,
    "total_frames": 3,
    "cadence": 2.0,
    "mode": "chime",
    "freq_ids": [0, 500],
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
def chime_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("chime")

    fakevis_buffer = runner.FakeVisBuffer(
        freq_ids=chime_global_params["freq_ids"],
        num_frames=chime_global_params["total_frames"],
    )

    dump_buffer = runner.DumpVisBuffer(str(tmpdir))

    test = runner.KotekanStageTester(
        "baselineCompression",
        chime_stage_params,
        fakevis_buffer,
        dump_buffer,
        chime_global_params,
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


def test_chime(chime_data):

    nvis_chime = 4 * (4 * 256 - 1) + 6 * 4 * 511

    # This is the typical number of entries per polarisation (for XX, XY and YY, not YX)
    np1 = 4 * 256 + 6 * 511

    for frame in chime_data:
        assert frame.vis.shape[0] == nvis_chime

        # Check that the entries in XX and XY are the same
        assert float_allclose(frame.vis[:np1], frame.vis[np1 : (2 * np1)])

        v1 = frame.vis[:np1]
        w1 = frame.weight[:np1]

        # Loop over all pairs of cylinders for XX
        for ci in range(4):
            for cj in range(ci, 4):

                # These numbers depend if we are within a cyl or not
                nv = 256 if ci == cj else 511  # Number of entries to compare
                lb = 0 if ci == cj else -255  # The most negative separation

                # A list of the feed separations in the NS dir
                d = np.arange(lb, 256)

                assert float_allclose(v1[:nv], (cj - ci + 1.0j * d))
                assert float_allclose(w1[:nv], (256.0 - np.abs(d)))

                v1 = v1[nv:]
                w1 = w1[nv:]
