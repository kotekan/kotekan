# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import pytest
import numpy as np
import h5py

from kotekan import visbuffer
from kotekan import runner

writer_params = {
    "num_elements": 4,
    "num_ev": 2,
    "cadence": 5.0,
    "total_frames": 10,  # One extra sample to ensure we actually get 256
    "freq": [3, 777, 554],
    "dataset_manager": {"use_dataset_broker": False},
}


@pytest.fixture(scope="module")
def written_data(tmpdir_factory):

    tmpdir = str(tmpdir_factory.mktemp("writer"))

    fakevis_buffer = runner.FakeVisBuffer(
        freq_ids=writer_params["freq"],
        num_frames=writer_params["total_frames"],
        cadence=writer_params["cadence"],
    )

    params = writer_params.copy()
    params["root_path"] = tmpdir

    test = runner.KotekanStageTester(
        "visWriter",
        {"node_mode": False, "file_type": "raw"},
        fakevis_buffer,
        None,
        params,
    )

    test.run()

    import glob

    files = sorted(glob.glob(tmpdir + "/20??????T??????Z_*_corr/*.meta"))

    yield [visbuffer.VisRaw(fname) for fname in files]


def test_vis(written_data):

    for vr in written_data:

        # Construct vis array
        vis = vr.data["vis"]

        # Extract metadata
        ftime = vr.time["fpga_count"]
        ctime = vr.time["ctime"]
        freq = np.array([f["centre"] for f in vr.index_map["freq"]])

        # Check the diagonals are correct
        pi = 0
        for ii in range(writer_params["num_elements"]):
            assert (vis[:, :, pi].imag == ii).all()
            pi += writer_params["num_elements"] - ii

        # Check the times are correct
        assert (vis[:, :, 0].real == ftime[:, np.newaxis].astype(np.float32)).all()
        assert (vis[:, :, 1].real == ctime[:, np.newaxis].astype(np.float32)).all()

        # Check the frequencies are correct
        vfreq = 800.0 - 400.0 * vis[:, :, 2].real / 1024
        assert (vfreq == freq[np.newaxis, :]).all()


def test_metadata(written_data):

    nt = writer_params["total_frames"]

    for vr in written_data:

        # Extract metadata
        ctime = vr.time["ctime"]
        freq = np.array([f["centre"] for f in vr.index_map["freq"]])
        input_a = np.array([p[0] for p in vr.index_map["prod"]])
        input_b = np.array([p[1] for p in vr.index_map["prod"]])

        # Check the number of samples has been written correctly
        assert vr.num_time == nt

        # Check the times
        assert np.allclose(np.diff(ctime), writer_params["cadence"])

        # Check the frequencies
        wfreq = 800.0 - 400.0 * np.array(writer_params["freq"]) / 1024
        assert (freq == wfreq).all()

        # Check the products
        ia, ib = np.triu_indices(writer_params["num_elements"])
        assert (input_a == ia).all()
        assert (input_b == ib).all()


def test_eigenvectors(written_data):

    for vr in written_data:

        nt = writer_params["total_frames"]
        nf = len(writer_params["freq"])
        ne = writer_params["num_ev"]
        ni = writer_params["num_elements"]

        evals = vr.data["eval"]
        evecs = vr.data["evec"]
        erms = vr.data["erms"]

        # Check datasets are present
        assert evals.shape == (nt, nf, ne)
        assert evecs.shape == (nt, nf, ne * ni)
        assert erms.shape == (nt, nf)

        evecs = evecs.reshape(nt, nf, ne, ni)

        im_ev = np.array(vr.index_map["ev"])

        print(im_ev, ne)

        # Check that the index map is there correctly
        assert (im_ev == np.arange(ne)).all()

        # Check that the datasets have the correct values
        assert (evals == np.arange(ne)[np.newaxis, np.newaxis, :]).all()
        assert (
            evecs.real == np.arange(ne)[np.newaxis, np.newaxis, :, np.newaxis]
        ).all()
        assert (
            evecs.imag == np.arange(ni)[np.newaxis, np.newaxis, np.newaxis, :]
        ).all()
        assert (erms == 1.0).all()
