# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import pytest
import numpy as np

from kotekan import runner
from kotekan import visutil

subset_params = {
    "num_elements": 16,
    "num_ev": 2,
    "total_frames": 128,
    "cadence": 5.0,
    "mode": "fill_ij",
    "freq_ids": [250],
    "buffer_depth": 5,
    "prod_subset_type": "only_inputs",
    "input_list": [1, 134],
    "dataset_manager": {"use_dataset_broker": False},
}

vis_params = {}


@pytest.fixture(scope="module")
def subset_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("subset")

    fakevis_buffer = runner.FakeVisBuffer(
        freq_ids=subset_params["freq_ids"], num_frames=subset_params["total_frames"]
    )

    write_buffer = runner.VisWriterBuffer(str(tmpdir), "raw")

    test = runner.KotekanStageTester(
        "prodSubset", vis_params, fakevis_buffer, write_buffer, subset_params
    )

    test.run()

    return write_buffer.load()


def only_inputs_condition(prod, input_list):

    inpa_in_list = False
    inpb_in_list = False
    for ipt in input_list:
        if prod.input_a == ipt:
            inpa_in_list = True
        if prod.input_b == ipt:
            inpb_in_list = True

    return inpa_in_list and inpb_in_list


def test_subset(subset_data):

    n_el = subset_params["num_elements"]
    num_prod = n_el * (n_el + 1) // 2

    vis = []
    for ii in range(num_prod):
        # With fill_ij, vis_ij = i+j*(1j)
        prod = visutil.icmap(ii, subset_params["num_elements"])
        if only_inputs_condition(prod, subset_params["input_list"]):
            vis.append(prod.input_a + 1j * prod.input_b)

    evecs = (
        np.arange(subset_params["num_ev"])[:, None]
        + 1.0j * np.arange(subset_params["num_elements"])[None, :]
    ).flatten()

    assert (subset_data.data["vis"] == np.array(vis)).all()
    assert (subset_data.data["eval"] == np.arange(subset_params["num_ev"])).all()
    assert (subset_data.data["evec"] == evecs).all()
    assert (subset_data.data["erms"] == 1.0).all()
