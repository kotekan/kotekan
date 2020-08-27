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
    "dataset_manager": {"use_dataset_broker": False},
}

vis_params = {}


@pytest.fixture(scope="module", params=[[1, 13], [0, 1, 14, 15, 20]])
def subset_data(request, tmpdir_factory):

    inputs = request.param

    tmpdir = tmpdir_factory.mktemp("subset")

    fakevis_buffer = runner.FakeVisBuffer(
        freq_ids=subset_params["freq_ids"], num_frames=subset_params["total_frames"]
    )

    write_buffer = runner.VisWriterBuffer(str(tmpdir), "raw")

    task_params = {"inputs": inputs}
    task_params.update(subset_params)

    test = runner.KotekanStageTester(
        "InputSubset", vis_params, fakevis_buffer, write_buffer, task_params
    )

    test.run()

    return inputs, write_buffer.load()


def test_metadata(request, subset_data):
    # Check that we have modified the input and product index maps correctly

    inputs, data = subset_data
    exp_inputs = [i for i in inputs if i < subset_params["num_elements"]]
    ninputs = len(exp_inputs)
    exp_prods = [[i, j] for i in range(ninputs) for j in range(i, ninputs)]

    input_ids = [i[0] for i in data.index_map["input"]]

    assert input_ids == exp_inputs
    assert data.index_map["prod"] == exp_prods


def test_data(subset_data):
    # Check that we have extracted the correct parts of the data

    inputs, data = subset_data

    n_el = subset_params["num_elements"]
    num_prod = n_el * (n_el + 1) // 2

    exp_inputs = np.array([i for i in inputs if i < subset_params["num_elements"]])
    ninput = len(exp_inputs)

    exp_vis = np.array(
        [
            (exp_inputs[ia] + exp_inputs[ib] * 1j)
            for ia in range(ninput)
            for ib in range(ia, ninput)
        ]
    )

    evecs = (
        np.arange(subset_params["num_ev"])[:, None] + 1.0j * exp_inputs[None, :]
    ).flatten()

    assert (data.data["vis"] == exp_vis).all()
    assert (data.data["eval"] == np.arange(subset_params["num_ev"])).all()
    assert (data.data["evec"] == evecs).all()
    assert (data.data["erms"] == 1.0).all()
