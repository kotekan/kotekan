# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import pytest
import numpy as np

from kotekan import runner

# Skip if LAPACK support not built into kotekan
if not runner.has_lapack():
    pytest.skip("LAPACK support not available.", allow_module_level=True)


default_params = {
    "num_elements": 200,
    "num_ev": 4,
    "total_frames": 16,
    "cadence": 10.0,
    "mode": "phase_ij",
    # 'mode': 'fill_ij',
    "freq": [0],
    "buffer_depth": 5,
    "num_diagonals_filled": 0,
    "dataset_manager": {"use_dataset_broker": False},
}


def run_eigenvis(tdir_factory, params=None):

    if not params:
        params = default_params

    tmpdir = tdir_factory.mktemp("eigenvis")

    fakevis_buffer = runner.FakeVisBuffer(
        freq_ids=params["freq"], num_frames=params["total_frames"], mode=params["mode"]
    )

    dump_buffer = runner.DumpVisBuffer(str(tmpdir))

    test = runner.KotekanStageTester(
        "eigenVis", {}, fakevis_buffer, dump_buffer, params
    )

    test.run()
    return dump_buffer.load()


def test_basic(tmpdir_factory):

    params = default_params
    num_elements = params["num_elements"]
    expected_evec_phase_fact = np.exp(1j * np.arange(num_elements))

    eigen_data = run_eigenvis(tmpdir_factory)
    for frame in eigen_data:
        largest_eval = frame.eval[0]
        largest_evec = frame.evec[:num_elements]
        assert abs(largest_eval - num_elements) / num_elements < 1e-6
        assert np.allclose(largest_evec / largest_evec[0], expected_evec_phase_fact)
        assert np.allclose(abs(largest_evec), 1 / np.sqrt(num_elements))
        zero_eval = frame.eval[1]
        assert zero_eval / num_elements < 1e-6
        # The erms error (below) may be larger due to accumulated error.
        # If this test is failing, it might be ok; some additional discussion is at
        # https://github.com/kotekan/kotekan/issues/1034.
        assert frame.erms < 1e-5


def test_filled(tmpdir_factory):

    params = dict(default_params)
    params["num_diagonals_filled"] = 10
    num_elements = params["num_elements"]
    expected_evec_phase = np.exp(1j * np.arange(num_elements))

    eigen_data = run_eigenvis(tmpdir_factory, params)
    for frame in eigen_data[8:]:
        largest_eval = frame.eval[0]
        largest_evec = frame.evec[:num_elements]
        assert abs(largest_eval - num_elements) / num_elements < 1e-4
        assert np.allclose(
            largest_evec / largest_evec[0], expected_evec_phase, rtol=1e-3
        )
        assert np.allclose(abs(largest_evec), 1 / np.sqrt(num_elements), rtol=1e-3)
        assert frame.erms < 1e-3


def test_input_excluded(tmpdir_factory):

    params = dict(default_params)
    params["exclude_inputs"] = [5, 10, 6]
    nexclude = len(params["exclude_inputs"])
    num_elements = params["num_elements"]
    expected_eval = num_elements - nexclude
    expected_evec_phase_fact = np.exp(1j * np.arange(num_elements))
    expected_evec_phase_fact[params["exclude_inputs"]] = 0

    eigen_data = run_eigenvis(tmpdir_factory, params)
    for frame in eigen_data:
        largest_eval = frame.eval[0]
        largest_evec = frame.evec[:num_elements]
        assert abs(largest_eval - expected_eval) / num_elements < 1e-6
        assert np.allclose(largest_evec / largest_evec[0], expected_evec_phase_fact)
