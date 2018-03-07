# import pytest
import numpy as np

import kotekan_runner

default_params = {
    'num_elements': 200,
    'num_eigenvectors': 4,
    'total_frames': 16,
    'cadence': 10.0,
    'mode': 'phase_ij',
    # 'mode': 'fill_ij',
    'freq': [0],
    'buffer_depth': 5,
    'num_diagonals_filled': 0
     }


def run_eigenvis(tdir_factory, params=None):

    if not params:
        params = default_params

    tmpdir = tdir_factory.mktemp("eigenvis")

    fakevis_buffer = kotekan_runner.FakeVisBuffer(
            freq_ids=params['freq'],
            num_frames=params['total_frames'],
            mode=params['mode'],
            )

    dump_buffer = kotekan_runner.DumpVisBuffer(str(tmpdir))

    test = kotekan_runner.KotekanProcessTester(
        'eigenVis', {},
        fakevis_buffer,
        dump_buffer,
        params
    )

    test.run()
    return dump_buffer.load()


def test_basic(tmpdir_factory):

    eigen_data = run_eigenvis(tmpdir_factory)
    params = default_params
    num_elements = params['num_elements']
    expected_evec_phase = np.exp(1j * np.arange(num_elements))

    for frame in eigen_data:
        largeset_eval = frame.evals[-1]
        largeset_evec = frame.evecs[-num_elements:]
        assert abs(largeset_eval - num_elements) / num_elements < 1e-6
        assert np.allclose(largeset_evec / largeset_evec[0],
                           expected_evec_phase)
        assert np.allclose(abs(largeset_evec), 1 / np.sqrt(num_elements))
        zero_eval = frame.evals[-2]
        assert zero_eval / num_elements < 1e-6
