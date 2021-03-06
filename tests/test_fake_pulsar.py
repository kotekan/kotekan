# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import pytest
import numpy as np

from kotekan import runner

pulsar_params = {
    "num_elements": 4,
    "num_ev": 4,
    "samples_per_data_set": 4000,  # 10 ms frames
    "num_frames": 4,  # One extra sample to ensure we actually get 256
    "block_size": 2,
    "pattern": "pulsar",
    "freq": 777,
    "coeff": [0.0, 0.0],
    "dm": 0.0,
    "t_ref": 58000.0,
    "phase_ref": 0.0,
    "rot_freq": 0.03e3,  # one period spans 3 frames
    "pulse_width": 1e-3,
    "gaussian_bgnd": False,
}

accumulate_params = pulsar_params.copy()
accumulate_params.update(
    {"num_gpu_frames": 1, "dataset_manager": {"use_dataset_broker": False}}
)


@pytest.fixture(scope="module")
def pulsar_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("accumulate")

    dump_buffer = runner.DumpVisBuffer(str(tmpdir))

    test = runner.KotekanStageTester(
        "visAccumulate",
        accumulate_params,
        runner.FakeGPUBuffer(**pulsar_params),
        dump_buffer,
        accumulate_params,
    )

    test.run()

    yield dump_buffer.load()


def test_pulsar(pulsar_data):
    assert len(pulsar_data) != 0
    # Should have one or two frames with a pulse
    real_part = 0
    for frame in pulsar_data:
        assert (frame.vis.imag == 0).all()
        real_part += frame.vis.real.sum()
    real_part /= frame.vis.shape[0] * 10
    assert real_part == 1.0 or real_part == 2.0
