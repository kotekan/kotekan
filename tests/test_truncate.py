# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import pytest
import numpy as np

from kotekan import visbuffer
from kotekan import runner

# Skip if OpenMP support not built into kotekan
if not runner.has_openmp():
    pytest.skip("OpenMP support not available.", allow_module_level=True)

trunc_params = {
    "fakevis_mode": "test_pattern_simple",
    "test_pattern_value": [0, 0],
    "cadence": 2.0,
    "total_frames": 10,
    "err_sq_lim": 0.003,
    "weight_fixed_precision": 0.001,
    "data_fixed_precision": 0.0001,
    "num_ev": 4,
    "num_elements": 4,
    "out_file": "/tmp/out.csv",
    "dataset_manager": {"use_dataset_broker": False},
}


@pytest.fixture(scope="module")
def vis_data(tmpdir_factory):
    """ Truncated visibilities """

    tmpdir = tmpdir_factory.mktemp("vis_data_t")

    fakevis_buffer = runner.FakeVisBuffer(
        num_frames=trunc_params["total_frames"],
        mode=trunc_params["fakevis_mode"],
        test_pattern_value=trunc_params["test_pattern_value"],
        cadence=trunc_params["cadence"],
    )

    in_dump_config = trunc_params.copy()
    in_dump_config["base_dir"] = str(tmpdir)
    in_dump_config["file_name"] = "fakevis"
    in_dump_config["file_ext"] = "dump"

    out_dump_buffer = runner.DumpVisBuffer(str(tmpdir))

    test = runner.KotekanStageTester(
        "VisTruncate",
        trunc_params,
        buffers_in=fakevis_buffer,
        buffers_out=out_dump_buffer,
        global_config=trunc_params,
        parallel_stage_type="rawFileWrite",
        parallel_stage_config=in_dump_config,
        noise=True,
    )

    test.run()

    yield (
        out_dump_buffer.load(),
        visbuffer.VisBuffer.load_files("%s/*fakevis*.dump" % str(tmpdir)),
    )


@pytest.fixture(scope="module")
def vis_data_zero_weights(tmpdir_factory):
    """ Truncated visibilities """

    tmpdir = tmpdir_factory.mktemp("vis_data_t")

    fakevis_buffer = runner.FakeVisBuffer(
        num_frames=trunc_params["total_frames"],
        mode=trunc_params["fakevis_mode"],
        cadence=trunc_params["cadence"],
        zero_weight=True,
    )

    in_dump_config = trunc_params.copy()
    in_dump_config["base_dir"] = str(tmpdir)
    in_dump_config["file_name"] = "fakevis"
    in_dump_config["file_ext"] = "dump"

    out_dump_buffer = runner.DumpVisBuffer(str(tmpdir))

    test = runner.KotekanStageTester(
        "VisTruncate",
        trunc_params,
        buffers_in=fakevis_buffer,
        buffers_out=out_dump_buffer,
        global_config=trunc_params,
        parallel_stage_type="rawFileWrite",
        parallel_stage_config=in_dump_config,
        noise="random",
    )

    test.run()

    yield (
        out_dump_buffer.load(),
        visbuffer.VisBuffer.load_files("%s/*fakevis*.dump" % str(dir)),
    )


def test_truncation(vis_data):
    n = trunc_params["num_elements"]

    for frame_t, frame in zip(vis_data[0], vis_data[1]):
        assert np.any(frame.vis != frame_t.vis)
        assert np.all(
            np.abs(frame.vis - frame_t.vis)
            <= np.sqrt(trunc_params["err_sq_lim"] / frame.weight)
        )
        assert np.any(frame.weight != frame_t.weight)
        assert np.all(
            np.abs(frame.weight - frame_t.weight)
            <= np.abs(frame.weight) * trunc_params["weight_fixed_precision"]
        )
        assert np.all(
            np.abs(frame.evec.real - frame_t.evec.real)
            <= np.abs(frame.evec.real) * trunc_params["data_fixed_precision"]
        )
        assert np.all(
            np.abs(frame.evec.imag - frame_t.evec.imag)
            <= np.abs(frame.evec.imag) * trunc_params["data_fixed_precision"]
        )
        assert np.all(
            np.abs(frame.gain.real - frame_t.gain.real)
            <= np.abs(frame.gain.real) * trunc_params["data_fixed_precision"]
        )
        assert np.all(
            np.abs(frame.gain.imag - frame_t.gain.imag)
            <= np.abs(frame.gain.imag) * trunc_params["data_fixed_precision"]
        )

        # test if RMSE of vis (normalised to variance) is within 5 sigma
        # of expected truncation error std deviation sqrt(err_sq_lim / 3)
        rmse = np.sqrt(
            np.mean(np.abs((frame.vis - frame_t.vis)) ** 2 * np.abs(frame.weight))
        )
        expected_rmse = np.sqrt(trunc_params["err_sq_lim"] / 3.0)
        five_sigma = 5 * expected_rmse / np.sqrt(len(frame.vis))
        assert np.all(np.abs(rmse - expected_rmse) < five_sigma)


def test_zero_weights(vis_data_zero_weights):
    n = trunc_params["num_elements"]

    for frame_t, frame in zip(vis_data_zero_weights[0], vis_data_zero_weights[1]):
        assert np.any(frame.vis != frame_t.vis)
        for i in range(0, int(n * (n + 1) * 0.5)):
            assert (
                np.abs(frame.vis[i].real - frame_t.vis[i].real)
                <= np.abs(frame.vis[i].real) * trunc_params["data_fixed_precision"]
            )
            assert (
                np.abs(frame.vis[i].imag - frame_t.vis[i].imag)
                <= np.abs(frame.vis[i].imag) * trunc_params["data_fixed_precision"]
            )
        assert np.all(frame.weight == frame_t.weight)
        assert np.all(frame.weight == 0.0)
        assert np.all(
            np.abs(frame.evec.real - frame_t.evec.real)
            <= np.abs(frame.evec.real) * trunc_params["data_fixed_precision"]
        )
        assert np.all(
            np.abs(frame.evec.imag - frame_t.evec.imag)
            <= np.abs(frame.evec.imag) * trunc_params["data_fixed_precision"]
        )
