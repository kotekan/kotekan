"""Validate setScatterIndices against a config-declared ndarray buffer.

Covers the framedesc conversion: the stage no longer *sets* its output
descriptor but *requires* (validates) the one the bufferFactory attaches from
an `kotekan_buffer: ndarray` declaration. The negative case confirms
require_frame_desc actually rejects a structurally-wrong declaration (a
transposed shape that has the same byte size) -- the property the
order-sensitive fengine conversions rely on.
"""
import pytest
import numpy as np
import h5py

from kotekan import runner

if not runner.has_hdf5():
    pytest.fail("HDF5 support not available; unable to run tests!")

NUM_DISHES = 8
NUM_POL = 2

stage_params = {
    "scatter_indices_name": "scatter_buffer",
    "num_dishes": NUM_DISHES,
    "num_polarizations": NUM_POL,
    "scatter_indices": list(range(NUM_POL * NUM_DISHES)),
}


def _global(extents, dimnames, tmpdir):
    return {
        "type": "config",
        "log_level": "INFO",
        "num_dishes": NUM_DISHES,
        "num_polarizations": NUM_POL,
        "scatter_buffer": {
            "kotekan_buffer": "ndarray",
            "num_frames": 1,
            "value_type": "int32",
            "quantity_name": "scatter_indices",
            "extents": extents,
            "dimnames": dimnames,
            "metadata_pool": "main_pool",
        },
        "write_scatter": {
            "kotekan_stage": "hdf5FileWrite",
            "base_dir": str(tmpdir),
            "prefix_hostname": False,
            "max_frames": "1",
            "file_name": "scatter_buffer",
            "in_buf": "scatter_buffer",
            "input_order": "CHIMEBeamformer",
        },
    }


def test_setScatterIndices_validates_matching_ndarray(tmpdir_factory):
    """Correct ndarray declaration: require_frame_desc passes, payload is written."""
    tmpdir = tmpdir_factory.mktemp("setScatterIndices_ok")
    g = _global(["num_polarizations", "num_dishes"], ["P", "D"], tmpdir)
    runner.KotekanStageTester("setScatterIndices", stage_params, None, None, g).run()

    buf = h5py.File(str(tmpdir) + "/scatter_buffer.00000000.h5", "r")["scatter_buffer"]
    assert buf.attrs["name"] == "scatter_indices"
    assert tuple(buf.shape) == (NUM_POL, NUM_DISHES)
    assert tuple(buf.attrs["dim_names"]) == ("P", "D")
    assert buf.attrs["type"] == "int32"
    assert np.all(np.asarray(buf).flatten() == np.arange(NUM_POL * NUM_DISHES))


def test_setScatterIndices_rejects_transposed_ndarray(tmpdir_factory):
    """Same byte size, transposed shape -> require_frame_desc must fatal."""
    tmpdir = tmpdir_factory.mktemp("setScatterIndices_bad")
    g = _global(["num_dishes", "num_polarizations"], ["D", "P"], tmpdir)
    runner.KotekanStageTester(
        "setScatterIndices", stage_params, None, None, g, expect_failure=True
    ).run()
