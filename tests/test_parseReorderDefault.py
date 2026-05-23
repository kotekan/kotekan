import pytest
import numpy as np
import numpy.random as random
import h5py

from kotekan import runner

if not runner.has_hdf5():
    pytest.fail("HDF5 support not available; unable to run tests!")

parseReorderDefault_params = {
    "out_buf": "reorder_buffer",
    "input_order": "CHIMECorrelator",
    "output_order": "CHIMEBeamformer",
    "name": "scatter_indices",
}

global_params = {
    "type": "config",
    "log_level": "INFO",
    "cpu_affinity": [0, 1],
    "num_elements": 2048,
    "num_dishes": 1024,
    "num_polarizations": 2,
    "num_cylinders": 4,
    # KotekanStageTester adds a "main_pool" section
    # Buffers
    "reorder_buffer": {
        "kotekan_buffer": "standard",
        "num_frames": 1,
        "frame_size": 2048 * 4,
        "metadata_pool": "main_pool",
    },
    # dump to disk to inspect manually
    "write_reorder_buffer": {
        "kotekan_stage": "hdf5FileWrite",
        "base_dir": None,  # set by Python code
        "prefix_hostname": False,
        "max_frames": "1",
        "file_name": "reorder_buffer",
        "in_buf": "reorder_buffer",
    },
    "input_reorder": None,  # set by Python code
}


@pytest.fixture(scope="module")
def scatter_indices(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("parseReorderDefault")

    global_params["write_reorder_buffer"]["base_dir"] = str(tmpdir)
    # file name of produced file
    fname = (
        global_params["write_reorder_buffer"]["base_dir"]
        + "/"
        + global_params["write_reorder_buffer"]["file_name"]
        + ".00000000.h5"
    )

    # make up some dummy data
    seed = 17
    rnd = random.Generator(random.PCG64(seed))

    station_ids = list(range(global_params["num_elements"]))
    adc_ids = [int(id) for id in rnd.permuted(station_ids)]
    feed_labels = ["FCC%06d" % id for id in rnd.permuted(station_ids)]

    input_reorder = list(zip(adc_ids, station_ids, feed_labels))
    global_params["input_reorder"] = input_reorder

    test = runner.KotekanStageTester(
        "parseReorderDefault", parseReorderDefault_params, None, None, global_params
    )
    test.run()

    yield h5py.File(fname, "r")["reorder_buffer"], station_ids, adc_ids


def test_scatter_indices(scatter_indices):

    reorder_buffer, station_ids, adc_ids = scatter_indices

    # metadata and array description
    assert reorder_buffer.attrs["name"] == "scatter_indices"
    assert np.all(reorder_buffer.shape == (2, 1024))
    assert np.all(reorder_buffer.attrs["dim_names"] == ("P", "D"))
    assert reorder_buffer.attrs["type"] == "int32"
    assert reorder_buffer.dtype == np.int32

    # payload
    mapping = np.empty_like(reorder_buffer)
    mapping[:] = -1
    for el in range(2048):
        adc_id = adc_ids[el]
        station_id = station_ids[el]
        # beamformer order from station_id (cylinder order index)
        cylinder = station_id // 512
        polarization = (station_id % 512) // 256
        dish = (station_id % 512) % 256
        mapping[adc_id // 1024][adc_id % 1024] = (
            polarization * 1024 + cylinder * 256 + dish
        )
    assert np.all(mapping != -1)
    assert np.all(mapping == reorder_buffer)
