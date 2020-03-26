# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import pytest
import numpy as np
import h5py
from kotekan import runner
from kotekan import visutil
from kotekan import visbuffer
import time
from pytest_localserver.http import WSGIServer
from flask import Flask, jsonify, request as flask_req
import base64

# Skip if HDF5 support not built into kotekan
if not runner.has_hdf5():
    pytest.skip("HDF5 support not available.", allow_module_level=True)

start_time = 1_500_000_000

old_timestamp = start_time - 10.0
new_timestamp = start_time + 5.0

old_update_id = f"gains{old_timestamp}"
new_update_id = f"gains{new_timestamp}"

transition_interval = 10.0

global_params = {
    "num_elements": 16,
    "num_ev": 4,
    "total_frames": 20,
    "start_time": start_time,
    "cadence": 1.0,
    "mode": "fill_ij",
    "freq_ids": [250],
    "buffer_depth": 8,
    "updatable_config": "/gains",
    "gains": {
        "kotekan_update_endpoint": "json",
        "start_time": old_timestamp,
        "update_id": old_update_id,
        "transition_interval": transition_interval,
    },
    "wait": True,
    "sleep_before": 2.0,
    "num_threads": 4,
    "dataset_manager": {"use_dataset_broker": False},
}


def gen_gains(filename, mult_factor, num_elements, freq):
    """Create gain file and return the gains and weights."""

    nfreq = len(freq)
    gain = (
        np.arange(nfreq)[:, None] * 1j * np.arange(num_elements)[None, :] * mult_factor
    ).astype(np.complex64)

    # Make some weights zero to test the behaviour of apply_gains
    weight = np.ones((nfreq, num_elements), dtype=np.bool8)
    weight[:, 1] = False
    weight[:, 3] = False

    with h5py.File(str(filename), "w") as f:

        dset = f.create_dataset("gain", data=gain)

        dset2 = f.create_dataset("weight", data=weight)
        dset2[...] = weight

        freq_ds = f.create_dataset("index_map/freq", (nfreq,), dtype="f")
        ipt_ds = f.create_dataset("index_map/input", (num_elements,), dtype="i")

        freq_ds[...] = freq
        ipt_ds[:] = np.arange(num_elements)

    return gain, weight


def encode_gains(gain, weight):
    # encode base64
    res = {
        "gain": {
            "dtype": "complex64",
            "shape": gain.shape,
            "data": base64.b64encode(gain.tobytes()).decode(),
        },
        "weight": {
            "dtype": "bool",
            "shape": weight.shape,
            "data": base64.b64encode(weight.tobytes()).decode(),
        },
    }
    return res


@pytest.fixture(scope="session")
def gain_path(tmp_path_factory):
    return tmp_path_factory.mktemp("gain")


@pytest.fixture(scope="session")
def old_gains(gain_path):

    # Get the name of the file to write
    fname = str(gain_path / f"{old_update_id}.h5")

    freq = np.linspace(800.0, 400.0, 1024)[global_params["freq_ids"]]

    return gen_gains(fname, 1.0, global_params["num_elements"], freq)


@pytest.fixture(scope="session")
def new_gains(gain_path):

    # Get the name of the file to write
    fname = str(gain_path / f"{new_update_id}.h5")

    freq = np.linspace(800.0, 400.0, 1024)[global_params["freq_ids"]]

    return gen_gains(fname, 2.0, global_params["num_elements"], freq)


@pytest.fixture(scope="session")
def cal_broker(request, old_gains, new_gains):
    # Create a basic flask server
    app = Flask("cal_broker")
    @app.route("/gain", methods=["POST"])
    def gain_app():
        content = flask_req.get_json()
        tag = content["update_id"]
        if tag == new_tag:
            gains = encode_gains(*new_gains)
        elif tag == old_tag:
            gains = encode_gains(*old_gains)
        else:
            raise Exception("Did not recognize tag {}.".format(tag))

        return jsonify(gains)

    # hand to localserver fixture
    server = WSGIServer(application=app)
    server.start()

    yield server

    server.stop()
    #request.addfinalizer(server.stop)


@pytest.fixture(scope="session", params=["file", "network"])
def apply_data(request, tmp_path_factory, gain_path, old_gains, new_gains, cal_broker):

    output_dir = str(tmp_path_factory.mktemp("output"))
    global_params["gains_dir"] = str(gain_path)

    # REST commands
    cmds = [
        [
            "post",
            "gains",
            {
                "update_id": new_update_id,
                "start_time": new_timestamp,
                "transition_interval": transition_interval,
            },
        ]
    ]

    fakevis_buffer = runner.FakeVisBuffer(
        freq_ids=global_params["freq_ids"],
        num_frames=global_params["total_frames"],
        wait=global_params["wait"],
    )

    out_dump_buffer = runner.DumpVisBuffer(output_dir)

    # Configuration for the direct dump of FakeVis's output
    fakevis_dump_conf = {
        "file_name": "fakevis_dump",
        "file_ext": "dump",
        "base_dir": output_dir,
    }

    host, port = cal_broker.server_address
    global_params.update({"broker_host": host, "broker_port": port})

    test = runner.KotekanStageTester(
        "applyGains",
        global_params,
        buffers_in=fakevis_buffer,
        buffers_out=out_dump_buffer,
        global_config=global_params,
        rest_commands=cmds,
        parallel_stage_type="rawFileWrite",
        parallel_stage_config=fakevis_dump_conf,
        read_from_file=(True if request.params == "file" else False),
    )

    test.run()

    in_dump = visbuffer.VisBuffer.load_files(f"{output_dir}/*fakevis_dump*.dump")

    return in_dump, out_dump_buffer.load()


def combine_gains(t_frame, transition_interval, new_ts, old_ts, new_gains, old_gains):
    if t_frame < old_ts:
        raise ValueError("Definitely shouldn't get in here.")
    elif t_frame < new_ts:
        return old_gains
    elif t_frame < new_ts + transition_interval:
        age = t_frame - new_ts
        new_coeff = age / transition_interval
        old_coeff = 1.0 - new_coeff
        return new_coeff * new_gains + old_coeff * old_gains
    else:
        return new_gains


def to_triangle(a, b):
    """Multiply a and b against each other and return the upper triange of the result."""
    return np.outer(a, b)[np.triu_indices(len(a))]


def test_metadata(apply_data):
    """Check that the stable metadata has not changed."""

    for input_frame, output_frame in zip(*apply_data):

        assert input_frame.metadata.freq_id == output_frame.metadata.freq_id
        assert input_frame.metadata.fpga_seq == output_frame.metadata.fpga_seq
        assert visutil.ts_to_double(input_frame.metadata.ctime) == visutil.ts_to_double(
            output_frame.metadata.ctime
        )


def test_eigen(apply_data):
    """Check that the eigensector has not been changed."""

    for input_frame, output_frame in zip(*apply_data):

        assert input_frame.erms == output_frame.erms
        assert (input_frame.eval == output_frame.eval).all()
        assert (input_frame.evec == output_frame.evec).all()


def test_gain(apply_data, old_gains, new_gains):
    """Check that the gains have been set and applied correctly.

    This reimplements the gain combination algorithm within python for testing.
    """

    # Preload the gain arrays
    old_gain_arr, _ = old_gains
    new_gain_arr, weight_arr = new_gains

    for input_frame, output_frame in zip(*apply_data):

        freq_ind = global_params["freq_ids"].index(output_frame.metadata.freq_id)

        # Select the current frequency
        old_gain = old_gain_arr[freq_ind]
        new_gain = new_gain_arr[freq_ind]
        gain_weight = weight_arr[freq_ind]

        # Combine the gains together
        output_frame_timestamp = visutil.ts_to_double(output_frame.metadata.ctime)
        gains = combine_gains(
            output_frame_timestamp,
            transition_interval,
            new_timestamp,
            old_timestamp,
            new_gain,
            old_gain,
        )

        # Construct the weight factors and deal with zero gains and weights
        zero_weight = np.logical_or(gain_weight == 0.0, gains == 0.0)
        gains = np.where(zero_weight, 1.0, gains)
        weight_factor = np.where(zero_weight, 0.0, 1.0 / np.abs(gains) ** 2)

        # Construct the expected visibility values. At the moment this is specific to the pattern
        exp_vis = input_frame.vis * to_triangle(gains, np.conj(gains))

        # Construct the expected weights for this frame
        exp_weight = input_frame.weight * to_triangle(weight_factor, weight_factor)

        assert output_frame.gain == pytest.approx(gains)
        assert output_frame.vis[:] == pytest.approx(exp_vis)
        assert output_frame.weight == pytest.approx(exp_weight)


def test_dataset_ids(apply_data):
    """Check that the dataset IDs have been changed from the input and that
    they change when the gain update is applied."""

    old_ds_id = None
    new_ds_id = None

    def dataset_id(frame):
        return bytes(frame.metadata.dataset_id)[::-1].hex()

    for input_frame, output_frame in zip(*apply_data):

        ds_id_in = dataset_id(input_frame)
        ds_id_out = dataset_id(output_frame)
        frame_timestamp = visutil.ts_to_double(output_frame.metadata.ctime)

        # Check that the dataset ID has been changed
        assert ds_id_in != ds_id_out

        if old_ds_id is None:
            old_ds_id = ds_id_out
        # Check that the old dataset IDs are identical before the gain update
        if frame_timestamp < new_timestamp:
            assert ds_id_out == old_ds_id

        # Check that the new dataset IDs are identical after the gain update,
        # and are different to the ones before
        if frame_timestamp >= new_timestamp:
            if new_ds_id is None:
                new_ds_id = ds_id_out
            assert ds_id_out != old_ds_id
            assert ds_id_out == new_ds_id
