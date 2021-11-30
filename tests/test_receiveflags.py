# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import pytest
import numpy as np
import pdb
import subprocess
from requests.exceptions import ConnectionError
import time

from kotekan import visbuffer
from kotekan import runner

params = {"num_kept_updates": 1}

global_params = {
    "wait": True,
    "dataset_manager": {"use_dataset_broker": False},
    "fakevis_mode": "fill_ij",
    "cadence": 1.0,
    "total_frames": 20,  # make it run long enough to receive REST commands
    "cadence": 0.1,
    "num_ev": 0,
    "num_elements": 5,
    "out_file": "/tmp/out.csv",
    "buffer_depth": 100,
    "some_static_part_of_the_config": 0,
    "dynamic_attributes": {
        "flagging": {
            "kotekan_update_endpoint": "json",
            "bad_inputs": [1, 4],
            "start_time": time.time(),
            "update_id": "initial_test_flags",
        }
    },
    "updatable_config": "/dynamic_attributes/flagging",
}

start_time = time.time()


def run_flagging(tmpdir_factory, cmds):
    """Receive Flags"""

    tmpdir = tmpdir_factory.mktemp("receiveflags")

    fakevis_buffer = runner.FakeVisBuffer(
        num_frames=global_params["total_frames"],
        mode=global_params["fakevis_mode"],
        cadence=global_params["cadence"],
        wait=global_params["wait"],
    )

    out_dump_buffer = runner.DumpVisBuffer(str(tmpdir))

    test = runner.KotekanStageTester(
        "ReceiveFlags",
        params,
        buffers_in=fakevis_buffer,
        buffers_out=out_dump_buffer,
        global_config=global_params,
        rest_commands=cmds,
        expect_failure=True,
    )

    test.run()

    return out_dump_buffer.load()


################################################################################
##                                                                            ##
## All following tests just send an old timestamp, effectively ignoring the   ##
## timestamp argument.                                                        ##
##                                                                            ##
################################################################################


def test_clear_flags(tmpdir_factory):
    n = global_params["num_elements"]
    num_prod = n * (n + 1) // 2
    flags_set = False

    # REST commands
    flags = []
    cmds = [
        [
            "post",
            "dynamic_attributes/flagging",
            {
                "bad_inputs": flags,
                "start_time": start_time,
                "update_id": "test_flag_update",
            },
        ]
    ]

    flags_dump = run_flagging(tmpdir_factory, cmds)

    for frame in flags_dump:
        assert frame.metadata.num_prod == num_prod

        assert n == len(frame.flags)

        # at some point the flags will change
        if not np.all(frame.flags == [1, 0, 1, 1, 0]):
            flags_set = True

        if flags_set:
            # from now on all flags will be set
            assert [1, 1, 1, 1, 1] == pytest.approx(frame.flags)


def test_too_many_flags(tmpdir_factory):
    n = global_params["num_elements"]

    # REST commands
    flags = [0, 1, 2, 3, 4, 5]
    cmds = [
        [
            "post",
            "dynamic_attributes/flagging",
            {
                "bad_inputs": flags,
                "start_time": start_time,
                "update_id": "test_flag_update",
            },
        ]
    ]

    flags_dump = run_flagging(tmpdir_factory, cmds)

    for frame in flags_dump:
        assert n == len(frame.flags)

        # flags should always stay at the initialization value
        assert np.all([1, 0, 1, 1, 0] == pytest.approx(frame.flags))


def test_one_flag(tmpdir_factory):
    global_params["num_elements"] = 1
    n = global_params["num_elements"]
    flags_set = False

    # REST commands
    flags = [0]
    cmds = [
        [
            "post",
            "dynamic_attributes/flagging",
            {
                "bad_inputs": flags,
                "start_time": start_time,
                "update_id": "test_flag_update",
            },
        ]
    ]
    global_params["dynamic_attributes"]["flagging"]["bad_inputs"] = []

    flags_dump = run_flagging(tmpdir_factory, cmds)

    for frame in flags_dump:
        assert n == len(frame.flags)

        # at some point the flags will change
        if not np.all(frame.flags == [1]):
            flags_set = True

        if flags_set:
            # from now on all flags will be set
            assert np.all([0] == pytest.approx(frame.flags))

    assert flags_set


def test_out_of_bounds_msg_flag(tmpdir_factory):
    global_params["num_elements"] = 1
    n = global_params["num_elements"]

    # REST commands
    flags = [2]
    cmds = [
        [
            "post",
            "dynamic_attributes/flagging",
            {
                "bad_inputs": flags,
                "start_time": start_time,
                "update_id": "test_flag_update",
            },
        ]
    ]
    global_params["dynamic_attributes"]["flagging"]["bad_inputs"] = [0]

    flags_dump = run_flagging(tmpdir_factory, cmds)

    for frame in flags_dump:
        assert n == len(frame.flags)

        # flaggs should always stay at the initialization value
        assert np.all([0] == pytest.approx(frame.flags))


def test_flags_data_type(tmpdir_factory):
    global_params["num_elements"] = 4
    n = global_params["num_elements"]

    # REST commands
    flags = ["false_flag"] * 4
    cmds = [
        [
            "post",
            "dynamic_attributes/flagging",
            {
                "bad_inputs": flags,
                "start_time": global_params["dynamic_attributes"]["flagging"][
                    "start_time"
                ],
                "update_id": "test_flag_update",
            },
        ]
    ]
    global_params["dynamic_attributes"]["flagging"]["bad_inputs"] = [0, 1]

    flags_dump = run_flagging(tmpdir_factory, cmds)

    for frame in flags_dump:
        assert 4 == len(frame.flags)

        # flags shouldn't change
        assert np.all([0, 0, 1, 1] == pytest.approx(frame.flags))


def test_flags_extra_arguments(tmpdir_factory):
    global_params["num_elements"] = 4

    # REST commands
    flags = [0, 1]
    cmds = [
        [
            "post",
            "dynamic_attributes/flagging",
            {
                "bad_inputs": flags,
                "foo": "bar",
                "start_time": global_params["dynamic_attributes"]["flagging"][
                    "start_time"
                ],
                "update_id": "test_flag_update",
            },
        ]
    ]
    global_params["dynamic_attributes"]["flagging"]["bad_inputs"] = [1, 2]

    flags_dump = run_flagging(tmpdir_factory, cmds)

    for frame in flags_dump:
        assert 4 == len(frame.flags)

        # flags shouldn't change
        assert np.all([1, 0, 0, 1] == pytest.approx(frame.flags))


def test_flags_wrong_argument(tmpdir_factory):
    global_params["num_elements"] = 4

    # REST commands
    flags = [0]
    cmds = [
        [
            "post",
            "dynamic_attributes/flagging",
            {
                "flags_with_a_typo": flags,
                "start_time": global_params["dynamic_attributes"]["flagging"][
                    "start_time"
                ],
                "update_id": "test_flag_update",
            },
        ]
    ]
    global_params["dynamic_attributes"]["flagging"]["bad_inputs"] = [0, 1]

    flags_dump = run_flagging(tmpdir_factory, cmds)

    for frame in flags_dump:
        assert 4 == len(frame.flags)

        # flags shouldn't change
        assert np.all([0, 0, 1, 1] == pytest.approx(frame.flags))


def test_flags_no_argument(tmpdir_factory):
    global_params["num_elements"] = 4

    # REST commands
    cmds = [["post", "dynamic_attributes/flagging", {}]]
    global_params["dynamic_attributes"]["flagging"]["bad_inputs"] = [0, 1]

    flags_dump = run_flagging(tmpdir_factory, cmds)

    for frame in flags_dump:
        assert 4 == len(frame.flags)

        # flags shouldn't change
        assert np.all([0, 0, 1, 1] == pytest.approx(frame.flags))


################################################################################
##                                                                            ##
## The following test is testing timestamps as well.                          ##
##                                                                            ##
################################################################################


def test_start_time(tmpdir_factory):
    global_params["total_frames"] = 200
    params["num_kept_updates"] = 5
    global_params["cadence"] = 0.1
    global_params["wait"] = True
    global_params["num_elements"] = 5
    global_params["dynamic_attributes"]["flagging"]["bad_inputs"] = [1, 4]
    flags_set = 0

    # REST commands
    flags = [
        global_params["dynamic_attributes"]["flagging"]["bad_inputs"],
        [1],
        [2],
        [3],
    ]
    frame_flags = [[1, 0, 1, 1, 0], [1, 0, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 0, 1]]
    ts = [time.time() + 15, time.time() + 17, time.time() + 18.5]
    cmds = [
        [
            "post",
            "dynamic_attributes/flagging",
            {
                "bad_inputs": flags[1],
                "start_time": ts[0],
                "update_id": "test_flag_update1",
            },
        ],
        [
            "post",
            "dynamic_attributes/flagging",
            {
                "bad_inputs": flags[2],
                "start_time": ts[1],
                "update_id": "test_flag_update2",
            },
        ],
        [
            "post",
            "dynamic_attributes/flagging",
            {
                "bad_inputs": flags[3],
                "start_time": ts[2],
                "update_id": "test_flag_update3",
            },
        ],
    ]

    flags_dump = run_flagging(tmpdir_factory, cmds)

    for frame in flags_dump:
        assert global_params["num_elements"] == len(frame.flags)

        # we know when the flags should change in this test
        frame_ts = frame.metadata.ctime.tv + frame.metadata.ctime.tv_nsec * 1e-9
        if flags_set < 3 and frame_ts >= ts[flags_set]:
            flags_set += 1

        assert np.all(frame_flags[flags_set] == pytest.approx(frame.flags))

    assert flags_set == 3


def test_start_time_out_of_order(tmpdir_factory):
    global_params["total_frames"] = 200
    params["num_kept_updates"] = 5
    global_params["cadence"] = 0.1
    global_params["num_elements"] = 5
    global_params["dynamic_attributes"]["flagging"]["bad_inputs"] = [1, 4]
    flags_set = 0

    # REST commands
    flags = [
        global_params["dynamic_attributes"]["flagging"]["bad_inputs"],
        [1],
        [2],
        [3],
    ]
    frame_flags = [[1, 0, 1, 1, 0], [1, 0, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 0, 1]]
    ts = [time.time() + 15, time.time() + 17, time.time() + 18.5]

    # reorder start_times
    cmds = [
        [
            "post",
            "dynamic_attributes/flagging",
            {
                "bad_inputs": flags[1],
                "start_time": ts[2],
                "update_id": "test_flag_update1",
            },
        ],
        [
            "post",
            "dynamic_attributes/flagging",
            {
                "bad_inputs": flags[2],
                "start_time": ts[0],
                "update_id": "test_flag_update2",
            },
        ],
        [
            "post",
            "dynamic_attributes/flagging",
            {
                "bad_inputs": flags[3],
                "start_time": ts[1],
                "update_id": "test_flag_update3",
            },
        ],
    ]
    frame_flags = [frame_flags[i] for i in [0, 2, 3, 1]]

    flags_dump = run_flagging(tmpdir_factory, cmds)

    for frame in flags_dump:
        assert global_params["num_elements"] == len(frame.flags)

        # we know when the flags should change in this test
        frame_ts = frame.metadata.ctime.tv + frame.metadata.ctime.tv_nsec * 1e-9
        if flags_set < 3 and frame_ts >= ts[flags_set]:
            flags_set += 1

        assert np.all(frame_flags[flags_set] == pytest.approx(frame.flags))

    assert flags_set == 3


def test_start_time_new_update(tmpdir_factory):
    global_params["total_frames"] = 200
    params["num_kept_updates"] = 5
    global_params["cadence"] = 0.1
    global_params["num_elements"] = 5
    flags_set = 0
    global_params["dynamic_attributes"]["flagging"]["bad_inputs"] = []

    # REST commands
    flags = [
        global_params["dynamic_attributes"]["flagging"]["bad_inputs"],
        [1],
        [2],
        [3],
    ]
    frame_flags = [[1, 1, 1, 1, 1], [1, 0, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 0, 1]]
    ts = [time.time() + 15, time.time() + 15, time.time() + 15]
    cmds = [
        [
            "post",
            "dynamic_attributes/flagging",
            {
                "bad_inputs": flags[1],
                "start_time": ts[0],
                "update_id": "test_flag_update1",
            },
        ],
        ["wait", 0.1, None],
        [
            "post",
            "dynamic_attributes/flagging",
            {
                "bad_inputs": flags[2],
                "start_time": ts[1],
                "update_id": "test_flag_update2",
            },
        ],
        ["wait", 0.1, None],
        [
            "post",
            "dynamic_attributes/flagging",
            {
                "bad_inputs": flags[3],
                "start_time": ts[2],
                "update_id": "test_flag_update3",
            },
        ],
        ["wait", 0.1, None],
        [
            "post",
            "dynamic_attributes/flagging",
            {
                "bad_inputs": flags[3],
                "start_time": ts[0],
                "update_id": "test_flag_update4",
            },
        ],
    ]
    flags_dump = run_flagging(tmpdir_factory, cmds)

    for frame in flags_dump:
        assert global_params["num_elements"] == len(frame.flags)

        # we know when the flags should change in this test
        frame_ts = frame.metadata.ctime.tv + frame.metadata.ctime.tv_nsec * 1e-9

        if flags_set < 1 and frame_ts >= ts[flags_set]:
            flags_set = 3

        assert np.all(frame_flags[flags_set] == pytest.approx(frame.flags))

    assert flags_set == 3


def test_flags_wrong_type(tmpdir_factory):
    global_params["total_frames"] = 20
    global_params["cadence"] = 0.1
    global_params["dynamic_attributes"]["flagging"]["bad_inputs"] = [1, 4]
    global_params["num_elements"] = 5
    n = global_params["num_elements"]
    num_prod = n * (n + 1) // 2
    flags_set = False

    # REST commands
    flags = [2, 3]
    cmds = [
        [
            "post",
            "dynamic_attributes/flagging",
            {
                "bad_inputs": start_time,
                "start_time": start_time,
                "update_id": "test_flag_update",
            },
        ]
    ]

    flags_dump = run_flagging(tmpdir_factory, cmds)

    for frame in flags_dump:
        assert frame.metadata.num_prod == num_prod

        assert n == len(frame.flags)

        # flags should stay the same
        assert np.all(frame.flags == [1, 0, 1, 1, 0])


def test_dset_id_change(tmpdir_factory):
    global_params["total_frames"] = 200
    params["num_kept_updates"] = 5
    global_params["cadence"] = 0.1
    global_params["wait"] = True
    global_params["num_elements"] = 5
    global_params["dynamic_attributes"]["flagging"]["bad_inputs"] = [1, 4]
    flags_set = 0

    # REST commands
    flags = [
        global_params["dynamic_attributes"]["flagging"]["bad_inputs"],
        [1],
        [2],
        [3],
    ]
    frame_flags = [[1, 0, 1, 1, 0], [1, 0, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 0, 1]]
    ts = [time.time() + 15, time.time() + 17, time.time() + 18.5]
    cmds = [
        [
            "post",
            "dynamic_attributes/flagging",
            {
                "bad_inputs": flags[1],
                "start_time": ts[0],
                "update_id": "test_flag_update1",
            },
        ],
        [
            "post",
            "dynamic_attributes/flagging",
            {
                "bad_inputs": flags[2],
                "start_time": ts[1],
                "update_id": "test_flag_update2",
            },
        ],
        [
            "post",
            "dynamic_attributes/flagging",
            {
                "bad_inputs": flags[3],
                "start_time": ts[2],
                "update_id": "test_flag_update3",
            },
        ],
    ]

    flags_dump = run_flagging(tmpdir_factory, cmds)

    def get_dset_id(frame):
        return bytes(frame.metadata.dataset_id)[::-1].hex()

    prev_dset_id = get_dset_id(flags_dump[0])

    for frame in flags_dump:
        assert global_params["num_elements"] == len(frame.flags)

        dset_id = get_dset_id(frame)

        # we know when the flags should change in this test
        frame_ts = frame.metadata.ctime.tv + frame.metadata.ctime.tv_nsec * 1e-9
        if flags_set < 3 and frame_ts >= ts[flags_set]:
            # Check that the dataset_id has changed when we expected it to
            assert prev_dset_id != dset_id
            flags_set += 1
        else:
            assert prev_dset_id == dset_id

        prev_dset_id = dset_id

    assert flags_set == 3
