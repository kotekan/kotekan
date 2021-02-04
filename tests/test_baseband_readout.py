# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

from dataclasses import dataclass
import os
import glob
import random
import pytest

import numpy as np
import h5py

from kotekan import runner

# Skip if HDF5 support not built into kotekan
if not runner.has_hdf5():
    pytest.skip("HDF5 support not available.", allow_module_level=True)


def is_docker():
    path = "/proc/self/cgroup"
    return (
        os.path.exists("/.dockerenv")
        or os.path.isfile(path)
        and any("docker" in line for line in open(path))
    )


if is_docker():
    pytest.skip("Does not work in Github Actions docker run.", allow_module_level=True)

default_params = {
    "max_dump_samples": 3500,
    "num_elements": 256,
    "total_frames": 60,
    "stream_id": 0,
    "buffer_depth": 20,
    "num_frames_buffer": 18,
    "type": "tpluse",
    "value": 153,
    "samples_per_data_set": 1024,
    "rest_mode": "step",
    "baseband_metadata_pool": {
        "kotekan_metadata_pool": "BasebandMetadata",
        "num_metadata_objects": 4096,
    },
}


@dataclass(frozen=False)
class EventDump:
    """Convenience class for reconstructing a complete baseband event from dumped BasebandBuffer"""

    event_id: int
    freq_id: int
    fpga_start_seq: int
    fpga_length: int

    @classmethod
    def from_metadata(cls, metadata):
        """Create an EventDump instance from the event's first dumped frame"""
        event_id = metadata.event_id
        freq_id = metadata.freq_id
        fpga_seq = metadata.fpga_seq
        valid_from = metadata.valid_from
        valid_to = metadata.valid_to
        data_start = fpga_seq + valid_from
        data_length = valid_to - valid_from
        return cls(event_id, freq_id, data_start, data_length)

    def extend(self, metadata):
        """Increase the duration of the event's data with another dumped frame
        Assumes that the dumped frames record continguous samples, with no holes caused by the output buffer's filling up.
        """
        fpga_seq = metadata.fpga_seq
        valid_from = metadata.valid_from
        valid_to = metadata.valid_to
        data_start = fpga_seq + valid_from
        data_length = valid_to - valid_from
        self.fpga_length += data_length


DATAGEN_PNAME = "fakenetwork"


def command_rest_frames(num_frames):
    return ("post", DATAGEN_PNAME + "/generate_test_data", {"num_frames": num_frames})


def command_trigger(start, length, event_id=123456, file_path="", dm=0, dm_error=0):
    if start < 0:
        start_unix_seconds = start
        start_unix_nano = 0
    else:
        start_unix_seconds = 0
        start_unix_nano = start * 2560

    data = {
        "event_id": event_id,
        "start_unix_seconds": start_unix_seconds,
        "start_unix_nano": start_unix_nano,
        "duration_nano": length * 2560,
        "dm": dm,
        "dm_error": dm_error,
        "file_path": file_path,
    }
    return ("post", "baseband", data)


def wait(wait_time):
    return ("wait", wait_time, None)


def run_baseband(tdir_factory, params=None, rest_commands=None, expect_a_failure=False):

    p = dict(default_params)
    tmpdir = tdir_factory.mktemp("baseband")

    if params:
        p.update(params)

    fake_buffer = runner.FakeNetworkBuffer(
        stage_name=DATAGEN_PNAME, num_frames=p["total_frames"], type=p["type"]
    )

    write_buffer = runner.DumpBasebandBuffer(str(tmpdir))
    test = runner.KotekanStageTester(
        "basebandReadout",
        {},
        fake_buffer,
        write_buffer,
        p,
        rest_commands=rest_commands,
        expect_failure=expect_a_failure,
    )

    test.run()

    dump_files = glob.glob(str(tmpdir) + "/*.h5")
    return write_buffer.load()


def collect_dumped_events(dump_frames):
    """Reconstructs a list of dumped BasebandBuffer frames into a list of `EventDump`s"""
    dumped_events = []
    for frame in dump_frames:
        event_id = frame.metadata.event_id
        if not dumped_events or dumped_events[-1].event_id != event_id:
            # start a new event
            dumped_events.append(EventDump.from_metadata(frame.metadata))
        else:
            # extend an existing one
            event = dumped_events[-1]
            event.extend(frame.metadata)
    return dumped_events


def test_max_samples(tmpdir_factory):
    """Test that the baseband dump length is truncated to max_dump_samples parameter"""
    rest_commands = [
        command_rest_frames(1),
        wait(0.5),
        command_rest_frames(5),
        command_trigger(1000, 3237),
        wait(0.1),
        command_rest_frames(20),
        # Give it some time to write the capture before shutdown.
        wait(0.5),
        command_rest_frames(5),
    ]
    params = {"total_frames": 30, "max_dump_samples": 2123}
    dump_frames = run_baseband(tmpdir_factory, params, rest_commands)
    dumped_events = collect_dumped_events(dump_frames)
    assert len(dumped_events) == 1

    dumped_event = dumped_events[0]
    assert dumped_event.fpga_start_seq == 1000
    assert dumped_event.fpga_length == params["max_dump_samples"]


def test_negative_start_time(tmpdir_factory):
    """Test using the 'save whatever you have' mode of the baseband dump

    Using -1 as the trigger start point initiates the dump using the oldest
    frame available in the buffers.
    """

    rest_commands = [
        command_rest_frames(1),
        wait(0.5),
        command_trigger(-1, 3237, 31),
        wait(0.1),
        command_rest_frames(25),
        # Give it some time to write the capture before shutdown.
        wait(1.0),
        command_rest_frames(5),
    ]
    params = {"total_frames": 30}
    dump_frames = run_baseband(tmpdir_factory, params, rest_commands)
    dumped_events = collect_dumped_events(dump_frames)
    assert len(dumped_events) == 1

    dumped_event = dumped_events[0]
    assert dumped_event.event_id == 31
    assert dumped_event.freq_id == 0
    assert dumped_event.fpga_start_seq == 0
    assert dumped_event.fpga_length == 3237


def test_basic(tmpdir_factory):

    rest_commands = [
        command_rest_frames(1),
        wait(0.5),
        command_rest_frames(5),
        command_trigger(1437, 1839, 10),
        wait(0.3),
        command_rest_frames(5),
        command_trigger(3457, 1237, 17),
        wait(0.3),
        command_rest_frames(5),
        command_trigger(5039, 1091, 31),
        wait(0.1),
        command_rest_frames(60),
    ]

    dump_frames = run_baseband(tmpdir_factory, {}, rest_commands)
    dumped_events = collect_dumped_events(dump_frames)

    baseband_requests = [
        cmd for cmd in rest_commands if cmd[0] == "post" and cmd[1] == "baseband"
    ]
    for i, event in enumerate(dumped_events):
        assert event.event_id == baseband_requests[i][2]["event_id"]
        assert event.freq_id == 0
        assert event.fpga_start_seq * 2560 == baseband_requests[i][2]["start_unix_nano"]
        assert event.fpga_length * 2560 == baseband_requests[i][2]["duration_nano"]


def test_missed(tmpdir_factory):

    good_trigger = (2437, 3123)
    rest_commands = [
        command_rest_frames(21),
        wait(0.5),
        command_trigger(*good_trigger),  # Catch part of this one.
        command_rest_frames(30),
        command_trigger(100, 100),  # Miss this one.
        command_trigger(1002, 112),  # Miss this one.
        command_trigger(1001, 300),  # Miss this one.
        command_trigger(81039, 7091),  # This one never arrives.
        command_rest_frames(10),
    ]
    dump_frames = run_baseband(tmpdir_factory, {}, rest_commands)
    dumped_events = collect_dumped_events(dump_frames)
    assert len(dumped_events) == 1

    dumped_event = dumped_events[0]
    assert (
        good_trigger[0]
        <= dumped_event.fpga_start_seq
        <= good_trigger[0] + good_trigger[1]
    )
    assert 0 < dumped_event.fpga_length <= good_trigger[1]


def test_overload_no_crash(tmpdir_factory):

    params = dict(default_params)
    params.update(
        {
            "write_throttle": 2.0,
            "samples_per_data_set": 16384,
            "num_elements": 128,
            "total_frames": 60,
        }
    )

    rest_commands = [command_rest_frames(1), wait(0.5)]
    random.seed()
    tf = params["total_frames"]
    spd = params["samples_per_data_set"]
    n = 30
    for ii in range(n):
        start = random.randrange(1, (ii * tf // n + 20) * spd)
        length = random.randrange(1, spd * 5)
        rest_commands += [command_trigger(start, length, (ii + 1))]
    rest_commands += [command_rest_frames(params["total_frames"])]

    run_baseband(tmpdir_factory, params, rest_commands)
