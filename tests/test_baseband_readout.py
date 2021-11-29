# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

from collections import defaultdict
from dataclasses import dataclass
import os
import glob
import random
import pytest

from kotekan import runner


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
    "telescope": "ICETelescope",
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
    event_start_seq: int
    event_end_seq: int
    num_elements: int
    fpga_start_seq: int
    fpga_length: int

    @classmethod
    def from_metadata(cls, metadata):
        """Create an EventDump instance from the event's first dumped frame"""
        event_id = metadata.event_id
        freq_id = metadata.freq_id
        event_start_seq = metadata.event_start_seq
        event_end_seq = metadata.event_end_seq
        fpga_seq = metadata.frame_fpga_seq
        num_elements = metadata.num_elements
        valid_to = metadata.valid_to
        return cls(
            event_id,
            freq_id,
            event_start_seq,
            event_end_seq,
            num_elements,
            fpga_seq,
            valid_to,
        )

    def extend(self, metadata):
        """Increase the duration of the event's data with another dumped frame
        Enforces that the dumped frames record continguous samples, with no holes caused by the output buffer's filling up.
        """
        # check structure matches
        assert self.event_id == metadata.event_id
        assert self.freq_id == metadata.freq_id
        assert self.event_start_seq == metadata.event_start_seq
        assert self.event_end_seq == metadata.event_end_seq
        assert self.num_elements == metadata.num_elements

        # check samples are continguous
        fpga_seq = metadata.frame_fpga_seq
        assert fpga_seq == self.fpga_start_seq + self.fpga_length

        self.fpga_length += metadata.valid_to


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

    write_buffer = runner.DumpBasebandBuffer(
        str(tmpdir),
        num_frames="buffer_depth * 4",
        frame_size="num_elements * samples_per_data_set / 4",
    )
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

    return write_buffer.load()


def stream_to_freq_id(stream, in_stream_freq_idx, num_freq_per_stream=1):
    """Python re-implementation of ICETelescope::stream_to_freq_id"""
    assert num_freq_per_stream in [1, 8]

    link_id = stream & 0xF
    slot_id = (stream & 0xF0) >> 4
    crate_id = (stream & 0xF00) >> 8
    unused = (stream & 0xF000) >> 12

    if num_freq_per_stream == 1:
        # CHIME: 128 ICEBoards, 2048 elements
        return crate_id * 16 + slot_id + link_id * 32 + unused * 256
    elif num_freq_per_stream == 8:
        # Pathfinder/HIRAX-256: 16 ICEBoards, 256 elements
        return slot_id + link_id * 16 + in_stream_freq_idx * 128


def collect_dumped_events(
    dump_frames, frame_size=default_params["samples_per_data_set"],
):
    """Reconstructs a list of dumped BasebandBuffer frames into a list of `EventDump`s"""
    dumped_events = []
    for frame_no, frame in enumerate(dump_frames):
        event_id = frame.metadata.event_id
        num_elements = frame.metadata.num_elements
        for j, val in enumerate(frame._buffer[frame.meta_size :]):
            if j >= frame.metadata.valid_to * num_elements:
                break
            # calculation used in `testDataGen` for method `tpluse`:
            time_idx = j // num_elements
            elem_idx = j % num_elements
            expected = (frame.metadata.frame_fpga_seq + time_idx + elem_idx) % 256
            assert (
                val == expected
            ), f"Baseband data mismatch at index {j}/{frame_no}, fpga_seq={frame.metadata.frame_fpga_seq}"

        if not dumped_events or dumped_events[-1].event_id != event_id:
            # start a new event
            dumped_events.append(EventDump.from_metadata(frame.metadata))
        else:
            # extend an existing one
            event = dumped_events[-1]
            event.extend(frame.metadata)
    return dumped_events


def collect_dumped_multi_freq_events(
    dump_frames,
    num_freq_per_stream,
    frame_size=default_params["samples_per_data_set"],
    stream_id=0,
):
    """Reconstructs a list of dumped BasebandBuffer frames into a list of `EventDump`s"""
    dumped_events = defaultdict(list)
    for frame_no, frame in enumerate(dump_frames):
        event_id = frame.metadata.event_id
        freq_id = frame.metadata.freq_id
        stream_freq_idx = [
            stream_to_freq_id(stream_id, i, num_freq_per_stream)
            for i in range(num_freq_per_stream)
        ].index(freq_id)
        num_elements = frame.metadata.num_elements
        for j, val in enumerate(frame._buffer[frame.meta_size :]):
            if j >= frame.metadata.valid_to * num_elements:
                break

            # in the case of multi-frequency dumps, we need to reconstruct
            # the "j" to be what it was in the testDataGen stage that was
            # the _input_ to baseband
            orig_fpga_seq = (frame.metadata.frame_fpga_seq // frame_size) * frame_size
            orig_j = (
                (frame.metadata.frame_fpga_seq % frame_size)
                * (num_elements * num_freq_per_stream)
                + stream_freq_idx * num_elements
                + (j // num_elements * (num_freq_per_stream * num_elements))
                + j % num_elements
            )

            # calculation used in `testDataGen` for method `tpluseplusfprime`:
            time_idx = orig_j // (num_freq_per_stream * num_elements)
            elem_idx = orig_j % num_elements
            expected = (
                2 * (orig_fpga_seq + time_idx) + 3 * freq_id + 5 * elem_idx
            ) % 256

            assert (
                val == expected
            ), f"Baseband data mismatch at freq_id={freq_id}, fpga_seq={frame.metadata.frame_fpga_seq} -> {orig_fpga_seq}, index {j}/{frame_no} -> {orig_j}"

        if (
            not dumped_events[freq_id]
            or dumped_events[freq_id][-1].event_id != event_id
        ):
            # if dumped_events[freq_id]:
            #     print(dumped_events[freq_id][-1])
            # start a new event
            dumped_events[freq_id].append(EventDump.from_metadata(frame.metadata))
        else:
            # extend an existing one
            event = dumped_events[freq_id][-1]
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


def test_8_multifreq(tmpdir_factory):
    # Eight frequencies, one stage.
    rest_commands = [
        command_rest_frames(1),  # generate 1 frame = 1024 time samples x 256 feeds
        wait(0.5),  # in seconds?
        command_trigger(
            1437, 1839, 10
        ),  # capture ~1839 time samples starting at t = 3.67 ms(=1437 x 2.56us) with event id=10
        command_trigger(20457, 3237, 17),  # similar to above
        command_trigger(41039, 2091, 31),
        wait(0.1),
        command_rest_frames(60),
    ]
    params = {
        "num_local_freq": 8,
        "type": "tpluseplusfprime",
        "stream_id": 2,
    }
    dump_frames = run_baseband(tmpdir_factory, params, rest_commands)
    dumped_multi_freq_events = collect_dumped_multi_freq_events(
        dump_frames,
        num_freq_per_stream=params["num_local_freq"],
        stream_id=params["stream_id"],
    )
    assert len(dumped_multi_freq_events) == params["num_local_freq"]

    baseband_requests = [
        cmd for cmd in rest_commands if cmd[0] == "post" and cmd[1] == "baseband"
    ]
    for freq_id in dumped_multi_freq_events:
        dumped_events = dumped_multi_freq_events[freq_id]
        for i, event in enumerate(dumped_events):
            assert event.event_id == baseband_requests[i][2]["event_id"]
            assert event.freq_id == freq_id
            assert (
                event.fpga_start_seq * 2560
                == baseband_requests[i][2]["start_unix_nano"]
            )
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
        {"samples_per_data_set": 16384, "num_elements": 128, "total_frames": 60,}
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
