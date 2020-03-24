# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import glob
import itertools
import pytest
from os import path
from kotekan import runner
from kotekan.frbbuffer import FrbPacket

global_params = {
    "buffer_depth": 7,
    # telescope parameters
    "num_elements": 2048,
    "num_local_freq": 1,
    "num_data_sets": 1,
    "samples_per_data_set": 49152,
    "num_gpus": 4,
    # Pulsar parameters
    "feed_sep_NS": 0.3048,
    "feed_sep_EW": 22.0,
    "num_beams": 10,
    "num_pol": 2,
    "timesamples_per_pulsar_packet": 625,
    "udp_pulsar_packet_size": 5032,
    "num_packet_per_stream": 80,
    "num_stream": 10,
    # constants
    "sizeof_float": 4,
    "sizeof_short": 2,
}


def psr_post_process_data(tmpdir_factory, missing_frames=None, test_env=None):
    """Runs the Kotekan pulsarPostProcess stage, optionally dropping frames

    If there are any frames to drop, the config will include a
    TestDropFrames stage, and drop the specified frame from the input
    to the pulsarPostProcess.
    """

    if test_env is None and not glob.glob(
        path.join(path.dirname(__file__), "data", "fake_psr_beamform_gpu_*.dump")
    ):
        pytest.skip("No test data and not requesting full generation")

    tmpdir = tmpdir_factory.mktemp("psr_post_process")

    lost_samples_buffer = runner.FakeLostSamplesBuffer(num_frames=10)

    dump_buffer = runner.DumpPsrPostProcessBuffer(str(tmpdir))

    input_buffers = [lost_samples_buffer]
    stage_params = {"pulsar_out_buf": dump_buffer.name}
    for i in range(4):
        beamform_buffer = runner.ReadRawBeamformBuffer(
            path.join(path.dirname(__file__), "data"), file_name="fake_psr_beamform_gpu"
        )
        beamform_buffer.buffer_block[beamform_buffer.name][
            "frame_size"
        ] = "samples_per_data_set * num_beams * num_pol * sizeof_float * 2"

        input_buffers.append(beamform_buffer)
        if missing_frames and missing_frames[i]:
            drop_frames_buffer = runner.DropFramesBuffer(
                beamform_buffer,
                missing_frames[i],
                frame_size=beamform_buffer.buffer_block[beamform_buffer.name][
                    "frame_size"
                ],
            )
            input_buffers.append(drop_frames_buffer)
            stage_params[
                "network_input_buffer_{:d}".format(i)
            ] = drop_frames_buffer.name
        else:
            stage_params["network_input_buffer_{:d}".format(i)] = beamform_buffer.name

    params = global_params.copy()

    test = runner.KotekanStageTester(
        "pulsarPostProcess", stage_params, input_buffers, dump_buffer, params
    )

    test.run()

    return dump_buffer


def test_psr_post_process(tmpdir_factory, request):
    test_env = request.config.getoption("-E", None)

    data = psr_post_process_data(tmpdir_factory, test_env=test_env)

    check_data(data)


def test_psr_post_process_missing(tmpdir_factory, request):
    test_env = request.config.getoption("-E", None)

    missing_frames = [[1], [0, 3, 4], [], []]
    data = psr_post_process_data(tmpdir_factory, missing_frames, test_env=test_env)

    check_data(data, missing_frames)


def test_psr_post_process_missing_1(tmpdir_factory, request):
    test_env = request.config.getoption("-E", None)

    missing_frames = [[1], [1], [1], [1]]
    data = psr_post_process_data(tmpdir_factory, missing_frames, test_env=test_env)

    check_data(data, missing_frames)


def check_data(data, missing_frames=[]):
    missing_frames = set(itertools.chain.from_iterable(missing_frames))
    actual_frames = data.load()

    input_fpga_seq_num = 0
    input_frame = 0

    output_fpga_seq_num = 0
    # output_fpga_seq_num = 625
    output_frame_seconds = (
        632718848  # (`0 - ctime(1 Jan 2018 UTC)`, truncated to 30 bits)
    )
    output_frame_nanoseconds = 0
    # output_frame_nanoseconds = 1600000  # 625 elapsed FPGA samples @ 2.56 us / sample

    expected_output = []
    output_in_progress = False
    start_new_stream = True
    while len(expected_output) <= len(actual_frames):
        print("Pass:", input_fpga_seq_num, output_fpga_seq_num)
        # catch up input to where output should start
        if input_fpga_seq_num + 49152 < output_fpga_seq_num:
            input_frame += 1
            input_fpga_seq_num += 49152
            print("Advance input:", input_fpga_seq_num)

        # handle missing frames
        while input_frame in missing_frames:
            if output_in_progress:
                print("Reset output")
                output_in_progress = False
            input_frame += 1
            input_fpga_seq_num += 49152
            print("Skip dropped input to :", input_fpga_seq_num)
            start_new_stream = True

        # if we get to here, we know we have a good input_frame that's advanced far enough to overlap with output_fpga_seq_num
        if output_in_progress:
            if output_fpga_seq_num + 50000 < input_fpga_seq_num + 49152:
                # completed output
                print(
                    "Completed output:",
                    output_fpga_seq_num,
                    output_frame_seconds,
                    output_frame_nanoseconds,
                )
                output_in_progress = False
                expected_output.append(
                    (
                        output_fpga_seq_num,
                        output_frame_seconds,
                        output_frame_nanoseconds,
                    )
                )
                output_fpga_seq_num += 50000
                output_frame_nanoseconds += 128000000
                if output_frame_nanoseconds >= 1e9:
                    output_frame_seconds += 1
                    output_frame_nanoseconds = output_frame_nanoseconds - 1e9
            else:
                input_frame += 1
                input_fpga_seq_num += 49152
                print("Advance input", input_fpga_seq_num)
        elif start_new_stream:
            # start outputting but align on a whole "data_frame" segment
            print("Align near", input_fpga_seq_num)
            output_frame_nanoseconds = input_fpga_seq_num * 2560
            offset = 1600000 - (output_frame_nanoseconds % 1600000)
            output_fpga_seq_num = input_fpga_seq_num + offset // 2560
            print("Move output by {} to {}".format(offset, output_fpga_seq_num))

            output_frame_nanoseconds += offset
            if output_frame_nanoseconds >= 1e9:
                output_frame_seconds += output_frame_nanoseconds // 1e9
                output_frame_nanoseconds = output_frame_nanoseconds % 1e9
            print("{}.{:09}".format(output_frame_seconds, output_frame_nanoseconds))
            output_in_progress = True
            print("Start output", output_fpga_seq_num)
            start_new_stream = False
        else:
            # continuing with the stream, but need to start a new output frame
            if output_fpga_seq_num >= input_fpga_seq_num:
                output_in_progress = True
                print("Start output", output_fpga_seq_num)
    print("Expected output frames:", expected_output)

    for frame, (fpga_seq_num, frame_seconds, frame_nanoseconds) in zip(
        actual_frames, expected_output
    ):
        beam_id = 0
        beam_frame = 1
        for pkt in frame:
            if beam_frame > 80:
                beam_frame = 1
                beam_id += 1
            if beam_frame == 1:
                seconds = frame_seconds
                nanoseconds = frame_nanoseconds

            assert pkt.header.legacy == 0
            assert pkt.header.invalid == 0

            assert pkt.header.seconds == seconds
            assert pkt.header.data_frame == int((nanoseconds / 1e9) / (625 * 2.56e-6))
            assert pkt.header.ref_epoch == 36

            assert pkt.header.frame_len == 629
            assert pkt.header.log_num_chan == 3
            assert pkt.header.vdif_version == 1

            assert pkt.header.station_id == 17240  # 'CX'==CHIME
            assert pkt.header.thread_id == 777
            assert pkt.header.bit_depth == 3
            assert pkt.header.data_type == 1

            assert pkt.header.beam_id == beam_id
            # print(pkt.header.scaling_factor)

            assert pkt.header.ra == 5361
            assert pkt.header.dec == 14464

            beam_frame += 1
            nanoseconds += 1600000
            if nanoseconds >= 1e9:
                seconds += 1
                nanoseconds -= 1e9
