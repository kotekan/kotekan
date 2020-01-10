# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import glob
import pytest
from os import path
from kotekan import runner
from kotekan.frbbuffer import FrbPacket

# Note that we're using a shortened beamformed data, using only 6144 `samples_per_data_set`
frb_params = {
    "buffer_depth": 7,
    "num_elements": 2048,
    "block_size": 2,
    "num_ev": 4,
    "num_freq_in_frame": 1,
    "num_local_freq": 1,
    "downsample_time": 3,
    "downsample_freq": 8,
    "factor_upchan": 128,
    "factor_upchan_out": 16,
    "num_beams_per_frb_packet": 4,
    "timesamples_per_frb_packet": 16,
    "num_frb_total_beams": 1024,
    "sizeof_float": 4,
    "sizeof_short": 2,
}


def frb_post_process_data(tmpdir_factory, missing_frames=None, test_env=None):
    """Runs the Kotekan using the short 6144 samples_per_dataset, optionally dropping frames

    This works out to just one packet per stream in a frame, and uses a single
    GPU, reading the beamforming inputs from files named starting with
    "fake_cpu_beamform".

    If there are any frames to drop, the config will include a
    TestDropFrames stage, and drop the specified frame from the input
    to the FrbPostProcess.

    """

    if test_env is None and not glob.glob(
        path.join(path.dirname(__file__), "data", "fake_cpu_beamform_*.dump")
    ):
        pytest.skip("No test data and not requesting full generation")

    tmpdir = tmpdir_factory.mktemp("frb_post_process")

    if test_env == "run_slow_cpu_tests":
        beamform_buffer = runner.FakeFrbBeamformBuffer(num_frames=7)
    elif test_env == "run_amd_gpu_tests":
        beamform_buffer = runner.FakeFrbBeamformBuffer(num_frames=7, cpu=False)
    elif test_env is None:
        beamform_buffer = runner.ReadRawBeamformBuffer(
            path.join(path.dirname(__file__), "data"), file_name="fake_cpu_beamform"
        )
    else:
        pytest.skip("Unknown environment {}".format(test_env))

    lost_samples_buffer = runner.FakeLostSamplesBuffer(num_frames=10)
    dump_buffer = runner.DumpFrbPostProcessBuffer(str(tmpdir))

    params = frb_params.copy()
    params["num_gpus"] = 1
    params["samples_per_data_set"] = 6144
    if missing_frames:
        drop_frames_buffer = runner.DropFramesBuffer(
            beamform_buffer,
            missing_frames,
            frame_size=beamform_buffer.buffer_block[beamform_buffer.name]["frame_size"],
        )
        input_buffers = [beamform_buffer, drop_frames_buffer, lost_samples_buffer]
        params["in_buf_0"] = drop_frames_buffer.name
    else:
        input_buffers = [beamform_buffer, lost_samples_buffer]
        params["in_buf_0"] = beamform_buffer.name
    params["lost_samples_buf"] = lost_samples_buffer.name

    test = runner.KotekanStageTester(
        "frbPostProcess", frb_params, input_buffers, dump_buffer, params
    )

    test.run()

    return dump_buffer


def frb_post_process_full_data(
    tmpdir_factory, num_gpus=1, missing_frames=None, test_env=None
):
    """Runs the Kotekan using the full 41952 samples_per_dataset, optionally dropping frames

    Uses an argument-specified number of GPUs, assuming the beamforming inputs
    are stored in files starting with "fake_cpu_beamform_long{GPU_ID}".

    If there are any frames to drop, the config will include a
    TestDropFrames stage, and drop the specified frame from the input
    to the FrbPostProcess.

    """

    if test_env is None and not glob.glob(
        path.join(path.dirname(__file__), "data", "fake_cpu_beamform_long[0-9]_*.dump")
    ):
        pytest.skip("No test data and not requesting full generation")

    tmpdir = tmpdir_factory.mktemp("frb_post_process")

    dump_buffer = runner.DumpFrbPostProcessBuffer(str(tmpdir))
    lost_samples_buffer = runner.FakeLostSamplesBuffer(num_frames=10)

    params = frb_params.copy()
    params["num_gpus"] = num_gpus
    params["lost_samples_buf"] = lost_samples_buffer.name
    params["samples_per_data_set"] = 49152

    input_buffers = [lost_samples_buffer]

    for i in range(num_gpus):
        if test_env == "run_slow_cpu_tests":
            beamform_buffer = runner.FakeFrbBeamformBuffer(num_frames=7)
        elif test_env == "run_slow_gpu_tests":
            beamform_buffer = runner.FakeFrbBeamformBuffer(num_frames=7, cpu=False)
        elif test_env is None:
            beamform_buffer = runner.ReadRawBeamformBuffer(
                path.join(path.dirname(__file__), "data"),
                file_name="fake_cpu_beamform_long{:d}".format(i),
            )
        else:
            pytest.skip("Unknown environment {}".format(test_env))

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
            params["in_buf_{:d}".format(i)] = drop_frames_buffer.name
        else:
            params["in_buf_{:d}".format(i)] = beamform_buffer.name

    test = runner.KotekanStageTester(
        "frbPostProcess", frb_params, input_buffers, dump_buffer, params
    )

    test.run()

    return dump_buffer


def test_frb_post_process(tmpdir_factory, request):
    test_env = request.config.getoption("-E", None)

    data = frb_post_process_data(tmpdir_factory, test_env=test_env)

    check_data(data, 1, 1)
    pkt = data.load()[0][0]
    # I know this because I traced the main loop of frbPostProcess
    # and recorded the very first value of scales and offsets:
    assert pkt.scales[0] == 4748.201171875
    assert pkt.offsets[0] == 484518.0625


def test_frb_post_process_missing(tmpdir_factory, request):
    test_env = request.config.getoption("-E", None)

    missing_frames = [0, 3, 4]
    data = frb_post_process_data(tmpdir_factory, missing_frames, test_env=test_env)
    check_data(data, 1, 1, missing_frames)


def test_frb_post_process_full_data(tmpdir_factory, request):
    test_env = request.config.getoption("-E", None)

    data = frb_post_process_full_data(tmpdir_factory, num_gpus=2, test_env=test_env)
    check_data(data, num_gpus=2, packets_per_stream=8)


def test_frb_post_process_full_data_missing(tmpdir_factory, request):
    test_env = request.config.getoption("-E", None)

    missing_frames = [[1], [0, 3, 4]]
    data = frb_post_process_full_data(
        tmpdir_factory, num_gpus=2, missing_frames=missing_frames, test_env=test_env
    )
    check_data(data, num_gpus=2, packets_per_stream=8, missing_frames=missing_frames)


def check_data(data, num_gpus, packets_per_stream, missing_frames=[]):
    if num_gpus > 1:
        import itertools

        missing_frames = set(itertools.chain.from_iterable(missing_frames))

    file_fpga_start = next_fpga_id = 0
    frame_id = 0
    for frame in data.load():
        next_beam = 0
        stream_packet = 0

        while frame_id in missing_frames:
            frame_id += 1
            file_fpga_start += 6144 * packets_per_stream

        for pkt in frame:
            if stream_packet == 0:
                next_fpga_id = file_fpga_start
                # adjust for FRB beam numbering (cylinder as thousands)
                stream_beams = [
                    b % 256 + b // 256 * 1000
                    for b in range(next_beam, next_beam + pkt.header.nbeams)
                ]

            assert pkt.header.version == 1
            assert pkt.header.nbeams == 4
            assert pkt.header.fpga_seq_num == next_fpga_id
            assert pkt.header.nfreq == num_gpus
            assert pkt.header.nupfreq == 16
            assert pkt.header.ntsamp == 16

            assert pkt.beam_ids[:] == stream_beams
            assert pkt.freq_ids[:] == [777] * num_gpus

            assert all([s != 0 for s in pkt.scales[:]])
            assert all([o != 0 for o in pkt.offsets[:]])

            next_fpga_id += 6144
            stream_packet += 1
            if stream_packet == packets_per_stream:
                stream_packet = 0
                next_beam += pkt.header.nbeams

        frame_id += 1
        file_fpga_start += 6144 * packets_per_stream
