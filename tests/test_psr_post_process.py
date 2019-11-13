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


def psr_post_process_data(tmpdir_factory, test_env=None):
    """Runs the Kotekan using the short 6144 samples_per_dataset, optionally dropping frames

    This works out to just one packet per stream in a frame, and uses a single
    GPU, reading the beamforming inputs from files named starting with
    "fake_cpu_beamform".
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
        stage_params["network_input_buffer_{:d}".format(i)] = beamform_buffer.name
        input_buffers.append(beamform_buffer)

    params = global_params.copy()

    test = runner.KotekanStageTester(
        "pulsarPostProcess", stage_params, input_buffers, dump_buffer, params
    )

    test.run()

    return dump_buffer


def test_psr_post_process(tmpdir_factory, request):
    test_env = request.config.getoption("-E", None)

    data = psr_post_process_data(tmpdir_factory, test_env=test_env)

    fpga_seq_num = 625  # skipped by the main_thread looking for 1s boundary
    frame_seconds = 632718848  # (`0 - ctime(1 Jan 2018 UTC)`, truncated to 30 bits)
    frame_nanoseconds = 1600000  # 625 elapsed FPGA samples @ 2.56 us / sample
    for frame in data.load():
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
            assert pkt.header.scaling_factor

            assert pkt.header.ra == 5361
            assert pkt.header.dec == 14464

            beam_frame += 1
            nanoseconds += 1600000
            if nanoseconds >= 1e9:
                seconds += 1
                nanoseconds -= 1e9

        fpga_seq_num += 500000  # 625 samples * 80 packets
        frame_nanoseconds += 128000000
        if frame_nanoseconds >= 1e9:
            frame_seconds += 1
            frame_nanoseconds -= 1e9

    # check_data(data, 1, 1)
    # pkt = data.load()[0][0]
    # # I know this because I traced the main loop of frbPostProcess
    # # and recorded the very first value of scales and offsets:
    # assert pkt.scales[0] == 4748.201171875
    # assert pkt.offsets[0] == 484518.0625
