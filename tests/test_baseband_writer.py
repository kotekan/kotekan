import glob
import io
import pytest

from kotekan import baseband_buffer, runner

global_params = {
    "max_dump_samples": 3500,
    "num_elements": 256,
    "total_frames": 60,
    "stream_id": 0,
    "buffer_depth": 6,
    "num_frames_buffer": 18,
    "samples_per_data_set": 1024,
    "baseband_metadata_pool": {
        "kotekan_metadata_pool": "BasebandMetadata",
        "num_metadata_objects": 4096,
    },
}
frame_size = global_params["frame_size"] = global_params["num_elements"] * global_params["samples_per_data_set"]


def run_kotekan(tmpdir_factory):
    frame_list = []
    for i in range(global_params["buffer_depth"]):
        frame_list.append(
            baseband_buffer.BasebandBuffer.new_from_params(
                event_id=12345, freq_id=0, frame_size=frame_size
            )
        )
        frame_list[i].metadata.fpga_seq = frame_size * i
        frame_list[i].metadata.valid_to = frame_size
    current_dir = str(tmpdir_factory.getbasetemp())
    read_buffer = runner.ReadBasebandBuffer(current_dir, frame_list)
    read_buffer.write()
    test = runner.KotekanStageTester(
        "BasebandWriter",
        {"root_path": current_dir},  # stage_config
        read_buffer,  # buffers_in
        None,  # buffers_out is None
        global_params,  # global_config
        expect_failure=True,  # because rawFileRead runs out of files to read
    )

    test.run()

    dump_files = sorted(
        glob.glob(current_dir + "/baseband_raw_12345/baseband_12345_*.data")
    )
    return dump_files


def test_start(tmpdir_factory):
    saved_files = run_kotekan(tmpdir_factory)
    assert len(saved_files) == 1

    metadata_size = baseband_buffer.BasebandBuffer.meta_size

    buf = bytearray(frame_size + metadata_size + 1)
    frame_index = 0
    with io.FileIO(saved_files[0], "rb") as fh:
        while (fh.readinto(buf)):
            # Check that the frame is valid
            assert buf[0] == 1

            # Check the frame metadata
            frame_metadata = baseband_buffer.BasebandMetadata.from_buffer(buf[1:])

            assert frame_metadata.event_id == 12345
            assert frame_metadata.freq_id == 0
            assert frame_metadata.fpga_seq == frame_index * frame_size
            assert frame_metadata.valid_to == frame_size

            frame_index += 1
    assert frame_index > 0
