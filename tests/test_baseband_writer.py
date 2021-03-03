import glob
import pytest

from kotekan import baseband_buffer, runner

global_params = {
    "max_dump_samples": 3500,
    "num_elements": 256,
    "total_frames": 60,
    "stream_id": 0,
    "buffer_depth": 20,
    "num_frames_buffer": 18,
    "samples_per_data_set": 1024,
    "baseband_metadata_pool": {
        "kotekan_metadata_pool": "BasebandMetadata",
        "num_metadata_objects": 4096,
    },
}


def run_kotekan(tmpdir_factory):
    frame_size = global_params["num_elements"] * global_params["samples_per_data_set"]
    frame_list = []
    for i in range(global_params["buffer_depth"]):
        frame_list.append(
            baseband_buffer.BasebandBuffer.new_from_params(
                event_id=12345, freq_id=0, frame_size=frame_size
            )
        )
        frame_list[i].metadata.fpga_seq = frame_size * i
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

    dump_files = sorted(glob.glob(current_dir + "/rawbaseband_buf_*.dump"))
    return dump_files


def test_start(tmpdir_factory):
    saved_files = run_kotekan(tmpdir_factory)
    assert saved_files
    for i, x in enumerate(
        [baseband_buffer.BasebandBuffer.from_file(f) for f in saved_files]
    ):
        print(i, x.metadata.event_id, x.metadata.freq_id, x.metadata.fpga_seq)
