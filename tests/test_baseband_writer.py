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
frame_size = global_params["frame_size"] = (
    global_params["num_elements"] * global_params["samples_per_data_set"]
)


def run_kotekan(tmpdir_factory, nfreqs=1):
    """Starts Kotekan with a simulated baseband stream from `nfreq` frequencies being fed into a single `basebandWriter` stage.

    The frequencies will have indexes [0:nfreq], and will be saved into raw baseband dump files in the `tmpdir_factory` subdirectory `baseband_raw_12345`, one frequency per file.

    Returns:
    --------
    Sorted list of filenames of the saved baseband files.
    """
    frame_list = []
    for i in range(global_params["buffer_depth"]):
        for freq_id in range(nfreqs):
            frame_list.append(
                baseband_buffer.BasebandBuffer.new_from_params(
                    event_id=12345, freq_id=freq_id, frame_size=frame_size
                )
            )
            frame_list[-1].metadata.fpga_seq = frame_size * i
            frame_list[-1].metadata.valid_to = frame_size
    frame_list[-1].metadata.valid_to -= 1234
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


def check_baseband_dump(file_name, freq_id=0):
    metadata_size = baseband_buffer.BasebandBuffer.meta_size

    buf = bytearray(frame_size + metadata_size)
    frame_index = 0
    with io.FileIO(file_name, "rb") as fh:
        final_frame = False
        while fh.readinto(buf):
            # Check the frame metadata
            frame_metadata = baseband_buffer.BasebandMetadata.from_buffer(buf)
            assert frame_metadata.event_id == 12345
            assert frame_metadata.freq_id == freq_id
            assert frame_metadata.fpga_seq == frame_index * frame_size

            if not final_frame:
                assert frame_metadata.valid_to <= frame_size
                if frame_metadata.valid_to < frame_size:
                    final_frame = True
            else:
                assert False, "No more event data is allowed after a non-full frame."

            frame_index += 1
    assert frame_index > 0


def test_simple(tmpdir_factory):
    """Check receiving a baseband dump with a single frequency and no dropped frames"""
    saved_files = run_kotekan(tmpdir_factory)
    assert len(saved_files) == 1

    check_baseband_dump(saved_files[0])


def test_multi_freq(tmpdir_factory):
    """Check receiving a baseband dump with a single frequency and no dropped frames"""
    saved_files = run_kotekan(tmpdir_factory, 3)
    assert len(saved_files) == 3

    for freq_id, file_name in enumerate(saved_files):
        check_baseband_dump(file_name, freq_id)
