import glob
import io
import pytest
import os

from kotekan import baseband_buffer, runner

global_params = {
    "max_dump_samples": 3500,
    "num_elements": 256,
    "total_frames": 60,
    "stream_id": 0,
    "buffer_depth": 6,
    "num_frames_buffer": 18,
    "samples_per_data_set": 512,
    "baseband_metadata_pool": {
        "kotekan_metadata_pool": "BasebandMetadata",
        "num_metadata_objects": 4096,
    },
}
frame_size = global_params["frame_size"] = (
    global_params["num_elements"] * global_params["samples_per_data_set"]
)


def generate_tpluse_data(fpga_seq, length, num_elements):
    """Simulates sample data using the "tpluse" algorithm from `testDataGen`

    Arguments:
    ----------
    fpga_seq
        starting FPGA sequence number
    length
        number of samples to generate
    num_elements
        number of telescope inputs

    Returns:
    --------
    A list of `length` samples
    """

    samples = []
    for j in range(length):
        val = (fpga_seq + j // num_elements + j % num_elements) % 256
        samples.append(val)
    return samples


def run_kotekan(tmpdir_factory, nfreqs=1):
    """Starts Kotekan with a simulated baseband stream from `nfreq` frequencies being fed into a single `basebandWriter` stage.

    The frequencies will have indexes [0:nfreq], and will be saved into raw baseband dump files in the `tmpdir_factory` subdirectory `baseband_raw_12345`, one frequency per file.

    Returns:
    --------
    Sorted list of filenames of the saved baseband files.
    """
    num_elements = global_params["num_elements"]
    samples_per_data_set = global_params["samples_per_data_set"]

    frame_list = []
    for i in range(global_params["buffer_depth"]):
        for freq_id in range(nfreqs):
            fpga_seq = frame_size * i
            frame_list.append(
                baseband_buffer.BasebandBuffer.new_from_params(
                    event_id=12345,
                    freq_id=freq_id,
                    num_elements=num_elements,
                    frame_size=frame_size,
                    frame_data=generate_tpluse_data(fpga_seq, frame_size, num_elements),
                )
            )
            frame_list[-1].metadata.frame_fpga_seq = frame_size * i
            frame_list[-1].metadata.valid_to = samples_per_data_set
    frame_list[-1].metadata.valid_to -= 17
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
    samples_per_data_set = global_params["samples_per_data_set"]
    num_elements = global_params["num_elements"]

    buf = bytearray(frame_size + metadata_size)
    frame_index = 0
    with io.FileIO(file_name, "rb") as fh:
        final_frame = False
        while fh.readinto(buf):
            # Check the frame metadata
            frame_metadata = baseband_buffer.BasebandMetadata.from_buffer(buf)
            assert frame_metadata.event_id == 12345
            assert frame_metadata.freq_id == freq_id
            assert frame_metadata.frame_fpga_seq == frame_index * frame_size
            if not final_frame:
                assert frame_metadata.valid_to <= samples_per_data_set
                if frame_metadata.valid_to < samples_per_data_set:
                    final_frame = True
            else:
                assert False, "No more event data is allowed after a non-full frame."

            # Check that the frame data matches tpluse-generated samples
            for j, val in enumerate(buf[metadata_size:]):
                if j >= frame_metadata.valid_to * num_elements:
                    break
                # calculation used in `testDataGen` for method `tpluse`:
                expected = (
                    frame_metadata.frame_fpga_seq + j // num_elements + j % num_elements
                ) % 256
                assert (
                    val == expected
                ), f"Baseband data mismatch at index {j}/{frame_index}, fpga_seq={frame_metadata.frame_fpga_seq}"

            frame_index += 1
    assert frame_index > 0


def test_simple(tmpdir_factory):
    """Check receiving a baseband dump with a single frequency and no dropped frames"""
    saved_files = run_kotekan(tmpdir_factory)
    assert len(saved_files) == 1

    check_baseband_dump(saved_files[0])
    os.system(f"rm -rf {saved_files[0]}")


def test_multi_freq(tmpdir_factory):
    """Check receiving a baseband dump with a single frequency and no dropped frames"""
    saved_files = run_kotekan(tmpdir_factory, 3)
    assert len(saved_files) == 3

    for freq_id, file_name in enumerate(saved_files):
        check_baseband_dump(file_name, freq_id)
