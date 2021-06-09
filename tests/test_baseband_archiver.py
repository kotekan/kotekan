import glob
import h5py
import io
import pytest

from kotekan.scripts import baseband_archiver
import kotekan


config = {
    "max_dump_samples": 3500,
    "num_elements": 28,
    "total_frames": 60,
    "stream_id": 0,
    "buffer_depth": 6,
    "num_frames_buffer": 18,
    "samples_per_data_set": 32,
    "baseband_metadata_pool": {
        "kotekan_metadata_pool": "BasebandMetadata",
        "num_metadata_objects": 4096,
    },
}
frame_size = config["frame_size"] = (
    config["num_elements"] * config["samples_per_data_set"]
)
num_elements = config["num_elements"]
config["input_reorder"] = [
    (num_elements - i - 1, i, f"FX{i :03d}") for i in range(num_elements)
]


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


def generate_baseband_raw_files(tmpdir_factory, nfreqs=1):
    """Starts Kotekan with a simulated baseband stream from `nfreq` frequencies being fed into a single `basebandWriter` stage.

    The frequencies will have indexes [0:nfreq], and will be saved into raw baseband dump files in the `tmpdir_factory` subdirectory `baseband_raw_12345`, one frequency per file.

    Returns:
    --------
    Sorted list of filenames of the saved baseband files.
    """
    num_elements = config["num_elements"]
    samples_per_data_set = config["samples_per_data_set"]
    current_dir = str(tmpdir_factory.mktemp("baseband_raw_12345"))

    for freq_id in range(nfreqs):
        frame_list = []
        for i in range(config["buffer_depth"]):
            frame_list.append(
                kotekan.baseband_buffer.BasebandBuffer.new_from_params(
                    event_id=12345,
                    freq_id=freq_id,
                    num_elements=num_elements,
                    frame_size=frame_size,
                    frame_data=generate_tpluse_data(
                        i * samples_per_data_set, frame_size, num_elements
                    ),
                )
            )
            frame_list[-1].metadata.event_start_seq = 0
            frame_list[-1].metadata.event_end_seq = (
                config["buffer_depth"] * samples_per_data_set
            )
            frame_list[-1].metadata.frame_fpga_seq = i * samples_per_data_set
            frame_list[-1].metadata.valid_to = samples_per_data_set
        # frame_list[-1].metadata.valid_to -= 17
        print("Saving:", i, current_dir)
        with open(f"{ current_dir }/baseband_12345_{ freq_id :d}.data", "wb+") as f:
            for frame in frame_list:
                f.write(frame._buffer)

    dump_files = sorted(glob.glob(current_dir + "/baseband_12345_*.data"))
    return dump_files


def check_baseband_dump(file_name, freq_id=0):
    metadata_size = kotekan.baseband_buffer.BasebandBuffer.meta_size
    samples_per_data_set = config["samples_per_data_set"]
    num_elements = config["num_elements"]

    buf = bytearray(frame_size + metadata_size)
    frame_index = 0
    with io.FileIO(file_name, "rb") as fh:
        final_frame = False
        while fh.readinto(buf):
            # Check the frame metadata
            frame_metadata = kotekan.baseband_buffer.BasebandMetadata.from_buffer(buf)
            assert frame_metadata.event_id == 12345
            assert frame_metadata.freq_id == freq_id
            assert frame_metadata.frame_fpga_seq == frame_index * samples_per_data_set

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


def check_baseband_archive(file_name):
    assert file_name
    with h5py.File(file_name, "r") as f:
        assert f.attrs["archive_version"] == "NT_3.1.0"
        assert f.attrs["git_version_tag"] == kotekan.__version__
        assert "collection_server" in f.attrs
        assert "system_user" in f.attrs
        assert "time0_fpga" in f.attrs
        assert "time0_ctime" in f.attrs
        assert "time0_ctime_offset" in f.attrs

        assert "first_packet_recv_time" in f.attrs
        assert "fpga0_ns" in f.attrs

        inputs = f["index_map/inputs"]
        assert len(inputs) == num_elements
        assert inputs.dtype.names == ("chan_id", "correlator_input")

        for i, val in enumerate(f["baseband"][:].flatten()):
            fpga_seq = i // num_elements
            j = i % num_elements
            expected = (fpga_seq + j // num_elements + j % num_elements) % 256
            assert (
                val == expected
            ), f"Baseband archive mismatch at index {i} ({fpga_seq}/{j})"


def test_single_freq(tmpdir_factory):
    """Check receiving a baseband dump with a single frequency and no dropped frames"""
    saved_files = generate_baseband_raw_files(tmpdir_factory)
    assert len(saved_files) == 1

    check_baseband_dump(saved_files[0])

    archive_file_name = baseband_archiver.process_raw_file(saved_files[0], config)
    print(archive_file_name)
    check_baseband_archive(archive_file_name)


def test_multi_freq(tmpdir_factory):
    """Check receiving a baseband dump with a single frequency and no dropped frames"""
    saved_files = generate_baseband_raw_files(tmpdir_factory, 3)
    assert len(saved_files) == 3

    for freq_id, file_name in enumerate(saved_files):
        check_baseband_dump(file_name, freq_id)
        archive_file_name = baseband_archiver.process_raw_file(file_name, config)
        check_baseband_archive(archive_file_name)
