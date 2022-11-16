"""Convert a baseband raw file into a standard baseband HDF5 archive
"""
import time
import click
import h5py
import io
import numpy as np
import os
import pwd
import yaml
from socket import gethostname
from typing import Dict
from datetime import datetime

freq0_MHz = None
df_MHz = None
nfreq = None
ny_zone = None
dt_ns = None

from kotekan import config, baseband_buffer, conv_backends, __version__


def parse_reorder_map(inputs_reorder):
    """Parses the reordering configuration section

    inputs_reorder: the corresponding section in the configuration

    Returns:
    --------
    Tuple containing a vector of the input reorder map and a vector of the input labels for the index map
    """
    adc_ids = []
    input_map = []
    for adc_id, chan_id, serial in inputs_reorder:
        adc_ids.append(adc_id)
        input_map.append((chan_id, serial))

    return adc_ids, input_map


def add_index_map(archive_file, config):
    num_elements = config.get("num_elements", 2048)
    adc_ids, input_map = parse_reorder_map(config["input_reorder"])
    assert len(adc_ids) == num_elements
    assert len(input_map) == len(adc_ids)

    inputs = np.zeros(
        (num_elements,),
        dtype=np.dtype([("chan_id", "<u2"), ("correlator_input", "S32")]),
    )
    for i, adc_id in enumerate(adc_ids):
        inputs[adc_id] = input_map[i]
    archive_file.create_dataset("index_map/input", data=inputs[:])


def set_sampling_params(config):
    """Reimplements ICETelescope::set_sampling_params"""
    sample_rate = config.get("sampling_rate", 800.0)
    fft_length = config.get("fft_length", 2048)
    zone = config.get("nyquist_zone", 2)

    global freq0_MHz, df_MHz, nfreq, ny_zone, dt_ns
    freq0_MHz = (zone / 2) * sample_rate
    df_MHz = (1 if zone % 2 else -1) * sample_rate / fft_length
    nfreq = fft_length / 2
    ny_zone = zone
    dt_ns = 1e3 / sample_rate * fft_length


def to_freq(freq_id):
    """Reimplements ICETelescope::to_freq"""
    if freq_id >= nfreq:
        raise ValueError(f"Bad freq_id: { freq_id } (< { nfreq })")

    return freq0_MHz + freq_id * df_MHz


def to_time(time0_ns, seq):
    """Reimplements ICETelescope::to_time"""
    time_ns = time0_ns + seq * dt_ns
    return time_ns / 1e9, time_ns % 1e9


def ts_to_double(ts_sec, ts_nsec):
    """Reimplements visUtil::ts_to_double"""
    return ts_sec + ts_nsec * 1e-9


def create_baseband_archive(
    frame_metadata: baseband_buffer.BasebandMetadata, root: str
):
    """Creates a standard baseband HDF5 archive for the event described by `frame_metadata`

    Returns
    -------
    (file, str):
        Pair of the handle to the created HDF5 file, open for writing, and its name
    """
    event_id, freq_id, fpga_seq = (
        frame_metadata.event_id,
        frame_metadata.freq_id,
        frame_metadata.frame_fpga_seq,
    )
    date = datetime.utcfromtimestamp(frame_metadata.time0_ctime).date()
    y = str(date.year).zfill(4)
    m = str(date.month).zfill(2)
    d = str(date.day).zfill(2)
    # confirm root path on the recv node.
    file_name = os.path.join(
        root, f"{y}/{m}/{d}/astro_{event_id}/baseband_{ event_id }_{ freq_id }.h5"
    )
    dir_name = os.path.dirname(file_name)
    if dir_name:
        if not os.path.exists(dir_name):
            if os.path.exists(dir_name.replace("astro", "rfi")):
                dir_name = dir_name.replace("astro", "rfi")
                file_name = file_name.replace("astro", "rfi")
        os.makedirs(dir_name, exist_ok=True)

    f = h5py.File(file_name, "w")
    f.attrs["archive_version"] = "NT_3.1.0"
    f.attrs["collection_server"] = gethostname()
    f.attrs["delta_time"] = 2.56e-06
    f.attrs["git_version_tag"] = __version__
    f.attrs["system_user"] = pwd.getpwuid(os.getuid()).pw_name
    f.attrs["event_id"] = event_id
    f.attrs["freq_id"] = freq_id
    f.attrs["freq"] = to_freq(freq_id)

    f.attrs["time0_fpga_count"] = np.uint64(frame_metadata.time0_fpga)
    f.attrs["time0_ctime"] = frame_metadata.time0_ctime
    f.attrs["time0_ctime_offset"] = frame_metadata.time0_ctime_offset
    f.attrs["type_time0_ctime"] = str(f.attrs["time0_ctime"].dtype)

    f.attrs["first_packet_recv_time"] = frame_metadata.first_packet_recv_time
    return f, file_name


def raw_baseband_frames(file_name: str, buf: bytes):
    """Iterates over frames in a raw baseband file"""
    with io.FileIO(file_name, "rb") as raw_file:
        while raw_file.readinto(buf):
            yield buf
        size = os.path.getsize(file_name)
        os.posix_fadvise(raw_file.fileno(), 0, size, os.POSIX_FADV_DONTNEED)


def process_raw_file(
    file_name: str,
    config: Dict[str, int],
    root: str,
    dry_run: bool = False,
    verbose: bool = False,
):
    """Convert raw file `file_name` to a standard baseband HDF5 archive"""
    metadata_size = baseband_buffer.BasebandBuffer.meta_size

    if nfreq is None:
        set_sampling_params(config)
    samples_per_data_set = config.get("samples_per_data_set", 512)
    if type(samples_per_data_set) is str:
        samples_per_data_set = eval(samples_per_data_set)
    num_elements = config.get("num_elements", 2048)
    if type(num_elements) is str:
        num_elements = eval(num_elements)
    frame_size = num_elements * samples_per_data_set
    buf = bytearray(frame_size + metadata_size)
    event_id = freq_id = None
    archive_file_name = archive_file = None
    frames_read = []
    clip_after = 0
    if verbose:
        print('samples_per_data_set:',samples_per_data_set)
        print('num_elements:',num_elements)
    for b in raw_baseband_frames(file_name, buf):
        # Check the frame metadata
        frame_metadata = baseband_buffer.BasebandMetadata.from_buffer(b)
        if verbose:
            print(
                file_name,
                frame_metadata.event_id,
                frame_metadata.freq_id,
                frame_metadata.frame_fpga_seq,
                frame_metadata.time0_fpga
            )

        if event_id is None:
            event_id, freq_id = (frame_metadata.event_id, frame_metadata.freq_id)

            if not dry_run:
                archive_file, archive_file_name = create_baseband_archive(
                    frame_metadata, root
                )
                add_index_map(archive_file, config)

            # Use the first captured sample (`time0_fpga`) as the start of the
            # event, rather than the frequency's nominal `event_start_seq`,
            # since the trigger may have been late
            event_fpga_start, event_fpga_end = (
                frame_metadata.time0_fpga,
                frame_metadata.event_end_seq,
            )
            if event_fpga_end >= event_fpga_start:
                event_fpga_len = event_fpga_end - event_fpga_start
            else:
                event_fpga_len = event_fpga_end
                event_fpga_end += event_fpga_start
            if verbose:
                print("Event length:", event_fpga_len)
                print("Num elements:", num_elements)

            baseband = np.zeros(shape=(event_fpga_len, num_elements), dtype=np.uint8)
            # sample_present = np.zeros(shape=(event_fpga_len,), dtype=bool)
        else:
            # Data validity check: all frames in the file should be for the same event and frequency
            assert event_id == frame_metadata.event_id
            assert freq_id == frame_metadata.freq_id

            assert event_fpga_start == frame_metadata.time0_fpga
            assert event_fpga_end == frame_metadata.event_end_seq

        fpga_seq = frame_metadata.frame_fpga_seq
        frames_read.append(fpga_seq)

        frame_start_idx = fpga_seq - event_fpga_start
        samples = buf[
            metadata_size : (metadata_size + frame_metadata.valid_to * num_elements)
        ]

        if verbose:
            print("Copying", frame_metadata.valid_to, "samples to", frame_start_idx)
            print("Samples:", frame_metadata.valid_to * num_elements)
            print(len(buf), len(samples))
        baseband[
            frame_start_idx : (frame_start_idx + frame_metadata.valid_to), :
        ] = np.array(
            buf[
                metadata_size : (metadata_size + frame_metadata.valid_to * num_elements)
            ]
        ).reshape(
            frame_metadata.valid_to, num_elements
        )
        # sample_present[
        #    frame_start_idx : (frame_start_idx + frame_metadata.valid_to)
        # ] = True
        clip_after += frame_metadata.valid_to

    baseband = baseband[
        :clip_after,
    ]
    if not dry_run:
        archive_file.create_dataset("baseband", data=baseband)
        archive_file["baseband"].attrs["axis"] = ["time", "input"]
        # archive_file.create_dataset("sample_present", data=sample_present)
        # archive_file["sample_present"].attrs["axis"] = ["time"]
        # archive_file["sample_present"].attrs["fill_value"] = False
    found = []
    for seq in sorted(frames_read):
        for f in found:
            if f[0] == seq + samples_per_data_set:
                f[0] = seq
                break
            elif f[1] == seq:
                f[1] = seq + samples_per_data_set
                break
        else:
            found.append([seq, seq + samples_per_data_set])
    found.sort()
    if verbose:
        print("Found:", found)
        print()

    missing = []
    if frame_metadata.time0_fpga != found[0][0]:
        missing.append([frame_metadata.time0_fpga, found[0][0]])
    for a, b in zip(found, found[1:]):
        assert a[1] <= b[0]
        missing.append([a[1], b[0]])
    if event_fpga_end > found[-1][1]:
        missing.append([found[-1][1], event_fpga_end])
    if verbose:
        print(missing)

    missing_len = 0
    for m in missing:
        missing_len += m[1] - m[0]

    if archive_file:
        size = os.path.getsize(archive_file_name)
        os.fsync(archive_file.id.get_vfd_handle())
        os.posix_fadvise(
            archive_file.id.get_vfd_handle(), 0, size, os.POSIX_FADV_DONTNEED
        )
        archive_file.close()
    return archive_file_name


def sample_present_stats(file_name):
    """Prints the percentage of valid samples in a baseband archive

    The calculation uses the mask stored in the file's `sample_present` dataset.

    Arguments:
    ----------
    file_name: str
        Path to the baseband archive HDF5 file
    """
    with h5py.File(file_name, "r") as archive_file:
        print(
            "Samples present: {:.1%}".format(
                archive_file["sample_present"][:].sum()
                / archive_file["sample_present"].size
            )
        )


def convert(
    file_name,
    config_file,
    root,
    stats=False,
    dry_run=False,
    verbose=False,
):
    """Main function to do the conversion at the level of specifying a conversion backend."""
    with open(config_file) as f:
        # generalize for .j2 files; return_dict=True preserves behavior for .yaml files.
        #config = yaml.safe_load(f)
        config_dict = config.load_config_file(config_file,return_dict=True)
    set_sampling_params(config_dict)

    archive_file_names = []
    for f in file_name:
        archive_file_name = process_raw_file(f, config_dict, root, dry_run, verbose)
        archive_file_names.append(archive_file_name)
        # if stats and not dry_run:
        #    sample_present_stats(archive_file_name)
    return archive_file_names


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("file_names", type=click.Path(exists=True), nargs=-1)
@click.option(
    "--stats",
    "-s",
    is_flag=True,
    default=False,
    help="Print samples-present stats (default: on)",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to Kotekan config file",
)
@click.option(
    "--root",
    "-r",
    type=click.Path(exists=True),
    help="Path containing a /yyyy/mm/dd tree",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Print diagnostic output (default: off)",
)
@click.option(
    "--dry-run",
    "-n",
    is_flag=True,
    default=False,
    help="Read the raw file but do not create the archive (default: off)",
)
def cli(file_names, stats, config, root, dry_run, verbose):
    """Convert a raw baseband file into an HDF5 baseband archive"""
    convert(file_names, config, root, stats, dry_run, verbose)


if __name__ == "__main__":
    cli()
