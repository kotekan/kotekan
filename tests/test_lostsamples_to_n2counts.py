"""Tests for `lostSampesToN2Counts` stage."""

import pytest

from kotekan import runner

# TODO: should this be a skip or a fail?
if not runner.has_hdf5():
    pytest.skip("HDF5 support not available; skipping...")

import h5py
import numpy as np

from pathlib import Path

global_params: dict = {
    "log_level": "DEBUG",
    "num_elements": 2048,  # CHIME
    "counts_ntiles": (2048 / 64) * (1 + 2048 / 64) / 2,
    "sizeof_int": 4,
}


class FakeRFIBuffer(runner.InputBuffer):
    """Create an input rfi mask using `testDataGen`.

    Parameters
    ----------
    samples_per_data_set : int
        Number of FPGA time samples per dataset.
    num_local_freq : int
        Number of frequencies present in this buffer.
    **kwargs : dict
        Parameters fed straight into the stage config.
    """

    _buf_ind: int = 0

    def __init__(self, samples_per_data_set: int, num_local_freq: int, **kwargs: dict):

        self.name = f"fake_rfi_buf_{self._buf_ind:,}"
        stage_name = f"fake_gen_rfi_{self._buf_ind:,}"

        self.__class__._buf_ind += 1

        self.buffer_block = {
            self.name: {
                "kotekan_buffer": "standard",
                "metadata_pool": "main_pool",
                "num_frames": "buffer_depth",
                "frame_size": "(samples_per_data_set * num_local_freq) / 8",
            }
        }

        stage_config = {
            "kotekan_stage": "testDataGen",
            "out_buf": self.name,
            "type": "const1x8",
            "value": 85,  # alternating 10101010...
            "name": stage_name,
            "array_shape": [samples_per_data_set // 1024, num_local_freq, 128],
            "dim_name": ["T8hi128", "F", "T8lo128"],
        }

        stage_config.update(**kwargs)

        self.stage_block = {stage_name: stage_config}


class FakeLostSamplesN2Buffer(runner.InputBuffer):
    """Create input `lost_samples` buffer using `testDataGen`.

    This is specific to the N2 pipeline and produces a single mask
    value per FPGA time sample.

    Parameters
    ----------
    samples_per_data_set : int
        Number of FPGA time samples.
    **kwargs : dict
        Additional parameters fed to the stage config.
    """

    _buf_ind: int = 0

    def __init__(self, samples_per_data_set: int, **kwargs: dict):
        self.name = f"fake_lost_samples_buf_{self._buf_ind}"
        stage_name = f"fake_gen_lost_samples_{self._buf_ind}"

        self.__class__._buf_ind += 1

        self.buffer_block = {
            self.name: {
                "kotekan_buffer": "standard",
                "metadata_pool": "main_pool",
                "num_frames": "buffer_depth",
                "frame_size": "samples_per_data_set",
            }
        }

        stage_config = {
            "kotekan_stage": "testDataGen",
            "out_buf": self.name,
            "type": "const8",
            "value": 0,  # no missing samples
            "name": stage_name,
            "array_shape": [samples_per_data_set],
            "dim_name": ["T"],
        }

        stage_config.update(**kwargs)

        self.stage_block = {stage_name: stage_config}


class DumpCountsBuffer(runner.OutputBuffer):
    """Consume a N2 Counts buffer and provide its content as `CountsBuffer` objects.

    Parameters
    ----------
    output_dir : str
        Temp. directory to output to. Dumped files are not removed.
    """

    _buf_ind = 0

    name = None

    def __init__(self, output_dir, num_elements=None):
        self.name = f"dumpcounts_buf{self._buf_ind}"
        stage_name = f"dump{self._buf_ind}"

        self.__class__._buf_ind += 1

        self.output_dir = output_dir
        self.num_elements = num_elements

        self.buffer_block = {
            self.name: {
                "kotekan_buffer": "ndarray",
                "metadata_pool": "main_pool",
                "num_frames": "buffer_depth",
                "value_type": "int32",
                "quantity_name": "n2k_counts",
                "extents": [
                    "samples_per_data_set / sub_integration_ntime",
                    "num_local_freq",
                    "counts_ntiles",
                    8,
                    8,
                ],
                "dimnames": ["Tc", "F", "D8Phi", "D8Plo1", "D8Plo2"],
            }
        }

        self.stage_block = {
            stage_name: {
                "kotekan_stage": "hdf5FileWrite",
                "in_buf": self.name,
                "file_name": self.name,
                "base_dir": output_dir,
                "prefix_hostname": False,
                "max_frames": "buffer_depth",
            }
        }

    def load(self):
        """Load the output data from the buffer.

        Returns
        -------
        dumps : lost of buffers
            Buffer output.
        """
        import glob

        files = glob.glob(f"{self.output_dir}/*{self.name}*.h5")

        dsets = []

        for file in files:
            # HDF5 file write produces a file called
            # <file_name>.<_buf_ind>.h5, so strip off
            # the number when accessing the dataset name
            dset_name = Path(file).stem.split(".")[0]

            dsets.append(h5py.File(file, "r")[dset_name])

        return dsets


def _generate_input_rfibuf(
    nsamples: int, nfreq: int
) -> tuple[FakeRFIBuffer, np.ndarray]:
    """Buffer (and matching array) representing a `uint1` RFI mask.

    Has shape (nsamples // 128 // 8, nfreq, 128).

    Parameters
    ----------
    nsamples
        Number of FPGA time samples
    nfreq
        Number of frequencies per node

    Returns
    -------
    buf : runner.FakeRFIBuffer
        Buffer instance filled with alternating ones and zeros.
    arr : np.ndarray
        Numpy array of same size filled with alternating ones and zeros.
    """
    buf = FakeRFIBuffer(samples_per_data_set=nsamples, num_local_freq=nfreq)

    # Creates an array of size nfreq * nsamples / 8 filled with
    # value 255, unpacked into a nfreq * nsamples array of ones
    arr = np.ones(nsamples * nfreq, dtype=np.uint8)
    arr[1::2] = 0  # Alternating 10101010... to match kotekan-generated

    return buf, arr


def _generate_input_lost_samples_bufs(
    nsamples: int, nbufs: int
) -> tuple[list[FakeLostSamplesN2Buffer], list[np.ndarray]]:
    """Buffers (and matching arrays) representing `uint8` lost samples masks.

    `nbufs` buffers of size `nsamples`.

    Parameters
    ----------
    nsamples
        Number of FPGA time samples
    nbufs
        Number of individual buffers to produce

    Returns
    -------
    buffers : list
        List of `FakeLostSamplesN2Buffer` objects.
    arrays : list
        List of numpy arrays matching the buffer data.
    """
    buffers = []

    # This is set to match the corresponding numpy array.
    # If this is changed, must change the array as well
    # fill_type_args: dict = {"type": "ramp", "value": 1}
    fill_type_args: dict = {}

    for n in range(nbufs):
        buffers.append(
            FakeLostSamplesN2Buffer(samples_per_data_set=nsamples, **fill_type_args)
        )

    # Create the matching numpy array
    # arr = (np.arange(nsamples) % 256).astype(np.uint8)
    arr = np.zeros(nsamples, dtype=np.uint8)

    return buffers, [arr.copy() for _ in range(nbufs)]


def accumulate_numpy(
    nsamples: int,
    nfreq: int,
    sub_integration_time: int,
    rfiarr: np.ndarray,
    lost_samples_arrs: list[np.ndarray],
) -> np.ndarray:
    """Combine and integrate array copies of input buffers.

    This should be used to test against the result of `accumulate`.
    Parameters
    ----------
    nsamples
        Number of FPGA time samples per frame
    nfreq
        Number of frequencies per RFI buffer
    sub_integration_time
        Number of time samples per subintegration
    rfiarr
        Array corresponding to the RFI mask. This is unpacked, so
        each byte represents a bit in the mask.
    lost_samples_arrs
        List of 1D arrays corresponding to lost_samples masks

    Returns
    -------
    expected_output : np.ndarray
        Array created by multiplying and accumulating the input masks
    """
    # Sort out the expected RFI buffer dimensions
    # This is the number of coarse times samples
    t_hi = nsamples // 1024

    # Split the array into coarse time samples and concatenate into
    # a single contiguous time axis (shape: freq, time)
    rtc = []
    for arr in np.split(rfiarr, t_hi):
        # Reshape the (freq, fine_time) array and append
        rtc.append(arr.reshape(nfreq, -1))

    # Concatenate across the frequency axis
    rtc = np.concatenate(rtc, axis=-1)

    # Sort out the expected number of frequencies represented
    # by each lost_samples buffer
    nbufs = len(lost_samples_arrs)

    if nfreq % nbufs:
        raise RuntimeError(
            f"Number of lost_samples buffers ({nbufs}) does not evenly divide frequencies ({nfreq})."
        )

    nfreq_per_buf = nfreq // nbufs

    expected_output = np.zeros(
        (nsamples // sub_integration_time, nfreq), dtype=np.int32
    )

    for ii, buf in enumerate(lost_samples_arrs):
        # reshape to sum over the last axis for accumulation
        ea = buf.reshape(-1, sub_integration_time)
        # Choose which frequencies to accumulate here
        sl = slice(ii * nfreq_per_buf, (ii + 1) * nfreq_per_buf)
        eb = rtc[sl].reshape(-1, sub_integration_time)
        # Should automatically broadcast length-1 axis in ea against
        # freq slice in eb. Need this slightly annoying double transpose to
        # handle varying axis lengths
        expected_output[:, sl] = np.atleast_2d(
            ((ea ^ 1) & eb).astype(np.int32).sum(axis=-1).T
        ).T

    return expected_output


def accumulate(
    tmpdir: str,
    nsamples: int,
    nfreq: int,
    sub_integration_ntime: int,
    rfibuf: FakeRFIBuffer,
    lost_samples_bufs: list[FakeLostSamplesN2Buffer],
) -> list[h5py.File]:
    """Get and process a combination of input buffers.

    Parameters
    ----------
    nsamples
        Number of FPGA time samples
    nfreq
        Number of frequencies
    nbufs
        Number of lost samples buffer to produce
    """
    # This needs to be in the global config to calculate the
    # counts buffer tile size
    gp = {
        **global_params,
        "sub_integration_ntime": sub_integration_ntime,
        "samples_per_data_set": nsamples,
        "num_local_freq": nfreq,
    }

    output_buf = DumpCountsBuffer(
        str(tmpdir), num_elements=global_params["num_elements"]
    )

    run_params = {
        "num_n2k_freq": nfreq,  # This has a different name from num_local_freq in the stage
        "sub_integration_ntime": sub_integration_ntime,
        "samples_per_data_set": nsamples,
        "num_elements": global_params["num_elements"],
        "rfi_mask_buf": rfibuf.name,
        "lost_samples_buffers": [buf.name for buf in lost_samples_bufs],
        "n2k_counts_buf": output_buf.name,
    }

    test = runner.KotekanStageTester(
        "lostSamplesToN2Counts",
        run_params,
        [rfibuf, *lost_samples_bufs],
        output_buf,
        gp,
    )

    test.run()

    return output_buf.load()


@pytest.mark.parametrize("nsamples", [2048])
@pytest.mark.parametrize("nfreq", [1, 3, 4])
@pytest.mark.parametrize("sub_integration_ntime", [1, 8, 128, 512])
def test_accumulate(tmpdir_factory, nsamples, nfreq, sub_integration_ntime):
    """Test the numerical correctness of the stage accumulation.

    These are integers and should be an exact match.
    """
    rfibuf, rfiarr = _generate_input_rfibuf(nsamples, nfreq)
    lost_samples_bufs, lost_samples_arrs = _generate_input_lost_samples_bufs(
        nsamples, nfreq
    )

    arr_accum = accumulate_numpy(
        nsamples, nfreq, sub_integration_ntime, rfiarr, lost_samples_arrs
    )

    tmpdir = tmpdir_factory.mktemp(f"tmp_{nsamples}_{nfreq}_{sub_integration_ntime}")

    _expected_shape = [
        nsamples // sub_integration_ntime,
        nfreq,
        global_params["counts_ntiles"],
        8,
        8,
    ]
    _expected_dim_names = ["Tc", "F", "D8Phi", "D8Plo1", "D8Plo2"]

    # This is a generator, so use a loop even if returning a single dset
    for dset in accumulate(
        tmpdir, nsamples, nfreq, sub_integration_ntime, rfibuf, lost_samples_bufs
    ):
        # Check that metadata is correct
        assert dset.attrs["name"] == "n2k_counts"
        assert dset.attrs["type"] == "int32"
        assert dset.attrs["time_downsampling_fpga"] == sub_integration_ntime
        assert (dset.attrs["dim_names"] == _expected_dim_names).all()

        # Check the payload
        arr = dset[:]
        assert arr.dtype == np.int32
        assert np.all([arrs == es for arrs, es in zip(arr.shape, _expected_shape)])
        # Check that the payload is as-expected and non-zero
        assert np.any(arr > 0)
        # Compare to the numpy-accumulated buffer
        assert np.all([a == b for a, b in zip(arr.shape[:2], arr_accum.shape)])
        assert (arr[:, :, 0, 0, 0] == arr_accum).all()
