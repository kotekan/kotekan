""" Python interface for validating that a VisRaw object contains valid and expected data"""

import numpy as np

from kotekan.visbuffer import VisRaw

test_patterns = ["default"]


def validate(vis_raw, config, pattern_name=""):
    """Validate an input `vis_raw` filled using `pattern_name` with `config`.

    Parameters
    ----------
    vis_raw : visbuffer.VisRaw
        The output of a Kotekan buffer.
    config : dict
        Structural parameters of the buffer and frames.
        Attributes: num_elements, num_ev, freq, total_frames
    pattern_name : str
        Name of the FakeVisPattern used to fill the buffer.

    vis_raw: VisRaw; pattern_name: str"""

    # Check that the config contains the required information
    for param in ["num_elements", "num_ev", "freq_ids", "total_frames"]:
        assert param in config.keys(), "parameter {} is missing from config".format(
            param
        )

    # Construct vis array
    vis = vis_raw.data["vis"]

    # Extract metadata
    ftime = vis_raw.time["fpga_count"]
    ctime = vis_raw.time["ctime"]
    freq = 800.0 - 400.0 * np.array(config["freq_ids"]) / 1024

    num_elements = config["num_elements"]
    num_ev = config["num_ev"]
    num_freq = len(config["freq_ids"])
    num_time = vis_raw.num_time
    total_frames = config["total_frames"]

    if pattern_name == "default":
        validate_vis(vis, num_elements, ftime, ctime, freq)
        validate_eigenvectors(vis_raw, num_time, num_freq, num_ev, num_elements)


def validate_vis(vis, num_elements, ftime, ctime, freq):
    """Tests that visibility array is populated with integers increasing from zero
    on the diagonal (imaginary part)
    and FPGA sequence number, timestamp, frequency, and frame ID in the first
    four elements (real part)
    the remaining elements are zero"""

    # Check that the diagonals are correct
    pi = 0
    for ii in range(num_elements):
        assert (vis[:, :, pi].imag == ii).all()
        pi += num_elements - ii

    # Check that the times are correct
    assert (vis[:, :, 0].real == ftime[:, np.newaxis].astype(np.float32)).all()
    assert (vis[:, :, 1].real == ctime[:, np.newaxis].astype(np.float32)).all()

    # Check that the frequencies are correct
    vfreq = 800.0 - 400.0 * vis[:, :, 2].real / 1024
    assert (vfreq == freq[np.newaxis, :]).all()


def validate_eigenvectors(vis_raw, num_time, num_freq, num_ev, num_elements):
    """
    Tests the structure of eigenvalues, eigenvectors, erms.
    """
    evals = vis_raw.data["eval"]
    evecs = vis_raw.data["evec"]
    erms = vis_raw.data["erms"]

    # Check datasets are present
    assert evals.shape == (num_time, num_freq, num_ev)
    assert evecs.shape == (num_time, num_freq, num_ev * num_elements)
    assert erms.shape == (num_time, num_freq)

    evecs = evecs.reshape(num_time, num_freq, num_ev, num_elements)

    # Check that the datasets have the correct values
    assert (evals == np.arange(num_ev)[np.newaxis, np.newaxis, :]).all()
    assert (
        evecs.real == np.arange(num_ev)[np.newaxis, np.newaxis, :, np.newaxis]
    ).all()
    assert (
        evecs.imag == np.arange(num_elements)[np.newaxis, np.newaxis, np.newaxis, :]
    ).all()
    assert (erms == 1.0).all()
