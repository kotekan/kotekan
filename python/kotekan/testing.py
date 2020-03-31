""" Python interface for validating that a VisRaw object contains valid and expected data"""

import numpy as np

from kotekan.visbuffer import VisRaw

def validate(vis_raw, config, pattern_name=''):
    '''Validate an input `vis_raw` filled using `pattern_name` with `config`.

    Parameters
    ----------
    vis_raw : visbuffer.VisRaw
        The output of a Kotekan buffer.
    config : dict
        Structural parameters of the buffer and frames.
        Attributes: num_elements, num_ev, freq, total_frames
    pattern_name : str
        Name of the FakeVisPattern used to fill the buffer.

    vis_raw: VisRaw; pattern_name: str'''


    # Check that the config contains the required information
    for param in ['num_elements', 'num_ev', 'freq_ids', 'total_frames']:
        assert param in config.keys(), 'parameter {} is missing from config'.format(param)

    # i believe this is for the "DefaultVisPattern
    # The visibility array is populated with integers increasing from zero
    # on the diagonal (imaginary part)
    # and FPGA sequence number, timestamp, frequency, and frame ID in the first
    # four elements (real part)
    # the remaining elements are zero
    # Construct vis array

    vis = vis_raw.data["vis"]

    # Extract metadata
    ftime = vis_raw.time["fpga_count"]
    ctime = vis_raw.time["ctime"]
    freq = 800.0 - 400.0 * np.array(config["freq_ids"]) / 1024
    input_a = np.array([p[0] for p in vis_raw.index_map["prod"]])
    input_b = np.array([p[1] for p in vis_raw.index_map["prod"]])

    num_elements = config["num_elements"]
    num_ev = config["num_ev"]
    num_freq = len(config["freq_ids"])
    num_time = vis_raw.num_time
    total_frames = config["total_frames"]

    # Check that the diagonals are correct
    pi = 0
    for ii in range(num_elements):
        assert (vis[:, :, pi].imag == ii).all()
        pi += num_elements - ii

    # Check that the times are correct
    assert (vis[:, :, 0].real == ftime[:, np.newaxis].astype(np.float32)).all()
    assert (vis[:, :, 1].real == ctime[:, np.newaxis].astype(np.float32)).all()

    if hasattr(config, "cadence"):
        assert np.allclose(np.diff(ctime), config["cadence"])

    # Check that the frequencies are correct
    vfreq = 800.0 - 400.0 * vis[:, :, 2].real / 1024
    assert (vfreq == freq[np.newaxis, :]).all()

    # Check the products
    ia, ib = np.triu_indices(num_elements)
    assert (input_a == ia).all()
    assert (input_b == ib).all()

    #### Test eigenvectors

    evals = vis_raw.data["eval"]
    evecs = vis_raw.data["evec"]
    erms = vis_raw.data["erms"]

    # Check datasets are present
    assert evals.shape == (num_time, num_freq, num_ev)
    assert evecs.shape == (num_time, num_freq, num_ev * num_elements)
    assert erms.shape == (num_time, num_freq)

    evecs = evecs.reshape(num_time, num_freq, num_ev, num_elements)

    im_ev = np.array(vis_raw.index_map["ev"])

    # Check that the index map is there correctly
    assert (im_ev == np.arange(num_ev)).all()

    # Check that the datasets have the correct values
    assert (evals == np.arange(num_ev)[np.newaxis, np.newaxis, :]).all()
    assert (
            evecs.real == np.arange(num_ev)[np.newaxis, np.newaxis, :, np.newaxis]
            ).all()
    assert (
            evecs.imag == np.arange(num_elements)[np.newaxis, np.newaxis, np.newaxis, :]
            ).all()
    assert (erms == 1.0).all()
