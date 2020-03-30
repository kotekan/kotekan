""" Python interface for validating that a VisRaw object contains valid and expected data"""

import numpy as np

from kotekan.visbuffer import VisRaw

def validate(vis_raw, pattern_name=''):
    '''vis_raw: VisRaw; pattern_name: str'''

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
    freq = np.array([f["centre"] for f in vis_raw.index_map["freq"]])
    num_elements = len(vis_raw.index_map["input"])
    input_a = np.array([p[0] for p in vis_raw.index_map["prod"]])
    input_b = np.array([p[1] for p in vis_raw.index_map["prod"]])

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

    # Check the products
    ia, ib = np.triu_indices(num_elements)
    assert (input_a == ia).all()
    assert (input_b == ib).all()
