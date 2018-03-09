
import pytest
import numpy as np

import kotekan_runner
import visutil

subset_params = {
    'num_elements': 16,
    'num_prod': 120,
    'num_eigenvectors': 2,
    'total_frames': 128,
    'cadence': 5.0,
    'mode':'fill_ij',
    'freq_ids': [250],
    'buffer_depth': 5
}

vis_params = {
    'prod_subset_type' : 'baseline',
    'max_ew_baseline' : 1, 
    'max_ns_baseline' : 63
}

@pytest.fixture(scope="module")
def subset_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("subset")

    fakevis_buffer = kotekan_runner.FakeVisBuffer(
        freq=subset_params['freq_ids'],
        num_frames=subset_params['total_frames']
    )

    dump_buffer = kotekan_runner.DumpVisBuffer(str(tmpdir))

    test = kotekan_runner.KotekanProcessTester(
        'prodSubset', vis_params,
        fakevis_buffer,
        dump_buffer,
        subset_params
    )

    test.run()

    yield dump_buffer.load()

def max_bl_condition(prod,xmax,ymax):

    x_sep = prod.input_a / 512 - prod.input_b / 512
    y_sep = prod.input_a % 256 - prod.input_b % 256
    if x_sep < 0: x_sep = - x_sep
    if y_sep < 0: y_sep = - y_sep

    return (x_sep <= xmax) and (y_sep <= ymax)



def test_subset(subset_data):

#    for frame in subset_data:
#        print frame.metadata.freq_id, frame.metadata.fpga_seq

    for frame in subset_data:
        # With fill_ij, vis_ij = i+j*(1j)
        vis = []
        for ii in range(subset_params['num_prod']):
            prod = visutil.icmap(ii,subset_params['num_elements'])
            if max_bl_condition(prod,
                                vis_params['max_ew_baseline'],
                                vis_params['max_ns_baseline']) :
                vis.append(prod.input_a+1j*prod.input_b)
        
        assert (frame.vis == np.array(vis)).all()
        assert (frame.evals == np.arange(
                subset_params['num_eigenvectors'])).all()
        evecs = (np.arange(subset_params['num_eigenvectors'],
                    dtype=complex)[:,None]*subset_params['num_elements']
            +np.arange(subset_params['num_elements'])[None,:]).flatten()
        assert (frame.evecs == evecs).all()
        assert (frame.rms == 1.)

