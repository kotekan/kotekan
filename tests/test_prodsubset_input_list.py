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
    'prod_subset_type' : 'input_list',
    'input_list' : [1,134], 
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


def input_list_condition(prod, input_list):
   
    prod_in_list = False
    for ipt in input_list :
        if ((prod.input_a==ipt) or (prod.input_b==ipt)):
            prod_in_list = True
            break

    return prod_in_list


def test_subset(subset_data):

#    for frame in subset_data:
#        print frame.metadata.freq_id, frame.metadata.fpga_seq

    for frame in subset_data:
        # With fill_ij, vis_ij = i+j*(1j)
        vis = []

        for ii in range(subset_params['num_prod']):
            prod = visutil.icmap(ii,subset_params['num_elements'])
            if input_list_condition(prod,
                                vis_params['input_list']) :
                vis.append(prod.input_a+1j*prod.input_b)

        assert (frame.vis == np.array(vis)).all()
        assert (frame.evals == np.arange(
                subset_params['num_eigenvectors'])).all()
        evecs = (np.arange(subset_params['num_eigenvectors'],
                    dtype=complex)[:,None]*subset_params['num_elements']
            +np.arange(subset_params['num_elements'])[None,:]).flatten()
        assert (frame.evecs == evecs).all()
        assert (frame.rms == 1.)

