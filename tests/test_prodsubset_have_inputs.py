import pytest
import numpy as np

import kotekan_runner
import visutil

subset_params = {
    'num_elements': 16,
    'num_ev': 2,
    'total_frames': 128,
    'cadence': 5.0,
    'mode': 'fill_ij',
    'freq_ids': [250],
    'buffer_depth': 5
}

vis_params = {
    'prod_subset_type': 'have_inputs',
    'input_list': [1, 134],
}

@pytest.fixture(scope="module")
def subset_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("subset")

    fakevis_buffer = kotekan_runner.FakeVisBuffer(
        freq_ids=subset_params['freq_ids'],
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


def have_inputs_condition(prod, input_list):

    prod_in_list = False
    for ipt in input_list :
        if ((prod.input_a==ipt) or (prod.input_b==ipt)):
            prod_in_list = True
            break

    return prod_in_list


def test_subset(subset_data):

#    for frame in subset_data:
#        print frame.metadata.freq_id, frame.metadata.fpga_seq

    n_el = subset_params['num_elements']
    num_prod = n_el * (n_el + 1) / 2

    for frame in subset_data:
        # With fill_ij, vis_ij = i+j*(1j)
        vis = []

        for ii in range(num_prod):
            prod = visutil.icmap(ii,subset_params['num_elements'])
            if have_inputs_condition(prod,
                                vis_params['input_list']) :
                vis.append(prod.input_a+1j*prod.input_b)

        assert (frame.vis == np.array(vis)).all()
        assert (frame.eval == np.arange(
                subset_params['num_ev'])).all()
        evecs = (np.arange(subset_params['num_ev'])[:, None] +
                 1.0J * np.arange(subset_params['num_elements'])[None, :]).flatten()
        assert (frame.evec == evecs).all()
        assert (frame.erms == 1.)

