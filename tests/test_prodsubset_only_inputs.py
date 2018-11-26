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
    'buffer_depth': 5,
    'prod_subset_type': 'only_inputs',
    'input_list': [1, 134],
}

vis_params = {}

@pytest.fixture(scope="module")
def subset_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("subset")

    fakevis_buffer = kotekan_runner.FakeVisBuffer(
        freq_ids=subset_params['freq_ids'],
        num_frames=subset_params['total_frames'],
        use_dataset_manager=True
    )

    write_buffer = kotekan_runner.VisWriterBuffer(
        str(tmpdir), "raw",
        subset_params['freq_ids'],
        extra_config={'use_dataset_manager': True})

    test = kotekan_runner.KotekanProcessTester(
        'prodSubset', vis_params,
        fakevis_buffer,
        write_buffer,
        subset_params
    )

    test.run()

    return write_buffer.load()


def only_inputs_condition(prod, input_list):

    inpa_in_list = False
    inpb_in_list = False
    for ipt in input_list :
        if (prod.input_a==ipt):
            inpa_in_list = True
        if (prod.input_b==ipt):
            inpb_in_list = True

    return (inpa_in_list and inpb_in_list)


def test_subset(subset_data):

    n_el = subset_params['num_elements']
    num_prod = n_el * (n_el + 1) / 2

    vis = []
    for ii in range(num_prod):
        # With fill_ij, vis_ij = i+j*(1j)
        prod = visutil.icmap(ii,subset_params['num_elements'])
        if only_inputs_condition(prod,
                            subset_params['input_list']) :
            vis.append(prod.input_a+1j*prod.input_b)

    evecs = (np.arange(subset_params['num_ev'])[:, None] +
             1.0J * np.arange(subset_params['num_elements'])[None, :]).flatten()

    for t in range(subset_params['total_frames']):
        for f in range(len(subset_params['freq_ids'])):
            frame = subset_data[t][f]
            assert (frame.vis == np.array(vis)).all()
            assert (frame.eval == np.arange(
                    subset_params['num_ev'])).all()
            assert (frame.evec == evecs).all()
            assert (frame.erms == 1.)

