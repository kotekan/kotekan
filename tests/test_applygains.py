import pytest
import numpy as np
import h5py
import kotekan_runner
import visutil
import time

print kotekan_runner.__file__

general_params = {
    'num_elements': 16,
    'num_ev': 4,
    'total_frames': 128,
    'cadence': 5.0,
    'mode': 'fill_ij',
    'freq_ids': [250],
    'buffer_depth': 5
}


@pytest.fixture(scope="module")
def gen_gains(tmpdir_factory, nelem=general_params['num_elements'], nfreq=1024):

    tmpgainsdir = tmpdir_factory.mktemp("gains")
    filepath = tmpgainsdir.join("gains.hdf5")
    f = h5py.File(str(filepath), "w")

    dset = f.create_dataset('gain', (nfreq, nelem), dtype='c8')
    gain = np.zeros((nfreq, nelem), dtype='c8')
    gain = (np.arange(nfreq, dtype='f2')[:, None] 
            * 1j*np.arange(nelem, dtype='f2')[None, :]+1.)
    dset[...] = gain

    dset2 = f.create_dataset('weight', (nfreq,), dtype='f')
    dset2[...] = np.arange(nfreq, dtype=float) * 0.5

    freq_ds = f.create_dataset('index_map/freq', (nfreq,), dtype='f')
    ipt_ds = f.create_dataset('index_map/input', (nelem,), dtype='i')

    freq_ds[...] = np.linspace(800., 400., nfreq, dtype=float)
    ipt_ds[:] = np.arange(nelem)

    # TODO: delete
    #gains_stt_time_ds = f.create_dataset('gains_stt_time', (1,), dtype='i')
    #gains_stt_time_ds[:] = np.array([int(time.time())], dtype=int)

    f.close()

    return filepath


@pytest.fixture(scope="module")
def apply_data(tmpdir_factory, gen_gains):

    tmpdir = tmpdir_factory.mktemp("apply")

    fakevis_buffer = kotekan_runner.FakeVisBuffer(
        freq_ids=general_params['freq_ids'],
        num_frames=general_params['total_frames']
    )

    dump_buffer = kotekan_runner.DumpVisBuffer(str(tmpdir))

    filepath = gen_gains
    apply_params = {
        'gains_path': str(filepath),
        'num_kept_updates': 4
    }

    test = kotekan_runner.KotekanProcessTester(
        'applyGains', apply_params,
        fakevis_buffer,
        dump_buffer,
        general_params
    )

    test.run()

    yield dump_buffer.load()


def load_gains(filepath, fr, ipt=None):

    gains = h5py.File(filepath, 'r')['gain']
    if ipt is None:
        return gains[fr]
    else:
        return gains[fr, ipt]


def test_apply(apply_data, gen_gains):

    filepath = gen_gains
    n_el = general_params['num_elements']
    num_prod = n_el * (n_el + 1) / 2

    for frame in apply_data:
        frqid = frame.metadata.freq_id
        gains = load_gains(filepath, frqid)
        expvis = np.zeros(num_prod, dtype=frame.vis[:].dtype)
        for ii in range(num_prod):
            prod = visutil.icmap(ii, general_params['num_elements'])
            # With fill_ij, vis_ij = i+j*(1j)
            expvis[ii] = ((prod.input_a + 1j*prod.input_b)
                          * (gains[prod.input_a])
                          * np.conj((gains[prod.input_b])))
        real_fracdiff = (frame.vis[:].real - expvis.real)/(frame.vis[:].real+1E-10)
        imag_fracdiff = (frame.vis[:].imag - expvis.imag)/(frame.vis[:].imag+1E-10)
        assert (real_fracdiff < 1E-5).all()
        assert (imag_fracdiff < 1E-5).all()
        assert (frame.eval == np.arange(
                general_params['num_ev'])).all()
        evecs = (np.arange(general_params['num_ev'])[:, None]
                 + 1.0J 
                 * np.arange(general_params['num_elements'])[None, :]).flatten()
        assert (frame.evec == evecs).all()
        assert (frame.erms == 1.)
        assert (frame.gain == gains).all()
        expweight = []
        for ii in range(n_el):
            for jj in range(ii, n_el):
                expweight.append(1./abs(gains[ii]*gains[jj])**2)
        weight_fracdiff = (frame.weight - expweight)/(frame.weight)
        assert (weight_fracdiff < 1E-5).all()
