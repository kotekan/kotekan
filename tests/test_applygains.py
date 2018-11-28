import pytest
import numpy as np
import h5py
import kotekan_runner
import visutil
import time

print kotekan_runner.__file__

old_tmstp = time.time() 
old_tag = "gains{0}".format(int(old_tmstp))

global_params = {
    'num_elements': 16,
    'num_ev': 4,
    'total_frames': 20,
    'cadence': 1.0,
    'mode': 'fill_ij',
    'freq_ids': [250],
    'buffer_depth': 8,
    'updatable_config': "/gains",
    'gains': {'kotekan_update_endpoint': "json",
              'start_time': old_tmstp,
              'tag': old_tag
    },
    'wait': True,
    'combine_gains_time': 10.,
    'num_threads': 4,
    'dataset_manager': {
        'use_dataset_broker': False
    },
}


def gen_gains(gains_dir, tag=None, mult_factor=1.,
              nelem=global_params['num_elements'], nfreq=1024):

    if tag is None:
        tag = global_params['gains']['tag']
    filepath = gains_dir.join(tag+'.hdf5')
    f = h5py.File(str(filepath), "w")

    dset = f.create_dataset('gain', (nfreq, nelem), dtype='c8')
    gain = (np.arange(nfreq, dtype='f2')[:, None] 
            * 1j*np.arange(nelem, dtype='f2')[None, :])
    dset[...] = gain * mult_factor

    dset2 = f.create_dataset('weight', (nfreq, nelem), dtype='f')
    weight = np.ones((nfreq, nelem), dtype=float)
    # Make some weights zero to test the behaviour of apply_gains
    weight[:, 1] = 0.
    weight[:, 3] = 0. 
    dset2[...] = weight

    freq_ds = f.create_dataset('index_map/freq', (nfreq,), dtype='f')
    ipt_ds = f.create_dataset('index_map/input', (nelem,), dtype='i')

    freq_ds[...] = np.linspace(800., 400., nfreq, dtype=float)
    ipt_ds[:] = np.arange(nelem)

    f.close()


def apply_data(cmds, tmpdir_factory):

    apply_dir = tmpdir_factory.mktemp("apply")

    fakevis_buffer = kotekan_runner.FakeVisBuffer(
        freq_ids=global_params['freq_ids'],
        num_frames=global_params['total_frames'],
        wait=global_params['wait']
    )

    out_dump_buffer = kotekan_runner.DumpVisBuffer(str(apply_dir))

    test = kotekan_runner.KotekanProcessTester(
        'applyGains', global_params,
        buffers_in=fakevis_buffer,
        buffers_out=out_dump_buffer,
        global_config=global_params,
        rest_commands=cmds
    )

    test.run()

    return out_dump_buffer.load()


def load_gains(filepath, fr, ipt=None):

    gains = h5py.File(filepath, 'r')['gain']
    if ipt is None:
        return gains[fr]
    else:
        return gains[fr, ipt]


def load_gain_weight(filepath, fr, ipt=None):

    gain_weight = h5py.File(filepath, 'r')['weight']
    if ipt is None:
        return gain_weight[fr]
    else:
        return gain_weight[fr, ipt]


def combine_gains(tframe, tcombine, new_tmstp, old_tmstp, 
                  new_gains, old_gains):
    if tframe < old_tmstp:
        # TODO: This should not happen. Add a warning or a raise?
        pass
    elif tframe < new_tmstp:
        return old_gains
    elif tframe < new_tmstp + tcombine:
        tpast = tframe - new_tmstp
        new_coef = tpast/tcombine
        old_coef = 1. - new_coef
        return new_coef * new_gains + old_coef * old_gains
    else:
        return new_gains


def test_apply(tmpdir_factory):

    gains_dir = tmpdir_factory.mktemp("gains")
    new_tmstp = time.time() + 5.
    new_tag = 'gains{0}'.format(int(new_tmstp))

    gen_gains(gains_dir)
    gen_gains(gains_dir, tag=new_tag, mult_factor=2.)
    old_filepath = gains_dir.join(old_tag+'.hdf5')
    new_filepath = gains_dir.join(new_tag+'.hdf5')

    global_params['gains_dir'] = str(gains_dir)
    tcombine = global_params['combine_gains_time']
    n_el = global_params['num_elements']
    num_prod = n_el * (n_el + 1) / 2

    # REST commands
    cmds = [["post", "gains", 
             {'tag': new_tag,
              'start_time': new_tmstp}]]
    gains_dump = apply_data(cmds, tmpdir_factory)

    for frame in gains_dump:
        frqid = frame.metadata.freq_id
        old_gains = load_gains(old_filepath, frqid)
        new_gains = load_gains(new_filepath, frqid)
        gain_weight = load_gain_weight(new_filepath, frqid)
        frame_tmstp = visutil.ts_to_double(frame.metadata.ctime)  
        gains = combine_gains(frame_tmstp, tcombine, new_tmstp, old_tmstp, 
                              new_gains, old_gains)

        weight_factor = np.ones(n_el)
        for ii in range(len(gain_weight)):
            if gain_weight[ii] == 0.:
                gains[ii] = 1.
                weight_factor[ii] = 0.
            elif abs(gains[ii]) == 0.:
                gains[ii] = 1.
                weight_factor[ii] = 0.
            else:
                weight_factor[ii] = 1./abs(gains[ii])**2

        expvis = np.zeros(num_prod, dtype=frame.vis[:].dtype)
        for ii in range(num_prod):
            prod = visutil.icmap(ii, global_params['num_elements'])
            # With fill_ij, vis_ij = i+j*(1j)
            expvis[ii] = ((prod.input_a + 1j*prod.input_b)
                          * (gains[prod.input_a])
                          * np.conj((gains[prod.input_b])))
        assert (abs(frame.vis[:].real - expvis.real)
                <= 1E-5*abs(frame.vis[:].real)).all()
        assert (abs(frame.vis[:].imag - expvis.imag)
                <= 1E-5*abs(frame.vis[:].imag)).all()

        assert (frame.eval == np.arange(
                global_params['num_ev'])).all()
        evecs = (np.arange(global_params['num_ev'])[:, None]
                 + 1.0J 
                 * np.arange(global_params['num_elements'])[None, :]).flatten()
        assert (frame.evec == evecs).all()
        assert (frame.erms == 1.)
        # This relies on the fact that the initial value 
        # of the gains is 1 in fakevis.
        assert (abs(frame.gain.real - gains.real) 
                <= 1E-5*abs(gains.real)).all()
        assert (abs(frame.gain.imag - gains.imag) 
                <= 1E-5*abs(gains.imag)).all()

        expweight = []
        for ii in range(n_el):
            for jj in range(ii, n_el):
                expweight.append(1.*weight_factor[ii]*weight_factor[jj])
        expweight = np.array(expweight)
        assert (abs(frame.weight - expweight)
                <= 1E-5*abs(frame.weight)).all()
