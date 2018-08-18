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
    'buffer_depth': 5,
    'updatable_config': "/dynamic_attributes/gains",
    'dynamic_attributes': {
        'flagging': {'kotekan_update_endpoint': "json",
                     'bad_inputs': [1, 4],
                     'timestamp': old_tmstp
                    },
        'gains': {'kotekan_update_endpoint': "json",
                  'gains_timestamp': old_tmstp,
                  'tag': old_tag
                 }
    },
    'wait': True,
    'num_kept_updates': 4,
    'combine_gains_time': 10.
}


def gen_gains(gains_dir, tag=None, mult_factor=1.,
              nelem=global_params['num_elements'], nfreq=1024):

    if tag is None:
        tag = global_params['dynamic_attributes']['gains']['tag']
    filepath = gains_dir.join(tag+'.hdf5')
    f = h5py.File(str(filepath), "w")

    dset = f.create_dataset('gain', (nfreq, nelem), dtype='c8')
    gain = np.zeros((nfreq, nelem), dtype='c8')
    gain = (np.arange(nfreq, dtype='f2')[:, None] 
            * 1j*np.arange(nelem, dtype='f2')[None, :]
            + 1. + 1j)
    dset[...] = gain * mult_factor

    dset2 = f.create_dataset('weight', (nfreq,), dtype='f')
    dset2[...] = np.arange(nfreq, dtype=float) * 0.5

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
    cmds = [["post", "dynamic_attributes/gains", 
             {'tag': new_tag,
              'gains_timestamp': new_tmstp}]]
    gains_dump = apply_data(cmds, tmpdir_factory)

    for frame in gains_dump:
        frqid = frame.metadata.freq_id
        old_gains = load_gains(old_filepath, frqid)
        new_gains = load_gains(new_filepath, frqid)
        frame_tmstp = visutil.ts_to_double(frame.metadata.ctime)  
        gains = combine_gains(frame_tmstp, tcombine, new_tmstp, old_tmstp, 
                              new_gains, old_gains)
        expvis = np.zeros(num_prod, dtype=frame.vis[:].dtype)
        for ii in range(num_prod):
            prod = visutil.icmap(ii, global_params['num_elements'])
            # With fill_ij, vis_ij = i+j*(1j)
            expvis[ii] = ((prod.input_a + 1j*prod.input_b)
                          * (gains[prod.input_a])
                          * np.conj((gains[prod.input_b])))
        real_fracdiff = abs(frame.vis[:].real 
                            - expvis.real)/abs(frame.vis[:].real+1E-10)
        imag_fracdiff = abs(frame.vis[:].imag 
                            - expvis.imag)/abs(frame.vis[:].imag+1E-10)
        assert (real_fracdiff < 1E-5).all()
        assert (imag_fracdiff < 1E-5).all()
        assert (frame.eval == np.arange(
                global_params['num_ev'])).all()
        evecs = (np.arange(global_params['num_ev'])[:, None]
                 + 1.0J 
                 * np.arange(global_params['num_elements'])[None, :]).flatten()
        assert (frame.evec == evecs).all()
        assert (frame.erms == 1.)
        # This relies on the fact that the initial value 
        # of the gains is 1 in fakevis.
        greal_fracdiff = abs(frame.gain.real - gains.real)/abs(gains.real)
        gimag_fracdiff = abs(frame.gain.imag - gains.imag)/abs(gains.imag)
        assert (greal_fracdiff < 1E-5).all()
        assert (gimag_fracdiff < 1E-5).all()
        expweight = []
        for ii in range(n_el):
            for jj in range(ii, n_el):
                expweight.append(1./abs(gains[ii]*gains[jj])**2)
        expweight = np.array(expweight)
        weight_fracdiff = abs(frame.weight - expweight)/(frame.weight)
        assert (weight_fracdiff < 1E-5).all()
