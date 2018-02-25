import subprocess
import os
import pytest
import numpy as np

import visbuffer


accumulate_params = {
    'num_elements': 4,
    'num_eigenvectors': 2,
    'samples': 32768,
    'int_frames': 64,
    'total_frames': 257,  # Go for one extra sample to ensure we actually get 256
    'block_size': 2,
    'freq': 777
}


@pytest.fixture(scope="module")
def accumulate_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("accumulate")

    config = """
---
type: config
log_level: info
num_elements: %(num_elements)i
num_local_freq: 1
samples_per_data_set: %(samples)i
buffer_depth: 4
num_gpu_frames: %(int_frames)i
block_size: %(block_size)i
cpu_affinity: []

# Metadata pool
main_pool:
    kotekan_metadata_pool: chimeMetadata
    num_metadata_objects: 5 * buffer_depth

vis_pool:
    kotekan_metadata_pool: visMetadata
    num_metadata_objects: 5 * buffer_depth

# Buffers
gpu_buf0:
    metadata_pool: main_pool
    num_frames: buffer_depth
    sizeof_int: 4
    frame_size: sizeof_int * num_local_freq * ((num_elements * num_elements) + (num_elements * block_size))
    kotekan_buffer: standard

vis_buf0:
    metadata_pool: vis_pool
    num_frames: buffer_depth
    sizeof_int: 4
    frame_size: 10 * sizeof_int * num_local_freq * num_elements * num_elements
    kotekan_buffer: standard

# Define the minimal processes to run the test
fakegpu0:
    pattern: accumulate
    num_frames: %(total_frames)i
    pre_accumulate: true
    kotekan_process: fakeGpuBuffer
    freq: %(freq)i
    wait: false
    out_buf: gpu_buf0

acc0:
    kotekan_process: visAccumulate
    in_buf: gpu_buf0
    out_buf: vis_buf0
    num_eigenvectors: %(num_eigenvectors)i

dump:
    kotekan_process: rawFileWrite
    base_dir: %(path)s
    file_name: vis
    file_ext: dump
    in_buf: vis_buf0
"""

    config = config % dict(accumulate_params, path=str(tmpdir))

    config_file = tmpdir.join("config.yaml")
    config_file.write(config)

    kotekan_dir = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                                "..", "build", "kotekan"))

    print kotekan_dir

    subprocess.check_call(["./kotekan", "-c", str(config_file)],
                          stdout=subprocess.PIPE, cwd=kotekan_dir)

    dump_data = visbuffer.VisBuffer.load_files(str(tmpdir) + '/*vis*.dump')

    yield dump_data


def test_structure(accumulate_data):

    n = accumulate_params['num_elements']

    # Check that each samples is the expected shape
    for dump in accumulate_data:
        assert dump.metadata.num_elements == n
        assert dump.metadata.num_prod == (n * (n + 1) / 2)
        assert dump.metadata.num_eigenvectors == accumulate_params['num_eigenvectors']

    # Check that we have the expected number of samples
    nsamp = accumulate_params['total_frames'] / accumulate_params['int_frames']
    assert len(accumulate_data) == nsamp


def test_metadata(accumulate_data):

    for dump in accumulate_data:
        assert dump.metadata.freq_id == accumulate_params['freq']
        assert dump.metadata.dataset_id == 0


def test_time(accumulate_data):

    delta_samp = accumulate_params['samples'] * accumulate_params['int_frames']

    for ii, dump in enumerate(accumulate_data):
        assert dump.metadata.fpga_seq == ii * delta_samp


def test_accumulate(accumulate_data):

    row, col = np.triu_indices(accumulate_params['num_elements'])

    pat = (row + 1.0J * col).astype(np.complex64)

    for dump in accumulate_data:

        assert (dump.vis == pat).all()
        assert (dump.weight == 8.0).all()
