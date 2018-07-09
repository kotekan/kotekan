import os
import itertools
import subprocess
import tempfile

import visbuffer


class KotekanRunner(object):
    """A lightweight class for running Kotekan from Python.

    Parameters
    ----------
    buffers : dict
        Dictionary containing all the buffers and their configuration (as
        dicts). Config is as it is in the config files.
    processs : dict
        Dictionary with all the process definitions.
    config : dict
        Global configuration at the root level.
    """

    def __init__(self, buffers=None, processes=None, config=None):

        self._buffers = buffers if buffers is not None else {}
        self._processes = processes if processes is not None else {}
        self._config = config if config is not None else {}

    def run(self):
        """Run kotekan.

        This configures kotekan by creating a temporary config file.
        """

        import yaml

        config_dict = yaml.load(default_config)
        config_dict.update(self._config)

        # At somepoint maybe do more specialised parsing and validation here
        config_dict.update(self._buffers)
        config_dict.update(self._processes)

        kotekan_dir = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                                    "..", "build", "kotekan"))

        with tempfile.NamedTemporaryFile() as fh:
            yaml.dump(config_dict, fh)
            fh.flush()
            print config_dict
            # Capture output and print it out
            self.output = subprocess.check_output(["./kotekan", "-c", fh.name],
                                                  cwd=kotekan_dir,
                                                  stderr=subprocess.STDOUT)
            print self.output


class InputBuffer(object):
    """Base class for an input buffer generator."""
    name = None


class OutputBuffer(object):
    """Base class for an output buffer consumer."""
    name = None


class FakeGPUBuffer(InputBuffer):
    """Create an input GPU format buffer and fill it using `fakeGPUBuffer`.

    Parameters
    ----------
    **kwargs : dict
        Parameters fed straight into the process config. `pattern` must be
        supplied.
    """
    _buf_ind = 0

    def __init__(self, **kwargs):

        self.name = 'fakegpu_buf%i' % self._buf_ind
        process_name = 'fakegpu%i' % self._buf_ind
        self.__class__._buf_ind += 1

        self.buffer_block = {
            self.name: {
                'kotekan_buffer': 'standard',
                'metadata_pool': 'main_pool',
                'num_frames': 'buffer_depth',
                'sizeof_int': 4,
                'frame_size': ('sizeof_int * num_local_freq * ((num_elements *'
                               ' num_elements) + (num_elements * block_size))')
            }
        }

        process_config = {
            'kotekan_process': 'fakeGpuBuffer',
            'out_buf': self.name,
            'freq': 0,
            'pre_accumulate': True,
            'wait': False
        }
        process_config.update(kwargs)

        self.process_block = {process_name: process_config}


class FakeVisBuffer(InputBuffer):
    """Create an input visBuffer format buffer and fill it using `fakeVis`.

    Parameters
    ----------
    **kwargs : dict
        Parameters fed straight into the process config.
    """
    _buf_ind = 0

    def __init__(self, **kwargs):

        self.name = 'fakevis_buf%i' % self._buf_ind
        process_name = 'fakevis%i' % self._buf_ind
        self.__class__._buf_ind += 1

        self.buffer_block = {
            self.name: {
                'kotekan_buffer': 'vis',
                'metadata_pool': 'vis_pool',
                'num_frames': 'buffer_depth',
            }
        }

        process_config = {
            'kotekan_process': 'fakeVis',
            'out_buf': self.name,
            'freq_ids': [0],
            'wait': False
        }
        process_config.update(kwargs)

        self.process_block = {process_name: process_config}


class ReadVisBuffer(InputBuffer):
    """Write down a visBuffer and reads it with rawFileRead.

    """
    _buf_ind = 0

    def __init__(self, input_dir, buffer_list):

        self.name = 'rawfileread_buf'
        process_name = 'rawfileread%i' % self._buf_ind
        self.__class__._buf_ind += 1

        self.input_dir = input_dir
        self.buffer_list = buffer_list

        self.buffer_block = {
            self.name: {
                'kotekan_buffer': 'vis',
                'metadata_pool': 'vis_pool',
                'num_frames': 'buffer_depth',
            }
        }

        process_config = {
            'kotekan_process': 'rawFileRead',
            'buf': self.name,
            'base_dir': input_dir,
            'file_ext': 'dump',
            'file_name': self.name
        }

        self.process_block = {process_name: process_config}

    def write(self):
        """Write a list of VisBuffer objects to disk.
        """
        visbuffer.VisBuffer.to_files(self.buffer_list,
                                     self.input_dir + '/' + self.name)


class DumpVisBuffer(OutputBuffer):
    """Consume a visBuffer and provide its contents at `VisBuffer` objects.

    Parameters
    ----------
    output_dir : string
        Temporary directory to output to. The dumped files are not removed.
    """

    _buf_ind = 0

    name = None

    def __init__(self, output_dir):

        self.name = 'dumpvis_buf%i' % self._buf_ind
        process_name = 'dump%i' % self._buf_ind
        self.__class__._buf_ind += 1

        self.output_dir = output_dir

        self.buffer_block = {
            self.name: {
                'kotekan_buffer': 'vis',
                'metadata_pool': 'vis_pool',
                'num_frames': 'buffer_depth',
            }
        }

        process_config = {
            'kotekan_process': 'rawFileWrite',
            'in_buf': self.name,
            'file_name': self.name,
            'file_ext': 'dump',
            'base_dir': output_dir
        }

        self.process_block = {process_name: process_config}

    def load(self):
        """Load the output data from the buffer.

        Returns
        -------
        dumps : list of VisBuffer
            The buffer output.
        """
        return visbuffer.VisBuffer.load_files("%s/*%s*.dump" %
                                              (self.output_dir, self.name))


class VisDebugBuffer(OutputBuffer):
    """visDebug.

    Parameters
    ----------
    interrupt_nframes  : int
        Passed directly to visDebug.
    """

    _buf_ind = 0

    name = None

    def __init__(self, interrupt_nframes=-1):

        self.name = 'visdebug_buf%i' % self._buf_ind
        process_name = 'visdebug%i' % self._buf_ind
        self.__class__._buf_ind += 1

        self.interrupt_nframes = interrupt_nframes

        self.buffer_block = {
            self.name: {
                'kotekan_buffer': 'vis',
                'metadata_pool': 'vis_pool',
                'num_frames': 'buffer_depth',
            }
        }

        process_config = {
            'kotekan_process': 'visDebug',
            'in_buf': self.name,
            'interrupt_nframes': self.interrupt_nframes
        }

        self.process_block = {process_name: process_config}


class KotekanProcessTester(KotekanRunner):
    """Construct a test around a single Kotekan process.

    This sets up a Kotekan run to test a specific process by connecting
    `InputBuffer` generators to its inputs and `OutputBuffer` consumers to
    its outputs.

    Parameters
    ----------
    process_type : string
        Type of the process to start (this must be the name registered in
        kotekan).
    process_config : dict
        Any configuration for the process.
    buffers_in : `InputBuffer` or list of
        Input buffers (and generator processes) to connect to the test process.
    buffers_out : `OutputBuffer` or list of
        Output buffers (and consumers processes) to connect.
    global_config : dict
        Any global configuration to run with.
    """

    def __init__(self, process_type, process_config, buffers_in,
                 buffers_out, global_config={}, parallel_process_type=None,
                 parallel_process_config={}):

        config = process_config.copy()
        parallel_config = parallel_process_config.copy()

        if buffers_in is None:
            buffers_in = []
        elif isinstance(buffers_in, (list, tuple)):
            config['in_bufs'] = [buf.name for buf in buffers_in]
            parallel_config['in_bufs'] = [buf.name for buf in buffers_in]
        else:
            config['in_buf'] = buffers_in.name
            parallel_config['in_buf'] = buffers_in.name
            buffers_in = [buffers_in]

        if buffers_out is None:
            buffers_out = []
        elif isinstance(buffers_out, (list, tuple)):
            config['out_bufs'] = [buf.name for buf in buffers_out]
        else:
            config['out_buf'] = buffers_out.name
            buffers_out = [buffers_out]

        config['kotekan_process'] = process_type

        process_block = {(process_type + "_test"): config}
        buffer_block = {}

        for buf in itertools.chain(buffers_in, buffers_out):
            process_block.update(buf.process_block)
            buffer_block.update(buf.buffer_block)

        if parallel_process_type is not None:
            parallel_config['kotekan_process'] = parallel_process_type
            process_block.update(
                {(parallel_process_type + "_test"): parallel_config})

        super(KotekanProcessTester, self).__init__(buffer_block, process_block,
                                                   global_config)


default_config = """
---
type: config
log_level: info
num_elements: 10
num_local_freq: 1
samples_per_data_set: 32768
buffer_depth: 4
num_gpu_frames: 64
block_size: 2
cpu_affinity: []

# Metadata pool
main_pool:
    kotekan_metadata_pool: chimeMetadata
    num_metadata_objects: 30 * buffer_depth

vis_pool:
    kotekan_metadata_pool: visMetadata
    num_metadata_objects: 30 * buffer_depth
"""
