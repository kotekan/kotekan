import os
import itertools
import subprocess
import tempfile
import time

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
    rest_commands : list
        REST commands to run packed as `(request_type, endpoint, json_data)`.
    """

    def __init__(self, buffers=None, processes=None, config=None,
                 rest_commands=None):

        self._buffers = buffers if buffers is not None else {}
        self._processes = processes if processes is not None else {}
        self._config = config if config is not None else {}
        self._rest_commands = (rest_commands if rest_commands is not None
                               else [])

    def run(self):
        """Run kotekan.

        This configures kotekan by creating a temporary config file.
        """

        import yaml

        rest_header = {"content-type": "application/json"}
        rest_addr = "http://localhost:12048/"

        config_dict = yaml.load(default_config)
        config_dict.update(self._config)

        # At somepoint maybe do more specialised parsing and validation here
        config_dict.update(self._buffers)
        config_dict.update(self._processes)

        kotekan_dir = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                                    "..", "build", "kotekan"))

        with tempfile.NamedTemporaryFile() as fh, \
             tempfile.NamedTemporaryFile() as f_out:

            yaml.dump(config_dict, fh)
            fh.flush()

            cmd = ["./kotekan", "-c", fh.name]
            p = subprocess.Popen(cmd, cwd=kotekan_dir,
                                 stdout=f_out, stderr=f_out)

            # Run any requested REST commands
            if self._rest_commands:
                import requests
                import json
                # Wait a moment for rest servers to start up.
                time.sleep(1)
                for rtype, endpoint, data in self._rest_commands:
                    if rtype == 'wait':
                        time.sleep(endpoint)
                        continue

                    try:
                        command = getattr(requests, rtype)
                    except AttributeError:
                        raise ValueError('REST command not found')

                    try:
                        command(rest_addr + endpoint,
                                headers=rest_header,
                                data=json.dumps(data))
                    except:
                        # print kotekan output if sending REST command fails
                        # (kotekan might have crashed and we want to know)
                        p.wait()
                        self.output = file(f_out.name).read()

                        # Print out the output from Kotekan for debugging
                        print self.output

                        # Throw an exception if we don't exit cleanly
                        if p.returncode:
                            raise subprocess.CalledProcessError(p.returncode, cmd)

                        print "Failed sending REST command: " + rtype + " to " + endpoint + " with data " + data


            # Wait for kotekan to finish and capture the output
            p.wait()
            self.output = file(f_out.name).read()

            # Print out the output from Kotekan for debugging
            print self.output

            # Throw an exception if we don't exit cleanly
            if p.returncode:
                raise subprocess.CalledProcessError(p.returncode, cmd)


class InputBuffer(object):
    """Base class for an input buffer generator."""
    name = None


class OutputBuffer(object):
    """Base class for an output buffer consumer."""
    name = None


class FakeNetworkBuffer(InputBuffer):
    """Create an input network format buffer and fill it using `testDataGen`.

    Parameters
    ----------
    **kwargs : dict
        Parameters fed straight into the process config. `type` must be
        supplied, as well as `value` for types other than "random".
    """
    _buf_ind = 0

    def __init__(self, **kwargs):

        self.name = 'fakenetwork_buf%i' % self._buf_ind
        if "process_name" in kwargs:
            process_name = kwargs['process_name']
        else:
            process_name = 'fakenetwork%i' % self._buf_ind
        self.__class__._buf_ind += 1

        self.buffer_block = {
            self.name: {
                'kotekan_buffer': 'standard',
                'metadata_pool': 'main_pool',
                'num_frames': 'buffer_depth',
                'frame_size': ('samples_per_data_set * num_elements'
                               '* num_local_freq * num_data_sets'),
            }
        }

        process_config = {
            'kotekan_process': 'testDataGen',
            'network_out_buf': self.name,
        }
        process_config.update(kwargs)

        self.process_block = {process_name: process_config}


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


class VisWriterBuffer(OutputBuffer):
    """Consume a visBuffer and provide its contents as raw or hdf5 file.

    Parameters
    ----------
    output_dir : string
        Temporary directory to output to. The dumped files are not removed.
    file_type : string
        File type to write into (see visWriter documentation)
    freq_ids : Array of Int.
        Frequency IDs
    in_buf : string
        Optionally specify the name of an input buffer instead of creating one.
    """

    _buf_ind = 0

    name = None

    def __init__(self, output_dir, file_type, freq_ids, in_buf=None):

        self.name = 'viswriter_buf%i' % self._buf_ind
        process_name = 'write%i' % self._buf_ind
        self.__class__._buf_ind += 1

        self.output_dir = output_dir

        if in_buf is None:
            self.buffer_block = {
                self.name: {
                    'kotekan_buffer': 'vis',
                    'metadata_pool': 'vis_pool',
                    'num_frames': 'buffer_depth',
                }
            }
            buf_name = self.name
        else:
            buf_name = in_buf
            self.buffer_block = {}

        process_config = {
            'kotekan_process': 'visWriter',
            'in_buf': buf_name,
            'file_name': self.name,
            'file_type': file_type,
            'root_path': output_dir,
            'write_ev': True,
            'node_mode': False,
            'freq_ids': freq_ids
        }

        self.process_block = {process_name: process_config}

    def load(self):
        """Load the output data from the buffer.

        Returns
        -------
        dumps : visbuffer.VisRaw object.
            The buffer output.
        """
        import glob

        # For now assume only one file is found
        # TODO: Might be nice to be able to check the file is the right one.
        # But visWriter creates the acquisition and file names on the flight
        flnm = glob.glob(self.output_dir+'/*/*.data')[0]
        return visbuffer.VisRaw(os.path.splitext(flnm)[0]).data

#        return visbuffer.VisBuffer.load_files("%s/*%s*.dump" %
#                                              (self.output_dir, self.name))


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
            'file_name': self.name,
            'end_interrupt': True
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


class ReadRawBuffer(InputBuffer):

    _buf_ind = 0

    def __init__(self, infile, chunk_size):

        self.name = "read_raw_buf{:d}".format(self._buf_ind)
        process_name = "read_raw{:d}".format(self._buf_ind)
        self.__class__._buf_ind += 1

        self.buffer_block = {
            self.name: {
                'kotekan_buffer': 'vis',
                'metadata_pool': 'vis_pool',
                'num_frames': 'buffer_depth',
            }
        }

        process_config = {
            'kotekan_process': 'visRawReader',
            'infile': infile,
            'out_buf': self.name,
            'chunk_size': chunk_size,
            'readahead_blocks': 4
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
    parallel_process_type : str
        Name of the process to be run in parallel with the process under test (It will use the same in buffers).
    parallel_process_config : dict
        any configurations to the parallel process
    """

    def __init__(self, process_type, process_config, buffers_in,
                 buffers_out, global_config={}, parallel_process_type=None,
                 parallel_process_config={}, rest_commands=None):

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
                {(parallel_process_type + "_test_parallel"): parallel_config})

        super(KotekanProcessTester, self).__init__(buffer_block, process_block,
                                                   global_config, rest_commands)


default_config = """
---
type: config
log_level: info
num_elements: 10
num_local_freq: 1
num_data_sets: 1
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
