"""Use Python to run a kotekan instance, particularly for testing.
"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility


import os
import itertools
import subprocess
import tempfile
import time
import json

from . import visbuffer


class KotekanRunner(object):
    """A lightweight class for running Kotekan from Python.

    Parameters
    ----------
    buffers : dict
        Dictionary containing all the buffers and their configuration (as
        dicts). Config is as it is in the config files.
    stages : dict
        Dictionary with all the stage definitions.
    config : dict
        Global configuration at the root level.
    rest_commands : list
        REST commands to run packed as `(request_type, endpoint, json_data)`.
    debug: bool
        Shows kotekan stdout and stderr before exit.
    rest_port: int
        Port to use for kotekan REST server. Set it to 0 to get a random free port.
        Default: 0.
    """

    @classmethod
    def kotekan_binary(cls):
        """Determine the kotekan binary to use."""
        build_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "build", "kotekan")
        )
        # If this path exists we are using a non installed version of the
        # kotekan python packages. If so we want to run the local kotekan
        # binary
        relative_path = os.path.join(build_dir, "kotekan")
        if os.path.exists(relative_path):
            return os.path.abspath(relative_path)
        else:
            return shutil.which("kotekan")

    @classmethod
    def kotekan_config(cls):
        """Get kotekan's build config."""
        cmd = "%s --version-json" % cls.kotekan_binary()
        version_string = subprocess.check_output(cmd.split()).decode()

        return json.loads(version_string)

    def __init__(
        self,
        buffers=None,
        stages=None,
        config=None,
        rest_commands=None,
        debug=False,
        expect_failure=False,
        rest_port=0,
    ):

        self._buffers = buffers if buffers is not None else {}
        self._stages = stages if stages is not None else {}
        self._config = config if config is not None else {}
        self._rest_commands = rest_commands if rest_commands is not None else []
        self.debug = debug
        self.expect_failure = expect_failure
        self.rest_port = rest_port
        self.return_code = 0

    def run(self):
        """Run kotekan.

        This configures kotekan by creating a temporary config file.
        """

        import yaml

        rest_header = {"content-type": "application/json"}
        rest_addr = "localhost:%d" % self.rest_port

        config_dict = yaml.safe_load(default_config)
        config_dict.update(self._config)

        # At somepoint maybe do more specialised parsing and validation here
        config_dict.update(self._buffers)
        config_dict.update(self._stages)

        # Set the working directory for the run
        build_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "build", "kotekan")
        )

        config_dict = fix_strings(config_dict)

        with tempfile.NamedTemporaryFile(
            mode="w"
        ) as fh, tempfile.NamedTemporaryFile() as f_out:

            yaml.safe_dump(config_dict, fh)
            print(yaml.safe_dump(config_dict))
            fh.flush()

            cmd = "%s -b %s -c %s" % (self.kotekan_binary(), rest_addr, fh.name)
            print(cmd)
            p = subprocess.Popen(cmd.split(), stdout=f_out, stderr=f_out)

            # Run any requested REST commands
            if self._rest_commands:
                import requests
                import json

                attempt = 0
                wait = 1

                # Wait for REST server to start
                while attempt < 10:

                    if attempt == 9:
                        print("Could not find kotekan REST server address in logs")
                        exit(1)

                    attempt += 1

                    # Wait a moment for rest servers to start up.
                    time.sleep(wait)

                    # If kotekan's REST server was started with a random port (0), we have to find out
                    # what that is from the logs
                    if self.rest_port == 0:
                        log = open(f_out.name, "r").read().split("\n")
                        rest_addr = None
                        for line in log:
                            if (
                                line[:43]
                                == "restServer: started server on address:port "
                            ):
                                rest_addr = line[43:]
                        if rest_addr:
                            print(
                                "Found REST server address in kotekan log: %s"
                                % rest_addr
                            )
                            break
                        else:
                            print(
                                "Could not find kotekan REST server address in logs. Increasing wait time..."
                            )
                            wait += 1

                # the requests module needs the address wrapped in http://*/
                rest_addr = "http://" + rest_addr + "/"

                for rtype, endpoint, data in self._rest_commands:
                    if rtype == "wait":
                        time.sleep(endpoint)
                        continue

                    try:
                        command = getattr(requests, rtype)
                    except AttributeError:
                        raise ValueError("REST command not found")

                    try:
                        command(
                            rest_addr + endpoint,
                            headers=rest_header,
                            data=json.dumps(data),
                        )
                    except:
                        # print kotekan output if sending REST command fails
                        # (kotekan might have crashed and we want to know)
                        p.wait()
                        self.output = open(f_out.name, "r")

                        # Print out the output from Kotekan for debugging
                        print(self.output.read())

                        # Throw an exception if we don't exit cleanly
                        if p.returncode:
                            raise subprocess.CalledProcessError(p.returncode, cmd)

                        print(
                            "Failed sending REST command: "
                            + rtype
                            + " to "
                            + endpoint
                            + " with data "
                            + str(data)
                        )

            while self.debug and None == p.poll():
                time.sleep(10)
                print(file(f_out.name).read())

            # Wait for kotekan to finish and capture the output
            p.wait()
            self.output = open(f_out.name, "r").read()

            # Print out the output from Kotekan for debugging
            print(self.output)

            # If failure is expected just report the exit code
            if self.expect_failure is True:
                print("Test failed as expected with exit code: " + str(p.returncode))
                self.return_code = p.returncode
            elif p.returncode:
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
        Parameters fed straight into the stage config. `type` must be
        supplied, as well as `value` for types other than "random".
    """

    _buf_ind = 0

    def __init__(self, **kwargs):

        self.name = "fakenetwork_buf%i" % self._buf_ind
        if "stage_name" in kwargs:
            stage_name = kwargs["stage_name"]
        else:
            stage_name = "fakenetwork%i" % self._buf_ind
        self.__class__._buf_ind += 1

        self.buffer_block = {
            self.name: {
                "kotekan_buffer": "standard",
                "metadata_pool": "main_pool",
                "num_frames": "buffer_depth",
                "frame_size": (
                    "samples_per_data_set * num_elements"
                    "* num_local_freq * num_data_sets"
                ),
            }
        }

        stage_config = {"kotekan_stage": "testDataGen", "out_buf": self.name}
        stage_config.update(kwargs)

        self.stage_block = {stage_name: stage_config}


class FakeGPUBuffer(InputBuffer):
    """Create an input GPU format buffer and fill it using `fakeGPUBuffer`.

    Parameters
    ----------
    **kwargs : dict
        Parameters fed straight into the stage config. `pattern` must be
        supplied.
    """

    _buf_ind = 0

    def __init__(self, **kwargs):

        self.name = "fakegpu_buf%i" % self._buf_ind
        stage_name = "fakegpu%i" % self._buf_ind
        self.__class__._buf_ind += 1

        self.buffer_block = {
            self.name: {
                "kotekan_buffer": "standard",
                "metadata_pool": "main_pool",
                "num_frames": "buffer_depth",
                "sizeof_int": 4,
                "frame_size": (
                    "sizeof_int * num_freq_in_frame * ((num_elements *"
                    " num_elements) + (num_elements * block_size))"
                ),
            }
        }

        stage_config = {
            "kotekan_stage": "FakeGpu",
            "out_buf": self.name,
            "freq": 0,
            "pre_accumulate": True,
            "wait": False,
        }
        stage_config.update(kwargs)

        self.stage_block = {stage_name: stage_config}


class FakeVisBuffer(InputBuffer):
    """Create an input visBuffer format buffer and fill it using `fakeVis`.

    Parameters
    ----------
    **kwargs : dict
        Parameters fed straight into the stage config.
    """

    _buf_ind = 0

    def __init__(self, **kwargs):

        self.name = "fakevis_buf%i" % self._buf_ind
        stage_name = "fakevis%i" % self._buf_ind
        self.__class__._buf_ind += 1

        self.buffer_block = {
            self.name: {
                "kotekan_buffer": "vis",
                "metadata_pool": "vis_pool",
                "num_frames": "buffer_depth",
            }
        }

        stage_config = {
            "kotekan_stage": "fakeVis",
            "out_buf": self.name,
            "freq_ids": [0],
            "wait": False,
        }
        stage_config.update(kwargs)

        self.stage_block = {stage_name: stage_config}


class VisWriterBuffer(OutputBuffer):
    """Consume a visBuffer and provide its contents as raw or hdf5 file.

    Parameters
    ----------
    output_dir : string
        Temporary directory to output to. The dumped files are not removed.
    file_type : string
        File type to write into (see visWriter documentation)
    in_buf : string
        Optionally specify the name of an input buffer instead of creating one.
    """

    _buf_ind = 0

    name = None

    def __init__(self, output_dir, file_type, in_buf=None, extra_config=None):

        self.name = "viswriter_buf%i" % self._buf_ind
        stage_name = "write%i" % self._buf_ind
        self.__class__._buf_ind += 1

        self.output_dir = output_dir

        if in_buf is None:
            self.buffer_block = {
                self.name: {
                    "kotekan_buffer": "vis",
                    "metadata_pool": "vis_pool",
                    "num_frames": "buffer_depth",
                }
            }
            buf_name = self.name
        else:
            buf_name = in_buf
            self.buffer_block = {}

        stage_config = {
            "kotekan_stage": "visWriter",
            "in_buf": buf_name,
            "file_name": self.name,
            "file_type": file_type,
            "root_path": output_dir,
            "node_mode": False,
        }
        if extra_config is not None:
            stage_config.update(extra_config)

        self.stage_block = {stage_name: stage_config}

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
        flnm = glob.glob(self.output_dir + "/*/*.data")[0]
        return visbuffer.VisRaw(os.path.splitext(flnm)[0])


class ReadVisBuffer(InputBuffer):
    """Write down a visBuffer and reads it with rawFileRead.

    """

    _buf_ind = 0

    def __init__(self, input_dir, buffer_list):

        self.name = "rawfileread_buf"
        stage_name = "rawfileread%i" % self._buf_ind
        self.__class__._buf_ind += 1

        self.input_dir = input_dir
        self.buffer_list = buffer_list

        self.buffer_block = {
            self.name: {
                "kotekan_buffer": "vis",
                "metadata_pool": "vis_pool",
                "num_frames": "buffer_depth",
            }
        }

        stage_config = {
            "kotekan_stage": "rawFileRead",
            "buf": self.name,
            "base_dir": input_dir,
            "file_ext": "dump",
            "file_name": self.name,
            "end_interrupt": True,
        }

        self.stage_block = {stage_name: stage_config}

    def write(self):
        """Write a list of VisBuffer objects to disk.
        """
        visbuffer.VisBuffer.to_files(self.buffer_list, self.input_dir + "/" + self.name)


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

        self.name = "dumpvis_buf%i" % self._buf_ind
        stage_name = "dump%i" % self._buf_ind
        self.__class__._buf_ind += 1

        self.output_dir = output_dir

        self.buffer_block = {
            self.name: {
                "kotekan_buffer": "vis",
                "metadata_pool": "vis_pool",
                "num_frames": "buffer_depth",
            }
        }

        stage_config = {
            "kotekan_stage": "rawFileWrite",
            "in_buf": self.name,
            "file_name": self.name,
            "file_ext": "dump",
            "base_dir": output_dir,
        }

        self.stage_block = {stage_name: stage_config}

    def load(self):
        """Load the output data from the buffer.

        Returns
        -------
        dumps : list of VisBuffer
            The buffer output.
        """
        return visbuffer.VisBuffer.load_files(
            "%s/*%s*.dump" % (self.output_dir, self.name)
        )


class ReadRawBuffer(InputBuffer):

    _buf_ind = 0

    def __init__(self, infile, chunk_size):

        self.name = "read_raw_buf{:d}".format(self._buf_ind)
        stage_name = "read_raw{:d}".format(self._buf_ind)
        self.__class__._buf_ind += 1

        self.buffer_block = {
            self.name: {
                "kotekan_buffer": "vis",
                "metadata_pool": "vis_pool",
                "num_frames": "buffer_depth",
            }
        }

        stage_config = {
            "kotekan_stage": "visRawReader",
            "infile": infile,
            "out_buf": self.name,
            "chunk_size": chunk_size,
            "readahead_blocks": 4,
        }

        self.stage_block = {stage_name: stage_config}


class KotekanStageTester(KotekanRunner):
    """Construct a test around a single Kotekan stage.

    This sets up a Kotekan run to test a specific stage by connecting
    `InputBuffer` generators to its inputs and `OutputBuffer` consumers to
    its outputs.

    Parameters
    ----------
    stage_type : string
        Type of the stage to start (this must be the name registered in
        kotekan).
    stage_config : dict
        Any configuration for the stage.
    buffers_in : `InputBuffer` or list of
        Input buffers (and generator stages) to connect to the test stage.
    buffers_out : `OutputBuffer` or list of
        Output buffers (and consumers stages) to connect.
    global_config : dict
        Any global configuration to run with.
    parallel_stage_type : str
        Name of the stage to be run in parallel with the stage under test (It will use the same in buffers).
    parallel_stage_config : dict
        any configurations to the parallel stage
    noise : string
        If it is not None, gaussian noise with SD=1 is added to the input,
        if it is "random" the random number generator will be initialized with
        a random seed.
    """

    def __init__(
        self,
        stage_type,
        stage_config,
        buffers_in,
        buffers_out,
        global_config={},
        parallel_stage_type=None,
        parallel_stage_config={},
        rest_commands=None,
        noise=False,
        expect_failure=False,
    ):

        config = stage_config.copy()
        parallel_config = parallel_stage_config.copy()
        noise_config = {}

        if noise:
            if buffers_in is None:
                buffers_in = []
            else:
                noise_config["in_buf"] = buffers_in.name
                buffers_in = [buffers_in]
            noise_config["kotekan_stage"] = "visNoise"
            noise_config["out_buf"] = "noise_buf"
            if noise == "random":
                noise_config["random"] = True
            noise_block = {("visNoise_test"): noise_config}
            config["in_buf"] = "noise_buf"
            parallel_config["in_buf"] = "noise_buf"
            noise_buffer = {
                "noise_buf": {
                    "kotekan_buffer": "vis",
                    "metadata_pool": "vis_pool",
                    "num_frames": "buffer_depth",
                }
            }
        else:
            if buffers_in is None:
                buffers_in = []
            elif isinstance(buffers_in, (list, tuple)):
                config["in_bufs"] = [buf.name for buf in buffers_in]
                parallel_config["in_bufs"] = [buf.name for buf in buffers_in]
            else:
                config["in_buf"] = buffers_in.name
                parallel_config["in_buf"] = buffers_in.name
                buffers_in = [buffers_in]

        if buffers_out is None:
            buffers_out = []
        elif isinstance(buffers_out, (list, tuple)):
            config["out_bufs"] = [buf.name for buf in buffers_out]
        else:
            config["out_buf"] = buffers_out.name
            buffers_out = [buffers_out]

        config["kotekan_stage"] = stage_type

        stage_block = {(stage_type + "_test"): config}
        buffer_block = {}

        for buf in itertools.chain(buffers_in, buffers_out):
            stage_block.update(buf.stage_block)
            buffer_block.update(buf.buffer_block)

        if parallel_stage_type is not None:
            parallel_config["kotekan_stage"] = parallel_stage_type
            stage_block.update(
                {(parallel_stage_type + "_test_parallel"): parallel_config}
            )

        if noise:
            stage_block.update(noise_block)
            buffer_block.update(noise_buffer)

        super(KotekanStageTester, self).__init__(
            buffer_block,
            stage_block,
            global_config,
            rest_commands,
            expect_failure=expect_failure,
        )


default_config = """
---
type: config
log_level: debug
num_elements: 10
num_freq_in_frame: 1
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
    "int_frames": 64,
"""


def fix_strings(d):

    import future.utils

    if isinstance(d, list):
        return [fix_strings(x) for x in d]

    if isinstance(d, dict):
        return {fix_strings(k): fix_strings(v) for k, v in d.items()}

    if isinstance(d, str):
        return future.utils.native(d)

    if isinstance(d, bytes):
        return future.utils.native(d.decode())

    return d


## Quick tests for kotekan's builds
def has_hdf5():
    """Is HDF5 support built in."""
    return KotekanRunner.kotekan_config()["cmake_build_settings"]["USE_HDF5"] == "ON"


def has_lapack():
    """Is LAPACK support built in."""
    return KotekanRunner.kotekan_config()["cmake_build_settings"]["USE_LAPACK"] == "ON"


def has_openmp():
    """Is OpenMP support build in."""
    return KotekanRunner.kotekan_config()["cmake_build_settings"]["USE_OMP"] == "ON"
