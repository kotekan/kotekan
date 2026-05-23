"""Use Python to run a kotekan instance, particularly for testing.
"""

# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility


import itertools
import json
import os
import sys
import shutil
import subprocess
import tempfile
import time
import warnings
import re
import numpy as np

from . import baseband_buffer
from . import visbuffer
from . import n2buffer
from . import chordbuffer
from . import frbbuffer
from . import psrbuffer


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
    gdb: bool
        Run in gdb and in a failure case produce a backtrace. Doesn't work if rest commands are supplied.
    """

    @classmethod
    def kotekan_binary(cls):
        """Determine the kotekan binary to use."""
        # If we are building using github actions, the binary may be located
        # in a specific build directory.
        if os.environ.get("KOTEKAN_BUILD_DIRNAME") is not None:
            build_dirname = os.environ.get("KOTEKAN_BUILD_DIRNAME")
        else:
            build_dirname = "build"

        build_dir = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__), "..", "..", build_dirname, "kotekan"
            )
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
        gdb=False,
        timeout=None,
    ):

        self._buffers = buffers if buffers is not None else {}
        self._stages = stages if stages is not None else {}
        self._config = config if config is not None else {}
        self._rest_commands = rest_commands if rest_commands is not None else []
        self.debug = debug
        self.expect_failure = expect_failure
        self.rest_port = rest_port
        self._gdb = gdb
        self.return_code = 0
        # Max seconds to wait for kotekan to exit on its own before we kill it.
        # Defaults to a little under pytest's per-test timeout (PYTEST_TIMEOUT, if
        # exported) so the runner fails with a clear message and cleans up the
        # process itself, rather than relying on pytest-timeout (which is
        # unreliable under xdist). Falls back to 120s outside CI.
        if timeout is None:
            _pytest_to = os.environ.get("PYTEST_TIMEOUT")
            timeout = 0.8 * float(_pytest_to) if _pytest_to else 120.0
        self._timeout = float(timeout)
        # Handle to the launched kotekan process, so run() can always reap it.
        self._process = None

    @staticmethod
    def _cleanup(p):
        """Ensure a launched kotekan process is stopped and reaped.

        Guards against orphaned kotekan processes (and the indefinite hangs and
        resource leaks they cause, especially under pytest-xdist) on every exit
        path from ``run()``.
        """
        # The gdb path uses subprocess.run(), which returns a CompletedProcess
        # with no poll()/terminate(); there is nothing to clean up there.
        if p is None or not hasattr(p, "poll"):
            return
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass

    def run(self):
        """Run kotekan, always terminating the kotekan process afterwards."""
        try:
            self._run()
        finally:
            self._cleanup(self._process)

    def _run(self):
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

        config_dict = fix_strings(config_dict)

        with tempfile.NamedTemporaryFile(
            mode="w"  # , delete=False
        ) as fh, tempfile.NamedTemporaryFile() as f_out:

            yaml.safe_dump(config_dict, fh)
            print(yaml.safe_dump(config_dict))
            fh.flush()

            if self._gdb and not self._rest_commands:
                if self._rest_commands:
                    warnings.warn(
                        "Sending REST commands is not supported when gdb=True."
                    )

                cmd = 'gdb %s -batch -ex "run -b %s -c %s" -ex "bt"' % (
                    self.kotekan_binary(),
                    rest_addr,
                    fh.name,
                )
                print(cmd)
                p = subprocess.run(cmd, stdout=f_out, stderr=f_out, shell=True)
            else:
                print("Config file %s" % fh.name, file=sys.stderr)
                cmd = "%s -b %s -c %s" % (self.kotekan_binary(), rest_addr, fh.name)
                print(cmd)
                p = subprocess.Popen(cmd.split(), stdout=f_out, stderr=f_out)
                self._process = p

            # Run any requested REST commands
            if self._rest_commands and not self._gdb:
                import requests

                attempt = 0
                wait = 0.2

                # Wait for REST server to start
                while attempt < 100:

                    if attempt == 99:
                        self._cleanup(p)
                        self.output = open(f_out.name, "r").read()
                        print(self.output)
                        raise RuntimeError(
                            "Could not find kotekan REST server address in logs. "
                            "Command: %s" % cmd
                        )

                    attempt += 1

                    # If kotekan already exited, its REST server will never appear.
                    if p.poll() is not None:
                        self.output = open(f_out.name, "r").read()
                        print(self.output)
                        raise RuntimeError(
                            "kotekan exited (code %s) before its REST server "
                            "started. Command: %s" % (p.returncode, cmd)
                        )

                    # Wait a moment for rest servers to start up (capped so a
                    # missing server can't make this loop run for many minutes).
                    time.sleep(min(wait, 2.0))

                    # If kotekan's REST server was started with a random port (0), we have to find out
                    # what that is from the logs
                    if self.rest_port == 0:
                        log = open(f_out.name, "r").read().split("\n")
                        rest_addr = None
                        for line in log:
                            match = re.search(
                                "^.*restServer: started server on address:port (.*)$",
                                line,
                            )
                            if match is not None:
                                rest_addr = match.group(1)
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

                # wait a moment for the restServer to start
                time.sleep(1)

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
                    except requests.RequestException:
                        # print kotekan output if sending REST command fails
                        # (kotekan might have crashed and we want to know)
                        try:
                            p.wait(timeout=self._timeout)
                        except subprocess.TimeoutExpired:
                            self._cleanup(p)
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
                print(open(f_out.name, "r").read())

            # Wait for kotekan to finish and capture the output. Bound the wait so
            # a kotekan that never exits fails the test quickly (and, under xdist,
            # does not stall the whole run) instead of blocking here indefinitely.
            if not self._gdb:
                try:
                    p.wait(timeout=self._timeout)
                except subprocess.TimeoutExpired:
                    self._cleanup(p)
                    self.output = open(f_out.name, "r").read()
                    print(self.output)
                    raise RuntimeError(
                        "kotekan did not exit within %g seconds; terminated it. "
                        "Command: %s" % (self._timeout, cmd)
                    )
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


class FakeLostSamplesBuffer(InputBuffer):
    """Provide an input buffer for the `lost_samples_buf` config parameter."""

    def __init__(self, **kwargs):
        self.name = "lost_samples_buffer"
        self.buffer_block = {
            self.name: {
                "kotekan_buffer": "standard",
                "sizeof_int": 4,
                "frame_size": "sizeof_int * num_freq_in_frame * ((num_elements * num_elements) + (num_elements * block_size))",
                "metadata_pool": "main_pool",
                "num_frames": "buffer_depth",
            }
        }
        stage_name = kwargs.get("stage_name", "fake_lost_samples")
        stage_config = {
            "kotekan_stage": "FakeGpu",
            "out_buf": self.name,
            "pattern": "lostsamples",
            "wait": "false",
            "freq": 0,
            "pre_accumulate": "true",
        }
        stage_config.update(kwargs)

        self.stage_block = {stage_name: stage_config}
        self.global_block = {"telescope": {"name": "fake"}}


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
    """Create an input GPU format buffer and fill it using `FakeGPU`.

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
        self.global_block = {"telescope": {"name": "fake"}}


class FakeN2KBuffers(InputBuffer):
    """Create N2K-output format buffers and fill using `testN2kGen` and `testDataGen`.

    Parameters
    ----------
    **kwargs : dict
        Parameters fed straight into the stage config. `pattern` must be
        supplied.
    """

    _buf_ind = 0

    def __init__(
        self,
        samples_per_data_set,
        num_local_freq,
        sub_integration_ntime,
        n2k_kwargs,
        rficounts_kwargs,
        plcounts_kwargs,
        rfiframemask_kwargs,
    ):

        self.name = "faken2k_corr_buf%i" % self._buf_ind
        self.counts_name = "faken2k_counts_buf%i" % self._buf_ind
        self.rficounts_name = "faken2k_rficounts_buf%i" % self._buf_ind
        self.plcounts_name = "faken2k_plcounts_buf%i" % self._buf_ind
        self.rfiframemask_name = "faken2k_rfiframemask_buf%i" % self._buf_ind

        n2k_stage_name = "faken2k_gen_%i" % self._buf_ind
        rfi_stage_name = "faken2k_gen_rfi_%i" % self._buf_ind
        rficount_stage_name = "faken2k_rficount_%i" % self._buf_ind
        plcounts_stage_name = "faken2k_plcounts_%i" % self._buf_ind
        rfiframemask_stage_name = "faken2k_gen_rfiframemask_%i" % self._buf_ind

        self.__class__._buf_ind += 1

        self.buffer_block = {
            self.name: {
                "kotekan_buffer": "standard",
                "metadata_pool": "chord_pool",
                "num_frames": "buffer_depth",
                "sizeof_float": 4,
                "vis_blocksize": 16,
                "vis_num_blocks_lin": "num_elements / vis_blocksize",
                "vis_num_blocks": "(vis_num_blocks_lin * (vis_num_blocks_lin + 1)) / 2",
                "frame_size": (
                    "(samples_per_data_set / sub_integration_ntime) * num_local_freq"
                    " * vis_num_blocks * vis_blocksize * vis_blocksize * 2 * sizeof_float"
                ),
            },
            self.counts_name: {
                "kotekan_buffer": "standard",
                "metadata_pool": "chord_pool",
                "num_frames": "buffer_depth",
                "sizeof_int": 4,
                "count_blocksize": 8,
                "count_num_blocks_lin": "(num_elements / 8) / count_blocksize",
                "count_num_blocks": "(count_num_blocks_lin * (count_num_blocks_lin + 1)) / 2",
                "frame_size": (
                    "(samples_per_data_set / sub_integration_ntime) * num_local_freq"
                    " * count_num_blocks * count_blocksize * count_blocksize * sizeof_int"
                ),
            },
            self.rficounts_name: {
                "kotekan_buffer": "standard",
                "metadata_pool": "chord_pool",
                "num_frames": "buffer_depth",
                "sizeof_int": 4,
                "frame_size": "(samples_per_data_set / sub_integration_ntime) * num_local_freq * sizeof_int",
            },
            self.plcounts_name: {
                "kotekan_buffer": "standard",
                "metadata_pool": "chord_pool",
                "num_frames": "buffer_depth",
                "sizeof_int": 4,
                "frame_size": "(samples_per_data_set / sub_integration_ntime) * num_local_freq * sizeof_int",
            },
            self.rfiframemask_name: {
                "kotekan_buffer": "standard",
                "metadata_pool": "chord_pool",
                "num_frames": "buffer_depth",
                "frame_size": "(samples_per_data_set / sub_integration_ntime) * num_local_freq",
            },
        }

        n2k_gen_config = {
            "kotekan_stage": "testN2kGen",
            "out_buf": self.name,
            "out_counts_buf": self.counts_name,
            "correlation_type": "const",
            "correlation_value": [1, -2],
            "counts_type": "const",
            "counts_value": "sub_integration_ntime",
            "mul_correlation_by_counts": True,
        }
        rficount_gen_config = {
            "kotekan_stage": "testLostCountsGen",
            "in_buf": self.counts_name,
            "out_buf": self.rficounts_name,
            "name": "RFImask_counts",
            "type": "const",
            "value": 0,
            "use_n2k_counts": True,
        }
        plcounts_gen_config = {
            "kotekan_stage": "testLostCountsGen",
            "in_buf": self.counts_name,
            "out_buf": self.plcounts_name,
            "name": "pl_lost_counts_scalar",
            "type": "const",
            "value": 0,
            "use_n2k_counts": True,
        }
        rfiframemask_gen_config = {
            "kotekan_stage": "testRFIFrameMaskGen",
            "out_buf": self.rfiframemask_name,
            "type": "const",
            "value": 1,
        }

        n2k_gen_config.update(n2k_kwargs)
        rficount_gen_config.update(rficounts_kwargs)
        plcounts_gen_config.update(plcounts_kwargs)
        rfiframemask_gen_config.update(rfiframemask_kwargs)

        self.stage_block = {
            n2k_stage_name: n2k_gen_config,
            rficount_stage_name: rficount_gen_config,
            plcounts_stage_name: plcounts_gen_config,
            rfiframemask_stage_name: rfiframemask_gen_config,
        }
        self.global_block = {
            "chord_pool": {
                "kotekan_metadata_pool": "chordMetadata",
                "num_metadata_objects": "6 * buffer_depth",
            }
        }


class FakeVisBuffer(InputBuffer):
    """Create an input visBuffer format buffer and fill it using `FakeVis`.

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
            "kotekan_stage": "FakeVis",
            "out_buf": self.name,
            "freq_ids": [0],
            "wait": False,
        }
        stage_config.update(kwargs)

        self.stage_block = {stage_name: stage_config}


class FakeN2Buffer(InputBuffer):
    """Create an input N2Buffer format buffer and fill it using `FakeN2`.

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
                "kotekan_buffer": "N2",
                "n2_layout": "FullUpperTri",
                "metadata_pool": "N2_pool",
                "num_frames": "buffer_depth",
            }
        }

        stage_config = {
            "kotekan_stage": "FakeN2",
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
        File type to write into (see VisWriter documentation)
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
            "kotekan_stage": "VisWriter",
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
        # But VisWriter creates the acquisition and file names on the flight
        flnm = glob.glob(self.output_dir + "/*/*.data")[0]
        return visbuffer.VisRaw.from_file(os.path.splitext(flnm)[0])


class ReadVisBuffer(InputBuffer):
    """Write down a visBuffer and reads it with rawFileRead."""

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
        """Write a list of VisBuffer objects to disk."""
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


class ReadChordBuffer(InputBuffer):
    """Write down a ChordBuffer and reads it with hdf5FileRead."""

    _buf_ind = 0

    def __init__(self, input_dir, buffer_list):

        self.name = "hdf5fileread_buf%i" % self._buf_ind
        stage_name = "hdf5fileread%i" % self._buf_ind
        self.__class__._buf_ind += 1

        self.input_dir = input_dir
        self.buffer_list = buffer_list

        self.buffer_block = {
            self.name: {
                "kotekan_buffer": "standard",
                "metadata_pool": "main_pool",
                "num_frames": "buffer_depth",
                "frame_size": buffer_list[0].data.nbytes,
            }
        }

        stage_config = {
            "kotekan_stage": "hdf5FileRead",
            "out_buf": self.name,
            "input_dir": input_dir,
            "file_name": self.name,
            "prefix_hostname": False,
            "prefix_host_rank": False,
        }

        self.stage_block = {stage_name: stage_config}

    def write(self):
        """Write a list of ChordBuffer objects to disk."""
        chordbuffer.ChordBuffer.to_files(
            self.buffer_list, self.input_dir + "/" + self.name
        )


class DumpChordBuffer(OutputBuffer):
    """Consume a Chord buffer and provide its content as `ChordBuffer` objects.

    Parameters
    ----------
    output_dir : str
        Temp. directory to output to. Dumped files are not removed.
    """

    _buf_ind = 0

    name = None

    def __init__(self, output_dir, shape, dtype, max_frames=-1):
        self.name = f"dumpchord_buf{self._buf_ind}"
        stage_name = f"dump{self._buf_ind}"

        self.__class__._buf_ind += 1

        self.output_dir = output_dir
        self.num_entries = int(np.prod(shape))
        self.max_frames = max_frames
        self.dtype = dtype

        self.buffer_block = {
            self.name: {
                "kotekan_buffer": "standard",
                "metadata_pool": "main_pool",
                "num_frames": "buffer_depth",
                "frame_size": self.num_entries * np.dtype(self.dtype).itemsize,
            }
        }

        self.stage_block = {
            stage_name: {
                "kotekan_stage": "hdf5FileWrite",
                "in_buf": self.name,
                "file_name": self.name,
                "base_dir": output_dir,
                "prefix_hostname": False,
                "max_frames": max_frames,
            }
        }

    def load(self):
        """Load the output data from the buffer.

        Returns
        -------
        dumps : lost of buffers
            Buffer output.
        """

        return chordbuffer.ChordBuffer.load_files(
            f"{self.output_dir}/*{self.name}*.h5", self.name
        )


class ReadN2Buffer(InputBuffer):
    """Write down an N2Buffer and reads it with rawFileRead."""

    _buf_ind = 0

    def __init__(self, input_dir, buffer_list):

        self.name = "rawfileread_buf"
        stage_name = "rawfileread%i" % self._buf_ind
        self.__class__._buf_ind += 1

        self.input_dir = input_dir
        self.buffer_list = buffer_list

        self.buffer_block = {
            self.name: {
                "kotekan_buffer": "N2",
                "n2_layout": "FullUpperTri",
                "metadata_pool": "N2_pool",
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
        """Write a list of VisBuffer objects to disk."""
        n2buffer.N2Buffer.to_files(self.buffer_list, self.input_dir + "/" + self.name)


class DumpN2Buffer(OutputBuffer):
    """Consume an N2Buffer and provide its contents as `N2Buffer` objects.
       Assumes FullUpperTri layout. (TODO: Maybe generalize later.)

    Parameters
    ----------
    output_dir : string
        Temporary directory to output to. The dumped files are not removed.
    """

    _buf_ind = 0

    name = None

    def __init__(
        self,
        output_dir,
        exit_after_n_files=0,
        num_elements=None,
        num_ev=None,
        num_freq=None,
    ):

        self.name = "dumpn2_buf%i" % self._buf_ind
        stage_name = "dump%i" % self._buf_ind
        self.__class__._buf_ind += 1

        self.output_dir = output_dir
        self.num_elements = num_elements
        if num_elements is not None:
            # Assume FullUpperTri layout
            self.num_prod = (num_elements * (num_elements + 1)) // 2
        else:
            self.num_prod = None
        self.num_ev = num_ev

        # If num_freq is given, size the buffer to hold one full accumulation
        # cycle (one frame per frequency) so the producer never stalls.

        num_freq_in_buffer = num_freq if num_freq is not None else 1

        self.buffer_block = {
            self.name: {
                "kotekan_buffer": "N2",
                "n2_layout": "FullUpperTri",
                "metadata_pool": "N2_pool",
                "num_frames": "buffer_depth * {:d}".format(num_freq_in_buffer),
            }
        }

        stage_config = {
            "kotekan_stage": "rawFileWrite",
            "in_buf": self.name,
            "file_name": self.name,
            "file_ext": "dump",
            "base_dir": output_dir,
            "exit_after_n_files": exit_after_n_files,
        }

        self.stage_block = {stage_name: stage_config}

    def load(self):
        """Load the output data from the buffer.

        Returns
        -------
        dumps : list of VisBuffer
            The buffer output.
        """
        return n2buffer.N2Buffer.load_files(
            "%s/*%s*.dump" % (self.output_dir, self.name),
            num_elements=self.num_elements,
            num_prod=self.num_prod,
            num_ev=self.num_ev,
        )


class FakeFrbBeamformBuffer(InputBuffer):
    """Create an input FRB beamform-format buffer using a fake testDataGen network data

    Parameters
    ----------
    cpu : bool (default: True)
        Use CPU beamformer (note: very slow for large inputs) instead of GPU kernels

    **kwargs : dict
        Parameters fed straight into the stage config.
    """

    _buf_ind = 0

    def __init__(self, cpu=True, **kwargs):

        self.name = "fakefrbbeamform_buf%i" % self._buf_ind
        gpu_input_name = "fakefrbbeamform_in_buf%i" % self._buf_ind
        gpu_input_stage_name = "fakefrbbeamform_in%i" % self._buf_ind
        stage_name = "fakefrbbeamform%i" % self._buf_ind
        read_gain_stage_name = "fakereadgain%i" % self._buf_ind
        gain_frb_buffer = "fakegain_frb_buf%i" % self._buf_ind
        gain_psr_buffer = "fakegain_psr_buf%i" % self._buf_ind
        num_psr_beams = 10
        gpu_id = self._buf_ind
        self.__class__._buf_ind += 1

        self.buffer_block = {
            gpu_input_name: {
                "num_frames": "buffer_depth",
                "frame_size": "samples_per_data_set * num_elements * num_local_freq * num_data_sets",
                "metadata_pool": "main_pool",
                "kotekan_buffer": "standard",
            },
            self.name: {
                "kotekan_buffer": "standard",
                "metadata_pool": "main_pool",
                "sizeof_float": 4,
                "frame_size": (
                    "num_data_sets * (samples_per_data_set/downsample_time/downsample_freq) * num_frb_total_beams * sizeof_float"
                ),
                "num_frames": "buffer_depth",
            },
        }

        cpu_stage_config = {
            "kotekan_stage": "gpuBeamformSimulate",
            "network_in_buf": gpu_input_name,
            "beam_out_buf": self.name,
            "northmost_beam": 90.0,
            "gain_dir": "../../gains/",
            "frb_missing_gains": [1.0, 1.0],
            "ew_spacing": [0, 0, 0, 0],
            # fmt: off
            "reorder_map": [
                32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59,
                96, 97, 98, 99, 104, 105, 106, 107, 112, 113, 114, 115, 120, 121, 122, 123,
                67, 66, 65, 64, 75, 74, 73, 72, 83, 82, 81, 80, 91, 90, 89, 88,
                3, 2, 1, 0, 11, 10, 9, 8, 19, 18, 17, 16, 27, 26, 25, 24,
                152, 153, 154, 155, 144, 145, 146, 147, 136, 137, 138, 139, 128, 129, 130, 131,
                216, 217, 218, 219, 208, 209, 210, 211, 200, 201, 202, 203, 192, 193, 194, 195,
                251, 250, 249, 248, 243, 242, 241, 240, 235, 234, 233, 232, 227, 226, 225, 224,
                187, 186, 185, 184, 179, 178, 177, 176, 171, 170, 169, 168, 163, 162, 161, 160,
                355, 354, 353, 352, 363, 362, 361, 360, 371, 370, 369, 368, 379, 378, 377, 376,
                291, 290, 289, 288, 299, 298, 297, 296, 307, 306, 305, 304, 315, 314, 313, 312,
                259, 258, 257, 256, 264, 265, 266, 267, 272, 273, 274, 275, 280, 281, 282, 283,
                323, 322, 321, 320, 331, 330, 329, 328, 339, 338, 337, 336, 347, 346, 345, 344,
                408, 409, 410, 411, 400, 401, 402, 403, 392, 393, 394, 395, 384, 385, 386, 387,
                472, 473, 474, 475, 464, 465, 466, 467, 456, 457, 458, 459, 448, 449, 450, 451,
                440, 441, 442, 443, 432, 433, 434, 435, 424, 425, 426, 427, 416, 417, 418, 419,
                504, 505, 506, 507, 496, 497, 498, 499, 488, 489, 490, 491, 480, 481, 482, 483,
                36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63,
                100, 101, 102, 103, 108, 109, 110, 111, 116, 117, 118, 119, 124, 125, 126, 127,
                71, 70, 69, 68, 79, 78, 77, 76, 87, 86, 85, 84, 95, 94, 93, 92,
                7, 6, 5, 4, 15, 14, 13, 12, 23, 22, 21, 20, 31, 30, 29, 28,
                156, 157, 158, 159, 148, 149, 150, 151, 140, 141, 142, 143, 132, 133, 134, 135,
                220, 221, 222, 223, 212, 213, 214, 215, 204, 205, 206, 207, 196, 197, 198, 199,
                255, 254, 253, 252, 247, 246, 245, 244, 239, 238, 237, 236, 231, 230, 229, 228,
                191, 190, 189, 188, 183, 182, 181, 180, 175, 174, 173, 172, 167, 166, 165, 164,
                359, 358, 357, 356, 367, 366, 365, 364, 375, 374, 373, 372, 383, 382, 381, 380,
                295, 294, 293, 292, 303, 302, 301, 300, 311, 310, 309, 308, 319, 318, 317, 316,
                263, 262, 261, 260, 268, 269, 270, 271, 276, 277, 278, 279, 284, 285, 286, 287,
                327, 326, 325, 324, 335, 334, 333, 332, 343, 342, 341, 340, 351, 350, 349, 348,
                412, 413, 414, 415, 404, 405, 406, 407, 396, 397, 398, 399, 388, 389, 390, 391,
                476, 477, 478, 479, 468, 469, 470, 471, 460, 461, 462, 463, 452, 453, 454, 455,
                444, 445, 446, 447, 436, 437, 438, 439, 428, 429, 430, 431, 420, 421, 422, 423,
                508, 509, 510, 511, 500, 501, 502, 503, 492, 493, 494, 495, 484, 485, 486, 487,
            ],
            # fmt: on
        }
        stage_config = cpu_stage_config
        stage_config.update(kwargs)

        self.stage_block = {
            stage_name: stage_config,
            gpu_input_stage_name: {
                "kotekan_stage": "testDataGen",
                "type": "random",
                "stream_id": 12432,
                "value": 153,
                "out_buf": gpu_input_name,
            },
        }
        if not cpu:
            self.buffer_block[gain_frb_buffer] = {
                "num_frames": "buffer_depth",
                "frame_size": "2048 * 2 * sizeof_float",
                "metadata_pool": "main_pool",
                "kotekan_buffer": "standard",
            }
            self.buffer_block[gain_psr_buffer] = {
                "num_frames": "buffer_depth",
                "frame_size": "2048 * 2 * num_beams * sizeof_float",
                "metadata_pool": "main_pool",
                "kotekan_buffer": "standard",
                "num_beams": num_psr_beams,
            }
            self.stage_block[read_gain_stage_name] = {
                "kotekan_stage": "ReadGain",
                "in_buf": gpu_input_name,
                "gain_frb_buf": gain_frb_buffer,
                "gain_psr_buf": gain_psr_buffer,
                "num_beams": num_psr_beams,
                "updatable_config": {
                    "gain_frb": "/updatable_config/frb_gain",
                    "gain_psr": "/updatable_config/pulsar_gain",
                },
                "frb_missing_gains": [1.0, 1.0],
            }
            self.stage_block["updatable_config"] = {
                "frb_gain": {
                    "kotekan_update_endpoint": "json",
                    "frb_gain_dir": "/nonexistent",
                },
                "pulsar_gain": {
                    "kotekan_update_endpoint": "json",
                    "pulsar_gain_dir": ["/nonexistent"] * num_psr_beams,
                },
            }


class ReadRawBeamformBuffer(InputBuffer):
    """Read a beamformed buffer saved to a file and stream it as an input

    Parameters
    ----------
    input_dir : string
        Directory to read the input files from.
    """

    _buf_ind = 0

    def __init__(self, input_dir, **kwargs):

        self.name = "rawfileread_buf%i" % self._buf_ind
        stage_name = "rawfileread%i" % self._buf_ind
        self.__class__._buf_ind += 1

        self.input_dir = input_dir

        self.buffer_block = {
            self.name: {
                "kotekan_buffer": "standard",
                "metadata_pool": "main_pool",
                "sizeof_float": 4,
                "frame_size": (
                    "num_data_sets * (samples_per_data_set/downsample_time/downsample_freq) * num_frb_total_beams * sizeof_float"
                ),
                "num_frames": "buffer_depth",
            }
        }

        stage_config = {
            "kotekan_stage": "rawFileRead",
            "buf": self.name,
            "base_dir": input_dir,
            "file_name": self.name,
            "file_ext": "dump",
        }
        stage_config.update(kwargs)

        self.stage_block = {stage_name: stage_config}


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
            "kotekan_stage": "VisRawReader",
            "infile": infile,
            "out_buf": self.name,
            "chunk_size": chunk_size,
            "readahead_blocks": 4,
        }

        self.stage_block = {stage_name: stage_config}


class DumpBasebandBuffer(OutputBuffer):
    """Consume a baseband output buffer and provide its contents at `BasebandBuffer` objects.

    Parameters
    ----------
    output_dir : string
        Temporary directory to output to. The dumped files are not removed.
    num_frames : int or string
        Number of frames in the buffer
    frame_size : int or string
        Frame size in bytes (default: "num_elements * samples_per_data_set")
    """

    _buf_ind = 0

    def __init__(
        self,
        output_dir,
        num_frames,
        frame_size="num_elements * samples_per_data_set",
        **kwargs,
    ):
        self.name = f"baseband_output_buffer_{ self.__class__._buf_ind }"
        self.__class__._buf_ind += 1
        self.output_dir = output_dir
        self.buffer_block = {
            self.name: {
                "kotekan_buffer": "standard",
                "metadata_pool": "baseband_metadata_pool",
                "num_frames": num_frames,
                "frame_size": frame_size,
            }
        }
        stage_name = kwargs.get("stage_name", "dumper_" + self.name)
        stage_config = {
            "kotekan_stage": "rawFileWrite",
            "in_buf": self.name,
            "file_name": self.name,
            "file_ext": "dump",
            "base_dir": self.output_dir,
            "prefix_hostname": False,
        }
        stage_config.update(kwargs)
        self.stage_block = {stage_name: stage_config}

    def load(self):
        """Load the output data from a BasebandReadout buffer.

        Returns
        -------
        dumps : list of BasebandBuffer
            The buffer output, one BasebandBuffer instance per frame
        """
        return baseband_buffer.BasebandBuffer.load_files(
            "{}/{}*.dump".format(self.output_dir, self.name)
        )


class ReadBasebandBuffer(InputBuffer):
    """Write down a BasebandBuffer and reads it with rawFileRead."""

    _buf_ind = 0

    def __init__(self, input_dir, buffer_list):

        self.name = "rawbaseband_buf"
        stage_name = "rawbaseband_read%i" % self._buf_ind
        self.__class__._buf_ind += 1

        self.input_dir = input_dir
        self.buffer_list = buffer_list

        self.buffer_block = {
            self.name: {
                "kotekan_buffer": "standard",
                "metadata_pool": "baseband_metadata_pool",
                "num_frames": "buffer_depth",
                "frame_size": "num_elements * samples_per_data_set",
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
        """Write a list of BasebandBuffer objects to disk."""
        baseband_buffer.BasebandBuffer.to_files(
            self.buffer_list, self.input_dir + "/" + self.name
        )


class DumpFrbPostProcessBuffer(InputBuffer):
    """Consume a FRB output buffer and provide its contents as a list of `frbbuffer.FrbPacket`s.

    Parameters
    ----------
    output_dir : string
        Temporary directory to output to. The dumped files are not removed.
    """

    def __init__(self, output_dir, **kwargs):
        self.name = "frb_post_process_buffer"
        self.output_dir = output_dir

        self.buffer_block = {
            self.name: {
                "num_frames": "buffer_depth + 4",
                "frame_size": "(samples_per_data_set / downsample_time / factor_upchan / timesamples_per_frb_packet)"
                "* 256 * (num_beams_per_frb_packet * num_gpus "
                "* factor_upchan_out * timesamples_per_frb_packet "
                "+ 24 + sizeof_short * num_beams_per_frb_packet + sizeof_short * num_gpus "
                "+ sizeof_float * num_beams_per_frb_packet * num_gpus "
                "+ sizeof_float * num_beams_per_frb_packet * num_gpus)",
                "sizeof_short": 2,
                "sizeof_float": 4,
                "kotekan_buffer": "standard",
                "metadata_pool": "main_pool",
            }
        }

        stage_name = kwargs.get("stage_name", "dump_frb_post_process")
        stage_config = {
            "kotekan_stage": "rawFileWrite",
            "in_buf": self.name,
            "file_name": self.name,
            "file_ext": "dump",
            "base_dir": self.output_dir,
        }
        stage_config.update(kwargs)

        self.stage_block = {stage_name: stage_config}

    def load(self):
        """Load the output data from the frbPostProcess buffer.

        Returns
        -------
        dumps : list of lists of FrbPacket
            The buffer output, one list per frame, a list of packets within the frame.
        """
        return frbbuffer.FrbPacket.load_files(
            "{}/*{}*.dump".format(self.output_dir, self.name)
        )


class DumpPsrPostProcessBuffer(InputBuffer):
    """Consume a Pulsar output buffer and provide its contents as a list of `psrbuffer.PsrPacket`s.

    Parameters
    ----------
    output_dir : string
        Temporary directory to output to. The dumped files are not removed.
    """

    def __init__(self, output_dir, **kwargs):
        self.name = "psr_post_process_buffer"
        self.output_dir = output_dir

        self.buffer_block = {
            self.name: {
                "num_frames": "buffer_depth + 4",
                "frame_size": "udp_pulsar_packet_size * num_stream * num_packet_per_stream",
                "kotekan_buffer": "standard",
                "metadata_pool": "main_pool",
            }
        }

        stage_name = kwargs.get("stage_name", "dump_psr_post_process")
        stage_config = {
            "kotekan_stage": "rawFileWrite",
            "in_buf": self.name,
            "file_name": self.name,
            "file_ext": "dump",
            "base_dir": self.output_dir,
        }
        stage_config.update(kwargs)

        self.stage_block = {stage_name: stage_config}

    def load(self):
        """Load the output data from the frbPostProcess buffer.

        Returns
        -------
        dumps : list of lists of PsrPacket
            The buffer output, one list per frame, a list of packets within the frame.
        """
        return psrbuffer.PsrPacket.load_files(
            "{}/*{}*.dump".format(self.output_dir, self.name)
        )


class DropFramesBuffer(InputBuffer):

    _buf_ind = 0

    def __init__(self, input_buffer, missing_frames=[], **kwargs):

        self.name = "dropframes_buf%i" % self._buf_ind
        stage_name = "dropframes%i" % self._buf_ind
        self.__class__._buf_ind += 1

        self.input_buffer = input_buffer

        self.buffer_block = {
            self.name: {
                "kotekan_buffer": "standard",
                "metadata_pool": "main_pool",
                "num_frames": "buffer_depth",
            }
        }
        self.buffer_block[self.name].update(kwargs)

        stage_config = {
            "kotekan_stage": "TestDropFrames",
            "missing_frames": missing_frames,
            "in_buf": input_buffer.name,
            "out_buf": self.name,
        }
        stage_config.update(kwargs)

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
    gdb: bool
        Run in gdb and in a failure case produce a backtrace. Doesn't work if rest commands are supplied.
    """

    def __init__(
        self,
        stage_type,
        stage_config,
        buffers_in,
        buffers_out,
        global_config=None,
        parallel_stage_type=None,
        parallel_stage_config={},
        rest_commands=None,
        noise=False,
        expect_failure=False,
        gdb=False,
    ):

        config = stage_config.copy()
        parallel_config = parallel_stage_config.copy()
        noise_config = {}
        if global_config is None:
            global_config = {}

        buffers_in_list = []

        if noise:
            if buffers_in is not None:
                noise_config["in_buf"] = buffers_in.name
                buffers_in_list = [buffers_in]
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
            if isinstance(buffers_in, dict):
                buffers_in_list = []
                for key, buf in buffers_in.items():
                    config[key] = buf.name
                    parallel_config[key] = buf.name
                    buffers_in_list.append(buf)
            elif isinstance(buffers_in, (list, tuple)):
                config["in_bufs"] = [buf.name for buf in buffers_in]
                parallel_config["in_bufs"] = [buf.name for buf in buffers_in]
                buffers_in_list = buffers_in
            elif buffers_in is not None:
                config["in_buf"] = buffers_in.name
                parallel_config["in_buf"] = buffers_in.name
                buffers_in_list = [buffers_in]

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

        for buf in itertools.chain(buffers_in_list, buffers_out):
            stage_block.update(buf.stage_block)
            buffer_block.update(buf.buffer_block)
            if hasattr(buf, "global_block"):
                global_config.update(buf.global_block)

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
            gdb=gdb,
        )


default_config = """
---
type: config
log_level: INFO
num_polarizations: 2
num_dishes: 5
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
    kotekan_metadata_pool: chordMetadata
    num_metadata_objects: 30 * buffer_depth

vis_pool:
    kotekan_metadata_pool: VisMetadata
    num_metadata_objects: 30 * buffer_depth
    int_frames: 64

N2_pool:
    kotekan_metadata_pool: N2Metadata
    num_metadata_objects: 30 * buffer_depth
    int_frames: 64
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


def has_hdf5():
    """ Check for HDF5 via registered stages """
    config = KotekanRunner.kotekan_config()

    available = set(config.get("available_stages", []))
    required = {"hdf5FileWrite", "hdf5FileRead", "VisWriter", "VisTranspose"}
    missing = required - available

    return len(missing) == 0


def has_eigenvis():
    """Check for eigenVis via registered stages."""

    config = KotekanRunner.kotekan_config()
    available = set(config.get("available_stages", []))
    return "eigenVis" in available


def has_openmp():
    """Is OpenMP support build in."""
    return KotekanRunner.kotekan_config()["cmake_build_settings"]["USE_OMP"] == "ON"
