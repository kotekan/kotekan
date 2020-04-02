""" Python interface for validating that a VisRaw object contains valid and expected data"""

import logging
import numpy as np
import time

from kotekan.shared_memory_buffer import SharedMemoryReader

logger = logging.getLogger(__name__)

test_patterns = ["default"]


def get_from_config(name, config):
    """
    Get value from any level of config dict.

    Parameters
    ----------
    name : str
        Name (key) of the value.
    config : dict
        Config dictionary.

    Returns
    -------
    Requested value.

    Raises
    ------
    ValueError
        If there is more than one entry with the requested name.
        Todo: offer passing in (partial) path as solution.
    """
    if not isinstance(config, dict):
        raise ValueError(
            "Expected 'config' of type 'dict' (got '{}').".format(type(config).__name__)
        )
    result = None
    try:
        result = config[name]
    except KeyError:
        for key, value in config.items():
            if isinstance(value, dict):
                recursive_result = get_from_config(name, value)
                if recursive_result is not None:
                    if result is None:
                        result = recursive_result
                    else:
                        raise ValueError(
                            "Config contained at least 2 entries named {}: {} and {}.".format(
                                name, recursive_result, result
                            )
                        )
            elif isinstance(value, list):
                for entry in value:
                    if isinstance(entry, dict):
                        recursive_result = get_from_config(name, entry)
                        if recursive_result is not None:
                            if result is None:
                                result = recursive_result
                            else:
                                raise ValueError(
                                    "Config contained at least 2 entries named {}: {} and {}.".format(
                                        name, recursive_result, result
                                    )
                                )
    return result


def validate(vis_raw, config, pattern_name=""):
    """Validate an input `vis_raw` filled using `pattern_name` with `config`.

    Parameters
    ----------
    vis_raw : visbuffer.VisRaw
        The output of a Kotekan buffer.
    config : dict
        Structural parameters of the buffer and frames.
        Attributes: num_elements, num_ev, freq, total_frames
    pattern_name : str
        Name of the FakeVisPattern used to fill the buffer.

    vis_raw: VisRaw; pattern_name: str"""

    # Construct vis array
    vis = vis_raw.data["vis"]

    # Extract metadata
    ftime = vis_raw.time["fpga_count"]
    ctime = vis_raw.time["ctime"]
    freq_ids = get_from_config("freq_ids", config)

    freq = 800.0 - 400.0 * np.array(freq_ids) / 1024

    num_elements = get_from_config("num_elements", config)
    num_ev = get_from_config("num_ev", config)
    num_freq = len(freq_ids)
    num_time = vis_raw.num_time

    if pattern_name == "default":
        validate_vis(vis, num_elements, ftime, ctime, freq)
        validate_eigenvectors(vis_raw, num_time, num_freq, num_ev, num_elements)


def validate_vis(vis, num_elements, ftime, ctime, freq):
    """Tests that visibility array is populated with integers increasing from zero
    on the diagonal (imaginary part)
    and FPGA sequence number, timestamp, frequency, and frame ID in the first
    four elements (real part)
    the remaining elements are zero"""

    # Check that the diagonals are correct
    pi = 0
    for ii in range(num_elements):
        assert (vis[:, :, pi].imag == ii).all(), 'expected: {}\n actual: {}'.format(ii, vis[:, :, pi].imag)
        pi += num_elements - ii

    # Check that the times are correct
    assert (vis[:, :, 0].real == ftime[:].astype(np.float32)).all(), 'expected: {}\n actual: {}'.format(ftime[:].astype(np.float32), vis[:, :, 0].real)
    assert (vis[:, :, 1].real == ctime[:].astype(np.float32)).all(), 'expected: {}\n actual: {}'.format(ctime[:].astype(np.float32), vis[:, :, 1].real)

    # Check that the frequencies are correct
    vfreq = 800.0 - 400.0 * vis[:, :, 2].real / 1024
    assert (vfreq == freq[np.newaxis, :]).all(), 'expected: {}\n actual: {}'.format(freq[np.newaxis, :], vfreq)


def validate_eigenvectors(vis_raw, num_time, num_freq, num_ev, num_elements):
    """
    Tests the structure of eigenvalues, eigenvectors, erms.
    """
    evals = vis_raw.data["eval"]
    evecs = vis_raw.data["evec"]
    erms = vis_raw.data["erms"]

    # Check datasets are present
    assert evals.shape == (num_time, num_freq, num_ev), 'expected: {}\n actual: {}'.format((num_time, num_freq, num_ev), evals.shape)
    assert evecs.shape == (num_time, num_freq, num_ev * num_elements), 'expected: {}\n actual: {}'.format((num_time, num_freq, num_ev * num_elements), evecs.shape)
    assert erms.shape == (num_time, num_freq), 'expected: {}\n actual: {}'.format((num_time, num_freq), erms.shape)

    evecs = evecs.reshape(num_time, num_freq, num_ev, num_elements)

    # Check that the datasets have the correct values
    assert (evals == np.arange(num_ev)[np.newaxis, np.newaxis, :]).all()
    assert (
        evecs.real == np.arange(num_ev)[np.newaxis, np.newaxis, :, np.newaxis]
    ).all()
    assert (
        evecs.imag == np.arange(num_elements)[np.newaxis, np.newaxis, np.newaxis, :]
    ).all()
    assert (erms == 1.0).all()


class ValidationFailed(Exception):
    """TheValidation failed."""

    pass


class SharedMemValidationTest:
    """
    A validation test starting a number of readers and validating read frames.

    Parameters
    ----------
    len_test : int
        Number of time slots to validate before ending the test.
    config : dict
        Kotekan config used.
    num_readers : int
        Number of readers to run.
    shared_memory_name : int
        Full path and file name of shared memory.
    view_sizes : int or list(int)
        Number of time slots to give access to. 0 means as many as possible (= size of shared memory buffer). If it should be different per reader, use a list of ints.
    test_pattern : str
        Name of the test pattern to expect in the frames.
    update_interval : int or list(int)
        Number of seconds to wait between update() calls. If it should be different per reader, use a list of ints.
    error_threshold : float
        Error thresholds in seconds. If a timing error greater than the threshold is detected,
        a `ValidationFailed` is raised. If set to a negative value, no exception is raised.

    Attributes
    ----------
    update_time : list(list(int))
        Number of seconds spent by each readers update() calls.
    delay : list(list(float))
        Delays of frames (time between frame creation and validation) for each reader.
    expected_delay : list(list(float))
        Expected delays of frames (time between frame creation and validation) for each reader.
        This includes an estimate of missed frames due to buffer and view sizes in config and the
        time it took the reader to update
    """

    def __init__(
        self,
        len_test,
        config,
        num_readers,
        shared_memory_name,
        view_sizes,
        test_pattern,
        update_interval,
        error_threshold,
    ):
        # search config for everything we need
        self.cadence = get_from_config("cadence", config)
        self.shm_size = get_from_config("nsamples", config)

        self.error_threshold = error_threshold
        self.config = config
        self.num_readers = num_readers
        self.len_test = len_test

        if test_pattern not in test_patterns:
            raise ValueError(
                "Test pattern '{}' is currently not supported, choose one of {}.".format(
                    test_pattern, test_patterns
                )
            )
        self.test_pattern = test_pattern

        if isinstance(view_sizes, int):
            self.view_sizes = [view_sizes] * num_readers
        else:
            self.view_sizes = view_sizes
        if len(self.view_sizes) != num_readers:
            raise ValueError(
                "Expected {} view sizes (= num_readers), but received {}.".format(
                    num_readers, len(self.view_sizes)
                )
            )

        if isinstance(update_interval, int):
            self.update_interval = [update_interval] * num_readers
        else:
            self.update_interval = update_interval
        if len(self.update_interval) != num_readers:
            raise ValueError(
                "Expected {} update intervals (= num_readers), but received {}.".format(
                    num_readers, len(self.update_interval)
                )
            )

        # Start readers and metrics
        self._readers = list()
        self.delay = list()
        self.expected_delay = list()
        self.update_time = list()
        self._next_check = [0] * self.num_readers
        for i in range(num_readers):
            view_size = self.view_sizes[i]
            self._readers.append(SharedMemoryReader(shared_memory_name, view_size))
            # allow the view size to be set/changed by the reader
            self.view_sizes[i] = self._readers[i].view_size
            self.delay.append([])
            self.expected_delay.append([])
            self.update_time.append([])

        self._first_update(shared_memory_name)
        self.validated_fpga_seqs = set()

    def _first_update(self, shared_memory_name):
        """Get a minimal first update to get a start time."""
        visraw = SharedMemoryReader(shared_memory_name, 1).update()
        assert visraw.time.shape[1] == visraw.num_freq
        times = np.unique(visraw.time)
        assert len(times) == 1
        timestamp = visraw.time[0, 0]
        self.start_time = timestamp["ctime"]
        fpga_seq = timestamp["fpga_count"]
        ctime = time.ctime(self.start_time)
        age = time.time() - self.start_time
        logger.info(
            "Starting validation with fpga_seq {} from {} (age {} seconds).".format(
                fpga_seq, ctime, age
            )
        )
        self._last_update_time = [self.start_time] * self.num_readers

    def run(self):
        while self.len_test > len(self.validated_fpga_seqs):
            next_check = min(self._next_check)
            logger.info("Waiting {}s for next update.".format(next_check))
            time.sleep(next_check)
            for r in range(self.num_readers):
                self._next_check[r] -= next_check
                if self._next_check[r] <= 0:
                    self.validated_fpga_seqs = self.validated_fpga_seqs.union(
                        self._check(r)
                    )
                    self._next_check[r] = self.update_interval[r]

    def _check(self, r):
        logger.info("Updating reader {}.".format(r))
        reader = self._readers[r]
        time0 = time.time()
        visraw = reader.update()
        self.update_time[r].append(time.time() - time0)

        self._validate_time(visraw, r)
        validate(visraw, self.config, self.test_pattern)

        return set(np.unique(visraw.time)[:]["fpga_count"])

    def _validate_time(self, visraw, r):
        age = time.time() - self._last_update_time[r]

        assert visraw.time.shape[1] == visraw.num_freq, "{} != {}".format(
            visraw.time.shape[1], visraw.num_freq
        )
        times = np.unique(visraw.time)
        assert len(times) == self.view_sizes[r], "{} != {} (times={})".format(
            len(times), self.view_sizes[r], times
        )

        # check first time slot
        timestamp = times[0]

        seconds = timestamp["ctime"]
        fpga_seq = timestamp["fpga_count"]
        ctime = time.ctime(seconds)
        logger.info(
            "Validating fpga_seq {} from {} (age {}s).".format(fpga_seq, ctime, age)
        )

        # determine if this age is acceptable
        cadence = self.cadence

        # calculate how much data we missed given cadence, limited buffer size and update frequency
        # TODO: handle case where no new frames since last update expected
        u_interval = self.update_interval[r]
        frames_per_update = np.ceil(u_interval // cadence)
        missed_frames = frames_per_update - self.shm_size
        if missed_frames < 0:
            missed_frames = 0
        missed_time = missed_frames * cadence
        expected_delay = missed_time + self.update_time[r][-1]
        error = age - expected_delay
        if error > cadence:
            logger.info("Time error: {}".format(error))
            if 0 < self.error_threshold < error:
                raise ValidationFailed(
                    "Time error of {}s in frame {} (age={})".format(
                        error, fpga_seq, age
                    )
                )
        self.expected_delay[r].append(expected_delay)
        self.delay[r].append(age)

        # check the rest
        for t in range(1, self.view_sizes[r]):
            timestamp = times[r]

            seconds = timestamp["ctime"]
            fpga_seq = timestamp["fpga_count"]
            ctime = time.ctime(seconds)
            logger.info(
                "Validating fpga_seq {} from {} (age {}s).".format(fpga_seq, ctime, age)
            )

            # determine if this age is acceptable
            cadence = self.cadence
            expected_delay = cadence + self.update_time[r][-1] + self.update_interval[r]
            error = age - expected_delay
            if error > 0:
                logger.warning("Time error: {}".format(error))
                if 0 < self.error_threshold < error:
                    raise ValidationFailed(
                        "Time error of {}s in frame {} (age={})".format(
                            error, fpga_seq, age
                        )
                    )
            self.delay[r].append(age)
            self.expected_delay[r].append(expected_delay)

        self._last_update_time[r] = seconds
