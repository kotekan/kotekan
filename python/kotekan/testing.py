""" Python interface for validating that a VisRaw object contains valid and expected data"""

import copy
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
        validate_vis(vis_raw, num_elements, ftime, ctime, freq)
        validate_eigenvectors(vis_raw, num_time, num_freq, num_ev, num_elements)


def validate_vis(vis_raw, num_elements, ftime, ctime, freq):
    """Tests that visibility array is populated with integers increasing from zero
    on the diagonal (imaginary part)
    and FPGA sequence number, timestamp, frequency, and frame ID in the first
    four elements (real part)
    the remaining elements are zero"""

    # Construct vis array
    vis = vis_raw.data["vis"]
    valid = vis_raw.valid_frames.astype(np.bool)

    def compare_valid(clause):
        return ~valid.T | clause.T

    # Check that the diagonals are correct
    pi = 0
    for ii in range(num_elements):
        assert (
            compare_valid(vis[:, :, pi].imag == ii)
        ).all(), "expected: {}\n actual: {}".format(ii, vis[:, :, pi].imag)
        pi += num_elements - ii

    # Check that the times are correct
    assert (
        compare_valid(vis[:, :, 0].real == ftime[:].astype(np.float32))
    ).all(), "expected: {}\n actual: {}".format(
        ftime[:].astype(np.float32), vis[:, :, 0].real
    )
    assert (
        compare_valid(vis[:, :, 1].real == ctime[:].astype(np.float32))
    ).all(), "expected: {}\n actual: {}".format(
        ctime[:].astype(np.float32), vis[:, :, 1].real
    )

    # Check that the frequencies are correct
    vfreq = 800.0 - 400.0 * vis[:, :, 2].real / 1024
    assert (
        compare_valid(vfreq == freq[np.newaxis, :])
    ).all(), "expected: {}\n actual: {}".format(freq[np.newaxis, :], vfreq)


def validate_eigenvectors(vis_raw, num_time, num_freq, num_ev, num_elements):
    """
    Tests the structure of eigenvalues, eigenvectors, erms.
    """

    valid = vis_raw.valid_frames.astype(np.bool)

    def compare_valid(clause):
        return ~valid.T | clause.T

    evals = vis_raw.data["eval"]
    evecs = vis_raw.data["evec"]
    erms = vis_raw.data["erms"]

    # Check datasets are present
    assert evals.shape == (
        num_time,
        num_freq,
        num_ev,
    ), "expected: {}\n actual: {}".format((num_time, num_freq, num_ev), evals.shape)
    assert evecs.shape == (
        num_time,
        num_freq,
        num_ev * num_elements,
    ), "expected: {}\n actual: {}".format(
        (num_time, num_freq, num_ev * num_elements), evecs.shape
    )
    assert erms.shape == (num_time, num_freq), "expected: {}\n actual: {}".format(
        (num_time, num_freq), erms.shape
    )

    evecs = evecs.reshape(num_time, num_freq, num_ev, num_elements)

    # Check that the datasets have the correct values
    assert (compare_valid(evals == np.arange(num_ev)[np.newaxis, np.newaxis, :])).all()
    assert (
        compare_valid(
            evecs.real == np.arange(num_ev)[np.newaxis, np.newaxis, :, np.newaxis]
        )
    ).all()
    assert (
        compare_valid(
            evecs.imag == np.arange(num_elements)[np.newaxis, np.newaxis, np.newaxis, :]
        )
    ).all()
    assert (compare_valid(erms == 1.0)).all()


class ValidationFailed(Exception):
    """TheValidation failed."""

    pass


def assert_equal(a, b):
    """Raise `ValidationFailed` if a != b."""
    if a != b:
        raise ValidationFailed("{} != {}".format(a, b))


class SharedMemValidationTest:
    """
    A validation test starting a number of readers and validating read frames.

    Parameters
    ----------
    len_test : int or list(int)
        Number of time slots to validate before ending the test. If it should be different per reader, use a list of ints.
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
    threshold_frame_age_error : float
        Error thresholds in seconds. If the difference between the age of a frame timestamp and
        an expected age (reading speed, update_interval) is greater than the threshold is detected,
        a `ValidationFailed` is raised. If set to a negative value, no exception is raised.
    threshold_cadence_error : float
        When checking difference in frame timestamps, this is the maximum allowed error. If an
        error greater than this is found, a `ValidationFailed` exception is raised. If set to a
        negative value, no exception is raised.

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
        threshold_frame_age_error,
        threshold_cadence_error,
    ):
        # search config for everything we need
        self.cadence = get_from_config("cadence", config)
        self.shm_size = get_from_config("num_samples", config)

        self.error_threshold = threshold_frame_age_error
        self.theshold_cadence_error = threshold_cadence_error
        self.config = config
        self.num_readers = num_readers

        if test_pattern not in test_patterns:
            raise ValueError(
                "Test pattern '{}' is currently not supported, choose one of {}.".format(
                    test_pattern, test_patterns
                )
            )
        self.test_pattern = test_pattern

        if isinstance(len_test, int):
            self.len_test = [len_test] * num_readers
        else:
            self.len_test = len_test
        if len(self.len_test) != num_readers:
            raise ValueError(
                "Expected {} test lengths (= num_readers), but received {}.".format(
                    num_readers, len(self.len_test)
                )
            )

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
        self.validated_fpga_seqs = [set()] * self.num_readers
        self._previous_invalid_timeslots = [0] * self.num_readers
        self._last_read = [0] * self.num_readers

    def _first_update(self, shared_memory_name):
        """Get a minimal first update to get a start time."""
        visraw = SharedMemoryReader(shared_memory_name, 1).update()
        if visraw.time.shape[1] != visraw.num_freq:
            raise ValidationFailed(
                "visRaw.time was expected to have size {} in frequency dimension, but has size {}.".format(
                    visraw.num_freq, visraw.time.shape[1]
                )
            )
        times = np.unique(visraw.time)
        assert_equal(len(times), 1)
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
        active_readers = set(range(self.num_readers))
        while len(active_readers) > 0:
            next_check = min([self._next_check[r] for r in active_readers])
            logger.info("Waiting {}s for next update.".format(next_check))
            time.sleep(next_check)
            for r in copy.copy(active_readers):
                self._next_check[r] -= next_check
                if self._next_check[r] <= 0:
                    self.validated_fpga_seqs[r] = self.validated_fpga_seqs[r].union(
                        self._check(r)
                    )
                    if len(self.validated_fpga_seqs[r]) >= self.len_test[r]:
                        logger.info("Reader {} done.".format(r))
                        active_readers.remove(r)
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

    def _validate_age(self, visraw, r):
        """Check if the difference between current time and the frame timestamps is acceptable.

        Parameters
        ----------
        visraw : VisRaw
            The data.
        r : int
            ID of the reader that gave us visraw.
        """
        now = time.time()
        times_valid = visraw.time[visraw.valid_frames.astype(np.bool)]
        assert_equal(visraw.time.shape[1], visraw.num_freq)
        assert_equal(visraw.time.shape[0], self.view_sizes[r])

        times = np.unique(times_valid)

        # check first time slot
        for timestamp in times:
            # timestamp = times[t]
            seconds = timestamp["ctime"]
            fpga_seq = timestamp["fpga_count"]
            if fpga_seq in self.validated_fpga_seqs[r]:
                logger.debug(
                    "Frames with fpga_seq={} already read before, skipping...".format(
                        fpga_seq
                    )
                )
                continue
            ctime = time.ctime(seconds)
            age = now - seconds
            logger.info(
                "Validating fpga_seq {} from {} (age {}s).".format(fpga_seq, ctime, age)
            )

            error = age - self.update_time[r][-1] - self.update_interval[r]
            if error > 0:
                msg = "Frames with fpga_seq={} read {:.3}s later than expected (age={:.3}, readerID={}).".format(
                    fpga_seq, error, age, r
                )
                if (
                    0 <= self.error_threshold < error
                    and len(self.validated_fpga_seqs[r]) > self.view_sizes[r]
                ):
                    raise ValidationFailed(msg)
                else:
                    logger.info(msg)

    def _validate_time(self, visraw, r):
        """Validation of all things time."""
        self._validate_age(visraw, r)
        self._validate_missing(visraw, r)

    def _validate_missing(self, visraw, r):
        """Check for missing frames and if this was to be expected."""

        now = time.time()
        valid = visraw.valid_frames.astype(np.bool)
        invalid_timeslots = np.sum(~valid) // visraw.num_freq
        times_valid = np.ma.masked_where(~valid, visraw.time, copy=False)
        times = np.unique(times_valid)

        # remove time slots we already checked last time from this check:
        # if the update interval of the reader is higher than the time it takes the writer to fill
        # the ringbuffer, we can expect big differences between those old, re-read values and
        # the rest. But that's not interesting...
        # TODO: this can be done in a nicer way. but I'm too tired...
        new_ts = []
        for t in range(len(times)):
            tt = times[t]
            if tt[0] and tt[0] not in self.validated_fpga_seqs[r]:
                new_ts.append(t)
        times = times[new_ts]

        if len(times) == 0:
            # nothing new to check
            return

        times.sort()
        timestamps = [t[1] for t in times]

        # check if there are missing times in between
        if len(timestamps) > 1:
            cadence_error = np.abs(np.diff(timestamps)) - self.cadence
            for error in cadence_error:
                if error > 0:
                    msg = "Difference between times in one VisRaw higher than cadence={}: {}".format(
                        self.cadence, error
                    )
                    if 0 <= self.theshold_cadence_error < error:
                        raise ValidationFailed(msg)
                    else:
                        logger.info(msg)

        # check for data missed between reads
        missed_time = timestamps[0] - self._last_update_time[r]
        missed_time_measured = now - self._last_read[r]
        frames_per_update = np.ceil(missed_time_measured / self.cadence)
        expected = (
            np.max((0, frames_per_update - self.shm_size))
            + invalid_timeslots
            + self._previous_invalid_timeslots[r]
            + self.cadence
        )
        self._previous_invalid_timeslots[r] = invalid_timeslots

        error = missed_time - expected
        if error > 0:
            msg = (
                "Between this read and the last one of reader {}, there is a gap in frame "
                "timestamps of {}s (we expected not more than {}s).".format(
                    r, missed_time, expected
                )
            )
            if (
                0 <= self.theshold_cadence_error * expected < error
                and len(self.validated_fpga_seqs[r]) > self.view_sizes[r]
            ):
                raise ValidationFailed(msg)
            else:
                logger.info(msg)

        # things we want to know next time
        self._last_read[r] = now
        self._last_update_time[r] = timestamps[-1]
