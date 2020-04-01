"""Python interface for the kotekan ring buffer using a shared memory region."""

import ctypes
import logging
import mmap
import numpy as np
import os
import posix_ipc

from kotekan.visbuffer import VisRaw

SIZE_UINT64_T = 8

logger = logging.getLogger(__name__)


class SharedMemoryError(Exception):
    """An error has occurred when trying to access the shared memory."""

    pass


class SharedMemoryReaderUsageError(Exception):
    """There was an error in the usage of SharedMemoryReader."""

    pass


class NoNewDataError(Exception):
    """The data in the shared memory has not changed since the last time checked."""

    pass


class Structure(ctypes.Structure):
    """Structural parameters of the shared memory reagion."""

    _fields_ = [
        ("num_writes", ctypes.c_uint64),
        ("num_time", ctypes.c_uint64),
        ("num_freq", ctypes.c_uint64),
        ("size_frame", ctypes.c_uint64),
        ("size_meta", ctypes.c_uint64),
        ("size_data", ctypes.c_uint64),
    ]


class SharedMemoryReader:
    """
    Reader for the shared memory region.

    Every time `update()` is called, the reader will return a view of the last time slots in the
    shared memory via a copy this module keeps. The number of time slots updated and made
    accessible is set by `view_size`.

    The reader keeps a copy of frames in a buffer between calls and only copies new frames on the
    next `update()` call.

    Parameters
    ----------
    semaphore_name : str
        Full path and file name of semaphore.
    shared_memory_name : int
        Full path and file name of shared memory.
    view_size : int
        Number of time slots to give access to. Default 0 (as many as possible).
    """

    # useful constant values
    num_structural_params = 6
    size_structural_data = SIZE_UINT64_T * num_structural_params
    size_access_record_entry = SIZE_UINT64_T
    valid_field_padding = 3
    size_valid_field = 1
    invalid_value = -1

    def __init__(self, semaphore_name, shared_memory_name, view_size):
        self.view_size = view_size

        # maps from access record timestamp to index (0..view_size)
        self._time_index_map = {}

        self.semaphore = posix_ipc.Semaphore(semaphore_name)

        self.shared_mem_name = shared_memory_name
        shared_mem = posix_ipc.SharedMemory(shared_memory_name)

        # 0 means entire file
        self.shared_mem = mmap.mmap(shared_mem.fd, 0, prot=mmap.PROT_READ)

        structure = self._read_structural_data()
        self.num_writes = structure.num_writes
        self.num_time = structure.num_time
        self.num_freq = structure.num_freq
        self.size_frame = structure.size_frame
        self.size_frame_meta = structure.size_meta
        self.size_frame_data = structure.size_data

        # index to write to next in self._data
        self._write_idx = 0

        if self.num_writes == self.invalid_value:
            raise SharedMemoryError(
                "The shared memory referenced by '{}' was marked as invalid by the writer.".format(
                    self.shared_mem_name
                )
            )

        self.len_data = self.num_freq * self.num_time

        if view_size == 0:
            logger.info(
                "Setting `view_size` to size of shared memory buffer ({}).".format(
                    self.num_time
                )
            )
            self.view_size = self.num_time
        elif self.num_time < self.view_size:
            logger.warning(
                "The value `view_size` of the SharedMemoryReader was set to {}, but the shared "
                "memory buffer only has size {}. Setting `view_size` to {}.".format(
                    self.view_size, self.num_time, self.num_time
                )
            )
            self.view_size = self.num_time

        self.size_access_record = self.len_data * self.size_access_record_entry
        self.pos_access_record = self.size_structural_data

        self.size_data = self.len_data * self.size_frame
        self.pos_data = self.pos_access_record + self.size_access_record

        self._initial_validation()

        os.close(shared_mem.fd)

        # Assign simplified frame structure to data in order to access valid field.
        # The valid field is 4byte-aligned.
        align_valid = 4
        self._frame_struct = np.dtype(
            {
                "names": ["valid", "metadata", "data"],
                "formats": [
                    np.uint8,
                    (np.void, self.size_frame_meta),
                    (np.void, self.size_frame_data),
                ],
                "offsets": [0, align_valid, align_valid + self.size_frame_meta],
                "itemsize": self.size_frame,
            }
        )

        # Create buffer for frames kept by this module
        self._data = np.ndarray(
            (self.view_size, self.num_freq), dtype=self._frame_struct
        )

        # initialize valid fields
        self._data["valid"][:, :] = 0

        logger.debug(
            "Created buffer for copy of data (size: {})".format(self._data.shape)
        )
        self._last_access_record = None

    def _initial_validation(self):
        shared_mem_size = (
            self.size_structural_data + self.size_access_record + self.size_data
        )
        if shared_mem_size != self.shared_mem.size():
            raise SharedMemoryError(
                "Expected shared memory to have size {} (but has {}).".format(
                    shared_mem_size, self.shared_mem.size()
                )
            )

    def _read_structural_data(self):
        return Structure.from_buffer_copy(self.shared_mem)

    def __del__(self):
        self.semaphore.release()
        try:
            self.semaphore.unlink()
        except posix_ipc.ExistentialError:
            logger.debug("Semaphore file did not exist when trying to unlink from it.")

        self.shared_mem.close()
        try:
            posix_ipc.unlink_shared_memory(self.shared_mem_name)
        except posix_ipc.ExistentialError:
            logger.debug(
                "Shared memory file did not exist when trying to unlink from it."
            )

    def update(self):
        """
        Read last time samples from the buffer.

        The number of time samples to read is set by the constructor.
        Only copies frames from the buffer if they haven't been read previously.

        The returned view is guaranteed to be sorted by the frames timestamps. It may be
        incomplete, though: Timesamples may be missing and frequencies may be missing. It may
        contain less timesamples than requested.

        Returns
        -------
        VisRaw or None
            Last n time samples, where n is set by `SharedMemoryReader()` or `None` if the shared
            memory buffer is empty.

        Raises
        ------
        SharedMemoryError
            If the shared memory is marked as invalid by the writer or if the structural parameters
            have changed.
        """
        self.shared_mem.seek(0)

        self._validate_shm()

        # get a data update from the ringbuffer
        access_record = self._access_record()

        times = self._filter_last(access_record, self.view_size)
        logger.debug("Reading last {} time slots: {}".format(self.view_size, times))

        self._copy_from_shm(times, access_record)

        # check if any data became invalid while reading it
        access_record_after_copy = self._access_record()
        invalid = np.where(access_record_after_copy != access_record)
        if len(invalid[0]) > 0:
            # get fpga_seq's of invalid frames
            fpga_seq_invalid = access_record[invalid]

            # filter out frames we didn't copy (including invalid ones)
            filter = [fpga_seq in self._time_index_map for fpga_seq in fpga_seq_invalid]
            fpga_seq_invalid = fpga_seq_invalid[filter]
            invalid = (invalid[0][filter], invalid[1][filter])

            if len(fpga_seq_invalid) > 0:
                logger.debug(
                    "{} frames became invalid while reading: (fpga_seq={}, freq_id={})".format(
                        len(invalid[0]), fpga_seq_invalid, invalid[1]
                    )
                )

                # translate time from timestamp to buffer index
                buf_idxs_invalid = (
                    [self._time_index_map[f] for f in fpga_seq_invalid],
                    invalid[1],
                )

                # mark as invalid
                self._data["valid"][buf_idxs_invalid] = 0

        if self._last_access_record is None:
            self._last_access_record = np.ndarray((self.num_time, self.num_freq))
        self._last_access_record[times, :] = access_record_after_copy[times, :]

        # TODO: make sure this works when there's no data in the buffer yet
        # if self._time_index_map == {}:
        #     return None

        return VisRaw.from_buffer(
            self._data, self.size_frame, self.view_size, self.num_freq
        )

    def _remove_oldest_time_slot(self):
        """
        Remove the oldest time slot from the time index map.

        Returns
        -------
        int
            Index of deleted time slot.
        """
        # it would be expensive to pop the from the index_map by value, so we might as well find it
        # by sorting the keys and verify that we are overwriting the oldest data in the buffer
        oldest = sorted(self._time_index_map.keys())[0]
        idx = self._time_index_map.pop(oldest)
        if idx != self._write_idx:
            raise SharedMemoryError(
                "Buffer index to overwrite next should have the oldest time"
                "stamp ({} at {}), but write index points to {} instead.".format(
                    oldest, idx, self._write_idx
                )
            )

        # increment
        self._write_idx = (self._write_idx + 1) % self.view_size

        # mark frames at all other frequencies for this time slot as invalid
        self._data["valid"][idx, :] = 0

        return idx

    def _find_free_time_slot(self):
        """
        Find a free time slot in the time index map.

        Returns
        -------
        int or None
            Index of free time slot.
        """
        idx = self._write_idx
        self._write_idx = (self._write_idx + 1) % self.view_size
        return idx

    def _filter_last(self, access_record, n):
        """
        Get the time indexes of the data for the n time slots with most recent changes.

        Parameters
        ----------
        access_record : numpy array
            Access record of the ring buffer.
        n : int
            Last n time samples to filer.

        Returns
        -------
        list(int)
            Time indexes of n time slots that were written to last. Sorted from old to new.

        Raises
        ------
        SharedMemoryError
            If duplicate time slots are found in the access record.
        """

        last_ts = self._max_ts_per_freq(access_record)
        last_ts, idxs = self._sort_timestamps(last_ts)
        self._check_for_identical_timestamps(last_ts)

        # return last n
        return idxs[-n:]

    def _max_ts_per_freq(self, access_record):
        """ Get the most recent timestamp for each time slot."""
        return access_record.max(axis=1)

    def _sort_timestamps(self, timestamps):
        timestamp_sorted, idxs = list(
            zip(
                *sorted(
                    [
                        (val, i)
                        for i, val in enumerate(timestamps)
                        if val != self.invalid_value
                    ]
                )
            )
        )
        return timestamp_sorted, idxs

    @staticmethod
    def _check_for_identical_timestamps(timestmaps):
        # there should not be multiple identical entries
        if len(timestmaps) != len(set(timestmaps)):
            raise SharedMemoryError(
                "Found duplicate timestamps in access record: {}".format(timestmaps)
            )

    def _copy_from_shm(self, times, access_record):

        # indexes of data to copy from shared mem
        idxs_shm = []

        # indexes for where to put the data in the buffer of this module
        # -> one list of indexes for time, one for frequency
        idxs_buf = ([], [])

        # gather all indexes
        for t in times:
            for f_i in range(self.num_freq):
                # check if this value should be copied: only if the access record changed and
                # is not set to invalid
                if access_record[t, f_i] != self.invalid_value and (
                    self._last_access_record is None
                    or access_record[t, f_i] > self._last_access_record[t, f_i]
                ):
                    logger.debug("Copying value at time={}, freq={}".format(t, f_i))
                    # check if this time slot is in local copy of data
                    try:
                        # this is a new frequency in a known time slot
                        t_i = self._time_index_map[access_record[t, f_i]]
                    except KeyError:
                        # this is a new time slot for the local buffer
                        if len(self._time_index_map) == self.view_size:
                            t_i = self._remove_oldest_time_slot()
                        else:
                            t_i = self._find_free_time_slot()
                        logger.debug(
                            "Setting time index map [{}] to {}.".format(
                                access_record[t, f_i], t_i
                            )
                        )
                        self._time_index_map[access_record[t, f_i]] = t_i

                    # remember index that we want to copy (from and to)
                    idxs_shm.append(t * self.num_freq + f_i)
                    idxs_buf[0].append(t_i)
                    idxs_buf[1].append(f_i)

        # get a view to the data section in the shared memory region
        tmp = np.ndarray(
            self.num_time * self.num_freq,
            self._frame_struct,
            self.shared_mem,
            self.pos_data,
            order="C",
        )
        # copy all the values at once
        self._data[idxs_buf[0], idxs_buf[1]] = tmp[idxs_shm].copy()

    def _access_record(self):
        with self.semaphore:
            record = np.ndarray(
                (self.num_time, self.num_freq),
                np.int64,
                self.shared_mem,
                self.pos_access_record,
                order="C",
            ).copy()
        return record

    def _validate_shm(self):
        """
        Validate the shared memory.

        Check structural parameters as well as the last-time-changed timestamp.

        Raises
        ------
        NoNewDataError
            If there has not been a recorded write since the last check.
        SharedMemoryError
            If the shared memory is marked as invalid by the writer or if the structural parameters
            have changed.
        """

        structure = self._read_structural_data()
        num_writes = structure.num_writes
        num_time = structure.num_time
        num_freq = structure.num_freq
        size_frame = structure.size_frame
        size_frame_meta = structure.size_meta
        size_frame_data = structure.size_data

        if num_writes == 0 and self.num_writes > 0:
            raise SharedMemoryError(
                "The shared memory referenced by '{}' was marked as invalid by the writer.".format(
                    self.shared_mem_name
                )
            )
        logger.debug("{} writes since last time.".format(num_writes - self.num_writes))
        self.num_writes = num_writes

        def _check_structure(old, new, name):
            """Compare old and new structural value and raise if not the same."""
            if old != new:
                raise SharedMemoryError(
                    "The structural value describing the {} has changed from {} to {} since last "
                    "read time.".format(name, old, new)
                )

        _check_structure(self.num_time, num_time, "number of time samples")
        _check_structure(self.num_freq, num_freq, "number of frequencies")
        _check_structure(self.size_frame, size_frame, "size of a frame")
        _check_structure(
            self.size_frame_meta, size_frame_meta, "size of the frame metadata"
        )
        _check_structure(
            self.size_frame_data, size_frame_data, "size of the frame data"
        )
