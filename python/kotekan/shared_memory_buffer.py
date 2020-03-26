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
        ("num_writes", ctypes.c_ulonglong),
        ("num_time", ctypes.c_ulonglong),
        ("num_freq", ctypes.c_ulonglong),
        ("size_frame", ctypes.c_ulonglong),
        ("size_meta", ctypes.c_ulonglong),
        ("size_data", ctypes.c_ulonglong),
    ]


class SharedMemoryReader:
    """
    Reader for the shared memory region.

    Parameters
    ----------
    semaphore_name : str
        Full path and file name of semaphore.
    shared_memory_name : int
        Full path and file name of shared memory.
    buffer_size : int
        Number of time slots to keep in local copy of data. Increase if large amount of data is to
        be read repeatedly. Decrease for less memory usage.
    """

    num_structural_params = 6
    size_structural_data = SIZE_UINT64_T * num_structural_params
    size_access_record_entry = SIZE_UINT64_T
    valid_field_padding = 3
    size_valid_field = 1
    invalid_value = -1

    def __init__(self, semaphore_name, shared_memory_name, buffer_size):
        if buffer_size < 1:
            raise NotImplementedError("Buffer size of 0 is not implemented yet.")
        self.buffer_size = buffer_size

        # maps from access record timestamp to index (0..buffer_size)
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

        if self.num_writes == self.invalid_value:
            raise SharedMemoryError(
                "The shared memory referenced by '{}' was marked as invalid by the writer.".format(
                    self.shared_mem_name
                )
            )

        self.len_data = self.num_freq * self.num_time

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
        self._data = np.ndarray((buffer_size, self.num_freq), dtype=self._frame_struct)

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

    def read_last(self, n):
        """
        Read last n time samples from the buffer.

        Only copies from buffer if not read previously.

        Parameters
        ----------
        n : int
            Number of time samples to read.

        Returns
        -------
        VisRaw
            Last n time samples.

        Raises
        ------
        SharedMemoryError
            If the shared memory is marked as invalid by the writer or if the structural parameters
            have changed.
        """
        if n > self.buffer_size:
            raise SharedMemoryReaderUsageError(
                "Can't read more than {0} last time slots, because buffer size is set to {0} "
                "(Tried to read last {1} time slots.).".format(self.buffer_size, n)
            )

        self.shared_mem.seek(0)

        self._validate_shm()

        # get a data update from the ringbuffer
        access_record = self._access_record()

        times = self._filter_last(access_record, n)
        logger.debug("Reading last {} time slots: {}".format(n, times))

        self._copy_from_shm(times, access_record)

        # check if any data became invalid while reading it
        access_record_after_copy = self._access_record()
        for t in times:
            for f in range(self.num_freq):
                if access_record_after_copy[t, f] != access_record[t, f]:
                    logger.debug(
                        "Data at t={}, f={} became invalid while reading it.".format(
                            t, f
                        )
                    )
                    self._data[self._time_index_map[t], f] = None
        if self._last_access_record is None:
            self._last_access_record = np.ndarray((self.num_time, self.num_freq))
        self._last_access_record[times, :] = access_record_after_copy[times, :]

        # return visRaw()
        return self._return_data_copy_last(n)

    def _remove_oldest_time_slot(self):
        """
        Remove the oldest time slot from the time index map.

        Returns
        -------
        int or None
            Index of deleted time slot.
        """
        oldest = sorted(self._time_index_map.keys())[-1]
        return self._time_index_map.pop(oldest)

    def _find_free_time_slot(self):
        """
        Find a free time slot in the time index map.

        Returns
        -------
        int or None
            Index of free time slot.
        """
        for i in range(self.buffer_size):
            if i not in self._time_index_map.values():
                logger.debug("Found free time slot: {}".format(i))
                return i
        return None

    def _return_data_copy_last(self, n):
        if not self._time_index_map:
            return None

        # sort time index map
        last_ts, idxs = list(zip(*sorted(self._time_index_map.items())))

        # select last n time slots
        idxs = idxs[-n:]
        return VisRaw.from_nparray(
            self._data[idxs, :],
            self.size_frame,
            len(idxs),
            self.num_freq,
            num_elements=7,
            num_stack=28,
            num_ev=0,
        )

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

    def read_since(self, timestamp):
        """
        Read all new time samples since a given time.

        Only copies from buffer if not read previously.

        Parameters
        ----------
        timestamp : int
            Timestamp of oldest sample to read.

        Returns
        -------
        VisRaw
            Time samples read from buffer.
        """
        self.shared_mem.seek(0)

        self._validate_shm()
        # get a data update from the ringbuffer
        access_record = self._access_record()

        times = self._filter_since(access_record, timestamp)
        logger.debug(
            "Reading {} time slots since {}: {}".format(len(times), timestamp, times)
        )

        self._copy_from_shm(times, access_record)

        # check if any data became invalid while reading it
        access_record_after_copy = self._access_record()
        for t in times:
            for f in range(self.num_freq):
                if access_record_after_copy[t, f] != access_record[t, f]:
                    self._data[self._time_index_map[t], f] = None
        if self._last_access_record is None:
            self._last_access_record = np.ndarray((self.num_time, self.num_freq))
        self._last_access_record[times, :] = access_record_after_copy[times, :]

        # return visRaw()
        return self._return_data_copy_since(timestamp)

    def _max_ts_per_freq(self, access_record):
        # get the most recent timestamp for each time slot
        last_ts = [None] * self.num_time
        for t in range(self.num_time):
            for f in range(self.num_freq):
                if (
                    last_ts[t] is None and access_record[t, f] != self.invalid_value
                ) or (last_ts[t] is None or access_record[t, f] > last_ts[t]):
                    last_ts[t] = access_record[t, f]
        return last_ts

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

    def _check_for_identical_timestamps(self, timestmaps):
        # there should not be multiple identical entries
        if len(timestmaps) != len(set(timestmaps)):
            raise SharedMemoryError(
                "Found duplicate timestamps in access record: {}".format(timestmaps)
            )

    def _filter_since(self, access_record, timestamp):
        last_ts = self._max_ts_per_freq(access_record)

        # remove invalid timestmaps
        while True:
            try:
                last_ts.remove(self.invalid_value)
            except ValueError:
                break

        self._check_for_identical_timestamps(last_ts)

        # return timestamps newer than the given one
        ll = [i for i, v in enumerate(last_ts) if v > timestamp]
        logger.debug(
            "Found {} time slots that are newer than {} in {}: {}.".format(
                len(ll), timestamp, last_ts, ll
            )
        )
        return ll

    def _return_data_copy_since(self, timestamp):
        if not self._time_index_map:
            return None

        # sort time index map
        last_ts, idxs = list(zip(*sorted(self._time_index_map.items())))

        # get indexes of timestamps newer than the given on
        idxs = [i for i, v in zip(idxs, last_ts) if v > timestamp]
        logger.debug(
            "Found {} time slots that are newer than {} in {}: {}.".format(
                len(idxs), timestamp, last_ts, idxs
            )
        )

        # select last n time slots
        return VisRaw.from_nparray(
            self._data[idxs, :],
            self.size_frame,
            len(idxs),
            self.num_freq,
            num_elements=7,
            num_stack=28,
            num_ev=0,
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
                        t_i = self._time_index_map[access_record[t, f_i]]
                    except KeyError:
                        if len(self._time_index_map) == self.buffer_size:
                            t_i = self._remove_oldest_time_slot()

                            # mark frames at all other frequencies for this time slot as invalid
                            self._data[t_i, :].valid = [0] * self.num_freq
                        else:
                            t_i = self._find_free_time_slot()
                            if t_i is None:
                                raise SharedMemoryError(
                                    "Can't find a free slot in buffer (has keys {}).".format(
                                        self._time_index_map.keys()
                                    )
                                )
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

    # def _update(self):
    # """
    #     Raises
    #     ------
    #     NoNewDataError
    #         If there has not been a recorded write since the last check.
    #     SharedMemoryError
    #         If the shared memory is marked as invalid by the writer or if the structural parameters
    #         have changed.
    # """
    #     self._validate_shm()
    #
    #     with self.semaphore:
    #         for freq_id in self.freq_ids:
    #             for time_slot in self.time_slots:
    #                 if _frame_changed():
    #                     struct.unpack_from("@c", memory_map_buf.read(1))[0]
