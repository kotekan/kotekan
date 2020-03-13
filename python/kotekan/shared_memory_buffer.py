"""Python interface for the kotekan ring buffer using a shared memory region."""

import logging
import mmap
import numpy as np
import os
import posix_ipc
import struct

FMT_UINT64_T = "Q"
SIZE_UINT64_T = 8

logger = logging.getLogger(__name__)


class SharedMemoryError(Exception):
    """An error has occurred when trying to access the shared memory."""

    pass


class NoNewDataError(Exception):
    """The data in the shared memory has not changed since the last time checked."""

    pass


class SharedMemoryReader:
    """Reader for the shared memory region."""

    num_structural_params = 6
    size_structural_data = SIZE_UINT64_T * num_structural_params
    size_access_record_entry = SIZE_UINT64_T
    valid_field_padding = 3
    size_valid_field = 1

    def __init__(self, semaphore_name, shared_memory_name):
        self.semaphore = posix_ipc.Semaphore(semaphore_name)

        self.shared_mem_name = shared_memory_name
        shared_mem = posix_ipc.SharedMemory(shared_memory_name)

        # 0 means entire file
        self.shared_mem = mmap.mmap(shared_mem.fd, 0, prot=mmap.PROT_READ)

        (
            self.time_of_last_change,
            self.num_time,
            self.num_freq,
            self.size_frame,
            self.size_frame_meta,
            self.size_frame_data,
        ) = self._read_structural_data()

        if self.num_writes == -1:
            raise SharedMemoryError(
                "The shared memory referenced by '{}' was marked as invalid by the writer.".format(
                    self.shared_mem_name
                )
            )

        self.len_data = self.num_freq * self.num_freq

        self.size_access_record = self.len_data * self.size_access_record_entry
        self.pos_access_record = self.size_structural_data

        self.size_data = self.len_data * self.num_time
        self.pos_data = self.pos_access_record + self.size_access_record

        os.close(shared_mem.fd)

    def _read_structural_data(self):
        num_writes = struct.unpack_from(
            FMT_UINT64_T, self.shared_mem.read(SIZE_UINT64_T)
        )[0]
        num_time = struct.unpack_from(
            FMT_UINT64_T, self.shared_mem.read(SIZE_UINT64_T)
        )[0]
        num_freq = struct.unpack_from(
            FMT_UINT64_T, self.shared_mem.read(SIZE_UINT64_T)
        )[0]
        size_frame = struct.unpack_from(
            FMT_UINT64_T, self.shared_mem.read(SIZE_UINT64_T)
        )[0]
        size_meta = struct.unpack_from(
            FMT_UINT64_T, self.shared_mem.read(SIZE_UINT64_T)
        )[0]
        size_data = struct.unpack_from(
            FMT_UINT64_T, self.shared_mem.read(SIZE_UINT64_T)
        )[0]
        return (
            num_writes,
            num_time,
            num_freq,
            size_frame,
            size_meta,
            size_data,
        )

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

    def read_last(n: int):
        pass
        # return visRaw()

    def read_new_since(self, timestamp):
        pass
        # return visRaw()

    def _access_record(self):
        record = np.ndarray(
            (self.num_freq, self.num_time),
            np.uint64,
            self.shared_mem,
            self.pos_access_record,
            order="C",
        )
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

        (
            num_writes,
            num_time,
            num_freq,
            size_frame,
            size_frame_meta,
            size_frame_data,
        ) = self._read_structural_data()

        if num_writes == -1:
            raise SharedMemoryError(
                "The shared memory referenced by '{}' was marked as invalid by the writer.".format(
                    self.shared_mem_name
                )
            )

        def _check_structure(old, new, name):
            """Compare old and new structural value and raise if not the same."""
            if old != new:
                raise SharedMemoryError(
                    "The structural value describing the {} has changed from {} to {} since last read time.".format(
                        name, old, new
                    )
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

        if self.num_writes == num_writes:
            raise NoNewDataError()
        self.num_writes = num_writes

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
