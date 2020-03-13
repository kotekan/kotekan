"""Python interface for the kotekan ring buffer using a shared memory region."""

import mmap
import os
import posix_ipc
import struct

FMT_UINT64_T = "@Q"
SIZE_UINT64_T = 8


class SharedMemoryReader:
    """Reader for the shared memory region."""

    num_structural_params = 5
    size_structural_data = SIZE_UINT64_T * num_structural_params

    def __init__(self, semaphore_name, shared_memory_name):
        self.semaphore = posix_ipc.Semaphore(semaphore_name)

        self.shared_mem_name = shared_memory_name
        shared_mem = posix_ipc.SharedMemory(shared_memory_name)

        # 0 means whole file
        self.structural_data = mmap.mmap(
            shared_mem.fd, self.size_structural_data, prot=mmap.PROT_READ
        )

        (
            self.num_time,
            self.num_freq,
            self.size_frame,
            self.size_meta,
            self.size_data,
        ) = self._read_structural_data()

        os.close(shared_mem.fd)

    def _read_structural_data(self):
        num_time = struct.unpack_from(
            FMT_UINT64_T, self.structural_data.read(SIZE_UINT64_T)
        )[0]
        num_freq = struct.unpack_from(
            FMT_UINT64_T, self.structural_data.read(SIZE_UINT64_T)
        )[0]
        size_frame = struct.unpack_from(
            FMT_UINT64_T, self.structural_data.read(SIZE_UINT64_T)
        )[0]
        size_meta = struct.unpack_from(
            FMT_UINT64_T, self.structural_data.read(SIZE_UINT64_T)
        )[0]
        size_data = struct.unpack_from(
            FMT_UINT64_T, self.structural_data.read(SIZE_UINT64_T)
        )[0]
        return (num_time, num_freq, size_frame, size_meta, size_data)

    def __del__(self):
        self.semaphore.release()
        self.semaphore.unlink()

        self.structural_data.close()
        posix_ipc.unlink_shared_memory(self.shared_mem_name)

    def read_last(n: int):
        pass
        # return visRaw()

    def read_new_since(self, timestamp):
        pass
        # return visRaw()

    # def _update(self):
    #     self._validate_shm()
    #
    #     with self.semaphore:
    #         for freq_id in self.freq_ids:
    #             for time_slot in self.time_slots:
    #                 if _frame_changed():
    #                     struct.unpack_from("@c", memory_map_buf.read(1))[0]
