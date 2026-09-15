"""Read an N2Buffer dump into python.
"""
import ctypes
import os
import io

import numpy as np
from kotekan import timespec
from kotekan import telescope


class N2Metadata(ctypes.Structure):
    """Wrap an N2Metadata struct."""

    _fields_ = [
        ("freq_id", ctypes.c_uint32),
        ("freq_MHz", ctypes.c_double),
        # Time index
        ("abs_time_idx", ctypes.c_uint64),
        # Earth orientation parameters
        ("time_center_eop", telescope.EOP),
        ("bin_eop", telescope.EOP),
        ("bin_start_ERA_deg", ctypes.c_double),
        ("bin_end_ERA_deg", ctypes.c_double),
        ("bin_start_ERAL_deg", ctypes.c_double),
        ("bin_end_ERAL_deg", ctypes.c_double),
        # FPGA timing
        ("fpga_start_tick", ctypes.c_uint64),
        ("frame_start_time_ns", ctypes.c_uint64),
        ("frame_length_fpga_ticks", ctypes.c_uint64),
        ("n_valid_fpga_ticks", ctypes.c_uint64),
        ("n_rfi_fpga_ticks", ctypes.c_uint64),
        ("n_rfi_only_fpga_ticks", ctypes.c_uint64),
        ("n_pl_fpga_ticks", ctypes.c_uint64),
        # RFI Excision
        ("rfi_frame_excision_enabled", ctypes.c_bool),
        ("rfi_frame_excision_num", ctypes.c_uint32),
        ("rfi_frame_excision_threshold", ctypes.c_float * 8),
        ("rfi_frame_excision_fraction", ctypes.c_float * 8),
        # For CHIME: dataset_id
        ("dataset_id", ctypes.c_uint64 * 2),
    ]


class N2Buffer(object):
    """Python representation of an N2Buffer dump.

    Access the data through the `vis`, `weight`, `eval`, `evec` and `erms`
    attributes which are all numpy arrays.

    Parameters
    ----------
    buffer : bytearray
        Memory to provide a view of.
    skip : int, optional
        Number of bytes to skip from the beginning of the buffer. Useful for
        raw dumps when the metadata size is given in the first four bytes.
    support_mode : {"scalar", "per_product_v1"}, optional
        Use "per_product_v1" for a dump with per-product counts. The default
        is "scalar"; raw dumps do not store this setting.
    """

    def __init__(
        self,
        buffer,
        skip=4,
        num_elements=None,
        num_prod=None,
        num_ev=None,
        support_mode="scalar",
    ):

        if support_mode not in ("scalar", "per_product_v1"):
            raise ValueError(f"Unknown N2 support_mode: {support_mode}")
        self.support_mode = support_mode
        # Use a view so metadata and array edits update the original buffer.
        self._buffer = memoryview(buffer)[skip:]
        self._num_elements = num_elements
        self._num_prod = num_prod
        self._num_ev = num_ev

        meta_size = ctypes.sizeof(N2Metadata)

        # print("Loading N2Buffer")
        # print("File size: {:d}".format(len(buffer)))
        # print("buffer size: {:d}".format(len(self._buffer)))
        # print("meta size: {:d}".format(meta_size))
        # print("data size: {:d}".format(len(self._buffer) - meta_size))

        if len(self._buffer) < meta_size:
            raise ValueError("Buffer too small to contain metadata.")

        self.metadata = N2Metadata.from_buffer(self._buffer[:meta_size])

        self._set_data_arrays()

    def _set_data_arrays(self):

        _data = self._buffer[ctypes.sizeof(N2Metadata) :]

        if (
            self._num_elements is not None
            and self._num_ev is not None
            and self._num_prod is not None
        ):
            num_elements = self._num_elements
            num_prod = self._num_prod
            num_ev = self._num_ev
        else:
            num_elements = self.metadata.num_elements
            num_prod = self.metadata.num_prod
            num_ev = self.metadata.num_ev

        layout = self.__class__.calculate_layout(
            num_elements, num_prod, num_ev, self.support_mode
        )

        if layout["size"] != len(_data):
            raise RuntimeError(
                "Received buffer length {0:d} (Total {1:d} - Metadata {2:d})".format(
                    len(_data), len(self._buffer), ctypes.sizeof(N2Metadata)
                )
                + " does not match expected size {0:d}".format(layout["size"])
                + " from num_elements {0:d} num_ev {1:d} num_prod {2:d}".format(
                    num_elements, num_ev, num_prod
                )
            )

        # Scalar frames use metadata.n_valid_fpga_ticks instead of this array.
        self.valid_fpga_ticks = np.empty(0, dtype=np.uint64)
        for member in layout["members"]:

            arr = np.frombuffer(
                _data[member["start"] : member["end"]], dtype=member["dtype"]
            )
            setattr(self, member["name"], arr)

    @classmethod
    def calculate_layout(cls, num_elements, num_prod, num_ev, support_mode="scalar"):
        """Calculate the buffer layout.

        Parameters
        ----------
        num_elements, num_prod, num_ev : int
            Length of each dimension.
        support_mode : {"scalar", "per_product_v1"}, optional
            Whether to append per-product counts. Default: "scalar".

        Returns
        -------
        layout : dict
            Structure of buffer.
        """

        if support_mode not in ("scalar", "per_product_v1"):
            raise ValueError(f"Unknown N2 support_mode: {support_mode}")
        if support_mode == "per_product_v1" and num_prod <= 0:
            raise ValueError("per_product_v1 requires at least one product")
        structure = [
            ("vis", np.complex64, num_prod),
            ("weight", np.float32, num_prod),
            ("flags", np.float32, num_elements),
            ("eval", np.float32, num_ev),
            ("evec", np.complex64, num_ev * num_elements),
            ("emethod", np.int32, 1),
            ("erms", np.float32, 1),
            ("radiometer_chi2", np.float32, 3),
            ("gain", np.complex64, num_elements),
            ("mask", np.uint8, num_elements),
        ]

        if support_mode == "per_product_v1":
            structure.append(("valid_fpga_ticks", np.uint64, num_prod))

        end = 0

        members = []
        maxalign = 0

        for name, dtype, num in structure:

            member = {}

            size = np.dtype(dtype).itemsize

            align = size // 2 if dtype in (np.complex64, np.complex128) else size

            # Update the maximum alignment
            maxalign = align if maxalign < align else maxalign

            member["start"] = _offset(end, align)
            end = member["start"] + num * size
            member["end"] = end
            member["size"] = num * size

            # make sure this dimension doesn't get squashed out if it's 1 (for everything but erms)
            if name == "erms":
                member["num"] = num
            else:
                member["num"] = (num,)

            member["dtype"] = dtype
            member["name"] = name

            members.append(member)

        struct_end = _offset(members[-1]["end"], maxalign)
        layout = {"size": struct_end, "members": members}

        return layout

    @classmethod
    def from_file(
        cls,
        filename,
        num_elements=None,
        num_prod=None,
        num_ev=None,
        support_mode="scalar",
    ):
        """Load a dump using the supplied dimensions and support mode."""
        filesize = os.path.getsize(filename)

        buf = bytearray(filesize)

        with io.FileIO(filename, "rb") as fh:
            fh.readinto(buf)

        return cls(
            buf,
            num_elements=num_elements,
            num_prod=num_prod,
            num_ev=num_ev,
            support_mode=support_mode,
        )

    @classmethod
    def load_files(
        cls,
        pattern,
        num_elements=None,
        num_prod=None,
        num_ev=None,
        support_mode="scalar",
    ):
        """Read a set of dump files as N2Buffers.

        Parameters
        ----------
        pattern : str
            A glob pattern to read.
        support_mode : {"scalar", "per_product_v1"}, optional
            Support mode shared by all files. Default: "scalar".

        Returns
        -------
        buffers : list of N2Buffers
        """
        import glob

        return [
            cls.from_file(
                fname,
                num_elements=num_elements,
                num_prod=num_prod,
                num_ev=num_ev,
                support_mode=support_mode,
            )
            for fname in sorted(glob.glob(pattern))
        ]

    @classmethod
    def to_files(cls, buffers, basename):
        """Write a list of buffers to disk.

        Parameters
        ----------
        buffers : list of N2Buffers
            Buffers to write.
        basename : str
            Basename for filenames.
        """
        pat = basename + "_%07d.dump"

        msize_c = ctypes.c_int(ctypes.sizeof(N2Metadata))

        for ii, buf in enumerate(buffers):

            with open(pat % ii, "wb+") as fh:
                fh.write(msize_c)
                fh.write(bytearray(buf._buffer))

    @classmethod
    def new_from_params(
        cls, num_elements, num_prod, num_ev, insert_size=True, support_mode="scalar"
    ):
        """Create an N2Buffer with its own zeroed memory.

        Parameters
        ----------
        num_elements, num_prod, num_ev : int
            Length of each dimension.
        support_mode : {"scalar", "per_product_v1"}, optional
            Whether to include per-product counts. Default: "scalar".

        Returns
        -------
        buffer : N2Buffer
        """

        layout = cls.calculate_layout(num_elements, num_prod, num_ev, support_mode)
        meta_size = ctypes.sizeof(N2Metadata)

        buf = np.zeros(meta_size + layout["size"], dtype=np.uint8)

        # N2Metadata does not store dimensions, so pass them to the view.
        return cls(
            buf,
            skip=0,
            num_elements=num_elements,
            num_prod=num_prod,
            num_ev=num_ev,
            support_mode=support_mode,
        )


def _offset(offset, align):
    """Calculate the start of a member with alignment `align` after `offset` within a
    struct."""
    return ((align - (offset % align)) % align) + offset
