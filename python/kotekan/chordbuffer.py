"""Read a ChordBuffer dump into python.
"""
from pathlib import Path
import numpy as np
import h5py as h5


class ChordBuffer(object):
    """Python representation of an ChordBuffer (a buffer with ChordMetadata).

    Access the data through data`, a numpy array.

    Parameters
    ----------
    data : numpy ndarray
        The ndarray data
    meta : dict
        The metadata parameters as a dict
    dset_name : str
        Name of this data in an HDF5 file
    """

    def __init__(self, data, meta):

        if not isinstance(data, np.ndarray):
            raise ValueError("data not an ndarray")

        if not isinstance(meta, dict):
            raise ValueError("meta not a dict")

        self.data = data
        self.metadata = meta

    @classmethod
    def from_file(cls, filename, dset_name):
        """Load a ChordBuffer from a kotekan hdf5 file."""

        with h5.File(filename, "r") as f:

            if dset_name not in f:
                raise RuntimeError(
                    "dataset {} is not in file {}".format(dset_name, filename)
                )

            dset = f[dset_name]
            data = dset[...]
            meta = {key: val for key, val in dset.attrs.items()}

        return cls(data, meta)

    @classmethod
    def load_files(cls, pattern, dset_name=None):
        """Read a set of hdf5 files as ChordBuffers.

        Parameters
        ----------
        pattern : str
            A globable pattern to read.
        dset_name : str, optional
            The dataset name in the HDF5 files.

        Returns
        -------
        buffers : list of ChordBuffers
        """
        import glob

        bufs = []

        for filename in sorted(glob.glob(pattern)):
            filepath = Path(filename)

            if dset_name is not None:
                my_dset_name = dset_name
            else:
                # If dset_name is not given, assume it is the filename,
                # which may have a trailing file index. e.g. <filename>.<index>.h5
                # This will fail if <filename> includes '.'
                my_dset_name = filepath.stem.split(".")[0]
            bufs.append(cls.from_file(filename, my_dset_name))

        return bufs

    @classmethod
    def to_files(cls, buffers, basename, dset_name=None):
        """Write a list of buffers to disk.

        Parameters
        ----------
        buffers : list of ChordBuffers
            Buffers to write.
        basename : str
            Basename for filenames.
        """
        pat = basename + ".%08d.h5"

        for ii, buf in enumerate(buffers):

            filename = Path(pat % ii)

            dset_name_ii = (
                dset_name if dset_name is not None else filename.stem.split(".")[0]
            )
            print("Writing dset {} to {}".format(dset_name_ii, pat % ii))

            with h5.File(pat % ii, "a") as f:

                dset = f.create_dataset(dset_name_ii, data=buf.data)
                for key, val in buf.metadata.items():
                    dset.attrs[key] = val


def get_metadata(name, type_name, dim_names):
    metadata = dict(
        chord_metadata_version=np.array([1, 0], dtype=np.int32),
        name=name,
        type=type_name,
        dim_names=dim_names,
    )
    return metadata
