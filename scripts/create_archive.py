"""Functions for transposing datasets of archive files.

The final step of creating the data archive files is to transpose the time axis
such that it is the fastest varying. This module performs this transpose.

"""

import click
import os
import sys
import subprocess
import tempfile
import yaml

# Maximum increase in noise (variance) from numerical truncation,
# with factor of 3 from uniform distribution of errors.
ERR_SQ_LIM = 3e-3
DATA_FIXED_PREC = 1e-4
WEIGHT_FIXED_PREC = 1e-3


def create_archive(infile, outfile, chunk, log_level='info',
                   buffer_depth=None, kotekan_dir=None, num_prod=None):
    """ Transform kotekan receiver raw output file into transposed and bitshuffle
        compressed archive file.
    """

    if buffer_depth is None:
        buffer_depth = chunk[0] * chunk[2] * 2

    # Base config
    # TODO: Should num_element be fixed?
    config = { 'log_level': log_level, 'num_elements': 2048, 'num_ev': 4,
               "num_local_freq": 1, "cpu_affinity": [] }

    # Buffers
    config.update( {
        'read_buffer': {
            'kotekan_buffer': 'vis',
            'metadata_pool': 'vis_pool',
            'num_frames': str(buffer_depth),
        },
        'trunc_buffer': {
            'kotekan_buffer': 'vis',
            'metadata_pool': 'vis_pool',
            'num_frames': str(buffer_depth),
        }
    })

    if num_prod is not None:
        config['read_buffer'].update({'num_prod': num_prod})
        config['trunc_buffer'].update({'num_prod': num_prod})

    # Metadata pool
    config.update({
        'vis_pool': {
            'kotekan_metadata_pool': 'visMetadata',
            'num_metadata_objects': str(20 * buffer_depth)
        }
    })

    # Reader process
    config.update( {
            'read_raw': {
                'kotekan_process': 'visRawReader',
                'infile': os.path.abspath(infile),
                'chunk_size': chunk,
                'out_buf': 'read_buffer',
                'readahead_blocks': 4
            }
        }
    )

    # Truncate process
    config.update( {
            'truncate': {
                'kotekan_process': 'visTruncate',
                'err_sq_lim': ERR_SQ_LIM,
                'data_fixed_precision': DATA_FIXED_PREC,
                'weight_fixed_precision': WEIGHT_FIXED_PREC,
                'in_buf': 'read_buffer',
                'out_buf': 'trunc_buffer'
            }
        }
    )

    # Transpose/write process
    config.update( {
            'transpose': {
                'kotekan_process': 'visTranspose',
                'in_buf': 'trunc_buffer',
                'chunk_size': chunk,
                'infile': os.path.abspath(infile),
                'outfile': os.path.abspath(outfile)
            }
        }
    )

    # select kotekan executable
    if kotekan_dir:
        kotekan_cmd = "./kotekan"
    else:
        kotekan_dir = "./"
        kotekan_cmd = "kotekan"

    with tempfile.NamedTemporaryFile() as fh:
        yaml.dump(config, fh)
        fh.flush()
        print config
        if kotekan_cmd == "./kotekan":
            subprocess.check_call([kotekan_cmd, "-c", fh.name],
                                  cwd=kotekan_dir)
        else:
            subprocess.check_call([kotekan_cmd, "-d", fh.name],
                                  cwd=kotekan_dir)


class MalformedData(Exception):
    """Raised when datasets have the wrong shape."""


class EmptyFile(Exception):
  """Raised when the file is empty."""


class CannotOpenFile(Exception):
    "Raised when unable to open a file."""
