#!/usr/bin/env python
import click
import os
import sys

# TODO: set up proper package
sys.path.append("../tests/")
from kotekan_runner import KotekanRunner

# (time, freq, prod) TODO: should maybe change this order
DEFAULT_CHUNK = (16,16,16)
ERR_SQ_LIM = 3. / 1000.
FIXED_PREC = 1. / 1000.

@click.command()
@click.option("--log-level", default='info')
@click.argument("infile")
@click.argument("outfile")
def create_archive(infile, outfile, log_level):

    bufs = {
        'read_buffer': {
            'kotekan_buffer': 'standard',
            'metadata_pool': 'vis_pool',
            'num_frames': 'buffer_depth',
            'sizeof_int': 4,
            'frame_size': '2 * sizeof_int * num_local_freq * num_elements * num_elements'
        },
        'trunc_buffer': {
            'kotekan_buffer': 'standard',
            'metadata_pool': 'vis_pool',
            'num_frames': 'buffer_depth',
            'sizeof_int': 4,
            'frame_size': '2 * sizeof_int * num_local_freq * num_elements * num_elements'
        }
    }

    # TODO: adapt to subsets
    config = { 'num_elements': 2048, 'num_prod': 2098176, 'log_level': log_level }

    proc = {}
    # Reader process
    proc.update( {
            'read_raw': {
                'kotekan_process': 'visRawReader',
                'filename': os.path.abspath(infile),
                'chunk_size': DEFAULT_CHUNK,
                'out_buf': 'read_buffer'
            }
        }
    )

    # Truncate process
    proc.update( {
            'truncate': {
                'kotekan_process': 'visTruncate',
                'err_sq_lim': ERR_SQ_LIM,
                'fixed_precision': FIXED_PREC,
                'in_buf': 'read_buffer',
                'out_buf': 'trunc_buffer'
            }
        }
    )

    # Transpose/write process
    proc.update( {
            'transpose': {
                'kotekan_process': 'visTranspose',
                'in_buf': 'trunc_buffer',
                'chunk_dim_time': DEFAULT_CHUNK[0],
                'chunk_dim_freq': DEFAULT_CHUNK[1],
                'md_filename': os.path.abspath(infile) + '.meta',
                'filename': os.path.abspath(outfile)
            }
        }
    )

    runner = KotekanRunner(buffers=bufs, processes=proc, config=config)
    runner.run()

if __name__ == '__main__':
    create_archive()
