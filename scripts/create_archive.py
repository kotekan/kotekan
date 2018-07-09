#!/usr/bin/env python
import click
import os
import sys
import subprocess
import tempfile
import yaml

# (freq, prod, time)
DEFAULT_CHUNK = (16,16,16)
ERR_SQ_LIM = 3e-3
DATA_FIXED_PREC = 1e-4
WEIGHT_FIXED_PREC = 1e-3

@click.command()
@click.option("--log-level", default='info', help="default: info")
@click.option("--chunk", nargs=3, type=int, default=DEFAULT_CHUNK,
              help="[freq prod time] chunk. default: 16 16 16")
@click.option("--buffer-depth", type=int, default=None, help="Specify buffer depth.")
@click.option("--system-kotekan", type=bool, default=False,
              help="Use the kotekan executable in the system path.")
@click.argument("infile")
@click.argument("outfile")
def create_archive(infile, outfile, log_level, chunk, buffer_depth, system_kotekan):
    """ Transform kotekan receiver raw output file into transposed and bitshuffle
        compressed archive file.
    """

    if buffer_depth is None:
        buffer_depth = chunk[0] * chunk[2]

    # Base config
    # TODO: Should num_element be fixed?
    config = { 'log_level': log_level, 'num_elements': 2048, "num_local_freq": 1, "cpu_affinity": [] }

    # Buffers
    config.update( {
        'read_buffer': {
            'kotekan_buffer': 'standard',
            'metadata_pool': 'vis_pool',
            'num_frames': str(buffer_depth),
            'sizeof_int': 4,
            'frame_size': '2 * sizeof_int * num_local_freq * num_elements * num_elements'
        },
        'trunc_buffer': {
            'kotekan_buffer': 'standard',
            'metadata_pool': 'vis_pool',
            'num_frames': str(buffer_depth),
            'sizeof_int': 4,
            'frame_size': '2 * sizeof_int * num_local_freq * num_elements * num_elements'
        }
    })

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

    # emulate KotekanRunner
    if system_kotekan:
        kotekan_dir = "./"
        kotekan_cmd = "kotekan"
    else:
        kotekan_dir = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                                    "..", "build", "kotekan"))
        kotekan_cmd = "./kotekan"

    with tempfile.NamedTemporaryFile() as fh:
        yaml.dump(config, fh)
        fh.flush()
        print config
        subprocess.check_call([kotekan_cmd, "-c", fh.name],
                              cwd=kotekan_dir)


if __name__ == '__main__':
    create_archive()
