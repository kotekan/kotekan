.. _asdf_zarr_frames_format:

========================================================
ASDF and Zarr frame dumps (asdfFileWrite, gdalFileWrite)
========================================================

Two sibling stages stream kotekan buffers to self-describing scientific
formats, one file (or directory) per frame:

* ``asdfFileWrite`` (``lib/stages/asdfFileWrite.cpp``) writes `ASDF
  <https://asdf-standard.readthedocs.io/>`_ files and accepts CHORD or N2
  metadata;
* ``gdalFileWrite`` (``lib/stages/gdalFileWrite.cpp``) writes Zarr v3
  stores through GDAL's multidimensional API and accepts CHORD metadata
  only.

.. note::
   This page documents the on-disk formats produced by
   ``lib/stages/asdfFileWrite.cpp`` and ``lib/stages/gdalFileWrite.cpp``;
   update it when changing the entries or attributes written there.

Both share the naming scheme (``base_dir`` is created if missing)::

    <base_dir>/[<hostname>_]<file_name>.<frame:08d>.asdf    (ASDF)
    <base_dir>/[<hostname>_]<file_name>.<frame:08d>.zarr    (Zarr; a directory)

with an 8-digit frame counter and the hostname prefix controlled by
``prefix_hostname`` (default true). Common configuration: ``base_dir``,
``file_name``, ``prefix_hostname``, ``max_frames`` (default -1 =
unlimited; on reaching it the stage shuts kotekan down), and
``skip_writing`` (consume without writing).

ASDF files (asdfFileWrite)
==========================

The ASDF tree contains one group keyed by the 8-digit frame number (e.g.
``"00000001"``). Array blocks are zlib-compressed (level 9). Within the
group, three entries are always present:

* ``telescope_name`` (string), ``seq_length_nsec`` (int, FPGA tick length
  in ns), ``gps_time_enabled`` (bool).

For **CHORD-metadata** buffers the group additionally contains:

* the frame's n-dimensional data array, keyed by the buffer name with the
  ``host_`` prefix and ``_buffer`` suffix stripped (e.g.
  ``host_upchan_buffer`` becomes ``upchan``), with the buffer's element
  type and dimensions;
* ``chord_metadata_version`` (int[2], currently {1, 0}); ``name`` and
  ``type`` (ndarray name and element type string); ``dim_names`` (string
  sequence);
* if present in the metadata: ``coarse_freq``, ``freq_upchan_factor``,
  ``freq_upchan_index`` (int sequences, one entry per frequency);
  ``fpga_seq_num`` and ``fpga_seq_time_nsec`` (first FPGA sequence number
  and its instrument time in ns); ``time_downsampling_fpga``.

For **N2-metadata** buffers (single-frequency visibility frames; see
:ref:`n2_vis_file_format` for field semantics) the group contains inline
arrays ``vis`` (complex64, :math:`N_p`), ``weights`` (float32,
:math:`N_p`), ``flags`` (float32, :math:`N_e`), ``eval`` (float32,
:math:`N_{ev}`), ``evec`` (complex64, :math:`N_{ev} \times N_e`), and
``gain`` (complex64; due to a bug at the time of writing this entry
contains a copy of the eigenvector data, not the gains), plus scalars
``emethod``, ``erms`` (stored as an integer), ``n_valid_fpga_ticks``,
``num_elements``, ``num_prod``, ``freq_id``.

Zarr stores (gdalFileWrite)
===========================

Each frame becomes a Zarr **v3** store (a directory) holding one
multidimensional array named after the buffer metadata's ndarray name, with
dimension names taken from the metadata. Arrays are compressed with Blosc
(zstd codec, level 9, bit-shuffle); chunks default to the frame extent,
with the first dimension split if a chunk would exceed 1 GB.

Root-group attributes mirror the ASDF entries: ``telescope_name``,
``seq_length_nsec``, ``gps_time_enabled`` (stored as int), ``name``,
``type``, ``chord_metadata_version``, and, when present in the metadata,
``coarse_freq``, ``freq_upchan_factor``, ``freq_upchan_index``,
``fpga_seq_num``, ``fpga_seq_time_nsec``, ``time_downsampling_fpga``.

The stage rejects N2-metadata buffers.
