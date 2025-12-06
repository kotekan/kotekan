.. _hdf5N2Write:

***********
hdf5N2Write
***********

Buffered writer for CHORD/CHIME N2 visibilities.  Frames arrive from an
``N2Buffer`` with ``N2Metadata``; the stage groups them by
``abs_time_idx / num_file_t``, buffers a full ``nfreq × nt`` block in memory,
and writes HDF5 files in one shot.  Missing frames stay zero-filled with
``frames_added`` showing which (f, t) pairs were present and ``frac_lost=1`` for
those gaps.  Work happens in ``<base_dir>/.partial/vis_<abs_idx>.h5`` and is
renamed to ``vis_<abs_idx>_<YYYYMMDDThhmmss>_<nsec>.h5`` using the earliest
``fpga_start_tick`` in the file.  If a finalized file already exists for a
window, late frames are dropped.  Files are finalized when full, after
``late_frame_grace_seconds`` of inactivity, or on shutdown.

.. doxygenclass:: hdf5N2Write
   :members:
   :undoc-members:

File layout
-----------

- Attributes: version (``CHORD_0.0``), ``file_mode`` (``CHORD``; ``CHIME`` moves
  flag-related datasets under ``/flags``), ``abs_file_idx``, ``num_file_t``,
  ``num_elements``, ``num_prod``, ``num_ev``, ``num_freq``, ``vis_layout``,
  telescope geometry (origin, orientations, dish maps), EOP tables, and
  ``num_file_f``.
- Index maps: ``/index_map/freq`` (MHz and width for each file frequency),
  ``/index_map/prod``, ``/index_map/grid_x_idx``, ``/index_map/grid_y_idx``,
  ``/index_map/feed_pos_disp_m``, ``/index_map/coelev_disp_deg``,
  ``/index_map/type``, ``/index_map/dish_positions_in_grid_coords``.
- Per-(f, p, t)/(f, t) datasets: ``/vis``, ``/vis_weight``, ``/eval``, ``/evec``,
  ``/erms``, ``/gain``, ``/flags``, ``/frac_lost``, ``/frac_rfi``,
  ``/frames_added``.
- Per-time metadata: ``/fpga_start_tick``, ``/frame_length_fpga_ticks``,
  ``/time_center_ut1_ns``, ``/bin_ut1_ns``, ``/bin_start_ERA_deg``,
  ``/bin_end_ERA_deg``, ``/bin_start_LAST``, ``/bin_end_LAST``.
- ``/config_json`` grows on flush with snapshots from ``configTracker``.

Chunking and compression
------------------------

- Chunk caps: ``blocksize_f`` (frequency), ``blocksize_p`` (product/element),
  ``blocksize_t`` (time).  Zero leaves the dimension size unchanged for chunking.
- Compression: ``compression="none"`` (default) or ``"deflate"`` (zlib; default
  level 4 when ``compression_level=0``).  ``use_bitshuffle=true`` adds the
  bitshuffle filter with backend ``compression`` (``"none"``, ``"zstd"``,
  ``"lz4"``); ``compression_level=0`` maps to level 9.  The required HDF5
  plugins must be available at runtime.

Configuration
-------------

``in_buf`` (string, required)
    N2 buffer supplying frames (``buffer_type`` must be ``"N2"``).
``base_dir`` (string, required)
    Output directory; ``<base_dir>`` and ``<base_dir>/.partial`` are created.
``num_file_t`` (uint, required)
    Number of time slots per file (``t_index = abs_time_idx % num_file_t``).
``blocksize_f`` (uint, default ``16``)
    Chunk cap for the frequency dimension.
``blocksize_p`` (uint, default ``16``)
    Chunk cap for product/element dimensions.
``blocksize_t`` (uint, default ``num_file_t``)
    Chunk cap for the time dimension.
``compression`` (string, default ``"none"``)
    ``"none"`` | ``"deflate"`` | ``"zstd"`` | ``"lz4"`` (backends when using
    bitshuffle).
``compression_level`` (uint, default ``0``)
    Codec level; ``0`` picks a stage default (4 for deflate, 9 for bitshuffle).
``use_bitshuffle`` (bool, default ``false``)
    Enable bitshuffle plus the selected backend codec.
``late_frame_grace_seconds`` (uint, default ``60``)
    Inactivity threshold before finalizing a partial file.
``max_frames`` (int, default ``-1``)
    Stop after this many frames; ``-1`` disables the limit.

Metrics
-------

- ``kotekan_hdf5N2Write_write_time_seconds``
- ``kotekan_hdf5N2Write_n_datasets``
- ``kotekan_hdf5N2Write_open_file_info`` / ``open_file_age_seconds`` /
  ``file_completion_fraction``
- ``kotekan_hdf5N2Write_add_frame_errors_total`` and
  ``last_add_frame_error_seconds`` (``reason``, ``abs_file_idx``, ``freq_id``,
  ``t_index``)
- ``kotekan_hdf5N2Write_finalize_failures_total``
- ``kotekan_hdf5N2Write_unfinalized_file``

Example
-------

.. code-block:: yaml

    hdf5_vis_write:
        kotekan_stage: hdf5N2Write
        in_buf: n2_merge_buffer
        base_dir: ./vis_data
        num_file_t: 10
        blocksize_f: 4
        blocksize_t: 5
        compression: deflate
        compression_level: 4
        late_frame_grace_seconds: 30

See ``tests/boost/test_hdf5N2Write.cpp`` for end-to-end expectations and file
layout checks.
