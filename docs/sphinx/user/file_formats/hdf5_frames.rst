.. _hdf5_frames_format:

=====================================
Per-frame HDF5 dumps (hdf5FileWrite)
=====================================

The ``hdf5FileWrite`` stage (``lib/stages/hdf5FileWrite.cpp``) streams any
kotekan buffer to HDF5. It is the workhorse dump stage of the F-engine and
test configurations. The file layout depends on the metadata type of the
input buffer:

* **CHORD metadata** (n-dimensional GPU-pipeline buffers): one extensible
  dataset per file, frames appended along the first axis;
* **N2 metadata** (visibility frames): one file per frame holding a fixed
  set of datasets and attributes.

.. note::
   This page documents the on-disk format produced by
   ``lib/stages/hdf5FileWrite.cpp``; update it when changing the datasets or
   attributes written there.

File naming
===========

Files are written to ``base_dir`` (created if missing) as::

    <base_dir>/[<hostname>_][x<rank:04d>_]<file_name>.<frame:08d>.h5

* ``<hostname>_`` is included when ``prefix_hostname`` is true (default);
* ``x<rank:04d>_`` is included when ``prefix_host_rank`` is true (rank from
  ``frequency_pool_rank``), used when several writers split a frequency
  range;
* ``<frame:08d>`` is an 8-digit frame counter, one file per frame. With
  ``create_single_file: true`` the counter is omitted and all frames go to
  one file.

Files are created in HDF5 v1.10 format with **SWMR** (single-writer
multiple-reader) write mode: they remain readable while being written, and
stay readable even if the writer did not close them cleanly (you may need
``h5clear -s FILENAME.h5`` to clear the writer flag after a crash).

CHORD-metadata layout
=====================

Each file contains a single dataset named after the ``file_name``
configuration value. Its rank and dimension names come from the buffer's
ndarray description; the first dimension is unlimited and grows as frames
are appended (one frame's extent per write; in per-frame mode each file
holds exactly one frame). The element type is the buffer's data type mapped
to HDF5.

The dataset is chunked (full frame extent, with the first dimension capped
so a chunk stays around 8 Mi elements) and compressed with the *bitshuffle*
filter (id 32008) with zstd backend at level 9.

Dataset attributes:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Attribute
     - Description
   * - ``telescope_name``
     - Telescope name from the running config.
   * - ``seq_length_nsec``
     - Duration of one FPGA tick, ns.
   * - ``gps_time_enabled``
     - Whether the FPGA timestamp is GPS-disciplined.
   * - ``chord_metadata_version``
     - int[2] metadata format version (currently {2, 0}).
   * - ``name``
     - The ndarray name from the buffer metadata.
   * - ``type``
     - Element data type as a string (e.g. ``float16``).
   * - ``dim_names``
     - String array of dimension names (e.g. ``["T", "F", "D", "P"]``).
   * - ``dim_scalings``
     - int64 array of per-dimension scaling factors: how many elements of
       the underlying, un-decimated axis one element along that dimension
       represents (1 for an axis that is neither decimated nor combined).
   * - ``fpga_seq_num``, ``fpga_seq_time_nsec``
     - First FPGA sequence number of the frame and its instrument time in
       ns (only if the metadata carries a sequence number).
   * - ``time_downsampling_fpga``
     - Total FPGA-tick downsampling factor of the time axis (if present).
   * - ``coarse_freq``, ``freq_upchan_factor``, ``freq_upchan_index``
     - Per-channel coarse frequency ids, upchannelization factors and
       indices (if present).
   * - ``rfi_frame_excision_enabled``, ``rfi_frame_excision_thresholds``
     - RFI frame-excision state (if present).
   * - ``num_polarizations``, ``num_dishes``
     - Telescope input counts (from the stage configuration).
   * - ``itrs_lat_deg``, ``itrs_lon_deg``
     - Telescope origin ITRS coordinates, degrees.
   * - ``grid_orientation``
     - Row-major 3x3 rotation, topocentric to grid frame.
   * - ``grid_size_x``, ``grid_size_y``
     - Dish grid dimensions.
   * - ``feed_separation_x_m``, ``feed_separation_y_m``
     - Grid spacing, metres.
   * - ``dish_grid_indices``
     - Per-dish (x, y) grid indices, int64 with shape
       (``num_dishes``, 2); (-1, -1) off the main array.
   * - ``feed_positions_m``
     - Per-element 3D feed positions in the grid frame, metres.

N2-metadata layout
==================

One file per frame (i.e. per frequency channel per accumulation bin). All
quantities refer to a single frequency and time bin; see
:ref:`n2_vis_file_format` for the semantics of the fields. Complex
quantities are stored as float32 with a trailing axis of length 2 (real,
imaginary) --- *not* as an HDF5 compound type.

Datasets (all bitshuffle+zstd compressed unless an extent is zero):

.. list-table::
   :header-rows: 1
   :widths: 24 30 14 32

   * - Dataset
     - Shape
     - Type
     - Description
   * - ``vis``
     - (:math:`N_p`, 2)
     - float32
     - Visibilities, upper-triangle product order, re/im.
   * - ``weight``
     - (:math:`N_p`)
     - float32
     - Inverse-variance visibility weights.
   * - ``flags``
     - (:math:`N_e`)
     - float32
     - Per-input flags.
   * - ``eval``
     - (:math:`N_{ev}`)
     - float32
     - Eigenvalues.
   * - ``evec``
     - (:math:`N_{ev}`, :math:`N_e`, 2)
     - float32
     - Eigenvectors, re/im.
   * - ``emethod``
     - (1)
     - int32
     - Eigen-decomposition method enum.
   * - ``erms``
     - (1)
     - float32
     - RMS of residual visibilities.
   * - ``gain``
     - (:math:`N_e`, 2)
     - float32
     - Applied per-input gains, re/im.
   * - ``radiometer_chi2``
     - (3)
     - float32
     - Radiometer test statistic per polarization pair (XX, XY, YY).

Root attributes: ``num_polarizations``, ``num_dishes``, ``num_elements``,
``num_prod``, ``num_ev``, ``input_order``, ``freq_id``, ``freq_MHz``,
``abs_time_idx``; the EOP at the data midpoint and at the nominal bin centre
as dotted scalar attributes (``time_center_eop.t_inst_ns``,
``time_center_eop.t_ut1_ns``, ``time_center_eop.delta_UT1_inst``,
``time_center_eop.ERA_deg``, ``time_center_eop.xp_as``,
``time_center_eop.yp_as``, and the same six under ``bin_eop.``);
``bin_start_ERA_deg``, ``bin_end_ERA_deg``, ``bin_start_ERAL_deg``,
``bin_end_ERAL_deg``; ``fpga_start_tick``, ``frame_start_time_ns``,
``frame_length_fpga_ticks``, ``n_valid_fpga_ticks``, ``n_rfi_fpga_ticks``,
``n_rfi_only_fpga_ticks``, ``n_pl_fpga_ticks``;
``rfi_frame_excision_enabled``, ``rfi_frame_excision_num``,
``rfi_frame_excision_threshold``, ``rfi_frame_excision_fraction``; and the
telescope geometry set (``itrs_lat_deg``, ``itrs_lon_deg``,
``grid_orientation``, ``grid_size_x``, ``grid_size_y``,
``feed_separation_x_m``, ``feed_separation_y_m``,
``main_array_grid_indices``, ``feed_positions_m``).

Configuration
=============

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Description
   * - ``base_dir``
     - required
     - Output directory (created if missing).
   * - ``file_name``
     - required
     - Base file name; also the dataset name in CHORD mode.
   * - ``prefix_hostname``
     - true
     - Prepend the hostname to file names.
   * - ``prefix_host_rank``
     - false
     - Prepend ``x<rank:04d>_`` (rank from ``frequency_pool_rank``).
   * - ``frequency_pool_rank``, ``frequency_pool_size``
     - 0, 1
     - This writer's rank/count when splitting work across writers.
   * - ``create_single_file``
     - false
     - Append all frames to one file (CHORD mode only) instead of one file
       per frame.
   * - ``use_compression``
     - true
     - Compress the data with bitshuffle+zstd (filter 32008).
   * - ``write_x_frames``, ``per_y_frames``
     - -1
     - Decimation: write only the first X out of every Y frames.
   * - ``max_frames``
     - -1
     - Stop (and shut kotekan down) after this many frames.
   * - ``skip_writing``
     - false
     - Consume frames without writing (for benchmarking).
   * - ``num_polarizations``, ``num_dishes``
     - required
     - Telescope input counts recorded in the attributes.

Reading the files requires the bitshuffle HDF5 plugin with zstd support
(``import hdf5plugin`` in Python).

Reading the files back
======================

CHORD-metadata files can be replayed into a kotekan buffer with the
``hdf5FileRead`` stage (``lib/stages/hdf5FileRead.cpp``). It reconstructs the
payload and the chord metadata from the dataset and its attributes, and aborts
if the telescope attributes disagree with the telescope of the replaying
session. The destination buffer must be declared with ``kotekan_buffer:
ndarray``; the reader validates the declared frame descriptor against the file.

* **Per-frame files**: point ``input_dir``/``file_name`` (and
  ``prefix_hostname``/``prefix_host_rank``) at the files written above. The
  reader starts at frame index 0 and stops when the next file is missing; a
  missing file at index 0 is fatal.
* **Single file**: additionally set ``read_single_file: true``. The reader
  splits axis 0 of the dataset into frames.

Because a single file stores the attributes only once (taken from the first
frame), some per-frame information is *not* preserved: ``frame_counter``,
``first_packet_recv_time``, the per-frame ``fpga_seq_num``, and the frame
boundary itself. The replay therefore reconstructs them:

* the axis-0 extent of one frame is the first extent declared for the
  reader's ``out_buf`` (the remaining extents, the element type and the
  labels must match the file);
* ``fpga_seq_num`` of frame *i* is ``fpga_seq_num + i * extents[0] *
  time_downsampling_fpga`` (a downsampling factor of 1 is assumed when the
  attribute is absent);
* the number of frames is ``axis0 / extents[0]``; a remainder is reported
  and ignored.

The writer must have finished before the reader starts, so the two have to run
in separate kotekan sessions: the whole single file, or the first per-frame
file, has to exist and be complete at startup. After a writer crash the file
may need ``h5clear -s FILENAME.h5`` first.

``hdf5FileReadSingleFile`` (``lib/stages/hdf5FileReadSingleFile.cpp``) is a
different, special-purpose stage: it replays only CHIME/F-engine baseband
voltages (rank-4 ``int4x2_swapped_withoffset`` data with a leading ``T`` axis
and an ``F`` axis), selecting a subset of the frequency channels and optionally
transposing time and frequency or combining dishes and polarizations. Use
``hdf5FileRead`` with ``read_single_file`` for everything else.
