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

    <base_dir>/[<capture start time>/][<hostname>_][x<rank:04d>_]<file_name>.<frame:08d>.h5

* ``<capture start time>/`` is included when ``timestamp_subdir`` is true (off by
  default). The sub-directory is created if missing and named after the UTC time of
  the first frame written, taken from the FPGA sequence number in its metadata, so a
  capture's files are grouped together without any extra configuration. The name is
  formatted with ``timestamp_subdir_format`` (default ``%Y%m%dT%H%M%SZ``, giving e.g.
  ``20260131T184500Z``); the time is rounded to the nearest second, because a capture
  scheduled on a whole second begins at the frame edge just before it;
* ``<hostname>_`` is included when ``prefix_hostname`` is true (default);
* ``x<rank:04d>_`` is included when ``prefix_host_rank`` is true (rank from
  ``frequency_pool_rank``), used when several writers split a frequency
  range;
* ``<frame:08d>`` is an 8-digit frame counter naming the first frame in the
  file. By default each file holds one frame; with ``frames_per_file: N``
  each file holds up to N consecutive frames (CHORD mode only). With
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
are appended (one frame's extent per write; a file holds one frame by
default, or up to ``frames_per_file`` frames). The element type is the
buffer's data type mapped to HDF5.

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
     - int[2] metadata format version (currently {1, 0}).
   * - ``name``
     - The ndarray name from the buffer metadata.
   * - ``type``
     - Element data type as a string (e.g. ``float16``).
   * - ``dim_names``
     - String array of dimension names (e.g. ``["T", "F", "D", "P"]``).
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
   * - ``num_polarizations``, ``num_dishes``, ``num_elements``
     - Telescope input counts (from the stage configuration).
   * - ``input_order``
     - Element ordering of the data (an ``ElementOrder`` name; see
       :ref:`element_order`).
   * - ``itrs_lat_deg``, ``itrs_lon_deg``
     - Telescope origin ITRS coordinates, degrees.
   * - ``grid_orientation``
     - Row-major 3x3 rotation, topocentric to grid frame.
   * - ``grid_size_x``, ``grid_size_y``
     - Dish grid dimensions.
   * - ``feed_separation_x_m``, ``feed_separation_y_m``
     - Grid spacing, metres.
   * - ``main_array_grid_indices``
     - Per-element (x, y) grid indices; (-1, -1) off the main array.
   * - ``feed_positions_m``
     - Per-element 3D feed positions in the grid frame, metres.

Because each node only writes the frequencies it processed, a full-band
dataset is spread over one set of files per node. ``python/scripts/
merge_baseband_freq.py`` stitches those back together: it checks that the
files describe the same observation and the same FPGA ticks, then writes a
single dataset whose frequency axis is the union of all of them (and whose
time axis concatenates each node's successive files). Run it with
``--dry-run`` to validate the inputs without writing, or ``--vds`` to build
an HDF5 virtual dataset that references the sources instead of copying
them.

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
   * - ``timestamp_subdir``
     - false
     - Write into a ``base_dir/<capture start time>`` sub-directory, named from the
       first frame's FPGA sequence number.
   * - ``timestamp_subdir_format``
     - ``%Y%m%dT%H%M%SZ``
     - ``strftime(3)`` format for that sub-directory name. Must not contain a
       path separator; whitespace is stripped.
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
   * - ``frames_per_file``
     - 1
     - Frames appended to each file before rolling over to a new one (CHORD
       mode only; mutually exclusive with ``create_single_file``).
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
   * - ``input_order``
     - required
     - Element ordering recorded in the attributes.

Reading the files requires the bitshuffle HDF5 plugin with zstd support
(``import hdf5plugin`` in Python).
