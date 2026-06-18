.. _n2_vis_file_format:

==========================================
N\ :sup:`2` visibility files (hdf5N2Write)
==========================================

This page describes the layout and contents of the HDF5 visibility files
produced by the kotekan :ref:`hdf5N2Write <hdf5N2Write>` stage: every dataset
and attribute, with its shape, HDF5 data type, axis labels, and meaning.
Array sizes are given symbolically; their values are set by the stage and
buffer configuration. For a worked example with concrete values, see the
deployment page below.

.. toctree::
    :maxdepth: 1

    n2_vis_hdf5_pathfinder

.. note::
   This page documents the on-disk format produced by
   ``lib/stages/hdf5N2Write.cpp``. If you add, remove, or change any dataset
   or attribute in that stage, update this page to match (see the note at the
   top of ``N2FileData::_open_or_create_file``).

Overview
========

The ``hdf5N2Write`` stage consumes single-frequency, fully accumulated
N\ :sup:`2` visibility frames (one frame per frequency channel per
accumulation bin, produced upstream by ``N2Accumulate``) and repacks them into
HDF5 files. Each file covers a fixed window of :math:`N_t` =
``num_file_t`` consecutive time bins across all :math:`N_f` science
frequency channels of the configured telescope. Within a file, arrays are
stored "transposed" with time as the *fastest-varying* (last) axis, i.e.
frequency-major, to allow large contiguous writes.

Files group :math:`N_t` consecutive accumulation bins by their absolute bin
index:

.. math::

   \texttt{abs_file_idx} = \left\lfloor \texttt{bin_abs_index} / N_t \right\rfloor,
   \qquad
   t\text{-slot in file} = \texttt{bin_abs_index} \bmod N_t .

The absolute bin index is assigned upstream by ``N2Accumulate`` and is
monotonic and safe across x/recv pipeline restarts. The index will reset
across f-engine restarts, however.
Depending on the accumulator configuration, bins
are aligned either to a fixed FPGA-tick grid or to a fixed
Earth-Rotation-Angle (ERA) grid (``bin_in_ERA: true``, with
``num_bins_per_rotation`` bins per Earth rotation and bin 0 starting at
ERA = 0 just before 2000 Jan 1 noon).

Directory layout and file naming
================================

At stage (receiver) startup, an *acquisition directory* is created under the
``base_dir``::

    <base_dir>/acq_YYYYMMDD_HHMMSS_NNNNNNNNN/

named from the UTC startup time (nanoseconds in the last field). Files being
filled live in a hidden staging directory
``<acq>/.partial/vis_<abs_file_idx>.h5`` and are renamed into the acquisition
directory once finalized. In the default ``CHORD`` file mode the final name
is::

    vis_<abs_file_idx:010d>_<YYYYMMDDT_HHMMSS>_<NNNNNNNNN>.h5

where the timestamp (UTC, with 9-digit nanosecond suffix) is the time of the
*earliest* FPGA tick contained in the file. (In the legacy ``CHIME`` file
mode the name is instead a 9-digit count of seconds since instrument start
plus a frequency suffix, ``NNNNNNNNN_0000.h5``.)

A file is finalized (flushed, closed, renamed) when (a) all
:math:`N_f \times N_t` (frequency, time) slots have been filled, (b) no new
frame has arrived for it for ``late_frame_grace_seconds`` while later data
flows, or (c) kotekan shuts down. Frames arriving after their window was
finalized are dropped. A file may therefore be *partially filled*: missing
(frequency, time) slots are zero-filled in all datasets, ``/frames_added`` is
0 there, and ``frac_lost`` stays at its default of 1.0 (see
`Completeness tracking and configuration snapshots`_).

Dimensions, conventions, and data types
=======================================

Dimension symbols
-----------------

.. list-table::
   :header-rows: 1
   :widths: 12 22 66

   * - Symbol
     - Axis label
     - Meaning
   * - :math:`N_f`
     - ``frequency``
     - science frequency channels (``num_file_f`` attribute; the telescope's
       science band)
   * - :math:`N_p`
     - ``product``
     - correlation products; :math:`N_e(N_e+1)/2` for the ``FullUpperTri``
       layout
   * - :math:`N_e`
     - ``element``
     - correlator inputs (typically 2 polarizations per dish)
   * - :math:`N_{ev}`
     - ``eigenval`` / ``eigenvec``
     - eigenvalue/eigenvector pairs (``num_ev``; may be 0)
   * - :math:`N_t`
     - ``time``
     - time bins per file (``num_file_t``)
   * - :math:`N_d`
     - ``dish``
     - dish slots in the telescope description (``num_dishes``)
   * - :math:`N_k`
     - ``threshold``
     - RFI excision threshold slots (``MAX_NUM_RFI_THRESHOLDS`` = 8)
   * - --
     - ``pol_product``
     - 3: polarization pairs (XX, XY, YY)
   * - --
     - ``xyz``
     - 3: Cartesian components

Conventions
-----------

**Axis labels.**
Every dataset created by the stage carries a string-array attribute ``axis``
naming its dimensions in order, e.g. ``/vis`` has
``axis = ["frequency", "product", "time"]``. (Some per-input tables under
``/index_map`` carry the axis label ``dish`` for historical reasons; see
below.)

**Frequency axis.**
Index :math:`f` along the ``frequency`` axis corresponds to FPGA channel
``freq_id`` :math:`= f_{\min} + f`, where :math:`f_{\min}` is the
telescope's first science channel. The physical centre frequency and width
of each channel are in ``/index_map/freq``. The telescope may have more
channels in total (``num_telescope_f``) than are written
(``num_file_f``).

**Product ordering (FullUpperTri).**
Products enumerate the upper triangle of the :math:`N_e \times N_e`
visibility matrix, including autocorrelations, row-major:
:math:`(0,0), (0,1), \dots, (0,N_e{-}1), (1,1), (1,2), \dots,
(N_e{-}1,N_e{-}1)`. The pair for product :math:`p` is stored in
``/index_map/prod`` as ``(input_a, input_b)`` with ``input_a <= input_b``;
``input_a`` is the row and ``input_b`` the column of the matrix entry
:math:`V_{ab}`. The lower triangle is the complex conjugate.

**Element ordering.**
The mapping from element index to physical input (dish, polarization) is
given by the ``input_order`` attribute; see :ref:`element_order` for the
defined orderings. Per-element grid positions are in the
``main_array_grid_indices`` and ``feed_positions_m`` attributes, in the
same element order.

**Time scales.**
Two time scales appear throughout: *instrument time* (``*_t_inst_ns``, also
``frame0_unix_ns``), nanoseconds since the UNIX epoch, defined by the
(ideally GPS-disciplined) FPGA clock; and *UT1* (``*_ut1_ns``), nanoseconds
since the J2000(UT1) epoch. Their difference is ``delta_UT1_inst``
(seconds). ERA is the Earth Rotation Angle in degrees :math:`[0, 360)`;
ERAL is the *local* ERA,
:math:`\mathrm{ERAL} = \mathrm{ERA} + \lambda_{\mathrm{ITRS}} + s'`, with
:math:`\lambda_{\mathrm{ITRS}}` the telescope's ITRS longitude. ``xp_as`` /
``yp_as`` are the polar motion components in arcseconds.

**FPGA ticks.**
One FPGA tick (sequence number increment) is one F-engine spectrum, of
duration ``fpga_seq_length_ns``. Tick 0 occurs at instrument time
``frame0_unix_ns``.

**Complex data.**
Complex datasets use an HDF5 compound type ``{r: float32, i: float32}``;
h5py reads them natively as ``complex64``.

**Sentinels.**
Quantities that no upstream stage populated hold sentinel values rather than
physical ones: ``erms`` = -1 or -FLT_MAX, ``gain`` = -1+0j,
``radiometer_chi2`` = -1, ``flags`` = 0, and ``eval`` / ``evec`` = 0 when no
eigensolver ran. Which of these are real in a given acquisition depends on
the deployed pipeline.

**CHIME file mode.**
With ``file_mode == CHIME`` (legacy), the flag-like datasets
(``vis_weight``, ``flags``, ``valid_fpga_count``, ``rfi_fpga_count``,
``rfi_only_fpga_count``, ``pl_fpga_count``, ``frac_lost``, ``frac_rfi``,
``frac_rfi_only``, ``frac_pl``) live under a ``/flags`` group instead of the
root. Everything below assumes the default ``CHORD`` mode (all datasets at
the root).

Root attributes
===============

All attributes are attached to the file's root group. Scalar HDF5 types are
given as NumPy-style names.

File identity and structure
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Attribute
     - Type
     - Description
   * - ``version``
     - string
     - File format version; ``"CHORD_0.0"``.
   * - ``file_mode``
     - string
     - ``"CHORD"`` or ``"CHIME"`` (see Conventions).
   * - ``abs_file_idx``
     - uint64
     - Absolute file index (= ``bin_abs_index`` / :math:`N_t`); also the
       first number in the file name.
   * - ``num_file_t``
     - uint64
     - Time bins per file, :math:`N_t`.
   * - ``num_file_f``
     - uint64
     - Science frequency channels in the file, :math:`N_f`.
   * - ``num_telescope_f``
     - uint64
     - Total telescope channels.
   * - ``num_elements``
     - uint32
     - Correlator inputs, :math:`N_e`.
   * - ``num_prod``
     - uint32
     - Correlation products, :math:`N_p`.
   * - ``num_ev``
     - uint32
     - Eigenpairs, :math:`N_{ev}`.
   * - ``n2_layout``
     - string
     - Visibility matrix packing: ``FullUpperTri``,
       ``RedundantBaselineAvg``, ``Autocorrelations``, ``InputANDMasked``,
       ``InputORMasked``, or ``GeneralSubset``.
   * - ``input_order``
     - string
     - Element ordering of the data (an ``ElementOrder`` name; see
       :ref:`element_order`).
   * - ``num_dishes``
     - uint64
     - Dish slots in the telescope description, :math:`N_d`.
   * - ``digital_gains_source_file``
     - string
     - Path of the gains file copied into ``/digital_gains`` (only present
       when gains are configured).
   * - ``digital_gains_source_file_modified``
     - string
     - Only present (value ``"true"``) if the gains file changed on disk
       between file creation and flush.

Telescope geometry and timing
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 16 54

   * - Attribute
     - Type
     - Description
   * - ``instrument_name``
     - string
     - Telescope name, e.g. ``"CHORDTelescope"``.
   * - ``nyquist_zone``
     - uint8
     - Sampling Nyquist zone.
   * - ``gps_time_enabled``
     - bool
     - True if the FPGA timestamp is GPS-disciplined (absolute times are
       trustworthy).
   * - ``frame0_unix_ns``
     - int64
     - Instrument time of FPGA tick 0, ns since the UNIX epoch.
   * - ``fpga_seq_length_ns``
     - uint64
     - Duration of one FPGA tick, ns.
   * - ``itrs_lat_deg``
     - float64
     - Telescope origin ITRS latitude, degrees.
   * - ``itrs_lon_deg``
     - float64
     - Telescope origin ITRS longitude, degrees.
   * - ``grid_orientation``
     - float64[9]
     - Row-major 3x3 rotation matrix from topocentric coordinates to the
       dish grid frame.
   * - ``dish_orientation``
     - float64[9]
     - Row-major 3x3 rotation matrix from topocentric coordinates to the
       dish pointing frame.
   * - ``grid_size_x``, ``grid_size_y``
     - uint64
     - Dish grid dimensions.
   * - ``feed_separation_x_m``, ``feed_separation_y_m``
     - float64
     - Grid spacing in metres.
   * - ``main_array_grid_indices``
     - int64[:math:`N_e`, 2]
     - Per-element (x, y) grid indices in element order; (-1, -1) for
       inputs not in the main array.
   * - ``feed_positions_m``
     - float64[:math:`N_e`, 3]
     - Per-element 3D feed positions in the grid frame, metres (includes
       per-feed displacements from the fiducial station position).
   * - ``dish_coelev_deg``
     - float64
     - Commanded dish co-elevation, degrees.

.. note::
   Files written by older kotekan builds may lack some of these attributes
   or use older names: ``itrs_lat_deg`` / ``itrs_lon_deg`` were previously
   ``origin_itrs_lat_deg`` / ``origin_itrs_lon_deg``, and ``input_order``,
   ``grid_size_x``, ``grid_size_y``, ``feed_separation_x_m``,
   ``feed_separation_y_m``, ``main_array_grid_indices``, and
   ``feed_positions_m`` were added later.

Earth orientation parameter (EOP) table
---------------------------------------

Six parallel array attributes store the EOP table that was loaded when the
file was created, one entry per tabulated epoch (typically daily entries
spanning the acquisition). All have the same length :math:`n`. They are
omitted entirely if the telescope has no EOP table loaded.

.. list-table::
   :header-rows: 1
   :widths: 30 14 56

   * - Attribute
     - Type
     - Description
   * - ``EOP_t_inst_ns``
     - int64[n]
     - Epochs, instrument time (ns, UNIX epoch).
   * - ``EOP_t_ut1_ns``
     - int64[n]
     - Epochs, UT1 (ns since J2000(UT1)).
   * - ``EOP_delta_UT1_inst``
     - float64[n]
     - UT1 - instrument time, seconds.
   * - ``EOP_ERA_deg``
     - float64[n]
     - Earth Rotation Angle, degrees.
   * - ``EOP_xp_as``
     - float64[n]
     - Polar motion :math:`x_p`, arcseconds.
   * - ``EOP_yp_as``
     - float64[n]
     - Polar motion :math:`y_p`, arcseconds.

Index maps (``/index_map``)
===========================

Static lookup tables describing the axes. The input tables (``grid_x_idx``
through ``label`` below) are copied verbatim from the telescope's
``dish_inputs`` configuration, which carries one entry per correlator input
(element); inputs not populated in the configuration hold type ``Fake``
(-1) and label ``"Fake"``. These tables are indexed by input even where the
``axis`` attribute says ``dish``.

.. list-table::
   :header-rows: 1
   :widths: 24 14 22 40

   * - Dataset
     - Shape
     - Type
     - Description
   * - ``freq``
     - (:math:`N_f`)
     - compound {centre: f8, width: f8}
     - Centre frequency and channel width in MHz of each ``frequency``
       index.
   * - ``prod``
     - (:math:`N_p`)
     - compound {input_a: u2, input_b: u2}
     - Element-index pair :math:`(a, b)`, :math:`a \le b`, for each
       ``product`` index (row/column of the visibility matrix).
   * - ``grid_x_idx``
     - (inputs)
     - int64
     - East--west dish grid column of each input.
   * - ``grid_y_idx``
     - (inputs)
     - int64
     - North--south dish grid row of each input.
   * - ``dish_positions_in_grid_coords``
     - (:math:`N_d`, 3)
     - float64
     - Dish positions in metres in the grid coordinate frame (grid index x
       dish separation); one entry per dish.
   * - ``feed_pos_disp_m``
     - (inputs, 3)
     - float64
     - Feed position displacement from the nominal grid position, metres.
   * - ``coelev_disp_deg``
     - (inputs)
     - float64
     - Co-elevation pointing offset from the commanded ``dish_coelev_deg``,
       degrees.
   * - ``type``
     - (inputs)
     - int32
     - Dish type enum: -1 = ``Fake`` (unpopulated input), 0 = ``ArrayDish``,
       1 = ``RFIDish``.
   * - ``label``
     - (inputs)
     - variable-length string
     - Human-readable input labels; ``"Fake"`` for unpopulated inputs.

Visibility and per-(frequency, time) datasets
=============================================

These carry the science data. Datasets marked **[c]** are written with the
configured compression filter (see
`Chunking, compression, and reading the files`_); all others are
uncompressed.

.. list-table::
   :header-rows: 1
   :widths: 24 22 12 42

   * - Dataset
     - Shape (axes)
     - Type
     - Description
   * - ``vis`` **[c]**
     - (:math:`N_f`, :math:`N_p`, :math:`N_t`)
     - complex64
     - Accumulated visibilities :math:`V_{ab}`, normalized per valid FPGA
       sample (i.e. the mean correlation over the samples actually
       accumulated). Product ordering per ``/index_map/prod``.
   * - ``vis_weight`` **[c]**
     - (:math:`N_f`, :math:`N_p`, :math:`N_t`)
     - float32
     - Statistical weight of each visibility, an inverse-variance estimate
       computed by ``N2Accumulate`` from the scatter of sub-integrations
       (estimator selected by the accumulator's ``variance_mode``). 0 where
       the visibility is missing or invalid.
   * - ``eval`` **[c]**
     - (:math:`N_f`, :math:`N_{ev}`, :math:`N_t`)
     - float32
     - Eigenvalues from a per-frequency eigen-decomposition of the
       visibility matrix. The dataset exists (with zero extent) even when
       :math:`N_{ev} = 0`.
   * - ``evec`` **[c]**
     - (:math:`N_f`, :math:`N_{ev}`, :math:`N_e`, :math:`N_t`)
     - complex64
     - Corresponding eigenvectors, packed (eigenpair, element).
   * - ``erms``
     - (:math:`N_f`, :math:`N_t`)
     - float32
     - RMS of residual visibilities after removing the modelled eigenmodes.
       Sentinel (-1 or -FLT_MAX) when not computed.
   * - ``gain``
     - (:math:`N_f`, :math:`N_e`, :math:`N_t`)
     - complex64
     - Complex gain applied to each input. Sentinel -1+0j when no gain
       application was recorded (the F-engine digital gains are in
       ``/digital_gains`` instead).
   * - ``flags``
     - (:math:`N_f`, :math:`N_e`, :math:`N_t`)
     - float32
     - Per-input flag/weight factors from upstream flagging; 0 when
       unpopulated.
   * - ``radiometer_chi2``
     - (:math:`N_f`, :math:`N_t`, 3)
     - float32
     - Radiometer-equation consistency statistic (chi-squared) per
       polarization product (XX, XY, YY). Sentinel -1 when the test stage is
       not running.
   * - ``frames_added``
     - (:math:`N_f`, :math:`N_t`)
     - uint8
     - 1 where an input frame was received and written into this
       (frequency, time) slot, 0 where the slot is zero-filled.
   * - ``valid_fpga_count``
     - (:math:`N_f`, :math:`N_t`)
     - uint64
     - Number of FPGA ticks actually accumulated into the bin (after RFI and
       packet-loss excision).
   * - ``rfi_fpga_count``
     - (:math:`N_f`, :math:`N_t`)
     - uint64
     - FPGA ticks excised by the RFI mask (may overlap packet loss).
   * - ``rfi_only_fpga_count``
     - (:math:`N_f`, :math:`N_t`)
     - uint64
     - FPGA ticks excised by RFI only (excluding ticks also lost to packet
       loss).
   * - ``pl_fpga_count``
     - (:math:`N_f`, :math:`N_t`)
     - uint64
     - FPGA ticks lost to packet loss.
   * - ``frac_lost``
     - (:math:`N_f`, :math:`N_t`)
     - float32
     - Fraction of the nominal bin not accumulated:
       1 - ``valid_fpga_count`` / ``frame_length_fpga_ticks``. Defaults to
       1.0 in slots with no received frame.
   * - ``frac_rfi``
     - (:math:`N_f`, :math:`N_t`)
     - float32
     - ``rfi_fpga_count`` / ``frame_length_fpga_ticks``.
   * - ``frac_rfi_only``
     - (:math:`N_f`, :math:`N_t`)
     - float32
     - ``rfi_only_fpga_count`` / ``frame_length_fpga_ticks``.
   * - ``frac_pl``
     - (:math:`N_f`, :math:`N_t`)
     - float32
     - ``pl_fpga_count`` / ``frame_length_fpga_ticks``.

Per-time-bin metadata datasets
==============================

All are 1-D of length :math:`N_t` (axis ``time``) unless noted. "Bin" values
refer to the *nominal* accumulation bin on the upstream bin grid;
"time-centre" values refer to the midpoint of the data actually accumulated.
Identical values are shared by all frequencies of a time slot (the writer
verifies consistency, within ns-level tolerance, as frames arrive).

.. list-table::
   :header-rows: 1
   :widths: 32 14 54

   * - Dataset
     - Type
     - Description
   * - ``fpga_start_tick``
     - uint64
     - FPGA sequence number of the first tick accumulated into the bin.
       Also determines the final file name (earliest tick in the file).
   * - ``frame_length_fpga_ticks``
     - uint64
     - Nominal accumulated length of the bin in FPGA ticks.
   * - ``bin_abs_index``
     - uint64
     - Absolute (restart-safe) accumulation bin counter (see Overview).
       Unfilled slots hold :math:`2^{64}-1`.
   * - ``time_center_t_inst_ns``
     - int64
     - Instrument time at the midpoint of the accumulated data, ns
       (UNIX epoch).
   * - ``time_center_ut1_ns``
     - int64
     - UT1 at the same midpoint, ns since J2000(UT1).
   * - ``bin_t_inst_ns``
     - int64
     - Instrument time at the nominal bin centre, ns (UNIX epoch).
   * - ``bin_ut1_ns``
     - int64
     - UT1 at the nominal bin centre, ns since J2000(UT1).
   * - ``bin_delta_ut1_inst``
     - float64
     - UT1 - instrument time at the bin centre, seconds.
   * - ``bin_ERA_deg``
     - float64
     - Earth Rotation Angle at the nominal bin centre, degrees
       :math:`[0, 360)`.
   * - ``bin_xp_as``, ``bin_yp_as``
     - float64
     - Polar motion :math:`x_p`, :math:`y_p` at the bin centre, arcseconds.
   * - ``bin_start_ERA_deg``, ``bin_end_ERA_deg``
     - float64
     - ERA at the nominal bin edges; with ERA binning these are exact
       multiples of the bin width.
   * - ``bin_start_ERAL_deg``, ``bin_end_ERAL_deg``
     - float64
     - Local ERA (ERA + ITRS longitude + :math:`s'`) at the bin edges,
       degrees.
   * - ``rfi_frame_excision_enabled``
     - bool
     - Whether second-stage (frame-level) RFI excision was active for this
       bin.
   * - ``rfi_frame_excision_num``
     - int32
     - Number of active excision thresholds (<= :math:`N_k`); only the
       first this many entries of the two datasets below are meaningful.
   * - ``rfi_frame_excision_threshold``
     - float32 (:math:`N_t`, :math:`N_k`)
     - Spectral-kurtosis thresholds, in sigma, used for excision.
   * - ``rfi_frame_excision_fraction``
     - float32 (:math:`N_t`, :math:`N_k`)
     - Fraction of samples above the corresponding threshold required to
       trigger excision.

Completeness tracking and configuration snapshots
=================================================

``/frames_added``
   (:math:`N_f`, :math:`N_t`), uint8. The authoritative mask of which
   (frequency, time) slots actually contain data. Files finalized by the
   grace-period or shutdown paths are partially filled; *always* check this
   mask (or equivalently ``frac_lost`` < 1) before interpreting zeros in the
   data datasets as measurements.

``/config_json``
   1-D, variable-length strings, unlimited maximum extent (chunk size 8),
   appended at every file flush. Each element is one JSON document recorded
   by kotekan's :ref:`Config Tracker <user_config>`: snapshots of the running
   kotekan configurations in the pipeline (the writer's process and upstream
   senders) and of the FPGA configuration fetched from the FPGA controller.
   Each document carries the keys ``config`` (the configuration tree
   itself), ``json_hash``, ``kotekan_version``, ``kotekan_git_commit_hash``,
   ``kotekan_build_branch``, and ``kotekan_cmake_options``. Empty if the
   tracker is disabled.

Digital gains group (``/digital_gains``)
========================================

Only present when gains are configured (``baseband_gain_file`` pointing to a
local gains archive, or ``baseband_gain_host_info`` naming an FPGA-controller
config block to fetch it from over HTTP; the fetched copy is cached as
``<acq>/.partial/baseband_gains.h5``). The gains archive is copied
*verbatim* (``H5Ocopy`` of its root group) into every output file, so its
internal layout follows the ``pychfpga`` ``DigitalGainArchive`` format, not
the conventions above. Typical contents:

**Group attributes** (from the source archive): ``acquisition_name``,
``archive_version``, ``collection_server``, ``instrument_name``, ``notes``,
``system_user``, ``type``, ``version``; plus ``selected_update_idx``
(int32), added by the writer: the index along ``update_time`` that was
current for this acquisition (configuration ``baseband_gain_update_idx`` =
-1 selects the latest entry).

.. list-table::
   :header-rows: 1
   :widths: 26 22 22 30

   * - Dataset
     - Shape
     - Type
     - Description
   * - ``gain_coeff``
     - (updates, channels, inputs)
     - complex64
     - Complex gain coefficient per update, FPGA channel (full band), and
       input.
   * - ``gain_exp``
     - (updates, inputs)
     - int32
     - Per-input binary exponent; the applied gain is
       ``gain_coeff`` x 2\ :sup:`gain_exp`.
   * - ``compute_time``
     - (updates, inputs)
     - float64
     - UNIX time at which each input's gains were computed.
   * - ``update_id``
     - (updates)
     - string
     - Identifier of each gain update.
   * - ``index_map/freq``
     - (channels)
     - compound {centre: f8, width: f8}
     - Centre/width in MHz of the FPGA channels (full band, not just the
       science band).
   * - ``index_map/input``
     - (inputs)
     - compound {chan_id: u2, correlator_input: S32}
     - Channel id and correlator input serial for each input.
   * - ``index_map/update_time``
     - (updates)
     - float64
     - UNIX time of each gain update.

The root attribute ``digital_gains_source_file`` records the source path.

Chunking, compression, and reading the files
============================================

Chunking
--------

One-dimensional datasets use a single chunk equal to their full extent.
Multi-dimensional datasets are chunked with each dimension capped by a
per-axis block size from the stage configuration: ``frequency`` by
``blocksize_f``, ``product`` and ``element`` by ``blocksize_p``, ``time`` by
``blocksize_t``; zero-length dimensions get chunk extent 1.

Compression
-----------

The four large datasets ``/vis``, ``/vis_weight``, ``/eval``, and ``/evec``
are written with the configured compression; all other datasets are
uncompressed. The options are:

* ``compression: none`` (default) --- no filter;
* ``compression: deflate`` --- zlib at ``compression_level`` (default 4);
* ``use_bitshuffle: true`` --- the *bitshuffle* HDF5 filter (filter id
  32008), with ``compression`` selecting the backend codec (``none``,
  ``lz4``, or ``zstd``).

When bitshuffle is used, readers must have the plugin available: in Python,
``import hdf5plugin`` (or having the bitshuffle plugin on
``HDF5_PLUGIN_PATH``) before reading those datasets with h5py. Note that
h5py's ``Dataset.compression`` property reports ``None`` for third-party
filters; inspect the dataset creation property list to see them.

Minimal reading example
-----------------------

.. code-block:: python

    import h5py, hdf5plugin  # hdf5plugin registers the bitshuffle filter
    import numpy as np

    with h5py.File("vis_<idx>_<timestamp>.h5") as f:
        freq = f["index_map/freq"]["centre"]      # MHz, (Nf,)
        prod = f["index_map/prod"][:]             # (input_a, input_b)
        good = f["frames_added"][:].astype(bool)  # (freq, time) validity
        vis  = f["vis"][:]                        # (freq, prod, time) complex64
        w    = f["vis_weight"][:]
        t    = f["time_center_t_inst_ns"][:] * 1e-9   # UNIX seconds
        vis[~good[:, None, :].repeat(vis.shape[1], 1)] = np.nan
