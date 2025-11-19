****************
hdf5VisWrite
****************

The ``hdf5VisWrite`` stage buffers visibility frames, which are transposed by
the ``visFileData`` class, and writes each file window as a single HDF5 product.
It relies on HDF5/HighFive and is a canonical way to persist N2
visibilities for downstream analysis.

.. doxygenclass:: hdf5VisWrite
   :members:
   :undoc-members:

Overview
--------

* Frames arrive from an ``N2Buffer`` and carry ``N2Metadata`` describing cadence,
  number of frequencies/products, etc.
* Frames are grouped into UTC-midnight-aligned windows of length ``file_seconds``
  (must evenly divide 86400).  Within a window the stage buffers an entire
  ``nfreq × nprod × ntimes`` block before writing.
* Each file is assembled inside ``<base_dir>/.partial`` and atomically renamed
  to its final location upon finalization.
* Missing frames simply leave zero/empty regions in the arrays; "late frames"
  (according to a grace period) are dropped once a final file already exists on disk.

File lifecycle
--------------

1. For every incoming frame, the stage computes the aligned file start time
   ``<midnight> + k * file_seconds`` and generates a filename of the form::

        <base_dir>/vis_<abs_idx>_YYYYMMDDThhmmss_<nsec>.h5

2. If a partial dataset for that window already exists it is reopened; otherwise
   the stage creates the directory structure, pre-allocates all arrays, and
   writes file-level attributes.
3. When a dataset becomes full, the data are flushed, the file handle is closed,
   and ``.partial/<name>`` is renamed into the final file.  On shutdown every
   open dataset is flushed the same way.
4. Datasets that have not received frames for ``late_frame_grace_seconds`` are
   also flushed/renamed to avoid leaking files if later frames never arrive.

File contents
-------------

Every file contains fixed arrays (dimensions shown as ``nfreq``, ``nprod``,
``ninput``, ``nev``, ``nt``):

- ``/vis_array`` (``nfreq × nprod × nt``) complex64 visibilities.
- ``/weights_array`` (``nfreq × nprod × nt``) float32 inverse-variance weights.
- ``/eval_array`` (``nfreq × nev × nt``) eigenvalues.
- ``/evec_array`` (``nfreq × nev × ninput × nt``) eigenvectors.
- ``/erms_array`` (``nfreq × nt``) float32 RMS per frequency.
- ``/gain_array`` and ``/flags_array`` (``nfreq × ninput × nt``) calibration
  gains and per-input flags.
- ``/frac_lost_array`` and ``/frac_rfi_array`` (``nfreq × nt``) lost/RFI fractions.
- ``/n_valid_fpga_ticks`` / ``/n_rfi_fpga_ticks`` (``nfreq × nt``) uint64 counters.
- Per-time datasets ``/fpga_start_tick``, ``/frame_start_time_ns``,
  ``/frame_length_fpga_ticks``, and ``/era_deg``.

File-level attributes store the dimensions (``num_elements``, ``num_prod``,
``num_ev``, ``num_freq``, ``frame_length_fpga_ticks``) so downstream tools can
validate the layout.

Chunking and compression
------------------------

Chunk sizes follow the configured block sizes: ``blocksize_f`` caps the frequency
dimension, ``blocksize_t`` caps the time dimension (``blocksize_p`` is reserved
for future product chunking and currently ignored).  Setting a block size to zero
leaves the HighFive/HDF5 defaults.

Compression options:

- ``compression = "none"`` (default) writes raw data.
- ``compression = "deflate"`` enables zlib/deflate; ``compression_level`` (0 =
  driver default) controls the zlib level.
- ``use_bitshuffle = true`` adds the bitshuffle filter; in that mode
  ``compression`` selects the backend (``"none"``, ``"zstd"``, ``"lz4"``) and
  ``compression_level`` controls the backend level (default 9).  Bitshuffle
  requires the relevant HDF5 plugin to be available at runtime.

Buffering, grace, and metrics
-----------------------------

* ``late_frame_grace_seconds`` controls how long a dataset may remain partially
  filled after the last write before it is finalized.  This protects against
  sources that never complete a window.
* ``max_frames`` (``-1`` for unlimited) is useful in CI or local tests to stop
  kotekan after a known number of frames.  When the limit is hit the stage calls
  ``exit_kotekan(CLEAN_EXIT)``.
* ``seq_length_nsec_override`` lets you override
  ``Telescope::seq_length_nsec()`` (handy in tests that lack a telescope block).
* Prometheus metrics ``kotekan_viswrite_write_time_seconds`` and
  ``kotekan_viswrite_n_datasets`` report, respectively, the duration of the most
  recent flush and how many datasets are currently open.

Configuration
-------------

``in_buf`` (string, required)
    Name of the ``N2Buffer`` supplying frames.
``base_dir`` (string, required)
    Output directory.  The stage creates it (plus ``.partial``) if necessary.
``blocksize_f`` (uint, default ``0``)
    Chunk length along the frequency axis; ``0`` lets HighFive choose.
``blocksize_p`` (uint, default ``0``)
    Reserved for future per-product chunking; currently ignored.
``blocksize_t`` (uint, default ``1``)
    Chunk length along the time axis.
``compression`` (string, default ``"none"``)
    ``"none"`` | ``"deflate"`` | ``"zstd"`` | ``"lz4"`` depending on whether
    you enable zlib or bitshuffle compression.
``compression_level`` (uint, default ``0``)
    Codec level; ``0`` tells the stage to use a sensible default (4 for zlib,
    9 for bitshuffle backends).
``use_bitshuffle`` (bool, default ``false``)
    Enables the bitshuffle+compression plugin.
``file_seconds`` (uint, default ``600``)
    Length of each file window.  Must evenly divide 86400; otherwise the stage
    aborts during startup.  Internally the number of frame slots is rounded up
    so that frames that start before the window end always fit.
``late_frame_grace_seconds`` (uint, default ``60``)
    Grace period before partially-complete files are flushed/renamed.
``max_frames`` (int, default ``-1``)
    Stop after a fixed number of frames; leave at ``-1`` for production.
``seq_length_nsec_override`` (uint, default ``0``)
    Override the telescope FPGA sequence length (mainly for tests).

Example
-------

CI exercises the stage in ``config/ci-tests/batch/test_hdf5VisWrite.yaml``; the core
snippet looks like::

    hdf5_vis_write:
        kotekan_stage: hdf5VisWrite
        in_buf: fakevis_buf
        base_dir: ./vis_ci_out
        blocksize_f: 1
        blocksize_t: 5
        file_seconds: 6
        late_frame_grace_seconds: 10

That configuration keeps the files tiny so tests complete quickly, but for real
pipelines longer ``file_seconds`` (e.g. > 600 s) will optimize I/O.

Monitoring and validation
-------------------------

* Watch ``kotekan_viswrite_*`` metrics plus the WARN logs that indicate why a
  frame was dropped or why file windows are being rounded up (<1 s cadence).
* ``tests/boost/test_hdf5VisWrite.cpp`` performs end-to-end validation of file
  naming, partial-file recovery, timeout behavior, and geometry metadata; it is
  a good reference when extending the stage.
* After a run, the presence of ``.partial`` files indicates that the pipeline
  stopped mid-window—look for matching WARN messages or bump
  ``late_frame_grace_seconds`` if that is expected.
