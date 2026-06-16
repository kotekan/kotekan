.. _chime_archive_format:

=========================================================
CHIME archive writers (VisWriter, HFBWriter, baseband)
=========================================================

The CHIME-era writer stages produce the CHIME correlator archive formats.
They are used by the ``chime_science_run_*`` configurations and are
documented here at summary level; the archive formats themselves are CHIME
conventions, processed downstream (e.g. by *gossec*) into the archive HDF5
format read by ``caput``/``ch_util``.

.. note::
   This page summarizes the output of ``lib/stages/BaseWriter.cpp`` and its
   subclasses (``VisWriter``, ``HFBWriter``) plus ``BasebandWriter`` and
   ``VisSharedMemWriter``; update it when changing those stages' output.

VisWriter and HFBWriter (BaseWriter)
====================================

Both derive from ``BaseWriter``, which manages *acquisitions*: directories
under ``root_path`` named from the time of the first sample,

.. code-block:: text

    <root_path>/<YYYYMMDDTHHMMSSZ>_<instrument_name>_corr/    (VisWriter)
    <root_path>/<YYYYMMDDTHHMMSSZ>_<instrument_name>_hfb/     (HFBWriter)

Within an acquisition, files are named by seconds since acquisition start:
``<sec:08d>_0000`` (VisWriter) or ``hfb_<sec:08d>_0000`` (HFBWriter), and
roll over after ``file_length`` time samples (default 1024). A sliding
``window`` (default 20) of recent samples is kept open for late data;
acquisitions close after ``acq_timeout`` (default 300 s) of inactivity.
Every open file has a companion lock file ``.<name>.lock`` containing the
writer PID; readers should skip locked files.

The on-disk format is selected by ``file_type``:

* ``raw`` / ``hfbraw`` --- a ``.meta`` + ``.data`` file pair (see below);
* ``hdf5`` / ``hdf5fast`` --- single ``.h5`` files (``hdf5fast``
  pre-allocates contiguous datasets and writes via direct I/O, recording
  the true sample count in a ``num_time`` attribute);
* ``ring`` --- like ``raw`` but a fixed-size ring that overwrites the
  oldest samples (used for live monitoring buffers).

**Raw format.** The ``.meta`` file is a msgpack-serialized JSON document
holding ``attributes`` (instrument and version info, e.g.
``archive_version: NT_3.1.0``, ``weight_type``, ``git_version_tag``),
``index_map`` (the ``freq``, ``input``, ``prod`` or ``stack``, ``ev``, and
``time`` axes), an optional ``reverse_map``, and a ``structure`` section
giving ``metadata_size``, ``data_size``, and the 4 KiB-aligned
``frame_size``, plus ``nfreq`` and ``ntime``. The ``.data`` file is a
dense (time, frequency) grid of slots of ``frame_size`` bytes at offset
``(time_ind * nfreq + freq_ind) * frame_size``, each slot containing a
1-byte valid flag, the frame metadata struct, and the frame payload;
missing samples remain zero-filled with valid flag 0.

Key configuration (BaseWriter): ``root_path``, ``file_type`` (default
``hdf5fast``), ``file_length``, ``window``, ``acq_timeout``,
``instrument_name`` (default ``chime``). HFBWriter additionally records the
beam and sub-frequency axes (``num_frb_total_beams``, ``factor_upchan``).

BasebandWriter
==============

``BasebandWriter`` records triggered baseband dumps. Each event/frequency
goes to its own raw file::

    <root_path>/baseband_raw_<event_id>/baseband_<event_id>_<freq_id>.data

Frames are appended as a 1-byte valid flag, the serialized
``BasebandMetadata``, and the raw baseband payload. An event's files are
closed after ``dump_timeout`` (default 60 s) without new data;
``max_frames_per_second`` (default 0 = off) throttles the write rate.

VisSharedMemWriter
==================

``VisSharedMemWriter`` is not a file writer: it exports the most recent
``num_samples`` visibility integrations into a POSIX shared-memory ring
buffer (by default under ``/dev/shm/<name>``) for live consumers. The
region holds a small structured header (write count and ring geometry), a
per-slot access record protected by a named semaphore ``sem.<name>``, and
the ring of frames (valid flag + metadata + data). A write count of 0
signals a graceful writer shutdown.
