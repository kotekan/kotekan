.. _raw_frames_format:

================================================
Raw frame dumps (rawFileWrite, nDiskFileWrite)
================================================

Two stages write kotekan buffer frames to disk as raw bytes, without any
container format.

.. note::
   This page documents the on-disk formats produced by
   ``lib/stages/rawFileWrite.cpp`` and ``lib/stages/nDiskFileWrite.cpp``;
   update it when changing the byte layout written there.

rawFileWrite
============

``rawFileWrite`` (``lib/stages/rawFileWrite.cpp``) dumps frames of any
buffer/metadata type. Files are written to ``base_dir`` as::

    <base_dir>/[<hostname>_]<file_name>_<file:07d>.<file_ext>

with a 7-digit file counter and the hostname prefix controlled by
``prefix_hostname`` (default true). Each file holds
``num_frames_per_file`` frames (default 1), appended sequentially. Each
frame is laid out as:

.. code-block:: text

    uint32   metadata_size      (0 if the frame has no metadata)
    bytes    metadata           (metadata_size bytes; the metadata object's
                                 binary serialization, e.g. chordMetadata)
    bytes    frame              (frame_size bytes of raw buffer content)

There is no file header; all frames in a file have the same sizes, so
readers can compute frames-per-file as
``file_size / (4 + metadata_size + frame_size)``. The companion
``rawFileRead`` stage replays these files into a pipeline (as written it
consumes the size prefix only once per file, so it only reads files
written with ``num_frames_per_file`` = 1 correctly).

Configuration: ``base_dir``, ``file_name``, ``file_ext`` (all required);
``num_frames_per_file`` (default 1); ``exit_after_n_files`` (default 0 =
unlimited); ``prefix_hostname`` (default true).

nDiskFileWrite
==============

``nDiskFileWrite`` (``lib/stages/nDiskFileWrite.cpp``) is a burst recorder
that stripes VDIF-format frames round-robin across ``num_disks`` drives.
Each frame is written verbatim (no header, no metadata) as one file::

    <disk_base>/<disk_set>/<disk_id>/<YYYYMMDDTHHMMSSZ>_<instrument_name>_vdif/<file:07d>.vdif

The dataset directory is named from the UTC start time; disk ``i`` receives
file numbers ``i``, ``i + num_disks``, ``i + 2 num_disks``, ... A
``settings.txt`` key=value file describing the capture (format version,
``num_freq``, ``num_inputs``, ``num_timesamples``, packet layout fields,
``start_time``, ``num_disks``, ``disk_set``, a free-form ``note``) is
written into the dataset directory on every disk when
``write_metadata_and_gains`` is true (default), along with copies of any
configured ``gain_files``. The companion ``nDiskFileRead`` stage replays a
capture.

Configuration: ``num_disks``, ``disk_base``, ``disk_set``,
``write_to_disk`` (all required); ``instrument_name`` (default
``no_name_set``); ``write_metadata_and_gains`` (default true);
``gain_files``; ``note``.
