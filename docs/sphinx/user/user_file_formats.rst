.. _user_file_formats:

************
File Formats
************

Reference documentation for the data files that kotekan writer stages produce
on disk. Each page describes one output format: every dataset and attribute,
its shape, type, and meaning, plus practical notes on reading the files.

These pages document the on-disk format and should be kept in sync with the
writer stages that produce them; writer stages contain notes
pointing back to the format pages.

.. toctree::
    :maxdepth: 1

    file_formats/n2_vis_hdf5
    file_formats/hdf5_frames
    file_formats/asdf_zarr_frames
    file_formats/raw_frames
    file_formats/chime_archive

JSON configuration snapshots written by ``configTrackerWriter`` are covered
with the Config Tracker in :ref:`user_config`.
