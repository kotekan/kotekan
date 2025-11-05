****************
gdalVisWrite
****************

Unified visibility writer using GDAL. Buffers a full file window of visibilities in memory
and writes large slabs to disk in one go. Supports Zarr and HDF5 containers selected at
runtime.

.. doxygenclass:: gdalVisWrite
   :members:
   :undoc-members:

Config
------

- `base_dir` (string) — Output directory; a `.partial/` subdir is used for atomic renames.
- `file_name` (string) — Base file name prefix.
- `prefix_hostname` (bool, default true) — Prefix files with the host name.
- `format` (string, default `zarr`) — `zarr` or `hdf5`.
- `compression` (string, default `none`) — `none`, `deflate`, `zstd`, or `blosc`.
- `compression_level` (int, default 0) — Compression level for `deflate`/`zstd`.
- `use_butshuffle` (bool, default false) — Request shuffle/"butshuffle" if supported by the driver.
- `blocksize_f` (uint, default 0) — Chunk size along frequency (0 = auto/minimal).
- `blocksize_p` (uint, default 0) — Chunk size along product (0 = full products).
- `blocksize_t` (uint, default 1) — Chunk size along time.
- `file_seconds` (uint, default 600) — File window length in seconds (must divide 86400).
- `late_frame_grace_seconds` (uint, default 60) — Grace period before finalizing partial files.
- `max_frames` (int, default -1) — Stop after N frames (-1 = unlimited).

Notes
-----

- The stage writes the same array names and dimensions independent of container format.
- Compression is applied per-array using commonly supported GDAL creation options
  (e.g., `COMPRESS=DEFLATE|ZSTD`, `LEVEL=<n>`, `SHUFFLE=YES`).
- ZIP storage for Zarr is not used.
