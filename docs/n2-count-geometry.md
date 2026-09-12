# Packet/RFI count geometry

`cudaPL1bitCorrelator` requires positive polarization and dish counts, with
`num_dishes` divisible by eight. The count kernel supports
`num_polarizations * (num_dishes / 8)` equal to 16 or 128: 64 or 512 dishes
with two polarizations. The constructor rejects other geometries before
registering buffers. Other kernel constraints still apply.

Station order is `[polarization, dish]`, with packet groups of eight adjacent
dishes. The count kernel intersects both packet masks with the common RFI
mask. It stores lower-triangular counts in row-major 8-by-8 tiles; correlation
tiles are 16-by-16.

The test covers valid geometries, invalid dimensions and overflow in
`n2_count_station_groups`. It compiles the host geometry check; it does not run
the GPU kernel.

```sh
python -m pytest --noconftest -q tests/test_n2_count_geometry.py
```
