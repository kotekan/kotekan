# GPU C++/CUDA core utils

This directory contains some of the files from
<https://github.com/kmsmith137/ksgpu> written by Kendrick Smith
<kmsmith@perimeterinstitute.ca>.

Files not used in Kotekan (tests etc.) have been removed.

This directory has the same layout as the upstream repository. To
update this directory, copy the respective files from the upstream
repository.

If you have `ksgpu` checked out in `~/ksgpu`, you can update files from the current directory via:
```
for x in generate_device_mma_hpp.py $(find include/ loose_ends/ src_lib/ -type f); do
  cp -v ~/ksgpu/$x $x;
done
```

