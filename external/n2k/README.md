Build instructions:
```
git clone https://github.com/kmsmith137/gputils
cd gputils
make -j
cd ..

git clone https://github.com/kmsmith137/n2k
cd n2k
make -j
./test-correlator
```

The n2k library contains:

  - An int4+4 GPU visibility matrix kernel with a boolean RFI mask.

  - GPU kernels which implement the high-speed RFI masking logic in CHIME/CHORD.

Some loose ends that I might fix later. Let me know if you'd like me to prioritize any of these.
See `n2k.hpp` for more info:

  - Currently, we only support 4+4 bit electric field samples, in the range [-7,7].
    If the value (-8) arises, then the output of the computation will be incorrect!

  - The kernel will segfault if run on a GPU which is not the cuda default device.

Contact: Kendrick Smith <kmsmith@perimeterinstitute.ca>
