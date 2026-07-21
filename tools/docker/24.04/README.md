# Kotekan docker environment instructions

The Dockerfile in this directory can be used to set up the environment necessary for
running and/or testing Kotekan. This Dockerfile is the one used in CI tests, and may
be used in production runs.

The file can be build with a cpu-only target (i.e. no NVIDIA drivers or CUDA toolkit),
`docker build --target cpu`, or will otherwise build the gpu target by default.
An additional `intel` target extends the cpu target with the Intel oneAPI C/C++
compilers (icx/icpx), `docker build --target intel`.
