# Self-test driver for the FRB beamformer. It uses the smallest production
# setup; `frb.jl` has no dedicated `selftest` setup because the self-test picks
# its own dish/beam/time indices rather than filling the whole output buffer.
# Run it via `./julia/bin/selftests.sh`, which works in a scratch directory so
# that the generated kernels under `julia/output/` are left alone.
include("setup_smallfinder.jl")
const U = 16
include("frb.jl")
main(; run_selftest=true)
