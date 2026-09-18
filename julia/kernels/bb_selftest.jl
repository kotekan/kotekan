# Self-test driver for the baseband beamformer. See `frb_selftest.jl`.
include("setup_smallfinder.jl")
include("bb.jl")
main(; run_selftest=true)
