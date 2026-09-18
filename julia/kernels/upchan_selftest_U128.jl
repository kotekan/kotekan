include("setup_selftest.jl")
const U = 128
include("upchan.jl")
main(; run_selftest=true)
