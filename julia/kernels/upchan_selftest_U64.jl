include("setup_selftest.jl")
const U = 64
include("upchan.jl")
main(; run_selftest=true)
