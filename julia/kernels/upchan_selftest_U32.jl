include("setup_selftest.jl")
const U = 32
include("upchan.jl")
main(; run_selftest=true)
