include("setup_selftest.jl")
const U = 16
include("upchan.jl")
# Both test cases always run, so that a failure of one still reports the other.
# `upchan_selftest.sh` relies on there being exactly one result line per case.
nerrors = main(; run_selftest=true, testcase=:tone)
nerrors += main(; run_selftest=true, testcase=:constant)
nerrors == 0 || error("*** SELF-TEST FAILED: $(nerrors) mismatches ***")
