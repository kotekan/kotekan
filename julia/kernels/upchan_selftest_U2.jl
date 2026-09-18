include("setup_selftest.jl")
const U = 2
include("upchan.jl")
# All test cases always run, so that a failure of one still reports the others.
# `upchan_selftest.sh` relies on there being exactly one result line per case.
nerrors = main(; run_selftest=true, testcase=:tone)
nerrors += main(; run_selftest=true, testcase=:constant)
nerrors += main(; run_selftest=true, testcase=:impulse)
# The same tone again, but with the time window wrapped around the end of the
# ring buffer, which is the only case that exercises a nonzero `Tmin`/`T̄min`.
nerrors += main(; run_selftest=true, testcase=:tone, twrap=true)
nerrors == 0 || error("*** SELF-TEST FAILED: $(nerrors) mismatches ***")
