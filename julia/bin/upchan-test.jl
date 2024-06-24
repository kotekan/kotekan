# using CairoMakie
# using SixelTerm

# Load the code
include("src/FEngine.jl")
using .FEngine

# ntaps = 4
# nsamples = 16384
# ntimes = 8192

# Set up some data, a monochromatic source in the middle between two channels
x = sinpi.(2 * (0.3e9 / 3.2e9 + 1 / 16386 / 2) * (0:(16384 * (8192 + 3) - 1)));
# First PFB
y = FEngine.upchan(x, 4, 16384);
# heatmap(abs.(y[1500:1600,:])')

# Compare the two neighbouring non-zero channels
maximum(abs, y[1537, :])
maximum(abs, y[1538, :])
# They are very similar!
maximum(abs, y[1538, :] - y[1537, :])
maximum(abs, y[1538, :] + y[1537, :])

# Load the code (again)
include("src/FEngine.jl")

z1 = FEngine.upchan(y[1537, :], 4, 8);
z2 = FEngine.upchan(y[1538, :], 4, 8);
maximum(abs, z1; dims=2)
maximum(abs, z2; dims=2)
