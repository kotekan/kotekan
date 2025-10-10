using CairoMakie
using DimensionalData
using HDF5
using Kotekan
using Printf
using SixelTerm

# Definitions
path = "/home/eschnett/src/kotekan/data/fengine_test_pathfinder"
host = "cx66"
quantity(U::Integer) = "upchan_U$(U)_voltage"
filename(quantity::AbstractString, iter::Integer) = "$(path)/$(host)_$(quantity).$(@sprintf "%08d" iter).h5"

# Read data
iters = 0:1                     # 0:23
Us = [2, 4, 8, 16, 32, 64]
Es = Dict([U => cat([read_hdf5(filename(quantity(U), iter)) for iter in iters]...; dims=:Tbar) for U in Us]);

# Reduce via RMS over all dimensions except Tbar and Fbar
ETFs = Dict([U => rms(Es[U]; dims=otherdims(Es[U], :Tbar, :Fbar)) for U in Us]);

# Downsample via max over Tbar and Fbar
ETFsmalls = Dict([U => bin_max(ETFs[U]; dims_lengths=[:Tbar => 512, :Fbar => 384]) for U in Us]);

for U in Us
    ETFsmall = ETFsmalls[U]
    fig = Figure(; size=(640, 480))
    ax = Axis(fig[1, 1]; title="Time/frequency diagram for U=$(U)", xlabel="time [ms]", ylabel="local channel")
    # TODO: read time scale and frequencies from file
    tscale = U * 1e+3 * 16384 / 3200.0e+6 # [ms]
    # fscale = 1e-6 * 3200.0e6 / 8192   # [MHz]
    obj = heatmap!(
        tscale * val(dims(ETFsmall, :Tbar)),
        # val(dims(ETFsmall, :Tbar)),
        val(dims(ETFsmall, :Fbar)),
        # fscale * coarse_freqs[val(dims(ETFsmall, :Fbar))],
        parent(permutedims(ETFsmall, [:Tbar, :Fbar]));
        colormap=:plasma,
        colorrange=(0, √2 * 7),
    )
    Colorbar(fig[1, 2], obj; label="voltage intensity")
    display(fig)
end

nothing
