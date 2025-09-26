using CairoMakie
using DimensionalData
using HDF5
using Kotekan
using Printf
using SixelTerm

# Definitions
path = "/home/eschnett/src/kotekan/data/fengine_test_pathfinder"
host = "cx66"
quantity() = "voltage"
filename(quantity::AbstractString, iter::Integer) = "$(path)/$(host)_$(quantity).$(@sprintf "%08d" iter).h5"

# Read data
iters = 0:1                     # 0:23
E = cat([read_hdf5(filename(quantity(), iter)) for iter in iters]...; dims=:T);
typeof(E)
eltype(E)
size(E)
dims(E)
size(E, :T)
size(E, :F)
size(E, :P)
size(E, :D)

# # Read metadata
# f = h5open(filename(quantity(), iters[begin]))
# make_dimvector(A::AbstractVector, dimname::Symbol) = DimArray(A, Dim{Symbol}(0:length(A)-1))
# coarse_freqs = make_dimvector(f[quantity]["coarse_freq"][], :F);

# Reduce via RMS over all dimensions except T and F
ETF = rms(E; dims=otherdims(E, :T, :F));

# Downsample via max over T and F
ETFsmall = bin_max(ETF; dims_lengths=[:T => 512, :F => 384]);

let
    fig = Figure(; size=(640, 480))
    ax = Axis(fig[1, 1]; title="Time/frequency diagram", xlabel="time [ms]", ylabel="local channel")
    # TODO: read time scale and frequencies from file
    tscale = 1e+3 * 16384 / 3200.0e+6 # [ms]
    # fscale = 1e-6 * 3200.0e6 / 8192   # [MHz]
    obj = heatmap!(
        tscale * val(dims(ETFsmall, :T)),
        # val(dims(ETF, :T)),
        val(dims(ETFsmall, :F)),
        # fscale * coarse_freqs[val(dims(ETFsmall, :F))],
        parent(permutedims(ETFsmall, [:T, :F]));
        colormap=:plasma,
        colorrange=(0, √2 * 7),
    )
    Colorbar(fig[1, 2], obj; label="voltage intensity")
    display(fig)
end;
