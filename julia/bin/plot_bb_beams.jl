using CairoMakie
using DimensionalData
using HDF5
using Kotekan
using Printf
using SixelTerm

function aspect!(fig::Figure, row::Integer, col::Integer, ratio_x_over_y::Real)
    if ratio_x_over_y > 4 / 3
        rowsize!(fig.layout, row, Aspect(1, inv(ratio_x_over_y)))
    else
        colsize!(fig.layout, col, Aspect(1, ratio_x_over_y))
    end
    return nothing
end

# Definitions
path = "/home/eschnett/src/kotekan/data/fengine_test_pathfinder"
host = "x0000"
quantity() = "bb_beams"
filename(quantity::AbstractString, iter::Integer) = "$(path)/$(host)_$(quantity).$(@sprintf "%08d" iter).h5"

# Read data
iters = 0:1                     # 0:24
J = cat([read_hdf5(filename(quantity(), iter)) for iter in iters]...; dims=:T);
typeof(J)
eltype(J)
size(J)
dims(J)
size(J, :T)
size(J, :F)
size(J, :P)
size(J, :B)

# Reduce via L2-norm over all dimensions except T, P, and B
JTPB = norm2(J; dims=otherdims(J, :T, :P, :B));

# Downsample via RMS over T
JTPBsmall = bin_rms(JTPB; dims_lengths=[:T => 512, :P => 2, :B => 16]);

# Combine B and P
JTPBmerged = mergedims(JTPBsmall, (:B, :P) => :BP);

let
    fig = Figure(; size=(640, 480))
    ax = Axis(fig[1, 1]; title="Time/beam diagram", xlabel="time [ms]", ylabel="polarization+beam")
    # TODO: read time scale from file
    tscale = 1e+3 * 16384 / 3200.0e+6 # [ms]
    obj = heatmap!(
        tscale * val(dims(JTPBmerged, :T)),
        [16*p+b for (b, p) in val(dims(JTPBmerged, :BP))],
        parent(permutedims(abs2.(JTPBmerged), [:T, :BP]));
        colormap=:plasma,
        colorrange=(0, (√2 * 7.5)^2),
    )
    Colorbar(fig[1, 2], obj; label="voltage intensity")
    display(fig)
end;

# # Read dish positions
# dish_positions = read_hdf5(filename("dish_positions", iters[begin]));
# 
# let
#     fig = Figure(; size=(640, 480))
#     ax = Axis(fig[1, 1]; title="Dishes", xlabel="east-west [m]", ylabel="north-south [m]")
#     dishsize = 6                # dish diameter [m]
#     min_ew, max_ew = extrema(@view dish_positions[1, :]) .+ (-dishsize, +dishsize)
#     min_ns, max_ns = extrema(@view dish_positions[2, :]) .+ (-dishsize, +dishsize)
#     max_E = maximum(Enorm2)
#     xlims!(ax, min_ew, max_ew)
#     ylims!(ax, min_ns, max_ns)
#     obj = scatter!(
#         ax,
#         parent(@view dish_positions[1, :]),
#         parent(@view dish_positions[2, :]);
#         color=reshape(parent(Enorm2), :),
#         colormap=:plasma,
#         colorrange=(0, 10),
#         marker=Circle,
#         markersize=dishsize,
#         markerspace=:data,
#     )
#     Colorbar(fig[1, 2], obj; label="amplitude")
#     aspect!(fig, 1, 1, (max_ew - min_ew) / (max_ns - min_ns))
#     display(fig)
# end;

# Reduce via RMS over all dimensions except B
JB = rms(JTPB; dims=otherdims(JTPB, :B));

# Read beam positions
bb_beam_positions = read_hdf5(filename("bb_beam_positions", iters[begin]));

# CHORD
c₀ = 299_792_458                # [m/s]
adc_freq = 3.2e+9               # [Hz]
num_samples_per_frame = 16384
dish_separation_ew = 6.3        # [m]
dish_separation_ns = 8.5        # [m]

lambda₀ = c₀ / (adc_freq / num_samples_per_frame) # [m]
channel = 7680
lambda = lambda₀ / channel      # [m]
beamsize_ew = lambda / dish_separation_ew # [rad]
beamsize_ns = lambda / dish_separation_ns # [rad]

let
    fig = Figure(; size=(640, 480))
    ax = Axis(fig[1, 1]; title="Baseband beams", xlabel="east-west [rad]", ylabel="north-south [rad]")

    plot_beamsize_ew = beamsize_ew / 2 # `/2` looks better
    plot_beamsize_ns = beamsize_ns / 2

    min_ew, max_ew = extrema(@view bb_beam_positions[1, :]) .+ (-plot_beamsize_ew, +plot_beamsize_ew)
    min_ns, max_ns = extrema(@view bb_beam_positions[2, :]) .+ (-plot_beamsize_ns, +plot_beamsize_ns)
    xlims!(ax, min_ew, max_ew)
    ylims!(ax, min_ns, max_ns)
    obj = scatter!(
        ax,
        parent(@view bb_beam_positions[1, :]),
        parent(@view bb_beam_positions[2, :]);
        color=reshape(parent(abs2.(JB)), :),
        colormap=:plasma,
        colorrange=(0, (√2 * 7)^2),
        marker=Circle,
        markersize=(plot_beamsize_ew, plot_beamsize_ns),
        markerspace=:data,
    )
    Colorbar(fig[1, 2], obj; label="baseband beam intensity")
    aspect!(fig, 1, 1, (max_ew - min_ew) / (max_ns - min_ns))
    display(fig)
end;

nothing
