using CairoMakie
using Kotekan
using LinearAlgebra
using SixelTerm

function dish_norm(A, p=2, freq=:)
    scale = p == Inf ? 1 : inv(length(@view A[begin, :, freq, :]))^inv(p)
    return [scale * norm((@view A[dish, :, freq, :]), p) for dish in 1:size(A, 1)]
end
function freq_norm(A, p=2)
    scale = p == Inf ? 1 : inv(length(@view A[:, :, begin, :]))^inv(p)
    return [scale * norm((@view A[:, :, freq, :]), p) for freq in 1:size(A, 3)]
end
function beam_norm(A, p=2, freq=:)
    scale = p == Inf ? 1 : inv(length(@view A[:, :, freq, begin]))^inv(p)
    return [scale * norm((@view A[:, :, freq, beam]), p) for beam in 1:size(A, 4)]
end

function aspect!(fig::Figure, row::Integer, col::Integer, ratio_x_over_y::Real)
    if ratio_x_over_y > 4 / 3
        rowsize!(fig.layout, row, Aspect(1, inv(ratio_x_over_y)))
    else
        colsize!(fig.layout, col, Aspect(1, ratio_x_over_y))
    end
    return nothing
end

prefix = "/tmp/fengine_pathfinder_test";
host = "indigo";
iter = "00000007"

freq = 1

dish_positions = read_kotekan("$(prefix)/$(host)_dish_positions.00000000.asdf", "dish_positions", ["EW/NS", "D"]);

E = read_kotekan("$(prefix)/$(host)_voltage.$(iter).asdf", "voltage", ["D", "P", "F", "T"]);
Enorm2 = dish_norm(E, 2, freq);

let
    fig = Figure(; size=(640, 480))
    ax = Axis(fig[1, 1]; title="Dishes (channel=$(freq-1))", xlabel="dish", ylabel="amplitude")
    ylims!(ax, (0, 10))
    barplot!(ax, 0:(length(Enorm2) - 1), Enorm2 .+ 0.1)
    display(fig)
end;

let
    fig = Figure(; size=(640, 480))
    ax = Axis(fig[1, 1]; title="Dishes (channel=$(freq-1))", xlabel="east-west [m]", ylabel="norh-south [m]")
    dishsize = 6                # dish diameter [m]
    min_ew, max_ew = extrema(@view dish_positions[1, :]) .+ (-dishsize, +dishsize)
    min_ns, max_ns = extrema(@view dish_positions[2, :]) .+ (-dishsize, +dishsize)
    xlims!(ax, min_ew, max_ew)
    ylims!(ax, min_ns, max_ns)
    obj = scatter!(
        ax,
        (@view dish_positions[1, :]),
        (@view dish_positions[2, :]);
        color=Enorm2,
        colormap=:plasma,
        colorrange=(0, 10),
        marker=Circle,
        markersize=dishsize,
        markerspace=:data,
    )
    Colorbar(fig[1, 2], obj; label="baseband beam intensity")
    aspect!(fig, 1, 1, (max_ew - min_ew) / (max_ns - min_ns))
    display(fig)
end;

beam_positions = read_kotekan("$(prefix)/$(host)_bb_beam_positions.00000000.asdf", "bb_beam_positions", ["EW/NS", "B"]);

A = read_kotekan("$(prefix)/$(host)_bb_phase.00000000.asdf", "bb_phase", ["C", "D", "B", "P", "F"]);
A = reinterpret(reshape, Complex{eltype(A)}, A);
s = read_kotekan("$(prefix)/$(host)_bb_shift.00000000.asdf", "bb_shift", ["B", "P", "F"]);

J = read_kotekan("$(prefix)/$(host)_bb_beams.$(iter).asdf", "bb_beams", ["T", "P", "F", "B"]);
Jnorm2 = beam_norm(J, 2, freq);

let
    fig = Figure(; size=(640, 480))
    ax = Axis(fig[1, 1]; title="Baseband beams (channel=$(freq-1))", xlabel="beam", ylabel="amplitude")
    ylims!(ax, (0, 10))
    barplot!(ax, 0:(length(Jnorm2) - 1), Jnorm2 .+ 0.1)
    display(fig)
end;

let
    fig = Figure(; size=(640, 480))
    ax = Axis(fig[1, 1]; title="Baseband beams (channel=$(freq-1))", xlabel="east-west [m]", ylabel="norh-south [m]")
    beamsize = 0.012            # beam diameter [rad]
    min_ew, max_ew = extrema(@view beam_positions[1, :]) .+ (-beamsize, +beamsize)
    min_ns, max_ns = extrema(@view beam_positions[2, :]) .+ (-beamsize, +beamsize)
    xlims!(ax, min_ew, max_ew)
    ylims!(ax, min_ns, max_ns)
    obj = scatter!(
        ax,
        (@view beam_positions[1, :]),
        (@view beam_positions[2, :]);
        color=Jnorm2,
        colormap=:plasma,
        colorrange=(0, 10),
        marker=Circle,
        markersize=beamsize,
        markerspace=:data,
    )
    Colorbar(fig[1, 2], obj; label="baseband beam intensity")
    aspect!(fig, 1, 1, (max_ew - min_ew) / (max_ns - min_ns))
    display(fig)
end;

nothing
