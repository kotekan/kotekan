using CairoMakie
using Kotekan
using LinearAlgebra
using SixelTerm

function freq_time_norm(A, p=2)
    scale = p == Inf ? 1 : inv(length(@view A[:, :, begin, begin]))^inv(p)
    return [scale * norm((@view A[:, :, freq, time]), p) for freq in 1:size(A, 3), time in 1:size(A, 4)]
end

prefix = "/tmp/fengine_pathfinder_frb";
host = "indigo";
iter0 = "00000000"
iter1 = "00000001"

E0 = read_kotekan("$(prefix)/$(host)_voltage.$(iter0).asdf", "voltage", ["D", "P", "F", "T"]);
E1 = read_kotekan("$(prefix)/$(host)_voltage.$(iter1).asdf", "voltage", ["D", "P", "F", "T"]);
E = cat(E0, E1; dims=4);

Enorm2 = freq_time_norm(E, 2);

Enorm2small = Enorm2 |> A -> reshape(A, (size(A, 1), 32, :)) |> A -> maximum(A; dims=2) |> A -> reshape(A, (size(A, 1), size(A, 3)));

let
    fig = Figure(; size=(640, 480))
    ax = Axis(fig[1, 1]; title="FRB time/frequency diagram", xlabel="time [ms]", ylabel="local channel")
    yscale = 1e+3 * 16384 / 3200e6
    heatmap!(yscale * 32 * (0:(size(Enorm2small, 2) - 1)), 0:(size(Enorm2small, 1) - 1), permutedims(Enorm2small))
    display(fig)
end;
