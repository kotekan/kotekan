using CairoMakie
using Kotekan
using LinearAlgebra
using SixelTerm

function freq_norm(A, p=2)
    scale = p == Inf ? 1 : 1 / length(@view A[:, :, begin, :])^(1 / p)
    return [scale * norm((@view A[:, :, freq, :]), p) for freq in 1:size(A, 3)]
end

prefix = "/tmp/fengine_pathfinder_test";
host = "indigo";
iter = "00000007"

E = read_kotekan("$(prefix)/$(host)_voltage.$(iter).asdf", "voltage", ["D", "P", "F", "T"]);

Enorm2 = freq_norm(E, 2);

let
    fig = Figure(; size=(640, 480))
    ax = Axis(fig[1, 1]; title="Coarse channels", xlabel="local channel", ylabel="amplitude")
    ylims!(ax, (0, 10))
    barplot!(ax, 0:(length(Enorm2) - 1), Enorm2 .+ 0.01)
    display(fig)
end;

Us = [2, 4, 8, 16, 32, 64];

Ebar = Dict(
    U => read_kotekan("$(prefix)/$(host)_upchan_U$(U)_voltage.$(iter).asdf", "upchan_U$(U)_voltage", ["D", "P", "Fbar", "Tbar"]) for
    U in Us
);

Ebarnorm2 = Dict(U => freq_norm(Ebar[U], 2) for U in Us);

for U in Us
    fig = Figure(; size=(640, 480))
    ax = Axis(fig[1, 1]; title="Fine channels U=$U", xlabel="fine channel", ylabel="amplitude")
    ylims!(ax, (0, 10))
    barplot!(ax, 0:(U - 1), Ebarnorm2[U][1:U] .+ 0.01)
    display(fig)
end;
