using CairoMakie
using Kotekan
using LinearAlgebra
using SixelTerm

function freq_norm(A, p=2)
    scale = p == Inf ? 1 : inv(length(@view A[:, :, begin, :]))^inv(p)
    return [scale * norm((@view A[:, :, freq, :]), p) for freq in 1:size(A, 3)]
end

prefix = "/tmp/fengine_pathfinder_test";
host = "indigo";
U = 64;

E = read_kotekan("$(prefix)/$(host)_voltage.00000000.asdf", "voltage", ["D", "P", "F", "T"]);

Enorm1 = freq_norm(E, 1);
Enorm2 = freq_norm(E, 2);
Enorminf = freq_norm(E, Inf);
barplot(Enorm2[1:20] .+ 0.01)

Ebar = read_kotekan("$(prefix)/$(host)_upchan_U$(U)_voltage.00000000.asdf", "upchan_U$(U)_voltage", ["D", "P", "Fbar", "Tbar"]);

Ebarnorm1 = freq_norm(Ebar, 1);
Ebarnorm2 = freq_norm(Ebar, 2);
Ebarnorminf = freq_norm(Ebar, Inf);
N = 1 * U   # 23*U
# barplot(Ebarnorm2[1:N])
barplot([u ÷ U + 1 for u in 0:(N - 1)], Ebarnorm2[1:N] .+ 0.01; dodge=[u % U + 1 for u in 0:(N - 1)], dodge_gap=2 / U)
barplot([u ÷ U + 1 for u in 0:(N - 1)], Ebarnorm2[N .+ (1:N)] .+ 0.01; dodge=[u % U + 1 for u in 0:(N - 1)], dodge_gap=2 / U)

###

scatterlines(real.(i2c.(array_E[1, 1, Emaxind[2], 1:100])))

include("src/FEngine.jl")
using .FEngine

y = FEngine.upchan(Complex{Float32}.(i2c.(array_E[1, 1, Emaxind[2], :])), 4, 2);
maximum(abs, y; dims=2)

# lines(imag.(i2c.(array_Ebar[1, 1, 12, 1:100])))
# lines(imag.(y[2, 1:100]))
