using CairoMakie
using DimensionalData
using Kotekan
using LinearAlgebra
using Printf
using SixelTerm

function cat_dimarrays(A::DimArray{T,D}, Bs::DimArray{T,D}...) where {T,D}
    dimnames = ntuple(d -> name(dims(A, d)), D)
    for B in Bs
        @assert ntuple(d -> name(dims(B, d)), D) == dimnames
    end
    data = cat(parent(A), [parent(B) for B in Bs]...; dims=D)
    R = DimArray(data, dimnames)
    return R
end

function freq_time_norm(A, p=2)
    scale = p == Inf ? 1 : inv(length(@view A[:, :, begin, begin]))^inv(p)
    return [scale * norm((@view A[:, :, freq, time]), p) for freq in 1:size(A, 3), time in 1:size(A, 4)]
end

function I_freq_time_norm(A, p=2)
    scale = p == Inf ? 1 : inv(length(@view A[begin, begin, :]))^inv(p)
    return [scale * norm((@view A[time, freq, :]), p) for time in 1:size(A, 1), freq in 1:size(A, 2)]
end

prefix = "/home/eschnett/src/kotekan/data/fengine_frb_pathfinder";
host = "cx66";



iters = 0:1;
E = let
    Es = [let
              iterstring = @sprintf "%08d" iter
              read_hdf5("$(prefix)/$(host)_voltage.$(iterstring).h5")
          end
          for iter in iters];
    cat_dimarrays(Es...)
end;

Enorm2 = freq_time_norm(E, 2);
Enorm2small = Enorm2 |> A -> reshape(A, (size(A, 1), 32, :)) |> A -> maximum(A; dims=2) |> A -> reshape(A, (size(A, 1), size(A, 3)));
let
    fig = Figure(; size=(640, 480))
    ax = Axis(fig[1, 1]; title="FRB time/frequency diagram", xlabel="time [ms]", ylabel="local channel")
    # TODO: check time scale
    tscale = 1e+3 * 16384 / 3200e+6 # [ms]
    obj = heatmap!(
        tscale * 32 * (0:(size(Enorm2small, 2) - 1)),
        0:(size(Enorm2small, 1) - 1),
        permutedims(Enorm2small);
        colormap=:plasma,
        colorrange=(0, √2 * 7),
    )
    Colorbar(fig[1, 2], obj; label="voltage intensity")
    display(fig)
end;



# TODO: check upchannelized voltage, frb 1 beams


iters = 0:0;
I = let
    Is = [let
              iterstring = @sprintf "%08d" iter
              read_hdf5("$(prefix)/$(host)_frb2_beams.$(iterstring).h5")
          end
          for iter in iters];
    cat_dimarrays(Is...)
end;

Inorm2 = I_freq_time_norm(I, 2);
let
    fig = Figure(; size=(640, 480))
    ax = Axis(fig[1, 1]; title="FRB time/frequency diagram", xlabel="time [ms]", ylabel="local channel")
    # TODO: fix time scale
    tscale = 1e+3 * 16384 / 3200e+6 # [ms]
    obj = heatmap!(
        tscale * 32 * (0:(size(Inorm2, 2) - 1)),
        0:(size(Inorm2, 1) - 1),
        #=permutedims=#(Inorm2);
        colormap=:plasma,
        colorrange=(0, √2 * 7),
    )
    Colorbar(fig[1, 2], obj; label="frb2 beams")
    display(fig)
end;



nothing
