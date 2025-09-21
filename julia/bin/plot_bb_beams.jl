using CairoMakie
using DimensionalData
using HDF5
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

function aspect!(fig::Figure, row::Integer, col::Integer, ratio_x_over_y::Real)
    if ratio_x_over_y > 4 / 3
        rowsize!(fig.layout, row, Aspect(1, inv(ratio_x_over_y)))
    else
        colsize!(fig.layout, col, Aspect(1, ratio_x_over_y))
    end
    return nothing
end

# function dish_norm(A, p=2, freq=:)
#     scale = p == Inf ? 1 : inv(length(@view A[begin, :, freq, :]))^inv(p)
#     return [scale * norm((@view A[dish, :, freq, :]), p) for dish in 1:size(A, 1)]
# end
# function freq_norm(A, p=2)
#     scale = p == Inf ? 1 : inv(length(@view A[:, :, begin, :]))^inv(p)
#     return [scale * norm((@view A[:, :, freq, :]), p) for freq in 1:size(A, 3)]
# end
# function beam_norm(A, p=2, freq=:)
#     scale = p == Inf ? 1 : inv(length(@view A[:, :, freq, begin]))^inv(p)
#     return [scale * norm((@view A[:, :, freq, beam]), p) for beam in 1:size(A, 4)]
# end

abs32(x) = abs(x)
abs32(x::Integer) = abs(Float32(x))
abs32(x::Complex{<:Integer}) = abs(Complex{Float32}(x))
abs232(x) = abs(x)
abs232(x::Integer) = abs2(Float32(x))
abs232(x::Complex{<:Integer}) = abs2(Complex{Float32}(x))

function LinearAlgebra.norm(A, p=2; dims)
    len = sum(Returns(1), A; dims=dims)[begin]
    if p == 1
        return maximum(abs32, A; dims=dims) / len
    elseif p == 2
        return sqrt.(sum(abs232, A; dims=dims) / len)
    else
        @assert false
    end
end



# telescope = "pathfinder";
telescope = "chord";
prefix = "/home/eschnett/src/kotekan/data/fengine_test_$(telescope)";
host = "cx66";

dish_index = let
    iterstring = "00000000"
    f = h5open("$(prefix)/$(host)_voltage.$(iterstring).h5")
    datasetname = keys(f)[begin]
    dataset = f[datasetname]
    n_dish_locations_ew = dataset["n_dish_locations_ew"][]
    n_dish_locations_ns = dataset["n_dish_locations_ns"][]
    DimArray(reshape(dataset["dish_index"][], (n_dish_locations_ew, n_dish_locations_ns)), (Dim{:EW}, Dim{:NS}))
end;

coarse_freq = let
    iterstring = "00000000"
    f = h5open("$(prefix)/$(host)_voltage.$(iterstring).h5")
    datasetname = keys(f)[begin]
    dataset = f[datasetname]
    dataset["coarse_freq"][]
end;



dish_positions = let
    iterstring = "00000000"
    read_hdf5("$(prefix)/$(host)_dish_positions.$(iterstring).h5")
end;

iters = 0:1;
E = let
    Es = [let
              iterstring = @sprintf "%08d" iter
              read_hdf5("$(prefix)/$(host)_voltage.$(iterstring).h5")
          end
          for iter in iters];
    cat_dimarrays(Es...)
end;

# Select one (or a range of) frequencies
# CHORD: freq ∈ [1, 3, 7, 12, 18, 26, 36]
freq = 1;
Enorm2 = norm(E[Dim{:F}(freq)]; dims=(:P, :T));
let
    fig = Figure(; size=(640, 480))
    ax = Axis(fig[1, 1]; title="Dishes (channel $(coarse_freq[freq]))", xlabel="dish", ylabel="amplitude")
    ylims!(ax, (0, 10))
    barplot!(ax, 0:(length(Enorm2) - 1), reshape(parent(Enorm2), :) .+ 0.1)
    display(fig)
end;

Enorm2 = norm(E; dims=(:D, :P, :T));
@assert length(Enorm2) == length(coarse_freq)
let
    fig = Figure(; size=(640, 480))
    ax = Axis(fig[1, 1]; title="Frequencies (channels)", xlabel="frequency channel", ylabel="voltage amplitude")
    ylims!(ax, (0, 10))
    barplot!(ax, coarse_freq, reshape(parent(Enorm2), :) .+ 0.1)
    display(fig)
end;

# Enorm2 = norm(E; dims=(:P, :F, :T));
Enorm2 = norm(E[Dim{:F}(freq)]; dims=(:P, :T));
let
    fig = Figure(; size=(640, 480))
    ax = Axis(fig[1, 1]; title="Dishes (channel $(coarse_freq[freq]))", xlabel="east-west [m]", ylabel="north-south [m]")
    dishsize = 6                # dish diameter [m]
    min_ew, max_ew = extrema(@view dish_positions[1, :]) .+ (-dishsize, +dishsize)
    min_ns, max_ns = extrema(@view dish_positions[2, :]) .+ (-dishsize, +dishsize)
    max_E = maximum(Enorm2)
    xlims!(ax, min_ew, max_ew)
    ylims!(ax, min_ns, max_ns)
    obj = scatter!(
        ax,
        parent(@view dish_positions[1, :]),
        parent(@view dish_positions[2, :]);
        color=reshape(parent(Enorm2), :),
        colormap=:plasma,
        colorrange=(0, 10),
        marker=Circle,
        markersize=dishsize,
        markerspace=:data,
    )
    Colorbar(fig[1, 2], obj; label="amplitude")
    aspect!(fig, 1, 1, (max_ew - min_ew) / (max_ns - min_ns))
    display(fig)
end;



beam_positions = read_hdf5("$(prefix)/$(host)_bb_beam_positions.00000000.h5");

# A = read_asdf("$(prefix)/$(host)_bb_phase.00000000.asdf", "bb_phase", ["C", "D", "B", "P", "F"]);
# A = reinterpret(reshape, Complex{eltype(A)}, A);
# s = read_asdf("$(prefix)/$(host)_bb_shift.00000000.asdf", "bb_shift", ["B", "P", "F"]);

iters = 0:1;
J = let
    Js = [let
              iterstring = @sprintf "%08d" iter
              read_hdf5("$(prefix)/$(host)_bb_beams.$(iterstring).h5")
          end
          for iter in iters];
    cat_dimarrays(Js...)
end;

Jnorm2 = norm(J[Dim{:F}(freq)]; dims=(:P, :T));
let
    fig = Figure(; size=(640, 480))
    ax = Axis(fig[1, 1]; title="Baseband beams (channel $(coarse_freq[freq]))", xlabel="beam", ylabel="amplitude")
    ylims!(ax, (0, 10))
    barplot!(ax, 0:(length(Jnorm2) - 1), reshape(parent(Jnorm2), :) .+ 0.1)
    display(fig)
end;

Jnorm2 = norm(J; dims=(:B, :P, :T));
@assert length(Jnorm2) == length(coarse_freq)
let
    fig = Figure(; size=(640, 480))
    ax = Axis(fig[1, 1]; title="Frequencies (channels)", xlabel="frequency channel", ylabel="baseband amplitude")
    ylims!(ax, (0, 10))
    barplot!(ax, coarse_freq, reshape(parent(Jnorm2), :) .+ 0.1)
    display(fig)
end;

# Jnorm2 = norm(J; dims=(:P, :F, :T));
Jnorm2 = norm(J[Dim{:F}(freq)]; dims=(:P, :T));
let
    fig = Figure(; size=(640, 480))
    ax = Axis(fig[1, 1]; title="Baseband beams (channel $(coarse_freq[freq]))", xlabel="east-west [rad]", ylabel="north-south [rad]")
    beamsize_ew = 0.015 / 2
    beamsize_ns = 0.010 / 2
    # # CHORD
    # c₀ = 299_792_458
    # adc_freq = 3.2e+9
    # num_samples_per_frame = 16384
    # dish_separation_ew = 6.3
    # dish_separation_ns = 8.5
    # 
    # lambda₀ = c₀ / (adc_freq / num_samples_per_frame)
    # channel = 7680 # coarse_freq[freq]
    # lambda = lambda₀ / channel
    # beamsize_ew = lambda / dish_separation_ew
    # beamsize_ns = lambda / dish_separation_ns

    min_ew, max_ew = extrema(@view beam_positions[1, :]) .+ (-beamsize_ew, +beamsize_ew)
    min_ns, max_ns = extrema(@view beam_positions[2, :]) .+ (-beamsize_ns, +beamsize_ns)
    xlims!(ax, min_ew, max_ew)
    ylims!(ax, min_ns, max_ns)
    obj = scatter!(
        ax,
        parent(@view beam_positions[1, :]),
        parent(@view beam_positions[2, :]);
        color=reshape(parent(Jnorm2), :),
        colormap=:plasma,
        colorrange=(0, 10),
        marker=Circle,
        markersize=(beamsize_ew, beamsize_ns),
        markerspace=:data,
    )
    Colorbar(fig[1, 2], obj; label="baseband beam intensity")
    aspect!(fig, 1, 1, (max_ew - min_ew) / (max_ns - min_ns))
    display(fig)
end;



nothing
