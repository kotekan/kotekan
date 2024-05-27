# This script reads HDF5 files produced by the config/tests/f_engine_chord_bb.yaml
# and makes amazing plots.
using ASDF2
using CUDASIMDTypes
using CairoMakie
# using ITerm2Images
using SixelTerm
using Statistics

# These should match settings in the YAML config file:
# - stage hdf5FileWrite's base_dir
setup = :pathfinder

dir = "/tmp/f_engine_$(setup)"
prefix = "indigo_"
iter = "00000000"

t2c(x::NTuple{2}) = Complex(x...)
i2t(x::Int4x2) = convert(NTuple{2,Int8}, x)
i2c(x::Int4x2) = t2c(i2t(x))

quantity_E = "voltage"
file_E = ASDF2.load_file("$(dir)/$(prefix)$(quantity_E).$(iter).asdf")
dataset_E = file_E.metadata[parse(Int, iter)]
@assert dataset_E["dim_names"] == ["T", "F", "P", "D"]
array_E = dataset_E["host_$(quantity_E)_buffer"][]
array_E::AbstractArray{UInt8,4}
array_E = reinterpret(Int4x2, array_E)
array_E::AbstractArray{Int4x2,4}
ndishs, npolrs, nfreqs, ntimes = size(array_E)

quantity_A = "bb_phase"
file_A = ASDF2.load_file("$(dir)/$(prefix)$(quantity_A).$(iter).asdf")
dataset_A = file_A.metadata[parse(Int, iter)]
@assert dataset_A["dim_names"] == ["F", "P", "B", "D", "C"]
array_A = dataset_A["host_$(quantity_A)_buffer"][]
array_A::AbstractArray{Int8,5}

# quantity_J0 = "expected_bb_beams"
# file_J0 = ASDF2.load_file("$(dir)/$(prefix)$(quantity_J0).$(iter).asdf")
# dataset_J0 = file_J0.metadata[parse(Int, iter)]
# @assert dataset_J0["dim_names"] == ["B", "F", "P", "T"]
# array_J0 = dataset_J0["host_$(quantity_J0)_buffer"][]
# array_J0::AbstractArray{UInt8,4}
# array_J0 = reinterpret(Int4x2, array_J0)
# array_J0::AbstractArray{Int4x2,4}
# ntimes′, npolrs′, nfreqs′, nbeams = size(array_J0)
# @assert (ntimes′, npolrs′, nfreqs′) == (ntimes, npolrs, nfreqs)

quantity_J = "bb_beams"
file_J = ASDF2.load_file("$(dir)/$(prefix)$(quantity_J).$(iter).asdf")
dataset_J = file_J.metadata[parse(Int, iter)]
@assert dataset_J["dim_names"] == ["B", "F", "P", "T"]
array_J = dataset_J["host_$(quantity_J)_buffer"][]
array_J::AbstractArray{UInt8,4}
array_J = reinterpret(Int4x2, array_J)
array_J::AbstractArray{Int4x2,4}
ntimes′, npolrs′, nfreqs′, nbeams = size(array_J)
@assert (ntimes′, npolrs′, nfreqs′, nbeams) == (ntimes, npolrs, nfreqs, nbeams)

# TODO: Read this from metadata
if setup === :chord
    ndishs_i = 32
    ndishs_j = 16
    dishsΔx = 6.3f0
    dishsΔy = 8.5f0
elseif setup === :pathfinder
    ndishs_i = 8
    ndishs_j = 8
    dishsΔx = 6.3f0
    dishsΔy = 8.5f0
elseif setup === :hirax
    ndishs_i = 16
    ndishs_j = 16
    dishsΔx = 6.5f0
    dishsΔy = 8.5f0
else
    @assert false
end

dishsi₀ = (ndishs_i - 1) / 2.0f0
dishsj₀ = (ndishs_j - 1) / 2.0f0
dishs_xlim = (dishsΔx * (0 - 1 / 2.0f0 - dishsi₀), dishsΔx * (ndishs_i - 1 + 1 / 2.0f0 - dishsi₀))
dishs_ylim = (dishsΔy * (0 - 1 / 2.0f0 - dishsj₀), dishsΔy * (ndishs_j - 1 + 1 / 2.0f0 - dishsj₀))
dishs_xsize = dishs_xlim[2] - dishs_xlim[1]
dishs_ysize = dishs_ylim[2] - dishs_ylim[1]

dishsx = Float32[]
dishsy = Float32[]
for dish in 1:ndishs
    dishi = (dish - 1) % ndishs_i
    dishj = (dish - 1) ÷ ndishs_i
    push!(dishsx, dishsΔx * (dishi - dishsi₀))
    push!(dishsy, dishsΔy * (dishj - dishsj₀))
end

# TODO: Read this from metadata
if setup === :chord
    nbeams_i = 12
    nbeams_j = 8
elseif setup === :pathfinder
    nbeams_i = 4
    nbeams_j = 4
elseif setup === :hirax
    nbeams_i = 4
    nbeams_j = 4
else
    @assert false
end
beamsΔx = 0.015f0
beamsΔy = 0.015f0

beamsi₀ = (nbeams_i - 1) / 2.0f0
beamsj₀ = (nbeams_j - 1) / 2.0f0
beams_xlim = (beamsΔx * (0 - 1 / 2.0f0 - beamsi₀), beamsΔx * (nbeams_i - 1 + 1 / 2.0f0 - beamsi₀))
beams_ylim = (beamsΔy * (0 - 1 / 2.0f0 - beamsj₀), beamsΔy * (nbeams_j - 1 + 1 / 2.0f0 - beamsj₀))
beams_xsize = beams_xlim[2] - beams_xlim[1]
beams_ysize = beams_ylim[2] - beams_ylim[1]

beamsx = Float32[]
beamsy = Float32[]
for beam in 1:nbeams
    beami = (beam - 1) % nbeams_i
    beamj = (beam - 1) ÷ nbeams_i
    push!(beamsx, beamsΔx * (beami - beamsi₀))
    push!(beamsy, beamsΔy * (beamj - beamsj₀))
end

# Most of the power is in frequency 1
# (TODO: Read this from metadata)
# freq = 1:nfreqs
freq = 1

function aspect!(fig::Figure, row::Integer, col::Integer, ratio_x_over_y)
    if ratio_x_over_y > 4 / 3
        rowsize!(fig.layout, row, Aspect(1, inv(ratio_x_over_y)))
    else
        colsize!(fig.layout, col, Aspect(1, ratio_x_over_y))
    end
    return nothing
end

# dataE = [abs2(Complex{Float32}(i2c(j))) for j in array_E];
# @show typeof(dataE)
# @show size(dataE)
# @show maximum.([@view dataE[:, :, f, :] for f in 1:nfreqs]);
# 
# dataJ = [abs2(Complex{Float32}(i2c(j))) for j in array_J];
# @show typeof(dataJ)
# @show size(dataJ)
# @show maximum.([@view dataJ[:, :, f, :] for f in 1:nfreqs]);

data = Float32[mean(abs2(real(Complex{Float32}(i2c(j)))) for j in view(array_E, dish, :, freq, :)) for dish in 1:ndishs]
# fig = Figure(; size=(1280, 960))
fig = Figure(; size=(640, 480))
ax = Axis(fig[1, 1]; title="$setup F-engine electric field", xlabel="x [m]", ylabel="y [m]")
xlims!(ax, dishs_xlim)
ylims!(ax, dishs_ylim)
obj = scatter!(ax, dishsx, dishsy; color=data, colormap=:plasma, markersize=480 / sqrt(2 * length(data)))
Colorbar(fig[1, 2], obj; label="|dish|₂")
# rowsize!(fig.layout, 1, Aspect(1, dishs_ysize / dishs_xsize))
# colsize!(fig.layout, 1, Aspect(1, dishs_xsize / dishs_ysize))
aspect!(fig, 1, 1, dishs_xsize / dishs_ysize)
display(fig)

data = Float32[
    mean(abs2(Complex{Float32}(i2c(j))) for j in view(array_J, :, polr, freq, beam)) for freq in 1:nfreqs, beam in 1:nbeams,
    polr in 1:npolrs
]
data = reshape(data, nfreqs, nbeams * npolrs)
# fig = Figure(; size=(1280, 960))
fig = Figure(; size=(640, 480))
ax = Axis(fig[1, 1]; title="$setup baseband beam spectra", xlabel="beam, polarization", ylabel="frequency channel")
xlims!(ax, (0 - 1 / 2, nbeams * npolrs - 1 / 2))
ylims!(ax, (0 - 1 / 2, nfreqs - 1 / 2))
obj = heatmap!(ax, 0:(nbeams * npolrs - 1), 0:(nfreqs - 1), data'; colormap=:plasma)
Colorbar(fig[1, 2], obj; label="baseband beam spectral intensity")
display(fig)

# data = Float32[mean(abs2(Complex{Float32}(i2c(j))) for j in view(array_J0, :, :, freq, beam)) for beam in 1:nbeams]
# fig = Figure(; size=(1280, 960))
# ax = Axis(fig[1, 1]; title="$setup expected baseband beams", xlabel="sky θx", ylabel="sky θy")
# xlims!(ax, beams_xlim)
# ylims!(ax, beams_ylim)
# obj = scatter!(ax, beamsx, beamsy; color=data, colormap=:plasma, markersize=960 / sqrt(2 * length(data)))
# Colorbar(fig[1, 2], obj; label="baseband beam intensity")
# # rowsize!(fig.layout, 1, Aspect(1, beams_ysize / beams_xsize))
# # colsize!(fig.layout, 1, Aspect(1, beams_xsize / beams_ysize))
# aspect!(fig, 1, 1, beams_xsize / beams_ysize)
# display(fig)

data = Float32[mean(abs2(Complex{Float32}(i2c(j))) for j in view(array_J, :, :, freq, beam)) for beam in 1:nbeams]
# fig = Figure(; size=(1280, 960))
fig = Figure(; size=(640, 480))
ax = Axis(fig[1, 1]; title="$setup X-engine baseband beams", xlabel="sky θx", ylabel="sky θy")
xlims!(ax, beams_xlim)
ylims!(ax, beams_ylim)
obj = scatter!(ax, beamsx, beamsy; color=data, colormap=:plasma, markersize=480 / sqrt(2 * length(data)))
Colorbar(fig[1, 2], obj; label="baseband beam intensity")
# rowsize!(fig.layout, 1, Aspect(1, beams_ysize / beams_xsize))
# colsize!(fig.layout, 1, Aspect(1, beams_xsize / beams_ysize))
aspect!(fig, 1, 1, beams_xsize / beams_ysize)
display(fig)

nothing
