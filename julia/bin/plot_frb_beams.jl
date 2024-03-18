using ASDF2
using CUDASIMDTypes
using CairoMakie
using SixelTerm
using Statistics

setup = :pathfinder

dir = "/tmp/f_engine_$(setup)"
prefix = "indigo_"
iter = "00000000"

t2c(x::NTuple{2}) = Complex(x...)
i2t(x::Int4x2) = convert(NTuple{2,Int8}, x)
i2c(x::Int4x2) = t2c(i2t(x))

quantity_Ebar = "upchan_voltage"
file_Ebar = ASDF2.load_file("$(dir)/$(prefix)$(quantity_Ebar).$(iter).asdf")
dataset_Ebar = file_Ebar.metadata[parse(Int, iter)]
@assert dataset_Ebar["dim_names"] == ["Tbar", "Fbar", "P", "D"]
array_Ebar = dataset_Ebar["host_$(quantity_Ebar)_buffer"][]
array_Ebar::AbstractArray{UInt8,4}
array_Ebar = reinterpret(Int4x2, array_Ebar)
array_Ebar::AbstractArray{Int4x2,4}
ndishs, npolrs, nfreqs, ntimes = size(array_Ebar)

quantity_W = "frb_phase"
file_W = ASDF2.load_file("$(dir)/$(prefix)$(quantity_W).$(iter).asdf")
dataset_W = file_W.metadata[parse(Int, iter)]
@assert dataset_W["dim_names"] == ["Fbar", "P", "dishN", "dishM", "C"]
array_W = dataset_W["host_$(quantity_W)_buffer"][]
array_W::AbstractArray{Float16,5}
array_W = reinterpret(Complex{Float16}, array_W)
@assert size(array_W, 1) == 1
array_W = reshape(array_W, size(array_W)[2:end])
array_W::AbstractArray{Complex{Float16},4}

# quantity_I0 = "expected_frb_beams"
# file_I0 = ASDF2.load_file("$(dir)/$(prefix)$(quantity_I0).$(iter).asdf")
# dataset_I0 = file_I0.metadata[parse(Int, iter)]
# @assert dataset_I0["dim_names"] == ["Ttilde", "Fbar", "beamQ", "beamP"]
# array_I0 = dataset_I0["host_$(quantity_I0)_buffer"][]
# array_I0::AbstractArray{Float16,4}
# nbeamps, nbeamqs, ntimebars, nfreqs′ = size(array_I0)
# @assert nfreqs′ == nfreqs

quantity_I = "frb_beams"
file_I = ASDF2.load_file("$(dir)/$(prefix)$(quantity_I).$(iter).asdf")
dataset_I = file_I.metadata[parse(Int, iter)]
@assert dataset_I["dim_names"] == ["Ttilde", "Fbar", "beamQ", "beamP"]
array_I = dataset_I["host_$(quantity_I)_buffer"][]
array_I::AbstractArray{Float16,4}
nbeamps, nbeamqs, ntimebars, nfreqs = size(array_I)
@assert nfreqs == nfreqs

array_I = map(x -> x == Inf ? prevfloat(typemax(x)) : x, array_I)

@assert dataset_Ebar["ndishes"] == ndishs

dish_index = dataset_Ebar["dish_index"][]
dish_index::AbstractArray{<:Integer,2}
num_dish_locations_N, num_dish_locations_M = size(dish_index)

dish_locations = fill((-1, -1), ndishs)
for locM in 0:(num_dish_locations_M - 1), locN in 0:(num_dish_locations_N - 1)
    dish = dish_index[locN + 1, locM + 1]
    if dish >= 0
        @assert dish_locations[dish + 1] == (-1, -1)
        dish_locations[dish + 1] = (locM, locN)
    end
end

dishsΔx = 6.3f0
dishsΔy = 8.5f0

ndishs_i = num_dish_locations_M
ndishs_j = num_dish_locations_N
dishsi₀ = (ndishs_i - 1) / 2.0f0
dishsj₀ = (ndishs_j - 1) / 2.0f0
dishs_xlim = (dishsΔx * (0 - 1 / 2.0f0 - dishsj₀), dishsΔx * (ndishs_j - 1 + 1 / 2.0f0 - dishsj₀))
dishs_ylim = (dishsΔy * (0 - 1 / 2.0f0 - dishsi₀), dishsΔy * (ndishs_i - 1 + 1 / 2.0f0 - dishsi₀))
dishs_xsize = dishs_xlim[2] - dishs_xlim[1]
dishs_ysize = dishs_ylim[2] - dishs_ylim[1]

dishsx = Float32[]
dishsy = Float32[]
for dish in 1:ndishs
    dishi = dish_locations[dish][1]
    dishj = dish_locations[dish][2]
    push!(dishsx, +dishsΔx * (dishj - dishsj₀))
    push!(dishsy, -dishsΔy * (dishi - dishsi₀))
end

frb_num_beams_P = 2 * num_dish_locations_M
frb_num_beams_Q = 2 * num_dish_locations_N
nbeams_i = frb_num_beams_P
nbeams_j = frb_num_beams_Q
nbeams = nbeams_i * nbeams_j
beamsΔx = 0.015f0               # ???
beamsΔy = 0.015f0

beamsi₀ = (nbeams_i - 1) / 2.0f0
beamsj₀ = (nbeams_j - 1) / 2.0f0
beams_xlim = (beamsΔx * (0 - 1 / 2.0f0 - beamsj₀), beamsΔx * (nbeams_j - 1 + 1 / 2.0f0 - beamsj₀))
beams_ylim = (beamsΔy * (0 - 1 / 2.0f0 - beamsi₀), beamsΔy * (nbeams_i - 1 + 1 / 2.0f0 - beamsi₀))
beams_xsize = beams_xlim[2] - beams_xlim[1]
beams_ysize = beams_ylim[2] - beams_ylim[1]

beamsx = Float32[]
beamsy = Float32[]
for beam in 1:nbeams
    beami = (beam - 1) % nbeams_i
    beamj = (beam - 1) ÷ nbeams_i
    push!(beamsx, beamsΔx * (beamj - beamsj₀))
    push!(beamsy, beamsΔy * (beami - beamsi₀))
end

# Most of the power is in frequency 89 (???)
# (TODO: Read this from metadata)
# freq = 1:nfreqs
if setup === :chord
    freq = 48
elseif setup === :pathfinder
    freq = 408
elseif setup === :hirax
    freq = 384
else
    @assert false
end

function aspect!(fig::Figure, row::Integer, col::Integer, ratio_x_over_y)
    if ratio_x_over_y > 4 / 3
        rowsize!(fig.layout, row, Aspect(1, inv(ratio_x_over_y)))
    else
        colsize!(fig.layout, col, Aspect(1, ratio_x_over_y))
    end
    return nothing
end

data = Float32[mean(abs2(real(Complex{Float32}(i2c(j)))) for j in view(array_Ebar, dish, :, freq, :)) for dish in 1:ndishs]
fig = Figure(; size=(1280, 960))
ax = Axis(fig[1, 1]; title="$setup F-engine electric field", xlabel="x [m]", ylabel="y [m]")
xlims!(ax, dishs_xlim)
ylims!(ax, dishs_ylim)
obj = scatter!(ax, dishsx, dishsy; color=data, colormap=:plasma, markersize=960 / sqrt(2 * length(data)))
Colorbar(fig[1, 2], obj; label="|dish|₂")
# rowsize!(fig.layout, 1, Aspect(1, dishs_ysize / dishs_xsize))
# colsize!(fig.layout, 1, Aspect(1, dishs_xsize / dishs_ysize))
@show dishs_xsize dishs_ysize
aspect!(fig, 1, 1, dishs_xsize / dishs_ysize)
display(fig)

freq_step = cld(nfreqs, 480)
data = Float32[
    mean(Float32(i) for i in view(array_I, beamp, beamq, :, freq:(min(nfreqs, freq + freq_step - 1)))) for beamp in 1:nbeamps,
    beamq in 1:nbeamqs, freq in 1:freq_step:nfreqs
]
data = reshape(data, nbeamps * nbeamqs, cld(nfreqs, freq_step))
fig = Figure(; size=(1280, 960))
ax = Axis(fig[1, 1]; title="$setup frb beam spectra", xlabel="beam", ylabel="frequency channel")
xlims!(ax, (0 - 1 / 2, nbeamps * nbeamqs - 1 / 2))
ylims!(ax, (0 - 1 / 2, nfreqs - 1 / 2))
obj = heatmap!(ax, 0:(nbeamps * nbeamqs - 1), 0:freq_step:(nfreqs - 1), data; color=data, colormap=:plasma)
Colorbar(fig[1, 2], obj; label="frb beam spectral intensity")
display(fig)

# data = Float32[mean(Float32(i) for i in view(array_I0, beamp, beamq, :, freq)) for beamq in 1:nbeamqs for beamp in 1:nbeamps]
# fig = Figure(; size=(1280, 960))
# ax = Axis(fig[1, 1]; title="$setup expected X-engine frb beams", xlabel="sky θx", ylabel="sky θy")
# xlims!(ax, beams_xlim)
# ylims!(ax, beams_ylim)
# obj = scatter!(ax, beamsx, beamsy; color=data, colormap=:plasma, markersize=960 / sqrt(2 * length(data)))
# Colorbar(fig[1, 2], obj; label="frb beam intensity")
# # rowsize!(fig.layout, 1, Aspect(1, beams_ysize / beams_xsize))
# # colsize!(fig.layout, 1, Aspect(1, beams_xsize / beams_ysize))
# aspect!(fig, 1, 1, beams_xsize / beams_ysize)
# display(fig)

data = Float32[mean(Float32(i) for i in view(array_I, beamp, beamq, :, freq)) for beamq in 1:nbeamqs for beamp in 1:nbeamps]
fig = Figure(; size=(1280, 960))
ax = Axis(fig[1, 1]; title="$setup X-engine frb beams", xlabel="sky θx", ylabel="sky θy")
xlims!(ax, beams_xlim)
ylims!(ax, beams_ylim)
obj = scatter!(ax, beamsx, beamsy; color=data, colormap=:plasma, markersize=960 / sqrt(2 * length(data)))
Colorbar(fig[1, 2], obj; label="frb beam intensity")
# rowsize!(fig.layout, 1, Aspect(1, beams_ysize / beams_xsize))
# colsize!(fig.layout, 1, Aspect(1, beams_xsize / beams_ysize))
aspect!(fig, 1, 1, beams_xsize / beams_ysize)
display(fig)

nothing
