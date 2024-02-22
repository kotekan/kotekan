using ASDF2
using CUDASIMDTypes
using CairoMakie
using SixelTerm

setup = :pathfinder

dir = "/tmp/f_engine_$(setup)_upchan"
prefix = "blue_"
iter = "00000000"

t2c(x::NTuple{2}) = Complex(x...)
i2t(x::Int4x2) = convert(NTuple{2,Int8}, x)
i2c(x::Int4x2) = t2c(i2t(x))

# TODO: Read this from metadata
U = 16

quantity_E = "voltage"
file_E = ASDF2.load_file("$(dir)/$(prefix)$(quantity_E).$(iter).asdf")
dataset_E = file_E.metadata[parse(Int, iter)]
@assert dataset_E["dim_names"] == ["T", "F", "P", "D"]
array_E = dataset_E["host_$(quantity_E)_buffer"][]
array_E::AbstractArray{UInt8,4}
array_E = reinterpret(Int4x2, array_E)
array_E::AbstractArray{Int4x2,4}
ndishs, npolrs, nfreqs, ntimes = size(array_E)

quantity_G = "upchan_gain"
file_G = ASDF2.load_file("$(dir)/$(prefix)$(quantity_G).$(iter).asdf")
dataset_G = file_G.metadata[parse(Int, iter)]
@assert dataset_G["dim_names"] == ["Fbar"]
array_G = dataset_G["host_$(quantity_G)_buffer"][]
# array_G::AbstractArray{UInt16,1}
# array_G = reinterpret(Float16, array_G)
array_G::AbstractArray{Float16,1}
ngains, = size(array_G)

@assert ngains == nfreqs * U

quantity_Ebar = "upchan_voltage"
file_Ebar = ASDF2.load_file("$(dir)/$(prefix)$(quantity_Ebar).$(iter).asdf")
dataset_Ebar = file_Ebar.metadata[parse(Int, iter)]
@assert dataset_Ebar["dim_names"] == ["Tbar", "Fbar", "P", "D"]
array_Ebar = dataset_Ebar["host_$(quantity_Ebar)_buffer"][]
array_Ebar::AbstractArray{UInt8,4}
array_Ebar = reinterpret(Int4x2, array_Ebar)
array_Ebar::AbstractArray{Int4x2,4}
ndishs′, npolrs′, nfreqbars, ntimebars = size(array_Ebar)

@assert (ndishs′, npolrs′, nfreqbars) == (ndishs, npolrs, nfreqs * U)
@assert ntimebars <= ntimes ÷ U

@assert dataset_E["ndishes"] == ndishs

dish_index = dataset_E["dish_index"][]
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

# Most of the power is in frequency 3
# (TODO: Read this from metadata)
# freq = 1:nfreqs
freq = 3
freqbar = 1:nfreqbars
# freqbar = 3 * U

function aspect!(fig::Figure, row::Integer, col::Integer, ratio_x_over_y)
    if ratio_x_over_y > 4 / 3
        rowsize!(fig.layout, row, Aspect(1, inv(ratio_x_over_y)))
    else
        colsize!(fig.layout, col, Aspect(1, ratio_x_over_y))
    end
    return nothing
end

data = Float32[
    sqrt(
        sum(abs2(real(Complex{Float32}(i2c(j)))) for j in view(array_E, dish, :, freq, :)) / length(view(array_E, dish, :, freq, :))
    ) for dish in 1:ndishs
]
fig = Figure(; size=(1280, 960))
ax = Axis(fig[1, 1]; title="$setup X-engine electric field", xlabel="x", ylabel="y")
xlims!(ax, dishs_xlim)
ylims!(ax, dishs_ylim)
obj = scatter!(ax, dishsx, dishsy; color=data, colormap=:plasma, markersize=960 / sqrt(2 * length(data)))
Colorbar(fig[1, 2], obj; label="|dish|₂")
aspect!(fig, 1, 1, dishs_xsize / dishs_ysize)
display(fig)

data = Float32[
    sqrt(
        sum(abs2(real(Complex{Float32}(i2c(j)))) for j in view(array_Ebar, dish, :, freqbar, :)) /
        length(view(array_Ebar, dish, :, freqbar, :)),
    ) for dish in 1:ndishs
]
fig = Figure(; size=(1280, 960))
ax = Axis(fig[1, 1]; title="$setup X-engine upchannelized electric field (U=$U)", xlabel="x", ylabel="y")
xlims!(ax, dishs_xlim)
ylims!(ax, dishs_ylim)
obj = scatter!(ax, dishsx, dishsy; color=data, colormap=:plasma, markersize=960 / sqrt(2 * length(data)))
Colorbar(fig[1, 2], obj; label="|dish|₂")
aspect!(fig, 1, 1, dishs_xsize / dishs_ysize)
display(fig)

nothing
