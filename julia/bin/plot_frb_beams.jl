using ASDF2
using CUDASIMDTypes
using CairoMakie
using SixelTerm

dir = "/tmp/f_engine_pathfinder_frb"
prefix = "blue_"
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

quantity_W = "frb_phase"
file_W = ASDF2.load_file("$(dir)/$(prefix)$(quantity_W).$(iter).asdf")
dataset_W = file_W.metadata[parse(Int, iter)]
@assert dataset_W["dim_names"] == ["F", "P", "dishN", "dishM", "C"]
array_W = dataset_W["host_$(quantity_W)_buffer"][]
array_W::AbstractArray{Float16,5}
array_W = reinterpret(Complex{Float16}, array_W)
@assert size(array_W, 1) == 1
array_W = reshape(array_W, size(array_W)[2:end])
array_W::AbstractArray{Complex{Float16},4}

quantity_I = "frb_intensity"
file_I = ASDF2.load_file("$(dir)/$(prefix)$(quantity_I).$(iter).asdf")
dataset_I = file_I.metadata[parse(Int, iter)]
@assert dataset_I["dim_names"] == ["F", "Tbar", "beamQ", "beamP"]
array_I = dataset_I["host_$(quantity_I)_buffer"][]
array_I::AbstractArray{Float16,4}
array_I = reinterpret(Float16, array_I)
array_I::AbstractArray{Float16,4}
nbeamps, nbeamqs, ntimebars, nfreqs′ = size(array_I)
@assert nfreqs′ == nfreqs

array_I = map(x -> x == Inf ? prevfloat(typemax(x)) : x, array_I)

# TODO: Read this from metadata
if false
    # CHORD
    num_dish_locations_M = 24
    num_dish_locations_N = 24
    #! format: off
    dish_locations = Int[
        0,0,0,1,0,2,0,3,0,4,0,5,0,6,0,7,0,8,0,9,0,10,0,11,0,12,0,13,0,14,0,15,0,16,0,17,0,18,0,19,0,20,0,21,0,22,0,23,
        1,0,1,1,1,2,1,3,1,4,1,5,1,6,1,7,1,8,1,9,1,10,1,11,1,12,1,13,1,14,1,15,1,16,1,17,1,18,1,19,1,20,1,21,1,22,1,23,
        2,0,2,1,2,2,2,3,2,4,2,5,2,6,2,7,2,8,2,9,2,10,2,11,2,12,2,13,2,14,2,15,2,16,2,17,2,18,2,19,2,20,2,21,2,22,2,23,
        3,0,3,1,3,2,3,3,3,4,3,5,3,6,3,7,3,8,3,9,3,10,3,11,3,12,3,13,3,14,3,15,3,16,3,17,3,18,3,19,3,20,3,21,3,22,3,23,
        4,0,4,1,4,2,4,3,4,4,4,5,4,6,4,7,4,8,4,9,4,10,4,11,4,12,4,13,4,14,4,15,4,16,4,17,4,18,4,19,4,20,4,21,4,22,4,23,
        5,0,5,1,5,2,5,3,5,4,5,5,5,6,5,7,5,8,5,9,5,10,5,11,5,12,5,13,5,14,5,15,5,16,5,17,5,18,5,19,5,20,5,21,5,22,5,23,
        6,0,6,1,6,2,6,3,6,4,6,5,6,6,6,7,6,8,6,9,6,10,6,11,6,12,6,13,6,14,6,15,6,16,6,17,6,18,6,19,6,20,6,21,6,22,6,23,
        7,0,7,1,7,2,7,3,7,4,7,5,7,6,7,7,7,8,7,9,7,10,7,11,7,12,7,13,7,14,7,15,7,16,7,17,7,18,7,19,7,20,7,21,7,22,7,23,
        8,0,8,1,8,2,8,3,8,4,8,5,8,6,8,7,8,8,8,9,8,10,8,11,8,12,8,13,8,14,8,15,8,16,8,17,8,18,8,19,8,20,8,21,8,22,8,23,
        9,0,9,1,9,2,9,3,9,4,9,5,9,6,9,7,9,8,9,9,9,10,9,11,9,12,9,13,9,14,9,15,9,16,9,17,9,18,9,19,9,20,9,21,9,22,9,23,
        10,0,10,1,10,2,10,3,10,4,10,5,10,6,10,7,10,8,10,9,10,10,10,11,10,12,10,13,10,14,10,15,10,16,10,17,10,18,10,19,10,20,10,21,10,22,10,23,
        11,0,11,1,11,2,11,3,11,4,11,5,11,6,11,7,11,8,11,9,11,10,11,11,11,12,11,13,11,14,11,15,11,16,11,17,11,18,11,19,11,20,11,21,11,22,11,23,
        12,0,12,1,12,2,12,3,12,4,12,5,12,6,12,7,12,8,12,9,12,10,12,11,12,12,12,13,12,14,12,15,12,16,12,17,12,18,12,19,12,20,12,21,12,22,12,23,
        13,0,13,1,13,2,13,3,13,4,13,5,13,6,13,7,13,8,13,9,13,10,13,11,13,12,13,13,13,14,13,15,13,16,13,17,13,18,13,19,13,20,13,21,13,22,13,23,
        14,0,14,1,14,2,14,3,14,4,14,5,14,6,14,7,14,8,14,9,14,10,14,11,14,12,14,13,14,14,14,15,14,16,14,17,14,18,14,19,14,20,14,21,14,22,14,23,
        15,0,15,1,15,2,15,3,15,4,15,5,15,6,15,7,15,8,15,9,15,10,15,11,15,12,15,13,15,14,15,15,15,16,15,17,15,18,15,19,15,20,15,21,15,22,15,23,
        16,0,16,1,16,2,16,3,16,4,16,5,16,6,16,7,16,8,16,9,16,10,16,11,16,12,16,13,16,14,16,15,16,16,16,17,16,18,16,19,16,20,16,21,16,22,16,23,
        17,0,17,1,17,2,17,3,17,4,17,5,17,6,17,7,17,8,17,9,17,10,17,11,17,12,17,13,17,14,17,15,17,16,17,17,17,18,17,19,17,20,17,21,17,22,17,23,
        18,0,18,1,18,2,18,3,18,4,18,5,18,6,18,7,18,8,18,9,18,10,18,11,18,12,18,13,18,14,18,15,18,16,18,17,18,18,18,19,18,20,18,21,18,22,18,23,
        19,0,19,1,19,2,19,3,19,4,19,5,19,6,19,7,19,8,19,9,19,10,19,11,19,12,19,13,19,14,19,15,19,16,19,17,19,18,19,19,19,20,19,21,19,22,19,23,
        20,0,20,1,20,2,20,3,20,4,20,5,20,6,20,7,20,8,20,9,20,10,20,11,20,12,20,13,20,14,20,15,20,16,20,17,20,18,20,19,20,20,20,21,20,22,20,23,
        21,0,21,1,21,2,21,3,21,4,21,5,21,6,21,7,21,8,21,9,21,10,21,11,21,12,21,13,21,14,21,15,21,16,21,17,21,18,21,19,21,20,21,21,21,22,21,23,
        22,0,22,1,22,2,22,3,22,4,22,5,22,6,22,7,22,8,22,9,22,10,22,11,22,12,22,13,22,14,22,15,22,16,22,17,22,18,22,19,22,20,22,21,22,22,22,23,
        23,0,23,1,23,2,23,3,23,4,23,5,23,6,23,7,23,8,23,9,23,10,23,11,23,12,23,13,23,14,23,15,23,16,23,17,23,18,23,19,23,20,23,21,23,22,23,23,
    ]
    #! format: on
else
    # Pathfinder
    num_dish_locations_M = 8
    num_dish_locations_N = 12
    dish_locations = Int[
        0,
        0,
        0,
        1,
        0,
        2,
        0,
        3,
        0,
        4,
        0,
        5,
        0,
        6,
        0,
        7,
        0,
        8,
        0,
        9,
        0,
        10,
        0,
        11,
        1,
        0,
        1,
        1,
        1,
        2,
        1,
        3,
        1,
        4,
        1,
        5,
        1,
        6,
        1,
        7,
        1,
        8,
        1,
        9,
        1,
        10,
        1,
        11,
        2,
        0,
        2,
        1,
        2,
        2,
        2,
        3,
        2,
        4,
        2,
        5,
        2,
        6,
        2,
        7,
        2,
        8,
        2,
        9,
        2,
        10,
        2,
        11,
        3,
        0,
        3,
        1,
        3,
        2,
        3,
        3,
        3,
        4,
        3,
        5,
        3,
        6,
        3,
        7,
        3,
        8,
        3,
        9,
        3,
        10,
        3,
        11,
        4,
        0,
        4,
        1,
        4,
        2,
        4,
        3,
        4,
        4,
        4,
        5,
        4,
        6,
        4,
        7,
        4,
        8,
        4,
        9,
        4,
        10,
        4,
        11,
        5,
        0,
        5,
        1,
        5,
        2,
        5,
        3,
        5,
        4,
        5,
        5,
        5,
        6,
        5,
        7,
        5,
        8,
        5,
        9,
        5,
        10,
        5,
        11,
        6,
        0,
        6,
        1,
        6,
        2,
        6,
        3,
        6,
        4,
        6,
        5,
        6,
        6,
        6,
        7,
        6,
        8,
        6,
        9,
        6,
        10,
        6,
        11,
        7,
        0,
        7,
        1,
        7,
        2,
        7,
        3,
        7,
        4,
        7,
        5,
        7,
        6,
        7,
        7,
        7,
        8,
        7,
        9,
        7,
        10,
        7,
        11,
    ]
end

# Check length
@assert length(dish_locations) == 2 * num_dish_locations_M * num_dish_locations_N
# Convert element type to tuples
dish_locations = reinterpret(NTuple{2,Int}, dish_locations)
# Drop dummy dishes
# num_dishes = 512
num_dishes = 64
@assert ndishs == num_dishes
dish_locations = dish_locations[1:num_dishes]
dishsΔx = 6.3f0
dishsΔy = 8.5f0

ndishs_i = num_dish_locations_M
ndishs_j = num_dish_locations_N
dishsi₀ = (ndishs_i - 1) / 2.0f0
dishsj₀ = (ndishs_j - 1) / 2.0f0
dishs_xlim = (dishsΔx * (0 - 1 / 2.0f0 - dishsi₀), dishsΔx * (ndishs_i - 1 + 1 / 2.0f0 - dishsi₀))
dishs_ylim = (dishsΔy * (0 - 1 / 2.0f0 - dishsj₀), dishsΔy * (ndishs_j - 1 + 1 / 2.0f0 - dishsj₀))
dishs_xsize = dishs_xlim[2] - dishs_xlim[1]
dishs_ysize = dishs_ylim[2] - dishs_ylim[1]

dishsx = Float32[]
dishsy = Float32[]
for dish in 1:ndishs
    dishi = dish_locations[dish][1]
    dishj = dish_locations[dish][2]
    push!(dishsx, dishsΔx * (dishi - dishsi₀))
    push!(dishsy, dishsΔy * (dishj - dishsj₀))
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

# Most of the power is in frequency 89 (???)
# (TODO: Read this from metadata)
# freq = 1:nfreqs
# Full CHORD
# freq = 48
# CHORD pathfinder
freq = 408

data = Float32[
    sqrt(
        sum(abs2(real(Complex{Float32}(i2c(j)))) for j in view(array_E, dish, :, freq, :)) / length(view(array_E, dish, :, freq, :))
    ) for dish in 1:ndishs
]
fig = Figure(; size=(1280, 960))
ax = Axis(fig[1, 1]; title="F-engine electric field", xlabel="x", ylabel="y")
xlims!(ax, dishs_xlim)
ylims!(ax, dishs_ylim)
obj = scatter!(ax, dishsx, dishsy; color=data, colormap=:plasma, markersize=960 / sqrt(2 * length(data)))
Colorbar(fig[1, 2], obj; label="|dish|₂")
# rowsize!(fig.layout, 1, Aspect(1, dishs_ysize / dishs_xsize))
colsize!(fig.layout, 1, Aspect(1, dishs_xsize / dishs_ysize))
display(fig)

data = Float32[
    sum(Float32(i) for i in view(array_I, beamp, beamq, :, freq)) / length(view(array_I, beamp, beamq, :, freq)) for
    beamq in 1:nbeamqs for beamp in 1:nbeamps
]
fig = Figure(; size=(1280, 960))
ax = Axis(fig[1, 1]; title="X-engine frb beams", xlabel="sky θx", ylabel="sky θy")
xlims!(ax, beams_xlim)
ylims!(ax, beams_ylim)
obj = scatter!(ax, beamsx, beamsy; color=data, colormap=:plasma, markersize=960 / sqrt(2 * length(data)))
Colorbar(fig[1, 2], obj; label="|baseband beam|₂")
# rowsize!(fig.layout, 1, Aspect(1, beams_ysize / beams_xsize))
colsize!(fig.layout, 1, Aspect(1, beams_xsize / beams_ysize))
display(fig)

nothing
