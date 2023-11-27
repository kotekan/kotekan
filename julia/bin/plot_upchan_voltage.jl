using ASDF2
using CUDASIMDTypes
using CairoMakie
using SixelTerm

dir = "/tmp"
prefix = "blue_"
iter = "00000000"

t2c(x::NTuple{2}) = Complex(x...)
i2t(x::Int4x2) = convert(NTuple{2,Int8}, x)
i2c(x::Int4x2) = t2c(i2t(x))

# TODO: Read this from metadata
U = 16

quantity_E = "voltage"
file_E = ASDF2.load_file("$(dir)/$(prefix)$(quantity_E).$iter.asdf")
dataset_E = file_E.metadata[parse(Int, iter)] # ["host_$(quantity_E)_buffer"]
@assert dataset_E["dim_names"] == ["T", "F", "P", "D"]
array_E = dataset_E["buffer"][]
array_E::AbstractArray{UInt8,4}
array_E = reinterpret(Int4x2, array_E)
array_E::AbstractArray{Int4x2,4}
ndishs, npolrs, nfreqs, ntimes = size(array_E)

quantity_G = "upchan_gain"
file_G = ASDF2.load_file("$(dir)/$(prefix)$(quantity_G).$iter.asdf")
dataset_G = file_G.metadata[parse(Int, iter)] # ["host_$(quantity_G)_buffer"]
@assert dataset_G["dim_names"] == ["Fbar"]
array_G = dataset_G["buffer"][]
# array_G::AbstractArray{UInt16,1}
# array_G = reinterpret(Float16, array_G)
array_G::AbstractArray{Float16,1}
ngains, = size(array_G)

@assert ngains == nfreqs * U

quantity_Ebar = "upchan_voltage"
file_Ebar = ASDF2.load_file("$(dir)/$(prefix)$(quantity_Ebar).$iter.asdf")
dataset_Ebar = file_Ebar.metadata[parse(Int, iter)]   # ["host_$(quantity_Ebar)_buffer"]
@assert dataset_Ebar["dim_names"] == ["Tbar", "Fbar", "P", "D"]
array_Ebar = dataset_Ebar["buffer"][]
array_Ebar::AbstractArray{UInt8,4}
array_Ebar = reinterpret(Int4x2, array_Ebar)
array_Ebar::AbstractArray{Int4x2,4}
ndishs′, npolrs′, nfreqbars, ntimebars = size(array_Ebar)

@assert (ndishs′, npolrs′, nfreqbars) == (ndishs, npolrs, nfreqs * U)
@assert ntimebars <= ntimes ÷ U

# TODO: Read this from metadata
ndishs_i = 32
ndishs_j = 16
dishsΔx = 6.3f0
dishsΔy = 8.5f0

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

# Most of the power is in frequency 3
# (TODO: Read this from metadata)
# freq = 1:nfreqs
freq = 3
freqbar = 1:nfreqbars
# freqbar = 3 * U

data = Float32[sqrt(sum(abs2(real(Complex{Float32}(i2c(j)))) for j in view(array_E, dish, :, freq, :)) /
                    length(view(array_E, dish, :, freq, :)))
               for dish in 1:ndishs]
fig = Figure(; resolution=(1280, 960))
ax = Axis(fig[1, 1]; title="F-engine electric field", xlabel="x", ylabel="y")
xlims!(ax, dishs_xlim)
ylims!(ax, dishs_ylim)
obj = scatter!(ax, dishsx, dishsy; color=data, colormap=:plasma, markersize=960 / sqrt(2 * length(data)))
Colorbar(fig[1, 2], obj; label="|dish|₂")
rowsize!(fig.layout, 1, Aspect(1, dishs_ysize / dishs_xsize))
display(fig)

data = Float32[sqrt(sum(abs2(real(Complex{Float32}(i2c(j)))) for j in view(array_Ebar, dish, :, freqbar, :)) /
                    length(view(array_Ebar, dish, :, freqbar, :)))
               for dish in 1:ndishs]
fig = Figure(; resolution=(1280, 960))
ax = Axis(fig[1, 1]; title="F-engine upchannelized electric field (U=$U)", xlabel="x", ylabel="y")
xlims!(ax, dishs_xlim)
ylims!(ax, dishs_ylim)
obj = scatter!(ax, dishsx, dishsy; color=data, colormap=:plasma, markersize=960 / sqrt(2 * length(data)))
Colorbar(fig[1, 2], obj; label="|dish|₂")
rowsize!(fig.layout, 1, Aspect(1, dishs_ysize / dishs_xsize))
display(fig)

nothing
