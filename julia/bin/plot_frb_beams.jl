using BFloat16s
using CUDASIMDTypes
using CairoMakie
using HDF5
using SixelTerm

dir = "/tmp"
prefix = "blue_"
iter = "0000000"

t2c(x::NTuple{2}) = Complex(x...)
i2t(x::Int4x2) = convert(NTuple{2,Int8}, x)
i2c(x::Int4x2) = t2c(i2t(x))

quantity_E = "upchan_voltage"
quantity_E′ = "voltage"
file_E = h5open("$(dir)/$(prefix)$(quantity_E).h5")
dataset_E = file_E[iter]["host_$(quantity_E′)_buffer"];
@assert dataset_E["dim_name"][] == ["T", "P", "F", "D"]
array_E = dataset_E[];
array_E::AbstractArray{UInt8,4};
array_E = reinterpret(Int4x2, array_E);
array_E::AbstractArray{Int4x2,4};
ndishs, nfreqs, npolrs, ntimes = size(array_E)

quantity_W = "frb_phase"
file_W = h5open("$(dir)/$(prefix)$(quantity_W).h5")
dataset_W = file_W[iter]["host_$(quantity_W)_buffer"];
@assert dataset_W["dim_name"][] == ["P", "F", "dishN", "dishM", "C"]
array_W = dataset_W[];
array_W::AbstractArray{UInt16,5};
array_W = reinterpret(Float16, array_W);
array_W::AbstractArray{Float16,5};

quantity_I = "frb_intensity"
file_I = h5open("$(dir)/$(prefix)$(quantity_I).h5")
dataset_I = file_I[iter]["host_$(quantity_I)_buffer"];
@assert dataset_I["dim_name"][] == ["F", "Tbar", "beamQ", "beamP"]
array_I = dataset_I[];
array_I::AbstractArray{UInt16,4};
array_I = reinterpret(Float16, array_I);
array_I::AbstractArray{Float16,4};
nbeamps, nbeamqs, ntimebars, nfreqs′ = size(array_I)
@assert nfreqs′ == nfreqs

CONTINUE HERE

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

# TODO: Read this from metadata
nbeams_i = 12
nbeams_j = 8
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

# Most of the power is in frequency 3
# (TODO: Read this from metadata)
# freq = 1:nfreqs
freq = 3

data = Float32[sqrt(sum(abs2(real(Complex{Float32}(i2c(j)))) for j in view(array_E, dish, freq, :, :)) /
                    length(view(array_E, dish, freq, :, :)))
               for dish in 1:ndishs]
fig = Figure(; resolution=(1280, 960));
ax = Axis(fig[1, 1]; title="F-engine electric field", xlabel="x", ylabel="y");
xlims!(ax, dishs_xlim)
ylims!(ax, dishs_ylim)
obj = scatter!(ax, dishsx, dishsy; color=data, colormap=:plasma, markersize=960 / sqrt(2 * length(data)));
Colorbar(fig[1, 2], obj; label="|dish|₂");
rowsize!(fig.layout, 1, Aspect(1, dishs_ysize / dishs_xsize))
display(fig)

data = Float32[sqrt(sum(abs2(Complex{Float32}(i2c(j))) for j in view(array_I0, :, :, freq, beam)) /
                    length(view(array_I0, :, :, freq, beam)))
               for beam in 1:nbeams]
fig = Figure(; resolution=(1280, 960));
ax = Axis(fig[1, 1]; title="expected baseband beams", xlabel="sky θx", ylabel="sky θy");
xlims!(ax, beams_xlim)
ylims!(ax, beams_ylim)
obj = scatter!(ax, beamsx, beamsy; color=data, colormap=:plasma, markersize=960 / sqrt(2 * length(data)));
Colorbar(fig[1, 2], obj; label="|baseband beam|₂");
rowsize!(fig.layout, 1, Aspect(1, beams_ysize / beams_xsize))
display(fig)

data = Float32[sqrt(sum(abs2(Complex{Float32}(i2c(j))) for j in view(array_I, :, :, freq, beam)) /
                    length(view(array_I, :, 1, freq, beam)))
               for beam in 1:nbeams]
fig = Figure(; resolution=(1280, 960));
ax = Axis(fig[1, 1]; title="X-engine baseband beams", xlabel="sky θx", ylabel="sky θy");
xlims!(ax, beams_xlim)
ylims!(ax, beams_ylim)
obj = scatter!(ax, beamsx, beamsy; color=data, colormap=:plasma, markersize=960 / sqrt(2 * length(data)));
Colorbar(fig[1, 2], obj; label="|baseband beam|₂");
rowsize!(fig.layout, 1, Aspect(1, beams_ysize / beams_xsize))
display(fig)

nothing
