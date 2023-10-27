module FEngine

# Simulate the dishes and the F-engine to generate inputs for the X-engine

# We are using SI units

# Real F-engine has 16k frequencies, 4 taps

using Base.Threads
using CUDASIMDTypes
using CairoMakie
using FFTW
using MappedArrays
using SixelTerm

################################################################################
# Constants

const Float = Float32

const c₀ = 299_792_458          # speed of light in vacuum

################################################################################
# Utilities

reim(x::Complex) = (x.re, x.im)

ftoi4(x::Complex{T}) where {T<:Real} = Int4x2(round.(Int8, clamp.(reim(x) .* T(7.5), -7, +7))...)

################################################################################
# Richard Shaw's PFB notes
# See < https://github.com/jrs65/pfb-inverse/blob/master/notes.ipynb>

"""Sinc window function.

Parameters
----------
ntaps : integer
    Number of taps.
lblock: integer
    Length of block.
    
Returns
-------
window : np.ndarray[ntaps * lblock]
"""
function sinc_window(::Type{T}, ntaps, lblock) where {T<:Real}
    coeff_length = T(π) * ntaps
    coeff_num_samples = ntaps * lblock

    # Sampling locations of sinc function
    X = LinRange(-coeff_length / 2, +coeff_length / 2, coeff_num_samples)

    # sinc function is sin(π* x) / (π * x), not sin(x) / x, so use X / π
    return sinc.(X / π)
end
sinc_window(ntaps, lblock) = sinc_window(typeof(0 / 1), ntaps, lblock)

# Same as np.hanning
hanning(::Type{T}, M) where {T<:Real} = T[(1 - cospi(2 * n / T(M - 1))) / 2 for n in 0:(M - 1)]
hanning(M) = hanning(typeof(0 / 1), M)

"""Hanning-sinc window function.

Parameters
----------
ntaps : integer
    Number of taps.
lblock: integer
    Length of block.
    
Returns
-------
window : np.ndarray[ntaps * lblock]
"""
sinc_hanning(::Type{T}, ntaps, lblock) where {T<:Real} = sinc_window(T, ntaps, lblock) .* hanning(T, ntaps * lblock)
sinc_hanning(ntaps, lblock) = sinc_hanning(typeof(0 / 1), ntaps, lblock)

"""Perform the CHIME PFB on a timestream.

Parameters
----------
timestream : np.ndarray
    Timestream to process
nfreq : int
    Number of frequencies we want out (probably should be odd
    number because of Nyquist)
ntaps : int
    Number of taps.

Returns
-------
pfb : np.ndarray[:, nfreq]
    Array of PFB frequencies.
"""
function pfb(timestream::AbstractVector{T}, nfreqs; ntaps=4, window=sinc_hanning) where {T<:Real}

    # Number of samples in a sub block
    lblock = 2 * (nfreqs - 1)

    # Number of blocks
    nblocks = timestream.size ÷ lblock - (ntaps - 1)

    # Define array for spectrum
    spec = Array{Complex{T}}(undef, nblocks, nfreqs)

    # Window function
    w = window(T, ntaps, lblock)

    # Iterate over blocks and perform the PFB
    @threads for bi in 1:nblocks
        # Cut out the correct timestream section
        ts_sec = @view timestream[((bi - 1) * lblock):((bi + ntaps - 1) * lblock - 1)]

        # Perform a real FFT(with applied window function)
        ft = rfft(ts_sec .* w)

        # Choose every n - th frequency
        spec[bi] = ft[begin:ntaps:end]
    end

    return spec::AbstractArray{Complex{T},2}
end

function pfb(samples::AbstractVector{<:AbstractArray{T,2}}; ntaps=4, window=sinc_hanning) where {T<:Real}
    @assert all(size(sample) == size(samples[begin]) for sample in samples)
    # Number of samples in a sub block
    lblock = size(samples[begin], 1)
    nfeeds = size(samples[begin], 2)
    nfreqs = lblock ÷ 2 + 1

    # Number of blocks
    nblocks = length(samples) - (ntaps - 1)
    @assert nblocks > 0

    # Define array for spectrum
    spec = Vector{Array{Complex{T},2}}(undef, nblocks)

    # Window function
    w = window(ntaps, lblock)

    ts_sec = Array{T,2}(undef, lblock * ntaps, nfeeds)
    P = plan_rfft(ts_sec, 1)

    # Iterate over blocks and perform the PFB
    for bi in 1:nblocks
        # Cut out the correct timestream section
        for ti in 1:ntaps
            for ai in 1:nfeeds, i in 1:lblock
                ts_sec[(ti - 1) * lblock + i, ai] = samples[bi + ntaps - ti][i, ai] * w[(ti - 1) * lblock + i]
            end
        end

        # Perform a real FFT(with applied window function)
        ft = P * ts_sec

        # Choose every n - th frequency
        spec[bi] = ft[begin:ntaps:end, :]
    end

    return spec::AbstractVector{<:AbstractArray{Complex{T},2}}
end

################################################################################
# Sources

# We represent the source as complex - valued function of time.The two
# complex components represent the two polarisations.

struct Source{F}
    fun::F
    sinx::Float
    siny::Float
end

function make_monochromatic_source(A::Complex{Float}, f₀::Float)
    function source(t::Float)
        return (A * cispi(2 * f₀ * t))::Complex{Float}
    end
    return source
end

################################################################################
# Dishes

struct Dishes
    locations::Vector{NTuple{2,Float}}
end

function make_dishes(dish_separation_x::Float, dish_separation_y::Float, ndishes_i::Int, ndishes_j::Int)
    # dish_separation_x = Float(6.3)  #  east-west
    # dish_separation_y = Float(8.5)  #  north-south
    i₀ = (1 + ndishes_i) / Float(2) #  centre
    j₀ = (1 + ndishes_j) / Float(2)
    # TODO: Use realistic dish layout
    locations = NTuple{2,Float}[let
                                    x = dish_separation_x * (i - i₀)
                                    y = dish_separation_y * (j - j₀)
                                    (x, y)
                                end
                                for j in 1:ndishes_j for i in 1:ndishes_i]
    return Dishes(locations)
end

################################################################################
# Sample

function dish_source(source::Source, dishx::Float, dishy::Float, t::Float)
    # TODO: Take beam shape into account
    return source.fun(t + source.sinx * dishx / c₀ + source.siny * dishy / c₀)::Complex{Float}
end

function sample_sources(::Type{T}, source::Source, dishes::Dishes, t₀::Float, Δt::Float, ntimes::Int) where {T<:Real}
    data = T[
             let
                 dishx, dishy = location
                 t = t₀ + Δt * (i - 1)
                 val = dish_source(source, dishx, dishy, t)
                 reim(val)[polr]
             end
             for i in 1:ntimes, location in dishes.locations, polr in 1:2
             ]
    return data::AbstractArray{T,3}
end

################################################################################
# F - engine

struct ADC
    Δt::Float
end

struct ADCFrame{T}
    t₀::Float
    Δt::Float
    data::Array{T,3}            # [time, dish, polr]
end

function plot(frame::ADCFrame{T}) where {T<:Real}
    data = reshape(frame.data, size(frame.data, 1), :)
    times = frame.t₀ .+ (axes(data, 1) .- 1) * frame.Δt
    dishes = axes(data, 2)

    fig = Figure(; resolution=(1280, 960))
    ax = Axis(fig[1, 1]; title="ADC sampled E-field", xlabel="time", ylabel="dish, polr")
    obj = heatmap!(ax, times, dishes, data; colormap=:plasma)
    Colorbar(fig[1, 2], obj; label="E-field")
    return fig
end

function adc_sample(::Type{T}, source::Source, dishes::Dishes, adc::ADC, ntimes::Int, nframes::Int) where {T<:Real}
    frames = Vector{ADCFrame{T}}(undef, nframes)
    @threads for n in 1:nframes
        Δt = adc.Δt
        t₀ = (n - 1) * ntimes * Δt
        data = sample_sources(T, source, dishes, t₀, adc.Δt, ntimes)
        frame = ADCFrame(t₀, Δt, data)
        frames[n] = frame
    end
    return frames::AbstractVector{ADCFrame{T}}
end

struct FFrame{T}
    t₀::Float
    Δt::Float
    f₀::Float
    Δf::Float
    data::Array{Complex{T},3} # [freq, dish, polr]
end

function plot(frame::FFrame{T}) where {T<:Real}
    data = reshape(frame.data, size(frame.data, 1), :)
    data = PermutedDimsArray(data, (2, 1))
    data = mappedarray(real, data)
    dishes = axes(data, 1)
    freqs = frame.f₀ .+ (axes(data, 2) .- 1) * frame.Δf

    fig = Figure(; resolution=(1280, 960))
    ax = Axis(fig[1, 1]; title="F-engine frame", xlabel="dish, polr", ylabel="freq")
    obj = heatmap!(ax, dishes, freqs, data; colormap=:plasma)
    Colorbar(fig[1, 2], obj; label="E-field (real part)")
    return fig
end

function f_engine(inframes::AbstractVector{ADCFrame{T}}, ntaps::Int) where {T<:Real}
    @assert all(size(frame.data) == size(inframes[begin].data) for frame in inframes)
    @assert all(frame.Δt == inframes[begin].Δt for frame in inframes)

    t₀ = inframes[begin].t₀
    Δt = inframes[begin].Δt

    # nframes = length(inframes)
    # npoints = size(inframes[begin], 1)
    # nfeeds = size(inframes[begin], 2)
    # 
    # # TODO: Memoize the plan
    # P = plan_rfft(Array{T}(undef, npoints * ntaps, nfeeds), 1)
    # outframes = Vector{Array{Complex{T }, 2 } }(undef, nframes)
    # for iframe in 1:nframes
    #     inframe = Array{T }(undef, npoints, ntaps, nfeeds)
    #     for itap in 1:ntaps
    #         if iframe - (itap - 1) ≥ 1
    #             inframe[:, itap, :] .= inframes[iframe - (itap - 1)]
    #         else
    #             inframe[:, itap, :] .= 0
    #         end
    #     end
    #     inframe = reshape(inframe, npoints * ntaps, nfeeds)
    #     
    #      # FFT
    #     outframe = P * inframe
    #     
    #      # FFT and corner turn
    #     outframes[n] = transpose(P * inframes[n])
    # end

    ntimes, ndishes, npolrs = size(inframes[begin].data)
    Δt′ = Δt * ntimes
    t₀′ = t₀ - Δt / 2 + Δt′ * ntaps / 2
    Δf′ = 1 / (ntimes * Δt)
    f₀′ = T(0)

    # Drop base frequency
    # TODO: Select interesting frequencies
    f₀′ += Δf′

    indata = [reshape(frame.data, size(frame.data, 1), :) for frame in inframes]::AbstractVector{<:AbstractArray{T,2}}
    outdata = pfb(indata; ntaps)
    nframes′ = length(outdata)
    outframes = Vector{FFrame{T}}(undef, nframes′)
    @threads for n in 1:nframes′
        # Drop base frequency
        outframes[n] = FFrame(t₀′ + (n - 1) * Δt′, Δt′, f₀′, Δf′,
                              reshape(outdata[n], size(outdata[n], 1), ndishes, npolrs)[(begin + 1):end, :, :])
    end

    return outframes::AbstractVector{FFrame{T}}
end

function quantize(::Type{I}, inframes::AbstractVector{<:FFrame}) where {I<:Integer}
    # TODO: add noise, then set the gain so that the variance is at 2.3 bits

    maxabs = maximum(abs.(maximum(abs.(inframe.data)) for inframe in inframes))
    outrange = 7

    scale = (2 * outrange + 1) / Float(2) / maxabs
    println("Scaling voltage by $scale")

    quantize1(x::Real) = clamp(round(I, scale * x), (-I(outrange)):(+I(outrange)))
    quantize(x::Complex) = Complex(quantize1(real(x)), quantize1(imag(x)))

    outframes = FFrame{I}[FFrame(inframe.t₀, inframe.Δt, inframe.f₀, inframe.Δf, quantize.(inframe.data))
                          for inframe in inframes]

    numvals = zeros(Int, 2 * outrange + 1)
    @threads for val in (-outrange):(+outrange)
        numvals[outrange + 1 + val] = sum(sum((real(x) == val) + (imag(x) == val) for x in outframe.data) for outframe in outframes)
    end
    println("Value range $(-outrange:+outrange), counts")
    print("    [")
    for val in (-outrange):(+outrange)
        print("$(numvals[outrange+1+val]), ")
    end
    println("]")

    return outframes::AbstractVector{FFrame{I}}
end

struct XFrame{T}
    t₀::Float
    Δt::Float
    f₀::Float
    Δf::Float
    data::Array{Complex{T},4} # [dish, polr, freq, time]
end

function plot(frame::XFrame{T}) where {T<:Real}
    freq = argmax(mappedarray(abs2, frame.data))[4]
    data = mappedarray(real, reshape((@view frame.data[:, :, freq, :]), size(frame.data, 1), :))
    times = frame.t₀ .+ (axes(data, 2) .- 1) * frame.Δt
    dishes = axes(data, 2)

    fig = Figure(; resolution=(1280, 960))
    ax = Axis(fig[1, 1]; title="X-engine frame", xlabel="time", ylabel="dish, polr")
    obj = heatmap!(ax, times, dishes, data; colormap=:plasma)
    Colorbar(fig[1, 2], obj; label="E-field (real part, freq=$(frame.f₀ + (freq-1) * frame.Δf))")
    return fig
end

function corner_turn(inframes::AbstractVector{FFrame{T}}, ntimes::Int) where {T<:Real}
    @assert length(inframes) % ntimes == 0

    inframes′ = reshape(inframes, ntimes, :)
    nframes = size(inframes′, 2)

    nfreqs, ndishes, npolrs = size(inframes′[begin, begin].data)

    outframes = Vector{XFrame{T}}(undef, nframes)
    @threads for n in 1:nframes
        outframes[n] = let
            t₀ = inframes′[begin, n].t₀
            Δt = inframes′[begin, n].Δt
            f₀ = inframes′[begin, n].f₀
            Δf = inframes′[begin, n].Δf
            data = Complex{T}[
                              inframes′[time, n].data[freq, dish, polr]
                              for dish in 1:ndishes, polr in 1:npolrs, freq in 1:nfreqs, time in 1:ntimes
                              ]
            XFrame{T}(t₀, Δt, f₀, Δf, data)
        end
    end

    return outframes::AbstractVector{XFrame{T}}
end

################################################################################
# Baseband beamformer

function bb(::Type{T}, A::AbstractArray{<:Complex,4}, E::AbstractArray{<:Complex,4}) where {T<:Real}
    # A: [dish, beam, polr, freq]
    # E: [dish, polr, freq, time]
    # J: [time, polr, freq, beam]
    ndishes, nbeams, npolrs, nfreqs = size(A)
    ndishes′, npolrs′, nfreqs′, ntimes = size(E)
    @assert (ndishes, npolrs, nfreqs) == (ndishes′, npolrs′, nfreqs′)
    J = Array{Complex{T},4}(undef, ntimes, npolrs, nfreqs, nbeams)
    @threads for freq in 1:nfreqs
        for beam in 1:nbeams, polr in 1:npolrs, time in 1:ntimes
            J[time, polr, freq, beam] = let
                J0 = zero(Complex{T})
                for dish in 1:ndishes
                    J0 += A[dish, beam, polr, freq] * E[dish, polr, freq, time]
                end
                J0
            end
        end
    end
    return J::AbstractArray{Complex{T},4}
end

function make_baseband_beams(nbeamsi::Int, nbeamsj::Int, Δθi::T, Δθj::T) where {T<:Real}
    sinxys = NTuple{2,T}[let
                             θi = Δθi * (beami - (1 + nbeamsi) / T(2))
                             θj = Δθj * (beamj - (1 + nbeamsj) / T(2))
                             (sin(θi), sin(θj))
                         end
                         for beamj in 1:nbeamsj for beami in 1:nbeamsi]
    return sinxys::Vector{NTuple{2,T}}
end

struct BBeams{T}
    t₀::Float
    Δt::Float
    f₀::Float
    Δf::Float
    sinxys::Vector{NTuple{2,Float}}
    phases::Array{Complex{T},4} # [dish, beam, polr, freq]
    data::Array{Complex{T},4}   # [time, polr, freq, beam]
end

# function plot(beams::BBeams{T}) where{T<:Real }
#     data = sum(abs2, beams.data; dims = 4)::AbstractArray{T, 4 }
#     data = reshape(data, size(data, 1) * size(data, 2), :)::AbstractArray{T, 2 }
#     data = PermutedDimsArray(data, (2, 1))
#     times = beams.t₀.+ (axes(data, 1).- 1) * beams.Δt
#     beams = axes(data, 2)
# 
#     fig = Figure(; resolution = (1280, 960))
#     ax = Axis(fig[1, 1]; title = "X-engine baseband beams", xlabel = "time", ylabel = "beam, polr")
#     obj = heatmap!(ax, times, beams, data; colormap = :plasma)
#     Colorbar(fig[1, 2], obj; label = "|baseband beam|²")
#     return fig
# end

function plot(beams::BBeams{T}) where {T<:Real}
    data = (sum(abs2, beams.data; dims=(1, 2, 3)) / prod(size(beams.data)[1:3]))::AbstractArray{T,4}
    data = reshape(data, size(data, 4))::AbstractArray{T,1}
    beamsx = map(xy -> asin(xy[1]), beams.sinxys)
    beamsy = map(xy -> asin(xy[2]), beams.sinxys)

    fig = Figure(; resolution=(1280, 960))
    ax = Axis(fig[1, 1]; title="X-engine baseband beams", xlabel="sky θx", ylabel="sky θy")
    obj = scatter!(ax, beamsx, beamsy; color=data, colormap=:plasma, markersize=960 / sqrt(2 * length(data)))
    Colorbar(fig[1, 2], obj; label="|baseband beam|²")
    colsize!(fig.layout, 1, Aspect(1, 1.0))
    return fig
end

function bb(::Type{T}, xframes::AbstractVector{<:XFrame}, dishes::Dishes, sinxys::Vector{NTuple{2,Float}}) where {T<:Real}
    ndishes, npolrs, nfreqs, ntimes = size(xframes[begin].data)
    @assert ndishes == length(dishes.locations)
    nbeams = length(sinxys)

    f₀ = xframes[begin].f₀
    Δf = xframes[begin].Δf

    # We choose `A` independent of polarization
    A = Complex{T}[
                   let
                       dishx, dishy = dishes.locations[dish]
                       sinx, siny = sinxys[beam]
                       Δt = sinx * dishx / c₀ + siny * dishy / c₀
                       f = f₀ + (freq - 1) * Δf
                       cispi(-2 * f * Δt)
                   end
                   for dish in 1:ndishes, beam in 1:nbeams, polr in 1:npolrs, freq in 1:nfreqs
                   ]

    beamss = BBeams{T}[]
    for xframe in xframes
        # E: [dish, polr, freq, time]
        E = xframe.data
        # J: [time, polr, freq, beam]
        J = bb(T, A, E)
        # A: [dish, beam, polr, freq]
        phases = A
        data = J
        beams = BBeams(xframe.t₀, xframe.Δt, xframe.f₀, xframe.Δf, sinxys, phases, data)
        push!(beamss, beams)
    end
    return beamss::AbstractVector{BBeams{T}}
end

function quantize(::Type{I}, inbeams::AbstractVector{<:BBeams}) where {I<:Integer}
    # TODO: add noise, then set the gain according to the variance

    # Quantize phases

    maxabs_phases = maximum(abs.(maximum(abs.(inbeam.phases)) for inbeam in inbeams))
    outrange_phases = 127
    scale_phases = (2 * outrange_phases + 1) / Float(2) / maxabs_phases
    println("Scaling phases by $scale_phases")

    quantize1_phases(x::Real) = I(clamp(round(Int, scale_phases * x), (-outrange_phases):(+outrange_phases)))
    quantize_phases(x::Complex) = Complex(quantize1_phases(real(x)), quantize1_phases(imag(x)))

    # Quantize beams

    maxabs_beams = maximum(abs.(maximum(abs.(inbeam.data)) for inbeam in inbeams))
    outrange_beams = 7
    scale_beams = (2 * outrange_beams + 1) / Float(2) / maxabs_beams
    println("Scaling beams by $scale_beams")

    quantize1_beams(x::Real) = I(clamp(round(Int, scale_beams * x), (-outrange_beams):(+outrange_beams)))
    quantize_beams(x::Complex) = Complex(quantize1_beams(real(x)), quantize1_beams(imag(x)))

    outbeams = BBeams{I}[BBeams(inbeam.t₀, inbeam.Δt, inbeam.f₀, inbeam.Δf, inbeam.sinxys, quantize_phases.(inbeam.phases),
                                quantize_beams.(inbeam.data)) for inbeam in inbeams]

    return outbeams::AbstractVector{BBeams{I}}
end

################################################################################
# Main script

function run(source_amplitude::Float, source_frequency::Float, source_position_x::Float, source_position_y::Float,
             dish_separation_x::Float, dish_separation_y::Float, ndishes_i::Int64, ndishes_j::Int64,
             adc_frequency::Float,
             ntaps::Int64, nfreq::Int64, ntimes::Int64,
             do_plot::Bool)
    source = let
        A = Complex{Float}(source_amplitude)
        f₀ = source_frequency       # Hz
        sinx = sin(source_position_x) # east-west
        siny = sin(source_position_y) # north-south
        Source(make_monochromatic_source(A, f₀), sinx, siny)
    end

    dishes = make_dishes(dish_separation_x, dish_separation_y, ndishes_i, ndishes_j)

    adc = let
        Δt = 1 / adc_frequency
        ADC(Δt)
    end

    # ntaps = 4 # Number of taps in PFB(fixed by F - engine))
    # nfreq = 2048 # Number of frequencies(fixed by F - engine)
    # ntimes = 32768 # Number of times in X - engine input

    # ntaps = 4 # Number of taps in PFB(fixed by F - engine))
    # nfreq = 64 # Number of frequencies(fixed by F - engine)
    # ntimes = 64 # Number of times in X - engine input

    nframes = ntimes + ntaps - 1

    aframes = adc_sample(Float, source, dishes, adc, 2 * nfreq, nframes)
    println("ADC output: $(length(aframes)) frames of size (ntimes, ndishes, npolrs)=$(size(aframes[begin].data)) t₀=$(aframes[begin].t₀) Δt=$(aframes[begin].Δt)")
    if do_plot
        fig = plot(aframes[begin])
        display(fig)
    end

    fframes = f_engine(aframes, ntaps)
    println("F-engine output: $(length(fframes)) frames of size (nfreqs, ndishes, npolrs)=$(size(fframes[begin].data)) t₀=$(fframes[begin].t₀) Δt=$(fframes[begin].Δt) f₀=$(fframes[begin].f₀) Δf=$(fframes[begin].Δf)")
    if do_plot
        fig = plot(fframes[begin])
        display(fig)
    end

    qframes = quantize(Int8, fframes)

    xframes = corner_turn(qframes, ntimes)
    global stored_xframes = xframes
    println("Corner turn output: $(length(xframes)) frames of size (ndishes, npolrs, nfreqs, ntimes)=$(size(xframes[begin].data)) t₀=$(xframes[begin].t₀) Δt=$(xframes[begin].Δt) f₀=$(xframes[begin].f₀) Δf=$(xframes[begin].Δf)")
    if do_plot
        fig = plot(xframes[begin])
        display(fig)
    end

    nbeamsi = 12
    nbeamsj = 8
    beamΔΘi = Float(0.015)
    beamΔΘj = Float(0.015)
    sinxys = make_baseband_beams(nbeamsi, nbeamsj, beamΔΘi, beamΔΘj)
    beams = bb(Float, xframes, dishes, sinxys)
    println("Baseband beams: $(length(beams)) frames of size (ntimes, npolrs, nfreqs, nbeams)=$(size(beams[begin].data)) t₀=$(beams[begin].t₀) Δt=$(beams[begin].Δt) f₀=$(beams[begin].f₀) Δf=$(beams[begin].Δf)")
    if do_plot
        fig = plot(beams[begin])
        display(fig)
    end

    qbeams = quantize(Int8, beams)
    global stored_beamss = qbeams

    return nothing
end

function fill_buffer_Int4!(ptr::Ptr{UInt8}, sz::Int64, data::AbstractArray{Complex{Int8}})
    @threads for i in 1:sz
        val = data[i]
        re, im = real(val), imag(val)
        re4 = (re % UInt8) & 0x0f
        im4 = (im % UInt8) & 0x0f
        cint4 = re4 << 0 | im4 << 4
        unsafe_store!(ptr, cint4, i)
    end
end

function fill_buffer!(ptr::Ptr{Complex{Int8}}, sz::Int64, data::AbstractArray{Complex{Int8}})
    @threads for i in 1:sz
        unsafe_store!(ptr, data[i], i)
    end
    return flush(stdout)
end

function setup(source_amplitude=Float(1.0), source_frequency=Float(0.3e+9), source_position_x=Float(0.02),
               source_position_y=Float(0.03),
               dish_separation_x=Float(6.3), dish_separation_y=Float(8.5), ndishes_i=8, ndishes_j=8,
               adc_frequency=Float(3.0e+9),
               ntaps=4, nfreq=64, ntimes=64)
    return run(Float(source_amplitude), Float(source_frequency), Float(source_position_x), Float(source_position_y),
               Float(dish_separation_x), Float(dish_separation_y), Int64(ndishes_i), Int64(ndishes_j),
               Float(adc_frequency),
               Int64(ntaps), Int64(nfreq), Int64(ntimes), false)
end

stored_xframes = nothing
function set_E(ptr::Ptr{UInt8}, sz::Int64, ndishes::Int64, npolrs::Int64, nfreqs::Int64, ntimes::Int64, frame_index::Int64)
    @show set_E ptr sz ndishes npolrs nfreqs ntimes frame_index
    xframes = stored_xframes::AbstractVector{<:XFrame}
    xframe = xframes[frame_index]::XFrame
    @assert length(xframe.data) == sz
    @assert size(xframe.data) == (ndishes, npolrs, nfreqs, ntimes)
    fill_buffer_Int4!(ptr, sz, xframe.data)
    return nothing
end

stored_beamss = nothing
function set_A(ptr::Ptr{UInt8}, sz::Int64, ndishs::Int64, nbeams::Int64, npolrs::Int64, nfreqs::Int64, frame_index::Int64)
    @show set_A ptr sz ndishs nbeams npolrs nfreqs frame_index
    beamss = stored_beamss::AbstractVector{<:BBeams}
    beams = beamss[frame_index]::BBeams
    if !(length(beams.phases) * sizeof(Complex{Int8}) == sz)
        @show length(beams.phases) * sizeof(Complex{Int8}) sz
    end
    if !(size(beams.phases) == (ndishs, nbeams, npolrs, nfreqs))
        @show size(beams.phases) (ndishs, nbeams, npolrs, nfreqs)
    end
    @assert length(beams.phases) * sizeof(Complex{Int8}) == sz
    @assert size(beams.phases) == (ndishs, nbeams, npolrs, nfreqs)
    fill_buffer!(Ptr{Complex{Int8}}(ptr), sz ÷ sizeof(Complex{Int8}), beams.phases)
    return nothing
end

function set_J(ptr::Ptr{UInt8}, sz::Int64, ntimes::Int64, npolrs::Int64, nfreqs::Int64, nbeams::Int64, frame_index::Int64)
    @show set_J ptr, sz ntimes npolrs nfreqs nbeams frame_index
    beamss = stored_beamss::AbstractVector{<:BBeams}
    beams = beamss[frame_index]::BBeams
    @assert length(beams.data) == sz
    @assert size(beams.data) == (ntimes, npolrs, nfreqs, nbeams)
    fill_buffer_Int4!(ptr, sz, beams.data)
    return nothing
end

end
