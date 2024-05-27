@show :FEngine0

module FEngine

# Simulate the dishes and the F-engine to generate inputs for the X-engine

# We are using SI units

# Real F-engine has 16k frequencies, 4 taps

using Base.Threads
using CUDASIMDTypes
#TODO using CairoMakie
using FFTW
using LinearAlgebra
using MappedArrays
#TODO using SixelTerm

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

    println("pfb: plan_rfft...")
    ts_secs = [Array{T,2}(undef, lblock * ntaps, nfeeds) for thr in 1:Threads.nthreads()]
    Ps = [plan_rfft(ts_secs[thr], 1) for thr in 1:Threads.nthreads()]
    fts = [Array{Complex{T},2}(undef, div(lblock * ntaps, 2) + 1, nfeeds) for thr in 1:Threads.nthreads()]

    # Iterate over blocks and perform the PFB
    println("pfb: ft...")
    @threads for bi in 1:nblocks
        thr = Threads.threadid()
        # thr == 1 && println("pfb: block $bi: sample...")
        ts_sec = ts_secs[thr]
        P = Ps[thr]
        ft = fts[thr]

        # Cut out the correct timestream section
        for ti in 1:ntaps
            for ai in 1:nfeeds, i in 1:lblock
                ts_sec[(ti - 1) * lblock + i, ai] = samples[bi + ntaps - ti][i, ai] * w[(ti - 1) * lblock + i]
            end
        end

        # Perform a real FFT(with applied window function)
        # thr == 1 && println("pfb: block $bi: FFT...")
        mul!(ft, P, ts_sec)

        # Choose every `n`th frequency
        # thr == 1 && println("pfb: block $bi: choose frequencies...")
        spec[bi] = ft[begin:ntaps:end, :]
    end

    println("pfb: done.")
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

function make_dishes(
    dish_separation_ew::Float,
    dish_separation_ns::Float,
    num_dish_locations_ew::Int,
    num_dish_locations_ns::Int,
    dish_locations::AbstractVector{NTuple{2,Int}},
)
    # dish_separation_ew = Float(6.3)  # east-west
    # dish_separation_ns = Float(8.5)  # north-south
    i_ew₀ = (num_dish_locations_ew - 1) / Float(2) # centre
    i_ns₀ = (num_dish_locations_ns - 1) / Float(2)
    locations = NTuple{2,Float}[
        let
            x_ew = dish_separation_ew * (i_ew - i_ew₀)
            x_ns = dish_separation_ns * (i_ns - i_ns₀)
            (x_ns, x_ew)
        end for (i_ns, i_ew) in dish_locations
    ]
    @show locations
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
        end for i in 1:ntimes, location in dishes.locations, polr in 1:2
    ]
    return data::AbstractArray{T,3}
end

################################################################################
# F-engine

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
    Δf::Float
    frequency_channels::Vector{Int}
    data::Array{Complex{T},3} # [dish, polr, freq]
end

function plot(frame::FFrame{T}) where {T<:Real}
    @assert false
    # TODO: The following lines assume data[freq, dish, polr], but the index order changed
    data = reshape(frame.data, size(frame.data, 1), :)
    data = PermutedDimsArray(data, (2, 1))
    data = mappedarray(real, data)
    dishes = axes(data, 1)
    freqs = frequency_channels[axes(data, 2)] * frame.Δf

    fig = Figure(; resolution=(1280, 960))
    ax = Axis(fig[1, 1]; title="F-engine frame", xlabel="dish, polr", ylabel="freq")
    obj = heatmap!(ax, dishes, freqs, data; colormap=:plasma)
    Colorbar(fig[1, 2], obj; label="E-field (real part)")
    return fig
end

function f_engine(inframes::AbstractVector{ADCFrame{T}}, ntaps::Int, frequency_channels::AbstractVector{<:Integer}) where {T<:Real}
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

    println("f_engine: reshape #1...")
    indata = [reshape(frame.data, size(frame.data, 1), :) for frame in inframes]::AbstractVector{<:AbstractArray{T,2}}
    channels = frequency_channels .+ 1
    println("f_engine: pfb...")
    outdata = pfb(indata; ntaps)
    println("f_engine: reshape #2...")
    nframes′ = length(outdata)
    outframes = Vector{FFrame{T}}(undef, nframes′)
    @threads for n in 1:nframes′
        # Select frequencies
        outframes[n] = let
            tmp = reshape(outdata[n], size(outdata[n], 1), ndishes, npolrs)
            data = Complex{T}[tmp[chan, dish, polr] for dish in 1:ndishes, polr in 1:npolrs, chan in channels]
            FFrame(t₀′ + (n - 1) * Δt′, Δt′, Δf′, frequency_channels, data)
        end
    end
    println("f_engine: done.")

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

    outframes = FFrame{I}[
        FFrame(inframe.t₀, inframe.Δt, inframe.Δf, inframe.frequency_channels, quantize.(inframe.data)) for inframe in inframes
    ]

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
    Δf::Float
    frequency_channels::Vector{Int}
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
    Colorbar(fig[1, 2], obj; label="E-field (real part, freq=$(frame.frequency_channels[freq] * frame.Δf))")
    return fig
end

function corner_turn(inframes::AbstractVector{FFrame{T}}, ntimes::Int) where {T<:Real}
    @assert length(inframes) % ntimes == 0

    inframes′ = reshape(inframes, ntimes, :)
    nframes = size(inframes′, 2)

    ndishes, npolrs, nfreqs = size(inframes′[begin, begin].data)

    outframes = Vector{XFrame{T}}(undef, nframes)
    @threads for n in 1:nframes
        outframes[n] = let
            t₀ = inframes′[begin, n].t₀
            Δt = inframes′[begin, n].Δt
            Δf = inframes′[begin, n].Δf
            frequency_channels = inframes′[begin, n].frequency_channels
            data = Complex{T}[
                inframes′[time, n].data[dish, polr, freq] for dish in 1:ndishes, polr in 1:npolrs, freq in 1:nfreqs,
                time in 1:ntimes
            ]
            XFrame{T}(t₀, Δt, Δf, frequency_channels, data)
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
    nbeamsi₀ = (nbeamsi - 1) / T(2) # centre
    nbeamsj₀ = (nbeamsj - 1) / T(2)
    sinxys = NTuple{2,T}[let
        θi = Δθi * (beami - nbeamsi₀)
        θj = Δθj * (beamj - nbeamsj₀)
        (sin(θi), sin(θj))
    end for beamj in 0:(nbeamsj - 1) for beami in 0:(nbeamsi - 1)]
    return sinxys::Vector{NTuple{2,T}}
end

struct BBBeams{T}
    t₀::Float
    Δt::Float
    Δf::Float
    frequency_channels::Vector{Int}
    sinxys::Vector{NTuple{2,Float}}
    phases::Array{Complex{T},4} # [dish, beam, polr, freq]
    data::Array{Complex{T},4}   # [time, polr, freq, beam]
end

# function plot(beams::BBBeams{T}) where{T<:Real }
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

function plot(beams::BBBeams{T}) where {T<:Real}
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

    Δf = xframes[begin].Δf
    frequency_channels = xframes[begin].frequency_channels

    # We choose `A` independent of polarization
    A = Complex{T}[
        let
            dishx, dishy = dishes.locations[dish]
            sinx, siny = sinxys[beam]
            Δt = sinx * dishx / c₀ + siny * dishy / c₀
            f = frequency_channels[freq] * Δf
            cispi(-2 * f * Δt)
        end for dish in 1:ndishes, beam in 1:nbeams, polr in 1:npolrs, freq in 1:nfreqs
    ]

    beamss = BBBeams{T}[]
    for xframe in xframes
        # E: [dish, polr, freq, time]
        E = xframe.data
        # J: [time, polr, freq, beam]
        J = bb(T, A, E)
        # A: [dish, beam, polr, freq]
        phases = A
        data = J
        beams = BBBeams(xframe.t₀, xframe.Δt, xframe.Δf, xframe.frequency_channels, sinxys, phases, data)
        push!(beamss, beams)
    end
    return beamss::AbstractVector{BBBeams{T}}
end

function quantize(::Type{I}, inbeams::AbstractVector{<:BBBeams}) where {I<:Integer}
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

    outbeams = BBBeams{I}[
        BBBeams(
            inbeam.t₀,
            inbeam.Δt,
            inbeam.Δf,
            inbeam.frequency_channels,
            inbeam.sinxys,
            quantize_phases.(inbeam.phases),
            quantize_beams.(inbeam.data),
        ) for inbeam in inbeams
    ]

    return outbeams::AbstractVector{BBBeams{I}}
end

################################################################################
# frb beamformer

function frb(
    ::Type{T}, S::AbstractVector{<:NTuple{2,<:Integer}}, W::AbstractArray{<:Complex,4}, E::AbstractArray{<:Complex,4}, Tds::Integer
) where {T<:Real}
    # S: [dish]
    # W: [dishM, dishN, polr, freq]
    # E: [dish, polr, freq, time]
    # I: [beamP, beamQ, time-bar, freq]
    ndishes, = size(S)
    ndishesM, ndishesN, npolrs, nfreqs = size(W)
    ndishes, npolrs′, nfreqs′, ntimes = size(E)
    @assert ndishes <= ndishesM * ndishesN
    @assert npolrs′ == npolrs
    @assert nfreqs′ == nfreqs
    nbeamsP = 2 * ndishesM
    nbeamsQ = 2 * ndishesN
    # @assert ntimes % Tds == 0
    ntimes_ds = ntimes ÷ Tds
    # Pre-calculate phase factors
    # eqn (4), probably needs more phase factors
    # TODO: Transpose `HPM`, `HQN`
    HPM = [cispi(2 * dishM * beamP / T(nbeamsP)) for dishM in 0:(ndishesM - 1), beamP in 0:(nbeamsP - 1)]
    HQN = [cispi(2 * dishN * beamQ / T(nbeamsQ)) for dishN in 0:(ndishesN - 1), beamQ in 0:(nbeamsQ - 1)]
    I = zeros(T, nbeamsP, nbeamsQ, ntimes_ds, nfreqs)
    Fs = Array{Complex{T},2}[Array{Complex{T},2}(undef, ndishesM, ndishesN) for tid in 1:Threads.threadpoolsize()]
    Xs = Array{Complex{T},2}[Array{Complex{T},2}(undef, ndishesN, nbeamsP) for tid in 1:Threads.threadpoolsize()]
    @threads for freq in 0:(nfreqs - 1)
        tid = threadid()
        F = Fs[tid]
        X = Xs[tid]
        F .= 0
        for time_ds in 0:(ntimes_ds - 1)
            for time in (time_ds * Tds):((time_ds + 1) * Tds - 1)
                for polr in 0:1
                    # 1. Dish gridding
                    for dish in 0:(ndishes - 1)
                        dishM, dishN = S[dish + 1]
                        F[dishM + 1, dishN + 1] =
                            W[dishM + 1, dishN + 1, polr + 1, freq + 1] * E[dish + 1, polr + 1, freq + 1, time + 1]
                    end
                    # 2. 2D FFT and accumulate
                    for dishN in 0:(ndishesN - 1)
                        for beamP in 0:(nbeamsP - 1)
                            X1 = zero(Complex{T})
                            for dishM in 0:(ndishesM - 1)
                                X1 += HPM[dishM + 1, beamP + 1] * F[dishM + 1, dishN + 1]
                            end
                            X[dishN + 1, beamP + 1] = X1
                        end
                    end
                    for beamP in 0:(nbeamsP - 1)
                        for beamQ in 0:(nbeamsQ - 1)
                            G1 = zero(Complex{T})
                            for dishN in 0:(ndishesN - 1)
                                G1 += HQN[dishN + 1, beamQ + 1] * X[dishN + 1, beamP + 1]
                            end
                            I[beamP + 1, beamQ + 1, time_ds + 1, freq + 1] += abs2(G1)
                        end
                    end
                end
            end
        end
    end
    return I::AbstractArray{T,4}
end

struct FRBBeams{T}
    t₀::Float
    Δt::Float
    Δf::Float
    frequency_channels::Vector{Int}
    phases::Array{Complex{T},4} # [dishM, dishN, polr, freq]
    data::Array{T,4}            # [beamP, beamQ, time_ds, freq]
end

function frb(
    ::Type{T},
    xframes::AbstractVector{<:XFrame},
    S::AbstractVector{<:NTuple{2,<:Integer}},
    num_dish_locations_M::Integer,
    num_dish_locations_N::Integer,
    Tds::Integer,
) where {T<:Real}
    ndishes, npolrs, nfreqs, ntimes = size(xframes[begin].data)
    @assert size(S) == (ndishes,)

    W = zeros(Complex{T}, num_dish_locations_M, num_dish_locations_N, npolrs, nfreqs)
    for dish in 0:(ndishes - 1)
        m, n = S[dish + 1]
        W[m + 1, n + 1, :, :] .= 1 / T(16)
    end

    beamss = FRBBeams{T}[]
    for xframe in xframes
        # E: [dish, polr, freq, time]
        E = xframe.data
        # I: [beamP, beamQ, time_ds, freq]
        I = frb(T, S, W, E, Tds)
        @assert size(I, 3) == size(E, 4) ÷ Tds
        # @assert size(I, 3) * Tds == size(E, 4)
        phases = W
        data = I
        beams = FRBBeams(xframe.t₀ * Tds, xframe.Δt * Tds, xframe.Δf, xframe.frequency_channels, phases, data)
        push!(beamss, beams)
    end
    return beamss::AbstractVector{FRBBeams{T}}
end

################################################################################
# Main script

function run(
    source_amplitude::Float,
    source_frequency::Float,
    source_position_ew::Float,
    source_position_ns::Float,
    num_dish_locations_ew::Int64,
    num_dish_locations_ns::Int64,
    dish_indices::Array{Int64,2},
    dish_separation_ew::Float,
    dish_separation_ns::Float,
    num_dishes::Int64,
    adc_frequency::Float,
    ntaps::Int64,
    nfreq::Int64,
    nchan::Int64,
    frequency_channels::Vector{Int64},
    ntimes::Int64,
    # ndishes_i::Int64,
    # ndishes_j::Int64,
    nbeams_i::Int64,
    nbeams_j::Int64,
    nframes::Int64,
    do_plot::Bool,
)
    source = let
        A = Complex{Float}(source_amplitude)
        f₀ = source_frequency            # Hz
        sin_ew = sin(source_position_ew) # east-west
        sin_ns = sin(source_position_ns) # north-south
        Source(make_monochromatic_source(A, f₀), sin_ew, sin_ns)
    end

    num_dish_locations = num_dish_locations_ew * num_dish_locations_ns
    dish_locations = fill((-1, -1), num_dishes)
    num_dishes_seen = 0
    @assert size(dish_indices) == (num_dish_locations_ew, num_dish_locations_ns)
    for loc_ns in 0:(num_dish_locations_ns - 1), loc_ew in 0:(num_dish_locations_ew - 1)
        dish = dish_indices[loc_ew + 1, loc_ns + 1]
        @assert dish == -1 || 0 <= dish < num_dishes
        if dish >= 0
            num_dishes_seen += 1
            @assert dish_locations[dish + 1] == (-1, -1)
            dish_locations[dish + 1] = (loc_ns, loc_ew)
        end
    end
    @assert num_dishes_seen == num_dishes

    dishes = make_dishes(dish_separation_ew, dish_separation_ns, num_dish_locations_ew, num_dish_locations_ns, dish_locations)

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

    nsamples = nframes * ntimes + ntaps - 1

    aframes = adc_sample(Float, source, dishes, adc, 2 * nfreq, nsamples)
    println(
        "ADC output: $(length(aframes)) frames of size (ntimes, ndishes, npolrs)=$(size(aframes[begin].data)) t₀=$(aframes[begin].t₀) Δt=$(aframes[begin].Δt)",
    )
    if do_plot
        fig = plot(aframes[begin])
        display(fig)
    end

    fframes = f_engine(aframes, ntaps, frequency_channels)
    println(
        "F-engine output: $(length(fframes)) frames of size (nchan, ndishes, npolrs)=$(size(fframes[begin].data)) t₀=$(fframes[begin].t₀) Δt=$(fframes[begin].Δt) Δf=$(fframes[begin].Δf)",
    )
    if do_plot
        fig = plot(fframes[begin])
        display(fig)
    end

    qframes = quantize(Int8, fframes)

    xframes = corner_turn(qframes, ntimes)
    global stored_xframes = xframes
    println(
        "Corner turn output: $(length(xframes)) frames of size (ndishes, npolrs, nchan, ntimes)=$(size(xframes[begin].data)) t₀=$(xframes[begin].t₀) Δt=$(xframes[begin].Δt) Δf=$(xframes[begin].Δf)",
    )
    if do_plot
        fig = plot(xframes[begin])
        display(fig)
    end

    beamΔΘi = Float(0.015)
    beamΔΘj = Float(0.015)
    sinxys = make_baseband_beams(nbeams_i, nbeams_j, beamΔΘi, beamΔΘj)
    bbbeams = bb(Float, xframes, dishes, sinxys)
    println(
        "Baseband beams: $(length(bbbeams)) frames of size (ntimes, npolrs, nchans, nbbbeams)=$(size(bbbeams[begin].data)) t₀=$(bbbeams[begin].t₀) Δt=$(bbbeams[begin].Δt) Δf=$(bbbeams[begin].Δf)",
    )
    if do_plot
        fig = plot(bbbeams[begin])
        display(fig)
    end

    qbbbeams = quantize(Int8, bbbeams)
    global stored_bbbeamss = qbbbeams

    Tds = 40
    frbbeams = frb(Float, xframes, dish_locations, num_dish_locations_ns, num_dish_locations_ew, Tds)
    println(
        "FRB beams: $(length(frbbeams)) frames of size (ntimes, nfrbbeamsP, nfrbbeamsQ, npolrs, nchans)=$(size(frbbeams[begin].data)) t₀=$(frbbeams[begin].t₀) Δt=$(frbbeams[begin].Δt) Δf=$(frbbeams[begin].Δf)",
    )
    if do_plot
        fig = plot(frbbeams[begin])
        display(fig)
    end

    global stored_frbbeamss = frbbeams

    return nothing
end

function fill_buffer_Int4!(ptr::Ptr{UInt8}, sz::Int64, data::AbstractArray{Complex{Int8}})
    @assert sz == length(data)  # sizeof(Complex{Int4}) should be 1
    @threads for i in 1:length(data)
        val = data[i]
        re, im = real(val), imag(val)
        re4 = (re % UInt8) & 0x0f
        im4 = (im % UInt8) & 0x0f
        cint4 = re4 << 0 | im4 << 4
        unsafe_store!(ptr, cint4, i)
    end
end

function fill_buffer!(ptr::Ptr, sz::Int64, data::AbstractArray)
    @assert sz == length(data) * sizeof(eltype(ptr))
    @threads for i in 1:length(data)
        unsafe_store!(ptr, data[i], i)
    end
end

function setup(
    source_amplitude=1.0,
    source_frequency=0.3e+9,
    source_position_ew=0.02,
    source_position_ns=0.03,
    num_dish_locations_ew=8,
    num_dish_locations_ns=8,
    dish_indices_ptr=Ptr{Cvoid}(), # [(m,n) for m in 0:M-1, n in 0:N-1],
    dish_separation_ew=6.3,
    dish_separation_ns=8.5,
    num_dishes=num_dish_locations_ew * num_dish_locations_ns,
    adc_frequency=3.0e+9,
    ntaps=4,
    nfreq=64,
    nchan=64,
    frequency_channels_ptr=Ptr{Cvoid}(),
    ntimes=64,
    # ndishes_i=8,
    # ndishes_j=8,
    nbeams_i=12,
    nbeams_j=8,
    nframes=1,
)
    println("Julia F-Engine setup:")
    println("    - source_amplitude:       $source_amplitude")
    println("    - source_frequency:       $source_frequency")
    println("    - source_position_ew:     $source_position_ew")
    println("    - source_position_ns:     $source_position_ns")
    println("    - num_dish_locations_ew:  $num_dish_locations_ew")
    println("    - num_dish_locations_ns:  $num_dish_locations_ns")
    println("    - dish_indices_ptr:       $dish_indices_ptr")
    println("    - dish_separation_ew:     $dish_separation_ew")
    println("    - dish_separation_ns:     $dish_separation_ns")
    println("    - num_dishes:             $num_dishes")
    println("    - adc_frequency:          $adc_frequency")
    println("    - ntaps:                  $ntaps")
    println("    - nfreq:                  $nfreq")
    println("    - nchan:                  $nchan")
    println("    - frequency_channels_ptr: $frequency_channels_ptr")
    println("    - ntimes:                 $ntimes")
    # println("    - ndishes_i:              $ndishes_i")
    # println("    - ndishes_j:              $ndishes_j")
    println("    - nbeams_i:               $nbeams_i")
    println("    - nbeams_j:               $nbeams_j")
    println("    - nframes:                $nframes")

    dish_indices = if dish_indices_ptr != Ptr{Cvoid}()
        Int64[
            unsafe_load(Ptr{Cint}(dish_indices_ptr), loc_ew + num_dish_locations_ew * loc_ns + 1) for
            loc_ew in 0:(num_dish_locations_ew - 1), loc_ns in 0:(num_dish_locations_ns - 1)
        ]
    else
        reshape(Int64.(0:(num_dish_locations_ew * num_dish_locations_ns - 1)), num_dish_locations_ew, num_dish_locations_ns)
    end
    dish_indices::Array{Int64,2}
    @assert size(dish_indices) == (num_dish_locations_ew, num_dish_locations_ns)

    frequency_channels = if frequency_channels_ptr != Ptr{Cvoid}()
        Int64[unsafe_load(Ptr{Cint}(frequency_channels_ptr), n) for n in 1:nchan]
    else
        Int64[n for n in 1:nchan]
    end

    return run(
        Float(source_amplitude),
        Float(source_frequency),
        Float(source_position_ew),
        Float(source_position_ns),
        Int64(num_dish_locations_ew),
        Int64(num_dish_locations_ns),
        dish_indices,
        Float(dish_separation_ew),
        Float(dish_separation_ns),
        Int64(num_dishes),
        Float(adc_frequency),
        Int64(ntaps),
        Int64(nfreq),
        Int64(nchan),
        frequency_channels,
        Int64(ntimes),
        # Int64(ndishes_i),
        # Int64(ndishes_j),
        Int64(nbeams_i),
        Int64(nbeams_j),
        Int64(nframes),
        false,
    )
end

stored_xframes = nothing
function set_E(ptr::Ptr{UInt8}, sz::Int64, ndishes::Int64, npolrs::Int64, nfreqs::Int64, ntimes::Int64, frame_index::Int64)
    xframes = stored_xframes::AbstractVector{<:XFrame}
    if frame_index ∉ axes(xframes)[1]
        println("Frame index $frame_index does not exist in E field")
    end
    xframe = xframes[frame_index]::XFrame
    @assert length(xframe.data) == sz
    @assert size(xframe.data) == (ndishes, npolrs, nfreqs, ntimes)
    fill_buffer_Int4!(ptr, sz, xframe.data)
    return nothing
end

stored_bbbeamss = nothing
function set_A(ptr::Ptr{UInt8}, sz::Int64, ndishs::Int64, nbbbeams::Int64, npolrs::Int64, nfreqs::Int64, frame_index::Int64)
    bbbeamss = stored_bbbeamss::AbstractVector{<:BBBeams{Int8}}
    if frame_index ∉ axes(bbbeamss)[1]
        println("Frame index $frame_index does not exist in A field")
    end
    bbbeams = bbbeamss[frame_index]::BBBeams{Int8}
    if !(length(bbbeams.phases) * sizeof(Complex{Int8}) == sz)
        @show length(bbbeams.phases) sizeof(Complex{Int8}) sz
    end
    if !(size(bbbeams.phases) == (ndishs, nbbbeams, npolrs, nfreqs))
        @show size(bbbeams.phases) (ndishs, nbbbeams, npolrs, nfreqs)
    end
    @assert length(bbbeams.phases) * sizeof(Complex{Int8}) == sz
    @assert size(bbbeams.phases) == (ndishs, nbbbeams, npolrs, nfreqs)
    fill_buffer!(Ptr{Complex{Int8}}(ptr), sz, bbbeams.phases)
    return nothing
end

function set_J(ptr::Ptr{UInt8}, sz::Int64, ntimes::Int64, npolrs::Int64, nfreqs::Int64, nbbbeams::Int64, frame_index::Int64)
    # @show set_J ptr sz ntimes npolrs nfreqs nbbbeams frame_index
    bbbeamss = stored_bbbeamss::AbstractVector{<:BBBeams{Int8}}
    if frame_index ∉ axes(bbbeamss)[1]
        println("Frame index $frame_index does not exist in J field")
    end
    bbbeams = bbbeamss[frame_index]::BBBeams{Int8}
    @assert length(bbbeams.data) * sizeof(Int8) == sz
    @assert size(bbbeams.data) == (ntimes, npolrs, nfreqs, nbbbeams)
    fill_buffer_Int4!(ptr, sz, bbbeams.data)
    return nothing
end

stored_frbbeamss = nothing
function set_W(ptr::Ptr{UInt8}, sz::Int64, ndishsM::Int64, ndishsN::Int64, npolrs::Int64, nfreqs::Int64, frame_index::Int64)
    # @show set_W ptr sz ndishsM ndishsN npolrs nfreqs frame_index
    frbbeamss = stored_frbbeamss::AbstractVector{<:FRBBeams{Float32}}
    if frame_index ∉ axes(frbbeamss)[1]
        println("Frame index $frame_index does not exist in W field")
    end
    frbbeams = frbbeamss[frame_index]::FRBBeams{Float32}
    if !(length(frbbeams.phases) * sizeof(Complex{Float16}) == sz)
        @show length(frbbeams.phases) sizeof(Complex{Float16}) sz
    end
    if !(size(frbbeams.phases) == (ndishsM, ndishsN, npolrs, nfreqs))
        @show size(frbbeams.phases) (ndishsM, ndishsN, npolrs, nfreqs)
    end
    @assert length(frbbeams.phases) * sizeof(Complex{Float16}) == sz
    @assert size(frbbeams.phases) == (ndishsM, ndishsN, npolrs, nfreqs)
    fill_buffer!(Ptr{Complex{Float16}}(ptr), sz, frbbeams.phases)
    return nothing
end

function set_I(
    ptr::Ptr{UInt8}, sz::Int64, nfrbbeams_i::Int64, nfrbbeams_j::Int64, ntimes_ds::Int64, nfreqs::Int64, frame_index::Int64
)
    # @show set_I ptr sz nfrbbeams_i nfrbbeams_j ntimes_ds nfreqs frame_index
    frbbeamss = stored_frbbeamss::AbstractVector{<:FRBBeams{Float32}}
    if frame_index ∉ axes(frbbeamss)[1]
        println("Frame index $frame_index does not exist in I field")
    end
    frbbeams = frbbeamss[frame_index]::FRBBeams{Float32}
    if !(length(frbbeams.data) * sizeof(Float16) == sz)
        @show length(frbbeams.data) sizeof(Float16) sz
    end
    if !(size(frbbeams.data) == (nfrbbeams_i, nfrbbeams_j, ntimes_ds, nfreqs))
        @show size(frbbeams.data) (nfrbbeams_i, nfrbbeams_j, ntimes_ds, nfreqs)
    end
    @assert length(frbbeams.data) * sizeof(Float16) == sz
    @assert size(frbbeams.data) == (nfrbbeams_i, nfrbbeams_j, ntimes_ds, nfreqs)
    fill_buffer!(Ptr{Float16}(ptr), sz, frbbeams.data)
    return nothing
end

end

@show :FEngine9

nothing
