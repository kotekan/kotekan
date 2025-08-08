module FEngine

# Simulate the dishes and the F-engine to generate inputs for the X-engine

# We are using SI units

# Real F-engine has 16k frequencies, 4 taps

using Base.Threads
using CUDA
using CUDA.CUFFT
using CUDASIMDTypes
using FFTW
using LinearAlgebra
using MappedArrays
using PrettyTables
using ProgressMeter
using TypedTables

################################################################################
# Constants

const Float = Float32

# const FUInt = Dict(1 => UInt8, 2 => UInt16, 4 => UInt32, 8 => UInt64, 16 => UInt128)[sizeof(Float)]
# const FInt = Dict(1 => Int8, 2 => Int16, 4 => Int32, 8 => Int64, 16 => Int128)[sizeof(Float)]

const c₀ = 299_792_458          # speed of light in vacuum [m/s]

################################################################################
# Utilities

c2i4(c::Complex) = Int4x2(real(c), imag(c))
reim(x::Complex) = (x.re, x.im)
ftoi4(x::Complex{T}) where {T<:Real} = Int4x2(round.(Int8, clamp.(reim(x) .* T(7.5), -7, +7))...)

c2i4_swapped_withoffset(c::Complex) = Int4x2(imag(c) ⊻ 0x8, real(c) ⊻ 0x8)

Base.clamp(val::Complex, lo, hi) = Complex(clamp(real(val), lo, hi), clamp(imag(val), lo, hi))
Base.round(::Type{T}, val::Complex) where {T} = Complex{T}(round(T, real(val)), round(T, imag(val)))

################################################################################
# Sources

# We represent the source as complex - valued function of time.The two
# complex components represent the two polarisations.

# The product f₀ * t can become too large to be represented accurately
# via single precision. We need to use double precision.

struct Noise
    A::Float
end

struct Source
    f₀::Float
    A::Complex{Float}
    sinx::Float
    siny::Float
end

struct DispersedSource
    t₀::Float
    t₁::Float
    f₀::Float
    f₁::Float
    Δf::Float
    A::Complex{Float}
    sinx::Float
    siny::Float
end

function eval_noise(noise::Noise)
    return noise.A * randn(Complex{Float})
end

function eval_source(source::Source, t::Float)
    # Add a random phase offset (100) to avoid syncing up all sources
    return source.A * cispi(2 * source.f₀ * (t + 100))
end

function eval_source(source::Source, Δt::Float, n::Integer)
    # Add a random phase offset to avoid syncing up all sources
    n += 100

    # We want to calculate:
    #     t = Δt * n
    #     α = mod(f₀ * t, 1)
    # The problem is that `f₀ * t` can be very large and requires double precision.
    # We use fixed-point arithmetic instead.

    # α = mod(f₀ * Δt * n, 1)
    # S := 2^32
    # α = mod(S * f₀ * Δt * n, S) / S

    logS = 8 * sizeof(UInt32)
    # ldexp is slow, ensure it can be evaluated at compile time
    S = ldexp(Float(1), logS)
    X = round(UInt32, source.f₀ * Δt * S)
    # This can overflow, giving a result mod S, which is exactly what we want
    Y = (X * n) % UInt32

    α = Float(Y) * inv(S)

    return source.A * cispi(2 * α)
end

function dispersed_frequency(dispersed_source::DispersedSource, t::Float)
    # relation:
    #     t(f) = ts + T / f^2
    #     f(t) = sqrt(T / (t - ts))
    # conditions:
    #     t(f0)=t0, t(f1)=t0
    # solution:
    #     T  = f0^2 f1^2 (t1 - t0) / (f0^2 - f1^2)
    #     ts = (f0^2 t0 - f1^2 t1) / (f0^2 - f1^2)

    t₀ = dispersed_source.t₀
    t₁ = dispersed_source.t₁
    f₀ = dispersed_source.f₀
    f₁ = dispersed_source.f₁

    T = f₀^2 * f₁^2 * (t₁ - t₀) / (f₀^2 - f₁^2)
    ts = (f₀^2 * t₀ - f₁^2 * t₁) / (f₀^2 - f₁^2)

    f = sqrt(T / (t - ts))

    return f
end

function eval_dispersed_source(dispersed_source::DispersedSource, t::Float64)
    t₀ = dispersed_source.t₀
    t₁ = dispersed_source.t₁
    A = dispersed_source.A
    t₀ <= t <= t₁ || return Complex{Float}(0)

    f = dispersed_frequency(dispersed_source, Float32(t))
    Δf = dispersed_source.Δf
    return A * Complex{Float}(cispi(2 * f * t))   # * exp(-(f / Δf)^2 / 2)
end

################################################################################
# Dishes

struct Dishes
    locations::Vector{NTuple{2,Float}} # (ew, ns)
end

function make_dishes(
    dish_separation_ew::Float,
    dish_separation_ns::Float,
    num_dish_locations_ew::Int,
    num_dish_locations_ns::Int,
    dish_locations::AbstractVector{NTuple{2,Int}},
)
    # Find centre
    i_ew₀ = (num_dish_locations_ew - 1) / Float(2)
    i_ns₀ = (num_dish_locations_ns - 1) / Float(2)
    locations = NTuple{2,Float}[
        let
            x_ew = dish_separation_ew * (i_ew - i_ew₀)
            x_ns = dish_separation_ns * (i_ns - i_ns₀)
            (x_ew, x_ns)
        end for (i_ew, i_ns) in dish_locations
    ]
    return Dishes(locations)
end

################################################################################
# F-engine: ADC

struct ADC
    t₀::Float
    Δt::Float
end

struct ADCFrame{T}
    t₀::Float
    Δt::Float
    data::Array{T,3}            # [time, dish, polr]
end

function adc_sample(
    ::Type{T},
    noise::Noise,
    sources::Vector{Source},
    dispersed_source::DispersedSource,
    dishes::Dishes,
    adc::ADC,
    time0::Int,
    ntimes::Int,
) where {T<:Real}
    t₀ = adc.t₀
    Δt = adc.Δt
    nsources = length(sources)
    ndishes = length(dishes.locations)
    npolrs = 2

    source_samples = Complex{T}[eval_source(sources[source], Δt, time0 + time - 1) for source in 1:nsources, time in 1:ntimes]
    dispersed_source_samples = Complex{T}[
        eval_dispersed_source(dispersed_source, Δt * Float64(time0 + time - 1)) for time in 1:ntimes
    ]

    data = Array{T}(undef, ntimes, ndishes, npolrs)
    # @showprogress desc = "ADC" dt = 1 @threads for dish in 1:ndishes
    for dish in 1:ndishes
        location = dishes.locations[dish]
        dishx, dishy = location
        ϕs = Complex{T}[
            let
                f₀ = source.f₀
                t₁ = t₀ - source.sinx * dishx / c₀ - source.siny * dishy / c₀
                cispi(2 * f₀ * t₁)
            end for source in sources
        ]
        dϕs = let
            f₀ = dispersed_source.f₀
            t₁ = t₀ - dispersed_source.sinx * dishx / c₀ - dispersed_source.siny * dishy / c₀
            cispi(2 * f₀ * t₁)
        end
        for time in 1:ntimes
            val =
                eval_noise(noise) +
                sum(ϕs[source] * source_samples[source, time] for source in 1:nsources; init=Complex{T}(0)) +
                dϕs * dispersed_source_samples[time]
            data[time, dish, 1] = real(val)
            data[time, dish, 2] = imag(val)
        end
    end

    return ADCFrame(t₀, Δt, data)
end

################################################################################
# F-engine: FFT

# See Richard Shaw's PFB notes:
# <https://github.com/jrs65/pfb-inverse/blob/master/notes.ipynb>

struct PFB
    ntaps::Int                  # 4
    nsamples::Int               # 16384
    frequency_channels::Vector{Int}
    function PFB(ntaps::Int, nsamples::Int, frequency_channels::Vector{Int})
        @assert ntaps > 0
        @assert nsamples > 0
        @assert nsamples % 2 == 0
        @assert all(0 .<= frequency_channels .<= nsamples ÷ 2)
        return new(ntaps, nsamples, frequency_channels)
    end
end

struct FFrame{T}
    t₀::Float
    Δt::Float
    Δf::Float
    frequency_channels::Vector{Int}
    data::Array{Complex{T},4}   # [freq, time, dish, polr]
end

# Base.sinc isn't inlined, probably too complex
sinc1(x) = iszero(x) ? one(x) : sinpi(x) / (π * x)

# sinc-Hanning weight function, eqn. (11), with `N = U+2`
function sinc_hanning(s, M, U)
    # # Naive
    # # @assert 0 <= s < M * U
    # s′ = (2 * s - (M * U - 1)) / Float(2 * (M * U - 1)) # normalized to [-1/2; +1/2]

    # Erik, maximum window width
    # @assert -1 < 2 * s′ < +1
    s′ = (2 * s - (M * U - 1)) / Float(2 * (M * U + 1)) # normalized to [-1/2; +1/2]

    # # Richard Shaw
    # # @assert -1 < 2 * s′ < +1
    # s′ = (2 * s - (M * U)) / Float(2 * (M * U)) # normalized to [-1/2; +1/2)
    # # @assert -1 <= 2 * s′ < +1

    # # Erik, correct limit for M->1, U->1
    # # @assert -1 < 2 * s′ < +1
    # s′ = (2 * s - (M * U - 1)) / Float(2 * (M * U)) # normalized to (-1/2; +1/2)
    # # @assert -1 < 2 * s′ < +1

    # ∫ cos² π s = 1/2
    # ∫ sinc 4 s ≈ 3.21083
    # ∫ (cos² π s) (sinc 4 s) ≈ 0.385521

    # return cospi(s′)^2
    # return sinc1(M * s′)
    return cospi(s′)^2 * sinc1(M * s′)
end

# First-stage PFB
function upchan(data::AbstractVector{T}, ntaps::Int, nsamples::Int) where {T<:Real}
    @assert ntaps > 0
    @assert nsamples > 0
    old_ntimes = length(data)
    @assert old_ntimes % nsamples == 0
    new_ntimes = old_ntimes ÷ nsamples - (ntaps - 1)
    new_nfreqs = (ntaps * nsamples) ÷ 2 + 1
    @assert new_ntimes > 0
    window = T[sinc_hanning(sample - 1, ntaps, nsamples) for sample in 1:(ntaps * nsamples)]
    input = Array{T}(undef, ntaps * nsamples, new_ntimes)
    output = Array{Complex{T}}(undef, new_nfreqs, new_ntimes)
    FFT = plan_rfft(input, 1)
    for new_time in 1:new_ntimes
        for sample in 1:(ntaps * nsamples)
            w = window[sample]
            input[sample, new_time] = w * data[(new_time - 1) * nsamples + sample]
        end
    end
    mul!(output, FFT, input)
    return output[begin:ntaps:end, :]
end

# Second-stage PFB
function upchan(data::AbstractVector{Complex{T}}, ntaps::Int, nsamples::Int) where {T<:Real}
    @assert ntaps > 0
    @assert nsamples > 0
    old_ntimes = length(data)
    @assert old_ntimes % nsamples == 0
    new_ntimes = old_ntimes ÷ nsamples - (ntaps - 1)
    @assert new_ntimes > 0
    new_nfreqs = nsamples
    window = T[sinc_hanning(sample - 1, ntaps, nsamples) for sample in 1:(ntaps * nsamples)]
    new_data = Array{Complex{T}}(undef, new_nfreqs, new_ntimes)
    for new_time in 1:new_ntimes
        for new_freq in 1:new_nfreqs
            res = zero(Complex{T})
            for sample in 1:(ntaps * nsamples)
                w = window[sample]
                # eqn (17)
                U = new_nfreqs
                S = nsamples
                u = new_freq - 1
                s = sample - 1
                ϕ = conj(cispi((2 * u - (U - 1)) * s % (2 * S) / T(S)))
                res += w * ϕ * data[(new_time - 1) * nsamples + sample]
            end
            new_data[new_freq, new_time] = res
        end
    end
    return new_data
end

function f_engine(pfb::PFB, adcframe::ADCFrame{T}) where {T<:Real}
    ntaps = pfb.ntaps
    nsamples = pfb.nsamples
    frequency_channels = pfb.frequency_channels
    @assert ntaps > 0
    @assert nsamples > 0

    t₀ = adcframe.t₀
    Δt = adcframe.Δt
    ntimes, ndishes, npolrs = size(adcframe.data)
    @assert ntimes % nsamples == 0

    Δt′ = Δt * nsamples
    t₀′ = t₀ - Δt / 2 + ntaps * Δt′ / 2
    Δf′ = 1 / Δt′

    ntimes′ = max(0, ntimes ÷ nsamples - pfb.ntaps + 1)

    window = T[sinc_hanning(sample - 1, ntaps, nsamples) for sample in 1:(ntaps * nsamples)]

    indatas = [Array{T}(undef, ntaps * nsamples) for thread in 1:Threads.nthreads()]
    outdatas = [Array{Complex{T}}(undef, ntaps * nsamples ÷ 2 + 1) for thread in 1:Threads.nthreads()]
    FFTs = [plan_rfft(indatas[thread], 1) for thread in 1:Threads.nthreads()]

    fdata = Array{Complex{T}}(undef, length(frequency_channels), ntimes′, ndishes, npolrs)
    # @showprogress desc = "PFB" dt = 1 @threads for time′_dish_polr in CartesianIndices((ntimes′, ndishes, npolrs))
    for time′_dish_polr in CartesianIndices((ntimes′, ndishes, npolrs))
        thread = Threads.threadid()
        FFT = FFTs[thread]
        indata = indatas[thread]
        outdata = outdatas[thread]

        time′, dish, polr = Tuple(time′_dish_polr)

        time0 = (time′ - 1) * nsamples + 1
        time1 = time0 + ntaps * nsamples - 1

        adcdata = @view adcframe.data[time0:time1, dish, polr]
        data = @view fdata[:, time′, dish, polr]

        for sample in 1:(ntaps * nsamples)
            w = window[sample] / (nsamples ÷ 2)
            indata[sample] = w * adcdata[sample]
        end
        mul!(outdata, FFT, indata)
        for freq in 1:length(frequency_channels)
            # Choose only every ntap-th frequency
            data[freq] = outdata[ntaps * frequency_channels[freq] + 1]
        end
    end
    @assert all(isfinite, fdata)

    return FFrame{T}(t₀′, Δt′, Δf′, frequency_channels, fdata)
end

# Transform 16 dishes simultaneously
function f_engine_16(pfb::PFB, adcframe::ADCFrame{T}) where {T<:Real}
    ntaps = pfb.ntaps
    nsamples = pfb.nsamples
    frequency_channels = pfb.frequency_channels
    @assert ntaps > 0
    @assert nsamples > 0

    t₀ = adcframe.t₀
    Δt = adcframe.Δt
    ntimes, ndishes, npolrs = size(adcframe.data)
    @assert ntimes % nsamples == 0

    Δt′ = Δt * nsamples
    t₀′ = t₀ - Δt / 2 + ntaps * Δt′ / 2
    Δf′ = 1 / (2 * Δt)

    ntimes′ = max(0, ntimes ÷ nsamples - pfb.ntaps + 1)

    groupsize = 16
    @assert ndishes % groupsize == 0

    indatas = [Array{T}(undef, groupsize, ntaps * nsamples) for thread in 1:Threads.nthreads()]
    outdatas = [Array{Complex{T}}(undef, groupsize, ntaps * nsamples ÷ 2 + 1) for thread in 1:Threads.nthreads()]
    FFTs = [plan_rfft(indatas[thread], 2) for thread in 1:Threads.nthreads()]

    fdata = Array{Complex{T}}(undef, length(frequency_channels), ntimes′, ndishes, npolrs)
    @showprogress desc = "PFB" dt = 1 @threads for time′_dish_polr in CartesianIndices((ntimes′, ndishes ÷ groupsize, npolrs))
        thread = Threads.threadid()
        FFT = FFTs[thread]
        indata = indatas[thread]
        outdata = outdatas[thread]

        time′, group, polr = Tuple(time′_dish_polr)
        dish0 = (group - 1) * groupsize
        dish1 = dish0 + groupsize

        time0 = (time′ - 1) * nsamples
        time1 = time0 + ntaps * nsamples

        adcdata = @view adcframe.data[(time0 + 1):time1, (dish0 + 1):dish1, polr]
        data = @view fdata[:, time′, (dish0 + 1):dish1, polr]

        for sample in 0:(ntaps * nsamples - 1)
            w = sinc_hanning(sample, ntaps, nsamples) / (nsamples ÷ 2)
            for dish in 0:(groupsize - 1)
                indata[dish + 1, sample + 1] = adcdata[sample + 1, dish + 1] * w
            end
        end
        mul!(outdata, FFT, indata)
        for freq in 1:length(frequency_channels)
            for dish in 1:groupsize
                # Choose only every ntap-th frequency
                data[freq, dish] = outdata[dish, ntaps * frequency_channels[freq] + 1]
            end
        end
    end

    return FFrame{T}(t₀′, Δt′, Δf′, frequency_channels, fdata)
end

function prepare_fft_input(indata::AbstractArray{T,3}, adcdata::AbstractArray{T,2}) where {T}
    ntaps_nsamples, ntimes′, nspectators = size(indata)
    @assert ntaps_nsamples == ntaps * nsamples
    for spectator in 1:nspectators
        for time′ in 1:ntimes′
            for sample in 0:(ntaps * nsamples - 1)
                w = sinc_hanning(sample, ntaps, nsamples) / (nsamples ÷ 2)
                indata[sample + 1, time′, spectator] = w * adcdata[time′ * nsamples + sample + 1, spectator]
            end
        end
    end
    return nothing
end

function f_engine_cuda(pfb::PFB, adcframe::ADCFrame{T}) where {T<:Real}
    ntaps = pfb.ntaps
    nsamples = pfb.nsamples
    frequency_channels = pfb.frequency_channels
    @assert ntaps > 0
    @assert nsamples > 0

    t₀ = adcframe.t₀
    Δt = adcframe.Δt
    ntimes, ndishes, npolrs = size(adcframe.data)
    @assert ntimes % nsamples == 0

    Δt′ = Δt * nsamples
    t₀′ = t₀ - Δt / 2 + ntaps * Δt′ / 2
    Δf′ = 1 / (2 * Δt)

    ntimes′ = max(0, ntimes ÷ nsamples - pfb.ntaps + 1)

    @assert CUDA.functional()
    adcdata = CuArray(adcframe.data)
    indata = CuArray{T}(undef, ntaps * nsamples, ntimes′, ndishes, npolrs)
    prepare_fft_input(indata, adcdata)

    outdatas = [Array{Complex{T}}(undef, ntaps * nsamples ÷ 2 + 1) for thread in 1:Threads.nthreads()]
    FFTs = [plan_rfft(indatas[thread], 1) for thread in 1:Threads.nthreads()]

    fdata = Array{Complex{T}}(undef, length(frequency_channels), ntimes′, ndishes, npolrs)
    @showprogress desc = "PFB" dt = 1 @threads for time′_dish_polr in CartesianIndices((ntimes′, ndishes, npolrs))
        thread = Threads.threadid()
        FFT = FFTs[thread]
        indata = indatas[thread]
        outdata = outdatas[thread]

        time′, dish, polr = Tuple(time′_dish_polr)

        time0 = (time′ - 1) * nsamples
        time1 = time0 + ntaps * nsamples

        adcdata = @view adcframe.data[(time0 + 1):time1, dish, polr]
        data = @view fdata[:, time′, dish, polr]

        for sample in 0:(ntaps * nsamples - 1)
            indata[sample + 1] = adcdata[sample + 1] * sinc_hanning(sample, ntaps, nsamples)
        end
        mul!(outdata, FFT, indata)
        for freq in 1:length(frequency_channels)
            # Choose only every ntap-th frequency
            data[freq] = outdata[ntaps * frequency_channels[freq] + 1]
        end
    end

    return FFrame{T}(t₀′, Δt′, Δf′, frequency_channels, fdata)
end

################################################################################
# F-engine: corner turn

struct XFrame{T}
    t₀::Float
    Δt::Float
    Δf::Float
    frequency_channels::Vector{Int}
    data::Array{Complex{T},4}   # [time, freq, dish, polr]
end

function corner_turn(fframe::FFrame{T}) where {T}
    fdata = fframe.data
    xdata = permutedims(fdata, (2, 1, 3, 4))
    return XFrame{T}(fframe.t₀, fframe.Δt, fframe.Δf, fframe.frequency_channels, xdata)
end

################################################################################
# F-engine: quantize

function quantize(::Type{I}, xframe::XFrame{T}) where {I<:Integer,T<:Real}
    xdata = xframe.data
    ntimes, nfreqs, ndishes, npolrs = size(xdata)

    # idata = similar(xdata, Complex{I})
    # @showprogress desc = "Quantize" dt = 1 @threads for freq_dish_polr in CartesianIndices((nfreqs, ndishes, npolrs))
    #     freq, dish, polr = Tuple(freq_dish_polr)
    # 
    #     xdata1 = @view xdata[:, freq, dish, polr]
    #     idata1 = @view idata[:, freq, dish, polr]
    # 
    #     norm2 = norm(xdata1, 2) / sqrt(T(length(xdata1)))
    #     # Set the noise level to 1/3
    #     scale = T(7.5) / T(3) / norm2
    # 
    #     idata1 .= round.(Int8, clamp.(scale * xdata1, T(-7), T(+7)))
    # end

    # maxabs = sqrt(maximum(abs2, xdata))
    # scale = T(7.5) / maxabs

    norm1 = norm(xdata, 1) / T(length(xdata))
    norm2 = norm(xdata, 2) / sqrt(T(length(xdata)))
    norminf = norm(xdata, Inf)
    println("    norm1:   $norm1")
    println("    norm2:   $norm2")
    println("    norminf: $norminf")

    scale = T(7.5)
    nclipped = sum(x -> (abs(real(x)) > 7.5) + (abs(imag(x)) > 7.5), scale * xdata)
    nclipped_fraction = round(nclipped / (2 * length(xdata)); sigdigits=2)
    idata = round.(I, clamp.(scale * xdata, T(-7), T(+7)))
    println("    nclipped: $nclipped (fraction $nclipped_fraction)")

    values = -7:+7
    counts = [sum(x -> (real(x) == val) + (imag(x) == val), idata) for val in values]
    percents = round.(counts * 100 / (2 * length(idata)); digits=1)
    stats = Table(; value=values, count=counts, percent=percents)
    println("Quantization statistics:")
    pretty_table(stats; header=["value", "count", "percent"], tf=tf_borderless)

    return XFrame{I}(xframe.t₀, xframe.Δt, xframe.Δf, xframe.frequency_channels, idata)
end

function quantize(::Type{I}, fframe::FFrame{T}) where {I<:Integer,T<:Real}
    fdata = fframe.data
    nfreqs, ntimes, ndishes, npolrs = size(fdata)

    scale = T(7.5)

    #TODO norm1 = norm(fdata, 1) / T(length(fdata))
    #TODO norm2 = norm(fdata, 2) / sqrt(T(length(fdata)))
    #TODO norminf = norm(fdata, Inf)
    #TODO nclipped = sum(x -> (abs(real(x)) > scale) + (abs(imag(x)) > scale), fdata)
    #TODO nclipped_fraction = nclipped / (2 * length(fdata))
    #TODO println("    norm1:   $norm1")
    #TODO println("    norm2:   $norm2")
    #TODO println("    norminf: $norminf")
    #TODO println("    nclipped: $nclipped (fraction $(round(nclipped_fraction; sigdigits=2)))")

    idata = round.(I, clamp.(scale * fdata, T(-7), T(+7)))

    #TODO values = -7:+7
    #TODO counts = [sum(x -> (real(x) == val) + (imag(x) == val), idata) for val in values]
    #TODO percents = round.(counts * 100 / (2 * length(idata)); digits=1)
    #TODO stats = Table(; value=values, count=counts, percent=percents)
    #TODO println("Quantization statistics:")
    #TODO pretty_table(stats; header=["value", "count", "percent"], tf=tf_borderless)

    return FFrame{I}(fframe.t₀, fframe.Δt, fframe.Δf, fframe.frequency_channels, idata)
end

################################################################################
# Baseband phases

struct BasebandBeams
    angles::Vector{NTuple{2,Float}} # (ew, ns), in radians
end

function make_baseband_beams(num_beams_ew::Int, num_beams_ns::Int, beam_separation_ew::Float, beam_separation_ns::Float)
    # Find centre
    i_ew₀ = (1 + num_beams_ew) / Float(2)
    i_ns₀ = (1 + num_beams_ns) / Float(2)
    angles = NTuple{2,Float}[let
        θ_ew = beam_separation_ew * (i_ew - i_ew₀)
        θ_ns = beam_separation_ns * (i_ns - i_ns₀)
        (θ_ew, θ_ns)
    end for i_ns in 1:num_beams_ns for i_ew in 1:num_beams_ew]
    return BasebandBeams(angles)
end

struct BasebandPhases
    phases::Array{Complex{Int8},4} # [dish, beam, polr, freq]
end

function baseband_phases(dishes::Dishes, frequencies::AbstractVector{Float}, beams::BasebandBeams)
    ndishes = length(dishes.locations)
    nbeams = length(beams.angles)
    npolrs = 2
    nfreqs = length(frequencies)

    # We choose `A` independent of polarization
    A = Complex{Int8}[
        let
            dish_ew, dish_ns = dishes.locations[dish]
            θ_ew, θ_ns, = beams.angles[beam]
            Δt = sin(θ_ew) * dish_ew / c₀ + sin(θ_ns) * dish_ns / c₀
            f = frequencies[freq]
            round(Int8, clamp(Float(127.5) * cispi(2 * f * Δt), Float(-127), Float(+127)))
        end for dish in 1:ndishes, beam in 1:nbeams, polr in 1:npolrs, freq in 1:nfreqs
    ]
    return BasebandPhases(A)
end

################################################################################
# F-engine: run

# This struct must be mutable so that it is heap-allocated and the
# calling C++ code can protect it from garbage collection
mutable struct Setup
    # Sources
    noise::Noise
    sources::Vector{Source}
    dispersed_source::DispersedSource

    # Dishes
    dishes::Dishes

    # ADC
    adc::ADC

    # PFB
    pfb::PFB

    # Baseband beams
    bb_beams::BasebandBeams
end

function setup(
    noise_amplitude::Float,
    source_channels::Vector{Float},
    source_amplitudes::Vector{Float},
    dispersed_source_start_time::Float,
    dispersed_source_end_time::Float,
    dispersed_source_start_frequency::Float,
    dispersed_source_end_frequency::Float,
    dispersed_source_linewidth::Float,
    dispersed_source_amplitude::Float,
    source_position_ew::Float,
    source_position_ns::Float,
    num_dish_locations_ew::Int64,
    num_dish_locations_ns::Int64,
    dish_indices::Array{Int64,2},
    dish_separation_ew::Float,
    dish_separation_ns::Float,
    ndishes::Int64,
    adc_frequency::Float,
    ntaps::Int64,
    nsamples::Int64,
    nfreqs::Int64,
    frequency_channels::Vector{Int64},
    ntimes::Int64,
    bb_num_beams_ew::Int64,
    bb_num_beams_ns::Int64,
    bb_beam_separation_ew::Float,
    bb_beam_separation_ns::Float,
    nframes::Int64,
)
    println("F-Engine setup:")
    println("    - noise_amplitude:                  $noise_amplitude")
    println("    - source_channels:                  $source_channels")
    println("    - source_amplitudes:                $source_amplitudes")
    println("    - dispersed_source_start_time:      $dispersed_source_start_time")
    println("    - dispersed_source_end_time:        $dispersed_source_end_time")
    println("    - dispersed_source_start_frequency: $dispersed_source_start_frequency")
    println("    - dispersed_source_end_frequency:   $dispersed_source_end_frequency")
    println("    - dispersed_source_linewidth:       $dispersed_source_linewidth")
    println("    - dispersed_source_amplitude:       $dispersed_source_amplitude")
    println("    - source_position_ew:               $source_position_ew")
    println("    - source_position_ns:               $source_position_ns")
    println("    - num_dish_locations_ew:            $num_dish_locations_ew")
    println("    - num_dish_locations_ns:            $num_dish_locations_ns")
    println("    - dish_indices:                     $dish_indices")
    println("    - dish_separation_ew:               $dish_separation_ew")
    println("    - dish_separation_ns:               $dish_separation_ns")
    println("    - ndishes:                          $ndishes")
    println("    - adc_frequency:                    $adc_frequency")
    println("    - ntaps:                            $ntaps")
    println("    - nsamples:                         $nsamples")
    println("    - nfreqs:                           $nfreqs")
    println("    - frequency_channels:               $frequency_channels")
    println("    - ntimes:                           $ntimes")
    println("    - bb_num_beams_ew:                  $bb_num_beams_ew")
    println("    - bb_num_beams_ns:                  $bb_num_beams_ns")
    println("    - bb_beam_separation_ew:            $bb_beam_separation_ew")
    println("    - bb_beam_separation_ns:            $bb_beam_separation_ns")
    println("    - nframes:                          $nframes")

    println("Setting up sources...")
    noise = Noise(noise_amplitude)
    sources = Source[
        let
            f₀ = channel * adc_frequency / nsamples
            A = Complex{Float}(amplitude)
            sin_ew = sin(source_position_ew) # east-west
            sin_ns = sin(source_position_ns) # north-south
            Source(f₀, A, sin_ew, sin_ns)
        end for (channel, amplitude) in zip(source_channels, source_amplitudes)
    ]
    dispersed_source = let
        t₀ = dispersed_source_start_time
        t₁ = dispersed_source_end_time
        f₀ = dispersed_source_start_frequency
        f₁ = dispersed_source_end_frequency
        Δf = dispersed_source_linewidth
        A = Complex{Float}(dispersed_source_amplitude)
        sin_ew = sin(source_position_ew) # east-west
        sin_ns = sin(source_position_ns) # north-south
        DispersedSource(t₀, t₁, f₀, f₁, Δf, A, sin_ew, sin_ns)
    end

    println("Setting up dishes...")
    dishes = let
        num_dish_locations = num_dish_locations_ew * num_dish_locations_ns
        dish_locations = fill((-1, -1), ndishes)
        ndishes_seen = 0
        @assert size(dish_indices) == (num_dish_locations_ew, num_dish_locations_ns)
        for loc_ns in 0:(num_dish_locations_ns - 1), loc_ew in 0:(num_dish_locations_ew - 1)
            dish = dish_indices[loc_ew + 1, loc_ns + 1]
            @assert dish == -1 || 0 <= dish < ndishes
            if dish >= 0
                ndishes_seen += 1
                @assert dish_locations[dish + 1] == (-1, -1)
                dish_locations[dish + 1] = (loc_ew, loc_ns)
            end
        end
        @assert ndishes_seen == ndishes
        make_dishes(dish_separation_ew, dish_separation_ns, num_dish_locations_ew, num_dish_locations_ns, dish_locations)
    end

    println("Setting up ADC...")
    adc = let
        t₀ = zero(adc_frequency)
        Δt = 1 / adc_frequency
        ADC(t₀, Δt)
    end

    println("Setting up PFB...")
    pfb = PFB(ntaps, nsamples, frequency_channels)

    println("Setting up baseband beams...")
    bb_beams = make_baseband_beams(bb_num_beams_ew, bb_num_beams_ns, bb_beam_separation_ew, bb_beam_separation_ns)

    return Setup(noise, sources, dispersed_source, dishes, adc, pfb, bb_beams)
end

# frequencies = mappedarray(c -> fframe.Δf * c, frequency_channels)
# A = baseband_phases(dishes, frequencies, bb_beams)

function setup(
    noise_amplitude=0.0,
    nsources=1,
    source_channels_ptr=Ptr{Cfloat}(),
    source_amplitudes_ptr=Ptr{Cfloat}(),
    dispersed_source_start_time=0.1,
    dispersed_source_end_time=2.5,
    dispersed_source_start_frequency=1500e6,
    dispersed_source_end_frequency=300e6,
    dispersed_source_linewidth=1.0,
    dispersed_source_amplitude=0.0,
    source_position_ew=0.02,
    source_position_ns=0.03,
    num_dish_locations_ew=8,
    num_dish_locations_ns=8,
    dish_indices_ptr=Ptr{Cint}(),
    dish_separation_ew=6.3,
    dish_separation_ns=8.5,
    ndishes=num_dish_locations_ew * num_dish_locations_ns,
    adc_frequency=3.0e+9,
    ntaps=4,
    nsamples=128,
    nfreqs=64,
    frequency_channels_ptr=Ptr{Cint}(),
    ntimes=64,
    bb_num_beams_ew=4,
    bb_num_beams_ns=4,
    bb_beam_separation_ew=0.015,
    bb_beam_separation_ns=0.015,
    nframes=2,
)
    source_channels_ptr = Ptr{Cfloat}(source_channels_ptr)
    source_amplitudes_ptr = Ptr{Cfloat}(source_amplitudes_ptr)
    dish_indices_ptr = Ptr{Cint}(dish_indices_ptr)
    frequency_channels_ptr = Ptr{Cint}(frequency_channels_ptr)

    source_channels = if source_channels_ptr != C_NULL
        Float[unsafe_load(source_channels_ptr, source) for source in 1:nsources]
    else
        Float[12 + source for source in 0:(nsources - 1)]
    end
    source_amplitudes = if source_amplitudes_ptr != C_NULL
        Float[unsafe_load(source_amplitudes_ptr, source) for source in 1:nsources]
    else
        Float[1 for source in 1:nsources]
    end

    dish_indices = if dish_indices_ptr != C_NULL
        Int64[
            unsafe_load(dish_indices_ptr, loc_ew + num_dish_locations_ew * loc_ns + 1) for
            loc_ew in 0:(num_dish_locations_ew - 1), loc_ns in 0:(num_dish_locations_ns - 1)
        ]
    else
        reshape(Int64.(0:(num_dish_locations_ew * num_dish_locations_ns - 1)), num_dish_locations_ew, num_dish_locations_ns)
    end
    dish_indices::Array{Int64,2}
    @assert size(dish_indices) == (num_dish_locations_ew, num_dish_locations_ns)

    frequency_channels = if frequency_channels_ptr != C_NULL
        Int64[unsafe_load(frequency_channels_ptr, n) for n in 1:nfreqs]
    else
        Int64[n for n in 1:nfreqs]
    end

    return setup(
        Float(noise_amplitude),
        source_channels::Vector{Float},
        source_amplitudes::Vector{Float},
        Float(dispersed_source_start_time),
        Float(dispersed_source_end_time),
        Float(dispersed_source_start_frequency),
        Float(dispersed_source_end_frequency),
        Float(dispersed_source_linewidth),
        Float(dispersed_source_amplitude),
        Float(source_position_ew),
        Float(source_position_ns),
        Int64(num_dish_locations_ew),
        Int64(num_dish_locations_ns),
        dish_indices::Array{Int64,2},
        Float(dish_separation_ew),
        Float(dish_separation_ns),
        Int64(ndishes),
        Float(adc_frequency),
        Int64(ntaps),
        Int64(nsamples),
        Int64(nfreqs),
        frequency_channels::Vector{Int64},
        Int64(ntimes),
        Int64(bb_num_beams_ew),
        Int64(bb_num_beams_ns),
        Float(bb_beam_separation_ew),
        Float(bb_beam_separation_ns),
        Int64(nframes),
    )
end

################################################################################
# Readout

# function fill_buffer_Int4!(ptr::Ptr{UInt8}, sz::Int64, data::AbstractArray{Complex{Int8}})
#     @assert sz == length(data)  # sizeof(Complex{Int4}) should be 1
#     @threads for i in 1:length(data)
#         val = data[i]
#         re, im = real(val), imag(val)
#         re4 = (re % UInt8) & 0x0f
#         im4 = (im % UInt8) & 0x0f
#         cint4 = re4 << 0 | im4 << 4
#         unsafe_store!(ptr, cint4, i)
#     end
# end
# 
# function fill_buffer!(ptr::Ptr, sz::Int64, data::AbstractArray)
#     @assert sz == length(data) * sizeof(eltype(ptr))
#     @threads for i in 1:length(data)
#         unsafe_store!(ptr, data[i], i)
#     end
# end

function set_dish_positions!(ptr::Ptr{UInt8}, sz::Int64, ndishes::Int64, setup::Setup)
    dishes = setup.dishes::Dishes
    locations_src = dishes.locations::Array{<:NTuple{2,<:Real},1}

    @assert (ndishes,) == size(locations_src)

    locations_dst = unsafe_wrap(Array, reinterpret(Ptr{NTuple{2,Float}}, ptr), (ndishes,))
    @assert sizeof(locations_dst) == sz

    @assert size(locations_dst) == size(locations_src)

    locations_dst .= locations_src

    return nothing
end

function set_bb_beam_positions!(ptr::Ptr{UInt8}, sz::Int64, nbbbeams::Int64, setup::Setup)
    bb_beams = setup.bb_beams::BasebandBeams
    angles_src = bb_beams.angles::Array{<:NTuple{2,<:Real},1}

    @assert (nbbbeams,) == size(angles_src)

    angles_dst = unsafe_wrap(Array, reinterpret(Ptr{NTuple{2,Float}}, ptr), (nbbbeams,))
    @assert sizeof(angles_dst) == sz
    @assert size(angles_dst) == size(angles_src)

    angles_dst .= angles_src

    return nothing
end

function set_A!(ptr::Ptr{UInt8}, sz::Int64, ndishes::Int64, nbbbeams::Int64, npolrs::Int64, nfreqs::Int64, setup::Setup)
    println("Calculating baseband phases...")
    Δf = inv(setup.adc.Δt) / setup.pfb.nsamples
    frequencies = mappedarray(c -> Δf * c, setup.pfb.frequency_channels)
    bbphases = baseband_phases(setup.dishes, frequencies, setup.bb_beams)
    Asrc = bbphases.phases::Array{<:Complex{<:Integer},4}

    @assert (ndishes, nbbbeams, npolrs, nfreqs) == size(Asrc)

    Adst = unsafe_wrap(Array, reinterpret(Ptr{Complex{Int8}}, ptr), (ndishes, nbbbeams, npolrs, nfreqs))
    @assert sizeof(Adst) == sz
    @assert size(Adst) == size(Asrc)
    Adst .= Asrc

    return nothing
end

function set_W(ptr::Ptr{UInt8}, sz::Int64, ndishsM::Int64, ndishsN::Int64, npolrs::Int64, nfreqs::Int64, frame_index::Int64)
    for i in 1:sz
        unsafe_store!(ptr, 0, i)
    end
    return nothing
end

function set_E!(
    ptr::Ptr{UInt8}, sz::Int64, ndishes::Int64, npolrs::Int64, nfreqs::Int64, ntimes::Int64, setup::Setup, frame_index::Int64
)
    E = unsafe_wrap(Array, reinterpret(Ptr{Int4x2}, ptr), (ndishes, npolrs, nfreqs, ntimes))
    @assert sizeof(E) == sz

    adc = setup.adc
    pfb = setup.pfb

    # TODO: Reuse memory, don't allocate, pass buffers in, this should save on GC time

    if ndishes == 64
        chunk_size = 256        # Pathfinder: use 32 threads
    elseif ndishes == 256
        chunk_size = 128        # HIRAX: use 32 threads
    elseif ndishes == 512
        chunk_size = 32         # CHORD: conserve memory
    elseif ndishes == 1024
        chunk_size = 32         # CHIME: conserve memory
    else
        chunk_size = 128
    end
    @assert ntimes % chunk_size == 0
    nchunks = ntimes ÷ chunk_size
    walltimes = zeros(4, nchunks)
    @showprogress desc = "Generating E-field" dt = 1 @threads for chunk_index in 1:nchunks
        # for chunk_index in 1:nchunks
        # println("Generating E-field (chunk $chunk_index/$nchunks)...")
        time0 = (chunk_index - 1) * chunk_size + 1
        time1 = time0 + chunk_size - 1

        # println("ADC sampling...")
        t0 = time()
        adc_time0 = (frame_index - 1) * ntimes + (chunk_index - 1) * chunk_size
        adc_ntimes = pfb.nsamples * (chunk_size + pfb.ntaps - 1)
        adcframe = adc_sample(Float, setup.noise, setup.sources, setup.dispersed_source, setup.dishes, adc, adc_time0, adc_ntimes)
        t1 = time()
        walltimes[1, chunk_index] = t1 - t0

        # println("PFB...")
        t0 = time()
        fframe = f_engine(pfb, adcframe)
        #TODO fframe = f_engine_16(pfb, adcframe)
        t1 = time()
        walltimes[2, chunk_index] = t1 - t0

        # println("Quantizing...")
        t0 = time()
        iframe = quantize(Int8, fframe)
        t1 = time()
        walltimes[3, chunk_index] = t1 - t0

        # println("Corner turn...")
        t0 = time()
        I = iframe.data
        permutedims!((@view E[:, :, :, time0:time1]), mappedarray(c2i4_swapped_withoffset, I), (3, 4, 1, 2))
        t1 = time()
        walltimes[4, chunk_index] = t1 - t0
    end

    # @show walltimes

    return nothing
end

function set_J(ptr::Ptr{UInt8}, sz::Int64, ntimes::Int64, npolrs::Int64, nfreqs::Int64, nbbbeams::Int64, frame_index::Int64)
    for i in 1:sz
        unsafe_store!(ptr, 0, i)
    end
    return nothing
end

function set_I(
    ptr::Ptr{UInt8}, sz::Int64, nfrbbeams_i::Int64, nfrbbeams_j::Int64, ntimes_ds::Int64, nfreqs::Int64, frame_index::Int64
)
    for i in 1:sz
        unsafe_store!(ptr, 0, i)
    end
    return nothing
end

end
