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

# const Float = Float32
const Float = Float64

const c₀ = 299_792_458          # speed of light in vacuum

################################################################################
# Utilities

c2i4(c::Complex) = Int4x2(real(c), imag(c))

reim(x::Complex) = (x.re, x.im)

ftoi4(x::Complex{T}) where {T<:Real} = Int4x2(round.(Int8, clamp.(reim(x) .* T(7.5), -7, +7))...)

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

function eval_noise(noise::Noise)
    return noise.A * randn(Complex{Float})
end

function eval_source(source::Source, dishx::Float, dishy::Float, t₀::Float)
    # Add a random phase offset (100) to avoid syncing up all sources
    t = t₀ + source.sinx * dishx / c₀ + source.siny * dishy / c₀ + 100
    return source.A * cispi(2 * source.f₀ * t)
end

function eval_sources(noise::Noise, sources::Vector{Source}, dishx::Float, dishy::Float, t₀::Float)
    return eval_noise(noise) + sum(eval_source(source, dishx, dishy, t₀) for source in sources)
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
    # Find centre
    i_ew₀ = (num_dish_locations_ew - 1) / Float(2)
    i_ns₀ = (num_dish_locations_ns - 1) / Float(2)
    locations = NTuple{2,Float}[
        let
            x_ew = dish_separation_ew * (i_ew - i_ew₀)
            x_ns = dish_separation_ns * (i_ns - i_ns₀)
            (x_ns, x_ew)
        end for (i_ns, i_ew) in dish_locations
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

function adc_sample(::Type{T}, source::Source, dishes::Dishes, adc::ADC, time::Int, dish::Int, polr::Int) where {T<:Real}
    t₀ = adc.t₀
    Δt = adc.Δt

    location = dishes.locations[dish]
    dishx, dishy = location
    t = t₀ + Δt * (time - 1)
    val = eval_source(source, dishx, dishy, t)
    res = reim(val)[polr]

    return res
end

function adc_sample(::Type{T}, noise::Noise, sources::Vector{Source}, dishes::Dishes, adc::ADC, ntimes::Int) where {T<:Real}
    t₀ = adc.t₀
    Δt = adc.Δt
    ndishes = length(dishes.locations)
    npolrs = 2

    # data = Array{T}(undef, ntimes, ndishes, npolrs)
    # @showprogress desc = "ADC" dt = 1 @threads for time_dish in CartesianIndices((ntimes, ndishes))
    #     time, dish = Tuple(time_dish)
    #     location = dishes.locations[dish]
    #     dishx, dishy = location
    #     t = t₀ + Δt * (time - 1)
    #     val = eval_sources(noise, sources, dishx, dishy, t)
    #     data[time, dish, 1] = real(val)
    #     data[time, dish, 2] = imag(val)
    # end

    data = Array{T}(undef, ntimes, ndishes, npolrs)
    @showprogress desc = "ADC" dt = 1 @threads for dish in 1:ndishes
        location = dishes.locations[dish]
        dishx, dishy = location
        for time in 1:ntimes
            t = t₀ + Δt * (time - 1)
            val = eval_sources(noise, sources, dishx, dishy, t)
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
    Δf′ = 1 / (2 * Δt)

    ntimes′ = max(0, ntimes ÷ nsamples - pfb.ntaps + 1)

    window = T[sinc_hanning(sample - 1, ntaps, nsamples) for sample in 1:(ntaps * nsamples)]

    indatas = [Array{T}(undef, ntaps * nsamples) for thread in 1:Threads.nthreads()]
    outdatas = [Array{Complex{T}}(undef, ntaps * nsamples ÷ 2 + 1) for thread in 1:Threads.nthreads()]
    FFTs = [plan_rfft(indatas[thread], 1) for thread in 1:Threads.nthreads()]

    fdata = Array{Complex{T}}(undef, length(frequency_channels), ntimes′, ndishes, npolrs)
    @showprogress desc = "PFB" dt = 1 @threads for time′_dish_polr in CartesianIndices((ntimes′, ndishes, npolrs))
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

Base.clamp(val::Complex, lo, hi) = Complex(clamp(real(val), lo, hi), clamp(imag(val), lo, hi))
Base.round(::Type{T}, val::Complex) where {T} = Complex{T}(round(T, real(val)), round(T, imag(val)))

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

################################################################################
# F-engine: run

function run(
    noise_amplitude::Float,
    source_channels::Vector{Float},
    source_amplitudes::Vector{Float},
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
    bb_num_beams_P::Int64,
    bb_num_beams_Q::Int64,
    nframes::Int64,
)
    println("Setting up sources...")
    noise = Noise(noise_amplitude)
    sources = Source[
        let
            A = Complex{Float}(amplitude)
            f₀ = channel * adc_frequency / nsamples
            sin_ew = sin(source_position_ew) # east-west
            sin_ns = sin(source_position_ns) # north-south
            Source(f₀, A, sin_ew, sin_ns)
        end for (channel, amplitude) in zip(source_channels, source_amplitudes)
    ]

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
                dish_locations[dish + 1] = (loc_ns, loc_ew)
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

    println("ADC sampling...")
    ntimes_total = nsamples * (nframes * ntimes + ntaps - 1)
    adcframe = adc_sample(Float, noise, sources, dishes, adc, ntimes_total)

    println("PFB FFT...")
    fframe = f_engine(pfb, adcframe)
    #TODO fframe = f_engine_16(pfb, adcframe)

    println("Corner turn...")
    xframe = corner_turn(fframe)

    println("Quantizing...")
    iframe = quantize(Int8, xframe)
    global stored_iframe = iframe

    npolrs = 2
    @assert size(iframe.data) == (ntimes * nframes, nfreqs, ndishes, npolrs)

    println("Done.")
    return nothing
end

function setup(
    noise_amplitude=0.0,
    nsources=1,
    source_channels_ptr=Ptr{Cfloat}(),
    source_amplitudes_ptr=Ptr{Cfloat}(),
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
    nbeams_i=12,
    nbeams_j=8,
    nframes=2,
)
    source_channels_ptr = Ptr{Cfloat}(source_channels_ptr)
    source_amplitudes_ptr = Ptr{Cfloat}(source_amplitudes_ptr)
    dish_indices_ptr = Ptr{Cint}(dish_indices_ptr)
    frequency_channels_ptr = Ptr{Cint}(frequency_channels_ptr)

    println("F-Engine setup:")
    println("    - noise_amplitude:        $noise_amplitude")
    println("    - nsources:               $nsources")
    println("    - source_channels_ptr:    $source_channels_ptr")
    println("    - source_amplitudes_ptr:  $source_amplitudes_ptr")
    println("    - source_position_ew:     $source_position_ew")
    println("    - source_position_ns:     $source_position_ns")
    println("    - num_dish_locations_ew:  $num_dish_locations_ew")
    println("    - num_dish_locations_ns:  $num_dish_locations_ns")
    println("    - dish_indices_ptr:       $dish_indices_ptr")
    println("    - dish_separation_ew:     $dish_separation_ew")
    println("    - dish_separation_ns:     $dish_separation_ns")
    println("    - ndishes:                $ndishes")
    println("    - adc_frequency:          $adc_frequency")
    println("    - ntaps:                  $ntaps")
    println("    - nsamples:               $nsamples")
    println("    - nfreqs:                 $nfreqs")
    println("    - frequency_channels_ptr: $frequency_channels_ptr")
    println("    - ntimes:                 $ntimes")
    println("    - nbeams_i:               $nbeams_i")
    println("    - nbeams_j:               $nbeams_j")
    println("    - nframes:                $nframes")

    source_channels = if source_channels_ptr != C_NULL
        Float[unsafe_load(source_channels_ptr, source) for source in 1:nsources]
    else
        Float[1536 + source for source in 0:(nsources - 1)]
    end
    source_amplitudes = if source_amplitudes_ptr != C_NULL
        Float[unsafe_load(source_amplitudes_ptr, source) for source in 1:nsources]
    else
        Float[1 for source in 1:nsources]
    end
    println("    - source_channels:        $source_channels")
    println("    - source_amplitudes:      $source_amplitudes")

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
    println("    - dish_indices:           $dish_indices")

    frequency_channels = if frequency_channels_ptr != C_NULL
        Int64[unsafe_load(frequency_channels_ptr, n) for n in 1:nfreqs]
    else
        Int64[n for n in 1:nfreqs]
    end
    println("    - frequency_channels:     $frequency_channels")

    return run(
        Float(noise_amplitude),
        source_channels::Vector{Float},
        source_amplitudes::Vector{Float},
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
        Int64(nbeams_i),
        Int64(nbeams_j),
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

function set_E(ptr::Ptr{UInt8}, sz::Int64, ndishes::Int64, npolrs::Int64, nfreqs::Int64, ntimes::Int64, frame_index::Int64)
    iframe = stored_iframe::XFrame
    idata = iframe.data::Array{<:Complex{<:Integer},4}

    @assert ndishes == size(idata, 3)
    @assert npolrs == size(idata, 4)
    @assert nfreqs == size(idata, 2)
    @assert 0 <= (frame_index - 1) * ntimes
    @assert frame_index * ntimes <= size(idata, 1)
    @assert sz == ndishes * npolrs * nfreqs * ntimes

    time0 = (frame_index - 1) * ntimes + 1
    time1 = time0 + ntimes - 1
    I = @view idata[time0:time1, :, :, :]
    E = unsafe_wrap(Array, reinterpret(Ptr{Int4x2}, ptr), (ndishes, npolrs, nfreqs, ntimes))
    @assert sizeof(E) == sz

    @assert size(E) == size(I)[[3, 4, 2, 1]]
    permutedims!(E, mappedarray(c2i4, I), (3, 4, 2, 1))

    return nothing
end

function set_A(ptr::Ptr{UInt8}, sz::Int64, ndishs::Int64, nbbbeams::Int64, npolrs::Int64, nfreqs::Int64, frame_index::Int64)
    for i in 1:sz
        unsafe_store!(ptr, 0, i)
    end
    return nothing
end

function set_J(ptr::Ptr{UInt8}, sz::Int64, ntimes::Int64, npolrs::Int64, nfreqs::Int64, nbbbeams::Int64, frame_index::Int64)
    for i in 1:sz
        unsafe_store!(ptr, 0, i)
    end
    return nothing
end

stored_frbbeamss = nothing
function set_W(ptr::Ptr{UInt8}, sz::Int64, ndishsM::Int64, ndishsN::Int64, npolrs::Int64, nfreqs::Int64, frame_index::Int64)
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
