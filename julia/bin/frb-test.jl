using Base.Threads
using CairoMakie
using FFTW
using ProgressMeter
using SixelTerm

include("../src/FEngine.jl")
using .FEngine

Float = Float32

gauss(x, W) = exp(-(x / W)^2 / 2)
logistic(x) = 1 / (1 + exp(-x))
lopass(x, x₀, Δx) = logistic((x₀ - x) / Δx)
hipass(x, x₀, Δx) = logistic((x - x₀) / Δx)

################################################################################
# Set up one FRB pulse with dispersion

adcfreq::Float = 3200e+6        # 3200 MHz, ADC sampling frequency
Δf::Float = 195.3125e+3         # 195.3125 kHz
Δt::Float = 5.12e-6             # 5.12 us

# We upchannelize and thus need to have a higher frequency resolution
scale = 64
ntimes = 8192 ÷ scale
nfreqs = 8192 * scale + 1

F::Float = adcfreq / 2          # 1500 MHz
T::Float = ntimes * scale * Δt  # 41.94304 ms

phystime(time::Integer) = time * T / ntimes
physfreq(freq::Integer) = freq * F / nfreqs

function time_delay(f::F) where {F<:Real}
    f == 0 && return F(0)

    t₀::F = 0.0e-3              # 0 ms
    t₁::F = 20.0e-3             # 20 ms
    f₀::F = 1500.0e+6           # 1500 MHz
    f₁::F = 300.0e+6            # 300 MHz

    T = (t₀ - t₁) / (1 / f₀^2 - 1 / f₁^2)
    ts = (t₁ / f₀^2 - t₀ / f₁^2) / (1 / f₀^2 - 1 / f₁^2)

    dt = ts + T / f^2
    return dt::F
end

# Time envelope
# Peaked at 5 ms with a width of 1 ms
time_envelope(t::Real) = gauss(t - T / 8, T / 16)

# Frequency envelope
# Cut off below 300 MHz (channel 1536) and above 1500 MHz (channel 7680)
freq_envelope(f::Real) = hipass(f, Float(250e+6), Float(50e+6)) * lopass(f, Float(1550e+6), Float(50e+6))

frb = let
    frb = Array{Complex{Float}}(undef, nfreqs, ntimes)
    @showprogress desc = "FRB" dt = 1 @threads for time in 1:ntimes
        for freq in 1:nfreqs
            t = phystime(time - 1)
            f = physfreq(freq - 1)
            dt = time_delay(f)
            t′ = t - dt
            frb[freq, time] = time_envelope(t′) * freq_envelope(f) * randn(Complex{Float})
        end
    end
    frb
end;

let
    fig = Figure(; size=(640, 480))
    ax = Axis(fig[1, 1]; title="Dispersed FRB pulse", xlabel="time [ms]", ylabel="frequency [MHz]")
    heatmap!(
        1.0e+3 * phystime(1) * (0:1:(ntimes - 1)),
        1.0e-6 * physfreq(1) * (0:1:(nfreqs - 1)),
        permutedims(abs.(@view frb[begin:128:end, :])),
    )
    display(fig)
end;

################################################################################
# Convert into time stream

samples = reshape(irfft(frb, 2 * (nfreqs - 1), 1), :);

let
    fig = Figure(; size=(640, 480))
    ax = Axis(fig[1, 1]; title="Dispersed FRB pulse", xlabel="time [ms]", ylabel="amplitude")
    scatter!(1.0e+3 * range(0, T; length=1024 + 1)[begin:(end - 1)], reshape(maximum(abs, reshape(samples, :, 1024); dims=1), :))
    display(fig)
end;

################################################################################
# PFB

frb2 = FEngine.upchan(samples, 4, 16384);

let
    fig = Figure(; size=(640, 480))
    ax = Axis(fig[1, 1]; title="Dispersed FRB pulse", xlabel="time [ms]", ylabel="frequency [MHz]")
    heatmap!(
        1.0e+3 * phystime(1) * (0:1:(ntimes - 1)),
        1.0e-6 * physfreq(1) * (0:1:(nfreqs - 1)),
        permutedims(abs.(@view frb2[begin:1:end, :])),
    )
    # scatter!(1.0e+3 * (T/8 .+ time_delay.(physfreq(1) * (0:(nfreqs - 1)))),
    #          1.0e-6 * physfreq(1) * (0:(nfreqs - 1)),
    #          )
    # xlims!(ax, 1.0e+3 .* phystime.((0, ntimes-1)))
    display(fig)
end;

################################################################################

nothing
