using PrettyTables
using TypedTables

# Load the code
include("../../lib/stages/FEngine.jl")
using .FEngine

function calibrate_upchan1(M, U, delta)
    adcfreq = 3.2e9
    freq = 0.3e9
    channel = 1536

    ntaps = 4
    nsamples = 16384
    ntimes = 16 * U

    # Set up some data, a monochromatic source in the middle between two channels
    x = sinpi.(2 * (freq / adcfreq + delta / nsamples) * (0:(nsamples * (ntimes + ntaps - 1) - 1)))
    # First PFB
    y = FEngine.upchan(x, ntaps, nsamples)
    y /= maximum(abs, y)

    # println("Coarse amplitude: ", maximum(abs, y[channel+1, :]))

    # Second PFB
    z = FEngine.upchan(y[channel + 1, :], M, U)

    channels = 1:U
    amplitudes = [maximum(abs, z[u, :]) for u in 1:U]
    stats = Table(; channels=channels, amplitudes=amplitudes)

    # println("Fine amplitudes M=$M U=$U δ=$delta:")
    # pretty_table(stats; header=["channel", "maxabs"], tf=tf_borderless)

    return amplitudes
end

function calibrate_upchan(M, Ulogmax)
    for Ulog in 0:Ulogmax
        U = 2^Ulog
        for u in 1:U
            delta = (u - (U + 1) / 2) / U
            amplitudes = calibrate_upchan1(4, U, delta)
            amplitude = amplitudes[u]
            println("U: $U   u: $u   A: $amplitude")
        end
    end
    return nothign
end

# calibrate_upchan(4, 7)
