module FRB

# Implement a 1D first-stage FRB beam forming algorithm via an FFT

using Base.Threads
using CUDA
using CUDA.CUFFT
using FFTW
using LinearAlgebra
using Printf

const Float = Float32

const ndishesi = 256
const ndishesj = 4
const npolarizations = 2

# We assume a sampling rate of 1600 MHz, 4096 ADC samples per time
# sample, upchannelization by a factor of 16, and a desired final time
# period of 1 ms
# Tds ≈ 1000 / (4096 / 1600 * 16)
const Tds = 25

@fastmath @inbounds function fft8!(Y::AbstractArray{Complex{T},3}, X::AbstractArray{Complex{T},3}) where {T<:Real}
    nspectators = size(Y, 3)
    @assert size(Y, 1) == size(X, 1) == 2 * ndishesi
    @assert size(Y, 2) == 2 * size(X, 2) == 2 * ndishesj
    @assert size(Y, 3) == size(X, 3) == nspectators

    r2 = sqrt(T(1) / 2)

    ϕ0 = T(1)
    ϕ1 = r2 + r2 * im
    ϕ2 = T(1) * im
    ϕ3 = -r2 + r2 * im
    ϕ4 = -T(1)
    ϕ5 = -r2 - r2 * im
    ϕ6 = -T(1) * im
    ϕ7 = r2 - r2 * im

    for spectator in 1:nspectators
        for dishi in 1:(2 * ndishesi)
            x0 = X[dishi, 1, spectator]
            x1 = X[dishi, 2, spectator]
            x2 = X[dishi, 3, spectator]
            x3 = X[dishi, 4, spectator]
            x4 = T(0)
            x5 = T(0)
            x6 = T(0)
            x7 = T(0)

            # yi = exp -ikx xi

            y0 = ((ϕ0 * x0 + ϕ0 * x1) + (ϕ0 * x2 + ϕ0 * x3)) + ((ϕ0 * x4 + ϕ0 * x5) + (ϕ0 * x6 + ϕ0 * x7))
            y1 = ((ϕ0 * x0 + ϕ1 * x1) + (ϕ2 * x2 + ϕ3 * x3)) + ((ϕ4 * x4 + ϕ5 * x5) + (ϕ6 * x6 + ϕ7 * x7))
            y2 = ((ϕ0 * x0 + ϕ2 * x1) + (ϕ4 * x2 + ϕ6 * x3)) + ((ϕ0 * x4 + ϕ2 * x5) + (ϕ4 * x6 + ϕ6 * x7))
            y3 = ((ϕ0 * x0 + ϕ3 * x1) + (ϕ6 * x2 + ϕ1 * x3)) + ((ϕ4 * x4 + ϕ7 * x5) + (ϕ2 * x6 + ϕ5 * x7))
            y4 = ((ϕ0 * x0 + ϕ4 * x1) + (ϕ0 * x2 + ϕ4 * x3)) + ((ϕ0 * x4 + ϕ4 * x5) + (ϕ0 * x6 + ϕ4 * x7))
            y5 = ((ϕ0 * x0 + ϕ5 * x1) + (ϕ2 * x2 + ϕ7 * x3)) + ((ϕ4 * x4 + ϕ1 * x5) + (ϕ6 * x6 + ϕ3 * x7))
            y6 = ((ϕ0 * x0 + ϕ6 * x1) + (ϕ4 * x2 + ϕ2 * x3)) + ((ϕ0 * x4 + ϕ6 * x5) + (ϕ4 * x6 + ϕ2 * x7))
            y7 = ((ϕ0 * x0 + ϕ7 * x1) + (ϕ6 * x2 + ϕ5 * x3)) + ((ϕ4 * x4 + ϕ3 * x5) + (ϕ2 * x6 + ϕ1 * x7))

            Y[dishi, 1, spectator] = y0
            Y[dishi, 2, spectator] = y1
            Y[dishi, 3, spectator] = y2
            Y[dishi, 4, spectator] = y3
            Y[dishi, 5, spectator] = y4
            Y[dishi, 6, spectator] = y5
            Y[dishi, 7, spectator] = y6
            Y[dishi, 8, spectator] = y7
        end
    end

    return Y
end

mutable struct State{T}
    inputi::Array{Complex{T},4}
    FFTi
    outputi::Array{Complex{T},4}
    outputj::Array{Complex{T},4}
    function State{T}() where {T}
        return new(
            Array{Complex{T},4}(undef, (0, 0, 0, 0)),
            nothing,
            Array{Complex{T},4}(undef, (0, 0, 0, 0)),
            Array{Complex{T},4}(undef, (0, 0, 0, 0)),
        )
    end
end

function frb1!(state::State{T}, I::AbstractArray{T,4}, E::AbstractArray{Complex{T},5}) where {T<:Real}
    ndishesi1, ndishesj1, npolarizations1, ntimes, nfrequencies = size(E)
    # checking assumptions
    @assert ndishesi1 == ndishesi
    @assert ndishesj1 == ndishesj
    @assert npolarizations1 == npolarizations
    # for convenience
    @assert ntimes % Tds == 0

    ntimesbar = ntimes ÷ Tds

    @assert size(I) == (2 * ndishesi, 2 * ndishesj, ntimesbar, nfrequencies)

    # Zero input because we only fill the lower half
    if size(state.inputi) != (2 * ndishesi, ndishesj, npolarizations, Tds)
        state.inputi = zeros(T, (2 * ndishesi, ndishesj, npolarizations, Tds))
    end
    inputi = state.inputi
    if state.FFTi === nothing
        state.FFTi = plan_fft(inputi, 1)
    end
    FFTi = state.FFTi
    if size(state.outputi) != size(inputi)
        state.outputi = similar(inputi)
    end
    outputi = state.outputi

    if size(state.outputj) != (2 * ndishesi, 2 * ndishesj, npolarizations, Tds)
        state.outputj = similar(outputi, (2 * ndishesi, 2 * ndishesj, npolarizations, Tds))
    end
    outputj = state.outputj

    for frequency in 1:nfrequencies, timebar in 1:ntimesbar
        timelo = (timebar - 1) * Tds + 1
        timehi = timelo + Tds - 1

        inputi[1:ndishesi, :, :, :] .= @view E[:, :, :, timelo:timehi, frequency]
        mul!(outputi, FFTi, inputi)

        # inputj[:, 1:ndishesj, :, :] .= outputi
        # mul!(outputj, FFTj, inputj)
        fft8!(reshape(outputj, 2 * ndishesi, 2 * ndishesj, :), reshape(outputi, 2 * ndishesi, ndishesj, :))

        # Calculate intensity and sum over polarizations and times
        I[:, :, timebar, frequency] .= sum(abs2, outputj; dims=(3, 4))
    end

    return nothing
end

################################################################################

@fastmath @inbounds function fft8_cuda!(Y::CuDeviceArray{T,3}, X::CuDeviceArray{Complex{T},4}) where {T<:Real}
    r2 = sqrt(T(1) / 2)

    ϕ0 = T(1)
    ϕ1 = r2 + r2 * im
    ϕ2 = T(1) * im
    ϕ3 = -r2 + r2 * im
    ϕ4 = -T(1)
    ϕ5 = -r2 - r2 * im
    ϕ6 = -T(1) * im
    ϕ7 = r2 - r2 * im

    dishi = threadIdx().x
    ntimes = npolarizations * Tds
    spectator = blockIdx().x

    y0::T = 0
    y1::T = 0
    y2::T = 0
    y3::T = 0
    y4::T = 0
    y5::T = 0
    y6::T = 0
    y7::T = 0

    for time in 1:ntimes
        x0 = X[dishi, 1, time, spectator]
        x1 = X[dishi, 2, time, spectator]
        x2 = X[dishi, 3, time, spectator]
        x3 = X[dishi, 4, time, spectator]
        x4 = T(0)
        x5 = T(0)
        x6 = T(0)
        x7 = T(0)

        # yi = exp -ikx xi

        z0 = ((ϕ0 * x0 + ϕ0 * x1) + (ϕ0 * x2 + ϕ0 * x3)) + ((ϕ0 * x4 + ϕ0 * x5) + (ϕ0 * x6 + ϕ0 * x7))
        z1 = ((ϕ0 * x0 + ϕ1 * x1) + (ϕ2 * x2 + ϕ3 * x3)) + ((ϕ4 * x4 + ϕ5 * x5) + (ϕ6 * x6 + ϕ7 * x7))
        z2 = ((ϕ0 * x0 + ϕ2 * x1) + (ϕ4 * x2 + ϕ6 * x3)) + ((ϕ0 * x4 + ϕ2 * x5) + (ϕ4 * x6 + ϕ6 * x7))
        z3 = ((ϕ0 * x0 + ϕ3 * x1) + (ϕ6 * x2 + ϕ1 * x3)) + ((ϕ4 * x4 + ϕ7 * x5) + (ϕ2 * x6 + ϕ5 * x7))
        z4 = ((ϕ0 * x0 + ϕ4 * x1) + (ϕ0 * x2 + ϕ4 * x3)) + ((ϕ0 * x4 + ϕ4 * x5) + (ϕ0 * x6 + ϕ4 * x7))
        z5 = ((ϕ0 * x0 + ϕ5 * x1) + (ϕ2 * x2 + ϕ7 * x3)) + ((ϕ4 * x4 + ϕ1 * x5) + (ϕ6 * x6 + ϕ3 * x7))
        z6 = ((ϕ0 * x0 + ϕ6 * x1) + (ϕ4 * x2 + ϕ2 * x3)) + ((ϕ0 * x4 + ϕ6 * x5) + (ϕ4 * x6 + ϕ2 * x7))
        z7 = ((ϕ0 * x0 + ϕ7 * x1) + (ϕ6 * x2 + ϕ5 * x3)) + ((ϕ4 * x4 + ϕ3 * x5) + (ϕ2 * x6 + ϕ1 * x7))

        y0 += abs2(z0)
        y1 += abs2(z1)
        y2 += abs2(z2)
        y3 += abs2(z3)
        y4 += abs2(z4)
        y5 += abs2(z5)
        y6 += abs2(z6)
        y7 += abs2(z7)
    end

    Y[dishi, 1, spectator] = y0
    Y[dishi, 2, spectator] = y1
    Y[dishi, 3, spectator] = y2
    Y[dishi, 4, spectator] = y3
    Y[dishi, 5, spectator] = y4
    Y[dishi, 6, spectator] = y5
    Y[dishi, 7, spectator] = y6
    Y[dishi, 8, spectator] = y7

    return nothing
end

function fft8_cuda!(Y::CuArray{T,3}, X::CuArray{Complex{T},4}) where {T<:Real}
    ntimes = size(Y, 3)
    nspectators = size(Y, 3)
    @assert size(Y, 1) == size(X, 1) == 2 * ndishesi
    @assert size(Y, 2) == 2 * size(X, 2) == 2 * ndishesj
    @assert size(X, 3) == npolarizations * Tds
    @assert size(Y, 3) == size(X, 4) == nspectators

    @cuda threads = 2 * ndishesi blocks = nspectators always_inline = true fastmath = true fft8_cuda!(Y, X)

    return Y
end

mutable struct CudaState{T}
    F::CuArray{Complex{T},2}
    FFTi
    G::CuArray{Complex{T},2}
    CudaState{T}() where {T} = new(CuArray{Complex{T},2}(undef, (0, 0)), nothing, CuArray{Complex{T},2}(undef, (0, 0)))
end

function frb1_cuda!(state::CudaState{T}, I::CuArray{T,4}, E::CuArray{Complex{T},5}) where {T<:Real}
    ndishesi1, ndishesj1, npolarizations1, ntimes, nfrequencies = size(E)
    # checking assumptions
    @assert ndishesi1 == ndishesi
    @assert ndishesj1 == ndishesj
    @assert npolarizations1 == npolarizations
    # for convenience
    @assert ntimes % Tds == 0

    ntimesbar = ntimes ÷ Tds

    @assert size(I) == (2 * ndishesi, 2 * ndishesj, ntimesbar, nfrequencies)

    E1 = reshape(E, (ndishesi, :))

    if size(state.F) != (2 * ndishesi, size(E1, 2))
        state.F = zeros(Complex{T}, (2 * ndishesi, size(E1, 2)))
    end
    F = state.F
    F[1:ndishesi, :] .= E1

    if state.FFTi === nothing
        state.FFTi = plan_fft(F, 1)
    end
    FFTi = state.FFTi
    if size(state.G) != (2 * ndishesi, size(E1, 2))
        state.G = similar(F, (2 * ndishesi, size(E1, 2)))
    end
    G = state.G
    mul!(G, FFTi, F)

    G1 = reshape(G, (2 * ndishesi, ndishesj, npolarizations * Tds, :))
    I1 = reshape(I, (2 * ndishesi, 2 * ndishesj, :))
    fft8_cuda!(I1, G1)

    return nothing
end

################################################################################

function test_frb1()
    println("FRB stage 1 on CPU, single-threaded")

    # 8192 time samples, upchannelized by 16
    ntimes = fld(8192 ÷ 16, Tds) * Tds
    # 16 frequencies per GPU, upchannelized by 16
    nfrequencies = 16 * 16

    println("Initializing...")
    E = 7 * randn(Complex{Float}, (ndishesi, ndishesj, npolarizations, ntimes, nfrequencies))
    I = similar(E, Float, (2 * ndishesi, 2 * ndishesj, ntimes ÷ Tds, nfrequencies))

    println("Preheating...")
    state = State{Float}()
    frb1!(state, I, E)

    println("Running...")
    t0 = time()
    frb1!(state, I, E)
    t1 = time()
    t = t1 - t0
    t_I = t / length(I)
    @printf "Run time: %.3f seconds, %.3f nanoseconds per I element\n" t 1.0e+9 * t_I

    println("Done.")
    return nothing
end

function test_frb1_cuda()
    println("FRB stage 1 with CUDA")

    # 8192 time samples, upchannelized by 16
    ntimes = fld(8192 ÷ 16, Tds) * Tds
    # 16 frequencies per GPU, upchannelized by 16
    nfrequencies = 16 * 16

    println("Initializing...")
    E = 7 * CUDA.randn(Complex{Float}, (ndishesi, ndishesj, npolarizations, ntimes, nfrequencies))
    I = similar(E, Float, (2 * ndishesi, 2 * ndishesj, ntimes ÷ Tds, nfrequencies))
    CUDA.synchronize()

    println("Preheating...")
    state = CudaState{Float}()
    frb1_cuda!(state, I, E)
    CUDA.synchronize()

    println("Running...")
    t0 = time()
    frb1_cuda!(state, I, E)
    CUDA.synchronize()
    t1 = time()
    t = t1 - t0
    t_I = t / length(I)
    @printf "Run time: %.3f seconds, %.3f nanoseconds per I element\n" t 1.0e+9 * t_I

    println("Done.")
    return nothing
end

end

FRB.test_frb1()
FRB.test_frb1_cuda()
