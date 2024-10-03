# CHORD 8-bit baseband beamformer
# <https://www.overleaf.com/project/6228adae742a3a2da1afe437l>

using CUDA
using CUDASIMDTypes
using IndexSpaces
using Mustache
using Random

const Memory = IndexSpaces.Memory

# const card = "A30"
const card = "A40"
# const card = "GeForce_RTX_4090"
# const card = "L40S"

if CUDA.functional()
    println("[Choosing CUDA device...]")
    CUDA.device!(0)
    println(name(device()))
    @assert replace(name(device()), ' ' => '_') == "NVIDIA_$card"
end

idiv(i::Integer, j::Integer) = (@assert iszero(i % j); i ÷ j)
# shift(x::Number, s) = (@assert s ≥ 0; s == 0 ? x : (x + (1 << (s - 1))) >> s)
shift(x::Number, s) = (@assert s ≥ 1; (x + (1 << (s - 1))) >> s)
shift(x::Complex, s) = Complex(shift(x.re, s), shift(x.im, s))
Base.clamp(x::Complex, a, b) = Complex(clamp(x.re, a, b), clamp(x.im, a, b))
Base.clamp(x::Complex, ab::UnitRange) = clamp(x, ab.start, ab.stop)

function shrink(value::Integer)
    typemin(Int32) <= value <= typemax(Int32) && return Int32(value)
    typemin(Int64) <= value <= typemax(Int64) && return Int64(value)
    return value
end
function shrinkmul(x::Integer, y::Symbol, ymax::Integer)
    @assert x >= 0 && ymax >= 0
    # We assume 0 <= y < ymax
    if x * (ymax - 1) <= typemax(Int32)
        # We can use 32-bit arithmetic        
        return :($(Int32(x)) * $y)
    elseif x * (ymax - 1) <= typemax(Int64)
        # We need to use 64-bit arithmetic        
        return :($(Int64(x)) * $y)
    else
        # Something is wrong
        @assert false
    end
end

@enum CHORDTag CplxTag BeamTag DishTag FreqTag PolrTag TimeTag ThreadTag WarpTag BlockTag

const Cplx = Index{Physics,CplxTag}
const Beam = Index{Physics,BeamTag}
const Dish = Index{Physics,DishTag}
const Freq = Index{Physics,FreqTag}
const Polr = Index{Physics,PolrTag}
const Time = Index{Physics,TimeTag}

# Setup

setup::Symbol
F::Integer
T::Integer

@static if setup ≡ :chord

    # Full CHORD
    const B = 96

    const T1_stride = 128
    const T2_stride = 32

    const Wb = idiv(B, 16)
    const Wd = 4
    const Wp = 1

elseif setup ≡ :hirax

    # HIRAX
    const B = 16                # 8...32

    const T1_stride = 128
    const T2_stride = 32

    const Wb = idiv(B, 8)
    const Wd = 2
    const Wp = 1

elseif setup ≡ :pathfinder

    # CHORD pathfinder
    const B = 16

    const T1_stride = 128
    const T2_stride = 32

    const Wb = idiv(B, 8)
    const Wd = 1
    const Wp = 2

elseif setup ≡ :chime

    # CHIME
    const B = 16

    const T1_stride = 128
    const T2_stride = 32

    const Wb = idiv(B, 16)
    const Wd = 4
    const Wp = 1

else
    @assert false
end

const Tout = idiv(T, 4)         # always process 1/4 of the ringbuffer at a time

# Since we introduced Tmin and Tmax, we don't support Bt != 1 any more
# const Bt = 16                   # distribute time samples over that many blocks
const Bt = 1

@assert T % Bt == 0
@assert T % Bt % T1_stride == 0

const Bp = idiv(P, Wp)

@assert T % T1_stride == 0
@assert T1_stride % T2_stride == 0

@assert Wp * Bp == P

@assert B % Wb == 0
@assert D % Wd == 0
@assert P % Wp == 0

@assert P % Bp == 0

const num_simd_bits = 32
const num_threads = 32
const num_warps = Wb * Wd * Wp
const num_blocks = Bt * Bp * F
const num_blocks_per_sm = 32 ÷ num_warps

const cacheline_length = 32     # counting in UInt32

const E_shared_type = Int4x8    # TODO: determine automatically
const E_shared_offset = 0
const E_shared_length = (idiv(D, 4) + 1) * 32 * Wp # TODO: calculate automatically

const Ju_shared_type = Int16x2  # TODO: determine automatically
const Ju_shared_offset = (E_shared_offset + E_shared_length + cacheline_length - 1) & -cacheline_length
const Ju_shared_length = (B + 4) * 32 * Wd * Wp # TODO: calculate automatically

const total_shared_length = (Ju_shared_offset + Ju_shared_length + cacheline_length - 1) & -cacheline_length
const shmem_bytes = sizeof(UInt32) * total_shared_length

const kernel_setup = KernelSetup(num_threads, num_warps, num_blocks, num_blocks_per_sm, shmem_bytes)

# Benchmark results:

# Setup for full CHORD on A40:
#
# benchmark-result:
#   kernel: "bb"
#   description: "baseband beamformer"
#   design-parameters:
#     number-of-beams: 96
#     number-of-complex-components: 2
#     number-of-dishes: 512
#     number-of-frequencies: 42
#     number-of-polarizations: 2
#     number-of-timesamples: 16384
#     sampling-time-μsec: 1.7066666666666668
#     shift-parameter-σ: 3
#   compile-parameters:
#     minthreads: 768
#     blocks_per_sm: 1
#   call-parameters:
#     threads: [32, 24]
#     blocks: [84]
#     shmem_bytes: 67712
#   result-μsec:
#     runtime: 3653.0
#     scaled-runtime: 1391.6
#     scaled-number-of-frequencies: 16
#     dataframe-length: 27962.0
#     dataframe-percent: 5.0

# Setup for full CHORD on A30:
#
# benchmark-result:
#   kernel: "bb"
#   description: "baseband beamformer"
#   design-parameters:
#     number-of-beams: 96
#     number-of-complex-components: 2
#     number-of-dishes: 512
#     number-of-frequencies: 28
#     number-of-polarizations: 2
#     number-of-timesamples: 32768
#     sampling-time-μsec: 1.7066666666666668
#     shift-parameter-σ: 3
#   compile-parameters:
#     minthreads: 768
#     blocks_per_sm: 1
#   call-parameters:
#     threads: [32, 24]
#     blocks: [56]
#     shmem_bytes: 67712
#   result-μsec:
#     runtime: 8804.8
#     scaled-runtime: 5031.3
#     scaled-number-of-frequencies: 16
#     dataframe-length: 55924.1
#     dataframe-percent: 9.0

# Setup for CHORD pathfinder on A40:
#
# benchmark-result:
#   kernel: "bb"
#   description: "baseband beamformer"
#   design-parameters:
#     number-of-beams: 16
#     number-of-complex-components: 2
#     number-of-dishes: 64
#     number-of-frequencies: 672
#     number-of-polarizations: 2
#     number-of-timesamples: 32768
#     sampling-time: 1.7
#     shift-parameter-σ: 2
#   compile-parameters:
#     minthreads: 128
#     blocks_per_sm: 8
#   call-parameters:
#     threads: [32, 4]
#     blocks: [672]
#     shmem_bytes: 9472
#   result-μsec:
#     runtime: 8524.7
#     scaled-runtime: 1623.8
#     scaled-number-of-frequencies: 128
#     dataframe-length: 55705.6
#     dataframe-percent: 2.9

# We need to shift the intermediate `Ju` results by `σ` bits to ensure
# they fit into 16 bits
const σ = trailing_zeros(idiv(D, Wd)) - 4
@assert σ ≥ 0

function make_bb_kernel()
    # Machine indices
    simd = SIMD(:simd, 1, num_simd_bits)
    thread = Thread(:thread, 1, num_threads)
    warp = Warp(:warp, 1, num_warps)
    block = Block(:block, 1, num_blocks)
    shared = Shared(:shared, 1, 131072)
    memory = Memory(:memory, 1, 2^32)

    loopT3 = Loop(:T3, 8, idiv(T2_stride, 8)) # 8 time samples enter the tensor core mma
    loopT2 = Loop(:T2, T2_stride, idiv(T1_stride, T2_stride))
    loopT1 = Loop(:T1, T1_stride, idiv(idiv(T, Bt), T1_stride))

    loopD = UnrolledLoop(:D, 4, idiv(D, 16 * Wd)) # 16 dishes enter the tensor core mma

    num_A_beam_registers = idiv(B, 8 * Wb)
    loopB = UnrolledLoop(:B, 8, num_A_beam_registers) # 8 beams enter the tensor core mma

    # Physics indices
    int4value = IntValue(:intvalue, 1, 4)
    int8value = IntValue(:intvalue, 1, 8)
    int16value = IntValue(:intvalue, 1, 16)
    int32value = IntValue(:intvalue, 1, 32)
    cplx = Cplx(:cplx, 1, C)
    beam = Beam(:beam, 1, B)
    dish = Dish(:dish, 1, D)
    freq = Freq(:freq, 1, F)
    polr = Polr(:polr, 1, P)
    time = Time(:time, 1, T)

    dish0 = Dish(:dish, 1, 2)
    dish01 = Dish(:dish, 1, 4)
    dish1 = Dish(:dish, 2, 2)
    dish1etc = Dish(:dish, 2, idiv(D, 2))
    dish2 = Dish(:dish, 4, 2)
    dish23 = Dish(:dish, 4, 4)
    dish234 = Dish(:dish, 4, 8)
    dish2345 = Dish(:dish, 4, 16)
    dish2etc = Dish(:dish, 4, idiv(D, 4))
    dish3 = Dish(:dish, 8, 2)
    dish34 = Dish(:dish, 8, 4)
    dish45 = Dish(:dish, 16, 4)
    dish456 = Dish(:dish, 16, 8)
    dish5 = Dish(:dish, 32, 2)
    dish56 = Dish(:dish, 32, 4)
    dish6 = Dish(:dish, 64, 2)
    dish67 = Dish(:dish, 64, 4)
    dish7 = Dish(:dish, 128, 2)
    dish78 = Dish(:dish, 128, 4)
    dish789 = Dish(:dish, 128, 8)
    dish89 = Dish(:dish, 256, 4)
    dish9 = Dish(:dish, 512, 2)

    beam0 = Beam(:beam, 1, 2)
    beam01 = Beam(:beam, 1, 4)
    beam012 = Beam(:beam, 1, 8)
    beam1 = Beam(:beam, 2, 2)
    beam12 = Beam(:beam, 2, 4)
    beam2 = Beam(:beam, 4, 2)
    beam2etc = Beam(:beam, 4, idiv(B, 4))
    beam3 = Beam(:beam, 8, 2)
    beam3etc = Beam(:beam, 8, idiv(B, 8))
    beam4etc = Beam(:beam, 16, idiv(B, 16))

    time0 = Time(:time, 1, 2)
    time01 = Time(:time, 1, 4)
    time012 = Time(:time, 1, 8)
    time01234 = Time(:time, 1, 32)
    time1 = Time(:time, 2, 2)
    time12 = Time(:time, 2, 4)
    time2 = Time(:time, 4, 2)
    time234 = Time(:time, 4, 8)
    time2etc = Time(:time, 4, idiv(T, 4))
    time3 = Time(:time, 8, 2)
    time34 = Time(:time, 8, 4)
    time4 = Time(:time, 16, 2)
    time56 = Time(:time, 32, 4)
    time7etc = Time(:time, 128, idiv(idiv(T, Bt), 128))

    # # Physics quantities
    # E = Quantity(:E, [cplx, dish, freq, polr, time, int4value])
    # A = Quantity(:A, [cplx, beam, dish, freq, polr, int8value])
    # J = Quantity(:J, [cplx, beam, freq, polr, time, int4value])

    # Memory layouts

    # E-matrix layout

    layout_E_memory = Layout([
        int4value => SIMD(:simd, 1, 4),
        cplx => SIMD(:simd, 4, 2),
        dish01 => SIMD(:simd, 4 * 2, 4),
        dish2etc => Memory(:memory, 1, idiv(D, 4)),
        polr => Memory(:memory, idiv(D, 4), P),
        freq => Memory(:memory, idiv(D, 4) * P, F),
        time => Memory(:memory, idiv(D, 4) * P * F, T),
    ])

    # A-matrix layout

    layout_A_memory = Layout([
        int8value => SIMD(:simd, 1, 8),
        cplx => SIMD(:simd, 8, 2),
        dish0 => SIMD(:simd, 8 * 2, 2),
        dish1etc => Memory(:memory, 1, idiv(D, 2)),
        beam => Memory(:memory, idiv(D, 2), B),
        polr => Memory(:memory, idiv(D, 2) * B, P),
        freq => Memory(:memory, idiv(D, 2) * B * P, F),
    ])

    # s layout

    layout_s_global = Layout([
        int32value => SIMD(:simd, 1, 32),
        beam => Memory(:memory, 1, B),
        polr => Memory(:memory, B, P),
        freq => Memory(:memory, B * P, F),
    ])

    # J-matrix layout

    layout_J_memory = Layout([
        int4value => SIMD(:simd, 1, 4),
        cplx => SIMD(:simd, 4, 2),
        time01 => SIMD(:simd, 8, 4),

        # time2etc => Memory(:memory, 1, idiv(T, 4)),
        # polr => Memory(:memory, idiv(T, 4), P),
        # freq => Memory(:memory, idiv(T, 4) * P, F),
        # beam => Memory(:memory, idiv(T, 4) * P * F, B),

        Time(:time, 4, idiv(Tout, 4)) => Memory(:memory, 1, idiv(Tout, 4)),
        polr => Memory(:memory, idiv(Tout, 4), P),
        freq => Memory(:memory, idiv(Tout, 4) * P, F),
        beam => Memory(:memory, idiv(Tout, 4) * P * F, B),

        # # We could use a larger chunk size than `T1_stride`
        # Time(:time, 4, idiv(T1_stride, 4)) => Memory(:memory, 1, idiv(T1_stride, 4)),
        # polr => Memory(:memory, idiv(T1_stride, 4), P),
        # freq => Memory(:memory, idiv(T1_stride, 4) * P, F),
        # beam => Memory(:memory, idiv(T1_stride, 4) * P * F, B),
        # Time(:time, T1_stride, idiv(T, T1_stride)) => Memory(:memory, idiv(T1_stride, 4) * P * F * B, idiv(T, T1_stride)),
    ])

    # info layout

    layout_info_memory = Layout([
        int32value => SIMD(:simd, 1, 32),
        Index{Physics,ThreadTag}(:thread, 1, num_threads) => Memory(:memory, 1, num_threads),
        Index{Physics,WarpTag}(:warp, 1, num_warps) => Memory(:memory, num_threads, num_warps),
        Index{Physics,BlockTag}(:block, 1, num_blocks) => Memory(:memory, num_threads * num_warps, num_blocks),
    ])

    # log layout

    layout_log_memory = Layout([
        int32value => SIMD(:simd, 1, 32), Index{Physics,BlockTag}(:block, 1, num_blocks) => Memory(:memory, 1, num_blocks)
    ])

    # Shared memory layouts

    # E-matrix layout

    # Section 3, eqn. (13)
    layout_E_shared = Layout([
        int4value => SIMD(:simd, 1, 4),
        cplx => SIMD(:simd, 4, 2),
        dish01 => SIMD(:simd, 4 * 2, 4),
        dish2etc => Shared(:shared, 1, idiv(D, 4)),
        time01234 => Shared(:shared, idiv(D, 4) + 1, 32),
        time56 => loopT2,
        time7etc => loopT1,
        Time(:time, idiv(T, Bt), Bt) => Block(:block, 1, Bt),
        Polr(:polr, 1, Bp) => Block(:block, Bt, Bp),
        Polr(:polr, Bp, Wp) => Shared(:shared, (idiv(D, 4) + 1) * 32, Wp),
        freq => Block(:block, Bt * Bp, F),
    ])

    # Ju layout

    layout_Ju_shared = Layout([
        int16value => SIMD(:simd, 1, 16),
        cplx => SIMD(:simd, 16, 2),
        beam => Shared(:shared, 1, B),
        time01234 => Shared(:shared, B + 4, 32),
        time56 => loopT2,
        time7etc => loopT1,
        Time(:time, idiv(T, Bt), Bt) => Block(:block, 1, Bt),
        Dish(:dish, idiv(D, Wd), Wd) => Shared(:shared, (B + 4) * 32, Wd),
        Polr(:polr, 1, Bp) => Block(:block, Bt, Bp),
        Polr(:polr, Bp, Wp) => Shared(:shared, (B + 4) * 32 * Wd, Wp),
        freq => Block(:block, Bt * Bp, F),
    ])

    # Register layouts

    # E-matrix layout

    # for copying from global to shared memory
    if D == 1024
        num_time_warps_for_Ecopy = prevpow(2, Wb)
        num_time_registers_for_Ecopy = idiv(8, num_time_warps_for_Ecopy)
        num_warps_for_Ecopy = Wd * num_time_warps_for_Ecopy
        @assert 4 * num_time_warps_for_Ecopy * num_time_registers_for_Ecopy == 32
        @assert 1 ≤ num_warps_for_Ecopy ≤ 32
        layout_E_registers = Layout([
            int4value => SIMD(:simd, 1, 4),
            cplx => SIMD(:simd, 4, C),
            dish01 => SIMD(:simd, 4 * C, 4),
            dish23 => Register(:dish, 4, 4),
            dish456 => Thread(:thread, 1, 8),
            dish78 => Warp(:warp, 1, Wd),
            dish9 => Register(:dish, 512, 2),
            time01 => Thread(:thread, 8, 4),
            Time(:time, 4, num_time_warps_for_Ecopy) => Warp(:warp, Wd, num_time_warps_for_Ecopy),
            Time(:time, 4 * num_time_warps_for_Ecopy, num_time_registers_for_Ecopy) =>
                Register(:time, 4 * num_time_warps_for_Ecopy, num_time_registers_for_Ecopy),
            time56 => loopT2,
            time7etc => loopT1,
            Time(:time, idiv(T, Bt), Bt) => Block(:block, 1, Bt),
            Polr(:polr, 1, P) => Block(:block, Bt, Bp),
            freq => Block(:block, Bt * Bp, F),
        ])
    elseif D == 512
        num_time_warps_for_Ecopy = prevpow(2, Wb)
        num_time_registers_for_Ecopy = idiv(8, num_time_warps_for_Ecopy)
        num_warps_for_Ecopy = Wd * num_time_warps_for_Ecopy
        @assert 4 * num_time_warps_for_Ecopy * num_time_registers_for_Ecopy == 32
        @assert 1 ≤ num_warps_for_Ecopy ≤ 32
        layout_E_registers = Layout([
            int4value => SIMD(:simd, 1, 4),
            cplx => SIMD(:simd, 4, C),
            dish01 => SIMD(:simd, 4 * C, 4),
            dish23 => Register(:dish, 4, 4),
            dish456 => Thread(:thread, 1, 8),
            dish78 => Warp(:warp, 1, Wd),
            time01 => Thread(:thread, 8, 4),
            Time(:time, 4, num_time_warps_for_Ecopy) => Warp(:warp, Wd, num_time_warps_for_Ecopy),
            Time(:time, 4 * num_time_warps_for_Ecopy, num_time_registers_for_Ecopy) =>
                Register(:time, 4 * num_time_warps_for_Ecopy, num_time_registers_for_Ecopy),
            time56 => loopT2,
            time7etc => loopT1,
            Time(:time, idiv(T, Bt), Bt) => Block(:block, 1, Bt),
            Polr(:polr, 1, P) => Block(:block, Bt, Bp),
            freq => Block(:block, Bt * Bp, F),
        ])
    elseif D == 256
        num_time_warps_for_Ecopy = prevpow(2, Wb)
        num_time_registers_for_Ecopy = idiv(8, num_time_warps_for_Ecopy)
        num_warps_for_Ecopy = Wd * num_time_warps_for_Ecopy
        @assert 4 * num_time_warps_for_Ecopy * num_time_registers_for_Ecopy == 32
        @assert 1 ≤ num_warps_for_Ecopy ≤ 32
        layout_E_registers = Layout([
            int4value => SIMD(:simd, 1, 4),
            cplx => SIMD(:simd, 4, C),
            dish01 => SIMD(:simd, 4 * C, 4),
            dish23 => Register(:dish, 4, 4),
            dish456 => Thread(:thread, 1, 8),
            dish7 => Warp(:warp, 1, Wd),
            time01 => Thread(:thread, 8, 4),
            Time(:time, 4, num_time_warps_for_Ecopy) => Warp(:warp, Wd, num_time_warps_for_Ecopy),
            Time(:time, 4 * num_time_warps_for_Ecopy, num_time_registers_for_Ecopy) =>
                Register(:time, 4 * num_time_warps_for_Ecopy, num_time_registers_for_Ecopy),
            time56 => loopT2,
            time7etc => loopT1,
            Time(:time, idiv(T, Bt), Bt) => Block(:block, 1, Bt),
            Polr(:polr, 1, P) => Block(:block, Bt, Bp),
            freq => Block(:block, Bt * Bp, F),
        ])
    elseif D == 64
        @assert Wd == 1
        num_time_warps_for_Ecopy = Wb
        num_time_iters_for_Ecopy = 1
        num_warps_for_Ecopy = Wd * num_time_warps_for_Ecopy * Wp
        @assert 1 ≤ num_warps_for_Ecopy ≤ 32
        layout_E_registers = Layout([
            int4value => SIMD(:simd, 1, 4),
            cplx => SIMD(:simd, 4, C),
            dish01 => SIMD(:simd, 4 * C, 4),
            dish23 => Register(:dish, 4, 4),
            dish45 => Thread(:thread, 1, 4),
            time012 => Thread(:thread, 4, 8),
            time3 => Register(:time, 8, 2),
            time4 => Warp(:warp, 1, Wb),
            time56 => loopT2,
            time7etc => loopT1,
            Time(:time, idiv(T, Bt), Bt) => Block(:block, 1, Bt),
            Polr(:polr, 1, P) => Warp(:warp, Wb, Wp),
            freq => Block(:block, Bt * Bp, F),
        ])
    else
        @assert false
    end

    # for multiplication with A
    @assert 4 * loopD.length * 4 * Wd == D
    layout_E0_registers = Layout([
        int4value => SIMD(:simd, 1, 4),
        cplx => SIMD(:simd, 4, C),
        dish01 => SIMD(:simd, 4 * C, 4), # mma input k, bits 01
        Dish(:dish, 4, loopD.length) => loopD,
        Dish(:dish, 4 * loopD.length, 4) => Thread(:thread, 1, 4),  # mma input k, bits 23
        Dish(:dish, 4 * loopD.length * 4, Wd) => Warp(:warp, 1, Wd),
        time012 => Thread(:thread, 4, 8), # mma input n, bits 012
        time34 => loopT3,
        time56 => loopT2,
        time7etc => loopT1,
        Time(:time, idiv(T, Bt), Bt) => Block(:block, 1, Bt),
        Polr(:polr, 1, Bp) => Block(:block, Bt, Bp),
        Polr(:polr, Bp, Wp) => Warp(:warp, Wd * Wb, Wp),
        freq => Block(:block, Bt * Bp, F),
    ])

    # A-matrix layout

    # Section 4, eqn. (17)
    @assert 4 * loopD.length * 4 * Wd == D
    @assert 8 * loopB.length * Wb == B
    layout_A_registers = Layout([
        int8value => SIMD(:simd, 1, 8),
        cplx => Register(:cplx, 1, C),
        dish01 => SIMD(:simd, 8, 4), # mma input k, bits 01
        Dish(:dish, 4, loopD.length) => Register(:dish, 4, loopD.length),
        Dish(:dish, 4 * loopD.length, 4) => Thread(:thread, 1, 4),  # mma input k, bits 23
        Dish(:dish, 4 * loopD.length * 4, Wd) => Warp(:warp, 1, Wd),
        beam012 => Thread(:thread, 4, 8), # mma output m, bits 012
        Beam(:beam, 8, loopB.length) => Register(:beam, 8, loopB.length),
        Beam(:beam, 8 * loopB.length, Wb) => Warp(:warp, Wd, Wb),
        Polr(:polr, 1, Bp) => Block(:block, Bt, Bp),
        Polr(:polr, Bp, Wp) => Warp(:warp, Wd * Wb, Wp),
        freq => Block(:block, Bt * Bp, F),
    ])

    # J-matrix layout

    # Section 5, eqn. (24)
    layout_J_registers = Layout([
        int32value => SIMD(:simd, 1, 32),
        cplx => Register(:cplx, 1, C),
        time012 => Thread(:thread, 1, 8),
        time34 => Register(:time, 8, 4),
        time56 => loopT2,
        time7etc => loopT1,
        Time(:time, idiv(T, Bt), Bt) => Block(:block, 1, Bt),
        beam01 => Thread(:thread, 8, 4),
        beam2etc => Warp(:warp, 1, Wd * Wb * Wp),
        Polr(:polr, 1, Bp) => Block(:block, Bt, Bp),
        Polr(:polr, Bp, Wp) => Register(:polr, Bp, Wp),
        freq => Block(:block, Bt * Bp, F),
    ])

    # s layout

    layout_s_registers = Layout([
        int32value => SIMD(:simd, 1, 32),
        beam01 => Thread(:thread, 8, 4),
        beam2etc => Warp(:warp, 1, idiv(B, 4)),
        Polr(:polr, 1, Bp) => Block(:block, Bt, Bp),
        Polr(:polr, Bp, Wp) => Register(:polr, Bp, Wp),
        freq => Block(:block, Bt * Bp, F),
    ])

    # info layout

    layout_info_registers = Layout([
        int32value => SIMD(:simd, 1, 32),
        Index{Physics,ThreadTag}(:thread, 1, num_threads) => Thread(:thread, 1, num_threads),
        Index{Physics,WarpTag}(:warp, 1, num_warps) => Warp(:warp, 1, num_warps),
        Index{Physics,BlockTag}(:block, 1, num_blocks) => Block(:block, 1, num_blocks),
    ])

    # log layout

    layout_log_registers = Layout([
        int32value => SIMD(:simd, 1, 32), Index{Physics,BlockTag}(:block, 1, num_blocks) => Block(:block, 1, num_blocks)
    ])

    # Generate code

    emitter = Emitter(kernel_setup)

    apply!(emitter, :info => layout_info_registers, 1i32)
    store!(emitter, :info_memory => layout_info_memory, :info)

    # Read parameters `Tmin`, `Tmax`
    if!(
        emitter, :(!(0i32 ≤ Tmin < $(Int32(T)) && Tmin ≤ Tmax < $(Int32(2 * T)) && (Tmax - Tmin) % $(Int32(T1_stride)) == 0i32))
    ) do emitter
        apply!(emitter, :info => layout_info_registers, 2i32)
        store!(emitter, :info_memory => layout_info_memory, :info)
        trap!(emitter)
        return nothing
    end

    load!(emitter, :s => layout_s_registers, :s_memory => layout_s_global)
    apply!(emitter, :s, [:s], (s,) -> :($s - $σ))

    if Wp == 1
        if!(emitter, :(!(0i32 < s < 32i32))) do emitter
            apply!(emitter, :info => layout_info_registers, 3i32)
            store!(emitter, :info_memory => layout_info_memory, :info)
            trap!(emitter)
            return nothing
        end
    elseif Wp == 2
        if!(emitter, :(!(0i32 < s_polr0 < 32i32 && 0i32 < s_polr1 < 32i32))) do emitter
            apply!(emitter, :info => layout_info_registers, 3i32)
            store!(emitter, :info_memory => layout_info_memory, :info)
            trap!(emitter)
            return nothing
        end
    else
        @assert false
    end

    if!(
        emitter,
        quote
            let
                thread = IndexSpaces.cuda_threadidx()
                warp = IndexSpaces.cuda_warpidx()
                thread == 0i32 && warp == 0i32
            end
        end,
    ) do emitter
        apply!(emitter, :logval => layout_log_registers, 0i32)
        store!(emitter, :log_memory => layout_log_memory, :logval)
        return nothing
    end
    sync_threads!(emitter)

    push!(
        emitter.statements,
        quote
            hasoverflow = false
        end,
    )

    if D == 1024
        layout_A0_registers = Layout([
            int8value => SIMD(:simd, 1, 8),
            cplx => SIMD(:simd, 8, 2),          # want register
            dish0 => SIMD(:simd, 16, 2),        # want simd3
            dish1 => Register(:cplx, 1, C),     # want simd4
            dish2345 => Register(:dish, 4, 16), # final
            dish67 => Thread(:thread, 1, 4),    # final
            dish89 => Warp(:warp, 1, Wd),       # final
            beam012 => Thread(:thread, 4, 8),   # final
            beam3 => Register(:beam, 8, 2),     # final
            beam4etc => Warp(:warp, Wd, Wb),    # final
            Polr(:polr, 1, Bp) => Block(:block, Bt, Bp),
            Polr(:polr, Bp, Wp) => Warp(:warp, Wd, Wp),
            freq => Block(:block, Bt * P, F),
        ])

        load!(emitter, :A => layout_A0_registers, :A_memory => layout_A_memory)
        permute!(emitter, :A, :A, Register(:cplx, 1, C), SIMD(:simd, 16, 2))
        permute!(emitter, :A, :A, Register(:cplx, 1, C), SIMD(:simd, 8, 2))
    elseif D == 512
        layout_A0_registers = Layout([
            int8value => SIMD(:simd, 1, 8),
            cplx => SIMD(:simd, 8, 2),       # want register
            dish0 => SIMD(:simd, 16, 2),     # want simd3
            dish1 => Register(:cplx, 1, C),  # want simd4
            dish2 => Register(:dish, 4, 2),  # final
            dish34 => Thread(:thread, 2, 4), # want register
            dish5 => Thread(:thread, 1, 2),  # final
            dish6 => Register(:dish, 8, 2),  # want thread2
            dish78 => Warp(:warp, 1, Wd),    # final
            beam0 => Register(:dish, 16, 2), # want thread4
            beam12 => Thread(:thread, 8, 4), # final
            beam3 => Register(:beam, 8, 2),  # final
            beam4etc => Warp(:warp, Wd, Wb),
            Polr(:polr, 1, Bp) => Block(:block, Bt, Bp),
            Polr(:polr, Bp, Wp) => Warp(:warp, Wd, Wp),
            freq => Block(:block, Bt * P, F),
        ])

        load!(emitter, :A => layout_A0_registers, :A_memory => layout_A_memory; align=16)
        permute!(emitter, :A, :A, Register(:cplx, 1, C), SIMD(:simd, 16, 2))
        permute!(emitter, :A, :A, Register(:cplx, 1, C), SIMD(:simd, 8, 2))
        permute!(emitter, :A, :A, Register(:dish, 8, 2), Thread(:thread, 2, 2))
        permute!(emitter, :A, :A, Register(:dish, 16, 2), Thread(:thread, 4, 2))
    elseif D == 256
        layout_A0_registers = Layout([
            int8value => SIMD(:simd, 1, 8),
            cplx => SIMD(:simd, 8, 2),       # want register
            dish0 => SIMD(:simd, 16, 2),     # want simd3
            dish1 => Register(:cplx, 1, C),  # want simd4
            dish2 => Register(:dish, 4, 2),  # final
            dish34 => Thread(:thread, 2, 4), # want register
            dish5 => Thread(:thread, 1, 2),  # final
            dish6 => Register(:dish, 8, 2),  # want thread2
            dish7 => Warp(:warp, 1, Wd),     # final
            beam0 => Register(:dish, 16, 2), # want thread4
            beam12 => Thread(:thread, 8, 4), # final
            beam3etc => Warp(:warp, Wd, Wb),
            Polr(:polr, 1, Bp) => Block(:block, Bt, Bp),
            Polr(:polr, Bp, Wp) => Warp(:warp, Wd * Wb, Wp),
            freq => Block(:block, Bt * P, F),
        ])

        load!(emitter, :A => layout_A0_registers, :A_memory => layout_A_memory; align=16)
        permute!(emitter, :A, :A, Register(:cplx, 1, C), SIMD(:simd, 16, 2))
        permute!(emitter, :A, :A, Register(:cplx, 1, C), SIMD(:simd, 8, 2))
        permute!(emitter, :A, :A, Register(:dish, 8, 2), Thread(:thread, 2, 2))
        permute!(emitter, :A, :A, Register(:dish, 16, 2), Thread(:thread, 4, 2))
    elseif D == 64
        @assert Wd == 1
        layout_A0_registers = Layout([
            int8value => SIMD(:simd, 1, 8),
            cplx => SIMD(:simd, 8, 2),       # want register
            dish0 => SIMD(:simd, 16, 2),     # want simd3
            dish1 => Register(:cplx, 1, C),  # want simd4
            dish2 => Register(:dish, 4, 2),  # final
            dish3 => Thread(:thread, 4, 2),  # want register
            dish45 => Thread(:thread, 1, 4), # final
            beam0 => Register(:dish, 8, 2),  # want thread2
            beam12 => Thread(:thread, 8, 4), # final
            beam3 => Warp(:warp, Wd, Wb),
            Polr(:polr, 1, Bp) => Block(:block, Bt, Bp),
            Polr(:polr, Bp, Wp) => Warp(:warp, Wd * Wb, Wp),
            freq => Block(:block, Bt * Bp, F),
        ])

        load!(emitter, :A => layout_A0_registers, :A_memory => layout_A_memory; align=16)
        permute!(emitter, :A, :A, Register(:cplx, 1, C), SIMD(:simd, 16, 2))
        permute!(emitter, :A, :A, Register(:cplx, 1, C), SIMD(:simd, 8, 2))
        permute!(emitter, :A, :A, Register(:dish, 8, 2), Thread(:thread, 4, 2))
    else
        @assert false
    end
    @assert emitter.environment[:A] == layout_A_registers

    loop!(emitter, Time(:time, loopT1.offset, loopT1.length) => loopT1) do emitter
        push!(
            emitter.statements,
            quote
                Tmin + T1 ≥ Tmax && break
            end,
        )

        loop!(emitter, Time(:time, loopT2.offset, loopT2.length) => loopT2) do emitter

            # Step 1: transferring global memory to shared memory
            if!(emitter, :(IndexSpaces.cuda_warpidx() < $(Int32(num_warps_for_Ecopy)))) do emitter
                load!(
                    emitter,
                    :E => layout_E_registers,
                    :E_memory => layout_E_memory;
                    align=16,
                    postprocess=addr -> :(
                        let
                            offset = $(shrinkmul(idiv(D, 4) * P * F, :Tmin, T))
                            length = $(shrink(idiv(D, 4) * P * F * T))
                            mod($addr + offset, length)
                        end
                    ),
                )
                store!(emitter, :E_shared => layout_E_shared, :E)
                return nothing
            end
            sync_threads!(emitter)

            # Step 2: matrix multiplication
            loop!(emitter, Time(:time, loopT3.offset, loopT3.length) => loopT3) do emitter
                unrolled_loop!(emitter, Beam(:beam, loopB.offset, loopB.length) => loopB) do emitter
                    select!(emitter, :AselB, :A, Register(:beam, loopB.offset, loopB.length) => loopB)

                    @assert 8 * loopB.length * Wb == B
                    layout_Jureim_registers = Layout([
                        int32value => SIMD(:simd, 1, 32),
                        Dish(:dish, idiv(D, Wd), Wd) => Warp(:warp, 1, Wd),
                        time0 => Register(:time, 1, 2), # mma spectator 0
                        time12 => Thread(:thread, 1, 4), # mma spectator 12
                        time34 => loopT3,
                        time56 => loopT2,
                        time7etc => loopT1,
                        Time(:time, idiv(T, Bt), Bt) => Block(:block, 1, Bt),
                        beam012 => Thread(:thread, 4, 8), # mma output 012
                        Beam(:beam, 8, loopB.length) => loopB,
                        Beam(:beam, 8 * loopB.length, Wb) => Warp(:warp, Wd, Wb),
                        Polr(:polr, 1, Bp) => Block(:block, Bt, Bp),
                        Polr(:polr, Bp, Wp) => Warp(:warp, Wd * Wb, Wp),
                        freq => Block(:block, Bt * Bp, F),
                    ])

                    apply!(emitter, :Jurepos => layout_Jureim_registers, Int32(0))
                    apply!(emitter, :Jureneg => layout_Jureim_registers, Int32(0))
                    apply!(emitter, :Juim => layout_Jureim_registers, Int32(0))

                    unrolled_loop!(emitter, Dish(:dish, loopD.offset, loopD.length) => loopD) do emitter
                        select!(emitter, :AselBD, :AselB, Register(:dish, loopD.offset, loopD.length) => loopD)
                        split!(emitter, [:Are, :Aim], :AselBD, cplx)

                        load!(emitter, :E0 => layout_E0_registers, :E_shared => layout_E_shared)

                        widen!(emitter, :E1, :E0, SIMD(:simd, 4, 2) => Register(:cplx, 1, C); swapped_withoffset=true)

                        split!(emitter, [:E1re, :E1im], :E1, cplx)

                        mma_beams = [beam0, beam1, beam2]
                        mma_dishes = [
                            Dish(:dish, 1, 2), Dish(:dish, 2, 2), Dish(:dish, 4 * loopD.length, 2), Dish(:dish, 8 * loopD.length, 2)
                        ]
                        mma_times = [time0, time1, time2]
                        mma_row_col_m8n8k16_s8!(
                            emitter,
                            :Jurepos,
                            :Are => (mma_beams, mma_dishes),
                            :E1re => (mma_dishes, mma_times),
                            :Jurepos => (mma_beams, mma_times),
                        )
                        mma_row_col_m8n8k16_s8!(
                            emitter,
                            :Jureneg,
                            :Aim => (mma_beams, mma_dishes),
                            :E1im => (mma_dishes, mma_times),
                            :Jureneg => (mma_beams, mma_times),
                        )
                        mma_row_col_m8n8k16_s8!(
                            emitter,
                            :Juim,
                            :Are => (mma_beams, mma_dishes),
                            :E1im => (mma_dishes, mma_times),
                            :Juim => (mma_beams, mma_times),
                        )
                        mma_row_col_m8n8k16_s8!(
                            emitter,
                            :Juim,
                            :Aim => (mma_beams, mma_dishes),
                            :E1re => (mma_dishes, mma_times),
                            :Juim => (mma_beams, mma_times),
                        )

                        return nothing
                    end

                    apply!(emitter, :Jure, [:Jurepos, :Jureneg], (Jurepos, Jureneg) -> :($Jurepos - $Jureneg))
                    merge!(emitter, :Ju, [:Jure, :Juim], cplx => Register(:cplx, 1, C))

                    # TODO: Break ties to even?
                    @assert σ ≥ 0
                    if σ == 0
                        # do nothing
                    else
                        apply!(emitter, :Ju, [:Ju], (Ju,) -> :(($Ju + $(Int32(1 << (σ - 1)))) >> $(UInt32(σ))))
                    end

                    # Note: `cvs_pack_s16` saturates, so we don't need to clamp
                    # Note: We wouldn't need to clamp anyway since the shift above prevents overflow
                    # apply!(emitter, :Ju, :Ju, Ju -> :(clamp($Ju, $(-Int32(0x7fff)):$(+Int32(0x7fff)))))
                    narrow!(emitter, :Ju, :Ju, Register(:cplx, 1, C) => SIMD(:simd, 16, 2))

                    store!(emitter, :Ju_shared => layout_Ju_shared, :Ju)

                    return nothing
                end

                return nothing
            end
            sync_threads!(emitter)

            # Step 3: reduce and quantize

            layout_Ju_registers = Layout([
                int16value => SIMD(:simd, 1, 16),
                cplx => SIMD(:simd, 16, C),
                Dish(:dish, 4 * loopD.length * 4, Wd) => Register(:dish, 4 * loopD.length * 4, Wd),
                beam01 => Thread(:thread, 8, 4),
                beam2etc => Warp(:warp, 1, Wd * Wb * Wp),
                time012 => Thread(:thread, 1, 8),
                time34 => Register(:time, 8, 4),
                time56 => loopT2,
                time7etc => loopT1,
                Time(:time, idiv(T, Bt), Bt) => Block(:block, 1, Bt),
                Polr(:polr, 1, Bp) => Block(:block, Bt, Bp),
                Polr(:polr, Bp, Wp) => Register(:polr, Bp, Wp),
                freq => Block(:block, Bt * Bp, F),
            ])

            load!(emitter, :Ju => layout_Ju_registers, :Ju_shared => layout_Ju_shared)
            widen!(emitter, :Ju, :Ju, SIMD(:simd, 16, 2) => Register(:cplx, 1, C))

            if D == 1024
                split!(emitter, [:Julo, :Juhi], :Ju, Dish(:dish, 256, 2))
                # TODO use add_sat
                apply!(emitter, :Ju, [:Julo, :Juhi], (Julo, Juhi) -> :($Julo + $Juhi))
                split!(emitter, [:Julo, :Juhi], :Ju, Dish(:dish, 512, 2))
                # TODO use add_sat
                apply!(emitter, :J, [:Julo, :Juhi], (Julo, Juhi) -> :($Julo + $Juhi))
            elseif D == 512
                split!(emitter, [:Julo, :Juhi], :Ju, Dish(:dish, 128, 2))
                # TODO use add_sat
                apply!(emitter, :Ju, [:Julo, :Juhi], (Julo, Juhi) -> :($Julo + $Juhi))
                split!(emitter, [:Julo, :Juhi], :Ju, Dish(:dish, 256, 2))
                # TODO use add_sat
                apply!(emitter, :J, [:Julo, :Juhi], (Julo, Juhi) -> :($Julo + $Juhi))
            elseif D == 256
                split!(emitter, [:Julo, :Juhi], :Ju, Dish(:dish, 128, 2))
                # TODO use add_sat
                apply!(emitter, :J, [:Julo, :Juhi], (Julo, Juhi) -> :($Julo + $Juhi))
            elseif D == 64
                apply!(emitter, :J, [:Ju], (Ju,) -> :($Ju))
            else
                @assert false
            end

            @assert emitter.environment[:J] == layout_J_registers

            apply!(emitter, :J, [:J, :s], (J, s) -> :(($J + ($(Int32(1)) << ($s % UInt32 - 0x1))) >> ($s % UInt32)))

            # TODO: Try this: Shift values left by 4, rely on saturation when converting, then shift right and mask (doesn't work)
            # TODO: Try this: Pack to Int16, the clamp, then pack to Int8 (doesn't work, no efficient 16-bit clamp)
            # apply!(emitter, :J, [:J], J -> :(clamp($J, ($(-Int32(0x7))):($(+Int32(0x7))))))
            apply!(emitter, :J, [:J], J -> quote
                let
                    Jnew = clamp($J, ($(-Int32(0x7))):($(+Int32(0x7))))
                    hasoverflow |= Jnew != $J
                    Jnew
                end
            end)
            narrow3!(
                emitter,
                :J,
                :J,
                Register(:cplx, 1, C) => SIMD(:simd, 4, 2),
                Register(:time, 8, 2) => SIMD(:simd, 8, 2),
                Register(:time, 16, 2) => SIMD(:simd, 16, 2),
            )

            unselect!(emitter, :Jper, :J, loopT2 => Register(:time, loopT2.offset, loopT2.length))

            return nothing
        end

        permute!(emitter, :Jper, :Jper, Register(:time, 32, 2), SIMD(:simd, 8, 2))
        permute!(emitter, :Jper, :Jper, Register(:time, 32, 2), Thread(:thread, 1, 2))
        permute!(emitter, :Jper, :Jper, Register(:time, 32, 2), SIMD(:simd, 8, 2))
        permute!(emitter, :Jper, :Jper, Register(:time, 64, 2), Thread(:thread, 2, 2))
        permute!(emitter, :Jper, :Jper, Register(:time, 64, 2), SIMD(:simd, 16, 2))
        permute!(emitter, :Jper, :Jper, Register(:time, 32, 2), Thread(:thread, 4, 2))
        permute!(emitter, :Jper, :Jper, Register(:time, 64, 2), Thread(:thread, 1, 2))

        store!(emitter, :J_memory => layout_J_memory, :Jper; align=16)

        return nothing
    end

    push!(
        emitter.statements,
        quote
            any_hasoverflow = sync_threads_or(hasoverflow)
        end,
    )
    if!(emitter, :any_hasoverflow) do emitter
        if!(
            emitter,
            quote
                let
                    thread = IndexSpaces.cuda_threadidx()
                    warp = IndexSpaces.cuda_warpidx()
                    thread == 0i32 && warp == 0i32
                end
            end,
        ) do emitter
            apply!(emitter, :logval => layout_log_registers, 1i32)
            store!(emitter, :log_memory => layout_log_memory, :logval)
        end
    end

    apply!(emitter, :info => layout_info_registers, 0i32)
    store!(emitter, :info_memory => layout_info_memory, :info)

    stmts = clean_code(
        quote
            @inbounds begin
                $(emitter.init_statements...)
                $(emitter.statements...)
            end
        end,
    )

    return stmts
end

println("[Creating bb kernel...]")
const bb_kernel = make_bb_kernel()
println("[Done creating bb kernel]")

@eval function bb(Tmin::Int32, Tmax::Int32, A_memory, E_memory, s_memory, J_memory, info_memory, log_memory)
    E_shared = @cuDynamicSharedMem($E_shared_type, $E_shared_length, $(sizeof(UInt32) * E_shared_offset))
    Ju_shared = @cuDynamicSharedMem($Ju_shared_type, $Ju_shared_length, $(sizeof(UInt32) * Ju_shared_offset))
    $bb_kernel
    return nothing
end

function main(; compile_only::Bool=false, output_kernel::Bool=false, run_selftest::Bool=false, nruns::Int=0)
    if !compile_only
        println("CHORD 8-bit baseband beamformer")
        println("J[t,p,f,b] = s[b,p,f] Σ[d] A[d,b,p,f] E[d,p,f,t]")
    end

    if output_kernel
        open("output-$card/bb_$setup.jl", "w") do fh
            println(fh, "# Julia source code for CUDA baseband beamformer")
            println(fh, "# This file has been generated automatically by `bb.jl`.")
            println(fh, "# Do not modify this file, your changes will be lost.")
            println(fh)
            println(fh, bb_kernel)
            return nothing
        end
    end

    if !compile_only
        println("Compiling kernel...")
    end

    num_threads = kernel_setup.num_threads
    num_warps = kernel_setup.num_warps
    num_blocks = kernel_setup.num_blocks
    num_blocks_per_sm = kernel_setup.num_blocks_per_sm
    shmem_bytes = kernel_setup.shmem_bytes
    @assert num_warps * num_blocks_per_sm ≤ 32 # (???)
    @assert shmem_bytes ≤ 100 * 1024 # NVIDIA A10/A40 have 100 kB shared memory
    kernel = @cuda launch = false minthreads = num_threads * num_warps blocks_per_sm = num_blocks_per_sm bb(
        Int32(0),
        Int32(0),
        CUDA.zeros(Int8x4, 0),
        CUDA.zeros(Int4x8, 0),
        CUDA.zeros(Int32, 0),
        CUDA.zeros(Int4x8, 0),
        CUDA.zeros(Int32, 0),
        CUDA.zeros(Int32, 0),
    )
    attributes(kernel.fun)[CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = shmem_bytes

    if compile_only
        return nothing
    end

    if output_kernel
        ptx = read("output-$card/bb_$setup.ptx", String)
        ptx = replace(ptx, r".extern .func gpu_([^;]*);"s => s".func gpu_\1.noreturn\n{\n\ttrap;\n}")
        open("output-$card/bb_$setup.ptx", "w") do fh
            println(fh, "// PTX kernel code for CUDA baseband beamformer")
            println(fh, "// This file has been generated automatically by `bb.jl`.")
            println(fh, "// Do not modify this file, your changes will be lost.")
            println(fh)
            write(fh, ptx)
            return nothing
        end
        sass = read("output-$card/bb_$setup.sass", String)
        open("output-$card/bb_$setup.sass", "w") do fh
            println(fh, "// SASS kernel code for CUDA baseband beamformer")
            println(fh, "// This file has been generated automatically by `bb.jl`.")
            println(fh, "// Do not modify this file, your changes will be lost.")
            println(fh)
            write(fh, sass)
            return nothing
        end
        kernel_symbol = match(r"\s\.globl\s+(\S+)"m, ptx).captures[1]
        open("output-$card/bb_$setup.yaml", "w") do fh
            println(fh, "# Metadata for the CUDA baseband beamformer")
            println(fh, "# This file has been generated automatically by `bb.jl`.")
            println(fh, "# Do not modify this file, your changes will be lost.")
            println(fh)
            print(
                fh,
                """
        --- !<tag:chord-observatory.ca/x-engine/kernel-description-1.0.0>
        kernel-description:
          name: "bb"
          description: "baseband beamformer"
          design-parameters:
            number-of-beams: $B
            number-of-complex-components: $C
            number-of-dishes: $D
            number-of-frequencies: $F
            number-of-polarizations: $P
            number-of-timesamples: $T
            sampling-time-μsec: $sampling_time_μsec
            shift-parameter-σ: $σ
          compile-parameters:
            minthreads: $(num_threads * num_warps)
            blocks_per_sm: $num_blocks_per_sm
          call-parameters:
            threads: [$num_threads, $num_warps]
            blocks: [$num_blocks]
            shmem_bytes: $shmem_bytes
          kernel-symbol: "$kernel_symbol"
          kernel-arguments:
            - name: "Tmin"
              intent: in
              type: Int32
            - name: "Tmax"
              intent: in
              type: Int32
            - name: "A"
              intent: in
              type: Int8
              indices: [C, D, B, P, F]
              shape: [$C, $D, $B, $P, $F]
              strides: [1, $C, $(C*D), $(C*D*B), $(C*D*B*P)]
            - name: "E"
              intent: in
              type: Int4
              indices: [C, D, P, F, T]
              shape: [$C, $D, $P, $F, $T]
              strides: [1, $C, $(C*D), $(C*D*P), $(C*D*P*F)]
            - name: "s"
              intent: in
              type: Int32
              indices: [B, P, F]
              shape: [$B, $P, $F]
              strides: [1, $B, $(B*P)]
            - name: "J"
              intent: out
              type: Int4
              indices: [C, T, P, F, B]
              shape: [$C, $Tout, $P, $F, $B]
              strides: [1, $C, $(C*Tout), $(C*Tout*P), $(C*Tout*P*F)]
            - name: "info"
              intent: out
              type: Int32
              indices: [thread, warp, block]
              shapes: [$num_threads, $num_warps, $num_blocks]
              strides: [1, $num_threads, $(num_threads*num_warps)]
            - name: "log"
              intent: inout
              type: Int32
              indices: [block]
              shapes: [$num_blocks]
              strides: [1]
        ...
        """,
            )
            return nothing
        end
        cxx = read("kernels/bb_template.cxx", String)
        cxx = Mustache.render(
            cxx,
            Dict(
                "kernel_name" => "BasebandBeamformer_$setup",
                "kernel_design_parameters" => [
                    Dict("type" => "int", "name" => "cuda_number_of_beams", "value" => "$B"),
                    Dict("type" => "int", "name" => "cuda_number_of_complex_components", "value" => "$C"),
                    Dict("type" => "int", "name" => "cuda_number_of_dishes", "value" => "$D"),
                    Dict("type" => "int", "name" => "cuda_number_of_frequencies", "value" => "$F"),
                    Dict("type" => "int", "name" => "cuda_number_of_polarizations", "value" => "$P"),
                    Dict("type" => "int", "name" => "cuda_number_of_timesamples", "value" => "$T"),
                    # Dict("type" => "int", "name" => "cuda_granularity_number_of_timesamples", "value" => "$T1_stride"),
                    Dict("type" => "int", "name" => "cuda_granularity_number_of_timesamples", "value" => "$Tout"),
                    Dict("type" => "int", "name" => "cuda_shift_parameter_sigma", "value" => "$σ"),
                ],
                "minthreads" => num_threads * num_warps,
                "num_blocks_per_sm" => num_blocks_per_sm,
                "num_threads" => num_threads,
                "num_warps" => num_warps,
                "num_blocks" => num_blocks,
                "shmem_bytes" => shmem_bytes,
                "kernel_symbol" => kernel_symbol,
                "kernel_arguments" => [
                    Dict(
                        "name" => "Tmin",
                        "kotekan_name" => "Tmin",
                        "type" => "int32",
                        "isoutput" => false,
                        "hasbuffer" => false,
                        "isscalar" => true,
                    ),
                    Dict(
                        "name" => "Tmax",
                        "kotekan_name" => "Tmax",
                        "type" => "int32",
                        "isoutput" => false,
                        "hasbuffer" => false,
                        "isscalar" => true,
                    ),
                    Dict(
                        "name" => "A",
                        "kotekan_name" => "gpu_mem_phase",
                        "type" => "int8",
                        "axes" => [
                            Dict("label" => "C", "length" => C),
                            Dict("label" => "D", "length" => D),
                            Dict("label" => "B", "length" => B),
                            Dict("label" => "P", "length" => P),
                            Dict("label" => "F", "length" => F),
                        ],
                        "isoutput" => false,
                        "hasbuffer" => true,
                        "isscalar" => false,
                    ),
                    Dict(
                        "name" => "E",
                        "kotekan_name" => "gpu_mem_voltage",
                        "type" => "int4p4chime",
                        "axes" => [
                            Dict("label" => "D", "length" => D),
                            Dict("label" => "P", "length" => P),
                            Dict("label" => "F", "length" => F),
                            Dict("label" => "T", "length" => T),
                        ],
                        "isoutput" => false,
                        "hasbuffer" => true,
                        "isscalar" => false,
                    ),
                    Dict(
                        "name" => "s",
                        "kotekan_name" => "gpu_mem_output_scaling",
                        "type" => "int32",
                        "axes" => [
                            Dict("label" => "B", "length" => B),
                            Dict("label" => "P", "length" => P),
                            Dict("label" => "F", "length" => F),
                        ],
                        "isoutput" => false,
                        "hasbuffer" => true,
                    ),
                    Dict(
                        "name" => "J",
                        "type" => "int4p4",
                        "kotekan_name" => "gpu_mem_formed_beams",
                        "axes" => [
                            Dict("label" => "T", "length" => Tout),
                            Dict("label" => "P", "length" => P),
                            Dict("label" => "F", "length" => F),
                            Dict("label" => "B", "length" => B),
                        ],
                        "isoutput" => true,
                        "hasbuffer" => true,
                        "isscalar" => false,
                    ),
                    Dict(
                        "name" => "info",
                        "kotekan_name" => "gpu_mem_info",
                        "type" => "int32",
                        "axes" => [
                            Dict("label" => "thread", "length" => num_threads),
                            Dict("label" => "warp", "length" => num_warps),
                            Dict("label" => "block", "length" => num_blocks),
                        ],
                        "isoutput" => true,
                        "hasbuffer" => false,
                        "isscalar" => false,
                    ),
                    Dict(
                        "name" => "log",
                        "kotekan_name" => "gpu_mem_log",
                        "type" => "int32",
                        "axes" => [Dict("label" => "block", "length" => num_blocks)],
                        "isoutput" => true,
                        "hasbuffer" => false,
                        "isscalar" => false,
                    ),
                ],
            ),
        )
        write("output-$card/bb_$setup.cxx", cxx)
    end

    println("Allocating input data...")

    # TODO: determine types and sizes automatically
    A_memory = Array{Int8x4}(undef, idiv(D, 2) * B * P * F)
    E_memory = Array{Int4x8}(undef, idiv(D, 4) * P * F * T)
    s_memory = Array{Int32}(undef, B * P * F)
    J_wanted = Array{Int4x8}(undef, idiv(Tout, 4) * P * F * B)
    info_wanted = Array{Int32}(undef, num_threads * num_warps * num_blocks)
    log_wanted = Array{Int32}(undef, num_blocks)

    println("Setting up input data...")
    map!(i -> zero(Int8x4), A_memory, A_memory)
    map!(i -> zero(Int4x8), E_memory, E_memory)
    map!(i -> Int32(σ + 1), s_memory, s_memory)
    map!(i -> zero(Int4x8), J_wanted, J_wanted)
    map!(i -> zero(Int32), info_wanted, info_wanted)
    map!(i -> zero(Int32), log_wanted, log_wanted)
    # map!(i -> rand(Int8x4), A_memory, A_memory)
    # map!(i -> rand(Int4x8), E_memory, E_memory)
    # map!(i -> rand(Int32(1):Int32(10)), s_memory, s_memory)

    Tmin = Int32(0)
    Tmax = Int32(Tout)

    input = :random
    if input ≡ :zero
        # do nothing
    elseif input ≡ :random
        Random.seed!(0)

        # Choose all s
        s_memory .= rand((σ + 1):(σ + 10), F * P * B)

        # Choose A and E
        for iter in 1:1000
            freq = 0
            polr = 0
            time = 0
            dish = 0
            beam = 0
            Eval = 0
            Aval = 0
            sval = 0
            Jval = 0
            while true
                freq = rand(0:(F - 1))
                polr = rand(0:(P - 1))
                time = rand(0:(T - 1))
                dish = rand(0:(D - 1))
                beam = rand(0:(B - 1))
                Eval = rand(-7:7) + im * rand(-7:7)
                Aval = rand(-127:127) + im * rand(-127:127)
                sval = s_memory[(freq * P + polr) * B + beam + 1]
                Juval = Aval * Eval
                Juval = shift(Juval, σ)
                @assert max(abs(Juval.re), abs(Juval.im)) ≤ 32767
                Jval = Juval
                Jval = shift(Jval, sval - σ)
                0 < max(abs(Jval.re), abs(Jval.im)) ≤ 7 && break
            end
            println("    freq=$freq polr=$polr time=$time dish=$dish beam=$beam Eval=$Eval Aval=$Aval sval=$sval Jval=$Jval")
            A_memory[(((freq * P + polr) * B + beam) * D + dish) ÷ 2 + 1] += if dish % 2 == 0
                Int8x4(Aval.re, Aval.im, 0, 0)
            elseif dish % 2 == 1
                Int8x4(0, 0, Aval.re, Aval.im)
            else
                @assert false
            end
            E_memory[(((time * F + freq) * P + polr) * D + dish) ÷ 4 + 1] += if dish % 4 == 0
                Int4x8(Eval.re, Eval.im, 0, 0, 0, 0, 0, 0)
            elseif dish % 4 == 1
                Int4x8(0, 0, Eval.re, Eval.im, 0, 0, 0, 0)
            elseif dish % 4 == 2
                Int4x8(0, 0, 0, 0, Eval.re, Eval.im, 0, 0)
            elseif dish % 4 == 3
                Int4x8(0, 0, 0, 0, 0, 0, Eval.re, Eval.im)
            else
                @assert false
            end
        end
    elseif input ≡ :random2
        Random.seed!(0)

        # Choose all s
        s_memory .= rand((σ + 1):(σ + 10), P * F * B)

        # Choose A and E
        A_memory .= rand(Int8x4, length(A_memory))
        E_memory .= rand(Int4x8, length(E_memory))

    else
        @assert false
    end

    if run_selftest
        println("Evaluating kernel on CPU...")
        Threads.@threads for b in 0:(B - 1)
            for f in 0:(F - 1), p in 0:(P - 1), t in 0:(Tout - 1)
                s = s_memory[(f * P + p) * B + b + 1]
                Ju = 0 + 0im
                for d in 0:(D - 1)
                    A = Complex{Int}(reinterpret(NTuple{2,Int8}, A_memory)[((f * P + p) * B + b) * D + d + 1]...)
                    E = Complex{Int}(convert(NTuple{2,Int8}, reinterpret(Int4x2, E_memory)[((t * F + f) * P + p) * D + d + 1])...)
                    Ju += A * E
                end
                Ju = shift(Ju, σ)
                @assert max(abs(Ju.re), abs(Ju.im)) ≤ 32767
                J = Ju
                J = shift(J, s - σ)
                reinterpret(Int4x2, J_wanted)[((b * F + f) * P + p) * T + t + 1] = Int4x2(
                    Int32(clamp(J.re, -7:+7)), Int32(clamp(J.im, -7:+7))
                )
            end
        end
    end

    println("Copying data from CPU to GPU...")
    A_cuda = CuArray(A_memory)
    E_cuda = CuArray(E_memory)
    s_cuda = CuArray(s_memory)
    J_cuda = CUDA.fill(Int4x8(-8, -8, -8, -8, -8, -8, -8, -8), idiv(Tout, 4) * P * F * B)
    info_cuda = CUDA.fill(-1i32, num_threads * num_warps * num_blocks)
    log_cuda = CUDA.fill(0i32, num_blocks)

    println("Running kernel...")
    kernel(
        Tmin,
        Tmax,
        A_cuda,
        E_cuda,
        s_cuda,
        J_cuda,
        info_cuda,
        log_cuda;
        threads=(num_threads, num_warps),
        blocks=num_blocks,
        shmem=shmem_bytes,
    )
    synchronize()

    if nruns > 0
        println("Starting $nruns benchmark runs...")
        stats = @timed begin
            for run in 1:nruns
                kernel(
                    Tmin,
                    Tmax,
                    A_cuda,
                    E_cuda,
                    s_cuda,
                    J_cuda,
                    info_cuda,
                    log_cuda;
                    threads=(num_threads, num_warps),
                    blocks=num_blocks,
                    shmem=shmem_bytes,
                )
            end
            synchronize()
        end
        println("Finished benchmark runs in $(stats.time) seconds.")
        # All times in μsec
        runtime = stats.time / nruns * 1.0e+6
        num_frequencies_scaled = F₀
        runtime_scaled = runtime / F * num_frequencies_scaled
        dataframe_length = T * sampling_time_μsec
        fraction = runtime_scaled / dataframe_length
        round1(x) = round(x; digits=1)
        println("""
        benchmark-result:
          kernel: "bb"
          description: "baseband beamformer"
          design-parameters:
            number-of-beams: $B
            number-of-complex-components: $C
            number-of-dishes: $D
            number-of-frequencies: $F
            number-of-polarizations: $P
            number-of-timesamples: $T
            sampling-time-μsec: $sampling_time_μsec
            shift-parameter-σ: $σ
          compile-parameters:
            minthreads: $(num_threads * num_warps)
            blocks_per_sm: $num_blocks_per_sm
          call-parameters:
            threads: [$num_threads, $num_warps]
            blocks: [$num_blocks]
            shmem_bytes: $shmem_bytes
          result-μsec:
            runtime: $(round1(runtime))
            scaled-runtime: $(round1(runtime_scaled))
            scaled-number-of-frequencies: $num_frequencies_scaled
            dataframe-length: $(round1(dataframe_length))
            dataframe-percent: $(round1(fraction * 100))
        """)
    end

    println("Copying data back from GPU to CPU...")
    J_memory = Array(J_cuda)
    info_memory = Array(info_cuda)
    log_memory = Array(log_cuda)
    @assert all(info_memory .== 0)
    #TODO @assert all(log_memory .== 0)

    if run_selftest
        println("Checking results...")
        error_count = 0
        checked_J = falses(B, Tout, P, F)
        for f in 0:(F - 1), p in 0:(P - 1), t in 0:4:(Tout - 1), b in 0:(B - 1)
            @assert !any(checked_J[b + 1, (t + 1):(t + 4), p + 1, f + 1])
            checked_J[b + 1, (t + 1):(t + 4), p + 1, f + 1] .= true
            J = J_memory[idiv(((b * F + f) * P + p) * Tout + t, 4) + 1]
            Jwant = J_wanted[idiv(((b * F + f) * P + p) * Tout + t, 4) + 1]
            if J ≠ Jwant
                if error_count ≤ 100
                    println("    ERROR: freq=$f polr=$p time=$t beam=$b J=$J Jwant=$Jwant")
                end
                error_count += 1
            end
        end
        @assert all(checked_J)
        println("    J: $error_count errors found")
    end

    println("Done.")
    return nothing
end

if CUDA.functional()
    # Output kernel
    open("output-$card/bb_$setup.ptx", "w") do fh
        redirect_stdout(fh) do
            @device_code_ptx main(; compile_only=true)
        end
    end
    open("output-$card/bb_$setup.sass", "w") do fh
        redirect_stdout(fh) do
            @device_code_sass main(; compile_only=true)
        end
    end
    # This call needs to happen after generating PTX code since it
    # modifies the generated PTX code
    main(; output_kernel=true)

    # # Run test
    # main(; run_selftest=true)

    # # Run benchmark
    # main(; nruns=10000)

    # # Regular run, also for profiling
    # main()
end
