# CHORD FRB beamformer
# <CHORD_FRB_beamformer.pdf>

using CUDA
using CUDASIMDTypes
using IndexSpaces
using Mustache
using Random
using StaticArrays

# const card = "A30"
const card = "A40"

if CUDA.functional()
    println("[Choosing CUDA device...]")
    CUDA.device!(0)
    println(name(device()))
    @assert name(device()) == "NVIDIA $card"
end

idiv(i::Integer, j::Integer) = (@assert iszero(i % j); i ÷ j)
# shift(x::Number, s) = (@assert s ≥ 1; (x + (1 << (s - 1))) >> s)
# shift(x::Complex, s) = Complex(shift(x.re, s), shift(x.im, s))
# Base.clamp(x::Complex, a, b) = Complex(clamp(x.re, a, b), clamp(x.im, a, b))
# Base.clamp(x::Complex, ab::UnitRange) = clamp(x, ab.start, ab.stop)

ilog2(i::Integer) = (@assert i == nextpow(2, i); trailing_zeros(i))

# Setup

# const setup = :chord
setup::Symbol

@static if setup ≡ :chord

    # CHORD Setup

    # Compile-time constants (section 4.4)

    # Full CHORD
    const D = 512
    const M = 24
    const N = 24
    const P = 2
    const F₀ = 16 * 16
    const F = 16 * 16           # benchmarking A30: 56; A40: 84

    const Touter = 48
    const Tinner = 4

    const Tds = 40              # downsampling factor

    const W = 24                # number of warps
    const B = 1                 # number of blocks per SM

elseif setup ≡ :hirax

    # HIRAX Setup

    # Compile-time constants (section 4.4)

    const D = 256
    const M = 16
    const N = 16
    const P = 2
    const F₀ = 64 * 16
    const F = 64 * 16

    const Touter = 64
    const Tinner = 8

    const Tds = 25              # downsampling factor

    const W = 16                # number of warps
    const B = 1                 # number of blocks per SM

elseif setup ≡ :pathfinder

    # CHORD pathfinder (case 2)
    const D = 64
    const M = 8
    const N = 12
    const P = 2
    const F₀ = 16 * 128
    const F = 16 * 128

    const Touter = 48
    const Tinner = 6

    const Tds = 40              # downsampling factor

    const W = 6                 # number of warps
    const B = 4                 # number of blocks per SM (TODO: check!)

else
    @assert false
end

const sampling_time_μsec = 16 * 4096 / (2 * 1200)
const C = 2
#TODO  const T = cld(32768 ÷ 16, Touter) * Touter
# const T = cld(2048 ÷ 16, Touter) * Touter
const T = 2048 ÷ 16 * 4
const T̄ = nextpow(2, fld(T, Tds))

const output_gain = 1 / (8 * Tds)

# Derived compile-time parameters (section 4.4)
const Mpad = nextpow(2, M)
const Npad = nextpow(2, N)

# Freg2 layout

const Mt = idiv(32, Npad)
const Mw = gcd(idiv(M, Mt), W)
const Mr = idiv(M, Mt * Mw)
const Tw = idiv(W, Mw)
const Tr = idiv(Touter, 2 * Tw)

# Fsh1 layout
const ΣF1 = D == 64 ? 260 : 257

# Fsh2 layout
const ΣF2 = 32 * (M + 1) + Mt

# Gsh layout
const ΣG1 = 2 * Mpad * Tinner + idiv(32, Npad)
const ΣG0 = Npad * ΣG1 + Mpad

# Registers/thread

const RF1 = cld(D, W)
const RF2 = Mr * Tr

# Shared memory bytes

const Fsh1_shmem_size = idiv(D * ΣF1, 8)
const Fsh2_shmem_size = N * ΣF2
const Gsh_shmem_size = ΣG0 * 2

# Machine setup

const num_simd_bits = 32
const num_threads = 32
const num_warps = W
const num_blocks = F
const num_blocks_per_sm = B

# Benchmark results:

# Setup for full CHORD on A40:
#
# benchmark-result:
#   kernel: "frb"
#   description: "FRB beamformer"
#   design-parameters:
#     beam-layout: [48, 48]
#     dish-layout: [24, 24]
#     downsampling-factor: 40
#     number-of-complex-components: 2
#     number-of-dishes: 512
#     number-of-frequencies: 84
#     number-of-polarizations: 2
#     number-of-timesamples: 2064
#     sampling-time-μsec: 27.30666666666667
#   compile-parameters:
#     minthreads: 768
#     blocks_per_sm: 1
#   call-parameters:
#     threads: [32, 24]
#     blocks: [84]
#     shmem_bytes: 76896
#   result-μsec:
#     runtime: 1796.2
#     scaled-runtime: 5474.2
#     scaled-number-of-frequencies: 256
#     dataframe-length: 56361.0
#     dataframe-percent: 9.7

# Setup for full CHORD on A30:
#
# benchmark-result:
#   kernel: "frb"
#   description: "FRB beamformer"
#   design-parameters:
#     beam-layout: [48, 48]
#     dish-layout: [24, 24]
#     downsampling-factor: 40
#     number-of-complex-components: 2
#     number-of-dishes: 512
#     number-of-frequencies: 56
#     number-of-polarizations: 2
#     number-of-timesamples: 2064
#     sampling-time-μsec: 27.30666666666667
#   compile-parameters:
#     minthreads: 768
#     blocks_per_sm: 1
#   call-parameters:
#     threads: [32, 24]
#     blocks: [56]
#     shmem_bytes: 76896
#   result-μsec:
#     runtime: 2311.7
#     scaled-runtime: 10568.0
#     scaled-number-of-frequencies: 256
#     dataframe-length: 56361.0
#     dataframe-percent: 18.8

# CHORD indices

@enum CHORDTag begin
    CplxTag
    DishTag
    MNTag
    DishMTag
    DishMLoTag
    DishMHiTag
    DishNTag
    DishNLoTag
    DishNHiTag
    BeamPTag
    BeamQTag
    FreqTag
    PolrTag
    TimeTag
    DSTimeTag
    ThreadTag
    WarpTag
    BlockTag
end

const Cplx = Index{Physics,CplxTag}
const Dish = Index{Physics,DishTag}
const MN = Index{Physics,MNTag}
const DishM = Index{Physics,DishMTag}
const DishMLo = Index{Physics,DishMLoTag}
const DishMHi = Index{Physics,DishMHiTag}
const DishN = Index{Physics,DishNTag}
const DishNLo = Index{Physics,DishNLoTag}
const DishNHi = Index{Physics,DishNHiTag}
const BeamP = Index{Physics,BeamPTag}
const BeamQ = Index{Physics,BeamQTag}
const Freq = Index{Physics,FreqTag}
const Polr = Index{Physics,PolrTag}
const Time = Index{Physics,TimeTag}
const DSTime = Index{Physics,DSTimeTag}

# const int4value = IntValue(:intvalue, 1, 4)
# const float16value = FloatValue(:floatvalue, 1, 16)

# Layouts

const layout_Tmin = Layout([IntValue(:intvalue, 1, 32) => SIMD(:simd, 1, 32)])
const layout_Tmax = Layout([IntValue(:intvalue, 1, 32) => SIMD(:simd, 1, 32)])
const layout_T̄min = Layout([IntValue(:intvalue, 1, 32) => SIMD(:simd, 1, 32)])
const layout_T̄max = Layout([IntValue(:intvalue, 1, 32) => SIMD(:simd, 1, 32)])

const layout_E_memory = Layout([
    IntValue(:intvalue, 1, 4) => SIMD(:simd, 1, 4),
    Cplx(:cplx, 1, C) => SIMD(:simd, 4, 2),
    Dish(:dish, 1, 4) => SIMD(:simd, 8, 4),
    Dish(:dish, 4, idiv(D, 4)) => Memory(:memory, 1, idiv(D, 4)),
    Polr(:polr, 1, P) => Memory(:memory, idiv(D, 4), P),
    Freq(:freq, 1, F) => Memory(:memory, idiv(D, 4) * P, F),
    Time(:time, 1, T) => Memory(:memory, idiv(D, 4) * F * P, T),
])

# We have M * N ≥ D dishes here. The additional ("dummy") dishes are initialized to zero.
const layout_Smn_memory = Layout([
    IntValue(:intvalue, 1, 16) => SIMD(:simd, 1, 16),
    MN(:mn, 1, 2) => SIMD(:simd, 16, 2),
    Dish(:dish, 1, M * N) => Memory(:memory, 1, M * N),
])

const layout_W_memory = Layout([
    FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
    Cplx(:cplx, 1, C) => SIMD(:simd, 16, 2),
    # TODO: Improve layout (index by dishes instead of dish grid)
    DishM(:dishM, 1, M) => Memory(:memory, 1, M),
    # DishMLo(:dishMLo, 1, idiv(M, 4)) => Memory(:memory, 1, idiv(M, 4)),
    # DishMHi(:dishMHi, 1, 4) => Memory(:memory, idiv(M, 4), 4),
    DishNLo(:dishNLo, 1, idiv(N, 4)) => Memory(:memory, M, idiv(N, 4)),
    DishNHi(:dishNHi, 1, 4) => Memory(:memory, M * idiv(N, 4), 4),
    Polr(:polr, 1, P) => Memory(:memory, M * N, P),
    Freq(:freq, 1, F) => Memory(:memory, M * N * P, F),
])

# I layout

const layout_I_memory = Layout([
    FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
    BeamP(:beamP, 1, 2) => SIMD(:simd, 16, 2),
    BeamP(:beamP, 2, M) => Memory(:memory, 1, M),
    BeamQ(:beamQ, 1, 2 * N) => Memory(:memory, M, 2 * N),
    Freq(:freq, 1, F) => Memory(:memory, M * 2 * N, F),
    DSTime(:dstime, 1, T̄) => Memory(:memory, M * 2 * N * F, T̄),
])

# info layout

const layout_info_memory = Layout([
    IntValue(:intvalue, 1, 32) => SIMD(:simd, 1, 32),
    Index{Physics,ThreadTag}(:thread, 1, num_threads) => Memory(:memory, 1, num_threads),
    Index{Physics,WarpTag}(:warp, 1, num_warps) => Memory(:memory, num_threads, num_warps),
    Index{Physics,BlockTag}(:block, 1, num_blocks) => Memory(:memory, num_threads * num_warps, num_blocks),
])

# Section 4.5, eqns. (54)+, (61)+
const layout_Fsh1_shared = Layout([
    IntValue(:intvalue, 1, 4) => SIMD(:simd, 1, 4),
    Cplx(:cplx, 1, C) => SIMD(:simd, 16, 2),
    Dish(:dish, 1, 8) => Shared(:shared, 32, 8),
    Dish(:dish, 8, idiv(D, 8)) => Shared(:shared, ΣF1, idiv(D, 8)),
    Freq(:freq, 1, F) => Block(:block, 1, F),
    Polr(:polr, 1, P) => SIMD(:simd, 4, 2),
    Time(:time, 1, idiv(Touter, 2)) => Shared(:shared, 1, idiv(Touter, 2)),
    Time(:time, idiv(Touter, 2), 2) => SIMD(:simd, 8, 2),
    Time(:time, Touter, fld(T, Touter)) => Loop(:t_outer, Touter, fld(T, Touter)),
])

# This `Fsh2` layout is used for writing.
# The dish locations are handled manually.
const layout_Fsh2_gridding_shared = Layout([
    IntValue(:intvalue, 1, 4) => SIMD(:simd, 1, 4),
    Cplx(:cplx, 1, C) => SIMD(:simd, 16, 2),
    Freq(:freq, 1, F) => Block(:block, 1, F),
    Polr(:polr, 1, P) => SIMD(:simd, 4, 2),
    Time(:time, 1, idiv(Touter, 2)) => Shared(:shared, 1, idiv(Touter, 2)),
    Time(:time, idiv(Touter, 2), 2) => SIMD(:simd, 8, 2),
    Time(:time, Touter, fld(T, Touter)) => Loop(:t_outer, Touter, fld(T, Touter)),
])

# Section 4.5, eqns. (67)+
# This `Fsh2` layout is used for reading
const layout_Fsh2_shared = Layout([
    IntValue(:intvalue, 1, 4) => SIMD(:simd, 1, 4),
    Cplx(:cplx, 1, C) => SIMD(:simd, 16, 2),
    DishM(:dishM, 1, M) => Shared(:shared, 33, M),
    # DishMLo(:dishMLo, 1, idiv(M, 4)) => Shared(:shared, 33, idiv(M, 4)),
    # DishMHi(:dishMHi, 1, 4) => Shared(:shared, 33 * idiv(M, 4), 4),
    # DishN(:dishN, 1, N) => Shared(:shared, ΣF2, N),
    DishNLo(:dishNLo, 1, idiv(N, 4)) => Shared(:shared, ΣF2, idiv(N, 4)),
    DishNHi(:dishNHi, 1, 4) => Shared(:shared, ΣF2 * idiv(N, 4), 4),
    Freq(:freq, 1, F) => Block(:block, 1, F),
    Polr(:polr, 1, P) => SIMD(:simd, 4, 2),
    Time(:time, 1, idiv(Touter, 2)) => Shared(:shared, 1, idiv(Touter, 2)),
    Time(:time, idiv(Touter, 2), 2) => SIMD(:simd, 8, 2),
    Time(:time, Touter, fld(T, Touter)) => Loop(:t_outer, Touter, fld(T, Touter)),
])

# Section 4.10, eqn. (76)
# This layout is used for writing `G`
const layout_Gsh_fft1_shared = Layout([
    FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
    Cplx(:cplx, 1, C) => SIMD(:simd, 16, 2),
    DishM(:dishM, 1, M) => Shared(:shared, 1, M),
    # DishMLo(:dishMLo, 1, idiv(M, 4)) => Shared(:shared, 1, idiv(M, 4)),
    # DishMHi(:dishMHi, 1, 4) => Shared(:shared, idiv(M, 4), 4),
    BeamQ(:beamQ, 1, 2) => Shared(:shared, ΣG0, 2),
    (
        if Npad == 32
            [
                BeamQ(:beamQ, 2, 2) => Shared(:shared, ΣG1 * 16, 2),
                BeamQ(:beamQ, 4, 2) => Shared(:shared, ΣG1 * 8, 2),
                BeamQ(:beamQ, 8, 2) => Shared(:shared, ΣG1 * 4, 2),
                BeamQ(:beamQ, 16, 2) => Shared(:shared, ΣG1 * 2, 2),
                BeamQ(:beamQ, 32, 2) => Shared(:shared, ΣG1 * 1, 2),
            ]
        elseif Npad == 16
            [
                BeamQ(:beamQ, 2, 2) => Shared(:shared, ΣG1 * 8, 2),
                BeamQ(:beamQ, 4, 2) => Shared(:shared, ΣG1 * 4, 2),
                BeamQ(:beamQ, 8, 2) => Shared(:shared, ΣG1 * 2, 2),
                BeamQ(:beamQ, 16, 2) => Shared(:shared, ΣG1 * 1, 2),
            ]
        elseif Npad == 8
            [
                BeamQ(:beamQ, 2, 2) => Shared(:shared, ΣG1 * 4, 2),
                BeamQ(:beamQ, 4, 2) => Shared(:shared, ΣG1 * 2, 2),
                BeamQ(:beamQ, 8, 2) => Shared(:shared, ΣG1 * 1, 2),
            ]
        else
            @assert false
        end
    )...,
    Freq(:freq, 1, F) => Block(:block, 1, F),
    Polr(:polr, 1, P) => Shared(:shared, Mpad, 2),
    Time(:time, 1, Tinner) => Shared(:shared, Mpad * 2, Tinner),
    # Time(:time, Tinner, idiv(Touter, Tinner)) => Loop(:t_inner, Tinner, idiv(Touter, Tinner)),
    Time(:time, Tinner, idiv(Touter, 2 * Tinner)) => Loop(:t_inner_lo, Tinner, idiv(Touter, 2 * Tinner)),
    Time(:time, idiv(Touter, 2), 2) => UnrolledLoop(:t_inner_hi, idiv(Touter, 2), 2),
    Time(:time, Touter, fld(T, Touter)) => Loop(:t_outer, Touter, fld(T, Touter)),
])
# This layout is used for reading `G`
const layout_Gsh_fft2_shared = let
    layout = copy(layout_Gsh_fft1_shared)
    @assert layout[DishM(:dishM, 1, M)] == Shared(:shared, 1, M)
    delete!(layout, DishM(:dishM, 1, M))
    layout[DishMLo(:dishMLo, 1, idiv(M, 4))] = Shared(:shared, 1, idiv(M, 4))
    layout[DishMHi(:dishMHi, 1, 4)] = Shared(:shared, idiv(M, 4), 4)
    layout
end

# info layout

const layout_info_registers = Layout([
    IntValue(:intvalue, 1, 32) => SIMD(:simd, 1, 32),
    Index{Physics,ThreadTag}(:thread, 1, num_threads) => Thread(:thread, 1, num_threads),
    Index{Physics,WarpTag}(:warp, 1, num_warps) => Warp(:warp, 1, num_warps),
    Index{Physics,BlockTag}(:block, 1, num_blocks) => Block(:block, 1, num_blocks),
])

# Machine indices

# const simd = SIMD(:simd, 1, num_simd_bits)
# const thread = Thread(:thread, 1, num_threads)
# const warp = Warp(:warp, 1, num_warps)
# const block = Block(:block, 1, num_blocks)
# const shared = Shared(:shared, 1, 99 * 1024)
# const memory = Memory(:memory, 1, 2^32)

const shmem_size = max(Fsh1_shmem_size, Fsh2_shmem_size, Gsh_shmem_size)
const shmem_bytes = 4 * shmem_size

const kernel_setup = KernelSetup(num_threads, num_warps, num_blocks, num_blocks_per_sm, shmem_bytes)

# Generate Code

# Copying global memory to shared memory (Fsh1) (section 4.6)
function copy_global_memory_to_Fsh1!(emitter)
    if setup === :chord || setup === :hirax

        # Eqn. (88)
        @assert D % 256 == 0
        @assert Touter % 2 == 0
        @assert (idiv(D, 256) * idiv(Touter, 2)) % W == 0
        # Eqn. (90)
        layout_E_registers = Layout([
            IntValue(:intvalue, 1, 4) => SIMD(:simd, 1, 4),
            Cplx(:cplx, 1, C) => SIMD(:simd, 4, 2),
            Dish(:dish, 1, 4) => SIMD(:simd, 8, 4),
            Dish(:dish, 4, 4) => Register(:dish, 4, 4),
            Dish(:dish, 16, 16) => Thread(:thread, 1, 16),
            Dish(:dish, 256, idiv(D, 256)) => Register(:dish, 256, idiv(D, 256)),
            Freq(:freq, 1, F) => Block(:block, 1, F),
            Polr(:polr, 1, P) => Thread(:thread, 16, 2),
            # Time(:time, 1, idiv(Touter, 2)) => Warp(:warp, 1, W),
            # Time(:time, idiv(Touter, 2), 2) => Register(:time, idiv(Touter, 2), 2),
            Time(:time, 1, W) => Warp(:warp, 1, W),
            Time(:time, W, idiv(Touter, W)) => Register(:time, W, idiv(Touter, W)),
            Time(:time, Touter, fld(T, Touter)) => Loop(:t_outer, Touter, fld(T, Touter)),
        ])
        load!(
            emitter,
            :E => layout_E_registers,
            :E_memory => layout_E_memory;
            align=16,
            postprocess=addr -> :(
                let
                    offset = $(Int32(idiv(D, 4) * P * F)) * Tmin
                    length = $(Int32(idiv(D, 4) * P * F * T))
                    mod($addr + offset, length)
                end
            ),
        )
        # Swap polr0, dish3
        permute!(emitter, :E, :E, Polr(:polr, 1, P), Dish(:dish, 8, 2))
        # E -> F shuffle
        # 1. swap polr0, cplx0
        # 2. swap timehi, dish0
        # 3. swap cplx0, dish1
        permute!(emitter, :E, :E, Polr(:polr, 1, P), Cplx(:cplx, 1, C))
        permute!(emitter, :E, :E, Time(:time, idiv(Touter, 2), 2), Dish(:dish, 1, 2))
        permute!(emitter, :E, :E, Cplx(:cplx, 1, C), Dish(:dish, 2, 2))
        store!(emitter, :Fsh1_shared => layout_Fsh1_shared, :E)

    elseif setup === :pathfinder

        # Eqn. (83)
        @assert D == 64
        @assert Touter % 8 == 0
        @assert idiv(Touter, 8) % W == 0
        # Eqn. (85)
        layout_E_registers = Layout([
            IntValue(:intvalue, 1, 4) => SIMD(:simd, 1, 4),
            Cplx(:cplx, 1, C) => SIMD(:simd, 4, 2),
            Dish(:dish, 1, 4) => SIMD(:simd, 8, 4),
            Dish(:dish, 4, 4) => Register(:dish, 4, 4),
            Dish(:dish, 16, 4) => Thread(:thread, 1, 4),
            Freq(:freq, 1, F) => Block(:block, 1, F),
            Polr(:polr, 1, P) => Thread(:thread, 4, 2),
            Time(:time, 1, 4) => Thread(:thread, 8, 4),
            Time(:time, 4, idiv(Touter, 8)) => Warp(:warp, 1, W),
            Time(:time, idiv(Touter, 2), 2) => Register(:time, idiv(Touter, 2), 2),
            Time(:time, Touter, fld(T, Touter)) => Loop(:t_outer, Touter, fld(T, Touter)),
        ])
        load!(
            emitter,
            :E => layout_E_registers,
            :E_memory => layout_E_memory;
            align=16,
            postprocess=addr -> :(
                let
                    offset = $(Int32(idiv(D, 4) * P * F)) * Tmin
                    length = $(Int32(idiv(D, 4) * P * F * T))
                    mod($addr + offset, length)
                end
            ),
        )
        # Swap polr0, dish3
        permute!(emitter, :E, :E, Polr(:polr, 1, P), Dish(:dish, 8, 2))
        # E -> F shuffle
        # 1. swap polr0, cplx0
        # 2. swap timehi, dish0
        # 3. swap cplx0, dish1
        permute!(emitter, :E, :E, Polr(:polr, 1, P), Cplx(:cplx, 1, C))
        permute!(emitter, :E, :E, Time(:time, idiv(Touter, 2), 2), Dish(:dish, 1, 2))
        permute!(emitter, :E, :E, Cplx(:cplx, 1, C), Dish(:dish, 2, 2))
        store!(emitter, :Fsh1_shared => layout_Fsh1_shared, :E)

    else
        @assert false
    end

    return nothing
end

# Reading shared memory (Fsh1) into registers (Freg1) (section 4.7)
function read_Fsh1!(emitter)
    layout_Freg1_registers = Layout([
        IntValue(:intvalue, 1, 4) => SIMD(:simd, 1, 4),
        Cplx(:cplx, 1, C) => SIMD(:simd, 16, 2),
        Dish(:dish, 1, W) => Warp(:warp, 1, W),
        Dish(:dish, W, RF1) => Register(:dish, W, RF1),
        Freq(:freq, 1, F) => Block(:block, 1, F),
        Polr(:polr, 1, 2) => SIMD(:simd, 4, 2),
        Time(:time, 1, idiv(Touter, 2)) => Thread(:thread, 1, idiv(Touter, 2)),
        Time(:time, idiv(Touter, 2), 2) => SIMD(:simd, 8, 2),
        Time(:time, Touter, fld(T, Touter)) => Loop(:t_outer, Touter, fld(T, Touter)),
    ])
    # This loads garbage for threadidx ≥ idiv(Touter, 2)
    load!(emitter, :Freg1 => layout_Freg1_registers, :Fsh1_shared => layout_Fsh1_shared)
    return nothing
end

# Writing shared memory (Fsh2) from registers (Freg1) (section 4.8)
function write_Fsh2!(emitter)
    # TODO: Skip writing dishes > D?
    # TODO: Skip writing garbage (threadidx ≥ idiv(Touter, 2))?
    broadcast!(emitter, :sd, :S, Register(:sd, 1, idiv(M * N, W)) => Thread(:thread, 1, idiv(M * N, W)))
    for i in 0:(idiv(M * N, W) - 1)
        layout = copy(emitter.environment[:Freg1])
        delete!(layout, Dish(:dish, W, RF1))
        emitter.environment[:Freg1′] = layout
        real_dish_value = Symbol(:Freg1_dish, string(W * i))
        zero_dish_value = zero(Int4x8)
        if i < D ÷ W
            # This is a real dish for all warps
            dish_value = real_dish_value
        elseif i < (D + W - 1) ÷ W
            # This is a real dish for some warps, and a dummy dish for other warps
            is_real_dish = quote
                warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, $num_warps)
                dish = warp + $(Int32(W)) * $(Int32(i))
                dish < $(Int32(D))
            end
            dish_value = :($is_real_dish ? $real_dish_value : $zero_dish_value)
        else
            # This is a dummy dish for all warps
            dish_value = zero_dish_value
        end
        push!(emitter.statements, :(Freg1′ = $dish_value))
        # push!(emitter.statements, :(@assert $(Symbol(:sd_sd, "$i")) ≠ 999999999i32))
        store!(emitter, :Fsh2_shared => layout_Fsh2_gridding_shared, :Freg1′; offset=Symbol(:sd_sd, "$i"))
    end

    return nothing
end

# Reading shared memory (Fsh2) into registers (Freg2) (section 4.9)
function read_Fsh2!(emitter)
    # Section 4.9, eqn. (99)+
    @assert Mw * Tw == W
    ν = trailing_zeros(Npad)
    νm = max(0, 5 - ν)
    νn = max(0, ν - 3)
    @assert νm + νn == 2
    @assert 2 * (1 << νn) == idiv(Npad, 4)
    # DishM
    @assert idiv(M, 4) ≤ Mw
    layout_Freg2_registers = Layout([
        IntValue(:intvalue, 1, 4) => SIMD(:simd, 1, 4),
        Cplx(:cplx, 1, C) => SIMD(:simd, 16, 2),
        DishM(:dishM, 1, Mt) => Thread(:thread, 8, Mt),
        DishM(:dishM, Mt, Mw) => Warp(:warp, 1, Mw),
        DishM(:dishM, Mt * Mw, Mr) => Register(:dishM, Mt * Mw, Mr),
        DishNLo(:dishNLo, 1, 2) => Thread(:thread, 4, 2),
        # Threads idiv(N,8) .. idiv(Npad,8) are padding
        DishNLo(:dishNLo, 2, idiv(Npad, 8)) => Thread(:thread, 8 * (1 << νm), idiv(Npad, 8)),
        DishNHi(:dishNHi, 1, 4) => Thread(:thread, 1, 4),
        Freq(:freq, 1, F) => Block(:block, 1, F),
        Polr(:polr, 1, P) => SIMD(:simd, 4, 2),
        Time(:time, 1, Tw) => Warp(:warp, Mw, Tw),
        Time(:time, Tw, idiv(Touter, 2 * Tw)) => Register(:time, Tw, Tr),
        Time(:time, idiv(Touter, 2), 2) => SIMD(:simd, 8, 2),
        Time(:time, Touter, fld(T, Touter)) => Loop(:t_outer, Touter, fld(T, Touter)),
    ])
    # This loads garbage for nlo ≥ idiv(N, 4)
    apply!(emitter, :Freg2 => layout_Freg2_registers, :(zero(Int4x8)))
    if!(
        emitter, :(
            let
                thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, $num_threads)
                nlo = 1i32 * (thread ÷ 4i32 % 2i32) + 2i32 * (thread ÷ $(Int32(8 * (1 << νm))) % $(Int32(idiv(Npad, 8))))
                nlo < $(Int32(idiv(N, 4)))
            end
        )
    ) do emitter
        load!(emitter, :Freg2 => layout_Freg2_registers, :Fsh2_shared => layout_Fsh2_shared)
        return nothing
    end
    return nothing
end

function setup_fft_coefficients!(
    emitter, Γ¹::Symbol, Γ²::Symbol, Γ³::Symbol, N::Int, mkDishLo, mkDishHi, mkBeam, mkSpectatorIn, mkSpectator
)
    # Section 3.3

    # TODO: Ensure we're converting warp bits into spectator bits

    # (39): high dish bits, low beam bits, no spectators needed
    Npad = nextpow(2, N)
    r = ilog2(Npad)
    @assert 3 ≤ r ≤ 5
    layout_Γ¹_registers = Layout([
        FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
        Cplx(:cplx_in, 1, C) => SIMD(:simd, 16, C),
        # Cplx(:cplx, 1, C) => Register(:cplx, 1, C),
        mkDishHi(1, 4) => Thread(:thread, 1, 4),
        mkBeam(1, 8) => Thread(:thread, 4, 8),
    ])
    push!(
        emitter.statements,
        quote
            (Γ¹_re_re, Γ¹_re_im, Γ¹_im_re, Γ¹_im_im) = let
                # Γ¹ is the mma A matrix
                # rows: i [beamp0, beamp1, beamp2, cplx0] [thread2, thread3, thread4, reg0]
                # cols: j [cplx0, mhi0, mhi1]             [simd4, thread0, thread1]
                #
                # Γ¹ = exp(π * im * c * v / 4)   (28)
                thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, $num_threads)
                c = thread % 4i32
                v = thread ÷ 4i32
                Γ¹ = cispi(((c * v) % 8i32 / 4.0f0) % 2.0f0)
                (+Γ¹.re, -Γ¹.im, +Γ¹.im, +Γ¹.re)
            end
        end,
    )
    apply!(emitter, :Γ¹_re => layout_Γ¹_registers, :(Float16x2(Γ¹_re_im, Γ¹_re_re)))
    apply!(emitter, :Γ¹_im => layout_Γ¹_registers, :(Float16x2(Γ¹_im_im, Γ¹_im_re)))
    merge!(emitter, Γ¹, [:Γ¹_re, :Γ¹_im], Cplx(:cplx, 1, C) => Register(:cplx, 1, C))

    # (41)
    # @assert trailing_zeros(Npad) == 5
    N4 = Int32(idiv(N, 4))
    N4pad = Int32(nextpow(2, N4))
    N4pad2 = Int32(idiv(N4pad, 2))
    layout_Γ²_registers = Layout([
        FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
        mkDishLo(1, 2) => SIMD(:simd, 16, 2),
        mkDishLo(2, N4pad2) => Thread(:thread, idiv(4, N4pad2), N4pad2),
        mkBeam(1, 8) => Thread(:thread, 4, 8),
        mkSpectator(1, idiv(4, N4pad2)) => Thread(:thread, 1, idiv(4, N4pad2)),
    ])
    push!(
        emitter.statements,
        quote
            (Γ²_d0_re, Γ²_d0_im, Γ²_d1_re, Γ²_d1_im) = let
                # Γ² = exp(π * im * d * v / N)   (29)
                thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, $num_threads)
                d0 = (thread % $N4pad2) * 2i32 + 0i32
                d1 = (thread % $N4pad2) * 2i32 + 1i32
                v = thread ÷ 4i32
                δ0 = δ1 = Γ²_d0 = d0 < $N4 ? cispi(((d0 * v) % $(Int32(2 * N)) / $(Float32(N))) % 2.0f0) : Complex(0.0f0)
                Γ²_d1 = d1 < $N4 ? cispi(((d1 * v) % $(Int32(2 * N)) / $(Float32(N))) % 2.0f0) : Complex(0.0f0)
                (Γ²_d0.re, Γ²_d0.im, Γ²_d1.re, Γ²_d1.im)
            end
        end,
    )
    apply!(emitter, :Γ²_re => layout_Γ²_registers, :(Float16x2(Γ²_d0_re, Γ²_d1_re)))
    apply!(emitter, :Γ²_im => layout_Γ²_registers, :(Float16x2(Γ²_d0_im, Γ²_d1_im)))
    merge!(emitter, Γ², [:Γ²_re, :Γ²_im], Cplx(:cplx, 1, C) => Register(:cplx, 1, C))

    # (45) - (47): low dish bits, high beam bits, with spectator bits
    if trailing_zeros(Npad) == 5
        layout_Γ³_registers = Layout([
            FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
            # Cplx(:cplx_in, 1, C) => Register(:cplx_in, 1, C),
            # Cplx(:cplx, 1, C) => Register(:cplx, 1, C),
            mkDishLo(1, 2) => SIMD(:simd, 16, 2),
            mkDishLo(2, 4) => Thread(:thread, 1, 4),
            mkBeam(8, 8) => Thread(:thread, 4, 8),
        ])
        push!(
            emitter.statements,
            quote
                (Γ³_d0_re_re, Γ³_d0_re_im, Γ³_d0_im_re, Γ³_d0_im_im, Γ³_d1_re_re, Γ³_d1_re_im, Γ³_d1_im_re, Γ³_d1_im_im) = let
                    # Γ³ is the mma A matrix
                    # rows: i [beamp3, beamp4, beamp5, cplx0] [thread2, thread3, thread4, reg0]
                    # cols: j [mlo0, mlo1, mlo2, cplx0]       [simd4, thread0, thread1, reg1]
                    #
                    # Γ³ = exp(8π * im * d * u / N)   (30)
                    thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, $num_threads)
                    d0 = (thread % 4i32) * 2i32 + 0i32
                    d1 = (thread % 4i32) * 2i32 + 1i32
                    u = (thread ÷ 4i32) % 8i32
                    Γ³_d0 = d0 < $N4 && u < $N4 ? cispi(((d0 * u) % $N4 / $(Float32(N / 8))) % 2.0f0) : Complex(0.0f0)
                    Γ³_d1 = d1 < $N4 && u < $N4 ? cispi(((d1 * u) % $N4 / $(Float32(N / 8))) % 2.0f0) : Complex(0.0f0)
                    (+Γ³_d0.re, -Γ³_d0.im, +Γ³_d0.im, +Γ³_d0.re, +Γ³_d1.re, -Γ³_d1.im, +Γ³_d1.im, +Γ³_d1.re)
                end
            end,
        )
        apply!(emitter, :Γ³_re_re => layout_Γ³_registers, :(Float16x2(Γ³_d0_re_re, Γ³_d1_re_re)))
        apply!(emitter, :Γ³_re_im => layout_Γ³_registers, :(Float16x2(Γ³_d0_re_im, Γ³_d1_re_im)))
        apply!(emitter, :Γ³_im_re => layout_Γ³_registers, :(Float16x2(Γ³_d0_im_re, Γ³_d1_im_re)))
        apply!(emitter, :Γ³_im_im => layout_Γ³_registers, :(Float16x2(Γ³_d0_im_im, Γ³_d1_im_im)))
        merge!(emitter, :Γ³_re, [:Γ³_re_re, :Γ³_re_im], Cplx(:cplx_in, 1, C) => Register(:cplx_in, 1, 2))
        merge!(emitter, :Γ³_im, [:Γ³_im_re, :Γ³_im_im], Cplx(:cplx_in, 1, C) => Register(:cplx_in, 1, 2))
        merge!(emitter, Γ³, [:Γ³_re, :Γ³_im], Cplx(:cplx, 1, C) => Register(:cplx, 1, 2))
    elseif trailing_zeros(Npad) == 4
        layout_Γ³_registers = Layout([
            FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
            # Cplx(:cplx, 1, C) => Register(:cplx, 1, C),
            mkDishLo(1, 2) => SIMD(:simd, 16, 2),
            mkDishLo(2, 2) => Thread(:thread, 1, 2),
            Cplx(:cplx_in, 1, C) => Thread(:thread, 2, 2),
            mkBeam(8, 4) => Thread(:thread, 4, 4),
            mkSpectator(1, 2) => Thread(:thread, 16, 2),
        ])
        push!(
            emitter.statements,
            quote
                (Γ³_d0_re, Γ³_d0_im, Γ³_d1_re, Γ³_d1_im) = let
                    # Γ³ is the mma A matrix
                    # rows: i [beamp3, beamp4, beamp5, cplx0] [thread2, thread3, thread4, reg0]
                    # cols: j [mlo0, mlo1, mlo2, cplx0]       [simd4, thread0, thread1, reg1]
                    #
                    # Γ³ = exp(8π * im * d * u / N)   (30)
                    thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, $num_threads)
                    d0 = (thread % 2i32) * 2i32 + 0i32
                    d1 = (thread % 2i32) * 2i32 + 1i32
                    u = (thread ÷ 4i32) % 4i32
                    Γ³_d0 = d0 < $N4 && u < $N4 ? cispi(((d0 * u) % $N4 / $(Float32(N / 8))) % 2.0f0) : Complex(0.0f0)
                    Γ³_d1 = d1 < $N4 && u < $N4 ? cispi(((d1 * u) % $N4 / $(Float32(N / 8))) % 2.0f0) : Complex(0.0f0)
                    reim = (thread ÷ 2i32) % 2i32
                    reim == 0i32 ? (+Γ³_d0.re, +Γ³_d0.im, +Γ³_d1.re, +Γ³_d1.im) : (-Γ³_d0.im, +Γ³_d0.re, -Γ³_d1.im, +Γ³_d1.re)
                end
            end,
        )
        apply!(emitter, :Γ³_re => layout_Γ³_registers, :(Float16x2(Γ³_d0_re, Γ³_d1_re)))
        apply!(emitter, :Γ³_im => layout_Γ³_registers, :(Float16x2(Γ³_d0_im, Γ³_d1_im)))
        merge!(emitter, Γ³, [:Γ³_re, :Γ³_im], Cplx(:cplx, 1, C) => Register(:cplx, 1, 2))
    elseif trailing_zeros(Npad) == 3
        layout_Γ³_registers = Layout([
            FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
            # Cplx(:cplx, 1, C) => Register(:cplx, 1, C),
            mkDishLo(1, 2) => SIMD(:simd, 16, 2),
            mkSpectatorIn(2, 2) => Thread(:thread, 1, 2),
            Cplx(:cplx_in, 1, C) => Thread(:thread, 2, 2),
            mkBeam(8, 2) => Thread(:thread, 4, 2),
            mkSpectator(1, 2) => Thread(:thread, 8, 2),
            mkSpectator(2, 2) => Thread(:thread, 16, 2),
        ])
        push!(
            emitter.statements,
            quote
                (Γ³_d0_re, Γ³_d0_im, Γ³_d1_re, Γ³_d1_im) = let
                    # Γ³ is the mma A matrix
                    # rows: i [beamp3, beamp4, beamp5, cplx0] [thread2, thread3, thread4, reg0]
                    # cols: j [mlo0, mlo1, mlo2, cplx0]       [simd4, thread0, thread1, reg1]
                    #
                    # Γ³ = exp(8π * im * d * u / N)   (30)
                    thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, $num_threads)
                    d0 = 0i32
                    d1 = 1i32
                    u = (thread ÷ 4i32) % 2i32
                    s1in = (thread ÷ 1i32) % 2i32
                    s1out = (thread ÷ 16i32) % 2i32
                    δ = s1in == s1out
                    Γ³_d0 = d0 < $N4 && u < $N4 && δ ? cispi(((d0 * u) % $N4 / $(Float32(N / 8))) % 2.0f0) : Complex(0.0f0)
                    Γ³_d1 = d1 < $N4 && u < $N4 && δ ? cispi(((d1 * u) % $N4 / $(Float32(N / 8))) % 2.0f0) : Complex(0.0f0)
                    reim = (thread ÷ 2i32) % 2i32
                    reim == 0i32 ? (+Γ³_d0.re, +Γ³_d0.im, +Γ³_d1.re, +Γ³_d1.im) : (-Γ³_d0.im, +Γ³_d0.re, -Γ³_d1.im, +Γ³_d1.re)
                end
            end,
        )
        apply!(emitter, :Γ³_re => layout_Γ³_registers, :(Float16x2(Γ³_d0_re, Γ³_d1_re)))
        apply!(emitter, :Γ³_im => layout_Γ³_registers, :(Float16x2(Γ³_d0_im, Γ³_d1_im)))
        merge!(emitter, Γ³, [:Γ³_re, :Γ³_im], Cplx(:cplx, 1, C) => Register(:cplx, 1, 2))
    else
        @assert false
    end

    return nothing
end

# First FFT (section 4.10)
function do_first_fft!(emitter)
    @assert Tinner % Tw == 0

    # # Here we widen all Freg2 values, but we will use only some of them in the `select!` call.
    # widen2!(
    #     emitter,
    #     :Freg2,
    #     :Freg2,
    #     SIMD(:simd, 4, 2) => Register(:polr, 1, P),
    #     SIMD(:simd, 8, 2) => Register(:time, idiv(Touter, 2), 2);
    #     newtype=FloatValue,
    # )
    # 
    # select!(emitter, :E, :Freg2, Register(:time, Tinner, idiv(Touter, Tinner)) => Loop(:t_inner, Tinner, idiv(Touter, Tinner)))

    select!(
        emitter,
        :Freg2′,
        :Freg2,
        Register(:time, Tinner, idiv(Touter, 2 * Tinner)) => Loop(:t_inner_lo, Tinner, idiv(Touter, 2 * Tinner)),
    )

    widen2!(
        emitter,
        :E′,
        :Freg2′,
        SIMD(:simd, 4, 2) => Register(:polr, 1, P),
        SIMD(:simd, 8, 2) => Register(:time, idiv(Touter, 2), 2);
        newtype=FloatValue,
    )

    select!(emitter, :E, :E′, Register(:time, idiv(Touter, 2), 2) => UnrolledLoop(:t_inner_hi, idiv(Touter, 2), 2))

    apply!(emitter, :WE, [:E, :W], (E, W) -> :(complex_mul($W, $E)))

    # Chapter 4.10 notation:
    #     G_mq = FT WE_mn
    # This is an FT that transforms n to q, with m as spectator index.

    # Chapter 3.3 notation:
    #
    #     n = (N/4) c + d      0 ≤ n < N     (21)
    #     q = 8 u + v          0 ≤ q < 2 N   (22)
    #
    #     Y_uvs = FT X_cds
    #
    #     Z_dvs = Γ¹_cv X_cds
    #     W_dvs = Γ²_dv Z_dvs
    #     Y_uvs = Γ³_du W_dvs

    # Register assignments:
    #     r = trailing_zeros(Npad)

    # Constants:
    #     Γ¹ = exp(π * im * c * v / 4)    (28)
    #     Γ² = exp(π * im * d * v / N)    (29)
    #     Γ³ = exp(8π * im * d * u / N)   (30)

    # Appendix C.1 notation:
    #
    #     C_ik = A_ij B_jk
    #     i:16
    #     j:8
    #     k:8
    #
    #     Z_dvs = Γ¹_cv X_cds
    #     C_ik  = A_ij  B_jk
    #
    #     i => v
    #     j => c
    #     k => ds

    apply!(emitter, :X, [:WE], (WE,) -> :($WE))

    # First step
    let
        # (37)
        layout_X_registers = Layout([
            FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
            Cplx(:cplx, 1, C) => SIMD(:simd, 16, 2),
            DishM(:dishM, 1, Mt) => Thread(:thread, 8, Mt),
            DishM(:dishM, Mt, Mw) => Warp(:warp, 1, Mw),
            DishM(:dishM, Mt * Mw, Mr) => Register(:dishM, Mt * Mw, Mr),
            DishNLo(:dishNLo, 1, 2) => Thread(:thread, 4, 2),
            DishNLo(:dishNLo, 2, idiv(Npad, 8)) => Thread(:thread, 8 * Mt, idiv(Npad, 8)),
            DishNHi(:dishNHi, 1, 4) => Thread(:thread, 1, 4),
            Polr(:polr, 1, P) => Register(:polr, 1, P),
            Freq(:freq, 1, F) => Block(:block, 1, F),
            Time(:time, 1, Tw) => Warp(:warp, Mw, Tw),
            Time(:time, Tw, idiv(Tinner, Tw)) => Register(:time, Tw, idiv(Tinner, Tw)),
            Time(:time, Tinner, idiv(Touter, 2 * Tinner)) => Loop(:t_inner_lo, Tinner, idiv(Touter, 2 * Tinner)),
            Time(:time, idiv(Touter, 2), 2) => UnrolledLoop(:t_inner_hi, idiv(Touter, 2), 2),
            Time(:time, Touter, fld(T, Touter)) => Loop(:t_outer, Touter, fld(T, Touter)),
        ])
        @assert emitter.environment[:X] == layout_X_registers

        # (40)
        ν = trailing_zeros(Npad)
        νm = max(0, 5 - ν)
        νn = max(0, ν - 3)
        @assert νm + νn == 2
        @assert 2 * (1 << νn) == idiv(Npad, 4)
        layout_Z_registers = Layout([
            FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
            Cplx(:cplx, 1, C) => Register(:cplx, 1, 2),
            DishM(:dishM, 1, Mt) => Thread(:thread, 1, Mt),
            DishM(:dishM, Mt, Mw) => Warp(:warp, 1, Mw),
            DishM(:dishM, Mt * Mw, idiv(M, Mt * Mw)) => Register(:dishM, Mt * Mw, Mr),
            DishNLo(:dishNLo, 1, 2) => SIMD(:simd, 16, 2),
            DishNLo(:dishNLo, 2, idiv(Npad, 8)) => Thread(:thread, Mt, idiv(Npad, 8)),
            BeamQ(:beamQ, 1, 8) => Thread(:thread, 4, 8),
            Freq(:freq, 1, F) => Block(:block, 1, F),
            Polr(:polr, 1, P) => Register(:polr, 1, P),
            Time(:time, 1, Tw) => Warp(:warp, Mw, Tw),
            # Time(:time, Tw, idiv(Touter, 2 * Tw)) => Register(:time, Tw, Tr),
            # Time(:time, idiv(Touter, 2), 2) => Register(:time, idiv(Touter, 2), 2),
            Time(:time, Tw, idiv(Tinner, Tw)) => Register(:time, Tw, idiv(Tinner, Tw)),
            # Time(:time, Tinner, idiv(Touter, Tinner)) => Loop(:t_inner, Tinner, idiv(Touter, Tinner)),
            Time(:time, Tinner, idiv(Touter, 2 * Tinner)) => Loop(:t_inner_lo, Tinner, idiv(Touter, 2 * Tinner)),
            Time(:time, idiv(Touter, 2), 2) => Loop(:t_inner_hi, idiv(Touter, 2), 2),
            Time(:time, Touter, fld(T, Touter)) => Loop(:t_outer, Touter, fld(T, Touter)),
        ])
        apply!(emitter, :Z => layout_Z_registers, :(zero(Float16x2)))

        # (38)
        let
            #TODO: Implement this cleanly, e.g. via a `rename!` function
            layout = copy(emitter.environment[:X])
            k = Cplx(:cplx, 1, C)
            k′ = Cplx(:cplx_in, 1, C)
            v = layout[k]
            delete!(layout, k)
            layout[k′] = v
            emitter.environment[:X] = layout
        end
        mma_is = [BeamQ(:beamQ, 1, 2), BeamQ(:beamQ, 2, 2), BeamQ(:beamQ, 4, 2), Cplx(:cplx, 1, 2)]
        mma_js = [Cplx(:cplx_in, 1, 2), DishNHi(:dishNHi, 1, 2), DishNHi(:dishNHi, 2, 2)]
        if trailing_zeros(Npad) == 5
            mma_ks = [DishNLo(:dishNLo, 1, 2), DishNLo(:dishNLo, 2, 2), DishNLo(:dishNLo, 4, 2)]
        elseif trailing_zeros(Npad) == 4
            mma_ks = [DishNLo(:dishNLo, 1, 2), DishM(:dishM, 1, 2), DishNLo(:dishNLo, 2, 2)]
        elseif trailing_zeros(Npad) == 3
            mma_ks = [DishNLo(:dishNLo, 1, 2), DishM(:dishM, 1, 2), DishM(:dishM, 2, 2)]
        else
            @assert false
        end
        mma_row_col_m16n8k8_f16!(emitter, :Z, :aΓ¹ => (mma_is, mma_js), :X => (mma_js, mma_ks), :Z => (mma_is, mma_ks))
    end

    # Second step
    # Section 3 `W` is called `V` here
    split!(emitter, [:aΓ²re, :aΓ²im], :aΓ², Register(:cplx, 1, 2))
    split!(emitter, [:Zre, :Zim], :Z, Register(:cplx, 1, 2))
    # TODO: Find a better set of conditions
    if trailing_zeros(Npad) == 5 || setup === :hirax
        apply!(emitter, :Vre, [:Zre, :Zim, :aΓ²re, :aΓ²im], (Zre, Zim, aΓ²re, aΓ²im) -> :(muladd($aΓ²re, $Zre, -$aΓ²im * $Zim)))
        apply!(emitter, :Vim, [:Zre, :Zim, :aΓ²re, :aΓ²im], (Zre, Zim, aΓ²re, aΓ²im) -> :(muladd($aΓ²re, $Zim, +$aΓ²im * $Zre)))
    elseif trailing_zeros(Npad) == 4
        # We can't handle partial index ranges in symbols. If there is
        # a `dishM` spectator index then the whole `dishM` index range
        # needs to be present. This should be corrected in
        # `IndexSpaces.apply!`.
        merge!(emitter, :aΓ²reM, [:aΓ²re, :aΓ²re], DishM(:dishM, 4, 2) => Register(:dishM, 4, 2))
        merge!(emitter, :aΓ²imM, [:aΓ²im, :aΓ²im], DishM(:dishM, 4, 2) => Register(:dishM, 4, 2))
        apply!(emitter, :Vre, [:Zre, :Zim, :aΓ²reM, :aΓ²imM], (Zre, Zim, aΓ²re, aΓ²im) -> :(muladd($aΓ²re, $Zre, -$aΓ²im * $Zim)))
        apply!(emitter, :Vim, [:Zre, :Zim, :aΓ²reM, :aΓ²imM], (Zre, Zim, aΓ²re, aΓ²im) -> :(muladd($aΓ²re, $Zim, +$aΓ²im * $Zre)))
    else
        @assert false
    end
    merge!(emitter, :V, [:Vre, :Vim], Cplx(:cplx, 1, 2) => Register(:cplx, 1, 2))

    # Third step
    let
        # (32)
        if trailing_zeros(Npad) == 5
            layout_V_registers = Layout([
                FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
                DishNLo(:dishNLo, 1, 2) => SIMD(:simd, 16, 2),
                Cplx(:cplx, 1, C) => Register(:cplx, 1, 2),
                DishNLo(:dishNLo, 2, 4) => Thread(:thread, 1, 4),
                BeamQ(:beamQ, 1, 8) => Thread(:thread, 4, 8),
                DishM(:dishM, Mt, Mw) => Warp(:warp, 1, Mw),
                DishM(:dishM, Mt * Mw, Mr) => Register(:dishM, Mt * Mw, Mr),
                Polr(:polr, 1, P) => Register(:polr, 1, P),
                Freq(:freq, 1, F) => Block(:block, 1, F),
                Time(:time, 1, Tw) => Warp(:warp, idiv(M, 4 * Mt) * Mw, Tw),
                Time(:time, Tw, idiv(Tinner, Tw)) => Register(:time, Tw, idiv(Tinner, Tw)),
                Time(:time, Tinner, idiv(Touter, 2 * Tinner)) => Loop(:t_inner_lo, Tinner, idiv(Touter, 2 * Tinner)),
                Time(:time, idiv(Touter, 2), 2) => Loop(:t_inner_hi, idiv(Touter, 2), 2),
                Time(:time, Touter, fld(T, Touter)) => Loop(:t_outer, Touter, fld(T, Touter)),
            ])
        elseif trailing_zeros(Npad) == 4
            layout_V_registers = Layout([
                FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
                DishNLo(:dishNLo, 1, 2) => SIMD(:simd, 16, 2),
                Cplx(:cplx, 1, C) => Register(:cplx, 1, 2),
                DishM(:dishM, 1, 2) => Thread(:thread, 1, 2),
                DishNLo(:dishNLo, 2, 2) => Thread(:thread, 2, 2),
                BeamQ(:beamQ, 1, 8) => Thread(:thread, 4, 8),
                DishM(:dishM, Mt, Mw) => Warp(:warp, 1, Mw),
                DishM(:dishM, Mt * Mw, Mr) => Register(:dishM, Mt * Mw, Mr),
                Polr(:polr, 1, P) => Register(:polr, 1, P),
                Freq(:freq, 1, F) => Block(:block, 1, F),
                Time(:time, 1, Tw) => Warp(:warp, Mw, Tw),
                Time(:time, Tw, idiv(Tinner, Tw)) => Register(:time, Tw, idiv(Tinner, Tw)),
                Time(:time, Tinner, idiv(Touter, 2 * Tinner)) => Loop(:t_inner_lo, Tinner, idiv(Touter, 2 * Tinner)),
                Time(:time, idiv(Touter, 2), 2) => Loop(:t_inner_hi, idiv(Touter, 2), 2),
                Time(:time, Touter, fld(T, Touter)) => Loop(:t_outer, Touter, fld(T, Touter)),
            ])
        else
            @assert false
        end
        @assert emitter.environment[:V] == layout_V_registers

        # (44)
        ν = trailing_zeros(Npad)
        νm = max(0, 5 - ν)
        νn = max(0, ν - 3)
        @assert νm + νn == 2
        @assert 2 * (1 << νn) == idiv(Npad, 4)
        layout_Y_registers = Layout([
            FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
            Cplx(:cplx, 1, C) => Register(:cplx, 1, 2),
            DishM(:dishM, 1, Mt) => Thread(:thread, idiv(32, Mt), Mt),
            DishM(:dishM, Mt, Mw) => Warp(:warp, 1, Mw),
            DishM(:dishM, Mt * Mw, Mr) => Register(:dishM, Mt * Mw, Mr),
            BeamQ(:beamQ, 1, 2) => SIMD(:simd, 16, 2),
            BeamQ(:beamQ, 2, 4) => Thread(:thread, 1, 4),
            BeamQ(:beamQ, 8, 1 << (νn + 1)) => Thread(:thread, 4, 1 << (νn + 1)),
            Freq(:freq, 1, F) => Block(:block, 1, F),
            Polr(:polr, 1, P) => Register(:polr, 1, P),
            Time(:time, 1, Tw) => Warp(:warp, Mw, Tw),
            # Time(:time, Tw, idiv(Touter, 2 * Tw)) => Register(:time, Tw, Tr),
            # Time(:time, idiv(Touter, 2), 2) => Register(:time, idiv(Touter, 2), 2),
            Time(:time, Tw, idiv(Tinner, Tw)) => Register(:time, Tw, idiv(Tinner, Tw)),
            # Time(:time, Tinner, idiv(Touter, Tinner)) => Loop(:t_inner, Tinner, idiv(Touter, Tinner)),
            Time(:time, Tinner, idiv(Touter, 2 * Tinner)) => Loop(:t_inner_lo, Tinner, idiv(Touter, 2 * Tinner)),
            Time(:time, idiv(Touter, 2), 2) => Loop(:t_inner_hi, idiv(Touter, 2), 2),
            Time(:time, Touter, fld(T, Touter)) => Loop(:t_outer, Touter, fld(T, Touter)),
        ])
        apply!(emitter, :Y => layout_Y_registers, :(zero(Float16x2)))

        # (43)
        if trailing_zeros(Npad) == 5
            mma_is = [BeamQ(:beamQ, 8, 2), BeamQ(:beamQ, 16, 2), BeamQ(:beamQ, 32, 2), Cplx(:cplx, 1, 2)]
            mma_js = [DishNLo(:dishNLo, 1, 2), DishNLo(:dishNLo, 2, 2), DishNLo(:dishNLo, 4, 2), Cplx(:cplx_in, 1, 2)]
            spectator = nothing
        elseif trailing_zeros(Npad) == 4
            mma_is = [BeamQ(:beamQ, 8, 2), BeamQ(:beamQ, 16, 2), DishM(:dishM, 1, 2), Cplx(:cplx, 1, 2)]
            mma_js = [DishNLo(:dishNLo, 1, 2), DishM(:dishM, 1, 2), DishNLo(:dishNLo, 2, 2), Cplx(:cplx_in, 1, 2)]
            spectator = DishM(:dishM, 1, 2)
        elseif trailing_zeros(Npad) == 3
            mma_is = [BeamQ(:beamQ, 8, 2), DishM(:dishM, 1, 2), DishM(:dishM, 2, 2), Cplx(:cplx, 1, 2)]
            mma_js = [DishNLo(:dishNLo, 1, 2), DishM(:dishM, 1, 2), DishM(:dishM_in, 2, 2), Cplx(:cplx_in, 1, 2)]
            spectator = DishM(:dishM, 1, 2)
        else
            @assert false
        end
        mma_ks = [BeamQ(:beamQ, 1, 2), BeamQ(:beamQ, 2, 2), BeamQ(:beamQ, 4, 2)]
        #TODO: Implement this cleaner, e.g. via a `rename!` function
        split!(emitter, [:Vre, :Vim], :V, Cplx(:cplx, 1, 2))
        merge!(emitter, :V, [:Vre, :Vim], Cplx(:cplx_in, 1, 2) => Register(:cplx_in, 1, 2))
        let
            #TODO: Implement this cleanly, e.g. via a `rename!` function
            layout = copy(emitter.environment[:V])
            for bit in 0:(4 - trailing_zeros(Npad))
                k = DishM(:dishM, 1 << bit, 2)
                if k ≠ spectator && k ∈ layout
                    k′ = DishM(:dishM_in, 1 << bit, 2)
                    v = layout[k]
                    delete!(layout, k)
                    layout[k′] = v
                end
            end
            emitter.environment[:V] = layout
        end
        if spectator === nothing
            # dense mma
            mma_row_col_m16n8k16_f16!(emitter, :Y, :aΓ³ => (mma_is, mma_js), :V => (mma_js, mma_ks), :Y => (mma_is, mma_ks))
        else
            # TODO: Find a better set of conditions
            if setup === :pathfinder
                # We can't handle partial index ranges in symbols. If there is
                # a `dishM` spectator index then the whole `dishM` index range
                # needs to be present. This should be corrected in
                # `IndexSpaces.apply!`.
                merge!(emitter, :aΓ³M, [:aΓ³, :aΓ³], DishM(:dishM, 4, 2) => Register(:dishM, 4, 2))
                # sparse mma
                mma_sp_row_col_m16n8k16_f16!(
                    emitter, :Y, :aΓ³M => (mma_is, mma_js), :V => (mma_js, mma_ks), :Y => (mma_is, mma_ks), spectator
                )
            else
                # sparse mma
                mma_sp_row_col_m16n8k16_f16!(
                    emitter, :Y, :aΓ³ => (mma_is, mma_js), :V => (mma_js, mma_ks), :Y => (mma_is, mma_ks), spectator
                )
            end
        end
    end

    apply!(emitter, :G, [:Y], (Y,) -> :($Y))

    # Write G to shared memory
    permute!(emitter, :G, :G, Register(:cplx, 1, 2), SIMD(:simd, 16, 2))
    # This writes superfluous data for 2*N ≤ beamQ < 2*Npad
    store!(emitter, :Gsh_shared => layout_Gsh_fft1_shared, :G)
    # if!(emitter, :(
    #     let
    #         thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, $num_threads)
    #         beamq = 2i32 * thread
    #         beamq < $(Int32(2 * N))
    #     end
    # )) do emitter
    #     store!(emitter, :Gsh_shared => layout_Gsh_fft1_shared, :G)
    #     nothing
    # end

    return nothing
end

# Second FFT (section 4.11)
function do_second_fft!(emitter)
    # Read one G-tile from shared memory
    # (123)
    if trailing_zeros(Mpad) == 5
        layout_G_registers = Layout([
            FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
            Cplx(:cplx, 1, C) => SIMD(:simd, 16, 2),
            DishMHi(:dishMHi, 1, 4) => Thread(:thread, 1, 4),
            DishMLo(:dishMLo, 1, 8) => Thread(:thread, 4, 8),
            # Is this an efficient warp layout (with reversed warp bits)?
            # This simplifies shared memory addressing; see `layout_Gsh_fft2_shared`.
            # TODO: Re-introduce this.
            # BeamQ(:beamQ, 2, 2) => Warp(:warp, 16, 2),
            # BeamQ(:beamQ, 4, 2) => Warp(:warp, 8, 2),
            # BeamQ(:beamQ, 8, 2) => Warp(:warp, 4, 2),
            # BeamQ(:beamQ, 16, 2) => Warp(:warp, 2, 2),
            # BeamQ(:beamQ, 32, 2) => Warp(:warp, 1, 2),
            BeamQ(:beamQ, 1, N) => Warp(:warp, 1, N),
            BeamQ(:beamQ, N, 2) => Register(:beamQ, N, 2),
            Freq(:freq, 1, F) => Block(:block, 1, F),
            # Polr(:polr, 1, P) => Loop(:polr, 1, P),
            Polr(:polr, 1, P) => Register(:polr, 1, P),
            Time(:time, 1, Tinner) => UnrolledLoop(:t, 1, Tinner),
            # Time(:time, Tinner, idiv(Touter, Tinner)) => Loop(:t_inner, Tinner, idiv(Touter, Tinner)),
            Time(:time, Tinner, idiv(Touter, 2 * Tinner)) => Loop(:t_inner_lo, Tinner, idiv(Touter, 2 * Tinner)),
            Time(:time, idiv(Touter, 2), 2) => UnrolledLoop(:t_inner_hi, idiv(Touter, 2), 2),
            Time(:time, Touter, fld(T, Touter)) => Loop(:t_outer, Touter, fld(T, Touter)),
        ])
    elseif trailing_zeros(Mpad) == 4
        layout_G_registers = Layout([
            FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
            Cplx(:cplx, 1, C) => SIMD(:simd, 16, 2),
            DishMHi(:dishMHi, 1, 4) => Thread(:thread, 1, 4),
            DishMLo(:dishMLo, 1, 2) => Thread(:thread, 4, 2),
            BeamQ(:beamQ, 1, 2) => Thread(:thread, 8, 2),
            DishMLo(:dishMLo, 2, 2) => Thread(:thread, 16, 2),
            # Is this an efficient warp layout (with reversed warp bits)?
            # This simplifies shared memory addressing; see `layout_Gsh_fft2_shared`.
            # TODO: Re-introduce this.
            # BeamQ(:beamQ, 2, 2) => Warp(:warp, 16, 2),
            # BeamQ(:beamQ, 4, 2) => Warp(:warp, 8, 2),
            # BeamQ(:beamQ, 8, 2) => Warp(:warp, 4, 2),
            # BeamQ(:beamQ, 16, 2) => Warp(:warp, 2, 2),
            # BeamQ(:beamQ, 32, 2) => Warp(:warp, 1, 2),
            BeamQ(:beamQ, 2, N) => Warp(:warp, 1, N),
            Freq(:freq, 1, F) => Block(:block, 1, F),
            # Polr(:polr, 1, P) => Loop(:polr, 1, P),
            Polr(:polr, 1, P) => Register(:polr, 1, P),
            Time(:time, 1, Tinner) => UnrolledLoop(:t, 1, Tinner),
            # Time(:time, Tinner, idiv(Touter, Tinner)) => Loop(:t_inner, Tinner, idiv(Touter, Tinner)),
            Time(:time, Tinner, idiv(Touter, 2 * Tinner)) => Loop(:t_inner_lo, Tinner, idiv(Touter, 2 * Tinner)),
            Time(:time, idiv(Touter, 2), 2) => UnrolledLoop(:t_inner_hi, idiv(Touter, 2), 2),
            Time(:time, Touter, fld(T, Touter)) => Loop(:t_outer, Touter, fld(T, Touter)),
        ])
    elseif trailing_zeros(Mpad) == 3
        layout_G_registers = Layout([
            FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
            Cplx(:cplx, 1, C) => SIMD(:simd, 16, 2),
            DishMHi(:dishMHi, 1, 4) => Thread(:thread, 1, 4),
            DishMLo(:dishMLo, 1, 2) => Thread(:thread, 4, 2),
            BeamQ(:beamQ, 1, 4) => Thread(:thread, 8, 4),
            # Is this an efficient warp layout (with reversed warp bits)?
            # This simplifies shared memory addressing; see `layout_Gsh_fft2_shared`.
            # TODO: Re-introduce this.
            # BeamQ(:beamQ, 2, 2) => Warp(:warp, 16, 2),
            # BeamQ(:beamQ, 4, 2) => Warp(:warp, 8, 2),
            # BeamQ(:beamQ, 8, 2) => Warp(:warp, 4, 2),
            # BeamQ(:beamQ, 16, 2) => Warp(:warp, 2, 2),
            # BeamQ(:beamQ, 32, 2) => Warp(:warp, 1, 2),
            BeamQ(:beamQ, 4, idiv(N, 2)) => Warp(:warp, 1, idiv(N, 2)),
            Freq(:freq, 1, F) => Block(:block, 1, F),
            # Polr(:polr, 1, P) => Loop(:polr, 1, P),
            Polr(:polr, 1, P) => Register(:polr, 1, P),
            Time(:time, 1, Tinner) => UnrolledLoop(:t, 1, Tinner),
            # Time(:time, Tinner, idiv(Touter, Tinner)) => Loop(:t_inner, Tinner, idiv(Touter, Tinner)),
            Time(:time, Tinner, idiv(Touter, 2 * Tinner)) => Loop(:t_inner_lo, Tinner, idiv(Touter, 2 * Tinner)),
            Time(:time, idiv(Touter, 2), 2) => UnrolledLoop(:t_inner_hi, idiv(Touter, 2), 2),
            Time(:time, Touter, fld(T, Touter)) => Loop(:t_outer, Touter, fld(T, Touter)),
        ])
    else
        @assert false
    end
    # This loads garbage for idiv(M, 4) ≤ mlo < idiv(Mpad, 4)
    apply!(emitter, :G => layout_G_registers, :(zero(Float16x2)))
    μ = trailing_zeros(Mpad)
    if!(
        emitter,
        if Mpad == M
            :true
        else
            quote
                let
                    thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, $num_threads)
                    mlo2_offset = $(Int32(1 << (8 - μ)))
                    mlo2_length = $(Int32(1 << (μ - 3)))
                    mlo = 1i32 * (thread ÷ 4i32 % 2i32) + 2i32 * (thread ÷ mlo2_offset % mlo2_length)
                    mlo < $(Int32(idiv(M, 4)))
                end
            end
        end,
    ) do emitter
        load!(emitter, :G => layout_G_registers, :Gsh_shared => layout_Gsh_fft2_shared)
        return nothing
    end

    # Chapter 4.11 notation:
    #     Ẽ_pq = FT G_mq
    # This is an FT that transforms m to p, with q as spectator index.

    # # (39)
    # layout_Γ¹_registers = Layout(
    #     Dict(
    #         FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
    #         Cplx(:cplx_in, 1, C) => SIMD(:simd, 16, 2), # mma j0
    #         Cplx(:cplx, 1, C) => Register(:cplx, 1, 2), # mma i3
    #         DishMHi(:dishMHi, 1, 4) => Thread(:thread, 1, 4), # mma j1, j2
    #         BeamP(:beamP, 1, 8) => Thread(:thread, 4, 8),
    #     ),
    # )
    # emitter.environment[:Γ¹] = layout_Γ¹_registers
    # 
    # # (41)
    # @assert trailing_zeros(Mpad) == 5
    # layout_Γ²_registers = Layout(
    #     Dict(
    #         FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
    #         Cplx(:cplx, 1, C) => Register(:cplx, 1, 2),
    #         DishMLo(:dishMLo, 1, 2) => SIMD(:simd, 16, 2),
    #         DishMLo(:dishMLo, 2, 4) => Thread(:thread, 1, 4),
    #         BeamP(:beamP, 1, 8) => Thread(:thread, 4, 8),
    #     ),
    # )
    # emitter.environment[:Γ²] = layout_Γ²_registers
    # 
    # # (45) - (47)
    # @assert trailing_zeros(Mpad) == 5
    # layout_Γ³_registers = Layout(
    #     Dict(
    #         FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
    #         Cplx(:cplx_in, 1, C) => Register(:cplx_in, 1, 2),
    #         Cplx(:cplx, 1, C) => Register(:cplx, 1, 2),
    #         DishMLo(:dishMLo, 1, 2) => SIMD(:simd, 16, 2),
    #         DishMLo(:dishMLo, 2, 4) => Thread(:thread, 1, 4),
    #         BeamP(:beamP, 8, 8) => Thread(:thread, 4, 8),
    #     ),
    # )
    # emitter.environment[:Γ³] = layout_Γ³_registers

    apply!(emitter, :X, [:G], (G,) -> :($G))

    Qt = idiv(32, Mpad)         # BeamQ threads
    Qw = gcd(idiv(2N, Qt), W)   # BeamQ warps
    Qr = idiv(2N, Qt * Qw)      # BeamQ registers

    # First step
    let
        # (37)
        layout_X_registers = Layout([
            FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
            Cplx(:cplx, 1, C) => SIMD(:simd, 16, 2),
            BeamQ(:beamQ, 1, Qt) => Thread(:thread, 8, Qt),
            BeamQ(:beamQ, Qt, Qw) => Warp(:warp, 1, Qw),
            BeamQ(:beamQ, Qt * Qw, Qr) => Register(:beamQ, Qt * Qw, Qr),
            DishMLo(:dishMLo, 1, 2) => Thread(:thread, 4, 2),
            DishMLo(:dishMLo, 2, idiv(Mpad, 8)) => Thread(:thread, 8 * Mt, idiv(Mpad, 8)),
            DishMHi(:dishMHi, 1, 4) => Thread(:thread, 1, 4),
            Polr(:polr, 1, P) => Register(:polr, 1, P),
            Freq(:freq, 1, F) => Block(:block, 1, F),
            Time(:time, 1, Tinner) => UnrolledLoop(:t, 1, Tinner),
            Time(:time, Tinner, idiv(Touter, 2 * Tinner)) => Loop(:t_inner_lo, Tinner, idiv(Touter, 2 * Tinner)),
            Time(:time, idiv(Touter, 2), 2) => UnrolledLoop(:t_inner_hi, idiv(Touter, 2), 2),
            Time(:time, Touter, fld(T, Touter)) => Loop(:t_outer, Touter, fld(T, Touter)),
        ])
        @assert emitter.environment[:X] == layout_X_registers

        # (40)
        @assert trailing_zeros(Mpad) ≥ 3
        layout_Z_registers = Layout([
            FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
            Cplx(:cplx, 1, C) => Register(:cplx, 1, 2),
            DishMLo(:dishMLo, 1, 2) => SIMD(:simd, 16, 2),
            DishMLo(:dishMLo, 2, idiv(Mpad, 8)) => Thread(:thread, Qt, idiv(Mpad, 8)),
            BeamP(:beamP, 1, 8) => Thread(:thread, 4, 8),
            BeamQ(:beamQ, 1, Qt) => Thread(:thread, 1, Qt),
            BeamQ(:beamQ, Qt, Qw) => Warp(:warp, 1, Qw),
            BeamQ(:beamQ, Qt * Qw, Qr) => Register(:beamQ, Qt * Qw, Qr),
            Freq(:freq, 1, F) => Block(:block, 1, F),
            # Polr(:polr, 1, P) => Loop(:polr, 1, P),
            Polr(:polr, 1, P) => Register(:polr, 1, P),
            Time(:time, 1, Tinner) => UnrolledLoop(:t, 1, Tinner),
            # Time(:time, Tinner, idiv(Touter, Tinner)) => Loop(:t_inner, Tinner, idiv(Touter, Tinner)),
            Time(:time, Tinner, idiv(Touter, 2 * Tinner)) => Loop(:t_inner_lo, Tinner, idiv(Touter, 2 * Tinner)),
            Time(:time, idiv(Touter, 2), 2) => Loop(:t_inner_hi, idiv(Touter, 2), 2),
            Time(:time, Touter, fld(T, Touter)) => Loop(:t_outer, Touter, fld(T, Touter)),
        ])
        apply!(emitter, :Z => layout_Z_registers, :(zero(Float16x2)))

        # (38)
        let
            #TODO: Implement this cleanly, e.g. via a `rename!` function
            layout = copy(emitter.environment[:X])
            k = Cplx(:cplx, 1, C)
            k′ = Cplx(:cplx_in, 1, C)
            v = layout[k]
            delete!(layout, k)
            layout[k′] = v
            emitter.environment[:X] = layout
        end
        mma_is = [BeamP(:beamP, 1, 2), BeamP(:beamP, 2, 2), BeamP(:beamP, 4, 2), Cplx(:cplx, 1, 2)]
        mma_js = [Cplx(:cplx_in, 1, 2), DishMHi(:dishMHi, 1, 2), DishMHi(:dishMHi, 2, 2)]
        if trailing_zeros(Mpad) == 5
            mma_ks = [DishMLo(:dishMLo, 1, 2), DishMLo(:dishMLo, 2, 2), DishMLo(:dishMLo, 4, 2)]
        elseif trailing_zeros(Mpad) == 4
            mma_ks = [DishMLo(:dishMLo, 1, 2), BeamQ(:beamQ, 1, 2), DishMLo(:dishMLo, 2, 2)]
        elseif trailing_zeros(Mpad) == 3
            mma_ks = [DishMLo(:dishMLo, 1, 2), BeamQ(:beamQ, 1, 2), BeamQ(:beamQ, 2, 2)]
        else
            @assert false
        end
        mma_row_col_m16n8k8_f16!(emitter, :Z, :bΓ¹ => (mma_is, mma_js), :X => (mma_js, mma_ks), :Z => (mma_is, mma_ks))
    end

    # Second step
    # Section 3 `W` is called `V` here
    split!(emitter, [:bΓ²re, :bΓ²im], :bΓ², Register(:cplx, 1, 2))
    split!(emitter, [:Zre, :Zim], :Z, Register(:cplx, 1, 2))
    apply!(emitter, :Vre, [:Zre, :Zim, :bΓ²re, :bΓ²im], (Zre, Zim, bΓ²re, bΓ²im) -> :(muladd($bΓ²re, $Zre, -$bΓ²im * $Zim)))
    apply!(emitter, :Vim, [:Zre, :Zim, :bΓ²re, :bΓ²im], (Zre, Zim, bΓ²re, bΓ²im) -> :(muladd($bΓ²re, $Zim, +$bΓ²im * $Zre)))
    merge!(emitter, :V, [:Vre, :Vim], Cplx(:cplx, 1, 2) => Register(:cplx, 1, 2))

    # Third step
    let
        # (44)
        μ = trailing_zeros(Mpad)
        μm = max(0, 5 - μ)
        μn = max(0, μ - 3)
        @assert μm + μn == 2
        @assert 2 * (1 << μn) == idiv(Mpad, 4)
        if trailing_zeros(Mpad) == 5
            layout_Y_registers = Layout([
                FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
                Cplx(:cplx, 1, C) => Register(:cplx, 1, 2),
                BeamP(:beamP, 1, 2) => SIMD(:simd, 16, 2),
                BeamP(:beamP, 2, 4) => Thread(:thread, 1, 4),
                BeamP(:beamP, 8, 8) => Thread(:thread, 4, 8),
                BeamQ(:beamQ, 1, W) => Warp(:warp, 1, W),
                BeamQ(:beamQ, W, idiv(2N, W)) => Register(:beamQ, W, idiv(2N, W)),
                Freq(:freq, 1, F) => Block(:block, 1, F),
                Polr(:polr, 1, P) => Register(:polr, 1, P),
                Time(:time, 1, Tinner) => UnrolledLoop(:t, 1, Tinner),
                Time(:time, Tinner, idiv(Touter, 2 * Tinner)) => Loop(:t_inner_lo, Tinner, idiv(Touter, 2 * Tinner)),
                Time(:time, idiv(Touter, 2), 2) => Loop(:t_inner_hi, idiv(Touter, 2), 2),
                Time(:time, Touter, fld(T, Touter)) => Loop(:t_outer, Touter, fld(T, Touter)),
            ])
        elseif trailing_zeros(Mpad) == 4
            layout_Y_registers = Layout([
                FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
                Cplx(:cplx, 1, C) => Register(:cplx, 1, 2),
                BeamP(:beamP, 1, 2) => SIMD(:simd, 16, 2),
                BeamP(:beamP, 2, 4) => Thread(:thread, 1, 4),
                BeamP(:beamP, 8, 4) => Thread(:thread, 4, 4),
                BeamQ(:beamQ, 1, 2) => Thread(:thread, 16, 2),
                BeamQ(:beamQ, 2, W) => Warp(:warp, 1, W),
                BeamQ(:beamQ, 2W, idiv(2N, 2W)) => Register(:beamQ, 2W, idiv(2N, 2W)),
                Freq(:freq, 1, F) => Block(:block, 1, F),
                Polr(:polr, 1, P) => Register(:polr, 1, P),
                Time(:time, 1, Tinner) => UnrolledLoop(:t, 1, Tinner),
                Time(:time, Tinner, idiv(Touter, 2 * Tinner)) => Loop(:t_inner_lo, Tinner, idiv(Touter, 2 * Tinner)),
                Time(:time, idiv(Touter, 2), 2) => Loop(:t_inner_hi, idiv(Touter, 2), 2),
                Time(:time, Touter, fld(T, Touter)) => Loop(:t_outer, Touter, fld(T, Touter)),
            ])
        elseif trailing_zeros(Mpad) == 3
            layout_Y_registers = Layout([
                FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
                Cplx(:cplx, 1, C) => Register(:cplx, 1, 2),
                BeamP(:beamP, 1, 2) => SIMD(:simd, 16, 2),
                BeamP(:beamP, 2, 4) => Thread(:thread, 1, 4),
                BeamP(:beamP, 8, 2) => Thread(:thread, 4, 2),
                BeamQ(:beamQ, 1, 4) => Thread(:thread, 8, 4),
                BeamQ(:beamQ, 4, W) => Warp(:warp, 1, W),
                BeamQ(:beamQ, 4W, idiv(2N, 4W)) => Register(:beamQ, 4W, idiv(2N, 4W)),
                Freq(:freq, 1, F) => Block(:block, 1, F),
                Polr(:polr, 1, P) => Register(:polr, 1, P),
                Time(:time, 1, Tinner) => UnrolledLoop(:t, 1, Tinner),
                Time(:time, Tinner, idiv(Touter, 2 * Tinner)) => Loop(:t_inner_lo, Tinner, idiv(Touter, 2 * Tinner)),
                Time(:time, idiv(Touter, 2), 2) => Loop(:t_inner_hi, idiv(Touter, 2), 2),
                Time(:time, Touter, fld(T, Touter)) => Loop(:t_outer, Touter, fld(T, Touter)),
            ])
        else
            @assert false
        end
        apply!(emitter, :Y => layout_Y_registers, :(zero(Float16x2)))

        # (43)
        if trailing_zeros(Mpad) == 5
            mma_is = [BeamP(:beamP, 8, 2), BeamP(:beamP, 16, 2), BeamP(:beamP, 32, 2), Cplx(:cplx, 1, 2)]
            mma_js = [DishMLo(:dishMLo, 1, 2), DishMLo(:dishMLo, 2, 2), DishMLo(:dishMLo, 4, 2), Cplx(:cplx_in, 1, 2)]
            spectator = nothing
        elseif trailing_zeros(Mpad) == 4
            mma_is = [BeamP(:beamP, 8, 2), BeamP(:beamP, 16, 2), BeamQ(:beamQ, 1, 2), Cplx(:cplx, 1, 2)]
            mma_js = [DishMLo(:dishMLo, 1, 2), BeamQ(:beamQ, 1, 2), DishMLo(:dishMLo, 2, 2), Cplx(:cplx_in, 1, 2)]
            spectator = BeamQ(:beamQ, 1, 2)
        elseif trailing_zeros(Mpad) == 3
            mma_is = [BeamP(:beamP, 8, 2), BeamQ(:beamQ, 1, 2), BeamQ(:beamQ, 2, 2), Cplx(:cplx, 1, 2)]
            mma_js = [DishMLo(:dishMLo, 1, 2), BeamQ(:beamQ, 1, 2), BeamQ(:beamQ_in, 2, 2), Cplx(:cplx_in, 1, 2)]
            spectator = BeamQ(:beamQ, 1, 2)
        else
            @assert false
        end
        mma_ks = [BeamP(:beamP, 1, 2), BeamP(:beamP, 2, 2), BeamP(:beamP, 4, 2)]
        #TODO: Implement this cleaner, e.g. via a `rename!` function
        split!(emitter, [:Vre, :Vim], :V, Cplx(:cplx, 1, 2))
        merge!(emitter, :V, [:Vre, :Vim], Cplx(:cplx_in, 1, 2) => Register(:cplx_in, 1, 2))
        let
            #TODO: Implement this cleanly, e.g. via a `rename!` function
            layout = copy(emitter.environment[:V])
            for bit in 0:(5 - trailing_zeros(2 * Mpad))
                k = BeamQ(:beamQ, 1 << bit, 2)
                if k ≠ spectator && k ∈ layout
                    k′ = BeamQ(:beamQ_in, 1 << bit, 2)
                    v = layout[k]
                    delete!(layout, k)
                    layout[k′] = v
                end
            end
            emitter.environment[:V] = layout
        end
        if spectator === nothing
            # dense mma
            mma_row_col_m16n8k16_f16!(emitter, :Y, :bΓ³ => (mma_is, mma_js), :V => (mma_js, mma_ks), :Y => (mma_is, mma_ks))
        else
            # sparse mma
            mma_sp_row_col_m16n8k16_f16!(
                emitter, :Y, :bΓ³ => (mma_is, mma_js), :V => (mma_js, mma_ks), :Y => (mma_is, mma_ks), spectator
            )
        end
    end

    apply!(emitter, :Ẽ, [:Y], (Y,) -> :($Y))

    split!(emitter, [:Ẽp0, :Ẽp1], :Ẽ, Polr(:polr, 1, P))
    split!(emitter, [:Ẽp0re, :Ẽp0im], :Ẽp0, Cplx(:cplx, 1, C))
    split!(emitter, [:Ẽp1re, :Ẽp1im], :Ẽp1, Cplx(:cplx, 1, C))

    apply!(
        emitter,
        :I,
        [:I, :Ẽp0re, :Ẽp0im, :Ẽp1re, :Ẽp1im],
        (I, Ẽp0re, Ẽp0im, Ẽp1re, Ẽp1im) -> :(
        # muladd($Ẽp1im, $Ẽp1im, muladd($Ẽp1re, $Ẽp1re, muladd($Ẽp0im, $Ẽp0im, muladd($Ẽp0re, $Ẽp0re, $I))))
        muladd(
            $(Float16x2(output_gain, output_gain)),
            muladd($Ẽp1im, $Ẽp1im, muladd($Ẽp1re, $Ẽp1re, muladd($Ẽp0im, $Ẽp0im, $Ẽp0re * $Ẽp0re))),
            $I,
        ));
        ignore=[Time(:time, 1, T)],
    )

    return nothing
end

function make_frb_kernel()
    emitter = Emitter(kernel_setup)

    # Generate code (section 4.3)

    apply!(emitter, :info => layout_info_registers, 1i32)
    store!(emitter, :info_memory => layout_info_memory, :info)

    # Read parameters `Tmin`, `Tmax`, `T̄min`, `T̄max`
    load!(emitter, :Tmin => layout_Tmin, :Tmin_memory => layout_Tmin)
    load!(emitter, :Tmax => layout_Tmax, :Tmax_memory => layout_Tmax)
    load!(emitter, :T̄min => layout_T̄min, :T̄min_memory => layout_T̄min)
    load!(emitter, :T̄max => layout_T̄max, :T̄max_memory => layout_T̄max)
    if!(
        emitter,
        :(
            !(
                0i32 ≤ Tmin ≤ Tmax ≤ $(Int32(2 * T)) &&
                (Tmax - Tmin) % $(Int32(Touter)) == 0i32 &&
                0i32 ≤ T̄min ≤ T̄max ≤ $(Int32(2 * T̄)) &&
                T̄max - T̄min == (Tmax - Tmin) ÷ $(Int32(Tds))
            )
        ),
    ) do emitter
        apply!(emitter, :info => layout_info_registers, 2i32)
        store!(emitter, :info_memory => layout_info_memory, :info)
        trap!(emitter)
        return nothing
    end

    setup_fft_coefficients!(
        emitter,
        :aΓ¹,
        :aΓ²,
        :aΓ³,
        N,
        (off, len) -> DishNLo(:dishNLo, off, len),
        (off, len) -> DishNHi(:dishNHi, off, len),
        (off, len) -> BeamQ(:beamQ, off, len),
        (off, len) -> DishM(:dishM_in, off, len),
        (off, len) -> DishM(:dishM, off, len),
    )
    setup_fft_coefficients!(
        emitter,
        :bΓ¹,
        :bΓ²,
        :bΓ³,
        M,
        (off, len) -> DishMLo(:dishMLo, off, len),
        (off, len) -> DishMHi(:dishMHi, off, len),
        (off, len) -> BeamP(:beamP, off, len),
        (off, len) -> BeamQ(:beamQ_in, off, len),
        (off, len) -> BeamQ(:beamQ, off, len),
    )

    let
        @assert (M * N) % W == 0    # eqn. (98)
        @assert (M * N) ÷ W ≤ 32    # eqn. (99)
        layout_S_registers = Layout([
            IntValue(:intvalue, 1, 32) => SIMD(:simd, 1, 32),
            Dish(:dish, 1, W) => Warp(:warp, 1, W),
            Dish(:dish, W, idiv(M * N, W)) => Thread(:thread, 1, idiv(M * N, W)),
        ])
        # This loads garbage for idiv(M * N, W) ≤ thread < 32
        apply!(emitter, :S => layout_S_registers, 999999999i32)
        @assert idiv(M * N, W) in [16, 24]
        @assert num_threads == 32
        if!(emitter, :(
            let
                thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, $num_threads)
                thread < $(Int32(idiv(M * N, W)))
            end
        )) do emitter
            layout_Smn_registers = Layout([
                IntValue(:intvalue, 1, 16) => SIMD(:simd, 1, 16),
                MN(:mn, 1, 2) => SIMD(:simd, 16, 2),
                Dish(:dish, 1, W) => Warp(:warp, 1, W),
                Dish(:dish, W, idiv(M * N, W)) => Thread(:thread, 1, idiv(M * N, W)),
            ])
            load!(emitter, :Smn => layout_Smn_registers, :Smn_memory => layout_Smn_memory)
            widen!(emitter, :Smn, :Smn, SIMD(:simd, 16, 2) => Register(:mn, 1, 2))
            split!(emitter, [:Sm, :Sn], :Smn, Register(:mn, 1, 2))
            apply!(emitter, :S, [:Sm, :Sn], (Sm, Sn) -> :(33i32 * Sm + $(Int32(ΣF2)) * Sn))
            return nothing
        end
    end

    let
        # This needs to be compatible with layout_Freg2_registers after widening
        @assert Mw * Tw == W
        ν = trailing_zeros(Npad)
        νm = max(0, 5 - ν)
        νn = max(0, ν - 3)
        @assert νm + νn == 2
        @assert 2 * (1 << νn) == idiv(Npad, 4)
        layout_W_registers = Layout([
            FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
            Cplx(:cplx, 1, C) => SIMD(:simd, 16, 2),
            DishM(:dishM, 1, Mt) => Thread(:thread, 8, Mt),
            DishM(:dishM, Mt, Mw) => Warp(:warp, 1, Mw),
            DishM(:dishM, Mt * Mw, Mr) => Register(:dishM, Mt * Mw, Mr),
            DishNLo(:dishNLo, 1, 2) => Thread(:thread, 4, 2),
            DishNLo(:dishNLo, 2, idiv(Npad, 8)) => Thread(:thread, 8 * (1 << νm), idiv(Npad, 8)),
            DishNHi(:dishNHi, 1, 4) => Thread(:thread, 1, 4),
            Freq(:freq, 1, F) => Block(:block, 1, F),
            Polr(:polr, 1, P) => Register(:polr, 1, P),
        ])
        # This loads garbage for idiv(N, 4) ≤ nlo ≤ idiv(Npad, 4)
        apply!(emitter, :W => layout_W_registers, :(zero(Float16x2)))
        if!(emitter, :(
            let
                thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, $num_threads)
                nlo = 2 * (thread ÷ $(Int32(8 * (1 << νm))))
                nlo < $(Int32(idiv(N, 4)))
            end
        )) do emitter
            load!(emitter, :W => layout_W_registers, :W_memory => layout_W_memory)
            return nothing
        end
    end

    let
        # Section 4.11, eqn. (128)
        Pt = Mpad
        Qt = idiv(32, Pt)
        Qw = W
        Qr = idiv(2 * N, Qt * Qw)
        @assert Qt * Qw * Qr == 2 * N
        @assert Pt * Qt == 32
        layout_I_registers = Layout([
            FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
            BeamP(:beamP, 1, 2) => SIMD(:simd, 16, 2),
            BeamP(:beamP, 2, Pt) => Thread(:thread, 1, Pt),
            BeamQ(:beamQ, 1, Qt) => Thread(:thread, Pt, Qt),
            BeamQ(:beamQ, Qt, Qw) => Warp(:warp, 1, W),
            BeamQ(:beamQ, Qt * Qw, Qr) => Register(:beamQ, Qt * Qw, Qr),
            Freq(:freq, 1, F) => Block(:block, 1, F),
            DSTime(:dstime, 1, T̄) => Loop(:dstime, 1, T̄),
        ])
        apply!(emitter, :I => layout_I_registers, :(zero(Float16x2)))
        push!(emitter.statements, :(dstime = $(0i32)))
        push!(emitter.statements, :(t_running = $(0i32)))
    end

    loop!(emitter, Time(:time, Touter, fld(T, Touter)) => Loop(:t_outer, Touter, fld(T, Touter))) do emitter
        push!(
            emitter.statements,
            quote
                Tmin + t_outer ≥ Tmax && break
            end,
        )

        block!(emitter) do emitter
            copy_global_memory_to_Fsh1!(emitter)
            sync_threads!(emitter)
            return nothing
        end

        block!(emitter) do emitter
            read_Fsh1!(emitter)
            sync_threads!(emitter)

            write_Fsh2!(emitter)
            sync_threads!(emitter)
            return nothing
        end

        block!(emitter) do emitter
            read_Fsh2!(emitter)
            sync_threads!(emitter)

            unrolled_loop!(emitter, Time(:time, idiv(Touter, 2), 2) => UnrolledLoop(:t_inner_hi, idiv(Touter, 2), 2)) do emitter
                # This loop should probably be unrolled for execution speed, but that increases compile time significantly
                # loop!(
                #     emitter, Time(:time, Tinner, idiv(Touter, Tinner)) => Loop(:t_inner, Tinner, idiv(Touter, Tinner))
                # ) do emitter
                loop!(
                    emitter, Time(:time, Tinner, idiv(Touter, 2 * Tinner)) => Loop(:t_inner_lo, Tinner, idiv(Touter, 2 * Tinner))
                ) do emitter

                    # 4.10 First FFT
                    # (111)
                    # unrolled_loop!(emitter, Time(:time, Tw, idiv(Tinner, Tw)) => Loop(:tau_tile, Tw, idiv(Tinner, Tw))) do emitter
                    #     unrolled_loop!(
                    #         emitter, DishM(:dishM, Mt * Mw, idiv(M, Mt * Mw)) => Loop(:dishM, Mt * Mw, idiv(M, Mt * Mw))
                    #     ) do emitter
                    #         unrolled_loop!(emitter, Polr(:polr, 1, P) => Loop(:polr, 1, P)) do emitter
                    #             do_fft1!(emitter)
                    # 
                    #             nothing
                    #         end
                    #         nothing
                    #     end
                    #     nothing
                    # end
                    do_first_fft!(emitter)
                    sync_threads!(emitter)

                    unrolled_loop!(emitter, Time(:time, 1, Tinner) => UnrolledLoop(:t, 1, Tinner)) do emitter

                        # 4.11 Second FFT
                        # loop!(emitter, Polr(:polr, 1, P) => Loop(:polr, 1, P)) do emitter
                        #     do_second_fft!(emitter)
                        #     nothing
                        # end
                        do_second_fft!(emitter)

                        push!(emitter.statements, :(t_running += $(1i32)))
                        # Skip this write-back for some of the `t` and `t_inner` iterations, depending on `% T_ds`
                        # (This condition will be evaluated at compile time)
                        if!(emitter, :((t_inner_hi + t + 1i32) % $(Int32(gcd(Tinner, Tds))) == 0i32)) do emitter
                            if!(emitter, :(t_running == $(Int32(Tds)))) do emitter
                                if!(
                                    emitter,
                                    :(
                                        let
                                            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, $num_threads)
                                            warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, $num_warps)
                                            p = 2i32 * thread
                                            q = 2i32 * warp
                                            0i32 ≤ p < $(Int32(2 * M)) && 0i32 ≤ q < $(Int32(2 * N))
                                        end
                                    ),
                                ) do emitter
                                    store!(
                                        emitter,
                                        :I_memory => layout_I_memory,
                                        :I;
                                        postprocess=addr -> quote
                                            let
                                                offset = $(Int32(M * 2 * N * F)) * T̄min
                                                length = $(Int32(M * 2 * N * F * T̄))
                                                mod($addr + offset, length)
                                            end
                                        end,
                                    )
                                    return nothing
                                end
                                apply!(emitter, :I, [:I], (I,) -> :(zero(Float16x2)))
                                push!(emitter.statements, :(t_running = $(0i32)))
                                push!(emitter.statements, :(dstime += $(1i32)))

                                return nothing
                            end

                            return nothing
                        end

                        return nothing
                    end
                    sync_threads!(emitter)

                    return nothing
                end
                return nothing
            end

            return nothing
        end

        return nothing
    end

    # # Write out partial I results if necessary
    # if T % Tds ≠ 0
    #     if!(emitter, :(
    #         let
    #             thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, $num_threads)
    #             warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, $num_warps)
    #             p = 2i32 * thread
    #             q = 2i32 * warp
    #             0i32 ≤ p < $(Int32(2 * M)) && 0i32 ≤ q < $(Int32(2 * N))
    #         end
    #     )) do emitter
    #         store!(emitter, :I_memory => layout_I_memory, :I)
    #         nothing
    #     end
    # end

    apply!(emitter, :info => layout_info_registers, 0i32)
    store!(emitter, :info_memory => layout_info_memory, :info)

    # Emit code

    stmts = clean_code(
        quote
            @fastmath @inbounds begin
                $(emitter.init_statements...)
                $(emitter.statements...)
            end
        end,
    )

    return stmts
end

println("[Creating frb kernel...]")
const frb_kernel = make_frb_kernel()
println("[Done creating frb kernel]")

@eval function frb(Tmin_memory, Tmax_memory, T̄min_memory, T̄max_memory, Smn_memory, W_memory, E_memory, I_memory, info_memory)
    shmem = @cuDynamicSharedMem(UInt8, shmem_bytes, 0)
    Fsh1_shared = reinterpret(Int4x8, shmem)
    Fsh2_shared = reinterpret(Int4x8, shmem)
    Gsh_shared = reinterpret(Float16x2, shmem)
    $frb_kernel
    return nothing
end

function main(; compile_only::Bool=false, output_kernel::Bool=false, run_selftest::Bool=false, nruns::Int=0, silent::Bool=false)
    !silent && println("CHORD FRB beamformer")

    if output_kernel
        open("output-$card/frb_$setup.jl", "w") do fh
            println(fh, "# Julia source code for CUDA frb beamformer")
            println(fh, "# This file has been generated automatically by `frb.jl`.")
            println(fh, "# Do not modify this file, your changes will be lost.")
            println(fh)
            println(fh, frb_kernel)
            return nothing
        end
    end

    !silent && println("Compiling kernel...")
    num_threads = kernel_setup.num_threads
    num_warps = kernel_setup.num_warps
    num_blocks = kernel_setup.num_blocks
    num_blocks_per_sm = kernel_setup.num_blocks_per_sm
    shmem_bytes = kernel_setup.shmem_bytes
    shmem_size = idiv(shmem_bytes, 4)
    @assert num_warps * num_blocks_per_sm ≤ 32 # (???)
    @assert shmem_bytes ≤ 99 * 1024 # NVIDIA A10/A40 have 99 kB shared memory
    kernel = @cuda launch = false minthreads = num_threads * num_warps blocks_per_sm = num_blocks_per_sm frb(
        CUDA.zeros(Int32, 0),
        CUDA.zeros(Int32, 0),
        CUDA.zeros(Int32, 0),
        CUDA.zeros(Int32, 0),
        CUDA.zeros(Int16x2, 0),
        CUDA.zeros(Float16x2, 0),
        CUDA.zeros(Int4x8, 0),
        CUDA.zeros(Float16x2, 0),
        CUDA.zeros(Int32, 0),
    )
    attributes(kernel.fun)[CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = shmem_bytes

    if compile_only
        return nothing
    end

    println("Allocating input data...")

    # TODO: determine types and sizes automatically
    Smn_memory = Array{Int16x2}(undef, M * N)
    W_memory = Array{Float16x2}(undef, M * N * F * P)
    E_memory = Array{Int4x8}(undef, idiv(D, 4) * P * F * T)
    I_wanted = Array{Float16x2}(undef, M * 2 * N * T̄ * F)
    info_wanted = Array{Int32}(undef, num_threads * num_warps * num_blocks)

    Random.seed!(0)
    niters = run_selftest ? 100 : 1
    found_error = false
    for iter in 1:niters
        if run_selftest
            println("Self-test iteration #$iter:")
        end

        println("Setting up input data...")
        map!(i -> zero(Int16x2), Smn_memory, Smn_memory)
        map!(i -> zero(Float16x2), W_memory, W_memory)
        map!(i -> zero(Int4x8), E_memory, E_memory)
        map!(i -> zero(Float16x2), I_wanted, I_wanted)
        map!(i -> zero(Int32), info_wanted, info_wanted)

        @show Tmin = 0
        @show Tmax = fld(idiv(T, 4), Touter) * Touter
        @show T̄min = 0
        @show T̄max = T̄min + fld(Tmax - Tmin, Tds)

        input = :random
        if input ≡ :zero
            # do nothing
        elseif input ≡ :random

            # Choose dish grid
            # Dishes 0:(D-1) are "real" dishes with E-field data.
            # Dishes D:(M*N-1) are "dummy" dishes where we set the E-field to zero.
            grid = [(m, n) for m in 0:(M - 1), n in 0:(N - 1)]
            # dish_grid = grid[1:(M * N)]
            dish_grid = grid[randperm(length(grid))]
            for d in 1:(M * N)
                Smn_memory[d] = Int16x2(dish_grid[d]...)
            end

            # Generate a uniform complex number in the unit disk. See
            # <https://stats.stackexchange.com/questions/481543/generating-random-points-uniformly-on-a-disk>.
            function uniform_in_disk()
                r = sqrt(rand(Float32))
                α = rand(Float32)
                c = r * cispi(2 * α)
                return c
            end
            uniform_factor() = (2 * rand(Float32) - 1)
            c2t(c::Complex) = (real(c), imag(c))
            t2c(t::NTuple{2}) = Complex(t[1], t[2])
            # Wvalue = 1 + 0im
            Wvalue = uniform_factor() * uniform_in_disk()
            # TODO: Set only on element of `W`, and this requires the dish gridding
            W_memory .= [Float16x2(c2t(Wvalue)...) for i in eachindex(W_memory)]

            dish = rand(0:(D - 1))
            freq = rand(0:(F - 1))
            polr = rand(0:(P - 1))
            time = rand(0:(T - 1))
            @show dish freq polr time
            Eidx = dish ÷ 4 + idiv(D, 4) * polr + idiv(D, 4) * P * freq + idiv(D, 4) * F * P * time
            Evalue = rand(-7:7) + im * rand(-7:7)
            Evalue8 = zero(SVector{8,Int8})
            Evalue8 = setindex(Evalue8, imag(Evalue), 2 * (dish % 4) + 0 + 1)
            Evalue8 = setindex(Evalue8, real(Evalue), 2 * (dish % 4) + 1 + 1)
            E_memory[Eidx + 1] = Int4x8(Evalue8...)
            @show Wvalue Evalue

            dstime = time ÷ Tds
            @show dstime
            for beamq in 0:(2 * N - 1), beamp in 0:(2 * M - 1)
                Iidx = beamp ÷ 2 + M * beamq + M * 2 * N * dstime + M * 2 * N * T̄ * freq
                dishm, dishn = dish_grid[dish + 1]
                # Eqn. (4)
                Ẽvalue = cispi((2 * dishm * beamp / Float32(2 * M) + 2 * dishn * beamq / Float32(2 * N)) % 2.0f0) * Wvalue * Evalue
                Ivalue = abs2(Ẽvalue)
                Ivalue2 = convert(NTuple{2,Float32}, I_wanted[Iidx + 1])
                Ivalue2 = setindex(Ivalue2, Ivalue, beamp % 2 + 1)
                I_wanted[Iidx + 1] = Float16x2(Ivalue2...)
            end
        end

        println("Copying data from CPU to GPU...")
        Tmin_cuda = CuArray(Int32[Tmin])
        Tmax_cuda = CuArray(Int32[Tmax])
        T̄min_cuda = CuArray(Int32[T̄min])
        T̄max_cuda = CuArray(Int32[T̄max])
        Smn_cuda = CuArray(Smn_memory)
        W_cuda = CuArray(W_memory)
        E_cuda = CuArray(E_memory)
        I_cuda = CUDA.fill(Float16x2(NaN, NaN), length(I_wanted))
        info_cuda = CUDA.fill(-1i32, length(info_wanted))

        println("Running kernel...")
        kernel(
            Tmin_cuda,
            Tmax_cuda,
            T̄min_cuda,
            T̄max_cuda,
            Smn_cuda,
            W_cuda,
            E_cuda,
            I_cuda,
            info_cuda;
            threads=(num_threads, num_warps),
            blocks=num_blocks,
            shmem=shmem_bytes,
        )
        synchronize()

        if nruns > 0
            stats = @timed begin
                for run in 1:nruns
                    kernel(
                        Tmin_cuda,
                        Tmax_cuda,
                        T̄min_cuda,
                        T̄max_cuda,
                        Smn_cuda,
                        W_cuda,
                        E_cuda,
                        I_cuda,
                        info_cuda;
                        threads=(num_threads, num_warps),
                        blocks=num_blocks,
                        shmem=shmem_bytes,
                    )
                end
                synchronize()
            end
            # All times in μsec
            runtime = stats.time / nruns * 1.0e+6
            num_frequencies_scaled = F₀
            runtime_scaled = runtime / F * num_frequencies_scaled
            dataframe_length = T * sampling_time_μsec
            fraction = runtime_scaled / dataframe_length
            round1(x) = round(x; digits=1)
            println("""
            benchmark-result:
              kernel: "frb"
              description: "FRB beamformer"
              design-parameters:
                beam-layout: [$(2*M), $(2*N)]
                dish-layout: [$M, $N]
                downsampling-factor: $Tds
                number-of-complex-components: $C
                number-of-dishes: $D
                number-of-frequencies: $F
                number-of-polarizations: $P
                number-of-timesamples: $T
                sampling-time-μsec: $sampling_time_μsec
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
        I_memory = Array(I_cuda)
        info_memory = Array(info_cuda)
        @assert all(info_memory .== 0)

        if run_selftest
            println("Checking results...")

            println("    I:")
            did_test_I_memory = falses(length(I_memory))
            for freq in 0:(F - 1), dstime in 0:(T̄ - 1), beamq in 0:(2 * N - 1), beamp in 0:(2 * M - 1)
                Iidx = beamp ÷ 2 + M * beamq + M * 2 * N * dstime + M * 2 * N * T̄ * freq
                if beamp % 2 == 0
                    @assert !did_test_I_memory[Iidx + 1]
                    did_test_I_memory[Iidx + 1] = true
                end
                have_value2 = convert(NTuple{2,Float32}, I_memory[Iidx + 1])
                want_value2 = convert(NTuple{2,Float32}, I_wanted[Iidx + 1])
                have_value = have_value2[beamp % 2 + 1]
                want_value = want_value2[beamp % 2 + 1]
                # if have_value ≠ want_value
                if !isapprox(have_value, want_value; atol=10 * eps(Float16), rtol=10 * eps(Float16))
                    println("        beamp=$beamp beamq=$beamq freq=$freq dstime=$dstime I=$have_value I₀=$want_value")
                    found_error = true
                end
            end
            @assert all(did_test_I_memory)

            found_error && break
        end
        if found_error
            println("*** FOUND ERROR DURING SELF-TEST ***")
        end
    end

    println("Done.")
    return nothing
end

function fix_ptx_kernel()
    ptx = read("output-$card/frb_$setup.ptx", String)
    ptx = replace(ptx, r".extern .func ([^;]*);"s => s".func \1.noreturn\n{\n\ttrap;\n}")
    open("output-$card/frb_$setup.ptx", "w") do fh
        println(fh, "// PTX kernel code for CUDA frb beamformer")
        println(fh, "// This file has been generated automatically by `frb.jl`.")
        println(fh, "// Do not modify this file, your changes will be lost.")
        println(fh)
        write(fh, ptx)
        return nothing
    end
    sass = read("output-$card/frb_$setup.sass", String)
    open("output-$card/frb_$setup.sass", "w") do fh
        println(fh, "// SASS kernel code for CUDA frb beamformer")
        println(fh, "// This file has been generated automatically by `frb.jl`.")
        println(fh, "// Do not modify this file, your changes will be lost.")
        println(fh)
        write(fh, sass)
        return nothing
    end
    kernel_symbol = match(r"\s\.globl\s+(\S+)"m, ptx).captures[1]
    open("output-$card/frb_$setup.yaml", "w") do fh
        println(fh, "# Metadata code for CUDA frb beamformer")
        println(fh, "# This file has been generated automatically by `frb.jl`.")
        println(fh, "# Do not modify this file, your changes will be lost.")
        println(fh)
        print(
            fh,
            """
    --- !<tag:chord-observatory.ca/x-engine/kernel-description-1.0.0>
    kernel-description:
      name: "frb"
      description: "FRB beamformer"
      design-parameters:
        beam-layout: [$(2*M), $(2*N)]
        dish-layout: [$M, $N]
        downsampling-factor: $Tds
        number-of-complex-components: $C
        number-of-dishes: $D
        number-of-frequencies: $F
        number-of-polarizations: $P
        number-of-timesamples: $T
        output-gain: $output_gain
        sampling-time-μsec: $sampling_time_μsec
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
          indices: []
          shape: []
          strides: []
        - name: "Tmax"
          intent: in
          type: Int32
          indices: []
          shape: []
          strides: []
        - name: "T̄min"
          intent: in
          type: Int32
          indices: []
          shape: []
          strides: []
        - name: "T̄max"
          intent: in
          type: Int32
          indices: []
          shape: []
          strides: []
        - name: "S"
          intent: in
          type: Int16
          indices: [MN, D]
          shape: [2, $(M*N)]
          strides: [1, 2]
        - name: "W"
          intent: in
          type: Float16
          indices: [C, dishM, dishN, P, F]
          shape: [$C, $M, $N, $P, $F]
          strides: [1, $C, $(C*M), $(C*M*N), $(C*M*N*P), $(C*M*N*P*F)]
        - name: "E"
          intent: in
          type: Int4
          indices: [C, D, P, F, T]
          shape: [$C, $D, $P, $F, $T]
          strides: [1, $C, $(C*D), $(C*D*P), $(C*D*P*F)]
        - name: "I"
          intent: out
          type: Float16
          indices: [beamP, beamQ, F, Tbar]
          shape: [$(2*M), $(2*N), $F, $(T̄)]
          strides: [1, $(2*M), $(2*M*2*N), $(2*M*2*N*F)]
        - name: "info"
          intent: out
          type: Int32
          indices: [thread, warp, block]
          shapes: [$num_threads, $num_warps, $num_blocks]
          strides: [1, $num_threads, $(num_threads*num_warps)]
    ...
    """,
        )
        return nothing
    end
    cxx = read("kernels/frb_template.cxx", String)
    cxx = Mustache.render(
        cxx,
        Dict(
            "kernel_name" => "FRBBeamformer_$setup",
            "kernel_design_parameters" => [
                Dict("type" => "int", "name" => "cuda_beam_layout_M", "value" => "$(2*M)"),
                Dict("type" => "int", "name" => "cuda_beam_layout_N", "value" => "$(2*N)"),
                Dict("type" => "int", "name" => "cuda_dish_layout_M", "value" => "$M"),
                Dict("type" => "int", "name" => "cuda_dish_layout_N", "value" => "$N"),
                Dict("type" => "int", "name" => "cuda_downsampling_factor", "value" => "$Tds"),
                Dict("type" => "int", "name" => "cuda_number_of_complex_components", "value" => "$C"),
                Dict("type" => "int", "name" => "cuda_number_of_dishes", "value" => "$D"),
                Dict("type" => "int", "name" => "cuda_number_of_frequencies", "value" => "$F"),
                Dict("type" => "int", "name" => "cuda_number_of_polarizations", "value" => "$P"),
                Dict("type" => "int", "name" => "cuda_number_of_timesamples", "value" => "$T"),
                Dict("type" => "int", "name" => "cuda_granularity_number_of_timesamples", "value" => "$Touter"),
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
                    "axes" => Dict[],
                    "isoutput" => false,
                    "hasbuffer" => false,
                ),
                Dict(
                    "name" => "Tmax",
                    "kotekan_name" => "Tmax",
                    "type" => "int32",
                    "axes" => Dict[],
                    "isoutput" => false,
                    "hasbuffer" => false,
                ),
                Dict(
                    "name" => "Tbarmin",
                    "kotekan_name" => "Tbarmin",
                    "type" => "int32",
                    "axes" => Dict[],
                    "isoutput" => false,
                    "hasbuffer" => false,
                ),
                Dict(
                    "name" => "Tbarmax",
                    "kotekan_name" => "Tbarmax",
                    "type" => "int32",
                    "axes" => Dict[],
                    "isoutput" => false,
                    "hasbuffer" => false,
                ),
                Dict(
                    "name" => "S",
                    "kotekan_name" => "gpu_mem_dishlayout",
                    "type" => "int16",
                    "axes" => [Dict("label" => "MN", "length" => 2), Dict("label" => "D", "length" => M * N)],
                    "isoutput" => false,
                    "hasbuffer" => false,
                ),
                Dict(
                    "name" => "W",
                    "kotekan_name" => "gpu_mem_phase",
                    "type" => "float16",
                    "axes" => [
                        Dict("label" => "C", "length" => C),
                        Dict("label" => "dishM", "length" => M),
                        Dict("label" => "dishN", "length" => N),
                        Dict("label" => "P", "length" => P),
                        Dict("label" => "F", "length" => F),
                    ],
                    "isoutput" => false,
                    "hasbuffer" => true,
                ),
                Dict(
                    "name" => "E",
                    "kotekan_name" => "gpu_mem_voltage",
                    "type" => "int4p4",
                    "axes" => [
                        Dict("label" => "D", "length" => D),
                        Dict("label" => "P", "length" => P),
                        Dict("label" => "F", "length" => F),
                        Dict("label" => "T", "length" => T),
                    ],
                    "isoutput" => false,
                    "hasbuffer" => true,
                ),
                Dict(
                    "name" => "I",
                    "kotekan_name" => "gpu_mem_beamgrid",
                    "type" => "float16",
                    "axes" => [
                        Dict("label" => "beamP", "length" => 2 * M),
                        Dict("label" => "beamQ", "length" => 2 * N),
                        Dict("label" => "F", "length" => F),
                        Dict("label" => "Tbar", "length" => T̄),
                    ],
                    "isoutput" => true,
                    "hasbuffer" => true,
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
                ),
            ],
        ),
    )
    write("output-$card/frb_$setup.cxx", cxx)
    return nothing
end

if CUDA.functional()
    # Output kernel
    main(; output_kernel=true)
    open("output-$card/frb_$setup.ptx", "w") do fh
        redirect_stdout(fh) do
            @device_code_ptx main(; compile_only=true, silent=true)
        end
    end
    open("output-$card/frb_$setup.sass", "w") do fh
        redirect_stdout(fh) do
            @device_code_sass main(; compile_only=true, silent=true)
        end
    end
    fix_ptx_kernel()

    # # Run test
    # main(; run_selftest=true)

    # # Run benchmark
    # main(; nruns=10000)

    # # Regular run, also for profiling
    # main()
end
