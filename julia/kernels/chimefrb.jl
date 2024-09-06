# CHIME FRB beamformer
# <CHIME_FRB_beamforming.pdf>

using CUDA
using CUDASIMDTypes
using IndexSpaces
using Mustache

const Memory = IndexSpaces.Memory

const card = "A40"

if CUDA.functional()
    println("[Choosing CUDA device...]")
    CUDA.device!(0)
    println(name(device()))
    @assert name(device()) == "NVIDIA $card"
end

idiv(i::Integer, j::Integer) = (@assert iszero(i % j); i ÷ j)

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

Base.isnan(f::Float16x2) = any(isnan, convert(NTuple{2,Float16}, f))

setup::Symbol
F::Integer
T::Integer
U::Integer

const F̄ = F_per_U[U] * U

const Tbar = T ÷ U
const Tds = idiv(Tds_U1, U)

const Fbar_W = F̄
const Fbar = U == 1 ? F : F̄

# Compile-time constants (section 4.4)
@static if setup ≡ :chime

    # CHIME Setup

    const M = 256               # north-south
    const N = 4                 # east-west

    const Treg = 1
    @assert Tds % Treg == 0

    const W = 8                 # number of warps
    const B = 2                 # number of blocks per SM

else
    @assert false
end

const Ttilde = 4 * 256

const output_gain = 1 / (8 * Tds)

# Machine setup

const num_simd_bits = 32
const num_threads = 32
const num_warps = W
const num_blocks = Fbar
const num_blocks_per_sm = B

# CHORD indices

@enum CHORDTag begin
    CplxTag
    DishTag
    BeamPTag
    BeamQTag
    FreqTag
    PolrTag
    TimeTag
    ThreadTag
    WarpTag
    BlockTag
end

const Cplx = Index{Physics,CplxTag}
const Dish = Index{Physics,DishTag}
const BeamP = Index{Physics,BeamPTag}
const BeamQ = Index{Physics,BeamQTag}
const Freq = Index{Physics,FreqTag}
const Polr = Index{Physics,PolrTag}
const Time = Index{Physics,TimeTag}

# Layouts

const layout_E_memory = Layout([
    IntValue(:intvalue, 1, 4) => SIMD(:simd, 1, 4),
    Cplx(:cplx, 1, C) => SIMD(:simd, 4, 2),
    Dish(:dish, 1, 4) => SIMD(:simd, 8, 4),
    Dish(:dish, 4, idiv(D, 4)) => Memory(:memory, 1, idiv(D, 4)),
    Polr(:polr, 1, P) => Memory(:memory, idiv(D, 4), P),
    Freq(:freq, 1, Fbar) => Memory(:memory, idiv(D, 4) * P, Fbar),
    Time(:time, 1, Tbar) => Memory(:memory, idiv(D, 4) * Fbar * P, Tbar),
])

const layout_W_memory = Layout([
    FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
    Cplx(:cplx, 1, C) => SIMD(:simd, 16, 2),
    Dish(:dish, 1, D) => Memory(:memory, 1, D),
    Polr(:polr, 1, P) => Memory(:memory, D, P),
    Freq(:freq, 1, Fbar_W) => Memory(:memory, D * P, Fbar_W),
])

# Y layout

const layout_Y_shared = Layout([
    FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
    BeamP(:beamP, 1, 2) => SIMD(:simd, 16, 2),
    BeamP(:beamP, idiv(2 * M, 32), 32) => Shared(:shared, 1, 32), # for 32 threads
    Cplx(:cplx, 1, C) => Shared(:shared, 32, C),
    BeamP(:beamP, 2, idiv(M, 32)) => Shared(:shared, 32 * C, idiv(M, 32)),
    Dish(:dish, 256, N) => Shared(:shared, C * M, N),
    Polr(:polr, 1, P) => Shared(:shared, C * M * N, P),
    Freq(:freq, 1, Fbar) => Block(:block, 1, Fbar),
    Time(:time, 1, Treg) => Shared(:shared, C * M * N * P, Treg),
    Time(:time, Treg, idiv(Tds, Treg)) => Loop(:time_inner, 1, idiv(Tds, Treg)),
    Time(:time, Tds, fld(Ttilde, Tds)) => Loop(:time_outer, Tds, fld(Ttilde, Tds)),
])
const Y_size = C * M * N * P * Treg

# I layout

const layout_I_memory = Layout([
    FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
    BeamP(:beamP, 1, 2) => SIMD(:simd, 16, 2),
    BeamP(:beamP, 2, M) => Memory(:memory, 1, M),
    BeamQ(:beamQ, 1, 2 * N) => Memory(:memory, M, 2 * N),
    # BeamQ(:beamQ, 1, 2) => Memory(:memory, 1, 2),
    # BeamQ(:beamQ, 2, 2) => Memory(:memory, 2, 2),
    # BeamP(:beamP, 16, 2) => Memory(:memory, 4, 2),
    # BeamP(:beamP, 32, 2) => Memory(:memory, 8, 2),
    # BeamP(:beamP, 64, 2) => Memory(:memory, 16, 2),
    # BeamP(:beamP, 128, 2) => Memory(:memory, 32, 2),
    # BeamP(:beamP, 256, 2) => Memory(:memory, 64, 2),
    # BeamQ(:beamQ, 4, 2) => Memory(:memory, 128, 2),
    # BeamP(:beamP, 2, 2) => Memory(:memory, 256, 2),
    # BeamP(:beamP, 4, 2) => Memory(:memory, 512, 2),
    # BeamP(:beamP, 8, 2) => Memory(:memory, 1024, 2),
    Freq(:freq, 1, Fbar) => Memory(:memory, M * 2 * N, Fbar),
    Time(:time, Tds, Ttilde) => Memory(:memory, M * 2 * N * Fbar, Ttilde),
])

# info layout

const layout_info_memory = Layout([
    IntValue(:intvalue, 1, 32) => SIMD(:simd, 1, 32),
    Index{Physics,ThreadTag}(:thread, 1, num_threads) => Memory(:memory, 1, num_threads),
    Index{Physics,WarpTag}(:warp, 1, num_warps) => Memory(:memory, num_threads, num_warps),
    Index{Physics,BlockTag}(:block, 1, num_blocks) => Memory(:memory, num_threads * num_warps, num_blocks),
])

const layout_info_registers = Layout([
    IntValue(:intvalue, 1, 32) => SIMD(:simd, 1, 32),
    Index{Physics,ThreadTag}(:thread, 1, num_threads) => Thread(:thread, 1, num_threads),
    Index{Physics,WarpTag}(:warp, 1, num_warps) => Warp(:warp, 1, num_warps),
    Index{Physics,BlockTag}(:block, 1, num_blocks) => Block(:block, 1, num_blocks),
])

const shmem_size = Y_size
const shmem_bytes = 4 * shmem_size

const kernel_setup = KernelSetup(num_threads, num_warps, num_blocks, num_blocks_per_sm, shmem_bytes)

function make_chimefrb_kernel()
    emitter = Emitter(kernel_setup)

    # Emit code

    apply!(emitter, :info => layout_info_registers, 1i32)
    store!(emitter, :info_memory => layout_info_memory, :info)

    # Check parameters `Tbarmin`, `Tbarmax`, `Ttildemin`, `Ttildemax`
    if!(
        emitter,
        :(
            !(
                0i32 ≤ Tbarmin < $(Int32(Tbar)) &&
                Tbarmin ≤ Tbarmax < $(Int32(2 * Tbar)) &&
                (Tbarmax - Tbarmin) % $(Int32(Tds)) == 0i32 &&
                0i32 ≤ Ttildemin < $(Int32(Ttilde)) &&
                Ttildemin ≤ Ttildemax < $(Int32(2 * Ttilde)) &&
                Ttildemax - Ttildemin == (Tbarmax - Tbarmin) ÷ $(Int32(Tds))
            )
        ),
    ) do emitter
        apply!(emitter, :info => layout_info_registers, 2i32)
        store!(emitter, :info_memory => layout_info_memory, :info)
        trap!(emitter)
        return nothing
    end

    # Calculate Γ1

    # (24)
    layout_Γ1_registers = Layout([
        FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
        Cplx(:cplx_in, 1, 2) => SIMD(:simd, 16, 2),
        # Dish(:dish, 1, 2) => Register(:dish, 1, 2),
        # Dish(:dish, 2, 2) => Register(:dish, 2, 2),
        Dish(:dish, 4, 2) => Register(:dish, 4, 2), # dummy index
        Dish(:dish, 64, 2) => Thread(:thread, 2, 2),
        Dish(:dish, 128, 2) => Thread(:thread, 1, 2),
        # Cplx(:cplx, 1, 2) => Register(:cplx, 1, 2),
        BeamP(:beamP, 1, 2) => Thread(:thread, 4, 2),
        BeamP(:beamP, 2, 2) => Thread(:thread, 8, 2),
        BeamP(:beamP, 4, 2) => Thread(:thread, 16, 2),
    ])
    push!(
        emitter.statements,
        quote
            (
                Γ1_dish0_0_dish1_0_cplx_in_0_cplx_0,
                Γ1_dish0_0_dish1_0_cplx_in_0_cplx_1,
                Γ1_dish0_0_dish1_0_cplx_in_1_cplx_0,
                Γ1_dish0_0_dish1_0_cplx_in_1_cplx_1,
                Γ1_dish0_1_dish1_0_cplx_in_0_cplx_0,
                Γ1_dish0_1_dish1_0_cplx_in_0_cplx_1,
                Γ1_dish0_1_dish1_0_cplx_in_1_cplx_0,
                Γ1_dish0_1_dish1_0_cplx_in_1_cplx_1,
                Γ1_dish0_0_dish1_1_cplx_in_0_cplx_0,
                Γ1_dish0_0_dish1_1_cplx_in_0_cplx_1,
                Γ1_dish0_0_dish1_1_cplx_in_1_cplx_0,
                Γ1_dish0_0_dish1_1_cplx_in_1_cplx_1,
                Γ1_dish0_1_dish1_1_cplx_in_0_cplx_0,
                Γ1_dish0_1_dish1_1_cplx_in_0_cplx_1,
                Γ1_dish0_1_dish1_1_cplx_in_1_cplx_0,
                Γ1_dish0_1_dish1_1_cplx_in_1_cplx_1,
            ) = let
                # (5): Γ = exp(2π * im * q * n / 512)
                # (8): q = {q_j} for j in [0, 1, 2]
                #      n = {n_k} for k in [0, 1, 6, 7]
                # (24)
                thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, $num_threads)
                thread0 = (thread ÷ 1i32 % 2i32)
                thread1 = (thread ÷ 2i32 % 2i32)
                thread2 = (thread ÷ 4i32 % 2i32)
                thread3 = (thread ÷ 8i32 % 2i32)
                thread4 = (thread ÷ 16i32 % 2i32)
                dish0_0 = 0i32
                dish0_1 = 1i32
                dish1_0 = 0i32
                dish1_1 = 1i32
                dish6 = thread1
                dish7 = thread0
                beamp0 = thread2
                beamp1 = thread3
                beamp2 = thread4
                n_dish0_0_dish1_0 = dish0_0 * 1i32 + dish1_0 * 2i32 + dish6 * 64i32 + dish7 * 128i32
                n_dish0_0_dish1_1 = dish0_0 * 1i32 + dish1_1 * 2i32 + dish6 * 64i32 + dish7 * 128i32
                n_dish0_1_dish1_0 = dish0_1 * 1i32 + dish1_0 * 2i32 + dish6 * 64i32 + dish7 * 128i32
                n_dish0_1_dish1_1 = dish0_1 * 1i32 + dish1_1 * 2i32 + dish6 * 64i32 + dish7 * 128i32
                q = beamp0 * 1i32 + beamp1 * 2i32 + beamp2 * 4i32
                Γ1_dish0_0_dish1_0 = cispi(q * n_dish0_0_dish1_0 % 512i32 * Float32(2 / 512))
                Γ1_dish0_0_dish1_1 = cispi(q * n_dish0_0_dish1_1 % 512i32 * Float32(2 / 512))
                Γ1_dish0_1_dish1_0 = cispi(q * n_dish0_1_dish1_0 % 512i32 * Float32(2 / 512))
                Γ1_dish0_1_dish1_1 = cispi(q * n_dish0_1_dish1_1 % 512i32 * Float32(2 / 512))
                (
                    +Γ1_dish0_0_dish1_0.re,
                    +Γ1_dish0_0_dish1_0.im,
                    -Γ1_dish0_0_dish1_0.im,
                    +Γ1_dish0_0_dish1_0.re,
                    +Γ1_dish0_1_dish1_0.re,
                    +Γ1_dish0_1_dish1_0.im,
                    -Γ1_dish0_1_dish1_0.im,
                    +Γ1_dish0_1_dish1_0.re,
                    +Γ1_dish0_0_dish1_1.re,
                    +Γ1_dish0_0_dish1_1.im,
                    -Γ1_dish0_0_dish1_1.im,
                    +Γ1_dish0_0_dish1_1.re,
                    +Γ1_dish0_1_dish1_1.re,
                    +Γ1_dish0_1_dish1_1.im,
                    -Γ1_dish0_1_dish1_1.im,
                    +Γ1_dish0_1_dish1_1.re,
                )
            end
        end,
    )

    apply!(
        emitter,
        :Γ1_dish0_0_dish1_0_cplx_0 => layout_Γ1_registers,
        :(Float16x2(Γ1_dish0_0_dish1_0_cplx_in_0_cplx_0, Γ1_dish0_0_dish1_0_cplx_in_1_cplx_0)),
    )
    apply!(
        emitter,
        :Γ1_dish0_0_dish1_0_cplx_1 => layout_Γ1_registers,
        :(Float16x2(Γ1_dish0_0_dish1_0_cplx_in_0_cplx_1, Γ1_dish0_0_dish1_0_cplx_in_1_cplx_1)),
    )
    apply!(
        emitter,
        :Γ1_dish0_0_dish1_1_cplx_0 => layout_Γ1_registers,
        :(Float16x2(Γ1_dish0_0_dish1_1_cplx_in_0_cplx_0, Γ1_dish0_0_dish1_1_cplx_in_1_cplx_0)),
    )
    apply!(
        emitter,
        :Γ1_dish0_0_dish1_1_cplx_1 => layout_Γ1_registers,
        :(Float16x2(Γ1_dish0_0_dish1_1_cplx_in_0_cplx_1, Γ1_dish0_0_dish1_1_cplx_in_1_cplx_1)),
    )
    apply!(
        emitter,
        :Γ1_dish0_1_dish1_0_cplx_0 => layout_Γ1_registers,
        :(Float16x2(Γ1_dish0_1_dish1_0_cplx_in_0_cplx_0, Γ1_dish0_1_dish1_0_cplx_in_1_cplx_0)),
    )
    apply!(
        emitter,
        :Γ1_dish0_1_dish1_0_cplx_1 => layout_Γ1_registers,
        :(Float16x2(Γ1_dish0_1_dish1_0_cplx_in_0_cplx_1, Γ1_dish0_1_dish1_0_cplx_in_1_cplx_1)),
    )
    apply!(
        emitter,
        :Γ1_dish0_1_dish1_1_cplx_0 => layout_Γ1_registers,
        :(Float16x2(Γ1_dish0_1_dish1_1_cplx_in_0_cplx_0, Γ1_dish0_1_dish1_1_cplx_in_1_cplx_0)),
    )
    apply!(
        emitter,
        :Γ1_dish0_1_dish1_1_cplx_1 => layout_Γ1_registers,
        :(Float16x2(Γ1_dish0_1_dish1_1_cplx_in_0_cplx_1, Γ1_dish0_1_dish1_1_cplx_in_1_cplx_1)),
    )

    merge!(
        emitter,
        :Γ1_dish0_0_dish1_0,
        [:Γ1_dish0_0_dish1_0_cplx_0, :Γ1_dish0_0_dish1_0_cplx_1],
        Cplx(:cplx, 1, 2) => Register(:cplx, 1, 2),
    )
    merge!(
        emitter,
        :Γ1_dish0_0_dish1_1,
        [:Γ1_dish0_0_dish1_1_cplx_0, :Γ1_dish0_0_dish1_1_cplx_1],
        Cplx(:cplx, 1, 2) => Register(:cplx, 1, 2),
    )
    merge!(
        emitter,
        :Γ1_dish0_1_dish1_0,
        [:Γ1_dish0_1_dish1_0_cplx_0, :Γ1_dish0_1_dish1_0_cplx_1],
        Cplx(:cplx, 1, 2) => Register(:cplx, 1, 2),
    )
    merge!(
        emitter,
        :Γ1_dish0_1_dish1_1,
        [:Γ1_dish0_1_dish1_1_cplx_0, :Γ1_dish0_1_dish1_1_cplx_1],
        Cplx(:cplx, 1, 2) => Register(:cplx, 1, 2),
    )

    merge!(emitter, :Γ1_dish0_0, [:Γ1_dish0_0_dish1_0, :Γ1_dish0_0_dish1_1], Dish(:dish, 2, 2) => Register(:dish, 2, 2))
    merge!(emitter, :Γ1_dish0_1, [:Γ1_dish0_1_dish1_0, :Γ1_dish0_1_dish1_1], Dish(:dish, 2, 2) => Register(:dish, 2, 2))

    merge!(emitter, :Γ1, [:Γ1_dish0_0, :Γ1_dish0_1], Dish(:dish, 1, 2) => Register(:dish, 1, 2))

    # Calculate Γ2

    # (25)
    layout_Γ2_registers = Layout([
        FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
        # Cplx(:cplx, 1, 2) => Register(:cplx, 1, 2),
        Dish(:dish, 1, 2) => Register(:dish, 1, 2), # dummy index
        Dish(:dish, 2, 2) => Register(:dish, 2, 2), # dummy index
        # Dish(:dish, 4, 2) => Register(:dish, 4, 2),
        Dish(:dish, 8, 2) => Thread(:thread, 2, 2),
        Dish(:dish, 16, 2) => Thread(:thread, 1, 2),
        Dish(:dish, 32, 2) => SIMD(:simd, 16, 2),
        BeamP(:beamP, 1, 2) => Thread(:thread, 4, 2),
        BeamP(:beamP, 2, 2) => Thread(:thread, 8, 2),
        BeamP(:beamP, 4, 2) => Thread(:thread, 16, 2),
    ])
    push!(
        emitter.statements,
        quote
            (
                Γ2_dish2_0_dish5_0_cplx_0,
                Γ2_dish2_0_dish5_0_cplx_1,
                Γ2_dish2_1_dish5_0_cplx_0,
                Γ2_dish2_1_dish5_0_cplx_1,
                Γ2_dish2_0_dish5_1_cplx_0,
                Γ2_dish2_0_dish5_1_cplx_1,
                Γ2_dish2_1_dish5_1_cplx_0,
                Γ2_dish2_1_dish5_1_cplx_1,
            ) = let
                # (5): Γ = exp(2π * im * q * n / 512)
                # (9): q = {q_j} for j in [0, 1, 2]
                #      n = {n_k} for k in [2, 3, 4, 5]
                # (25)
                thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, $num_threads)
                thread0 = (thread ÷ 1i32 % 2i32)
                thread1 = (thread ÷ 2i32 % 2i32)
                thread2 = (thread ÷ 4i32 % 2i32)
                thread3 = (thread ÷ 8i32 % 2i32)
                thread4 = (thread ÷ 16i32 % 2i32)
                dish2_0 = 0i32
                dish2_1 = 1i32
                dish3 = thread1
                dish4 = thread0
                dish5_0 = 0i32
                dish5_1 = 1i32
                beamp0 = thread2
                beamp1 = thread3
                beamp2 = thread4
                n_dish2_0_dish5_0 = dish2_0 * 4i32 + dish3 * 8i32 + dish4 * 16i32 + dish5_0 * 32i32
                n_dish2_0_dish5_1 = dish2_0 * 4i32 + dish3 * 8i32 + dish4 * 16i32 + dish5_1 * 32i32
                n_dish2_1_dish5_0 = dish2_1 * 4i32 + dish3 * 8i32 + dish4 * 16i32 + dish5_0 * 32i32
                n_dish2_1_dish5_1 = dish2_1 * 4i32 + dish3 * 8i32 + dish4 * 16i32 + dish5_1 * 32i32
                q = beamp0 * 1i32 + beamp1 * 2i32 + beamp2 * 4i32
                Γ2_dish2_0_dish5_0 = cispi(q * n_dish2_0_dish5_0 % 512i32 * Float32(2 / 512))
                Γ2_dish2_0_dish5_1 = cispi(q * n_dish2_0_dish5_1 % 512i32 * Float32(2 / 512))
                Γ2_dish2_1_dish5_0 = cispi(q * n_dish2_1_dish5_0 % 512i32 * Float32(2 / 512))
                Γ2_dish2_1_dish5_1 = cispi(q * n_dish2_1_dish5_1 % 512i32 * Float32(2 / 512))
                (
                    Γ2_dish2_0_dish5_0.re,
                    Γ2_dish2_0_dish5_0.im,
                    Γ2_dish2_1_dish5_0.re,
                    Γ2_dish2_1_dish5_0.im,
                    Γ2_dish2_1_dish5_0.re,
                    Γ2_dish2_0_dish5_0.im,
                    Γ2_dish2_0_dish5_0.re,
                    Γ2_dish2_1_dish5_0.im,
                    Γ2_dish2_1_dish5_0.re,
                    Γ2_dish2_0_dish5_1.im,
                    Γ2_dish2_0_dish5_1.re,
                    Γ2_dish2_1_dish5_1.im,
                    Γ2_dish2_0_dish5_1.re,
                    Γ2_dish2_0_dish5_1.im,
                    Γ2_dish2_1_dish5_1.re,
                    Γ2_dish2_1_dish5_1.im,
                )
            end
        end,
    )

    apply!(emitter, :Γ2_dish2_0_cplx_0 => layout_Γ2_registers, :(Float16x2(Γ2_dish2_0_dish5_0_cplx_0, Γ2_dish2_0_dish5_1_cplx_0)))
    apply!(emitter, :Γ2_dish2_0_cplx_1 => layout_Γ2_registers, :(Float16x2(Γ2_dish2_0_dish5_0_cplx_1, Γ2_dish2_0_dish5_1_cplx_1)))
    apply!(emitter, :Γ2_dish2_1_cplx_0 => layout_Γ2_registers, :(Float16x2(Γ2_dish2_1_dish5_0_cplx_0, Γ2_dish2_1_dish5_1_cplx_0)))
    apply!(emitter, :Γ2_dish2_1_cplx_1 => layout_Γ2_registers, :(Float16x2(Γ2_dish2_1_dish5_0_cplx_1, Γ2_dish2_1_dish5_1_cplx_1)))

    merge!(emitter, :Γ2_dish2_0, [:Γ2_dish2_0_cplx_0, :Γ2_dish2_0_cplx_1], Cplx(:cplx, 1, 2) => Register(:cplx, 1, 2))
    merge!(emitter, :Γ2_dish2_1, [:Γ2_dish2_1_cplx_0, :Γ2_dish2_1_cplx_1], Cplx(:cplx, 1, 2) => Register(:cplx, 1, 2))

    merge!(emitter, :Γ2, [:Γ2_dish2_0, :Γ2_dish2_1], Dish(:dish, 4, 2) => Register(:dish, 4, 2))

    # Calculate Γ3

    # (26)
    layout_Γ3_registers = Layout([
        FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
        # Cplx(:cplx_in, 1, 2) => Register(:cplx_in, 1, 2),
        Dish(:dish, 1, 2) => Register(:dish, 1, 2), # dummy index
        Dish(:dish, 2, 2) => Register(:dish, 2, 2), # dummy index
        Dish(:dish, 4, 2) => Register(:dish, 4, 2), # dummy index
        Dish(:dish, 8, 2) => Thread(:thread, 2, 2),
        Dish(:dish, 16, 2) => Thread(:thread, 1, 2),
        Dish(:dish, 32, 2) => SIMD(:simd, 16, 2),
        # Cplx(:cplx, 1, 2) => Register(:cplx, 1, 2),
        BeamP(:beamP, 8, 2) => Thread(:thread, 4, 2),
        BeamP(:beamP, 16, 2) => Thread(:thread, 8, 2),
        BeamP(:beamP, 32, 2) => Thread(:thread, 16, 2),
    ])
    push!(
        emitter.statements,
        quote
            (
                Γ3_dish5_0_cplx_in_0_cplx_0,
                Γ3_dish5_0_cplx_in_1_cplx_0,
                Γ3_dish5_0_cplx_in_0_cplx_1,
                Γ3_dish5_0_cplx_in_1_cplx_1,
                Γ3_dish5_1_cplx_in_0_cplx_0,
                Γ3_dish5_1_cplx_in_1_cplx_0,
                Γ3_dish5_1_cplx_in_0_cplx_1,
                Γ3_dish5_1_cplx_in_1_cplx_1,
            ) = let
                # (5): Γ = exp(2π * im * q * n / 512)
                # (10): q = {q_j} for j in [3, 4, 5]
                #       n = {n_k} for k in [3, 4, 5n]
                # (26)
                thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, $num_threads)
                thread0 = (thread ÷ 1i32 % 2i32)
                thread1 = (thread ÷ 2i32 % 2i32)
                thread2 = (thread ÷ 4i32 % 2i32)
                thread3 = (thread ÷ 8i32 % 2i32)
                thread4 = (thread ÷ 16i32 % 2i32)
                dish3 = thread1
                dish4 = thread0
                dish5_0 = 0i32
                dish5_1 = 1i32
                beamp3 = thread2
                beamp4 = thread3
                beamp5 = thread4
                n_dish5_0 = dish3 * 8i32 + dish4 * 16i32 + dish5_0 * 32i32
                n_dish5_1 = dish3 * 8i32 + dish4 * 16i32 + dish5_1 * 32i32
                q = beamp3 * 8i32 + beamp4 * 16i32 + beamp5 * 32i32
                Γ3_dish5_0 = cispi(q * n_dish5_0 % 512i32 * Float32(2 / 512))
                Γ3_dish5_1 = cispi(q * n_dish5_1 % 512i32 * Float32(2 / 512))
                (
                    +Γ3_dish5_0.re,
                    +Γ3_dish5_0.im,
                    -Γ3_dish5_0.im,
                    +Γ3_dish5_0.re,
                    +Γ3_dish5_1.re,
                    +Γ3_dish5_1.im,
                    -Γ3_dish5_1.im,
                    +Γ3_dish5_1.re,
                )
            end
        end,
    )

    apply!(
        emitter, :Γ3_cplx_in_0_cplx_0 => layout_Γ3_registers, :(Float16x2(Γ3_dish5_0_cplx_in_0_cplx_0, Γ3_dish5_1_cplx_in_0_cplx_0))
    )
    apply!(
        emitter, :Γ3_cplx_in_0_cplx_1 => layout_Γ3_registers, :(Float16x2(Γ3_dish5_0_cplx_in_0_cplx_1, Γ3_dish5_1_cplx_in_0_cplx_1))
    )
    apply!(
        emitter, :Γ3_cplx_in_1_cplx_0 => layout_Γ3_registers, :(Float16x2(Γ3_dish5_0_cplx_in_1_cplx_0, Γ3_dish5_1_cplx_in_1_cplx_0))
    )
    apply!(
        emitter, :Γ3_cplx_in_1_cplx_1 => layout_Γ3_registers, :(Float16x2(Γ3_dish5_0_cplx_in_1_cplx_1, Γ3_dish5_1_cplx_in_1_cplx_1))
    )

    merge!(emitter, :Γ3_cplx_in_0, [:Γ3_cplx_in_0_cplx_0, :Γ3_cplx_in_0_cplx_1], Cplx(:cplx, 1, 2) => Register(:cplx, 1, 2))
    merge!(emitter, :Γ3_cplx_in_1, [:Γ3_cplx_in_1_cplx_0, :Γ3_cplx_in_1_cplx_1], Cplx(:cplx, 1, 2) => Register(:cplx, 1, 2))

    merge!(emitter, :Γ3, [:Γ3_cplx_in_0, :Γ3_cplx_in_1], Cplx(:cplx_in, 1, 2) => Register(:cplx_in, 1, 2))

    # Calculate Γ4

    # (27)
    layout_Γ4_registers = Layout([
        FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
        # Cplx(:cplx, 1, 2) => Register(:cplx, 1, 2),
        Dish(:dish, 1, 2) => Thread(:thread, 2, 2),
        Dish(:dish, 2, 2) => Thread(:thread, 1, 2),
        Dish(:dish, 4, 2) => SIMD(:simd, 16, 2),
        BeamP(:beamP, 1, 2) => Register(:beamP, 1, 2), # dummy index
        BeamP(:beamP, 2, 2) => Register(:beamP, 2, 2), # dummy index
        BeamP(:beamP, 4, 2) => Register(:beamP, 4, 2), # dummy index
        BeamP(:beamP, 8, 2) => Thread(:thread, 4, 2),
        BeamP(:beamP, 16, 2) => Thread(:thread, 8, 2),
        BeamP(:beamP, 32, 2) => Thread(:thread, 16, 2),
    ])
    push!(
        emitter.statements,
        quote
            (Γ4_dish2_0_cplx_0, Γ4_dish2_0_cplx_1, Γ4_dish2_1_cplx_0, Γ4_dish2_1_cplx_1) = let
                # (5): Γ = exp(2π * im * q * n / 512)
                # (11): q = {q_j} for j in [3, 4, 5]
                #       n = {n_k} for k in [3, 4, 5]
                # (27)
                thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, $num_threads)
                thread0 = (thread ÷ 1i32 % 2i32)
                thread1 = (thread ÷ 2i32 % 2i32)
                thread2 = (thread ÷ 4i32 % 2i32)
                thread3 = (thread ÷ 8i32 % 2i32)
                thread4 = (thread ÷ 16i32 % 2i32)
                dish0 = thread1
                dish1 = thread0
                dish2_0 = 0i32
                dish2_1 = 1i32
                beamp3 = thread2
                beamp4 = thread3
                beamp5 = thread4
                n_dish2_0 = dish0 * 1i32 + dish1 * 2i32 + dish2_0 * 4i32
                n_dish2_1 = dish0 * 1i32 + dish1 * 2i32 + dish2_1 * 4i32
                q = beamp3 * 8i32 + beamp4 * 16i32 + beamp5 * 32i32
                Γ4_dish2_0 = cispi(q * n_dish2_0 % 512i32 * Float32(2 / 512))
                Γ4_dish2_1 = cispi(q * n_dish2_1 % 512i32 * Float32(2 / 512))
                (
                    Γ4_dish2_0.re,
                    Γ4_dish2_0.im,
                    Γ4_dish2_1.re,
                    Γ4_dish2_1.im,
                    Γ4_dish2_1.re,
                    Γ4_dish2_0.im,
                    Γ4_dish2_0.re,
                    Γ4_dish2_1.im,
                    Γ4_dish2_1.re,
                )
            end
        end,
    )

    apply!(emitter, :Γ4_cplx_0 => layout_Γ4_registers, :(Float16x2(Γ4_dish2_0_cplx_0, Γ4_dish2_1_cplx_0)))
    apply!(emitter, :Γ4_cplx_1 => layout_Γ4_registers, :(Float16x2(Γ4_dish2_0_cplx_1, Γ4_dish2_1_cplx_1)))

    merge!(emitter, :Γ4, [:Γ4_cplx_0, :Γ4_cplx_1], Cplx(:cplx, 1, 2) => Register(:cplx, 1, 2))

    # Calculate Γ5

    # (28)
    layout_Γ5_registers = Layout([
        FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
        # Cplx(:cplx_in, 1, 2) => Register(:cplx_in, 1, 2),
        Dish(:dish, 1, 2) => Thread(:thread, 2, 2),
        Dish(:dish, 2, 2) => Thread(:thread, 1, 2),
        Dish(:dish, 4, 2) => SIMD(:simd, 16, 2),
        # Cplx(:cplx, 1, 2) => Register(:cplx, 1, 2),
        BeamP(:beamP, 1, 2) => Register(:beamP, 1, 2), # dummy index
        BeamP(:beamP, 2, 2) => Register(:beamP, 2, 2), # dummy index
        BeamP(:beamP, 4, 2) => Register(:beamP, 4, 2), # dummy index
        BeamP(:beamP, 64, 2) => Thread(:thread, 4, 2),
        BeamP(:beamP, 128, 2) => Thread(:thread, 8, 2),
        BeamP(:beamP, 256, 2) => Thread(:thread, 16, 2),
    ])
    push!(
        emitter.statements,
        quote
            (
                Γ5_dish2_0_cplx_in_0_cplx_0,
                Γ5_dish2_0_cplx_in_1_cplx_0,
                Γ5_dish2_0_cplx_in_0_cplx_1,
                Γ5_dish2_0_cplx_in_1_cplx_1,
                Γ5_dish2_1_cplx_in_0_cplx_0,
                Γ5_dish2_1_cplx_in_1_cplx_0,
                Γ5_dish2_1_cplx_in_0_cplx_1,
                Γ5_dish2_1_cplx_in_1_cplx_1,
            ) = let
                # (5): Γ = exp(2π * im * q * n / 512)
                # (12): q = {q_j} for j in [0, 1, 2]
                #       n = {n_k} for k in [6, 7, 8]
                # (28)
                thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, $num_threads)
                thread0 = (thread ÷ 1i32 % 2i32)
                thread1 = (thread ÷ 2i32 % 2i32)
                thread2 = (thread ÷ 4i32 % 2i32)
                thread3 = (thread ÷ 8i32 % 2i32)
                thread4 = (thread ÷ 16i32 % 2i32)
                dish0 = thread1
                dish1 = thread0
                dish2_0 = 0i32
                dish2_1 = 1i32
                beamp6 = thread2
                beamp7 = thread3
                beamp8 = thread4
                n_dish2_0 = dish0 * 1i32 + dish1 * 2i32 + dish2_0 * 4i32
                n_dish2_1 = dish0 * 1i32 + dish1 * 2i32 + dish2_1 * 4i32
                q = beamp6 * 64i32 + beamp7 * 128i32 + beamp8 * 256i32
                Γ5_dish2_0 = cispi(q * n_dish2_0 % 512i32 * Float32(2 / 512))
                Γ5_dish2_1 = cispi(q * n_dish2_1 % 512i32 * Float32(2 / 512))
                (
                    +Γ5_dish2_0.re,
                    +Γ5_dish2_0.im,
                    -Γ5_dish2_0.im,
                    +Γ5_dish2_0.re,
                    +Γ5_dish2_1.re,
                    +Γ5_dish2_1.im,
                    -Γ5_dish2_1.im,
                    +Γ5_dish2_1.re,
                )
            end
        end,
    )

    apply!(
        emitter, :Γ5_cplx_in_0_cplx_0 => layout_Γ5_registers, :(Float16x2(Γ5_dish2_0_cplx_in_0_cplx_0, Γ5_dish2_1_cplx_in_0_cplx_0))
    )
    apply!(
        emitter, :Γ5_cplx_in_0_cplx_1 => layout_Γ5_registers, :(Float16x2(Γ5_dish2_0_cplx_in_0_cplx_1, Γ5_dish2_1_cplx_in_0_cplx_1))
    )
    apply!(
        emitter, :Γ5_cplx_in_1_cplx_0 => layout_Γ5_registers, :(Float16x2(Γ5_dish2_0_cplx_in_1_cplx_0, Γ5_dish2_1_cplx_in_1_cplx_0))
    )
    apply!(
        emitter, :Γ5_cplx_in_1_cplx_1 => layout_Γ5_registers, :(Float16x2(Γ5_dish2_0_cplx_in_1_cplx_1, Γ5_dish2_1_cplx_in_1_cplx_1))
    )

    merge!(emitter, :Γ5_cplx_in_0, [:Γ5_cplx_in_0_cplx_0, :Γ5_cplx_in_0_cplx_1], Cplx(:cplx, 1, 2) => Register(:cplx, 1, 2))
    merge!(emitter, :Γ5_cplx_in_1, [:Γ5_cplx_in_1_cplx_0, :Γ5_cplx_in_1_cplx_1], Cplx(:cplx, 1, 2) => Register(:cplx, 1, 2))

    merge!(emitter, :Γ5, [:Γ5_cplx_in_0, :Γ5_cplx_in_1], Cplx(:cplx_in, 1, 2) => Register(:cplx_in, 1, 2))

    # Load W

    layout_W_registers = Layout([
        FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
        Cplx(:cplx, 1, C) => SIMD(:simd, 16, 2),
        Dish(:dish, 1, 2) => Register(:dish, 1, 2),
        Dish(:dish, 2, 2) => Register(:dish, 2, 2),
        Dish(:dish, 4, 2) => Register(:dish, 4, 2),
        Dish(:dish, 8, 2) => Thread(:thread, 16, 2),
        Dish(:dish, 16, 2) => Thread(:thread, 8, 2),
        Dish(:dish, 32, 2) => Thread(:thread, 4, 2),
        Dish(:dish, 64, 2) => Thread(:thread, 2, 2),
        Dish(:dish, 128, 2) => Thread(:thread, 1, 2),
        Dish(:dish, 256, 4) => Warp(:warp, 1, 4),
        Polr(:polr, 1, P) => Warp(:warp, 4, 2),
        Freq(:freq, 1, Fbar) => Block(:block, 1, Fbar),
    ])
    load!(emitter, :W => layout_W_registers, :W_memory => layout_W_memory)

    # Main loop

    loop!(emitter, Time(:time, Tds, fld(Ttilde, Tds)) => Loop(:time_outer, Tds, fld(Ttilde, Tds))) do emitter
        # Note: This layout is very inefficient for writing to global memory
        layout_I_registers = Layout([
            FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
            BeamP(:beamP, 1, 2) => SIMD(:simd, 16, 2),
            BeamP(:beamP, 2, 2) => Warp(:warp, 1, 2),
            BeamP(:beamP, 4, 2) => Warp(:warp, 2, 2),
            BeamP(:beamP, 8, 2) => Warp(:warp, 4, 2),
            BeamP(:beamP, 16, 2) => Thread(:thread, 1, 2),
            BeamP(:beamP, 32, 2) => Thread(:thread, 2, 2),
            BeamP(:beamP, 64, 2) => Thread(:thread, 4, 2),
            BeamP(:beamP, 128, 2) => Thread(:thread, 8, 2),
            BeamP(:beamP, 256, 2) => Thread(:thread, 16, 2),
            BeamQ(:beamQ, 1, 2) => Register(:beamQ, 1, 2),
            BeamQ(:beamQ, 2, 2) => Register(:beamQ, 2, 2),
            BeamQ(:beamQ, 4, 2) => Register(:beamQ, 4, 2),
            Freq(:freq, 1, Fbar) => Block(:block, 1, Fbar),
            Time(:time, Tds, fld(Ttilde, Tds)) => Loop(:time_outer, Tds, fld(Ttilde, Tds)),
        ])
        apply!(emitter, :I => layout_I_registers, :(zero(Float16x2)))

        loop!(emitter, Time(:time, Treg, idiv(Tds, Treg)) => Loop(:time_inner, 1, idiv(Tds, Treg))) do emitter

            # Load E

            @assert D == 1024
            @assert P == 2
            @assert W == 8

            if Treg == 1
                layout_E_registers = Layout([
                    IntValue(:intvalue, 1, 4) => SIMD(:simd, 1, 4),
                    Cplx(:cplx, 1, C) => SIMD(:simd, 4, 2),
                    Dish(:dish, 1, 4) => SIMD(:simd, 8, 4),
                    Dish(:dish, 4, 2) => Register(:dish, 4, 2),
                    Dish(:dish, 8, 2) => Thread(:thread, 16, 2),
                    Dish(:dish, 16, 2) => Thread(:thread, 8, 2),
                    Dish(:dish, 32, 2) => Thread(:thread, 4, 2),
                    Dish(:dish, 64, 2) => Thread(:thread, 2, 2),
                    Dish(:dish, 128, 2) => Thread(:thread, 1, 2),
                    Dish(:dish, 256, 4) => Warp(:warp, 1, 4),
                    Polr(:polr, 1, P) => Warp(:warp, 4, 2),
                    Freq(:freq, 1, Fbar) => Block(:block, 1, Fbar),
                    Time(:time, 1, Tds) => Loop(:time_inner, 1, Tds),
                    Time(:time, Tds, fld(Ttilde, Tds)) => Loop(:time_outer, Tds, fld(Ttilde, Tds)),
                ])
                load!(
                    emitter,
                    :E => layout_E_registers,
                    :E_memory => layout_E_memory;
                    align=8,
                    postprocess=addr -> :(
                        let
                            offset = $(shrinkmul(idiv(D, 4) * P * Fbar, :Tbarmin, Tbar))
                            length = $(shrink(idiv(D, 4) * P * Fbar * Tbar))
                            mod($addr + offset, length)
                        end
                    ),
                )

            elseif Treg == 2
                layout_E_registers = Layout([
                    IntValue(:intvalue, 1, 4) => SIMD(:simd, 1, 4),
                    Cplx(:cplx, 1, C) => SIMD(:simd, 4, 2),
                    Dish(:dish, 1, 4) => SIMD(:simd, 8, 4),
                    Dish(:dish, 4, 2) => Register(:dish, 4, 2),
                    Dish(:dish, 8, 2) => Register(:time, 1, Treg),
                    Dish(:dish, 16, 2) => Thread(:thread, 8, 2),
                    Dish(:dish, 32, 2) => Thread(:thread, 4, 2),
                    Dish(:dish, 64, 2) => Thread(:thread, 2, 2),
                    Dish(:dish, 128, 2) => Thread(:thread, 1, 2),
                    Dish(:dish, 256, 4) => Warp(:warp, 1, 4),
                    Polr(:polr, 1, P) => Warp(:warp, 4, 2),
                    Freq(:freq, 1, Fbar) => Block(:block, 1, Fbar),
                    Time(:time, 1, Treg) => Thread(:thread, 16, 2),
                    Time(:time, Treg, idiv(Tds, Treg)) => Loop(:time_inner, 1, idiv(Tds, Treg)),
                    Time(:time, Tds, fld(Ttilde, Tds)) => Loop(:time_outer, Tds, fld(Ttilde, Tds)),
                ])
                load!(
                    emitter,
                    :E0 => layout_E_registers,
                    :E_memory => layout_E_memory;
                    align=16,
                    postprocess=addr -> :(
                        let
                            offset = $(shrinkmul(idiv(D, 4) * P * Fbar, :Tbarmin, Tbar))
                            length = $(shrink(idiv(D, 4) * P * Fbar * Tbar))
                            mod($addr + offset, length)
                        end
                    ),
                )
                permute!(emitter, :E, :E0, Dish(:dish, 8, 2), Time(:time, 1, 2))

            else
                @assert false
            end

            # Convert to float16

            widen2!(
                emitter,
                :X0,
                :E,
                SIMD(:simd, 4, 2) => Register(:dish, 2, 2),
                SIMD(:simd, 8, 2) => Register(:dish, 1, 2);
                newtype=FloatValue,
            )
            permute!(emitter, :X1, :X0, Cplx(:cplx, 1, 2), Dish(:dish, 2, 2))

            # Scale by input gain

            apply!(emitter, :X, [:X1, :W], (X1, W) -> :(complex_mul($W, $X1)))

            # (13)
            layout_X_registers = Layout([
                FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
                Cplx(:cplx, 1, C) => SIMD(:simd, 16, 2),
                Dish(:dish, 1, 2) => Register(:dish, 1, 2),
                Dish(:dish, 2, 2) => Register(:dish, 2, 2),
                Dish(:dish, 4, 2) => Register(:dish, 4, 2),
                Dish(:dish, 8, 2) => Thread(:thread, 16, 2),
                Dish(:dish, 16, 2) => Thread(:thread, 8, 2),
                Dish(:dish, 32, 2) => Thread(:thread, 4, 2),
                Dish(:dish, 64, 2) => Thread(:thread, 2, 2),
                Dish(:dish, 128, 2) => Thread(:thread, 1, 2),
                Dish(:dish, 256, 4) => Warp(:warp, 1, 4),
                Polr(:polr, 1, P) => Warp(:warp, 4, 2),
                Freq(:freq, 1, Fbar) => Block(:block, 1, Fbar),
                Time(:time, 1, Treg) => Register(:time, 1, Treg),
                Time(:time, Treg, idiv(Tds, Treg)) => Loop(:time_inner, 1, idiv(Tds, Treg)),
                Time(:time, Tds, fld(Ttilde, Tds)) => Loop(:time_outer, Tds, fld(Ttilde, Tds)),
            ])
            @assert emitter.environment[:X] == layout_X_registers

            # Step 1: (8)

            # (14)
            layout_Z1_registers = Layout([
                FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
                Cplx(:cplx, 1, C) => Register(:cplx, 1, C),
                Dish(:dish, 1, 2) => Register(:dish, 1, 2),
                Dish(:dish, 2, 2) => Register(:dish, 2, 2),
                Dish(:dish, 4, 2) => Register(:dish, 4, 2),
                Dish(:dish, 8, 2) => Thread(:thread, 2, 2),
                Dish(:dish, 16, 2) => Thread(:thread, 1, 2),
                Dish(:dish, 32, 2) => SIMD(:simd, 16, 2),
                BeamP(:beamP, 1, 2) => Thread(:thread, 4, 2),
                BeamP(:beamP, 2, 2) => Thread(:thread, 8, 2),
                BeamP(:beamP, 4, 2) => Thread(:thread, 16, 2),
                Dish(:dish, 256, 4) => Warp(:warp, 1, 4),
                Polr(:polr, 1, P) => Warp(:warp, 4, 2),
                Freq(:freq, 1, Fbar) => Block(:block, 1, Fbar),
                Time(:time, 1, Treg) => Register(:time, 1, Treg),
                Time(:time, Treg, idiv(Tds, Treg)) => Loop(:time_inner, 1, idiv(Tds, Treg)),
                Time(:time, Tds, fld(Ttilde, Tds)) => Loop(:time_outer, Tds, fld(Ttilde, Tds)),
            ])
            apply!(emitter, :Z1 => layout_Z1_registers, :(zero(Float16x2)))
            let
                layout = copy(emitter.environment[:X])
                k = Cplx(:cplx, 1, C)
                k′ = Cplx(:cplx_in, 1, C)
                v = layout[k]
                delete!(layout, k)
                layout[k′] = v
                emitter.environment[:X] = layout
            end
            mma_is = [BeamP(:beamP, 1, 2), BeamP(:beamP, 2, 2), BeamP(:beamP, 4, 2), Cplx(:cplx, 1, 2)]
            mma_js = [Cplx(:cplx_in, 1, 2), Dish(:dish, 128, 2), Dish(:dish, 64, 2)]
            mma_ks = [Dish(:dish, 32, 2), Dish(:dish, 16, 2), Dish(:dish, 8, 2)]
            mma_row_col_m16n8k8_f16!(emitter, :Z1, :Γ1 => (mma_is, mma_js), :X => (mma_js, mma_ks), :Z1 => (mma_is, mma_ks))

            # Step 2: (9)

            split!(emitter, [:Γ2_re, :Γ2_im], :Γ2, Register(:cplx, 1, 2))
            split!(emitter, [:Z1_re, :Z1_im], :Z1, Register(:cplx, 1, 2))
            apply!(
                emitter,
                :Z2_re,
                [:Z1_re, :Z1_im, :Γ2_re, :Γ2_im],
                (Z1_re, Z1_im, Γ2_re, Γ2_im) -> :(muladd($Γ2_re, $Z1_re, -$Γ2_im * $Z1_im)),
            )
            apply!(
                emitter,
                :Z2_im,
                [:Z1_re, :Z1_im, :Γ2_re, :Γ2_im],
                (Z1_re, Z1_im, Γ2_re, Γ2_im) -> :(muladd($Γ2_re, $Z1_im, +$Γ2_im * $Z1_re)),
            )
            merge!(emitter, :Z2, [:Z2_re, :Z2_im], Cplx(:cplx, 1, 2) => Register(:cplx, 1, 2))

            # (14)
            layout_Z2_registers = layout_Z1_registers
            @assert emitter.environment[:Z2] == layout_Z2_registers

            # Step 3: (10)

            # (15)
            layout_Z3′_registers = Layout([
                FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
                Cplx(:cplx, 1, C) => Register(:cplx, 1, C),
                Dish(:dish, 1, 2) => Register(:dish, 1, 2),
                Dish(:dish, 2, 2) => Register(:dish, 2, 2),
                Dish(:dish, 4, 2) => Register(:dish, 4, 2),
                BeamP(:beamP, 1, 2) => SIMD(:simd, 16, 2),
                BeamP(:beamP, 2, 2) => Thread(:thread, 1, 2),
                BeamP(:beamP, 4, 2) => Thread(:thread, 2, 2),
                BeamP(:beamP, 8, 2) => Thread(:thread, 4, 2),
                BeamP(:beamP, 16, 2) => Thread(:thread, 8, 2),
                BeamP(:beamP, 32, 2) => Thread(:thread, 16, 2),
                Dish(:dish, 256, 4) => Warp(:warp, 1, 4),
                Polr(:polr, 1, P) => Warp(:warp, 4, 2),
                Freq(:freq, 1, Fbar) => Block(:block, 1, Fbar),
                Time(:time, 1, Treg) => Register(:time, 1, Treg),
                Time(:time, Treg, idiv(Tds, Treg)) => Loop(:time_inner, 1, idiv(Tds, Treg)),
                Time(:time, Tds, fld(Ttilde, Tds)) => Loop(:time_outer, Tds, fld(Ttilde, Tds)),
            ])
            apply!(emitter, :Z3′ => layout_Z3′_registers, :(zero(Float16x2)))
            split!(emitter, [:Z2_re, :Z2_im], :Z2, Register(:cplx, 1, 2))
            merge!(emitter, :Z2, [:Z2_re, :Z2_im], Cplx(:cplx_in, 1, 2) => Register(:cplx, 1, 2))
            mma_is = [BeamP(:beamP, 8, 2), BeamP(:beamP, 16, 2), BeamP(:beamP, 32, 2), Cplx(:cplx, 1, 2)]
            mma_js = [Dish(:dish, 32, 2), Dish(:dish, 16, 2), Dish(:dish, 8, 2), Cplx(:cplx_in, 1, 2)]
            mma_ks = [BeamP(:beamP, 1, 2), BeamP(:beamP, 2, 2), BeamP(:beamP, 4, 2)]
            mma_row_col_m16n8k16_f16!(emitter, :Z3′, :Γ3 => (mma_is, mma_js), :Z2 => (mma_js, mma_ks), :Z3′ => (mma_is, mma_ks))

            # (16)
            layout_Z3_registers = Layout([
                FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
                Cplx(:cplx, 1, C) => Register(:cplx, 1, C),
                Dish(:dish, 1, 2) => Thread(:thread, 2, 2),
                Dish(:dish, 2, 2) => Thread(:thread, 1, 2),
                Dish(:dish, 4, 2) => SIMD(:simd, 16, 2),
                BeamP(:beamP, 1, 2) => Register(:beamP, 1, 2),
                BeamP(:beamP, 2, 2) => Register(:beamP, 2, 2),
                BeamP(:beamP, 4, 2) => Register(:beamP, 4, 2),
                BeamP(:beamP, 8, 2) => Thread(:thread, 4, 2),
                BeamP(:beamP, 16, 2) => Thread(:thread, 8, 2),
                BeamP(:beamP, 32, 2) => Thread(:thread, 16, 2),
                Dish(:dish, 256, 4) => Warp(:warp, 1, 4),
                Polr(:polr, 1, P) => Warp(:warp, 4, 2),
                Freq(:freq, 1, Fbar) => Block(:block, 1, Fbar),
                Time(:time, 1, Treg) => Register(:time, 1, Treg),
                Time(:time, Treg, idiv(Tds, Treg)) => Loop(:time_inner, 1, idiv(Tds, Treg)),
                Time(:time, Tds, fld(Ttilde, Tds)) => Loop(:time_outer, Tds, fld(Ttilde, Tds)),
            ])

            # initial:
            #     dish0 dish1 dish2 beamp0 beamp1 beamp2
            #     reg0  reg1  reg2  simd   thr0   thr1

            permute!(emitter, :Z3′1, :Z3′, Register(:dish, 4, 2), SIMD(:simd, 16, 2))
            split!(emitter, [:Z3′1_beam0_0, :Z3′1_beam0_1], :Z3′1, Register(:dish, 4, 2))
            merge!(emitter, :Z3′1, [:Z3′1_beam0_0, :Z3′1_beam0_1], BeamP(:beamP, 1, 2) => Register(:beamP, 1, 2))
            #     dish0 dish1 dish2 beamp0 beamp1 beamp2
            #     reg0  reg1  simd  reg2   thr0   thr1

            permute!(emitter, :Z3′2, :Z3′1, Register(:dish, 2, 2), Thread(:thread, 1, 2))
            split!(emitter, [:Z3′2_beam1_0, :Z3′2_beam1_1], :Z3′2, Register(:dish, 2, 2))
            merge!(emitter, :Z3′2, [:Z3′2_beam1_0, :Z3′2_beam1_1], BeamP(:beamP, 2, 2) => Register(:beamP, 2, 2))
            #     dish0 dish1 dish2 beamp0 beamp1 beamp2
            #     reg0  thr0  simd  reg2   reg1   thr1

            permute!(emitter, :Z3, :Z3′2, Register(:dish, 1, 2), Thread(:thread, 2, 2))
            split!(emitter, [:Z3_beam2_0, :Z3_beam2_1], :Z3, Register(:dish, 1, 2))
            merge!(emitter, :Z3, [:Z3_beam2_0, :Z3_beam2_1], BeamP(:beamP, 4, 2) => Register(:beamP, 4, 2))
            #     dish0 dish1 dish2 beamp0 beamp1 beamp2
            #     thr1  thr0  simd  reg2   reg1   reg0

            # final:
            #     dish0 dish1 dish2 beamp0 beamp1 beamp2
            #     thr1  thr0  simd  reg?   reg?   reg?
            @assert emitter.environment[:Z3] == layout_Z3_registers

            # Step 4: (11)

            split!(emitter, [:Γ4_re, :Γ4_im], :Γ4, Register(:cplx, 1, 2))
            split!(emitter, [:Z3_re, :Z3_im], :Z3, Register(:cplx, 1, 2))
            apply!(
                emitter,
                :Z4_re,
                [:Z3_re, :Z3_im, :Γ4_re, :Γ4_im],
                (Z3_re, Z3_im, Γ4_re, Γ4_im) -> :(muladd($Γ4_re, $Z3_re, -$Γ4_im * $Z3_im)),
            )
            apply!(
                emitter,
                :Z4_im,
                [:Z3_re, :Z3_im, :Γ4_re, :Γ4_im],
                (Z3_re, Z3_im, Γ4_re, Γ4_im) -> :(muladd($Γ4_re, $Z3_im, +$Γ4_im * $Z3_re)),
            )
            merge!(emitter, :Z4, [:Z4_re, :Z4_im], Cplx(:cplx, 1, 2) => Register(:cplx, 1, 2))

            # (16)
            layout_Z4_registers = layout_Z3_registers
            @assert emitter.environment[:Z4] == layout_Z4_registers

            # Step 5: (12)

            # (17)
            layout_Y′_registers = Layout([
                FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
                Cplx(:cplx, 1, C) => Register(:cplx, 1, C),
                BeamP(:beamP, 1, 2) => Register(:beamP, 1, 2),
                BeamP(:beamP, 2, 2) => Register(:beamP, 2, 2),
                BeamP(:beamP, 4, 2) => Register(:beamP, 4, 2),
                BeamP(:beamP, 8, 2) => SIMD(:simd, 16, 2),
                BeamP(:beamP, 16, 2) => Thread(:thread, 1, 2),
                BeamP(:beamP, 32, 2) => Thread(:thread, 2, 2),
                BeamP(:beamP, 64, 2) => Thread(:thread, 4, 2),
                BeamP(:beamP, 128, 2) => Thread(:thread, 8, 2),
                BeamP(:beamP, 256, 2) => Thread(:thread, 16, 2),
                Dish(:dish, 256, 4) => Warp(:warp, 1, 4),
                Polr(:polr, 1, P) => Warp(:warp, 4, 2),
                Freq(:freq, 1, Fbar) => Block(:block, 1, Fbar),
                Time(:time, 1, Treg) => Register(:time, 1, Treg),
                Time(:time, Treg, idiv(Tds, Treg)) => Loop(:time_inner, 1, idiv(Tds, Treg)),
                Time(:time, Tds, fld(Ttilde, Tds)) => Loop(:time_outer, Tds, fld(Ttilde, Tds)),
            ])
            apply!(emitter, :Y′ => layout_Y′_registers, :(zero(Float16x2)))
            split!(emitter, [:Z4_re, :Z4_im], :Z4, Register(:cplx, 1, 2))
            merge!(emitter, :Z4, [:Z4_re, :Z4_im], Cplx(:cplx_in, 1, 2) => Register(:cplx, 1, 2))
            mma_is = [BeamP(:beamP, 64, 2), BeamP(:beamP, 128, 2), BeamP(:beamP, 256, 2), Cplx(:cplx, 1, 2)]
            mma_js = [Dish(:dish, 4, 2), Dish(:dish, 2, 2), Dish(:dish, 1, 2), Cplx(:cplx_in, 1, 2)]
            mma_ks = [BeamP(:beamP, 8, 2), BeamP(:beamP, 16, 2), BeamP(:beamP, 32, 2)]
            mma_row_col_m16n8k16_f16!(emitter, :Y′, :Γ5 => (mma_is, mma_js), :Z4 => (mma_js, mma_ks), :Y′ => (mma_is, mma_ks))

            # (18)
            layout_Y_registers = Layout([
                FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
                # Cplx(:cplx, 1, C) => SIMD(:simd, 16, 2),
                # BeamP(:beamP, 1, 2) => Register(:beamP, 1, 2),
                Cplx(:cplx, 1, C) => Register(:cplx, 1, 2),
                BeamP(:beamP, 1, 2) => SIMD(:simd, 16, 2),
                BeamP(:beamP, 2, 2) => Register(:beamP, 2, 2),
                BeamP(:beamP, 4, 2) => Register(:beamP, 4, 2),
                BeamP(:beamP, 8, 2) => Register(:beamP, 8, 2),
                BeamP(:beamP, 16, 2) => Thread(:thread, 1, 2),
                BeamP(:beamP, 32, 2) => Thread(:thread, 2, 2),
                BeamP(:beamP, 64, 2) => Thread(:thread, 4, 2),
                BeamP(:beamP, 128, 2) => Thread(:thread, 8, 2),
                BeamP(:beamP, 256, 2) => Thread(:thread, 16, 2),
                Dish(:dish, 256, 4) => Warp(:warp, 1, 4),
                Polr(:polr, 1, P) => Warp(:warp, 4, 2),
                Freq(:freq, 1, Fbar) => Block(:block, 1, Fbar),
                Time(:time, 1, Treg) => Register(:time, 1, Treg),
                Time(:time, Treg, idiv(Tds, Treg)) => Loop(:time_inner, 1, idiv(Tds, Treg)),
                Time(:time, Tds, fld(Ttilde, Tds)) => Loop(:time_outer, Tds, fld(Ttilde, Tds)),
            ])

            # permute!(emitter, :Y, :Y′, Register(:cplx, 1, 2), SIMD(:simd, 16, 2))
            # split!(emitter, [:Y_beam3_0, :Y_beam3_1], :Y, Register(:cplx, 1, 2))
            # merge!(emitter, :Y, [:Y_beam3_0, :Y_beam3_1], BeamP(:beamP, 8, 2) => Register(:beamP, 8, 2))
            permute!(emitter, :Y, :Y′, Register(:beamP, 1, 2), SIMD(:simd, 16, 2))
            split!(emitter, [:Y_beam3_0, :Y_beam3_1], :Y, Register(:beamP, 1, 2))
            merge!(emitter, :Y, [:Y_beam3_0, :Y_beam3_1], BeamP(:beamP, 8, 2) => Register(:beamP, 8, 2))

            @assert emitter.environment[:Y] == layout_Y_registers

            # Transpose Y via shared memory
            # TODO: avoid bank conflicts

            store!(emitter, :Y_shared => layout_Y_shared, :Y)
            sync_threads!(emitter)

            layout_X_registers = Layout([
                FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
                Cplx(:cplx, 1, C) => Register(:cplx, 1, 2),
                BeamP(:beamP, 1, 2) => SIMD(:simd, 16, 2),
                BeamP(:beamP, 2, 2) => Warp(:warp, 1, 2),
                BeamP(:beamP, 4, 2) => Warp(:warp, 2, 2),
                BeamP(:beamP, 8, 2) => Warp(:warp, 4, 2),
                BeamP(:beamP, 16, 2) => Thread(:thread, 1, 2),
                BeamP(:beamP, 32, 2) => Thread(:thread, 2, 2),
                BeamP(:beamP, 64, 2) => Thread(:thread, 4, 2),
                BeamP(:beamP, 128, 2) => Thread(:thread, 8, 2),
                BeamP(:beamP, 256, 2) => Thread(:thread, 16, 2),
                Dish(:dish, 256, 4) => Register(:dish, 256, 4),
                Polr(:polr, 1, P) => Register(:polr, 1, P),
                Freq(:freq, 1, Fbar) => Block(:block, 1, Fbar),
                Time(:time, 1, Treg) => Register(:time, 1, Treg),
                Time(:time, Treg, idiv(Tds, Treg)) => Loop(:time_inner, 1, idiv(Tds, Treg)),
                Time(:time, Tds, fld(Ttilde, Tds)) => Loop(:time_outer, Tds, fld(Ttilde, Tds)),
            ])
            load!(emitter, :X => layout_X_registers, :Y_shared => layout_Y_shared)

            # FFT in N direction
            split!(emitter, [:X_re, :X_im], :X, Cplx(:cplx, 1, 2))
            split!(emitter, [:X_dish0_re, :X_dish1_re, :X_dish2_re, :X_dish3_re], :X_re, Dish(:dish, 256, 4))
            split!(emitter, [:X_dish0_im, :X_dish1_im, :X_dish2_im, :X_dish3_im], :X_im, Dish(:dish, 256, 4))

            layout_Y_registers = Layout([
                FloatValue(:floatvalue, 1, 16) => SIMD(:simd, 1, 16),
                # Cplx(:cplx, 1, C) => Register(:cplx, 1, 2),
                BeamP(:beamP, 1, 2) => SIMD(:simd, 16, 2),
                BeamP(:beamP, 2, 2) => Warp(:warp, 1, 2),
                BeamP(:beamP, 4, 2) => Warp(:warp, 2, 2),
                BeamP(:beamP, 8, 2) => Warp(:warp, 4, 2),
                BeamP(:beamP, 16, 2) => Thread(:thread, 1, 2),
                BeamP(:beamP, 32, 2) => Thread(:thread, 2, 2),
                BeamP(:beamP, 64, 2) => Thread(:thread, 4, 2),
                BeamP(:beamP, 128, 2) => Thread(:thread, 8, 2),
                BeamP(:beamP, 256, 2) => Thread(:thread, 16, 2),
                # BeamQ(:beamQ, 1, 2) => Register(:beamQ, 1, 2), 
                # BeamQ(:beamQ, 2, 2) => Register(:beamQ, 2, 2),
                # BeamQ(:beamQ, 4, 2) => Register(:beamQ, 4, 2),
                Polr(:polr, 1, P) => Register(:polr, 1, P),
                Freq(:freq, 1, Fbar) => Block(:block, 1, Fbar),
                Time(:time, 1, Treg) => Register(:time, 1, Treg),
                Time(:time, Treg, idiv(Tds, Treg)) => Loop(:time_inner, 1, idiv(Tds, Treg)),
                Time(:time, Tds, fld(Ttilde, Tds)) => Loop(:time_outer, Tds, fld(Ttilde, Tds)),
            ])

            function apply_phase(n, X)
                sqrt_half = Float16x2(sqrt(0.5f0), sqrt(0.5f0))
                Xre, Xim = X
                n % 8 == 0 && return :(+$Xre), :(+$Xim)
                n % 8 == 1 && return :(+$sqrt_half * $Xre), :(+$sqrt_half * $Xim)
                n % 8 == 2 && return :(-$Xim), :(+$Xre)
                n % 8 == 3 && return :(-$sqrt_half * $Xim), :(+$sqrt_half * $Xre)
                n % 8 == 4 && return :(-$Xre), :(-$Xim)
                n % 8 == 5 && return :(-$sqrt_half * $Xre), :(-$sqrt_half * $Xim)
                n % 8 == 6 && return :(+$Xim), :(-$Xre)
                n % 8 == 7 && return :(+$sqrt_half * $Xim), :(-$sqrt_half * $Xre)
                @assert false
            end

            function calc_beam(q, dishes, reim)
                return quote
                    $(apply_phase(q * 0, dishes[1])[reim]) +
                    $(apply_phase(q * 1, dishes[2])[reim]) +
                    $(apply_phase(q * 2, dishes[3])[reim]) +
                    $(apply_phase(q * 3, dishes[4])[reim])
                end
            end

            function simplify(expr)
                @label restart
                if expr isa Expr
                    # Is there a sum?
                    if expr.head === :call && length(expr.args) >= 3 && expr.args[1] === :+
                        for n in 2:length(expr.args)
                            expr1 = expr.args[n]
                            # Is there a product?
                            if expr1 isa Expr && expr1.head === :call && length(expr1.args) >= 3 && expr1.args[1] === :*
                                @assert length(expr1.args) == 3
                                expr = Expr(
                                    :call,
                                    :muladd,
                                    expr1.args[2],
                                    expr1.args[3],
                                    Expr(:call, :+, expr.args[2:(n - 1)]..., expr.args[(n + 1):end]...),
                                )
                                @goto restart
                            end
                        end
                    end
                    # recurse
                    return Expr(simplify(expr.head), simplify.(expr.args)...)
                else
                    # do nothing
                    return expr
                end
            end

            for q in 0:7, reim in 1:2
                apply!(
                    emitter,
                    Symbol(:Y_beamQ, q, :_, ("re", "im")[reim]),
                    [:X_dish0_re, :X_dish1_re, :X_dish2_re, :X_dish3_re, :X_dish0_im, :X_dish1_im, :X_dish2_im, :X_dish3_im],
                    (X_dish0_re, X_dish1_re, X_dish2_re, X_dish3_re, X_dish0_im, X_dish1_im, X_dish2_im, X_dish3_im) -> simplify(
                        calc_beam(
                            q,
                            (
                                (X_dish0_re, X_dish0_im),
                                (X_dish1_re, X_dish1_im),
                                (X_dish2_re, X_dish2_im),
                                (X_dish3_re, X_dish3_im),
                            ),
                            reim,
                        ),
                    ),
                )
            end

            merge!(
                emitter,
                :Y_re,
                [:Y_beamQ0_re, :Y_beamQ1_re, :Y_beamQ2_re, :Y_beamQ3_re, :Y_beamQ4_re, :Y_beamQ5_re, :Y_beamQ6_re, :Y_beamQ7_re],
                BeamQ(:beamQ, 1, 8) => Register(:beamQ, 1, 8),
            )
            merge!(
                emitter,
                :Y_im,
                [:Y_beamQ0_im, :Y_beamQ1_im, :Y_beamQ2_im, :Y_beamQ3_im, :Y_beamQ4_im, :Y_beamQ5_im, :Y_beamQ6_im, :Y_beamQ7_im],
                BeamQ(:beamQ, 1, 8) => Register(:beamQ, 1, 8),
            )

            # Calculate intensity

            split!(emitter, [:Y_polr0_re, :Y_polr1_re], :Y_re, Polr(:polr, 1, 2))
            split!(emitter, [:Y_polr0_im, :Y_polr1_im], :Y_im, Polr(:polr, 1, 2))

            if Treg == 1
                apply!(
                    emitter,
                    :I,
                    [:I, :Y_polr0_re, :Y_polr0_im, :Y_polr1_re, :Y_polr1_im],
                    (I, Y_polr0_re, Y_polr0_im, Y_polr1_re, Y_polr1_im) -> :(muladd(
                        $(Float16x2(output_gain, output_gain)),
                        muladd(
                            $Y_polr1_im,
                            $Y_polr1_im,
                            muladd($Y_polr1_re, $Y_polr1_re, muladd($Y_polr0_im, $Y_polr0_im, $Y_polr0_re * $Y_polr0_re)),
                        ),
                        $I,
                    ));
                    ignore=[Time(:time, Treg, idiv(Tds, Treg))],
                )

            elseif Treg == 2
                split!(emitter, [:Y_time0_polr0_re, :Y_time1_polr0_re], :Y_polr0_re, Time(:time, 1, 2))
                split!(emitter, [:Y_time0_polr0_im, :Y_time1_polr0_im], :Y_polr0_im, Time(:time, 1, 2))
                split!(emitter, [:Y_time0_polr1_re, :Y_time1_polr1_re], :Y_polr1_re, Time(:time, 1, 2))
                split!(emitter, [:Y_time0_polr1_im, :Y_time1_polr1_im], :Y_polr1_im, Time(:time, 1, 2))

                apply!(
                    emitter,
                    :I,
                    [
                        :I,
                        :Y_time0_polr0_re,
                        :Y_time0_polr0_im,
                        :Y_time0_polr1_re,
                        :Y_time0_polr1_im,
                        :Y_time1_polr0_re,
                        :Y_time1_polr0_im,
                        :Y_time1_polr1_re,
                        :Y_time1_polr1_im,
                    ],
                    (
                        I,
                        Y_time0_polr0_re,
                        Y_time0_polr0_im,
                        Y_time0_polr1_re,
                        Y_time0_polr1_im,
                        Y_time1_polr0_re,
                        Y_time1_polr0_im,
                        Y_time1_polr1_re,
                        Y_time1_polr1_im,
                    ) -> :(muladd(
                        $(Float16x2(output_gain, output_gain)),
                        muladd(
                            $Y_time1_polr1_im,
                            $Y_time1_polr1_im,
                            muladd(
                                $Y_time1_polr1_re,
                                $Y_time1_polr1_re,
                                muladd(
                                    $Y_time1_polr0_im,
                                    $Y_time1_polr0_im,
                                    muladd(
                                        $Y_time1_polr0_re,
                                        $Y_time1_polr0_re,
                                        muladd(
                                            $Y_time0_polr1_im,
                                            $Y_time0_polr1_im,
                                            muladd(
                                                $Y_time0_polr1_re,
                                                $Y_time0_polr1_re,
                                                muladd($Y_time0_polr0_im, $Y_time0_polr0_im, $Y_time0_polr0_re * $Y_time0_polr0_re),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                        $I,
                    ));
                    ignore=[Time(:time, Treg, idiv(Tds, Treg))],
                )

            else
                @assert false
            end

            sync_threads!(emitter)

            return nothing
        end

        # Store result

        store!(
            emitter,
            :I_memory => layout_I_memory,
            :I;
            # align=16,
            postprocess=addr -> quote
                let
                    offset = $(Int32(M * 2 * N * Fbar)) * Ttildemin
                    length = $(Int32(M * 2 * N * Fbar * Ttilde))
                    mod($addr + offset, length)
                end
            end,
        )

        return nothing
    end

    # Done.

    apply!(emitter, :info => layout_info_registers, 0i32)
    store!(emitter, :info_memory => layout_info_memory, :info)

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

println("[Creating chimefrb kernel...]")
const chimefrb_kernel = make_chimefrb_kernel()
println("[Done creating chimefrb kernel]")

@eval function chimefrb(
    Tbarmin::Int32, Tbarmax::Int32, Ttildemin::Int32, Ttildemax::Int32, W_memory, E_memory, I_memory, info_memory
)
    shmem = @cuDynamicSharedMem(UInt8, shmem_bytes, 0)
    Y_shared = reinterpret(Float16x2, shmem)

    $chimefrb_kernel
    return nothing
end

function main(; compile_only::Bool=false, output_kernel::Bool=false, nruns::Int=0, silent::Bool=false)
    !silent && println("CHIME FRB beamformer")

    if output_kernel
        open("output-$card/chimefrb_$(setup)_U$(U).jl", "w") do fh
            println(fh, "# Julia source code for CUDA chimefrb beamformer")
            println(fh, "# This file has been generated automatically by `chimefrb.jl`.")
            println(fh, "# Do not modify this file, your changes will be lost.")
            println(fh)
            println(fh, chimefrb_kernel)
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
    @assert shmem_bytes ≤ 100 * 1024 # NVIDIA A10/A40 have 100 kB shared memory
    kernel = @cuda launch = false minthreads = num_threads * num_warps blocks_per_sm = num_blocks_per_sm chimefrb(
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
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

    W_memory = Array{Float16x2}(undef, M * N * Fbar * P)
    E_memory = Array{Int4x8}(undef, idiv(D, 4) * P * Fbar * Tbar)
    I_memory = Array{Float16x2}(undef, M * 2 * N * Fbar * Ttilde)
    info_memory = Array{Int32}(undef, num_threads * num_warps * num_blocks)

    println("Setting up input data...")
    map!(f -> Float16x2(1, 1), W_memory, W_memory)
    map!(i -> Int4x8(-4, -3, -2, -1, 0, 1, 2, 3), E_memory, E_memory)

    Ttildemin = Int32(0)
    Ttildemax = Int32(fld(fld(Tbar, 4), Tds))

    Tbarmin = Int32(0)
    Tbarmax = Int32(Ttildemax * Tds)

    println("   Fixed kernel parameters:")
    println("       Fbar:   $Fbar   (number of frequencies)")
    println("       Tbar:   $Tbar   (maximum number of input time samples)")
    println("       Ttilde: $Ttilde   (maximum number of output time samples)")
    println("       Tds:    $Tds   (downsampling factor)")
    println("    Dynamic kernel parameters:")
    println("    Tbarmin:   $Tbarmin   (input time samples)")
    println("    Tbarmax:   $Tbarmax")
    println("    Ttildemin: $Ttildemin   (output time samples)")
    println("    Ttildemax: $Ttildemax")

    @assert 0i32 ≤ Tbarmin < Int32(Tbar)
    @assert Tbarmin ≤ Tbarmax < Int32(2 * Tbar)
    @assert (Tbarmax - Tbarmin) % Int32(Tds) == 0i32
    @assert 0i32 ≤ Ttildemin < Int32(Ttilde)
    @assert Ttildemin ≤ Ttildemax < Int32(2 * Ttilde)
    @assert Ttildemax - Ttildemin == (Tbarmax - Tbarmin) ÷ Int32(Tds)

    println("Copying data from CPU to GPU...")
    W_cuda = CuArray(W_memory)
    E_cuda = CuArray(E_memory)
    I_cuda = CUDA.fill(Float16x2(NaN, NaN), length(I_memory))
    info_cuda = CUDA.fill(-1i32, length(info_memory))

    println("Running kernel...")
    kernel(
        Tbarmin,
        Tbarmax,
        Ttildemin,
        Ttildemax,
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
        for i in 1:nruns
            stats = @timed begin
                for run in 1:nruns
                    kernel(
                        Tbarmin,
                        Tbarmax,
                        Ttildemin,
                        Ttildemax,
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
            # All times in msec
            runtime = stats.time / nruns * 1.0e+3
            println("    Kernel run time: $runtime msec")
        end
    end

    println("Copying data back from GPU to CPU...")
    I_memory = Array(I_cuda)
    @assert all(!isnan, (@view I_memory[1:(M * 2 * N * Fbar * Ttildemax)]))
    info_memory = Array(info_cuda)
    @assert all(info_memory .== 0)

    println("Done.")
    return nothing
end

function fix_ptx_kernel()
    ptx = read("output-$card/chimefrb_$(setup)_U$(U).ptx", String)
    ptx = replace(ptx, r".extern .func gpu_([^;]*);"s => s".func gpu_\1.noreturn\n{\n\ttrap;\n}")
    open("output-$card/chimefrb_$(setup)_U$(U).ptx", "w") do fh
        println(fh, "// PTX kernel code for CUDA chimefrb beamformer")
        println(fh, "// This file has been generated automatically by `chimefrb.jl`.")
        println(fh, "// Do not modify this file, your changes will be lost.")
        println(fh)
        write(fh, ptx)
        return nothing
    end
    sass = read("output-$card/chimefrb_$(setup)_U$(U).sass", String)
    open("output-$card/chimefrb_$(setup)_U$(U).sass", "w") do fh
        println(fh, "// SASS kernel code for CUDA chimefrb beamformer")
        println(fh, "// This file has been generated automatically by `chimefrb.jl`.")
        println(fh, "// Do not modify this file, your changes will be lost.")
        println(fh)
        write(fh, sass)
        return nothing
    end
    kernel_symbol = match(r"\s\.globl\s+(\S+)"m, ptx).captures[1]
    open("output-$card/chimefrb_$(setup)_U$(U).yaml", "w") do fh
        println(fh, "# Metadata code for CUDA chimefrb beamformer")
        println(fh, "# This file has been generated automatically by `chimefrb.jl`.")
        println(fh, "# Do not modify this file, your changes will be lost.")
        println(fh)
        print(
            fh,
            """
    --- !<tag:chord-observatory.ca/x-engine/kernel-description-1.0.0>
    kernel-description:
      name: "chimefrb"
      description: "CHIMEFRB beamformer"
      design-parameters:
        beam-layout: [$(2*M), $(2*N)]
        dish-layout: [$M, $N]
        downsampling-factor: $Tds
        number-of-complex-components: $C
        number-of-dishes: $D
        number-of-frequencies: $Fbar
        number-of-polarizations: $P
        number-of-timesamples: $Tbar
        output-gain: $output_gain
        sampling-time-μsec: $sampling_time_μsec
        upchannelization-factor: $U
      compile-parameters:
        minthreads: $(num_threads * num_warps)
        blocks_per_sm: $num_blocks_per_sm
      call-parameters:
        threads: [$num_threads, $num_warps]
        blocks: [$num_blocks]
        shmem_bytes: $shmem_bytes
      kernel-symbol: "$kernel_symbol"
      kernel-arguments:
        - name: "Tbarmin"
          intent: in
          type: Int32
        - name: "Tbarmax"
          intent: in
          type: Int32
        - name: "Ttildemin"
          intent: in
          type: Int32
        - name: "Ttildemax"
          intent: in
          type: Int32
        - name: "W"
          intent: in
          type: Float16
          indices: [C, dishM, dishN, P, Fbar]
          shape: [$C, $M, $N, $P, $Fbar_W]
          strides: [1, $C, $(C*M), $(C*M*N), $(C*M*N*P), $(C*M*N*P*Fbar_W)]
        - name: "Ē"
          intent: in
          type: Int4
          indices: [C, D, P, Fbar, Tbar]
          shape: [$C, $D, $P, $Fbar, $Tbar]
          strides: [1, $C, $(C*D), $(C*D*P), $(C*D*P*Fbar)]
        - name: "I"
          intent: out
          type: Float16
          indices: [beamP, beamQ, Fbar, Tbar]
          shape: [$(2*M), $(2*N), $Fbar, $Ttilde]
          strides: [1, $(2*M), $(2*M*2*N), $(2*M*2*N*Fbar)]
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
    cxx = read("kernels/chimefrb_template.cxx", String)
    cxx = Mustache.render(
        cxx,
        Dict(
            "kernel_name" => "CHIMEFRBBeamformer_$(setup)_U$(U)",
            "upchannelization_factor" => "$U",
            "downsampling_factor" => "$Tds",
            "kernel_design_parameters" => [
                Dict("type" => "int", "name" => "cuda_beam_layout_M", "value" => "$(2*M)"),
                Dict("type" => "int", "name" => "cuda_beam_layout_N", "value" => "$(2*N)"),
                Dict("type" => "int", "name" => "cuda_dish_layout_M", "value" => "$M"),
                Dict("type" => "int", "name" => "cuda_dish_layout_N", "value" => "$N"),
                Dict("type" => "int", "name" => "cuda_downsampling_factor", "value" => "$Tds"),
                Dict("type" => "int", "name" => "cuda_number_of_complex_components", "value" => "$C"),
                Dict("type" => "int", "name" => "cuda_number_of_dishes", "value" => "$D"),
                Dict("type" => "int", "name" => "cuda_number_of_frequencies", "value" => "$Fbar"),
                Dict("type" => "int", "name" => "cuda_number_of_polarizations", "value" => "$P"),
                Dict("type" => "int", "name" => "cuda_number_of_timesamples", "value" => "$Tbar"),
                Dict("type" => "int", "name" => "cuda_granularity_number_of_timesamples", "value" => "$Tds"),
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
                    "name" => "Tbarmin",
                    "kotekan_name" => "Tbarmin",
                    "type" => "int32",
                    "isoutput" => false,
                    "hasbuffer" => false,
                    "isscalar" => true,
                ),
                Dict(
                    "name" => "Tbarmax",
                    "kotekan_name" => "Tbarmax",
                    "type" => "int32",
                    "isoutput" => false,
                    "hasbuffer" => false,
                    "isscalar" => true,
                ),
                Dict(
                    "name" => "Ttildemin",
                    "kotekan_name" => "Ttildemin",
                    "type" => "int32",
                    "isoutput" => false,
                    "hasbuffer" => false,
                    "isscalar" => true,
                ),
                Dict(
                    "name" => "Ttildemax",
                    "kotekan_name" => "Ttildemax",
                    "type" => "int32",
                    "isoutput" => false,
                    "hasbuffer" => false,
                    "isscalar" => true,
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
                        Dict("label" => "F", "length" => Fbar_W),
                    ],
                    "isoutput" => false,
                    "hasbuffer" => true,
                    "isscalar" => false,
                ),
                Dict(
                    "name" => "Ebar",
                    "kotekan_name" => "gpu_mem_voltage",
                    "type" => "int4p4chime",
                    "axes" => [
                        Dict("label" => "D", "length" => D),
                        Dict("label" => "P", "length" => P),
                        Dict("label" => "Fbar", "length" => Fbar),
                        Dict("label" => "Tbar", "length" => Tbar),
                    ],
                    "isoutput" => false,
                    "hasbuffer" => true,
                    "isscalar" => false,
                ),
                Dict(
                    "name" => "I",
                    "kotekan_name" => "gpu_mem_beamgrid",
                    "type" => "float16",
                    "axes" => [
                        Dict("label" => "beamP", "length" => 2 * M),
                        Dict("label" => "beamQ", "length" => 2 * N),
                        Dict("label" => "Fbar", "length" => Fbar),
                        Dict("label" => "Ttilde", "length" => Ttilde),
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
            ],
        ),
    )
    write("output-$card/chimefrb_$(setup)_U$(U).cxx", cxx)
    return nothing
end

if CUDA.functional()
    # Output kernel
    main(; output_kernel=true)
    open("output-$card/chimefrb_$(setup)_U$(U).ptx", "w") do fh
        redirect_stdout(fh) do
            @device_code_ptx main(; compile_only=true, silent=true)
        end
    end
    open("output-$card/chimefrb_$(setup)_U$(U).sass", "w") do fh
        redirect_stdout(fh) do
            @device_code_sass main(; compile_only=true, silent=true)
        end
    end
    fix_ptx_kernel()

    # # Run benchmark
    # main(; nruns=100)

    # # Regular run, also for profiling
    # main()
end
