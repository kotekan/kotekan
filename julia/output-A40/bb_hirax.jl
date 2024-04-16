# Julia source code for CUDA baseband beamformer
# This file has been generated automatically by `bb.jl`.
# Do not modify this file, your changes will be lost.

@inbounds begin #= /localhome/eschnett/src/kotekan/julia/kernels/bb.jl:892 =#
    info = 1
    if true
        info_memory[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) % 128) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 4) % 4) * 32) + 0) + 0x01] =
            info
    end
    if !(0i32 ≤ Tmin ≤ Tmax ≤ 131072 && (Tmax - Tmin) % 128 == 0i32)
        info = 2
        if true
            info_memory[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) % 128) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 4) % 4) * 32) + 0) + 0x01] =
                info
        end
        IndexSpaces.cuda_trap()
    end
    s = s_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 2) % 64) % 64) * 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 2) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 4) * 4) % 16) + 0x01]
    s = s - 3
    if !(0i32 < s < 32i32)
        info = 2
        if true
            info_memory[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) % 128) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 4) % 4) * 32) + 0) + 0x01] =
                info
        end
        IndexSpaces.cuda_trap()
    end
    (A_cplx0_dish0, A_cplx1_dish0, A_cplx0_dish4, A_cplx1_dish4) = IndexSpaces.unsafe_load4_global(
        A_memory,
        (
            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 2) % 64) % 64) * 4096 +
            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 2) % 2) * 2048 +
            (
                (
                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) * 2 +
                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) ÷ 2) % 2) * 8
                ) % 16
            ) * 128 +
            (
                (
                    (
                        ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 4) * 8 +
                        (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 32
                    ) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128
                ) ÷ 2
            ) % 128
        ) + 1i32,
    )
    (A_cplx0_dish8, A_cplx1_dish8, A_cplx0_dish12, A_cplx1_dish12) = IndexSpaces.unsafe_load4_global(
        A_memory,
        (
            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 2) % 64) % 64) * 4096 +
            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 2) % 2) * 2048 +
            (
                (
                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) * 2 +
                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) ÷ 2) % 2) * 8
                ) % 16
            ) * 128 +
            (
                (
                    (
                        (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 4) * 8 + 64) +
                        (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 32
                    ) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128
                ) ÷ 2
            ) % 128
        ) + 1i32,
    )
    (A_cplx0_dish16, A_cplx1_dish16, A_cplx0_dish20, A_cplx1_dish20) = IndexSpaces.unsafe_load4_global(
        A_memory,
        (
            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 2) % 64) % 64) * 4096 +
            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 2) % 2) * 2048 +
            (
                (
                    (
                        ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) * 2 +
                        ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) ÷ 2) % 2) * 8
                    ) + 1
                ) % 16
            ) * 128 +
            (
                (
                    (
                        ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 4) * 8 +
                        (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 32
                    ) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128
                ) ÷ 2
            ) % 128
        ) + 1i32,
    )
    (A_cplx0_dish24, A_cplx1_dish24, A_cplx0_dish28, A_cplx1_dish28) = IndexSpaces.unsafe_load4_global(
        A_memory,
        (
            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 2) % 64) % 64) * 4096 +
            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 2) % 2) * 2048 +
            (
                (
                    (
                        ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) * 2 +
                        ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) ÷ 2) % 2) * 8
                    ) + 1
                ) % 16
            ) * 128 +
            (
                (
                    (
                        (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 4) * 8 + 64) +
                        (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 32
                    ) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128
                ) ÷ 2
            ) % 128
        ) + 1i32,
    )
    (A_cplx0_dish0, A_cplx1_dish0) = (
        IndexSpaces.get_lo16(A_cplx0_dish0, A_cplx1_dish0), IndexSpaces.get_hi16(A_cplx0_dish0, A_cplx1_dish0)
    )
    (A_cplx0_dish4, A_cplx1_dish4) = (
        IndexSpaces.get_lo16(A_cplx0_dish4, A_cplx1_dish4), IndexSpaces.get_hi16(A_cplx0_dish4, A_cplx1_dish4)
    )
    (A_cplx0_dish8, A_cplx1_dish8) = (
        IndexSpaces.get_lo16(A_cplx0_dish8, A_cplx1_dish8), IndexSpaces.get_hi16(A_cplx0_dish8, A_cplx1_dish8)
    )
    (A_cplx0_dish12, A_cplx1_dish12) = (
        IndexSpaces.get_lo16(A_cplx0_dish12, A_cplx1_dish12), IndexSpaces.get_hi16(A_cplx0_dish12, A_cplx1_dish12)
    )
    (A_cplx0_dish16, A_cplx1_dish16) = (
        IndexSpaces.get_lo16(A_cplx0_dish16, A_cplx1_dish16), IndexSpaces.get_hi16(A_cplx0_dish16, A_cplx1_dish16)
    )
    (A_cplx0_dish20, A_cplx1_dish20) = (
        IndexSpaces.get_lo16(A_cplx0_dish20, A_cplx1_dish20), IndexSpaces.get_hi16(A_cplx0_dish20, A_cplx1_dish20)
    )
    (A_cplx0_dish24, A_cplx1_dish24) = (
        IndexSpaces.get_lo16(A_cplx0_dish24, A_cplx1_dish24), IndexSpaces.get_hi16(A_cplx0_dish24, A_cplx1_dish24)
    )
    (A_cplx0_dish28, A_cplx1_dish28) = (
        IndexSpaces.get_lo16(A_cplx0_dish28, A_cplx1_dish28), IndexSpaces.get_hi16(A_cplx0_dish28, A_cplx1_dish28)
    )
    (A_cplx0_dish0, A_cplx1_dish0) = (
        IndexSpaces.get_lo8(A_cplx0_dish0, A_cplx1_dish0), IndexSpaces.get_hi8(A_cplx0_dish0, A_cplx1_dish0)
    )
    (A_cplx0_dish4, A_cplx1_dish4) = (
        IndexSpaces.get_lo8(A_cplx0_dish4, A_cplx1_dish4), IndexSpaces.get_hi8(A_cplx0_dish4, A_cplx1_dish4)
    )
    (A_cplx0_dish8, A_cplx1_dish8) = (
        IndexSpaces.get_lo8(A_cplx0_dish8, A_cplx1_dish8), IndexSpaces.get_hi8(A_cplx0_dish8, A_cplx1_dish8)
    )
    (A_cplx0_dish12, A_cplx1_dish12) = (
        IndexSpaces.get_lo8(A_cplx0_dish12, A_cplx1_dish12), IndexSpaces.get_hi8(A_cplx0_dish12, A_cplx1_dish12)
    )
    (A_cplx0_dish16, A_cplx1_dish16) = (
        IndexSpaces.get_lo8(A_cplx0_dish16, A_cplx1_dish16), IndexSpaces.get_hi8(A_cplx0_dish16, A_cplx1_dish16)
    )
    (A_cplx0_dish20, A_cplx1_dish20) = (
        IndexSpaces.get_lo8(A_cplx0_dish20, A_cplx1_dish20), IndexSpaces.get_hi8(A_cplx0_dish20, A_cplx1_dish20)
    )
    (A_cplx0_dish24, A_cplx1_dish24) = (
        IndexSpaces.get_lo8(A_cplx0_dish24, A_cplx1_dish24), IndexSpaces.get_hi8(A_cplx0_dish24, A_cplx1_dish24)
    )
    (A_cplx0_dish28, A_cplx1_dish28) = (
        IndexSpaces.get_lo8(A_cplx0_dish28, A_cplx1_dish28), IndexSpaces.get_hi8(A_cplx0_dish28, A_cplx1_dish28)
    )
    is_lo_thread = IndexSpaces.cuda_threadidx() & 0x00000002 == 0x00
    (A_cplx0_dish0, A_cplx0_dish8) = let
        src = if is_lo_thread
            A_cplx0_dish8
        else
            A_cplx0_dish0
        end
        dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
        if is_lo_thread
            (A_cplx0_dish0, dst)
        else
            (dst, A_cplx0_dish8)
        end
    end
    (A_cplx1_dish0, A_cplx1_dish8) = let
        src = if is_lo_thread
            A_cplx1_dish8
        else
            A_cplx1_dish0
        end
        dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
        if is_lo_thread
            (A_cplx1_dish0, dst)
        else
            (dst, A_cplx1_dish8)
        end
    end
    (A_cplx0_dish4, A_cplx0_dish12) = let
        src = if is_lo_thread
            A_cplx0_dish12
        else
            A_cplx0_dish4
        end
        dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
        if is_lo_thread
            (A_cplx0_dish4, dst)
        else
            (dst, A_cplx0_dish12)
        end
    end
    (A_cplx1_dish4, A_cplx1_dish12) = let
        src = if is_lo_thread
            A_cplx1_dish12
        else
            A_cplx1_dish4
        end
        dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
        if is_lo_thread
            (A_cplx1_dish4, dst)
        else
            (dst, A_cplx1_dish12)
        end
    end
    (A_cplx0_dish16, A_cplx0_dish24) = let
        src = if is_lo_thread
            A_cplx0_dish24
        else
            A_cplx0_dish16
        end
        dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
        if is_lo_thread
            (A_cplx0_dish16, dst)
        else
            (dst, A_cplx0_dish24)
        end
    end
    (A_cplx1_dish16, A_cplx1_dish24) = let
        src = if is_lo_thread
            A_cplx1_dish24
        else
            A_cplx1_dish16
        end
        dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
        if is_lo_thread
            (A_cplx1_dish16, dst)
        else
            (dst, A_cplx1_dish24)
        end
    end
    (A_cplx0_dish20, A_cplx0_dish28) = let
        src = if is_lo_thread
            A_cplx0_dish28
        else
            A_cplx0_dish20
        end
        dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
        if is_lo_thread
            (A_cplx0_dish20, dst)
        else
            (dst, A_cplx0_dish28)
        end
    end
    (A_cplx1_dish20, A_cplx1_dish28) = let
        src = if is_lo_thread
            A_cplx1_dish28
        else
            A_cplx1_dish20
        end
        dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
        if is_lo_thread
            (A_cplx1_dish20, dst)
        else
            (dst, A_cplx1_dish28)
        end
    end
    is_lo_thread = IndexSpaces.cuda_threadidx() & 0x00000004 == 0x00
    (A_cplx0_dish0, A_cplx0_dish16) = let
        src = if is_lo_thread
            A_cplx0_dish16
        else
            A_cplx0_dish0
        end
        dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
        if is_lo_thread
            (A_cplx0_dish0, dst)
        else
            (dst, A_cplx0_dish16)
        end
    end
    (A_cplx1_dish0, A_cplx1_dish16) = let
        src = if is_lo_thread
            A_cplx1_dish16
        else
            A_cplx1_dish0
        end
        dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
        if is_lo_thread
            (A_cplx1_dish0, dst)
        else
            (dst, A_cplx1_dish16)
        end
    end
    (A_cplx0_dish4, A_cplx0_dish20) = let
        src = if is_lo_thread
            A_cplx0_dish20
        else
            A_cplx0_dish4
        end
        dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
        if is_lo_thread
            (A_cplx0_dish4, dst)
        else
            (dst, A_cplx0_dish20)
        end
    end
    (A_cplx1_dish4, A_cplx1_dish20) = let
        src = if is_lo_thread
            A_cplx1_dish20
        else
            A_cplx1_dish4
        end
        dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
        if is_lo_thread
            (A_cplx1_dish4, dst)
        else
            (dst, A_cplx1_dish20)
        end
    end
    (A_cplx0_dish8, A_cplx0_dish24) = let
        src = if is_lo_thread
            A_cplx0_dish24
        else
            A_cplx0_dish8
        end
        dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
        if is_lo_thread
            (A_cplx0_dish8, dst)
        else
            (dst, A_cplx0_dish24)
        end
    end
    (A_cplx1_dish8, A_cplx1_dish24) = let
        src = if is_lo_thread
            A_cplx1_dish24
        else
            A_cplx1_dish8
        end
        dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
        if is_lo_thread
            (A_cplx1_dish8, dst)
        else
            (dst, A_cplx1_dish24)
        end
    end
    (A_cplx0_dish12, A_cplx0_dish28) = let
        src = if is_lo_thread
            A_cplx0_dish28
        else
            A_cplx0_dish12
        end
        dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
        if is_lo_thread
            (A_cplx0_dish12, dst)
        else
            (dst, A_cplx0_dish28)
        end
    end
    (A_cplx1_dish12, A_cplx1_dish28) = let
        src = if is_lo_thread
            A_cplx1_dish28
        else
            A_cplx1_dish12
        end
        dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
        if is_lo_thread
            (A_cplx1_dish12, dst)
        else
            (dst, A_cplx1_dish28)
        end
    end
    for T1 in 0:128:65535
        Tmin + T1 ≥ Tmax && break
        Jper_time0 = zero(Int4x8)
        Jper_time32 = zero(Int4x8)
        Jper_time64 = zero(Int4x8)
        Jper_time96 = zero(Int4x8)
        for T2 in 0:32:127
            if IndexSpaces.cuda_warpidx() < 4
                (E_dish0_time0, E_dish4_time0, E_dish8_time0, E_dish12_time0) = IndexSpaces.unsafe_load4_global(
                    E_memory,
                    let
                        offset = 8192 * Tmin
                        length = 536870912
                        mod(
                            (
                                (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 2) % 64) % 64) * 128 +
                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 2) % 2) * 64 +
                                (
                                    (
                                        (
                                            (
                                                ((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 +
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4
                                            ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) ÷ 2) % 2) * 4
                                        ) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128
                                    ) % 65536
                                ) * 8192 +
                                (
                                    (
                                        (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16 +
                                        (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128
                                    ) ÷ 4
                                ) % 64
                            ) + offset,
                            length,
                        )
                    end + 1i32,
                )
                (E_dish0_time8, E_dish4_time8, E_dish8_time8, E_dish12_time8) = IndexSpaces.unsafe_load4_global(
                    E_memory,
                    let
                        offset = 8192 * Tmin
                        length = 536870912
                        mod(
                            (
                                (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 2) % 64) % 64) * 128 +
                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 2) % 2) * 64 +
                                (
                                    (
                                        (
                                            (
                                                (
                                                    ((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 +
                                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4
                                                ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) ÷ 2) % 2) * 4
                                            ) + 8
                                        ) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128
                                    ) % 65536
                                ) * 8192 +
                                (
                                    (
                                        (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16 +
                                        (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128
                                    ) ÷ 4
                                ) % 64
                            ) + offset,
                            length,
                        )
                    end + 1i32,
                )
                (E_dish0_time16, E_dish4_time16, E_dish8_time16, E_dish12_time16) = IndexSpaces.unsafe_load4_global(
                    E_memory,
                    let
                        offset = 8192 * Tmin
                        length = 536870912
                        mod(
                            (
                                (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 2) % 64) % 64) * 128 +
                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 2) % 2) * 64 +
                                (
                                    (
                                        (
                                            (
                                                (
                                                    ((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 +
                                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4
                                                ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) ÷ 2) % 2) * 4
                                            ) + 16
                                        ) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128
                                    ) % 65536
                                ) * 8192 +
                                (
                                    (
                                        (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16 +
                                        (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128
                                    ) ÷ 4
                                ) % 64
                            ) + offset,
                            length,
                        )
                    end + 1i32,
                )
                (E_dish0_time24, E_dish4_time24, E_dish8_time24, E_dish12_time24) = IndexSpaces.unsafe_load4_global(
                    E_memory,
                    let
                        offset = 8192 * Tmin
                        length = 536870912
                        mod(
                            (
                                (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 2) % 64) % 64) * 128 +
                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 2) % 2) * 64 +
                                (
                                    (
                                        (
                                            (
                                                (
                                                    ((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 +
                                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4
                                                ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) ÷ 2) % 2) * 4
                                            ) + 24
                                        ) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128
                                    ) % 65536
                                ) * 8192 +
                                (
                                    (
                                        (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16 +
                                        (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128
                                    ) ÷ 4
                                ) % 64
                            ) + offset,
                            length,
                        )
                    end + 1i32,
                )
                if true
                    E_shared[((((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) ÷ 2) % 2) * 4) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128) ÷ 4) % 64) + 0) + 0x01] =
                        E_dish0_time0
                end
                if true
                    E_shared[((((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) ÷ 2) % 2) * 4) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16 + 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128) ÷ 4) % 64) + 0) + 0x01] =
                        E_dish4_time0
                end
                if true
                    E_shared[((((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) ÷ 2) % 2) * 4) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16 + 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128) ÷ 4) % 64) + 0) + 0x01] =
                        E_dish8_time0
                end
                if true
                    E_shared[((((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) ÷ 2) % 2) * 4) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16 + 12) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128) ÷ 4) % 64) + 0) + 0x01] =
                        E_dish12_time0
                end
                if true
                    E_shared[(((((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) ÷ 2) % 2) * 4) + 8) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128) ÷ 4) % 64) + 0) + 0x01] =
                        E_dish0_time8
                end
                if true
                    E_shared[(((((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) ÷ 2) % 2) * 4) + 8) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16 + 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128) ÷ 4) % 64) + 0) + 0x01] =
                        E_dish4_time8
                end
                if true
                    E_shared[(((((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) ÷ 2) % 2) * 4) + 8) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16 + 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128) ÷ 4) % 64) + 0) + 0x01] =
                        E_dish8_time8
                end
                if true
                    E_shared[(((((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) ÷ 2) % 2) * 4) + 8) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16 + 12) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128) ÷ 4) % 64) + 0) + 0x01] =
                        E_dish12_time8
                end
                if true
                    E_shared[(((((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) ÷ 2) % 2) * 4) + 16) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128) ÷ 4) % 64) + 0) + 0x01] =
                        E_dish0_time16
                end
                if true
                    E_shared[(((((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) ÷ 2) % 2) * 4) + 16) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16 + 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128) ÷ 4) % 64) + 0) + 0x01] =
                        E_dish4_time16
                end
                if true
                    E_shared[(((((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) ÷ 2) % 2) * 4) + 16) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16 + 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128) ÷ 4) % 64) + 0) + 0x01] =
                        E_dish8_time16
                end
                if true
                    E_shared[(((((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) ÷ 2) % 2) * 4) + 16) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16 + 12) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128) ÷ 4) % 64) + 0) + 0x01] =
                        E_dish12_time16
                end
                if true
                    E_shared[(((((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) ÷ 2) % 2) * 4) + 24) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128) ÷ 4) % 64) + 0) + 0x01] =
                        E_dish0_time24
                end
                if true
                    E_shared[(((((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) ÷ 2) % 2) * 4) + 24) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16 + 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128) ÷ 4) % 64) + 0) + 0x01] =
                        E_dish4_time24
                end
                if true
                    E_shared[(((((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) ÷ 2) % 2) * 4) + 24) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16 + 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128) ÷ 4) % 64) + 0) + 0x01] =
                        E_dish8_time24
                end
                if true
                    E_shared[(((((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) ÷ 2) % 2) * 4) + 24) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16 + 12) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128) ÷ 4) % 64) + 0) + 0x01] =
                        E_dish12_time24
                end
            end
            IndexSpaces.cuda_sync_threads()
            for T3 in 0:8:31
                let
                    B = 0
                    AselB_cplx0_dish0 = A_cplx0_dish0
                    AselB_cplx1_dish0 = A_cplx1_dish0
                    AselB_cplx0_dish4 = A_cplx0_dish4
                    AselB_cplx1_dish4 = A_cplx1_dish4
                    AselB_cplx0_dish8 = A_cplx0_dish8
                    AselB_cplx1_dish8 = A_cplx1_dish8
                    AselB_cplx0_dish12 = A_cplx0_dish12
                    AselB_cplx1_dish12 = A_cplx1_dish12
                    AselB_cplx0_dish16 = A_cplx0_dish16
                    AselB_cplx1_dish16 = A_cplx1_dish16
                    AselB_cplx0_dish20 = A_cplx0_dish20
                    AselB_cplx1_dish20 = A_cplx1_dish20
                    AselB_cplx0_dish24 = A_cplx0_dish24
                    AselB_cplx1_dish24 = A_cplx1_dish24
                    AselB_cplx0_dish28 = A_cplx0_dish28
                    AselB_cplx1_dish28 = A_cplx1_dish28
                    Jurepos_time0 = 0
                    Jurepos_time1 = 0
                    Jureneg_time0 = 0
                    Jureneg_time1 = 0
                    Juim_time0 = 0
                    Juim_time1 = 0
                    let
                        D = 0
                        AselBD_cplx0 = AselB_cplx0_dish0
                        AselBD_cplx1 = AselB_cplx1_dish0
                        Are = AselBD_cplx0
                        Aim = AselBD_cplx1
                        E0 = E_shared[(((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) + ((IndexSpaces.assume_inrange(T3, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 65 + (((((D ÷ 4) % 8) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128) ÷ 4) % 64) + 0x01]
                        (E1_cplx0, E1_cplx1) = convert(NTuple{2,Int8x4}, E0)
                        E1re = E1_cplx0
                        E1im = E1_cplx1
                        (Jurepos_time0, Jurepos_time1) = IndexSpaces.mma_m8n8k16(Are, E1re, (Jurepos_time0, Jurepos_time1))
                        (Jureneg_time0, Jureneg_time1) = IndexSpaces.mma_m8n8k16(Aim, E1im, (Jureneg_time0, Jureneg_time1))
                        (Juim_time0, Juim_time1) = IndexSpaces.mma_m8n8k16(Are, E1im, (Juim_time0, Juim_time1))
                        (Juim_time0, Juim_time1) = IndexSpaces.mma_m8n8k16(Aim, E1re, (Juim_time0, Juim_time1))
                    end
                    let
                        D = 4
                        AselBD_cplx0 = AselB_cplx0_dish4
                        AselBD_cplx1 = AselB_cplx1_dish4
                        Are = AselBD_cplx0
                        Aim = AselBD_cplx1
                        E0 = E_shared[(((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) + ((IndexSpaces.assume_inrange(T3, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 65 + (((((D ÷ 4) % 8) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128) ÷ 4) % 64) + 0x01]
                        (E1_cplx0, E1_cplx1) = convert(NTuple{2,Int8x4}, E0)
                        E1re = E1_cplx0
                        E1im = E1_cplx1
                        (Jurepos_time0, Jurepos_time1) = IndexSpaces.mma_m8n8k16(Are, E1re, (Jurepos_time0, Jurepos_time1))
                        (Jureneg_time0, Jureneg_time1) = IndexSpaces.mma_m8n8k16(Aim, E1im, (Jureneg_time0, Jureneg_time1))
                        (Juim_time0, Juim_time1) = IndexSpaces.mma_m8n8k16(Are, E1im, (Juim_time0, Juim_time1))
                        (Juim_time0, Juim_time1) = IndexSpaces.mma_m8n8k16(Aim, E1re, (Juim_time0, Juim_time1))
                    end
                    let
                        D = 8
                        AselBD_cplx0 = AselB_cplx0_dish8
                        AselBD_cplx1 = AselB_cplx1_dish8
                        Are = AselBD_cplx0
                        Aim = AselBD_cplx1
                        E0 = E_shared[(((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) + ((IndexSpaces.assume_inrange(T3, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 65 + (((((D ÷ 4) % 8) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128) ÷ 4) % 64) + 0x01]
                        (E1_cplx0, E1_cplx1) = convert(NTuple{2,Int8x4}, E0)
                        E1re = E1_cplx0
                        E1im = E1_cplx1
                        (Jurepos_time0, Jurepos_time1) = IndexSpaces.mma_m8n8k16(Are, E1re, (Jurepos_time0, Jurepos_time1))
                        (Jureneg_time0, Jureneg_time1) = IndexSpaces.mma_m8n8k16(Aim, E1im, (Jureneg_time0, Jureneg_time1))
                        (Juim_time0, Juim_time1) = IndexSpaces.mma_m8n8k16(Are, E1im, (Juim_time0, Juim_time1))
                        (Juim_time0, Juim_time1) = IndexSpaces.mma_m8n8k16(Aim, E1re, (Juim_time0, Juim_time1))
                    end
                    let
                        D = 12
                        AselBD_cplx0 = AselB_cplx0_dish12
                        AselBD_cplx1 = AselB_cplx1_dish12
                        Are = AselBD_cplx0
                        Aim = AselBD_cplx1
                        E0 = E_shared[(((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) + ((IndexSpaces.assume_inrange(T3, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 65 + (((((D ÷ 4) % 8) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128) ÷ 4) % 64) + 0x01]
                        (E1_cplx0, E1_cplx1) = convert(NTuple{2,Int8x4}, E0)
                        E1re = E1_cplx0
                        E1im = E1_cplx1
                        (Jurepos_time0, Jurepos_time1) = IndexSpaces.mma_m8n8k16(Are, E1re, (Jurepos_time0, Jurepos_time1))
                        (Jureneg_time0, Jureneg_time1) = IndexSpaces.mma_m8n8k16(Aim, E1im, (Jureneg_time0, Jureneg_time1))
                        (Juim_time0, Juim_time1) = IndexSpaces.mma_m8n8k16(Are, E1im, (Juim_time0, Juim_time1))
                        (Juim_time0, Juim_time1) = IndexSpaces.mma_m8n8k16(Aim, E1re, (Juim_time0, Juim_time1))
                    end
                    let
                        D = 16
                        AselBD_cplx0 = AselB_cplx0_dish16
                        AselBD_cplx1 = AselB_cplx1_dish16
                        Are = AselBD_cplx0
                        Aim = AselBD_cplx1
                        E0 = E_shared[(((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) + ((IndexSpaces.assume_inrange(T3, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 65 + (((((D ÷ 4) % 8) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128) ÷ 4) % 64) + 0x01]
                        (E1_cplx0, E1_cplx1) = convert(NTuple{2,Int8x4}, E0)
                        E1re = E1_cplx0
                        E1im = E1_cplx1
                        (Jurepos_time0, Jurepos_time1) = IndexSpaces.mma_m8n8k16(Are, E1re, (Jurepos_time0, Jurepos_time1))
                        (Jureneg_time0, Jureneg_time1) = IndexSpaces.mma_m8n8k16(Aim, E1im, (Jureneg_time0, Jureneg_time1))
                        (Juim_time0, Juim_time1) = IndexSpaces.mma_m8n8k16(Are, E1im, (Juim_time0, Juim_time1))
                        (Juim_time0, Juim_time1) = IndexSpaces.mma_m8n8k16(Aim, E1re, (Juim_time0, Juim_time1))
                    end
                    let
                        D = 20
                        AselBD_cplx0 = AselB_cplx0_dish20
                        AselBD_cplx1 = AselB_cplx1_dish20
                        Are = AselBD_cplx0
                        Aim = AselBD_cplx1
                        E0 = E_shared[(((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) + ((IndexSpaces.assume_inrange(T3, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 65 + (((((D ÷ 4) % 8) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128) ÷ 4) % 64) + 0x01]
                        (E1_cplx0, E1_cplx1) = convert(NTuple{2,Int8x4}, E0)
                        E1re = E1_cplx0
                        E1im = E1_cplx1
                        (Jurepos_time0, Jurepos_time1) = IndexSpaces.mma_m8n8k16(Are, E1re, (Jurepos_time0, Jurepos_time1))
                        (Jureneg_time0, Jureneg_time1) = IndexSpaces.mma_m8n8k16(Aim, E1im, (Jureneg_time0, Jureneg_time1))
                        (Juim_time0, Juim_time1) = IndexSpaces.mma_m8n8k16(Are, E1im, (Juim_time0, Juim_time1))
                        (Juim_time0, Juim_time1) = IndexSpaces.mma_m8n8k16(Aim, E1re, (Juim_time0, Juim_time1))
                    end
                    let
                        D = 24
                        AselBD_cplx0 = AselB_cplx0_dish24
                        AselBD_cplx1 = AselB_cplx1_dish24
                        Are = AselBD_cplx0
                        Aim = AselBD_cplx1
                        E0 = E_shared[(((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) + ((IndexSpaces.assume_inrange(T3, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 65 + (((((D ÷ 4) % 8) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128) ÷ 4) % 64) + 0x01]
                        (E1_cplx0, E1_cplx1) = convert(NTuple{2,Int8x4}, E0)
                        E1re = E1_cplx0
                        E1im = E1_cplx1
                        (Jurepos_time0, Jurepos_time1) = IndexSpaces.mma_m8n8k16(Are, E1re, (Jurepos_time0, Jurepos_time1))
                        (Jureneg_time0, Jureneg_time1) = IndexSpaces.mma_m8n8k16(Aim, E1im, (Jureneg_time0, Jureneg_time1))
                        (Juim_time0, Juim_time1) = IndexSpaces.mma_m8n8k16(Are, E1im, (Juim_time0, Juim_time1))
                        (Juim_time0, Juim_time1) = IndexSpaces.mma_m8n8k16(Aim, E1re, (Juim_time0, Juim_time1))
                    end
                    let
                        D = 28
                        AselBD_cplx0 = AselB_cplx0_dish28
                        AselBD_cplx1 = AselB_cplx1_dish28
                        Are = AselBD_cplx0
                        Aim = AselBD_cplx1
                        E0 = E_shared[(((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) + ((IndexSpaces.assume_inrange(T3, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 65 + (((((D ÷ 4) % 8) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128) ÷ 4) % 64) + 0x01]
                        (E1_cplx0, E1_cplx1) = convert(NTuple{2,Int8x4}, E0)
                        E1re = E1_cplx0
                        E1im = E1_cplx1
                        (Jurepos_time0, Jurepos_time1) = IndexSpaces.mma_m8n8k16(Are, E1re, (Jurepos_time0, Jurepos_time1))
                        (Jureneg_time0, Jureneg_time1) = IndexSpaces.mma_m8n8k16(Aim, E1im, (Jureneg_time0, Jureneg_time1))
                        (Juim_time0, Juim_time1) = IndexSpaces.mma_m8n8k16(Are, E1im, (Juim_time0, Juim_time1))
                        (Juim_time0, Juim_time1) = IndexSpaces.mma_m8n8k16(Aim, E1re, (Juim_time0, Juim_time1))
                    end
                    Jure_time0 = Jurepos_time0 - Jureneg_time0
                    Jure_time1 = Jurepos_time1 - Jureneg_time1
                    Ju_cplx0_time0 = Jure_time0
                    Ju_cplx1_time0 = Juim_time0
                    Ju_cplx0_time1 = Jure_time1
                    Ju_cplx1_time1 = Juim_time1
                    Ju_cplx0_time0 = (Ju_cplx0_time0 + 4) >> 0x00000003
                    Ju_cplx1_time0 = (Ju_cplx1_time0 + 4) >> 0x00000003
                    Ju_cplx0_time1 = (Ju_cplx0_time1 + 4) >> 0x00000003
                    Ju_cplx1_time1 = (Ju_cplx1_time1 + 4) >> 0x00000003
                    Ju_time0 = Int16x2((Ju_cplx0_time0, Ju_cplx1_time0))
                    Ju_time1 = Int16x2((Ju_cplx0_time1, Ju_cplx1_time1))
                    if true
                        Ju_shared[((((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + ((IndexSpaces.assume_inrange(T3, 0, 8, 32) ÷ 8) % 4) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 2) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 20 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) ÷ 2) % 2) * 8) % 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128) ÷ 128) % 2) * 640) + 0) + 0x01] =
                            Ju_time0
                    end
                    if true
                        Ju_shared[(((((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + ((IndexSpaces.assume_inrange(T3, 0, 8, 32) ÷ 8) % 4) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 2) + 1) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 20 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) ÷ 2) % 2) * 8) % 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 2) * 128) ÷ 128) % 2) * 640) + 0) + 0x01] =
                            Ju_time1
                    end
                end
            end
            IndexSpaces.cuda_sync_threads()
            Ju_dish0_time0 = Ju_shared[((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 20 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 4) * 4) % 16 + 0) + 0x01]
            Ju_dish128_time0 = Ju_shared[((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 20 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 4) * 4) % 16 + 640) + 0x01]
            Ju_dish0_time8 = Ju_shared[(((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) + 8) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 20 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 4) * 4) % 16 + 0) + 0x01]
            Ju_dish128_time8 = Ju_shared[(((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) + 8) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 20 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 4) * 4) % 16 + 640) + 0x01]
            Ju_dish0_time16 = Ju_shared[(((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) + 16) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 20 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 4) * 4) % 16 + 0) + 0x01]
            Ju_dish128_time16 = Ju_shared[(((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) + 16) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 20 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 4) * 4) % 16 + 640) + 0x01]
            Ju_dish0_time24 = Ju_shared[(((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) + 24) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 20 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 4) * 4) % 16 + 0) + 0x01]
            Ju_dish128_time24 = Ju_shared[(((((((IndexSpaces.assume_inrange(T2, 0, 32, 128) ÷ 32) % 4) * 32 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) + 24) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128) % 32) * 20 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 4) * 4) % 16 + 640) + 0x01]
            (Ju_cplx0_dish0_time0, Ju_cplx1_dish0_time0) = convert(NTuple{2,Int32}, Ju_dish0_time0)
            (Ju_cplx0_dish128_time0, Ju_cplx1_dish128_time0) = convert(NTuple{2,Int32}, Ju_dish128_time0)
            (Ju_cplx0_dish0_time8, Ju_cplx1_dish0_time8) = convert(NTuple{2,Int32}, Ju_dish0_time8)
            (Ju_cplx0_dish128_time8, Ju_cplx1_dish128_time8) = convert(NTuple{2,Int32}, Ju_dish128_time8)
            (Ju_cplx0_dish0_time16, Ju_cplx1_dish0_time16) = convert(NTuple{2,Int32}, Ju_dish0_time16)
            (Ju_cplx0_dish128_time16, Ju_cplx1_dish128_time16) = convert(NTuple{2,Int32}, Ju_dish128_time16)
            (Ju_cplx0_dish0_time24, Ju_cplx1_dish0_time24) = convert(NTuple{2,Int32}, Ju_dish0_time24)
            (Ju_cplx0_dish128_time24, Ju_cplx1_dish128_time24) = convert(NTuple{2,Int32}, Ju_dish128_time24)
            Julo_cplx0_time0 = Ju_cplx0_dish0_time0
            Juhi_cplx0_time0 = Ju_cplx0_dish128_time0
            Julo_cplx1_time0 = Ju_cplx1_dish0_time0
            Juhi_cplx1_time0 = Ju_cplx1_dish128_time0
            Julo_cplx0_time8 = Ju_cplx0_dish0_time8
            Juhi_cplx0_time8 = Ju_cplx0_dish128_time8
            Julo_cplx1_time8 = Ju_cplx1_dish0_time8
            Juhi_cplx1_time8 = Ju_cplx1_dish128_time8
            Julo_cplx0_time16 = Ju_cplx0_dish0_time16
            Juhi_cplx0_time16 = Ju_cplx0_dish128_time16
            Julo_cplx1_time16 = Ju_cplx1_dish0_time16
            Juhi_cplx1_time16 = Ju_cplx1_dish128_time16
            Julo_cplx0_time24 = Ju_cplx0_dish0_time24
            Juhi_cplx0_time24 = Ju_cplx0_dish128_time24
            Julo_cplx1_time24 = Ju_cplx1_dish0_time24
            Juhi_cplx1_time24 = Ju_cplx1_dish128_time24
            J_cplx0_time0 = Julo_cplx0_time0 + Juhi_cplx0_time0
            J_cplx1_time0 = Julo_cplx1_time0 + Juhi_cplx1_time0
            J_cplx0_time8 = Julo_cplx0_time8 + Juhi_cplx0_time8
            J_cplx1_time8 = Julo_cplx1_time8 + Juhi_cplx1_time8
            J_cplx0_time16 = Julo_cplx0_time16 + Juhi_cplx0_time16
            J_cplx1_time16 = Julo_cplx1_time16 + Juhi_cplx1_time16
            J_cplx0_time24 = Julo_cplx0_time24 + Juhi_cplx0_time24
            J_cplx1_time24 = Julo_cplx1_time24 + Juhi_cplx1_time24
            J_cplx0_time0 = (J_cplx0_time0 + 1 << (s % UInt32 - 0x01)) >> (s % UInt32)
            J_cplx1_time0 = (J_cplx1_time0 + 1 << (s % UInt32 - 0x01)) >> (s % UInt32)
            J_cplx0_time8 = (J_cplx0_time8 + 1 << (s % UInt32 - 0x01)) >> (s % UInt32)
            J_cplx1_time8 = (J_cplx1_time8 + 1 << (s % UInt32 - 0x01)) >> (s % UInt32)
            J_cplx0_time16 = (J_cplx0_time16 + 1 << (s % UInt32 - 0x01)) >> (s % UInt32)
            J_cplx1_time16 = (J_cplx1_time16 + 1 << (s % UInt32 - 0x01)) >> (s % UInt32)
            J_cplx0_time24 = (J_cplx0_time24 + 1 << (s % UInt32 - 0x01)) >> (s % UInt32)
            J_cplx1_time24 = (J_cplx1_time24 + 1 << (s % UInt32 - 0x01)) >> (s % UInt32)
            J_cplx0_time0 = clamp(J_cplx0_time0, -7:7)
            J_cplx1_time0 = clamp(J_cplx1_time0, -7:7)
            J_cplx0_time8 = clamp(J_cplx0_time8, -7:7)
            J_cplx1_time8 = clamp(J_cplx1_time8, -7:7)
            J_cplx0_time16 = clamp(J_cplx0_time16, -7:7)
            J_cplx1_time16 = clamp(J_cplx1_time16, -7:7)
            J_cplx0_time24 = clamp(J_cplx0_time24, -7:7)
            J_cplx1_time24 = clamp(J_cplx1_time24, -7:7)
            J = Int4x8(
                J_cplx0_time0,
                J_cplx1_time0,
                J_cplx0_time8,
                J_cplx1_time8,
                J_cplx0_time16,
                J_cplx1_time16,
                J_cplx0_time24,
                J_cplx1_time24,
            )
            if T2 == 0
                Jper_time0 = J
            end
            if T2 == 32
                Jper_time32 = J
            end
            if T2 == 64
                Jper_time64 = J
            end
            if T2 == 96
                Jper_time96 = J
            end
        end
        (Jper_time0, Jper_time32) = (IndexSpaces.get_lo8(Jper_time0, Jper_time32), IndexSpaces.get_hi8(Jper_time0, Jper_time32))
        (Jper_time64, Jper_time96) = (IndexSpaces.get_lo8(Jper_time64, Jper_time96), IndexSpaces.get_hi8(Jper_time64, Jper_time96))
        is_lo_thread = IndexSpaces.cuda_threadidx() & 0x00000001 == 0x00
        (Jper_time0, Jper_time32) = let
            src = if is_lo_thread
                Jper_time32
            else
                Jper_time0
            end
            dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000001)
            if is_lo_thread
                (Jper_time0, dst)
            else
                (dst, Jper_time32)
            end
        end
        (Jper_time64, Jper_time96) = let
            src = if is_lo_thread
                Jper_time96
            else
                Jper_time64
            end
            dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000001)
            if is_lo_thread
                (Jper_time64, dst)
            else
                (dst, Jper_time96)
            end
        end
        (Jper_time0, Jper_time32) = (IndexSpaces.get_lo8(Jper_time0, Jper_time32), IndexSpaces.get_hi8(Jper_time0, Jper_time32))
        (Jper_time64, Jper_time96) = (IndexSpaces.get_lo8(Jper_time64, Jper_time96), IndexSpaces.get_hi8(Jper_time64, Jper_time96))
        is_lo_thread = IndexSpaces.cuda_threadidx() & 0x00000002 == 0x00
        (Jper_time0, Jper_time64) = let
            src = if is_lo_thread
                Jper_time64
            else
                Jper_time0
            end
            dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
            if is_lo_thread
                (Jper_time0, dst)
            else
                (dst, Jper_time64)
            end
        end
        (Jper_time32, Jper_time96) = let
            src = if is_lo_thread
                Jper_time96
            else
                Jper_time32
            end
            dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
            if is_lo_thread
                (Jper_time32, dst)
            else
                (dst, Jper_time96)
            end
        end
        (Jper_time0, Jper_time64) = (IndexSpaces.get_lo16(Jper_time0, Jper_time64), IndexSpaces.get_hi16(Jper_time0, Jper_time64))
        (Jper_time32, Jper_time96) = (
            IndexSpaces.get_lo16(Jper_time32, Jper_time96), IndexSpaces.get_hi16(Jper_time32, Jper_time96)
        )
        is_lo_thread = IndexSpaces.cuda_threadidx() & 0x00000004 == 0x00
        (Jper_time0, Jper_time32) = let
            src = if is_lo_thread
                Jper_time32
            else
                Jper_time0
            end
            dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
            if is_lo_thread
                (Jper_time0, dst)
            else
                (dst, Jper_time32)
            end
        end
        (Jper_time64, Jper_time96) = let
            src = if is_lo_thread
                Jper_time96
            else
                Jper_time64
            end
            dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
            if is_lo_thread
                (Jper_time64, dst)
            else
                (dst, Jper_time96)
            end
        end
        is_lo_thread = IndexSpaces.cuda_threadidx() & 0x00000001 == 0x00
        (Jper_time0, Jper_time64) = let
            src = if is_lo_thread
                Jper_time64
            else
                Jper_time0
            end
            dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000001)
            if is_lo_thread
                (Jper_time0, dst)
            else
                (dst, Jper_time64)
            end
        end
        (Jper_time32, Jper_time96) = let
            src = if is_lo_thread
                Jper_time96
            else
                Jper_time32
            end
            dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000001)
            if is_lo_thread
                (Jper_time32, dst)
            else
                (dst, Jper_time96)
            end
        end
        if true
            IndexSpaces.unsafe_store4_global!(
                J_memory,
                (
                    (
                        (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 2) % 64) % 64) * 8192 +
                        (
                            (
                                (
                                    (
                                        (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 16 +
                                        ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 32
                                    ) + ((IndexSpaces.assume_inrange(T1, 0, 128, 65536) ÷ 128) % 512) * 128
                                ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 2) * 64
                            ) ÷ 4
                        ) % 4096 +
                        ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 2) % 2) * 4096 +
                        (
                            (
                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4 +
                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 4) * 4
                            ) % 16
                        ) * 524288
                    ) + 0
                ) + 0x01,
                (Jper_time0, Jper_time32, Jper_time64, Jper_time96),
            )
        end
    end
    info = 0
    if true
        info_memory[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) % 128) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 4) % 4) % 4) * 32) + 0) + 0x01] =
            info
    end
end
