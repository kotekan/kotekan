# Julia source code for CUDA baseband beamformer
# This file has been generated automatically by `bb.jl`.
# Do not modify this file, your changes will be lost.

@inbounds begin #= /home/eschnett/src/kotekan/julia/kernels/bb.jl:1038 =#
    info = 1
    info_memory[(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384), 1), 384), 1), 1), 384), 128), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 1), 32), 1), 1), 32), 1), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1), 4), 1), 1), 4), 32)) + 0) + 0x01] =
        info
    if !(0i32 ≤ Tmin < 32768 && (Tmin ≤ Tmax < 65536 && (Tmax - Tmin) % 128 == 0i32))
        info = 2
        info_memory[(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384), 1), 384), 1), 1), 384), 128), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 1), 32), 1), 1), 32), 1), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1), 4), 1), 1), 4), 32)) + 0) + 0x01] =
            info
        IndexSpaces.cuda_trap()
    end
    S_polr0 = S_memory[IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1), 4), 4), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 8), 4), 1)), 1), 16), 1), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 1), 2), 1), 1), 2), 16), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384), 1), 384), 1), 1), 384), 32)) + 0x01]
    S_polr1 = S_memory[IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1), 4), 4), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 8), 4), 1)), 1), 16), 1), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(1::Int32, 1), 2), 1), 1), 2), 16), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384), 1), 384), 1), 1), 384), 32)) + 0x01]
    if !(0i32 ≤ S_polr0 && 0i32 ≤ S_polr1)
        info = 3
        info_memory[(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384), 1), 384), 1), 1), 384), 128), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 1), 32), 1), 1), 32), 1), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1), 4), 1), 1), 4), 32)) + 0) + 0x01] =
            info
        IndexSpaces.cuda_trap()
    end
    if let
        thread = IndexSpaces.cuda_threadidx()
        warp = IndexSpaces.cuda_warpidx()
        thread == 0i32 && warp == 0i32
    end
        logval = 0
        log_memory[(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384), 1), 384), 1), 1), 384), 1) + 0) + 0x01] =
            logval
    end
    IndexSpaces.cuda_sync_threads()
    hasoverflow = false
    (A_cplx0_dish0, A_cplx1_dish0, A_cplx0_dish4, A_cplx1_dish4) = IndexSpaces.unsafe_load4(
        A_memory,
        IndexSpaces.add(
            IndexSpaces.mul(
                IndexSpaces.imod(
                    IndexSpaces.idiv(
                        IndexSpaces.add(
                            IndexSpaces.mul(
                                IndexSpaces.imod(
                                    IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 4), 2
                                ),
                                8,
                            ),
                            IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 4), 2), 4),
                            IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 1), 2), 2),
                            IndexSpaces.mul(
                                IndexSpaces.imod(
                                    IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 1), 4
                                ),
                                16,
                            ),
                        ),
                        2,
                    ),
                    32,
                ),
                1,
            ),
            IndexSpaces.mul(
                IndexSpaces.imod(
                    IndexSpaces.idiv(
                        IndexSpaces.add(
                            IndexSpaces.mul(
                                IndexSpaces.imod(
                                    IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1), 2
                                ),
                                8,
                            ),
                            IndexSpaces.mul(
                                IndexSpaces.imod(
                                    IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 8), 4
                                ),
                                2,
                            ),
                            IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 8), 2), 1),
                        ),
                        1,
                    ),
                    16,
                ),
                32,
            ),
            IndexSpaces.mul(
                IndexSpaces.imod(
                    IndexSpaces.idiv(
                        IndexSpaces.mul(
                            IndexSpaces.imod(
                                IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 2), 2
                            ),
                            1,
                        ),
                        1,
                    ),
                    2,
                ),
                512,
            ),
            IndexSpaces.mul(
                IndexSpaces.imod(
                    IndexSpaces.idiv(
                        IndexSpaces.mul(
                            IndexSpaces.imod(
                                IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384), 1), 384
                            ),
                            1,
                        ),
                        1,
                    ),
                    384,
                ),
                1024,
            ),
        ) + 0x01,
    )
    (A_cplx0_dish8, A_cplx1_dish8, A_cplx0_dish12, A_cplx1_dish12) = IndexSpaces.unsafe_load4(
        A_memory,
        IndexSpaces.add(
            IndexSpaces.mul(
                IndexSpaces.imod(
                    IndexSpaces.idiv(
                        IndexSpaces.add(
                            IndexSpaces.mul(
                                IndexSpaces.imod(
                                    IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 4), 2
                                ),
                                8,
                            ),
                            IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(8::Int32, 4), 2), 4),
                            IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 1), 2), 2),
                            IndexSpaces.mul(
                                IndexSpaces.imod(
                                    IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 1), 4
                                ),
                                16,
                            ),
                        ),
                        2,
                    ),
                    32,
                ),
                1,
            ),
            IndexSpaces.mul(
                IndexSpaces.imod(
                    IndexSpaces.idiv(
                        IndexSpaces.add(
                            IndexSpaces.mul(
                                IndexSpaces.imod(
                                    IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1), 2
                                ),
                                8,
                            ),
                            IndexSpaces.mul(
                                IndexSpaces.imod(
                                    IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 8), 4
                                ),
                                2,
                            ),
                            IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(8::Int32, 8), 2), 1),
                        ),
                        1,
                    ),
                    16,
                ),
                32,
            ),
            IndexSpaces.mul(
                IndexSpaces.imod(
                    IndexSpaces.idiv(
                        IndexSpaces.mul(
                            IndexSpaces.imod(
                                IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 2), 2
                            ),
                            1,
                        ),
                        1,
                    ),
                    2,
                ),
                512,
            ),
            IndexSpaces.mul(
                IndexSpaces.imod(
                    IndexSpaces.idiv(
                        IndexSpaces.mul(
                            IndexSpaces.imod(
                                IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384), 1), 384
                            ),
                            1,
                        ),
                        1,
                    ),
                    384,
                ),
                1024,
            ),
        ) + 0x01,
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
    is_lo_thread = IndexSpaces.cuda_threadidx() & 0x00000004 == 0x00
    (A_cplx0_dish0, A_cplx0_dish8) = let
        src = if is_lo_thread
            A_cplx0_dish8
        else
            A_cplx0_dish0
        end
        dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
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
        dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
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
        dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
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
        dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
        if is_lo_thread
            (A_cplx1_dish4, dst)
        else
            (dst, A_cplx1_dish12)
        end
    end
    for T1 in 0:128:32767
        Tmin + T1 ≥ Tmax && break
        Jper_polr0_time0 = zero(Int4x8)
        Jper_polr0_time32 = zero(Int4x8)
        Jper_polr0_time64 = zero(Int4x8)
        Jper_polr0_time96 = zero(Int4x8)
        Jper_polr1_time0 = zero(Int4x8)
        Jper_polr1_time32 = zero(Int4x8)
        Jper_polr1_time64 = zero(Int4x8)
        Jper_polr1_time96 = zero(Int4x8)
        for T2 in 0:32:127
            if IndexSpaces.cuda_warpidx() < 4
                (E_dish0_time0, E_dish4_time0, E_dish8_time0, E_dish12_time0) = IndexSpaces.unsafe_load4(
                    E_memory,
                    let
                        offset = 12288 * Tmin
                        length = 402653184
                        mod(
                            IndexSpaces.add(
                                IndexSpaces.mul(
                                    IndexSpaces.imod(
                                        IndexSpaces.idiv(
                                            IndexSpaces.mul(
                                                IndexSpaces.imod(
                                                    IndexSpaces.idiv(
                                                        IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384), 1
                                                    ),
                                                    384,
                                                ),
                                                1,
                                            ),
                                            1,
                                        ),
                                        384,
                                    ),
                                    32,
                                ),
                                IndexSpaces.mul(
                                    IndexSpaces.imod(
                                        IndexSpaces.idiv(
                                            IndexSpaces.mul(
                                                IndexSpaces.imod(
                                                    IndexSpaces.idiv(
                                                        IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 2
                                                    ),
                                                    2,
                                                ),
                                                1,
                                            ),
                                            1,
                                        ),
                                        2,
                                    ),
                                    16,
                                ),
                                IndexSpaces.mul(
                                    IndexSpaces.imod(
                                        IndexSpaces.idiv(
                                            IndexSpaces.add(
                                                IndexSpaces.mul(
                                                    IndexSpaces.imod(
                                                        IndexSpaces.idiv(
                                                            IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1
                                                        ),
                                                        2,
                                                    ),
                                                    16,
                                                ),
                                                IndexSpaces.mul(
                                                    IndexSpaces.imod(
                                                        IndexSpaces.idiv(IndexSpaces.assume_inrange(T2::Int32, 0, 32, 128), 32), 4
                                                    ),
                                                    32,
                                                ),
                                                IndexSpaces.mul(
                                                    IndexSpaces.imod(
                                                        IndexSpaces.idiv(IndexSpaces.assume_inrange(T1::Int32, 0, 128, 32768), 128),
                                                        256,
                                                    ),
                                                    128,
                                                ),
                                                IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 8), 2), 8),
                                                IndexSpaces.mul(
                                                    IndexSpaces.imod(
                                                        IndexSpaces.idiv(
                                                            IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32),
                                                            4,
                                                        ),
                                                        8,
                                                    ),
                                                    1,
                                                ),
                                            ),
                                            1,
                                        ),
                                        32768,
                                    ),
                                    12288,
                                ),
                                IndexSpaces.mul(
                                    IndexSpaces.imod(
                                        IndexSpaces.idiv(
                                            IndexSpaces.add(
                                                IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 4), 4), 4),
                                                IndexSpaces.mul(
                                                    IndexSpaces.imod(
                                                        IndexSpaces.idiv(
                                                            IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32),
                                                            1,
                                                        ),
                                                        4,
                                                    ),
                                                    16,
                                                ),
                                            ),
                                            4,
                                        ),
                                        16,
                                    ),
                                    1,
                                ),
                            ) + offset,
                            length,
                        )
                    end + 0x01,
                )
                (E_dish0_time8, E_dish4_time8, E_dish8_time8, E_dish12_time8) = IndexSpaces.unsafe_load4(
                    E_memory,
                    let
                        offset = 12288 * Tmin
                        length = 402653184
                        mod(
                            IndexSpaces.add(
                                IndexSpaces.mul(
                                    IndexSpaces.imod(
                                        IndexSpaces.idiv(
                                            IndexSpaces.mul(
                                                IndexSpaces.imod(
                                                    IndexSpaces.idiv(
                                                        IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384), 1
                                                    ),
                                                    384,
                                                ),
                                                1,
                                            ),
                                            1,
                                        ),
                                        384,
                                    ),
                                    32,
                                ),
                                IndexSpaces.mul(
                                    IndexSpaces.imod(
                                        IndexSpaces.idiv(
                                            IndexSpaces.mul(
                                                IndexSpaces.imod(
                                                    IndexSpaces.idiv(
                                                        IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 2
                                                    ),
                                                    2,
                                                ),
                                                1,
                                            ),
                                            1,
                                        ),
                                        2,
                                    ),
                                    16,
                                ),
                                IndexSpaces.mul(
                                    IndexSpaces.imod(
                                        IndexSpaces.idiv(
                                            IndexSpaces.add(
                                                IndexSpaces.mul(
                                                    IndexSpaces.imod(
                                                        IndexSpaces.idiv(
                                                            IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1
                                                        ),
                                                        2,
                                                    ),
                                                    16,
                                                ),
                                                IndexSpaces.mul(
                                                    IndexSpaces.imod(
                                                        IndexSpaces.idiv(IndexSpaces.assume_inrange(T2::Int32, 0, 32, 128), 32), 4
                                                    ),
                                                    32,
                                                ),
                                                IndexSpaces.mul(
                                                    IndexSpaces.imod(
                                                        IndexSpaces.idiv(IndexSpaces.assume_inrange(T1::Int32, 0, 128, 32768), 128),
                                                        256,
                                                    ),
                                                    128,
                                                ),
                                                IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(8::Int32, 8), 2), 8),
                                                IndexSpaces.mul(
                                                    IndexSpaces.imod(
                                                        IndexSpaces.idiv(
                                                            IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32),
                                                            4,
                                                        ),
                                                        8,
                                                    ),
                                                    1,
                                                ),
                                            ),
                                            1,
                                        ),
                                        32768,
                                    ),
                                    12288,
                                ),
                                IndexSpaces.mul(
                                    IndexSpaces.imod(
                                        IndexSpaces.idiv(
                                            IndexSpaces.add(
                                                IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 4), 4), 4),
                                                IndexSpaces.mul(
                                                    IndexSpaces.imod(
                                                        IndexSpaces.idiv(
                                                            IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32),
                                                            1,
                                                        ),
                                                        4,
                                                    ),
                                                    16,
                                                ),
                                            ),
                                            4,
                                        ),
                                        16,
                                    ),
                                    1,
                                ),
                            ) + offset,
                            length,
                        )
                    end + 0x01,
                )
                E_shared[(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1), 2), 16), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(T2::Int32, 0, 32, 128), 32), 4), 32), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(T1::Int32, 0, 128, 32768), 128), 256), 128), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 8), 2), 8), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 4), 8), 1)), 1), 32), 17), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 2), 2), 1), 1), 2), 544), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 4), 4), 4), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 1), 4), 16)), 4), 16), 1)) + 0) + 0x01] =
                    E_dish0_time0
                E_shared[(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1), 2), 16), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(T2::Int32, 0, 32, 128), 32), 4), 32), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(T1::Int32, 0, 128, 32768), 128), 256), 128), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 8), 2), 8), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 4), 8), 1)), 1), 32), 17), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 2), 2), 1), 1), 2), 544), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(4::Int32, 4), 4), 4), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 1), 4), 16)), 4), 16), 1)) + 0) + 0x01] =
                    E_dish4_time0
                E_shared[(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1), 2), 16), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(T2::Int32, 0, 32, 128), 32), 4), 32), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(T1::Int32, 0, 128, 32768), 128), 256), 128), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 8), 2), 8), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 4), 8), 1)), 1), 32), 17), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 2), 2), 1), 1), 2), 544), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(8::Int32, 4), 4), 4), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 1), 4), 16)), 4), 16), 1)) + 0) + 0x01] =
                    E_dish8_time0
                E_shared[(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1), 2), 16), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(T2::Int32, 0, 32, 128), 32), 4), 32), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(T1::Int32, 0, 128, 32768), 128), 256), 128), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 8), 2), 8), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 4), 8), 1)), 1), 32), 17), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 2), 2), 1), 1), 2), 544), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(12::Int32, 4), 4), 4), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 1), 4), 16)), 4), 16), 1)) + 0) + 0x01] =
                    E_dish12_time0
                E_shared[(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1), 2), 16), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(T2::Int32, 0, 32, 128), 32), 4), 32), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(T1::Int32, 0, 128, 32768), 128), 256), 128), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(8::Int32, 8), 2), 8), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 4), 8), 1)), 1), 32), 17), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 2), 2), 1), 1), 2), 544), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 4), 4), 4), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 1), 4), 16)), 4), 16), 1)) + 0) + 0x01] =
                    E_dish0_time8
                E_shared[(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1), 2), 16), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(T2::Int32, 0, 32, 128), 32), 4), 32), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(T1::Int32, 0, 128, 32768), 128), 256), 128), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(8::Int32, 8), 2), 8), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 4), 8), 1)), 1), 32), 17), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 2), 2), 1), 1), 2), 544), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(4::Int32, 4), 4), 4), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 1), 4), 16)), 4), 16), 1)) + 0) + 0x01] =
                    E_dish4_time8
                E_shared[(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1), 2), 16), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(T2::Int32, 0, 32, 128), 32), 4), 32), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(T1::Int32, 0, 128, 32768), 128), 256), 128), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(8::Int32, 8), 2), 8), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 4), 8), 1)), 1), 32), 17), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 2), 2), 1), 1), 2), 544), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(8::Int32, 4), 4), 4), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 1), 4), 16)), 4), 16), 1)) + 0) + 0x01] =
                    E_dish8_time8
                E_shared[(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1), 2), 16), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(T2::Int32, 0, 32, 128), 32), 4), 32), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(T1::Int32, 0, 128, 32768), 128), 256), 128), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(8::Int32, 8), 2), 8), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 4), 8), 1)), 1), 32), 17), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 2), 2), 1), 1), 2), 544), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(12::Int32, 4), 4), 4), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 1), 4), 16)), 4), 16), 1)) + 0) + 0x01] =
                    E_dish12_time8
            end
            IndexSpaces.cuda_sync_threads()
            for T3 in 0:8:31
                let B = 0
                    AselB_cplx0_dish0 = A_cplx0_dish0
                    AselB_cplx1_dish0 = A_cplx1_dish0
                    AselB_cplx0_dish4 = A_cplx0_dish4
                    AselB_cplx1_dish4 = A_cplx1_dish4
                    AselB_cplx0_dish8 = A_cplx0_dish8
                    AselB_cplx1_dish8 = A_cplx1_dish8
                    AselB_cplx0_dish12 = A_cplx0_dish12
                    AselB_cplx1_dish12 = A_cplx1_dish12
                    Jurepos_time0 = 0
                    Jurepos_time1 = 0
                    Jureneg_time0 = 0
                    Jureneg_time1 = 0
                    Juim_time0 = 0
                    Juim_time1 = 0
                    let D = 0
                        AselBD_cplx0 = AselB_cplx0_dish0
                        AselBD_cplx1 = AselB_cplx1_dish0
                        Are = AselBD_cplx0
                        Aim = AselBD_cplx1
                        E0 = E_shared[IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(T2::Int32, 0, 32, 128), 32), 4), 32), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(T1::Int32, 0, 128, 32768), 128), 256), 128), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 4), 8), 1), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(T3::Int32, 0, 8, 32), 8), 4), 8)), 1), 32), 17), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 2), 2), 1), 1), 2), 544), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(D::Int32, 4), 4), 4), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 1), 4), 16)), 4), 16), 1)) + 0x01]
                        (E1_cplx0, E1_cplx1) = convert_swapped_withoffset(NTuple{2,Int8x4}, E0)
                        E1re = E1_cplx0
                        E1im = E1_cplx1
                        (Jurepos_time0, Jurepos_time1) = IndexSpaces.mma_m8n8k16(Are, E1re, (Jurepos_time0, Jurepos_time1))
                        (Jureneg_time0, Jureneg_time1) = IndexSpaces.mma_m8n8k16(Aim, E1im, (Jureneg_time0, Jureneg_time1))
                        (Juim_time0, Juim_time1) = IndexSpaces.mma_m8n8k16(Are, E1im, (Juim_time0, Juim_time1))
                        (Juim_time0, Juim_time1) = IndexSpaces.mma_m8n8k16(Aim, E1re, (Juim_time0, Juim_time1))
                    end
                    let D = 4
                        AselBD_cplx0 = AselB_cplx0_dish4
                        AselBD_cplx1 = AselB_cplx1_dish4
                        Are = AselBD_cplx0
                        Aim = AselBD_cplx1
                        E0 = E_shared[IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(T2::Int32, 0, 32, 128), 32), 4), 32), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(T1::Int32, 0, 128, 32768), 128), 256), 128), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 4), 8), 1), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(T3::Int32, 0, 8, 32), 8), 4), 8)), 1), 32), 17), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 2), 2), 1), 1), 2), 544), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(D::Int32, 4), 4), 4), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 1), 4), 16)), 4), 16), 1)) + 0x01]
                        (E1_cplx0, E1_cplx1) = convert_swapped_withoffset(NTuple{2,Int8x4}, E0)
                        E1re = E1_cplx0
                        E1im = E1_cplx1
                        (Jurepos_time0, Jurepos_time1) = IndexSpaces.mma_m8n8k16(Are, E1re, (Jurepos_time0, Jurepos_time1))
                        (Jureneg_time0, Jureneg_time1) = IndexSpaces.mma_m8n8k16(Aim, E1im, (Jureneg_time0, Jureneg_time1))
                        (Juim_time0, Juim_time1) = IndexSpaces.mma_m8n8k16(Are, E1im, (Juim_time0, Juim_time1))
                        (Juim_time0, Juim_time1) = IndexSpaces.mma_m8n8k16(Aim, E1re, (Juim_time0, Juim_time1))
                    end
                    let D = 8
                        AselBD_cplx0 = AselB_cplx0_dish8
                        AselBD_cplx1 = AselB_cplx1_dish8
                        Are = AselBD_cplx0
                        Aim = AselBD_cplx1
                        E0 = E_shared[IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(T2::Int32, 0, 32, 128), 32), 4), 32), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(T1::Int32, 0, 128, 32768), 128), 256), 128), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 4), 8), 1), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(T3::Int32, 0, 8, 32), 8), 4), 8)), 1), 32), 17), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 2), 2), 1), 1), 2), 544), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(D::Int32, 4), 4), 4), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 1), 4), 16)), 4), 16), 1)) + 0x01]
                        (E1_cplx0, E1_cplx1) = convert_swapped_withoffset(NTuple{2,Int8x4}, E0)
                        E1re = E1_cplx0
                        E1im = E1_cplx1
                        (Jurepos_time0, Jurepos_time1) = IndexSpaces.mma_m8n8k16(Are, E1re, (Jurepos_time0, Jurepos_time1))
                        (Jureneg_time0, Jureneg_time1) = IndexSpaces.mma_m8n8k16(Aim, E1im, (Jureneg_time0, Jureneg_time1))
                        (Juim_time0, Juim_time1) = IndexSpaces.mma_m8n8k16(Are, E1im, (Juim_time0, Juim_time1))
                        (Juim_time0, Juim_time1) = IndexSpaces.mma_m8n8k16(Aim, E1re, (Juim_time0, Juim_time1))
                    end
                    let D = 12
                        AselBD_cplx0 = AselB_cplx0_dish12
                        AselBD_cplx1 = AselB_cplx1_dish12
                        Are = AselBD_cplx0
                        Aim = AselBD_cplx1
                        E0 = E_shared[IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(T2::Int32, 0, 32, 128), 32), 4), 32), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(T1::Int32, 0, 128, 32768), 128), 256), 128), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 4), 8), 1), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(T3::Int32, 0, 8, 32), 8), 4), 8)), 1), 32), 17), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 2), 2), 1), 1), 2), 544), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(D::Int32, 4), 4), 4), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 1), 4), 16)), 4), 16), 1)) + 0x01]
                        (E1_cplx0, E1_cplx1) = convert_swapped_withoffset(NTuple{2,Int8x4}, E0)
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
                    IndexSpaces.unsafe_store2!(
                        Ju_shared,
                        (
                            IndexSpaces.add(
                                IndexSpaces.mul(
                                    IndexSpaces.imod(
                                        IndexSpaces.idiv(
                                            IndexSpaces.add(
                                                IndexSpaces.mul(
                                                    IndexSpaces.imod(
                                                        IndexSpaces.idiv(
                                                            IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1
                                                        ),
                                                        2,
                                                    ),
                                                    8,
                                                ),
                                                IndexSpaces.mul(
                                                    IndexSpaces.imod(
                                                        IndexSpaces.idiv(
                                                            IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32),
                                                            4,
                                                        ),
                                                        8,
                                                    ),
                                                    1,
                                                ),
                                            ),
                                            1,
                                        ),
                                        16,
                                    ),
                                    2,
                                ),
                                IndexSpaces.mul(
                                    IndexSpaces.imod(
                                        IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 1), 2), 1), 1),
                                        2,
                                    ),
                                    1,
                                ),
                                IndexSpaces.mul(
                                    IndexSpaces.imod(
                                        IndexSpaces.idiv(
                                            IndexSpaces.mul(
                                                IndexSpaces.imod(
                                                    IndexSpaces.idiv(
                                                        IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 2
                                                    ),
                                                    2,
                                                ),
                                                1,
                                            ),
                                            1,
                                        ),
                                        2,
                                    ),
                                    1152,
                                ),
                                IndexSpaces.mul(
                                    IndexSpaces.imod(
                                        IndexSpaces.idiv(
                                            IndexSpaces.add(
                                                IndexSpaces.mul(
                                                    IndexSpaces.imod(
                                                        IndexSpaces.idiv(
                                                            IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32),
                                                            1,
                                                        ),
                                                        4,
                                                    ),
                                                    2,
                                                ),
                                                IndexSpaces.mul(
                                                    IndexSpaces.imod(
                                                        IndexSpaces.idiv(IndexSpaces.assume_inrange(T2::Int32, 0, 32, 128), 32), 4
                                                    ),
                                                    32,
                                                ),
                                                IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 1), 2), 1),
                                                IndexSpaces.mul(
                                                    IndexSpaces.imod(
                                                        IndexSpaces.idiv(IndexSpaces.assume_inrange(T1::Int32, 0, 128, 32768), 128),
                                                        256,
                                                    ),
                                                    128,
                                                ),
                                                IndexSpaces.mul(
                                                    IndexSpaces.imod(
                                                        IndexSpaces.idiv(IndexSpaces.assume_inrange(T3::Int32, 0, 8, 32), 8), 4
                                                    ),
                                                    8,
                                                ),
                                            ),
                                            1,
                                        ),
                                        32,
                                    ),
                                    36,
                                ),
                            ) + 0
                        ) + 0x01,
                        (Ju_cplx0_time0, Ju_cplx1_time0),
                    )
                    IndexSpaces.unsafe_store2!(
                        Ju_shared,
                        (
                            IndexSpaces.add(
                                IndexSpaces.mul(
                                    IndexSpaces.imod(
                                        IndexSpaces.idiv(
                                            IndexSpaces.add(
                                                IndexSpaces.mul(
                                                    IndexSpaces.imod(
                                                        IndexSpaces.idiv(
                                                            IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1
                                                        ),
                                                        2,
                                                    ),
                                                    8,
                                                ),
                                                IndexSpaces.mul(
                                                    IndexSpaces.imod(
                                                        IndexSpaces.idiv(
                                                            IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32),
                                                            4,
                                                        ),
                                                        8,
                                                    ),
                                                    1,
                                                ),
                                            ),
                                            1,
                                        ),
                                        16,
                                    ),
                                    2,
                                ),
                                IndexSpaces.mul(
                                    IndexSpaces.imod(
                                        IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 1), 2), 1), 1),
                                        2,
                                    ),
                                    1,
                                ),
                                IndexSpaces.mul(
                                    IndexSpaces.imod(
                                        IndexSpaces.idiv(
                                            IndexSpaces.mul(
                                                IndexSpaces.imod(
                                                    IndexSpaces.idiv(
                                                        IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 2
                                                    ),
                                                    2,
                                                ),
                                                1,
                                            ),
                                            1,
                                        ),
                                        2,
                                    ),
                                    1152,
                                ),
                                IndexSpaces.mul(
                                    IndexSpaces.imod(
                                        IndexSpaces.idiv(
                                            IndexSpaces.add(
                                                IndexSpaces.mul(
                                                    IndexSpaces.imod(
                                                        IndexSpaces.idiv(
                                                            IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32),
                                                            1,
                                                        ),
                                                        4,
                                                    ),
                                                    2,
                                                ),
                                                IndexSpaces.mul(
                                                    IndexSpaces.imod(
                                                        IndexSpaces.idiv(IndexSpaces.assume_inrange(T2::Int32, 0, 32, 128), 32), 4
                                                    ),
                                                    32,
                                                ),
                                                IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(1::Int32, 1), 2), 1),
                                                IndexSpaces.mul(
                                                    IndexSpaces.imod(
                                                        IndexSpaces.idiv(IndexSpaces.assume_inrange(T1::Int32, 0, 128, 32768), 128),
                                                        256,
                                                    ),
                                                    128,
                                                ),
                                                IndexSpaces.mul(
                                                    IndexSpaces.imod(
                                                        IndexSpaces.idiv(IndexSpaces.assume_inrange(T3::Int32, 0, 8, 32), 8), 4
                                                    ),
                                                    8,
                                                ),
                                            ),
                                            1,
                                        ),
                                        32,
                                    ),
                                    36,
                                ),
                            ) + 0
                        ) + 0x01,
                        (Ju_cplx0_time1, Ju_cplx1_time1),
                    )
                end
            end
            IndexSpaces.cuda_sync_threads()
            (Ju_cplx0_polr0_time0, Ju_cplx1_polr0_time0) = IndexSpaces.unsafe_load2(
                Ju_shared,
                IndexSpaces.add(
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(
                                IndexSpaces.add(
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1
                                            ),
                                            4,
                                        ),
                                        4,
                                    ),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 8
                                            ),
                                            4,
                                        ),
                                        1,
                                    ),
                                ),
                                1,
                            ),
                            16,
                        ),
                        2,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 1), 2), 1), 1), 2
                        ),
                        1,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 1), 2), 1), 1), 2
                        ),
                        1152,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(
                                IndexSpaces.add(
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 1
                                            ),
                                            8,
                                        ),
                                        1,
                                    ),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(IndexSpaces.assume_inrange(T2::Int32, 0, 32, 128), 32), 4
                                        ),
                                        32,
                                    ),
                                    IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 8), 4), 8),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(IndexSpaces.assume_inrange(T1::Int32, 0, 128, 32768), 128), 256
                                        ),
                                        128,
                                    ),
                                ),
                                1,
                            ),
                            32,
                        ),
                        36,
                    ),
                ) + 0x01,
            )
            (Ju_cplx0_polr1_time0, Ju_cplx1_polr1_time0) = IndexSpaces.unsafe_load2(
                Ju_shared,
                IndexSpaces.add(
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(
                                IndexSpaces.add(
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1
                                            ),
                                            4,
                                        ),
                                        4,
                                    ),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 8
                                            ),
                                            4,
                                        ),
                                        1,
                                    ),
                                ),
                                1,
                            ),
                            16,
                        ),
                        2,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 1), 2), 1), 1), 2
                        ),
                        1,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(1::Int32, 1), 2), 1), 1), 2
                        ),
                        1152,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(
                                IndexSpaces.add(
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 1
                                            ),
                                            8,
                                        ),
                                        1,
                                    ),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(IndexSpaces.assume_inrange(T2::Int32, 0, 32, 128), 32), 4
                                        ),
                                        32,
                                    ),
                                    IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 8), 4), 8),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(IndexSpaces.assume_inrange(T1::Int32, 0, 128, 32768), 128), 256
                                        ),
                                        128,
                                    ),
                                ),
                                1,
                            ),
                            32,
                        ),
                        36,
                    ),
                ) + 0x01,
            )
            (Ju_cplx0_polr0_time8, Ju_cplx1_polr0_time8) = IndexSpaces.unsafe_load2(
                Ju_shared,
                IndexSpaces.add(
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(
                                IndexSpaces.add(
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1
                                            ),
                                            4,
                                        ),
                                        4,
                                    ),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 8
                                            ),
                                            4,
                                        ),
                                        1,
                                    ),
                                ),
                                1,
                            ),
                            16,
                        ),
                        2,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 1), 2), 1), 1), 2
                        ),
                        1,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 1), 2), 1), 1), 2
                        ),
                        1152,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(
                                IndexSpaces.add(
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 1
                                            ),
                                            8,
                                        ),
                                        1,
                                    ),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(IndexSpaces.assume_inrange(T2::Int32, 0, 32, 128), 32), 4
                                        ),
                                        32,
                                    ),
                                    IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(8::Int32, 8), 4), 8),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(IndexSpaces.assume_inrange(T1::Int32, 0, 128, 32768), 128), 256
                                        ),
                                        128,
                                    ),
                                ),
                                1,
                            ),
                            32,
                        ),
                        36,
                    ),
                ) + 0x01,
            )
            (Ju_cplx0_polr1_time8, Ju_cplx1_polr1_time8) = IndexSpaces.unsafe_load2(
                Ju_shared,
                IndexSpaces.add(
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(
                                IndexSpaces.add(
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1
                                            ),
                                            4,
                                        ),
                                        4,
                                    ),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 8
                                            ),
                                            4,
                                        ),
                                        1,
                                    ),
                                ),
                                1,
                            ),
                            16,
                        ),
                        2,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 1), 2), 1), 1), 2
                        ),
                        1,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(1::Int32, 1), 2), 1), 1), 2
                        ),
                        1152,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(
                                IndexSpaces.add(
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 1
                                            ),
                                            8,
                                        ),
                                        1,
                                    ),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(IndexSpaces.assume_inrange(T2::Int32, 0, 32, 128), 32), 4
                                        ),
                                        32,
                                    ),
                                    IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(8::Int32, 8), 4), 8),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(IndexSpaces.assume_inrange(T1::Int32, 0, 128, 32768), 128), 256
                                        ),
                                        128,
                                    ),
                                ),
                                1,
                            ),
                            32,
                        ),
                        36,
                    ),
                ) + 0x01,
            )
            (Ju_cplx0_polr0_time16, Ju_cplx1_polr0_time16) = IndexSpaces.unsafe_load2(
                Ju_shared,
                IndexSpaces.add(
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(
                                IndexSpaces.add(
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1
                                            ),
                                            4,
                                        ),
                                        4,
                                    ),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 8
                                            ),
                                            4,
                                        ),
                                        1,
                                    ),
                                ),
                                1,
                            ),
                            16,
                        ),
                        2,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 1), 2), 1), 1), 2
                        ),
                        1,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 1), 2), 1), 1), 2
                        ),
                        1152,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(
                                IndexSpaces.add(
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 1
                                            ),
                                            8,
                                        ),
                                        1,
                                    ),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(IndexSpaces.assume_inrange(T2::Int32, 0, 32, 128), 32), 4
                                        ),
                                        32,
                                    ),
                                    IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(16::Int32, 8), 4), 8),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(IndexSpaces.assume_inrange(T1::Int32, 0, 128, 32768), 128), 256
                                        ),
                                        128,
                                    ),
                                ),
                                1,
                            ),
                            32,
                        ),
                        36,
                    ),
                ) + 0x01,
            )
            (Ju_cplx0_polr1_time16, Ju_cplx1_polr1_time16) = IndexSpaces.unsafe_load2(
                Ju_shared,
                IndexSpaces.add(
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(
                                IndexSpaces.add(
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1
                                            ),
                                            4,
                                        ),
                                        4,
                                    ),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 8
                                            ),
                                            4,
                                        ),
                                        1,
                                    ),
                                ),
                                1,
                            ),
                            16,
                        ),
                        2,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 1), 2), 1), 1), 2
                        ),
                        1,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(1::Int32, 1), 2), 1), 1), 2
                        ),
                        1152,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(
                                IndexSpaces.add(
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 1
                                            ),
                                            8,
                                        ),
                                        1,
                                    ),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(IndexSpaces.assume_inrange(T2::Int32, 0, 32, 128), 32), 4
                                        ),
                                        32,
                                    ),
                                    IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(16::Int32, 8), 4), 8),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(IndexSpaces.assume_inrange(T1::Int32, 0, 128, 32768), 128), 256
                                        ),
                                        128,
                                    ),
                                ),
                                1,
                            ),
                            32,
                        ),
                        36,
                    ),
                ) + 0x01,
            )
            (Ju_cplx0_polr0_time24, Ju_cplx1_polr0_time24) = IndexSpaces.unsafe_load2(
                Ju_shared,
                IndexSpaces.add(
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(
                                IndexSpaces.add(
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1
                                            ),
                                            4,
                                        ),
                                        4,
                                    ),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 8
                                            ),
                                            4,
                                        ),
                                        1,
                                    ),
                                ),
                                1,
                            ),
                            16,
                        ),
                        2,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 1), 2), 1), 1), 2
                        ),
                        1,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 1), 2), 1), 1), 2
                        ),
                        1152,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(
                                IndexSpaces.add(
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 1
                                            ),
                                            8,
                                        ),
                                        1,
                                    ),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(IndexSpaces.assume_inrange(T2::Int32, 0, 32, 128), 32), 4
                                        ),
                                        32,
                                    ),
                                    IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(24::Int32, 8), 4), 8),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(IndexSpaces.assume_inrange(T1::Int32, 0, 128, 32768), 128), 256
                                        ),
                                        128,
                                    ),
                                ),
                                1,
                            ),
                            32,
                        ),
                        36,
                    ),
                ) + 0x01,
            )
            (Ju_cplx0_polr1_time24, Ju_cplx1_polr1_time24) = IndexSpaces.unsafe_load2(
                Ju_shared,
                IndexSpaces.add(
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(
                                IndexSpaces.add(
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1
                                            ),
                                            4,
                                        ),
                                        4,
                                    ),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 8
                                            ),
                                            4,
                                        ),
                                        1,
                                    ),
                                ),
                                1,
                            ),
                            16,
                        ),
                        2,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 1), 2), 1), 1), 2
                        ),
                        1,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(1::Int32, 1), 2), 1), 1), 2
                        ),
                        1152,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(
                                IndexSpaces.add(
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 1
                                            ),
                                            8,
                                        ),
                                        1,
                                    ),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(IndexSpaces.assume_inrange(T2::Int32, 0, 32, 128), 32), 4
                                        ),
                                        32,
                                    ),
                                    IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(24::Int32, 8), 4), 8),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(IndexSpaces.assume_inrange(T1::Int32, 0, 128, 32768), 128), 256
                                        ),
                                        128,
                                    ),
                                ),
                                1,
                            ),
                            32,
                        ),
                        36,
                    ),
                ) + 0x01,
            )
            J_cplx0_polr0_time0 = Ju_cplx0_polr0_time0
            J_cplx1_polr0_time0 = Ju_cplx1_polr0_time0
            J_cplx0_polr1_time0 = Ju_cplx0_polr1_time0
            J_cplx1_polr1_time0 = Ju_cplx1_polr1_time0
            J_cplx0_polr0_time8 = Ju_cplx0_polr0_time8
            J_cplx1_polr0_time8 = Ju_cplx1_polr0_time8
            J_cplx0_polr1_time8 = Ju_cplx0_polr1_time8
            J_cplx1_polr1_time8 = Ju_cplx1_polr1_time8
            J_cplx0_polr0_time16 = Ju_cplx0_polr0_time16
            J_cplx1_polr0_time16 = Ju_cplx1_polr0_time16
            J_cplx0_polr1_time16 = Ju_cplx0_polr1_time16
            J_cplx1_polr1_time16 = Ju_cplx1_polr1_time16
            J_cplx0_polr0_time24 = Ju_cplx0_polr0_time24
            J_cplx1_polr0_time24 = Ju_cplx1_polr0_time24
            J_cplx0_polr1_time24 = Ju_cplx0_polr1_time24
            J_cplx1_polr1_time24 = Ju_cplx1_polr1_time24
            J_cplx0_polr0_time0 = let
                Jscaled = (Int64(J_cplx0_polr0_time0) * Int64(S_polr0) + 8388608) >> 0x18
                Jnew = clamp(Jscaled, -7:7)
                hasoverflow |= Jnew != Jscaled
                Jnew % Int32
            end
            J_cplx1_polr0_time0 = let
                Jscaled = (Int64(J_cplx1_polr0_time0) * Int64(S_polr0) + 8388608) >> 0x18
                Jnew = clamp(Jscaled, -7:7)
                hasoverflow |= Jnew != Jscaled
                Jnew % Int32
            end
            J_cplx0_polr1_time0 = let
                Jscaled = (Int64(J_cplx0_polr1_time0) * Int64(S_polr1) + 8388608) >> 0x18
                Jnew = clamp(Jscaled, -7:7)
                hasoverflow |= Jnew != Jscaled
                Jnew % Int32
            end
            J_cplx1_polr1_time0 = let
                Jscaled = (Int64(J_cplx1_polr1_time0) * Int64(S_polr1) + 8388608) >> 0x18
                Jnew = clamp(Jscaled, -7:7)
                hasoverflow |= Jnew != Jscaled
                Jnew % Int32
            end
            J_cplx0_polr0_time8 = let
                Jscaled = (Int64(J_cplx0_polr0_time8) * Int64(S_polr0) + 8388608) >> 0x18
                Jnew = clamp(Jscaled, -7:7)
                hasoverflow |= Jnew != Jscaled
                Jnew % Int32
            end
            J_cplx1_polr0_time8 = let
                Jscaled = (Int64(J_cplx1_polr0_time8) * Int64(S_polr0) + 8388608) >> 0x18
                Jnew = clamp(Jscaled, -7:7)
                hasoverflow |= Jnew != Jscaled
                Jnew % Int32
            end
            J_cplx0_polr1_time8 = let
                Jscaled = (Int64(J_cplx0_polr1_time8) * Int64(S_polr1) + 8388608) >> 0x18
                Jnew = clamp(Jscaled, -7:7)
                hasoverflow |= Jnew != Jscaled
                Jnew % Int32
            end
            J_cplx1_polr1_time8 = let
                Jscaled = (Int64(J_cplx1_polr1_time8) * Int64(S_polr1) + 8388608) >> 0x18
                Jnew = clamp(Jscaled, -7:7)
                hasoverflow |= Jnew != Jscaled
                Jnew % Int32
            end
            J_cplx0_polr0_time16 = let
                Jscaled = (Int64(J_cplx0_polr0_time16) * Int64(S_polr0) + 8388608) >> 0x18
                Jnew = clamp(Jscaled, -7:7)
                hasoverflow |= Jnew != Jscaled
                Jnew % Int32
            end
            J_cplx1_polr0_time16 = let
                Jscaled = (Int64(J_cplx1_polr0_time16) * Int64(S_polr0) + 8388608) >> 0x18
                Jnew = clamp(Jscaled, -7:7)
                hasoverflow |= Jnew != Jscaled
                Jnew % Int32
            end
            J_cplx0_polr1_time16 = let
                Jscaled = (Int64(J_cplx0_polr1_time16) * Int64(S_polr1) + 8388608) >> 0x18
                Jnew = clamp(Jscaled, -7:7)
                hasoverflow |= Jnew != Jscaled
                Jnew % Int32
            end
            J_cplx1_polr1_time16 = let
                Jscaled = (Int64(J_cplx1_polr1_time16) * Int64(S_polr1) + 8388608) >> 0x18
                Jnew = clamp(Jscaled, -7:7)
                hasoverflow |= Jnew != Jscaled
                Jnew % Int32
            end
            J_cplx0_polr0_time24 = let
                Jscaled = (Int64(J_cplx0_polr0_time24) * Int64(S_polr0) + 8388608) >> 0x18
                Jnew = clamp(Jscaled, -7:7)
                hasoverflow |= Jnew != Jscaled
                Jnew % Int32
            end
            J_cplx1_polr0_time24 = let
                Jscaled = (Int64(J_cplx1_polr0_time24) * Int64(S_polr0) + 8388608) >> 0x18
                Jnew = clamp(Jscaled, -7:7)
                hasoverflow |= Jnew != Jscaled
                Jnew % Int32
            end
            J_cplx0_polr1_time24 = let
                Jscaled = (Int64(J_cplx0_polr1_time24) * Int64(S_polr1) + 8388608) >> 0x18
                Jnew = clamp(Jscaled, -7:7)
                hasoverflow |= Jnew != Jscaled
                Jnew % Int32
            end
            J_cplx1_polr1_time24 = let
                Jscaled = (Int64(J_cplx1_polr1_time24) * Int64(S_polr1) + 8388608) >> 0x18
                Jnew = clamp(Jscaled, -7:7)
                hasoverflow |= Jnew != Jscaled
                Jnew % Int32
            end
            J_polr0 = Int4x8(
                J_cplx1_polr0_time0,
                J_cplx0_polr0_time0,
                J_cplx1_polr0_time8,
                J_cplx0_polr0_time8,
                J_cplx1_polr0_time16,
                J_cplx0_polr0_time16,
                J_cplx1_polr0_time24,
                J_cplx0_polr0_time24,
            )
            J_polr0 ⊻= Int4x8(0x88888888)
            J_polr1 = Int4x8(
                J_cplx1_polr1_time0,
                J_cplx0_polr1_time0,
                J_cplx1_polr1_time8,
                J_cplx0_polr1_time8,
                J_cplx1_polr1_time16,
                J_cplx0_polr1_time16,
                J_cplx1_polr1_time24,
                J_cplx0_polr1_time24,
            )
            J_polr1 ⊻= Int4x8(0x88888888)
            if T2 == 0
                Jper_polr0_time0 = J_polr0
            end
            if T2 == 32
                Jper_polr0_time32 = J_polr0
            end
            if T2 == 64
                Jper_polr0_time64 = J_polr0
            end
            if T2 == 96
                Jper_polr0_time96 = J_polr0
            end
            if T2 == 0
                Jper_polr1_time0 = J_polr1
            end
            if T2 == 32
                Jper_polr1_time32 = J_polr1
            end
            if T2 == 64
                Jper_polr1_time64 = J_polr1
            end
            if T2 == 96
                Jper_polr1_time96 = J_polr1
            end
        end
        (Jper_polr0_time0, Jper_polr0_time32) = (
            IndexSpaces.get_lo8(Jper_polr0_time0, Jper_polr0_time32), IndexSpaces.get_hi8(Jper_polr0_time0, Jper_polr0_time32)
        )
        (Jper_polr1_time0, Jper_polr1_time32) = (
            IndexSpaces.get_lo8(Jper_polr1_time0, Jper_polr1_time32), IndexSpaces.get_hi8(Jper_polr1_time0, Jper_polr1_time32)
        )
        (Jper_polr0_time64, Jper_polr0_time96) = (
            IndexSpaces.get_lo8(Jper_polr0_time64, Jper_polr0_time96), IndexSpaces.get_hi8(Jper_polr0_time64, Jper_polr0_time96)
        )
        (Jper_polr1_time64, Jper_polr1_time96) = (
            IndexSpaces.get_lo8(Jper_polr1_time64, Jper_polr1_time96), IndexSpaces.get_hi8(Jper_polr1_time64, Jper_polr1_time96)
        )
        is_lo_thread = IndexSpaces.cuda_threadidx() & 0x00000001 == 0x00
        (Jper_polr0_time0, Jper_polr0_time32) = let
            src = if is_lo_thread
                Jper_polr0_time32
            else
                Jper_polr0_time0
            end
            dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000001)
            if is_lo_thread
                (Jper_polr0_time0, dst)
            else
                (dst, Jper_polr0_time32)
            end
        end
        (Jper_polr1_time0, Jper_polr1_time32) = let
            src = if is_lo_thread
                Jper_polr1_time32
            else
                Jper_polr1_time0
            end
            dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000001)
            if is_lo_thread
                (Jper_polr1_time0, dst)
            else
                (dst, Jper_polr1_time32)
            end
        end
        (Jper_polr0_time64, Jper_polr0_time96) = let
            src = if is_lo_thread
                Jper_polr0_time96
            else
                Jper_polr0_time64
            end
            dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000001)
            if is_lo_thread
                (Jper_polr0_time64, dst)
            else
                (dst, Jper_polr0_time96)
            end
        end
        (Jper_polr1_time64, Jper_polr1_time96) = let
            src = if is_lo_thread
                Jper_polr1_time96
            else
                Jper_polr1_time64
            end
            dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000001)
            if is_lo_thread
                (Jper_polr1_time64, dst)
            else
                (dst, Jper_polr1_time96)
            end
        end
        (Jper_polr0_time0, Jper_polr0_time32) = (
            IndexSpaces.get_lo8(Jper_polr0_time0, Jper_polr0_time32), IndexSpaces.get_hi8(Jper_polr0_time0, Jper_polr0_time32)
        )
        (Jper_polr1_time0, Jper_polr1_time32) = (
            IndexSpaces.get_lo8(Jper_polr1_time0, Jper_polr1_time32), IndexSpaces.get_hi8(Jper_polr1_time0, Jper_polr1_time32)
        )
        (Jper_polr0_time64, Jper_polr0_time96) = (
            IndexSpaces.get_lo8(Jper_polr0_time64, Jper_polr0_time96), IndexSpaces.get_hi8(Jper_polr0_time64, Jper_polr0_time96)
        )
        (Jper_polr1_time64, Jper_polr1_time96) = (
            IndexSpaces.get_lo8(Jper_polr1_time64, Jper_polr1_time96), IndexSpaces.get_hi8(Jper_polr1_time64, Jper_polr1_time96)
        )
        is_lo_thread = IndexSpaces.cuda_threadidx() & 0x00000002 == 0x00
        (Jper_polr0_time0, Jper_polr0_time64) = let
            src = if is_lo_thread
                Jper_polr0_time64
            else
                Jper_polr0_time0
            end
            dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
            if is_lo_thread
                (Jper_polr0_time0, dst)
            else
                (dst, Jper_polr0_time64)
            end
        end
        (Jper_polr1_time0, Jper_polr1_time64) = let
            src = if is_lo_thread
                Jper_polr1_time64
            else
                Jper_polr1_time0
            end
            dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
            if is_lo_thread
                (Jper_polr1_time0, dst)
            else
                (dst, Jper_polr1_time64)
            end
        end
        (Jper_polr0_time32, Jper_polr0_time96) = let
            src = if is_lo_thread
                Jper_polr0_time96
            else
                Jper_polr0_time32
            end
            dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
            if is_lo_thread
                (Jper_polr0_time32, dst)
            else
                (dst, Jper_polr0_time96)
            end
        end
        (Jper_polr1_time32, Jper_polr1_time96) = let
            src = if is_lo_thread
                Jper_polr1_time96
            else
                Jper_polr1_time32
            end
            dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
            if is_lo_thread
                (Jper_polr1_time32, dst)
            else
                (dst, Jper_polr1_time96)
            end
        end
        (Jper_polr0_time0, Jper_polr0_time64) = (
            IndexSpaces.get_lo16(Jper_polr0_time0, Jper_polr0_time64), IndexSpaces.get_hi16(Jper_polr0_time0, Jper_polr0_time64)
        )
        (Jper_polr1_time0, Jper_polr1_time64) = (
            IndexSpaces.get_lo16(Jper_polr1_time0, Jper_polr1_time64), IndexSpaces.get_hi16(Jper_polr1_time0, Jper_polr1_time64)
        )
        (Jper_polr0_time32, Jper_polr0_time96) = (
            IndexSpaces.get_lo16(Jper_polr0_time32, Jper_polr0_time96), IndexSpaces.get_hi16(Jper_polr0_time32, Jper_polr0_time96)
        )
        (Jper_polr1_time32, Jper_polr1_time96) = (
            IndexSpaces.get_lo16(Jper_polr1_time32, Jper_polr1_time96), IndexSpaces.get_hi16(Jper_polr1_time32, Jper_polr1_time96)
        )
        is_lo_thread = IndexSpaces.cuda_threadidx() & 0x00000004 == 0x00
        (Jper_polr0_time0, Jper_polr0_time32) = let
            src = if is_lo_thread
                Jper_polr0_time32
            else
                Jper_polr0_time0
            end
            dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
            if is_lo_thread
                (Jper_polr0_time0, dst)
            else
                (dst, Jper_polr0_time32)
            end
        end
        (Jper_polr1_time0, Jper_polr1_time32) = let
            src = if is_lo_thread
                Jper_polr1_time32
            else
                Jper_polr1_time0
            end
            dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
            if is_lo_thread
                (Jper_polr1_time0, dst)
            else
                (dst, Jper_polr1_time32)
            end
        end
        (Jper_polr0_time64, Jper_polr0_time96) = let
            src = if is_lo_thread
                Jper_polr0_time96
            else
                Jper_polr0_time64
            end
            dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
            if is_lo_thread
                (Jper_polr0_time64, dst)
            else
                (dst, Jper_polr0_time96)
            end
        end
        (Jper_polr1_time64, Jper_polr1_time96) = let
            src = if is_lo_thread
                Jper_polr1_time96
            else
                Jper_polr1_time64
            end
            dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
            if is_lo_thread
                (Jper_polr1_time64, dst)
            else
                (dst, Jper_polr1_time96)
            end
        end
        is_lo_thread = IndexSpaces.cuda_threadidx() & 0x00000001 == 0x00
        (Jper_polr0_time0, Jper_polr0_time64) = let
            src = if is_lo_thread
                Jper_polr0_time64
            else
                Jper_polr0_time0
            end
            dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000001)
            if is_lo_thread
                (Jper_polr0_time0, dst)
            else
                (dst, Jper_polr0_time64)
            end
        end
        (Jper_polr1_time0, Jper_polr1_time64) = let
            src = if is_lo_thread
                Jper_polr1_time64
            else
                Jper_polr1_time0
            end
            dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000001)
            if is_lo_thread
                (Jper_polr1_time0, dst)
            else
                (dst, Jper_polr1_time64)
            end
        end
        (Jper_polr0_time32, Jper_polr0_time96) = let
            src = if is_lo_thread
                Jper_polr0_time96
            else
                Jper_polr0_time32
            end
            dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000001)
            if is_lo_thread
                (Jper_polr0_time32, dst)
            else
                (dst, Jper_polr0_time96)
            end
        end
        (Jper_polr1_time32, Jper_polr1_time96) = let
            src = if is_lo_thread
                Jper_polr1_time96
            else
                Jper_polr1_time32
            end
            dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000001)
            if is_lo_thread
                (Jper_polr1_time32, dst)
            else
                (dst, Jper_polr1_time96)
            end
        end
        IndexSpaces.unsafe_store4!(
            J_memory,
            (
                IndexSpaces.add(
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(
                                IndexSpaces.add(
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1
                                            ),
                                            4,
                                        ),
                                        4,
                                    ),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 8
                                            ),
                                            4,
                                        ),
                                        1,
                                    ),
                                ),
                                1,
                            ),
                            16,
                        ),
                        1572864,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(
                                IndexSpaces.add(
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 1
                                            ),
                                            2,
                                        ),
                                        16,
                                    ),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 4
                                            ),
                                            2,
                                        ),
                                        32,
                                    ),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(IndexSpaces.assume_inrange(T1::Int32, 0, 128, 32768), 128), 256
                                        ),
                                        128,
                                    ),
                                    IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 32), 4), 4),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 2
                                            ),
                                            2,
                                        ),
                                        64,
                                    ),
                                ),
                                4,
                            ),
                            2048,
                        ),
                        1,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 1), 2), 1), 1), 2
                        ),
                        2048,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(
                                IndexSpaces.mul(
                                    IndexSpaces.imod(
                                        IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384), 1),
                                        384,
                                    ),
                                    1,
                                ),
                                1,
                            ),
                            384,
                        ),
                        4096,
                    ),
                ) + 0
            ) + 0x01,
            (Jper_polr0_time0, Jper_polr0_time32, Jper_polr0_time64, Jper_polr0_time96),
        )
        IndexSpaces.unsafe_store4!(
            J_memory,
            (
                IndexSpaces.add(
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(
                                IndexSpaces.add(
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1
                                            ),
                                            4,
                                        ),
                                        4,
                                    ),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 8
                                            ),
                                            4,
                                        ),
                                        1,
                                    ),
                                ),
                                1,
                            ),
                            16,
                        ),
                        1572864,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(
                                IndexSpaces.add(
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 1
                                            ),
                                            2,
                                        ),
                                        16,
                                    ),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 4
                                            ),
                                            2,
                                        ),
                                        32,
                                    ),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(IndexSpaces.assume_inrange(T1::Int32, 0, 128, 32768), 128), 256
                                        ),
                                        128,
                                    ),
                                    IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(0::Int32, 32), 4), 4),
                                    IndexSpaces.mul(
                                        IndexSpaces.imod(
                                            IndexSpaces.idiv(
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 2
                                            ),
                                            2,
                                        ),
                                        64,
                                    ),
                                ),
                                4,
                            ),
                            2048,
                        ),
                        1,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(1::Int32, 1), 2), 1), 1), 2
                        ),
                        2048,
                    ),
                    IndexSpaces.mul(
                        IndexSpaces.imod(
                            IndexSpaces.idiv(
                                IndexSpaces.mul(
                                    IndexSpaces.imod(
                                        IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384), 1),
                                        384,
                                    ),
                                    1,
                                ),
                                1,
                            ),
                            384,
                        ),
                        4096,
                    ),
                ) + 0
            ) + 0x01,
            (Jper_polr1_time0, Jper_polr1_time32, Jper_polr1_time64, Jper_polr1_time96),
        )
    end
    any_hasoverflow = sync_threads_or(hasoverflow)
    if any_hasoverflow
        if let
            thread = IndexSpaces.cuda_threadidx()
            warp = IndexSpaces.cuda_warpidx()
            thread == 0i32 && warp == 0i32
        end
            logval = 1
            log_memory[(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384), 1), 384), 1), 1), 384), 1) + 0) + 0x01] =
                logval
        end
    end
    info = 0
    info_memory[(IndexSpaces.add(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384), 1), 384), 1), 1), 384), 128), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32), 1), 32), 1), 1), 32), 1), IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.mul(IndexSpaces.imod(IndexSpaces.idiv(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 4), 1), 4), 1), 1), 4), 32)) + 0) + 0x01] =
        info
end
