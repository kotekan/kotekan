# Julia source code for CUDA frb beamformer
# This file has been generated automatically by `frb.jl`.
# Do not modify this file, your changes will be lost.

@fastmath @inbounds(
    begin #= /localhome/eschnett/src/kotekan/julia/kernels/frb.jl:1861 =#
        info = 1
        if true
            info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                info
        end
        if !(
            0i32 ≤ Tmin < 256 && (
                Tmin ≤ Tmax < 512 &&
                ((Tmax - Tmin) % 48 == 0i32 && (0i32 ≤ T̄min < 64 && (T̄min ≤ T̄max < 128 && T̄max - T̄min == (Tmax - Tmin) ÷ 40)))
            )
        )
            info = 2
            if true
                info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                    info
            end
            IndexSpaces.cuda_trap()
        end
        if !(
            0i32 ≤ Fmin ≤ Fmax ≤ 6144 && (
                (Fmax - Fmin) % 128 == 0i32 &&
                (0i32 ≤ F̄min ≤ F̄max ≤ 6144 && ((F̄max - F̄min) % 128 == 0i32 && F̄max - F̄min == Fmax - Fmin))
            )
        )
            info = 3
            if true
                info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                    info
            end
            IndexSpaces.cuda_trap()
        end
        (Γ¹_re_re, Γ¹_re_im, Γ¹_im_re, Γ¹_im_im) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            c = thread % (4i32)
            v = thread ÷ (4i32)
            Γ¹ = cispi((((c * v) % (8i32)) / 4.0f0) % 2.0f0)
            (+(Γ¹.re), -(Γ¹.im), +(Γ¹.im), +(Γ¹.re))
        end
        Γ¹_re = Float16x2(Γ¹_re_im, Γ¹_re_re)
        Γ¹_im = Float16x2(Γ¹_im_im, Γ¹_im_re)
        aΓ¹_cplx0 = Γ¹_re
        aΓ¹_cplx1 = Γ¹_im
        (Γ²_d0_re, Γ²_d0_im, Γ²_d1_re, Γ²_d1_im) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            d0 = (thread % 4) * (2i32) + 0i32
            d1 = (thread % 4) * (2i32) + 1i32
            v = thread ÷ (4i32)
            δ0 = (δ1 = (Γ²_d0 = if d0 < 6
                cispi((((d0 * v) % 48) / 24.0f0) % 2.0f0)
            else
                Complex(0.0f0)
            end))
            Γ²_d1 = if d1 < 6
                cispi((((d1 * v) % 48) / 24.0f0) % 2.0f0)
            else
                Complex(0.0f0)
            end
            (Γ²_d0.re, Γ²_d0.im, Γ²_d1.re, Γ²_d1.im)
        end
        Γ²_re = Float16x2(Γ²_d0_re, Γ²_d1_re)
        Γ²_im = Float16x2(Γ²_d0_im, Γ²_d1_im)
        aΓ²_cplx0 = Γ²_re
        aΓ²_cplx1 = Γ²_im
        (Γ³_d0_re_re, Γ³_d0_re_im, Γ³_d0_im_re, Γ³_d0_im_im, Γ³_d1_re_re, Γ³_d1_re_im, Γ³_d1_im_re, Γ³_d1_im_im) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            d0 = (thread % (4i32)) * (2i32) + 0i32
            d1 = (thread % (4i32)) * (2i32) + 1i32
            u = (thread ÷ (4i32)) % (8i32)
            Γ³_d0 = if d0 < 6 && u < 6
                cispi((((d0 * u) % 6) / 3.0f0) % 2.0f0)
            else
                Complex(0.0f0)
            end
            Γ³_d1 = if d1 < 6 && u < 6
                cispi((((d1 * u) % 6) / 3.0f0) % 2.0f0)
            else
                Complex(0.0f0)
            end
            (+(Γ³_d0.re), -(Γ³_d0.im), +(Γ³_d0.im), +(Γ³_d0.re), +(Γ³_d1.re), -(Γ³_d1.im), +(Γ³_d1.im), +(Γ³_d1.re))
        end
        Γ³_re_re = Float16x2(Γ³_d0_re_re, Γ³_d1_re_re)
        Γ³_re_im = Float16x2(Γ³_d0_re_im, Γ³_d1_re_im)
        Γ³_im_re = Float16x2(Γ³_d0_im_re, Γ³_d1_im_re)
        Γ³_im_im = Float16x2(Γ³_d0_im_im, Γ³_d1_im_im)
        Γ³_re_cplx_in0 = Γ³_re_re
        Γ³_re_cplx_in1 = Γ³_re_im
        Γ³_im_cplx_in0 = Γ³_im_re
        Γ³_im_cplx_in1 = Γ³_im_im
        aΓ³_cplx0_cplx_in0 = Γ³_re_cplx_in0
        aΓ³_cplx1_cplx_in0 = Γ³_im_cplx_in0
        aΓ³_cplx0_cplx_in1 = Γ³_re_cplx_in1
        aΓ³_cplx1_cplx_in1 = Γ³_im_cplx_in1
        (Γ¹_re_re, Γ¹_re_im, Γ¹_im_re, Γ¹_im_im) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            c = thread % (4i32)
            v = thread ÷ (4i32)
            Γ¹ = cispi((((c * v) % (8i32)) / 4.0f0) % 2.0f0)
            (+(Γ¹.re), -(Γ¹.im), +(Γ¹.im), +(Γ¹.re))
        end
        Γ¹_re = Float16x2(Γ¹_re_im, Γ¹_re_re)
        Γ¹_im = Float16x2(Γ¹_im_im, Γ¹_im_re)
        bΓ¹_cplx0 = Γ¹_re
        bΓ¹_cplx1 = Γ¹_im
        (Γ²_d0_re, Γ²_d0_im, Γ²_d1_re, Γ²_d1_im) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            d0 = (thread % 4) * (2i32) + 0i32
            d1 = (thread % 4) * (2i32) + 1i32
            v = thread ÷ (4i32)
            δ0 = (δ1 = (Γ²_d0 = if d0 < 6
                cispi((((d0 * v) % 48) / 24.0f0) % 2.0f0)
            else
                Complex(0.0f0)
            end))
            Γ²_d1 = if d1 < 6
                cispi((((d1 * v) % 48) / 24.0f0) % 2.0f0)
            else
                Complex(0.0f0)
            end
            (Γ²_d0.re, Γ²_d0.im, Γ²_d1.re, Γ²_d1.im)
        end
        Γ²_re = Float16x2(Γ²_d0_re, Γ²_d1_re)
        Γ²_im = Float16x2(Γ²_d0_im, Γ²_d1_im)
        bΓ²_cplx0 = Γ²_re
        bΓ²_cplx1 = Γ²_im
        (Γ³_d0_re_re, Γ³_d0_re_im, Γ³_d0_im_re, Γ³_d0_im_im, Γ³_d1_re_re, Γ³_d1_re_im, Γ³_d1_im_re, Γ³_d1_im_im) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            d0 = (thread % (4i32)) * (2i32) + 0i32
            d1 = (thread % (4i32)) * (2i32) + 1i32
            u = (thread ÷ (4i32)) % (8i32)
            Γ³_d0 = if d0 < 6 && u < 6
                cispi((((d0 * u) % 6) / 3.0f0) % 2.0f0)
            else
                Complex(0.0f0)
            end
            Γ³_d1 = if d1 < 6 && u < 6
                cispi((((d1 * u) % 6) / 3.0f0) % 2.0f0)
            else
                Complex(0.0f0)
            end
            (+(Γ³_d0.re), -(Γ³_d0.im), +(Γ³_d0.im), +(Γ³_d0.re), +(Γ³_d1.re), -(Γ³_d1.im), +(Γ³_d1.im), +(Γ³_d1.re))
        end
        Γ³_re_re = Float16x2(Γ³_d0_re_re, Γ³_d1_re_re)
        Γ³_re_im = Float16x2(Γ³_d0_re_im, Γ³_d1_re_im)
        Γ³_im_re = Float16x2(Γ³_d0_im_re, Γ³_d1_im_re)
        Γ³_im_im = Float16x2(Γ³_d0_im_im, Γ³_d1_im_im)
        Γ³_re_cplx_in0 = Γ³_re_re
        Γ³_re_cplx_in1 = Γ³_re_im
        Γ³_im_cplx_in0 = Γ³_im_re
        Γ³_im_cplx_in1 = Γ³_im_im
        bΓ³_cplx0_cplx_in0 = Γ³_re_cplx_in0
        bΓ³_cplx1_cplx_in0 = Γ³_im_cplx_in0
        bΓ³_cplx0_cplx_in1 = Γ³_re_cplx_in1
        bΓ³_cplx1_cplx_in1 = Γ³_im_cplx_in1
        S = 999999999
        if let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            thread < 24
        end
            Smn = Smn_memory[(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24) * 24) % 576 + 0x01]
            (Smn_mn0, Smn_mn1) = convert(NTuple{2,Int32}, Smn)
            Sm = Smn_mn0
            Sn = Smn_mn1
            if !(0i32 ≤ Sm < 24 && 0i32 ≤ Sn < 24)
                CUDA.@cuprintf "thread=%d warp=%d block=%d Sm=%d Sn=%d\n" Cint((threadIdx()).x - 1) Cint((threadIdx()).y - 1) Cint(
                    (blockIdx()).x - 1
                ) Cint(Sm) Cint(Sn)                    #= /localhome/eschnett/src/kotekan/julia/kernels/frb.jl:1643 =#
                info = 4
                if true
                    info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                        info
                end
                IndexSpaces.cuda_trap()
            end
            S = (33i32) * Sm + 801 * Sn
        end
        W_polr0 = zero(Float16x2)
        W_polr1 = zero(Float16x2)
        if let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            nlo = 2 * (thread ÷ 8)
            nlo < 6
        end
            W_polr0 = W_memory[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 1152 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 144 + 0) + 0x01]
            W_polr1 = W_memory[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 1152 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 144 + 576) + 0x01]
        end
        I_beamQ0 = zero(Float16x2)
        I_beamQ24 = zero(Float16x2)
        dstime = 0
        t_running = 0
        for t_outer in 0:48:239
            Tmin + t_outer ≥ Tmax && break
            let
                (E_dish0_time0, E_dish4_time0, E_dish8_time0, E_dish12_time0) = IndexSpaces.unsafe_load4_global(
                    E_memory,
                    let
                        offset = 1572864 * Tmin + 256 * Fmin
                        length = 402653184
                        mod(
                            (
                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 256 +
                                (
                                    (
                                        IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 +
                                        ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48
                                    ) % 256
                                ) * 1572864 +
                                (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) ÷ 4) % 128 +
                                (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 2) * 128
                            ) + offset,
                            length,
                        )
                    end + 1i32,
                )
                (E_dish256_time0, E_dish260_time0, E_dish264_time0, E_dish268_time0) = IndexSpaces.unsafe_load4_global(
                    E_memory,
                    let
                        offset = 1572864 * Tmin + 256 * Fmin
                        length = 402653184
                        mod(
                            (
                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 256 +
                                (
                                    (
                                        IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 +
                                        ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48
                                    ) % 256
                                ) * 1572864 +
                                ((256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) ÷ 4) % 128 +
                                (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 2) * 128
                            ) + offset,
                            length,
                        )
                    end + 1i32,
                )
                (E_dish0_time24, E_dish4_time24, E_dish8_time24, E_dish12_time24) = IndexSpaces.unsafe_load4_global(
                    E_memory,
                    let
                        offset = 1572864 * Tmin + 256 * Fmin
                        length = 402653184
                        mod(
                            (
                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 256 +
                                (
                                    (
                                        (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 24) +
                                        ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48
                                    ) % 256
                                ) * 1572864 +
                                (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) ÷ 4) % 128 +
                                (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 2) * 128
                            ) + offset,
                            length,
                        )
                    end + 1i32,
                )
                (E_dish256_time24, E_dish260_time24, E_dish264_time24, E_dish268_time24) = IndexSpaces.unsafe_load4_global(
                    E_memory,
                    let
                        offset = 1572864 * Tmin + 256 * Fmin
                        length = 402653184
                        mod(
                            (
                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 256 +
                                (
                                    (
                                        (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 24) +
                                        ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48
                                    ) % 256
                                ) * 1572864 +
                                ((256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) ÷ 4) % 128 +
                                (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 2) * 128
                            ) + offset,
                            length,
                        )
                    end + 1i32,
                )
                is_lo_thread = IndexSpaces.cuda_threadidx() & 0x00000010 == 0x00
                (E_dish0_time0, E_dish8_time0) = let
                    src = if is_lo_thread
                        E_dish8_time0
                    else
                        E_dish0_time0
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                    if is_lo_thread
                        (E_dish0_time0, dst)
                    else
                        (dst, E_dish8_time0)
                    end
                end
                (E_dish4_time0, E_dish12_time0) = let
                    src = if is_lo_thread
                        E_dish12_time0
                    else
                        E_dish4_time0
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                    if is_lo_thread
                        (E_dish4_time0, dst)
                    else
                        (dst, E_dish12_time0)
                    end
                end
                (E_dish256_time0, E_dish264_time0) = let
                    src = if is_lo_thread
                        E_dish264_time0
                    else
                        E_dish256_time0
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                    if is_lo_thread
                        (E_dish256_time0, dst)
                    else
                        (dst, E_dish264_time0)
                    end
                end
                (E_dish260_time0, E_dish268_time0) = let
                    src = if is_lo_thread
                        E_dish268_time0
                    else
                        E_dish260_time0
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                    if is_lo_thread
                        (E_dish260_time0, dst)
                    else
                        (dst, E_dish268_time0)
                    end
                end
                (E_dish0_time24, E_dish8_time24) = let
                    src = if is_lo_thread
                        E_dish8_time24
                    else
                        E_dish0_time24
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                    if is_lo_thread
                        (E_dish0_time24, dst)
                    else
                        (dst, E_dish8_time24)
                    end
                end
                (E_dish4_time24, E_dish12_time24) = let
                    src = if is_lo_thread
                        E_dish12_time24
                    else
                        E_dish4_time24
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                    if is_lo_thread
                        (E_dish4_time24, dst)
                    else
                        (dst, E_dish12_time24)
                    end
                end
                (E_dish256_time24, E_dish264_time24) = let
                    src = if is_lo_thread
                        E_dish264_time24
                    else
                        E_dish256_time24
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                    if is_lo_thread
                        (E_dish256_time24, dst)
                    else
                        (dst, E_dish264_time24)
                    end
                end
                (E_dish260_time24, E_dish268_time24) = let
                    src = if is_lo_thread
                        E_dish268_time24
                    else
                        E_dish260_time24
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                    if is_lo_thread
                        (E_dish260_time24, dst)
                    else
                        (dst, E_dish268_time24)
                    end
                end
                (E_dish0_time0, E_dish8_time0) = (
                    IndexSpaces.get_lo4(E_dish0_time0, E_dish8_time0), IndexSpaces.get_hi4(E_dish0_time0, E_dish8_time0)
                )
                (E_dish4_time0, E_dish12_time0) = (
                    IndexSpaces.get_lo4(E_dish4_time0, E_dish12_time0), IndexSpaces.get_hi4(E_dish4_time0, E_dish12_time0)
                )
                (E_dish256_time0, E_dish264_time0) = (
                    IndexSpaces.get_lo4(E_dish256_time0, E_dish264_time0), IndexSpaces.get_hi4(E_dish256_time0, E_dish264_time0)
                )
                (E_dish260_time0, E_dish268_time0) = (
                    IndexSpaces.get_lo4(E_dish260_time0, E_dish268_time0), IndexSpaces.get_hi4(E_dish260_time0, E_dish268_time0)
                )
                (E_dish0_time24, E_dish8_time24) = (
                    IndexSpaces.get_lo4(E_dish0_time24, E_dish8_time24), IndexSpaces.get_hi4(E_dish0_time24, E_dish8_time24)
                )
                (E_dish4_time24, E_dish12_time24) = (
                    IndexSpaces.get_lo4(E_dish4_time24, E_dish12_time24), IndexSpaces.get_hi4(E_dish4_time24, E_dish12_time24)
                )
                (E_dish256_time24, E_dish264_time24) = (
                    IndexSpaces.get_lo4(E_dish256_time24, E_dish264_time24), IndexSpaces.get_hi4(E_dish256_time24, E_dish264_time24)
                )
                (E_dish260_time24, E_dish268_time24) = (
                    IndexSpaces.get_lo4(E_dish260_time24, E_dish268_time24), IndexSpaces.get_hi4(E_dish260_time24, E_dish268_time24)
                )
                (E_dish0_time0, E_dish0_time24) = (
                    IndexSpaces.get_lo8(E_dish0_time0, E_dish0_time24), IndexSpaces.get_hi8(E_dish0_time0, E_dish0_time24)
                )
                (E_dish4_time0, E_dish4_time24) = (
                    IndexSpaces.get_lo8(E_dish4_time0, E_dish4_time24), IndexSpaces.get_hi8(E_dish4_time0, E_dish4_time24)
                )
                (E_dish8_time0, E_dish8_time24) = (
                    IndexSpaces.get_lo8(E_dish8_time0, E_dish8_time24), IndexSpaces.get_hi8(E_dish8_time0, E_dish8_time24)
                )
                (E_dish12_time0, E_dish12_time24) = (
                    IndexSpaces.get_lo8(E_dish12_time0, E_dish12_time24), IndexSpaces.get_hi8(E_dish12_time0, E_dish12_time24)
                )
                (E_dish256_time0, E_dish256_time24) = (
                    IndexSpaces.get_lo8(E_dish256_time0, E_dish256_time24), IndexSpaces.get_hi8(E_dish256_time0, E_dish256_time24)
                )
                (E_dish260_time0, E_dish260_time24) = (
                    IndexSpaces.get_lo8(E_dish260_time0, E_dish260_time24), IndexSpaces.get_hi8(E_dish260_time0, E_dish260_time24)
                )
                (E_dish264_time0, E_dish264_time24) = (
                    IndexSpaces.get_lo8(E_dish264_time0, E_dish264_time24), IndexSpaces.get_hi8(E_dish264_time0, E_dish264_time24)
                )
                (E_dish268_time0, E_dish268_time24) = (
                    IndexSpaces.get_lo8(E_dish268_time0, E_dish268_time24), IndexSpaces.get_hi8(E_dish268_time0, E_dish268_time24)
                )
                (E_dish0_time0, E_dish8_time0) = (
                    IndexSpaces.get_lo16(E_dish0_time0, E_dish8_time0), IndexSpaces.get_hi16(E_dish0_time0, E_dish8_time0)
                )
                (E_dish4_time0, E_dish12_time0) = (
                    IndexSpaces.get_lo16(E_dish4_time0, E_dish12_time0), IndexSpaces.get_hi16(E_dish4_time0, E_dish12_time0)
                )
                (E_dish256_time0, E_dish264_time0) = (
                    IndexSpaces.get_lo16(E_dish256_time0, E_dish264_time0), IndexSpaces.get_hi16(E_dish256_time0, E_dish264_time0)
                )
                (E_dish260_time0, E_dish268_time0) = (
                    IndexSpaces.get_lo16(E_dish260_time0, E_dish268_time0), IndexSpaces.get_hi16(E_dish260_time0, E_dish268_time0)
                )
                (E_dish0_time24, E_dish8_time24) = (
                    IndexSpaces.get_lo16(E_dish0_time24, E_dish8_time24), IndexSpaces.get_hi16(E_dish0_time24, E_dish8_time24)
                )
                (E_dish4_time24, E_dish12_time24) = (
                    IndexSpaces.get_lo16(E_dish4_time24, E_dish12_time24), IndexSpaces.get_hi16(E_dish4_time24, E_dish12_time24)
                )
                (E_dish256_time24, E_dish264_time24) = (
                    IndexSpaces.get_lo16(E_dish256_time24, E_dish264_time24),
                    IndexSpaces.get_hi16(E_dish256_time24, E_dish264_time24),
                )
                (E_dish260_time24, E_dish268_time24) = (
                    IndexSpaces.get_lo16(E_dish260_time24, E_dish268_time24),
                    IndexSpaces.get_hi16(E_dish260_time24, E_dish268_time24),
                )
                if true
                    Fsh1_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) % 8) * 32 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) ÷ 8) % 64) * 257) + 0) + 0x01] =
                        E_dish0_time0
                end
                if true
                    Fsh1_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + (((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) % 8) * 32 + ((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) ÷ 8) % 64) * 257) + 0) + 0x01] =
                        E_dish4_time0
                end
                if true
                    Fsh1_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 2) % 8) * 32 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 2) ÷ 8) % 64) * 257) + 0) + 0x01] =
                        E_dish8_time0
                end
                if true
                    Fsh1_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 2) % 8) * 32 + (((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 2) ÷ 8) % 64) * 257) + 0) + 0x01] =
                        E_dish12_time0
                end
                if true
                    Fsh1_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8 + 256) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) % 8) * 32 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8 + 256) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) ÷ 8) % 64) * 257) + 0) + 0x01] =
                        E_dish256_time0
                end
                if true
                    Fsh1_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) + 256) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) % 8) * 32 + (((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) + 256) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) ÷ 8) % 64) * 257) + 0) + 0x01] =
                        E_dish260_time0
                end
                if true
                    Fsh1_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8 + 256) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 2) % 8) * 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8 + 256) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 2) ÷ 8) % 64) * 257) + 0) + 0x01] =
                        E_dish264_time0
                end
                if true
                    Fsh1_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + (((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) + 256) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 2) % 8) * 32 + ((((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) + 256) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 2) ÷ 8) % 64) * 257) + 0) + 0x01] =
                        E_dish268_time0
                end
                if true
                    Fsh1_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + (((1 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) % 8) * 32 + ((((1 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) ÷ 8) % 64) * 257) + 0) + 0x01] =
                        E_dish0_time24
                end
                if true
                    Fsh1_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + (((5 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) % 8) * 32 + ((((5 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) ÷ 8) % 64) * 257) + 0) + 0x01] =
                        E_dish4_time24
                end
                if true
                    Fsh1_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((((1 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 2) % 8) * 32 + (((((1 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 2) ÷ 8) % 64) * 257) + 0) + 0x01] =
                        E_dish8_time24
                end
                if true
                    Fsh1_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((((5 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 2) % 8) * 32 + (((((5 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 2) ÷ 8) % 64) * 257) + 0) + 0x01] =
                        E_dish12_time24
                end
                if true
                    Fsh1_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((((1 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) + 256) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) % 8) * 32 + (((((1 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) + 256) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) ÷ 8) % 64) * 257) + 0) + 0x01] =
                        E_dish256_time24
                end
                if true
                    Fsh1_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((((5 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) + 256) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) % 8) * 32 + (((((5 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) + 256) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) ÷ 8) % 64) * 257) + 0) + 0x01] =
                        E_dish260_time24
                end
                if true
                    Fsh1_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + (((((1 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) + 256) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 2) % 8) * 32 + ((((((1 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) + 256) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 2) ÷ 8) % 64) * 257) + 0) + 0x01] =
                        E_dish264_time24
                end
                if true
                    Fsh1_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + (((((5 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) + 256) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 2) % 8) * 32 + ((((((5 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) + 256) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 2) ÷ 8) % 64) * 257) + 0) + 0x01] =
                        E_dish268_time24
                end
                IndexSpaces.cuda_sync_threads()
            end
            let
                Freg1_dish0 = Fsh1_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 8) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 64) * 257) + 0x01]
                Freg1_dish24 = Fsh1_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 8) * 32 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 64) * 257) + 0x01]
                Freg1_dish48 = Fsh1_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 8) * 32 + (((48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 64) * 257) + 0x01]
                Freg1_dish72 = Fsh1_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((72 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 8) * 32 + (((72 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 64) * 257) + 0x01]
                Freg1_dish96 = Fsh1_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((96 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 8) * 32 + (((96 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 64) * 257) + 0x01]
                Freg1_dish120 = Fsh1_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((120 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 8) * 32 + (((120 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 64) * 257) + 0x01]
                Freg1_dish144 = Fsh1_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((144 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 8) * 32 + (((144 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 64) * 257) + 0x01]
                Freg1_dish168 = Fsh1_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((168 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 8) * 32 + (((168 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 64) * 257) + 0x01]
                Freg1_dish192 = Fsh1_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((192 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 8) * 32 + (((192 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 64) * 257) + 0x01]
                Freg1_dish216 = Fsh1_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((216 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 8) * 32 + (((216 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 64) * 257) + 0x01]
                Freg1_dish240 = Fsh1_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((240 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 8) * 32 + (((240 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 64) * 257) + 0x01]
                Freg1_dish264 = Fsh1_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((264 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 8) * 32 + (((264 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 64) * 257) + 0x01]
                Freg1_dish288 = Fsh1_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((288 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 8) * 32 + (((288 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 64) * 257) + 0x01]
                Freg1_dish312 = Fsh1_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((312 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 8) * 32 + (((312 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 64) * 257) + 0x01]
                Freg1_dish336 = Fsh1_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((336 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 8) * 32 + (((336 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 64) * 257) + 0x01]
                Freg1_dish360 = Fsh1_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((360 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 8) * 32 + (((360 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 64) * 257) + 0x01]
                Freg1_dish384 = Fsh1_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((384 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 8) * 32 + (((384 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 64) * 257) + 0x01]
                Freg1_dish408 = Fsh1_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((408 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 8) * 32 + (((408 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 64) * 257) + 0x01]
                Freg1_dish432 = Fsh1_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((432 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 8) * 32 + (((432 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 64) * 257) + 0x01]
                Freg1_dish456 = Fsh1_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((456 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 8) * 32 + (((456 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 64) * 257) + 0x01]
                Freg1_dish480 = Fsh1_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((480 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 8) * 32 + (((480 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 64) * 257) + 0x01]
                Freg1_dish504 = Fsh1_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((504 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 8) * 32 + (((504 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 64) * 257) + 0x01]
                IndexSpaces.cuda_sync_threads()
                sd_sd0 = IndexSpaces.cuda_shfl_sync(0xffffffff, S, 0)
                sd_sd1 = IndexSpaces.cuda_shfl_sync(0xffffffff, S, 1)
                sd_sd2 = IndexSpaces.cuda_shfl_sync(0xffffffff, S, 2)
                sd_sd3 = IndexSpaces.cuda_shfl_sync(0xffffffff, S, 3)
                sd_sd4 = IndexSpaces.cuda_shfl_sync(0xffffffff, S, 4)
                sd_sd5 = IndexSpaces.cuda_shfl_sync(0xffffffff, S, 5)
                sd_sd6 = IndexSpaces.cuda_shfl_sync(0xffffffff, S, 6)
                sd_sd7 = IndexSpaces.cuda_shfl_sync(0xffffffff, S, 7)
                sd_sd8 = IndexSpaces.cuda_shfl_sync(0xffffffff, S, 8)
                sd_sd9 = IndexSpaces.cuda_shfl_sync(0xffffffff, S, 9)
                sd_sd10 = IndexSpaces.cuda_shfl_sync(0xffffffff, S, 10)
                sd_sd11 = IndexSpaces.cuda_shfl_sync(0xffffffff, S, 11)
                sd_sd12 = IndexSpaces.cuda_shfl_sync(0xffffffff, S, 12)
                sd_sd13 = IndexSpaces.cuda_shfl_sync(0xffffffff, S, 13)
                sd_sd14 = IndexSpaces.cuda_shfl_sync(0xffffffff, S, 14)
                sd_sd15 = IndexSpaces.cuda_shfl_sync(0xffffffff, S, 15)
                sd_sd16 = IndexSpaces.cuda_shfl_sync(0xffffffff, S, 16)
                sd_sd17 = IndexSpaces.cuda_shfl_sync(0xffffffff, S, 17)
                sd_sd18 = IndexSpaces.cuda_shfl_sync(0xffffffff, S, 18)
                sd_sd19 = IndexSpaces.cuda_shfl_sync(0xffffffff, S, 19)
                sd_sd20 = IndexSpaces.cuda_shfl_sync(0xffffffff, S, 20)
                sd_sd21 = IndexSpaces.cuda_shfl_sync(0xffffffff, S, 21)
                sd_sd22 = IndexSpaces.cuda_shfl_sync(0xffffffff, S, 22)
                sd_sd23 = IndexSpaces.cuda_shfl_sync(0xffffffff, S, 23)
                Freg1′ = Freg1_dish0
                if sd_sd0 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + sd_sd0) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish24
                if sd_sd1 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + sd_sd1) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish48
                if sd_sd2 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + sd_sd2) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish72
                if sd_sd3 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + sd_sd3) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish96
                if sd_sd4 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + sd_sd4) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish120
                if sd_sd5 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + sd_sd5) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish144
                if sd_sd6 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + sd_sd6) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish168
                if sd_sd7 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + sd_sd7) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish192
                if sd_sd8 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + sd_sd8) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish216
                if sd_sd9 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + sd_sd9) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish240
                if sd_sd10 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + sd_sd10) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish264
                if sd_sd11 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + sd_sd11) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish288
                if sd_sd12 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + sd_sd12) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish312
                if sd_sd13 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + sd_sd13) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish336
                if sd_sd14 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + sd_sd14) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish360
                if sd_sd15 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + sd_sd15) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish384
                if sd_sd16 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + sd_sd16) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish408
                if sd_sd17 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + sd_sd17) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish432
                if sd_sd18 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + sd_sd18) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish456
                if sd_sd19 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + sd_sd19) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish480
                if sd_sd20 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + sd_sd20) + 0x01] =
                        Freg1′
                end
                Freg1′ = if warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24), dish = warp + 24 * 21, dish < 512
                    Freg1_dish504
                else
                    Int4x8(0, 0, 0, 0, 0, 0, 0, 0)
                end
                if sd_sd21 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + sd_sd21) + 0x01] =
                        Freg1′
                end
                Freg1′ = Int4x8(0, 0, 0, 0, 0, 0, 0, 0)
                if sd_sd22 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + sd_sd22) + 0x01] =
                        Freg1′
                end
                Freg1′ = Int4x8(0, 0, 0, 0, 0, 0, 0, 0)
                if sd_sd23 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + sd_sd23) + 0x01] =
                        Freg1′
                end
                IndexSpaces.cuda_sync_threads()
            end
            let
                Freg2_time0 = zero(Int4x8)
                Freg2_time1 = zero(Int4x8)
                Freg2_time2 = zero(Int4x8)
                Freg2_time3 = zero(Int4x8)
                Freg2_time4 = zero(Int4x8)
                Freg2_time5 = zero(Int4x8)
                Freg2_time6 = zero(Int4x8)
                Freg2_time7 = zero(Int4x8)
                Freg2_time8 = zero(Int4x8)
                Freg2_time9 = zero(Int4x8)
                Freg2_time10 = zero(Int4x8)
                Freg2_time11 = zero(Int4x8)
                Freg2_time12 = zero(Int4x8)
                Freg2_time13 = zero(Int4x8)
                Freg2_time14 = zero(Int4x8)
                Freg2_time15 = zero(Int4x8)
                Freg2_time16 = zero(Int4x8)
                Freg2_time17 = zero(Int4x8)
                Freg2_time18 = zero(Int4x8)
                Freg2_time19 = zero(Int4x8)
                Freg2_time20 = zero(Int4x8)
                Freg2_time21 = zero(Int4x8)
                Freg2_time22 = zero(Int4x8)
                Freg2_time23 = zero(Int4x8)
                if let
                    thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                    nlo = (1i32) * ((thread ÷ (4i32)) % (2i32)) + (2i32) * ((thread ÷ 8) % 4)
                    nlo < 6
                end
                    Freg2_time0 = Fsh2_shared[((((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 33 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806) + 0x01]
                    Freg2_time1 = Fsh2_shared[((1 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 33 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806) + 0x01]
                    Freg2_time2 = Fsh2_shared[((2 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 33 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806) + 0x01]
                    Freg2_time3 = Fsh2_shared[((3 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 33 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806) + 0x01]
                    Freg2_time4 = Fsh2_shared[((4 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 33 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806) + 0x01]
                    Freg2_time5 = Fsh2_shared[((5 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 33 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806) + 0x01]
                    Freg2_time6 = Fsh2_shared[((6 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 33 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806) + 0x01]
                    Freg2_time7 = Fsh2_shared[((7 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 33 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806) + 0x01]
                    Freg2_time8 = Fsh2_shared[((8 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 33 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806) + 0x01]
                    Freg2_time9 = Fsh2_shared[((9 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 33 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806) + 0x01]
                    Freg2_time10 = Fsh2_shared[((10 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 33 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806) + 0x01]
                    Freg2_time11 = Fsh2_shared[((11 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 33 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806) + 0x01]
                    Freg2_time12 = Fsh2_shared[((12 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 33 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806) + 0x01]
                    Freg2_time13 = Fsh2_shared[((13 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 33 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806) + 0x01]
                    Freg2_time14 = Fsh2_shared[((14 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 33 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806) + 0x01]
                    Freg2_time15 = Fsh2_shared[((15 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 33 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806) + 0x01]
                    Freg2_time16 = Fsh2_shared[((16 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 33 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806) + 0x01]
                    Freg2_time17 = Fsh2_shared[((17 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 33 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806) + 0x01]
                    Freg2_time18 = Fsh2_shared[((18 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 33 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806) + 0x01]
                    Freg2_time19 = Fsh2_shared[((19 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 33 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806) + 0x01]
                    Freg2_time20 = Fsh2_shared[((20 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 33 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806) + 0x01]
                    Freg2_time21 = Fsh2_shared[((21 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 33 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806) + 0x01]
                    Freg2_time22 = Fsh2_shared[((22 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 33 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806) + 0x01]
                    Freg2_time23 = Fsh2_shared[((23 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 33 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806) + 0x01]
                end
                IndexSpaces.cuda_sync_threads()
                let
                    t_inner_hi = 0
                    for t_inner_lo in 0:4:23
                        Freg2′_time0 = zero(Int4x8)
                        if t_inner_lo == 0
                            Freg2′_time0 = Freg2_time0
                        end
                        if t_inner_lo == 4
                            Freg2′_time0 = Freg2_time4
                        end
                        if t_inner_lo == 8
                            Freg2′_time0 = Freg2_time8
                        end
                        if t_inner_lo == 12
                            Freg2′_time0 = Freg2_time12
                        end
                        if t_inner_lo == 16
                            Freg2′_time0 = Freg2_time16
                        end
                        if t_inner_lo == 20
                            Freg2′_time0 = Freg2_time20
                        end
                        Freg2′_time1 = zero(Int4x8)
                        if t_inner_lo == 0
                            Freg2′_time1 = Freg2_time1
                        end
                        if t_inner_lo == 4
                            Freg2′_time1 = Freg2_time5
                        end
                        if t_inner_lo == 8
                            Freg2′_time1 = Freg2_time9
                        end
                        if t_inner_lo == 12
                            Freg2′_time1 = Freg2_time13
                        end
                        if t_inner_lo == 16
                            Freg2′_time1 = Freg2_time17
                        end
                        if t_inner_lo == 20
                            Freg2′_time1 = Freg2_time21
                        end
                        Freg2′_time2 = zero(Int4x8)
                        if t_inner_lo == 0
                            Freg2′_time2 = Freg2_time2
                        end
                        if t_inner_lo == 4
                            Freg2′_time2 = Freg2_time6
                        end
                        if t_inner_lo == 8
                            Freg2′_time2 = Freg2_time10
                        end
                        if t_inner_lo == 12
                            Freg2′_time2 = Freg2_time14
                        end
                        if t_inner_lo == 16
                            Freg2′_time2 = Freg2_time18
                        end
                        if t_inner_lo == 20
                            Freg2′_time2 = Freg2_time22
                        end
                        Freg2′_time3 = zero(Int4x8)
                        if t_inner_lo == 0
                            Freg2′_time3 = Freg2_time3
                        end
                        if t_inner_lo == 4
                            Freg2′_time3 = Freg2_time7
                        end
                        if t_inner_lo == 8
                            Freg2′_time3 = Freg2_time11
                        end
                        if t_inner_lo == 12
                            Freg2′_time3 = Freg2_time15
                        end
                        if t_inner_lo == 16
                            Freg2′_time3 = Freg2_time19
                        end
                        if t_inner_lo == 20
                            Freg2′_time3 = Freg2_time23
                        end
                        (E′_polr0_time0, E′_polr1_time0, E′_polr0_time24, E′_polr1_time24) = convert(
                            NTuple{4,Float16x2}, Freg2′_time0
                        )
                        (E′_polr0_time1, E′_polr1_time1, E′_polr0_time25, E′_polr1_time25) = convert(
                            NTuple{4,Float16x2}, Freg2′_time1
                        )
                        (E′_polr0_time2, E′_polr1_time2, E′_polr0_time26, E′_polr1_time26) = convert(
                            NTuple{4,Float16x2}, Freg2′_time2
                        )
                        (E′_polr0_time3, E′_polr1_time3, E′_polr0_time27, E′_polr1_time27) = convert(
                            NTuple{4,Float16x2}, Freg2′_time3
                        )
                        E_polr0_time0 = E′_polr0_time0
                        E_polr1_time0 = E′_polr1_time0
                        E_polr0_time1 = E′_polr0_time1
                        E_polr1_time1 = E′_polr1_time1
                        E_polr0_time2 = E′_polr0_time2
                        E_polr1_time2 = E′_polr1_time2
                        E_polr0_time3 = E′_polr0_time3
                        E_polr1_time3 = E′_polr1_time3
                        WE_polr0_time0 = complex_mul(W_polr0, E_polr0_time0)
                        WE_polr1_time0 = complex_mul(W_polr1, E_polr1_time0)
                        WE_polr0_time1 = complex_mul(W_polr0, E_polr0_time1)
                        WE_polr1_time1 = complex_mul(W_polr1, E_polr1_time1)
                        WE_polr0_time2 = complex_mul(W_polr0, E_polr0_time2)
                        WE_polr1_time2 = complex_mul(W_polr1, E_polr1_time2)
                        WE_polr0_time3 = complex_mul(W_polr0, E_polr0_time3)
                        WE_polr1_time3 = complex_mul(W_polr1, E_polr1_time3)
                        X_polr0_time0 = WE_polr0_time0
                        X_polr1_time0 = WE_polr1_time0
                        X_polr0_time1 = WE_polr0_time1
                        X_polr1_time1 = WE_polr1_time1
                        X_polr0_time2 = WE_polr0_time2
                        X_polr1_time2 = WE_polr1_time2
                        X_polr0_time3 = WE_polr0_time3
                        X_polr1_time3 = WE_polr1_time3
                        Z_cplx0_polr0_time0 = zero(Float16x2)
                        Z_cplx1_polr0_time0 = zero(Float16x2)
                        Z_cplx0_polr1_time0 = zero(Float16x2)
                        Z_cplx1_polr1_time0 = zero(Float16x2)
                        Z_cplx0_polr0_time1 = zero(Float16x2)
                        Z_cplx1_polr0_time1 = zero(Float16x2)
                        Z_cplx0_polr1_time1 = zero(Float16x2)
                        Z_cplx1_polr1_time1 = zero(Float16x2)
                        Z_cplx0_polr0_time2 = zero(Float16x2)
                        Z_cplx1_polr0_time2 = zero(Float16x2)
                        Z_cplx0_polr1_time2 = zero(Float16x2)
                        Z_cplx1_polr1_time2 = zero(Float16x2)
                        Z_cplx0_polr0_time3 = zero(Float16x2)
                        Z_cplx1_polr0_time3 = zero(Float16x2)
                        Z_cplx0_polr1_time3 = zero(Float16x2)
                        Z_cplx1_polr1_time3 = zero(Float16x2)
                        (Z_cplx0_polr0_time0, Z_cplx1_polr0_time0) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr0_time0, (Z_cplx0_polr0_time0, Z_cplx1_polr0_time0)
                        )
                        (Z_cplx0_polr1_time0, Z_cplx1_polr1_time0) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr1_time0, (Z_cplx0_polr1_time0, Z_cplx1_polr1_time0)
                        )
                        (Z_cplx0_polr0_time1, Z_cplx1_polr0_time1) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr0_time1, (Z_cplx0_polr0_time1, Z_cplx1_polr0_time1)
                        )
                        (Z_cplx0_polr1_time1, Z_cplx1_polr1_time1) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr1_time1, (Z_cplx0_polr1_time1, Z_cplx1_polr1_time1)
                        )
                        (Z_cplx0_polr0_time2, Z_cplx1_polr0_time2) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr0_time2, (Z_cplx0_polr0_time2, Z_cplx1_polr0_time2)
                        )
                        (Z_cplx0_polr1_time2, Z_cplx1_polr1_time2) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr1_time2, (Z_cplx0_polr1_time2, Z_cplx1_polr1_time2)
                        )
                        (Z_cplx0_polr0_time3, Z_cplx1_polr0_time3) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr0_time3, (Z_cplx0_polr0_time3, Z_cplx1_polr0_time3)
                        )
                        (Z_cplx0_polr1_time3, Z_cplx1_polr1_time3) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr1_time3, (Z_cplx0_polr1_time3, Z_cplx1_polr1_time3)
                        )
                        aΓ²re = aΓ²_cplx0
                        aΓ²im = aΓ²_cplx1
                        Zre_polr0_time0 = Z_cplx0_polr0_time0
                        Zim_polr0_time0 = Z_cplx1_polr0_time0
                        Zre_polr1_time0 = Z_cplx0_polr1_time0
                        Zim_polr1_time0 = Z_cplx1_polr1_time0
                        Zre_polr0_time1 = Z_cplx0_polr0_time1
                        Zim_polr0_time1 = Z_cplx1_polr0_time1
                        Zre_polr1_time1 = Z_cplx0_polr1_time1
                        Zim_polr1_time1 = Z_cplx1_polr1_time1
                        Zre_polr0_time2 = Z_cplx0_polr0_time2
                        Zim_polr0_time2 = Z_cplx1_polr0_time2
                        Zre_polr1_time2 = Z_cplx0_polr1_time2
                        Zim_polr1_time2 = Z_cplx1_polr1_time2
                        Zre_polr0_time3 = Z_cplx0_polr0_time3
                        Zim_polr0_time3 = Z_cplx1_polr0_time3
                        Zre_polr1_time3 = Z_cplx0_polr1_time3
                        Zim_polr1_time3 = Z_cplx1_polr1_time3
                        Vre_polr0_time0 = muladd(aΓ²re, Zre_polr0_time0, -aΓ²im * Zim_polr0_time0)
                        Vre_polr1_time0 = muladd(aΓ²re, Zre_polr1_time0, -aΓ²im * Zim_polr1_time0)
                        Vre_polr0_time1 = muladd(aΓ²re, Zre_polr0_time1, -aΓ²im * Zim_polr0_time1)
                        Vre_polr1_time1 = muladd(aΓ²re, Zre_polr1_time1, -aΓ²im * Zim_polr1_time1)
                        Vre_polr0_time2 = muladd(aΓ²re, Zre_polr0_time2, -aΓ²im * Zim_polr0_time2)
                        Vre_polr1_time2 = muladd(aΓ²re, Zre_polr1_time2, -aΓ²im * Zim_polr1_time2)
                        Vre_polr0_time3 = muladd(aΓ²re, Zre_polr0_time3, -aΓ²im * Zim_polr0_time3)
                        Vre_polr1_time3 = muladd(aΓ²re, Zre_polr1_time3, -aΓ²im * Zim_polr1_time3)
                        Vim_polr0_time0 = muladd(aΓ²re, Zim_polr0_time0, +aΓ²im * Zre_polr0_time0)
                        Vim_polr1_time0 = muladd(aΓ²re, Zim_polr1_time0, +aΓ²im * Zre_polr1_time0)
                        Vim_polr0_time1 = muladd(aΓ²re, Zim_polr0_time1, +aΓ²im * Zre_polr0_time1)
                        Vim_polr1_time1 = muladd(aΓ²re, Zim_polr1_time1, +aΓ²im * Zre_polr1_time1)
                        Vim_polr0_time2 = muladd(aΓ²re, Zim_polr0_time2, +aΓ²im * Zre_polr0_time2)
                        Vim_polr1_time2 = muladd(aΓ²re, Zim_polr1_time2, +aΓ²im * Zre_polr1_time2)
                        Vim_polr0_time3 = muladd(aΓ²re, Zim_polr0_time3, +aΓ²im * Zre_polr0_time3)
                        Vim_polr1_time3 = muladd(aΓ²re, Zim_polr1_time3, +aΓ²im * Zre_polr1_time3)
                        V_cplx0_polr0_time0 = Vre_polr0_time0
                        V_cplx1_polr0_time0 = Vim_polr0_time0
                        V_cplx0_polr1_time0 = Vre_polr1_time0
                        V_cplx1_polr1_time0 = Vim_polr1_time0
                        V_cplx0_polr0_time1 = Vre_polr0_time1
                        V_cplx1_polr0_time1 = Vim_polr0_time1
                        V_cplx0_polr1_time1 = Vre_polr1_time1
                        V_cplx1_polr1_time1 = Vim_polr1_time1
                        V_cplx0_polr0_time2 = Vre_polr0_time2
                        V_cplx1_polr0_time2 = Vim_polr0_time2
                        V_cplx0_polr1_time2 = Vre_polr1_time2
                        V_cplx1_polr1_time2 = Vim_polr1_time2
                        V_cplx0_polr0_time3 = Vre_polr0_time3
                        V_cplx1_polr0_time3 = Vim_polr0_time3
                        V_cplx0_polr1_time3 = Vre_polr1_time3
                        V_cplx1_polr1_time3 = Vim_polr1_time3
                        Y_cplx0_polr0_time0 = zero(Float16x2)
                        Y_cplx1_polr0_time0 = zero(Float16x2)
                        Y_cplx0_polr1_time0 = zero(Float16x2)
                        Y_cplx1_polr1_time0 = zero(Float16x2)
                        Y_cplx0_polr0_time1 = zero(Float16x2)
                        Y_cplx1_polr0_time1 = zero(Float16x2)
                        Y_cplx0_polr1_time1 = zero(Float16x2)
                        Y_cplx1_polr1_time1 = zero(Float16x2)
                        Y_cplx0_polr0_time2 = zero(Float16x2)
                        Y_cplx1_polr0_time2 = zero(Float16x2)
                        Y_cplx0_polr1_time2 = zero(Float16x2)
                        Y_cplx1_polr1_time2 = zero(Float16x2)
                        Y_cplx0_polr0_time3 = zero(Float16x2)
                        Y_cplx1_polr0_time3 = zero(Float16x2)
                        Y_cplx0_polr1_time3 = zero(Float16x2)
                        Y_cplx1_polr1_time3 = zero(Float16x2)
                        Vre_polr0_time0 = V_cplx0_polr0_time0
                        Vim_polr0_time0 = V_cplx1_polr0_time0
                        Vre_polr1_time0 = V_cplx0_polr1_time0
                        Vim_polr1_time0 = V_cplx1_polr1_time0
                        Vre_polr0_time1 = V_cplx0_polr0_time1
                        Vim_polr0_time1 = V_cplx1_polr0_time1
                        Vre_polr1_time1 = V_cplx0_polr1_time1
                        Vim_polr1_time1 = V_cplx1_polr1_time1
                        Vre_polr0_time2 = V_cplx0_polr0_time2
                        Vim_polr0_time2 = V_cplx1_polr0_time2
                        Vre_polr1_time2 = V_cplx0_polr1_time2
                        Vim_polr1_time2 = V_cplx1_polr1_time2
                        Vre_polr0_time3 = V_cplx0_polr0_time3
                        Vim_polr0_time3 = V_cplx1_polr0_time3
                        Vre_polr1_time3 = V_cplx0_polr1_time3
                        Vim_polr1_time3 = V_cplx1_polr1_time3
                        V_cplx_in0_polr0_time0 = Vre_polr0_time0
                        V_cplx_in1_polr0_time0 = Vim_polr0_time0
                        V_cplx_in0_polr1_time0 = Vre_polr1_time0
                        V_cplx_in1_polr1_time0 = Vim_polr1_time0
                        V_cplx_in0_polr0_time1 = Vre_polr0_time1
                        V_cplx_in1_polr0_time1 = Vim_polr0_time1
                        V_cplx_in0_polr1_time1 = Vre_polr1_time1
                        V_cplx_in1_polr1_time1 = Vim_polr1_time1
                        V_cplx_in0_polr0_time2 = Vre_polr0_time2
                        V_cplx_in1_polr0_time2 = Vim_polr0_time2
                        V_cplx_in0_polr1_time2 = Vre_polr1_time2
                        V_cplx_in1_polr1_time2 = Vim_polr1_time2
                        V_cplx_in0_polr0_time3 = Vre_polr0_time3
                        V_cplx_in1_polr0_time3 = Vim_polr0_time3
                        V_cplx_in0_polr1_time3 = Vre_polr1_time3
                        V_cplx_in1_polr1_time3 = Vim_polr1_time3
                        (Y_cplx0_polr0_time0, Y_cplx1_polr0_time0) = IndexSpaces.mma_m16n8k16(
                            (aΓ³_cplx0_cplx_in0, aΓ³_cplx1_cplx_in0, aΓ³_cplx0_cplx_in1, aΓ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr0_time0, V_cplx_in1_polr0_time0),
                            (Y_cplx0_polr0_time0, Y_cplx1_polr0_time0),
                        )
                        (Y_cplx0_polr1_time0, Y_cplx1_polr1_time0) = IndexSpaces.mma_m16n8k16(
                            (aΓ³_cplx0_cplx_in0, aΓ³_cplx1_cplx_in0, aΓ³_cplx0_cplx_in1, aΓ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr1_time0, V_cplx_in1_polr1_time0),
                            (Y_cplx0_polr1_time0, Y_cplx1_polr1_time0),
                        )
                        (Y_cplx0_polr0_time1, Y_cplx1_polr0_time1) = IndexSpaces.mma_m16n8k16(
                            (aΓ³_cplx0_cplx_in0, aΓ³_cplx1_cplx_in0, aΓ³_cplx0_cplx_in1, aΓ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr0_time1, V_cplx_in1_polr0_time1),
                            (Y_cplx0_polr0_time1, Y_cplx1_polr0_time1),
                        )
                        (Y_cplx0_polr1_time1, Y_cplx1_polr1_time1) = IndexSpaces.mma_m16n8k16(
                            (aΓ³_cplx0_cplx_in0, aΓ³_cplx1_cplx_in0, aΓ³_cplx0_cplx_in1, aΓ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr1_time1, V_cplx_in1_polr1_time1),
                            (Y_cplx0_polr1_time1, Y_cplx1_polr1_time1),
                        )
                        (Y_cplx0_polr0_time2, Y_cplx1_polr0_time2) = IndexSpaces.mma_m16n8k16(
                            (aΓ³_cplx0_cplx_in0, aΓ³_cplx1_cplx_in0, aΓ³_cplx0_cplx_in1, aΓ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr0_time2, V_cplx_in1_polr0_time2),
                            (Y_cplx0_polr0_time2, Y_cplx1_polr0_time2),
                        )
                        (Y_cplx0_polr1_time2, Y_cplx1_polr1_time2) = IndexSpaces.mma_m16n8k16(
                            (aΓ³_cplx0_cplx_in0, aΓ³_cplx1_cplx_in0, aΓ³_cplx0_cplx_in1, aΓ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr1_time2, V_cplx_in1_polr1_time2),
                            (Y_cplx0_polr1_time2, Y_cplx1_polr1_time2),
                        )
                        (Y_cplx0_polr0_time3, Y_cplx1_polr0_time3) = IndexSpaces.mma_m16n8k16(
                            (aΓ³_cplx0_cplx_in0, aΓ³_cplx1_cplx_in0, aΓ³_cplx0_cplx_in1, aΓ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr0_time3, V_cplx_in1_polr0_time3),
                            (Y_cplx0_polr0_time3, Y_cplx1_polr0_time3),
                        )
                        (Y_cplx0_polr1_time3, Y_cplx1_polr1_time3) = IndexSpaces.mma_m16n8k16(
                            (aΓ³_cplx0_cplx_in0, aΓ³_cplx1_cplx_in0, aΓ³_cplx0_cplx_in1, aΓ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr1_time3, V_cplx_in1_polr1_time3),
                            (Y_cplx0_polr1_time3, Y_cplx1_polr1_time3),
                        )
                        G_cplx0_polr0_time0 = Y_cplx0_polr0_time0
                        G_cplx1_polr0_time0 = Y_cplx1_polr0_time0
                        G_cplx0_polr1_time0 = Y_cplx0_polr1_time0
                        G_cplx1_polr1_time0 = Y_cplx1_polr1_time0
                        G_cplx0_polr0_time1 = Y_cplx0_polr0_time1
                        G_cplx1_polr0_time1 = Y_cplx1_polr0_time1
                        G_cplx0_polr1_time1 = Y_cplx0_polr1_time1
                        G_cplx1_polr1_time1 = Y_cplx1_polr1_time1
                        G_cplx0_polr0_time2 = Y_cplx0_polr0_time2
                        G_cplx1_polr0_time2 = Y_cplx1_polr0_time2
                        G_cplx0_polr1_time2 = Y_cplx0_polr1_time2
                        G_cplx1_polr1_time2 = Y_cplx1_polr1_time2
                        G_cplx0_polr0_time3 = Y_cplx0_polr0_time3
                        G_cplx1_polr0_time3 = Y_cplx1_polr0_time3
                        G_cplx0_polr1_time3 = Y_cplx0_polr1_time3
                        G_cplx1_polr1_time3 = Y_cplx1_polr1_time3
                        (G_cplx0_polr0_time0, G_cplx1_polr0_time0) = (
                            IndexSpaces.get_lo16(G_cplx0_polr0_time0, G_cplx1_polr0_time0),
                            IndexSpaces.get_hi16(G_cplx0_polr0_time0, G_cplx1_polr0_time0),
                        )
                        (G_cplx0_polr1_time0, G_cplx1_polr1_time0) = (
                            IndexSpaces.get_lo16(G_cplx0_polr1_time0, G_cplx1_polr1_time0),
                            IndexSpaces.get_hi16(G_cplx0_polr1_time0, G_cplx1_polr1_time0),
                        )
                        (G_cplx0_polr0_time1, G_cplx1_polr0_time1) = (
                            IndexSpaces.get_lo16(G_cplx0_polr0_time1, G_cplx1_polr0_time1),
                            IndexSpaces.get_hi16(G_cplx0_polr0_time1, G_cplx1_polr0_time1),
                        )
                        (G_cplx0_polr1_time1, G_cplx1_polr1_time1) = (
                            IndexSpaces.get_lo16(G_cplx0_polr1_time1, G_cplx1_polr1_time1),
                            IndexSpaces.get_hi16(G_cplx0_polr1_time1, G_cplx1_polr1_time1),
                        )
                        (G_cplx0_polr0_time2, G_cplx1_polr0_time2) = (
                            IndexSpaces.get_lo16(G_cplx0_polr0_time2, G_cplx1_polr0_time2),
                            IndexSpaces.get_hi16(G_cplx0_polr0_time2, G_cplx1_polr0_time2),
                        )
                        (G_cplx0_polr1_time2, G_cplx1_polr1_time2) = (
                            IndexSpaces.get_lo16(G_cplx0_polr1_time2, G_cplx1_polr1_time2),
                            IndexSpaces.get_hi16(G_cplx0_polr1_time2, G_cplx1_polr1_time2),
                        )
                        (G_cplx0_polr0_time3, G_cplx1_polr0_time3) = (
                            IndexSpaces.get_lo16(G_cplx0_polr0_time3, G_cplx1_polr0_time3),
                            IndexSpaces.get_hi16(G_cplx0_polr0_time3, G_cplx1_polr0_time3),
                        )
                        (G_cplx0_polr1_time3, G_cplx1_polr1_time3) = (
                            IndexSpaces.get_lo16(G_cplx0_polr1_time3, G_cplx1_polr1_time3),
                            IndexSpaces.get_hi16(G_cplx0_polr1_time3, G_cplx1_polr1_time3),
                        )
                        IndexSpaces.cuda_sync_threads()
                        let
                            t = 0
                            G_beamQ0_polr0 = zero(Float16x2)
                            G_beamQ24_polr0 = zero(Float16x2)
                            G_beamQ0_polr1 = zero(Float16x2)
                            G_beamQ24_polr1 = zero(Float16x2)
                            if let
                                thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                mlo2_offset = 8
                                mlo2_length = 4
                                mlo = (1i32) * ((thread ÷ (4i32)) % (2i32)) + (2i32) * ((thread ÷ mlo2_offset) % mlo2_length)
                                mlo < 6
                            end
                                G_beamQ0_polr0 = Gsh_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 0 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ24_polr0 = Gsh_shared[((((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 0 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ0_polr1 = Gsh_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ24_polr1 = Gsh_shared[((((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 32 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                            end
                            X_beamQ0_polr0 = G_beamQ0_polr0
                            X_beamQ24_polr0 = G_beamQ24_polr0
                            X_beamQ0_polr1 = G_beamQ0_polr1
                            X_beamQ24_polr1 = G_beamQ24_polr1
                            Z_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ24_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ24_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ24_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Z_beamQ24_cplx1_polr1 = zero(Float16x2)
                            (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ0_polr0, (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0)
                            )
                            (Z_beamQ24_cplx0_polr0, Z_beamQ24_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ24_polr0, (Z_beamQ24_cplx0_polr0, Z_beamQ24_cplx1_polr0)
                            )
                            (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ0_polr1, (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1)
                            )
                            (Z_beamQ24_cplx0_polr1, Z_beamQ24_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ24_polr1, (Z_beamQ24_cplx0_polr1, Z_beamQ24_cplx1_polr1)
                            )
                            bΓ²re = bΓ²_cplx0
                            bΓ²im = bΓ²_cplx1
                            Zre_beamQ0_polr0 = Z_beamQ0_cplx0_polr0
                            Zim_beamQ0_polr0 = Z_beamQ0_cplx1_polr0
                            Zre_beamQ24_polr0 = Z_beamQ24_cplx0_polr0
                            Zim_beamQ24_polr0 = Z_beamQ24_cplx1_polr0
                            Zre_beamQ0_polr1 = Z_beamQ0_cplx0_polr1
                            Zim_beamQ0_polr1 = Z_beamQ0_cplx1_polr1
                            Zre_beamQ24_polr1 = Z_beamQ24_cplx0_polr1
                            Zim_beamQ24_polr1 = Z_beamQ24_cplx1_polr1
                            Vre_beamQ0_polr0 = muladd(bΓ²re, Zre_beamQ0_polr0, -bΓ²im * Zim_beamQ0_polr0)
                            Vre_beamQ24_polr0 = muladd(bΓ²re, Zre_beamQ24_polr0, -bΓ²im * Zim_beamQ24_polr0)
                            Vre_beamQ0_polr1 = muladd(bΓ²re, Zre_beamQ0_polr1, -bΓ²im * Zim_beamQ0_polr1)
                            Vre_beamQ24_polr1 = muladd(bΓ²re, Zre_beamQ24_polr1, -bΓ²im * Zim_beamQ24_polr1)
                            Vim_beamQ0_polr0 = muladd(bΓ²re, Zim_beamQ0_polr0, +bΓ²im * Zre_beamQ0_polr0)
                            Vim_beamQ24_polr0 = muladd(bΓ²re, Zim_beamQ24_polr0, +bΓ²im * Zre_beamQ24_polr0)
                            Vim_beamQ0_polr1 = muladd(bΓ²re, Zim_beamQ0_polr1, +bΓ²im * Zre_beamQ0_polr1)
                            Vim_beamQ24_polr1 = muladd(bΓ²re, Zim_beamQ24_polr1, +bΓ²im * Zre_beamQ24_polr1)
                            V_beamQ0_cplx0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx1_polr0 = Vim_beamQ0_polr0
                            V_beamQ24_cplx0_polr0 = Vre_beamQ24_polr0
                            V_beamQ24_cplx1_polr0 = Vim_beamQ24_polr0
                            V_beamQ0_cplx0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx1_polr1 = Vim_beamQ0_polr1
                            V_beamQ24_cplx0_polr1 = Vre_beamQ24_polr1
                            V_beamQ24_cplx1_polr1 = Vim_beamQ24_polr1
                            Y_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ24_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ24_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ24_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Y_beamQ24_cplx1_polr1 = zero(Float16x2)
                            Vre_beamQ0_polr0 = V_beamQ0_cplx0_polr0
                            Vim_beamQ0_polr0 = V_beamQ0_cplx1_polr0
                            Vre_beamQ24_polr0 = V_beamQ24_cplx0_polr0
                            Vim_beamQ24_polr0 = V_beamQ24_cplx1_polr0
                            Vre_beamQ0_polr1 = V_beamQ0_cplx0_polr1
                            Vim_beamQ0_polr1 = V_beamQ0_cplx1_polr1
                            Vre_beamQ24_polr1 = V_beamQ24_cplx0_polr1
                            Vim_beamQ24_polr1 = V_beamQ24_cplx1_polr1
                            V_beamQ0_cplx_in0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx_in1_polr0 = Vim_beamQ0_polr0
                            V_beamQ24_cplx_in0_polr0 = Vre_beamQ24_polr0
                            V_beamQ24_cplx_in1_polr0 = Vim_beamQ24_polr0
                            V_beamQ0_cplx_in0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx_in1_polr1 = Vim_beamQ0_polr1
                            V_beamQ24_cplx_in0_polr1 = Vre_beamQ24_polr1
                            V_beamQ24_cplx_in1_polr1 = Vim_beamQ24_polr1
                            (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr0, V_beamQ0_cplx_in1_polr0),
                                (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0),
                            )
                            (Y_beamQ24_cplx0_polr0, Y_beamQ24_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ24_cplx_in0_polr0, V_beamQ24_cplx_in1_polr0),
                                (Y_beamQ24_cplx0_polr0, Y_beamQ24_cplx1_polr0),
                            )
                            (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr1, V_beamQ0_cplx_in1_polr1),
                                (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1),
                            )
                            (Y_beamQ24_cplx0_polr1, Y_beamQ24_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ24_cplx_in0_polr1, V_beamQ24_cplx_in1_polr1),
                                (Y_beamQ24_cplx0_polr1, Y_beamQ24_cplx1_polr1),
                            )
                            Ẽ_beamQ0_cplx0_polr0 = Y_beamQ0_cplx0_polr0
                            Ẽ_beamQ24_cplx0_polr0 = Y_beamQ24_cplx0_polr0
                            Ẽ_beamQ0_cplx1_polr0 = Y_beamQ0_cplx1_polr0
                            Ẽ_beamQ24_cplx1_polr0 = Y_beamQ24_cplx1_polr0
                            Ẽ_beamQ0_cplx0_polr1 = Y_beamQ0_cplx0_polr1
                            Ẽ_beamQ24_cplx0_polr1 = Y_beamQ24_cplx0_polr1
                            Ẽ_beamQ0_cplx1_polr1 = Y_beamQ0_cplx1_polr1
                            Ẽ_beamQ24_cplx1_polr1 = Y_beamQ24_cplx1_polr1
                            Ẽp0_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr0
                            Ẽp1_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr1
                            Ẽp0_beamQ24_cplx0 = Ẽ_beamQ24_cplx0_polr0
                            Ẽp1_beamQ24_cplx0 = Ẽ_beamQ24_cplx0_polr1
                            Ẽp0_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr0
                            Ẽp1_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr1
                            Ẽp0_beamQ24_cplx1 = Ẽ_beamQ24_cplx1_polr0
                            Ẽp1_beamQ24_cplx1 = Ẽ_beamQ24_cplx1_polr1
                            Ẽp0re_beamQ0 = Ẽp0_beamQ0_cplx0
                            Ẽp0im_beamQ0 = Ẽp0_beamQ0_cplx1
                            Ẽp0re_beamQ24 = Ẽp0_beamQ24_cplx0
                            Ẽp0im_beamQ24 = Ẽp0_beamQ24_cplx1
                            Ẽp1re_beamQ0 = Ẽp1_beamQ0_cplx0
                            Ẽp1im_beamQ0 = Ẽp1_beamQ0_cplx1
                            Ẽp1re_beamQ24 = Ẽp1_beamQ24_cplx0
                            Ẽp1im_beamQ24 = Ẽp1_beamQ24_cplx1
                            I_beamQ0 = muladd(
                                Float16x2(0.003124237f0, 0.003124237f0),
                                muladd(
                                    Ẽp1im_beamQ0,
                                    Ẽp1im_beamQ0,
                                    muladd(
                                        Ẽp1re_beamQ0, Ẽp1re_beamQ0, muladd(Ẽp0im_beamQ0, Ẽp0im_beamQ0, Ẽp0re_beamQ0 * Ẽp0re_beamQ0)
                                    ),
                                ),
                                I_beamQ0,
                            )
                            I_beamQ24 = muladd(
                                Float16x2(0.003124237f0, 0.003124237f0),
                                muladd(
                                    Ẽp1im_beamQ24,
                                    Ẽp1im_beamQ24,
                                    muladd(
                                        Ẽp1re_beamQ24,
                                        Ẽp1re_beamQ24,
                                        muladd(Ẽp0im_beamQ24, Ẽp0im_beamQ24, Ẽp0re_beamQ24 * Ẽp0re_beamQ24),
                                    ),
                                ),
                                I_beamQ24,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 4 == 0i32
                                if t_running == 40
                                    if let
                                        thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                        warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24)
                                        p = (2i32) * thread
                                        q = (2i32) * warp
                                        0i32 ≤ p < 48 && 0i32 ≤ q < 48
                                    end
                                        if true
                                            I_memory[let
                                                offset = 7077888 * T̄min + 1152 * F̄min
                                                length = 452984832
                                                mod(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 1152 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 64) % 64) % 64) * 7077888 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 48) * 24) + 0) + offset, length)
                                            end + 0x01] = I_beamQ0
                                        end
                                        if true
                                            I_memory[let
                                                offset = 7077888 * T̄min + 1152 * F̄min
                                                length = 452984832
                                                mod(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 1152 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 64) % 64) % 64) * 7077888 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 24) % 48) * 24) + 0) + offset, length)
                                            end + 0x01] = I_beamQ24
                                        end
                                    end
                                    I_beamQ0 = zero(Float16x2)
                                    I_beamQ24 = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let
                            t = 1
                            G_beamQ0_polr0 = zero(Float16x2)
                            G_beamQ24_polr0 = zero(Float16x2)
                            G_beamQ0_polr1 = zero(Float16x2)
                            G_beamQ24_polr1 = zero(Float16x2)
                            if let
                                thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                mlo2_offset = 8
                                mlo2_length = 4
                                mlo = (1i32) * ((thread ÷ (4i32)) % (2i32)) + (2i32) * ((thread ÷ mlo2_offset) % mlo2_length)
                                mlo < 6
                            end
                                G_beamQ0_polr0 = Gsh_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 0 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ24_polr0 = Gsh_shared[((((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 0 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ0_polr1 = Gsh_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ24_polr1 = Gsh_shared[((((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 32 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                            end
                            X_beamQ0_polr0 = G_beamQ0_polr0
                            X_beamQ24_polr0 = G_beamQ24_polr0
                            X_beamQ0_polr1 = G_beamQ0_polr1
                            X_beamQ24_polr1 = G_beamQ24_polr1
                            Z_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ24_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ24_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ24_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Z_beamQ24_cplx1_polr1 = zero(Float16x2)
                            (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ0_polr0, (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0)
                            )
                            (Z_beamQ24_cplx0_polr0, Z_beamQ24_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ24_polr0, (Z_beamQ24_cplx0_polr0, Z_beamQ24_cplx1_polr0)
                            )
                            (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ0_polr1, (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1)
                            )
                            (Z_beamQ24_cplx0_polr1, Z_beamQ24_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ24_polr1, (Z_beamQ24_cplx0_polr1, Z_beamQ24_cplx1_polr1)
                            )
                            bΓ²re = bΓ²_cplx0
                            bΓ²im = bΓ²_cplx1
                            Zre_beamQ0_polr0 = Z_beamQ0_cplx0_polr0
                            Zim_beamQ0_polr0 = Z_beamQ0_cplx1_polr0
                            Zre_beamQ24_polr0 = Z_beamQ24_cplx0_polr0
                            Zim_beamQ24_polr0 = Z_beamQ24_cplx1_polr0
                            Zre_beamQ0_polr1 = Z_beamQ0_cplx0_polr1
                            Zim_beamQ0_polr1 = Z_beamQ0_cplx1_polr1
                            Zre_beamQ24_polr1 = Z_beamQ24_cplx0_polr1
                            Zim_beamQ24_polr1 = Z_beamQ24_cplx1_polr1
                            Vre_beamQ0_polr0 = muladd(bΓ²re, Zre_beamQ0_polr0, -bΓ²im * Zim_beamQ0_polr0)
                            Vre_beamQ24_polr0 = muladd(bΓ²re, Zre_beamQ24_polr0, -bΓ²im * Zim_beamQ24_polr0)
                            Vre_beamQ0_polr1 = muladd(bΓ²re, Zre_beamQ0_polr1, -bΓ²im * Zim_beamQ0_polr1)
                            Vre_beamQ24_polr1 = muladd(bΓ²re, Zre_beamQ24_polr1, -bΓ²im * Zim_beamQ24_polr1)
                            Vim_beamQ0_polr0 = muladd(bΓ²re, Zim_beamQ0_polr0, +bΓ²im * Zre_beamQ0_polr0)
                            Vim_beamQ24_polr0 = muladd(bΓ²re, Zim_beamQ24_polr0, +bΓ²im * Zre_beamQ24_polr0)
                            Vim_beamQ0_polr1 = muladd(bΓ²re, Zim_beamQ0_polr1, +bΓ²im * Zre_beamQ0_polr1)
                            Vim_beamQ24_polr1 = muladd(bΓ²re, Zim_beamQ24_polr1, +bΓ²im * Zre_beamQ24_polr1)
                            V_beamQ0_cplx0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx1_polr0 = Vim_beamQ0_polr0
                            V_beamQ24_cplx0_polr0 = Vre_beamQ24_polr0
                            V_beamQ24_cplx1_polr0 = Vim_beamQ24_polr0
                            V_beamQ0_cplx0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx1_polr1 = Vim_beamQ0_polr1
                            V_beamQ24_cplx0_polr1 = Vre_beamQ24_polr1
                            V_beamQ24_cplx1_polr1 = Vim_beamQ24_polr1
                            Y_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ24_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ24_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ24_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Y_beamQ24_cplx1_polr1 = zero(Float16x2)
                            Vre_beamQ0_polr0 = V_beamQ0_cplx0_polr0
                            Vim_beamQ0_polr0 = V_beamQ0_cplx1_polr0
                            Vre_beamQ24_polr0 = V_beamQ24_cplx0_polr0
                            Vim_beamQ24_polr0 = V_beamQ24_cplx1_polr0
                            Vre_beamQ0_polr1 = V_beamQ0_cplx0_polr1
                            Vim_beamQ0_polr1 = V_beamQ0_cplx1_polr1
                            Vre_beamQ24_polr1 = V_beamQ24_cplx0_polr1
                            Vim_beamQ24_polr1 = V_beamQ24_cplx1_polr1
                            V_beamQ0_cplx_in0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx_in1_polr0 = Vim_beamQ0_polr0
                            V_beamQ24_cplx_in0_polr0 = Vre_beamQ24_polr0
                            V_beamQ24_cplx_in1_polr0 = Vim_beamQ24_polr0
                            V_beamQ0_cplx_in0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx_in1_polr1 = Vim_beamQ0_polr1
                            V_beamQ24_cplx_in0_polr1 = Vre_beamQ24_polr1
                            V_beamQ24_cplx_in1_polr1 = Vim_beamQ24_polr1
                            (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr0, V_beamQ0_cplx_in1_polr0),
                                (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0),
                            )
                            (Y_beamQ24_cplx0_polr0, Y_beamQ24_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ24_cplx_in0_polr0, V_beamQ24_cplx_in1_polr0),
                                (Y_beamQ24_cplx0_polr0, Y_beamQ24_cplx1_polr0),
                            )
                            (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr1, V_beamQ0_cplx_in1_polr1),
                                (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1),
                            )
                            (Y_beamQ24_cplx0_polr1, Y_beamQ24_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ24_cplx_in0_polr1, V_beamQ24_cplx_in1_polr1),
                                (Y_beamQ24_cplx0_polr1, Y_beamQ24_cplx1_polr1),
                            )
                            Ẽ_beamQ0_cplx0_polr0 = Y_beamQ0_cplx0_polr0
                            Ẽ_beamQ24_cplx0_polr0 = Y_beamQ24_cplx0_polr0
                            Ẽ_beamQ0_cplx1_polr0 = Y_beamQ0_cplx1_polr0
                            Ẽ_beamQ24_cplx1_polr0 = Y_beamQ24_cplx1_polr0
                            Ẽ_beamQ0_cplx0_polr1 = Y_beamQ0_cplx0_polr1
                            Ẽ_beamQ24_cplx0_polr1 = Y_beamQ24_cplx0_polr1
                            Ẽ_beamQ0_cplx1_polr1 = Y_beamQ0_cplx1_polr1
                            Ẽ_beamQ24_cplx1_polr1 = Y_beamQ24_cplx1_polr1
                            Ẽp0_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr0
                            Ẽp1_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr1
                            Ẽp0_beamQ24_cplx0 = Ẽ_beamQ24_cplx0_polr0
                            Ẽp1_beamQ24_cplx0 = Ẽ_beamQ24_cplx0_polr1
                            Ẽp0_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr0
                            Ẽp1_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr1
                            Ẽp0_beamQ24_cplx1 = Ẽ_beamQ24_cplx1_polr0
                            Ẽp1_beamQ24_cplx1 = Ẽ_beamQ24_cplx1_polr1
                            Ẽp0re_beamQ0 = Ẽp0_beamQ0_cplx0
                            Ẽp0im_beamQ0 = Ẽp0_beamQ0_cplx1
                            Ẽp0re_beamQ24 = Ẽp0_beamQ24_cplx0
                            Ẽp0im_beamQ24 = Ẽp0_beamQ24_cplx1
                            Ẽp1re_beamQ0 = Ẽp1_beamQ0_cplx0
                            Ẽp1im_beamQ0 = Ẽp1_beamQ0_cplx1
                            Ẽp1re_beamQ24 = Ẽp1_beamQ24_cplx0
                            Ẽp1im_beamQ24 = Ẽp1_beamQ24_cplx1
                            I_beamQ0 = muladd(
                                Float16x2(0.003124237f0, 0.003124237f0),
                                muladd(
                                    Ẽp1im_beamQ0,
                                    Ẽp1im_beamQ0,
                                    muladd(
                                        Ẽp1re_beamQ0, Ẽp1re_beamQ0, muladd(Ẽp0im_beamQ0, Ẽp0im_beamQ0, Ẽp0re_beamQ0 * Ẽp0re_beamQ0)
                                    ),
                                ),
                                I_beamQ0,
                            )
                            I_beamQ24 = muladd(
                                Float16x2(0.003124237f0, 0.003124237f0),
                                muladd(
                                    Ẽp1im_beamQ24,
                                    Ẽp1im_beamQ24,
                                    muladd(
                                        Ẽp1re_beamQ24,
                                        Ẽp1re_beamQ24,
                                        muladd(Ẽp0im_beamQ24, Ẽp0im_beamQ24, Ẽp0re_beamQ24 * Ẽp0re_beamQ24),
                                    ),
                                ),
                                I_beamQ24,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 4 == 0i32
                                if t_running == 40
                                    if let
                                        thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                        warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24)
                                        p = (2i32) * thread
                                        q = (2i32) * warp
                                        0i32 ≤ p < 48 && 0i32 ≤ q < 48
                                    end
                                        if true
                                            I_memory[let
                                                offset = 7077888 * T̄min + 1152 * F̄min
                                                length = 452984832
                                                mod(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 1152 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 64) % 64) % 64) * 7077888 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 48) * 24) + 0) + offset, length)
                                            end + 0x01] = I_beamQ0
                                        end
                                        if true
                                            I_memory[let
                                                offset = 7077888 * T̄min + 1152 * F̄min
                                                length = 452984832
                                                mod(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 1152 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 64) % 64) % 64) * 7077888 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 24) % 48) * 24) + 0) + offset, length)
                                            end + 0x01] = I_beamQ24
                                        end
                                    end
                                    I_beamQ0 = zero(Float16x2)
                                    I_beamQ24 = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let
                            t = 2
                            G_beamQ0_polr0 = zero(Float16x2)
                            G_beamQ24_polr0 = zero(Float16x2)
                            G_beamQ0_polr1 = zero(Float16x2)
                            G_beamQ24_polr1 = zero(Float16x2)
                            if let
                                thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                mlo2_offset = 8
                                mlo2_length = 4
                                mlo = (1i32) * ((thread ÷ (4i32)) % (2i32)) + (2i32) * ((thread ÷ mlo2_offset) % mlo2_length)
                                mlo < 6
                            end
                                G_beamQ0_polr0 = Gsh_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 0 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ24_polr0 = Gsh_shared[((((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 0 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ0_polr1 = Gsh_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ24_polr1 = Gsh_shared[((((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 32 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                            end
                            X_beamQ0_polr0 = G_beamQ0_polr0
                            X_beamQ24_polr0 = G_beamQ24_polr0
                            X_beamQ0_polr1 = G_beamQ0_polr1
                            X_beamQ24_polr1 = G_beamQ24_polr1
                            Z_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ24_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ24_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ24_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Z_beamQ24_cplx1_polr1 = zero(Float16x2)
                            (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ0_polr0, (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0)
                            )
                            (Z_beamQ24_cplx0_polr0, Z_beamQ24_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ24_polr0, (Z_beamQ24_cplx0_polr0, Z_beamQ24_cplx1_polr0)
                            )
                            (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ0_polr1, (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1)
                            )
                            (Z_beamQ24_cplx0_polr1, Z_beamQ24_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ24_polr1, (Z_beamQ24_cplx0_polr1, Z_beamQ24_cplx1_polr1)
                            )
                            bΓ²re = bΓ²_cplx0
                            bΓ²im = bΓ²_cplx1
                            Zre_beamQ0_polr0 = Z_beamQ0_cplx0_polr0
                            Zim_beamQ0_polr0 = Z_beamQ0_cplx1_polr0
                            Zre_beamQ24_polr0 = Z_beamQ24_cplx0_polr0
                            Zim_beamQ24_polr0 = Z_beamQ24_cplx1_polr0
                            Zre_beamQ0_polr1 = Z_beamQ0_cplx0_polr1
                            Zim_beamQ0_polr1 = Z_beamQ0_cplx1_polr1
                            Zre_beamQ24_polr1 = Z_beamQ24_cplx0_polr1
                            Zim_beamQ24_polr1 = Z_beamQ24_cplx1_polr1
                            Vre_beamQ0_polr0 = muladd(bΓ²re, Zre_beamQ0_polr0, -bΓ²im * Zim_beamQ0_polr0)
                            Vre_beamQ24_polr0 = muladd(bΓ²re, Zre_beamQ24_polr0, -bΓ²im * Zim_beamQ24_polr0)
                            Vre_beamQ0_polr1 = muladd(bΓ²re, Zre_beamQ0_polr1, -bΓ²im * Zim_beamQ0_polr1)
                            Vre_beamQ24_polr1 = muladd(bΓ²re, Zre_beamQ24_polr1, -bΓ²im * Zim_beamQ24_polr1)
                            Vim_beamQ0_polr0 = muladd(bΓ²re, Zim_beamQ0_polr0, +bΓ²im * Zre_beamQ0_polr0)
                            Vim_beamQ24_polr0 = muladd(bΓ²re, Zim_beamQ24_polr0, +bΓ²im * Zre_beamQ24_polr0)
                            Vim_beamQ0_polr1 = muladd(bΓ²re, Zim_beamQ0_polr1, +bΓ²im * Zre_beamQ0_polr1)
                            Vim_beamQ24_polr1 = muladd(bΓ²re, Zim_beamQ24_polr1, +bΓ²im * Zre_beamQ24_polr1)
                            V_beamQ0_cplx0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx1_polr0 = Vim_beamQ0_polr0
                            V_beamQ24_cplx0_polr0 = Vre_beamQ24_polr0
                            V_beamQ24_cplx1_polr0 = Vim_beamQ24_polr0
                            V_beamQ0_cplx0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx1_polr1 = Vim_beamQ0_polr1
                            V_beamQ24_cplx0_polr1 = Vre_beamQ24_polr1
                            V_beamQ24_cplx1_polr1 = Vim_beamQ24_polr1
                            Y_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ24_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ24_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ24_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Y_beamQ24_cplx1_polr1 = zero(Float16x2)
                            Vre_beamQ0_polr0 = V_beamQ0_cplx0_polr0
                            Vim_beamQ0_polr0 = V_beamQ0_cplx1_polr0
                            Vre_beamQ24_polr0 = V_beamQ24_cplx0_polr0
                            Vim_beamQ24_polr0 = V_beamQ24_cplx1_polr0
                            Vre_beamQ0_polr1 = V_beamQ0_cplx0_polr1
                            Vim_beamQ0_polr1 = V_beamQ0_cplx1_polr1
                            Vre_beamQ24_polr1 = V_beamQ24_cplx0_polr1
                            Vim_beamQ24_polr1 = V_beamQ24_cplx1_polr1
                            V_beamQ0_cplx_in0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx_in1_polr0 = Vim_beamQ0_polr0
                            V_beamQ24_cplx_in0_polr0 = Vre_beamQ24_polr0
                            V_beamQ24_cplx_in1_polr0 = Vim_beamQ24_polr0
                            V_beamQ0_cplx_in0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx_in1_polr1 = Vim_beamQ0_polr1
                            V_beamQ24_cplx_in0_polr1 = Vre_beamQ24_polr1
                            V_beamQ24_cplx_in1_polr1 = Vim_beamQ24_polr1
                            (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr0, V_beamQ0_cplx_in1_polr0),
                                (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0),
                            )
                            (Y_beamQ24_cplx0_polr0, Y_beamQ24_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ24_cplx_in0_polr0, V_beamQ24_cplx_in1_polr0),
                                (Y_beamQ24_cplx0_polr0, Y_beamQ24_cplx1_polr0),
                            )
                            (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr1, V_beamQ0_cplx_in1_polr1),
                                (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1),
                            )
                            (Y_beamQ24_cplx0_polr1, Y_beamQ24_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ24_cplx_in0_polr1, V_beamQ24_cplx_in1_polr1),
                                (Y_beamQ24_cplx0_polr1, Y_beamQ24_cplx1_polr1),
                            )
                            Ẽ_beamQ0_cplx0_polr0 = Y_beamQ0_cplx0_polr0
                            Ẽ_beamQ24_cplx0_polr0 = Y_beamQ24_cplx0_polr0
                            Ẽ_beamQ0_cplx1_polr0 = Y_beamQ0_cplx1_polr0
                            Ẽ_beamQ24_cplx1_polr0 = Y_beamQ24_cplx1_polr0
                            Ẽ_beamQ0_cplx0_polr1 = Y_beamQ0_cplx0_polr1
                            Ẽ_beamQ24_cplx0_polr1 = Y_beamQ24_cplx0_polr1
                            Ẽ_beamQ0_cplx1_polr1 = Y_beamQ0_cplx1_polr1
                            Ẽ_beamQ24_cplx1_polr1 = Y_beamQ24_cplx1_polr1
                            Ẽp0_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr0
                            Ẽp1_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr1
                            Ẽp0_beamQ24_cplx0 = Ẽ_beamQ24_cplx0_polr0
                            Ẽp1_beamQ24_cplx0 = Ẽ_beamQ24_cplx0_polr1
                            Ẽp0_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr0
                            Ẽp1_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr1
                            Ẽp0_beamQ24_cplx1 = Ẽ_beamQ24_cplx1_polr0
                            Ẽp1_beamQ24_cplx1 = Ẽ_beamQ24_cplx1_polr1
                            Ẽp0re_beamQ0 = Ẽp0_beamQ0_cplx0
                            Ẽp0im_beamQ0 = Ẽp0_beamQ0_cplx1
                            Ẽp0re_beamQ24 = Ẽp0_beamQ24_cplx0
                            Ẽp0im_beamQ24 = Ẽp0_beamQ24_cplx1
                            Ẽp1re_beamQ0 = Ẽp1_beamQ0_cplx0
                            Ẽp1im_beamQ0 = Ẽp1_beamQ0_cplx1
                            Ẽp1re_beamQ24 = Ẽp1_beamQ24_cplx0
                            Ẽp1im_beamQ24 = Ẽp1_beamQ24_cplx1
                            I_beamQ0 = muladd(
                                Float16x2(0.003124237f0, 0.003124237f0),
                                muladd(
                                    Ẽp1im_beamQ0,
                                    Ẽp1im_beamQ0,
                                    muladd(
                                        Ẽp1re_beamQ0, Ẽp1re_beamQ0, muladd(Ẽp0im_beamQ0, Ẽp0im_beamQ0, Ẽp0re_beamQ0 * Ẽp0re_beamQ0)
                                    ),
                                ),
                                I_beamQ0,
                            )
                            I_beamQ24 = muladd(
                                Float16x2(0.003124237f0, 0.003124237f0),
                                muladd(
                                    Ẽp1im_beamQ24,
                                    Ẽp1im_beamQ24,
                                    muladd(
                                        Ẽp1re_beamQ24,
                                        Ẽp1re_beamQ24,
                                        muladd(Ẽp0im_beamQ24, Ẽp0im_beamQ24, Ẽp0re_beamQ24 * Ẽp0re_beamQ24),
                                    ),
                                ),
                                I_beamQ24,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 4 == 0i32
                                if t_running == 40
                                    if let
                                        thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                        warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24)
                                        p = (2i32) * thread
                                        q = (2i32) * warp
                                        0i32 ≤ p < 48 && 0i32 ≤ q < 48
                                    end
                                        if true
                                            I_memory[let
                                                offset = 7077888 * T̄min + 1152 * F̄min
                                                length = 452984832
                                                mod(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 1152 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 64) % 64) % 64) * 7077888 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 48) * 24) + 0) + offset, length)
                                            end + 0x01] = I_beamQ0
                                        end
                                        if true
                                            I_memory[let
                                                offset = 7077888 * T̄min + 1152 * F̄min
                                                length = 452984832
                                                mod(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 1152 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 64) % 64) % 64) * 7077888 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 24) % 48) * 24) + 0) + offset, length)
                                            end + 0x01] = I_beamQ24
                                        end
                                    end
                                    I_beamQ0 = zero(Float16x2)
                                    I_beamQ24 = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let
                            t = 3
                            G_beamQ0_polr0 = zero(Float16x2)
                            G_beamQ24_polr0 = zero(Float16x2)
                            G_beamQ0_polr1 = zero(Float16x2)
                            G_beamQ24_polr1 = zero(Float16x2)
                            if let
                                thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                mlo2_offset = 8
                                mlo2_length = 4
                                mlo = (1i32) * ((thread ÷ (4i32)) % (2i32)) + (2i32) * ((thread ÷ mlo2_offset) % mlo2_length)
                                mlo < 6
                            end
                                G_beamQ0_polr0 = Gsh_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 0 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ24_polr0 = Gsh_shared[((((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 0 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ0_polr1 = Gsh_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ24_polr1 = Gsh_shared[((((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 32 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                            end
                            X_beamQ0_polr0 = G_beamQ0_polr0
                            X_beamQ24_polr0 = G_beamQ24_polr0
                            X_beamQ0_polr1 = G_beamQ0_polr1
                            X_beamQ24_polr1 = G_beamQ24_polr1
                            Z_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ24_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ24_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ24_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Z_beamQ24_cplx1_polr1 = zero(Float16x2)
                            (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ0_polr0, (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0)
                            )
                            (Z_beamQ24_cplx0_polr0, Z_beamQ24_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ24_polr0, (Z_beamQ24_cplx0_polr0, Z_beamQ24_cplx1_polr0)
                            )
                            (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ0_polr1, (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1)
                            )
                            (Z_beamQ24_cplx0_polr1, Z_beamQ24_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ24_polr1, (Z_beamQ24_cplx0_polr1, Z_beamQ24_cplx1_polr1)
                            )
                            bΓ²re = bΓ²_cplx0
                            bΓ²im = bΓ²_cplx1
                            Zre_beamQ0_polr0 = Z_beamQ0_cplx0_polr0
                            Zim_beamQ0_polr0 = Z_beamQ0_cplx1_polr0
                            Zre_beamQ24_polr0 = Z_beamQ24_cplx0_polr0
                            Zim_beamQ24_polr0 = Z_beamQ24_cplx1_polr0
                            Zre_beamQ0_polr1 = Z_beamQ0_cplx0_polr1
                            Zim_beamQ0_polr1 = Z_beamQ0_cplx1_polr1
                            Zre_beamQ24_polr1 = Z_beamQ24_cplx0_polr1
                            Zim_beamQ24_polr1 = Z_beamQ24_cplx1_polr1
                            Vre_beamQ0_polr0 = muladd(bΓ²re, Zre_beamQ0_polr0, -bΓ²im * Zim_beamQ0_polr0)
                            Vre_beamQ24_polr0 = muladd(bΓ²re, Zre_beamQ24_polr0, -bΓ²im * Zim_beamQ24_polr0)
                            Vre_beamQ0_polr1 = muladd(bΓ²re, Zre_beamQ0_polr1, -bΓ²im * Zim_beamQ0_polr1)
                            Vre_beamQ24_polr1 = muladd(bΓ²re, Zre_beamQ24_polr1, -bΓ²im * Zim_beamQ24_polr1)
                            Vim_beamQ0_polr0 = muladd(bΓ²re, Zim_beamQ0_polr0, +bΓ²im * Zre_beamQ0_polr0)
                            Vim_beamQ24_polr0 = muladd(bΓ²re, Zim_beamQ24_polr0, +bΓ²im * Zre_beamQ24_polr0)
                            Vim_beamQ0_polr1 = muladd(bΓ²re, Zim_beamQ0_polr1, +bΓ²im * Zre_beamQ0_polr1)
                            Vim_beamQ24_polr1 = muladd(bΓ²re, Zim_beamQ24_polr1, +bΓ²im * Zre_beamQ24_polr1)
                            V_beamQ0_cplx0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx1_polr0 = Vim_beamQ0_polr0
                            V_beamQ24_cplx0_polr0 = Vre_beamQ24_polr0
                            V_beamQ24_cplx1_polr0 = Vim_beamQ24_polr0
                            V_beamQ0_cplx0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx1_polr1 = Vim_beamQ0_polr1
                            V_beamQ24_cplx0_polr1 = Vre_beamQ24_polr1
                            V_beamQ24_cplx1_polr1 = Vim_beamQ24_polr1
                            Y_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ24_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ24_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ24_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Y_beamQ24_cplx1_polr1 = zero(Float16x2)
                            Vre_beamQ0_polr0 = V_beamQ0_cplx0_polr0
                            Vim_beamQ0_polr0 = V_beamQ0_cplx1_polr0
                            Vre_beamQ24_polr0 = V_beamQ24_cplx0_polr0
                            Vim_beamQ24_polr0 = V_beamQ24_cplx1_polr0
                            Vre_beamQ0_polr1 = V_beamQ0_cplx0_polr1
                            Vim_beamQ0_polr1 = V_beamQ0_cplx1_polr1
                            Vre_beamQ24_polr1 = V_beamQ24_cplx0_polr1
                            Vim_beamQ24_polr1 = V_beamQ24_cplx1_polr1
                            V_beamQ0_cplx_in0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx_in1_polr0 = Vim_beamQ0_polr0
                            V_beamQ24_cplx_in0_polr0 = Vre_beamQ24_polr0
                            V_beamQ24_cplx_in1_polr0 = Vim_beamQ24_polr0
                            V_beamQ0_cplx_in0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx_in1_polr1 = Vim_beamQ0_polr1
                            V_beamQ24_cplx_in0_polr1 = Vre_beamQ24_polr1
                            V_beamQ24_cplx_in1_polr1 = Vim_beamQ24_polr1
                            (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr0, V_beamQ0_cplx_in1_polr0),
                                (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0),
                            )
                            (Y_beamQ24_cplx0_polr0, Y_beamQ24_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ24_cplx_in0_polr0, V_beamQ24_cplx_in1_polr0),
                                (Y_beamQ24_cplx0_polr0, Y_beamQ24_cplx1_polr0),
                            )
                            (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr1, V_beamQ0_cplx_in1_polr1),
                                (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1),
                            )
                            (Y_beamQ24_cplx0_polr1, Y_beamQ24_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ24_cplx_in0_polr1, V_beamQ24_cplx_in1_polr1),
                                (Y_beamQ24_cplx0_polr1, Y_beamQ24_cplx1_polr1),
                            )
                            Ẽ_beamQ0_cplx0_polr0 = Y_beamQ0_cplx0_polr0
                            Ẽ_beamQ24_cplx0_polr0 = Y_beamQ24_cplx0_polr0
                            Ẽ_beamQ0_cplx1_polr0 = Y_beamQ0_cplx1_polr0
                            Ẽ_beamQ24_cplx1_polr0 = Y_beamQ24_cplx1_polr0
                            Ẽ_beamQ0_cplx0_polr1 = Y_beamQ0_cplx0_polr1
                            Ẽ_beamQ24_cplx0_polr1 = Y_beamQ24_cplx0_polr1
                            Ẽ_beamQ0_cplx1_polr1 = Y_beamQ0_cplx1_polr1
                            Ẽ_beamQ24_cplx1_polr1 = Y_beamQ24_cplx1_polr1
                            Ẽp0_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr0
                            Ẽp1_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr1
                            Ẽp0_beamQ24_cplx0 = Ẽ_beamQ24_cplx0_polr0
                            Ẽp1_beamQ24_cplx0 = Ẽ_beamQ24_cplx0_polr1
                            Ẽp0_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr0
                            Ẽp1_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr1
                            Ẽp0_beamQ24_cplx1 = Ẽ_beamQ24_cplx1_polr0
                            Ẽp1_beamQ24_cplx1 = Ẽ_beamQ24_cplx1_polr1
                            Ẽp0re_beamQ0 = Ẽp0_beamQ0_cplx0
                            Ẽp0im_beamQ0 = Ẽp0_beamQ0_cplx1
                            Ẽp0re_beamQ24 = Ẽp0_beamQ24_cplx0
                            Ẽp0im_beamQ24 = Ẽp0_beamQ24_cplx1
                            Ẽp1re_beamQ0 = Ẽp1_beamQ0_cplx0
                            Ẽp1im_beamQ0 = Ẽp1_beamQ0_cplx1
                            Ẽp1re_beamQ24 = Ẽp1_beamQ24_cplx0
                            Ẽp1im_beamQ24 = Ẽp1_beamQ24_cplx1
                            I_beamQ0 = muladd(
                                Float16x2(0.003124237f0, 0.003124237f0),
                                muladd(
                                    Ẽp1im_beamQ0,
                                    Ẽp1im_beamQ0,
                                    muladd(
                                        Ẽp1re_beamQ0, Ẽp1re_beamQ0, muladd(Ẽp0im_beamQ0, Ẽp0im_beamQ0, Ẽp0re_beamQ0 * Ẽp0re_beamQ0)
                                    ),
                                ),
                                I_beamQ0,
                            )
                            I_beamQ24 = muladd(
                                Float16x2(0.003124237f0, 0.003124237f0),
                                muladd(
                                    Ẽp1im_beamQ24,
                                    Ẽp1im_beamQ24,
                                    muladd(
                                        Ẽp1re_beamQ24,
                                        Ẽp1re_beamQ24,
                                        muladd(Ẽp0im_beamQ24, Ẽp0im_beamQ24, Ẽp0re_beamQ24 * Ẽp0re_beamQ24),
                                    ),
                                ),
                                I_beamQ24,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 4 == 0i32
                                if t_running == 40
                                    if let
                                        thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                        warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24)
                                        p = (2i32) * thread
                                        q = (2i32) * warp
                                        0i32 ≤ p < 48 && 0i32 ≤ q < 48
                                    end
                                        if true
                                            I_memory[let
                                                offset = 7077888 * T̄min + 1152 * F̄min
                                                length = 452984832
                                                mod(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 1152 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 64) % 64) % 64) * 7077888 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 48) * 24) + 0) + offset, length)
                                            end + 0x01] = I_beamQ0
                                        end
                                        if true
                                            I_memory[let
                                                offset = 7077888 * T̄min + 1152 * F̄min
                                                length = 452984832
                                                mod(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 1152 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 64) % 64) % 64) * 7077888 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 24) % 48) * 24) + 0) + offset, length)
                                            end + 0x01] = I_beamQ24
                                        end
                                    end
                                    I_beamQ0 = zero(Float16x2)
                                    I_beamQ24 = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        IndexSpaces.cuda_sync_threads()
                    end
                end
                let
                    t_inner_hi = 24
                    for t_inner_lo in 0:4:23
                        Freg2′_time0 = zero(Int4x8)
                        if t_inner_lo == 0
                            Freg2′_time0 = Freg2_time0
                        end
                        if t_inner_lo == 4
                            Freg2′_time0 = Freg2_time4
                        end
                        if t_inner_lo == 8
                            Freg2′_time0 = Freg2_time8
                        end
                        if t_inner_lo == 12
                            Freg2′_time0 = Freg2_time12
                        end
                        if t_inner_lo == 16
                            Freg2′_time0 = Freg2_time16
                        end
                        if t_inner_lo == 20
                            Freg2′_time0 = Freg2_time20
                        end
                        Freg2′_time1 = zero(Int4x8)
                        if t_inner_lo == 0
                            Freg2′_time1 = Freg2_time1
                        end
                        if t_inner_lo == 4
                            Freg2′_time1 = Freg2_time5
                        end
                        if t_inner_lo == 8
                            Freg2′_time1 = Freg2_time9
                        end
                        if t_inner_lo == 12
                            Freg2′_time1 = Freg2_time13
                        end
                        if t_inner_lo == 16
                            Freg2′_time1 = Freg2_time17
                        end
                        if t_inner_lo == 20
                            Freg2′_time1 = Freg2_time21
                        end
                        Freg2′_time2 = zero(Int4x8)
                        if t_inner_lo == 0
                            Freg2′_time2 = Freg2_time2
                        end
                        if t_inner_lo == 4
                            Freg2′_time2 = Freg2_time6
                        end
                        if t_inner_lo == 8
                            Freg2′_time2 = Freg2_time10
                        end
                        if t_inner_lo == 12
                            Freg2′_time2 = Freg2_time14
                        end
                        if t_inner_lo == 16
                            Freg2′_time2 = Freg2_time18
                        end
                        if t_inner_lo == 20
                            Freg2′_time2 = Freg2_time22
                        end
                        Freg2′_time3 = zero(Int4x8)
                        if t_inner_lo == 0
                            Freg2′_time3 = Freg2_time3
                        end
                        if t_inner_lo == 4
                            Freg2′_time3 = Freg2_time7
                        end
                        if t_inner_lo == 8
                            Freg2′_time3 = Freg2_time11
                        end
                        if t_inner_lo == 12
                            Freg2′_time3 = Freg2_time15
                        end
                        if t_inner_lo == 16
                            Freg2′_time3 = Freg2_time19
                        end
                        if t_inner_lo == 20
                            Freg2′_time3 = Freg2_time23
                        end
                        (E′_polr0_time0, E′_polr1_time0, E′_polr0_time24, E′_polr1_time24) = convert(
                            NTuple{4,Float16x2}, Freg2′_time0
                        )
                        (E′_polr0_time1, E′_polr1_time1, E′_polr0_time25, E′_polr1_time25) = convert(
                            NTuple{4,Float16x2}, Freg2′_time1
                        )
                        (E′_polr0_time2, E′_polr1_time2, E′_polr0_time26, E′_polr1_time26) = convert(
                            NTuple{4,Float16x2}, Freg2′_time2
                        )
                        (E′_polr0_time3, E′_polr1_time3, E′_polr0_time27, E′_polr1_time27) = convert(
                            NTuple{4,Float16x2}, Freg2′_time3
                        )
                        E_polr0_time0 = E′_polr0_time24
                        E_polr1_time0 = E′_polr1_time24
                        E_polr0_time1 = E′_polr0_time25
                        E_polr1_time1 = E′_polr1_time25
                        E_polr0_time2 = E′_polr0_time26
                        E_polr1_time2 = E′_polr1_time26
                        E_polr0_time3 = E′_polr0_time27
                        E_polr1_time3 = E′_polr1_time27
                        WE_polr0_time0 = complex_mul(W_polr0, E_polr0_time0)
                        WE_polr1_time0 = complex_mul(W_polr1, E_polr1_time0)
                        WE_polr0_time1 = complex_mul(W_polr0, E_polr0_time1)
                        WE_polr1_time1 = complex_mul(W_polr1, E_polr1_time1)
                        WE_polr0_time2 = complex_mul(W_polr0, E_polr0_time2)
                        WE_polr1_time2 = complex_mul(W_polr1, E_polr1_time2)
                        WE_polr0_time3 = complex_mul(W_polr0, E_polr0_time3)
                        WE_polr1_time3 = complex_mul(W_polr1, E_polr1_time3)
                        X_polr0_time0 = WE_polr0_time0
                        X_polr1_time0 = WE_polr1_time0
                        X_polr0_time1 = WE_polr0_time1
                        X_polr1_time1 = WE_polr1_time1
                        X_polr0_time2 = WE_polr0_time2
                        X_polr1_time2 = WE_polr1_time2
                        X_polr0_time3 = WE_polr0_time3
                        X_polr1_time3 = WE_polr1_time3
                        Z_cplx0_polr0_time0 = zero(Float16x2)
                        Z_cplx1_polr0_time0 = zero(Float16x2)
                        Z_cplx0_polr1_time0 = zero(Float16x2)
                        Z_cplx1_polr1_time0 = zero(Float16x2)
                        Z_cplx0_polr0_time1 = zero(Float16x2)
                        Z_cplx1_polr0_time1 = zero(Float16x2)
                        Z_cplx0_polr1_time1 = zero(Float16x2)
                        Z_cplx1_polr1_time1 = zero(Float16x2)
                        Z_cplx0_polr0_time2 = zero(Float16x2)
                        Z_cplx1_polr0_time2 = zero(Float16x2)
                        Z_cplx0_polr1_time2 = zero(Float16x2)
                        Z_cplx1_polr1_time2 = zero(Float16x2)
                        Z_cplx0_polr0_time3 = zero(Float16x2)
                        Z_cplx1_polr0_time3 = zero(Float16x2)
                        Z_cplx0_polr1_time3 = zero(Float16x2)
                        Z_cplx1_polr1_time3 = zero(Float16x2)
                        (Z_cplx0_polr0_time0, Z_cplx1_polr0_time0) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr0_time0, (Z_cplx0_polr0_time0, Z_cplx1_polr0_time0)
                        )
                        (Z_cplx0_polr1_time0, Z_cplx1_polr1_time0) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr1_time0, (Z_cplx0_polr1_time0, Z_cplx1_polr1_time0)
                        )
                        (Z_cplx0_polr0_time1, Z_cplx1_polr0_time1) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr0_time1, (Z_cplx0_polr0_time1, Z_cplx1_polr0_time1)
                        )
                        (Z_cplx0_polr1_time1, Z_cplx1_polr1_time1) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr1_time1, (Z_cplx0_polr1_time1, Z_cplx1_polr1_time1)
                        )
                        (Z_cplx0_polr0_time2, Z_cplx1_polr0_time2) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr0_time2, (Z_cplx0_polr0_time2, Z_cplx1_polr0_time2)
                        )
                        (Z_cplx0_polr1_time2, Z_cplx1_polr1_time2) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr1_time2, (Z_cplx0_polr1_time2, Z_cplx1_polr1_time2)
                        )
                        (Z_cplx0_polr0_time3, Z_cplx1_polr0_time3) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr0_time3, (Z_cplx0_polr0_time3, Z_cplx1_polr0_time3)
                        )
                        (Z_cplx0_polr1_time3, Z_cplx1_polr1_time3) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr1_time3, (Z_cplx0_polr1_time3, Z_cplx1_polr1_time3)
                        )
                        aΓ²re = aΓ²_cplx0
                        aΓ²im = aΓ²_cplx1
                        Zre_polr0_time0 = Z_cplx0_polr0_time0
                        Zim_polr0_time0 = Z_cplx1_polr0_time0
                        Zre_polr1_time0 = Z_cplx0_polr1_time0
                        Zim_polr1_time0 = Z_cplx1_polr1_time0
                        Zre_polr0_time1 = Z_cplx0_polr0_time1
                        Zim_polr0_time1 = Z_cplx1_polr0_time1
                        Zre_polr1_time1 = Z_cplx0_polr1_time1
                        Zim_polr1_time1 = Z_cplx1_polr1_time1
                        Zre_polr0_time2 = Z_cplx0_polr0_time2
                        Zim_polr0_time2 = Z_cplx1_polr0_time2
                        Zre_polr1_time2 = Z_cplx0_polr1_time2
                        Zim_polr1_time2 = Z_cplx1_polr1_time2
                        Zre_polr0_time3 = Z_cplx0_polr0_time3
                        Zim_polr0_time3 = Z_cplx1_polr0_time3
                        Zre_polr1_time3 = Z_cplx0_polr1_time3
                        Zim_polr1_time3 = Z_cplx1_polr1_time3
                        Vre_polr0_time0 = muladd(aΓ²re, Zre_polr0_time0, -aΓ²im * Zim_polr0_time0)
                        Vre_polr1_time0 = muladd(aΓ²re, Zre_polr1_time0, -aΓ²im * Zim_polr1_time0)
                        Vre_polr0_time1 = muladd(aΓ²re, Zre_polr0_time1, -aΓ²im * Zim_polr0_time1)
                        Vre_polr1_time1 = muladd(aΓ²re, Zre_polr1_time1, -aΓ²im * Zim_polr1_time1)
                        Vre_polr0_time2 = muladd(aΓ²re, Zre_polr0_time2, -aΓ²im * Zim_polr0_time2)
                        Vre_polr1_time2 = muladd(aΓ²re, Zre_polr1_time2, -aΓ²im * Zim_polr1_time2)
                        Vre_polr0_time3 = muladd(aΓ²re, Zre_polr0_time3, -aΓ²im * Zim_polr0_time3)
                        Vre_polr1_time3 = muladd(aΓ²re, Zre_polr1_time3, -aΓ²im * Zim_polr1_time3)
                        Vim_polr0_time0 = muladd(aΓ²re, Zim_polr0_time0, +aΓ²im * Zre_polr0_time0)
                        Vim_polr1_time0 = muladd(aΓ²re, Zim_polr1_time0, +aΓ²im * Zre_polr1_time0)
                        Vim_polr0_time1 = muladd(aΓ²re, Zim_polr0_time1, +aΓ²im * Zre_polr0_time1)
                        Vim_polr1_time1 = muladd(aΓ²re, Zim_polr1_time1, +aΓ²im * Zre_polr1_time1)
                        Vim_polr0_time2 = muladd(aΓ²re, Zim_polr0_time2, +aΓ²im * Zre_polr0_time2)
                        Vim_polr1_time2 = muladd(aΓ²re, Zim_polr1_time2, +aΓ²im * Zre_polr1_time2)
                        Vim_polr0_time3 = muladd(aΓ²re, Zim_polr0_time3, +aΓ²im * Zre_polr0_time3)
                        Vim_polr1_time3 = muladd(aΓ²re, Zim_polr1_time3, +aΓ²im * Zre_polr1_time3)
                        V_cplx0_polr0_time0 = Vre_polr0_time0
                        V_cplx1_polr0_time0 = Vim_polr0_time0
                        V_cplx0_polr1_time0 = Vre_polr1_time0
                        V_cplx1_polr1_time0 = Vim_polr1_time0
                        V_cplx0_polr0_time1 = Vre_polr0_time1
                        V_cplx1_polr0_time1 = Vim_polr0_time1
                        V_cplx0_polr1_time1 = Vre_polr1_time1
                        V_cplx1_polr1_time1 = Vim_polr1_time1
                        V_cplx0_polr0_time2 = Vre_polr0_time2
                        V_cplx1_polr0_time2 = Vim_polr0_time2
                        V_cplx0_polr1_time2 = Vre_polr1_time2
                        V_cplx1_polr1_time2 = Vim_polr1_time2
                        V_cplx0_polr0_time3 = Vre_polr0_time3
                        V_cplx1_polr0_time3 = Vim_polr0_time3
                        V_cplx0_polr1_time3 = Vre_polr1_time3
                        V_cplx1_polr1_time3 = Vim_polr1_time3
                        Y_cplx0_polr0_time0 = zero(Float16x2)
                        Y_cplx1_polr0_time0 = zero(Float16x2)
                        Y_cplx0_polr1_time0 = zero(Float16x2)
                        Y_cplx1_polr1_time0 = zero(Float16x2)
                        Y_cplx0_polr0_time1 = zero(Float16x2)
                        Y_cplx1_polr0_time1 = zero(Float16x2)
                        Y_cplx0_polr1_time1 = zero(Float16x2)
                        Y_cplx1_polr1_time1 = zero(Float16x2)
                        Y_cplx0_polr0_time2 = zero(Float16x2)
                        Y_cplx1_polr0_time2 = zero(Float16x2)
                        Y_cplx0_polr1_time2 = zero(Float16x2)
                        Y_cplx1_polr1_time2 = zero(Float16x2)
                        Y_cplx0_polr0_time3 = zero(Float16x2)
                        Y_cplx1_polr0_time3 = zero(Float16x2)
                        Y_cplx0_polr1_time3 = zero(Float16x2)
                        Y_cplx1_polr1_time3 = zero(Float16x2)
                        Vre_polr0_time0 = V_cplx0_polr0_time0
                        Vim_polr0_time0 = V_cplx1_polr0_time0
                        Vre_polr1_time0 = V_cplx0_polr1_time0
                        Vim_polr1_time0 = V_cplx1_polr1_time0
                        Vre_polr0_time1 = V_cplx0_polr0_time1
                        Vim_polr0_time1 = V_cplx1_polr0_time1
                        Vre_polr1_time1 = V_cplx0_polr1_time1
                        Vim_polr1_time1 = V_cplx1_polr1_time1
                        Vre_polr0_time2 = V_cplx0_polr0_time2
                        Vim_polr0_time2 = V_cplx1_polr0_time2
                        Vre_polr1_time2 = V_cplx0_polr1_time2
                        Vim_polr1_time2 = V_cplx1_polr1_time2
                        Vre_polr0_time3 = V_cplx0_polr0_time3
                        Vim_polr0_time3 = V_cplx1_polr0_time3
                        Vre_polr1_time3 = V_cplx0_polr1_time3
                        Vim_polr1_time3 = V_cplx1_polr1_time3
                        V_cplx_in0_polr0_time0 = Vre_polr0_time0
                        V_cplx_in1_polr0_time0 = Vim_polr0_time0
                        V_cplx_in0_polr1_time0 = Vre_polr1_time0
                        V_cplx_in1_polr1_time0 = Vim_polr1_time0
                        V_cplx_in0_polr0_time1 = Vre_polr0_time1
                        V_cplx_in1_polr0_time1 = Vim_polr0_time1
                        V_cplx_in0_polr1_time1 = Vre_polr1_time1
                        V_cplx_in1_polr1_time1 = Vim_polr1_time1
                        V_cplx_in0_polr0_time2 = Vre_polr0_time2
                        V_cplx_in1_polr0_time2 = Vim_polr0_time2
                        V_cplx_in0_polr1_time2 = Vre_polr1_time2
                        V_cplx_in1_polr1_time2 = Vim_polr1_time2
                        V_cplx_in0_polr0_time3 = Vre_polr0_time3
                        V_cplx_in1_polr0_time3 = Vim_polr0_time3
                        V_cplx_in0_polr1_time3 = Vre_polr1_time3
                        V_cplx_in1_polr1_time3 = Vim_polr1_time3
                        (Y_cplx0_polr0_time0, Y_cplx1_polr0_time0) = IndexSpaces.mma_m16n8k16(
                            (aΓ³_cplx0_cplx_in0, aΓ³_cplx1_cplx_in0, aΓ³_cplx0_cplx_in1, aΓ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr0_time0, V_cplx_in1_polr0_time0),
                            (Y_cplx0_polr0_time0, Y_cplx1_polr0_time0),
                        )
                        (Y_cplx0_polr1_time0, Y_cplx1_polr1_time0) = IndexSpaces.mma_m16n8k16(
                            (aΓ³_cplx0_cplx_in0, aΓ³_cplx1_cplx_in0, aΓ³_cplx0_cplx_in1, aΓ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr1_time0, V_cplx_in1_polr1_time0),
                            (Y_cplx0_polr1_time0, Y_cplx1_polr1_time0),
                        )
                        (Y_cplx0_polr0_time1, Y_cplx1_polr0_time1) = IndexSpaces.mma_m16n8k16(
                            (aΓ³_cplx0_cplx_in0, aΓ³_cplx1_cplx_in0, aΓ³_cplx0_cplx_in1, aΓ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr0_time1, V_cplx_in1_polr0_time1),
                            (Y_cplx0_polr0_time1, Y_cplx1_polr0_time1),
                        )
                        (Y_cplx0_polr1_time1, Y_cplx1_polr1_time1) = IndexSpaces.mma_m16n8k16(
                            (aΓ³_cplx0_cplx_in0, aΓ³_cplx1_cplx_in0, aΓ³_cplx0_cplx_in1, aΓ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr1_time1, V_cplx_in1_polr1_time1),
                            (Y_cplx0_polr1_time1, Y_cplx1_polr1_time1),
                        )
                        (Y_cplx0_polr0_time2, Y_cplx1_polr0_time2) = IndexSpaces.mma_m16n8k16(
                            (aΓ³_cplx0_cplx_in0, aΓ³_cplx1_cplx_in0, aΓ³_cplx0_cplx_in1, aΓ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr0_time2, V_cplx_in1_polr0_time2),
                            (Y_cplx0_polr0_time2, Y_cplx1_polr0_time2),
                        )
                        (Y_cplx0_polr1_time2, Y_cplx1_polr1_time2) = IndexSpaces.mma_m16n8k16(
                            (aΓ³_cplx0_cplx_in0, aΓ³_cplx1_cplx_in0, aΓ³_cplx0_cplx_in1, aΓ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr1_time2, V_cplx_in1_polr1_time2),
                            (Y_cplx0_polr1_time2, Y_cplx1_polr1_time2),
                        )
                        (Y_cplx0_polr0_time3, Y_cplx1_polr0_time3) = IndexSpaces.mma_m16n8k16(
                            (aΓ³_cplx0_cplx_in0, aΓ³_cplx1_cplx_in0, aΓ³_cplx0_cplx_in1, aΓ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr0_time3, V_cplx_in1_polr0_time3),
                            (Y_cplx0_polr0_time3, Y_cplx1_polr0_time3),
                        )
                        (Y_cplx0_polr1_time3, Y_cplx1_polr1_time3) = IndexSpaces.mma_m16n8k16(
                            (aΓ³_cplx0_cplx_in0, aΓ³_cplx1_cplx_in0, aΓ³_cplx0_cplx_in1, aΓ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr1_time3, V_cplx_in1_polr1_time3),
                            (Y_cplx0_polr1_time3, Y_cplx1_polr1_time3),
                        )
                        G_cplx0_polr0_time0 = Y_cplx0_polr0_time0
                        G_cplx1_polr0_time0 = Y_cplx1_polr0_time0
                        G_cplx0_polr1_time0 = Y_cplx0_polr1_time0
                        G_cplx1_polr1_time0 = Y_cplx1_polr1_time0
                        G_cplx0_polr0_time1 = Y_cplx0_polr0_time1
                        G_cplx1_polr0_time1 = Y_cplx1_polr0_time1
                        G_cplx0_polr1_time1 = Y_cplx0_polr1_time1
                        G_cplx1_polr1_time1 = Y_cplx1_polr1_time1
                        G_cplx0_polr0_time2 = Y_cplx0_polr0_time2
                        G_cplx1_polr0_time2 = Y_cplx1_polr0_time2
                        G_cplx0_polr1_time2 = Y_cplx0_polr1_time2
                        G_cplx1_polr1_time2 = Y_cplx1_polr1_time2
                        G_cplx0_polr0_time3 = Y_cplx0_polr0_time3
                        G_cplx1_polr0_time3 = Y_cplx1_polr0_time3
                        G_cplx0_polr1_time3 = Y_cplx0_polr1_time3
                        G_cplx1_polr1_time3 = Y_cplx1_polr1_time3
                        (G_cplx0_polr0_time0, G_cplx1_polr0_time0) = (
                            IndexSpaces.get_lo16(G_cplx0_polr0_time0, G_cplx1_polr0_time0),
                            IndexSpaces.get_hi16(G_cplx0_polr0_time0, G_cplx1_polr0_time0),
                        )
                        (G_cplx0_polr1_time0, G_cplx1_polr1_time0) = (
                            IndexSpaces.get_lo16(G_cplx0_polr1_time0, G_cplx1_polr1_time0),
                            IndexSpaces.get_hi16(G_cplx0_polr1_time0, G_cplx1_polr1_time0),
                        )
                        (G_cplx0_polr0_time1, G_cplx1_polr0_time1) = (
                            IndexSpaces.get_lo16(G_cplx0_polr0_time1, G_cplx1_polr0_time1),
                            IndexSpaces.get_hi16(G_cplx0_polr0_time1, G_cplx1_polr0_time1),
                        )
                        (G_cplx0_polr1_time1, G_cplx1_polr1_time1) = (
                            IndexSpaces.get_lo16(G_cplx0_polr1_time1, G_cplx1_polr1_time1),
                            IndexSpaces.get_hi16(G_cplx0_polr1_time1, G_cplx1_polr1_time1),
                        )
                        (G_cplx0_polr0_time2, G_cplx1_polr0_time2) = (
                            IndexSpaces.get_lo16(G_cplx0_polr0_time2, G_cplx1_polr0_time2),
                            IndexSpaces.get_hi16(G_cplx0_polr0_time2, G_cplx1_polr0_time2),
                        )
                        (G_cplx0_polr1_time2, G_cplx1_polr1_time2) = (
                            IndexSpaces.get_lo16(G_cplx0_polr1_time2, G_cplx1_polr1_time2),
                            IndexSpaces.get_hi16(G_cplx0_polr1_time2, G_cplx1_polr1_time2),
                        )
                        (G_cplx0_polr0_time3, G_cplx1_polr0_time3) = (
                            IndexSpaces.get_lo16(G_cplx0_polr0_time3, G_cplx1_polr0_time3),
                            IndexSpaces.get_hi16(G_cplx0_polr0_time3, G_cplx1_polr0_time3),
                        )
                        (G_cplx0_polr1_time3, G_cplx1_polr1_time3) = (
                            IndexSpaces.get_lo16(G_cplx0_polr1_time3, G_cplx1_polr1_time3),
                            IndexSpaces.get_hi16(G_cplx0_polr1_time3, G_cplx1_polr1_time3),
                        )
                        IndexSpaces.cuda_sync_threads()
                        let
                            t = 0
                            G_beamQ0_polr0 = zero(Float16x2)
                            G_beamQ24_polr0 = zero(Float16x2)
                            G_beamQ0_polr1 = zero(Float16x2)
                            G_beamQ24_polr1 = zero(Float16x2)
                            if let
                                thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                mlo2_offset = 8
                                mlo2_length = 4
                                mlo = (1i32) * ((thread ÷ (4i32)) % (2i32)) + (2i32) * ((thread ÷ mlo2_offset) % mlo2_length)
                                mlo < 6
                            end
                                G_beamQ0_polr0 = Gsh_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 0 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ24_polr0 = Gsh_shared[((((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 0 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ0_polr1 = Gsh_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ24_polr1 = Gsh_shared[((((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 32 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                            end
                            X_beamQ0_polr0 = G_beamQ0_polr0
                            X_beamQ24_polr0 = G_beamQ24_polr0
                            X_beamQ0_polr1 = G_beamQ0_polr1
                            X_beamQ24_polr1 = G_beamQ24_polr1
                            Z_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ24_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ24_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ24_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Z_beamQ24_cplx1_polr1 = zero(Float16x2)
                            (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ0_polr0, (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0)
                            )
                            (Z_beamQ24_cplx0_polr0, Z_beamQ24_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ24_polr0, (Z_beamQ24_cplx0_polr0, Z_beamQ24_cplx1_polr0)
                            )
                            (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ0_polr1, (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1)
                            )
                            (Z_beamQ24_cplx0_polr1, Z_beamQ24_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ24_polr1, (Z_beamQ24_cplx0_polr1, Z_beamQ24_cplx1_polr1)
                            )
                            bΓ²re = bΓ²_cplx0
                            bΓ²im = bΓ²_cplx1
                            Zre_beamQ0_polr0 = Z_beamQ0_cplx0_polr0
                            Zim_beamQ0_polr0 = Z_beamQ0_cplx1_polr0
                            Zre_beamQ24_polr0 = Z_beamQ24_cplx0_polr0
                            Zim_beamQ24_polr0 = Z_beamQ24_cplx1_polr0
                            Zre_beamQ0_polr1 = Z_beamQ0_cplx0_polr1
                            Zim_beamQ0_polr1 = Z_beamQ0_cplx1_polr1
                            Zre_beamQ24_polr1 = Z_beamQ24_cplx0_polr1
                            Zim_beamQ24_polr1 = Z_beamQ24_cplx1_polr1
                            Vre_beamQ0_polr0 = muladd(bΓ²re, Zre_beamQ0_polr0, -bΓ²im * Zim_beamQ0_polr0)
                            Vre_beamQ24_polr0 = muladd(bΓ²re, Zre_beamQ24_polr0, -bΓ²im * Zim_beamQ24_polr0)
                            Vre_beamQ0_polr1 = muladd(bΓ²re, Zre_beamQ0_polr1, -bΓ²im * Zim_beamQ0_polr1)
                            Vre_beamQ24_polr1 = muladd(bΓ²re, Zre_beamQ24_polr1, -bΓ²im * Zim_beamQ24_polr1)
                            Vim_beamQ0_polr0 = muladd(bΓ²re, Zim_beamQ0_polr0, +bΓ²im * Zre_beamQ0_polr0)
                            Vim_beamQ24_polr0 = muladd(bΓ²re, Zim_beamQ24_polr0, +bΓ²im * Zre_beamQ24_polr0)
                            Vim_beamQ0_polr1 = muladd(bΓ²re, Zim_beamQ0_polr1, +bΓ²im * Zre_beamQ0_polr1)
                            Vim_beamQ24_polr1 = muladd(bΓ²re, Zim_beamQ24_polr1, +bΓ²im * Zre_beamQ24_polr1)
                            V_beamQ0_cplx0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx1_polr0 = Vim_beamQ0_polr0
                            V_beamQ24_cplx0_polr0 = Vre_beamQ24_polr0
                            V_beamQ24_cplx1_polr0 = Vim_beamQ24_polr0
                            V_beamQ0_cplx0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx1_polr1 = Vim_beamQ0_polr1
                            V_beamQ24_cplx0_polr1 = Vre_beamQ24_polr1
                            V_beamQ24_cplx1_polr1 = Vim_beamQ24_polr1
                            Y_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ24_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ24_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ24_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Y_beamQ24_cplx1_polr1 = zero(Float16x2)
                            Vre_beamQ0_polr0 = V_beamQ0_cplx0_polr0
                            Vim_beamQ0_polr0 = V_beamQ0_cplx1_polr0
                            Vre_beamQ24_polr0 = V_beamQ24_cplx0_polr0
                            Vim_beamQ24_polr0 = V_beamQ24_cplx1_polr0
                            Vre_beamQ0_polr1 = V_beamQ0_cplx0_polr1
                            Vim_beamQ0_polr1 = V_beamQ0_cplx1_polr1
                            Vre_beamQ24_polr1 = V_beamQ24_cplx0_polr1
                            Vim_beamQ24_polr1 = V_beamQ24_cplx1_polr1
                            V_beamQ0_cplx_in0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx_in1_polr0 = Vim_beamQ0_polr0
                            V_beamQ24_cplx_in0_polr0 = Vre_beamQ24_polr0
                            V_beamQ24_cplx_in1_polr0 = Vim_beamQ24_polr0
                            V_beamQ0_cplx_in0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx_in1_polr1 = Vim_beamQ0_polr1
                            V_beamQ24_cplx_in0_polr1 = Vre_beamQ24_polr1
                            V_beamQ24_cplx_in1_polr1 = Vim_beamQ24_polr1
                            (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr0, V_beamQ0_cplx_in1_polr0),
                                (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0),
                            )
                            (Y_beamQ24_cplx0_polr0, Y_beamQ24_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ24_cplx_in0_polr0, V_beamQ24_cplx_in1_polr0),
                                (Y_beamQ24_cplx0_polr0, Y_beamQ24_cplx1_polr0),
                            )
                            (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr1, V_beamQ0_cplx_in1_polr1),
                                (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1),
                            )
                            (Y_beamQ24_cplx0_polr1, Y_beamQ24_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ24_cplx_in0_polr1, V_beamQ24_cplx_in1_polr1),
                                (Y_beamQ24_cplx0_polr1, Y_beamQ24_cplx1_polr1),
                            )
                            Ẽ_beamQ0_cplx0_polr0 = Y_beamQ0_cplx0_polr0
                            Ẽ_beamQ24_cplx0_polr0 = Y_beamQ24_cplx0_polr0
                            Ẽ_beamQ0_cplx1_polr0 = Y_beamQ0_cplx1_polr0
                            Ẽ_beamQ24_cplx1_polr0 = Y_beamQ24_cplx1_polr0
                            Ẽ_beamQ0_cplx0_polr1 = Y_beamQ0_cplx0_polr1
                            Ẽ_beamQ24_cplx0_polr1 = Y_beamQ24_cplx0_polr1
                            Ẽ_beamQ0_cplx1_polr1 = Y_beamQ0_cplx1_polr1
                            Ẽ_beamQ24_cplx1_polr1 = Y_beamQ24_cplx1_polr1
                            Ẽp0_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr0
                            Ẽp1_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr1
                            Ẽp0_beamQ24_cplx0 = Ẽ_beamQ24_cplx0_polr0
                            Ẽp1_beamQ24_cplx0 = Ẽ_beamQ24_cplx0_polr1
                            Ẽp0_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr0
                            Ẽp1_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr1
                            Ẽp0_beamQ24_cplx1 = Ẽ_beamQ24_cplx1_polr0
                            Ẽp1_beamQ24_cplx1 = Ẽ_beamQ24_cplx1_polr1
                            Ẽp0re_beamQ0 = Ẽp0_beamQ0_cplx0
                            Ẽp0im_beamQ0 = Ẽp0_beamQ0_cplx1
                            Ẽp0re_beamQ24 = Ẽp0_beamQ24_cplx0
                            Ẽp0im_beamQ24 = Ẽp0_beamQ24_cplx1
                            Ẽp1re_beamQ0 = Ẽp1_beamQ0_cplx0
                            Ẽp1im_beamQ0 = Ẽp1_beamQ0_cplx1
                            Ẽp1re_beamQ24 = Ẽp1_beamQ24_cplx0
                            Ẽp1im_beamQ24 = Ẽp1_beamQ24_cplx1
                            I_beamQ0 = muladd(
                                Float16x2(0.003124237f0, 0.003124237f0),
                                muladd(
                                    Ẽp1im_beamQ0,
                                    Ẽp1im_beamQ0,
                                    muladd(
                                        Ẽp1re_beamQ0, Ẽp1re_beamQ0, muladd(Ẽp0im_beamQ0, Ẽp0im_beamQ0, Ẽp0re_beamQ0 * Ẽp0re_beamQ0)
                                    ),
                                ),
                                I_beamQ0,
                            )
                            I_beamQ24 = muladd(
                                Float16x2(0.003124237f0, 0.003124237f0),
                                muladd(
                                    Ẽp1im_beamQ24,
                                    Ẽp1im_beamQ24,
                                    muladd(
                                        Ẽp1re_beamQ24,
                                        Ẽp1re_beamQ24,
                                        muladd(Ẽp0im_beamQ24, Ẽp0im_beamQ24, Ẽp0re_beamQ24 * Ẽp0re_beamQ24),
                                    ),
                                ),
                                I_beamQ24,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 4 == 0i32
                                if t_running == 40
                                    if let
                                        thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                        warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24)
                                        p = (2i32) * thread
                                        q = (2i32) * warp
                                        0i32 ≤ p < 48 && 0i32 ≤ q < 48
                                    end
                                        if true
                                            I_memory[let
                                                offset = 7077888 * T̄min + 1152 * F̄min
                                                length = 452984832
                                                mod(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 1152 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 64) % 64) % 64) * 7077888 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 48) * 24) + 0) + offset, length)
                                            end + 0x01] = I_beamQ0
                                        end
                                        if true
                                            I_memory[let
                                                offset = 7077888 * T̄min + 1152 * F̄min
                                                length = 452984832
                                                mod(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 1152 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 64) % 64) % 64) * 7077888 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 24) % 48) * 24) + 0) + offset, length)
                                            end + 0x01] = I_beamQ24
                                        end
                                    end
                                    I_beamQ0 = zero(Float16x2)
                                    I_beamQ24 = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let
                            t = 1
                            G_beamQ0_polr0 = zero(Float16x2)
                            G_beamQ24_polr0 = zero(Float16x2)
                            G_beamQ0_polr1 = zero(Float16x2)
                            G_beamQ24_polr1 = zero(Float16x2)
                            if let
                                thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                mlo2_offset = 8
                                mlo2_length = 4
                                mlo = (1i32) * ((thread ÷ (4i32)) % (2i32)) + (2i32) * ((thread ÷ mlo2_offset) % mlo2_length)
                                mlo < 6
                            end
                                G_beamQ0_polr0 = Gsh_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 0 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ24_polr0 = Gsh_shared[((((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 0 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ0_polr1 = Gsh_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ24_polr1 = Gsh_shared[((((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 32 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                            end
                            X_beamQ0_polr0 = G_beamQ0_polr0
                            X_beamQ24_polr0 = G_beamQ24_polr0
                            X_beamQ0_polr1 = G_beamQ0_polr1
                            X_beamQ24_polr1 = G_beamQ24_polr1
                            Z_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ24_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ24_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ24_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Z_beamQ24_cplx1_polr1 = zero(Float16x2)
                            (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ0_polr0, (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0)
                            )
                            (Z_beamQ24_cplx0_polr0, Z_beamQ24_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ24_polr0, (Z_beamQ24_cplx0_polr0, Z_beamQ24_cplx1_polr0)
                            )
                            (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ0_polr1, (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1)
                            )
                            (Z_beamQ24_cplx0_polr1, Z_beamQ24_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ24_polr1, (Z_beamQ24_cplx0_polr1, Z_beamQ24_cplx1_polr1)
                            )
                            bΓ²re = bΓ²_cplx0
                            bΓ²im = bΓ²_cplx1
                            Zre_beamQ0_polr0 = Z_beamQ0_cplx0_polr0
                            Zim_beamQ0_polr0 = Z_beamQ0_cplx1_polr0
                            Zre_beamQ24_polr0 = Z_beamQ24_cplx0_polr0
                            Zim_beamQ24_polr0 = Z_beamQ24_cplx1_polr0
                            Zre_beamQ0_polr1 = Z_beamQ0_cplx0_polr1
                            Zim_beamQ0_polr1 = Z_beamQ0_cplx1_polr1
                            Zre_beamQ24_polr1 = Z_beamQ24_cplx0_polr1
                            Zim_beamQ24_polr1 = Z_beamQ24_cplx1_polr1
                            Vre_beamQ0_polr0 = muladd(bΓ²re, Zre_beamQ0_polr0, -bΓ²im * Zim_beamQ0_polr0)
                            Vre_beamQ24_polr0 = muladd(bΓ²re, Zre_beamQ24_polr0, -bΓ²im * Zim_beamQ24_polr0)
                            Vre_beamQ0_polr1 = muladd(bΓ²re, Zre_beamQ0_polr1, -bΓ²im * Zim_beamQ0_polr1)
                            Vre_beamQ24_polr1 = muladd(bΓ²re, Zre_beamQ24_polr1, -bΓ²im * Zim_beamQ24_polr1)
                            Vim_beamQ0_polr0 = muladd(bΓ²re, Zim_beamQ0_polr0, +bΓ²im * Zre_beamQ0_polr0)
                            Vim_beamQ24_polr0 = muladd(bΓ²re, Zim_beamQ24_polr0, +bΓ²im * Zre_beamQ24_polr0)
                            Vim_beamQ0_polr1 = muladd(bΓ²re, Zim_beamQ0_polr1, +bΓ²im * Zre_beamQ0_polr1)
                            Vim_beamQ24_polr1 = muladd(bΓ²re, Zim_beamQ24_polr1, +bΓ²im * Zre_beamQ24_polr1)
                            V_beamQ0_cplx0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx1_polr0 = Vim_beamQ0_polr0
                            V_beamQ24_cplx0_polr0 = Vre_beamQ24_polr0
                            V_beamQ24_cplx1_polr0 = Vim_beamQ24_polr0
                            V_beamQ0_cplx0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx1_polr1 = Vim_beamQ0_polr1
                            V_beamQ24_cplx0_polr1 = Vre_beamQ24_polr1
                            V_beamQ24_cplx1_polr1 = Vim_beamQ24_polr1
                            Y_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ24_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ24_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ24_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Y_beamQ24_cplx1_polr1 = zero(Float16x2)
                            Vre_beamQ0_polr0 = V_beamQ0_cplx0_polr0
                            Vim_beamQ0_polr0 = V_beamQ0_cplx1_polr0
                            Vre_beamQ24_polr0 = V_beamQ24_cplx0_polr0
                            Vim_beamQ24_polr0 = V_beamQ24_cplx1_polr0
                            Vre_beamQ0_polr1 = V_beamQ0_cplx0_polr1
                            Vim_beamQ0_polr1 = V_beamQ0_cplx1_polr1
                            Vre_beamQ24_polr1 = V_beamQ24_cplx0_polr1
                            Vim_beamQ24_polr1 = V_beamQ24_cplx1_polr1
                            V_beamQ0_cplx_in0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx_in1_polr0 = Vim_beamQ0_polr0
                            V_beamQ24_cplx_in0_polr0 = Vre_beamQ24_polr0
                            V_beamQ24_cplx_in1_polr0 = Vim_beamQ24_polr0
                            V_beamQ0_cplx_in0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx_in1_polr1 = Vim_beamQ0_polr1
                            V_beamQ24_cplx_in0_polr1 = Vre_beamQ24_polr1
                            V_beamQ24_cplx_in1_polr1 = Vim_beamQ24_polr1
                            (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr0, V_beamQ0_cplx_in1_polr0),
                                (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0),
                            )
                            (Y_beamQ24_cplx0_polr0, Y_beamQ24_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ24_cplx_in0_polr0, V_beamQ24_cplx_in1_polr0),
                                (Y_beamQ24_cplx0_polr0, Y_beamQ24_cplx1_polr0),
                            )
                            (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr1, V_beamQ0_cplx_in1_polr1),
                                (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1),
                            )
                            (Y_beamQ24_cplx0_polr1, Y_beamQ24_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ24_cplx_in0_polr1, V_beamQ24_cplx_in1_polr1),
                                (Y_beamQ24_cplx0_polr1, Y_beamQ24_cplx1_polr1),
                            )
                            Ẽ_beamQ0_cplx0_polr0 = Y_beamQ0_cplx0_polr0
                            Ẽ_beamQ24_cplx0_polr0 = Y_beamQ24_cplx0_polr0
                            Ẽ_beamQ0_cplx1_polr0 = Y_beamQ0_cplx1_polr0
                            Ẽ_beamQ24_cplx1_polr0 = Y_beamQ24_cplx1_polr0
                            Ẽ_beamQ0_cplx0_polr1 = Y_beamQ0_cplx0_polr1
                            Ẽ_beamQ24_cplx0_polr1 = Y_beamQ24_cplx0_polr1
                            Ẽ_beamQ0_cplx1_polr1 = Y_beamQ0_cplx1_polr1
                            Ẽ_beamQ24_cplx1_polr1 = Y_beamQ24_cplx1_polr1
                            Ẽp0_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr0
                            Ẽp1_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr1
                            Ẽp0_beamQ24_cplx0 = Ẽ_beamQ24_cplx0_polr0
                            Ẽp1_beamQ24_cplx0 = Ẽ_beamQ24_cplx0_polr1
                            Ẽp0_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr0
                            Ẽp1_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr1
                            Ẽp0_beamQ24_cplx1 = Ẽ_beamQ24_cplx1_polr0
                            Ẽp1_beamQ24_cplx1 = Ẽ_beamQ24_cplx1_polr1
                            Ẽp0re_beamQ0 = Ẽp0_beamQ0_cplx0
                            Ẽp0im_beamQ0 = Ẽp0_beamQ0_cplx1
                            Ẽp0re_beamQ24 = Ẽp0_beamQ24_cplx0
                            Ẽp0im_beamQ24 = Ẽp0_beamQ24_cplx1
                            Ẽp1re_beamQ0 = Ẽp1_beamQ0_cplx0
                            Ẽp1im_beamQ0 = Ẽp1_beamQ0_cplx1
                            Ẽp1re_beamQ24 = Ẽp1_beamQ24_cplx0
                            Ẽp1im_beamQ24 = Ẽp1_beamQ24_cplx1
                            I_beamQ0 = muladd(
                                Float16x2(0.003124237f0, 0.003124237f0),
                                muladd(
                                    Ẽp1im_beamQ0,
                                    Ẽp1im_beamQ0,
                                    muladd(
                                        Ẽp1re_beamQ0, Ẽp1re_beamQ0, muladd(Ẽp0im_beamQ0, Ẽp0im_beamQ0, Ẽp0re_beamQ0 * Ẽp0re_beamQ0)
                                    ),
                                ),
                                I_beamQ0,
                            )
                            I_beamQ24 = muladd(
                                Float16x2(0.003124237f0, 0.003124237f0),
                                muladd(
                                    Ẽp1im_beamQ24,
                                    Ẽp1im_beamQ24,
                                    muladd(
                                        Ẽp1re_beamQ24,
                                        Ẽp1re_beamQ24,
                                        muladd(Ẽp0im_beamQ24, Ẽp0im_beamQ24, Ẽp0re_beamQ24 * Ẽp0re_beamQ24),
                                    ),
                                ),
                                I_beamQ24,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 4 == 0i32
                                if t_running == 40
                                    if let
                                        thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                        warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24)
                                        p = (2i32) * thread
                                        q = (2i32) * warp
                                        0i32 ≤ p < 48 && 0i32 ≤ q < 48
                                    end
                                        if true
                                            I_memory[let
                                                offset = 7077888 * T̄min + 1152 * F̄min
                                                length = 452984832
                                                mod(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 1152 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 64) % 64) % 64) * 7077888 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 48) * 24) + 0) + offset, length)
                                            end + 0x01] = I_beamQ0
                                        end
                                        if true
                                            I_memory[let
                                                offset = 7077888 * T̄min + 1152 * F̄min
                                                length = 452984832
                                                mod(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 1152 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 64) % 64) % 64) * 7077888 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 24) % 48) * 24) + 0) + offset, length)
                                            end + 0x01] = I_beamQ24
                                        end
                                    end
                                    I_beamQ0 = zero(Float16x2)
                                    I_beamQ24 = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let
                            t = 2
                            G_beamQ0_polr0 = zero(Float16x2)
                            G_beamQ24_polr0 = zero(Float16x2)
                            G_beamQ0_polr1 = zero(Float16x2)
                            G_beamQ24_polr1 = zero(Float16x2)
                            if let
                                thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                mlo2_offset = 8
                                mlo2_length = 4
                                mlo = (1i32) * ((thread ÷ (4i32)) % (2i32)) + (2i32) * ((thread ÷ mlo2_offset) % mlo2_length)
                                mlo < 6
                            end
                                G_beamQ0_polr0 = Gsh_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 0 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ24_polr0 = Gsh_shared[((((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 0 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ0_polr1 = Gsh_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ24_polr1 = Gsh_shared[((((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 32 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                            end
                            X_beamQ0_polr0 = G_beamQ0_polr0
                            X_beamQ24_polr0 = G_beamQ24_polr0
                            X_beamQ0_polr1 = G_beamQ0_polr1
                            X_beamQ24_polr1 = G_beamQ24_polr1
                            Z_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ24_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ24_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ24_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Z_beamQ24_cplx1_polr1 = zero(Float16x2)
                            (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ0_polr0, (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0)
                            )
                            (Z_beamQ24_cplx0_polr0, Z_beamQ24_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ24_polr0, (Z_beamQ24_cplx0_polr0, Z_beamQ24_cplx1_polr0)
                            )
                            (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ0_polr1, (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1)
                            )
                            (Z_beamQ24_cplx0_polr1, Z_beamQ24_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ24_polr1, (Z_beamQ24_cplx0_polr1, Z_beamQ24_cplx1_polr1)
                            )
                            bΓ²re = bΓ²_cplx0
                            bΓ²im = bΓ²_cplx1
                            Zre_beamQ0_polr0 = Z_beamQ0_cplx0_polr0
                            Zim_beamQ0_polr0 = Z_beamQ0_cplx1_polr0
                            Zre_beamQ24_polr0 = Z_beamQ24_cplx0_polr0
                            Zim_beamQ24_polr0 = Z_beamQ24_cplx1_polr0
                            Zre_beamQ0_polr1 = Z_beamQ0_cplx0_polr1
                            Zim_beamQ0_polr1 = Z_beamQ0_cplx1_polr1
                            Zre_beamQ24_polr1 = Z_beamQ24_cplx0_polr1
                            Zim_beamQ24_polr1 = Z_beamQ24_cplx1_polr1
                            Vre_beamQ0_polr0 = muladd(bΓ²re, Zre_beamQ0_polr0, -bΓ²im * Zim_beamQ0_polr0)
                            Vre_beamQ24_polr0 = muladd(bΓ²re, Zre_beamQ24_polr0, -bΓ²im * Zim_beamQ24_polr0)
                            Vre_beamQ0_polr1 = muladd(bΓ²re, Zre_beamQ0_polr1, -bΓ²im * Zim_beamQ0_polr1)
                            Vre_beamQ24_polr1 = muladd(bΓ²re, Zre_beamQ24_polr1, -bΓ²im * Zim_beamQ24_polr1)
                            Vim_beamQ0_polr0 = muladd(bΓ²re, Zim_beamQ0_polr0, +bΓ²im * Zre_beamQ0_polr0)
                            Vim_beamQ24_polr0 = muladd(bΓ²re, Zim_beamQ24_polr0, +bΓ²im * Zre_beamQ24_polr0)
                            Vim_beamQ0_polr1 = muladd(bΓ²re, Zim_beamQ0_polr1, +bΓ²im * Zre_beamQ0_polr1)
                            Vim_beamQ24_polr1 = muladd(bΓ²re, Zim_beamQ24_polr1, +bΓ²im * Zre_beamQ24_polr1)
                            V_beamQ0_cplx0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx1_polr0 = Vim_beamQ0_polr0
                            V_beamQ24_cplx0_polr0 = Vre_beamQ24_polr0
                            V_beamQ24_cplx1_polr0 = Vim_beamQ24_polr0
                            V_beamQ0_cplx0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx1_polr1 = Vim_beamQ0_polr1
                            V_beamQ24_cplx0_polr1 = Vre_beamQ24_polr1
                            V_beamQ24_cplx1_polr1 = Vim_beamQ24_polr1
                            Y_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ24_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ24_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ24_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Y_beamQ24_cplx1_polr1 = zero(Float16x2)
                            Vre_beamQ0_polr0 = V_beamQ0_cplx0_polr0
                            Vim_beamQ0_polr0 = V_beamQ0_cplx1_polr0
                            Vre_beamQ24_polr0 = V_beamQ24_cplx0_polr0
                            Vim_beamQ24_polr0 = V_beamQ24_cplx1_polr0
                            Vre_beamQ0_polr1 = V_beamQ0_cplx0_polr1
                            Vim_beamQ0_polr1 = V_beamQ0_cplx1_polr1
                            Vre_beamQ24_polr1 = V_beamQ24_cplx0_polr1
                            Vim_beamQ24_polr1 = V_beamQ24_cplx1_polr1
                            V_beamQ0_cplx_in0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx_in1_polr0 = Vim_beamQ0_polr0
                            V_beamQ24_cplx_in0_polr0 = Vre_beamQ24_polr0
                            V_beamQ24_cplx_in1_polr0 = Vim_beamQ24_polr0
                            V_beamQ0_cplx_in0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx_in1_polr1 = Vim_beamQ0_polr1
                            V_beamQ24_cplx_in0_polr1 = Vre_beamQ24_polr1
                            V_beamQ24_cplx_in1_polr1 = Vim_beamQ24_polr1
                            (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr0, V_beamQ0_cplx_in1_polr0),
                                (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0),
                            )
                            (Y_beamQ24_cplx0_polr0, Y_beamQ24_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ24_cplx_in0_polr0, V_beamQ24_cplx_in1_polr0),
                                (Y_beamQ24_cplx0_polr0, Y_beamQ24_cplx1_polr0),
                            )
                            (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr1, V_beamQ0_cplx_in1_polr1),
                                (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1),
                            )
                            (Y_beamQ24_cplx0_polr1, Y_beamQ24_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ24_cplx_in0_polr1, V_beamQ24_cplx_in1_polr1),
                                (Y_beamQ24_cplx0_polr1, Y_beamQ24_cplx1_polr1),
                            )
                            Ẽ_beamQ0_cplx0_polr0 = Y_beamQ0_cplx0_polr0
                            Ẽ_beamQ24_cplx0_polr0 = Y_beamQ24_cplx0_polr0
                            Ẽ_beamQ0_cplx1_polr0 = Y_beamQ0_cplx1_polr0
                            Ẽ_beamQ24_cplx1_polr0 = Y_beamQ24_cplx1_polr0
                            Ẽ_beamQ0_cplx0_polr1 = Y_beamQ0_cplx0_polr1
                            Ẽ_beamQ24_cplx0_polr1 = Y_beamQ24_cplx0_polr1
                            Ẽ_beamQ0_cplx1_polr1 = Y_beamQ0_cplx1_polr1
                            Ẽ_beamQ24_cplx1_polr1 = Y_beamQ24_cplx1_polr1
                            Ẽp0_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr0
                            Ẽp1_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr1
                            Ẽp0_beamQ24_cplx0 = Ẽ_beamQ24_cplx0_polr0
                            Ẽp1_beamQ24_cplx0 = Ẽ_beamQ24_cplx0_polr1
                            Ẽp0_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr0
                            Ẽp1_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr1
                            Ẽp0_beamQ24_cplx1 = Ẽ_beamQ24_cplx1_polr0
                            Ẽp1_beamQ24_cplx1 = Ẽ_beamQ24_cplx1_polr1
                            Ẽp0re_beamQ0 = Ẽp0_beamQ0_cplx0
                            Ẽp0im_beamQ0 = Ẽp0_beamQ0_cplx1
                            Ẽp0re_beamQ24 = Ẽp0_beamQ24_cplx0
                            Ẽp0im_beamQ24 = Ẽp0_beamQ24_cplx1
                            Ẽp1re_beamQ0 = Ẽp1_beamQ0_cplx0
                            Ẽp1im_beamQ0 = Ẽp1_beamQ0_cplx1
                            Ẽp1re_beamQ24 = Ẽp1_beamQ24_cplx0
                            Ẽp1im_beamQ24 = Ẽp1_beamQ24_cplx1
                            I_beamQ0 = muladd(
                                Float16x2(0.003124237f0, 0.003124237f0),
                                muladd(
                                    Ẽp1im_beamQ0,
                                    Ẽp1im_beamQ0,
                                    muladd(
                                        Ẽp1re_beamQ0, Ẽp1re_beamQ0, muladd(Ẽp0im_beamQ0, Ẽp0im_beamQ0, Ẽp0re_beamQ0 * Ẽp0re_beamQ0)
                                    ),
                                ),
                                I_beamQ0,
                            )
                            I_beamQ24 = muladd(
                                Float16x2(0.003124237f0, 0.003124237f0),
                                muladd(
                                    Ẽp1im_beamQ24,
                                    Ẽp1im_beamQ24,
                                    muladd(
                                        Ẽp1re_beamQ24,
                                        Ẽp1re_beamQ24,
                                        muladd(Ẽp0im_beamQ24, Ẽp0im_beamQ24, Ẽp0re_beamQ24 * Ẽp0re_beamQ24),
                                    ),
                                ),
                                I_beamQ24,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 4 == 0i32
                                if t_running == 40
                                    if let
                                        thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                        warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24)
                                        p = (2i32) * thread
                                        q = (2i32) * warp
                                        0i32 ≤ p < 48 && 0i32 ≤ q < 48
                                    end
                                        if true
                                            I_memory[let
                                                offset = 7077888 * T̄min + 1152 * F̄min
                                                length = 452984832
                                                mod(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 1152 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 64) % 64) % 64) * 7077888 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 48) * 24) + 0) + offset, length)
                                            end + 0x01] = I_beamQ0
                                        end
                                        if true
                                            I_memory[let
                                                offset = 7077888 * T̄min + 1152 * F̄min
                                                length = 452984832
                                                mod(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 1152 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 64) % 64) % 64) * 7077888 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 24) % 48) * 24) + 0) + offset, length)
                                            end + 0x01] = I_beamQ24
                                        end
                                    end
                                    I_beamQ0 = zero(Float16x2)
                                    I_beamQ24 = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let
                            t = 3
                            G_beamQ0_polr0 = zero(Float16x2)
                            G_beamQ24_polr0 = zero(Float16x2)
                            G_beamQ0_polr1 = zero(Float16x2)
                            G_beamQ24_polr1 = zero(Float16x2)
                            if let
                                thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                mlo2_offset = 8
                                mlo2_length = 4
                                mlo = (1i32) * ((thread ÷ (4i32)) % (2i32)) + (2i32) * ((thread ÷ mlo2_offset) % mlo2_length)
                                mlo < 6
                            end
                                G_beamQ0_polr0 = Gsh_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 0 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ24_polr0 = Gsh_shared[((((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 0 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ0_polr1 = Gsh_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ24_polr1 = Gsh_shared[((((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 2) % 2) * 4112 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 4) % 2) * 2056 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 2) * 1028 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((((((t_inner_hi ÷ 24) % 2) * 24 + t % 4) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 240) ÷ 48) % 5) * 48) % 4) * 64 + ((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 2) * 8256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 16) % 2) * 514 + 32 + (((24 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 32) % 2) * 257) + 0x01]
                            end
                            X_beamQ0_polr0 = G_beamQ0_polr0
                            X_beamQ24_polr0 = G_beamQ24_polr0
                            X_beamQ0_polr1 = G_beamQ0_polr1
                            X_beamQ24_polr1 = G_beamQ24_polr1
                            Z_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ24_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ24_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ24_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Z_beamQ24_cplx1_polr1 = zero(Float16x2)
                            (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ0_polr0, (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0)
                            )
                            (Z_beamQ24_cplx0_polr0, Z_beamQ24_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ24_polr0, (Z_beamQ24_cplx0_polr0, Z_beamQ24_cplx1_polr0)
                            )
                            (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ0_polr1, (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1)
                            )
                            (Z_beamQ24_cplx0_polr1, Z_beamQ24_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_beamQ24_polr1, (Z_beamQ24_cplx0_polr1, Z_beamQ24_cplx1_polr1)
                            )
                            bΓ²re = bΓ²_cplx0
                            bΓ²im = bΓ²_cplx1
                            Zre_beamQ0_polr0 = Z_beamQ0_cplx0_polr0
                            Zim_beamQ0_polr0 = Z_beamQ0_cplx1_polr0
                            Zre_beamQ24_polr0 = Z_beamQ24_cplx0_polr0
                            Zim_beamQ24_polr0 = Z_beamQ24_cplx1_polr0
                            Zre_beamQ0_polr1 = Z_beamQ0_cplx0_polr1
                            Zim_beamQ0_polr1 = Z_beamQ0_cplx1_polr1
                            Zre_beamQ24_polr1 = Z_beamQ24_cplx0_polr1
                            Zim_beamQ24_polr1 = Z_beamQ24_cplx1_polr1
                            Vre_beamQ0_polr0 = muladd(bΓ²re, Zre_beamQ0_polr0, -bΓ²im * Zim_beamQ0_polr0)
                            Vre_beamQ24_polr0 = muladd(bΓ²re, Zre_beamQ24_polr0, -bΓ²im * Zim_beamQ24_polr0)
                            Vre_beamQ0_polr1 = muladd(bΓ²re, Zre_beamQ0_polr1, -bΓ²im * Zim_beamQ0_polr1)
                            Vre_beamQ24_polr1 = muladd(bΓ²re, Zre_beamQ24_polr1, -bΓ²im * Zim_beamQ24_polr1)
                            Vim_beamQ0_polr0 = muladd(bΓ²re, Zim_beamQ0_polr0, +bΓ²im * Zre_beamQ0_polr0)
                            Vim_beamQ24_polr0 = muladd(bΓ²re, Zim_beamQ24_polr0, +bΓ²im * Zre_beamQ24_polr0)
                            Vim_beamQ0_polr1 = muladd(bΓ²re, Zim_beamQ0_polr1, +bΓ²im * Zre_beamQ0_polr1)
                            Vim_beamQ24_polr1 = muladd(bΓ²re, Zim_beamQ24_polr1, +bΓ²im * Zre_beamQ24_polr1)
                            V_beamQ0_cplx0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx1_polr0 = Vim_beamQ0_polr0
                            V_beamQ24_cplx0_polr0 = Vre_beamQ24_polr0
                            V_beamQ24_cplx1_polr0 = Vim_beamQ24_polr0
                            V_beamQ0_cplx0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx1_polr1 = Vim_beamQ0_polr1
                            V_beamQ24_cplx0_polr1 = Vre_beamQ24_polr1
                            V_beamQ24_cplx1_polr1 = Vim_beamQ24_polr1
                            Y_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ24_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ24_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ24_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Y_beamQ24_cplx1_polr1 = zero(Float16x2)
                            Vre_beamQ0_polr0 = V_beamQ0_cplx0_polr0
                            Vim_beamQ0_polr0 = V_beamQ0_cplx1_polr0
                            Vre_beamQ24_polr0 = V_beamQ24_cplx0_polr0
                            Vim_beamQ24_polr0 = V_beamQ24_cplx1_polr0
                            Vre_beamQ0_polr1 = V_beamQ0_cplx0_polr1
                            Vim_beamQ0_polr1 = V_beamQ0_cplx1_polr1
                            Vre_beamQ24_polr1 = V_beamQ24_cplx0_polr1
                            Vim_beamQ24_polr1 = V_beamQ24_cplx1_polr1
                            V_beamQ0_cplx_in0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx_in1_polr0 = Vim_beamQ0_polr0
                            V_beamQ24_cplx_in0_polr0 = Vre_beamQ24_polr0
                            V_beamQ24_cplx_in1_polr0 = Vim_beamQ24_polr0
                            V_beamQ0_cplx_in0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx_in1_polr1 = Vim_beamQ0_polr1
                            V_beamQ24_cplx_in0_polr1 = Vre_beamQ24_polr1
                            V_beamQ24_cplx_in1_polr1 = Vim_beamQ24_polr1
                            (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr0, V_beamQ0_cplx_in1_polr0),
                                (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0),
                            )
                            (Y_beamQ24_cplx0_polr0, Y_beamQ24_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ24_cplx_in0_polr0, V_beamQ24_cplx_in1_polr0),
                                (Y_beamQ24_cplx0_polr0, Y_beamQ24_cplx1_polr0),
                            )
                            (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr1, V_beamQ0_cplx_in1_polr1),
                                (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1),
                            )
                            (Y_beamQ24_cplx0_polr1, Y_beamQ24_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (bΓ³_cplx0_cplx_in0, bΓ³_cplx1_cplx_in0, bΓ³_cplx0_cplx_in1, bΓ³_cplx1_cplx_in1),
                                (V_beamQ24_cplx_in0_polr1, V_beamQ24_cplx_in1_polr1),
                                (Y_beamQ24_cplx0_polr1, Y_beamQ24_cplx1_polr1),
                            )
                            Ẽ_beamQ0_cplx0_polr0 = Y_beamQ0_cplx0_polr0
                            Ẽ_beamQ24_cplx0_polr0 = Y_beamQ24_cplx0_polr0
                            Ẽ_beamQ0_cplx1_polr0 = Y_beamQ0_cplx1_polr0
                            Ẽ_beamQ24_cplx1_polr0 = Y_beamQ24_cplx1_polr0
                            Ẽ_beamQ0_cplx0_polr1 = Y_beamQ0_cplx0_polr1
                            Ẽ_beamQ24_cplx0_polr1 = Y_beamQ24_cplx0_polr1
                            Ẽ_beamQ0_cplx1_polr1 = Y_beamQ0_cplx1_polr1
                            Ẽ_beamQ24_cplx1_polr1 = Y_beamQ24_cplx1_polr1
                            Ẽp0_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr0
                            Ẽp1_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr1
                            Ẽp0_beamQ24_cplx0 = Ẽ_beamQ24_cplx0_polr0
                            Ẽp1_beamQ24_cplx0 = Ẽ_beamQ24_cplx0_polr1
                            Ẽp0_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr0
                            Ẽp1_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr1
                            Ẽp0_beamQ24_cplx1 = Ẽ_beamQ24_cplx1_polr0
                            Ẽp1_beamQ24_cplx1 = Ẽ_beamQ24_cplx1_polr1
                            Ẽp0re_beamQ0 = Ẽp0_beamQ0_cplx0
                            Ẽp0im_beamQ0 = Ẽp0_beamQ0_cplx1
                            Ẽp0re_beamQ24 = Ẽp0_beamQ24_cplx0
                            Ẽp0im_beamQ24 = Ẽp0_beamQ24_cplx1
                            Ẽp1re_beamQ0 = Ẽp1_beamQ0_cplx0
                            Ẽp1im_beamQ0 = Ẽp1_beamQ0_cplx1
                            Ẽp1re_beamQ24 = Ẽp1_beamQ24_cplx0
                            Ẽp1im_beamQ24 = Ẽp1_beamQ24_cplx1
                            I_beamQ0 = muladd(
                                Float16x2(0.003124237f0, 0.003124237f0),
                                muladd(
                                    Ẽp1im_beamQ0,
                                    Ẽp1im_beamQ0,
                                    muladd(
                                        Ẽp1re_beamQ0, Ẽp1re_beamQ0, muladd(Ẽp0im_beamQ0, Ẽp0im_beamQ0, Ẽp0re_beamQ0 * Ẽp0re_beamQ0)
                                    ),
                                ),
                                I_beamQ0,
                            )
                            I_beamQ24 = muladd(
                                Float16x2(0.003124237f0, 0.003124237f0),
                                muladd(
                                    Ẽp1im_beamQ24,
                                    Ẽp1im_beamQ24,
                                    muladd(
                                        Ẽp1re_beamQ24,
                                        Ẽp1re_beamQ24,
                                        muladd(Ẽp0im_beamQ24, Ẽp0im_beamQ24, Ẽp0re_beamQ24 * Ẽp0re_beamQ24),
                                    ),
                                ),
                                I_beamQ24,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 4 == 0i32
                                if t_running == 40
                                    if let
                                        thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                        warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24)
                                        p = (2i32) * thread
                                        q = (2i32) * warp
                                        0i32 ≤ p < 48 && 0i32 ≤ q < 48
                                    end
                                        if true
                                            I_memory[let
                                                offset = 7077888 * T̄min + 1152 * F̄min
                                                length = 452984832
                                                mod(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 1152 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 64) % 64) % 64) * 7077888 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 48) * 24) + 0) + offset, length)
                                            end + 0x01] = I_beamQ0
                                        end
                                        if true
                                            I_memory[let
                                                offset = 7077888 * T̄min + 1152 * F̄min
                                                length = 452984832
                                                mod(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 1152 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 64) % 64) % 64) * 7077888 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 24) % 48) * 24) + 0) + offset, length)
                                            end + 0x01] = I_beamQ24
                                        end
                                    end
                                    I_beamQ0 = zero(Float16x2)
                                    I_beamQ24 = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        IndexSpaces.cuda_sync_threads()
                    end
                end
            end
        end
        info = 0
        if true
            info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 6144) % 6144) % 6144) * 768 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                info
        end
    end
)
