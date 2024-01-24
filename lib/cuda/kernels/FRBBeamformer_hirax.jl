# Julia source code for CUDA frb beamformer
# This file has been generated automatically by `frb.jl`.
# Do not modify this file, your changes will be lost.

@fastmath @inbounds(
    begin #= /home/eschnett/src/kotekan/julia/kernels/frb.jl:1705 =#
        info = 1
        if true
            info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) % 16) * 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 1024) % 1024) % 1024) * 512 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                info
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
            d0 = (thread % 2) * (2i32) + 0i32
            d1 = (thread % 2) * (2i32) + 1i32
            v = thread ÷ (4i32)
            δ0 = (δ1 = (Γ²_d0 = if d0 < 4
                cispi((((d0 * v) % 32) / 16.0f0) % 2.0f0)
            else
                Complex(0.0f0)
            end))
            Γ²_d1 = if d1 < 4
                cispi((((d1 * v) % 32) / 16.0f0) % 2.0f0)
            else
                Complex(0.0f0)
            end
            (Γ²_d0.re, Γ²_d0.im, Γ²_d1.re, Γ²_d1.im)
        end
        Γ²_re = Float16x2(Γ²_d0_re, Γ²_d1_re)
        Γ²_im = Float16x2(Γ²_d0_im, Γ²_d1_im)
        aΓ²_cplx0 = Γ²_re
        aΓ²_cplx1 = Γ²_im
        (Γ³_d0_re, Γ³_d0_im, Γ³_d1_re, Γ³_d1_im) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            d0 = (thread % (2i32)) * (2i32) + 0i32
            d1 = (thread % (2i32)) * (2i32) + 1i32
            u = (thread ÷ (4i32)) % (4i32)
            Γ³_d0 = if d0 < 4 && u < 4
                cispi((((d0 * u) % 4) / 2.0f0) % 2.0f0)
            else
                Complex(0.0f0)
            end
            Γ³_d1 = if d1 < 4 && u < 4
                cispi((((d1 * u) % 4) / 2.0f0) % 2.0f0)
            else
                Complex(0.0f0)
            end
            reim = (thread ÷ (2i32)) % (2i32)
            if reim == 0i32
                (+(Γ³_d0.re), +(Γ³_d0.im), +(Γ³_d1.re), +(Γ³_d1.im))
            else
                (-(Γ³_d0.im), +(Γ³_d0.re), -(Γ³_d1.im), +(Γ³_d1.re))
            end
        end
        Γ³_re = Float16x2(Γ³_d0_re, Γ³_d1_re)
        Γ³_im = Float16x2(Γ³_d0_im, Γ³_d1_im)
        aΓ³_cplx0 = Γ³_re
        aΓ³_cplx1 = Γ³_im
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
            d0 = (thread % 2) * (2i32) + 0i32
            d1 = (thread % 2) * (2i32) + 1i32
            v = thread ÷ (4i32)
            δ0 = (δ1 = (Γ²_d0 = if d0 < 4
                cispi((((d0 * v) % 32) / 16.0f0) % 2.0f0)
            else
                Complex(0.0f0)
            end))
            Γ²_d1 = if d1 < 4
                cispi((((d1 * v) % 32) / 16.0f0) % 2.0f0)
            else
                Complex(0.0f0)
            end
            (Γ²_d0.re, Γ²_d0.im, Γ²_d1.re, Γ²_d1.im)
        end
        Γ²_re = Float16x2(Γ²_d0_re, Γ²_d1_re)
        Γ²_im = Float16x2(Γ²_d0_im, Γ²_d1_im)
        bΓ²_cplx0 = Γ²_re
        bΓ²_cplx1 = Γ²_im
        (Γ³_d0_re, Γ³_d0_im, Γ³_d1_re, Γ³_d1_im) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            d0 = (thread % (2i32)) * (2i32) + 0i32
            d1 = (thread % (2i32)) * (2i32) + 1i32
            u = (thread ÷ (4i32)) % (4i32)
            Γ³_d0 = if d0 < 4 && u < 4
                cispi((((d0 * u) % 4) / 2.0f0) % 2.0f0)
            else
                Complex(0.0f0)
            end
            Γ³_d1 = if d1 < 4 && u < 4
                cispi((((d1 * u) % 4) / 2.0f0) % 2.0f0)
            else
                Complex(0.0f0)
            end
            reim = (thread ÷ (2i32)) % (2i32)
            if reim == 0i32
                (+(Γ³_d0.re), +(Γ³_d0.im), +(Γ³_d1.re), +(Γ³_d1.im))
            else
                (-(Γ³_d0.im), +(Γ³_d0.re), -(Γ³_d1.im), +(Γ³_d1.re))
            end
        end
        Γ³_re = Float16x2(Γ³_d0_re, Γ³_d1_re)
        Γ³_im = Float16x2(Γ³_d0_im, Γ³_d1_im)
        bΓ³_cplx0 = Γ³_re
        bΓ³_cplx1 = Γ³_im
        S = 999999999
        if let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            thread < 16
        end
            Smn = Smn_memory[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) % 256 + 0x01]
            (Smn_mn0, Smn_mn1) = convert(NTuple{2,Int32}, Smn)
            Sm = Smn_mn0
            Sn = Smn_mn1
            S = (33i32) * Sm + 546 * Sn
        end
        W_polr0 = zero(Float16x2)
        W_polr1 = zero(Float16x2)
        if let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            nlo = 2 * (thread ÷ 16)
            nlo < 4
        end
            W_polr0 = W_memory[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) % 4) * 16 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 1024) % 1024) % 1024) * 512) + 0x01]
            W_polr1 = W_memory[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) % 4) * 16 + 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 1024) % 1024) % 1024) * 512) + 0x01]
        end
        I = zero(Float16x2)
        dstime = 0
        t_running = 0
        for t_outer in 0:64:2047
            let
                (E_dish0_time0, E_dish4_time0, E_dish8_time0, E_dish12_time0) = IndexSpaces.unsafe_load4_global(
                    E_memory,
                    (
                        (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) ÷ 4) % 64 +
                        (
                            (
                                IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16 +
                                ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64
                            ) % 2048
                        ) * 131072 +
                        (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 2) * 64 +
                        ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 1024) % 1024) % 1024) * 128
                    ) + 1i32,
                )
                (E_dish0_time16, E_dish4_time16, E_dish8_time16, E_dish12_time16) = IndexSpaces.unsafe_load4_global(
                    E_memory,
                    (
                        (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) ÷ 4) % 64 +
                        (
                            (
                                (16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) +
                                ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64
                            ) % 2048
                        ) * 131072 +
                        (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 2) * 64 +
                        ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 1024) % 1024) % 1024) * 128
                    ) + 1i32,
                )
                (E_dish0_time32, E_dish4_time32, E_dish8_time32, E_dish12_time32) = IndexSpaces.unsafe_load4_global(
                    E_memory,
                    (
                        (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) ÷ 4) % 64 +
                        (
                            (
                                (32 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) +
                                ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64
                            ) % 2048
                        ) * 131072 +
                        (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 2) * 64 +
                        ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 1024) % 1024) % 1024) * 128
                    ) + 1i32,
                )
                (E_dish0_time48, E_dish4_time48, E_dish8_time48, E_dish12_time48) = IndexSpaces.unsafe_load4_global(
                    E_memory,
                    (
                        (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) ÷ 4) % 64 +
                        (
                            (
                                (48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) +
                                ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64
                            ) % 2048
                        ) * 131072 +
                        (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 2) * 64 +
                        ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 1024) % 1024) % 1024) * 128
                    ) + 1i32,
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
                (E_dish0_time16, E_dish8_time16) = let
                    src = if is_lo_thread
                        E_dish8_time16
                    else
                        E_dish0_time16
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                    if is_lo_thread
                        (E_dish0_time16, dst)
                    else
                        (dst, E_dish8_time16)
                    end
                end
                (E_dish4_time16, E_dish12_time16) = let
                    src = if is_lo_thread
                        E_dish12_time16
                    else
                        E_dish4_time16
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                    if is_lo_thread
                        (E_dish4_time16, dst)
                    else
                        (dst, E_dish12_time16)
                    end
                end
                (E_dish0_time32, E_dish8_time32) = let
                    src = if is_lo_thread
                        E_dish8_time32
                    else
                        E_dish0_time32
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                    if is_lo_thread
                        (E_dish0_time32, dst)
                    else
                        (dst, E_dish8_time32)
                    end
                end
                (E_dish4_time32, E_dish12_time32) = let
                    src = if is_lo_thread
                        E_dish12_time32
                    else
                        E_dish4_time32
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                    if is_lo_thread
                        (E_dish4_time32, dst)
                    else
                        (dst, E_dish12_time32)
                    end
                end
                (E_dish0_time48, E_dish8_time48) = let
                    src = if is_lo_thread
                        E_dish8_time48
                    else
                        E_dish0_time48
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                    if is_lo_thread
                        (E_dish0_time48, dst)
                    else
                        (dst, E_dish8_time48)
                    end
                end
                (E_dish4_time48, E_dish12_time48) = let
                    src = if is_lo_thread
                        E_dish12_time48
                    else
                        E_dish4_time48
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                    if is_lo_thread
                        (E_dish4_time48, dst)
                    else
                        (dst, E_dish12_time48)
                    end
                end
                (E_dish0_time0, E_dish8_time0) = (
                    IndexSpaces.get_lo4(E_dish0_time0, E_dish8_time0), IndexSpaces.get_hi4(E_dish0_time0, E_dish8_time0)
                )
                (E_dish4_time0, E_dish12_time0) = (
                    IndexSpaces.get_lo4(E_dish4_time0, E_dish12_time0), IndexSpaces.get_hi4(E_dish4_time0, E_dish12_time0)
                )
                (E_dish0_time16, E_dish8_time16) = (
                    IndexSpaces.get_lo4(E_dish0_time16, E_dish8_time16), IndexSpaces.get_hi4(E_dish0_time16, E_dish8_time16)
                )
                (E_dish4_time16, E_dish12_time16) = (
                    IndexSpaces.get_lo4(E_dish4_time16, E_dish12_time16), IndexSpaces.get_hi4(E_dish4_time16, E_dish12_time16)
                )
                (E_dish0_time32, E_dish8_time32) = (
                    IndexSpaces.get_lo4(E_dish0_time32, E_dish8_time32), IndexSpaces.get_hi4(E_dish0_time32, E_dish8_time32)
                )
                (E_dish4_time32, E_dish12_time32) = (
                    IndexSpaces.get_lo4(E_dish4_time32, E_dish12_time32), IndexSpaces.get_hi4(E_dish4_time32, E_dish12_time32)
                )
                (E_dish0_time48, E_dish8_time48) = (
                    IndexSpaces.get_lo4(E_dish0_time48, E_dish8_time48), IndexSpaces.get_hi4(E_dish0_time48, E_dish8_time48)
                )
                (E_dish4_time48, E_dish12_time48) = (
                    IndexSpaces.get_lo4(E_dish4_time48, E_dish12_time48), IndexSpaces.get_hi4(E_dish4_time48, E_dish12_time48)
                )
                (E_dish0_time0, E_dish0_time32) = (
                    IndexSpaces.get_lo8(E_dish0_time0, E_dish0_time32), IndexSpaces.get_hi8(E_dish0_time0, E_dish0_time32)
                )
                (E_dish4_time0, E_dish4_time32) = (
                    IndexSpaces.get_lo8(E_dish4_time0, E_dish4_time32), IndexSpaces.get_hi8(E_dish4_time0, E_dish4_time32)
                )
                (E_dish8_time0, E_dish8_time32) = (
                    IndexSpaces.get_lo8(E_dish8_time0, E_dish8_time32), IndexSpaces.get_hi8(E_dish8_time0, E_dish8_time32)
                )
                (E_dish12_time0, E_dish12_time32) = (
                    IndexSpaces.get_lo8(E_dish12_time0, E_dish12_time32), IndexSpaces.get_hi8(E_dish12_time0, E_dish12_time32)
                )
                (E_dish0_time16, E_dish0_time48) = (
                    IndexSpaces.get_lo8(E_dish0_time16, E_dish0_time48), IndexSpaces.get_hi8(E_dish0_time16, E_dish0_time48)
                )
                (E_dish4_time16, E_dish4_time48) = (
                    IndexSpaces.get_lo8(E_dish4_time16, E_dish4_time48), IndexSpaces.get_hi8(E_dish4_time16, E_dish4_time48)
                )
                (E_dish8_time16, E_dish8_time48) = (
                    IndexSpaces.get_lo8(E_dish8_time16, E_dish8_time48), IndexSpaces.get_hi8(E_dish8_time16, E_dish8_time48)
                )
                (E_dish12_time16, E_dish12_time48) = (
                    IndexSpaces.get_lo8(E_dish12_time16, E_dish12_time48), IndexSpaces.get_hi8(E_dish12_time16, E_dish12_time48)
                )
                (E_dish0_time0, E_dish8_time0) = (
                    IndexSpaces.get_lo16(E_dish0_time0, E_dish8_time0), IndexSpaces.get_hi16(E_dish0_time0, E_dish8_time0)
                )
                (E_dish4_time0, E_dish12_time0) = (
                    IndexSpaces.get_lo16(E_dish4_time0, E_dish12_time0), IndexSpaces.get_hi16(E_dish4_time0, E_dish12_time0)
                )
                (E_dish0_time16, E_dish8_time16) = (
                    IndexSpaces.get_lo16(E_dish0_time16, E_dish8_time16), IndexSpaces.get_hi16(E_dish0_time16, E_dish8_time16)
                )
                (E_dish4_time16, E_dish12_time16) = (
                    IndexSpaces.get_lo16(E_dish4_time16, E_dish12_time16), IndexSpaces.get_hi16(E_dish4_time16, E_dish12_time16)
                )
                (E_dish0_time32, E_dish8_time32) = (
                    IndexSpaces.get_lo16(E_dish0_time32, E_dish8_time32), IndexSpaces.get_hi16(E_dish0_time32, E_dish8_time32)
                )
                (E_dish4_time32, E_dish12_time32) = (
                    IndexSpaces.get_lo16(E_dish4_time32, E_dish12_time32), IndexSpaces.get_hi16(E_dish4_time32, E_dish12_time32)
                )
                (E_dish0_time48, E_dish8_time48) = (
                    IndexSpaces.get_lo16(E_dish0_time48, E_dish8_time48), IndexSpaces.get_hi16(E_dish0_time48, E_dish8_time48)
                )
                (E_dish4_time48, E_dish12_time48) = (
                    IndexSpaces.get_lo16(E_dish4_time48, E_dish12_time48), IndexSpaces.get_hi16(E_dish4_time48, E_dish12_time48)
                )
                if true
                    Fsh1_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 32) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish0_time0
                end
                if true
                    Fsh1_shared[((((((4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 32) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + (((4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish4_time0
                end
                if true
                    Fsh1_shared[((((((2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 32) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + (((2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish8_time0
                end
                if true
                    Fsh1_shared[((((((6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 32) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + (((6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish12_time0
                end
                if true
                    Fsh1_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 32) * 257 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16 + 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish0_time16
                end
                if true
                    Fsh1_shared[((((((4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 32) * 257 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16 + 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + (((4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish4_time16
                end
                if true
                    Fsh1_shared[((((((2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 32) * 257 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16 + 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + (((2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish8_time16
                end
                if true
                    Fsh1_shared[((((((6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 32) * 257 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16 + 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + (((6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish12_time16
                end
                if true
                    Fsh1_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16 + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 32) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16 + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish0_time32
                end
                if true
                    Fsh1_shared[(((((((4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 32) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + ((((4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish4_time32
                end
                if true
                    Fsh1_shared[(((((((2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 32) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + ((((2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish8_time32
                end
                if true
                    Fsh1_shared[(((((((6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 32) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + ((((6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish12_time32
                end
                if true
                    Fsh1_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16 + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 32) * 257 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16 + 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16 + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish0_time48
                end
                if true
                    Fsh1_shared[(((((((4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 32) * 257 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16 + 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + ((((4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish4_time48
                end
                if true
                    Fsh1_shared[(((((((2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 32) * 257 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16 + 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + ((((2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish8_time48
                end
                if true
                    Fsh1_shared[(((((((6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 32) * 257 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16 + 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + ((((6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish12_time48
                end
                IndexSpaces.cuda_sync_threads()
            end
            let
                Freg1_dish0 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) ÷ 8) % 32) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) % 8) * 32) + 0x01]
                Freg1_dish16 = Fsh1_shared[((((16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) ÷ 8) % 32) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + ((16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) % 8) * 32) + 0x01]
                Freg1_dish32 = Fsh1_shared[((((32 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) ÷ 8) % 32) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + ((32 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) % 8) * 32) + 0x01]
                Freg1_dish48 = Fsh1_shared[((((48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) ÷ 8) % 32) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + ((48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) % 8) * 32) + 0x01]
                Freg1_dish64 = Fsh1_shared[((((64 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) ÷ 8) % 32) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + ((64 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) % 8) * 32) + 0x01]
                Freg1_dish80 = Fsh1_shared[((((80 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) ÷ 8) % 32) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + ((80 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) % 8) * 32) + 0x01]
                Freg1_dish96 = Fsh1_shared[((((96 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) ÷ 8) % 32) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + ((96 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) % 8) * 32) + 0x01]
                Freg1_dish112 = Fsh1_shared[((((112 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) ÷ 8) % 32) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + ((112 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) % 8) * 32) + 0x01]
                Freg1_dish128 = Fsh1_shared[((((128 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) ÷ 8) % 32) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + ((128 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) % 8) * 32) + 0x01]
                Freg1_dish144 = Fsh1_shared[((((144 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) ÷ 8) % 32) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + ((144 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) % 8) * 32) + 0x01]
                Freg1_dish160 = Fsh1_shared[((((160 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) ÷ 8) % 32) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + ((160 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) % 8) * 32) + 0x01]
                Freg1_dish176 = Fsh1_shared[((((176 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) ÷ 8) % 32) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + ((176 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) % 8) * 32) + 0x01]
                Freg1_dish192 = Fsh1_shared[((((192 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) ÷ 8) % 32) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + ((192 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) % 8) * 32) + 0x01]
                Freg1_dish208 = Fsh1_shared[((((208 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) ÷ 8) % 32) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + ((208 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) % 8) * 32) + 0x01]
                Freg1_dish224 = Fsh1_shared[((((224 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) ÷ 8) % 32) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + ((224 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) % 8) * 32) + 0x01]
                Freg1_dish240 = Fsh1_shared[((((240 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) ÷ 8) % 32) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + ((240 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) % 8) * 32) + 0x01]
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
                Freg1′ = Freg1_dish0
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + sd_sd0) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish16
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + sd_sd1) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish32
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + sd_sd2) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish48
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + sd_sd3) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish64
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + sd_sd4) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish80
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + sd_sd5) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish96
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + sd_sd6) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish112
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + sd_sd7) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish128
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + sd_sd8) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish144
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + sd_sd9) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish160
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + sd_sd10) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish176
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + sd_sd11) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish192
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + sd_sd12) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish208
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + sd_sd13) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish224
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + sd_sd14) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish240
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + sd_sd15) + 0x01] =
                        Freg1′
                end
                IndexSpaces.cuda_sync_threads()
            end
            let
                Freg2_time0 = zero(Int4x8)
                Freg2_time2 = zero(Int4x8)
                Freg2_time4 = zero(Int4x8)
                Freg2_time6 = zero(Int4x8)
                Freg2_time8 = zero(Int4x8)
                Freg2_time10 = zero(Int4x8)
                Freg2_time12 = zero(Int4x8)
                Freg2_time14 = zero(Int4x8)
                Freg2_time16 = zero(Int4x8)
                Freg2_time18 = zero(Int4x8)
                Freg2_time20 = zero(Int4x8)
                Freg2_time22 = zero(Int4x8)
                Freg2_time24 = zero(Int4x8)
                Freg2_time26 = zero(Int4x8)
                Freg2_time28 = zero(Int4x8)
                Freg2_time30 = zero(Int4x8)
                if let
                    thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                    nlo = (1i32) * ((thread ÷ (4i32)) % (2i32)) + (2i32) * ((thread ÷ 16) % 2)
                    nlo < 4
                end
                    Freg2_time0 = Fsh2_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) % 16) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 2184 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2 + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) % 4) * 546) + 0x01]
                    Freg2_time2 = Fsh2_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) % 16) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 2184 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2 + 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) % 4) * 546) + 0x01]
                    Freg2_time4 = Fsh2_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) % 16) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 2184 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2 + 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) % 4) * 546) + 0x01]
                    Freg2_time6 = Fsh2_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) % 16) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 2184 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2 + 6) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) % 4) * 546) + 0x01]
                    Freg2_time8 = Fsh2_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) % 16) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 2184 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2 + 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) % 4) * 546) + 0x01]
                    Freg2_time10 = Fsh2_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) % 16) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 2184 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2 + 10) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) % 4) * 546) + 0x01]
                    Freg2_time12 = Fsh2_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) % 16) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 2184 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2 + 12) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) % 4) * 546) + 0x01]
                    Freg2_time14 = Fsh2_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) % 16) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 2184 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2 + 14) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) % 4) * 546) + 0x01]
                    Freg2_time16 = Fsh2_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) % 16) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 2184 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2 + 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) % 4) * 546) + 0x01]
                    Freg2_time18 = Fsh2_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) % 16) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 2184 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2 + 18) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) % 4) * 546) + 0x01]
                    Freg2_time20 = Fsh2_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) % 16) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 2184 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2 + 20) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) % 4) * 546) + 0x01]
                    Freg2_time22 = Fsh2_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) % 16) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 2184 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2 + 22) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) % 4) * 546) + 0x01]
                    Freg2_time24 = Fsh2_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) % 16) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 2184 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2 + 24) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) % 4) * 546) + 0x01]
                    Freg2_time26 = Fsh2_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) % 16) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 2184 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2 + 26) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) % 4) * 546) + 0x01]
                    Freg2_time28 = Fsh2_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) % 16) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 2184 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2 + 28) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) % 4) * 546) + 0x01]
                    Freg2_time30 = Fsh2_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) % 16) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 2184 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2 + 30) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) % 4) * 546) + 0x01]
                end
                IndexSpaces.cuda_sync_threads()
                let
                    t_inner_hi = 0
                    for t_inner_lo in 0:8:31
                        Freg2′_time0 = zero(Int4x8)
                        if t_inner_lo == 0
                            Freg2′_time0 = Freg2_time0
                        end
                        if t_inner_lo == 8
                            Freg2′_time0 = Freg2_time8
                        end
                        if t_inner_lo == 16
                            Freg2′_time0 = Freg2_time16
                        end
                        if t_inner_lo == 24
                            Freg2′_time0 = Freg2_time24
                        end
                        Freg2′_time2 = zero(Int4x8)
                        if t_inner_lo == 0
                            Freg2′_time2 = Freg2_time2
                        end
                        if t_inner_lo == 8
                            Freg2′_time2 = Freg2_time10
                        end
                        if t_inner_lo == 16
                            Freg2′_time2 = Freg2_time18
                        end
                        if t_inner_lo == 24
                            Freg2′_time2 = Freg2_time26
                        end
                        Freg2′_time4 = zero(Int4x8)
                        if t_inner_lo == 0
                            Freg2′_time4 = Freg2_time4
                        end
                        if t_inner_lo == 8
                            Freg2′_time4 = Freg2_time12
                        end
                        if t_inner_lo == 16
                            Freg2′_time4 = Freg2_time20
                        end
                        if t_inner_lo == 24
                            Freg2′_time4 = Freg2_time28
                        end
                        Freg2′_time6 = zero(Int4x8)
                        if t_inner_lo == 0
                            Freg2′_time6 = Freg2_time6
                        end
                        if t_inner_lo == 8
                            Freg2′_time6 = Freg2_time14
                        end
                        if t_inner_lo == 16
                            Freg2′_time6 = Freg2_time22
                        end
                        if t_inner_lo == 24
                            Freg2′_time6 = Freg2_time30
                        end
                        (E′_polr0_time0, E′_polr1_time0, E′_polr0_time32, E′_polr1_time32) = convert(
                            NTuple{4,Float16x2}, Freg2′_time0
                        )
                        (E′_polr0_time2, E′_polr1_time2, E′_polr0_time34, E′_polr1_time34) = convert(
                            NTuple{4,Float16x2}, Freg2′_time2
                        )
                        (E′_polr0_time4, E′_polr1_time4, E′_polr0_time36, E′_polr1_time36) = convert(
                            NTuple{4,Float16x2}, Freg2′_time4
                        )
                        (E′_polr0_time6, E′_polr1_time6, E′_polr0_time38, E′_polr1_time38) = convert(
                            NTuple{4,Float16x2}, Freg2′_time6
                        )
                        E_polr0_time0 = E′_polr0_time0
                        E_polr1_time0 = E′_polr1_time0
                        E_polr0_time2 = E′_polr0_time2
                        E_polr1_time2 = E′_polr1_time2
                        E_polr0_time4 = E′_polr0_time4
                        E_polr1_time4 = E′_polr1_time4
                        E_polr0_time6 = E′_polr0_time6
                        E_polr1_time6 = E′_polr1_time6
                        WE_polr0_time0 = complex_mul(W_polr0, E_polr0_time0)
                        WE_polr1_time0 = complex_mul(W_polr1, E_polr1_time0)
                        WE_polr0_time2 = complex_mul(W_polr0, E_polr0_time2)
                        WE_polr1_time2 = complex_mul(W_polr1, E_polr1_time2)
                        WE_polr0_time4 = complex_mul(W_polr0, E_polr0_time4)
                        WE_polr1_time4 = complex_mul(W_polr1, E_polr1_time4)
                        WE_polr0_time6 = complex_mul(W_polr0, E_polr0_time6)
                        WE_polr1_time6 = complex_mul(W_polr1, E_polr1_time6)
                        X_polr0_time0 = WE_polr0_time0
                        X_polr1_time0 = WE_polr1_time0
                        X_polr0_time2 = WE_polr0_time2
                        X_polr1_time2 = WE_polr1_time2
                        X_polr0_time4 = WE_polr0_time4
                        X_polr1_time4 = WE_polr1_time4
                        X_polr0_time6 = WE_polr0_time6
                        X_polr1_time6 = WE_polr1_time6
                        Z_cplx0_polr0_time0 = zero(Float16x2)
                        Z_cplx1_polr0_time0 = zero(Float16x2)
                        Z_cplx0_polr1_time0 = zero(Float16x2)
                        Z_cplx1_polr1_time0 = zero(Float16x2)
                        Z_cplx0_polr0_time2 = zero(Float16x2)
                        Z_cplx1_polr0_time2 = zero(Float16x2)
                        Z_cplx0_polr1_time2 = zero(Float16x2)
                        Z_cplx1_polr1_time2 = zero(Float16x2)
                        Z_cplx0_polr0_time4 = zero(Float16x2)
                        Z_cplx1_polr0_time4 = zero(Float16x2)
                        Z_cplx0_polr1_time4 = zero(Float16x2)
                        Z_cplx1_polr1_time4 = zero(Float16x2)
                        Z_cplx0_polr0_time6 = zero(Float16x2)
                        Z_cplx1_polr0_time6 = zero(Float16x2)
                        Z_cplx0_polr1_time6 = zero(Float16x2)
                        Z_cplx1_polr1_time6 = zero(Float16x2)
                        (Z_cplx0_polr0_time0, Z_cplx1_polr0_time0) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr0_time0, (Z_cplx0_polr0_time0, Z_cplx1_polr0_time0)
                        )
                        (Z_cplx0_polr1_time0, Z_cplx1_polr1_time0) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr1_time0, (Z_cplx0_polr1_time0, Z_cplx1_polr1_time0)
                        )
                        (Z_cplx0_polr0_time2, Z_cplx1_polr0_time2) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr0_time2, (Z_cplx0_polr0_time2, Z_cplx1_polr0_time2)
                        )
                        (Z_cplx0_polr1_time2, Z_cplx1_polr1_time2) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr1_time2, (Z_cplx0_polr1_time2, Z_cplx1_polr1_time2)
                        )
                        (Z_cplx0_polr0_time4, Z_cplx1_polr0_time4) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr0_time4, (Z_cplx0_polr0_time4, Z_cplx1_polr0_time4)
                        )
                        (Z_cplx0_polr1_time4, Z_cplx1_polr1_time4) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr1_time4, (Z_cplx0_polr1_time4, Z_cplx1_polr1_time4)
                        )
                        (Z_cplx0_polr0_time6, Z_cplx1_polr0_time6) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr0_time6, (Z_cplx0_polr0_time6, Z_cplx1_polr0_time6)
                        )
                        (Z_cplx0_polr1_time6, Z_cplx1_polr1_time6) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr1_time6, (Z_cplx0_polr1_time6, Z_cplx1_polr1_time6)
                        )
                        aΓ²re = aΓ²_cplx0
                        aΓ²im = aΓ²_cplx1
                        Zre_polr0_time0 = Z_cplx0_polr0_time0
                        Zim_polr0_time0 = Z_cplx1_polr0_time0
                        Zre_polr1_time0 = Z_cplx0_polr1_time0
                        Zim_polr1_time0 = Z_cplx1_polr1_time0
                        Zre_polr0_time2 = Z_cplx0_polr0_time2
                        Zim_polr0_time2 = Z_cplx1_polr0_time2
                        Zre_polr1_time2 = Z_cplx0_polr1_time2
                        Zim_polr1_time2 = Z_cplx1_polr1_time2
                        Zre_polr0_time4 = Z_cplx0_polr0_time4
                        Zim_polr0_time4 = Z_cplx1_polr0_time4
                        Zre_polr1_time4 = Z_cplx0_polr1_time4
                        Zim_polr1_time4 = Z_cplx1_polr1_time4
                        Zre_polr0_time6 = Z_cplx0_polr0_time6
                        Zim_polr0_time6 = Z_cplx1_polr0_time6
                        Zre_polr1_time6 = Z_cplx0_polr1_time6
                        Zim_polr1_time6 = Z_cplx1_polr1_time6
                        Vre_polr0_time0 = muladd(aΓ²re, Zre_polr0_time0, -aΓ²im * Zim_polr0_time0)
                        Vre_polr1_time0 = muladd(aΓ²re, Zre_polr1_time0, -aΓ²im * Zim_polr1_time0)
                        Vre_polr0_time2 = muladd(aΓ²re, Zre_polr0_time2, -aΓ²im * Zim_polr0_time2)
                        Vre_polr1_time2 = muladd(aΓ²re, Zre_polr1_time2, -aΓ²im * Zim_polr1_time2)
                        Vre_polr0_time4 = muladd(aΓ²re, Zre_polr0_time4, -aΓ²im * Zim_polr0_time4)
                        Vre_polr1_time4 = muladd(aΓ²re, Zre_polr1_time4, -aΓ²im * Zim_polr1_time4)
                        Vre_polr0_time6 = muladd(aΓ²re, Zre_polr0_time6, -aΓ²im * Zim_polr0_time6)
                        Vre_polr1_time6 = muladd(aΓ²re, Zre_polr1_time6, -aΓ²im * Zim_polr1_time6)
                        Vim_polr0_time0 = muladd(aΓ²re, Zim_polr0_time0, +aΓ²im * Zre_polr0_time0)
                        Vim_polr1_time0 = muladd(aΓ²re, Zim_polr1_time0, +aΓ²im * Zre_polr1_time0)
                        Vim_polr0_time2 = muladd(aΓ²re, Zim_polr0_time2, +aΓ²im * Zre_polr0_time2)
                        Vim_polr1_time2 = muladd(aΓ²re, Zim_polr1_time2, +aΓ²im * Zre_polr1_time2)
                        Vim_polr0_time4 = muladd(aΓ²re, Zim_polr0_time4, +aΓ²im * Zre_polr0_time4)
                        Vim_polr1_time4 = muladd(aΓ²re, Zim_polr1_time4, +aΓ²im * Zre_polr1_time4)
                        Vim_polr0_time6 = muladd(aΓ²re, Zim_polr0_time6, +aΓ²im * Zre_polr0_time6)
                        Vim_polr1_time6 = muladd(aΓ²re, Zim_polr1_time6, +aΓ²im * Zre_polr1_time6)
                        V_cplx0_polr0_time0 = Vre_polr0_time0
                        V_cplx1_polr0_time0 = Vim_polr0_time0
                        V_cplx0_polr1_time0 = Vre_polr1_time0
                        V_cplx1_polr1_time0 = Vim_polr1_time0
                        V_cplx0_polr0_time2 = Vre_polr0_time2
                        V_cplx1_polr0_time2 = Vim_polr0_time2
                        V_cplx0_polr1_time2 = Vre_polr1_time2
                        V_cplx1_polr1_time2 = Vim_polr1_time2
                        V_cplx0_polr0_time4 = Vre_polr0_time4
                        V_cplx1_polr0_time4 = Vim_polr0_time4
                        V_cplx0_polr1_time4 = Vre_polr1_time4
                        V_cplx1_polr1_time4 = Vim_polr1_time4
                        V_cplx0_polr0_time6 = Vre_polr0_time6
                        V_cplx1_polr0_time6 = Vim_polr0_time6
                        V_cplx0_polr1_time6 = Vre_polr1_time6
                        V_cplx1_polr1_time6 = Vim_polr1_time6
                        Y_cplx0_polr0_time0 = zero(Float16x2)
                        Y_cplx1_polr0_time0 = zero(Float16x2)
                        Y_cplx0_polr1_time0 = zero(Float16x2)
                        Y_cplx1_polr1_time0 = zero(Float16x2)
                        Y_cplx0_polr0_time2 = zero(Float16x2)
                        Y_cplx1_polr0_time2 = zero(Float16x2)
                        Y_cplx0_polr1_time2 = zero(Float16x2)
                        Y_cplx1_polr1_time2 = zero(Float16x2)
                        Y_cplx0_polr0_time4 = zero(Float16x2)
                        Y_cplx1_polr0_time4 = zero(Float16x2)
                        Y_cplx0_polr1_time4 = zero(Float16x2)
                        Y_cplx1_polr1_time4 = zero(Float16x2)
                        Y_cplx0_polr0_time6 = zero(Float16x2)
                        Y_cplx1_polr0_time6 = zero(Float16x2)
                        Y_cplx0_polr1_time6 = zero(Float16x2)
                        Y_cplx1_polr1_time6 = zero(Float16x2)
                        Vre_polr0_time0 = V_cplx0_polr0_time0
                        Vim_polr0_time0 = V_cplx1_polr0_time0
                        Vre_polr1_time0 = V_cplx0_polr1_time0
                        Vim_polr1_time0 = V_cplx1_polr1_time0
                        Vre_polr0_time2 = V_cplx0_polr0_time2
                        Vim_polr0_time2 = V_cplx1_polr0_time2
                        Vre_polr1_time2 = V_cplx0_polr1_time2
                        Vim_polr1_time2 = V_cplx1_polr1_time2
                        Vre_polr0_time4 = V_cplx0_polr0_time4
                        Vim_polr0_time4 = V_cplx1_polr0_time4
                        Vre_polr1_time4 = V_cplx0_polr1_time4
                        Vim_polr1_time4 = V_cplx1_polr1_time4
                        Vre_polr0_time6 = V_cplx0_polr0_time6
                        Vim_polr0_time6 = V_cplx1_polr0_time6
                        Vre_polr1_time6 = V_cplx0_polr1_time6
                        Vim_polr1_time6 = V_cplx1_polr1_time6
                        V_cplx_in0_polr0_time0 = Vre_polr0_time0
                        V_cplx_in1_polr0_time0 = Vim_polr0_time0
                        V_cplx_in0_polr1_time0 = Vre_polr1_time0
                        V_cplx_in1_polr1_time0 = Vim_polr1_time0
                        V_cplx_in0_polr0_time2 = Vre_polr0_time2
                        V_cplx_in1_polr0_time2 = Vim_polr0_time2
                        V_cplx_in0_polr1_time2 = Vre_polr1_time2
                        V_cplx_in1_polr1_time2 = Vim_polr1_time2
                        V_cplx_in0_polr0_time4 = Vre_polr0_time4
                        V_cplx_in1_polr0_time4 = Vim_polr0_time4
                        V_cplx_in0_polr1_time4 = Vre_polr1_time4
                        V_cplx_in1_polr1_time4 = Vim_polr1_time4
                        V_cplx_in0_polr0_time6 = Vre_polr0_time6
                        V_cplx_in1_polr0_time6 = Vim_polr0_time6
                        V_cplx_in0_polr1_time6 = Vre_polr1_time6
                        V_cplx_in1_polr1_time6 = Vim_polr1_time6
                        (Y_cplx0_polr0_time0, Y_cplx1_polr0_time0) = let
                            e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            thread = IndexSpaces.cuda_threadidx()
                            threadgroup = thread ÷ 4
                            e = if threadgroup == 0i32
                                e0
                            elseif threadgroup == 1i32
                                e1
                            elseif threadgroup == 2i32
                                e2
                            elseif threadgroup == 3i32
                                e3
                            elseif threadgroup == 4i32
                                e4
                            elseif threadgroup == 5i32
                                e5
                            elseif threadgroup == 6i32
                                e6
                            elseif threadgroup == 7i32
                                e7
                            end
                            IndexSpaces.mma_sp_m16n8k16(
                                (aΓ³_cplx0, aΓ³_cplx1)::NTuple{2,Float16x2},
                                (V_cplx_in0_polr0_time0, V_cplx_in1_polr0_time0)::NTuple{2,Float16x2},
                                (Y_cplx0_polr0_time0, Y_cplx1_polr0_time0)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        (Y_cplx0_polr1_time0, Y_cplx1_polr1_time0) = let
                            e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            thread = IndexSpaces.cuda_threadidx()
                            threadgroup = thread ÷ 4
                            e = if threadgroup == 0i32
                                e0
                            elseif threadgroup == 1i32
                                e1
                            elseif threadgroup == 2i32
                                e2
                            elseif threadgroup == 3i32
                                e3
                            elseif threadgroup == 4i32
                                e4
                            elseif threadgroup == 5i32
                                e5
                            elseif threadgroup == 6i32
                                e6
                            elseif threadgroup == 7i32
                                e7
                            end
                            IndexSpaces.mma_sp_m16n8k16(
                                (aΓ³_cplx0, aΓ³_cplx1)::NTuple{2,Float16x2},
                                (V_cplx_in0_polr1_time0, V_cplx_in1_polr1_time0)::NTuple{2,Float16x2},
                                (Y_cplx0_polr1_time0, Y_cplx1_polr1_time0)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        (Y_cplx0_polr0_time2, Y_cplx1_polr0_time2) = let
                            e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            thread = IndexSpaces.cuda_threadidx()
                            threadgroup = thread ÷ 4
                            e = if threadgroup == 0i32
                                e0
                            elseif threadgroup == 1i32
                                e1
                            elseif threadgroup == 2i32
                                e2
                            elseif threadgroup == 3i32
                                e3
                            elseif threadgroup == 4i32
                                e4
                            elseif threadgroup == 5i32
                                e5
                            elseif threadgroup == 6i32
                                e6
                            elseif threadgroup == 7i32
                                e7
                            end
                            IndexSpaces.mma_sp_m16n8k16(
                                (aΓ³_cplx0, aΓ³_cplx1)::NTuple{2,Float16x2},
                                (V_cplx_in0_polr0_time2, V_cplx_in1_polr0_time2)::NTuple{2,Float16x2},
                                (Y_cplx0_polr0_time2, Y_cplx1_polr0_time2)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        (Y_cplx0_polr1_time2, Y_cplx1_polr1_time2) = let
                            e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            thread = IndexSpaces.cuda_threadidx()
                            threadgroup = thread ÷ 4
                            e = if threadgroup == 0i32
                                e0
                            elseif threadgroup == 1i32
                                e1
                            elseif threadgroup == 2i32
                                e2
                            elseif threadgroup == 3i32
                                e3
                            elseif threadgroup == 4i32
                                e4
                            elseif threadgroup == 5i32
                                e5
                            elseif threadgroup == 6i32
                                e6
                            elseif threadgroup == 7i32
                                e7
                            end
                            IndexSpaces.mma_sp_m16n8k16(
                                (aΓ³_cplx0, aΓ³_cplx1)::NTuple{2,Float16x2},
                                (V_cplx_in0_polr1_time2, V_cplx_in1_polr1_time2)::NTuple{2,Float16x2},
                                (Y_cplx0_polr1_time2, Y_cplx1_polr1_time2)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        (Y_cplx0_polr0_time4, Y_cplx1_polr0_time4) = let
                            e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            thread = IndexSpaces.cuda_threadidx()
                            threadgroup = thread ÷ 4
                            e = if threadgroup == 0i32
                                e0
                            elseif threadgroup == 1i32
                                e1
                            elseif threadgroup == 2i32
                                e2
                            elseif threadgroup == 3i32
                                e3
                            elseif threadgroup == 4i32
                                e4
                            elseif threadgroup == 5i32
                                e5
                            elseif threadgroup == 6i32
                                e6
                            elseif threadgroup == 7i32
                                e7
                            end
                            IndexSpaces.mma_sp_m16n8k16(
                                (aΓ³_cplx0, aΓ³_cplx1)::NTuple{2,Float16x2},
                                (V_cplx_in0_polr0_time4, V_cplx_in1_polr0_time4)::NTuple{2,Float16x2},
                                (Y_cplx0_polr0_time4, Y_cplx1_polr0_time4)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        (Y_cplx0_polr1_time4, Y_cplx1_polr1_time4) = let
                            e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            thread = IndexSpaces.cuda_threadidx()
                            threadgroup = thread ÷ 4
                            e = if threadgroup == 0i32
                                e0
                            elseif threadgroup == 1i32
                                e1
                            elseif threadgroup == 2i32
                                e2
                            elseif threadgroup == 3i32
                                e3
                            elseif threadgroup == 4i32
                                e4
                            elseif threadgroup == 5i32
                                e5
                            elseif threadgroup == 6i32
                                e6
                            elseif threadgroup == 7i32
                                e7
                            end
                            IndexSpaces.mma_sp_m16n8k16(
                                (aΓ³_cplx0, aΓ³_cplx1)::NTuple{2,Float16x2},
                                (V_cplx_in0_polr1_time4, V_cplx_in1_polr1_time4)::NTuple{2,Float16x2},
                                (Y_cplx0_polr1_time4, Y_cplx1_polr1_time4)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        (Y_cplx0_polr0_time6, Y_cplx1_polr0_time6) = let
                            e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            thread = IndexSpaces.cuda_threadidx()
                            threadgroup = thread ÷ 4
                            e = if threadgroup == 0i32
                                e0
                            elseif threadgroup == 1i32
                                e1
                            elseif threadgroup == 2i32
                                e2
                            elseif threadgroup == 3i32
                                e3
                            elseif threadgroup == 4i32
                                e4
                            elseif threadgroup == 5i32
                                e5
                            elseif threadgroup == 6i32
                                e6
                            elseif threadgroup == 7i32
                                e7
                            end
                            IndexSpaces.mma_sp_m16n8k16(
                                (aΓ³_cplx0, aΓ³_cplx1)::NTuple{2,Float16x2},
                                (V_cplx_in0_polr0_time6, V_cplx_in1_polr0_time6)::NTuple{2,Float16x2},
                                (Y_cplx0_polr0_time6, Y_cplx1_polr0_time6)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        (Y_cplx0_polr1_time6, Y_cplx1_polr1_time6) = let
                            e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            thread = IndexSpaces.cuda_threadidx()
                            threadgroup = thread ÷ 4
                            e = if threadgroup == 0i32
                                e0
                            elseif threadgroup == 1i32
                                e1
                            elseif threadgroup == 2i32
                                e2
                            elseif threadgroup == 3i32
                                e3
                            elseif threadgroup == 4i32
                                e4
                            elseif threadgroup == 5i32
                                e5
                            elseif threadgroup == 6i32
                                e6
                            elseif threadgroup == 7i32
                                e7
                            end
                            IndexSpaces.mma_sp_m16n8k16(
                                (aΓ³_cplx0, aΓ³_cplx1)::NTuple{2,Float16x2},
                                (V_cplx_in0_polr1_time6, V_cplx_in1_polr1_time6)::NTuple{2,Float16x2},
                                (Y_cplx0_polr1_time6, Y_cplx1_polr1_time6)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        G_cplx0_polr0_time0 = Y_cplx0_polr0_time0
                        G_cplx1_polr0_time0 = Y_cplx1_polr0_time0
                        G_cplx0_polr1_time0 = Y_cplx0_polr1_time0
                        G_cplx1_polr1_time0 = Y_cplx1_polr1_time0
                        G_cplx0_polr0_time2 = Y_cplx0_polr0_time2
                        G_cplx1_polr0_time2 = Y_cplx1_polr0_time2
                        G_cplx0_polr1_time2 = Y_cplx0_polr1_time2
                        G_cplx1_polr1_time2 = Y_cplx1_polr1_time2
                        G_cplx0_polr0_time4 = Y_cplx0_polr0_time4
                        G_cplx1_polr0_time4 = Y_cplx1_polr0_time4
                        G_cplx0_polr1_time4 = Y_cplx0_polr1_time4
                        G_cplx1_polr1_time4 = Y_cplx1_polr1_time4
                        G_cplx0_polr0_time6 = Y_cplx0_polr0_time6
                        G_cplx1_polr0_time6 = Y_cplx1_polr0_time6
                        G_cplx0_polr1_time6 = Y_cplx0_polr1_time6
                        G_cplx1_polr1_time6 = Y_cplx1_polr1_time6
                        (G_cplx0_polr0_time0, G_cplx1_polr0_time0) = (
                            IndexSpaces.get_lo16(G_cplx0_polr0_time0, G_cplx1_polr0_time0),
                            IndexSpaces.get_hi16(G_cplx0_polr0_time0, G_cplx1_polr0_time0),
                        )
                        (G_cplx0_polr1_time0, G_cplx1_polr1_time0) = (
                            IndexSpaces.get_lo16(G_cplx0_polr1_time0, G_cplx1_polr1_time0),
                            IndexSpaces.get_hi16(G_cplx0_polr1_time0, G_cplx1_polr1_time0),
                        )
                        (G_cplx0_polr0_time2, G_cplx1_polr0_time2) = (
                            IndexSpaces.get_lo16(G_cplx0_polr0_time2, G_cplx1_polr0_time2),
                            IndexSpaces.get_hi16(G_cplx0_polr0_time2, G_cplx1_polr0_time2),
                        )
                        (G_cplx0_polr1_time2, G_cplx1_polr1_time2) = (
                            IndexSpaces.get_lo16(G_cplx0_polr1_time2, G_cplx1_polr1_time2),
                            IndexSpaces.get_hi16(G_cplx0_polr1_time2, G_cplx1_polr1_time2),
                        )
                        (G_cplx0_polr0_time4, G_cplx1_polr0_time4) = (
                            IndexSpaces.get_lo16(G_cplx0_polr0_time4, G_cplx1_polr0_time4),
                            IndexSpaces.get_hi16(G_cplx0_polr0_time4, G_cplx1_polr0_time4),
                        )
                        (G_cplx0_polr1_time4, G_cplx1_polr1_time4) = (
                            IndexSpaces.get_lo16(G_cplx0_polr1_time4, G_cplx1_polr1_time4),
                            IndexSpaces.get_hi16(G_cplx0_polr1_time4, G_cplx1_polr1_time4),
                        )
                        (G_cplx0_polr0_time6, G_cplx1_polr0_time6) = (
                            IndexSpaces.get_lo16(G_cplx0_polr0_time6, G_cplx1_polr0_time6),
                            IndexSpaces.get_hi16(G_cplx0_polr0_time6, G_cplx1_polr0_time6),
                        )
                        (G_cplx0_polr1_time6, G_cplx1_polr1_time6) = (
                            IndexSpaces.get_lo16(G_cplx0_polr1_time6, G_cplx1_polr1_time6),
                            IndexSpaces.get_hi16(G_cplx0_polr1_time6, G_cplx1_polr1_time6),
                        )
                        if true
                            Gsh_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx0_polr0_time0
                        end
                        if true
                            Gsh_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx1_polr0_time0
                        end
                        if true
                            Gsh_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx0_polr1_time0
                        end
                        if true
                            Gsh_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx1_polr1_time0
                        end
                        if true
                            Gsh_shared[(((((((2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx0_polr0_time2
                        end
                        if true
                            Gsh_shared[(((((((2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx1_polr0_time2
                        end
                        if true
                            Gsh_shared[(((((((2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx0_polr1_time2
                        end
                        if true
                            Gsh_shared[(((((((2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx1_polr1_time2
                        end
                        if true
                            Gsh_shared[(((((((4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx0_polr0_time4
                        end
                        if true
                            Gsh_shared[(((((((4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx1_polr0_time4
                        end
                        if true
                            Gsh_shared[(((((((4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx0_polr1_time4
                        end
                        if true
                            Gsh_shared[(((((((4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx1_polr1_time4
                        end
                        if true
                            Gsh_shared[(((((((6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx0_polr0_time6
                        end
                        if true
                            Gsh_shared[(((((((6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx1_polr0_time6
                        end
                        if true
                            Gsh_shared[(((((((6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx0_polr1_time6
                        end
                        if true
                            Gsh_shared[(((((((6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx1_polr1_time6
                        end
                        IndexSpaces.cuda_sync_threads()
                        let
                            t = 0
                            G_polr0 = zero(Float16x2)
                            G_polr1 = zero(Float16x2)
                            if true
                                G_polr0 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                                G_polr1 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                            end
                            X_polr0 = G_polr0
                            X_polr1 = G_polr1
                            Z_cplx0_polr0 = zero(Float16x2)
                            Z_cplx1_polr0 = zero(Float16x2)
                            Z_cplx0_polr1 = zero(Float16x2)
                            Z_cplx1_polr1 = zero(Float16x2)
                            (Z_cplx0_polr0, Z_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr0, (Z_cplx0_polr0, Z_cplx1_polr0)
                            )
                            (Z_cplx0_polr1, Z_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr1, (Z_cplx0_polr1, Z_cplx1_polr1)
                            )
                            bΓ²re = bΓ²_cplx0
                            bΓ²im = bΓ²_cplx1
                            Zre_polr0 = Z_cplx0_polr0
                            Zim_polr0 = Z_cplx1_polr0
                            Zre_polr1 = Z_cplx0_polr1
                            Zim_polr1 = Z_cplx1_polr1
                            Vre_polr0 = muladd(bΓ²re, Zre_polr0, -bΓ²im * Zim_polr0)
                            Vre_polr1 = muladd(bΓ²re, Zre_polr1, -bΓ²im * Zim_polr1)
                            Vim_polr0 = muladd(bΓ²re, Zim_polr0, +bΓ²im * Zre_polr0)
                            Vim_polr1 = muladd(bΓ²re, Zim_polr1, +bΓ²im * Zre_polr1)
                            V_cplx0_polr0 = Vre_polr0
                            V_cplx1_polr0 = Vim_polr0
                            V_cplx0_polr1 = Vre_polr1
                            V_cplx1_polr1 = Vim_polr1
                            Y_cplx0_polr0 = zero(Float16x2)
                            Y_cplx1_polr0 = zero(Float16x2)
                            Y_cplx0_polr1 = zero(Float16x2)
                            Y_cplx1_polr1 = zero(Float16x2)
                            Vre_polr0 = V_cplx0_polr0
                            Vim_polr0 = V_cplx1_polr0
                            Vre_polr1 = V_cplx0_polr1
                            Vim_polr1 = V_cplx1_polr1
                            V_cplx_in0_polr0 = Vre_polr0
                            V_cplx_in1_polr0 = Vim_polr0
                            V_cplx_in0_polr1 = Vre_polr1
                            V_cplx_in1_polr1 = Vim_polr1
                            (Y_cplx0_polr0, Y_cplx1_polr0) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr0, V_cplx_in1_polr0)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr0, Y_cplx1_polr0)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            (Y_cplx0_polr1, Y_cplx1_polr1) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr1, V_cplx_in1_polr1)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr1, Y_cplx1_polr1)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            Ẽ_cplx0_polr0 = Y_cplx0_polr0
                            Ẽ_cplx1_polr0 = Y_cplx1_polr0
                            Ẽ_cplx0_polr1 = Y_cplx0_polr1
                            Ẽ_cplx1_polr1 = Y_cplx1_polr1
                            Ẽp0_cplx0 = Ẽ_cplx0_polr0
                            Ẽp1_cplx0 = Ẽ_cplx0_polr1
                            Ẽp0_cplx1 = Ẽ_cplx1_polr0
                            Ẽp1_cplx1 = Ẽ_cplx1_polr1
                            Ẽp0re = Ẽp0_cplx0
                            Ẽp0im = Ẽp0_cplx1
                            Ẽp1re = Ẽp1_cplx0
                            Ẽp1im = Ẽp1_cplx1
                            I = muladd(
                                Float16x2(0.005001068f0, 0.005001068f0),
                                muladd(Ẽp1im, Ẽp1im, muladd(Ẽp1re, Ẽp1re, muladd(Ẽp0im, Ẽp0im, Ẽp0re * Ẽp0re))),
                                I,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 1 == 0i32
                                if t_running == 25
                                    if let
                                        thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                        warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16)
                                        p = (2i32) * thread
                                        q = (2i32) * warp
                                        0i32 ≤ p < 32 && 0i32 ≤ q < 32
                                    end
                                        if true
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 16 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 32) * 16 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 81) % 81) % 81) * 512 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 1024) % 1024) % 1024) * 41472) + 0) + 0x01] =
                                                I
                                        end
                                    end
                                    I = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let
                            t = 1
                            G_polr0 = zero(Float16x2)
                            G_polr1 = zero(Float16x2)
                            if true
                                G_polr0 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                                G_polr1 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                            end
                            X_polr0 = G_polr0
                            X_polr1 = G_polr1
                            Z_cplx0_polr0 = zero(Float16x2)
                            Z_cplx1_polr0 = zero(Float16x2)
                            Z_cplx0_polr1 = zero(Float16x2)
                            Z_cplx1_polr1 = zero(Float16x2)
                            (Z_cplx0_polr0, Z_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr0, (Z_cplx0_polr0, Z_cplx1_polr0)
                            )
                            (Z_cplx0_polr1, Z_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr1, (Z_cplx0_polr1, Z_cplx1_polr1)
                            )
                            bΓ²re = bΓ²_cplx0
                            bΓ²im = bΓ²_cplx1
                            Zre_polr0 = Z_cplx0_polr0
                            Zim_polr0 = Z_cplx1_polr0
                            Zre_polr1 = Z_cplx0_polr1
                            Zim_polr1 = Z_cplx1_polr1
                            Vre_polr0 = muladd(bΓ²re, Zre_polr0, -bΓ²im * Zim_polr0)
                            Vre_polr1 = muladd(bΓ²re, Zre_polr1, -bΓ²im * Zim_polr1)
                            Vim_polr0 = muladd(bΓ²re, Zim_polr0, +bΓ²im * Zre_polr0)
                            Vim_polr1 = muladd(bΓ²re, Zim_polr1, +bΓ²im * Zre_polr1)
                            V_cplx0_polr0 = Vre_polr0
                            V_cplx1_polr0 = Vim_polr0
                            V_cplx0_polr1 = Vre_polr1
                            V_cplx1_polr1 = Vim_polr1
                            Y_cplx0_polr0 = zero(Float16x2)
                            Y_cplx1_polr0 = zero(Float16x2)
                            Y_cplx0_polr1 = zero(Float16x2)
                            Y_cplx1_polr1 = zero(Float16x2)
                            Vre_polr0 = V_cplx0_polr0
                            Vim_polr0 = V_cplx1_polr0
                            Vre_polr1 = V_cplx0_polr1
                            Vim_polr1 = V_cplx1_polr1
                            V_cplx_in0_polr0 = Vre_polr0
                            V_cplx_in1_polr0 = Vim_polr0
                            V_cplx_in0_polr1 = Vre_polr1
                            V_cplx_in1_polr1 = Vim_polr1
                            (Y_cplx0_polr0, Y_cplx1_polr0) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr0, V_cplx_in1_polr0)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr0, Y_cplx1_polr0)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            (Y_cplx0_polr1, Y_cplx1_polr1) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr1, V_cplx_in1_polr1)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr1, Y_cplx1_polr1)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            Ẽ_cplx0_polr0 = Y_cplx0_polr0
                            Ẽ_cplx1_polr0 = Y_cplx1_polr0
                            Ẽ_cplx0_polr1 = Y_cplx0_polr1
                            Ẽ_cplx1_polr1 = Y_cplx1_polr1
                            Ẽp0_cplx0 = Ẽ_cplx0_polr0
                            Ẽp1_cplx0 = Ẽ_cplx0_polr1
                            Ẽp0_cplx1 = Ẽ_cplx1_polr0
                            Ẽp1_cplx1 = Ẽ_cplx1_polr1
                            Ẽp0re = Ẽp0_cplx0
                            Ẽp0im = Ẽp0_cplx1
                            Ẽp1re = Ẽp1_cplx0
                            Ẽp1im = Ẽp1_cplx1
                            I = muladd(
                                Float16x2(0.005001068f0, 0.005001068f0),
                                muladd(Ẽp1im, Ẽp1im, muladd(Ẽp1re, Ẽp1re, muladd(Ẽp0im, Ẽp0im, Ẽp0re * Ẽp0re))),
                                I,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 1 == 0i32
                                if t_running == 25
                                    if let
                                        thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                        warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16)
                                        p = (2i32) * thread
                                        q = (2i32) * warp
                                        0i32 ≤ p < 32 && 0i32 ≤ q < 32
                                    end
                                        if true
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 16 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 32) * 16 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 81) % 81) % 81) * 512 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 1024) % 1024) % 1024) * 41472) + 0) + 0x01] =
                                                I
                                        end
                                    end
                                    I = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let
                            t = 2
                            G_polr0 = zero(Float16x2)
                            G_polr1 = zero(Float16x2)
                            if true
                                G_polr0 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                                G_polr1 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                            end
                            X_polr0 = G_polr0
                            X_polr1 = G_polr1
                            Z_cplx0_polr0 = zero(Float16x2)
                            Z_cplx1_polr0 = zero(Float16x2)
                            Z_cplx0_polr1 = zero(Float16x2)
                            Z_cplx1_polr1 = zero(Float16x2)
                            (Z_cplx0_polr0, Z_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr0, (Z_cplx0_polr0, Z_cplx1_polr0)
                            )
                            (Z_cplx0_polr1, Z_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr1, (Z_cplx0_polr1, Z_cplx1_polr1)
                            )
                            bΓ²re = bΓ²_cplx0
                            bΓ²im = bΓ²_cplx1
                            Zre_polr0 = Z_cplx0_polr0
                            Zim_polr0 = Z_cplx1_polr0
                            Zre_polr1 = Z_cplx0_polr1
                            Zim_polr1 = Z_cplx1_polr1
                            Vre_polr0 = muladd(bΓ²re, Zre_polr0, -bΓ²im * Zim_polr0)
                            Vre_polr1 = muladd(bΓ²re, Zre_polr1, -bΓ²im * Zim_polr1)
                            Vim_polr0 = muladd(bΓ²re, Zim_polr0, +bΓ²im * Zre_polr0)
                            Vim_polr1 = muladd(bΓ²re, Zim_polr1, +bΓ²im * Zre_polr1)
                            V_cplx0_polr0 = Vre_polr0
                            V_cplx1_polr0 = Vim_polr0
                            V_cplx0_polr1 = Vre_polr1
                            V_cplx1_polr1 = Vim_polr1
                            Y_cplx0_polr0 = zero(Float16x2)
                            Y_cplx1_polr0 = zero(Float16x2)
                            Y_cplx0_polr1 = zero(Float16x2)
                            Y_cplx1_polr1 = zero(Float16x2)
                            Vre_polr0 = V_cplx0_polr0
                            Vim_polr0 = V_cplx1_polr0
                            Vre_polr1 = V_cplx0_polr1
                            Vim_polr1 = V_cplx1_polr1
                            V_cplx_in0_polr0 = Vre_polr0
                            V_cplx_in1_polr0 = Vim_polr0
                            V_cplx_in0_polr1 = Vre_polr1
                            V_cplx_in1_polr1 = Vim_polr1
                            (Y_cplx0_polr0, Y_cplx1_polr0) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr0, V_cplx_in1_polr0)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr0, Y_cplx1_polr0)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            (Y_cplx0_polr1, Y_cplx1_polr1) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr1, V_cplx_in1_polr1)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr1, Y_cplx1_polr1)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            Ẽ_cplx0_polr0 = Y_cplx0_polr0
                            Ẽ_cplx1_polr0 = Y_cplx1_polr0
                            Ẽ_cplx0_polr1 = Y_cplx0_polr1
                            Ẽ_cplx1_polr1 = Y_cplx1_polr1
                            Ẽp0_cplx0 = Ẽ_cplx0_polr0
                            Ẽp1_cplx0 = Ẽ_cplx0_polr1
                            Ẽp0_cplx1 = Ẽ_cplx1_polr0
                            Ẽp1_cplx1 = Ẽ_cplx1_polr1
                            Ẽp0re = Ẽp0_cplx0
                            Ẽp0im = Ẽp0_cplx1
                            Ẽp1re = Ẽp1_cplx0
                            Ẽp1im = Ẽp1_cplx1
                            I = muladd(
                                Float16x2(0.005001068f0, 0.005001068f0),
                                muladd(Ẽp1im, Ẽp1im, muladd(Ẽp1re, Ẽp1re, muladd(Ẽp0im, Ẽp0im, Ẽp0re * Ẽp0re))),
                                I,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 1 == 0i32
                                if t_running == 25
                                    if let
                                        thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                        warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16)
                                        p = (2i32) * thread
                                        q = (2i32) * warp
                                        0i32 ≤ p < 32 && 0i32 ≤ q < 32
                                    end
                                        if true
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 16 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 32) * 16 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 81) % 81) % 81) * 512 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 1024) % 1024) % 1024) * 41472) + 0) + 0x01] =
                                                I
                                        end
                                    end
                                    I = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let
                            t = 3
                            G_polr0 = zero(Float16x2)
                            G_polr1 = zero(Float16x2)
                            if true
                                G_polr0 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                                G_polr1 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                            end
                            X_polr0 = G_polr0
                            X_polr1 = G_polr1
                            Z_cplx0_polr0 = zero(Float16x2)
                            Z_cplx1_polr0 = zero(Float16x2)
                            Z_cplx0_polr1 = zero(Float16x2)
                            Z_cplx1_polr1 = zero(Float16x2)
                            (Z_cplx0_polr0, Z_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr0, (Z_cplx0_polr0, Z_cplx1_polr0)
                            )
                            (Z_cplx0_polr1, Z_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr1, (Z_cplx0_polr1, Z_cplx1_polr1)
                            )
                            bΓ²re = bΓ²_cplx0
                            bΓ²im = bΓ²_cplx1
                            Zre_polr0 = Z_cplx0_polr0
                            Zim_polr0 = Z_cplx1_polr0
                            Zre_polr1 = Z_cplx0_polr1
                            Zim_polr1 = Z_cplx1_polr1
                            Vre_polr0 = muladd(bΓ²re, Zre_polr0, -bΓ²im * Zim_polr0)
                            Vre_polr1 = muladd(bΓ²re, Zre_polr1, -bΓ²im * Zim_polr1)
                            Vim_polr0 = muladd(bΓ²re, Zim_polr0, +bΓ²im * Zre_polr0)
                            Vim_polr1 = muladd(bΓ²re, Zim_polr1, +bΓ²im * Zre_polr1)
                            V_cplx0_polr0 = Vre_polr0
                            V_cplx1_polr0 = Vim_polr0
                            V_cplx0_polr1 = Vre_polr1
                            V_cplx1_polr1 = Vim_polr1
                            Y_cplx0_polr0 = zero(Float16x2)
                            Y_cplx1_polr0 = zero(Float16x2)
                            Y_cplx0_polr1 = zero(Float16x2)
                            Y_cplx1_polr1 = zero(Float16x2)
                            Vre_polr0 = V_cplx0_polr0
                            Vim_polr0 = V_cplx1_polr0
                            Vre_polr1 = V_cplx0_polr1
                            Vim_polr1 = V_cplx1_polr1
                            V_cplx_in0_polr0 = Vre_polr0
                            V_cplx_in1_polr0 = Vim_polr0
                            V_cplx_in0_polr1 = Vre_polr1
                            V_cplx_in1_polr1 = Vim_polr1
                            (Y_cplx0_polr0, Y_cplx1_polr0) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr0, V_cplx_in1_polr0)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr0, Y_cplx1_polr0)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            (Y_cplx0_polr1, Y_cplx1_polr1) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr1, V_cplx_in1_polr1)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr1, Y_cplx1_polr1)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            Ẽ_cplx0_polr0 = Y_cplx0_polr0
                            Ẽ_cplx1_polr0 = Y_cplx1_polr0
                            Ẽ_cplx0_polr1 = Y_cplx0_polr1
                            Ẽ_cplx1_polr1 = Y_cplx1_polr1
                            Ẽp0_cplx0 = Ẽ_cplx0_polr0
                            Ẽp1_cplx0 = Ẽ_cplx0_polr1
                            Ẽp0_cplx1 = Ẽ_cplx1_polr0
                            Ẽp1_cplx1 = Ẽ_cplx1_polr1
                            Ẽp0re = Ẽp0_cplx0
                            Ẽp0im = Ẽp0_cplx1
                            Ẽp1re = Ẽp1_cplx0
                            Ẽp1im = Ẽp1_cplx1
                            I = muladd(
                                Float16x2(0.005001068f0, 0.005001068f0),
                                muladd(Ẽp1im, Ẽp1im, muladd(Ẽp1re, Ẽp1re, muladd(Ẽp0im, Ẽp0im, Ẽp0re * Ẽp0re))),
                                I,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 1 == 0i32
                                if t_running == 25
                                    if let
                                        thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                        warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16)
                                        p = (2i32) * thread
                                        q = (2i32) * warp
                                        0i32 ≤ p < 32 && 0i32 ≤ q < 32
                                    end
                                        if true
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 16 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 32) * 16 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 81) % 81) % 81) * 512 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 1024) % 1024) % 1024) * 41472) + 0) + 0x01] =
                                                I
                                        end
                                    end
                                    I = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let
                            t = 4
                            G_polr0 = zero(Float16x2)
                            G_polr1 = zero(Float16x2)
                            if true
                                G_polr0 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                                G_polr1 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                            end
                            X_polr0 = G_polr0
                            X_polr1 = G_polr1
                            Z_cplx0_polr0 = zero(Float16x2)
                            Z_cplx1_polr0 = zero(Float16x2)
                            Z_cplx0_polr1 = zero(Float16x2)
                            Z_cplx1_polr1 = zero(Float16x2)
                            (Z_cplx0_polr0, Z_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr0, (Z_cplx0_polr0, Z_cplx1_polr0)
                            )
                            (Z_cplx0_polr1, Z_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr1, (Z_cplx0_polr1, Z_cplx1_polr1)
                            )
                            bΓ²re = bΓ²_cplx0
                            bΓ²im = bΓ²_cplx1
                            Zre_polr0 = Z_cplx0_polr0
                            Zim_polr0 = Z_cplx1_polr0
                            Zre_polr1 = Z_cplx0_polr1
                            Zim_polr1 = Z_cplx1_polr1
                            Vre_polr0 = muladd(bΓ²re, Zre_polr0, -bΓ²im * Zim_polr0)
                            Vre_polr1 = muladd(bΓ²re, Zre_polr1, -bΓ²im * Zim_polr1)
                            Vim_polr0 = muladd(bΓ²re, Zim_polr0, +bΓ²im * Zre_polr0)
                            Vim_polr1 = muladd(bΓ²re, Zim_polr1, +bΓ²im * Zre_polr1)
                            V_cplx0_polr0 = Vre_polr0
                            V_cplx1_polr0 = Vim_polr0
                            V_cplx0_polr1 = Vre_polr1
                            V_cplx1_polr1 = Vim_polr1
                            Y_cplx0_polr0 = zero(Float16x2)
                            Y_cplx1_polr0 = zero(Float16x2)
                            Y_cplx0_polr1 = zero(Float16x2)
                            Y_cplx1_polr1 = zero(Float16x2)
                            Vre_polr0 = V_cplx0_polr0
                            Vim_polr0 = V_cplx1_polr0
                            Vre_polr1 = V_cplx0_polr1
                            Vim_polr1 = V_cplx1_polr1
                            V_cplx_in0_polr0 = Vre_polr0
                            V_cplx_in1_polr0 = Vim_polr0
                            V_cplx_in0_polr1 = Vre_polr1
                            V_cplx_in1_polr1 = Vim_polr1
                            (Y_cplx0_polr0, Y_cplx1_polr0) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr0, V_cplx_in1_polr0)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr0, Y_cplx1_polr0)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            (Y_cplx0_polr1, Y_cplx1_polr1) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr1, V_cplx_in1_polr1)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr1, Y_cplx1_polr1)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            Ẽ_cplx0_polr0 = Y_cplx0_polr0
                            Ẽ_cplx1_polr0 = Y_cplx1_polr0
                            Ẽ_cplx0_polr1 = Y_cplx0_polr1
                            Ẽ_cplx1_polr1 = Y_cplx1_polr1
                            Ẽp0_cplx0 = Ẽ_cplx0_polr0
                            Ẽp1_cplx0 = Ẽ_cplx0_polr1
                            Ẽp0_cplx1 = Ẽ_cplx1_polr0
                            Ẽp1_cplx1 = Ẽ_cplx1_polr1
                            Ẽp0re = Ẽp0_cplx0
                            Ẽp0im = Ẽp0_cplx1
                            Ẽp1re = Ẽp1_cplx0
                            Ẽp1im = Ẽp1_cplx1
                            I = muladd(
                                Float16x2(0.005001068f0, 0.005001068f0),
                                muladd(Ẽp1im, Ẽp1im, muladd(Ẽp1re, Ẽp1re, muladd(Ẽp0im, Ẽp0im, Ẽp0re * Ẽp0re))),
                                I,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 1 == 0i32
                                if t_running == 25
                                    if let
                                        thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                        warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16)
                                        p = (2i32) * thread
                                        q = (2i32) * warp
                                        0i32 ≤ p < 32 && 0i32 ≤ q < 32
                                    end
                                        if true
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 16 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 32) * 16 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 81) % 81) % 81) * 512 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 1024) % 1024) % 1024) * 41472) + 0) + 0x01] =
                                                I
                                        end
                                    end
                                    I = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let
                            t = 5
                            G_polr0 = zero(Float16x2)
                            G_polr1 = zero(Float16x2)
                            if true
                                G_polr0 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                                G_polr1 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                            end
                            X_polr0 = G_polr0
                            X_polr1 = G_polr1
                            Z_cplx0_polr0 = zero(Float16x2)
                            Z_cplx1_polr0 = zero(Float16x2)
                            Z_cplx0_polr1 = zero(Float16x2)
                            Z_cplx1_polr1 = zero(Float16x2)
                            (Z_cplx0_polr0, Z_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr0, (Z_cplx0_polr0, Z_cplx1_polr0)
                            )
                            (Z_cplx0_polr1, Z_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr1, (Z_cplx0_polr1, Z_cplx1_polr1)
                            )
                            bΓ²re = bΓ²_cplx0
                            bΓ²im = bΓ²_cplx1
                            Zre_polr0 = Z_cplx0_polr0
                            Zim_polr0 = Z_cplx1_polr0
                            Zre_polr1 = Z_cplx0_polr1
                            Zim_polr1 = Z_cplx1_polr1
                            Vre_polr0 = muladd(bΓ²re, Zre_polr0, -bΓ²im * Zim_polr0)
                            Vre_polr1 = muladd(bΓ²re, Zre_polr1, -bΓ²im * Zim_polr1)
                            Vim_polr0 = muladd(bΓ²re, Zim_polr0, +bΓ²im * Zre_polr0)
                            Vim_polr1 = muladd(bΓ²re, Zim_polr1, +bΓ²im * Zre_polr1)
                            V_cplx0_polr0 = Vre_polr0
                            V_cplx1_polr0 = Vim_polr0
                            V_cplx0_polr1 = Vre_polr1
                            V_cplx1_polr1 = Vim_polr1
                            Y_cplx0_polr0 = zero(Float16x2)
                            Y_cplx1_polr0 = zero(Float16x2)
                            Y_cplx0_polr1 = zero(Float16x2)
                            Y_cplx1_polr1 = zero(Float16x2)
                            Vre_polr0 = V_cplx0_polr0
                            Vim_polr0 = V_cplx1_polr0
                            Vre_polr1 = V_cplx0_polr1
                            Vim_polr1 = V_cplx1_polr1
                            V_cplx_in0_polr0 = Vre_polr0
                            V_cplx_in1_polr0 = Vim_polr0
                            V_cplx_in0_polr1 = Vre_polr1
                            V_cplx_in1_polr1 = Vim_polr1
                            (Y_cplx0_polr0, Y_cplx1_polr0) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr0, V_cplx_in1_polr0)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr0, Y_cplx1_polr0)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            (Y_cplx0_polr1, Y_cplx1_polr1) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr1, V_cplx_in1_polr1)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr1, Y_cplx1_polr1)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            Ẽ_cplx0_polr0 = Y_cplx0_polr0
                            Ẽ_cplx1_polr0 = Y_cplx1_polr0
                            Ẽ_cplx0_polr1 = Y_cplx0_polr1
                            Ẽ_cplx1_polr1 = Y_cplx1_polr1
                            Ẽp0_cplx0 = Ẽ_cplx0_polr0
                            Ẽp1_cplx0 = Ẽ_cplx0_polr1
                            Ẽp0_cplx1 = Ẽ_cplx1_polr0
                            Ẽp1_cplx1 = Ẽ_cplx1_polr1
                            Ẽp0re = Ẽp0_cplx0
                            Ẽp0im = Ẽp0_cplx1
                            Ẽp1re = Ẽp1_cplx0
                            Ẽp1im = Ẽp1_cplx1
                            I = muladd(
                                Float16x2(0.005001068f0, 0.005001068f0),
                                muladd(Ẽp1im, Ẽp1im, muladd(Ẽp1re, Ẽp1re, muladd(Ẽp0im, Ẽp0im, Ẽp0re * Ẽp0re))),
                                I,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 1 == 0i32
                                if t_running == 25
                                    if let
                                        thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                        warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16)
                                        p = (2i32) * thread
                                        q = (2i32) * warp
                                        0i32 ≤ p < 32 && 0i32 ≤ q < 32
                                    end
                                        if true
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 16 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 32) * 16 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 81) % 81) % 81) * 512 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 1024) % 1024) % 1024) * 41472) + 0) + 0x01] =
                                                I
                                        end
                                    end
                                    I = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let
                            t = 6
                            G_polr0 = zero(Float16x2)
                            G_polr1 = zero(Float16x2)
                            if true
                                G_polr0 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                                G_polr1 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                            end
                            X_polr0 = G_polr0
                            X_polr1 = G_polr1
                            Z_cplx0_polr0 = zero(Float16x2)
                            Z_cplx1_polr0 = zero(Float16x2)
                            Z_cplx0_polr1 = zero(Float16x2)
                            Z_cplx1_polr1 = zero(Float16x2)
                            (Z_cplx0_polr0, Z_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr0, (Z_cplx0_polr0, Z_cplx1_polr0)
                            )
                            (Z_cplx0_polr1, Z_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr1, (Z_cplx0_polr1, Z_cplx1_polr1)
                            )
                            bΓ²re = bΓ²_cplx0
                            bΓ²im = bΓ²_cplx1
                            Zre_polr0 = Z_cplx0_polr0
                            Zim_polr0 = Z_cplx1_polr0
                            Zre_polr1 = Z_cplx0_polr1
                            Zim_polr1 = Z_cplx1_polr1
                            Vre_polr0 = muladd(bΓ²re, Zre_polr0, -bΓ²im * Zim_polr0)
                            Vre_polr1 = muladd(bΓ²re, Zre_polr1, -bΓ²im * Zim_polr1)
                            Vim_polr0 = muladd(bΓ²re, Zim_polr0, +bΓ²im * Zre_polr0)
                            Vim_polr1 = muladd(bΓ²re, Zim_polr1, +bΓ²im * Zre_polr1)
                            V_cplx0_polr0 = Vre_polr0
                            V_cplx1_polr0 = Vim_polr0
                            V_cplx0_polr1 = Vre_polr1
                            V_cplx1_polr1 = Vim_polr1
                            Y_cplx0_polr0 = zero(Float16x2)
                            Y_cplx1_polr0 = zero(Float16x2)
                            Y_cplx0_polr1 = zero(Float16x2)
                            Y_cplx1_polr1 = zero(Float16x2)
                            Vre_polr0 = V_cplx0_polr0
                            Vim_polr0 = V_cplx1_polr0
                            Vre_polr1 = V_cplx0_polr1
                            Vim_polr1 = V_cplx1_polr1
                            V_cplx_in0_polr0 = Vre_polr0
                            V_cplx_in1_polr0 = Vim_polr0
                            V_cplx_in0_polr1 = Vre_polr1
                            V_cplx_in1_polr1 = Vim_polr1
                            (Y_cplx0_polr0, Y_cplx1_polr0) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr0, V_cplx_in1_polr0)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr0, Y_cplx1_polr0)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            (Y_cplx0_polr1, Y_cplx1_polr1) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr1, V_cplx_in1_polr1)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr1, Y_cplx1_polr1)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            Ẽ_cplx0_polr0 = Y_cplx0_polr0
                            Ẽ_cplx1_polr0 = Y_cplx1_polr0
                            Ẽ_cplx0_polr1 = Y_cplx0_polr1
                            Ẽ_cplx1_polr1 = Y_cplx1_polr1
                            Ẽp0_cplx0 = Ẽ_cplx0_polr0
                            Ẽp1_cplx0 = Ẽ_cplx0_polr1
                            Ẽp0_cplx1 = Ẽ_cplx1_polr0
                            Ẽp1_cplx1 = Ẽ_cplx1_polr1
                            Ẽp0re = Ẽp0_cplx0
                            Ẽp0im = Ẽp0_cplx1
                            Ẽp1re = Ẽp1_cplx0
                            Ẽp1im = Ẽp1_cplx1
                            I = muladd(
                                Float16x2(0.005001068f0, 0.005001068f0),
                                muladd(Ẽp1im, Ẽp1im, muladd(Ẽp1re, Ẽp1re, muladd(Ẽp0im, Ẽp0im, Ẽp0re * Ẽp0re))),
                                I,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 1 == 0i32
                                if t_running == 25
                                    if let
                                        thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                        warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16)
                                        p = (2i32) * thread
                                        q = (2i32) * warp
                                        0i32 ≤ p < 32 && 0i32 ≤ q < 32
                                    end
                                        if true
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 16 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 32) * 16 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 81) % 81) % 81) * 512 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 1024) % 1024) % 1024) * 41472) + 0) + 0x01] =
                                                I
                                        end
                                    end
                                    I = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let
                            t = 7
                            G_polr0 = zero(Float16x2)
                            G_polr1 = zero(Float16x2)
                            if true
                                G_polr0 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                                G_polr1 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                            end
                            X_polr0 = G_polr0
                            X_polr1 = G_polr1
                            Z_cplx0_polr0 = zero(Float16x2)
                            Z_cplx1_polr0 = zero(Float16x2)
                            Z_cplx0_polr1 = zero(Float16x2)
                            Z_cplx1_polr1 = zero(Float16x2)
                            (Z_cplx0_polr0, Z_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr0, (Z_cplx0_polr0, Z_cplx1_polr0)
                            )
                            (Z_cplx0_polr1, Z_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr1, (Z_cplx0_polr1, Z_cplx1_polr1)
                            )
                            bΓ²re = bΓ²_cplx0
                            bΓ²im = bΓ²_cplx1
                            Zre_polr0 = Z_cplx0_polr0
                            Zim_polr0 = Z_cplx1_polr0
                            Zre_polr1 = Z_cplx0_polr1
                            Zim_polr1 = Z_cplx1_polr1
                            Vre_polr0 = muladd(bΓ²re, Zre_polr0, -bΓ²im * Zim_polr0)
                            Vre_polr1 = muladd(bΓ²re, Zre_polr1, -bΓ²im * Zim_polr1)
                            Vim_polr0 = muladd(bΓ²re, Zim_polr0, +bΓ²im * Zre_polr0)
                            Vim_polr1 = muladd(bΓ²re, Zim_polr1, +bΓ²im * Zre_polr1)
                            V_cplx0_polr0 = Vre_polr0
                            V_cplx1_polr0 = Vim_polr0
                            V_cplx0_polr1 = Vre_polr1
                            V_cplx1_polr1 = Vim_polr1
                            Y_cplx0_polr0 = zero(Float16x2)
                            Y_cplx1_polr0 = zero(Float16x2)
                            Y_cplx0_polr1 = zero(Float16x2)
                            Y_cplx1_polr1 = zero(Float16x2)
                            Vre_polr0 = V_cplx0_polr0
                            Vim_polr0 = V_cplx1_polr0
                            Vre_polr1 = V_cplx0_polr1
                            Vim_polr1 = V_cplx1_polr1
                            V_cplx_in0_polr0 = Vre_polr0
                            V_cplx_in1_polr0 = Vim_polr0
                            V_cplx_in0_polr1 = Vre_polr1
                            V_cplx_in1_polr1 = Vim_polr1
                            (Y_cplx0_polr0, Y_cplx1_polr0) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr0, V_cplx_in1_polr0)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr0, Y_cplx1_polr0)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            (Y_cplx0_polr1, Y_cplx1_polr1) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr1, V_cplx_in1_polr1)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr1, Y_cplx1_polr1)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            Ẽ_cplx0_polr0 = Y_cplx0_polr0
                            Ẽ_cplx1_polr0 = Y_cplx1_polr0
                            Ẽ_cplx0_polr1 = Y_cplx0_polr1
                            Ẽ_cplx1_polr1 = Y_cplx1_polr1
                            Ẽp0_cplx0 = Ẽ_cplx0_polr0
                            Ẽp1_cplx0 = Ẽ_cplx0_polr1
                            Ẽp0_cplx1 = Ẽ_cplx1_polr0
                            Ẽp1_cplx1 = Ẽ_cplx1_polr1
                            Ẽp0re = Ẽp0_cplx0
                            Ẽp0im = Ẽp0_cplx1
                            Ẽp1re = Ẽp1_cplx0
                            Ẽp1im = Ẽp1_cplx1
                            I = muladd(
                                Float16x2(0.005001068f0, 0.005001068f0),
                                muladd(Ẽp1im, Ẽp1im, muladd(Ẽp1re, Ẽp1re, muladd(Ẽp0im, Ẽp0im, Ẽp0re * Ẽp0re))),
                                I,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 1 == 0i32
                                if t_running == 25
                                    if let
                                        thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                        warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16)
                                        p = (2i32) * thread
                                        q = (2i32) * warp
                                        0i32 ≤ p < 32 && 0i32 ≤ q < 32
                                    end
                                        if true
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 16 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 32) * 16 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 81) % 81) % 81) * 512 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 1024) % 1024) % 1024) * 41472) + 0) + 0x01] =
                                                I
                                        end
                                    end
                                    I = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        IndexSpaces.cuda_sync_threads()
                    end
                end
                let
                    t_inner_hi = 32
                    for t_inner_lo in 0:8:31
                        Freg2′_time0 = zero(Int4x8)
                        if t_inner_lo == 0
                            Freg2′_time0 = Freg2_time0
                        end
                        if t_inner_lo == 8
                            Freg2′_time0 = Freg2_time8
                        end
                        if t_inner_lo == 16
                            Freg2′_time0 = Freg2_time16
                        end
                        if t_inner_lo == 24
                            Freg2′_time0 = Freg2_time24
                        end
                        Freg2′_time2 = zero(Int4x8)
                        if t_inner_lo == 0
                            Freg2′_time2 = Freg2_time2
                        end
                        if t_inner_lo == 8
                            Freg2′_time2 = Freg2_time10
                        end
                        if t_inner_lo == 16
                            Freg2′_time2 = Freg2_time18
                        end
                        if t_inner_lo == 24
                            Freg2′_time2 = Freg2_time26
                        end
                        Freg2′_time4 = zero(Int4x8)
                        if t_inner_lo == 0
                            Freg2′_time4 = Freg2_time4
                        end
                        if t_inner_lo == 8
                            Freg2′_time4 = Freg2_time12
                        end
                        if t_inner_lo == 16
                            Freg2′_time4 = Freg2_time20
                        end
                        if t_inner_lo == 24
                            Freg2′_time4 = Freg2_time28
                        end
                        Freg2′_time6 = zero(Int4x8)
                        if t_inner_lo == 0
                            Freg2′_time6 = Freg2_time6
                        end
                        if t_inner_lo == 8
                            Freg2′_time6 = Freg2_time14
                        end
                        if t_inner_lo == 16
                            Freg2′_time6 = Freg2_time22
                        end
                        if t_inner_lo == 24
                            Freg2′_time6 = Freg2_time30
                        end
                        (E′_polr0_time0, E′_polr1_time0, E′_polr0_time32, E′_polr1_time32) = convert(
                            NTuple{4,Float16x2}, Freg2′_time0
                        )
                        (E′_polr0_time2, E′_polr1_time2, E′_polr0_time34, E′_polr1_time34) = convert(
                            NTuple{4,Float16x2}, Freg2′_time2
                        )
                        (E′_polr0_time4, E′_polr1_time4, E′_polr0_time36, E′_polr1_time36) = convert(
                            NTuple{4,Float16x2}, Freg2′_time4
                        )
                        (E′_polr0_time6, E′_polr1_time6, E′_polr0_time38, E′_polr1_time38) = convert(
                            NTuple{4,Float16x2}, Freg2′_time6
                        )
                        E_polr0_time0 = E′_polr0_time32
                        E_polr1_time0 = E′_polr1_time32
                        E_polr0_time2 = E′_polr0_time34
                        E_polr1_time2 = E′_polr1_time34
                        E_polr0_time4 = E′_polr0_time36
                        E_polr1_time4 = E′_polr1_time36
                        E_polr0_time6 = E′_polr0_time38
                        E_polr1_time6 = E′_polr1_time38
                        WE_polr0_time0 = complex_mul(W_polr0, E_polr0_time0)
                        WE_polr1_time0 = complex_mul(W_polr1, E_polr1_time0)
                        WE_polr0_time2 = complex_mul(W_polr0, E_polr0_time2)
                        WE_polr1_time2 = complex_mul(W_polr1, E_polr1_time2)
                        WE_polr0_time4 = complex_mul(W_polr0, E_polr0_time4)
                        WE_polr1_time4 = complex_mul(W_polr1, E_polr1_time4)
                        WE_polr0_time6 = complex_mul(W_polr0, E_polr0_time6)
                        WE_polr1_time6 = complex_mul(W_polr1, E_polr1_time6)
                        X_polr0_time0 = WE_polr0_time0
                        X_polr1_time0 = WE_polr1_time0
                        X_polr0_time2 = WE_polr0_time2
                        X_polr1_time2 = WE_polr1_time2
                        X_polr0_time4 = WE_polr0_time4
                        X_polr1_time4 = WE_polr1_time4
                        X_polr0_time6 = WE_polr0_time6
                        X_polr1_time6 = WE_polr1_time6
                        Z_cplx0_polr0_time0 = zero(Float16x2)
                        Z_cplx1_polr0_time0 = zero(Float16x2)
                        Z_cplx0_polr1_time0 = zero(Float16x2)
                        Z_cplx1_polr1_time0 = zero(Float16x2)
                        Z_cplx0_polr0_time2 = zero(Float16x2)
                        Z_cplx1_polr0_time2 = zero(Float16x2)
                        Z_cplx0_polr1_time2 = zero(Float16x2)
                        Z_cplx1_polr1_time2 = zero(Float16x2)
                        Z_cplx0_polr0_time4 = zero(Float16x2)
                        Z_cplx1_polr0_time4 = zero(Float16x2)
                        Z_cplx0_polr1_time4 = zero(Float16x2)
                        Z_cplx1_polr1_time4 = zero(Float16x2)
                        Z_cplx0_polr0_time6 = zero(Float16x2)
                        Z_cplx1_polr0_time6 = zero(Float16x2)
                        Z_cplx0_polr1_time6 = zero(Float16x2)
                        Z_cplx1_polr1_time6 = zero(Float16x2)
                        (Z_cplx0_polr0_time0, Z_cplx1_polr0_time0) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr0_time0, (Z_cplx0_polr0_time0, Z_cplx1_polr0_time0)
                        )
                        (Z_cplx0_polr1_time0, Z_cplx1_polr1_time0) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr1_time0, (Z_cplx0_polr1_time0, Z_cplx1_polr1_time0)
                        )
                        (Z_cplx0_polr0_time2, Z_cplx1_polr0_time2) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr0_time2, (Z_cplx0_polr0_time2, Z_cplx1_polr0_time2)
                        )
                        (Z_cplx0_polr1_time2, Z_cplx1_polr1_time2) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr1_time2, (Z_cplx0_polr1_time2, Z_cplx1_polr1_time2)
                        )
                        (Z_cplx0_polr0_time4, Z_cplx1_polr0_time4) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr0_time4, (Z_cplx0_polr0_time4, Z_cplx1_polr0_time4)
                        )
                        (Z_cplx0_polr1_time4, Z_cplx1_polr1_time4) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr1_time4, (Z_cplx0_polr1_time4, Z_cplx1_polr1_time4)
                        )
                        (Z_cplx0_polr0_time6, Z_cplx1_polr0_time6) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr0_time6, (Z_cplx0_polr0_time6, Z_cplx1_polr0_time6)
                        )
                        (Z_cplx0_polr1_time6, Z_cplx1_polr1_time6) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_polr1_time6, (Z_cplx0_polr1_time6, Z_cplx1_polr1_time6)
                        )
                        aΓ²re = aΓ²_cplx0
                        aΓ²im = aΓ²_cplx1
                        Zre_polr0_time0 = Z_cplx0_polr0_time0
                        Zim_polr0_time0 = Z_cplx1_polr0_time0
                        Zre_polr1_time0 = Z_cplx0_polr1_time0
                        Zim_polr1_time0 = Z_cplx1_polr1_time0
                        Zre_polr0_time2 = Z_cplx0_polr0_time2
                        Zim_polr0_time2 = Z_cplx1_polr0_time2
                        Zre_polr1_time2 = Z_cplx0_polr1_time2
                        Zim_polr1_time2 = Z_cplx1_polr1_time2
                        Zre_polr0_time4 = Z_cplx0_polr0_time4
                        Zim_polr0_time4 = Z_cplx1_polr0_time4
                        Zre_polr1_time4 = Z_cplx0_polr1_time4
                        Zim_polr1_time4 = Z_cplx1_polr1_time4
                        Zre_polr0_time6 = Z_cplx0_polr0_time6
                        Zim_polr0_time6 = Z_cplx1_polr0_time6
                        Zre_polr1_time6 = Z_cplx0_polr1_time6
                        Zim_polr1_time6 = Z_cplx1_polr1_time6
                        Vre_polr0_time0 = muladd(aΓ²re, Zre_polr0_time0, -aΓ²im * Zim_polr0_time0)
                        Vre_polr1_time0 = muladd(aΓ²re, Zre_polr1_time0, -aΓ²im * Zim_polr1_time0)
                        Vre_polr0_time2 = muladd(aΓ²re, Zre_polr0_time2, -aΓ²im * Zim_polr0_time2)
                        Vre_polr1_time2 = muladd(aΓ²re, Zre_polr1_time2, -aΓ²im * Zim_polr1_time2)
                        Vre_polr0_time4 = muladd(aΓ²re, Zre_polr0_time4, -aΓ²im * Zim_polr0_time4)
                        Vre_polr1_time4 = muladd(aΓ²re, Zre_polr1_time4, -aΓ²im * Zim_polr1_time4)
                        Vre_polr0_time6 = muladd(aΓ²re, Zre_polr0_time6, -aΓ²im * Zim_polr0_time6)
                        Vre_polr1_time6 = muladd(aΓ²re, Zre_polr1_time6, -aΓ²im * Zim_polr1_time6)
                        Vim_polr0_time0 = muladd(aΓ²re, Zim_polr0_time0, +aΓ²im * Zre_polr0_time0)
                        Vim_polr1_time0 = muladd(aΓ²re, Zim_polr1_time0, +aΓ²im * Zre_polr1_time0)
                        Vim_polr0_time2 = muladd(aΓ²re, Zim_polr0_time2, +aΓ²im * Zre_polr0_time2)
                        Vim_polr1_time2 = muladd(aΓ²re, Zim_polr1_time2, +aΓ²im * Zre_polr1_time2)
                        Vim_polr0_time4 = muladd(aΓ²re, Zim_polr0_time4, +aΓ²im * Zre_polr0_time4)
                        Vim_polr1_time4 = muladd(aΓ²re, Zim_polr1_time4, +aΓ²im * Zre_polr1_time4)
                        Vim_polr0_time6 = muladd(aΓ²re, Zim_polr0_time6, +aΓ²im * Zre_polr0_time6)
                        Vim_polr1_time6 = muladd(aΓ²re, Zim_polr1_time6, +aΓ²im * Zre_polr1_time6)
                        V_cplx0_polr0_time0 = Vre_polr0_time0
                        V_cplx1_polr0_time0 = Vim_polr0_time0
                        V_cplx0_polr1_time0 = Vre_polr1_time0
                        V_cplx1_polr1_time0 = Vim_polr1_time0
                        V_cplx0_polr0_time2 = Vre_polr0_time2
                        V_cplx1_polr0_time2 = Vim_polr0_time2
                        V_cplx0_polr1_time2 = Vre_polr1_time2
                        V_cplx1_polr1_time2 = Vim_polr1_time2
                        V_cplx0_polr0_time4 = Vre_polr0_time4
                        V_cplx1_polr0_time4 = Vim_polr0_time4
                        V_cplx0_polr1_time4 = Vre_polr1_time4
                        V_cplx1_polr1_time4 = Vim_polr1_time4
                        V_cplx0_polr0_time6 = Vre_polr0_time6
                        V_cplx1_polr0_time6 = Vim_polr0_time6
                        V_cplx0_polr1_time6 = Vre_polr1_time6
                        V_cplx1_polr1_time6 = Vim_polr1_time6
                        Y_cplx0_polr0_time0 = zero(Float16x2)
                        Y_cplx1_polr0_time0 = zero(Float16x2)
                        Y_cplx0_polr1_time0 = zero(Float16x2)
                        Y_cplx1_polr1_time0 = zero(Float16x2)
                        Y_cplx0_polr0_time2 = zero(Float16x2)
                        Y_cplx1_polr0_time2 = zero(Float16x2)
                        Y_cplx0_polr1_time2 = zero(Float16x2)
                        Y_cplx1_polr1_time2 = zero(Float16x2)
                        Y_cplx0_polr0_time4 = zero(Float16x2)
                        Y_cplx1_polr0_time4 = zero(Float16x2)
                        Y_cplx0_polr1_time4 = zero(Float16x2)
                        Y_cplx1_polr1_time4 = zero(Float16x2)
                        Y_cplx0_polr0_time6 = zero(Float16x2)
                        Y_cplx1_polr0_time6 = zero(Float16x2)
                        Y_cplx0_polr1_time6 = zero(Float16x2)
                        Y_cplx1_polr1_time6 = zero(Float16x2)
                        Vre_polr0_time0 = V_cplx0_polr0_time0
                        Vim_polr0_time0 = V_cplx1_polr0_time0
                        Vre_polr1_time0 = V_cplx0_polr1_time0
                        Vim_polr1_time0 = V_cplx1_polr1_time0
                        Vre_polr0_time2 = V_cplx0_polr0_time2
                        Vim_polr0_time2 = V_cplx1_polr0_time2
                        Vre_polr1_time2 = V_cplx0_polr1_time2
                        Vim_polr1_time2 = V_cplx1_polr1_time2
                        Vre_polr0_time4 = V_cplx0_polr0_time4
                        Vim_polr0_time4 = V_cplx1_polr0_time4
                        Vre_polr1_time4 = V_cplx0_polr1_time4
                        Vim_polr1_time4 = V_cplx1_polr1_time4
                        Vre_polr0_time6 = V_cplx0_polr0_time6
                        Vim_polr0_time6 = V_cplx1_polr0_time6
                        Vre_polr1_time6 = V_cplx0_polr1_time6
                        Vim_polr1_time6 = V_cplx1_polr1_time6
                        V_cplx_in0_polr0_time0 = Vre_polr0_time0
                        V_cplx_in1_polr0_time0 = Vim_polr0_time0
                        V_cplx_in0_polr1_time0 = Vre_polr1_time0
                        V_cplx_in1_polr1_time0 = Vim_polr1_time0
                        V_cplx_in0_polr0_time2 = Vre_polr0_time2
                        V_cplx_in1_polr0_time2 = Vim_polr0_time2
                        V_cplx_in0_polr1_time2 = Vre_polr1_time2
                        V_cplx_in1_polr1_time2 = Vim_polr1_time2
                        V_cplx_in0_polr0_time4 = Vre_polr0_time4
                        V_cplx_in1_polr0_time4 = Vim_polr0_time4
                        V_cplx_in0_polr1_time4 = Vre_polr1_time4
                        V_cplx_in1_polr1_time4 = Vim_polr1_time4
                        V_cplx_in0_polr0_time6 = Vre_polr0_time6
                        V_cplx_in1_polr0_time6 = Vim_polr0_time6
                        V_cplx_in0_polr1_time6 = Vre_polr1_time6
                        V_cplx_in1_polr1_time6 = Vim_polr1_time6
                        (Y_cplx0_polr0_time0, Y_cplx1_polr0_time0) = let
                            e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            thread = IndexSpaces.cuda_threadidx()
                            threadgroup = thread ÷ 4
                            e = if threadgroup == 0i32
                                e0
                            elseif threadgroup == 1i32
                                e1
                            elseif threadgroup == 2i32
                                e2
                            elseif threadgroup == 3i32
                                e3
                            elseif threadgroup == 4i32
                                e4
                            elseif threadgroup == 5i32
                                e5
                            elseif threadgroup == 6i32
                                e6
                            elseif threadgroup == 7i32
                                e7
                            end
                            IndexSpaces.mma_sp_m16n8k16(
                                (aΓ³_cplx0, aΓ³_cplx1)::NTuple{2,Float16x2},
                                (V_cplx_in0_polr0_time0, V_cplx_in1_polr0_time0)::NTuple{2,Float16x2},
                                (Y_cplx0_polr0_time0, Y_cplx1_polr0_time0)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        (Y_cplx0_polr1_time0, Y_cplx1_polr1_time0) = let
                            e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            thread = IndexSpaces.cuda_threadidx()
                            threadgroup = thread ÷ 4
                            e = if threadgroup == 0i32
                                e0
                            elseif threadgroup == 1i32
                                e1
                            elseif threadgroup == 2i32
                                e2
                            elseif threadgroup == 3i32
                                e3
                            elseif threadgroup == 4i32
                                e4
                            elseif threadgroup == 5i32
                                e5
                            elseif threadgroup == 6i32
                                e6
                            elseif threadgroup == 7i32
                                e7
                            end
                            IndexSpaces.mma_sp_m16n8k16(
                                (aΓ³_cplx0, aΓ³_cplx1)::NTuple{2,Float16x2},
                                (V_cplx_in0_polr1_time0, V_cplx_in1_polr1_time0)::NTuple{2,Float16x2},
                                (Y_cplx0_polr1_time0, Y_cplx1_polr1_time0)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        (Y_cplx0_polr0_time2, Y_cplx1_polr0_time2) = let
                            e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            thread = IndexSpaces.cuda_threadidx()
                            threadgroup = thread ÷ 4
                            e = if threadgroup == 0i32
                                e0
                            elseif threadgroup == 1i32
                                e1
                            elseif threadgroup == 2i32
                                e2
                            elseif threadgroup == 3i32
                                e3
                            elseif threadgroup == 4i32
                                e4
                            elseif threadgroup == 5i32
                                e5
                            elseif threadgroup == 6i32
                                e6
                            elseif threadgroup == 7i32
                                e7
                            end
                            IndexSpaces.mma_sp_m16n8k16(
                                (aΓ³_cplx0, aΓ³_cplx1)::NTuple{2,Float16x2},
                                (V_cplx_in0_polr0_time2, V_cplx_in1_polr0_time2)::NTuple{2,Float16x2},
                                (Y_cplx0_polr0_time2, Y_cplx1_polr0_time2)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        (Y_cplx0_polr1_time2, Y_cplx1_polr1_time2) = let
                            e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            thread = IndexSpaces.cuda_threadidx()
                            threadgroup = thread ÷ 4
                            e = if threadgroup == 0i32
                                e0
                            elseif threadgroup == 1i32
                                e1
                            elseif threadgroup == 2i32
                                e2
                            elseif threadgroup == 3i32
                                e3
                            elseif threadgroup == 4i32
                                e4
                            elseif threadgroup == 5i32
                                e5
                            elseif threadgroup == 6i32
                                e6
                            elseif threadgroup == 7i32
                                e7
                            end
                            IndexSpaces.mma_sp_m16n8k16(
                                (aΓ³_cplx0, aΓ³_cplx1)::NTuple{2,Float16x2},
                                (V_cplx_in0_polr1_time2, V_cplx_in1_polr1_time2)::NTuple{2,Float16x2},
                                (Y_cplx0_polr1_time2, Y_cplx1_polr1_time2)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        (Y_cplx0_polr0_time4, Y_cplx1_polr0_time4) = let
                            e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            thread = IndexSpaces.cuda_threadidx()
                            threadgroup = thread ÷ 4
                            e = if threadgroup == 0i32
                                e0
                            elseif threadgroup == 1i32
                                e1
                            elseif threadgroup == 2i32
                                e2
                            elseif threadgroup == 3i32
                                e3
                            elseif threadgroup == 4i32
                                e4
                            elseif threadgroup == 5i32
                                e5
                            elseif threadgroup == 6i32
                                e6
                            elseif threadgroup == 7i32
                                e7
                            end
                            IndexSpaces.mma_sp_m16n8k16(
                                (aΓ³_cplx0, aΓ³_cplx1)::NTuple{2,Float16x2},
                                (V_cplx_in0_polr0_time4, V_cplx_in1_polr0_time4)::NTuple{2,Float16x2},
                                (Y_cplx0_polr0_time4, Y_cplx1_polr0_time4)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        (Y_cplx0_polr1_time4, Y_cplx1_polr1_time4) = let
                            e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            thread = IndexSpaces.cuda_threadidx()
                            threadgroup = thread ÷ 4
                            e = if threadgroup == 0i32
                                e0
                            elseif threadgroup == 1i32
                                e1
                            elseif threadgroup == 2i32
                                e2
                            elseif threadgroup == 3i32
                                e3
                            elseif threadgroup == 4i32
                                e4
                            elseif threadgroup == 5i32
                                e5
                            elseif threadgroup == 6i32
                                e6
                            elseif threadgroup == 7i32
                                e7
                            end
                            IndexSpaces.mma_sp_m16n8k16(
                                (aΓ³_cplx0, aΓ³_cplx1)::NTuple{2,Float16x2},
                                (V_cplx_in0_polr1_time4, V_cplx_in1_polr1_time4)::NTuple{2,Float16x2},
                                (Y_cplx0_polr1_time4, Y_cplx1_polr1_time4)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        (Y_cplx0_polr0_time6, Y_cplx1_polr0_time6) = let
                            e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            thread = IndexSpaces.cuda_threadidx()
                            threadgroup = thread ÷ 4
                            e = if threadgroup == 0i32
                                e0
                            elseif threadgroup == 1i32
                                e1
                            elseif threadgroup == 2i32
                                e2
                            elseif threadgroup == 3i32
                                e3
                            elseif threadgroup == 4i32
                                e4
                            elseif threadgroup == 5i32
                                e5
                            elseif threadgroup == 6i32
                                e6
                            elseif threadgroup == 7i32
                                e7
                            end
                            IndexSpaces.mma_sp_m16n8k16(
                                (aΓ³_cplx0, aΓ³_cplx1)::NTuple{2,Float16x2},
                                (V_cplx_in0_polr0_time6, V_cplx_in1_polr0_time6)::NTuple{2,Float16x2},
                                (Y_cplx0_polr0_time6, Y_cplx1_polr0_time6)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        (Y_cplx0_polr1_time6, Y_cplx1_polr1_time6) = let
                            e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                            e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                            thread = IndexSpaces.cuda_threadidx()
                            threadgroup = thread ÷ 4
                            e = if threadgroup == 0i32
                                e0
                            elseif threadgroup == 1i32
                                e1
                            elseif threadgroup == 2i32
                                e2
                            elseif threadgroup == 3i32
                                e3
                            elseif threadgroup == 4i32
                                e4
                            elseif threadgroup == 5i32
                                e5
                            elseif threadgroup == 6i32
                                e6
                            elseif threadgroup == 7i32
                                e7
                            end
                            IndexSpaces.mma_sp_m16n8k16(
                                (aΓ³_cplx0, aΓ³_cplx1)::NTuple{2,Float16x2},
                                (V_cplx_in0_polr1_time6, V_cplx_in1_polr1_time6)::NTuple{2,Float16x2},
                                (Y_cplx0_polr1_time6, Y_cplx1_polr1_time6)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        G_cplx0_polr0_time0 = Y_cplx0_polr0_time0
                        G_cplx1_polr0_time0 = Y_cplx1_polr0_time0
                        G_cplx0_polr1_time0 = Y_cplx0_polr1_time0
                        G_cplx1_polr1_time0 = Y_cplx1_polr1_time0
                        G_cplx0_polr0_time2 = Y_cplx0_polr0_time2
                        G_cplx1_polr0_time2 = Y_cplx1_polr0_time2
                        G_cplx0_polr1_time2 = Y_cplx0_polr1_time2
                        G_cplx1_polr1_time2 = Y_cplx1_polr1_time2
                        G_cplx0_polr0_time4 = Y_cplx0_polr0_time4
                        G_cplx1_polr0_time4 = Y_cplx1_polr0_time4
                        G_cplx0_polr1_time4 = Y_cplx0_polr1_time4
                        G_cplx1_polr1_time4 = Y_cplx1_polr1_time4
                        G_cplx0_polr0_time6 = Y_cplx0_polr0_time6
                        G_cplx1_polr0_time6 = Y_cplx1_polr0_time6
                        G_cplx0_polr1_time6 = Y_cplx0_polr1_time6
                        G_cplx1_polr1_time6 = Y_cplx1_polr1_time6
                        (G_cplx0_polr0_time0, G_cplx1_polr0_time0) = (
                            IndexSpaces.get_lo16(G_cplx0_polr0_time0, G_cplx1_polr0_time0),
                            IndexSpaces.get_hi16(G_cplx0_polr0_time0, G_cplx1_polr0_time0),
                        )
                        (G_cplx0_polr1_time0, G_cplx1_polr1_time0) = (
                            IndexSpaces.get_lo16(G_cplx0_polr1_time0, G_cplx1_polr1_time0),
                            IndexSpaces.get_hi16(G_cplx0_polr1_time0, G_cplx1_polr1_time0),
                        )
                        (G_cplx0_polr0_time2, G_cplx1_polr0_time2) = (
                            IndexSpaces.get_lo16(G_cplx0_polr0_time2, G_cplx1_polr0_time2),
                            IndexSpaces.get_hi16(G_cplx0_polr0_time2, G_cplx1_polr0_time2),
                        )
                        (G_cplx0_polr1_time2, G_cplx1_polr1_time2) = (
                            IndexSpaces.get_lo16(G_cplx0_polr1_time2, G_cplx1_polr1_time2),
                            IndexSpaces.get_hi16(G_cplx0_polr1_time2, G_cplx1_polr1_time2),
                        )
                        (G_cplx0_polr0_time4, G_cplx1_polr0_time4) = (
                            IndexSpaces.get_lo16(G_cplx0_polr0_time4, G_cplx1_polr0_time4),
                            IndexSpaces.get_hi16(G_cplx0_polr0_time4, G_cplx1_polr0_time4),
                        )
                        (G_cplx0_polr1_time4, G_cplx1_polr1_time4) = (
                            IndexSpaces.get_lo16(G_cplx0_polr1_time4, G_cplx1_polr1_time4),
                            IndexSpaces.get_hi16(G_cplx0_polr1_time4, G_cplx1_polr1_time4),
                        )
                        (G_cplx0_polr0_time6, G_cplx1_polr0_time6) = (
                            IndexSpaces.get_lo16(G_cplx0_polr0_time6, G_cplx1_polr0_time6),
                            IndexSpaces.get_hi16(G_cplx0_polr0_time6, G_cplx1_polr0_time6),
                        )
                        (G_cplx0_polr1_time6, G_cplx1_polr1_time6) = (
                            IndexSpaces.get_lo16(G_cplx0_polr1_time6, G_cplx1_polr1_time6),
                            IndexSpaces.get_hi16(G_cplx0_polr1_time6, G_cplx1_polr1_time6),
                        )
                        if true
                            Gsh_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx0_polr0_time0
                        end
                        if true
                            Gsh_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx1_polr0_time0
                        end
                        if true
                            Gsh_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx0_polr1_time0
                        end
                        if true
                            Gsh_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx1_polr1_time0
                        end
                        if true
                            Gsh_shared[(((((((2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx0_polr0_time2
                        end
                        if true
                            Gsh_shared[(((((((2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx1_polr0_time2
                        end
                        if true
                            Gsh_shared[(((((((2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx0_polr1_time2
                        end
                        if true
                            Gsh_shared[(((((((2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx1_polr1_time2
                        end
                        if true
                            Gsh_shared[(((((((4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx0_polr0_time4
                        end
                        if true
                            Gsh_shared[(((((((4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx1_polr0_time4
                        end
                        if true
                            Gsh_shared[(((((((4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx0_polr1_time4
                        end
                        if true
                            Gsh_shared[(((((((4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx1_polr1_time4
                        end
                        if true
                            Gsh_shared[(((((((6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx0_polr0_time6
                        end
                        if true
                            Gsh_shared[(((((((6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx1_polr0_time6
                        end
                        if true
                            Gsh_shared[(((((((6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx0_polr1_time6
                        end
                        if true
                            Gsh_shared[(((((((6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 8) % 2) + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 32, 64) ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 16 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 4) % 2) * 1032 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) % 2) * 4144 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 16) % 2) * 258 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 8) % 2) * 516 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 2) * 2064) + 0) + 0x01] =
                                G_cplx1_polr1_time6
                        end
                        IndexSpaces.cuda_sync_threads()
                        let
                            t = 0
                            G_polr0 = zero(Float16x2)
                            G_polr1 = zero(Float16x2)
                            if true
                                G_polr0 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                                G_polr1 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                            end
                            X_polr0 = G_polr0
                            X_polr1 = G_polr1
                            Z_cplx0_polr0 = zero(Float16x2)
                            Z_cplx1_polr0 = zero(Float16x2)
                            Z_cplx0_polr1 = zero(Float16x2)
                            Z_cplx1_polr1 = zero(Float16x2)
                            (Z_cplx0_polr0, Z_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr0, (Z_cplx0_polr0, Z_cplx1_polr0)
                            )
                            (Z_cplx0_polr1, Z_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr1, (Z_cplx0_polr1, Z_cplx1_polr1)
                            )
                            bΓ²re = bΓ²_cplx0
                            bΓ²im = bΓ²_cplx1
                            Zre_polr0 = Z_cplx0_polr0
                            Zim_polr0 = Z_cplx1_polr0
                            Zre_polr1 = Z_cplx0_polr1
                            Zim_polr1 = Z_cplx1_polr1
                            Vre_polr0 = muladd(bΓ²re, Zre_polr0, -bΓ²im * Zim_polr0)
                            Vre_polr1 = muladd(bΓ²re, Zre_polr1, -bΓ²im * Zim_polr1)
                            Vim_polr0 = muladd(bΓ²re, Zim_polr0, +bΓ²im * Zre_polr0)
                            Vim_polr1 = muladd(bΓ²re, Zim_polr1, +bΓ²im * Zre_polr1)
                            V_cplx0_polr0 = Vre_polr0
                            V_cplx1_polr0 = Vim_polr0
                            V_cplx0_polr1 = Vre_polr1
                            V_cplx1_polr1 = Vim_polr1
                            Y_cplx0_polr0 = zero(Float16x2)
                            Y_cplx1_polr0 = zero(Float16x2)
                            Y_cplx0_polr1 = zero(Float16x2)
                            Y_cplx1_polr1 = zero(Float16x2)
                            Vre_polr0 = V_cplx0_polr0
                            Vim_polr0 = V_cplx1_polr0
                            Vre_polr1 = V_cplx0_polr1
                            Vim_polr1 = V_cplx1_polr1
                            V_cplx_in0_polr0 = Vre_polr0
                            V_cplx_in1_polr0 = Vim_polr0
                            V_cplx_in0_polr1 = Vre_polr1
                            V_cplx_in1_polr1 = Vim_polr1
                            (Y_cplx0_polr0, Y_cplx1_polr0) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr0, V_cplx_in1_polr0)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr0, Y_cplx1_polr0)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            (Y_cplx0_polr1, Y_cplx1_polr1) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr1, V_cplx_in1_polr1)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr1, Y_cplx1_polr1)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            Ẽ_cplx0_polr0 = Y_cplx0_polr0
                            Ẽ_cplx1_polr0 = Y_cplx1_polr0
                            Ẽ_cplx0_polr1 = Y_cplx0_polr1
                            Ẽ_cplx1_polr1 = Y_cplx1_polr1
                            Ẽp0_cplx0 = Ẽ_cplx0_polr0
                            Ẽp1_cplx0 = Ẽ_cplx0_polr1
                            Ẽp0_cplx1 = Ẽ_cplx1_polr0
                            Ẽp1_cplx1 = Ẽ_cplx1_polr1
                            Ẽp0re = Ẽp0_cplx0
                            Ẽp0im = Ẽp0_cplx1
                            Ẽp1re = Ẽp1_cplx0
                            Ẽp1im = Ẽp1_cplx1
                            I = muladd(
                                Float16x2(0.005001068f0, 0.005001068f0),
                                muladd(Ẽp1im, Ẽp1im, muladd(Ẽp1re, Ẽp1re, muladd(Ẽp0im, Ẽp0im, Ẽp0re * Ẽp0re))),
                                I,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 1 == 0i32
                                if t_running == 25
                                    if let
                                        thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                        warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16)
                                        p = (2i32) * thread
                                        q = (2i32) * warp
                                        0i32 ≤ p < 32 && 0i32 ≤ q < 32
                                    end
                                        if true
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 16 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 32) * 16 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 81) % 81) % 81) * 512 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 1024) % 1024) % 1024) * 41472) + 0) + 0x01] =
                                                I
                                        end
                                    end
                                    I = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let
                            t = 1
                            G_polr0 = zero(Float16x2)
                            G_polr1 = zero(Float16x2)
                            if true
                                G_polr0 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                                G_polr1 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                            end
                            X_polr0 = G_polr0
                            X_polr1 = G_polr1
                            Z_cplx0_polr0 = zero(Float16x2)
                            Z_cplx1_polr0 = zero(Float16x2)
                            Z_cplx0_polr1 = zero(Float16x2)
                            Z_cplx1_polr1 = zero(Float16x2)
                            (Z_cplx0_polr0, Z_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr0, (Z_cplx0_polr0, Z_cplx1_polr0)
                            )
                            (Z_cplx0_polr1, Z_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr1, (Z_cplx0_polr1, Z_cplx1_polr1)
                            )
                            bΓ²re = bΓ²_cplx0
                            bΓ²im = bΓ²_cplx1
                            Zre_polr0 = Z_cplx0_polr0
                            Zim_polr0 = Z_cplx1_polr0
                            Zre_polr1 = Z_cplx0_polr1
                            Zim_polr1 = Z_cplx1_polr1
                            Vre_polr0 = muladd(bΓ²re, Zre_polr0, -bΓ²im * Zim_polr0)
                            Vre_polr1 = muladd(bΓ²re, Zre_polr1, -bΓ²im * Zim_polr1)
                            Vim_polr0 = muladd(bΓ²re, Zim_polr0, +bΓ²im * Zre_polr0)
                            Vim_polr1 = muladd(bΓ²re, Zim_polr1, +bΓ²im * Zre_polr1)
                            V_cplx0_polr0 = Vre_polr0
                            V_cplx1_polr0 = Vim_polr0
                            V_cplx0_polr1 = Vre_polr1
                            V_cplx1_polr1 = Vim_polr1
                            Y_cplx0_polr0 = zero(Float16x2)
                            Y_cplx1_polr0 = zero(Float16x2)
                            Y_cplx0_polr1 = zero(Float16x2)
                            Y_cplx1_polr1 = zero(Float16x2)
                            Vre_polr0 = V_cplx0_polr0
                            Vim_polr0 = V_cplx1_polr0
                            Vre_polr1 = V_cplx0_polr1
                            Vim_polr1 = V_cplx1_polr1
                            V_cplx_in0_polr0 = Vre_polr0
                            V_cplx_in1_polr0 = Vim_polr0
                            V_cplx_in0_polr1 = Vre_polr1
                            V_cplx_in1_polr1 = Vim_polr1
                            (Y_cplx0_polr0, Y_cplx1_polr0) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr0, V_cplx_in1_polr0)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr0, Y_cplx1_polr0)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            (Y_cplx0_polr1, Y_cplx1_polr1) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr1, V_cplx_in1_polr1)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr1, Y_cplx1_polr1)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            Ẽ_cplx0_polr0 = Y_cplx0_polr0
                            Ẽ_cplx1_polr0 = Y_cplx1_polr0
                            Ẽ_cplx0_polr1 = Y_cplx0_polr1
                            Ẽ_cplx1_polr1 = Y_cplx1_polr1
                            Ẽp0_cplx0 = Ẽ_cplx0_polr0
                            Ẽp1_cplx0 = Ẽ_cplx0_polr1
                            Ẽp0_cplx1 = Ẽ_cplx1_polr0
                            Ẽp1_cplx1 = Ẽ_cplx1_polr1
                            Ẽp0re = Ẽp0_cplx0
                            Ẽp0im = Ẽp0_cplx1
                            Ẽp1re = Ẽp1_cplx0
                            Ẽp1im = Ẽp1_cplx1
                            I = muladd(
                                Float16x2(0.005001068f0, 0.005001068f0),
                                muladd(Ẽp1im, Ẽp1im, muladd(Ẽp1re, Ẽp1re, muladd(Ẽp0im, Ẽp0im, Ẽp0re * Ẽp0re))),
                                I,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 1 == 0i32
                                if t_running == 25
                                    if let
                                        thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                        warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16)
                                        p = (2i32) * thread
                                        q = (2i32) * warp
                                        0i32 ≤ p < 32 && 0i32 ≤ q < 32
                                    end
                                        if true
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 16 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 32) * 16 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 81) % 81) % 81) * 512 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 1024) % 1024) % 1024) * 41472) + 0) + 0x01] =
                                                I
                                        end
                                    end
                                    I = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let
                            t = 2
                            G_polr0 = zero(Float16x2)
                            G_polr1 = zero(Float16x2)
                            if true
                                G_polr0 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                                G_polr1 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                            end
                            X_polr0 = G_polr0
                            X_polr1 = G_polr1
                            Z_cplx0_polr0 = zero(Float16x2)
                            Z_cplx1_polr0 = zero(Float16x2)
                            Z_cplx0_polr1 = zero(Float16x2)
                            Z_cplx1_polr1 = zero(Float16x2)
                            (Z_cplx0_polr0, Z_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr0, (Z_cplx0_polr0, Z_cplx1_polr0)
                            )
                            (Z_cplx0_polr1, Z_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr1, (Z_cplx0_polr1, Z_cplx1_polr1)
                            )
                            bΓ²re = bΓ²_cplx0
                            bΓ²im = bΓ²_cplx1
                            Zre_polr0 = Z_cplx0_polr0
                            Zim_polr0 = Z_cplx1_polr0
                            Zre_polr1 = Z_cplx0_polr1
                            Zim_polr1 = Z_cplx1_polr1
                            Vre_polr0 = muladd(bΓ²re, Zre_polr0, -bΓ²im * Zim_polr0)
                            Vre_polr1 = muladd(bΓ²re, Zre_polr1, -bΓ²im * Zim_polr1)
                            Vim_polr0 = muladd(bΓ²re, Zim_polr0, +bΓ²im * Zre_polr0)
                            Vim_polr1 = muladd(bΓ²re, Zim_polr1, +bΓ²im * Zre_polr1)
                            V_cplx0_polr0 = Vre_polr0
                            V_cplx1_polr0 = Vim_polr0
                            V_cplx0_polr1 = Vre_polr1
                            V_cplx1_polr1 = Vim_polr1
                            Y_cplx0_polr0 = zero(Float16x2)
                            Y_cplx1_polr0 = zero(Float16x2)
                            Y_cplx0_polr1 = zero(Float16x2)
                            Y_cplx1_polr1 = zero(Float16x2)
                            Vre_polr0 = V_cplx0_polr0
                            Vim_polr0 = V_cplx1_polr0
                            Vre_polr1 = V_cplx0_polr1
                            Vim_polr1 = V_cplx1_polr1
                            V_cplx_in0_polr0 = Vre_polr0
                            V_cplx_in1_polr0 = Vim_polr0
                            V_cplx_in0_polr1 = Vre_polr1
                            V_cplx_in1_polr1 = Vim_polr1
                            (Y_cplx0_polr0, Y_cplx1_polr0) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr0, V_cplx_in1_polr0)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr0, Y_cplx1_polr0)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            (Y_cplx0_polr1, Y_cplx1_polr1) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr1, V_cplx_in1_polr1)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr1, Y_cplx1_polr1)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            Ẽ_cplx0_polr0 = Y_cplx0_polr0
                            Ẽ_cplx1_polr0 = Y_cplx1_polr0
                            Ẽ_cplx0_polr1 = Y_cplx0_polr1
                            Ẽ_cplx1_polr1 = Y_cplx1_polr1
                            Ẽp0_cplx0 = Ẽ_cplx0_polr0
                            Ẽp1_cplx0 = Ẽ_cplx0_polr1
                            Ẽp0_cplx1 = Ẽ_cplx1_polr0
                            Ẽp1_cplx1 = Ẽ_cplx1_polr1
                            Ẽp0re = Ẽp0_cplx0
                            Ẽp0im = Ẽp0_cplx1
                            Ẽp1re = Ẽp1_cplx0
                            Ẽp1im = Ẽp1_cplx1
                            I = muladd(
                                Float16x2(0.005001068f0, 0.005001068f0),
                                muladd(Ẽp1im, Ẽp1im, muladd(Ẽp1re, Ẽp1re, muladd(Ẽp0im, Ẽp0im, Ẽp0re * Ẽp0re))),
                                I,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 1 == 0i32
                                if t_running == 25
                                    if let
                                        thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                        warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16)
                                        p = (2i32) * thread
                                        q = (2i32) * warp
                                        0i32 ≤ p < 32 && 0i32 ≤ q < 32
                                    end
                                        if true
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 16 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 32) * 16 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 81) % 81) % 81) * 512 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 1024) % 1024) % 1024) * 41472) + 0) + 0x01] =
                                                I
                                        end
                                    end
                                    I = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let
                            t = 3
                            G_polr0 = zero(Float16x2)
                            G_polr1 = zero(Float16x2)
                            if true
                                G_polr0 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                                G_polr1 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                            end
                            X_polr0 = G_polr0
                            X_polr1 = G_polr1
                            Z_cplx0_polr0 = zero(Float16x2)
                            Z_cplx1_polr0 = zero(Float16x2)
                            Z_cplx0_polr1 = zero(Float16x2)
                            Z_cplx1_polr1 = zero(Float16x2)
                            (Z_cplx0_polr0, Z_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr0, (Z_cplx0_polr0, Z_cplx1_polr0)
                            )
                            (Z_cplx0_polr1, Z_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr1, (Z_cplx0_polr1, Z_cplx1_polr1)
                            )
                            bΓ²re = bΓ²_cplx0
                            bΓ²im = bΓ²_cplx1
                            Zre_polr0 = Z_cplx0_polr0
                            Zim_polr0 = Z_cplx1_polr0
                            Zre_polr1 = Z_cplx0_polr1
                            Zim_polr1 = Z_cplx1_polr1
                            Vre_polr0 = muladd(bΓ²re, Zre_polr0, -bΓ²im * Zim_polr0)
                            Vre_polr1 = muladd(bΓ²re, Zre_polr1, -bΓ²im * Zim_polr1)
                            Vim_polr0 = muladd(bΓ²re, Zim_polr0, +bΓ²im * Zre_polr0)
                            Vim_polr1 = muladd(bΓ²re, Zim_polr1, +bΓ²im * Zre_polr1)
                            V_cplx0_polr0 = Vre_polr0
                            V_cplx1_polr0 = Vim_polr0
                            V_cplx0_polr1 = Vre_polr1
                            V_cplx1_polr1 = Vim_polr1
                            Y_cplx0_polr0 = zero(Float16x2)
                            Y_cplx1_polr0 = zero(Float16x2)
                            Y_cplx0_polr1 = zero(Float16x2)
                            Y_cplx1_polr1 = zero(Float16x2)
                            Vre_polr0 = V_cplx0_polr0
                            Vim_polr0 = V_cplx1_polr0
                            Vre_polr1 = V_cplx0_polr1
                            Vim_polr1 = V_cplx1_polr1
                            V_cplx_in0_polr0 = Vre_polr0
                            V_cplx_in1_polr0 = Vim_polr0
                            V_cplx_in0_polr1 = Vre_polr1
                            V_cplx_in1_polr1 = Vim_polr1
                            (Y_cplx0_polr0, Y_cplx1_polr0) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr0, V_cplx_in1_polr0)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr0, Y_cplx1_polr0)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            (Y_cplx0_polr1, Y_cplx1_polr1) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr1, V_cplx_in1_polr1)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr1, Y_cplx1_polr1)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            Ẽ_cplx0_polr0 = Y_cplx0_polr0
                            Ẽ_cplx1_polr0 = Y_cplx1_polr0
                            Ẽ_cplx0_polr1 = Y_cplx0_polr1
                            Ẽ_cplx1_polr1 = Y_cplx1_polr1
                            Ẽp0_cplx0 = Ẽ_cplx0_polr0
                            Ẽp1_cplx0 = Ẽ_cplx0_polr1
                            Ẽp0_cplx1 = Ẽ_cplx1_polr0
                            Ẽp1_cplx1 = Ẽ_cplx1_polr1
                            Ẽp0re = Ẽp0_cplx0
                            Ẽp0im = Ẽp0_cplx1
                            Ẽp1re = Ẽp1_cplx0
                            Ẽp1im = Ẽp1_cplx1
                            I = muladd(
                                Float16x2(0.005001068f0, 0.005001068f0),
                                muladd(Ẽp1im, Ẽp1im, muladd(Ẽp1re, Ẽp1re, muladd(Ẽp0im, Ẽp0im, Ẽp0re * Ẽp0re))),
                                I,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 1 == 0i32
                                if t_running == 25
                                    if let
                                        thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                        warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16)
                                        p = (2i32) * thread
                                        q = (2i32) * warp
                                        0i32 ≤ p < 32 && 0i32 ≤ q < 32
                                    end
                                        if true
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 16 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 32) * 16 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 81) % 81) % 81) * 512 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 1024) % 1024) % 1024) * 41472) + 0) + 0x01] =
                                                I
                                        end
                                    end
                                    I = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let
                            t = 4
                            G_polr0 = zero(Float16x2)
                            G_polr1 = zero(Float16x2)
                            if true
                                G_polr0 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                                G_polr1 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                            end
                            X_polr0 = G_polr0
                            X_polr1 = G_polr1
                            Z_cplx0_polr0 = zero(Float16x2)
                            Z_cplx1_polr0 = zero(Float16x2)
                            Z_cplx0_polr1 = zero(Float16x2)
                            Z_cplx1_polr1 = zero(Float16x2)
                            (Z_cplx0_polr0, Z_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr0, (Z_cplx0_polr0, Z_cplx1_polr0)
                            )
                            (Z_cplx0_polr1, Z_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr1, (Z_cplx0_polr1, Z_cplx1_polr1)
                            )
                            bΓ²re = bΓ²_cplx0
                            bΓ²im = bΓ²_cplx1
                            Zre_polr0 = Z_cplx0_polr0
                            Zim_polr0 = Z_cplx1_polr0
                            Zre_polr1 = Z_cplx0_polr1
                            Zim_polr1 = Z_cplx1_polr1
                            Vre_polr0 = muladd(bΓ²re, Zre_polr0, -bΓ²im * Zim_polr0)
                            Vre_polr1 = muladd(bΓ²re, Zre_polr1, -bΓ²im * Zim_polr1)
                            Vim_polr0 = muladd(bΓ²re, Zim_polr0, +bΓ²im * Zre_polr0)
                            Vim_polr1 = muladd(bΓ²re, Zim_polr1, +bΓ²im * Zre_polr1)
                            V_cplx0_polr0 = Vre_polr0
                            V_cplx1_polr0 = Vim_polr0
                            V_cplx0_polr1 = Vre_polr1
                            V_cplx1_polr1 = Vim_polr1
                            Y_cplx0_polr0 = zero(Float16x2)
                            Y_cplx1_polr0 = zero(Float16x2)
                            Y_cplx0_polr1 = zero(Float16x2)
                            Y_cplx1_polr1 = zero(Float16x2)
                            Vre_polr0 = V_cplx0_polr0
                            Vim_polr0 = V_cplx1_polr0
                            Vre_polr1 = V_cplx0_polr1
                            Vim_polr1 = V_cplx1_polr1
                            V_cplx_in0_polr0 = Vre_polr0
                            V_cplx_in1_polr0 = Vim_polr0
                            V_cplx_in0_polr1 = Vre_polr1
                            V_cplx_in1_polr1 = Vim_polr1
                            (Y_cplx0_polr0, Y_cplx1_polr0) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr0, V_cplx_in1_polr0)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr0, Y_cplx1_polr0)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            (Y_cplx0_polr1, Y_cplx1_polr1) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr1, V_cplx_in1_polr1)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr1, Y_cplx1_polr1)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            Ẽ_cplx0_polr0 = Y_cplx0_polr0
                            Ẽ_cplx1_polr0 = Y_cplx1_polr0
                            Ẽ_cplx0_polr1 = Y_cplx0_polr1
                            Ẽ_cplx1_polr1 = Y_cplx1_polr1
                            Ẽp0_cplx0 = Ẽ_cplx0_polr0
                            Ẽp1_cplx0 = Ẽ_cplx0_polr1
                            Ẽp0_cplx1 = Ẽ_cplx1_polr0
                            Ẽp1_cplx1 = Ẽ_cplx1_polr1
                            Ẽp0re = Ẽp0_cplx0
                            Ẽp0im = Ẽp0_cplx1
                            Ẽp1re = Ẽp1_cplx0
                            Ẽp1im = Ẽp1_cplx1
                            I = muladd(
                                Float16x2(0.005001068f0, 0.005001068f0),
                                muladd(Ẽp1im, Ẽp1im, muladd(Ẽp1re, Ẽp1re, muladd(Ẽp0im, Ẽp0im, Ẽp0re * Ẽp0re))),
                                I,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 1 == 0i32
                                if t_running == 25
                                    if let
                                        thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                        warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16)
                                        p = (2i32) * thread
                                        q = (2i32) * warp
                                        0i32 ≤ p < 32 && 0i32 ≤ q < 32
                                    end
                                        if true
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 16 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 32) * 16 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 81) % 81) % 81) * 512 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 1024) % 1024) % 1024) * 41472) + 0) + 0x01] =
                                                I
                                        end
                                    end
                                    I = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let
                            t = 5
                            G_polr0 = zero(Float16x2)
                            G_polr1 = zero(Float16x2)
                            if true
                                G_polr0 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                                G_polr1 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                            end
                            X_polr0 = G_polr0
                            X_polr1 = G_polr1
                            Z_cplx0_polr0 = zero(Float16x2)
                            Z_cplx1_polr0 = zero(Float16x2)
                            Z_cplx0_polr1 = zero(Float16x2)
                            Z_cplx1_polr1 = zero(Float16x2)
                            (Z_cplx0_polr0, Z_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr0, (Z_cplx0_polr0, Z_cplx1_polr0)
                            )
                            (Z_cplx0_polr1, Z_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr1, (Z_cplx0_polr1, Z_cplx1_polr1)
                            )
                            bΓ²re = bΓ²_cplx0
                            bΓ²im = bΓ²_cplx1
                            Zre_polr0 = Z_cplx0_polr0
                            Zim_polr0 = Z_cplx1_polr0
                            Zre_polr1 = Z_cplx0_polr1
                            Zim_polr1 = Z_cplx1_polr1
                            Vre_polr0 = muladd(bΓ²re, Zre_polr0, -bΓ²im * Zim_polr0)
                            Vre_polr1 = muladd(bΓ²re, Zre_polr1, -bΓ²im * Zim_polr1)
                            Vim_polr0 = muladd(bΓ²re, Zim_polr0, +bΓ²im * Zre_polr0)
                            Vim_polr1 = muladd(bΓ²re, Zim_polr1, +bΓ²im * Zre_polr1)
                            V_cplx0_polr0 = Vre_polr0
                            V_cplx1_polr0 = Vim_polr0
                            V_cplx0_polr1 = Vre_polr1
                            V_cplx1_polr1 = Vim_polr1
                            Y_cplx0_polr0 = zero(Float16x2)
                            Y_cplx1_polr0 = zero(Float16x2)
                            Y_cplx0_polr1 = zero(Float16x2)
                            Y_cplx1_polr1 = zero(Float16x2)
                            Vre_polr0 = V_cplx0_polr0
                            Vim_polr0 = V_cplx1_polr0
                            Vre_polr1 = V_cplx0_polr1
                            Vim_polr1 = V_cplx1_polr1
                            V_cplx_in0_polr0 = Vre_polr0
                            V_cplx_in1_polr0 = Vim_polr0
                            V_cplx_in0_polr1 = Vre_polr1
                            V_cplx_in1_polr1 = Vim_polr1
                            (Y_cplx0_polr0, Y_cplx1_polr0) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr0, V_cplx_in1_polr0)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr0, Y_cplx1_polr0)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            (Y_cplx0_polr1, Y_cplx1_polr1) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr1, V_cplx_in1_polr1)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr1, Y_cplx1_polr1)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            Ẽ_cplx0_polr0 = Y_cplx0_polr0
                            Ẽ_cplx1_polr0 = Y_cplx1_polr0
                            Ẽ_cplx0_polr1 = Y_cplx0_polr1
                            Ẽ_cplx1_polr1 = Y_cplx1_polr1
                            Ẽp0_cplx0 = Ẽ_cplx0_polr0
                            Ẽp1_cplx0 = Ẽ_cplx0_polr1
                            Ẽp0_cplx1 = Ẽ_cplx1_polr0
                            Ẽp1_cplx1 = Ẽ_cplx1_polr1
                            Ẽp0re = Ẽp0_cplx0
                            Ẽp0im = Ẽp0_cplx1
                            Ẽp1re = Ẽp1_cplx0
                            Ẽp1im = Ẽp1_cplx1
                            I = muladd(
                                Float16x2(0.005001068f0, 0.005001068f0),
                                muladd(Ẽp1im, Ẽp1im, muladd(Ẽp1re, Ẽp1re, muladd(Ẽp0im, Ẽp0im, Ẽp0re * Ẽp0re))),
                                I,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 1 == 0i32
                                if t_running == 25
                                    if let
                                        thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                        warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16)
                                        p = (2i32) * thread
                                        q = (2i32) * warp
                                        0i32 ≤ p < 32 && 0i32 ≤ q < 32
                                    end
                                        if true
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 16 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 32) * 16 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 81) % 81) % 81) * 512 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 1024) % 1024) % 1024) * 41472) + 0) + 0x01] =
                                                I
                                        end
                                    end
                                    I = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let
                            t = 6
                            G_polr0 = zero(Float16x2)
                            G_polr1 = zero(Float16x2)
                            if true
                                G_polr0 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                                G_polr1 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                            end
                            X_polr0 = G_polr0
                            X_polr1 = G_polr1
                            Z_cplx0_polr0 = zero(Float16x2)
                            Z_cplx1_polr0 = zero(Float16x2)
                            Z_cplx0_polr1 = zero(Float16x2)
                            Z_cplx1_polr1 = zero(Float16x2)
                            (Z_cplx0_polr0, Z_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr0, (Z_cplx0_polr0, Z_cplx1_polr0)
                            )
                            (Z_cplx0_polr1, Z_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr1, (Z_cplx0_polr1, Z_cplx1_polr1)
                            )
                            bΓ²re = bΓ²_cplx0
                            bΓ²im = bΓ²_cplx1
                            Zre_polr0 = Z_cplx0_polr0
                            Zim_polr0 = Z_cplx1_polr0
                            Zre_polr1 = Z_cplx0_polr1
                            Zim_polr1 = Z_cplx1_polr1
                            Vre_polr0 = muladd(bΓ²re, Zre_polr0, -bΓ²im * Zim_polr0)
                            Vre_polr1 = muladd(bΓ²re, Zre_polr1, -bΓ²im * Zim_polr1)
                            Vim_polr0 = muladd(bΓ²re, Zim_polr0, +bΓ²im * Zre_polr0)
                            Vim_polr1 = muladd(bΓ²re, Zim_polr1, +bΓ²im * Zre_polr1)
                            V_cplx0_polr0 = Vre_polr0
                            V_cplx1_polr0 = Vim_polr0
                            V_cplx0_polr1 = Vre_polr1
                            V_cplx1_polr1 = Vim_polr1
                            Y_cplx0_polr0 = zero(Float16x2)
                            Y_cplx1_polr0 = zero(Float16x2)
                            Y_cplx0_polr1 = zero(Float16x2)
                            Y_cplx1_polr1 = zero(Float16x2)
                            Vre_polr0 = V_cplx0_polr0
                            Vim_polr0 = V_cplx1_polr0
                            Vre_polr1 = V_cplx0_polr1
                            Vim_polr1 = V_cplx1_polr1
                            V_cplx_in0_polr0 = Vre_polr0
                            V_cplx_in1_polr0 = Vim_polr0
                            V_cplx_in0_polr1 = Vre_polr1
                            V_cplx_in1_polr1 = Vim_polr1
                            (Y_cplx0_polr0, Y_cplx1_polr0) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr0, V_cplx_in1_polr0)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr0, Y_cplx1_polr0)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            (Y_cplx0_polr1, Y_cplx1_polr1) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr1, V_cplx_in1_polr1)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr1, Y_cplx1_polr1)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            Ẽ_cplx0_polr0 = Y_cplx0_polr0
                            Ẽ_cplx1_polr0 = Y_cplx1_polr0
                            Ẽ_cplx0_polr1 = Y_cplx0_polr1
                            Ẽ_cplx1_polr1 = Y_cplx1_polr1
                            Ẽp0_cplx0 = Ẽ_cplx0_polr0
                            Ẽp1_cplx0 = Ẽ_cplx0_polr1
                            Ẽp0_cplx1 = Ẽ_cplx1_polr0
                            Ẽp1_cplx1 = Ẽ_cplx1_polr1
                            Ẽp0re = Ẽp0_cplx0
                            Ẽp0im = Ẽp0_cplx1
                            Ẽp1re = Ẽp1_cplx0
                            Ẽp1im = Ẽp1_cplx1
                            I = muladd(
                                Float16x2(0.005001068f0, 0.005001068f0),
                                muladd(Ẽp1im, Ẽp1im, muladd(Ẽp1re, Ẽp1re, muladd(Ẽp0im, Ẽp0im, Ẽp0re * Ẽp0re))),
                                I,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 1 == 0i32
                                if t_running == 25
                                    if let
                                        thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                        warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16)
                                        p = (2i32) * thread
                                        q = (2i32) * warp
                                        0i32 ≤ p < 32 && 0i32 ≤ q < 32
                                    end
                                        if true
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 16 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 32) * 16 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 81) % 81) % 81) * 512 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 1024) % 1024) % 1024) * 41472) + 0) + 0x01] =
                                                I
                                        end
                                    end
                                    I = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let
                            t = 7
                            G_polr0 = zero(Float16x2)
                            G_polr1 = zero(Float16x2)
                            if true
                                G_polr0 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 0 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                                G_polr1 = Gsh_shared[(((((t % 8 + ((IndexSpaces.assume_inrange(t_inner_lo, 0, 8, 32) ÷ 8) % 4) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 64, 2048) ÷ 64) % 32) * 64) + ((t_inner_hi ÷ 32) % 2) * 32) % 8) * 32 + 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 4 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 2) * 1032 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) % 2) * 4144 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 16) % 2) * 258 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 8) % 2) * 516 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 2064) + 0x01]
                            end
                            X_polr0 = G_polr0
                            X_polr1 = G_polr1
                            Z_cplx0_polr0 = zero(Float16x2)
                            Z_cplx1_polr0 = zero(Float16x2)
                            Z_cplx0_polr1 = zero(Float16x2)
                            Z_cplx1_polr1 = zero(Float16x2)
                            (Z_cplx0_polr0, Z_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr0, (Z_cplx0_polr0, Z_cplx1_polr0)
                            )
                            (Z_cplx0_polr1, Z_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (bΓ¹_cplx0, bΓ¹_cplx1), X_polr1, (Z_cplx0_polr1, Z_cplx1_polr1)
                            )
                            bΓ²re = bΓ²_cplx0
                            bΓ²im = bΓ²_cplx1
                            Zre_polr0 = Z_cplx0_polr0
                            Zim_polr0 = Z_cplx1_polr0
                            Zre_polr1 = Z_cplx0_polr1
                            Zim_polr1 = Z_cplx1_polr1
                            Vre_polr0 = muladd(bΓ²re, Zre_polr0, -bΓ²im * Zim_polr0)
                            Vre_polr1 = muladd(bΓ²re, Zre_polr1, -bΓ²im * Zim_polr1)
                            Vim_polr0 = muladd(bΓ²re, Zim_polr0, +bΓ²im * Zre_polr0)
                            Vim_polr1 = muladd(bΓ²re, Zim_polr1, +bΓ²im * Zre_polr1)
                            V_cplx0_polr0 = Vre_polr0
                            V_cplx1_polr0 = Vim_polr0
                            V_cplx0_polr1 = Vre_polr1
                            V_cplx1_polr1 = Vim_polr1
                            Y_cplx0_polr0 = zero(Float16x2)
                            Y_cplx1_polr0 = zero(Float16x2)
                            Y_cplx0_polr1 = zero(Float16x2)
                            Y_cplx1_polr1 = zero(Float16x2)
                            Vre_polr0 = V_cplx0_polr0
                            Vim_polr0 = V_cplx1_polr0
                            Vre_polr1 = V_cplx0_polr1
                            Vim_polr1 = V_cplx1_polr1
                            V_cplx_in0_polr0 = Vre_polr0
                            V_cplx_in1_polr0 = Vim_polr0
                            V_cplx_in0_polr1 = Vre_polr1
                            V_cplx_in1_polr1 = Vim_polr1
                            (Y_cplx0_polr0, Y_cplx1_polr0) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr0, V_cplx_in1_polr0)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr0, Y_cplx1_polr0)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            (Y_cplx0_polr1, Y_cplx1_polr1) = let
                                e0 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e1 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e2 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e3 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e4 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e5 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e6 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e7 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                thread = IndexSpaces.cuda_threadidx()
                                threadgroup = thread ÷ 4
                                e = if threadgroup == 0i32
                                    e0
                                elseif threadgroup == 1i32
                                    e1
                                elseif threadgroup == 2i32
                                    e2
                                elseif threadgroup == 3i32
                                    e3
                                elseif threadgroup == 4i32
                                    e4
                                elseif threadgroup == 5i32
                                    e5
                                elseif threadgroup == 6i32
                                    e6
                                elseif threadgroup == 7i32
                                    e7
                                end
                                IndexSpaces.mma_sp_m16n8k16(
                                    (bΓ³_cplx0, bΓ³_cplx1)::NTuple{2,Float16x2},
                                    (V_cplx_in0_polr1, V_cplx_in1_polr1)::NTuple{2,Float16x2},
                                    (Y_cplx0_polr1, Y_cplx1_polr1)::NTuple{2,Float16x2},
                                    e::Int2x16,
                                    0i32,
                                )::NTuple{2,Float16x2}
                            end
                            Ẽ_cplx0_polr0 = Y_cplx0_polr0
                            Ẽ_cplx1_polr0 = Y_cplx1_polr0
                            Ẽ_cplx0_polr1 = Y_cplx0_polr1
                            Ẽ_cplx1_polr1 = Y_cplx1_polr1
                            Ẽp0_cplx0 = Ẽ_cplx0_polr0
                            Ẽp1_cplx0 = Ẽ_cplx0_polr1
                            Ẽp0_cplx1 = Ẽ_cplx1_polr0
                            Ẽp1_cplx1 = Ẽ_cplx1_polr1
                            Ẽp0re = Ẽp0_cplx0
                            Ẽp0im = Ẽp0_cplx1
                            Ẽp1re = Ẽp1_cplx0
                            Ẽp1im = Ẽp1_cplx1
                            I = muladd(
                                Float16x2(0.005001068f0, 0.005001068f0),
                                muladd(Ẽp1im, Ẽp1im, muladd(Ẽp1re, Ẽp1re, muladd(Ẽp0im, Ẽp0im, Ẽp0re * Ẽp0re))),
                                I,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 1 == 0i32
                                if t_running == 25
                                    if let
                                        thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                        warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16)
                                        p = (2i32) * thread
                                        q = (2i32) * warp
                                        0i32 ≤ p < 32 && 0i32 ≤ q < 32
                                    end
                                        if true
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 2) ÷ 2) % 16 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 32) * 16 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 81) % 81) % 81) * 512 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 1024) % 1024) % 1024) * 41472) + 0) + 0x01] =
                                                I
                                        end
                                    end
                                    I = zero(Float16x2)
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
            info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) % 16) * 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 1024) % 1024) % 1024) * 512 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                info
        end
    end
)