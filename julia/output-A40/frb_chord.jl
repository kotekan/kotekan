@fastmath @inbounds(
    begin #= /home/dstn/kotekan/julia/kernels/frb.jl:1456 =#
        info = 1
        if true
            info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 256) % 256) * 768 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32) + 0) + 0x01] =
                info
        end
        (Γ¹_re_re, Γ¹_re_im, Γ¹_im_re, Γ¹_im_im) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            c = thread % (4i32)
            v = thread ÷ (4i32)
            Γ¹ = cispi((((c * v) % 8) / 4.0f0) % 2.0f0)
            (+(Γ¹.re), -(Γ¹.im), +(Γ¹.im), +(Γ¹.re))
        end
        Γ¹_re = Float16x2(Γ¹_re_im, Γ¹_re_re)
        Γ¹_im = Float16x2(Γ¹_im_im, Γ¹_im_re)
        Γ¹_cplx0 = Γ¹_re
        Γ¹_cplx1 = Γ¹_im
        (Γ²_d0_re, Γ²_d0_im, Γ²_d1_re, Γ²_d1_im) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            d0 = (thread % (4i32)) * (2i32) + 0i32
            d1 = (thread % (4i32)) * (2i32) + 1i32
            v = thread ÷ (4i32)
            Γ²_d0 = if d0 < 6
                cispi((((d0 * v) % 48) / 24.0f0) % 2.0f0)
            else
                Complex(0.0f0)
            end
            Γ²_d1 = if d1 < 6
                cispi((((d1 * v) % 48) / 24.0f0) % 2.0f0)
            else
                Complex(0.0f0)
            end
            (Γ²_d0.re, Γ²_d0.im, Γ²_d1.re, Γ²_d1.im)
        end
        Γ²_re = Float16x2(Γ²_d0_re, Γ²_d1_re)
        Γ²_im = Float16x2(Γ²_d0_im, Γ²_d1_im)
        Γ²_cplx0 = Γ²_re
        Γ²_cplx1 = Γ²_im
        (Γ³_d0_re_re, Γ³_d0_re_im, Γ³_d0_im_re, Γ³_d0_im_im, Γ³_d1_re_re, Γ³_d1_re_im, Γ³_d1_im_re, Γ³_d1_im_im) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            d0 = (thread % (4i32)) * (2i32) + 0i32
            d1 = (thread % (4i32)) * (2i32) + 1i32
            u = thread ÷ (4i32)
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
        Γ³_cplx0_cplx_in0 = Γ³_re_cplx_in0
        Γ³_cplx1_cplx_in0 = Γ³_im_cplx_in0
        Γ³_cplx0_cplx_in1 = Γ³_re_cplx_in1
        Γ³_cplx1_cplx_in1 = Γ³_im_cplx_in1
        S = 999999999
        if let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            thread < 24
        end
            Smn = Smn_memory[(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24) * 24) % 576 + 0x01]
            (Smn_mn0, Smn_mn1) = convert(NTuple{2,Int32}, Smn)
            Sm = Smn_mn0
            Sn = Smn_mn1
            S = (33i32) * Sm + 801 * Sn
        end
        W_polr0 = zero(Float16x2)
        W_polr1 = zero(Float16x2)
        if let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            nlo = 2 * (thread ÷ 8)
            nlo < 6
        end
            W_polr0 = W_memory[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 144 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + 0 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 256) % 256) * 1152 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6) + 0x01]
            W_polr1 = W_memory[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 144 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + 576 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 256) % 256) * 1152 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6) + 0x01]
        end
        I_beamQ0 = zero(Float16x2)
        I_beamQ1 = zero(Float16x2)
        dstime = 0
        t_running = 0
        for t_outer in 0:48:2063
            let
                (E_dish0_time0, E_dish4_time0, E_dish8_time0, E_dish12_time0) = IndexSpaces.unsafe_load4_global(
                    E_memory,
                    (
                        (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 2) * 128 +
                        (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) ÷ 4) % 128 +
                        ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 256) % 256) * 256 +
                        (
                            (
                                ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 +
                                IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24
                            ) % 2064
                        ) * 65536
                    ) + 1i32,
                )
                (E_dish256_time0, E_dish260_time0, E_dish264_time0, E_dish268_time0) = IndexSpaces.unsafe_load4_global(
                    E_memory,
                    (
                        (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 2) * 128 +
                        ((256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) ÷ 4) % 128 +
                        ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 256) % 256) * 256 +
                        (
                            (
                                ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 +
                                IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24
                            ) % 2064
                        ) * 65536
                    ) + 1i32,
                )
                (E_dish0_time24, E_dish4_time24, E_dish8_time24, E_dish12_time24) = IndexSpaces.unsafe_load4_global(
                    E_memory,
                    (
                        (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 2) * 128 +
                        (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) ÷ 4) % 128 +
                        ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 256) % 256) * 256 +
                        (
                            (
                                (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + 24) +
                                IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24
                            ) % 2064
                        ) * 65536
                    ) + 1i32,
                )
                (E_dish256_time24, E_dish260_time24, E_dish264_time24, E_dish268_time24) = IndexSpaces.unsafe_load4_global(
                    E_memory,
                    (
                        (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) % 2) * 128 +
                        ((256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) ÷ 4) % 128 +
                        ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 256) % 256) * 256 +
                        (
                            (
                                (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + 24) +
                                IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24
                            ) % 2064
                        ) * 65536
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
                    Fsh1_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 64) * 257 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish0_time0
                end
                if true
                    Fsh1_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16 + 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 64) * 257 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16 + 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish4_time0
                end
                if true
                    Fsh1_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16 + 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 64) * 257 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16 + 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish8_time0
                end
                if true
                    Fsh1_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16 + 4) + 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 64) * 257 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16 + 4) + 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish12_time0
                end
                if true
                    Fsh1_shared[((((((256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 64) * 257 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24 + (((256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish256_time0
                end
                if true
                    Fsh1_shared[(((((((256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 64) * 257 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24 + ((((256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish260_time0
                end
                if true
                    Fsh1_shared[(((((((256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 64) * 257 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24 + ((((256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish264_time0
                end
                if true
                    Fsh1_shared[((((((((256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 4) + 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 64) * 257 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24 + (((((256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 4) + 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish268_time0
                end
                if true
                    Fsh1_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16 + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 64) * 257 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16 + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish0_time24
                end
                if true
                    Fsh1_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16 + 4) + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 64) * 257 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16 + 4) + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish4_time24
                end
                if true
                    Fsh1_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16 + 2) + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 64) * 257 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16 + 2) + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish8_time24
                end
                if true
                    Fsh1_shared[(((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16 + 4) + 2) + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 64) * 257 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16 + 4) + 2) + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish12_time24
                end
                if true
                    Fsh1_shared[(((((((256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 64) * 257 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24 + ((((256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish256_time24
                end
                if true
                    Fsh1_shared[((((((((256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 4) + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 64) * 257 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24 + (((((256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 4) + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish260_time24
                end
                if true
                    Fsh1_shared[((((((((256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 2) + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 64) * 257 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24 + (((((256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 2) + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish264_time24
                end
                if true
                    Fsh1_shared[(((((((((256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 4) + 2) + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) ÷ 8) % 64) * 257 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24 + ((((((256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) + 4) + 2) + 1) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) % 8) * 32) + 0) + 0x01] =
                        E_dish268_time24
                end
                IndexSpaces.cuda_sync_threads()
            end
            let
                Freg1_dish0 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) ÷ 8) % 64) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 8) * 32) + 0x01]
                Freg1_dish24 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 24) ÷ 8) % 64) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 24) % 8) * 32) + 0x01]
                Freg1_dish48 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 48) ÷ 8) % 64) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 48) % 8) * 32) + 0x01]
                Freg1_dish72 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 72) ÷ 8) % 64) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 72) % 8) * 32) + 0x01]
                Freg1_dish96 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 96) ÷ 8) % 64) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 96) % 8) * 32) + 0x01]
                Freg1_dish120 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 120) ÷ 8) % 64) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 120) % 8) * 32) + 0x01]
                Freg1_dish144 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 144) ÷ 8) % 64) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 144) % 8) * 32) + 0x01]
                Freg1_dish168 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 168) ÷ 8) % 64) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 168) % 8) * 32) + 0x01]
                Freg1_dish192 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 192) ÷ 8) % 64) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 192) % 8) * 32) + 0x01]
                Freg1_dish216 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 216) ÷ 8) % 64) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 216) % 8) * 32) + 0x01]
                Freg1_dish240 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 240) ÷ 8) % 64) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 240) % 8) * 32) + 0x01]
                Freg1_dish264 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 264) ÷ 8) % 64) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 264) % 8) * 32) + 0x01]
                Freg1_dish288 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 288) ÷ 8) % 64) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 288) % 8) * 32) + 0x01]
                Freg1_dish312 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 312) ÷ 8) % 64) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 312) % 8) * 32) + 0x01]
                Freg1_dish336 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 336) ÷ 8) % 64) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 336) % 8) * 32) + 0x01]
                Freg1_dish360 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 360) ÷ 8) % 64) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 360) % 8) * 32) + 0x01]
                Freg1_dish384 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 384) ÷ 8) % 64) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 384) % 8) * 32) + 0x01]
                Freg1_dish408 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 408) ÷ 8) % 64) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 408) % 8) * 32) + 0x01]
                Freg1_dish432 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 432) ÷ 8) % 64) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 432) % 8) * 32) + 0x01]
                Freg1_dish456 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 456) ÷ 8) % 64) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 456) % 8) * 32) + 0x01]
                Freg1_dish480 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 480) ÷ 8) % 64) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 480) % 8) * 32) + 0x01]
                Freg1_dish504 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 504) ÷ 8) % 64) * 257 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24 + 504) % 8) * 32) + 0x01]
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
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + sd_sd0) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish24
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + sd_sd1) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish48
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + sd_sd2) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish72
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + sd_sd3) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish96
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + sd_sd4) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish120
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + sd_sd5) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish144
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + sd_sd6) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish168
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + sd_sd7) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish192
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + sd_sd8) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish216
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + sd_sd9) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish240
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + sd_sd10) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish264
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + sd_sd11) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish288
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + sd_sd12) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish312
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + sd_sd13) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish336
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + sd_sd14) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish360
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + sd_sd15) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish384
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + sd_sd16) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish408
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + sd_sd17) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish432
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + sd_sd18) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish456
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + sd_sd19) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish480
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + sd_sd20) + 0x01] =
                        Freg1′
                end
                Freg1′ = if warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24), dish = warp + 24 * 21, dish < 512
                    Freg1_dish504
                else
                    Int4x8(0, 0, 0, 0, 0, 0, 0, 0)
                end
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + sd_sd21) + 0x01] =
                        Freg1′
                end
                Freg1′ = Int4x8(0, 0, 0, 0, 0, 0, 0, 0)
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + sd_sd22) + 0x01] =
                        Freg1′
                end
                Freg1′ = Int4x8(0, 0, 0, 0, 0, 0, 0, 0)
                if true
                    Fsh2_shared[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 24 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24 + sd_sd23) + 0x01] =
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
                    thread ÷ 8 < 3
                end
                    Freg2_time0 = Fsh2_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 198 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6) * 33 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) % 24) + 0x01]
                    Freg2_time1 = Fsh2_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 198 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6) * 33 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + 1) % 24) + 0x01]
                    Freg2_time2 = Fsh2_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 198 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6) * 33 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + 2) % 24) + 0x01]
                    Freg2_time3 = Fsh2_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 198 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6) * 33 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + 3) % 24) + 0x01]
                    Freg2_time4 = Fsh2_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 198 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6) * 33 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + 4) % 24) + 0x01]
                    Freg2_time5 = Fsh2_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 198 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6) * 33 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + 5) % 24) + 0x01]
                    Freg2_time6 = Fsh2_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 198 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6) * 33 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + 6) % 24) + 0x01]
                    Freg2_time7 = Fsh2_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 198 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6) * 33 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + 7) % 24) + 0x01]
                    Freg2_time8 = Fsh2_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 198 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6) * 33 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + 8) % 24) + 0x01]
                    Freg2_time9 = Fsh2_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 198 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6) * 33 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + 9) % 24) + 0x01]
                    Freg2_time10 = Fsh2_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 198 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6) * 33 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + 10) % 24) + 0x01]
                    Freg2_time11 = Fsh2_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 198 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6) * 33 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + 11) % 24) + 0x01]
                    Freg2_time12 = Fsh2_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 198 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6) * 33 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + 12) % 24) + 0x01]
                    Freg2_time13 = Fsh2_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 198 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6) * 33 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + 13) % 24) + 0x01]
                    Freg2_time14 = Fsh2_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 198 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6) * 33 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + 14) % 24) + 0x01]
                    Freg2_time15 = Fsh2_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 198 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6) * 33 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + 15) % 24) + 0x01]
                    Freg2_time16 = Fsh2_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 198 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6) * 33 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + 16) % 24) + 0x01]
                    Freg2_time17 = Fsh2_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 198 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6) * 33 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + 17) % 24) + 0x01]
                    Freg2_time18 = Fsh2_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 198 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6) * 33 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + 18) % 24) + 0x01]
                    Freg2_time19 = Fsh2_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 198 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6) * 33 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + 19) % 24) + 0x01]
                    Freg2_time20 = Fsh2_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 198 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6) * 33 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + 20) % 24) + 0x01]
                    Freg2_time21 = Fsh2_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 198 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6) * 33 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + 21) % 24) + 0x01]
                    Freg2_time22 = Fsh2_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 198 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6) * 33 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + 22) % 24) + 0x01]
                    Freg2_time23 = Fsh2_shared[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 4806 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6) * 801 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 198 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6) * 33 + (((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48 + 23) % 24) + 0x01]
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
                            (Γ¹_cplx0, Γ¹_cplx1), X_polr0_time0, (Z_cplx0_polr0_time0, Z_cplx1_polr0_time0)
                        )
                        (Z_cplx0_polr1_time0, Z_cplx1_polr1_time0) = IndexSpaces.mma_m16n8k8(
                            (Γ¹_cplx0, Γ¹_cplx1), X_polr1_time0, (Z_cplx0_polr1_time0, Z_cplx1_polr1_time0)
                        )
                        (Z_cplx0_polr0_time1, Z_cplx1_polr0_time1) = IndexSpaces.mma_m16n8k8(
                            (Γ¹_cplx0, Γ¹_cplx1), X_polr0_time1, (Z_cplx0_polr0_time1, Z_cplx1_polr0_time1)
                        )
                        (Z_cplx0_polr1_time1, Z_cplx1_polr1_time1) = IndexSpaces.mma_m16n8k8(
                            (Γ¹_cplx0, Γ¹_cplx1), X_polr1_time1, (Z_cplx0_polr1_time1, Z_cplx1_polr1_time1)
                        )
                        (Z_cplx0_polr0_time2, Z_cplx1_polr0_time2) = IndexSpaces.mma_m16n8k8(
                            (Γ¹_cplx0, Γ¹_cplx1), X_polr0_time2, (Z_cplx0_polr0_time2, Z_cplx1_polr0_time2)
                        )
                        (Z_cplx0_polr1_time2, Z_cplx1_polr1_time2) = IndexSpaces.mma_m16n8k8(
                            (Γ¹_cplx0, Γ¹_cplx1), X_polr1_time2, (Z_cplx0_polr1_time2, Z_cplx1_polr1_time2)
                        )
                        (Z_cplx0_polr0_time3, Z_cplx1_polr0_time3) = IndexSpaces.mma_m16n8k8(
                            (Γ¹_cplx0, Γ¹_cplx1), X_polr0_time3, (Z_cplx0_polr0_time3, Z_cplx1_polr0_time3)
                        )
                        (Z_cplx0_polr1_time3, Z_cplx1_polr1_time3) = IndexSpaces.mma_m16n8k8(
                            (Γ¹_cplx0, Γ¹_cplx1), X_polr1_time3, (Z_cplx0_polr1_time3, Z_cplx1_polr1_time3)
                        )
                        Γ²re = Γ²_cplx0
                        Γ²im = Γ²_cplx1
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
                        Vre_polr0_time0 = muladd(Γ²re, Zre_polr0_time0, -Γ²im * Zim_polr0_time0)
                        Vre_polr1_time0 = muladd(Γ²re, Zre_polr1_time0, -Γ²im * Zim_polr1_time0)
                        Vre_polr0_time1 = muladd(Γ²re, Zre_polr0_time1, -Γ²im * Zim_polr0_time1)
                        Vre_polr1_time1 = muladd(Γ²re, Zre_polr1_time1, -Γ²im * Zim_polr1_time1)
                        Vre_polr0_time2 = muladd(Γ²re, Zre_polr0_time2, -Γ²im * Zim_polr0_time2)
                        Vre_polr1_time2 = muladd(Γ²re, Zre_polr1_time2, -Γ²im * Zim_polr1_time2)
                        Vre_polr0_time3 = muladd(Γ²re, Zre_polr0_time3, -Γ²im * Zim_polr0_time3)
                        Vre_polr1_time3 = muladd(Γ²re, Zre_polr1_time3, -Γ²im * Zim_polr1_time3)
                        Vim_polr0_time0 = muladd(Γ²re, Zim_polr0_time0, +Γ²im * Zre_polr0_time0)
                        Vim_polr1_time0 = muladd(Γ²re, Zim_polr1_time0, +Γ²im * Zre_polr1_time0)
                        Vim_polr0_time1 = muladd(Γ²re, Zim_polr0_time1, +Γ²im * Zre_polr0_time1)
                        Vim_polr1_time1 = muladd(Γ²re, Zim_polr1_time1, +Γ²im * Zre_polr1_time1)
                        Vim_polr0_time2 = muladd(Γ²re, Zim_polr0_time2, +Γ²im * Zre_polr0_time2)
                        Vim_polr1_time2 = muladd(Γ²re, Zim_polr1_time2, +Γ²im * Zre_polr1_time2)
                        Vim_polr0_time3 = muladd(Γ²re, Zim_polr0_time3, +Γ²im * Zre_polr0_time3)
                        Vim_polr1_time3 = muladd(Γ²re, Zim_polr1_time3, +Γ²im * Zre_polr1_time3)
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
                            (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr0_time0, V_cplx_in1_polr0_time0),
                            (Y_cplx0_polr0_time0, Y_cplx1_polr0_time0),
                        )
                        (Y_cplx0_polr1_time0, Y_cplx1_polr1_time0) = IndexSpaces.mma_m16n8k16(
                            (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr1_time0, V_cplx_in1_polr1_time0),
                            (Y_cplx0_polr1_time0, Y_cplx1_polr1_time0),
                        )
                        (Y_cplx0_polr0_time1, Y_cplx1_polr0_time1) = IndexSpaces.mma_m16n8k16(
                            (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr0_time1, V_cplx_in1_polr0_time1),
                            (Y_cplx0_polr0_time1, Y_cplx1_polr0_time1),
                        )
                        (Y_cplx0_polr1_time1, Y_cplx1_polr1_time1) = IndexSpaces.mma_m16n8k16(
                            (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr1_time1, V_cplx_in1_polr1_time1),
                            (Y_cplx0_polr1_time1, Y_cplx1_polr1_time1),
                        )
                        (Y_cplx0_polr0_time2, Y_cplx1_polr0_time2) = IndexSpaces.mma_m16n8k16(
                            (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr0_time2, V_cplx_in1_polr0_time2),
                            (Y_cplx0_polr0_time2, Y_cplx1_polr0_time2),
                        )
                        (Y_cplx0_polr1_time2, Y_cplx1_polr1_time2) = IndexSpaces.mma_m16n8k16(
                            (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr1_time2, V_cplx_in1_polr1_time2),
                            (Y_cplx0_polr1_time2, Y_cplx1_polr1_time2),
                        )
                        (Y_cplx0_polr0_time3, Y_cplx1_polr0_time3) = IndexSpaces.mma_m16n8k16(
                            (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr0_time3, V_cplx_in1_polr0_time3),
                            (Y_cplx0_polr0_time3, Y_cplx1_polr0_time3),
                        )
                        (Y_cplx0_polr1_time3, Y_cplx1_polr1_time3) = IndexSpaces.mma_m16n8k16(
                            (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
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
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 16) % 2) * 514 + 0 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 2) * 4112 + (((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx0_polr0_time0
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 16) % 2) * 514 + 0 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 2) % 2) * 4112 + (((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx1_polr0_time0
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 16) % 2) * 514 + 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 2) * 4112 + (((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx0_polr1_time0
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 16) % 2) * 514 + 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 2) % 2) * 4112 + (((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx1_polr1_time0
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 16) % 2) * 514 + 0 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + 1) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx0_polr0_time1
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 16) % 2) * 514 + 0 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + 1) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx1_polr0_time1
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 16) % 2) * 514 + 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + 1) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx0_polr1_time1
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 16) % 2) * 514 + 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + 1) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx1_polr1_time1
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 16) % 2) * 514 + 0 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx0_polr0_time2
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 16) % 2) * 514 + 0 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx1_polr0_time2
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 16) % 2) * 514 + 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx0_polr1_time2
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 16) % 2) * 514 + 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx1_polr1_time2
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 16) % 2) * 514 + 0 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + 3) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx0_polr0_time3
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 16) % 2) * 514 + 0 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + 3) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx1_polr0_time3
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 16) % 2) * 514 + 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + 3) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx0_polr1_time3
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 16) % 2) * 514 + 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + 3) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx1_polr1_time3
                        end
                        IndexSpaces.cuda_sync_threads()
                        let
                            t = 0
                            G_beamQ0_polr0 = zero(Float16x2)
                            G_beamQ1_polr0 = zero(Float16x2)
                            G_beamQ0_polr1 = zero(Float16x2)
                            G_beamQ1_polr1 = zero(Float16x2)
                            if let
                                thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                mlo = thread ÷ (4i32)
                                mlo < 6
                            end
                                G_beamQ0_polr0 = Gsh_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 0 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ1_polr0 = Gsh_shared[((((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 0 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ0_polr1 = Gsh_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ1_polr1 = Gsh_shared[((((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 32 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                            end
                            X_beamQ0_polr0 = G_beamQ0_polr0
                            X_beamQ1_polr0 = G_beamQ1_polr0
                            X_beamQ0_polr1 = G_beamQ0_polr1
                            X_beamQ1_polr1 = G_beamQ1_polr1
                            Z_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ1_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ1_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ1_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Z_beamQ1_cplx1_polr1 = zero(Float16x2)
                            (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ0_polr0, (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0)
                            )
                            (Z_beamQ1_cplx0_polr0, Z_beamQ1_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ1_polr0, (Z_beamQ1_cplx0_polr0, Z_beamQ1_cplx1_polr0)
                            )
                            (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ0_polr1, (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1)
                            )
                            (Z_beamQ1_cplx0_polr1, Z_beamQ1_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ1_polr1, (Z_beamQ1_cplx0_polr1, Z_beamQ1_cplx1_polr1)
                            )
                            Γ²re = Γ²_cplx0
                            Γ²im = Γ²_cplx1
                            Zre_beamQ0_polr0 = Z_beamQ0_cplx0_polr0
                            Zim_beamQ0_polr0 = Z_beamQ0_cplx1_polr0
                            Zre_beamQ1_polr0 = Z_beamQ1_cplx0_polr0
                            Zim_beamQ1_polr0 = Z_beamQ1_cplx1_polr0
                            Zre_beamQ0_polr1 = Z_beamQ0_cplx0_polr1
                            Zim_beamQ0_polr1 = Z_beamQ0_cplx1_polr1
                            Zre_beamQ1_polr1 = Z_beamQ1_cplx0_polr1
                            Zim_beamQ1_polr1 = Z_beamQ1_cplx1_polr1
                            Vre_beamQ0_polr0 = muladd(Γ²re, Zre_beamQ0_polr0, -Γ²im * Zim_beamQ0_polr0)
                            Vre_beamQ1_polr0 = muladd(Γ²re, Zre_beamQ1_polr0, -Γ²im * Zim_beamQ1_polr0)
                            Vre_beamQ0_polr1 = muladd(Γ²re, Zre_beamQ0_polr1, -Γ²im * Zim_beamQ0_polr1)
                            Vre_beamQ1_polr1 = muladd(Γ²re, Zre_beamQ1_polr1, -Γ²im * Zim_beamQ1_polr1)
                            Vim_beamQ0_polr0 = muladd(Γ²re, Zim_beamQ0_polr0, +Γ²im * Zre_beamQ0_polr0)
                            Vim_beamQ1_polr0 = muladd(Γ²re, Zim_beamQ1_polr0, +Γ²im * Zre_beamQ1_polr0)
                            Vim_beamQ0_polr1 = muladd(Γ²re, Zim_beamQ0_polr1, +Γ²im * Zre_beamQ0_polr1)
                            Vim_beamQ1_polr1 = muladd(Γ²re, Zim_beamQ1_polr1, +Γ²im * Zre_beamQ1_polr1)
                            V_beamQ0_cplx0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx1_polr0 = Vim_beamQ0_polr0
                            V_beamQ1_cplx0_polr0 = Vre_beamQ1_polr0
                            V_beamQ1_cplx1_polr0 = Vim_beamQ1_polr0
                            V_beamQ0_cplx0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx1_polr1 = Vim_beamQ0_polr1
                            V_beamQ1_cplx0_polr1 = Vre_beamQ1_polr1
                            V_beamQ1_cplx1_polr1 = Vim_beamQ1_polr1
                            Y_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ1_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ1_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ1_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Y_beamQ1_cplx1_polr1 = zero(Float16x2)
                            Vre_beamQ0_polr0 = V_beamQ0_cplx0_polr0
                            Vim_beamQ0_polr0 = V_beamQ0_cplx1_polr0
                            Vre_beamQ1_polr0 = V_beamQ1_cplx0_polr0
                            Vim_beamQ1_polr0 = V_beamQ1_cplx1_polr0
                            Vre_beamQ0_polr1 = V_beamQ0_cplx0_polr1
                            Vim_beamQ0_polr1 = V_beamQ0_cplx1_polr1
                            Vre_beamQ1_polr1 = V_beamQ1_cplx0_polr1
                            Vim_beamQ1_polr1 = V_beamQ1_cplx1_polr1
                            V_beamQ0_cplx_in0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx_in1_polr0 = Vim_beamQ0_polr0
                            V_beamQ1_cplx_in0_polr0 = Vre_beamQ1_polr0
                            V_beamQ1_cplx_in1_polr0 = Vim_beamQ1_polr0
                            V_beamQ0_cplx_in0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx_in1_polr1 = Vim_beamQ0_polr1
                            V_beamQ1_cplx_in0_polr1 = Vre_beamQ1_polr1
                            V_beamQ1_cplx_in1_polr1 = Vim_beamQ1_polr1
                            (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr0, V_beamQ0_cplx_in1_polr0),
                                (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0),
                            )
                            (Y_beamQ1_cplx0_polr0, Y_beamQ1_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ1_cplx_in0_polr0, V_beamQ1_cplx_in1_polr0),
                                (Y_beamQ1_cplx0_polr0, Y_beamQ1_cplx1_polr0),
                            )
                            (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr1, V_beamQ0_cplx_in1_polr1),
                                (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1),
                            )
                            (Y_beamQ1_cplx0_polr1, Y_beamQ1_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ1_cplx_in0_polr1, V_beamQ1_cplx_in1_polr1),
                                (Y_beamQ1_cplx0_polr1, Y_beamQ1_cplx1_polr1),
                            )
                            Ẽ_beamQ0_cplx0_polr0 = Y_beamQ0_cplx0_polr0
                            Ẽ_beamQ1_cplx0_polr0 = Y_beamQ1_cplx0_polr0
                            Ẽ_beamQ0_cplx1_polr0 = Y_beamQ0_cplx1_polr0
                            Ẽ_beamQ1_cplx1_polr0 = Y_beamQ1_cplx1_polr0
                            Ẽ_beamQ0_cplx0_polr1 = Y_beamQ0_cplx0_polr1
                            Ẽ_beamQ1_cplx0_polr1 = Y_beamQ1_cplx0_polr1
                            Ẽ_beamQ0_cplx1_polr1 = Y_beamQ0_cplx1_polr1
                            Ẽ_beamQ1_cplx1_polr1 = Y_beamQ1_cplx1_polr1
                            Ẽp0_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr0
                            Ẽp1_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr1
                            Ẽp0_beamQ1_cplx0 = Ẽ_beamQ1_cplx0_polr0
                            Ẽp1_beamQ1_cplx0 = Ẽ_beamQ1_cplx0_polr1
                            Ẽp0_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr0
                            Ẽp1_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr1
                            Ẽp0_beamQ1_cplx1 = Ẽ_beamQ1_cplx1_polr0
                            Ẽp1_beamQ1_cplx1 = Ẽ_beamQ1_cplx1_polr1
                            Ẽp0re_beamQ0 = Ẽp0_beamQ0_cplx0
                            Ẽp0im_beamQ0 = Ẽp0_beamQ0_cplx1
                            Ẽp0re_beamQ1 = Ẽp0_beamQ1_cplx0
                            Ẽp0im_beamQ1 = Ẽp0_beamQ1_cplx1
                            Ẽp1re_beamQ0 = Ẽp1_beamQ0_cplx0
                            Ẽp1im_beamQ0 = Ẽp1_beamQ0_cplx1
                            Ẽp1re_beamQ1 = Ẽp1_beamQ1_cplx0
                            Ẽp1im_beamQ1 = Ẽp1_beamQ1_cplx1
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
                            I_beamQ1 = muladd(
                                Float16x2(0.003124237f0, 0.003124237f0),
                                muladd(
                                    Ẽp1im_beamQ1,
                                    Ẽp1im_beamQ1,
                                    muladd(
                                        Ẽp1re_beamQ1, Ẽp1re_beamQ1, muladd(Ẽp0im_beamQ1, Ẽp0im_beamQ1, Ẽp0re_beamQ1 * Ẽp0re_beamQ1)
                                    ),
                                ),
                                I_beamQ1,
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
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 48) * 24 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 51) % 51) % 51) * 1152 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 256) % 256) * 58752) + 0) + 0x01] =
                                                I_beamQ0
                                        end
                                        if true
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2 + 1) % 48) * 24 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 51) % 51) % 51) * 1152 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 256) % 256) * 58752) + 0) + 0x01] =
                                                I_beamQ1
                                        end
                                    end
                                    I_beamQ0 = zero(Float16x2)
                                    I_beamQ1 = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let
                            t = 1
                            G_beamQ0_polr0 = zero(Float16x2)
                            G_beamQ1_polr0 = zero(Float16x2)
                            G_beamQ0_polr1 = zero(Float16x2)
                            G_beamQ1_polr1 = zero(Float16x2)
                            if let
                                thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                mlo = thread ÷ (4i32)
                                mlo < 6
                            end
                                G_beamQ0_polr0 = Gsh_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 0 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ1_polr0 = Gsh_shared[((((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 0 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ0_polr1 = Gsh_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ1_polr1 = Gsh_shared[((((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 32 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                            end
                            X_beamQ0_polr0 = G_beamQ0_polr0
                            X_beamQ1_polr0 = G_beamQ1_polr0
                            X_beamQ0_polr1 = G_beamQ0_polr1
                            X_beamQ1_polr1 = G_beamQ1_polr1
                            Z_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ1_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ1_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ1_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Z_beamQ1_cplx1_polr1 = zero(Float16x2)
                            (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ0_polr0, (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0)
                            )
                            (Z_beamQ1_cplx0_polr0, Z_beamQ1_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ1_polr0, (Z_beamQ1_cplx0_polr0, Z_beamQ1_cplx1_polr0)
                            )
                            (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ0_polr1, (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1)
                            )
                            (Z_beamQ1_cplx0_polr1, Z_beamQ1_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ1_polr1, (Z_beamQ1_cplx0_polr1, Z_beamQ1_cplx1_polr1)
                            )
                            Γ²re = Γ²_cplx0
                            Γ²im = Γ²_cplx1
                            Zre_beamQ0_polr0 = Z_beamQ0_cplx0_polr0
                            Zim_beamQ0_polr0 = Z_beamQ0_cplx1_polr0
                            Zre_beamQ1_polr0 = Z_beamQ1_cplx0_polr0
                            Zim_beamQ1_polr0 = Z_beamQ1_cplx1_polr0
                            Zre_beamQ0_polr1 = Z_beamQ0_cplx0_polr1
                            Zim_beamQ0_polr1 = Z_beamQ0_cplx1_polr1
                            Zre_beamQ1_polr1 = Z_beamQ1_cplx0_polr1
                            Zim_beamQ1_polr1 = Z_beamQ1_cplx1_polr1
                            Vre_beamQ0_polr0 = muladd(Γ²re, Zre_beamQ0_polr0, -Γ²im * Zim_beamQ0_polr0)
                            Vre_beamQ1_polr0 = muladd(Γ²re, Zre_beamQ1_polr0, -Γ²im * Zim_beamQ1_polr0)
                            Vre_beamQ0_polr1 = muladd(Γ²re, Zre_beamQ0_polr1, -Γ²im * Zim_beamQ0_polr1)
                            Vre_beamQ1_polr1 = muladd(Γ²re, Zre_beamQ1_polr1, -Γ²im * Zim_beamQ1_polr1)
                            Vim_beamQ0_polr0 = muladd(Γ²re, Zim_beamQ0_polr0, +Γ²im * Zre_beamQ0_polr0)
                            Vim_beamQ1_polr0 = muladd(Γ²re, Zim_beamQ1_polr0, +Γ²im * Zre_beamQ1_polr0)
                            Vim_beamQ0_polr1 = muladd(Γ²re, Zim_beamQ0_polr1, +Γ²im * Zre_beamQ0_polr1)
                            Vim_beamQ1_polr1 = muladd(Γ²re, Zim_beamQ1_polr1, +Γ²im * Zre_beamQ1_polr1)
                            V_beamQ0_cplx0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx1_polr0 = Vim_beamQ0_polr0
                            V_beamQ1_cplx0_polr0 = Vre_beamQ1_polr0
                            V_beamQ1_cplx1_polr0 = Vim_beamQ1_polr0
                            V_beamQ0_cplx0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx1_polr1 = Vim_beamQ0_polr1
                            V_beamQ1_cplx0_polr1 = Vre_beamQ1_polr1
                            V_beamQ1_cplx1_polr1 = Vim_beamQ1_polr1
                            Y_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ1_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ1_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ1_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Y_beamQ1_cplx1_polr1 = zero(Float16x2)
                            Vre_beamQ0_polr0 = V_beamQ0_cplx0_polr0
                            Vim_beamQ0_polr0 = V_beamQ0_cplx1_polr0
                            Vre_beamQ1_polr0 = V_beamQ1_cplx0_polr0
                            Vim_beamQ1_polr0 = V_beamQ1_cplx1_polr0
                            Vre_beamQ0_polr1 = V_beamQ0_cplx0_polr1
                            Vim_beamQ0_polr1 = V_beamQ0_cplx1_polr1
                            Vre_beamQ1_polr1 = V_beamQ1_cplx0_polr1
                            Vim_beamQ1_polr1 = V_beamQ1_cplx1_polr1
                            V_beamQ0_cplx_in0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx_in1_polr0 = Vim_beamQ0_polr0
                            V_beamQ1_cplx_in0_polr0 = Vre_beamQ1_polr0
                            V_beamQ1_cplx_in1_polr0 = Vim_beamQ1_polr0
                            V_beamQ0_cplx_in0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx_in1_polr1 = Vim_beamQ0_polr1
                            V_beamQ1_cplx_in0_polr1 = Vre_beamQ1_polr1
                            V_beamQ1_cplx_in1_polr1 = Vim_beamQ1_polr1
                            (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr0, V_beamQ0_cplx_in1_polr0),
                                (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0),
                            )
                            (Y_beamQ1_cplx0_polr0, Y_beamQ1_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ1_cplx_in0_polr0, V_beamQ1_cplx_in1_polr0),
                                (Y_beamQ1_cplx0_polr0, Y_beamQ1_cplx1_polr0),
                            )
                            (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr1, V_beamQ0_cplx_in1_polr1),
                                (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1),
                            )
                            (Y_beamQ1_cplx0_polr1, Y_beamQ1_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ1_cplx_in0_polr1, V_beamQ1_cplx_in1_polr1),
                                (Y_beamQ1_cplx0_polr1, Y_beamQ1_cplx1_polr1),
                            )
                            Ẽ_beamQ0_cplx0_polr0 = Y_beamQ0_cplx0_polr0
                            Ẽ_beamQ1_cplx0_polr0 = Y_beamQ1_cplx0_polr0
                            Ẽ_beamQ0_cplx1_polr0 = Y_beamQ0_cplx1_polr0
                            Ẽ_beamQ1_cplx1_polr0 = Y_beamQ1_cplx1_polr0
                            Ẽ_beamQ0_cplx0_polr1 = Y_beamQ0_cplx0_polr1
                            Ẽ_beamQ1_cplx0_polr1 = Y_beamQ1_cplx0_polr1
                            Ẽ_beamQ0_cplx1_polr1 = Y_beamQ0_cplx1_polr1
                            Ẽ_beamQ1_cplx1_polr1 = Y_beamQ1_cplx1_polr1
                            Ẽp0_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr0
                            Ẽp1_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr1
                            Ẽp0_beamQ1_cplx0 = Ẽ_beamQ1_cplx0_polr0
                            Ẽp1_beamQ1_cplx0 = Ẽ_beamQ1_cplx0_polr1
                            Ẽp0_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr0
                            Ẽp1_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr1
                            Ẽp0_beamQ1_cplx1 = Ẽ_beamQ1_cplx1_polr0
                            Ẽp1_beamQ1_cplx1 = Ẽ_beamQ1_cplx1_polr1
                            Ẽp0re_beamQ0 = Ẽp0_beamQ0_cplx0
                            Ẽp0im_beamQ0 = Ẽp0_beamQ0_cplx1
                            Ẽp0re_beamQ1 = Ẽp0_beamQ1_cplx0
                            Ẽp0im_beamQ1 = Ẽp0_beamQ1_cplx1
                            Ẽp1re_beamQ0 = Ẽp1_beamQ0_cplx0
                            Ẽp1im_beamQ0 = Ẽp1_beamQ0_cplx1
                            Ẽp1re_beamQ1 = Ẽp1_beamQ1_cplx0
                            Ẽp1im_beamQ1 = Ẽp1_beamQ1_cplx1
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
                            I_beamQ1 = muladd(
                                Float16x2(0.003124237f0, 0.003124237f0),
                                muladd(
                                    Ẽp1im_beamQ1,
                                    Ẽp1im_beamQ1,
                                    muladd(
                                        Ẽp1re_beamQ1, Ẽp1re_beamQ1, muladd(Ẽp0im_beamQ1, Ẽp0im_beamQ1, Ẽp0re_beamQ1 * Ẽp0re_beamQ1)
                                    ),
                                ),
                                I_beamQ1,
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
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 48) * 24 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 51) % 51) % 51) * 1152 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 256) % 256) * 58752) + 0) + 0x01] =
                                                I_beamQ0
                                        end
                                        if true
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2 + 1) % 48) * 24 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 51) % 51) % 51) * 1152 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 256) % 256) * 58752) + 0) + 0x01] =
                                                I_beamQ1
                                        end
                                    end
                                    I_beamQ0 = zero(Float16x2)
                                    I_beamQ1 = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let
                            t = 2
                            G_beamQ0_polr0 = zero(Float16x2)
                            G_beamQ1_polr0 = zero(Float16x2)
                            G_beamQ0_polr1 = zero(Float16x2)
                            G_beamQ1_polr1 = zero(Float16x2)
                            if let
                                thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                mlo = thread ÷ (4i32)
                                mlo < 6
                            end
                                G_beamQ0_polr0 = Gsh_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 0 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ1_polr0 = Gsh_shared[((((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 0 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ0_polr1 = Gsh_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ1_polr1 = Gsh_shared[((((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 32 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                            end
                            X_beamQ0_polr0 = G_beamQ0_polr0
                            X_beamQ1_polr0 = G_beamQ1_polr0
                            X_beamQ0_polr1 = G_beamQ0_polr1
                            X_beamQ1_polr1 = G_beamQ1_polr1
                            Z_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ1_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ1_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ1_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Z_beamQ1_cplx1_polr1 = zero(Float16x2)
                            (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ0_polr0, (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0)
                            )
                            (Z_beamQ1_cplx0_polr0, Z_beamQ1_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ1_polr0, (Z_beamQ1_cplx0_polr0, Z_beamQ1_cplx1_polr0)
                            )
                            (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ0_polr1, (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1)
                            )
                            (Z_beamQ1_cplx0_polr1, Z_beamQ1_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ1_polr1, (Z_beamQ1_cplx0_polr1, Z_beamQ1_cplx1_polr1)
                            )
                            Γ²re = Γ²_cplx0
                            Γ²im = Γ²_cplx1
                            Zre_beamQ0_polr0 = Z_beamQ0_cplx0_polr0
                            Zim_beamQ0_polr0 = Z_beamQ0_cplx1_polr0
                            Zre_beamQ1_polr0 = Z_beamQ1_cplx0_polr0
                            Zim_beamQ1_polr0 = Z_beamQ1_cplx1_polr0
                            Zre_beamQ0_polr1 = Z_beamQ0_cplx0_polr1
                            Zim_beamQ0_polr1 = Z_beamQ0_cplx1_polr1
                            Zre_beamQ1_polr1 = Z_beamQ1_cplx0_polr1
                            Zim_beamQ1_polr1 = Z_beamQ1_cplx1_polr1
                            Vre_beamQ0_polr0 = muladd(Γ²re, Zre_beamQ0_polr0, -Γ²im * Zim_beamQ0_polr0)
                            Vre_beamQ1_polr0 = muladd(Γ²re, Zre_beamQ1_polr0, -Γ²im * Zim_beamQ1_polr0)
                            Vre_beamQ0_polr1 = muladd(Γ²re, Zre_beamQ0_polr1, -Γ²im * Zim_beamQ0_polr1)
                            Vre_beamQ1_polr1 = muladd(Γ²re, Zre_beamQ1_polr1, -Γ²im * Zim_beamQ1_polr1)
                            Vim_beamQ0_polr0 = muladd(Γ²re, Zim_beamQ0_polr0, +Γ²im * Zre_beamQ0_polr0)
                            Vim_beamQ1_polr0 = muladd(Γ²re, Zim_beamQ1_polr0, +Γ²im * Zre_beamQ1_polr0)
                            Vim_beamQ0_polr1 = muladd(Γ²re, Zim_beamQ0_polr1, +Γ²im * Zre_beamQ0_polr1)
                            Vim_beamQ1_polr1 = muladd(Γ²re, Zim_beamQ1_polr1, +Γ²im * Zre_beamQ1_polr1)
                            V_beamQ0_cplx0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx1_polr0 = Vim_beamQ0_polr0
                            V_beamQ1_cplx0_polr0 = Vre_beamQ1_polr0
                            V_beamQ1_cplx1_polr0 = Vim_beamQ1_polr0
                            V_beamQ0_cplx0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx1_polr1 = Vim_beamQ0_polr1
                            V_beamQ1_cplx0_polr1 = Vre_beamQ1_polr1
                            V_beamQ1_cplx1_polr1 = Vim_beamQ1_polr1
                            Y_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ1_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ1_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ1_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Y_beamQ1_cplx1_polr1 = zero(Float16x2)
                            Vre_beamQ0_polr0 = V_beamQ0_cplx0_polr0
                            Vim_beamQ0_polr0 = V_beamQ0_cplx1_polr0
                            Vre_beamQ1_polr0 = V_beamQ1_cplx0_polr0
                            Vim_beamQ1_polr0 = V_beamQ1_cplx1_polr0
                            Vre_beamQ0_polr1 = V_beamQ0_cplx0_polr1
                            Vim_beamQ0_polr1 = V_beamQ0_cplx1_polr1
                            Vre_beamQ1_polr1 = V_beamQ1_cplx0_polr1
                            Vim_beamQ1_polr1 = V_beamQ1_cplx1_polr1
                            V_beamQ0_cplx_in0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx_in1_polr0 = Vim_beamQ0_polr0
                            V_beamQ1_cplx_in0_polr0 = Vre_beamQ1_polr0
                            V_beamQ1_cplx_in1_polr0 = Vim_beamQ1_polr0
                            V_beamQ0_cplx_in0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx_in1_polr1 = Vim_beamQ0_polr1
                            V_beamQ1_cplx_in0_polr1 = Vre_beamQ1_polr1
                            V_beamQ1_cplx_in1_polr1 = Vim_beamQ1_polr1
                            (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr0, V_beamQ0_cplx_in1_polr0),
                                (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0),
                            )
                            (Y_beamQ1_cplx0_polr0, Y_beamQ1_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ1_cplx_in0_polr0, V_beamQ1_cplx_in1_polr0),
                                (Y_beamQ1_cplx0_polr0, Y_beamQ1_cplx1_polr0),
                            )
                            (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr1, V_beamQ0_cplx_in1_polr1),
                                (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1),
                            )
                            (Y_beamQ1_cplx0_polr1, Y_beamQ1_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ1_cplx_in0_polr1, V_beamQ1_cplx_in1_polr1),
                                (Y_beamQ1_cplx0_polr1, Y_beamQ1_cplx1_polr1),
                            )
                            Ẽ_beamQ0_cplx0_polr0 = Y_beamQ0_cplx0_polr0
                            Ẽ_beamQ1_cplx0_polr0 = Y_beamQ1_cplx0_polr0
                            Ẽ_beamQ0_cplx1_polr0 = Y_beamQ0_cplx1_polr0
                            Ẽ_beamQ1_cplx1_polr0 = Y_beamQ1_cplx1_polr0
                            Ẽ_beamQ0_cplx0_polr1 = Y_beamQ0_cplx0_polr1
                            Ẽ_beamQ1_cplx0_polr1 = Y_beamQ1_cplx0_polr1
                            Ẽ_beamQ0_cplx1_polr1 = Y_beamQ0_cplx1_polr1
                            Ẽ_beamQ1_cplx1_polr1 = Y_beamQ1_cplx1_polr1
                            Ẽp0_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr0
                            Ẽp1_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr1
                            Ẽp0_beamQ1_cplx0 = Ẽ_beamQ1_cplx0_polr0
                            Ẽp1_beamQ1_cplx0 = Ẽ_beamQ1_cplx0_polr1
                            Ẽp0_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr0
                            Ẽp1_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr1
                            Ẽp0_beamQ1_cplx1 = Ẽ_beamQ1_cplx1_polr0
                            Ẽp1_beamQ1_cplx1 = Ẽ_beamQ1_cplx1_polr1
                            Ẽp0re_beamQ0 = Ẽp0_beamQ0_cplx0
                            Ẽp0im_beamQ0 = Ẽp0_beamQ0_cplx1
                            Ẽp0re_beamQ1 = Ẽp0_beamQ1_cplx0
                            Ẽp0im_beamQ1 = Ẽp0_beamQ1_cplx1
                            Ẽp1re_beamQ0 = Ẽp1_beamQ0_cplx0
                            Ẽp1im_beamQ0 = Ẽp1_beamQ0_cplx1
                            Ẽp1re_beamQ1 = Ẽp1_beamQ1_cplx0
                            Ẽp1im_beamQ1 = Ẽp1_beamQ1_cplx1
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
                            I_beamQ1 = muladd(
                                Float16x2(0.003124237f0, 0.003124237f0),
                                muladd(
                                    Ẽp1im_beamQ1,
                                    Ẽp1im_beamQ1,
                                    muladd(
                                        Ẽp1re_beamQ1, Ẽp1re_beamQ1, muladd(Ẽp0im_beamQ1, Ẽp0im_beamQ1, Ẽp0re_beamQ1 * Ẽp0re_beamQ1)
                                    ),
                                ),
                                I_beamQ1,
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
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 48) * 24 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 51) % 51) % 51) * 1152 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 256) % 256) * 58752) + 0) + 0x01] =
                                                I_beamQ0
                                        end
                                        if true
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2 + 1) % 48) * 24 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 51) % 51) % 51) * 1152 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 256) % 256) * 58752) + 0) + 0x01] =
                                                I_beamQ1
                                        end
                                    end
                                    I_beamQ0 = zero(Float16x2)
                                    I_beamQ1 = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let
                            t = 3
                            G_beamQ0_polr0 = zero(Float16x2)
                            G_beamQ1_polr0 = zero(Float16x2)
                            G_beamQ0_polr1 = zero(Float16x2)
                            G_beamQ1_polr1 = zero(Float16x2)
                            if let
                                thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                mlo = thread ÷ (4i32)
                                mlo < 6
                            end
                                G_beamQ0_polr0 = Gsh_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 0 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ1_polr0 = Gsh_shared[((((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 0 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ0_polr1 = Gsh_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ1_polr1 = Gsh_shared[((((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 32 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                            end
                            X_beamQ0_polr0 = G_beamQ0_polr0
                            X_beamQ1_polr0 = G_beamQ1_polr0
                            X_beamQ0_polr1 = G_beamQ0_polr1
                            X_beamQ1_polr1 = G_beamQ1_polr1
                            Z_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ1_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ1_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ1_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Z_beamQ1_cplx1_polr1 = zero(Float16x2)
                            (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ0_polr0, (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0)
                            )
                            (Z_beamQ1_cplx0_polr0, Z_beamQ1_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ1_polr0, (Z_beamQ1_cplx0_polr0, Z_beamQ1_cplx1_polr0)
                            )
                            (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ0_polr1, (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1)
                            )
                            (Z_beamQ1_cplx0_polr1, Z_beamQ1_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ1_polr1, (Z_beamQ1_cplx0_polr1, Z_beamQ1_cplx1_polr1)
                            )
                            Γ²re = Γ²_cplx0
                            Γ²im = Γ²_cplx1
                            Zre_beamQ0_polr0 = Z_beamQ0_cplx0_polr0
                            Zim_beamQ0_polr0 = Z_beamQ0_cplx1_polr0
                            Zre_beamQ1_polr0 = Z_beamQ1_cplx0_polr0
                            Zim_beamQ1_polr0 = Z_beamQ1_cplx1_polr0
                            Zre_beamQ0_polr1 = Z_beamQ0_cplx0_polr1
                            Zim_beamQ0_polr1 = Z_beamQ0_cplx1_polr1
                            Zre_beamQ1_polr1 = Z_beamQ1_cplx0_polr1
                            Zim_beamQ1_polr1 = Z_beamQ1_cplx1_polr1
                            Vre_beamQ0_polr0 = muladd(Γ²re, Zre_beamQ0_polr0, -Γ²im * Zim_beamQ0_polr0)
                            Vre_beamQ1_polr0 = muladd(Γ²re, Zre_beamQ1_polr0, -Γ²im * Zim_beamQ1_polr0)
                            Vre_beamQ0_polr1 = muladd(Γ²re, Zre_beamQ0_polr1, -Γ²im * Zim_beamQ0_polr1)
                            Vre_beamQ1_polr1 = muladd(Γ²re, Zre_beamQ1_polr1, -Γ²im * Zim_beamQ1_polr1)
                            Vim_beamQ0_polr0 = muladd(Γ²re, Zim_beamQ0_polr0, +Γ²im * Zre_beamQ0_polr0)
                            Vim_beamQ1_polr0 = muladd(Γ²re, Zim_beamQ1_polr0, +Γ²im * Zre_beamQ1_polr0)
                            Vim_beamQ0_polr1 = muladd(Γ²re, Zim_beamQ0_polr1, +Γ²im * Zre_beamQ0_polr1)
                            Vim_beamQ1_polr1 = muladd(Γ²re, Zim_beamQ1_polr1, +Γ²im * Zre_beamQ1_polr1)
                            V_beamQ0_cplx0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx1_polr0 = Vim_beamQ0_polr0
                            V_beamQ1_cplx0_polr0 = Vre_beamQ1_polr0
                            V_beamQ1_cplx1_polr0 = Vim_beamQ1_polr0
                            V_beamQ0_cplx0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx1_polr1 = Vim_beamQ0_polr1
                            V_beamQ1_cplx0_polr1 = Vre_beamQ1_polr1
                            V_beamQ1_cplx1_polr1 = Vim_beamQ1_polr1
                            Y_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ1_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ1_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ1_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Y_beamQ1_cplx1_polr1 = zero(Float16x2)
                            Vre_beamQ0_polr0 = V_beamQ0_cplx0_polr0
                            Vim_beamQ0_polr0 = V_beamQ0_cplx1_polr0
                            Vre_beamQ1_polr0 = V_beamQ1_cplx0_polr0
                            Vim_beamQ1_polr0 = V_beamQ1_cplx1_polr0
                            Vre_beamQ0_polr1 = V_beamQ0_cplx0_polr1
                            Vim_beamQ0_polr1 = V_beamQ0_cplx1_polr1
                            Vre_beamQ1_polr1 = V_beamQ1_cplx0_polr1
                            Vim_beamQ1_polr1 = V_beamQ1_cplx1_polr1
                            V_beamQ0_cplx_in0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx_in1_polr0 = Vim_beamQ0_polr0
                            V_beamQ1_cplx_in0_polr0 = Vre_beamQ1_polr0
                            V_beamQ1_cplx_in1_polr0 = Vim_beamQ1_polr0
                            V_beamQ0_cplx_in0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx_in1_polr1 = Vim_beamQ0_polr1
                            V_beamQ1_cplx_in0_polr1 = Vre_beamQ1_polr1
                            V_beamQ1_cplx_in1_polr1 = Vim_beamQ1_polr1
                            (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr0, V_beamQ0_cplx_in1_polr0),
                                (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0),
                            )
                            (Y_beamQ1_cplx0_polr0, Y_beamQ1_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ1_cplx_in0_polr0, V_beamQ1_cplx_in1_polr0),
                                (Y_beamQ1_cplx0_polr0, Y_beamQ1_cplx1_polr0),
                            )
                            (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr1, V_beamQ0_cplx_in1_polr1),
                                (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1),
                            )
                            (Y_beamQ1_cplx0_polr1, Y_beamQ1_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ1_cplx_in0_polr1, V_beamQ1_cplx_in1_polr1),
                                (Y_beamQ1_cplx0_polr1, Y_beamQ1_cplx1_polr1),
                            )
                            Ẽ_beamQ0_cplx0_polr0 = Y_beamQ0_cplx0_polr0
                            Ẽ_beamQ1_cplx0_polr0 = Y_beamQ1_cplx0_polr0
                            Ẽ_beamQ0_cplx1_polr0 = Y_beamQ0_cplx1_polr0
                            Ẽ_beamQ1_cplx1_polr0 = Y_beamQ1_cplx1_polr0
                            Ẽ_beamQ0_cplx0_polr1 = Y_beamQ0_cplx0_polr1
                            Ẽ_beamQ1_cplx0_polr1 = Y_beamQ1_cplx0_polr1
                            Ẽ_beamQ0_cplx1_polr1 = Y_beamQ0_cplx1_polr1
                            Ẽ_beamQ1_cplx1_polr1 = Y_beamQ1_cplx1_polr1
                            Ẽp0_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr0
                            Ẽp1_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr1
                            Ẽp0_beamQ1_cplx0 = Ẽ_beamQ1_cplx0_polr0
                            Ẽp1_beamQ1_cplx0 = Ẽ_beamQ1_cplx0_polr1
                            Ẽp0_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr0
                            Ẽp1_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr1
                            Ẽp0_beamQ1_cplx1 = Ẽ_beamQ1_cplx1_polr0
                            Ẽp1_beamQ1_cplx1 = Ẽ_beamQ1_cplx1_polr1
                            Ẽp0re_beamQ0 = Ẽp0_beamQ0_cplx0
                            Ẽp0im_beamQ0 = Ẽp0_beamQ0_cplx1
                            Ẽp0re_beamQ1 = Ẽp0_beamQ1_cplx0
                            Ẽp0im_beamQ1 = Ẽp0_beamQ1_cplx1
                            Ẽp1re_beamQ0 = Ẽp1_beamQ0_cplx0
                            Ẽp1im_beamQ0 = Ẽp1_beamQ0_cplx1
                            Ẽp1re_beamQ1 = Ẽp1_beamQ1_cplx0
                            Ẽp1im_beamQ1 = Ẽp1_beamQ1_cplx1
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
                            I_beamQ1 = muladd(
                                Float16x2(0.003124237f0, 0.003124237f0),
                                muladd(
                                    Ẽp1im_beamQ1,
                                    Ẽp1im_beamQ1,
                                    muladd(
                                        Ẽp1re_beamQ1, Ẽp1re_beamQ1, muladd(Ẽp0im_beamQ1, Ẽp0im_beamQ1, Ẽp0re_beamQ1 * Ẽp0re_beamQ1)
                                    ),
                                ),
                                I_beamQ1,
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
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 48) * 24 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 51) % 51) % 51) * 1152 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 256) % 256) * 58752) + 0) + 0x01] =
                                                I_beamQ0
                                        end
                                        if true
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2 + 1) % 48) * 24 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 51) % 51) % 51) * 1152 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 256) % 256) * 58752) + 0) + 0x01] =
                                                I_beamQ1
                                        end
                                    end
                                    I_beamQ0 = zero(Float16x2)
                                    I_beamQ1 = zero(Float16x2)
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
                            (Γ¹_cplx0, Γ¹_cplx1), X_polr0_time0, (Z_cplx0_polr0_time0, Z_cplx1_polr0_time0)
                        )
                        (Z_cplx0_polr1_time0, Z_cplx1_polr1_time0) = IndexSpaces.mma_m16n8k8(
                            (Γ¹_cplx0, Γ¹_cplx1), X_polr1_time0, (Z_cplx0_polr1_time0, Z_cplx1_polr1_time0)
                        )
                        (Z_cplx0_polr0_time1, Z_cplx1_polr0_time1) = IndexSpaces.mma_m16n8k8(
                            (Γ¹_cplx0, Γ¹_cplx1), X_polr0_time1, (Z_cplx0_polr0_time1, Z_cplx1_polr0_time1)
                        )
                        (Z_cplx0_polr1_time1, Z_cplx1_polr1_time1) = IndexSpaces.mma_m16n8k8(
                            (Γ¹_cplx0, Γ¹_cplx1), X_polr1_time1, (Z_cplx0_polr1_time1, Z_cplx1_polr1_time1)
                        )
                        (Z_cplx0_polr0_time2, Z_cplx1_polr0_time2) = IndexSpaces.mma_m16n8k8(
                            (Γ¹_cplx0, Γ¹_cplx1), X_polr0_time2, (Z_cplx0_polr0_time2, Z_cplx1_polr0_time2)
                        )
                        (Z_cplx0_polr1_time2, Z_cplx1_polr1_time2) = IndexSpaces.mma_m16n8k8(
                            (Γ¹_cplx0, Γ¹_cplx1), X_polr1_time2, (Z_cplx0_polr1_time2, Z_cplx1_polr1_time2)
                        )
                        (Z_cplx0_polr0_time3, Z_cplx1_polr0_time3) = IndexSpaces.mma_m16n8k8(
                            (Γ¹_cplx0, Γ¹_cplx1), X_polr0_time3, (Z_cplx0_polr0_time3, Z_cplx1_polr0_time3)
                        )
                        (Z_cplx0_polr1_time3, Z_cplx1_polr1_time3) = IndexSpaces.mma_m16n8k8(
                            (Γ¹_cplx0, Γ¹_cplx1), X_polr1_time3, (Z_cplx0_polr1_time3, Z_cplx1_polr1_time3)
                        )
                        Γ²re = Γ²_cplx0
                        Γ²im = Γ²_cplx1
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
                        Vre_polr0_time0 = muladd(Γ²re, Zre_polr0_time0, -Γ²im * Zim_polr0_time0)
                        Vre_polr1_time0 = muladd(Γ²re, Zre_polr1_time0, -Γ²im * Zim_polr1_time0)
                        Vre_polr0_time1 = muladd(Γ²re, Zre_polr0_time1, -Γ²im * Zim_polr0_time1)
                        Vre_polr1_time1 = muladd(Γ²re, Zre_polr1_time1, -Γ²im * Zim_polr1_time1)
                        Vre_polr0_time2 = muladd(Γ²re, Zre_polr0_time2, -Γ²im * Zim_polr0_time2)
                        Vre_polr1_time2 = muladd(Γ²re, Zre_polr1_time2, -Γ²im * Zim_polr1_time2)
                        Vre_polr0_time3 = muladd(Γ²re, Zre_polr0_time3, -Γ²im * Zim_polr0_time3)
                        Vre_polr1_time3 = muladd(Γ²re, Zre_polr1_time3, -Γ²im * Zim_polr1_time3)
                        Vim_polr0_time0 = muladd(Γ²re, Zim_polr0_time0, +Γ²im * Zre_polr0_time0)
                        Vim_polr1_time0 = muladd(Γ²re, Zim_polr1_time0, +Γ²im * Zre_polr1_time0)
                        Vim_polr0_time1 = muladd(Γ²re, Zim_polr0_time1, +Γ²im * Zre_polr0_time1)
                        Vim_polr1_time1 = muladd(Γ²re, Zim_polr1_time1, +Γ²im * Zre_polr1_time1)
                        Vim_polr0_time2 = muladd(Γ²re, Zim_polr0_time2, +Γ²im * Zre_polr0_time2)
                        Vim_polr1_time2 = muladd(Γ²re, Zim_polr1_time2, +Γ²im * Zre_polr1_time2)
                        Vim_polr0_time3 = muladd(Γ²re, Zim_polr0_time3, +Γ²im * Zre_polr0_time3)
                        Vim_polr1_time3 = muladd(Γ²re, Zim_polr1_time3, +Γ²im * Zre_polr1_time3)
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
                            (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr0_time0, V_cplx_in1_polr0_time0),
                            (Y_cplx0_polr0_time0, Y_cplx1_polr0_time0),
                        )
                        (Y_cplx0_polr1_time0, Y_cplx1_polr1_time0) = IndexSpaces.mma_m16n8k16(
                            (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr1_time0, V_cplx_in1_polr1_time0),
                            (Y_cplx0_polr1_time0, Y_cplx1_polr1_time0),
                        )
                        (Y_cplx0_polr0_time1, Y_cplx1_polr0_time1) = IndexSpaces.mma_m16n8k16(
                            (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr0_time1, V_cplx_in1_polr0_time1),
                            (Y_cplx0_polr0_time1, Y_cplx1_polr0_time1),
                        )
                        (Y_cplx0_polr1_time1, Y_cplx1_polr1_time1) = IndexSpaces.mma_m16n8k16(
                            (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr1_time1, V_cplx_in1_polr1_time1),
                            (Y_cplx0_polr1_time1, Y_cplx1_polr1_time1),
                        )
                        (Y_cplx0_polr0_time2, Y_cplx1_polr0_time2) = IndexSpaces.mma_m16n8k16(
                            (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr0_time2, V_cplx_in1_polr0_time2),
                            (Y_cplx0_polr0_time2, Y_cplx1_polr0_time2),
                        )
                        (Y_cplx0_polr1_time2, Y_cplx1_polr1_time2) = IndexSpaces.mma_m16n8k16(
                            (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr1_time2, V_cplx_in1_polr1_time2),
                            (Y_cplx0_polr1_time2, Y_cplx1_polr1_time2),
                        )
                        (Y_cplx0_polr0_time3, Y_cplx1_polr0_time3) = IndexSpaces.mma_m16n8k16(
                            (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                            (V_cplx_in0_polr0_time3, V_cplx_in1_polr0_time3),
                            (Y_cplx0_polr0_time3, Y_cplx1_polr0_time3),
                        )
                        (Y_cplx0_polr1_time3, Y_cplx1_polr1_time3) = IndexSpaces.mma_m16n8k16(
                            (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
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
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 16) % 2) * 514 + 0 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 2) * 4112 + (((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx0_polr0_time0
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 16) % 2) * 514 + 0 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 2) % 2) * 4112 + (((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx1_polr0_time0
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 16) % 2) * 514 + 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 2) * 4112 + (((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx0_polr1_time0
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 16) % 2) * 514 + 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 2) % 2) * 4112 + (((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx1_polr1_time0
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 16) % 2) * 514 + 0 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + 1) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx0_polr0_time1
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 16) % 2) * 514 + 0 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + 1) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx1_polr0_time1
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 16) % 2) * 514 + 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + 1) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx0_polr1_time1
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 16) % 2) * 514 + 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + 1) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx1_polr1_time1
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 16) % 2) * 514 + 0 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx0_polr0_time2
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 16) % 2) * 514 + 0 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx1_polr0_time2
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 16) % 2) * 514 + 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx0_polr1_time2
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 16) % 2) * 514 + 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx1_polr1_time2
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 16) % 2) * 514 + 0 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + 3) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx0_polr0_time3
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 16) % 2) * 514 + 0 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + 3) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx1_polr0_time3
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 16) % 2) * 514 + 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + 3) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx0_polr1_time3
                        end
                        if true
                            Gsh_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 16) % 2) * 514 + 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + 3) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 4) % 2) * 2056 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) ÷ 6) % 4) % 4) * 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 6) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2 + 1) ÷ 32) % 2) * 257) + 0) + 0x01] =
                                G_cplx1_polr1_time3
                        end
                        IndexSpaces.cuda_sync_threads()
                        let
                            t = 0
                            G_beamQ0_polr0 = zero(Float16x2)
                            G_beamQ1_polr0 = zero(Float16x2)
                            G_beamQ0_polr1 = zero(Float16x2)
                            G_beamQ1_polr1 = zero(Float16x2)
                            if let
                                thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                mlo = thread ÷ (4i32)
                                mlo < 6
                            end
                                G_beamQ0_polr0 = Gsh_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 0 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ1_polr0 = Gsh_shared[((((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 0 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ0_polr1 = Gsh_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ1_polr1 = Gsh_shared[((((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 32 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                            end
                            X_beamQ0_polr0 = G_beamQ0_polr0
                            X_beamQ1_polr0 = G_beamQ1_polr0
                            X_beamQ0_polr1 = G_beamQ0_polr1
                            X_beamQ1_polr1 = G_beamQ1_polr1
                            Z_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ1_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ1_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ1_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Z_beamQ1_cplx1_polr1 = zero(Float16x2)
                            (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ0_polr0, (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0)
                            )
                            (Z_beamQ1_cplx0_polr0, Z_beamQ1_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ1_polr0, (Z_beamQ1_cplx0_polr0, Z_beamQ1_cplx1_polr0)
                            )
                            (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ0_polr1, (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1)
                            )
                            (Z_beamQ1_cplx0_polr1, Z_beamQ1_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ1_polr1, (Z_beamQ1_cplx0_polr1, Z_beamQ1_cplx1_polr1)
                            )
                            Γ²re = Γ²_cplx0
                            Γ²im = Γ²_cplx1
                            Zre_beamQ0_polr0 = Z_beamQ0_cplx0_polr0
                            Zim_beamQ0_polr0 = Z_beamQ0_cplx1_polr0
                            Zre_beamQ1_polr0 = Z_beamQ1_cplx0_polr0
                            Zim_beamQ1_polr0 = Z_beamQ1_cplx1_polr0
                            Zre_beamQ0_polr1 = Z_beamQ0_cplx0_polr1
                            Zim_beamQ0_polr1 = Z_beamQ0_cplx1_polr1
                            Zre_beamQ1_polr1 = Z_beamQ1_cplx0_polr1
                            Zim_beamQ1_polr1 = Z_beamQ1_cplx1_polr1
                            Vre_beamQ0_polr0 = muladd(Γ²re, Zre_beamQ0_polr0, -Γ²im * Zim_beamQ0_polr0)
                            Vre_beamQ1_polr0 = muladd(Γ²re, Zre_beamQ1_polr0, -Γ²im * Zim_beamQ1_polr0)
                            Vre_beamQ0_polr1 = muladd(Γ²re, Zre_beamQ0_polr1, -Γ²im * Zim_beamQ0_polr1)
                            Vre_beamQ1_polr1 = muladd(Γ²re, Zre_beamQ1_polr1, -Γ²im * Zim_beamQ1_polr1)
                            Vim_beamQ0_polr0 = muladd(Γ²re, Zim_beamQ0_polr0, +Γ²im * Zre_beamQ0_polr0)
                            Vim_beamQ1_polr0 = muladd(Γ²re, Zim_beamQ1_polr0, +Γ²im * Zre_beamQ1_polr0)
                            Vim_beamQ0_polr1 = muladd(Γ²re, Zim_beamQ0_polr1, +Γ²im * Zre_beamQ0_polr1)
                            Vim_beamQ1_polr1 = muladd(Γ²re, Zim_beamQ1_polr1, +Γ²im * Zre_beamQ1_polr1)
                            V_beamQ0_cplx0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx1_polr0 = Vim_beamQ0_polr0
                            V_beamQ1_cplx0_polr0 = Vre_beamQ1_polr0
                            V_beamQ1_cplx1_polr0 = Vim_beamQ1_polr0
                            V_beamQ0_cplx0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx1_polr1 = Vim_beamQ0_polr1
                            V_beamQ1_cplx0_polr1 = Vre_beamQ1_polr1
                            V_beamQ1_cplx1_polr1 = Vim_beamQ1_polr1
                            Y_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ1_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ1_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ1_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Y_beamQ1_cplx1_polr1 = zero(Float16x2)
                            Vre_beamQ0_polr0 = V_beamQ0_cplx0_polr0
                            Vim_beamQ0_polr0 = V_beamQ0_cplx1_polr0
                            Vre_beamQ1_polr0 = V_beamQ1_cplx0_polr0
                            Vim_beamQ1_polr0 = V_beamQ1_cplx1_polr0
                            Vre_beamQ0_polr1 = V_beamQ0_cplx0_polr1
                            Vim_beamQ0_polr1 = V_beamQ0_cplx1_polr1
                            Vre_beamQ1_polr1 = V_beamQ1_cplx0_polr1
                            Vim_beamQ1_polr1 = V_beamQ1_cplx1_polr1
                            V_beamQ0_cplx_in0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx_in1_polr0 = Vim_beamQ0_polr0
                            V_beamQ1_cplx_in0_polr0 = Vre_beamQ1_polr0
                            V_beamQ1_cplx_in1_polr0 = Vim_beamQ1_polr0
                            V_beamQ0_cplx_in0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx_in1_polr1 = Vim_beamQ0_polr1
                            V_beamQ1_cplx_in0_polr1 = Vre_beamQ1_polr1
                            V_beamQ1_cplx_in1_polr1 = Vim_beamQ1_polr1
                            (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr0, V_beamQ0_cplx_in1_polr0),
                                (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0),
                            )
                            (Y_beamQ1_cplx0_polr0, Y_beamQ1_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ1_cplx_in0_polr0, V_beamQ1_cplx_in1_polr0),
                                (Y_beamQ1_cplx0_polr0, Y_beamQ1_cplx1_polr0),
                            )
                            (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr1, V_beamQ0_cplx_in1_polr1),
                                (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1),
                            )
                            (Y_beamQ1_cplx0_polr1, Y_beamQ1_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ1_cplx_in0_polr1, V_beamQ1_cplx_in1_polr1),
                                (Y_beamQ1_cplx0_polr1, Y_beamQ1_cplx1_polr1),
                            )
                            Ẽ_beamQ0_cplx0_polr0 = Y_beamQ0_cplx0_polr0
                            Ẽ_beamQ1_cplx0_polr0 = Y_beamQ1_cplx0_polr0
                            Ẽ_beamQ0_cplx1_polr0 = Y_beamQ0_cplx1_polr0
                            Ẽ_beamQ1_cplx1_polr0 = Y_beamQ1_cplx1_polr0
                            Ẽ_beamQ0_cplx0_polr1 = Y_beamQ0_cplx0_polr1
                            Ẽ_beamQ1_cplx0_polr1 = Y_beamQ1_cplx0_polr1
                            Ẽ_beamQ0_cplx1_polr1 = Y_beamQ0_cplx1_polr1
                            Ẽ_beamQ1_cplx1_polr1 = Y_beamQ1_cplx1_polr1
                            Ẽp0_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr0
                            Ẽp1_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr1
                            Ẽp0_beamQ1_cplx0 = Ẽ_beamQ1_cplx0_polr0
                            Ẽp1_beamQ1_cplx0 = Ẽ_beamQ1_cplx0_polr1
                            Ẽp0_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr0
                            Ẽp1_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr1
                            Ẽp0_beamQ1_cplx1 = Ẽ_beamQ1_cplx1_polr0
                            Ẽp1_beamQ1_cplx1 = Ẽ_beamQ1_cplx1_polr1
                            Ẽp0re_beamQ0 = Ẽp0_beamQ0_cplx0
                            Ẽp0im_beamQ0 = Ẽp0_beamQ0_cplx1
                            Ẽp0re_beamQ1 = Ẽp0_beamQ1_cplx0
                            Ẽp0im_beamQ1 = Ẽp0_beamQ1_cplx1
                            Ẽp1re_beamQ0 = Ẽp1_beamQ0_cplx0
                            Ẽp1im_beamQ0 = Ẽp1_beamQ0_cplx1
                            Ẽp1re_beamQ1 = Ẽp1_beamQ1_cplx0
                            Ẽp1im_beamQ1 = Ẽp1_beamQ1_cplx1
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
                            I_beamQ1 = muladd(
                                Float16x2(0.003124237f0, 0.003124237f0),
                                muladd(
                                    Ẽp1im_beamQ1,
                                    Ẽp1im_beamQ1,
                                    muladd(
                                        Ẽp1re_beamQ1, Ẽp1re_beamQ1, muladd(Ẽp0im_beamQ1, Ẽp0im_beamQ1, Ẽp0re_beamQ1 * Ẽp0re_beamQ1)
                                    ),
                                ),
                                I_beamQ1,
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
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 48) * 24 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 51) % 51) % 51) * 1152 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 256) % 256) * 58752) + 0) + 0x01] =
                                                I_beamQ0
                                        end
                                        if true
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2 + 1) % 48) * 24 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 51) % 51) % 51) * 1152 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 256) % 256) * 58752) + 0) + 0x01] =
                                                I_beamQ1
                                        end
                                    end
                                    I_beamQ0 = zero(Float16x2)
                                    I_beamQ1 = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let
                            t = 1
                            G_beamQ0_polr0 = zero(Float16x2)
                            G_beamQ1_polr0 = zero(Float16x2)
                            G_beamQ0_polr1 = zero(Float16x2)
                            G_beamQ1_polr1 = zero(Float16x2)
                            if let
                                thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                mlo = thread ÷ (4i32)
                                mlo < 6
                            end
                                G_beamQ0_polr0 = Gsh_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 0 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ1_polr0 = Gsh_shared[((((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 0 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ0_polr1 = Gsh_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ1_polr1 = Gsh_shared[((((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 32 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                            end
                            X_beamQ0_polr0 = G_beamQ0_polr0
                            X_beamQ1_polr0 = G_beamQ1_polr0
                            X_beamQ0_polr1 = G_beamQ0_polr1
                            X_beamQ1_polr1 = G_beamQ1_polr1
                            Z_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ1_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ1_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ1_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Z_beamQ1_cplx1_polr1 = zero(Float16x2)
                            (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ0_polr0, (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0)
                            )
                            (Z_beamQ1_cplx0_polr0, Z_beamQ1_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ1_polr0, (Z_beamQ1_cplx0_polr0, Z_beamQ1_cplx1_polr0)
                            )
                            (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ0_polr1, (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1)
                            )
                            (Z_beamQ1_cplx0_polr1, Z_beamQ1_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ1_polr1, (Z_beamQ1_cplx0_polr1, Z_beamQ1_cplx1_polr1)
                            )
                            Γ²re = Γ²_cplx0
                            Γ²im = Γ²_cplx1
                            Zre_beamQ0_polr0 = Z_beamQ0_cplx0_polr0
                            Zim_beamQ0_polr0 = Z_beamQ0_cplx1_polr0
                            Zre_beamQ1_polr0 = Z_beamQ1_cplx0_polr0
                            Zim_beamQ1_polr0 = Z_beamQ1_cplx1_polr0
                            Zre_beamQ0_polr1 = Z_beamQ0_cplx0_polr1
                            Zim_beamQ0_polr1 = Z_beamQ0_cplx1_polr1
                            Zre_beamQ1_polr1 = Z_beamQ1_cplx0_polr1
                            Zim_beamQ1_polr1 = Z_beamQ1_cplx1_polr1
                            Vre_beamQ0_polr0 = muladd(Γ²re, Zre_beamQ0_polr0, -Γ²im * Zim_beamQ0_polr0)
                            Vre_beamQ1_polr0 = muladd(Γ²re, Zre_beamQ1_polr0, -Γ²im * Zim_beamQ1_polr0)
                            Vre_beamQ0_polr1 = muladd(Γ²re, Zre_beamQ0_polr1, -Γ²im * Zim_beamQ0_polr1)
                            Vre_beamQ1_polr1 = muladd(Γ²re, Zre_beamQ1_polr1, -Γ²im * Zim_beamQ1_polr1)
                            Vim_beamQ0_polr0 = muladd(Γ²re, Zim_beamQ0_polr0, +Γ²im * Zre_beamQ0_polr0)
                            Vim_beamQ1_polr0 = muladd(Γ²re, Zim_beamQ1_polr0, +Γ²im * Zre_beamQ1_polr0)
                            Vim_beamQ0_polr1 = muladd(Γ²re, Zim_beamQ0_polr1, +Γ²im * Zre_beamQ0_polr1)
                            Vim_beamQ1_polr1 = muladd(Γ²re, Zim_beamQ1_polr1, +Γ²im * Zre_beamQ1_polr1)
                            V_beamQ0_cplx0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx1_polr0 = Vim_beamQ0_polr0
                            V_beamQ1_cplx0_polr0 = Vre_beamQ1_polr0
                            V_beamQ1_cplx1_polr0 = Vim_beamQ1_polr0
                            V_beamQ0_cplx0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx1_polr1 = Vim_beamQ0_polr1
                            V_beamQ1_cplx0_polr1 = Vre_beamQ1_polr1
                            V_beamQ1_cplx1_polr1 = Vim_beamQ1_polr1
                            Y_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ1_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ1_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ1_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Y_beamQ1_cplx1_polr1 = zero(Float16x2)
                            Vre_beamQ0_polr0 = V_beamQ0_cplx0_polr0
                            Vim_beamQ0_polr0 = V_beamQ0_cplx1_polr0
                            Vre_beamQ1_polr0 = V_beamQ1_cplx0_polr0
                            Vim_beamQ1_polr0 = V_beamQ1_cplx1_polr0
                            Vre_beamQ0_polr1 = V_beamQ0_cplx0_polr1
                            Vim_beamQ0_polr1 = V_beamQ0_cplx1_polr1
                            Vre_beamQ1_polr1 = V_beamQ1_cplx0_polr1
                            Vim_beamQ1_polr1 = V_beamQ1_cplx1_polr1
                            V_beamQ0_cplx_in0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx_in1_polr0 = Vim_beamQ0_polr0
                            V_beamQ1_cplx_in0_polr0 = Vre_beamQ1_polr0
                            V_beamQ1_cplx_in1_polr0 = Vim_beamQ1_polr0
                            V_beamQ0_cplx_in0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx_in1_polr1 = Vim_beamQ0_polr1
                            V_beamQ1_cplx_in0_polr1 = Vre_beamQ1_polr1
                            V_beamQ1_cplx_in1_polr1 = Vim_beamQ1_polr1
                            (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr0, V_beamQ0_cplx_in1_polr0),
                                (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0),
                            )
                            (Y_beamQ1_cplx0_polr0, Y_beamQ1_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ1_cplx_in0_polr0, V_beamQ1_cplx_in1_polr0),
                                (Y_beamQ1_cplx0_polr0, Y_beamQ1_cplx1_polr0),
                            )
                            (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr1, V_beamQ0_cplx_in1_polr1),
                                (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1),
                            )
                            (Y_beamQ1_cplx0_polr1, Y_beamQ1_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ1_cplx_in0_polr1, V_beamQ1_cplx_in1_polr1),
                                (Y_beamQ1_cplx0_polr1, Y_beamQ1_cplx1_polr1),
                            )
                            Ẽ_beamQ0_cplx0_polr0 = Y_beamQ0_cplx0_polr0
                            Ẽ_beamQ1_cplx0_polr0 = Y_beamQ1_cplx0_polr0
                            Ẽ_beamQ0_cplx1_polr0 = Y_beamQ0_cplx1_polr0
                            Ẽ_beamQ1_cplx1_polr0 = Y_beamQ1_cplx1_polr0
                            Ẽ_beamQ0_cplx0_polr1 = Y_beamQ0_cplx0_polr1
                            Ẽ_beamQ1_cplx0_polr1 = Y_beamQ1_cplx0_polr1
                            Ẽ_beamQ0_cplx1_polr1 = Y_beamQ0_cplx1_polr1
                            Ẽ_beamQ1_cplx1_polr1 = Y_beamQ1_cplx1_polr1
                            Ẽp0_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr0
                            Ẽp1_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr1
                            Ẽp0_beamQ1_cplx0 = Ẽ_beamQ1_cplx0_polr0
                            Ẽp1_beamQ1_cplx0 = Ẽ_beamQ1_cplx0_polr1
                            Ẽp0_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr0
                            Ẽp1_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr1
                            Ẽp0_beamQ1_cplx1 = Ẽ_beamQ1_cplx1_polr0
                            Ẽp1_beamQ1_cplx1 = Ẽ_beamQ1_cplx1_polr1
                            Ẽp0re_beamQ0 = Ẽp0_beamQ0_cplx0
                            Ẽp0im_beamQ0 = Ẽp0_beamQ0_cplx1
                            Ẽp0re_beamQ1 = Ẽp0_beamQ1_cplx0
                            Ẽp0im_beamQ1 = Ẽp0_beamQ1_cplx1
                            Ẽp1re_beamQ0 = Ẽp1_beamQ0_cplx0
                            Ẽp1im_beamQ0 = Ẽp1_beamQ0_cplx1
                            Ẽp1re_beamQ1 = Ẽp1_beamQ1_cplx0
                            Ẽp1im_beamQ1 = Ẽp1_beamQ1_cplx1
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
                            I_beamQ1 = muladd(
                                Float16x2(0.003124237f0, 0.003124237f0),
                                muladd(
                                    Ẽp1im_beamQ1,
                                    Ẽp1im_beamQ1,
                                    muladd(
                                        Ẽp1re_beamQ1, Ẽp1re_beamQ1, muladd(Ẽp0im_beamQ1, Ẽp0im_beamQ1, Ẽp0re_beamQ1 * Ẽp0re_beamQ1)
                                    ),
                                ),
                                I_beamQ1,
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
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 48) * 24 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 51) % 51) % 51) * 1152 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 256) % 256) * 58752) + 0) + 0x01] =
                                                I_beamQ0
                                        end
                                        if true
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2 + 1) % 48) * 24 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 51) % 51) % 51) * 1152 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 256) % 256) * 58752) + 0) + 0x01] =
                                                I_beamQ1
                                        end
                                    end
                                    I_beamQ0 = zero(Float16x2)
                                    I_beamQ1 = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let
                            t = 2
                            G_beamQ0_polr0 = zero(Float16x2)
                            G_beamQ1_polr0 = zero(Float16x2)
                            G_beamQ0_polr1 = zero(Float16x2)
                            G_beamQ1_polr1 = zero(Float16x2)
                            if let
                                thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                mlo = thread ÷ (4i32)
                                mlo < 6
                            end
                                G_beamQ0_polr0 = Gsh_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 0 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ1_polr0 = Gsh_shared[((((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 0 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ0_polr1 = Gsh_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ1_polr1 = Gsh_shared[((((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 32 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                            end
                            X_beamQ0_polr0 = G_beamQ0_polr0
                            X_beamQ1_polr0 = G_beamQ1_polr0
                            X_beamQ0_polr1 = G_beamQ0_polr1
                            X_beamQ1_polr1 = G_beamQ1_polr1
                            Z_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ1_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ1_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ1_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Z_beamQ1_cplx1_polr1 = zero(Float16x2)
                            (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ0_polr0, (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0)
                            )
                            (Z_beamQ1_cplx0_polr0, Z_beamQ1_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ1_polr0, (Z_beamQ1_cplx0_polr0, Z_beamQ1_cplx1_polr0)
                            )
                            (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ0_polr1, (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1)
                            )
                            (Z_beamQ1_cplx0_polr1, Z_beamQ1_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ1_polr1, (Z_beamQ1_cplx0_polr1, Z_beamQ1_cplx1_polr1)
                            )
                            Γ²re = Γ²_cplx0
                            Γ²im = Γ²_cplx1
                            Zre_beamQ0_polr0 = Z_beamQ0_cplx0_polr0
                            Zim_beamQ0_polr0 = Z_beamQ0_cplx1_polr0
                            Zre_beamQ1_polr0 = Z_beamQ1_cplx0_polr0
                            Zim_beamQ1_polr0 = Z_beamQ1_cplx1_polr0
                            Zre_beamQ0_polr1 = Z_beamQ0_cplx0_polr1
                            Zim_beamQ0_polr1 = Z_beamQ0_cplx1_polr1
                            Zre_beamQ1_polr1 = Z_beamQ1_cplx0_polr1
                            Zim_beamQ1_polr1 = Z_beamQ1_cplx1_polr1
                            Vre_beamQ0_polr0 = muladd(Γ²re, Zre_beamQ0_polr0, -Γ²im * Zim_beamQ0_polr0)
                            Vre_beamQ1_polr0 = muladd(Γ²re, Zre_beamQ1_polr0, -Γ²im * Zim_beamQ1_polr0)
                            Vre_beamQ0_polr1 = muladd(Γ²re, Zre_beamQ0_polr1, -Γ²im * Zim_beamQ0_polr1)
                            Vre_beamQ1_polr1 = muladd(Γ²re, Zre_beamQ1_polr1, -Γ²im * Zim_beamQ1_polr1)
                            Vim_beamQ0_polr0 = muladd(Γ²re, Zim_beamQ0_polr0, +Γ²im * Zre_beamQ0_polr0)
                            Vim_beamQ1_polr0 = muladd(Γ²re, Zim_beamQ1_polr0, +Γ²im * Zre_beamQ1_polr0)
                            Vim_beamQ0_polr1 = muladd(Γ²re, Zim_beamQ0_polr1, +Γ²im * Zre_beamQ0_polr1)
                            Vim_beamQ1_polr1 = muladd(Γ²re, Zim_beamQ1_polr1, +Γ²im * Zre_beamQ1_polr1)
                            V_beamQ0_cplx0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx1_polr0 = Vim_beamQ0_polr0
                            V_beamQ1_cplx0_polr0 = Vre_beamQ1_polr0
                            V_beamQ1_cplx1_polr0 = Vim_beamQ1_polr0
                            V_beamQ0_cplx0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx1_polr1 = Vim_beamQ0_polr1
                            V_beamQ1_cplx0_polr1 = Vre_beamQ1_polr1
                            V_beamQ1_cplx1_polr1 = Vim_beamQ1_polr1
                            Y_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ1_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ1_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ1_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Y_beamQ1_cplx1_polr1 = zero(Float16x2)
                            Vre_beamQ0_polr0 = V_beamQ0_cplx0_polr0
                            Vim_beamQ0_polr0 = V_beamQ0_cplx1_polr0
                            Vre_beamQ1_polr0 = V_beamQ1_cplx0_polr0
                            Vim_beamQ1_polr0 = V_beamQ1_cplx1_polr0
                            Vre_beamQ0_polr1 = V_beamQ0_cplx0_polr1
                            Vim_beamQ0_polr1 = V_beamQ0_cplx1_polr1
                            Vre_beamQ1_polr1 = V_beamQ1_cplx0_polr1
                            Vim_beamQ1_polr1 = V_beamQ1_cplx1_polr1
                            V_beamQ0_cplx_in0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx_in1_polr0 = Vim_beamQ0_polr0
                            V_beamQ1_cplx_in0_polr0 = Vre_beamQ1_polr0
                            V_beamQ1_cplx_in1_polr0 = Vim_beamQ1_polr0
                            V_beamQ0_cplx_in0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx_in1_polr1 = Vim_beamQ0_polr1
                            V_beamQ1_cplx_in0_polr1 = Vre_beamQ1_polr1
                            V_beamQ1_cplx_in1_polr1 = Vim_beamQ1_polr1
                            (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr0, V_beamQ0_cplx_in1_polr0),
                                (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0),
                            )
                            (Y_beamQ1_cplx0_polr0, Y_beamQ1_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ1_cplx_in0_polr0, V_beamQ1_cplx_in1_polr0),
                                (Y_beamQ1_cplx0_polr0, Y_beamQ1_cplx1_polr0),
                            )
                            (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr1, V_beamQ0_cplx_in1_polr1),
                                (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1),
                            )
                            (Y_beamQ1_cplx0_polr1, Y_beamQ1_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ1_cplx_in0_polr1, V_beamQ1_cplx_in1_polr1),
                                (Y_beamQ1_cplx0_polr1, Y_beamQ1_cplx1_polr1),
                            )
                            Ẽ_beamQ0_cplx0_polr0 = Y_beamQ0_cplx0_polr0
                            Ẽ_beamQ1_cplx0_polr0 = Y_beamQ1_cplx0_polr0
                            Ẽ_beamQ0_cplx1_polr0 = Y_beamQ0_cplx1_polr0
                            Ẽ_beamQ1_cplx1_polr0 = Y_beamQ1_cplx1_polr0
                            Ẽ_beamQ0_cplx0_polr1 = Y_beamQ0_cplx0_polr1
                            Ẽ_beamQ1_cplx0_polr1 = Y_beamQ1_cplx0_polr1
                            Ẽ_beamQ0_cplx1_polr1 = Y_beamQ0_cplx1_polr1
                            Ẽ_beamQ1_cplx1_polr1 = Y_beamQ1_cplx1_polr1
                            Ẽp0_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr0
                            Ẽp1_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr1
                            Ẽp0_beamQ1_cplx0 = Ẽ_beamQ1_cplx0_polr0
                            Ẽp1_beamQ1_cplx0 = Ẽ_beamQ1_cplx0_polr1
                            Ẽp0_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr0
                            Ẽp1_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr1
                            Ẽp0_beamQ1_cplx1 = Ẽ_beamQ1_cplx1_polr0
                            Ẽp1_beamQ1_cplx1 = Ẽ_beamQ1_cplx1_polr1
                            Ẽp0re_beamQ0 = Ẽp0_beamQ0_cplx0
                            Ẽp0im_beamQ0 = Ẽp0_beamQ0_cplx1
                            Ẽp0re_beamQ1 = Ẽp0_beamQ1_cplx0
                            Ẽp0im_beamQ1 = Ẽp0_beamQ1_cplx1
                            Ẽp1re_beamQ0 = Ẽp1_beamQ0_cplx0
                            Ẽp1im_beamQ0 = Ẽp1_beamQ0_cplx1
                            Ẽp1re_beamQ1 = Ẽp1_beamQ1_cplx0
                            Ẽp1im_beamQ1 = Ẽp1_beamQ1_cplx1
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
                            I_beamQ1 = muladd(
                                Float16x2(0.003124237f0, 0.003124237f0),
                                muladd(
                                    Ẽp1im_beamQ1,
                                    Ẽp1im_beamQ1,
                                    muladd(
                                        Ẽp1re_beamQ1, Ẽp1re_beamQ1, muladd(Ẽp0im_beamQ1, Ẽp0im_beamQ1, Ẽp0re_beamQ1 * Ẽp0re_beamQ1)
                                    ),
                                ),
                                I_beamQ1,
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
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 48) * 24 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 51) % 51) % 51) * 1152 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 256) % 256) * 58752) + 0) + 0x01] =
                                                I_beamQ0
                                        end
                                        if true
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2 + 1) % 48) * 24 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 51) % 51) % 51) * 1152 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 256) % 256) * 58752) + 0) + 0x01] =
                                                I_beamQ1
                                        end
                                    end
                                    I_beamQ0 = zero(Float16x2)
                                    I_beamQ1 = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let
                            t = 3
                            G_beamQ0_polr0 = zero(Float16x2)
                            G_beamQ1_polr0 = zero(Float16x2)
                            G_beamQ0_polr1 = zero(Float16x2)
                            G_beamQ1_polr1 = zero(Float16x2)
                            if let
                                thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                                mlo = thread ÷ (4i32)
                                mlo < 6
                            end
                                G_beamQ0_polr0 = Gsh_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 0 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ1_polr0 = Gsh_shared[((((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 0 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ0_polr1 = Gsh_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                                G_beamQ1_polr1 = Gsh_shared[((((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 8) % 2) * 1028 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 16) % 2) * 514 + 32 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 2) % 2) * 4112 + ((((((IndexSpaces.assume_inrange(t_inner_lo, 0, 4, 24) ÷ 4) % 6) * 4 + t % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 48, 2064) ÷ 48) % 43) * 48) + ((IndexSpaces.assume_inrange(t_inner_hi, 0, 24, 48) ÷ 24) % 2) * 24) % 4) * 64 + ((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 2) * 8256 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 4) % 2) * 2056 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) % 4) * 6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 8) % 6 + (((1 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) ÷ 32) % 2) * 257) + 0x01]
                            end
                            X_beamQ0_polr0 = G_beamQ0_polr0
                            X_beamQ1_polr0 = G_beamQ1_polr0
                            X_beamQ0_polr1 = G_beamQ0_polr1
                            X_beamQ1_polr1 = G_beamQ1_polr1
                            Z_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ1_cplx0_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ1_cplx1_polr0 = zero(Float16x2)
                            Z_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ1_cplx0_polr1 = zero(Float16x2)
                            Z_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Z_beamQ1_cplx1_polr1 = zero(Float16x2)
                            (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ0_polr0, (Z_beamQ0_cplx0_polr0, Z_beamQ0_cplx1_polr0)
                            )
                            (Z_beamQ1_cplx0_polr0, Z_beamQ1_cplx1_polr0) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ1_polr0, (Z_beamQ1_cplx0_polr0, Z_beamQ1_cplx1_polr0)
                            )
                            (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ0_polr1, (Z_beamQ0_cplx0_polr1, Z_beamQ0_cplx1_polr1)
                            )
                            (Z_beamQ1_cplx0_polr1, Z_beamQ1_cplx1_polr1) = IndexSpaces.mma_m16n8k8(
                                (Γ¹_cplx0, Γ¹_cplx1), X_beamQ1_polr1, (Z_beamQ1_cplx0_polr1, Z_beamQ1_cplx1_polr1)
                            )
                            Γ²re = Γ²_cplx0
                            Γ²im = Γ²_cplx1
                            Zre_beamQ0_polr0 = Z_beamQ0_cplx0_polr0
                            Zim_beamQ0_polr0 = Z_beamQ0_cplx1_polr0
                            Zre_beamQ1_polr0 = Z_beamQ1_cplx0_polr0
                            Zim_beamQ1_polr0 = Z_beamQ1_cplx1_polr0
                            Zre_beamQ0_polr1 = Z_beamQ0_cplx0_polr1
                            Zim_beamQ0_polr1 = Z_beamQ0_cplx1_polr1
                            Zre_beamQ1_polr1 = Z_beamQ1_cplx0_polr1
                            Zim_beamQ1_polr1 = Z_beamQ1_cplx1_polr1
                            Vre_beamQ0_polr0 = muladd(Γ²re, Zre_beamQ0_polr0, -Γ²im * Zim_beamQ0_polr0)
                            Vre_beamQ1_polr0 = muladd(Γ²re, Zre_beamQ1_polr0, -Γ²im * Zim_beamQ1_polr0)
                            Vre_beamQ0_polr1 = muladd(Γ²re, Zre_beamQ0_polr1, -Γ²im * Zim_beamQ0_polr1)
                            Vre_beamQ1_polr1 = muladd(Γ²re, Zre_beamQ1_polr1, -Γ²im * Zim_beamQ1_polr1)
                            Vim_beamQ0_polr0 = muladd(Γ²re, Zim_beamQ0_polr0, +Γ²im * Zre_beamQ0_polr0)
                            Vim_beamQ1_polr0 = muladd(Γ²re, Zim_beamQ1_polr0, +Γ²im * Zre_beamQ1_polr0)
                            Vim_beamQ0_polr1 = muladd(Γ²re, Zim_beamQ0_polr1, +Γ²im * Zre_beamQ0_polr1)
                            Vim_beamQ1_polr1 = muladd(Γ²re, Zim_beamQ1_polr1, +Γ²im * Zre_beamQ1_polr1)
                            V_beamQ0_cplx0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx1_polr0 = Vim_beamQ0_polr0
                            V_beamQ1_cplx0_polr0 = Vre_beamQ1_polr0
                            V_beamQ1_cplx1_polr0 = Vim_beamQ1_polr0
                            V_beamQ0_cplx0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx1_polr1 = Vim_beamQ0_polr1
                            V_beamQ1_cplx0_polr1 = Vre_beamQ1_polr1
                            V_beamQ1_cplx1_polr1 = Vim_beamQ1_polr1
                            Y_beamQ0_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ1_cplx0_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ1_cplx1_polr0 = zero(Float16x2)
                            Y_beamQ0_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ1_cplx0_polr1 = zero(Float16x2)
                            Y_beamQ0_cplx1_polr1 = zero(Float16x2)
                            Y_beamQ1_cplx1_polr1 = zero(Float16x2)
                            Vre_beamQ0_polr0 = V_beamQ0_cplx0_polr0
                            Vim_beamQ0_polr0 = V_beamQ0_cplx1_polr0
                            Vre_beamQ1_polr0 = V_beamQ1_cplx0_polr0
                            Vim_beamQ1_polr0 = V_beamQ1_cplx1_polr0
                            Vre_beamQ0_polr1 = V_beamQ0_cplx0_polr1
                            Vim_beamQ0_polr1 = V_beamQ0_cplx1_polr1
                            Vre_beamQ1_polr1 = V_beamQ1_cplx0_polr1
                            Vim_beamQ1_polr1 = V_beamQ1_cplx1_polr1
                            V_beamQ0_cplx_in0_polr0 = Vre_beamQ0_polr0
                            V_beamQ0_cplx_in1_polr0 = Vim_beamQ0_polr0
                            V_beamQ1_cplx_in0_polr0 = Vre_beamQ1_polr0
                            V_beamQ1_cplx_in1_polr0 = Vim_beamQ1_polr0
                            V_beamQ0_cplx_in0_polr1 = Vre_beamQ0_polr1
                            V_beamQ0_cplx_in1_polr1 = Vim_beamQ0_polr1
                            V_beamQ1_cplx_in0_polr1 = Vre_beamQ1_polr1
                            V_beamQ1_cplx_in1_polr1 = Vim_beamQ1_polr1
                            (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr0, V_beamQ0_cplx_in1_polr0),
                                (Y_beamQ0_cplx0_polr0, Y_beamQ0_cplx1_polr0),
                            )
                            (Y_beamQ1_cplx0_polr0, Y_beamQ1_cplx1_polr0) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ1_cplx_in0_polr0, V_beamQ1_cplx_in1_polr0),
                                (Y_beamQ1_cplx0_polr0, Y_beamQ1_cplx1_polr0),
                            )
                            (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ0_cplx_in0_polr1, V_beamQ0_cplx_in1_polr1),
                                (Y_beamQ0_cplx0_polr1, Y_beamQ0_cplx1_polr1),
                            )
                            (Y_beamQ1_cplx0_polr1, Y_beamQ1_cplx1_polr1) = IndexSpaces.mma_m16n8k16(
                                (Γ³_cplx0_cplx_in0, Γ³_cplx1_cplx_in0, Γ³_cplx0_cplx_in1, Γ³_cplx1_cplx_in1),
                                (V_beamQ1_cplx_in0_polr1, V_beamQ1_cplx_in1_polr1),
                                (Y_beamQ1_cplx0_polr1, Y_beamQ1_cplx1_polr1),
                            )
                            Ẽ_beamQ0_cplx0_polr0 = Y_beamQ0_cplx0_polr0
                            Ẽ_beamQ1_cplx0_polr0 = Y_beamQ1_cplx0_polr0
                            Ẽ_beamQ0_cplx1_polr0 = Y_beamQ0_cplx1_polr0
                            Ẽ_beamQ1_cplx1_polr0 = Y_beamQ1_cplx1_polr0
                            Ẽ_beamQ0_cplx0_polr1 = Y_beamQ0_cplx0_polr1
                            Ẽ_beamQ1_cplx0_polr1 = Y_beamQ1_cplx0_polr1
                            Ẽ_beamQ0_cplx1_polr1 = Y_beamQ0_cplx1_polr1
                            Ẽ_beamQ1_cplx1_polr1 = Y_beamQ1_cplx1_polr1
                            Ẽp0_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr0
                            Ẽp1_beamQ0_cplx0 = Ẽ_beamQ0_cplx0_polr1
                            Ẽp0_beamQ1_cplx0 = Ẽ_beamQ1_cplx0_polr0
                            Ẽp1_beamQ1_cplx0 = Ẽ_beamQ1_cplx0_polr1
                            Ẽp0_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr0
                            Ẽp1_beamQ0_cplx1 = Ẽ_beamQ0_cplx1_polr1
                            Ẽp0_beamQ1_cplx1 = Ẽ_beamQ1_cplx1_polr0
                            Ẽp1_beamQ1_cplx1 = Ẽ_beamQ1_cplx1_polr1
                            Ẽp0re_beamQ0 = Ẽp0_beamQ0_cplx0
                            Ẽp0im_beamQ0 = Ẽp0_beamQ0_cplx1
                            Ẽp0re_beamQ1 = Ẽp0_beamQ1_cplx0
                            Ẽp0im_beamQ1 = Ẽp0_beamQ1_cplx1
                            Ẽp1re_beamQ0 = Ẽp1_beamQ0_cplx0
                            Ẽp1im_beamQ0 = Ẽp1_beamQ0_cplx1
                            Ẽp1re_beamQ1 = Ẽp1_beamQ1_cplx0
                            Ẽp1im_beamQ1 = Ẽp1_beamQ1_cplx1
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
                            I_beamQ1 = muladd(
                                Float16x2(0.003124237f0, 0.003124237f0),
                                muladd(
                                    Ẽp1im_beamQ1,
                                    Ẽp1im_beamQ1,
                                    muladd(
                                        Ẽp1re_beamQ1, Ẽp1re_beamQ1, muladd(Ẽp0im_beamQ1, Ẽp0im_beamQ1, Ẽp0re_beamQ1 * Ẽp0re_beamQ1)
                                    ),
                                ),
                                I_beamQ1,
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
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2) % 48) * 24 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 51) % 51) % 51) * 1152 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 256) % 256) * 58752) + 0) + 0x01] =
                                                I_beamQ0
                                        end
                                        if true
                                            I_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) * 2 + 1) % 48) * 24 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) * 2) ÷ 2) % 24 + ((IndexSpaces.assume_inrange(dstime, 0, 1, 51) % 51) % 51) * 1152 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 256) % 256) * 58752) + 0) + 0x01] =
                                                I_beamQ1
                                        end
                                    end
                                    I_beamQ0 = zero(Float16x2)
                                    I_beamQ1 = zero(Float16x2)
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
            info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 256) % 256) * 768 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 24) % 24) % 24) * 32) + 0) + 0x01] =
                info
        end
    end
)
