# Julia source code for CUDA frb beamformer
# This file has been generated automatically by `frb.jl`.
# Do not modify this file, your changes will be lost.

@fastmath @inbounds(
    begin #= /localhome/eschnett/src/kotekan/julia/kernels/frb.jl:1857 =#
        info = 1
        if true
            info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 384) * 192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) % 6) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32) + 0) + 0x01] =
                info
        end
        if !(
            0i32 ≤ Tbarmin < 32768 && (
                Tbarmin ≤ Tbarmax < 65536 && (
                    (Tbarmax - Tbarmin) % 48 == 0i32 && (
                        0i32 ≤ Ttildemin < 512 &&
                        (Ttildemin ≤ Ttildemax < 1024 && Ttildemax - Ttildemin == (Tbarmax - Tbarmin) ÷ 256)
                    )
                )
            )
        )
            info = 2
            if true
                info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 384) * 192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) % 6) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32) + 0) + 0x01] =
                    info
            end
            IndexSpaces.cuda_trap()
        end
        if !(
            0i32 ≤ Fbar_in_min ≤ Fbar_in_max ≤ 384 && (
                (Fbar_in_max - Fbar_in_min) % 1 == 0i32 && (
                    0i32 ≤ Fbar_out_min ≤ Fbar_out_max ≤ 4096 &&
                    ((Fbar_out_max - Fbar_out_min) % 1 == 0i32 && Fbar_out_max - Fbar_out_min == Fbar_in_max - Fbar_in_min)
                )
            )
        )
            info = 3
            if true
                info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 384) * 192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) % 6) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32) + 0) + 0x01] =
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
            d0 = (thread % 2) * (2i32) + 0i32
            d1 = (thread % 2) * (2i32) + 1i32
            v = thread ÷ (4i32)
            δ0 = (δ1 = (Γ²_d0 = if d0 < 3
                cispi((((d0 * v) % 24) / 12.0f0) % 2.0f0)
            else
                Complex(0.0f0)
            end))
            Γ²_d1 = if d1 < 3
                cispi((((d1 * v) % 24) / 12.0f0) % 2.0f0)
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
            Γ³_d0 = if d0 < 3 && u < 3
                cispi((((d0 * u) % 3) / 1.5f0) % 2.0f0)
            else
                Complex(0.0f0)
            end
            Γ³_d1 = if d1 < 3 && u < 3
                cispi((((d1 * u) % 3) / 1.5f0) % 2.0f0)
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
            d0 = (thread % 1) * (2i32) + 0i32
            d1 = (thread % 1) * (2i32) + 1i32
            v = thread ÷ (4i32)
            δ0 = (δ1 = (Γ²_d0 = if d0 < 2
                cispi((((d0 * v) % 16) / 8.0f0) % 2.0f0)
            else
                Complex(0.0f0)
            end))
            Γ²_d1 = if d1 < 2
                cispi((((d1 * v) % 16) / 8.0f0) % 2.0f0)
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
            d0 = 0i32
            d1 = 1i32
            u = (thread ÷ (4i32)) % (2i32)
            s1in = (thread ÷ (1i32)) % (2i32)
            s1out = (thread ÷ (16i32)) % (2i32)
            δ = s1in == s1out
            Γ³_d0 = if d0 < 2 && (u < 2 && δ)
                cispi((((d0 * u) % 2) / 1.0f0) % 2.0f0)
            else
                Complex(0.0f0)
            end
            Γ³_d1 = if d1 < 2 && (u < 2 && δ)
                cispi((((d1 * u) % 2) / 1.0f0) % 2.0f0)
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
            Smn = Smn_memory[(IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 16) * 6) % 96 + 0x01]
            (Smn_mn0, Smn_mn1) = convert(NTuple{2,Int32}, Smn)
            Sm = Smn_mn0
            Sn = Smn_mn1
            if !(0i32 ≤ Sm < 8 && 0i32 ≤ Sn < 12)
                CUDA.@cuprintf "thread=%d warp=%d block=%d Sm=%d Sn=%d\n" Cint((threadIdx()).x - 1) Cint((threadIdx()).y - 1) Cint(
                    (blockIdx()).x - 1
                ) Cint(Sm) Cint(Sn)                    #= /localhome/eschnett/src/kotekan/julia/kernels/frb.jl:1625 =#
                info = 4
                if true
                    info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 384) * 192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) % 6) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32) + 0) + 0x01] =
                        info
                end
                IndexSpaces.cuda_trap()
            end
            S = (33i32) * Sm + 290 * Sn
        end
        W_dishM0_polr0 = zero(Float16x2)
        W_dishM4_polr0 = zero(Float16x2)
        W_dishM0_polr1 = zero(Float16x2)
        W_dishM4_polr1 = zero(Float16x2)
        if let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            nlo = 2 * (thread ÷ 16)
            nlo < 3
        end
            W_dishM0_polr0 = W_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2) % 3) * 8 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2 + ((0::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 2) * 2) % 8 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 128) * 192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 24 + ((0::Int32 % 2) % 2) * 96) + 0x01]
            W_dishM4_polr0 = W_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2) % 3) * 8 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2 + ((4::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 2) * 2) % 8 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 128) * 192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 24 + ((0::Int32 % 2) % 2) * 96) + 0x01]
            W_dishM0_polr1 = W_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2) % 3) * 8 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2 + ((0::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 2) * 2) % 8 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 128) * 192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 24 + ((1::Int32 % 2) % 2) * 96) + 0x01]
            W_dishM4_polr1 = W_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2) % 3) * 8 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2 + ((4::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 2) * 2) % 8 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 128) * 192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 24 + ((1::Int32 % 2) % 2) * 96) + 0x01]
        end
        I = zero(Float16x2)
        dstime = 0
        t_running = 0
        for t_outer in 0:48:32735
            Tbarmin + t_outer ≥ Tbarmax && break
            let
                (E_dish0_time0, E_dish4_time0, E_dish8_time0, E_dish12_time0) = IndexSpaces.unsafe_load4_global(
                    E_memory,
                    let
                        offset = 12288 * Tbarmin + 32 * Fbar_in_min
                        length = 402653184
                        mod(
                            (
                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 384) * 32 +
                                (
                                    (
                                        (
                                            (
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 +
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4
                                            ) + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48
                                        ) + ((0::Int32 ÷ 24) % 2) * 24
                                    ) % 32768
                                ) * 12288 +
                                (
                                    (
                                        ((0::Int32 ÷ 4) % 4) * 4 +
                                        (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16
                                    ) ÷ 4
                                ) % 16 +
                                (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 2) * 16
                            ) + offset,
                            length,
                        )
                    end + 1i32,
                )
                (E_dish0_time24, E_dish4_time24, E_dish8_time24, E_dish12_time24) = IndexSpaces.unsafe_load4_global(
                    E_memory,
                    let
                        offset = 12288 * Tbarmin + 32 * Fbar_in_min
                        length = 402653184
                        mod(
                            (
                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 384) * 32 +
                                (
                                    (
                                        (
                                            (
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 +
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4
                                            ) + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48
                                        ) + ((24::Int32 ÷ 24) % 2) * 24
                                    ) % 32768
                                ) * 12288 +
                                (
                                    (
                                        ((0::Int32 ÷ 4) % 4) * 4 +
                                        (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16
                                    ) ÷ 4
                                ) % 16 +
                                (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 2) * 16
                            ) + offset,
                            length,
                        )
                    end + 1i32,
                )
                is_lo_thread = IndexSpaces.cuda_threadidx() & 0x00000004 == 0x00
                (E_dish0_time0, E_dish8_time0) = let
                    src = if is_lo_thread
                        E_dish8_time0
                    else
                        E_dish0_time0
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
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
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                    if is_lo_thread
                        (E_dish4_time0, dst)
                    else
                        (dst, E_dish12_time0)
                    end
                end
                (E_dish0_time24, E_dish8_time24) = let
                    src = if is_lo_thread
                        E_dish8_time24
                    else
                        E_dish0_time24
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
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
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                    if is_lo_thread
                        (E_dish4_time24, dst)
                    else
                        (dst, E_dish12_time24)
                    end
                end
                (E_dish0_time0, E_dish8_time0) = (
                    IndexSpaces.get_lo4(E_dish0_time0, E_dish8_time0), IndexSpaces.get_hi4(E_dish0_time0, E_dish8_time0)
                )
                (E_dish4_time0, E_dish12_time0) = (
                    IndexSpaces.get_lo4(E_dish4_time0, E_dish12_time0), IndexSpaces.get_hi4(E_dish4_time0, E_dish12_time0)
                )
                (E_dish0_time24, E_dish8_time24) = (
                    IndexSpaces.get_lo4(E_dish0_time24, E_dish8_time24), IndexSpaces.get_hi4(E_dish0_time24, E_dish8_time24)
                )
                (E_dish4_time24, E_dish12_time24) = (
                    IndexSpaces.get_lo4(E_dish4_time24, E_dish12_time24), IndexSpaces.get_hi4(E_dish4_time24, E_dish12_time24)
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
                (E_dish0_time0, E_dish8_time0) = (
                    IndexSpaces.get_lo16(E_dish0_time0, E_dish8_time0), IndexSpaces.get_hi16(E_dish0_time0, E_dish8_time0)
                )
                (E_dish4_time0, E_dish12_time0) = (
                    IndexSpaces.get_lo16(E_dish4_time0, E_dish12_time0), IndexSpaces.get_hi16(E_dish4_time0, E_dish12_time0)
                )
                (E_dish0_time24, E_dish8_time24) = (
                    IndexSpaces.get_lo16(E_dish0_time24, E_dish8_time24), IndexSpaces.get_hi16(E_dish0_time24, E_dish8_time24)
                )
                (E_dish4_time24, E_dish12_time24) = (
                    IndexSpaces.get_lo16(E_dish4_time24, E_dish12_time24), IndexSpaces.get_hi16(E_dish4_time24, E_dish12_time24)
                )
                if true
                    Fsh1_shared[(((((((((0::Int32 ÷ 24) % 2 + ((0::Int32 ÷ 8) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8) + ((0::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16) ÷ 8) % 8) * 260 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) % 24 + ((((((0::Int32 ÷ 24) % 2 + ((0::Int32 ÷ 8) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8) + ((0::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16) % 8) * 32) + 0) + 0x01] =
                        E_dish0_time0
                end
                if true
                    Fsh1_shared[(((((((((0::Int32 ÷ 24) % 2 + ((4::Int32 ÷ 8) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8) + ((4::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16) ÷ 8) % 8) * 260 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) % 24 + ((((((0::Int32 ÷ 24) % 2 + ((4::Int32 ÷ 8) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8) + ((4::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16) % 8) * 32) + 0) + 0x01] =
                        E_dish4_time0
                end
                if true
                    Fsh1_shared[(((((((((0::Int32 ÷ 24) % 2 + ((8::Int32 ÷ 8) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8) + ((8::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16) ÷ 8) % 8) * 260 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) % 24 + ((((((0::Int32 ÷ 24) % 2 + ((8::Int32 ÷ 8) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8) + ((8::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16) % 8) * 32) + 0) + 0x01] =
                        E_dish8_time0
                end
                if true
                    Fsh1_shared[(((((((((0::Int32 ÷ 24) % 2 + ((12::Int32 ÷ 8) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8) + ((12::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16) ÷ 8) % 8) * 260 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) % 24 + ((((((0::Int32 ÷ 24) % 2 + ((12::Int32 ÷ 8) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8) + ((12::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16) % 8) * 32) + 0) + 0x01] =
                        E_dish12_time0
                end
                if true
                    Fsh1_shared[(((((((((24::Int32 ÷ 24) % 2 + ((0::Int32 ÷ 8) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8) + ((0::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16) ÷ 8) % 8) * 260 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) % 24 + ((((((24::Int32 ÷ 24) % 2 + ((0::Int32 ÷ 8) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8) + ((0::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16) % 8) * 32) + 0) + 0x01] =
                        E_dish0_time24
                end
                if true
                    Fsh1_shared[(((((((((24::Int32 ÷ 24) % 2 + ((4::Int32 ÷ 8) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8) + ((4::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16) ÷ 8) % 8) * 260 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) % 24 + ((((((24::Int32 ÷ 24) % 2 + ((4::Int32 ÷ 8) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8) + ((4::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16) % 8) * 32) + 0) + 0x01] =
                        E_dish4_time24
                end
                if true
                    Fsh1_shared[(((((((((24::Int32 ÷ 24) % 2 + ((8::Int32 ÷ 8) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8) + ((8::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16) ÷ 8) % 8) * 260 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) % 24 + ((((((24::Int32 ÷ 24) % 2 + ((8::Int32 ÷ 8) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8) + ((8::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16) % 8) * 32) + 0) + 0x01] =
                        E_dish8_time24
                end
                if true
                    Fsh1_shared[(((((((((24::Int32 ÷ 24) % 2 + ((12::Int32 ÷ 8) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8) + ((12::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16) ÷ 8) % 8) * 260 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) % 24 + ((((((24::Int32 ÷ 24) % 2 + ((12::Int32 ÷ 8) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8) + ((12::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16) % 8) * 32) + 0) + 0x01] =
                        E_dish12_time24
                end
                IndexSpaces.cuda_sync_threads()
            end
            let
                Freg1_dish0 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6 + ((0::Int32 ÷ 6) % 11) * 6) ÷ 8) % 8) * 260 + (((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 24) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6 + ((0::Int32 ÷ 6) % 11) * 6) % 8) * 32) + 0x01]
                Freg1_dish6 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6 + ((6::Int32 ÷ 6) % 11) * 6) ÷ 8) % 8) * 260 + (((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 24) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6 + ((6::Int32 ÷ 6) % 11) * 6) % 8) * 32) + 0x01]
                Freg1_dish12 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6 + ((12::Int32 ÷ 6) % 11) * 6) ÷ 8) % 8) * 260 + (((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 24) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6 + ((12::Int32 ÷ 6) % 11) * 6) % 8) * 32) + 0x01]
                Freg1_dish18 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6 + ((18::Int32 ÷ 6) % 11) * 6) ÷ 8) % 8) * 260 + (((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 24) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6 + ((18::Int32 ÷ 6) % 11) * 6) % 8) * 32) + 0x01]
                Freg1_dish24 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6 + ((24::Int32 ÷ 6) % 11) * 6) ÷ 8) % 8) * 260 + (((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 24) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6 + ((24::Int32 ÷ 6) % 11) * 6) % 8) * 32) + 0x01]
                Freg1_dish30 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6 + ((30::Int32 ÷ 6) % 11) * 6) ÷ 8) % 8) * 260 + (((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 24) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6 + ((30::Int32 ÷ 6) % 11) * 6) % 8) * 32) + 0x01]
                Freg1_dish36 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6 + ((36::Int32 ÷ 6) % 11) * 6) ÷ 8) % 8) * 260 + (((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 24) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6 + ((36::Int32 ÷ 6) % 11) * 6) % 8) * 32) + 0x01]
                Freg1_dish42 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6 + ((42::Int32 ÷ 6) % 11) * 6) ÷ 8) % 8) * 260 + (((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 24) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6 + ((42::Int32 ÷ 6) % 11) * 6) % 8) * 32) + 0x01]
                Freg1_dish48 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6 + ((48::Int32 ÷ 6) % 11) * 6) ÷ 8) % 8) * 260 + (((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 24) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6 + ((48::Int32 ÷ 6) % 11) * 6) % 8) * 32) + 0x01]
                Freg1_dish54 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6 + ((54::Int32 ÷ 6) % 11) * 6) ÷ 8) % 8) * 260 + (((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 24) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6 + ((54::Int32 ÷ 6) % 11) * 6) % 8) * 32) + 0x01]
                Freg1_dish60 = Fsh1_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6 + ((60::Int32 ÷ 6) % 11) * 6) ÷ 8) % 8) * 260 + (((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 24) % 24 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6 + ((60::Int32 ÷ 6) % 11) * 6) % 8) * 32) + 0x01]
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
                if sd_sd0 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 384) * 192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) % 6) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 24) % 24 + sd_sd0) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish6
                if sd_sd1 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 384) * 192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) % 6) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 24) % 24 + sd_sd1) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish12
                if sd_sd2 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 384) * 192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) % 6) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 24) % 24 + sd_sd2) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish18
                if sd_sd3 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 384) * 192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) % 6) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 24) % 24 + sd_sd3) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish24
                if sd_sd4 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 384) * 192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) % 6) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 24) % 24 + sd_sd4) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish30
                if sd_sd5 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 384) * 192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) % 6) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 24) % 24 + sd_sd5) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish36
                if sd_sd6 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 384) * 192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) % 6) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 24) % 24 + sd_sd6) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish42
                if sd_sd7 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 384) * 192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) % 6) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 24) % 24 + sd_sd7) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish48
                if sd_sd8 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 384) * 192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) % 6) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 24) % 24 + sd_sd8) + 0x01] =
                        Freg1′
                end
                Freg1′ = Freg1_dish54
                if sd_sd9 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 384) * 192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) % 6) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 24) % 24 + sd_sd9) + 0x01] =
                        Freg1′
                end
                Freg1′ = if warp = IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 6), dish = warp + 6 * 10, dish < 64
                    Freg1_dish60
                else
                    Int4x8(0, 0, 0, 0, 0, 0, 0, 0)
                end
                if sd_sd10 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 384) * 192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) % 6) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 24) % 24 + sd_sd10) + 0x01] =
                        Freg1′
                end
                Freg1′ = Int4x8(0, 0, 0, 0, 0, 0, 0, 0)
                if sd_sd11 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 384) * 192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) % 6) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 24) % 24 + sd_sd11) + 0x01] =
                        Freg1′
                end
                Freg1′ = Int4x8(0, 0, 0, 0, 0, 0, 0, 0)
                if sd_sd12 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 384) * 192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) % 6) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 24) % 24 + sd_sd12) + 0x01] =
                        Freg1′
                end
                Freg1′ = Int4x8(0, 0, 0, 0, 0, 0, 0, 0)
                if sd_sd13 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 384) * 192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) % 6) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 24) % 24 + sd_sd13) + 0x01] =
                        Freg1′
                end
                Freg1′ = Int4x8(0, 0, 0, 0, 0, 0, 0, 0)
                if sd_sd14 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 384) * 192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) % 6) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 24) % 24 + sd_sd14) + 0x01] =
                        Freg1′
                end
                Freg1′ = Int4x8(0, 0, 0, 0, 0, 0, 0, 0)
                if sd_sd15 == 999999999i32
                    info = 5
                    if true
                        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 384) * 192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) % 6) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32) + 0) + 0x01] =
                            info
                    end
                    IndexSpaces.cuda_trap()
                end
                if true
                    Fsh2_shared[((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 24) % 24 + sd_sd15) + 0x01] =
                        Freg1′
                end
                IndexSpaces.cuda_sync_threads()
            end
            let
                Freg2_dishM0_time0 = zero(Int4x8)
                Freg2_dishM4_time0 = zero(Int4x8)
                Freg2_dishM0_time3 = zero(Int4x8)
                Freg2_dishM4_time3 = zero(Int4x8)
                Freg2_dishM0_time6 = zero(Int4x8)
                Freg2_dishM4_time6 = zero(Int4x8)
                Freg2_dishM0_time9 = zero(Int4x8)
                Freg2_dishM4_time9 = zero(Int4x8)
                Freg2_dishM0_time12 = zero(Int4x8)
                Freg2_dishM4_time12 = zero(Int4x8)
                Freg2_dishM0_time15 = zero(Int4x8)
                Freg2_dishM4_time15 = zero(Int4x8)
                Freg2_dishM0_time18 = zero(Int4x8)
                Freg2_dishM4_time18 = zero(Int4x8)
                Freg2_dishM0_time21 = zero(Int4x8)
                Freg2_dishM4_time21 = zero(Int4x8)
                if let
                    thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
                    nlo = (1i32) * ((thread ÷ (4i32)) % (2i32)) + (2i32) * ((thread ÷ 16) % 2)
                    nlo < 3
                end
                    Freg2_dishM0_time0 = Fsh2_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 3) * 290 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2 + ((0::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 2) * 2) % 8) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 870 + ((((0::Int32 ÷ 3) % 8) * 3 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) ÷ 2) % 3) + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) % 24) + 0x01]
                    Freg2_dishM4_time0 = Fsh2_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 3) * 290 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2 + ((4::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 2) * 2) % 8) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 870 + ((((0::Int32 ÷ 3) % 8) * 3 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) ÷ 2) % 3) + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) % 24) + 0x01]
                    Freg2_dishM0_time3 = Fsh2_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 3) * 290 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2 + ((0::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 2) * 2) % 8) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 870 + ((((3::Int32 ÷ 3) % 8) * 3 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) ÷ 2) % 3) + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) % 24) + 0x01]
                    Freg2_dishM4_time3 = Fsh2_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 3) * 290 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2 + ((4::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 2) * 2) % 8) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 870 + ((((3::Int32 ÷ 3) % 8) * 3 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) ÷ 2) % 3) + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) % 24) + 0x01]
                    Freg2_dishM0_time6 = Fsh2_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 3) * 290 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2 + ((0::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 2) * 2) % 8) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 870 + ((((6::Int32 ÷ 3) % 8) * 3 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) ÷ 2) % 3) + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) % 24) + 0x01]
                    Freg2_dishM4_time6 = Fsh2_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 3) * 290 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2 + ((4::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 2) * 2) % 8) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 870 + ((((6::Int32 ÷ 3) % 8) * 3 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) ÷ 2) % 3) + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) % 24) + 0x01]
                    Freg2_dishM0_time9 = Fsh2_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 3) * 290 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2 + ((0::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 2) * 2) % 8) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 870 + ((((9::Int32 ÷ 3) % 8) * 3 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) ÷ 2) % 3) + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) % 24) + 0x01]
                    Freg2_dishM4_time9 = Fsh2_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 3) * 290 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2 + ((4::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 2) * 2) % 8) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 870 + ((((9::Int32 ÷ 3) % 8) * 3 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) ÷ 2) % 3) + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) % 24) + 0x01]
                    Freg2_dishM0_time12 = Fsh2_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 3) * 290 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2 + ((0::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 2) * 2) % 8) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 870 + ((((12::Int32 ÷ 3) % 8) * 3 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) ÷ 2) % 3) + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) % 24) + 0x01]
                    Freg2_dishM4_time12 = Fsh2_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 3) * 290 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2 + ((4::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 2) * 2) % 8) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 870 + ((((12::Int32 ÷ 3) % 8) * 3 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) ÷ 2) % 3) + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) % 24) + 0x01]
                    Freg2_dishM0_time15 = Fsh2_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 3) * 290 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2 + ((0::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 2) * 2) % 8) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 870 + ((((15::Int32 ÷ 3) % 8) * 3 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) ÷ 2) % 3) + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) % 24) + 0x01]
                    Freg2_dishM4_time15 = Fsh2_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 3) * 290 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2 + ((4::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 2) * 2) % 8) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 870 + ((((15::Int32 ÷ 3) % 8) * 3 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) ÷ 2) % 3) + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) % 24) + 0x01]
                    Freg2_dishM0_time18 = Fsh2_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 3) * 290 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2 + ((0::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 2) * 2) % 8) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 870 + ((((18::Int32 ÷ 3) % 8) * 3 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) ÷ 2) % 3) + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) % 24) + 0x01]
                    Freg2_dishM4_time18 = Fsh2_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 3) * 290 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2 + ((4::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 2) * 2) % 8) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 870 + ((((18::Int32 ÷ 3) % 8) * 3 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) ÷ 2) % 3) + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) % 24) + 0x01]
                    Freg2_dishM0_time21 = Fsh2_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 3) * 290 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2 + ((0::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 2) * 2) % 8) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 870 + ((((21::Int32 ÷ 3) % 8) * 3 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) ÷ 2) % 3) + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) % 24) + 0x01]
                    Freg2_dishM4_time21 = Fsh2_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 3) * 290 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2 + ((4::Int32 ÷ 4) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 2) * 2) % 8) * 33 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 870 + ((((21::Int32 ÷ 3) % 8) * 3 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) ÷ 2) % 3) + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) % 24) + 0x01]
                end
                IndexSpaces.cuda_sync_threads()
                let t_inner_hi = 0
                    for t_inner_lo in 0:6:23
                        Freg2′_dishM0_time0 = zero(Int4x8)
                        if t_inner_lo == 0
                            Freg2′_dishM0_time0 = Freg2_dishM0_time0
                        end
                        if t_inner_lo == 6
                            Freg2′_dishM0_time0 = Freg2_dishM0_time6
                        end
                        if t_inner_lo == 12
                            Freg2′_dishM0_time0 = Freg2_dishM0_time12
                        end
                        if t_inner_lo == 18
                            Freg2′_dishM0_time0 = Freg2_dishM0_time18
                        end
                        Freg2′_dishM4_time0 = zero(Int4x8)
                        if t_inner_lo == 0
                            Freg2′_dishM4_time0 = Freg2_dishM4_time0
                        end
                        if t_inner_lo == 6
                            Freg2′_dishM4_time0 = Freg2_dishM4_time6
                        end
                        if t_inner_lo == 12
                            Freg2′_dishM4_time0 = Freg2_dishM4_time12
                        end
                        if t_inner_lo == 18
                            Freg2′_dishM4_time0 = Freg2_dishM4_time18
                        end
                        Freg2′_dishM0_time3 = zero(Int4x8)
                        if t_inner_lo == 0
                            Freg2′_dishM0_time3 = Freg2_dishM0_time3
                        end
                        if t_inner_lo == 6
                            Freg2′_dishM0_time3 = Freg2_dishM0_time9
                        end
                        if t_inner_lo == 12
                            Freg2′_dishM0_time3 = Freg2_dishM0_time15
                        end
                        if t_inner_lo == 18
                            Freg2′_dishM0_time3 = Freg2_dishM0_time21
                        end
                        Freg2′_dishM4_time3 = zero(Int4x8)
                        if t_inner_lo == 0
                            Freg2′_dishM4_time3 = Freg2_dishM4_time3
                        end
                        if t_inner_lo == 6
                            Freg2′_dishM4_time3 = Freg2_dishM4_time9
                        end
                        if t_inner_lo == 12
                            Freg2′_dishM4_time3 = Freg2_dishM4_time15
                        end
                        if t_inner_lo == 18
                            Freg2′_dishM4_time3 = Freg2_dishM4_time21
                        end
                        (E′_dishM0_polr0_time0, E′_dishM0_polr1_time0, E′_dishM0_polr0_time24, E′_dishM0_polr1_time24) = convert(
                            NTuple{4,Float16x2}, Freg2′_dishM0_time0
                        )
                        (E′_dishM4_polr0_time0, E′_dishM4_polr1_time0, E′_dishM4_polr0_time24, E′_dishM4_polr1_time24) = convert(
                            NTuple{4,Float16x2}, Freg2′_dishM4_time0
                        )
                        (E′_dishM0_polr0_time3, E′_dishM0_polr1_time3, E′_dishM0_polr0_time27, E′_dishM0_polr1_time27) = convert(
                            NTuple{4,Float16x2}, Freg2′_dishM0_time3
                        )
                        (E′_dishM4_polr0_time3, E′_dishM4_polr1_time3, E′_dishM4_polr0_time27, E′_dishM4_polr1_time27) = convert(
                            NTuple{4,Float16x2}, Freg2′_dishM4_time3
                        )
                        E_dishM0_polr0_time0 = E′_dishM0_polr0_time0
                        E_dishM4_polr0_time0 = E′_dishM4_polr0_time0
                        E_dishM0_polr1_time0 = E′_dishM0_polr1_time0
                        E_dishM4_polr1_time0 = E′_dishM4_polr1_time0
                        E_dishM0_polr0_time3 = E′_dishM0_polr0_time3
                        E_dishM4_polr0_time3 = E′_dishM4_polr0_time3
                        E_dishM0_polr1_time3 = E′_dishM0_polr1_time3
                        E_dishM4_polr1_time3 = E′_dishM4_polr1_time3
                        WE_dishM0_polr0_time0 = complex_mul(W_dishM0_polr0, E_dishM0_polr0_time0)
                        WE_dishM4_polr0_time0 = complex_mul(W_dishM4_polr0, E_dishM4_polr0_time0)
                        WE_dishM0_polr1_time0 = complex_mul(W_dishM0_polr1, E_dishM0_polr1_time0)
                        WE_dishM4_polr1_time0 = complex_mul(W_dishM4_polr1, E_dishM4_polr1_time0)
                        WE_dishM0_polr0_time3 = complex_mul(W_dishM0_polr0, E_dishM0_polr0_time3)
                        WE_dishM4_polr0_time3 = complex_mul(W_dishM4_polr0, E_dishM4_polr0_time3)
                        WE_dishM0_polr1_time3 = complex_mul(W_dishM0_polr1, E_dishM0_polr1_time3)
                        WE_dishM4_polr1_time3 = complex_mul(W_dishM4_polr1, E_dishM4_polr1_time3)
                        X_dishM0_polr0_time0 = WE_dishM0_polr0_time0
                        X_dishM4_polr0_time0 = WE_dishM4_polr0_time0
                        X_dishM0_polr1_time0 = WE_dishM0_polr1_time0
                        X_dishM4_polr1_time0 = WE_dishM4_polr1_time0
                        X_dishM0_polr0_time3 = WE_dishM0_polr0_time3
                        X_dishM4_polr0_time3 = WE_dishM4_polr0_time3
                        X_dishM0_polr1_time3 = WE_dishM0_polr1_time3
                        X_dishM4_polr1_time3 = WE_dishM4_polr1_time3
                        Z_cplx0_dishM0_polr0_time0 = zero(Float16x2)
                        Z_cplx1_dishM0_polr0_time0 = zero(Float16x2)
                        Z_cplx0_dishM4_polr0_time0 = zero(Float16x2)
                        Z_cplx1_dishM4_polr0_time0 = zero(Float16x2)
                        Z_cplx0_dishM0_polr1_time0 = zero(Float16x2)
                        Z_cplx1_dishM0_polr1_time0 = zero(Float16x2)
                        Z_cplx0_dishM4_polr1_time0 = zero(Float16x2)
                        Z_cplx1_dishM4_polr1_time0 = zero(Float16x2)
                        Z_cplx0_dishM0_polr0_time3 = zero(Float16x2)
                        Z_cplx1_dishM0_polr0_time3 = zero(Float16x2)
                        Z_cplx0_dishM4_polr0_time3 = zero(Float16x2)
                        Z_cplx1_dishM4_polr0_time3 = zero(Float16x2)
                        Z_cplx0_dishM0_polr1_time3 = zero(Float16x2)
                        Z_cplx1_dishM0_polr1_time3 = zero(Float16x2)
                        Z_cplx0_dishM4_polr1_time3 = zero(Float16x2)
                        Z_cplx1_dishM4_polr1_time3 = zero(Float16x2)
                        (Z_cplx0_dishM0_polr0_time0, Z_cplx1_dishM0_polr0_time0) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_dishM0_polr0_time0, (Z_cplx0_dishM0_polr0_time0, Z_cplx1_dishM0_polr0_time0)
                        )
                        (Z_cplx0_dishM4_polr0_time0, Z_cplx1_dishM4_polr0_time0) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_dishM4_polr0_time0, (Z_cplx0_dishM4_polr0_time0, Z_cplx1_dishM4_polr0_time0)
                        )
                        (Z_cplx0_dishM0_polr1_time0, Z_cplx1_dishM0_polr1_time0) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_dishM0_polr1_time0, (Z_cplx0_dishM0_polr1_time0, Z_cplx1_dishM0_polr1_time0)
                        )
                        (Z_cplx0_dishM4_polr1_time0, Z_cplx1_dishM4_polr1_time0) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_dishM4_polr1_time0, (Z_cplx0_dishM4_polr1_time0, Z_cplx1_dishM4_polr1_time0)
                        )
                        (Z_cplx0_dishM0_polr0_time3, Z_cplx1_dishM0_polr0_time3) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_dishM0_polr0_time3, (Z_cplx0_dishM0_polr0_time3, Z_cplx1_dishM0_polr0_time3)
                        )
                        (Z_cplx0_dishM4_polr0_time3, Z_cplx1_dishM4_polr0_time3) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_dishM4_polr0_time3, (Z_cplx0_dishM4_polr0_time3, Z_cplx1_dishM4_polr0_time3)
                        )
                        (Z_cplx0_dishM0_polr1_time3, Z_cplx1_dishM0_polr1_time3) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_dishM0_polr1_time3, (Z_cplx0_dishM0_polr1_time3, Z_cplx1_dishM0_polr1_time3)
                        )
                        (Z_cplx0_dishM4_polr1_time3, Z_cplx1_dishM4_polr1_time3) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_dishM4_polr1_time3, (Z_cplx0_dishM4_polr1_time3, Z_cplx1_dishM4_polr1_time3)
                        )
                        aΓ²re = aΓ²_cplx0
                        aΓ²im = aΓ²_cplx1
                        Zre_dishM0_polr0_time0 = Z_cplx0_dishM0_polr0_time0
                        Zim_dishM0_polr0_time0 = Z_cplx1_dishM0_polr0_time0
                        Zre_dishM4_polr0_time0 = Z_cplx0_dishM4_polr0_time0
                        Zim_dishM4_polr0_time0 = Z_cplx1_dishM4_polr0_time0
                        Zre_dishM0_polr1_time0 = Z_cplx0_dishM0_polr1_time0
                        Zim_dishM0_polr1_time0 = Z_cplx1_dishM0_polr1_time0
                        Zre_dishM4_polr1_time0 = Z_cplx0_dishM4_polr1_time0
                        Zim_dishM4_polr1_time0 = Z_cplx1_dishM4_polr1_time0
                        Zre_dishM0_polr0_time3 = Z_cplx0_dishM0_polr0_time3
                        Zim_dishM0_polr0_time3 = Z_cplx1_dishM0_polr0_time3
                        Zre_dishM4_polr0_time3 = Z_cplx0_dishM4_polr0_time3
                        Zim_dishM4_polr0_time3 = Z_cplx1_dishM4_polr0_time3
                        Zre_dishM0_polr1_time3 = Z_cplx0_dishM0_polr1_time3
                        Zim_dishM0_polr1_time3 = Z_cplx1_dishM0_polr1_time3
                        Zre_dishM4_polr1_time3 = Z_cplx0_dishM4_polr1_time3
                        Zim_dishM4_polr1_time3 = Z_cplx1_dishM4_polr1_time3
                        aΓ²reM_dishM0 = aΓ²re
                        aΓ²reM_dishM4 = aΓ²re
                        aΓ²imM_dishM0 = aΓ²im
                        aΓ²imM_dishM4 = aΓ²im
                        Vre_dishM0_polr0_time0 = muladd(
                            aΓ²reM_dishM0, Zre_dishM0_polr0_time0, -aΓ²imM_dishM0 * Zim_dishM0_polr0_time0
                        )
                        Vre_dishM4_polr0_time0 = muladd(
                            aΓ²reM_dishM4, Zre_dishM4_polr0_time0, -aΓ²imM_dishM4 * Zim_dishM4_polr0_time0
                        )
                        Vre_dishM0_polr1_time0 = muladd(
                            aΓ²reM_dishM0, Zre_dishM0_polr1_time0, -aΓ²imM_dishM0 * Zim_dishM0_polr1_time0
                        )
                        Vre_dishM4_polr1_time0 = muladd(
                            aΓ²reM_dishM4, Zre_dishM4_polr1_time0, -aΓ²imM_dishM4 * Zim_dishM4_polr1_time0
                        )
                        Vre_dishM0_polr0_time3 = muladd(
                            aΓ²reM_dishM0, Zre_dishM0_polr0_time3, -aΓ²imM_dishM0 * Zim_dishM0_polr0_time3
                        )
                        Vre_dishM4_polr0_time3 = muladd(
                            aΓ²reM_dishM4, Zre_dishM4_polr0_time3, -aΓ²imM_dishM4 * Zim_dishM4_polr0_time3
                        )
                        Vre_dishM0_polr1_time3 = muladd(
                            aΓ²reM_dishM0, Zre_dishM0_polr1_time3, -aΓ²imM_dishM0 * Zim_dishM0_polr1_time3
                        )
                        Vre_dishM4_polr1_time3 = muladd(
                            aΓ²reM_dishM4, Zre_dishM4_polr1_time3, -aΓ²imM_dishM4 * Zim_dishM4_polr1_time3
                        )
                        Vim_dishM0_polr0_time0 = muladd(
                            aΓ²reM_dishM0, Zim_dishM0_polr0_time0, +aΓ²imM_dishM0 * Zre_dishM0_polr0_time0
                        )
                        Vim_dishM4_polr0_time0 = muladd(
                            aΓ²reM_dishM4, Zim_dishM4_polr0_time0, +aΓ²imM_dishM4 * Zre_dishM4_polr0_time0
                        )
                        Vim_dishM0_polr1_time0 = muladd(
                            aΓ²reM_dishM0, Zim_dishM0_polr1_time0, +aΓ²imM_dishM0 * Zre_dishM0_polr1_time0
                        )
                        Vim_dishM4_polr1_time0 = muladd(
                            aΓ²reM_dishM4, Zim_dishM4_polr1_time0, +aΓ²imM_dishM4 * Zre_dishM4_polr1_time0
                        )
                        Vim_dishM0_polr0_time3 = muladd(
                            aΓ²reM_dishM0, Zim_dishM0_polr0_time3, +aΓ²imM_dishM0 * Zre_dishM0_polr0_time3
                        )
                        Vim_dishM4_polr0_time3 = muladd(
                            aΓ²reM_dishM4, Zim_dishM4_polr0_time3, +aΓ²imM_dishM4 * Zre_dishM4_polr0_time3
                        )
                        Vim_dishM0_polr1_time3 = muladd(
                            aΓ²reM_dishM0, Zim_dishM0_polr1_time3, +aΓ²imM_dishM0 * Zre_dishM0_polr1_time3
                        )
                        Vim_dishM4_polr1_time3 = muladd(
                            aΓ²reM_dishM4, Zim_dishM4_polr1_time3, +aΓ²imM_dishM4 * Zre_dishM4_polr1_time3
                        )
                        V_cplx0_dishM0_polr0_time0 = Vre_dishM0_polr0_time0
                        V_cplx1_dishM0_polr0_time0 = Vim_dishM0_polr0_time0
                        V_cplx0_dishM4_polr0_time0 = Vre_dishM4_polr0_time0
                        V_cplx1_dishM4_polr0_time0 = Vim_dishM4_polr0_time0
                        V_cplx0_dishM0_polr1_time0 = Vre_dishM0_polr1_time0
                        V_cplx1_dishM0_polr1_time0 = Vim_dishM0_polr1_time0
                        V_cplx0_dishM4_polr1_time0 = Vre_dishM4_polr1_time0
                        V_cplx1_dishM4_polr1_time0 = Vim_dishM4_polr1_time0
                        V_cplx0_dishM0_polr0_time3 = Vre_dishM0_polr0_time3
                        V_cplx1_dishM0_polr0_time3 = Vim_dishM0_polr0_time3
                        V_cplx0_dishM4_polr0_time3 = Vre_dishM4_polr0_time3
                        V_cplx1_dishM4_polr0_time3 = Vim_dishM4_polr0_time3
                        V_cplx0_dishM0_polr1_time3 = Vre_dishM0_polr1_time3
                        V_cplx1_dishM0_polr1_time3 = Vim_dishM0_polr1_time3
                        V_cplx0_dishM4_polr1_time3 = Vre_dishM4_polr1_time3
                        V_cplx1_dishM4_polr1_time3 = Vim_dishM4_polr1_time3
                        Y_cplx0_dishM0_polr0_time0 = zero(Float16x2)
                        Y_cplx1_dishM0_polr0_time0 = zero(Float16x2)
                        Y_cplx0_dishM4_polr0_time0 = zero(Float16x2)
                        Y_cplx1_dishM4_polr0_time0 = zero(Float16x2)
                        Y_cplx0_dishM0_polr1_time0 = zero(Float16x2)
                        Y_cplx1_dishM0_polr1_time0 = zero(Float16x2)
                        Y_cplx0_dishM4_polr1_time0 = zero(Float16x2)
                        Y_cplx1_dishM4_polr1_time0 = zero(Float16x2)
                        Y_cplx0_dishM0_polr0_time3 = zero(Float16x2)
                        Y_cplx1_dishM0_polr0_time3 = zero(Float16x2)
                        Y_cplx0_dishM4_polr0_time3 = zero(Float16x2)
                        Y_cplx1_dishM4_polr0_time3 = zero(Float16x2)
                        Y_cplx0_dishM0_polr1_time3 = zero(Float16x2)
                        Y_cplx1_dishM0_polr1_time3 = zero(Float16x2)
                        Y_cplx0_dishM4_polr1_time3 = zero(Float16x2)
                        Y_cplx1_dishM4_polr1_time3 = zero(Float16x2)
                        Vre_dishM0_polr0_time0 = V_cplx0_dishM0_polr0_time0
                        Vim_dishM0_polr0_time0 = V_cplx1_dishM0_polr0_time0
                        Vre_dishM4_polr0_time0 = V_cplx0_dishM4_polr0_time0
                        Vim_dishM4_polr0_time0 = V_cplx1_dishM4_polr0_time0
                        Vre_dishM0_polr1_time0 = V_cplx0_dishM0_polr1_time0
                        Vim_dishM0_polr1_time0 = V_cplx1_dishM0_polr1_time0
                        Vre_dishM4_polr1_time0 = V_cplx0_dishM4_polr1_time0
                        Vim_dishM4_polr1_time0 = V_cplx1_dishM4_polr1_time0
                        Vre_dishM0_polr0_time3 = V_cplx0_dishM0_polr0_time3
                        Vim_dishM0_polr0_time3 = V_cplx1_dishM0_polr0_time3
                        Vre_dishM4_polr0_time3 = V_cplx0_dishM4_polr0_time3
                        Vim_dishM4_polr0_time3 = V_cplx1_dishM4_polr0_time3
                        Vre_dishM0_polr1_time3 = V_cplx0_dishM0_polr1_time3
                        Vim_dishM0_polr1_time3 = V_cplx1_dishM0_polr1_time3
                        Vre_dishM4_polr1_time3 = V_cplx0_dishM4_polr1_time3
                        Vim_dishM4_polr1_time3 = V_cplx1_dishM4_polr1_time3
                        V_cplx_in0_dishM0_polr0_time0 = Vre_dishM0_polr0_time0
                        V_cplx_in1_dishM0_polr0_time0 = Vim_dishM0_polr0_time0
                        V_cplx_in0_dishM4_polr0_time0 = Vre_dishM4_polr0_time0
                        V_cplx_in1_dishM4_polr0_time0 = Vim_dishM4_polr0_time0
                        V_cplx_in0_dishM0_polr1_time0 = Vre_dishM0_polr1_time0
                        V_cplx_in1_dishM0_polr1_time0 = Vim_dishM0_polr1_time0
                        V_cplx_in0_dishM4_polr1_time0 = Vre_dishM4_polr1_time0
                        V_cplx_in1_dishM4_polr1_time0 = Vim_dishM4_polr1_time0
                        V_cplx_in0_dishM0_polr0_time3 = Vre_dishM0_polr0_time3
                        V_cplx_in1_dishM0_polr0_time3 = Vim_dishM0_polr0_time3
                        V_cplx_in0_dishM4_polr0_time3 = Vre_dishM4_polr0_time3
                        V_cplx_in1_dishM4_polr0_time3 = Vim_dishM4_polr0_time3
                        V_cplx_in0_dishM0_polr1_time3 = Vre_dishM0_polr1_time3
                        V_cplx_in1_dishM0_polr1_time3 = Vim_dishM0_polr1_time3
                        V_cplx_in0_dishM4_polr1_time3 = Vre_dishM4_polr1_time3
                        V_cplx_in1_dishM4_polr1_time3 = Vim_dishM4_polr1_time3
                        aΓ³M_cplx0_dishM0 = aΓ³_cplx0
                        aΓ³M_cplx0_dishM4 = aΓ³_cplx0
                        aΓ³M_cplx1_dishM0 = aΓ³_cplx1
                        aΓ³M_cplx1_dishM4 = aΓ³_cplx1
                        (Y_cplx0_dishM0_polr0_time0, Y_cplx1_dishM0_polr0_time0) = let
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
                                (aΓ³M_cplx0_dishM0, aΓ³M_cplx1_dishM0)::NTuple{2,Float16x2},
                                (V_cplx_in0_dishM0_polr0_time0, V_cplx_in1_dishM0_polr0_time0)::NTuple{2,Float16x2},
                                (Y_cplx0_dishM0_polr0_time0, Y_cplx1_dishM0_polr0_time0)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        (Y_cplx0_dishM4_polr0_time0, Y_cplx1_dishM4_polr0_time0) = let
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
                                (aΓ³M_cplx0_dishM4, aΓ³M_cplx1_dishM4)::NTuple{2,Float16x2},
                                (V_cplx_in0_dishM4_polr0_time0, V_cplx_in1_dishM4_polr0_time0)::NTuple{2,Float16x2},
                                (Y_cplx0_dishM4_polr0_time0, Y_cplx1_dishM4_polr0_time0)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        (Y_cplx0_dishM0_polr1_time0, Y_cplx1_dishM0_polr1_time0) = let
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
                                (aΓ³M_cplx0_dishM0, aΓ³M_cplx1_dishM0)::NTuple{2,Float16x2},
                                (V_cplx_in0_dishM0_polr1_time0, V_cplx_in1_dishM0_polr1_time0)::NTuple{2,Float16x2},
                                (Y_cplx0_dishM0_polr1_time0, Y_cplx1_dishM0_polr1_time0)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        (Y_cplx0_dishM4_polr1_time0, Y_cplx1_dishM4_polr1_time0) = let
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
                                (aΓ³M_cplx0_dishM4, aΓ³M_cplx1_dishM4)::NTuple{2,Float16x2},
                                (V_cplx_in0_dishM4_polr1_time0, V_cplx_in1_dishM4_polr1_time0)::NTuple{2,Float16x2},
                                (Y_cplx0_dishM4_polr1_time0, Y_cplx1_dishM4_polr1_time0)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        (Y_cplx0_dishM0_polr0_time3, Y_cplx1_dishM0_polr0_time3) = let
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
                                (aΓ³M_cplx0_dishM0, aΓ³M_cplx1_dishM0)::NTuple{2,Float16x2},
                                (V_cplx_in0_dishM0_polr0_time3, V_cplx_in1_dishM0_polr0_time3)::NTuple{2,Float16x2},
                                (Y_cplx0_dishM0_polr0_time3, Y_cplx1_dishM0_polr0_time3)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        (Y_cplx0_dishM4_polr0_time3, Y_cplx1_dishM4_polr0_time3) = let
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
                                (aΓ³M_cplx0_dishM4, aΓ³M_cplx1_dishM4)::NTuple{2,Float16x2},
                                (V_cplx_in0_dishM4_polr0_time3, V_cplx_in1_dishM4_polr0_time3)::NTuple{2,Float16x2},
                                (Y_cplx0_dishM4_polr0_time3, Y_cplx1_dishM4_polr0_time3)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        (Y_cplx0_dishM0_polr1_time3, Y_cplx1_dishM0_polr1_time3) = let
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
                                (aΓ³M_cplx0_dishM0, aΓ³M_cplx1_dishM0)::NTuple{2,Float16x2},
                                (V_cplx_in0_dishM0_polr1_time3, V_cplx_in1_dishM0_polr1_time3)::NTuple{2,Float16x2},
                                (Y_cplx0_dishM0_polr1_time3, Y_cplx1_dishM0_polr1_time3)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        (Y_cplx0_dishM4_polr1_time3, Y_cplx1_dishM4_polr1_time3) = let
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
                                (aΓ³M_cplx0_dishM4, aΓ³M_cplx1_dishM4)::NTuple{2,Float16x2},
                                (V_cplx_in0_dishM4_polr1_time3, V_cplx_in1_dishM4_polr1_time3)::NTuple{2,Float16x2},
                                (Y_cplx0_dishM4_polr1_time3, Y_cplx1_dishM4_polr1_time3)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        G_cplx0_dishM0_polr0_time0 = Y_cplx0_dishM0_polr0_time0
                        G_cplx1_dishM0_polr0_time0 = Y_cplx1_dishM0_polr0_time0
                        G_cplx0_dishM4_polr0_time0 = Y_cplx0_dishM4_polr0_time0
                        G_cplx1_dishM4_polr0_time0 = Y_cplx1_dishM4_polr0_time0
                        G_cplx0_dishM0_polr1_time0 = Y_cplx0_dishM0_polr1_time0
                        G_cplx1_dishM0_polr1_time0 = Y_cplx1_dishM0_polr1_time0
                        G_cplx0_dishM4_polr1_time0 = Y_cplx0_dishM4_polr1_time0
                        G_cplx1_dishM4_polr1_time0 = Y_cplx1_dishM4_polr1_time0
                        G_cplx0_dishM0_polr0_time3 = Y_cplx0_dishM0_polr0_time3
                        G_cplx1_dishM0_polr0_time3 = Y_cplx1_dishM0_polr0_time3
                        G_cplx0_dishM4_polr0_time3 = Y_cplx0_dishM4_polr0_time3
                        G_cplx1_dishM4_polr0_time3 = Y_cplx1_dishM4_polr0_time3
                        G_cplx0_dishM0_polr1_time3 = Y_cplx0_dishM0_polr1_time3
                        G_cplx1_dishM0_polr1_time3 = Y_cplx1_dishM0_polr1_time3
                        G_cplx0_dishM4_polr1_time3 = Y_cplx0_dishM4_polr1_time3
                        G_cplx1_dishM4_polr1_time3 = Y_cplx1_dishM4_polr1_time3
                        (G_cplx0_dishM0_polr0_time0, G_cplx1_dishM0_polr0_time0) = (
                            IndexSpaces.get_lo16(G_cplx0_dishM0_polr0_time0, G_cplx1_dishM0_polr0_time0),
                            IndexSpaces.get_hi16(G_cplx0_dishM0_polr0_time0, G_cplx1_dishM0_polr0_time0),
                        )
                        (G_cplx0_dishM4_polr0_time0, G_cplx1_dishM4_polr0_time0) = (
                            IndexSpaces.get_lo16(G_cplx0_dishM4_polr0_time0, G_cplx1_dishM4_polr0_time0),
                            IndexSpaces.get_hi16(G_cplx0_dishM4_polr0_time0, G_cplx1_dishM4_polr0_time0),
                        )
                        (G_cplx0_dishM0_polr1_time0, G_cplx1_dishM0_polr1_time0) = (
                            IndexSpaces.get_lo16(G_cplx0_dishM0_polr1_time0, G_cplx1_dishM0_polr1_time0),
                            IndexSpaces.get_hi16(G_cplx0_dishM0_polr1_time0, G_cplx1_dishM0_polr1_time0),
                        )
                        (G_cplx0_dishM4_polr1_time0, G_cplx1_dishM4_polr1_time0) = (
                            IndexSpaces.get_lo16(G_cplx0_dishM4_polr1_time0, G_cplx1_dishM4_polr1_time0),
                            IndexSpaces.get_hi16(G_cplx0_dishM4_polr1_time0, G_cplx1_dishM4_polr1_time0),
                        )
                        (G_cplx0_dishM0_polr0_time3, G_cplx1_dishM0_polr0_time3) = (
                            IndexSpaces.get_lo16(G_cplx0_dishM0_polr0_time3, G_cplx1_dishM0_polr0_time3),
                            IndexSpaces.get_hi16(G_cplx0_dishM0_polr0_time3, G_cplx1_dishM0_polr0_time3),
                        )
                        (G_cplx0_dishM4_polr0_time3, G_cplx1_dishM4_polr0_time3) = (
                            IndexSpaces.get_lo16(G_cplx0_dishM4_polr0_time3, G_cplx1_dishM4_polr0_time3),
                            IndexSpaces.get_hi16(G_cplx0_dishM4_polr0_time3, G_cplx1_dishM4_polr0_time3),
                        )
                        (G_cplx0_dishM0_polr1_time3, G_cplx1_dishM0_polr1_time3) = (
                            IndexSpaces.get_lo16(G_cplx0_dishM0_polr1_time3, G_cplx1_dishM0_polr1_time3),
                            IndexSpaces.get_hi16(G_cplx0_dishM0_polr1_time3, G_cplx1_dishM0_polr1_time3),
                        )
                        (G_cplx0_dishM4_polr1_time3, G_cplx1_dishM4_polr1_time3) = (
                            IndexSpaces.get_lo16(G_cplx0_dishM4_polr1_time3, G_cplx1_dishM4_polr1_time3),
                            IndexSpaces.get_hi16(G_cplx0_dishM4_polr1_time3, G_cplx1_dishM4_polr1_time3),
                        )
                        IndexSpaces.cuda_sync_threads()
                        let t = 0
                            G_polr0 = zero(Float16x2)
                            G_polr1 = zero(Float16x2)
                            if true
                                G_polr0 = Gsh_shared[(((((t::Int32 % 6 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) + ((IndexSpaces.assume_inrange(t_inner_lo::Int32, 0, 6, 24) ÷ 6) % 4) * 6) + ((t_inner_hi::Int32 ÷ 24) % 2) * 24) % 6) * 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 4) % 2) * 392 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 2 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 2) * 1576 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 16) % 2) * 98 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 2 + ((0::Int32 % 2) % 2) * 8 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 2) % 2) * 784 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 8) % 2) * 196) + 0x01]
                                G_polr1 = Gsh_shared[(((((t::Int32 % 6 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) + ((IndexSpaces.assume_inrange(t_inner_lo::Int32, 0, 6, 24) ÷ 6) % 4) * 6) + ((t_inner_hi::Int32 ÷ 24) % 2) * 24) % 6) * 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 4) % 2) * 392 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 2 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 2) * 1576 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 16) % 2) * 98 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 2 + ((1::Int32 % 2) % 2) * 8 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 2) % 2) * 784 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 8) % 2) * 196) + 0x01]
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
                                e2 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e3 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e4 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e5 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
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
                                e2 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e3 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e4 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e5 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
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
                                Float16x2(0.00048828125f0, 0.00048828125f0),
                                muladd(Ẽp1im, Ẽp1im, muladd(Ẽp1re, Ẽp1re, muladd(Ẽp0im, Ẽp0im, Ẽp0re * Ẽp0re))),
                                I,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 2 == 0i32
                                if t_running == 256
                                    if 0i32 ≤
                                       +(
                                           ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0i32, 32) ÷ 1) % 8) * 2
                                       ) <
                                       16 &&
                                        0i32 ≤
                                       ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0i32, 6) ÷ 1) % 6) * 4 +
                                       ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0i32, 32) ÷ 8) % 4) * 1 <
                                       24
                                        I_memory[let
                                            offset = 786432 * Ttildemin + 192 * Fbar_out_min
                                            length = 402653184
                                            mod((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 2) ÷ 2) % 8 + ((IndexSpaces.assume_inrange(dstime::Int32, 0, 1, 512) % 512) % 512) * 786432 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 4096) * 192 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 24) * 8) + 0) + offset, length)
                                        end + 0x01] = I
                                    end
                                    I = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let t = 1
                            G_polr0 = zero(Float16x2)
                            G_polr1 = zero(Float16x2)
                            if true
                                G_polr0 = Gsh_shared[(((((t::Int32 % 6 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) + ((IndexSpaces.assume_inrange(t_inner_lo::Int32, 0, 6, 24) ÷ 6) % 4) * 6) + ((t_inner_hi::Int32 ÷ 24) % 2) * 24) % 6) * 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 4) % 2) * 392 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 2 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 2) * 1576 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 16) % 2) * 98 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 2 + ((0::Int32 % 2) % 2) * 8 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 2) % 2) * 784 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 8) % 2) * 196) + 0x01]
                                G_polr1 = Gsh_shared[(((((t::Int32 % 6 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) + ((IndexSpaces.assume_inrange(t_inner_lo::Int32, 0, 6, 24) ÷ 6) % 4) * 6) + ((t_inner_hi::Int32 ÷ 24) % 2) * 24) % 6) * 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 4) % 2) * 392 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 2 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 2) * 1576 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 16) % 2) * 98 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 2 + ((1::Int32 % 2) % 2) * 8 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 2) % 2) * 784 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 8) % 2) * 196) + 0x01]
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
                                e2 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e3 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e4 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e5 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
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
                                e2 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e3 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e4 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e5 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
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
                                Float16x2(0.00048828125f0, 0.00048828125f0),
                                muladd(Ẽp1im, Ẽp1im, muladd(Ẽp1re, Ẽp1re, muladd(Ẽp0im, Ẽp0im, Ẽp0re * Ẽp0re))),
                                I,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 2 == 0i32
                                if t_running == 256
                                    if 0i32 ≤
                                       +(
                                           ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0i32, 32) ÷ 1) % 8) * 2
                                       ) <
                                       16 &&
                                        0i32 ≤
                                       ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0i32, 6) ÷ 1) % 6) * 4 +
                                       ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0i32, 32) ÷ 8) % 4) * 1 <
                                       24
                                        I_memory[let
                                            offset = 786432 * Ttildemin + 192 * Fbar_out_min
                                            length = 402653184
                                            mod((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 2) ÷ 2) % 8 + ((IndexSpaces.assume_inrange(dstime::Int32, 0, 1, 512) % 512) % 512) * 786432 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 4096) * 192 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 24) * 8) + 0) + offset, length)
                                        end + 0x01] = I
                                    end
                                    I = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let t = 2
                            G_polr0 = zero(Float16x2)
                            G_polr1 = zero(Float16x2)
                            if true
                                G_polr0 = Gsh_shared[(((((t::Int32 % 6 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) + ((IndexSpaces.assume_inrange(t_inner_lo::Int32, 0, 6, 24) ÷ 6) % 4) * 6) + ((t_inner_hi::Int32 ÷ 24) % 2) * 24) % 6) * 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 4) % 2) * 392 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 2 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 2) * 1576 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 16) % 2) * 98 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 2 + ((0::Int32 % 2) % 2) * 8 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 2) % 2) * 784 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 8) % 2) * 196) + 0x01]
                                G_polr1 = Gsh_shared[(((((t::Int32 % 6 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) + ((IndexSpaces.assume_inrange(t_inner_lo::Int32, 0, 6, 24) ÷ 6) % 4) * 6) + ((t_inner_hi::Int32 ÷ 24) % 2) * 24) % 6) * 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 4) % 2) * 392 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 2 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 2) * 1576 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 16) % 2) * 98 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 2 + ((1::Int32 % 2) % 2) * 8 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 2) % 2) * 784 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 8) % 2) * 196) + 0x01]
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
                                e2 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e3 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e4 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e5 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
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
                                e2 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e3 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e4 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e5 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
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
                                Float16x2(0.00048828125f0, 0.00048828125f0),
                                muladd(Ẽp1im, Ẽp1im, muladd(Ẽp1re, Ẽp1re, muladd(Ẽp0im, Ẽp0im, Ẽp0re * Ẽp0re))),
                                I,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 2 == 0i32
                                if t_running == 256
                                    if 0i32 ≤
                                       +(
                                           ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0i32, 32) ÷ 1) % 8) * 2
                                       ) <
                                       16 &&
                                        0i32 ≤
                                       ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0i32, 6) ÷ 1) % 6) * 4 +
                                       ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0i32, 32) ÷ 8) % 4) * 1 <
                                       24
                                        I_memory[let
                                            offset = 786432 * Ttildemin + 192 * Fbar_out_min
                                            length = 402653184
                                            mod((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 2) ÷ 2) % 8 + ((IndexSpaces.assume_inrange(dstime::Int32, 0, 1, 512) % 512) % 512) * 786432 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 4096) * 192 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 24) * 8) + 0) + offset, length)
                                        end + 0x01] = I
                                    end
                                    I = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let t = 3
                            G_polr0 = zero(Float16x2)
                            G_polr1 = zero(Float16x2)
                            if true
                                G_polr0 = Gsh_shared[(((((t::Int32 % 6 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) + ((IndexSpaces.assume_inrange(t_inner_lo::Int32, 0, 6, 24) ÷ 6) % 4) * 6) + ((t_inner_hi::Int32 ÷ 24) % 2) * 24) % 6) * 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 4) % 2) * 392 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 2 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 2) * 1576 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 16) % 2) * 98 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 2 + ((0::Int32 % 2) % 2) * 8 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 2) % 2) * 784 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 8) % 2) * 196) + 0x01]
                                G_polr1 = Gsh_shared[(((((t::Int32 % 6 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) + ((IndexSpaces.assume_inrange(t_inner_lo::Int32, 0, 6, 24) ÷ 6) % 4) * 6) + ((t_inner_hi::Int32 ÷ 24) % 2) * 24) % 6) * 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 4) % 2) * 392 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 2 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 2) * 1576 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 16) % 2) * 98 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 2 + ((1::Int32 % 2) % 2) * 8 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 2) % 2) * 784 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 8) % 2) * 196) + 0x01]
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
                                e2 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e3 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e4 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e5 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
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
                                e2 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e3 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e4 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e5 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
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
                                Float16x2(0.00048828125f0, 0.00048828125f0),
                                muladd(Ẽp1im, Ẽp1im, muladd(Ẽp1re, Ẽp1re, muladd(Ẽp0im, Ẽp0im, Ẽp0re * Ẽp0re))),
                                I,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 2 == 0i32
                                if t_running == 256
                                    if 0i32 ≤
                                       +(
                                           ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0i32, 32) ÷ 1) % 8) * 2
                                       ) <
                                       16 &&
                                        0i32 ≤
                                       ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0i32, 6) ÷ 1) % 6) * 4 +
                                       ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0i32, 32) ÷ 8) % 4) * 1 <
                                       24
                                        I_memory[let
                                            offset = 786432 * Ttildemin + 192 * Fbar_out_min
                                            length = 402653184
                                            mod((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 2) ÷ 2) % 8 + ((IndexSpaces.assume_inrange(dstime::Int32, 0, 1, 512) % 512) % 512) * 786432 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 4096) * 192 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 24) * 8) + 0) + offset, length)
                                        end + 0x01] = I
                                    end
                                    I = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let t = 4
                            G_polr0 = zero(Float16x2)
                            G_polr1 = zero(Float16x2)
                            if true
                                G_polr0 = Gsh_shared[(((((t::Int32 % 6 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) + ((IndexSpaces.assume_inrange(t_inner_lo::Int32, 0, 6, 24) ÷ 6) % 4) * 6) + ((t_inner_hi::Int32 ÷ 24) % 2) * 24) % 6) * 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 4) % 2) * 392 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 2 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 2) * 1576 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 16) % 2) * 98 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 2 + ((0::Int32 % 2) % 2) * 8 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 2) % 2) * 784 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 8) % 2) * 196) + 0x01]
                                G_polr1 = Gsh_shared[(((((t::Int32 % 6 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) + ((IndexSpaces.assume_inrange(t_inner_lo::Int32, 0, 6, 24) ÷ 6) % 4) * 6) + ((t_inner_hi::Int32 ÷ 24) % 2) * 24) % 6) * 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 4) % 2) * 392 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 2 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 2) * 1576 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 16) % 2) * 98 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 2 + ((1::Int32 % 2) % 2) * 8 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 2) % 2) * 784 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 8) % 2) * 196) + 0x01]
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
                                e2 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e3 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e4 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e5 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
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
                                e2 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e3 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e4 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e5 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
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
                                Float16x2(0.00048828125f0, 0.00048828125f0),
                                muladd(Ẽp1im, Ẽp1im, muladd(Ẽp1re, Ẽp1re, muladd(Ẽp0im, Ẽp0im, Ẽp0re * Ẽp0re))),
                                I,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 2 == 0i32
                                if t_running == 256
                                    if 0i32 ≤
                                       +(
                                           ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0i32, 32) ÷ 1) % 8) * 2
                                       ) <
                                       16 &&
                                        0i32 ≤
                                       ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0i32, 6) ÷ 1) % 6) * 4 +
                                       ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0i32, 32) ÷ 8) % 4) * 1 <
                                       24
                                        I_memory[let
                                            offset = 786432 * Ttildemin + 192 * Fbar_out_min
                                            length = 402653184
                                            mod((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 2) ÷ 2) % 8 + ((IndexSpaces.assume_inrange(dstime::Int32, 0, 1, 512) % 512) % 512) * 786432 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 4096) * 192 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 24) * 8) + 0) + offset, length)
                                        end + 0x01] = I
                                    end
                                    I = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let t = 5
                            G_polr0 = zero(Float16x2)
                            G_polr1 = zero(Float16x2)
                            if true
                                G_polr0 = Gsh_shared[(((((t::Int32 % 6 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) + ((IndexSpaces.assume_inrange(t_inner_lo::Int32, 0, 6, 24) ÷ 6) % 4) * 6) + ((t_inner_hi::Int32 ÷ 24) % 2) * 24) % 6) * 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 4) % 2) * 392 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 2 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 2) * 1576 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 16) % 2) * 98 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 2 + ((0::Int32 % 2) % 2) * 8 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 2) % 2) * 784 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 8) % 2) * 196) + 0x01]
                                G_polr1 = Gsh_shared[(((((t::Int32 % 6 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) + ((IndexSpaces.assume_inrange(t_inner_lo::Int32, 0, 6, 24) ÷ 6) % 4) * 6) + ((t_inner_hi::Int32 ÷ 24) % 2) * 24) % 6) * 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 4) % 2) * 392 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 2 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 2) * 1576 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 16) % 2) * 98 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 2 + ((1::Int32 % 2) % 2) * 8 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 2) % 2) * 784 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 8) % 2) * 196) + 0x01]
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
                                e2 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e3 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e4 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e5 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
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
                                e2 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e3 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e4 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e5 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
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
                                Float16x2(0.00048828125f0, 0.00048828125f0),
                                muladd(Ẽp1im, Ẽp1im, muladd(Ẽp1re, Ẽp1re, muladd(Ẽp0im, Ẽp0im, Ẽp0re * Ẽp0re))),
                                I,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 2 == 0i32
                                if t_running == 256
                                    if 0i32 ≤
                                       +(
                                           ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0i32, 32) ÷ 1) % 8) * 2
                                       ) <
                                       16 &&
                                        0i32 ≤
                                       ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0i32, 6) ÷ 1) % 6) * 4 +
                                       ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0i32, 32) ÷ 8) % 4) * 1 <
                                       24
                                        I_memory[let
                                            offset = 786432 * Ttildemin + 192 * Fbar_out_min
                                            length = 402653184
                                            mod((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 2) ÷ 2) % 8 + ((IndexSpaces.assume_inrange(dstime::Int32, 0, 1, 512) % 512) % 512) * 786432 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 4096) * 192 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 24) * 8) + 0) + offset, length)
                                        end + 0x01] = I
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
                let t_inner_hi = 24
                    for t_inner_lo in 0:6:23
                        Freg2′_dishM0_time0 = zero(Int4x8)
                        if t_inner_lo == 0
                            Freg2′_dishM0_time0 = Freg2_dishM0_time0
                        end
                        if t_inner_lo == 6
                            Freg2′_dishM0_time0 = Freg2_dishM0_time6
                        end
                        if t_inner_lo == 12
                            Freg2′_dishM0_time0 = Freg2_dishM0_time12
                        end
                        if t_inner_lo == 18
                            Freg2′_dishM0_time0 = Freg2_dishM0_time18
                        end
                        Freg2′_dishM4_time0 = zero(Int4x8)
                        if t_inner_lo == 0
                            Freg2′_dishM4_time0 = Freg2_dishM4_time0
                        end
                        if t_inner_lo == 6
                            Freg2′_dishM4_time0 = Freg2_dishM4_time6
                        end
                        if t_inner_lo == 12
                            Freg2′_dishM4_time0 = Freg2_dishM4_time12
                        end
                        if t_inner_lo == 18
                            Freg2′_dishM4_time0 = Freg2_dishM4_time18
                        end
                        Freg2′_dishM0_time3 = zero(Int4x8)
                        if t_inner_lo == 0
                            Freg2′_dishM0_time3 = Freg2_dishM0_time3
                        end
                        if t_inner_lo == 6
                            Freg2′_dishM0_time3 = Freg2_dishM0_time9
                        end
                        if t_inner_lo == 12
                            Freg2′_dishM0_time3 = Freg2_dishM0_time15
                        end
                        if t_inner_lo == 18
                            Freg2′_dishM0_time3 = Freg2_dishM0_time21
                        end
                        Freg2′_dishM4_time3 = zero(Int4x8)
                        if t_inner_lo == 0
                            Freg2′_dishM4_time3 = Freg2_dishM4_time3
                        end
                        if t_inner_lo == 6
                            Freg2′_dishM4_time3 = Freg2_dishM4_time9
                        end
                        if t_inner_lo == 12
                            Freg2′_dishM4_time3 = Freg2_dishM4_time15
                        end
                        if t_inner_lo == 18
                            Freg2′_dishM4_time3 = Freg2_dishM4_time21
                        end
                        (E′_dishM0_polr0_time0, E′_dishM0_polr1_time0, E′_dishM0_polr0_time24, E′_dishM0_polr1_time24) = convert(
                            NTuple{4,Float16x2}, Freg2′_dishM0_time0
                        )
                        (E′_dishM4_polr0_time0, E′_dishM4_polr1_time0, E′_dishM4_polr0_time24, E′_dishM4_polr1_time24) = convert(
                            NTuple{4,Float16x2}, Freg2′_dishM4_time0
                        )
                        (E′_dishM0_polr0_time3, E′_dishM0_polr1_time3, E′_dishM0_polr0_time27, E′_dishM0_polr1_time27) = convert(
                            NTuple{4,Float16x2}, Freg2′_dishM0_time3
                        )
                        (E′_dishM4_polr0_time3, E′_dishM4_polr1_time3, E′_dishM4_polr0_time27, E′_dishM4_polr1_time27) = convert(
                            NTuple{4,Float16x2}, Freg2′_dishM4_time3
                        )
                        E_dishM0_polr0_time0 = E′_dishM0_polr0_time24
                        E_dishM4_polr0_time0 = E′_dishM4_polr0_time24
                        E_dishM0_polr1_time0 = E′_dishM0_polr1_time24
                        E_dishM4_polr1_time0 = E′_dishM4_polr1_time24
                        E_dishM0_polr0_time3 = E′_dishM0_polr0_time27
                        E_dishM4_polr0_time3 = E′_dishM4_polr0_time27
                        E_dishM0_polr1_time3 = E′_dishM0_polr1_time27
                        E_dishM4_polr1_time3 = E′_dishM4_polr1_time27
                        WE_dishM0_polr0_time0 = complex_mul(W_dishM0_polr0, E_dishM0_polr0_time0)
                        WE_dishM4_polr0_time0 = complex_mul(W_dishM4_polr0, E_dishM4_polr0_time0)
                        WE_dishM0_polr1_time0 = complex_mul(W_dishM0_polr1, E_dishM0_polr1_time0)
                        WE_dishM4_polr1_time0 = complex_mul(W_dishM4_polr1, E_dishM4_polr1_time0)
                        WE_dishM0_polr0_time3 = complex_mul(W_dishM0_polr0, E_dishM0_polr0_time3)
                        WE_dishM4_polr0_time3 = complex_mul(W_dishM4_polr0, E_dishM4_polr0_time3)
                        WE_dishM0_polr1_time3 = complex_mul(W_dishM0_polr1, E_dishM0_polr1_time3)
                        WE_dishM4_polr1_time3 = complex_mul(W_dishM4_polr1, E_dishM4_polr1_time3)
                        X_dishM0_polr0_time0 = WE_dishM0_polr0_time0
                        X_dishM4_polr0_time0 = WE_dishM4_polr0_time0
                        X_dishM0_polr1_time0 = WE_dishM0_polr1_time0
                        X_dishM4_polr1_time0 = WE_dishM4_polr1_time0
                        X_dishM0_polr0_time3 = WE_dishM0_polr0_time3
                        X_dishM4_polr0_time3 = WE_dishM4_polr0_time3
                        X_dishM0_polr1_time3 = WE_dishM0_polr1_time3
                        X_dishM4_polr1_time3 = WE_dishM4_polr1_time3
                        Z_cplx0_dishM0_polr0_time0 = zero(Float16x2)
                        Z_cplx1_dishM0_polr0_time0 = zero(Float16x2)
                        Z_cplx0_dishM4_polr0_time0 = zero(Float16x2)
                        Z_cplx1_dishM4_polr0_time0 = zero(Float16x2)
                        Z_cplx0_dishM0_polr1_time0 = zero(Float16x2)
                        Z_cplx1_dishM0_polr1_time0 = zero(Float16x2)
                        Z_cplx0_dishM4_polr1_time0 = zero(Float16x2)
                        Z_cplx1_dishM4_polr1_time0 = zero(Float16x2)
                        Z_cplx0_dishM0_polr0_time3 = zero(Float16x2)
                        Z_cplx1_dishM0_polr0_time3 = zero(Float16x2)
                        Z_cplx0_dishM4_polr0_time3 = zero(Float16x2)
                        Z_cplx1_dishM4_polr0_time3 = zero(Float16x2)
                        Z_cplx0_dishM0_polr1_time3 = zero(Float16x2)
                        Z_cplx1_dishM0_polr1_time3 = zero(Float16x2)
                        Z_cplx0_dishM4_polr1_time3 = zero(Float16x2)
                        Z_cplx1_dishM4_polr1_time3 = zero(Float16x2)
                        (Z_cplx0_dishM0_polr0_time0, Z_cplx1_dishM0_polr0_time0) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_dishM0_polr0_time0, (Z_cplx0_dishM0_polr0_time0, Z_cplx1_dishM0_polr0_time0)
                        )
                        (Z_cplx0_dishM4_polr0_time0, Z_cplx1_dishM4_polr0_time0) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_dishM4_polr0_time0, (Z_cplx0_dishM4_polr0_time0, Z_cplx1_dishM4_polr0_time0)
                        )
                        (Z_cplx0_dishM0_polr1_time0, Z_cplx1_dishM0_polr1_time0) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_dishM0_polr1_time0, (Z_cplx0_dishM0_polr1_time0, Z_cplx1_dishM0_polr1_time0)
                        )
                        (Z_cplx0_dishM4_polr1_time0, Z_cplx1_dishM4_polr1_time0) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_dishM4_polr1_time0, (Z_cplx0_dishM4_polr1_time0, Z_cplx1_dishM4_polr1_time0)
                        )
                        (Z_cplx0_dishM0_polr0_time3, Z_cplx1_dishM0_polr0_time3) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_dishM0_polr0_time3, (Z_cplx0_dishM0_polr0_time3, Z_cplx1_dishM0_polr0_time3)
                        )
                        (Z_cplx0_dishM4_polr0_time3, Z_cplx1_dishM4_polr0_time3) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_dishM4_polr0_time3, (Z_cplx0_dishM4_polr0_time3, Z_cplx1_dishM4_polr0_time3)
                        )
                        (Z_cplx0_dishM0_polr1_time3, Z_cplx1_dishM0_polr1_time3) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_dishM0_polr1_time3, (Z_cplx0_dishM0_polr1_time3, Z_cplx1_dishM0_polr1_time3)
                        )
                        (Z_cplx0_dishM4_polr1_time3, Z_cplx1_dishM4_polr1_time3) = IndexSpaces.mma_m16n8k8(
                            (aΓ¹_cplx0, aΓ¹_cplx1), X_dishM4_polr1_time3, (Z_cplx0_dishM4_polr1_time3, Z_cplx1_dishM4_polr1_time3)
                        )
                        aΓ²re = aΓ²_cplx0
                        aΓ²im = aΓ²_cplx1
                        Zre_dishM0_polr0_time0 = Z_cplx0_dishM0_polr0_time0
                        Zim_dishM0_polr0_time0 = Z_cplx1_dishM0_polr0_time0
                        Zre_dishM4_polr0_time0 = Z_cplx0_dishM4_polr0_time0
                        Zim_dishM4_polr0_time0 = Z_cplx1_dishM4_polr0_time0
                        Zre_dishM0_polr1_time0 = Z_cplx0_dishM0_polr1_time0
                        Zim_dishM0_polr1_time0 = Z_cplx1_dishM0_polr1_time0
                        Zre_dishM4_polr1_time0 = Z_cplx0_dishM4_polr1_time0
                        Zim_dishM4_polr1_time0 = Z_cplx1_dishM4_polr1_time0
                        Zre_dishM0_polr0_time3 = Z_cplx0_dishM0_polr0_time3
                        Zim_dishM0_polr0_time3 = Z_cplx1_dishM0_polr0_time3
                        Zre_dishM4_polr0_time3 = Z_cplx0_dishM4_polr0_time3
                        Zim_dishM4_polr0_time3 = Z_cplx1_dishM4_polr0_time3
                        Zre_dishM0_polr1_time3 = Z_cplx0_dishM0_polr1_time3
                        Zim_dishM0_polr1_time3 = Z_cplx1_dishM0_polr1_time3
                        Zre_dishM4_polr1_time3 = Z_cplx0_dishM4_polr1_time3
                        Zim_dishM4_polr1_time3 = Z_cplx1_dishM4_polr1_time3
                        aΓ²reM_dishM0 = aΓ²re
                        aΓ²reM_dishM4 = aΓ²re
                        aΓ²imM_dishM0 = aΓ²im
                        aΓ²imM_dishM4 = aΓ²im
                        Vre_dishM0_polr0_time0 = muladd(
                            aΓ²reM_dishM0, Zre_dishM0_polr0_time0, -aΓ²imM_dishM0 * Zim_dishM0_polr0_time0
                        )
                        Vre_dishM4_polr0_time0 = muladd(
                            aΓ²reM_dishM4, Zre_dishM4_polr0_time0, -aΓ²imM_dishM4 * Zim_dishM4_polr0_time0
                        )
                        Vre_dishM0_polr1_time0 = muladd(
                            aΓ²reM_dishM0, Zre_dishM0_polr1_time0, -aΓ²imM_dishM0 * Zim_dishM0_polr1_time0
                        )
                        Vre_dishM4_polr1_time0 = muladd(
                            aΓ²reM_dishM4, Zre_dishM4_polr1_time0, -aΓ²imM_dishM4 * Zim_dishM4_polr1_time0
                        )
                        Vre_dishM0_polr0_time3 = muladd(
                            aΓ²reM_dishM0, Zre_dishM0_polr0_time3, -aΓ²imM_dishM0 * Zim_dishM0_polr0_time3
                        )
                        Vre_dishM4_polr0_time3 = muladd(
                            aΓ²reM_dishM4, Zre_dishM4_polr0_time3, -aΓ²imM_dishM4 * Zim_dishM4_polr0_time3
                        )
                        Vre_dishM0_polr1_time3 = muladd(
                            aΓ²reM_dishM0, Zre_dishM0_polr1_time3, -aΓ²imM_dishM0 * Zim_dishM0_polr1_time3
                        )
                        Vre_dishM4_polr1_time3 = muladd(
                            aΓ²reM_dishM4, Zre_dishM4_polr1_time3, -aΓ²imM_dishM4 * Zim_dishM4_polr1_time3
                        )
                        Vim_dishM0_polr0_time0 = muladd(
                            aΓ²reM_dishM0, Zim_dishM0_polr0_time0, +aΓ²imM_dishM0 * Zre_dishM0_polr0_time0
                        )
                        Vim_dishM4_polr0_time0 = muladd(
                            aΓ²reM_dishM4, Zim_dishM4_polr0_time0, +aΓ²imM_dishM4 * Zre_dishM4_polr0_time0
                        )
                        Vim_dishM0_polr1_time0 = muladd(
                            aΓ²reM_dishM0, Zim_dishM0_polr1_time0, +aΓ²imM_dishM0 * Zre_dishM0_polr1_time0
                        )
                        Vim_dishM4_polr1_time0 = muladd(
                            aΓ²reM_dishM4, Zim_dishM4_polr1_time0, +aΓ²imM_dishM4 * Zre_dishM4_polr1_time0
                        )
                        Vim_dishM0_polr0_time3 = muladd(
                            aΓ²reM_dishM0, Zim_dishM0_polr0_time3, +aΓ²imM_dishM0 * Zre_dishM0_polr0_time3
                        )
                        Vim_dishM4_polr0_time3 = muladd(
                            aΓ²reM_dishM4, Zim_dishM4_polr0_time3, +aΓ²imM_dishM4 * Zre_dishM4_polr0_time3
                        )
                        Vim_dishM0_polr1_time3 = muladd(
                            aΓ²reM_dishM0, Zim_dishM0_polr1_time3, +aΓ²imM_dishM0 * Zre_dishM0_polr1_time3
                        )
                        Vim_dishM4_polr1_time3 = muladd(
                            aΓ²reM_dishM4, Zim_dishM4_polr1_time3, +aΓ²imM_dishM4 * Zre_dishM4_polr1_time3
                        )
                        V_cplx0_dishM0_polr0_time0 = Vre_dishM0_polr0_time0
                        V_cplx1_dishM0_polr0_time0 = Vim_dishM0_polr0_time0
                        V_cplx0_dishM4_polr0_time0 = Vre_dishM4_polr0_time0
                        V_cplx1_dishM4_polr0_time0 = Vim_dishM4_polr0_time0
                        V_cplx0_dishM0_polr1_time0 = Vre_dishM0_polr1_time0
                        V_cplx1_dishM0_polr1_time0 = Vim_dishM0_polr1_time0
                        V_cplx0_dishM4_polr1_time0 = Vre_dishM4_polr1_time0
                        V_cplx1_dishM4_polr1_time0 = Vim_dishM4_polr1_time0
                        V_cplx0_dishM0_polr0_time3 = Vre_dishM0_polr0_time3
                        V_cplx1_dishM0_polr0_time3 = Vim_dishM0_polr0_time3
                        V_cplx0_dishM4_polr0_time3 = Vre_dishM4_polr0_time3
                        V_cplx1_dishM4_polr0_time3 = Vim_dishM4_polr0_time3
                        V_cplx0_dishM0_polr1_time3 = Vre_dishM0_polr1_time3
                        V_cplx1_dishM0_polr1_time3 = Vim_dishM0_polr1_time3
                        V_cplx0_dishM4_polr1_time3 = Vre_dishM4_polr1_time3
                        V_cplx1_dishM4_polr1_time3 = Vim_dishM4_polr1_time3
                        Y_cplx0_dishM0_polr0_time0 = zero(Float16x2)
                        Y_cplx1_dishM0_polr0_time0 = zero(Float16x2)
                        Y_cplx0_dishM4_polr0_time0 = zero(Float16x2)
                        Y_cplx1_dishM4_polr0_time0 = zero(Float16x2)
                        Y_cplx0_dishM0_polr1_time0 = zero(Float16x2)
                        Y_cplx1_dishM0_polr1_time0 = zero(Float16x2)
                        Y_cplx0_dishM4_polr1_time0 = zero(Float16x2)
                        Y_cplx1_dishM4_polr1_time0 = zero(Float16x2)
                        Y_cplx0_dishM0_polr0_time3 = zero(Float16x2)
                        Y_cplx1_dishM0_polr0_time3 = zero(Float16x2)
                        Y_cplx0_dishM4_polr0_time3 = zero(Float16x2)
                        Y_cplx1_dishM4_polr0_time3 = zero(Float16x2)
                        Y_cplx0_dishM0_polr1_time3 = zero(Float16x2)
                        Y_cplx1_dishM0_polr1_time3 = zero(Float16x2)
                        Y_cplx0_dishM4_polr1_time3 = zero(Float16x2)
                        Y_cplx1_dishM4_polr1_time3 = zero(Float16x2)
                        Vre_dishM0_polr0_time0 = V_cplx0_dishM0_polr0_time0
                        Vim_dishM0_polr0_time0 = V_cplx1_dishM0_polr0_time0
                        Vre_dishM4_polr0_time0 = V_cplx0_dishM4_polr0_time0
                        Vim_dishM4_polr0_time0 = V_cplx1_dishM4_polr0_time0
                        Vre_dishM0_polr1_time0 = V_cplx0_dishM0_polr1_time0
                        Vim_dishM0_polr1_time0 = V_cplx1_dishM0_polr1_time0
                        Vre_dishM4_polr1_time0 = V_cplx0_dishM4_polr1_time0
                        Vim_dishM4_polr1_time0 = V_cplx1_dishM4_polr1_time0
                        Vre_dishM0_polr0_time3 = V_cplx0_dishM0_polr0_time3
                        Vim_dishM0_polr0_time3 = V_cplx1_dishM0_polr0_time3
                        Vre_dishM4_polr0_time3 = V_cplx0_dishM4_polr0_time3
                        Vim_dishM4_polr0_time3 = V_cplx1_dishM4_polr0_time3
                        Vre_dishM0_polr1_time3 = V_cplx0_dishM0_polr1_time3
                        Vim_dishM0_polr1_time3 = V_cplx1_dishM0_polr1_time3
                        Vre_dishM4_polr1_time3 = V_cplx0_dishM4_polr1_time3
                        Vim_dishM4_polr1_time3 = V_cplx1_dishM4_polr1_time3
                        V_cplx_in0_dishM0_polr0_time0 = Vre_dishM0_polr0_time0
                        V_cplx_in1_dishM0_polr0_time0 = Vim_dishM0_polr0_time0
                        V_cplx_in0_dishM4_polr0_time0 = Vre_dishM4_polr0_time0
                        V_cplx_in1_dishM4_polr0_time0 = Vim_dishM4_polr0_time0
                        V_cplx_in0_dishM0_polr1_time0 = Vre_dishM0_polr1_time0
                        V_cplx_in1_dishM0_polr1_time0 = Vim_dishM0_polr1_time0
                        V_cplx_in0_dishM4_polr1_time0 = Vre_dishM4_polr1_time0
                        V_cplx_in1_dishM4_polr1_time0 = Vim_dishM4_polr1_time0
                        V_cplx_in0_dishM0_polr0_time3 = Vre_dishM0_polr0_time3
                        V_cplx_in1_dishM0_polr0_time3 = Vim_dishM0_polr0_time3
                        V_cplx_in0_dishM4_polr0_time3 = Vre_dishM4_polr0_time3
                        V_cplx_in1_dishM4_polr0_time3 = Vim_dishM4_polr0_time3
                        V_cplx_in0_dishM0_polr1_time3 = Vre_dishM0_polr1_time3
                        V_cplx_in1_dishM0_polr1_time3 = Vim_dishM0_polr1_time3
                        V_cplx_in0_dishM4_polr1_time3 = Vre_dishM4_polr1_time3
                        V_cplx_in1_dishM4_polr1_time3 = Vim_dishM4_polr1_time3
                        aΓ³M_cplx0_dishM0 = aΓ³_cplx0
                        aΓ³M_cplx0_dishM4 = aΓ³_cplx0
                        aΓ³M_cplx1_dishM0 = aΓ³_cplx1
                        aΓ³M_cplx1_dishM4 = aΓ³_cplx1
                        (Y_cplx0_dishM0_polr0_time0, Y_cplx1_dishM0_polr0_time0) = let
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
                                (aΓ³M_cplx0_dishM0, aΓ³M_cplx1_dishM0)::NTuple{2,Float16x2},
                                (V_cplx_in0_dishM0_polr0_time0, V_cplx_in1_dishM0_polr0_time0)::NTuple{2,Float16x2},
                                (Y_cplx0_dishM0_polr0_time0, Y_cplx1_dishM0_polr0_time0)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        (Y_cplx0_dishM4_polr0_time0, Y_cplx1_dishM4_polr0_time0) = let
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
                                (aΓ³M_cplx0_dishM4, aΓ³M_cplx1_dishM4)::NTuple{2,Float16x2},
                                (V_cplx_in0_dishM4_polr0_time0, V_cplx_in1_dishM4_polr0_time0)::NTuple{2,Float16x2},
                                (Y_cplx0_dishM4_polr0_time0, Y_cplx1_dishM4_polr0_time0)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        (Y_cplx0_dishM0_polr1_time0, Y_cplx1_dishM0_polr1_time0) = let
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
                                (aΓ³M_cplx0_dishM0, aΓ³M_cplx1_dishM0)::NTuple{2,Float16x2},
                                (V_cplx_in0_dishM0_polr1_time0, V_cplx_in1_dishM0_polr1_time0)::NTuple{2,Float16x2},
                                (Y_cplx0_dishM0_polr1_time0, Y_cplx1_dishM0_polr1_time0)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        (Y_cplx0_dishM4_polr1_time0, Y_cplx1_dishM4_polr1_time0) = let
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
                                (aΓ³M_cplx0_dishM4, aΓ³M_cplx1_dishM4)::NTuple{2,Float16x2},
                                (V_cplx_in0_dishM4_polr1_time0, V_cplx_in1_dishM4_polr1_time0)::NTuple{2,Float16x2},
                                (Y_cplx0_dishM4_polr1_time0, Y_cplx1_dishM4_polr1_time0)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        (Y_cplx0_dishM0_polr0_time3, Y_cplx1_dishM0_polr0_time3) = let
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
                                (aΓ³M_cplx0_dishM0, aΓ³M_cplx1_dishM0)::NTuple{2,Float16x2},
                                (V_cplx_in0_dishM0_polr0_time3, V_cplx_in1_dishM0_polr0_time3)::NTuple{2,Float16x2},
                                (Y_cplx0_dishM0_polr0_time3, Y_cplx1_dishM0_polr0_time3)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        (Y_cplx0_dishM4_polr0_time3, Y_cplx1_dishM4_polr0_time3) = let
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
                                (aΓ³M_cplx0_dishM4, aΓ³M_cplx1_dishM4)::NTuple{2,Float16x2},
                                (V_cplx_in0_dishM4_polr0_time3, V_cplx_in1_dishM4_polr0_time3)::NTuple{2,Float16x2},
                                (Y_cplx0_dishM4_polr0_time3, Y_cplx1_dishM4_polr0_time3)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        (Y_cplx0_dishM0_polr1_time3, Y_cplx1_dishM0_polr1_time3) = let
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
                                (aΓ³M_cplx0_dishM0, aΓ³M_cplx1_dishM0)::NTuple{2,Float16x2},
                                (V_cplx_in0_dishM0_polr1_time3, V_cplx_in1_dishM0_polr1_time3)::NTuple{2,Float16x2},
                                (Y_cplx0_dishM0_polr1_time3, Y_cplx1_dishM0_polr1_time3)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        (Y_cplx0_dishM4_polr1_time3, Y_cplx1_dishM4_polr1_time3) = let
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
                                (aΓ³M_cplx0_dishM4, aΓ³M_cplx1_dishM4)::NTuple{2,Float16x2},
                                (V_cplx_in0_dishM4_polr1_time3, V_cplx_in1_dishM4_polr1_time3)::NTuple{2,Float16x2},
                                (Y_cplx0_dishM4_polr1_time3, Y_cplx1_dishM4_polr1_time3)::NTuple{2,Float16x2},
                                e::Int2x16,
                                0i32,
                            )::NTuple{2,Float16x2}
                        end
                        G_cplx0_dishM0_polr0_time0 = Y_cplx0_dishM0_polr0_time0
                        G_cplx1_dishM0_polr0_time0 = Y_cplx1_dishM0_polr0_time0
                        G_cplx0_dishM4_polr0_time0 = Y_cplx0_dishM4_polr0_time0
                        G_cplx1_dishM4_polr0_time0 = Y_cplx1_dishM4_polr0_time0
                        G_cplx0_dishM0_polr1_time0 = Y_cplx0_dishM0_polr1_time0
                        G_cplx1_dishM0_polr1_time0 = Y_cplx1_dishM0_polr1_time0
                        G_cplx0_dishM4_polr1_time0 = Y_cplx0_dishM4_polr1_time0
                        G_cplx1_dishM4_polr1_time0 = Y_cplx1_dishM4_polr1_time0
                        G_cplx0_dishM0_polr0_time3 = Y_cplx0_dishM0_polr0_time3
                        G_cplx1_dishM0_polr0_time3 = Y_cplx1_dishM0_polr0_time3
                        G_cplx0_dishM4_polr0_time3 = Y_cplx0_dishM4_polr0_time3
                        G_cplx1_dishM4_polr0_time3 = Y_cplx1_dishM4_polr0_time3
                        G_cplx0_dishM0_polr1_time3 = Y_cplx0_dishM0_polr1_time3
                        G_cplx1_dishM0_polr1_time3 = Y_cplx1_dishM0_polr1_time3
                        G_cplx0_dishM4_polr1_time3 = Y_cplx0_dishM4_polr1_time3
                        G_cplx1_dishM4_polr1_time3 = Y_cplx1_dishM4_polr1_time3
                        (G_cplx0_dishM0_polr0_time0, G_cplx1_dishM0_polr0_time0) = (
                            IndexSpaces.get_lo16(G_cplx0_dishM0_polr0_time0, G_cplx1_dishM0_polr0_time0),
                            IndexSpaces.get_hi16(G_cplx0_dishM0_polr0_time0, G_cplx1_dishM0_polr0_time0),
                        )
                        (G_cplx0_dishM4_polr0_time0, G_cplx1_dishM4_polr0_time0) = (
                            IndexSpaces.get_lo16(G_cplx0_dishM4_polr0_time0, G_cplx1_dishM4_polr0_time0),
                            IndexSpaces.get_hi16(G_cplx0_dishM4_polr0_time0, G_cplx1_dishM4_polr0_time0),
                        )
                        (G_cplx0_dishM0_polr1_time0, G_cplx1_dishM0_polr1_time0) = (
                            IndexSpaces.get_lo16(G_cplx0_dishM0_polr1_time0, G_cplx1_dishM0_polr1_time0),
                            IndexSpaces.get_hi16(G_cplx0_dishM0_polr1_time0, G_cplx1_dishM0_polr1_time0),
                        )
                        (G_cplx0_dishM4_polr1_time0, G_cplx1_dishM4_polr1_time0) = (
                            IndexSpaces.get_lo16(G_cplx0_dishM4_polr1_time0, G_cplx1_dishM4_polr1_time0),
                            IndexSpaces.get_hi16(G_cplx0_dishM4_polr1_time0, G_cplx1_dishM4_polr1_time0),
                        )
                        (G_cplx0_dishM0_polr0_time3, G_cplx1_dishM0_polr0_time3) = (
                            IndexSpaces.get_lo16(G_cplx0_dishM0_polr0_time3, G_cplx1_dishM0_polr0_time3),
                            IndexSpaces.get_hi16(G_cplx0_dishM0_polr0_time3, G_cplx1_dishM0_polr0_time3),
                        )
                        (G_cplx0_dishM4_polr0_time3, G_cplx1_dishM4_polr0_time3) = (
                            IndexSpaces.get_lo16(G_cplx0_dishM4_polr0_time3, G_cplx1_dishM4_polr0_time3),
                            IndexSpaces.get_hi16(G_cplx0_dishM4_polr0_time3, G_cplx1_dishM4_polr0_time3),
                        )
                        (G_cplx0_dishM0_polr1_time3, G_cplx1_dishM0_polr1_time3) = (
                            IndexSpaces.get_lo16(G_cplx0_dishM0_polr1_time3, G_cplx1_dishM0_polr1_time3),
                            IndexSpaces.get_hi16(G_cplx0_dishM0_polr1_time3, G_cplx1_dishM0_polr1_time3),
                        )
                        (G_cplx0_dishM4_polr1_time3, G_cplx1_dishM4_polr1_time3) = (
                            IndexSpaces.get_lo16(G_cplx0_dishM4_polr1_time3, G_cplx1_dishM4_polr1_time3),
                            IndexSpaces.get_hi16(G_cplx0_dishM4_polr1_time3, G_cplx1_dishM4_polr1_time3),
                        )
                        IndexSpaces.cuda_sync_threads()
                        let t = 0
                            G_polr0 = zero(Float16x2)
                            G_polr1 = zero(Float16x2)
                            if true
                                G_polr0 = Gsh_shared[(((((t::Int32 % 6 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) + ((IndexSpaces.assume_inrange(t_inner_lo::Int32, 0, 6, 24) ÷ 6) % 4) * 6) + ((t_inner_hi::Int32 ÷ 24) % 2) * 24) % 6) * 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 4) % 2) * 392 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 2 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 2) * 1576 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 16) % 2) * 98 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 2 + ((0::Int32 % 2) % 2) * 8 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 2) % 2) * 784 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 8) % 2) * 196) + 0x01]
                                G_polr1 = Gsh_shared[(((((t::Int32 % 6 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) + ((IndexSpaces.assume_inrange(t_inner_lo::Int32, 0, 6, 24) ÷ 6) % 4) * 6) + ((t_inner_hi::Int32 ÷ 24) % 2) * 24) % 6) * 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 4) % 2) * 392 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 2 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 2) * 1576 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 16) % 2) * 98 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 2 + ((1::Int32 % 2) % 2) * 8 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 2) % 2) * 784 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 8) % 2) * 196) + 0x01]
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
                                e2 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e3 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e4 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e5 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
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
                                e2 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e3 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e4 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e5 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
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
                                Float16x2(0.00048828125f0, 0.00048828125f0),
                                muladd(Ẽp1im, Ẽp1im, muladd(Ẽp1re, Ẽp1re, muladd(Ẽp0im, Ẽp0im, Ẽp0re * Ẽp0re))),
                                I,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 2 == 0i32
                                if t_running == 256
                                    if 0i32 ≤
                                       +(
                                           ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0i32, 32) ÷ 1) % 8) * 2
                                       ) <
                                       16 &&
                                        0i32 ≤
                                       ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0i32, 6) ÷ 1) % 6) * 4 +
                                       ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0i32, 32) ÷ 8) % 4) * 1 <
                                       24
                                        I_memory[let
                                            offset = 786432 * Ttildemin + 192 * Fbar_out_min
                                            length = 402653184
                                            mod((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 2) ÷ 2) % 8 + ((IndexSpaces.assume_inrange(dstime::Int32, 0, 1, 512) % 512) % 512) * 786432 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 4096) * 192 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 24) * 8) + 0) + offset, length)
                                        end + 0x01] = I
                                    end
                                    I = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let t = 1
                            G_polr0 = zero(Float16x2)
                            G_polr1 = zero(Float16x2)
                            if true
                                G_polr0 = Gsh_shared[(((((t::Int32 % 6 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) + ((IndexSpaces.assume_inrange(t_inner_lo::Int32, 0, 6, 24) ÷ 6) % 4) * 6) + ((t_inner_hi::Int32 ÷ 24) % 2) * 24) % 6) * 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 4) % 2) * 392 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 2 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 2) * 1576 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 16) % 2) * 98 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 2 + ((0::Int32 % 2) % 2) * 8 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 2) % 2) * 784 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 8) % 2) * 196) + 0x01]
                                G_polr1 = Gsh_shared[(((((t::Int32 % 6 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) + ((IndexSpaces.assume_inrange(t_inner_lo::Int32, 0, 6, 24) ÷ 6) % 4) * 6) + ((t_inner_hi::Int32 ÷ 24) % 2) * 24) % 6) * 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 4) % 2) * 392 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 2 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 2) * 1576 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 16) % 2) * 98 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 2 + ((1::Int32 % 2) % 2) * 8 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 2) % 2) * 784 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 8) % 2) * 196) + 0x01]
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
                                e2 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e3 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e4 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e5 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
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
                                e2 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e3 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e4 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e5 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
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
                                Float16x2(0.00048828125f0, 0.00048828125f0),
                                muladd(Ẽp1im, Ẽp1im, muladd(Ẽp1re, Ẽp1re, muladd(Ẽp0im, Ẽp0im, Ẽp0re * Ẽp0re))),
                                I,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 2 == 0i32
                                if t_running == 256
                                    if 0i32 ≤
                                       +(
                                           ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0i32, 32) ÷ 1) % 8) * 2
                                       ) <
                                       16 &&
                                        0i32 ≤
                                       ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0i32, 6) ÷ 1) % 6) * 4 +
                                       ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0i32, 32) ÷ 8) % 4) * 1 <
                                       24
                                        I_memory[let
                                            offset = 786432 * Ttildemin + 192 * Fbar_out_min
                                            length = 402653184
                                            mod((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 2) ÷ 2) % 8 + ((IndexSpaces.assume_inrange(dstime::Int32, 0, 1, 512) % 512) % 512) * 786432 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 4096) * 192 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 24) * 8) + 0) + offset, length)
                                        end + 0x01] = I
                                    end
                                    I = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let t = 2
                            G_polr0 = zero(Float16x2)
                            G_polr1 = zero(Float16x2)
                            if true
                                G_polr0 = Gsh_shared[(((((t::Int32 % 6 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) + ((IndexSpaces.assume_inrange(t_inner_lo::Int32, 0, 6, 24) ÷ 6) % 4) * 6) + ((t_inner_hi::Int32 ÷ 24) % 2) * 24) % 6) * 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 4) % 2) * 392 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 2 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 2) * 1576 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 16) % 2) * 98 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 2 + ((0::Int32 % 2) % 2) * 8 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 2) % 2) * 784 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 8) % 2) * 196) + 0x01]
                                G_polr1 = Gsh_shared[(((((t::Int32 % 6 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) + ((IndexSpaces.assume_inrange(t_inner_lo::Int32, 0, 6, 24) ÷ 6) % 4) * 6) + ((t_inner_hi::Int32 ÷ 24) % 2) * 24) % 6) * 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 4) % 2) * 392 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 2 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 2) * 1576 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 16) % 2) * 98 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 2 + ((1::Int32 % 2) % 2) * 8 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 2) % 2) * 784 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 8) % 2) * 196) + 0x01]
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
                                e2 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e3 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e4 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e5 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
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
                                e2 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e3 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e4 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e5 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
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
                                Float16x2(0.00048828125f0, 0.00048828125f0),
                                muladd(Ẽp1im, Ẽp1im, muladd(Ẽp1re, Ẽp1re, muladd(Ẽp0im, Ẽp0im, Ẽp0re * Ẽp0re))),
                                I,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 2 == 0i32
                                if t_running == 256
                                    if 0i32 ≤
                                       +(
                                           ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0i32, 32) ÷ 1) % 8) * 2
                                       ) <
                                       16 &&
                                        0i32 ≤
                                       ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0i32, 6) ÷ 1) % 6) * 4 +
                                       ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0i32, 32) ÷ 8) % 4) * 1 <
                                       24
                                        I_memory[let
                                            offset = 786432 * Ttildemin + 192 * Fbar_out_min
                                            length = 402653184
                                            mod((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 2) ÷ 2) % 8 + ((IndexSpaces.assume_inrange(dstime::Int32, 0, 1, 512) % 512) % 512) * 786432 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 4096) * 192 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 24) * 8) + 0) + offset, length)
                                        end + 0x01] = I
                                    end
                                    I = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let t = 3
                            G_polr0 = zero(Float16x2)
                            G_polr1 = zero(Float16x2)
                            if true
                                G_polr0 = Gsh_shared[(((((t::Int32 % 6 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) + ((IndexSpaces.assume_inrange(t_inner_lo::Int32, 0, 6, 24) ÷ 6) % 4) * 6) + ((t_inner_hi::Int32 ÷ 24) % 2) * 24) % 6) * 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 4) % 2) * 392 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 2 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 2) * 1576 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 16) % 2) * 98 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 2 + ((0::Int32 % 2) % 2) * 8 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 2) % 2) * 784 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 8) % 2) * 196) + 0x01]
                                G_polr1 = Gsh_shared[(((((t::Int32 % 6 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) + ((IndexSpaces.assume_inrange(t_inner_lo::Int32, 0, 6, 24) ÷ 6) % 4) * 6) + ((t_inner_hi::Int32 ÷ 24) % 2) * 24) % 6) * 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 4) % 2) * 392 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 2 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 2) * 1576 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 16) % 2) * 98 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 2 + ((1::Int32 % 2) % 2) * 8 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 2) % 2) * 784 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 8) % 2) * 196) + 0x01]
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
                                e2 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e3 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e4 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e5 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
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
                                e2 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e3 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e4 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e5 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
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
                                Float16x2(0.00048828125f0, 0.00048828125f0),
                                muladd(Ẽp1im, Ẽp1im, muladd(Ẽp1re, Ẽp1re, muladd(Ẽp0im, Ẽp0im, Ẽp0re * Ẽp0re))),
                                I,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 2 == 0i32
                                if t_running == 256
                                    if 0i32 ≤
                                       +(
                                           ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0i32, 32) ÷ 1) % 8) * 2
                                       ) <
                                       16 &&
                                        0i32 ≤
                                       ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0i32, 6) ÷ 1) % 6) * 4 +
                                       ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0i32, 32) ÷ 8) % 4) * 1 <
                                       24
                                        I_memory[let
                                            offset = 786432 * Ttildemin + 192 * Fbar_out_min
                                            length = 402653184
                                            mod((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 2) ÷ 2) % 8 + ((IndexSpaces.assume_inrange(dstime::Int32, 0, 1, 512) % 512) % 512) * 786432 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 4096) * 192 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 24) * 8) + 0) + offset, length)
                                        end + 0x01] = I
                                    end
                                    I = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let t = 4
                            G_polr0 = zero(Float16x2)
                            G_polr1 = zero(Float16x2)
                            if true
                                G_polr0 = Gsh_shared[(((((t::Int32 % 6 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) + ((IndexSpaces.assume_inrange(t_inner_lo::Int32, 0, 6, 24) ÷ 6) % 4) * 6) + ((t_inner_hi::Int32 ÷ 24) % 2) * 24) % 6) * 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 4) % 2) * 392 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 2 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 2) * 1576 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 16) % 2) * 98 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 2 + ((0::Int32 % 2) % 2) * 8 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 2) % 2) * 784 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 8) % 2) * 196) + 0x01]
                                G_polr1 = Gsh_shared[(((((t::Int32 % 6 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) + ((IndexSpaces.assume_inrange(t_inner_lo::Int32, 0, 6, 24) ÷ 6) % 4) * 6) + ((t_inner_hi::Int32 ÷ 24) % 2) * 24) % 6) * 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 4) % 2) * 392 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 2 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 2) * 1576 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 16) % 2) * 98 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 2 + ((1::Int32 % 2) % 2) * 8 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 2) % 2) * 784 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 8) % 2) * 196) + 0x01]
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
                                e2 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e3 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e4 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e5 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
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
                                e2 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e3 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e4 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e5 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
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
                                Float16x2(0.00048828125f0, 0.00048828125f0),
                                muladd(Ẽp1im, Ẽp1im, muladd(Ẽp1re, Ẽp1re, muladd(Ẽp0im, Ẽp0im, Ẽp0re * Ẽp0re))),
                                I,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 2 == 0i32
                                if t_running == 256
                                    if 0i32 ≤
                                       +(
                                           ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0i32, 32) ÷ 1) % 8) * 2
                                       ) <
                                       16 &&
                                        0i32 ≤
                                       ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0i32, 6) ÷ 1) % 6) * 4 +
                                       ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0i32, 32) ÷ 8) % 4) * 1 <
                                       24
                                        I_memory[let
                                            offset = 786432 * Ttildemin + 192 * Fbar_out_min
                                            length = 402653184
                                            mod((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 2) ÷ 2) % 8 + ((IndexSpaces.assume_inrange(dstime::Int32, 0, 1, 512) % 512) % 512) * 786432 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 4096) * 192 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 24) * 8) + 0) + offset, length)
                                        end + 0x01] = I
                                    end
                                    I = zero(Float16x2)
                                    t_running = 0
                                    dstime += 1
                                end
                            end
                        end
                        let t = 5
                            G_polr0 = zero(Float16x2)
                            G_polr1 = zero(Float16x2)
                            if true
                                G_polr0 = Gsh_shared[(((((t::Int32 % 6 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) + ((IndexSpaces.assume_inrange(t_inner_lo::Int32, 0, 6, 24) ÷ 6) % 4) * 6) + ((t_inner_hi::Int32 ÷ 24) % 2) * 24) % 6) * 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 4) % 2) * 392 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 2 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 2) * 1576 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 16) % 2) * 98 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 2 + ((0::Int32 % 2) % 2) * 8 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 2) % 2) * 784 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 8) % 2) * 196) + 0x01]
                                G_polr1 = Gsh_shared[(((((t::Int32 % 6 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 48, 32736) ÷ 48) % 682) * 48) + ((IndexSpaces.assume_inrange(t_inner_lo::Int32, 0, 6, 24) ÷ 6) % 4) * 6) + ((t_inner_hi::Int32 ÷ 24) % 2) * 24) % 6) * 16 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 4) % 2) * 392 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) % 2 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 2) * 1576 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 16) % 2) * 98 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) % 4) * 2 + ((1::Int32 % 2) % 2) * 8 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 2) % 2) * 784 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) ÷ 8) % 2) * 196) + 0x01]
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
                                e2 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e3 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e4 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e5 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
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
                                e2 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e3 = Int2x16(-2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1)
                                e4 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
                                e5 = Int2x16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
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
                                Float16x2(0.00048828125f0, 0.00048828125f0),
                                muladd(Ẽp1im, Ẽp1im, muladd(Ẽp1re, Ẽp1re, muladd(Ẽp0im, Ẽp0im, Ẽp0re * Ẽp0re))),
                                I,
                            )
                            t_running += 1
                            if (t_inner_hi + t + 1i32) % 2 == 0i32
                                if t_running == 256
                                    if 0i32 ≤
                                       +(
                                           ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0i32, 32) ÷ 1) % 8) * 2
                                       ) <
                                       16 &&
                                        0i32 ≤
                                       ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0i32, 6) ÷ 1) % 6) * 4 +
                                       ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0i32, 32) ÷ 8) % 4) * 1 <
                                       24
                                        I_memory[let
                                            offset = 786432 * Ttildemin + 192 * Fbar_out_min
                                            length = 402653184
                                            mod((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 2) ÷ 2) % 8 + ((IndexSpaces.assume_inrange(dstime::Int32, 0, 1, 512) % 512) % 512) * 786432 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 4096) * 192 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) * 4 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) % 24) * 8) + 0) + offset, length)
                                        end + 0x01] = I
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
            info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 384) % 384) % 384) * 192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 6) % 6) % 6) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32) + 0) + 0x01] =
                info
        end
    end
)
