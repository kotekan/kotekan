# Julia source code for the CUDA upchannelizer
# This file has been generated automatically by `upchan.jl`.
# Do not modify this file, your changes will be lost.

@fastmath @inbounds(
    begin #= /localhome/eschnett/src/kotekan/julia/kernels/upchan.jl:1469 =#
        info = 1
        if true
            info_memory[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) % 384) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 8) % 8) * 32) + 0) + 0x01] =
                info
        end
        if !(
            0i32 ≤ Tmin < 32768 && (
                Tmin ≤ Tmax < 65536 && (
                    (Tmax - Tmin) % 256 == 0i32 &&
                    (0i32 ≤ T̄min < 4096 && (T̄min ≤ T̄max < 8192 && ((T̄max - T̄min) + 3) % 32 == 0i32))
                )
            )
        )
            info = 2
            if true
                info_memory[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) % 384) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 8) % 8) * 32) + 0) + 0x01] =
                    info
            end
            IndexSpaces.cuda_trap()
        end
        if !(0i32 ≤ Fmin ≤ Fmax ≤ F)
            info = 3
            if true
                info_memory[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) % 384) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 8) % 8) * 32) + 0) + 0x01] =
                    info
            end
            IndexSpaces.cuda_trap()
        end
        F_ringbuf_mtaps0 = zero(Int4x8)
        F_ringbuf_mtaps1 = zero(Int4x8)
        F_ringbuf_mtaps2 = zero(Int4x8)
        Gains = G_memory[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) ÷ 2) % 256 + 0x01]
        (Wpfb0_m0, Wpfb1_m0) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            time0 = 0 + thread2time(thread)
            time1 = time0 + 4
            s0 = time0 + 0
            s1 = time1 + 0
            W0 = 0.06636127f0 * Wkernel(s0, 4, 8)
            W1 = 0.06636127f0 * Wkernel(s1, 4, 8)
            (W0, W1)
        end
        Wpfb_m0_t0 = Float16x2(Wpfb0_m0, Wpfb1_m0)
        (Wpfb0_m1, Wpfb1_m1) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            time0 = 0 + thread2time(thread)
            time1 = time0 + 4
            s0 = time0 + 8
            s1 = time1 + 8
            W0 = 0.06636127f0 * Wkernel(s0, 4, 8)
            W1 = 0.06636127f0 * Wkernel(s1, 4, 8)
            (W0, W1)
        end
        Wpfb_m1_t0 = Float16x2(Wpfb0_m1, Wpfb1_m1)
        (Wpfb0_m2, Wpfb1_m2) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            time0 = 0 + thread2time(thread)
            time1 = time0 + 4
            s0 = time0 + 16
            s1 = time1 + 16
            W0 = 0.06636127f0 * Wkernel(s0, 4, 8)
            W1 = 0.06636127f0 * Wkernel(s1, 4, 8)
            (W0, W1)
        end
        Wpfb_m2_t0 = Float16x2(Wpfb0_m2, Wpfb1_m2)
        (Wpfb0_m3, Wpfb1_m3) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            time0 = 0 + thread2time(thread)
            time1 = time0 + 4
            s0 = time0 + 24
            s1 = time1 + 24
            W0 = 0.06636127f0 * Wkernel(s0, 4, 8)
            W1 = 0.06636127f0 * Wkernel(s1, 4, 8)
            (W0, W1)
        end
        Wpfb_m3_t0 = Float16x2(Wpfb0_m3, Wpfb1_m3)
        Wpfb_m0 = Wpfb_m0_t0
        Wpfb_m1 = Wpfb_m1_t0
        Wpfb_m2 = Wpfb_m2_t0
        Wpfb_m3 = Wpfb_m3_t0
        Wpfb_mtaps0 = Wpfb_m0
        Wpfb_mtaps1 = Wpfb_m1
        Wpfb_mtaps2 = Wpfb_m2
        Wpfb_mtaps3 = Wpfb_m3
        (X0, X1) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            time0 = thread2time(thread)
            time1 = time0 + 4
            X0 = cispi(((time0 * Int32(U - 1)) % Int32(2U)) / Float32(U))
            X1 = cispi(((time1 * Int32(U - 1)) % Int32(2U)) / Float32(U))
            (X0, X1)
        end
        Xre = Float16x2(real(X0), real(X1))
        Xim = Float16x2(imag(X0), imag(X1))
        X_cplx0 = Xre
        X_cplx1 = Xim
        (Γ¹0, Γ¹1) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            thread0 = (thread ÷ (1i32)) % (2i32)
            thread1 = (thread ÷ (2i32)) % (2i32)
            thread2 = (thread ÷ (4i32)) % (2i32)
            thread3 = (thread ÷ (8i32)) % (2i32)
            thread4 = (thread ÷ (16i32)) % (2i32)
            timehi0 = (4i32) * (0i32) + (2i32) * thread1 + (1i32) * thread0
            timehi1 = (4i32) * (1i32) + (2i32) * thread1 + (1i32) * thread0
            dish_in0 = 0i32
            dish_in1 = 0i32
            freqlo = (1i32) * thread2 + (2i32) * thread4 + (4i32) * thread3
            dish = 0i32
            delta0 = dish == dish_in0
            delta1 = dish == dish_in1
            (Γ¹0, Γ¹1) = (
                delta0 * cispi((((-2i32) * timehi0 * freqlo) / 8.0f0) % 2.0f0),
                delta1 * cispi((((-2i32) * timehi1 * freqlo) / 8.0f0) % 2.0f0),
            )
            (Γ¹0, Γ¹1)
        end
        Γ¹rere = Float16x2(real(Γ¹0), real(Γ¹1))
        Γ¹reim = Float16x2(-(imag(Γ¹0)), -(imag(Γ¹1)))
        Γ¹imre = Float16x2(imag(Γ¹0), imag(Γ¹1))
        Γ¹imim = Float16x2(real(Γ¹0), real(Γ¹1))
        Γ¹re_cplx_in0 = Γ¹rere
        Γ¹re_cplx_in1 = Γ¹reim
        Γ¹im_cplx_in0 = Γ¹imre
        Γ¹im_cplx_in1 = Γ¹imim
        Γ¹_cplx0_cplx_in0 = Γ¹re_cplx_in0
        Γ¹_cplx1_cplx_in0 = Γ¹im_cplx_in0
        Γ¹_cplx0_cplx_in1 = Γ¹re_cplx_in1
        Γ¹_cplx1_cplx_in1 = Γ¹im_cplx_in1
        (Γ³0, Γ³1) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            thread0 = (thread ÷ (1i32)) % (2i32)
            thread1 = (thread ÷ (2i32)) % (2i32)
            thread2 = (thread ÷ (4i32)) % (2i32)
            thread3 = (thread ÷ (8i32)) % (2i32)
            thread4 = (thread ÷ (16i32)) % (2i32)
            timelo0 = 0i32
            timelo1 = 0i32
            dish_in0 = (1i32) * (0i32) + (2i32) * thread1 + (4i32) * thread0
            dish_in1 = (1i32) * (1i32) + (2i32) * thread1 + (4i32) * thread0
            freqhi = 0i32
            dish = (1i32) * thread2 + (2i32) * thread4 + (4i32) * thread3
            delta0 = dish == dish_in0
            delta1 = dish == dish_in1
            (Γ³0, Γ³1) = (
                delta0 * cispi((((-2i32) * timelo0 * freqhi) / 1.0f0) % 2.0f0),
                delta1 * cispi((((-2i32) * timelo1 * freqhi) / 1.0f0) % 2.0f0),
            )
            (Γ³0, Γ³1)
        end
        Γ³rere = Float16x2(real(Γ³0), real(Γ³1))
        Γ³reim = Float16x2(-(imag(Γ³0)), -(imag(Γ³1)))
        Γ³imre = Float16x2(imag(Γ³0), imag(Γ³1))
        Γ³imim = Float16x2(real(Γ³0), real(Γ³1))
        Γ³re_cplx_in0 = Γ³rere
        Γ³re_cplx_in1 = Γ³reim
        Γ³im_cplx_in0 = Γ³imre
        Γ³im_cplx_in1 = Γ³imim
        Γ³_cplx0_cplx_in0 = Γ³re_cplx_in0
        Γ³_cplx1_cplx_in0 = Γ³im_cplx_in0
        Γ³_cplx0_cplx_in1 = Γ³re_cplx_in1
        Γ³_cplx1_cplx_in1 = Γ³im_cplx_in1
        Γ³_cplx0_cplx_in0_dish0 = Γ³_cplx0_cplx_in0
        Γ³_cplx0_cplx_in0_dish1 = Γ³_cplx0_cplx_in0
        Γ³_cplx1_cplx_in0_dish0 = Γ³_cplx1_cplx_in0
        Γ³_cplx1_cplx_in0_dish1 = Γ³_cplx1_cplx_in0
        Γ³_cplx0_cplx_in1_dish0 = Γ³_cplx0_cplx_in1
        Γ³_cplx0_cplx_in1_dish1 = Γ³_cplx0_cplx_in1
        Γ³_cplx1_cplx_in1_dish0 = Γ³_cplx1_cplx_in1
        Γ³_cplx1_cplx_in1_dish1 = Γ³_cplx1_cplx_in1
        for t_outer in 0:256:32767
            Tmin + t_outer ≥ Tmax && break
            (E_dish0_time0, E_dish4_time0, E_dish8_time0, E_dish12_time0) = IndexSpaces.unsafe_load4_global(
                E_memory,
                let
                    offset = 12288 * Tmin + 32 * Fmin
                    length = 402653184
                    mod(
                        (
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) % 384) * 32 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 +
                            (
                                (
                                    (
                                        (
                                            (
                                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 4 +
                                                ((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256
                                            ) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2
                                        ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2
                                    ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8
                                ) % 32768
                            ) * 12288 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16
                        ) + offset,
                        length,
                    )
                end + 1i32,
            )
            (E_dish0_time32, E_dish4_time32, E_dish8_time32, E_dish12_time32) = IndexSpaces.unsafe_load4_global(
                E_memory,
                let
                    offset = 12288 * Tmin + 32 * Fmin
                    length = 402653184
                    mod(
                        (
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) % 384) * 32 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 +
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) *
                                                    4 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256
                                                ) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2
                                            ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2
                                        ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8
                                    ) + 32
                                ) % 32768
                            ) * 12288 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16
                        ) + offset,
                        length,
                    )
                end + 1i32,
            )
            (E_dish0_time64, E_dish4_time64, E_dish8_time64, E_dish12_time64) = IndexSpaces.unsafe_load4_global(
                E_memory,
                let
                    offset = 12288 * Tmin + 32 * Fmin
                    length = 402653184
                    mod(
                        (
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) % 384) * 32 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 +
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) *
                                                    4 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256
                                                ) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2
                                            ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2
                                        ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8
                                    ) + 64
                                ) % 32768
                            ) * 12288 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16
                        ) + offset,
                        length,
                    )
                end + 1i32,
            )
            (E_dish0_time96, E_dish4_time96, E_dish8_time96, E_dish12_time96) = IndexSpaces.unsafe_load4_global(
                E_memory,
                let
                    offset = 12288 * Tmin + 32 * Fmin
                    length = 402653184
                    mod(
                        (
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) % 384) * 32 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 +
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) *
                                                    4 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256
                                                ) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2
                                            ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2
                                        ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8
                                    ) + 96
                                ) % 32768
                            ) * 12288 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16
                        ) + offset,
                        length,
                    )
                end + 1i32,
            )
            (E_dish0_time128, E_dish4_time128, E_dish8_time128, E_dish12_time128) = IndexSpaces.unsafe_load4_global(
                E_memory,
                let
                    offset = 12288 * Tmin + 32 * Fmin
                    length = 402653184
                    mod(
                        (
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) % 384) * 32 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 +
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) *
                                                    4 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256
                                                ) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2
                                            ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2
                                        ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8
                                    ) + 128
                                ) % 32768
                            ) * 12288 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16
                        ) + offset,
                        length,
                    )
                end + 1i32,
            )
            (E_dish0_time160, E_dish4_time160, E_dish8_time160, E_dish12_time160) = IndexSpaces.unsafe_load4_global(
                E_memory,
                let
                    offset = 12288 * Tmin + 32 * Fmin
                    length = 402653184
                    mod(
                        (
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) % 384) * 32 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 +
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) *
                                                    4 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256
                                                ) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2
                                            ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2
                                        ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8
                                    ) + 160
                                ) % 32768
                            ) * 12288 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16
                        ) + offset,
                        length,
                    )
                end + 1i32,
            )
            (E_dish0_time192, E_dish4_time192, E_dish8_time192, E_dish12_time192) = IndexSpaces.unsafe_load4_global(
                E_memory,
                let
                    offset = 12288 * Tmin + 32 * Fmin
                    length = 402653184
                    mod(
                        (
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) % 384) * 32 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 +
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) *
                                                    4 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256
                                                ) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2
                                            ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2
                                        ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8
                                    ) + 192
                                ) % 32768
                            ) * 12288 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16
                        ) + offset,
                        length,
                    )
                end + 1i32,
            )
            (E_dish0_time224, E_dish4_time224, E_dish8_time224, E_dish12_time224) = IndexSpaces.unsafe_load4_global(
                E_memory,
                let
                    offset = 12288 * Tmin + 32 * Fmin
                    length = 402653184
                    mod(
                        (
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) % 384) * 32 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 +
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) *
                                                    4 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256
                                                ) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2
                                            ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2
                                        ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8
                                    ) + 224
                                ) % 32768
                            ) * 12288 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16
                        ) + offset,
                        length,
                    )
                end + 1i32,
            )
            is_lo_thread = IndexSpaces.cuda_threadidx() & 0x00000008 == 0x00
            (E1_dish0_time0, E1_dish8_time0) = let
                src = if is_lo_thread
                    E_dish8_time0
                else
                    E_dish0_time0
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E_dish0_time0, dst)
                else
                    (dst, E_dish8_time0)
                end
            end
            (E1_dish4_time0, E1_dish12_time0) = let
                src = if is_lo_thread
                    E_dish12_time0
                else
                    E_dish4_time0
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E_dish4_time0, dst)
                else
                    (dst, E_dish12_time0)
                end
            end
            (E1_dish0_time32, E1_dish8_time32) = let
                src = if is_lo_thread
                    E_dish8_time32
                else
                    E_dish0_time32
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E_dish0_time32, dst)
                else
                    (dst, E_dish8_time32)
                end
            end
            (E1_dish4_time32, E1_dish12_time32) = let
                src = if is_lo_thread
                    E_dish12_time32
                else
                    E_dish4_time32
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E_dish4_time32, dst)
                else
                    (dst, E_dish12_time32)
                end
            end
            (E1_dish0_time64, E1_dish8_time64) = let
                src = if is_lo_thread
                    E_dish8_time64
                else
                    E_dish0_time64
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E_dish0_time64, dst)
                else
                    (dst, E_dish8_time64)
                end
            end
            (E1_dish4_time64, E1_dish12_time64) = let
                src = if is_lo_thread
                    E_dish12_time64
                else
                    E_dish4_time64
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E_dish4_time64, dst)
                else
                    (dst, E_dish12_time64)
                end
            end
            (E1_dish0_time96, E1_dish8_time96) = let
                src = if is_lo_thread
                    E_dish8_time96
                else
                    E_dish0_time96
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E_dish0_time96, dst)
                else
                    (dst, E_dish8_time96)
                end
            end
            (E1_dish4_time96, E1_dish12_time96) = let
                src = if is_lo_thread
                    E_dish12_time96
                else
                    E_dish4_time96
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E_dish4_time96, dst)
                else
                    (dst, E_dish12_time96)
                end
            end
            (E1_dish0_time128, E1_dish8_time128) = let
                src = if is_lo_thread
                    E_dish8_time128
                else
                    E_dish0_time128
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E_dish0_time128, dst)
                else
                    (dst, E_dish8_time128)
                end
            end
            (E1_dish4_time128, E1_dish12_time128) = let
                src = if is_lo_thread
                    E_dish12_time128
                else
                    E_dish4_time128
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E_dish4_time128, dst)
                else
                    (dst, E_dish12_time128)
                end
            end
            (E1_dish0_time160, E1_dish8_time160) = let
                src = if is_lo_thread
                    E_dish8_time160
                else
                    E_dish0_time160
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E_dish0_time160, dst)
                else
                    (dst, E_dish8_time160)
                end
            end
            (E1_dish4_time160, E1_dish12_time160) = let
                src = if is_lo_thread
                    E_dish12_time160
                else
                    E_dish4_time160
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E_dish4_time160, dst)
                else
                    (dst, E_dish12_time160)
                end
            end
            (E1_dish0_time192, E1_dish8_time192) = let
                src = if is_lo_thread
                    E_dish8_time192
                else
                    E_dish0_time192
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E_dish0_time192, dst)
                else
                    (dst, E_dish8_time192)
                end
            end
            (E1_dish4_time192, E1_dish12_time192) = let
                src = if is_lo_thread
                    E_dish12_time192
                else
                    E_dish4_time192
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E_dish4_time192, dst)
                else
                    (dst, E_dish12_time192)
                end
            end
            (E1_dish0_time224, E1_dish8_time224) = let
                src = if is_lo_thread
                    E_dish8_time224
                else
                    E_dish0_time224
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E_dish0_time224, dst)
                else
                    (dst, E_dish8_time224)
                end
            end
            (E1_dish4_time224, E1_dish12_time224) = let
                src = if is_lo_thread
                    E_dish12_time224
                else
                    E_dish4_time224
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E_dish4_time224, dst)
                else
                    (dst, E_dish12_time224)
                end
            end
            E1lo_dish0_time0 = E1_dish0_time0
            E1hi_dish0_time0 = E1_dish8_time0
            E1lo_dish4_time0 = E1_dish4_time0
            E1hi_dish4_time0 = E1_dish12_time0
            E1lo_dish0_time32 = E1_dish0_time32
            E1hi_dish0_time32 = E1_dish8_time32
            E1lo_dish4_time32 = E1_dish4_time32
            E1hi_dish4_time32 = E1_dish12_time32
            E1lo_dish0_time64 = E1_dish0_time64
            E1hi_dish0_time64 = E1_dish8_time64
            E1lo_dish4_time64 = E1_dish4_time64
            E1hi_dish4_time64 = E1_dish12_time64
            E1lo_dish0_time96 = E1_dish0_time96
            E1hi_dish0_time96 = E1_dish8_time96
            E1lo_dish4_time96 = E1_dish4_time96
            E1hi_dish4_time96 = E1_dish12_time96
            E1lo_dish0_time128 = E1_dish0_time128
            E1hi_dish0_time128 = E1_dish8_time128
            E1lo_dish4_time128 = E1_dish4_time128
            E1hi_dish4_time128 = E1_dish12_time128
            E1lo_dish0_time160 = E1_dish0_time160
            E1hi_dish0_time160 = E1_dish8_time160
            E1lo_dish4_time160 = E1_dish4_time160
            E1hi_dish4_time160 = E1_dish12_time160
            E1lo_dish0_time192 = E1_dish0_time192
            E1hi_dish0_time192 = E1_dish8_time192
            E1lo_dish4_time192 = E1_dish4_time192
            E1hi_dish4_time192 = E1_dish12_time192
            E1lo_dish0_time224 = E1_dish0_time224
            E1hi_dish0_time224 = E1_dish8_time224
            E1lo_dish4_time224 = E1_dish4_time224
            E1hi_dish4_time224 = E1_dish12_time224
            E1_dish0_time0 = E1lo_dish0_time0
            E1_dish0_time4 = E1hi_dish0_time0
            E1_dish4_time0 = E1lo_dish4_time0
            E1_dish4_time4 = E1hi_dish4_time0
            E1_dish0_time32 = E1lo_dish0_time32
            E1_dish0_time36 = E1hi_dish0_time32
            E1_dish4_time32 = E1lo_dish4_time32
            E1_dish4_time36 = E1hi_dish4_time32
            E1_dish0_time64 = E1lo_dish0_time64
            E1_dish0_time68 = E1hi_dish0_time64
            E1_dish4_time64 = E1lo_dish4_time64
            E1_dish4_time68 = E1hi_dish4_time64
            E1_dish0_time96 = E1lo_dish0_time96
            E1_dish0_time100 = E1hi_dish0_time96
            E1_dish4_time96 = E1lo_dish4_time96
            E1_dish4_time100 = E1hi_dish4_time96
            E1_dish0_time128 = E1lo_dish0_time128
            E1_dish0_time132 = E1hi_dish0_time128
            E1_dish4_time128 = E1lo_dish4_time128
            E1_dish4_time132 = E1hi_dish4_time128
            E1_dish0_time160 = E1lo_dish0_time160
            E1_dish0_time164 = E1hi_dish0_time160
            E1_dish4_time160 = E1lo_dish4_time160
            E1_dish4_time164 = E1hi_dish4_time160
            E1_dish0_time192 = E1lo_dish0_time192
            E1_dish0_time196 = E1hi_dish0_time192
            E1_dish4_time192 = E1lo_dish4_time192
            E1_dish4_time196 = E1hi_dish4_time192
            E1_dish0_time224 = E1lo_dish0_time224
            E1_dish0_time228 = E1hi_dish0_time224
            E1_dish4_time224 = E1lo_dish4_time224
            E1_dish4_time228 = E1hi_dish4_time224
            (E2_dish0_time0, E2_dish0_time4) = (
                IndexSpaces.get_lo16(E1_dish0_time0, E1_dish0_time4), IndexSpaces.get_hi16(E1_dish0_time0, E1_dish0_time4)
            )
            (E2_dish4_time0, E2_dish4_time4) = (
                IndexSpaces.get_lo16(E1_dish4_time0, E1_dish4_time4), IndexSpaces.get_hi16(E1_dish4_time0, E1_dish4_time4)
            )
            (E2_dish0_time32, E2_dish0_time36) = (
                IndexSpaces.get_lo16(E1_dish0_time32, E1_dish0_time36), IndexSpaces.get_hi16(E1_dish0_time32, E1_dish0_time36)
            )
            (E2_dish4_time32, E2_dish4_time36) = (
                IndexSpaces.get_lo16(E1_dish4_time32, E1_dish4_time36), IndexSpaces.get_hi16(E1_dish4_time32, E1_dish4_time36)
            )
            (E2_dish0_time64, E2_dish0_time68) = (
                IndexSpaces.get_lo16(E1_dish0_time64, E1_dish0_time68), IndexSpaces.get_hi16(E1_dish0_time64, E1_dish0_time68)
            )
            (E2_dish4_time64, E2_dish4_time68) = (
                IndexSpaces.get_lo16(E1_dish4_time64, E1_dish4_time68), IndexSpaces.get_hi16(E1_dish4_time64, E1_dish4_time68)
            )
            (E2_dish0_time96, E2_dish0_time100) = (
                IndexSpaces.get_lo16(E1_dish0_time96, E1_dish0_time100), IndexSpaces.get_hi16(E1_dish0_time96, E1_dish0_time100)
            )
            (E2_dish4_time96, E2_dish4_time100) = (
                IndexSpaces.get_lo16(E1_dish4_time96, E1_dish4_time100), IndexSpaces.get_hi16(E1_dish4_time96, E1_dish4_time100)
            )
            (E2_dish0_time128, E2_dish0_time132) = (
                IndexSpaces.get_lo16(E1_dish0_time128, E1_dish0_time132), IndexSpaces.get_hi16(E1_dish0_time128, E1_dish0_time132)
            )
            (E2_dish4_time128, E2_dish4_time132) = (
                IndexSpaces.get_lo16(E1_dish4_time128, E1_dish4_time132), IndexSpaces.get_hi16(E1_dish4_time128, E1_dish4_time132)
            )
            (E2_dish0_time160, E2_dish0_time164) = (
                IndexSpaces.get_lo16(E1_dish0_time160, E1_dish0_time164), IndexSpaces.get_hi16(E1_dish0_time160, E1_dish0_time164)
            )
            (E2_dish4_time160, E2_dish4_time164) = (
                IndexSpaces.get_lo16(E1_dish4_time160, E1_dish4_time164), IndexSpaces.get_hi16(E1_dish4_time160, E1_dish4_time164)
            )
            (E2_dish0_time192, E2_dish0_time196) = (
                IndexSpaces.get_lo16(E1_dish0_time192, E1_dish0_time196), IndexSpaces.get_hi16(E1_dish0_time192, E1_dish0_time196)
            )
            (E2_dish4_time192, E2_dish4_time196) = (
                IndexSpaces.get_lo16(E1_dish4_time192, E1_dish4_time196), IndexSpaces.get_hi16(E1_dish4_time192, E1_dish4_time196)
            )
            (E2_dish0_time224, E2_dish0_time228) = (
                IndexSpaces.get_lo16(E1_dish0_time224, E1_dish0_time228), IndexSpaces.get_hi16(E1_dish0_time224, E1_dish0_time228)
            )
            (E2_dish4_time224, E2_dish4_time228) = (
                IndexSpaces.get_lo16(E1_dish4_time224, E1_dish4_time228), IndexSpaces.get_hi16(E1_dish4_time224, E1_dish4_time228)
            )
            E2lo_dish0_time0 = E2_dish0_time0
            E2hi_dish0_time0 = E2_dish0_time4
            E2lo_dish4_time0 = E2_dish4_time0
            E2hi_dish4_time0 = E2_dish4_time4
            E2lo_dish0_time32 = E2_dish0_time32
            E2hi_dish0_time32 = E2_dish0_time36
            E2lo_dish4_time32 = E2_dish4_time32
            E2hi_dish4_time32 = E2_dish4_time36
            E2lo_dish0_time64 = E2_dish0_time64
            E2hi_dish0_time64 = E2_dish0_time68
            E2lo_dish4_time64 = E2_dish4_time64
            E2hi_dish4_time64 = E2_dish4_time68
            E2lo_dish0_time96 = E2_dish0_time96
            E2hi_dish0_time96 = E2_dish0_time100
            E2lo_dish4_time96 = E2_dish4_time96
            E2hi_dish4_time96 = E2_dish4_time100
            E2lo_dish0_time128 = E2_dish0_time128
            E2hi_dish0_time128 = E2_dish0_time132
            E2lo_dish4_time128 = E2_dish4_time128
            E2hi_dish4_time128 = E2_dish4_time132
            E2lo_dish0_time160 = E2_dish0_time160
            E2hi_dish0_time160 = E2_dish0_time164
            E2lo_dish4_time160 = E2_dish4_time160
            E2hi_dish4_time160 = E2_dish4_time164
            E2lo_dish0_time192 = E2_dish0_time192
            E2hi_dish0_time192 = E2_dish0_time196
            E2lo_dish4_time192 = E2_dish4_time192
            E2hi_dish4_time192 = E2_dish4_time196
            E2lo_dish0_time224 = E2_dish0_time224
            E2hi_dish0_time224 = E2_dish0_time228
            E2lo_dish4_time224 = E2_dish4_time224
            E2hi_dish4_time224 = E2_dish4_time228
            E2_dish0_time0 = E2lo_dish0_time0
            E2_dish2_time0 = E2hi_dish0_time0
            E2_dish4_time0 = E2lo_dish4_time0
            E2_dish6_time0 = E2hi_dish4_time0
            E2_dish0_time32 = E2lo_dish0_time32
            E2_dish2_time32 = E2hi_dish0_time32
            E2_dish4_time32 = E2lo_dish4_time32
            E2_dish6_time32 = E2hi_dish4_time32
            E2_dish0_time64 = E2lo_dish0_time64
            E2_dish2_time64 = E2hi_dish0_time64
            E2_dish4_time64 = E2lo_dish4_time64
            E2_dish6_time64 = E2hi_dish4_time64
            E2_dish0_time96 = E2lo_dish0_time96
            E2_dish2_time96 = E2hi_dish0_time96
            E2_dish4_time96 = E2lo_dish4_time96
            E2_dish6_time96 = E2hi_dish4_time96
            E2_dish0_time128 = E2lo_dish0_time128
            E2_dish2_time128 = E2hi_dish0_time128
            E2_dish4_time128 = E2lo_dish4_time128
            E2_dish6_time128 = E2hi_dish4_time128
            E2_dish0_time160 = E2lo_dish0_time160
            E2_dish2_time160 = E2hi_dish0_time160
            E2_dish4_time160 = E2lo_dish4_time160
            E2_dish6_time160 = E2hi_dish4_time160
            E2_dish0_time192 = E2lo_dish0_time192
            E2_dish2_time192 = E2hi_dish0_time192
            E2_dish4_time192 = E2lo_dish4_time192
            E2_dish6_time192 = E2hi_dish4_time192
            E2_dish0_time224 = E2lo_dish0_time224
            E2_dish2_time224 = E2hi_dish0_time224
            E2_dish4_time224 = E2lo_dish4_time224
            E2_dish6_time224 = E2hi_dish4_time224
            F_dish0_time0 = E2_dish0_time0
            F_dish2_time0 = E2_dish2_time0
            F_dish4_time0 = E2_dish4_time0
            F_dish6_time0 = E2_dish6_time0
            F_dish0_time32 = E2_dish0_time32
            F_dish2_time32 = E2_dish2_time32
            F_dish4_time32 = E2_dish4_time32
            F_dish6_time32 = E2_dish6_time32
            F_dish0_time64 = E2_dish0_time64
            F_dish2_time64 = E2_dish2_time64
            F_dish4_time64 = E2_dish4_time64
            F_dish6_time64 = E2_dish6_time64
            F_dish0_time96 = E2_dish0_time96
            F_dish2_time96 = E2_dish2_time96
            F_dish4_time96 = E2_dish4_time96
            F_dish6_time96 = E2_dish6_time96
            F_dish0_time128 = E2_dish0_time128
            F_dish2_time128 = E2_dish2_time128
            F_dish4_time128 = E2_dish4_time128
            F_dish6_time128 = E2_dish6_time128
            F_dish0_time160 = E2_dish0_time160
            F_dish2_time160 = E2_dish2_time160
            F_dish4_time160 = E2_dish4_time160
            F_dish6_time160 = E2_dish6_time160
            F_dish0_time192 = E2_dish0_time192
            F_dish2_time192 = E2_dish2_time192
            F_dish4_time192 = E2_dish4_time192
            F_dish6_time192 = E2_dish6_time192
            F_dish0_time224 = E2_dish0_time224
            F_dish2_time224 = E2_dish2_time224
            F_dish4_time224 = E2_dish4_time224
            F_dish6_time224 = E2_dish6_time224
            if true
                F_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) % 2) * 130 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish0_time0
            end
            if true
                F_shared[(((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) % 2) * 130 + ((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish2_time0
            end
            if true
                F_shared[(((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) % 2) * 130 + ((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish4_time0
            end
            if true
                F_shared[(((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) % 2) * 130 + ((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish6_time0
            end
            if true
                F_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 32) % 2) * 130 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 32) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 32) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish0_time32
            end
            if true
                F_shared[(((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 32) % 2) * 130 + ((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 32) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 32) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish2_time32
            end
            if true
                F_shared[(((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 32) % 2) * 130 + ((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 32) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 32) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish4_time32
            end
            if true
                F_shared[(((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 32) % 2) * 130 + ((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 32) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 32) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish6_time32
            end
            if true
                F_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 64) % 2) * 130 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 64) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 64) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish0_time64
            end
            if true
                F_shared[(((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 64) % 2) * 130 + ((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 64) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 64) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish2_time64
            end
            if true
                F_shared[(((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 64) % 2) * 130 + ((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 64) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 64) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish4_time64
            end
            if true
                F_shared[(((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 64) % 2) * 130 + ((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 64) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 64) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish6_time64
            end
            if true
                F_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 96) % 2) * 130 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 96) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 96) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish0_time96
            end
            if true
                F_shared[(((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 96) % 2) * 130 + ((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 96) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 96) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish2_time96
            end
            if true
                F_shared[(((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 96) % 2) * 130 + ((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 96) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 96) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish4_time96
            end
            if true
                F_shared[(((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 96) % 2) * 130 + ((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 96) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 96) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish6_time96
            end
            if true
                F_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 128) % 2) * 130 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 128) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 128) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish0_time128
            end
            if true
                F_shared[(((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 128) % 2) * 130 + ((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 128) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 128) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish2_time128
            end
            if true
                F_shared[(((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 128) % 2) * 130 + ((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 128) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 128) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish4_time128
            end
            if true
                F_shared[(((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 128) % 2) * 130 + ((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 128) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 128) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish6_time128
            end
            if true
                F_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 160) % 2) * 130 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 160) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 160) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish0_time160
            end
            if true
                F_shared[(((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 160) % 2) * 130 + ((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 160) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 160) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish2_time160
            end
            if true
                F_shared[(((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 160) % 2) * 130 + ((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 160) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 160) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish4_time160
            end
            if true
                F_shared[(((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 160) % 2) * 130 + ((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 160) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 160) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish6_time160
            end
            if true
                F_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 192) % 2) * 130 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 192) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 192) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish0_time192
            end
            if true
                F_shared[(((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 192) % 2) * 130 + ((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 192) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 192) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish2_time192
            end
            if true
                F_shared[(((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 192) % 2) * 130 + ((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 192) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 192) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish4_time192
            end
            if true
                F_shared[(((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 192) % 2) * 130 + ((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 192) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 192) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish6_time192
            end
            if true
                F_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 224) % 2) * 130 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 224) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 224) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish0_time224
            end
            if true
                F_shared[(((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 224) % 2) * 130 + ((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 224) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 224) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish2_time224
            end
            if true
                F_shared[(((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 224) % 2) * 130 + ((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 224) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 224) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish4_time224
            end
            if true
                F_shared[(((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 224) % 2) * 130 + ((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 224) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 224) ÷ 8) % 32) * 289) + 0) + 0x01] =
                    F_dish6_time224
            end
            IndexSpaces.cuda_sync_threads()
            for t_inner in 0:8:255
                let
                    dish = 0
                    F_in = F_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 8) * 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 32) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 16) ÷ 4) % 16 + (((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 8, 256) ÷ 8) % 32) * 8) % 2) * 130 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 8) * 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 32) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 16) ÷ 2) % 2) * 32 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 8, 256) ÷ 8) % 32) * 8) ÷ 2) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 8, 256) ÷ 8) % 32) * 8) ÷ 8) % 32) * 289) + 0x01]
                    (E_cplx0_dish0, E_cplx1_dish0, E_cplx0_dish1, E_cplx1_dish1) = convert(NTuple{4,Float16x2}, F_in)
                    W_m0 = Wpfb_mtaps0
                    W_m1 = Wpfb_mtaps1
                    W_m2 = Wpfb_mtaps2
                    W_m3 = Wpfb_mtaps3
                    E2_cplx0_dish0 = -W_m3 * E_cplx0_dish0
                    E2_cplx1_dish0 = -W_m3 * E_cplx1_dish0
                    E2_cplx0_dish1 = -W_m3 * E_cplx0_dish1
                    E2_cplx1_dish1 = -W_m3 * E_cplx1_dish1
                    F_ringbuf_m0 = F_ringbuf_mtaps0
                    F_ringbuf_m1 = F_ringbuf_mtaps1
                    F_ringbuf_m2 = F_ringbuf_mtaps2
                    (E_ringbuf_m0_cplx0_dish0, E_ringbuf_m0_cplx1_dish0, E_ringbuf_m0_cplx0_dish1, E_ringbuf_m0_cplx1_dish1) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m0
                    )
                    E2_cplx0_dish0 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish0, E2_cplx0_dish0)
                    E2_cplx1_dish0 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish0, E2_cplx1_dish0)
                    E2_cplx0_dish1 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish1, E2_cplx0_dish1)
                    E2_cplx1_dish1 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish1, E2_cplx1_dish1)
                    (E_ringbuf_m1_cplx0_dish0, E_ringbuf_m1_cplx1_dish0, E_ringbuf_m1_cplx0_dish1, E_ringbuf_m1_cplx1_dish1) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m1
                    )
                    E2_cplx0_dish0 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish0, E2_cplx0_dish0)
                    E2_cplx1_dish0 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish0, E2_cplx1_dish0)
                    E2_cplx0_dish1 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish1, E2_cplx0_dish1)
                    E2_cplx1_dish1 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish1, E2_cplx1_dish1)
                    (E_ringbuf_m2_cplx0_dish0, E_ringbuf_m2_cplx1_dish0, E_ringbuf_m2_cplx0_dish1, E_ringbuf_m2_cplx1_dish1) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m2
                    )
                    E2_cplx0_dish0 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish0, E2_cplx0_dish0)
                    E2_cplx1_dish0 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish0, E2_cplx1_dish0)
                    E2_cplx0_dish1 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish1, E2_cplx0_dish1)
                    E2_cplx1_dish1 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish1, E2_cplx1_dish1)
                    E2re_dish0 = E2_cplx0_dish0
                    E2im_dish0 = E2_cplx1_dish0
                    E2re_dish1 = E2_cplx0_dish1
                    E2im_dish1 = E2_cplx1_dish1
                    Xre = X_cplx0
                    Xim = X_cplx1
                    E3re_dish0 = muladd(Xre, E2re_dish0, -Xim * E2im_dish0)
                    E3re_dish1 = muladd(Xre, E2re_dish1, -Xim * E2im_dish1)
                    E3im_dish0 = muladd(Xre, E2im_dish0, Xim * E2re_dish0)
                    E3im_dish1 = muladd(Xre, E2im_dish1, Xim * E2re_dish1)
                    E3_cplx0_dish0 = E3re_dish0
                    E3_cplx1_dish0 = E3im_dish0
                    E3_cplx0_dish1 = E3re_dish1
                    E3_cplx1_dish1 = E3im_dish1
                    XX_cplx0_dish0 = E3_cplx0_dish0
                    XX_cplx1_dish0 = E3_cplx1_dish0
                    XX_cplx0_dish1 = E3_cplx0_dish1
                    XX_cplx1_dish1 = E3_cplx1_dish1
                    XXre_dish0 = XX_cplx0_dish0
                    XXim_dish0 = XX_cplx1_dish0
                    XXre_dish1 = XX_cplx0_dish1
                    XXim_dish1 = XX_cplx1_dish1
                    XX_cplx_in0_dish0 = XXre_dish0
                    XX_cplx_in1_dish0 = XXim_dish0
                    XX_cplx_in0_dish1 = XXre_dish1
                    XX_cplx_in1_dish1 = XXim_dish1
                    WW_cplx0_dish0 = zero(Float16x2)
                    WW_cplx1_dish0 = zero(Float16x2)
                    WW_cplx0_dish1 = zero(Float16x2)
                    WW_cplx1_dish1 = zero(Float16x2)
                    (WW_cplx0_dish0, WW_cplx1_dish0) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish0, XX_cplx_in1_dish0),
                        (WW_cplx0_dish0, WW_cplx1_dish0),
                    )
                    (WW_cplx0_dish1, WW_cplx1_dish1) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish1, XX_cplx_in1_dish1),
                        (WW_cplx0_dish1, WW_cplx1_dish1),
                    )
                    ZZ_cplx0_dish0 = WW_cplx0_dish0
                    ZZ_cplx1_dish0 = WW_cplx1_dish0
                    ZZ_cplx0_dish1 = WW_cplx0_dish1
                    ZZ_cplx1_dish1 = WW_cplx1_dish1
                    ZZre_dish0 = ZZ_cplx0_dish0
                    ZZim_dish0 = ZZ_cplx1_dish0
                    ZZre_dish1 = ZZ_cplx0_dish1
                    ZZim_dish1 = ZZ_cplx1_dish1
                    ZZ_cplx_in0_dish0 = ZZre_dish0
                    ZZ_cplx_in1_dish0 = ZZim_dish0
                    ZZ_cplx_in0_dish1 = ZZre_dish1
                    ZZ_cplx_in1_dish1 = ZZim_dish1
                    YY_cplx0_dish0 = zero(Float16x2)
                    YY_cplx1_dish0 = zero(Float16x2)
                    YY_cplx0_dish1 = zero(Float16x2)
                    YY_cplx1_dish1 = zero(Float16x2)
                    (YY_cplx0_dish0, YY_cplx1_dish0) = IndexSpaces.mma_m16n8k16(
                        (Γ³_cplx0_cplx_in0_dish0, Γ³_cplx1_cplx_in0_dish0, Γ³_cplx0_cplx_in1_dish0, Γ³_cplx1_cplx_in1_dish0),
                        (ZZ_cplx_in0_dish0, ZZ_cplx_in1_dish0),
                        (YY_cplx0_dish0, YY_cplx1_dish0),
                    )
                    (YY_cplx0_dish1, YY_cplx1_dish1) = IndexSpaces.mma_m16n8k16(
                        (Γ³_cplx0_cplx_in0_dish1, Γ³_cplx1_cplx_in0_dish1, Γ³_cplx0_cplx_in1_dish1, Γ³_cplx1_cplx_in1_dish1),
                        (ZZ_cplx_in0_dish1, ZZ_cplx_in1_dish1),
                        (YY_cplx0_dish1, YY_cplx1_dish1),
                    )
                    E4_cplx0_dish0 = YY_cplx0_dish0
                    E4_cplx1_dish0 = YY_cplx1_dish0
                    E4_cplx0_dish1 = YY_cplx0_dish1
                    E4_cplx1_dish1 = YY_cplx1_dish1
                    E5_cplx0_dish0 = Gains * E4_cplx0_dish0
                    E5_cplx1_dish0 = Gains * E4_cplx1_dish0
                    E5_cplx0_dish1 = Gains * E4_cplx0_dish1
                    E5_cplx1_dish1 = Gains * E4_cplx1_dish1
                    E5_cplx0_dish0 = clamp(E5_cplx0_dish0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish0 = clamp(E5_cplx1_dish0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish1 = clamp(E5_cplx0_dish1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish1 = clamp(E5_cplx1_dish1, Float16x2(-7, -7), Float16x2(7, 7))
                    F̄_out = Int4x8((E5_cplx0_dish0, E5_cplx1_dish0, E5_cplx0_dish1, E5_cplx1_dish1))
                    if true
                        F̄_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 8) * 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 32) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 4) ÷ 2) % 4) * 65 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 8) * 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 32) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) % 2) * 16 + (((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(t_inner, 0, 8, 256) ÷ 8) % 32) * 8) ÷ 8) % 32) * 289) + 0) + 0x01] =
                            F̄_out
                    end
                    F_ringbuf_m0 = F_ringbuf_mtaps0
                    F_ringbuf_m1 = F_ringbuf_mtaps1
                    F_ringbuf_m2 = F_ringbuf_mtaps2
                    F_ringbuf_m0 = F_ringbuf_m1
                    F_ringbuf_m1 = F_ringbuf_m2
                    F_ringbuf_m2 = F_in
                    F_ringbuf_mtaps0 = F_ringbuf_m0
                    F_ringbuf_mtaps1 = F_ringbuf_m1
                    F_ringbuf_mtaps2 = F_ringbuf_m2
                end
            end
            IndexSpaces.cuda_sync_threads()
            Ē_dish0_time0 = F̄_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + (((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish2_time0 = F̄_shared[((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + ((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + (((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish4_time0 = F̄_shared[((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + ((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + (((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish6_time0 = F̄_shared[((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + ((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + (((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish0_time32 = F̄_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 32) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish2_time32 = F̄_shared[((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + ((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 32) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish4_time32 = F̄_shared[((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + ((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 32) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish6_time32 = F̄_shared[((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + ((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 32) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish0_time64 = F̄_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 64) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish2_time64 = F̄_shared[((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + ((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 64) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish4_time64 = F̄_shared[((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + ((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 64) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish6_time64 = F̄_shared[((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + ((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 64) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish0_time96 = F̄_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 96) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish2_time96 = F̄_shared[((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + ((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 96) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish4_time96 = F̄_shared[((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + ((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 96) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish6_time96 = F̄_shared[((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + ((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 96) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish0_time128 = F̄_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 128) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish2_time128 = F̄_shared[((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + ((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 128) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish4_time128 = F̄_shared[((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + ((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 128) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish6_time128 = F̄_shared[((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + ((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 128) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish0_time160 = F̄_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 160) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish2_time160 = F̄_shared[((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + ((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 160) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish4_time160 = F̄_shared[((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + ((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 160) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish6_time160 = F̄_shared[((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + ((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 160) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish0_time192 = F̄_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 192) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish2_time192 = F̄_shared[((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + ((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 192) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish4_time192 = F̄_shared[((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + ((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 192) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish6_time192 = F̄_shared[((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + ((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 192) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish0_time224 = F̄_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 224) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish2_time224 = F̄_shared[((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + ((((2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 224) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish4_time224 = F̄_shared[((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + ((((4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 224) ÷ 8) % 32) * 289) + 0x01]
            Ē_dish6_time224 = F̄_shared[((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) ÷ 2) % 4) * 65 + ((((6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + ((((((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8) + 224) ÷ 8) % 32) * 289) + 0x01]
            (Ē1_dish0_time0, Ē1_dish2_time0) = (
                IndexSpaces.get_lo16(Ē_dish0_time0, Ē_dish2_time0), IndexSpaces.get_hi16(Ē_dish0_time0, Ē_dish2_time0)
            )
            (Ē1_dish4_time0, Ē1_dish6_time0) = (
                IndexSpaces.get_lo16(Ē_dish4_time0, Ē_dish6_time0), IndexSpaces.get_hi16(Ē_dish4_time0, Ē_dish6_time0)
            )
            (Ē1_dish0_time32, Ē1_dish2_time32) = (
                IndexSpaces.get_lo16(Ē_dish0_time32, Ē_dish2_time32), IndexSpaces.get_hi16(Ē_dish0_time32, Ē_dish2_time32)
            )
            (Ē1_dish4_time32, Ē1_dish6_time32) = (
                IndexSpaces.get_lo16(Ē_dish4_time32, Ē_dish6_time32), IndexSpaces.get_hi16(Ē_dish4_time32, Ē_dish6_time32)
            )
            (Ē1_dish0_time64, Ē1_dish2_time64) = (
                IndexSpaces.get_lo16(Ē_dish0_time64, Ē_dish2_time64), IndexSpaces.get_hi16(Ē_dish0_time64, Ē_dish2_time64)
            )
            (Ē1_dish4_time64, Ē1_dish6_time64) = (
                IndexSpaces.get_lo16(Ē_dish4_time64, Ē_dish6_time64), IndexSpaces.get_hi16(Ē_dish4_time64, Ē_dish6_time64)
            )
            (Ē1_dish0_time96, Ē1_dish2_time96) = (
                IndexSpaces.get_lo16(Ē_dish0_time96, Ē_dish2_time96), IndexSpaces.get_hi16(Ē_dish0_time96, Ē_dish2_time96)
            )
            (Ē1_dish4_time96, Ē1_dish6_time96) = (
                IndexSpaces.get_lo16(Ē_dish4_time96, Ē_dish6_time96), IndexSpaces.get_hi16(Ē_dish4_time96, Ē_dish6_time96)
            )
            (Ē1_dish0_time128, Ē1_dish2_time128) = (
                IndexSpaces.get_lo16(Ē_dish0_time128, Ē_dish2_time128), IndexSpaces.get_hi16(Ē_dish0_time128, Ē_dish2_time128)
            )
            (Ē1_dish4_time128, Ē1_dish6_time128) = (
                IndexSpaces.get_lo16(Ē_dish4_time128, Ē_dish6_time128), IndexSpaces.get_hi16(Ē_dish4_time128, Ē_dish6_time128)
            )
            (Ē1_dish0_time160, Ē1_dish2_time160) = (
                IndexSpaces.get_lo16(Ē_dish0_time160, Ē_dish2_time160), IndexSpaces.get_hi16(Ē_dish0_time160, Ē_dish2_time160)
            )
            (Ē1_dish4_time160, Ē1_dish6_time160) = (
                IndexSpaces.get_lo16(Ē_dish4_time160, Ē_dish6_time160), IndexSpaces.get_hi16(Ē_dish4_time160, Ē_dish6_time160)
            )
            (Ē1_dish0_time192, Ē1_dish2_time192) = (
                IndexSpaces.get_lo16(Ē_dish0_time192, Ē_dish2_time192), IndexSpaces.get_hi16(Ē_dish0_time192, Ē_dish2_time192)
            )
            (Ē1_dish4_time192, Ē1_dish6_time192) = (
                IndexSpaces.get_lo16(Ē_dish4_time192, Ē_dish6_time192), IndexSpaces.get_hi16(Ē_dish4_time192, Ē_dish6_time192)
            )
            (Ē1_dish0_time224, Ē1_dish2_time224) = (
                IndexSpaces.get_lo16(Ē_dish0_time224, Ē_dish2_time224), IndexSpaces.get_hi16(Ē_dish0_time224, Ē_dish2_time224)
            )
            (Ē1_dish4_time224, Ē1_dish6_time224) = (
                IndexSpaces.get_lo16(Ē_dish4_time224, Ē_dish6_time224), IndexSpaces.get_hi16(Ē_dish4_time224, Ē_dish6_time224)
            )
            Ē1lo_dish0_time0 = Ē1_dish0_time0
            Ē1hi_dish0_time0 = Ē1_dish2_time0
            Ē1lo_dish4_time0 = Ē1_dish4_time0
            Ē1hi_dish4_time0 = Ē1_dish6_time0
            Ē1lo_dish0_time32 = Ē1_dish0_time32
            Ē1hi_dish0_time32 = Ē1_dish2_time32
            Ē1lo_dish4_time32 = Ē1_dish4_time32
            Ē1hi_dish4_time32 = Ē1_dish6_time32
            Ē1lo_dish0_time64 = Ē1_dish0_time64
            Ē1hi_dish0_time64 = Ē1_dish2_time64
            Ē1lo_dish4_time64 = Ē1_dish4_time64
            Ē1hi_dish4_time64 = Ē1_dish6_time64
            Ē1lo_dish0_time96 = Ē1_dish0_time96
            Ē1hi_dish0_time96 = Ē1_dish2_time96
            Ē1lo_dish4_time96 = Ē1_dish4_time96
            Ē1hi_dish4_time96 = Ē1_dish6_time96
            Ē1lo_dish0_time128 = Ē1_dish0_time128
            Ē1hi_dish0_time128 = Ē1_dish2_time128
            Ē1lo_dish4_time128 = Ē1_dish4_time128
            Ē1hi_dish4_time128 = Ē1_dish6_time128
            Ē1lo_dish0_time160 = Ē1_dish0_time160
            Ē1hi_dish0_time160 = Ē1_dish2_time160
            Ē1lo_dish4_time160 = Ē1_dish4_time160
            Ē1hi_dish4_time160 = Ē1_dish6_time160
            Ē1lo_dish0_time192 = Ē1_dish0_time192
            Ē1hi_dish0_time192 = Ē1_dish2_time192
            Ē1lo_dish4_time192 = Ē1_dish4_time192
            Ē1hi_dish4_time192 = Ē1_dish6_time192
            Ē1lo_dish0_time224 = Ē1_dish0_time224
            Ē1hi_dish0_time224 = Ē1_dish2_time224
            Ē1lo_dish4_time224 = Ē1_dish4_time224
            Ē1hi_dish4_time224 = Ē1_dish6_time224
            Ē1_dish0_freq0_time0 = Ē1lo_dish0_time0
            Ē1_dish0_freq1_time0 = Ē1hi_dish0_time0
            Ē1_dish4_freq0_time0 = Ē1lo_dish4_time0
            Ē1_dish4_freq1_time0 = Ē1hi_dish4_time0
            Ē1_dish0_freq0_time32 = Ē1lo_dish0_time32
            Ē1_dish0_freq1_time32 = Ē1hi_dish0_time32
            Ē1_dish4_freq0_time32 = Ē1lo_dish4_time32
            Ē1_dish4_freq1_time32 = Ē1hi_dish4_time32
            Ē1_dish0_freq0_time64 = Ē1lo_dish0_time64
            Ē1_dish0_freq1_time64 = Ē1hi_dish0_time64
            Ē1_dish4_freq0_time64 = Ē1lo_dish4_time64
            Ē1_dish4_freq1_time64 = Ē1hi_dish4_time64
            Ē1_dish0_freq0_time96 = Ē1lo_dish0_time96
            Ē1_dish0_freq1_time96 = Ē1hi_dish0_time96
            Ē1_dish4_freq0_time96 = Ē1lo_dish4_time96
            Ē1_dish4_freq1_time96 = Ē1hi_dish4_time96
            Ē1_dish0_freq0_time128 = Ē1lo_dish0_time128
            Ē1_dish0_freq1_time128 = Ē1hi_dish0_time128
            Ē1_dish4_freq0_time128 = Ē1lo_dish4_time128
            Ē1_dish4_freq1_time128 = Ē1hi_dish4_time128
            Ē1_dish0_freq0_time160 = Ē1lo_dish0_time160
            Ē1_dish0_freq1_time160 = Ē1hi_dish0_time160
            Ē1_dish4_freq0_time160 = Ē1lo_dish4_time160
            Ē1_dish4_freq1_time160 = Ē1hi_dish4_time160
            Ē1_dish0_freq0_time192 = Ē1lo_dish0_time192
            Ē1_dish0_freq1_time192 = Ē1hi_dish0_time192
            Ē1_dish4_freq0_time192 = Ē1lo_dish4_time192
            Ē1_dish4_freq1_time192 = Ē1hi_dish4_time192
            Ē1_dish0_freq0_time224 = Ē1lo_dish0_time224
            Ē1_dish0_freq1_time224 = Ē1hi_dish0_time224
            Ē1_dish4_freq0_time224 = Ē1lo_dish4_time224
            Ē1_dish4_freq1_time224 = Ē1hi_dish4_time224
            is_lo_thread = IndexSpaces.cuda_threadidx() & 0x00000008 == 0x00
            (Ē2_dish0_freq0_time0, Ē2_dish0_freq1_time0) = let
                src = if is_lo_thread
                    Ē1_dish0_freq1_time0
                else
                    Ē1_dish0_freq0_time0
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (Ē1_dish0_freq0_time0, dst)
                else
                    (dst, Ē1_dish0_freq1_time0)
                end
            end
            (Ē2_dish4_freq0_time0, Ē2_dish4_freq1_time0) = let
                src = if is_lo_thread
                    Ē1_dish4_freq1_time0
                else
                    Ē1_dish4_freq0_time0
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (Ē1_dish4_freq0_time0, dst)
                else
                    (dst, Ē1_dish4_freq1_time0)
                end
            end
            (Ē2_dish0_freq0_time32, Ē2_dish0_freq1_time32) = let
                src = if is_lo_thread
                    Ē1_dish0_freq1_time32
                else
                    Ē1_dish0_freq0_time32
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (Ē1_dish0_freq0_time32, dst)
                else
                    (dst, Ē1_dish0_freq1_time32)
                end
            end
            (Ē2_dish4_freq0_time32, Ē2_dish4_freq1_time32) = let
                src = if is_lo_thread
                    Ē1_dish4_freq1_time32
                else
                    Ē1_dish4_freq0_time32
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (Ē1_dish4_freq0_time32, dst)
                else
                    (dst, Ē1_dish4_freq1_time32)
                end
            end
            (Ē2_dish0_freq0_time64, Ē2_dish0_freq1_time64) = let
                src = if is_lo_thread
                    Ē1_dish0_freq1_time64
                else
                    Ē1_dish0_freq0_time64
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (Ē1_dish0_freq0_time64, dst)
                else
                    (dst, Ē1_dish0_freq1_time64)
                end
            end
            (Ē2_dish4_freq0_time64, Ē2_dish4_freq1_time64) = let
                src = if is_lo_thread
                    Ē1_dish4_freq1_time64
                else
                    Ē1_dish4_freq0_time64
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (Ē1_dish4_freq0_time64, dst)
                else
                    (dst, Ē1_dish4_freq1_time64)
                end
            end
            (Ē2_dish0_freq0_time96, Ē2_dish0_freq1_time96) = let
                src = if is_lo_thread
                    Ē1_dish0_freq1_time96
                else
                    Ē1_dish0_freq0_time96
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (Ē1_dish0_freq0_time96, dst)
                else
                    (dst, Ē1_dish0_freq1_time96)
                end
            end
            (Ē2_dish4_freq0_time96, Ē2_dish4_freq1_time96) = let
                src = if is_lo_thread
                    Ē1_dish4_freq1_time96
                else
                    Ē1_dish4_freq0_time96
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (Ē1_dish4_freq0_time96, dst)
                else
                    (dst, Ē1_dish4_freq1_time96)
                end
            end
            (Ē2_dish0_freq0_time128, Ē2_dish0_freq1_time128) = let
                src = if is_lo_thread
                    Ē1_dish0_freq1_time128
                else
                    Ē1_dish0_freq0_time128
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (Ē1_dish0_freq0_time128, dst)
                else
                    (dst, Ē1_dish0_freq1_time128)
                end
            end
            (Ē2_dish4_freq0_time128, Ē2_dish4_freq1_time128) = let
                src = if is_lo_thread
                    Ē1_dish4_freq1_time128
                else
                    Ē1_dish4_freq0_time128
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (Ē1_dish4_freq0_time128, dst)
                else
                    (dst, Ē1_dish4_freq1_time128)
                end
            end
            (Ē2_dish0_freq0_time160, Ē2_dish0_freq1_time160) = let
                src = if is_lo_thread
                    Ē1_dish0_freq1_time160
                else
                    Ē1_dish0_freq0_time160
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (Ē1_dish0_freq0_time160, dst)
                else
                    (dst, Ē1_dish0_freq1_time160)
                end
            end
            (Ē2_dish4_freq0_time160, Ē2_dish4_freq1_time160) = let
                src = if is_lo_thread
                    Ē1_dish4_freq1_time160
                else
                    Ē1_dish4_freq0_time160
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (Ē1_dish4_freq0_time160, dst)
                else
                    (dst, Ē1_dish4_freq1_time160)
                end
            end
            (Ē2_dish0_freq0_time192, Ē2_dish0_freq1_time192) = let
                src = if is_lo_thread
                    Ē1_dish0_freq1_time192
                else
                    Ē1_dish0_freq0_time192
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (Ē1_dish0_freq0_time192, dst)
                else
                    (dst, Ē1_dish0_freq1_time192)
                end
            end
            (Ē2_dish4_freq0_time192, Ē2_dish4_freq1_time192) = let
                src = if is_lo_thread
                    Ē1_dish4_freq1_time192
                else
                    Ē1_dish4_freq0_time192
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (Ē1_dish4_freq0_time192, dst)
                else
                    (dst, Ē1_dish4_freq1_time192)
                end
            end
            (Ē2_dish0_freq0_time224, Ē2_dish0_freq1_time224) = let
                src = if is_lo_thread
                    Ē1_dish0_freq1_time224
                else
                    Ē1_dish0_freq0_time224
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (Ē1_dish0_freq0_time224, dst)
                else
                    (dst, Ē1_dish0_freq1_time224)
                end
            end
            (Ē2_dish4_freq0_time224, Ē2_dish4_freq1_time224) = let
                src = if is_lo_thread
                    Ē1_dish4_freq1_time224
                else
                    Ē1_dish4_freq0_time224
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (Ē1_dish4_freq0_time224, dst)
                else
                    (dst, Ē1_dish4_freq1_time224)
                end
            end
            Ē2lo_dish0_time0 = Ē2_dish0_freq0_time0
            Ē2hi_dish0_time0 = Ē2_dish0_freq1_time0
            Ē2lo_dish4_time0 = Ē2_dish4_freq0_time0
            Ē2hi_dish4_time0 = Ē2_dish4_freq1_time0
            Ē2lo_dish0_time32 = Ē2_dish0_freq0_time32
            Ē2hi_dish0_time32 = Ē2_dish0_freq1_time32
            Ē2lo_dish4_time32 = Ē2_dish4_freq0_time32
            Ē2hi_dish4_time32 = Ē2_dish4_freq1_time32
            Ē2lo_dish0_time64 = Ē2_dish0_freq0_time64
            Ē2hi_dish0_time64 = Ē2_dish0_freq1_time64
            Ē2lo_dish4_time64 = Ē2_dish4_freq0_time64
            Ē2hi_dish4_time64 = Ē2_dish4_freq1_time64
            Ē2lo_dish0_time96 = Ē2_dish0_freq0_time96
            Ē2hi_dish0_time96 = Ē2_dish0_freq1_time96
            Ē2lo_dish4_time96 = Ē2_dish4_freq0_time96
            Ē2hi_dish4_time96 = Ē2_dish4_freq1_time96
            Ē2lo_dish0_time128 = Ē2_dish0_freq0_time128
            Ē2hi_dish0_time128 = Ē2_dish0_freq1_time128
            Ē2lo_dish4_time128 = Ē2_dish4_freq0_time128
            Ē2hi_dish4_time128 = Ē2_dish4_freq1_time128
            Ē2lo_dish0_time160 = Ē2_dish0_freq0_time160
            Ē2hi_dish0_time160 = Ē2_dish0_freq1_time160
            Ē2lo_dish4_time160 = Ē2_dish4_freq0_time160
            Ē2hi_dish4_time160 = Ē2_dish4_freq1_time160
            Ē2lo_dish0_time192 = Ē2_dish0_freq0_time192
            Ē2hi_dish0_time192 = Ē2_dish0_freq1_time192
            Ē2lo_dish4_time192 = Ē2_dish4_freq0_time192
            Ē2hi_dish4_time192 = Ē2_dish4_freq1_time192
            Ē2lo_dish0_time224 = Ē2_dish0_freq0_time224
            Ē2hi_dish0_time224 = Ē2_dish0_freq1_time224
            Ē2lo_dish4_time224 = Ē2_dish4_freq0_time224
            Ē2hi_dish4_time224 = Ē2_dish4_freq1_time224
            Ē3_dish0_time0 = Ē2lo_dish0_time0
            Ē3_dish8_time0 = Ē2hi_dish0_time0
            Ē3_dish4_time0 = Ē2lo_dish4_time0
            Ē3_dish12_time0 = Ē2hi_dish4_time0
            Ē3_dish0_time32 = Ē2lo_dish0_time32
            Ē3_dish8_time32 = Ē2hi_dish0_time32
            Ē3_dish4_time32 = Ē2lo_dish4_time32
            Ē3_dish12_time32 = Ē2hi_dish4_time32
            Ē3_dish0_time64 = Ē2lo_dish0_time64
            Ē3_dish8_time64 = Ē2hi_dish0_time64
            Ē3_dish4_time64 = Ē2lo_dish4_time64
            Ē3_dish12_time64 = Ē2hi_dish4_time64
            Ē3_dish0_time96 = Ē2lo_dish0_time96
            Ē3_dish8_time96 = Ē2hi_dish0_time96
            Ē3_dish4_time96 = Ē2lo_dish4_time96
            Ē3_dish12_time96 = Ē2hi_dish4_time96
            Ē3_dish0_time128 = Ē2lo_dish0_time128
            Ē3_dish8_time128 = Ē2hi_dish0_time128
            Ē3_dish4_time128 = Ē2lo_dish4_time128
            Ē3_dish12_time128 = Ē2hi_dish4_time128
            Ē3_dish0_time160 = Ē2lo_dish0_time160
            Ē3_dish8_time160 = Ē2hi_dish0_time160
            Ē3_dish4_time160 = Ē2lo_dish4_time160
            Ē3_dish12_time160 = Ē2hi_dish4_time160
            Ē3_dish0_time192 = Ē2lo_dish0_time192
            Ē3_dish8_time192 = Ē2hi_dish0_time192
            Ē3_dish4_time192 = Ē2lo_dish4_time192
            Ē3_dish12_time192 = Ē2hi_dish4_time192
            Ē3_dish0_time224 = Ē2lo_dish0_time224
            Ē3_dish8_time224 = Ē2hi_dish0_time224
            Ē3_dish4_time224 = Ē2lo_dish4_time224
            Ē3_dish12_time224 = Ē2hi_dish4_time224
            if ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0i32, 8) ÷ 2) % (2i32)) * 8 +
               ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0i32, 8) ÷ 4) % (2i32)) * 16 +
               ((0 ÷ 32) % (2i32)) * 32 ≥ 24
                IndexSpaces.unsafe_store4_global!(
                    Ē_memory,
                    let
                        offset = 16384 * T̄min - 49152
                        length = 67108864
                        mod(
                            (
                                (
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 +
                                    (
                                        (
                                            (
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8 +
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4
                                            ) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4
                                        ) % 512
                                    ) * 32 +
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 +
                                    (
                                        (
                                            (
                                                ((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 +
                                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8
                                            ) ÷ 8
                                        ) % 4096
                                    ) * 16384
                                ) + 0
                            ) + offset,
                            length,
                        )
                    end + 0x01,
                    (Ē3_dish0_time0, Ē3_dish4_time0, Ē3_dish8_time0, Ē3_dish12_time0),
                )
            end
            if ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0i32, 8) ÷ 2) % (2i32)) * 8 +
               ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0i32, 8) ÷ 4) % (2i32)) * 16 +
               ((32 ÷ 32) % (2i32)) * 32 ≥ 24
                IndexSpaces.unsafe_store4_global!(
                    Ē_memory,
                    let
                        offset = 16384 * T̄min - 49152
                        length = 67108864
                        mod(
                            (
                                (
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 +
                                    (
                                        (
                                            (
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8 +
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4
                                            ) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4
                                        ) % 512
                                    ) * 32 +
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 +
                                    (
                                        (
                                            (
                                                (
                                                    ((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 +
                                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8
                                                ) + 32
                                            ) ÷ 8
                                        ) % 4096
                                    ) * 16384
                                ) + 0
                            ) + offset,
                            length,
                        )
                    end + 0x01,
                    (Ē3_dish0_time32, Ē3_dish4_time32, Ē3_dish8_time32, Ē3_dish12_time32),
                )
            end
            if ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0i32, 8) ÷ 2) % (2i32)) * 8 +
               ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0i32, 8) ÷ 4) % (2i32)) * 16 +
               ((64 ÷ 32) % (2i32)) * 32 ≥ 24
                IndexSpaces.unsafe_store4_global!(
                    Ē_memory,
                    let
                        offset = 16384 * T̄min - 49152
                        length = 67108864
                        mod(
                            (
                                (
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 +
                                    (
                                        (
                                            (
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8 +
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4
                                            ) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4
                                        ) % 512
                                    ) * 32 +
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 +
                                    (
                                        (
                                            (
                                                (
                                                    ((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 +
                                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8
                                                ) + 64
                                            ) ÷ 8
                                        ) % 4096
                                    ) * 16384
                                ) + 0
                            ) + offset,
                            length,
                        )
                    end + 0x01,
                    (Ē3_dish0_time64, Ē3_dish4_time64, Ē3_dish8_time64, Ē3_dish12_time64),
                )
            end
            if ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0i32, 8) ÷ 2) % (2i32)) * 8 +
               ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0i32, 8) ÷ 4) % (2i32)) * 16 +
               ((96 ÷ 32) % (2i32)) * 32 ≥ 24
                IndexSpaces.unsafe_store4_global!(
                    Ē_memory,
                    let
                        offset = 16384 * T̄min - 49152
                        length = 67108864
                        mod(
                            (
                                (
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 +
                                    (
                                        (
                                            (
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8 +
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4
                                            ) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4
                                        ) % 512
                                    ) * 32 +
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 +
                                    (
                                        (
                                            (
                                                (
                                                    ((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 +
                                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8
                                                ) + 96
                                            ) ÷ 8
                                        ) % 4096
                                    ) * 16384
                                ) + 0
                            ) + offset,
                            length,
                        )
                    end + 0x01,
                    (Ē3_dish0_time96, Ē3_dish4_time96, Ē3_dish8_time96, Ē3_dish12_time96),
                )
            end
            if ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0i32, 8) ÷ 2) % (2i32)) * 8 +
               ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0i32, 8) ÷ 4) % (2i32)) * 16 +
               ((128 ÷ 32) % (2i32)) * 32 ≥ 24
                IndexSpaces.unsafe_store4_global!(
                    Ē_memory,
                    let
                        offset = 16384 * T̄min - 49152
                        length = 67108864
                        mod(
                            (
                                (
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 +
                                    (
                                        (
                                            (
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8 +
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4
                                            ) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4
                                        ) % 512
                                    ) * 32 +
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 +
                                    (
                                        (
                                            (
                                                (
                                                    ((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 +
                                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8
                                                ) + 128
                                            ) ÷ 8
                                        ) % 4096
                                    ) * 16384
                                ) + 0
                            ) + offset,
                            length,
                        )
                    end + 0x01,
                    (Ē3_dish0_time128, Ē3_dish4_time128, Ē3_dish8_time128, Ē3_dish12_time128),
                )
            end
            if ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0i32, 8) ÷ 2) % (2i32)) * 8 +
               ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0i32, 8) ÷ 4) % (2i32)) * 16 +
               ((160 ÷ 32) % (2i32)) * 32 ≥ 24
                IndexSpaces.unsafe_store4_global!(
                    Ē_memory,
                    let
                        offset = 16384 * T̄min - 49152
                        length = 67108864
                        mod(
                            (
                                (
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 +
                                    (
                                        (
                                            (
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8 +
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4
                                            ) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4
                                        ) % 512
                                    ) * 32 +
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 +
                                    (
                                        (
                                            (
                                                (
                                                    ((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 +
                                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8
                                                ) + 160
                                            ) ÷ 8
                                        ) % 4096
                                    ) * 16384
                                ) + 0
                            ) + offset,
                            length,
                        )
                    end + 0x01,
                    (Ē3_dish0_time160, Ē3_dish4_time160, Ē3_dish8_time160, Ē3_dish12_time160),
                )
            end
            if ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0i32, 8) ÷ 2) % (2i32)) * 8 +
               ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0i32, 8) ÷ 4) % (2i32)) * 16 +
               ((192 ÷ 32) % (2i32)) * 32 ≥ 24
                IndexSpaces.unsafe_store4_global!(
                    Ē_memory,
                    let
                        offset = 16384 * T̄min - 49152
                        length = 67108864
                        mod(
                            (
                                (
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 +
                                    (
                                        (
                                            (
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8 +
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4
                                            ) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4
                                        ) % 512
                                    ) * 32 +
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 +
                                    (
                                        (
                                            (
                                                (
                                                    ((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 +
                                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8
                                                ) + 192
                                            ) ÷ 8
                                        ) % 4096
                                    ) * 16384
                                ) + 0
                            ) + offset,
                            length,
                        )
                    end + 0x01,
                    (Ē3_dish0_time192, Ē3_dish4_time192, Ē3_dish8_time192, Ē3_dish12_time192),
                )
            end
            if ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0i32, 8) ÷ 2) % (2i32)) * 8 +
               ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0i32, 8) ÷ 4) % (2i32)) * 16 +
               ((224 ÷ 32) % (2i32)) * 32 ≥ 24
                IndexSpaces.unsafe_store4_global!(
                    Ē_memory,
                    let
                        offset = 16384 * T̄min - 49152
                        length = 67108864
                        mod(
                            (
                                (
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 +
                                    (
                                        (
                                            (
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) * 8 +
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4
                                            ) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4
                                        ) % 512
                                    ) * 32 +
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 +
                                    (
                                        (
                                            (
                                                (
                                                    ((IndexSpaces.assume_inrange(t_outer, 0, 256, 32768) ÷ 256) % 128) * 256 +
                                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 4) * 8
                                                ) + 224
                                            ) ÷ 8
                                        ) % 4096
                                    ) * 16384
                                ) + 0
                            ) + offset,
                            length,
                        )
                    end + 0x01,
                    (Ē3_dish0_time224, Ē3_dish4_time224, Ē3_dish8_time224, Ē3_dish12_time224),
                )
            end
        end
        info = 0
        if true
            info_memory[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 384) % 384) % 384) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 8) % 8) * 32) + 0) + 0x01] =
                info
        end
    end
)
