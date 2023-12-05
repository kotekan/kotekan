@fastmath @inbounds(
    begin #= /home/eschnett/src/kotekan/julia/kernels/upchan.jl:1416 =#
        info = 1
        if true
            info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) % 128) * 256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 8) % 8) * 32) + 0) + 0x01] =
                info
        end
        Tmin = Tmin_memory[0 + 0x01]
        Tmax = Tmax_memory[0 + 0x01]
        T̄min = T̄min_memory[0 + 0x01]
        T̄max = T̄max_memory[0 + 0x01]
        if !(
            0i32 ≤ Tmin ≤ Tmax ≤ 262144 &&
            ((Tmax - Tmin) % 256 == 0i32 && (0i32 ≤ T̄min ≤ T̄max ≤ 32768 && ((T̄max - T̄min) + 3) % 32 == 0i32))
        )
            info = 2
            if true
                info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) % 128) * 256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 8) % 8) * 32) + 0) + 0x01] =
                    info
            end
            IndexSpaces.cuda_trap()
        end
        F_ringbuf_mtaps0 = zero(Int4x8)
        F_ringbuf_mtaps1 = zero(Int4x8)
        F_ringbuf_mtaps2 = zero(Int4x8)
        Gains = G_memory[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 2) * 2) ÷ 2) % 64 + 0x01]
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
            X0 = cispi(((time0 * 7) / Float32(U)) % 2.0f0)
            X1 = cispi(((time1 * 7) / Float32(U)) % 2.0f0)
            (X0, X1)
        end
        Xre = Float16x2(real(X0), real(X1))
        Xim = Float16x2(imag(X0), imag(X1))
        X_cplx0 = Xre
        X_cplx1 = Xim
        (Γ¹0, Γ¹1) = let
            k = Ubits
            @assert U == 2^k                    #= /home/eschnett/src/kotekan/julia/kernels/upchan.jl:642 =#
            m = 3
            n = k - m
            @assert 0 ≤ m                    #= /home/eschnett/src/kotekan/julia/kernels/upchan.jl:645 =#
            @assert 0 ≤ n                    #= /home/eschnett/src/kotekan/julia/kernels/upchan.jl:646 =#
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            thread0 = (thread ÷ (1i32)) % (2i32)
            thread1 = (thread ÷ (2i32)) % (2i32)
            thread2 = (thread ÷ (4i32)) % (2i32)
            thread3 = (thread ÷ (8i32)) % (2i32)
            thread4 = (thread ÷ (16i32)) % (2i32)
            timehi0 = (1i32) * thread0 + (2i32) * thread1
            timehi1 = timehi0 + 4i32
            freqlo = (1i32) * thread2 + (2i32) * thread3 + (4i32) * thread4
            (Γ¹0, Γ¹1) = (
                cispi((((-2i32) * timehi0 * freqlo) / Float32(2^m)) % 2.0f0),
                cispi((((-2i32) * timehi1 * freqlo) / Float32(2^m)) % 2.0f0),
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
            k = Ubits
            @assert U == 2^k                    #= /home/eschnett/src/kotekan/julia/kernels/upchan.jl:766 =#
            m = 3
            n = k - m
            @assert 0 ≤ m                    #= /home/eschnett/src/kotekan/julia/kernels/upchan.jl:769 =#
            @assert 0 ≤ n                    #= /home/eschnett/src/kotekan/julia/kernels/upchan.jl:770 =#
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            thread0 = (thread ÷ (1i32)) % (2i32)
            thread1 = (thread ÷ (2i32)) % (2i32)
            thread2 = (thread ÷ (4i32)) % (2i32)
            thread3 = (thread ÷ (8i32)) % (2i32)
            thread4 = (thread ÷ (16i32)) % (2i32)
            if U == 8
                timelo0 = 0i32
                timelo1 = timelo0
            elseif U == 16
                timelo0 = 0i32
                timelo1 = timelo0 + 1i32
            elseif U == 32
                timelo0 = (1i32) * thread1
                timelo1 = timelo0 + 2i32
            elseif U == 64
                timelo0 = (2i32) * thread1 + (1i32) * thread0
                timelo1 = timelo0 + 4i32
            elseif U == 128
                timelo0 = (4i32) * thread1 + (2i32) * thread0
                timelo1 = timelo0 + 4i32
            else
                @assert false                        #= /home/eschnett/src/kotekan/julia/kernels/upchan.jl:793 =#
            end
            if U == 8
                freqhi = 0i32
                dish_in = (1i32) * thread1 + (2i32) * thread0 + (4i32) * thread2
                dish = (1i32) * thread2 + (2i32) * thread4 + (4i32) * thread3
            elseif U == 16
                freqhi = (1i32) * thread2
                dish_in = (1i32) * thread1 + (2i32) * thread0
                dish = (1i32) * thread4 + (2i32) * thread3
            elseif U == 32
                freqhi = (1i32) * thread2 + (2i32) * thread4
                dish_in = (1i32) * thread0
                dish = (1i32) * thread3
            elseif U == 64
                freqhi = (1i32) * thread2 + (2i32) * thread4 + (4i32) * thread3
                dish_in = 0i32
                dish = 0i32
            elseif U == 128
                freqhi = (1i32) * thread2 + (2i32) * thread4 + (4i32) * thread3
                dish_in = 0i32
                dish = 0i32
            else
                @assert false                        #= /home/eschnett/src/kotekan/julia/kernels/upchan.jl:816 =#
            end
            delta = dish == dish_in
            (Γ³0, Γ³1) = (
                delta * cispi((((-2i32) * timelo0 * freqhi) / Float32(2^n)) % 2.0f0),
                delta * cispi((((-2i32) * timelo1 * freqhi) / Float32(2^n)) % 2.0f0),
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
        for t_outer in 0:256:131071
            Tmin + t_outer ≥ Tmax && break
            (E_dish0_time0, E_dish4_time0, E_dish8_time0, E_dish12_time0) = IndexSpaces.unsafe_load4_global(
                E_memory,
                let
                    offset = 4096 * Tmin
                    length = 536870912
                    mod(
                        (
                            (
                                (
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16 +
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128
                                ) ÷ 4
                            ) % 128 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 4) % 2) % 2) * 128 +
                            (
                                (
                                    (
                                        (
                                            (
                                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8 +
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2
                                            ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2
                                        ) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256
                                    ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 4
                                ) % 131072
                            ) * 4096 +
                            ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 8) % 16) * 8) % 16) * 256
                        ) + offset,
                        length,
                    )
                end + 1i32,
            )
            (E_dish0_time64, E_dish4_time64, E_dish8_time64, E_dish12_time64) = IndexSpaces.unsafe_load4_global(
                E_memory,
                let
                    offset = 4096 * Tmin
                    length = 536870912
                    mod(
                        (
                            (
                                (
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16 +
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128
                                ) ÷ 4
                            ) % 128 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 4) % 2) % 2) * 128 +
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    64 +
                                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8
                                                ) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2
                                            ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2
                                        ) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256
                                    ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 4
                                ) % 131072
                            ) * 4096 +
                            ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 8) % 16) * 8) % 16) * 256
                        ) + offset,
                        length,
                    )
                end + 1i32,
            )
            (E_dish0_time128, E_dish4_time128, E_dish8_time128, E_dish12_time128) = IndexSpaces.unsafe_load4_global(
                E_memory,
                let
                    offset = 4096 * Tmin
                    length = 536870912
                    mod(
                        (
                            (
                                (
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16 +
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128
                                ) ÷ 4
                            ) % 128 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 4) % 2) % 2) * 128 +
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    128 +
                                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8
                                                ) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2
                                            ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2
                                        ) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256
                                    ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 4
                                ) % 131072
                            ) * 4096 +
                            ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 8) % 16) * 8) % 16) * 256
                        ) + offset,
                        length,
                    )
                end + 1i32,
            )
            (E_dish0_time192, E_dish4_time192, E_dish8_time192, E_dish12_time192) = IndexSpaces.unsafe_load4_global(
                E_memory,
                let
                    offset = 4096 * Tmin
                    length = 536870912
                    mod(
                        (
                            (
                                (
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16 +
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128
                                ) ÷ 4
                            ) % 128 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 4) % 2) % 2) * 128 +
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    192 +
                                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8
                                                ) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2
                                            ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2
                                        ) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256
                                    ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 4
                                ) % 131072
                            ) * 4096 +
                            ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 8) % 16) * 8) % 16) * 256
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
            E1lo_dish0_time0 = E1_dish0_time0
            E1hi_dish0_time0 = E1_dish8_time0
            E1lo_dish4_time0 = E1_dish4_time0
            E1hi_dish4_time0 = E1_dish12_time0
            E1lo_dish0_time64 = E1_dish0_time64
            E1hi_dish0_time64 = E1_dish8_time64
            E1lo_dish4_time64 = E1_dish4_time64
            E1hi_dish4_time64 = E1_dish12_time64
            E1lo_dish0_time128 = E1_dish0_time128
            E1hi_dish0_time128 = E1_dish8_time128
            E1lo_dish4_time128 = E1_dish4_time128
            E1hi_dish4_time128 = E1_dish12_time128
            E1lo_dish0_time192 = E1_dish0_time192
            E1hi_dish0_time192 = E1_dish8_time192
            E1lo_dish4_time192 = E1_dish4_time192
            E1hi_dish4_time192 = E1_dish12_time192
            E1_dish0_time0 = E1lo_dish0_time0
            E1_dish0_time4 = E1hi_dish0_time0
            E1_dish4_time0 = E1lo_dish4_time0
            E1_dish4_time4 = E1hi_dish4_time0
            E1_dish0_time64 = E1lo_dish0_time64
            E1_dish0_time68 = E1hi_dish0_time64
            E1_dish4_time64 = E1lo_dish4_time64
            E1_dish4_time68 = E1hi_dish4_time64
            E1_dish0_time128 = E1lo_dish0_time128
            E1_dish0_time132 = E1hi_dish0_time128
            E1_dish4_time128 = E1lo_dish4_time128
            E1_dish4_time132 = E1hi_dish4_time128
            E1_dish0_time192 = E1lo_dish0_time192
            E1_dish0_time196 = E1hi_dish0_time192
            E1_dish4_time192 = E1lo_dish4_time192
            E1_dish4_time196 = E1hi_dish4_time192
            (E2_dish0_time0, E2_dish0_time4) = (
                IndexSpaces.get_lo16(E1_dish0_time0, E1_dish0_time4), IndexSpaces.get_hi16(E1_dish0_time0, E1_dish0_time4)
            )
            (E2_dish4_time0, E2_dish4_time4) = (
                IndexSpaces.get_lo16(E1_dish4_time0, E1_dish4_time4), IndexSpaces.get_hi16(E1_dish4_time0, E1_dish4_time4)
            )
            (E2_dish0_time64, E2_dish0_time68) = (
                IndexSpaces.get_lo16(E1_dish0_time64, E1_dish0_time68), IndexSpaces.get_hi16(E1_dish0_time64, E1_dish0_time68)
            )
            (E2_dish4_time64, E2_dish4_time68) = (
                IndexSpaces.get_lo16(E1_dish4_time64, E1_dish4_time68), IndexSpaces.get_hi16(E1_dish4_time64, E1_dish4_time68)
            )
            (E2_dish0_time128, E2_dish0_time132) = (
                IndexSpaces.get_lo16(E1_dish0_time128, E1_dish0_time132), IndexSpaces.get_hi16(E1_dish0_time128, E1_dish0_time132)
            )
            (E2_dish4_time128, E2_dish4_time132) = (
                IndexSpaces.get_lo16(E1_dish4_time128, E1_dish4_time132), IndexSpaces.get_hi16(E1_dish4_time128, E1_dish4_time132)
            )
            (E2_dish0_time192, E2_dish0_time196) = (
                IndexSpaces.get_lo16(E1_dish0_time192, E1_dish0_time196), IndexSpaces.get_hi16(E1_dish0_time192, E1_dish0_time196)
            )
            (E2_dish4_time192, E2_dish4_time196) = (
                IndexSpaces.get_lo16(E1_dish4_time192, E1_dish4_time196), IndexSpaces.get_hi16(E1_dish4_time192, E1_dish4_time196)
            )
            E2lo_dish0_time0 = E2_dish0_time0
            E2hi_dish0_time0 = E2_dish0_time4
            E2lo_dish4_time0 = E2_dish4_time0
            E2hi_dish4_time0 = E2_dish4_time4
            E2lo_dish0_time64 = E2_dish0_time64
            E2hi_dish0_time64 = E2_dish0_time68
            E2lo_dish4_time64 = E2_dish4_time64
            E2hi_dish4_time64 = E2_dish4_time68
            E2lo_dish0_time128 = E2_dish0_time128
            E2hi_dish0_time128 = E2_dish0_time132
            E2lo_dish4_time128 = E2_dish4_time128
            E2hi_dish4_time128 = E2_dish4_time132
            E2lo_dish0_time192 = E2_dish0_time192
            E2hi_dish0_time192 = E2_dish0_time196
            E2lo_dish4_time192 = E2_dish4_time192
            E2hi_dish4_time192 = E2_dish4_time196
            E2_dish0_time0 = E2lo_dish0_time0
            E2_dish2_time0 = E2hi_dish0_time0
            E2_dish4_time0 = E2lo_dish4_time0
            E2_dish6_time0 = E2hi_dish4_time0
            E2_dish0_time64 = E2lo_dish0_time64
            E2_dish2_time64 = E2hi_dish0_time64
            E2_dish4_time64 = E2lo_dish4_time64
            E2_dish6_time64 = E2hi_dish4_time64
            E2_dish0_time128 = E2lo_dish0_time128
            E2_dish2_time128 = E2hi_dish0_time128
            E2_dish4_time128 = E2lo_dish4_time128
            E2_dish6_time128 = E2hi_dish4_time128
            E2_dish0_time192 = E2lo_dish0_time192
            E2_dish2_time192 = E2hi_dish0_time192
            E2_dish4_time192 = E2lo_dish4_time192
            E2_dish6_time192 = E2hi_dish4_time192
            F_dish0_time0 = E2_dish0_time0
            F_dish2_time0 = E2_dish2_time0
            F_dish4_time0 = E2_dish4_time0
            F_dish6_time0 = E2_dish6_time0
            F_dish0_time64 = E2_dish0_time64
            F_dish2_time64 = E2_dish2_time64
            F_dish4_time64 = E2_dish4_time64
            F_dish6_time64 = E2_dish6_time64
            F_dish0_time128 = E2_dish0_time128
            F_dish2_time128 = E2_dish2_time128
            F_dish4_time128 = E2_dish4_time128
            F_dish6_time128 = E2_dish6_time128
            F_dish0_time192 = E2_dish0_time192
            F_dish2_time192 = E2_dish2_time192
            F_dish4_time192 = E2_dish4_time192
            F_dish6_time192 = E2_dish6_time192
            if true
                F_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) ÷ 2) % 2) * 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) % 2) * 130 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 2) % 2) * 65 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) ÷ 4) % 32) + 0) + 0x01] =
                    F_dish0_time0
            end
            if true
                F_shared[(((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 2) ÷ 2) % 2) * 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) % 2) * 130 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 2) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 2) ÷ 4) % 32) + 0) + 0x01] =
                    F_dish2_time0
            end
            if true
                F_shared[(((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 4) ÷ 2) % 2) * 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) % 2) * 130 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 2) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 4) ÷ 4) % 32) + 0) + 0x01] =
                    F_dish4_time0
            end
            if true
                F_shared[(((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 6) ÷ 2) % 2) * 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) % 2) * 130 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 2) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 6) ÷ 4) % 32) + 0) + 0x01] =
                    F_dish6_time0
            end
            if true
                F_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) ÷ 2) % 2) * 32 + ((((((64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + (((((64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) % 2) * 130 + ((((((64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 2) % 2) * 65 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) ÷ 4) % 32) + 0) + 0x01] =
                    F_dish0_time64
            end
            if true
                F_shared[(((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 2) ÷ 2) % 2) * 32 + ((((((64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + (((((64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) % 2) * 130 + ((((((64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 2) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 2) ÷ 4) % 32) + 0) + 0x01] =
                    F_dish2_time64
            end
            if true
                F_shared[(((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 4) ÷ 2) % 2) * 32 + ((((((64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + (((((64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) % 2) * 130 + ((((((64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 2) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 4) ÷ 4) % 32) + 0) + 0x01] =
                    F_dish4_time64
            end
            if true
                F_shared[(((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 6) ÷ 2) % 2) * 32 + ((((((64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + (((((64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) % 2) * 130 + ((((((64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 2) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 6) ÷ 4) % 32) + 0) + 0x01] =
                    F_dish6_time64
            end
            if true
                F_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) ÷ 2) % 2) * 32 + ((((((128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + (((((128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) % 2) * 130 + ((((((128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 2) % 2) * 65 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) ÷ 4) % 32) + 0) + 0x01] =
                    F_dish0_time128
            end
            if true
                F_shared[(((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 2) ÷ 2) % 2) * 32 + ((((((128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + (((((128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) % 2) * 130 + ((((((128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 2) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 2) ÷ 4) % 32) + 0) + 0x01] =
                    F_dish2_time128
            end
            if true
                F_shared[(((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 4) ÷ 2) % 2) * 32 + ((((((128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + (((((128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) % 2) * 130 + ((((((128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 2) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 4) ÷ 4) % 32) + 0) + 0x01] =
                    F_dish4_time128
            end
            if true
                F_shared[(((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 6) ÷ 2) % 2) * 32 + ((((((128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + (((((128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) % 2) * 130 + ((((((128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 2) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 6) ÷ 4) % 32) + 0) + 0x01] =
                    F_dish6_time128
            end
            if true
                F_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) ÷ 2) % 2) * 32 + ((((((192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + (((((192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) % 2) * 130 + ((((((192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 2) % 2) * 65 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) ÷ 4) % 32) + 0) + 0x01] =
                    F_dish0_time192
            end
            if true
                F_shared[(((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 2) ÷ 2) % 2) * 32 + ((((((192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + (((((192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) % 2) * 130 + ((((((192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 2) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 2) ÷ 4) % 32) + 0) + 0x01] =
                    F_dish2_time192
            end
            if true
                F_shared[(((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 4) ÷ 2) % 2) * 32 + ((((((192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + (((((192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) % 2) * 130 + ((((((192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 2) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 4) ÷ 4) % 32) + 0) + 0x01] =
                    F_dish4_time192
            end
            if true
                F_shared[(((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 6) ÷ 2) % 2) * 32 + ((((((192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + (((((192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) % 2) * 130 + ((((((192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 2) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 6) ÷ 4) % 32) + 0) + 0x01] =
                    F_dish6_time192
            end
            IndexSpaces.cuda_sync_threads()
            for t_inner in 0:8:255
                let
                    dish = 0
                    F_in = F_shared[(((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 32) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 8) * 2) ÷ 2) % 2) * 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4 + ((IndexSpaces.assume_inrange(t_inner, 0, 8, 256) ÷ 8) % 32) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4 + ((IndexSpaces.assume_inrange(t_inner, 0, 8, 256) ÷ 8) % 32) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) % 2) * 130 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4 + ((IndexSpaces.assume_inrange(t_inner, 0, 8, 256) ÷ 8) % 32) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 2) % 2) * 65 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 32) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 8) * 2) ÷ 4) % 32) + 0x01]
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
                        F̄_shared[((((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 32) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 8) * 2) ÷ 2) % 2) * 32 + (((((IndexSpaces.assume_inrange(t_inner, 0, 8, 256) ÷ 8) % 32) * 8 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 8) % 16) * 8) ÷ 2) % 4) * 65 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 32) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 8) * 2) ÷ 4) % 32) + 0) + 0x01] =
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
            Ē_dish0_time0 = F̄_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) ÷ 2) % 2) * 32 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 8) % 16) * 8) ÷ 2) % 4) * 65 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) ÷ 4) % 32) + 0x01]
            Ē_dish2_time0 = F̄_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 2) ÷ 2) % 2) * 32 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 8) % 16) * 8) ÷ 2) % 4) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 2) ÷ 4) % 32) + 0x01]
            Ē_dish4_time0 = F̄_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 4) ÷ 2) % 2) * 32 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 8) % 16) * 8) ÷ 2) % 4) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 4) ÷ 4) % 32) + 0x01]
            Ē_dish6_time0 = F̄_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 6) ÷ 2) % 2) * 32 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 8) % 16) * 8) ÷ 2) % 4) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 6) ÷ 4) % 32) + 0x01]
            Ē_dish0_time64 = F̄_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) ÷ 2) % 2) * 32 + ((((64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 8) % 16) * 8) ÷ 2) % 4) * 65 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) ÷ 4) % 32) + 0x01]
            Ē_dish2_time64 = F̄_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 2) ÷ 2) % 2) * 32 + ((((64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 8) % 16) * 8) ÷ 2) % 4) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 2) ÷ 4) % 32) + 0x01]
            Ē_dish4_time64 = F̄_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 4) ÷ 2) % 2) * 32 + ((((64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 8) % 16) * 8) ÷ 2) % 4) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 4) ÷ 4) % 32) + 0x01]
            Ē_dish6_time64 = F̄_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 6) ÷ 2) % 2) * 32 + ((((64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 8) % 16) * 8) ÷ 2) % 4) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 6) ÷ 4) % 32) + 0x01]
            Ē_dish0_time128 = F̄_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) ÷ 2) % 2) * 32 + ((((128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 8) % 16) * 8) ÷ 2) % 4) * 65 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) ÷ 4) % 32) + 0x01]
            Ē_dish2_time128 = F̄_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 2) ÷ 2) % 2) * 32 + ((((128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 8) % 16) * 8) ÷ 2) % 4) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 2) ÷ 4) % 32) + 0x01]
            Ē_dish4_time128 = F̄_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 4) ÷ 2) % 2) * 32 + ((((128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 8) % 16) * 8) ÷ 2) % 4) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 4) ÷ 4) % 32) + 0x01]
            Ē_dish6_time128 = F̄_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 6) ÷ 2) % 2) * 32 + ((((128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 8) % 16) * 8) ÷ 2) % 4) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 6) ÷ 4) % 32) + 0x01]
            Ē_dish0_time192 = F̄_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) ÷ 2) % 2) * 32 + ((((192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 8) % 16) * 8) ÷ 2) % 4) * 65 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) ÷ 4) % 32) + 0x01]
            Ē_dish2_time192 = F̄_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 2) ÷ 2) % 2) * 32 + ((((192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 8) % 16) * 8) ÷ 2) % 4) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 2) ÷ 4) % 32) + 0x01]
            Ē_dish4_time192 = F̄_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 4) ÷ 2) % 2) * 32 + ((((192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 8) % 16) * 8) ÷ 2) % 4) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 4) ÷ 4) % 32) + 0x01]
            Ē_dish6_time192 = F̄_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 6) ÷ 2) % 2) * 32 + ((((192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 32) * 289 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 8) % 16) * 8) ÷ 2) % 4) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128) + 6) ÷ 4) % 32) + 0x01]
            (Ē1_dish0_time0, Ē1_dish2_time0) = (
                IndexSpaces.get_lo16(Ē_dish0_time0, Ē_dish2_time0), IndexSpaces.get_hi16(Ē_dish0_time0, Ē_dish2_time0)
            )
            (Ē1_dish4_time0, Ē1_dish6_time0) = (
                IndexSpaces.get_lo16(Ē_dish4_time0, Ē_dish6_time0), IndexSpaces.get_hi16(Ē_dish4_time0, Ē_dish6_time0)
            )
            (Ē1_dish0_time64, Ē1_dish2_time64) = (
                IndexSpaces.get_lo16(Ē_dish0_time64, Ē_dish2_time64), IndexSpaces.get_hi16(Ē_dish0_time64, Ē_dish2_time64)
            )
            (Ē1_dish4_time64, Ē1_dish6_time64) = (
                IndexSpaces.get_lo16(Ē_dish4_time64, Ē_dish6_time64), IndexSpaces.get_hi16(Ē_dish4_time64, Ē_dish6_time64)
            )
            (Ē1_dish0_time128, Ē1_dish2_time128) = (
                IndexSpaces.get_lo16(Ē_dish0_time128, Ē_dish2_time128), IndexSpaces.get_hi16(Ē_dish0_time128, Ē_dish2_time128)
            )
            (Ē1_dish4_time128, Ē1_dish6_time128) = (
                IndexSpaces.get_lo16(Ē_dish4_time128, Ē_dish6_time128), IndexSpaces.get_hi16(Ē_dish4_time128, Ē_dish6_time128)
            )
            (Ē1_dish0_time192, Ē1_dish2_time192) = (
                IndexSpaces.get_lo16(Ē_dish0_time192, Ē_dish2_time192), IndexSpaces.get_hi16(Ē_dish0_time192, Ē_dish2_time192)
            )
            (Ē1_dish4_time192, Ē1_dish6_time192) = (
                IndexSpaces.get_lo16(Ē_dish4_time192, Ē_dish6_time192), IndexSpaces.get_hi16(Ē_dish4_time192, Ē_dish6_time192)
            )
            Ē1lo_dish0_time0 = Ē1_dish0_time0
            Ē1hi_dish0_time0 = Ē1_dish2_time0
            Ē1lo_dish4_time0 = Ē1_dish4_time0
            Ē1hi_dish4_time0 = Ē1_dish6_time0
            Ē1lo_dish0_time64 = Ē1_dish0_time64
            Ē1hi_dish0_time64 = Ē1_dish2_time64
            Ē1lo_dish4_time64 = Ē1_dish4_time64
            Ē1hi_dish4_time64 = Ē1_dish6_time64
            Ē1lo_dish0_time128 = Ē1_dish0_time128
            Ē1hi_dish0_time128 = Ē1_dish2_time128
            Ē1lo_dish4_time128 = Ē1_dish4_time128
            Ē1hi_dish4_time128 = Ē1_dish6_time128
            Ē1lo_dish0_time192 = Ē1_dish0_time192
            Ē1hi_dish0_time192 = Ē1_dish2_time192
            Ē1lo_dish4_time192 = Ē1_dish4_time192
            Ē1hi_dish4_time192 = Ē1_dish6_time192
            Ē1_dish0_freq0_time0 = Ē1lo_dish0_time0
            Ē1_dish0_freq1_time0 = Ē1hi_dish0_time0
            Ē1_dish4_freq0_time0 = Ē1lo_dish4_time0
            Ē1_dish4_freq1_time0 = Ē1hi_dish4_time0
            Ē1_dish0_freq0_time64 = Ē1lo_dish0_time64
            Ē1_dish0_freq1_time64 = Ē1hi_dish0_time64
            Ē1_dish4_freq0_time64 = Ē1lo_dish4_time64
            Ē1_dish4_freq1_time64 = Ē1hi_dish4_time64
            Ē1_dish0_freq0_time128 = Ē1lo_dish0_time128
            Ē1_dish0_freq1_time128 = Ē1hi_dish0_time128
            Ē1_dish4_freq0_time128 = Ē1lo_dish4_time128
            Ē1_dish4_freq1_time128 = Ē1hi_dish4_time128
            Ē1_dish0_freq0_time192 = Ē1lo_dish0_time192
            Ē1_dish0_freq1_time192 = Ē1hi_dish0_time192
            Ē1_dish4_freq0_time192 = Ē1lo_dish4_time192
            Ē1_dish4_freq1_time192 = Ē1hi_dish4_time192
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
            Ē2lo_dish0_time0 = Ē2_dish0_freq0_time0
            Ē2hi_dish0_time0 = Ē2_dish0_freq1_time0
            Ē2lo_dish4_time0 = Ē2_dish4_freq0_time0
            Ē2hi_dish4_time0 = Ē2_dish4_freq1_time0
            Ē2lo_dish0_time64 = Ē2_dish0_freq0_time64
            Ē2hi_dish0_time64 = Ē2_dish0_freq1_time64
            Ē2lo_dish4_time64 = Ē2_dish4_freq0_time64
            Ē2hi_dish4_time64 = Ē2_dish4_freq1_time64
            Ē2lo_dish0_time128 = Ē2_dish0_freq0_time128
            Ē2hi_dish0_time128 = Ē2_dish0_freq1_time128
            Ē2lo_dish4_time128 = Ē2_dish4_freq0_time128
            Ē2hi_dish4_time128 = Ē2_dish4_freq1_time128
            Ē2lo_dish0_time192 = Ē2_dish0_freq0_time192
            Ē2hi_dish0_time192 = Ē2_dish0_freq1_time192
            Ē2lo_dish4_time192 = Ē2_dish4_freq0_time192
            Ē2hi_dish4_time192 = Ē2_dish4_freq1_time192
            Ē3_dish0_time0 = Ē2lo_dish0_time0
            Ē3_dish8_time0 = Ē2hi_dish0_time0
            Ē3_dish4_time0 = Ē2lo_dish4_time0
            Ē3_dish12_time0 = Ē2hi_dish4_time0
            Ē3_dish0_time64 = Ē2lo_dish0_time64
            Ē3_dish8_time64 = Ē2hi_dish0_time64
            Ē3_dish4_time64 = Ē2lo_dish4_time64
            Ē3_dish12_time64 = Ē2hi_dish4_time64
            Ē3_dish0_time128 = Ē2lo_dish0_time128
            Ē3_dish8_time128 = Ē2hi_dish0_time128
            Ē3_dish4_time128 = Ē2lo_dish4_time128
            Ē3_dish12_time128 = Ē2hi_dish4_time128
            Ē3_dish0_time192 = Ē2lo_dish0_time192
            Ē3_dish8_time192 = Ē2hi_dish0_time192
            Ē3_dish4_time192 = Ē2lo_dish4_time192
            Ē3_dish12_time192 = Ē2hi_dish4_time192
            if ((0 + ((IndexSpaces.cuda_warpidx() ÷ 2) % (2i32)) * 8) + ((IndexSpaces.cuda_warpidx() ÷ 4) % (2i32)) * 16) +
               ((IndexSpaces.cuda_warpidx() ÷ 8) % (2i32)) * 32 ≥ 24
                IndexSpaces.unsafe_store4_global!(
                    Ē_memory,
                    let
                        offset = 32768 * T̄min - 98304
                        length = 536870912
                        mod(
                            (
                                (
                                    (
                                        (
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16 +
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128
                                        ) ÷ 4
                                    ) % 128 +
                                    (
                                        (
                                            (
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4 +
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4
                                            ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 8) % 16) * 8
                                        ) % 128
                                    ) * 256 +
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 4) % 2) % 2) * 128 +
                                    (
                                        (
                                            (
                                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8 +
                                                ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256
                                            ) ÷ 8
                                        ) % 16384
                                    ) * 32768
                                ) + 0
                            ) + offset,
                            length,
                        )
                    end + 0x01,
                    (Ē3_dish0_time0, Ē3_dish4_time0, Ē3_dish8_time0, Ē3_dish12_time0),
                )
            end
            if ((0 + ((IndexSpaces.cuda_warpidx() ÷ 2) % (2i32)) * 8) + ((IndexSpaces.cuda_warpidx() ÷ 4) % (2i32)) * 16) +
               ((IndexSpaces.cuda_warpidx() ÷ 8) % (2i32)) * 32 ≥ 24
                IndexSpaces.unsafe_store4_global!(
                    Ē_memory,
                    let
                        offset = 32768 * T̄min - 98304
                        length = 536870912
                        mod(
                            (
                                (
                                    (
                                        (
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16 +
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128
                                        ) ÷ 4
                                    ) % 128 +
                                    (
                                        (
                                            (
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4 +
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4
                                            ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 8) % 16) * 8
                                        ) % 128
                                    ) * 256 +
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 4) % 2) % 2) * 128 +
                                    (
                                        (
                                            (
                                                (
                                                    64 +
                                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8
                                                ) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256
                                            ) ÷ 8
                                        ) % 16384
                                    ) * 32768
                                ) + 0
                            ) + offset,
                            length,
                        )
                    end + 0x01,
                    (Ē3_dish0_time64, Ē3_dish4_time64, Ē3_dish8_time64, Ē3_dish12_time64),
                )
            end
            if ((0 + ((IndexSpaces.cuda_warpidx() ÷ 2) % (2i32)) * 8) + ((IndexSpaces.cuda_warpidx() ÷ 4) % (2i32)) * 16) +
               ((IndexSpaces.cuda_warpidx() ÷ 8) % (2i32)) * 32 ≥ 24
                IndexSpaces.unsafe_store4_global!(
                    Ē_memory,
                    let
                        offset = 32768 * T̄min - 98304
                        length = 536870912
                        mod(
                            (
                                (
                                    (
                                        (
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16 +
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128
                                        ) ÷ 4
                                    ) % 128 +
                                    (
                                        (
                                            (
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4 +
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4
                                            ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 8) % 16) * 8
                                        ) % 128
                                    ) * 256 +
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 4) % 2) % 2) * 128 +
                                    (
                                        (
                                            (
                                                (
                                                    128 +
                                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8
                                                ) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256
                                            ) ÷ 8
                                        ) % 16384
                                    ) * 32768
                                ) + 0
                            ) + offset,
                            length,
                        )
                    end + 0x01,
                    (Ē3_dish0_time128, Ē3_dish4_time128, Ē3_dish8_time128, Ē3_dish12_time128),
                )
            end
            if ((0 + ((IndexSpaces.cuda_warpidx() ÷ 2) % (2i32)) * 8) + ((IndexSpaces.cuda_warpidx() ÷ 4) % (2i32)) * 16) +
               ((IndexSpaces.cuda_warpidx() ÷ 8) % (2i32)) * 32 ≥ 24
                IndexSpaces.unsafe_store4_global!(
                    Ē_memory,
                    let
                        offset = 32768 * T̄min - 98304
                        length = 536870912
                        mod(
                            (
                                (
                                    (
                                        (
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16 +
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 4) * 128
                                        ) ÷ 4
                                    ) % 128 +
                                    (
                                        (
                                            (
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4 +
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 2) * 4
                                            ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 8) % 16) * 8
                                        ) % 128
                                    ) * 256 +
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) ÷ 4) % 2) % 2) * 128 +
                                    (
                                        (
                                            (
                                                (
                                                    192 +
                                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) ÷ 2) % 8) * 8
                                                ) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256
                                            ) ÷ 8
                                        ) % 16384
                                    ) * 32768
                                ) + 0
                            ) + offset,
                            length,
                        )
                    end + 0x01,
                    (Ē3_dish0_time192, Ē3_dish4_time192, Ē3_dish8_time192, Ē3_dish12_time192),
                )
            end
        end
        info = 0
        if true
            info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) % 128) * 256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 8) % 8) % 8) * 32) + 0) + 0x01] =
                info
        end
    end
)
