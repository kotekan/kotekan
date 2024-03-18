# Julia source code for the CUDA upchannelizer
# This file has been generated automatically by `upchan.jl`.
# Do not modify this file, your changes will be lost.

@fastmath @inbounds(
    begin #= /home/dstn/kotekan/julia/kernels/upchan.jl:1400 =#
        info = 1
        if true
            info_memory[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 256) % 256) * 512 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) % 16) * 32) + 0) + 0x01] =
                info
        end
        Tmin = Tmin_memory[0 + 0x01]
        Tmax = Tmax_memory[0 + 0x01]
        T̄min = T̄min_memory[0 + 0x01]
        T̄max = T̄max_memory[0 + 0x01]
        if !(
            0i32 ≤ Tmin ≤ Tmax ≤ 16384 &&
            ((Tmax - Tmin) % 256 == 0i32 && (0i32 ≤ T̄min ≤ T̄max ≤ 1024 && ((T̄max - T̄min) + 3) % 16 == 0i32))
        )
            info = 2
            if true
                info_memory[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 256) % 256) * 512 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) % 16) * 32) + 0) + 0x01] =
                    info
            end
            IndexSpaces.cuda_trap()
        end
        F_ringbuf_mtaps0 = zero(Int4x8)
        F_ringbuf_mtaps1 = zero(Int4x8)
        F_ringbuf_mtaps2 = zero(Int4x8)
        Gains = G_memory[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 4) % 64) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 8) ÷ 2) % 512 + 0x01]
        (Wpfb0_m0, Wpfb1_m0) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            time0 = 0 + thread2time(thread)
            time1 = time0 + 8
            s0 = time0 + 0
            s1 = time1 + 0
            W0 = 0.033601265f0 * Wkernel(s0, 4, 16)
            W1 = 0.033601265f0 * Wkernel(s1, 4, 16)
            (W0, W1)
        end
        Wpfb_m0_t0 = Float16x2(Wpfb0_m0, Wpfb1_m0)
        (Wpfb0_m1, Wpfb1_m1) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            time0 = 0 + thread2time(thread)
            time1 = time0 + 8
            s0 = time0 + 16
            s1 = time1 + 16
            W0 = 0.033601265f0 * Wkernel(s0, 4, 16)
            W1 = 0.033601265f0 * Wkernel(s1, 4, 16)
            (W0, W1)
        end
        Wpfb_m1_t0 = Float16x2(Wpfb0_m1, Wpfb1_m1)
        (Wpfb0_m2, Wpfb1_m2) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            time0 = 0 + thread2time(thread)
            time1 = time0 + 8
            s0 = time0 + 32
            s1 = time1 + 32
            W0 = 0.033601265f0 * Wkernel(s0, 4, 16)
            W1 = 0.033601265f0 * Wkernel(s1, 4, 16)
            (W0, W1)
        end
        Wpfb_m2_t0 = Float16x2(Wpfb0_m2, Wpfb1_m2)
        (Wpfb0_m3, Wpfb1_m3) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            time0 = 0 + thread2time(thread)
            time1 = time0 + 8
            s0 = time0 + 48
            s1 = time1 + 48
            W0 = 0.033601265f0 * Wkernel(s0, 4, 16)
            W1 = 0.033601265f0 * Wkernel(s1, 4, 16)
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
            time1 = time0 + 8
            X0 = cispi(((time0 * 15) / Float32(U)) % 2.0f0)
            X1 = cispi(((time1 * 15) / Float32(U)) % 2.0f0)
            (X0, X1)
        end
        Xre = Float16x2(real(X0), real(X1))
        Xim = Float16x2(imag(X0), imag(X1))
        X_cplx0 = Xre
        X_cplx1 = Xim
        (Γ¹0, Γ¹1) = let
            k = 4
            @assert 16 == 2^k                    #= /home/dstn/kotekan/julia/kernels/upchan.jl:666 =#
            m = 3
            n = k - m
            @assert 0 ≤ m                    #= /home/dstn/kotekan/julia/kernels/upchan.jl:669 =#
            @assert 0 ≤ n                    #= /home/dstn/kotekan/julia/kernels/upchan.jl:670 =#
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            thread0 = (thread ÷ (1i32)) % (2i32)
            thread1 = (thread ÷ (2i32)) % (2i32)
            thread2 = (thread ÷ (4i32)) % (2i32)
            thread3 = (thread ÷ (8i32)) % (2i32)
            thread4 = (thread ÷ (16i32)) % (2i32)
            if false
                timehi0 = (4i32) * (0i32)
                timehi1 = (4i32) * (1i32)
                dish_in0 = (1i32) * thread1 + (2i32) * thread0
                dish_in1 = (1i32) * thread1 + (2i32) * thread0
            elseif false
                timehi0 = (4i32) * (0i32) + (2i32) * thread1
                timehi1 = (4i32) * (1i32) + (2i32) * thread1
                dish_in0 = (1i32) * thread0
                dish_in1 = (1i32) * thread0
            elseif true
                timehi0 = (4i32) * (0i32) + (2i32) * thread1 + (1i32) * thread0
                timehi1 = (4i32) * (1i32) + (2i32) * thread1 + (1i32) * thread0
                dish_in0 = 0i32
                dish_in1 = 0i32
            else
                @assert false                        #= /home/dstn/kotekan/julia/kernels/upchan.jl:693 =#
            end
            if false
                freqlo = (1i32) * thread2
                dish = (1i32) * thread4 + (2i32) * thread3
            elseif false
                freqlo = (1i32) * thread2 + (2i32) * thread4
                dish = (1i32) * thread3
            elseif true
                freqlo = (1i32) * thread2 + (2i32) * thread4 + (4i32) * thread3
                dish = 0i32
            else
                @assert false                        #= /home/dstn/kotekan/julia/kernels/upchan.jl:705 =#
            end
            delta0 = dish == dish_in0
            delta1 = dish == dish_in1
            (Γ¹0, Γ¹1) = (
                delta0 * cispi((((-2i32) * timehi0 * freqlo) / Float32(2^m)) % 2.0f0),
                delta1 * cispi((((-2i32) * timehi1 * freqlo) / Float32(2^m)) % 2.0f0),
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
        (Γ²0, Γ²1) = let
            k = 4
            @assert 16 == 2^k                    #= /home/dstn/kotekan/julia/kernels/upchan.jl:744 =#
            m = 3
            n = k - m
            @assert 0 ≤ m                    #= /home/dstn/kotekan/julia/kernels/upchan.jl:747 =#
            @assert 0 ≤ n                    #= /home/dstn/kotekan/julia/kernels/upchan.jl:748 =#
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            thread0 = (thread ÷ (1i32)) % (2i32)
            thread1 = (thread ÷ (2i32)) % (2i32)
            thread2 = (thread ÷ (4i32)) % (2i32)
            thread3 = (thread ÷ (8i32)) % (2i32)
            thread4 = (thread ÷ (16i32)) % (2i32)
            if false
                timelo0 = 0i32
                timelo1 = 0i32
            elseif false
                timelo0 = 0i32
                timelo1 = 0i32
            elseif true
                timelo0 = (1i32) * (0i32)
                timelo1 = (1i32) * (1i32)
            elseif false
                timelo0 = (2i32) * (0i32) + (1i32) * thread1
                timelo1 = (2i32) * (1i32) + (1i32) * thread1
            elseif false
                timelo0 = (4i32) * (0i32) + (2i32) * thread1 + (1i32) * thread0
                timelo1 = (4i32) * (1i32) + (2i32) * thread1 + (1i32) * thread0
            elseif false
                timelo0 = (8i32) * (0i32) + (4i32) * thread1 + (2i32) * thread0
                timelo1 = (8i32) * (1i32) + (4i32) * thread1 + (2i32) * thread0
            else
                @assert false                        #= /home/dstn/kotekan/julia/kernels/upchan.jl:774 =#
            end
            freqlo = (1i32) * thread2 + (2i32) * thread4 + (4i32) * thread3
            (Γ²0, Γ²1) = (
                cispi((((-2i32) * timelo0 * freqlo) / Float32(2^(m + n))) % 2.0f0),
                cispi((((-2i32) * timelo1 * freqlo) / Float32(2^(m + n))) % 2.0f0),
            )
            (Γ²0, Γ²1)
        end
        Γ²re = Float16x2(real(Γ²0), real(Γ²1))
        Γ²im = Float16x2(imag(Γ²0), imag(Γ²1))
        Γ²_cplx0 = Γ²re
        Γ²_cplx1 = Γ²im
        (Γ³0, Γ³1) = let
            k = 4
            @assert 16 == 2^k                    #= /home/dstn/kotekan/julia/kernels/upchan.jl:809 =#
            m = 3
            n = k - m
            @assert 0 ≤ m                    #= /home/dstn/kotekan/julia/kernels/upchan.jl:812 =#
            @assert 0 ≤ n                    #= /home/dstn/kotekan/julia/kernels/upchan.jl:813 =#
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            thread0 = (thread ÷ (1i32)) % (2i32)
            thread1 = (thread ÷ (2i32)) % (2i32)
            thread2 = (thread ÷ (4i32)) % (2i32)
            thread3 = (thread ÷ (8i32)) % (2i32)
            thread4 = (thread ÷ (16i32)) % (2i32)
            if false
                timelo0 = 0i32
                timelo1 = 0i32
                dish_in0 = (1i32) * (0i32) + (2i32) * thread1 + (4i32) * thread0
                dish_in1 = (1i32) * (1i32) + (2i32) * thread1 + (4i32) * thread0
            elseif false
                timelo0 = 0i32
                timelo1 = 0i32
                dish_in0 = (1i32) * (0i32) + (2i32) * thread1 + (4i32) * thread0
                dish_in1 = (1i32) * (1i32) + (2i32) * thread1 + (4i32) * thread0
            elseif true
                timelo0 = (1i32) * (0i32)
                timelo1 = (1i32) * (1i32)
                dish_in0 = (1i32) * thread1 + (2i32) * thread0
                dish_in1 = (1i32) * thread1 + (2i32) * thread0
            elseif false
                timelo0 = (2i32) * (0i32) + (1i32) * thread1
                timelo1 = (2i32) * (1i32) + (1i32) * thread1
                dish_in0 = (1i32) * thread0
                dish_in1 = (1i32) * thread0
            elseif false
                timelo0 = (4i32) * (0i32) + (2i32) * thread1 + (1i32) * thread0
                timelo1 = (4i32) * (1i32) + (2i32) * thread1 + (1i32) * thread0
                dish_in0 = 0i32
                dish_in1 = 0i32
            elseif false
                timelo0 = (8i32) * (0i32) + (4i32) * thread1 + (2i32) * thread0
                timelo1 = (8i32) * (1i32) + (4i32) * thread1 + (2i32) * thread0
                dish_in0 = 0i32
                dish_in1 = 0i32
            else
                @assert false                        #= /home/dstn/kotekan/julia/kernels/upchan.jl:851 =#
            end
            if false
                freqhi = 0i32
                dish = (1i32) * thread2 + (2i32) * thread4 + (4i32) * thread3
            elseif false
                freqhi = 0i32
                dish = (1i32) * thread2 + (2i32) * thread4 + (4i32) * thread3
            elseif true
                freqhi = (1i32) * thread2
                dish = (1i32) * thread4 + (2i32) * thread3
            elseif false
                freqhi = (1i32) * thread2 + (2i32) * thread4
                dish = (1i32) * thread3
            elseif false
                freqhi = (1i32) * thread2 + (2i32) * thread4 + (4i32) * thread3
                dish = 0i32
            elseif false
                freqhi = (1i32) * thread2 + (2i32) * thread4 + (4i32) * thread3
                dish = 0i32
            else
                @assert false                        #= /home/dstn/kotekan/julia/kernels/upchan.jl:872 =#
            end
            delta0 = dish == dish_in0
            delta1 = dish == dish_in1
            (Γ³0, Γ³1) = (
                delta0 * cispi((((-2i32) * timelo0 * freqhi) / Float32(2^n)) % 2.0f0),
                delta1 * cispi((((-2i32) * timelo1 * freqhi) / Float32(2^n)) % 2.0f0),
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
        for t_outer in 0:256:8191
            Tmin + t_outer ≥ Tmax && break
            (E_dish0_time0, E_dish4_time0, E_dish8_time0, E_dish12_time0) = IndexSpaces.unsafe_load4_global(
                E_memory,
                let
                    offset = 8192 * Tmin
                    length = 67108864
                    mod(
                        (
                            ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 4) % 64) * 16) % 64) * 128 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 2) % 2) % 2) * 64 +
                            (
                                (
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128 +
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16
                                ) ÷ 4
                            ) % 64 +
                            (
                                (
                                    (
                                        (
                                            (
                                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 +
                                                IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4
                                            ) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256
                                        ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8
                                    ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4
                                ) % 8192
                            ) * 8192
                        ) + offset,
                        length,
                    )
                end + 1i32,
            )
            (E_dish0_time64, E_dish4_time64, E_dish8_time64, E_dish12_time64) = IndexSpaces.unsafe_load4_global(
                E_memory,
                let
                    offset = 8192 * Tmin
                    length = 67108864
                    mod(
                        (
                            ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 4) % 64) * 16) % 64) * 128 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 2) % 2) % 2) * 64 +
                            (
                                (
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128 +
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16
                                ) ÷ 4
                            ) % 64 +
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 +
                                                    IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4
                                                ) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256
                                            ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8
                                        ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4
                                    ) + 64
                                ) % 8192
                            ) * 8192
                        ) + offset,
                        length,
                    )
                end + 1i32,
            )
            (E_dish0_time128, E_dish4_time128, E_dish8_time128, E_dish12_time128) = IndexSpaces.unsafe_load4_global(
                E_memory,
                let
                    offset = 8192 * Tmin
                    length = 67108864
                    mod(
                        (
                            ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 4) % 64) * 16) % 64) * 128 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 2) % 2) % 2) * 64 +
                            (
                                (
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128 +
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16
                                ) ÷ 4
                            ) % 64 +
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 +
                                                    IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4
                                                ) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256
                                            ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8
                                        ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4
                                    ) + 128
                                ) % 8192
                            ) * 8192
                        ) + offset,
                        length,
                    )
                end + 1i32,
            )
            (E_dish0_time192, E_dish4_time192, E_dish8_time192, E_dish12_time192) = IndexSpaces.unsafe_load4_global(
                E_memory,
                let
                    offset = 8192 * Tmin
                    length = 67108864
                    mod(
                        (
                            ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 4) % 64) * 16) % 64) * 128 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 2) % 2) % 2) * 64 +
                            (
                                (
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128 +
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16
                                ) ÷ 4
                            ) % 64 +
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 +
                                                    IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4
                                                ) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256
                                            ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8
                                        ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4
                                    ) + 192
                                ) % 8192
                            ) * 8192
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
            E1_dish0_time8 = E1hi_dish0_time0
            E1_dish4_time0 = E1lo_dish4_time0
            E1_dish4_time8 = E1hi_dish4_time0
            E1_dish0_time64 = E1lo_dish0_time64
            E1_dish0_time72 = E1hi_dish0_time64
            E1_dish4_time64 = E1lo_dish4_time64
            E1_dish4_time72 = E1hi_dish4_time64
            E1_dish0_time128 = E1lo_dish0_time128
            E1_dish0_time136 = E1hi_dish0_time128
            E1_dish4_time128 = E1lo_dish4_time128
            E1_dish4_time136 = E1hi_dish4_time128
            E1_dish0_time192 = E1lo_dish0_time192
            E1_dish0_time200 = E1hi_dish0_time192
            E1_dish4_time192 = E1lo_dish4_time192
            E1_dish4_time200 = E1hi_dish4_time192
            (E2_dish0_time0, E2_dish0_time8) = (
                IndexSpaces.get_lo16(E1_dish0_time0, E1_dish0_time8), IndexSpaces.get_hi16(E1_dish0_time0, E1_dish0_time8)
            )
            (E2_dish4_time0, E2_dish4_time8) = (
                IndexSpaces.get_lo16(E1_dish4_time0, E1_dish4_time8), IndexSpaces.get_hi16(E1_dish4_time0, E1_dish4_time8)
            )
            (E2_dish0_time64, E2_dish0_time72) = (
                IndexSpaces.get_lo16(E1_dish0_time64, E1_dish0_time72), IndexSpaces.get_hi16(E1_dish0_time64, E1_dish0_time72)
            )
            (E2_dish4_time64, E2_dish4_time72) = (
                IndexSpaces.get_lo16(E1_dish4_time64, E1_dish4_time72), IndexSpaces.get_hi16(E1_dish4_time64, E1_dish4_time72)
            )
            (E2_dish0_time128, E2_dish0_time136) = (
                IndexSpaces.get_lo16(E1_dish0_time128, E1_dish0_time136), IndexSpaces.get_hi16(E1_dish0_time128, E1_dish0_time136)
            )
            (E2_dish4_time128, E2_dish4_time136) = (
                IndexSpaces.get_lo16(E1_dish4_time128, E1_dish4_time136), IndexSpaces.get_hi16(E1_dish4_time128, E1_dish4_time136)
            )
            (E2_dish0_time192, E2_dish0_time200) = (
                IndexSpaces.get_lo16(E1_dish0_time192, E1_dish0_time200), IndexSpaces.get_hi16(E1_dish0_time192, E1_dish0_time200)
            )
            (E2_dish4_time192, E2_dish4_time200) = (
                IndexSpaces.get_lo16(E1_dish4_time192, E1_dish4_time200), IndexSpaces.get_hi16(E1_dish4_time192, E1_dish4_time200)
            )
            E2lo_dish0_time0 = E2_dish0_time0
            E2hi_dish0_time0 = E2_dish0_time8
            E2lo_dish4_time0 = E2_dish4_time0
            E2hi_dish4_time0 = E2_dish4_time8
            E2lo_dish0_time64 = E2_dish0_time64
            E2hi_dish0_time64 = E2_dish0_time72
            E2lo_dish4_time64 = E2_dish4_time64
            E2hi_dish4_time64 = E2_dish4_time72
            E2lo_dish0_time128 = E2_dish0_time128
            E2hi_dish0_time128 = E2_dish0_time136
            E2lo_dish4_time128 = E2_dish4_time128
            E2hi_dish4_time128 = E2_dish4_time136
            E2lo_dish0_time192 = E2_dish0_time192
            E2hi_dish0_time192 = E2_dish0_time200
            E2lo_dish4_time192 = E2_dish4_time192
            E2hi_dish4_time192 = E2_dish4_time200
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
                F_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) % 2) * 260 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) ÷ 16) % 16) * 545 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) ÷ 2) % 2) * 130 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) ÷ 4) % 2) * 65) + 0) + 0x01] =
                    F_dish0_time0
            end
            if true
                F_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) ÷ 16) % 16) * 545 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) ÷ 2) % 2) * 130 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) ÷ 4) % 2) * 65) + 0) + 0x01] =
                    F_dish2_time0
            end
            if true
                F_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) ÷ 16) % 16) * 545 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) ÷ 2) % 2) * 130 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) ÷ 4) % 2) * 65) + 0) + 0x01] =
                    F_dish4_time0
            end
            if true
                F_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 6) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 6) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) ÷ 16) % 16) * 545 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) ÷ 2) % 2) * 130 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) ÷ 4) % 2) * 65) + 0) + 0x01] =
                    F_dish6_time0
            end
            if true
                F_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 64) % 2) * 260 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 64) ÷ 16) % 16) * 545 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 64) ÷ 2) % 2) * 130 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 64) ÷ 4) % 2) * 65) + 0) + 0x01] =
                    F_dish0_time64
            end
            if true
                F_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 64) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 64) ÷ 16) % 16) * 545 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 64) ÷ 2) % 2) * 130 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 64) ÷ 4) % 2) * 65) + 0) + 0x01] =
                    F_dish2_time64
            end
            if true
                F_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 64) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 64) ÷ 16) % 16) * 545 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 64) ÷ 2) % 2) * 130 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 64) ÷ 4) % 2) * 65) + 0) + 0x01] =
                    F_dish4_time64
            end
            if true
                F_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 6) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 64) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 6) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 64) ÷ 16) % 16) * 545 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 64) ÷ 2) % 2) * 130 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 64) ÷ 4) % 2) * 65) + 0) + 0x01] =
                    F_dish6_time64
            end
            if true
                F_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 128) % 2) * 260 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 128) ÷ 16) % 16) * 545 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 128) ÷ 2) % 2) * 130 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 128) ÷ 4) % 2) * 65) + 0) + 0x01] =
                    F_dish0_time128
            end
            if true
                F_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 128) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 128) ÷ 16) % 16) * 545 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 128) ÷ 2) % 2) * 130 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 128) ÷ 4) % 2) * 65) + 0) + 0x01] =
                    F_dish2_time128
            end
            if true
                F_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 128) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 128) ÷ 16) % 16) * 545 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 128) ÷ 2) % 2) * 130 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 128) ÷ 4) % 2) * 65) + 0) + 0x01] =
                    F_dish4_time128
            end
            if true
                F_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 6) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 128) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 6) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 128) ÷ 16) % 16) * 545 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 128) ÷ 2) % 2) * 130 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 128) ÷ 4) % 2) * 65) + 0) + 0x01] =
                    F_dish6_time128
            end
            if true
                F_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 192) % 2) * 260 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 192) ÷ 16) % 16) * 545 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 192) ÷ 2) % 2) * 130 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 192) ÷ 4) % 2) * 65) + 0) + 0x01] =
                    F_dish0_time192
            end
            if true
                F_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 192) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 192) ÷ 16) % 16) * 545 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 192) ÷ 2) % 2) * 130 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 192) ÷ 4) % 2) * 65) + 0) + 0x01] =
                    F_dish2_time192
            end
            if true
                F_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 192) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 192) ÷ 16) % 16) * 545 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 192) ÷ 2) % 2) * 130 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 192) ÷ 4) % 2) * 65) + 0) + 0x01] =
                    F_dish4_time192
            end
            if true
                F_shared[((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 6) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 192) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 6) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 192) ÷ 16) % 16) * 545 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 192) ÷ 2) % 2) * 130 + ((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 4) + 192) ÷ 4) % 2) * 65) + 0) + 0x01] =
                    F_dish6_time192
            end
            IndexSpaces.cuda_sync_threads()
            for t_inner in 0:16:255
                let
                    dish = 0
                    F_in = F_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 64) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 2) + ((IndexSpaces.assume_inrange(t_inner, 0, 16, 256) ÷ 16) % 16) * 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) % 2) * 260 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 64) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 32 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 2) + ((IndexSpaces.assume_inrange(t_inner, 0, 16, 256) ÷ 16) % 16) * 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) ÷ 16) % 16) * 545 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 2) + ((IndexSpaces.assume_inrange(t_inner, 0, 16, 256) ÷ 16) % 16) * 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) ÷ 2) % 2) * 130 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 2) + ((IndexSpaces.assume_inrange(t_inner, 0, 16, 256) ÷ 16) % 16) * 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) ÷ 4) % 2) * 65) + 0x01]
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
                    Γ²re = Γ²_cplx0
                    Γ²im = Γ²_cplx1
                    WWre_dish0 = WW_cplx0_dish0
                    WWim_dish0 = WW_cplx1_dish0
                    WWre_dish1 = WW_cplx0_dish1
                    WWim_dish1 = WW_cplx1_dish1
                    ZZre_dish0 = muladd(Γ²re, WWre_dish0, -Γ²im * WWim_dish0)
                    ZZre_dish1 = muladd(Γ²re, WWre_dish1, -Γ²im * WWim_dish1)
                    ZZim_dish0 = muladd(Γ²re, WWim_dish0, Γ²im * WWre_dish0)
                    ZZim_dish1 = muladd(Γ²re, WWim_dish1, Γ²im * WWre_dish1)
                    ZZ_cplx0_dish0 = ZZre_dish0
                    ZZ_cplx1_dish0 = ZZim_dish0
                    ZZ_cplx0_dish1 = ZZre_dish1
                    ZZ_cplx1_dish1 = ZZim_dish1
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
                        F̄_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 64) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 32 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 64) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 32 + (((((IndexSpaces.assume_inrange(t_inner, 0, 16, 256) ÷ 16) % 16) * 16 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) ÷ 16) % 16) * 545 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 4) % 64) * 16) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 8) ÷ 2) % 8) * 65) + 0) + 0x01] =
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
            Ē_dish0_time0 = F̄_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) ÷ 16) % 16) * 545 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 4) % 64) * 16) ÷ 2) % 8) * 65) + 0x01]
            Ē_dish2_time0 = F̄_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) ÷ 16) % 16) * 545 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 4) % 64) * 16) ÷ 2) % 8) * 65) + 0x01]
            Ē_dish4_time0 = F̄_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) ÷ 16) % 16) * 545 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 4) % 64) * 16) ÷ 2) % 8) * 65) + 0x01]
            Ē_dish6_time0 = F̄_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 6) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 6) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) ÷ 16) % 16) * 545 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 4) % 64) * 16) ÷ 2) % 8) * 65) + 0x01]
            Ē_dish0_time64 = F̄_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + 64) ÷ 16) % 16) * 545 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 4) % 64) * 16) ÷ 2) % 8) * 65) + 0x01]
            Ē_dish2_time64 = F̄_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + 64) ÷ 16) % 16) * 545 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 4) % 64) * 16) ÷ 2) % 8) * 65) + 0x01]
            Ē_dish4_time64 = F̄_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + 64) ÷ 16) % 16) * 545 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 4) % 64) * 16) ÷ 2) % 8) * 65) + 0x01]
            Ē_dish6_time64 = F̄_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 6) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 6) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + 64) ÷ 16) % 16) * 545 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 4) % 64) * 16) ÷ 2) % 8) * 65) + 0x01]
            Ē_dish0_time128 = F̄_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + 128) ÷ 16) % 16) * 545 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 4) % 64) * 16) ÷ 2) % 8) * 65) + 0x01]
            Ē_dish2_time128 = F̄_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + 128) ÷ 16) % 16) * 545 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 4) % 64) * 16) ÷ 2) % 8) * 65) + 0x01]
            Ē_dish4_time128 = F̄_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + 128) ÷ 16) % 16) * 545 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 4) % 64) * 16) ÷ 2) % 8) * 65) + 0x01]
            Ē_dish6_time128 = F̄_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 6) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 6) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + 128) ÷ 16) % 16) * 545 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 4) % 64) * 16) ÷ 2) % 8) * 65) + 0x01]
            Ē_dish0_time192 = F̄_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + 192) ÷ 16) % 16) * 545 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 4) % 64) * 16) ÷ 2) % 8) * 65) + 0x01]
            Ē_dish2_time192 = F̄_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + 192) ÷ 16) % 16) * 545 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 4) % 64) * 16) ÷ 2) % 8) * 65) + 0x01]
            Ē_dish4_time192 = F̄_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + 192) ÷ 16) % 16) * 545 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 4) % 64) * 16) ÷ 2) % 8) * 65) + 0x01]
            Ē_dish6_time192 = F̄_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 6) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 4) % 32 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128) + 6) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16) ÷ 2) % 2) * 32 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256) + 192) ÷ 16) % 16) * 545 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 4) % 64) * 16) ÷ 2) % 8) * 65) + 0x01]
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
            if ((0 + ((IndexSpaces.cuda_warpidx() ÷ 4) % (2i32)) * 16) + ((IndexSpaces.cuda_warpidx() ÷ 8) % (2i32)) * 32) +
               ((0 ÷ 64) % (2i32)) * 64 ≥ 48
                IndexSpaces.unsafe_store4_global!(
                    Ē_memory,
                    let
                        offset = 131072 * T̄min - 393216
                        length = 67108864
                        mod(
                            (
                                (
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 2) % 2) % 2) * 64 +
                                    (
                                        (
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128 +
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16
                                        ) ÷ 4
                                    ) % 64 +
                                    (
                                        (
                                            (
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) * 4 +
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4
                                            ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 4) % 64) * 16
                                        ) % 1024
                                    ) * 128 +
                                    (
                                        (
                                            (
                                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 +
                                                ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256
                                            ) ÷ 16
                                        ) % 512
                                    ) * 131072
                                ) + 0
                            ) + offset,
                            length,
                        )
                    end + 0x01,
                    (Ē3_dish0_time0, Ē3_dish4_time0, Ē3_dish8_time0, Ē3_dish12_time0),
                )
            end
            if ((0 + ((IndexSpaces.cuda_warpidx() ÷ 4) % (2i32)) * 16) + ((IndexSpaces.cuda_warpidx() ÷ 8) % (2i32)) * 32) +
               ((64 ÷ 64) % (2i32)) * 64 ≥ 48
                IndexSpaces.unsafe_store4_global!(
                    Ē_memory,
                    let
                        offset = 131072 * T̄min - 393216
                        length = 67108864
                        mod(
                            (
                                (
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 2) % 2) % 2) * 64 +
                                    (
                                        (
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128 +
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16
                                        ) ÷ 4
                                    ) % 64 +
                                    (
                                        (
                                            (
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) * 4 +
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4
                                            ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 4) % 64) * 16
                                        ) % 1024
                                    ) * 128 +
                                    (
                                        (
                                            (
                                                (
                                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 +
                                                    ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256
                                                ) + 64
                                            ) ÷ 16
                                        ) % 512
                                    ) * 131072
                                ) + 0
                            ) + offset,
                            length,
                        )
                    end + 0x01,
                    (Ē3_dish0_time64, Ē3_dish4_time64, Ē3_dish8_time64, Ē3_dish12_time64),
                )
            end
            if ((0 + ((IndexSpaces.cuda_warpidx() ÷ 4) % (2i32)) * 16) + ((IndexSpaces.cuda_warpidx() ÷ 8) % (2i32)) * 32) +
               ((128 ÷ 64) % (2i32)) * 64 ≥ 48
                IndexSpaces.unsafe_store4_global!(
                    Ē_memory,
                    let
                        offset = 131072 * T̄min - 393216
                        length = 67108864
                        mod(
                            (
                                (
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 2) % 2) % 2) * 64 +
                                    (
                                        (
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128 +
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16
                                        ) ÷ 4
                                    ) % 64 +
                                    (
                                        (
                                            (
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) * 4 +
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4
                                            ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 4) % 64) * 16
                                        ) % 1024
                                    ) * 128 +
                                    (
                                        (
                                            (
                                                (
                                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 +
                                                    ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256
                                                ) + 128
                                            ) ÷ 16
                                        ) % 512
                                    ) * 131072
                                ) + 0
                            ) + offset,
                            length,
                        )
                    end + 0x01,
                    (Ē3_dish0_time128, Ē3_dish4_time128, Ē3_dish8_time128, Ē3_dish12_time128),
                )
            end
            if ((0 + ((IndexSpaces.cuda_warpidx() ÷ 4) % (2i32)) * 16) + ((IndexSpaces.cuda_warpidx() ÷ 8) % (2i32)) * 32) +
               ((192 ÷ 64) % (2i32)) * 64 ≥ 48
                IndexSpaces.unsafe_store4_global!(
                    Ē_memory,
                    let
                        offset = 131072 * T̄min - 393216
                        length = 67108864
                        mod(
                            (
                                (
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 2) % 2) % 2) * 64 +
                                    (
                                        (
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 2) * 128 +
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 8) * 16
                                        ) ÷ 4
                                    ) % 64 +
                                    (
                                        (
                                            (
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 4) * 4 +
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4
                                            ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) ÷ 4) % 64) * 16
                                        ) % 1024
                                    ) * 128 +
                                    (
                                        (
                                            (
                                                (
                                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 4) % 4) * 16 +
                                                    ((IndexSpaces.assume_inrange(t_outer, 0, 256, 8192) ÷ 256) % 32) * 256
                                                ) + 192
                                            ) ÷ 16
                                        ) % 512
                                    ) * 131072
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
            info_memory[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 256) % 256) % 256) * 512 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) % 16) * 32) + 0) + 0x01] =
                info
        end
    end
)
