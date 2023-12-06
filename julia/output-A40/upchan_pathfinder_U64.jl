@fastmath @inbounds(
    begin #= /home/eschnett/src/kotekan/julia/kernels/upchan.jl:1316 =#
        info = 1
        if true
            info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) % 16) * 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) % 128) * 512 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                info
        end
        Tmin = Tmin_memory[0 + 0x01]
        Tmax = Tmax_memory[0 + 0x01]
        T̄min = T̄min_memory[0 + 0x01]
        T̄max = T̄max_memory[0 + 0x01]
        if !(
            0i32 ≤ Tmin ≤ Tmax ≤ 262144 &&
            ((Tmax - Tmin) % 256 == 0i32 && (0i32 ≤ T̄min ≤ T̄max ≤ 4096 && ((T̄max - T̄min) + 3) % 4 == 0i32))
        )
            info = 2
            if true
                info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) % 16) * 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) % 128) * 512 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                    info
            end
            IndexSpaces.cuda_trap()
        end
        F_ringbuf_dish0_mtaps0_polr0 = zero(Int4x8)
        F_ringbuf_dish32_mtaps0_polr0 = zero(Int4x8)
        F_ringbuf_dish0_mtaps1_polr0 = zero(Int4x8)
        F_ringbuf_dish32_mtaps1_polr0 = zero(Int4x8)
        F_ringbuf_dish0_mtaps2_polr0 = zero(Int4x8)
        F_ringbuf_dish32_mtaps2_polr0 = zero(Int4x8)
        F_ringbuf_dish0_mtaps0_polr1 = zero(Int4x8)
        F_ringbuf_dish32_mtaps0_polr1 = zero(Int4x8)
        F_ringbuf_dish0_mtaps1_polr1 = zero(Int4x8)
        F_ringbuf_dish32_mtaps1_polr1 = zero(Int4x8)
        F_ringbuf_dish0_mtaps2_polr1 = zero(Int4x8)
        F_ringbuf_dish32_mtaps2_polr1 = zero(Int4x8)
        Gains = G_memory[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 2) * 2) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 8) ÷ 2) % 4096 + 0x01]
        (Wpfb0_m0, Wpfb1_m0) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            time0 = 0 + thread2time(thread)
            time1 = time0 + 32
            s0 = time0 + 0
            s1 = time1 + 0
            W0 = 0.008481575f0 * Wkernel(s0, 4, 64)
            W1 = 0.008481575f0 * Wkernel(s1, 4, 64)
            (W0, W1)
        end
        Wpfb_m0_t0 = Float16x2(Wpfb0_m0, Wpfb1_m0)
        (Wpfb0_m1, Wpfb1_m1) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            time0 = 0 + thread2time(thread)
            time1 = time0 + 32
            s0 = time0 + 64
            s1 = time1 + 64
            W0 = 0.008481575f0 * Wkernel(s0, 4, 64)
            W1 = 0.008481575f0 * Wkernel(s1, 4, 64)
            (W0, W1)
        end
        Wpfb_m1_t0 = Float16x2(Wpfb0_m1, Wpfb1_m1)
        (Wpfb0_m2, Wpfb1_m2) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            time0 = 0 + thread2time(thread)
            time1 = time0 + 32
            s0 = time0 + 128
            s1 = time1 + 128
            W0 = 0.008481575f0 * Wkernel(s0, 4, 64)
            W1 = 0.008481575f0 * Wkernel(s1, 4, 64)
            (W0, W1)
        end
        Wpfb_m2_t0 = Float16x2(Wpfb0_m2, Wpfb1_m2)
        (Wpfb0_m3, Wpfb1_m3) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            time0 = 0 + thread2time(thread)
            time1 = time0 + 32
            s0 = time0 + 192
            s1 = time1 + 192
            W0 = 0.008481575f0 * Wkernel(s0, 4, 64)
            W1 = 0.008481575f0 * Wkernel(s1, 4, 64)
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
            time1 = time0 + 32
            X0 = cispi(((time0 * 63) / Float32(U)) % 2.0f0)
            X1 = cispi(((time1 * 63) / Float32(U)) % 2.0f0)
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
        (Γ²0, Γ²1) = let
            k = Ubits
            @assert U == 2^k                    #= /home/eschnett/src/kotekan/julia/kernels/upchan.jl:705 =#
            m = 3
            n = k - m
            @assert 0 ≤ m                    #= /home/eschnett/src/kotekan/julia/kernels/upchan.jl:708 =#
            @assert 0 ≤ n                    #= /home/eschnett/src/kotekan/julia/kernels/upchan.jl:709 =#
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
                @assert false                        #= /home/eschnett/src/kotekan/julia/kernels/upchan.jl:732 =#
            end
            freqlo = (1i32) * thread2 + (2i32) * thread3 + (4i32) * thread4
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
        Γ³_cplx0_cplx_in0_dish0 = Γ³_cplx0_cplx_in0_dish0
        Γ³_cplx0_cplx_in0_dish32 = Γ³_cplx0_cplx_in0_dish0
        Γ³_cplx1_cplx_in0_dish0 = Γ³_cplx1_cplx_in0_dish0
        Γ³_cplx1_cplx_in0_dish32 = Γ³_cplx1_cplx_in0_dish0
        Γ³_cplx0_cplx_in1_dish0 = Γ³_cplx0_cplx_in1_dish0
        Γ³_cplx0_cplx_in1_dish32 = Γ³_cplx0_cplx_in1_dish0
        Γ³_cplx1_cplx_in1_dish0 = Γ³_cplx1_cplx_in1_dish0
        Γ³_cplx1_cplx_in1_dish32 = Γ³_cplx1_cplx_in1_dish0
        Γ³_cplx0_cplx_in0_dish1 = Γ³_cplx0_cplx_in0_dish1
        Γ³_cplx0_cplx_in0_dish33 = Γ³_cplx0_cplx_in0_dish1
        Γ³_cplx1_cplx_in0_dish1 = Γ³_cplx1_cplx_in0_dish1
        Γ³_cplx1_cplx_in0_dish33 = Γ³_cplx1_cplx_in0_dish1
        Γ³_cplx0_cplx_in1_dish1 = Γ³_cplx0_cplx_in1_dish1
        Γ³_cplx0_cplx_in1_dish33 = Γ³_cplx0_cplx_in1_dish1
        Γ³_cplx1_cplx_in1_dish1 = Γ³_cplx1_cplx_in1_dish1
        Γ³_cplx1_cplx_in1_dish33 = Γ³_cplx1_cplx_in1_dish1
        Γ³_cplx0_cplx_in0_dish0_polr0 = Γ³_cplx0_cplx_in0_dish0
        Γ³_cplx0_cplx_in0_dish0_polr1 = Γ³_cplx0_cplx_in0_dish0
        Γ³_cplx1_cplx_in0_dish0_polr0 = Γ³_cplx1_cplx_in0_dish0
        Γ³_cplx1_cplx_in0_dish0_polr1 = Γ³_cplx1_cplx_in0_dish0
        Γ³_cplx0_cplx_in1_dish0_polr0 = Γ³_cplx0_cplx_in1_dish0
        Γ³_cplx0_cplx_in1_dish0_polr1 = Γ³_cplx0_cplx_in1_dish0
        Γ³_cplx1_cplx_in1_dish0_polr0 = Γ³_cplx1_cplx_in1_dish0
        Γ³_cplx1_cplx_in1_dish0_polr1 = Γ³_cplx1_cplx_in1_dish0
        Γ³_cplx0_cplx_in0_dish1_polr0 = Γ³_cplx0_cplx_in0_dish1
        Γ³_cplx0_cplx_in0_dish1_polr1 = Γ³_cplx0_cplx_in0_dish1
        Γ³_cplx1_cplx_in0_dish1_polr0 = Γ³_cplx1_cplx_in0_dish1
        Γ³_cplx1_cplx_in0_dish1_polr1 = Γ³_cplx1_cplx_in0_dish1
        Γ³_cplx0_cplx_in1_dish1_polr0 = Γ³_cplx0_cplx_in1_dish1
        Γ³_cplx0_cplx_in1_dish1_polr1 = Γ³_cplx0_cplx_in1_dish1
        Γ³_cplx1_cplx_in1_dish1_polr0 = Γ³_cplx1_cplx_in1_dish1
        Γ³_cplx1_cplx_in1_dish1_polr1 = Γ³_cplx1_cplx_in1_dish1
        Γ³_cplx0_cplx_in0_dish32_polr0 = Γ³_cplx0_cplx_in0_dish32
        Γ³_cplx0_cplx_in0_dish32_polr1 = Γ³_cplx0_cplx_in0_dish32
        Γ³_cplx1_cplx_in0_dish32_polr0 = Γ³_cplx1_cplx_in0_dish32
        Γ³_cplx1_cplx_in0_dish32_polr1 = Γ³_cplx1_cplx_in0_dish32
        Γ³_cplx0_cplx_in1_dish32_polr0 = Γ³_cplx0_cplx_in1_dish32
        Γ³_cplx0_cplx_in1_dish32_polr1 = Γ³_cplx0_cplx_in1_dish32
        Γ³_cplx1_cplx_in1_dish32_polr0 = Γ³_cplx1_cplx_in1_dish32
        Γ³_cplx1_cplx_in1_dish32_polr1 = Γ³_cplx1_cplx_in1_dish32
        Γ³_cplx0_cplx_in0_dish33_polr0 = Γ³_cplx0_cplx_in0_dish33
        Γ³_cplx0_cplx_in0_dish33_polr1 = Γ³_cplx0_cplx_in0_dish33
        Γ³_cplx1_cplx_in0_dish33_polr0 = Γ³_cplx1_cplx_in0_dish33
        Γ³_cplx1_cplx_in0_dish33_polr1 = Γ³_cplx1_cplx_in0_dish33
        Γ³_cplx0_cplx_in1_dish33_polr0 = Γ³_cplx0_cplx_in1_dish33
        Γ³_cplx0_cplx_in1_dish33_polr1 = Γ³_cplx0_cplx_in1_dish33
        Γ³_cplx1_cplx_in1_dish33_polr0 = Γ³_cplx1_cplx_in1_dish33
        Γ³_cplx1_cplx_in1_dish33_polr1 = Γ³_cplx1_cplx_in1_dish33
        for t_outer in 0:256:131071
            Tmin + t_outer ≥ Tmax && break
            (E_dish0, E_dish4, E_dish8, E_dish12) = IndexSpaces.unsafe_load4_global(
                E_memory,
                let
                    offset = 4096 * Tmin
                    length = 536870912
                    mod(
                        (
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) * 64) % 128) * 32 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16 +
                            (
                                (
                                    (
                                        (
                                            (
                                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 +
                                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16
                                            ) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 32
                                        ) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16
                                    ) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256
                                ) % 131072
                            ) * 4096
                        ) + offset,
                        length,
                    )
                end + 1i32,
            )
            is_lo_thread = IndexSpaces.cuda_threadidx() & 0x00000008 == 0x00
            (E1_dish0, E1_dish8) = let
                src = if is_lo_thread
                    E_dish8
                else
                    E_dish0
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E_dish0, dst)
                else
                    (dst, E_dish8)
                end
            end
            (E1_dish4, E1_dish12) = let
                src = if is_lo_thread
                    E_dish12
                else
                    E_dish4
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E_dish4, dst)
                else
                    (dst, E_dish12)
                end
            end
            E1lo_dish0 = E1_dish0
            E1hi_dish0 = E1_dish8
            E1lo_dish4 = E1_dish4
            E1hi_dish4 = E1_dish12
            E1_dish0_time0 = E1lo_dish0
            E1_dish0_time32 = E1hi_dish0
            E1_dish4_time0 = E1lo_dish4
            E1_dish4_time32 = E1hi_dish4
            (E2_dish0_time0, E2_dish0_time32) = (
                IndexSpaces.get_lo16(E1_dish0_time0, E1_dish0_time32), IndexSpaces.get_hi16(E1_dish0_time0, E1_dish0_time32)
            )
            (E2_dish4_time0, E2_dish4_time32) = (
                IndexSpaces.get_lo16(E1_dish4_time0, E1_dish4_time32), IndexSpaces.get_hi16(E1_dish4_time0, E1_dish4_time32)
            )
            E2lo_dish0 = E2_dish0_time0
            E2hi_dish0 = E2_dish0_time32
            E2lo_dish4 = E2_dish4_time0
            E2hi_dish4 = E2_dish4_time32
            E2_dish0 = E2lo_dish0
            E2_dish2 = E2hi_dish0
            E2_dish4 = E2lo_dish4
            E2_dish6 = E2hi_dish4
            F_dish0 = E2_dish0
            F_dish2 = E2_dish2
            F_dish4 = E2_dish4
            F_dish6 = E2_dish6
            if true
                F_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 16) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) % 2) * 1040 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 2) % 2) * 520 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 4) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 2) * 130) + 0) + 0x01] =
                    F_dish0
            end
            if true
                F_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + 2) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 64) % 4) * 2081 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + 2) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 16) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) % 2) * 1040 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 2) % 2) * 520 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 4) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 2) * 130) + 0) + 0x01] =
                    F_dish2
            end
            if true
                F_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + 4) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 64) % 4) * 2081 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + 4) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 16) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) % 2) * 1040 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 2) % 2) * 520 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 4) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 2) * 130) + 0) + 0x01] =
                    F_dish4
            end
            if true
                F_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + 6) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 64) % 4) * 2081 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + 6) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 16) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) % 2) * 1040 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 2) % 2) * 520 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 4) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16) + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 8) % 2) * 130) + 0) + 0x01] =
                    F_dish6
            end
            IndexSpaces.cuda_sync_threads()
            for t_inner in 0:64:255
                let
                    dish = 0
                    F_in_dish0_polr0 = F_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 32 + 0 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 16) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) % 2) * 1040 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 2) % 2) * 520 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 4) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 8) % 2) * 130) + 0x01]
                    F_in_dish32_polr0 = F_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 2) % 2) * 32 + 0 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 16) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) % 2) * 1040 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 2) % 2) * 520 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 4) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 8) % 2) * 130) + 0x01]
                    F_in_dish0_polr1 = F_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 32 + 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 16) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) % 2) * 1040 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 2) % 2) * 520 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 4) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 8) % 2) * 130) + 0x01]
                    F_in_dish32_polr1 = F_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 2) % 2) * 32 + 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 16) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) % 2) * 1040 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 2) % 2) * 520 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 4) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 8) % 2) * 130) + 0x01]
                    (E_cplx0_dish0_polr0, E_cplx1_dish0_polr0, E_cplx0_dish1_polr0, E_cplx1_dish1_polr0) = convert(
                        NTuple{4,Float16x2}, F_in_dish0_polr0
                    )
                    (E_cplx0_dish32_polr0, E_cplx1_dish32_polr0, E_cplx0_dish33_polr0, E_cplx1_dish33_polr0) = convert(
                        NTuple{4,Float16x2}, F_in_dish32_polr0
                    )
                    (E_cplx0_dish0_polr1, E_cplx1_dish0_polr1, E_cplx0_dish1_polr1, E_cplx1_dish1_polr1) = convert(
                        NTuple{4,Float16x2}, F_in_dish0_polr1
                    )
                    (E_cplx0_dish32_polr1, E_cplx1_dish32_polr1, E_cplx0_dish33_polr1, E_cplx1_dish33_polr1) = convert(
                        NTuple{4,Float16x2}, F_in_dish32_polr1
                    )
                    W_m0 = Wpfb_mtaps0
                    W_m1 = Wpfb_mtaps1
                    W_m2 = Wpfb_mtaps2
                    W_m3 = Wpfb_mtaps3
                    E2_cplx0_dish0_polr0 = -W_m3 * E_cplx0_dish0_polr0
                    E2_cplx1_dish0_polr0 = -W_m3 * E_cplx1_dish0_polr0
                    E2_cplx0_dish1_polr0 = -W_m3 * E_cplx0_dish1_polr0
                    E2_cplx1_dish1_polr0 = -W_m3 * E_cplx1_dish1_polr0
                    E2_cplx0_dish32_polr0 = -W_m3 * E_cplx0_dish32_polr0
                    E2_cplx1_dish32_polr0 = -W_m3 * E_cplx1_dish32_polr0
                    E2_cplx0_dish33_polr0 = -W_m3 * E_cplx0_dish33_polr0
                    E2_cplx1_dish33_polr0 = -W_m3 * E_cplx1_dish33_polr0
                    E2_cplx0_dish0_polr1 = -W_m3 * E_cplx0_dish0_polr1
                    E2_cplx1_dish0_polr1 = -W_m3 * E_cplx1_dish0_polr1
                    E2_cplx0_dish1_polr1 = -W_m3 * E_cplx0_dish1_polr1
                    E2_cplx1_dish1_polr1 = -W_m3 * E_cplx1_dish1_polr1
                    E2_cplx0_dish32_polr1 = -W_m3 * E_cplx0_dish32_polr1
                    E2_cplx1_dish32_polr1 = -W_m3 * E_cplx1_dish32_polr1
                    E2_cplx0_dish33_polr1 = -W_m3 * E_cplx0_dish33_polr1
                    E2_cplx1_dish33_polr1 = -W_m3 * E_cplx1_dish33_polr1
                    F_ringbuf_m0_dish0_polr0 = F_ringbuf_dish0_mtaps0_polr0
                    F_ringbuf_m1_dish0_polr0 = F_ringbuf_dish0_mtaps1_polr0
                    F_ringbuf_m2_dish0_polr0 = F_ringbuf_dish0_mtaps2_polr0
                    F_ringbuf_m0_dish32_polr0 = F_ringbuf_dish32_mtaps0_polr0
                    F_ringbuf_m1_dish32_polr0 = F_ringbuf_dish32_mtaps1_polr0
                    F_ringbuf_m2_dish32_polr0 = F_ringbuf_dish32_mtaps2_polr0
                    F_ringbuf_m0_dish0_polr1 = F_ringbuf_dish0_mtaps0_polr1
                    F_ringbuf_m1_dish0_polr1 = F_ringbuf_dish0_mtaps1_polr1
                    F_ringbuf_m2_dish0_polr1 = F_ringbuf_dish0_mtaps2_polr1
                    F_ringbuf_m0_dish32_polr1 = F_ringbuf_dish32_mtaps0_polr1
                    F_ringbuf_m1_dish32_polr1 = F_ringbuf_dish32_mtaps1_polr1
                    F_ringbuf_m2_dish32_polr1 = F_ringbuf_dish32_mtaps2_polr1
                    (E_ringbuf_m0_cplx0_dish0_polr0, E_ringbuf_m0_cplx1_dish0_polr0, E_ringbuf_m0_cplx0_dish1_polr0, E_ringbuf_m0_cplx1_dish1_polr0) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m0_dish0_polr0
                    )
                    (E_ringbuf_m0_cplx0_dish32_polr0, E_ringbuf_m0_cplx1_dish32_polr0, E_ringbuf_m0_cplx0_dish33_polr0, E_ringbuf_m0_cplx1_dish33_polr0) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m0_dish32_polr0
                    )
                    (E_ringbuf_m0_cplx0_dish0_polr1, E_ringbuf_m0_cplx1_dish0_polr1, E_ringbuf_m0_cplx0_dish1_polr1, E_ringbuf_m0_cplx1_dish1_polr1) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m0_dish0_polr1
                    )
                    (E_ringbuf_m0_cplx0_dish32_polr1, E_ringbuf_m0_cplx1_dish32_polr1, E_ringbuf_m0_cplx0_dish33_polr1, E_ringbuf_m0_cplx1_dish33_polr1) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m0_dish32_polr1
                    )
                    E2_cplx0_dish0_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish0_polr0, E2_cplx0_dish0_polr0)
                    E2_cplx1_dish0_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish0_polr0, E2_cplx1_dish0_polr0)
                    E2_cplx0_dish1_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish1_polr0, E2_cplx0_dish1_polr0)
                    E2_cplx1_dish1_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish1_polr0, E2_cplx1_dish1_polr0)
                    E2_cplx0_dish32_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish32_polr0, E2_cplx0_dish32_polr0)
                    E2_cplx1_dish32_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish32_polr0, E2_cplx1_dish32_polr0)
                    E2_cplx0_dish33_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish33_polr0, E2_cplx0_dish33_polr0)
                    E2_cplx1_dish33_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish33_polr0, E2_cplx1_dish33_polr0)
                    E2_cplx0_dish0_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish0_polr1, E2_cplx0_dish0_polr1)
                    E2_cplx1_dish0_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish0_polr1, E2_cplx1_dish0_polr1)
                    E2_cplx0_dish1_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish1_polr1, E2_cplx0_dish1_polr1)
                    E2_cplx1_dish1_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish1_polr1, E2_cplx1_dish1_polr1)
                    E2_cplx0_dish32_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish32_polr1, E2_cplx0_dish32_polr1)
                    E2_cplx1_dish32_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish32_polr1, E2_cplx1_dish32_polr1)
                    E2_cplx0_dish33_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish33_polr1, E2_cplx0_dish33_polr1)
                    E2_cplx1_dish33_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish33_polr1, E2_cplx1_dish33_polr1)
                    (E_ringbuf_m1_cplx0_dish0_polr0, E_ringbuf_m1_cplx1_dish0_polr0, E_ringbuf_m1_cplx0_dish1_polr0, E_ringbuf_m1_cplx1_dish1_polr0) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m1_dish0_polr0
                    )
                    (E_ringbuf_m1_cplx0_dish32_polr0, E_ringbuf_m1_cplx1_dish32_polr0, E_ringbuf_m1_cplx0_dish33_polr0, E_ringbuf_m1_cplx1_dish33_polr0) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m1_dish32_polr0
                    )
                    (E_ringbuf_m1_cplx0_dish0_polr1, E_ringbuf_m1_cplx1_dish0_polr1, E_ringbuf_m1_cplx0_dish1_polr1, E_ringbuf_m1_cplx1_dish1_polr1) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m1_dish0_polr1
                    )
                    (E_ringbuf_m1_cplx0_dish32_polr1, E_ringbuf_m1_cplx1_dish32_polr1, E_ringbuf_m1_cplx0_dish33_polr1, E_ringbuf_m1_cplx1_dish33_polr1) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m1_dish32_polr1
                    )
                    E2_cplx0_dish0_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish0_polr0, E2_cplx0_dish0_polr0)
                    E2_cplx1_dish0_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish0_polr0, E2_cplx1_dish0_polr0)
                    E2_cplx0_dish1_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish1_polr0, E2_cplx0_dish1_polr0)
                    E2_cplx1_dish1_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish1_polr0, E2_cplx1_dish1_polr0)
                    E2_cplx0_dish32_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish32_polr0, E2_cplx0_dish32_polr0)
                    E2_cplx1_dish32_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish32_polr0, E2_cplx1_dish32_polr0)
                    E2_cplx0_dish33_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish33_polr0, E2_cplx0_dish33_polr0)
                    E2_cplx1_dish33_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish33_polr0, E2_cplx1_dish33_polr0)
                    E2_cplx0_dish0_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish0_polr1, E2_cplx0_dish0_polr1)
                    E2_cplx1_dish0_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish0_polr1, E2_cplx1_dish0_polr1)
                    E2_cplx0_dish1_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish1_polr1, E2_cplx0_dish1_polr1)
                    E2_cplx1_dish1_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish1_polr1, E2_cplx1_dish1_polr1)
                    E2_cplx0_dish32_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish32_polr1, E2_cplx0_dish32_polr1)
                    E2_cplx1_dish32_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish32_polr1, E2_cplx1_dish32_polr1)
                    E2_cplx0_dish33_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish33_polr1, E2_cplx0_dish33_polr1)
                    E2_cplx1_dish33_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish33_polr1, E2_cplx1_dish33_polr1)
                    (E_ringbuf_m2_cplx0_dish0_polr0, E_ringbuf_m2_cplx1_dish0_polr0, E_ringbuf_m2_cplx0_dish1_polr0, E_ringbuf_m2_cplx1_dish1_polr0) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m2_dish0_polr0
                    )
                    (E_ringbuf_m2_cplx0_dish32_polr0, E_ringbuf_m2_cplx1_dish32_polr0, E_ringbuf_m2_cplx0_dish33_polr0, E_ringbuf_m2_cplx1_dish33_polr0) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m2_dish32_polr0
                    )
                    (E_ringbuf_m2_cplx0_dish0_polr1, E_ringbuf_m2_cplx1_dish0_polr1, E_ringbuf_m2_cplx0_dish1_polr1, E_ringbuf_m2_cplx1_dish1_polr1) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m2_dish0_polr1
                    )
                    (E_ringbuf_m2_cplx0_dish32_polr1, E_ringbuf_m2_cplx1_dish32_polr1, E_ringbuf_m2_cplx0_dish33_polr1, E_ringbuf_m2_cplx1_dish33_polr1) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m2_dish32_polr1
                    )
                    E2_cplx0_dish0_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish0_polr0, E2_cplx0_dish0_polr0)
                    E2_cplx1_dish0_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish0_polr0, E2_cplx1_dish0_polr0)
                    E2_cplx0_dish1_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish1_polr0, E2_cplx0_dish1_polr0)
                    E2_cplx1_dish1_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish1_polr0, E2_cplx1_dish1_polr0)
                    E2_cplx0_dish32_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish32_polr0, E2_cplx0_dish32_polr0)
                    E2_cplx1_dish32_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish32_polr0, E2_cplx1_dish32_polr0)
                    E2_cplx0_dish33_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish33_polr0, E2_cplx0_dish33_polr0)
                    E2_cplx1_dish33_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish33_polr0, E2_cplx1_dish33_polr0)
                    E2_cplx0_dish0_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish0_polr1, E2_cplx0_dish0_polr1)
                    E2_cplx1_dish0_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish0_polr1, E2_cplx1_dish0_polr1)
                    E2_cplx0_dish1_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish1_polr1, E2_cplx0_dish1_polr1)
                    E2_cplx1_dish1_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish1_polr1, E2_cplx1_dish1_polr1)
                    E2_cplx0_dish32_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish32_polr1, E2_cplx0_dish32_polr1)
                    E2_cplx1_dish32_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish32_polr1, E2_cplx1_dish32_polr1)
                    E2_cplx0_dish33_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish33_polr1, E2_cplx0_dish33_polr1)
                    E2_cplx1_dish33_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish33_polr1, E2_cplx1_dish33_polr1)
                    E2re_dish0_polr0 = E2_cplx0_dish0_polr0
                    E2im_dish0_polr0 = E2_cplx1_dish0_polr0
                    E2re_dish1_polr0 = E2_cplx0_dish1_polr0
                    E2im_dish1_polr0 = E2_cplx1_dish1_polr0
                    E2re_dish32_polr0 = E2_cplx0_dish32_polr0
                    E2im_dish32_polr0 = E2_cplx1_dish32_polr0
                    E2re_dish33_polr0 = E2_cplx0_dish33_polr0
                    E2im_dish33_polr0 = E2_cplx1_dish33_polr0
                    E2re_dish0_polr1 = E2_cplx0_dish0_polr1
                    E2im_dish0_polr1 = E2_cplx1_dish0_polr1
                    E2re_dish1_polr1 = E2_cplx0_dish1_polr1
                    E2im_dish1_polr1 = E2_cplx1_dish1_polr1
                    E2re_dish32_polr1 = E2_cplx0_dish32_polr1
                    E2im_dish32_polr1 = E2_cplx1_dish32_polr1
                    E2re_dish33_polr1 = E2_cplx0_dish33_polr1
                    E2im_dish33_polr1 = E2_cplx1_dish33_polr1
                    Xre = X_cplx0
                    Xim = X_cplx1
                    E3re_dish0_polr0 = muladd(Xre, E2re_dish0_polr0, -Xim * E2im_dish0_polr0)
                    E3re_dish1_polr0 = muladd(Xre, E2re_dish1_polr0, -Xim * E2im_dish1_polr0)
                    E3re_dish32_polr0 = muladd(Xre, E2re_dish32_polr0, -Xim * E2im_dish32_polr0)
                    E3re_dish33_polr0 = muladd(Xre, E2re_dish33_polr0, -Xim * E2im_dish33_polr0)
                    E3re_dish0_polr1 = muladd(Xre, E2re_dish0_polr1, -Xim * E2im_dish0_polr1)
                    E3re_dish1_polr1 = muladd(Xre, E2re_dish1_polr1, -Xim * E2im_dish1_polr1)
                    E3re_dish32_polr1 = muladd(Xre, E2re_dish32_polr1, -Xim * E2im_dish32_polr1)
                    E3re_dish33_polr1 = muladd(Xre, E2re_dish33_polr1, -Xim * E2im_dish33_polr1)
                    E3im_dish0_polr0 = muladd(Xre, E2im_dish0_polr0, Xim * E2re_dish0_polr0)
                    E3im_dish1_polr0 = muladd(Xre, E2im_dish1_polr0, Xim * E2re_dish1_polr0)
                    E3im_dish32_polr0 = muladd(Xre, E2im_dish32_polr0, Xim * E2re_dish32_polr0)
                    E3im_dish33_polr0 = muladd(Xre, E2im_dish33_polr0, Xim * E2re_dish33_polr0)
                    E3im_dish0_polr1 = muladd(Xre, E2im_dish0_polr1, Xim * E2re_dish0_polr1)
                    E3im_dish1_polr1 = muladd(Xre, E2im_dish1_polr1, Xim * E2re_dish1_polr1)
                    E3im_dish32_polr1 = muladd(Xre, E2im_dish32_polr1, Xim * E2re_dish32_polr1)
                    E3im_dish33_polr1 = muladd(Xre, E2im_dish33_polr1, Xim * E2re_dish33_polr1)
                    E3_cplx0_dish0_polr0 = E3re_dish0_polr0
                    E3_cplx1_dish0_polr0 = E3im_dish0_polr0
                    E3_cplx0_dish1_polr0 = E3re_dish1_polr0
                    E3_cplx1_dish1_polr0 = E3im_dish1_polr0
                    E3_cplx0_dish32_polr0 = E3re_dish32_polr0
                    E3_cplx1_dish32_polr0 = E3im_dish32_polr0
                    E3_cplx0_dish33_polr0 = E3re_dish33_polr0
                    E3_cplx1_dish33_polr0 = E3im_dish33_polr0
                    E3_cplx0_dish0_polr1 = E3re_dish0_polr1
                    E3_cplx1_dish0_polr1 = E3im_dish0_polr1
                    E3_cplx0_dish1_polr1 = E3re_dish1_polr1
                    E3_cplx1_dish1_polr1 = E3im_dish1_polr1
                    E3_cplx0_dish32_polr1 = E3re_dish32_polr1
                    E3_cplx1_dish32_polr1 = E3im_dish32_polr1
                    E3_cplx0_dish33_polr1 = E3re_dish33_polr1
                    E3_cplx1_dish33_polr1 = E3im_dish33_polr1
                    XX_cplx0_dish0_polr0 = E3_cplx0_dish0_polr0
                    XX_cplx1_dish0_polr0 = E3_cplx1_dish0_polr0
                    XX_cplx0_dish1_polr0 = E3_cplx0_dish1_polr0
                    XX_cplx1_dish1_polr0 = E3_cplx1_dish1_polr0
                    XX_cplx0_dish32_polr0 = E3_cplx0_dish32_polr0
                    XX_cplx1_dish32_polr0 = E3_cplx1_dish32_polr0
                    XX_cplx0_dish33_polr0 = E3_cplx0_dish33_polr0
                    XX_cplx1_dish33_polr0 = E3_cplx1_dish33_polr0
                    XX_cplx0_dish0_polr1 = E3_cplx0_dish0_polr1
                    XX_cplx1_dish0_polr1 = E3_cplx1_dish0_polr1
                    XX_cplx0_dish1_polr1 = E3_cplx0_dish1_polr1
                    XX_cplx1_dish1_polr1 = E3_cplx1_dish1_polr1
                    XX_cplx0_dish32_polr1 = E3_cplx0_dish32_polr1
                    XX_cplx1_dish32_polr1 = E3_cplx1_dish32_polr1
                    XX_cplx0_dish33_polr1 = E3_cplx0_dish33_polr1
                    XX_cplx1_dish33_polr1 = E3_cplx1_dish33_polr1
                    XXre_dish0_polr0 = XX_cplx0_dish0_polr0
                    XXim_dish0_polr0 = XX_cplx1_dish0_polr0
                    XXre_dish1_polr0 = XX_cplx0_dish1_polr0
                    XXim_dish1_polr0 = XX_cplx1_dish1_polr0
                    XXre_dish32_polr0 = XX_cplx0_dish32_polr0
                    XXim_dish32_polr0 = XX_cplx1_dish32_polr0
                    XXre_dish33_polr0 = XX_cplx0_dish33_polr0
                    XXim_dish33_polr0 = XX_cplx1_dish33_polr0
                    XXre_dish0_polr1 = XX_cplx0_dish0_polr1
                    XXim_dish0_polr1 = XX_cplx1_dish0_polr1
                    XXre_dish1_polr1 = XX_cplx0_dish1_polr1
                    XXim_dish1_polr1 = XX_cplx1_dish1_polr1
                    XXre_dish32_polr1 = XX_cplx0_dish32_polr1
                    XXim_dish32_polr1 = XX_cplx1_dish32_polr1
                    XXre_dish33_polr1 = XX_cplx0_dish33_polr1
                    XXim_dish33_polr1 = XX_cplx1_dish33_polr1
                    XX_cplx_in0_dish0_polr0 = XXre_dish0_polr0
                    XX_cplx_in1_dish0_polr0 = XXim_dish0_polr0
                    XX_cplx_in0_dish1_polr0 = XXre_dish1_polr0
                    XX_cplx_in1_dish1_polr0 = XXim_dish1_polr0
                    XX_cplx_in0_dish32_polr0 = XXre_dish32_polr0
                    XX_cplx_in1_dish32_polr0 = XXim_dish32_polr0
                    XX_cplx_in0_dish33_polr0 = XXre_dish33_polr0
                    XX_cplx_in1_dish33_polr0 = XXim_dish33_polr0
                    XX_cplx_in0_dish0_polr1 = XXre_dish0_polr1
                    XX_cplx_in1_dish0_polr1 = XXim_dish0_polr1
                    XX_cplx_in0_dish1_polr1 = XXre_dish1_polr1
                    XX_cplx_in1_dish1_polr1 = XXim_dish1_polr1
                    XX_cplx_in0_dish32_polr1 = XXre_dish32_polr1
                    XX_cplx_in1_dish32_polr1 = XXim_dish32_polr1
                    XX_cplx_in0_dish33_polr1 = XXre_dish33_polr1
                    XX_cplx_in1_dish33_polr1 = XXim_dish33_polr1
                    WW_cplx0_dish0_polr0 = zero(Float16x2)
                    WW_cplx1_dish0_polr0 = zero(Float16x2)
                    WW_cplx0_dish1_polr0 = zero(Float16x2)
                    WW_cplx1_dish1_polr0 = zero(Float16x2)
                    WW_cplx0_dish32_polr0 = zero(Float16x2)
                    WW_cplx1_dish32_polr0 = zero(Float16x2)
                    WW_cplx0_dish33_polr0 = zero(Float16x2)
                    WW_cplx1_dish33_polr0 = zero(Float16x2)
                    WW_cplx0_dish0_polr1 = zero(Float16x2)
                    WW_cplx1_dish0_polr1 = zero(Float16x2)
                    WW_cplx0_dish1_polr1 = zero(Float16x2)
                    WW_cplx1_dish1_polr1 = zero(Float16x2)
                    WW_cplx0_dish32_polr1 = zero(Float16x2)
                    WW_cplx1_dish32_polr1 = zero(Float16x2)
                    WW_cplx0_dish33_polr1 = zero(Float16x2)
                    WW_cplx1_dish33_polr1 = zero(Float16x2)
                    (WW_cplx0_dish0_polr0, WW_cplx1_dish0_polr0) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish0_polr0, XX_cplx_in1_dish0_polr0),
                        (WW_cplx0_dish0_polr0, WW_cplx1_dish0_polr0),
                    )
                    (WW_cplx0_dish1_polr0, WW_cplx1_dish1_polr0) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish1_polr0, XX_cplx_in1_dish1_polr0),
                        (WW_cplx0_dish1_polr0, WW_cplx1_dish1_polr0),
                    )
                    (WW_cplx0_dish32_polr0, WW_cplx1_dish32_polr0) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish32_polr0, XX_cplx_in1_dish32_polr0),
                        (WW_cplx0_dish32_polr0, WW_cplx1_dish32_polr0),
                    )
                    (WW_cplx0_dish33_polr0, WW_cplx1_dish33_polr0) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish33_polr0, XX_cplx_in1_dish33_polr0),
                        (WW_cplx0_dish33_polr0, WW_cplx1_dish33_polr0),
                    )
                    (WW_cplx0_dish0_polr1, WW_cplx1_dish0_polr1) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish0_polr1, XX_cplx_in1_dish0_polr1),
                        (WW_cplx0_dish0_polr1, WW_cplx1_dish0_polr1),
                    )
                    (WW_cplx0_dish1_polr1, WW_cplx1_dish1_polr1) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish1_polr1, XX_cplx_in1_dish1_polr1),
                        (WW_cplx0_dish1_polr1, WW_cplx1_dish1_polr1),
                    )
                    (WW_cplx0_dish32_polr1, WW_cplx1_dish32_polr1) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish32_polr1, XX_cplx_in1_dish32_polr1),
                        (WW_cplx0_dish32_polr1, WW_cplx1_dish32_polr1),
                    )
                    (WW_cplx0_dish33_polr1, WW_cplx1_dish33_polr1) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish33_polr1, XX_cplx_in1_dish33_polr1),
                        (WW_cplx0_dish33_polr1, WW_cplx1_dish33_polr1),
                    )
                    Γ²re = Γ²_cplx0
                    Γ²im = Γ²_cplx1
                    WWre_dish0_polr0 = WW_cplx0_dish0_polr0
                    WWim_dish0_polr0 = WW_cplx1_dish0_polr0
                    WWre_dish1_polr0 = WW_cplx0_dish1_polr0
                    WWim_dish1_polr0 = WW_cplx1_dish1_polr0
                    WWre_dish32_polr0 = WW_cplx0_dish32_polr0
                    WWim_dish32_polr0 = WW_cplx1_dish32_polr0
                    WWre_dish33_polr0 = WW_cplx0_dish33_polr0
                    WWim_dish33_polr0 = WW_cplx1_dish33_polr0
                    WWre_dish0_polr1 = WW_cplx0_dish0_polr1
                    WWim_dish0_polr1 = WW_cplx1_dish0_polr1
                    WWre_dish1_polr1 = WW_cplx0_dish1_polr1
                    WWim_dish1_polr1 = WW_cplx1_dish1_polr1
                    WWre_dish32_polr1 = WW_cplx0_dish32_polr1
                    WWim_dish32_polr1 = WW_cplx1_dish32_polr1
                    WWre_dish33_polr1 = WW_cplx0_dish33_polr1
                    WWim_dish33_polr1 = WW_cplx1_dish33_polr1
                    ZZre_dish0_polr0 = muladd(Γ²re, WWre_dish0_polr0, -Γ²im * WWim_dish0_polr0)
                    ZZre_dish1_polr0 = muladd(Γ²re, WWre_dish1_polr0, -Γ²im * WWim_dish1_polr0)
                    ZZre_dish32_polr0 = muladd(Γ²re, WWre_dish32_polr0, -Γ²im * WWim_dish32_polr0)
                    ZZre_dish33_polr0 = muladd(Γ²re, WWre_dish33_polr0, -Γ²im * WWim_dish33_polr0)
                    ZZre_dish0_polr1 = muladd(Γ²re, WWre_dish0_polr1, -Γ²im * WWim_dish0_polr1)
                    ZZre_dish1_polr1 = muladd(Γ²re, WWre_dish1_polr1, -Γ²im * WWim_dish1_polr1)
                    ZZre_dish32_polr1 = muladd(Γ²re, WWre_dish32_polr1, -Γ²im * WWim_dish32_polr1)
                    ZZre_dish33_polr1 = muladd(Γ²re, WWre_dish33_polr1, -Γ²im * WWim_dish33_polr1)
                    ZZim_dish0_polr0 = muladd(Γ²re, WWim_dish0_polr0, Γ²im * WWre_dish0_polr0)
                    ZZim_dish1_polr0 = muladd(Γ²re, WWim_dish1_polr0, Γ²im * WWre_dish1_polr0)
                    ZZim_dish32_polr0 = muladd(Γ²re, WWim_dish32_polr0, Γ²im * WWre_dish32_polr0)
                    ZZim_dish33_polr0 = muladd(Γ²re, WWim_dish33_polr0, Γ²im * WWre_dish33_polr0)
                    ZZim_dish0_polr1 = muladd(Γ²re, WWim_dish0_polr1, Γ²im * WWre_dish0_polr1)
                    ZZim_dish1_polr1 = muladd(Γ²re, WWim_dish1_polr1, Γ²im * WWre_dish1_polr1)
                    ZZim_dish32_polr1 = muladd(Γ²re, WWim_dish32_polr1, Γ²im * WWre_dish32_polr1)
                    ZZim_dish33_polr1 = muladd(Γ²re, WWim_dish33_polr1, Γ²im * WWre_dish33_polr1)
                    ZZ_cplx0_dish0_polr0 = ZZre_dish0_polr0
                    ZZ_cplx1_dish0_polr0 = ZZim_dish0_polr0
                    ZZ_cplx0_dish1_polr0 = ZZre_dish1_polr0
                    ZZ_cplx1_dish1_polr0 = ZZim_dish1_polr0
                    ZZ_cplx0_dish32_polr0 = ZZre_dish32_polr0
                    ZZ_cplx1_dish32_polr0 = ZZim_dish32_polr0
                    ZZ_cplx0_dish33_polr0 = ZZre_dish33_polr0
                    ZZ_cplx1_dish33_polr0 = ZZim_dish33_polr0
                    ZZ_cplx0_dish0_polr1 = ZZre_dish0_polr1
                    ZZ_cplx1_dish0_polr1 = ZZim_dish0_polr1
                    ZZ_cplx0_dish1_polr1 = ZZre_dish1_polr1
                    ZZ_cplx1_dish1_polr1 = ZZim_dish1_polr1
                    ZZ_cplx0_dish32_polr1 = ZZre_dish32_polr1
                    ZZ_cplx1_dish32_polr1 = ZZim_dish32_polr1
                    ZZ_cplx0_dish33_polr1 = ZZre_dish33_polr1
                    ZZ_cplx1_dish33_polr1 = ZZim_dish33_polr1
                    ZZre_dish0_polr0 = ZZ_cplx0_dish0_polr0
                    ZZim_dish0_polr0 = ZZ_cplx1_dish0_polr0
                    ZZre_dish1_polr0 = ZZ_cplx0_dish1_polr0
                    ZZim_dish1_polr0 = ZZ_cplx1_dish1_polr0
                    ZZre_dish32_polr0 = ZZ_cplx0_dish32_polr0
                    ZZim_dish32_polr0 = ZZ_cplx1_dish32_polr0
                    ZZre_dish33_polr0 = ZZ_cplx0_dish33_polr0
                    ZZim_dish33_polr0 = ZZ_cplx1_dish33_polr0
                    ZZre_dish0_polr1 = ZZ_cplx0_dish0_polr1
                    ZZim_dish0_polr1 = ZZ_cplx1_dish0_polr1
                    ZZre_dish1_polr1 = ZZ_cplx0_dish1_polr1
                    ZZim_dish1_polr1 = ZZ_cplx1_dish1_polr1
                    ZZre_dish32_polr1 = ZZ_cplx0_dish32_polr1
                    ZZim_dish32_polr1 = ZZ_cplx1_dish32_polr1
                    ZZre_dish33_polr1 = ZZ_cplx0_dish33_polr1
                    ZZim_dish33_polr1 = ZZ_cplx1_dish33_polr1
                    ZZ_cplx_in0_dish0_polr0 = ZZre_dish0_polr0
                    ZZ_cplx_in1_dish0_polr0 = ZZim_dish0_polr0
                    ZZ_cplx_in0_dish1_polr0 = ZZre_dish1_polr0
                    ZZ_cplx_in1_dish1_polr0 = ZZim_dish1_polr0
                    ZZ_cplx_in0_dish32_polr0 = ZZre_dish32_polr0
                    ZZ_cplx_in1_dish32_polr0 = ZZim_dish32_polr0
                    ZZ_cplx_in0_dish33_polr0 = ZZre_dish33_polr0
                    ZZ_cplx_in1_dish33_polr0 = ZZim_dish33_polr0
                    ZZ_cplx_in0_dish0_polr1 = ZZre_dish0_polr1
                    ZZ_cplx_in1_dish0_polr1 = ZZim_dish0_polr1
                    ZZ_cplx_in0_dish1_polr1 = ZZre_dish1_polr1
                    ZZ_cplx_in1_dish1_polr1 = ZZim_dish1_polr1
                    ZZ_cplx_in0_dish32_polr1 = ZZre_dish32_polr1
                    ZZ_cplx_in1_dish32_polr1 = ZZim_dish32_polr1
                    ZZ_cplx_in0_dish33_polr1 = ZZre_dish33_polr1
                    ZZ_cplx_in1_dish33_polr1 = ZZim_dish33_polr1
                    YY_cplx0_dish0_polr0 = zero(Float16x2)
                    YY_cplx1_dish0_polr0 = zero(Float16x2)
                    YY_cplx0_dish1_polr0 = zero(Float16x2)
                    YY_cplx1_dish1_polr0 = zero(Float16x2)
                    YY_cplx0_dish32_polr0 = zero(Float16x2)
                    YY_cplx1_dish32_polr0 = zero(Float16x2)
                    YY_cplx0_dish33_polr0 = zero(Float16x2)
                    YY_cplx1_dish33_polr0 = zero(Float16x2)
                    YY_cplx0_dish0_polr1 = zero(Float16x2)
                    YY_cplx1_dish0_polr1 = zero(Float16x2)
                    YY_cplx0_dish1_polr1 = zero(Float16x2)
                    YY_cplx1_dish1_polr1 = zero(Float16x2)
                    YY_cplx0_dish32_polr1 = zero(Float16x2)
                    YY_cplx1_dish32_polr1 = zero(Float16x2)
                    YY_cplx0_dish33_polr1 = zero(Float16x2)
                    YY_cplx1_dish33_polr1 = zero(Float16x2)
                    (YY_cplx0_dish0_polr0, YY_cplx1_dish0_polr0) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish0_polr0,
                            Γ³_cplx1_cplx_in0_dish0_polr0,
                            Γ³_cplx0_cplx_in1_dish0_polr0,
                            Γ³_cplx1_cplx_in1_dish0_polr0,
                        ),
                        (ZZ_cplx_in0_dish0_polr0, ZZ_cplx_in1_dish0_polr0),
                        (YY_cplx0_dish0_polr0, YY_cplx1_dish0_polr0),
                    )
                    (YY_cplx0_dish1_polr0, YY_cplx1_dish1_polr0) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish1_polr0,
                            Γ³_cplx1_cplx_in0_dish1_polr0,
                            Γ³_cplx0_cplx_in1_dish1_polr0,
                            Γ³_cplx1_cplx_in1_dish1_polr0,
                        ),
                        (ZZ_cplx_in0_dish1_polr0, ZZ_cplx_in1_dish1_polr0),
                        (YY_cplx0_dish1_polr0, YY_cplx1_dish1_polr0),
                    )
                    (YY_cplx0_dish32_polr0, YY_cplx1_dish32_polr0) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish32_polr0,
                            Γ³_cplx1_cplx_in0_dish32_polr0,
                            Γ³_cplx0_cplx_in1_dish32_polr0,
                            Γ³_cplx1_cplx_in1_dish32_polr0,
                        ),
                        (ZZ_cplx_in0_dish32_polr0, ZZ_cplx_in1_dish32_polr0),
                        (YY_cplx0_dish32_polr0, YY_cplx1_dish32_polr0),
                    )
                    (YY_cplx0_dish33_polr0, YY_cplx1_dish33_polr0) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish33_polr0,
                            Γ³_cplx1_cplx_in0_dish33_polr0,
                            Γ³_cplx0_cplx_in1_dish33_polr0,
                            Γ³_cplx1_cplx_in1_dish33_polr0,
                        ),
                        (ZZ_cplx_in0_dish33_polr0, ZZ_cplx_in1_dish33_polr0),
                        (YY_cplx0_dish33_polr0, YY_cplx1_dish33_polr0),
                    )
                    (YY_cplx0_dish0_polr1, YY_cplx1_dish0_polr1) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish0_polr1,
                            Γ³_cplx1_cplx_in0_dish0_polr1,
                            Γ³_cplx0_cplx_in1_dish0_polr1,
                            Γ³_cplx1_cplx_in1_dish0_polr1,
                        ),
                        (ZZ_cplx_in0_dish0_polr1, ZZ_cplx_in1_dish0_polr1),
                        (YY_cplx0_dish0_polr1, YY_cplx1_dish0_polr1),
                    )
                    (YY_cplx0_dish1_polr1, YY_cplx1_dish1_polr1) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish1_polr1,
                            Γ³_cplx1_cplx_in0_dish1_polr1,
                            Γ³_cplx0_cplx_in1_dish1_polr1,
                            Γ³_cplx1_cplx_in1_dish1_polr1,
                        ),
                        (ZZ_cplx_in0_dish1_polr1, ZZ_cplx_in1_dish1_polr1),
                        (YY_cplx0_dish1_polr1, YY_cplx1_dish1_polr1),
                    )
                    (YY_cplx0_dish32_polr1, YY_cplx1_dish32_polr1) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish32_polr1,
                            Γ³_cplx1_cplx_in0_dish32_polr1,
                            Γ³_cplx0_cplx_in1_dish32_polr1,
                            Γ³_cplx1_cplx_in1_dish32_polr1,
                        ),
                        (ZZ_cplx_in0_dish32_polr1, ZZ_cplx_in1_dish32_polr1),
                        (YY_cplx0_dish32_polr1, YY_cplx1_dish32_polr1),
                    )
                    (YY_cplx0_dish33_polr1, YY_cplx1_dish33_polr1) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish33_polr1,
                            Γ³_cplx1_cplx_in0_dish33_polr1,
                            Γ³_cplx0_cplx_in1_dish33_polr1,
                            Γ³_cplx1_cplx_in1_dish33_polr1,
                        ),
                        (ZZ_cplx_in0_dish33_polr1, ZZ_cplx_in1_dish33_polr1),
                        (YY_cplx0_dish33_polr1, YY_cplx1_dish33_polr1),
                    )
                    E4_cplx0_dish0_polr0 = YY_cplx0_dish0_polr0
                    E4_cplx1_dish0_polr0 = YY_cplx1_dish0_polr0
                    E4_cplx0_dish1_polr0 = YY_cplx0_dish1_polr0
                    E4_cplx1_dish1_polr0 = YY_cplx1_dish1_polr0
                    E4_cplx0_dish32_polr0 = YY_cplx0_dish32_polr0
                    E4_cplx1_dish32_polr0 = YY_cplx1_dish32_polr0
                    E4_cplx0_dish33_polr0 = YY_cplx0_dish33_polr0
                    E4_cplx1_dish33_polr0 = YY_cplx1_dish33_polr0
                    E4_cplx0_dish0_polr1 = YY_cplx0_dish0_polr1
                    E4_cplx1_dish0_polr1 = YY_cplx1_dish0_polr1
                    E4_cplx0_dish1_polr1 = YY_cplx0_dish1_polr1
                    E4_cplx1_dish1_polr1 = YY_cplx1_dish1_polr1
                    E4_cplx0_dish32_polr1 = YY_cplx0_dish32_polr1
                    E4_cplx1_dish32_polr1 = YY_cplx1_dish32_polr1
                    E4_cplx0_dish33_polr1 = YY_cplx0_dish33_polr1
                    E4_cplx1_dish33_polr1 = YY_cplx1_dish33_polr1
                    E5_cplx0_dish0_polr0 = Gains * E4_cplx0_dish0_polr0
                    E5_cplx1_dish0_polr0 = Gains * E4_cplx1_dish0_polr0
                    E5_cplx0_dish1_polr0 = Gains * E4_cplx0_dish1_polr0
                    E5_cplx1_dish1_polr0 = Gains * E4_cplx1_dish1_polr0
                    E5_cplx0_dish32_polr0 = Gains * E4_cplx0_dish32_polr0
                    E5_cplx1_dish32_polr0 = Gains * E4_cplx1_dish32_polr0
                    E5_cplx0_dish33_polr0 = Gains * E4_cplx0_dish33_polr0
                    E5_cplx1_dish33_polr0 = Gains * E4_cplx1_dish33_polr0
                    E5_cplx0_dish0_polr1 = Gains * E4_cplx0_dish0_polr1
                    E5_cplx1_dish0_polr1 = Gains * E4_cplx1_dish0_polr1
                    E5_cplx0_dish1_polr1 = Gains * E4_cplx0_dish1_polr1
                    E5_cplx1_dish1_polr1 = Gains * E4_cplx1_dish1_polr1
                    E5_cplx0_dish32_polr1 = Gains * E4_cplx0_dish32_polr1
                    E5_cplx1_dish32_polr1 = Gains * E4_cplx1_dish32_polr1
                    E5_cplx0_dish33_polr1 = Gains * E4_cplx0_dish33_polr1
                    E5_cplx1_dish33_polr1 = Gains * E4_cplx1_dish33_polr1
                    E5_cplx0_dish0_polr0 = clamp(E5_cplx0_dish0_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish0_polr0 = clamp(E5_cplx1_dish0_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish1_polr0 = clamp(E5_cplx0_dish1_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish1_polr0 = clamp(E5_cplx1_dish1_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish32_polr0 = clamp(E5_cplx0_dish32_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish32_polr0 = clamp(E5_cplx1_dish32_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish33_polr0 = clamp(E5_cplx0_dish33_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish33_polr0 = clamp(E5_cplx1_dish33_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish0_polr1 = clamp(E5_cplx0_dish0_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish0_polr1 = clamp(E5_cplx1_dish0_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish1_polr1 = clamp(E5_cplx0_dish1_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish1_polr1 = clamp(E5_cplx1_dish1_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish32_polr1 = clamp(E5_cplx0_dish32_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish32_polr1 = clamp(E5_cplx1_dish32_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish33_polr1 = clamp(E5_cplx0_dish33_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish33_polr1 = clamp(E5_cplx1_dish33_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    F̄_out_dish0_polr0 = Int4x8((
                        E5_cplx0_dish0_polr0, E5_cplx1_dish0_polr0, E5_cplx0_dish1_polr0, E5_cplx1_dish1_polr0
                    ))
                    F̄_out_dish32_polr0 = Int4x8((
                        E5_cplx0_dish32_polr0, E5_cplx1_dish32_polr0, E5_cplx0_dish33_polr0, E5_cplx1_dish33_polr0
                    ))
                    F̄_out_dish0_polr1 = Int4x8((
                        E5_cplx0_dish0_polr1, E5_cplx1_dish0_polr1, E5_cplx0_dish1_polr1, E5_cplx1_dish1_polr1
                    ))
                    F̄_out_dish32_polr1 = Int4x8((
                        E5_cplx0_dish32_polr1, E5_cplx1_dish32_polr1, E5_cplx0_dish33_polr1, E5_cplx1_dish33_polr1
                    ))
                    if true
                        F̄_shared[(((((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 8) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 2) * 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) * 64) ÷ 2) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 32 + 0 + (((((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 16) + 0) + 0x01] =
                            F̄_out_dish0_polr0
                    end
                    if true
                        F̄_shared[(((((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 8) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 2) * 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) * 64) ÷ 2) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 2) % 2) * 32 + 0 + (((((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 4) % 16) + 0) + 0x01] =
                            F̄_out_dish32_polr0
                    end
                    if true
                        F̄_shared[(((((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 8) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 2) * 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) * 64) ÷ 2) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 32 + 16 + (((((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 16) + 0) + 0x01] =
                            F̄_out_dish0_polr1
                    end
                    if true
                        F̄_shared[(((((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 8) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 2) * 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) * 64) ÷ 2) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 2) % 2) * 32 + 16 + (((((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 4) % 16) + 0) + 0x01] =
                            F̄_out_dish32_polr1
                    end
                    F_ringbuf_m0_dish0_polr0 = F_ringbuf_dish0_mtaps0_polr0
                    F_ringbuf_m1_dish0_polr0 = F_ringbuf_dish0_mtaps1_polr0
                    F_ringbuf_m2_dish0_polr0 = F_ringbuf_dish0_mtaps2_polr0
                    F_ringbuf_m0_dish32_polr0 = F_ringbuf_dish32_mtaps0_polr0
                    F_ringbuf_m1_dish32_polr0 = F_ringbuf_dish32_mtaps1_polr0
                    F_ringbuf_m2_dish32_polr0 = F_ringbuf_dish32_mtaps2_polr0
                    F_ringbuf_m0_dish0_polr1 = F_ringbuf_dish0_mtaps0_polr1
                    F_ringbuf_m1_dish0_polr1 = F_ringbuf_dish0_mtaps1_polr1
                    F_ringbuf_m2_dish0_polr1 = F_ringbuf_dish0_mtaps2_polr1
                    F_ringbuf_m0_dish32_polr1 = F_ringbuf_dish32_mtaps0_polr1
                    F_ringbuf_m1_dish32_polr1 = F_ringbuf_dish32_mtaps1_polr1
                    F_ringbuf_m2_dish32_polr1 = F_ringbuf_dish32_mtaps2_polr1
                    F_ringbuf_m0_dish0_polr0 = F_ringbuf_m1_dish0_polr0
                    F_ringbuf_m0_dish32_polr0 = F_ringbuf_m1_dish32_polr0
                    F_ringbuf_m0_dish0_polr1 = F_ringbuf_m1_dish0_polr1
                    F_ringbuf_m0_dish32_polr1 = F_ringbuf_m1_dish32_polr1
                    F_ringbuf_m1_dish0_polr0 = F_ringbuf_m2_dish0_polr0
                    F_ringbuf_m1_dish32_polr0 = F_ringbuf_m2_dish32_polr0
                    F_ringbuf_m1_dish0_polr1 = F_ringbuf_m2_dish0_polr1
                    F_ringbuf_m1_dish32_polr1 = F_ringbuf_m2_dish32_polr1
                    F_ringbuf_m2_dish0_polr0 = F_in_dish0_polr0
                    F_ringbuf_m2_dish32_polr0 = F_in_dish32_polr0
                    F_ringbuf_m2_dish0_polr1 = F_in_dish0_polr1
                    F_ringbuf_m2_dish32_polr1 = F_in_dish32_polr1
                    F_ringbuf_dish0_mtaps0_polr0 = F_ringbuf_m0_dish0_polr0
                    F_ringbuf_dish0_mtaps1_polr0 = F_ringbuf_m1_dish0_polr0
                    F_ringbuf_dish0_mtaps2_polr0 = F_ringbuf_m2_dish0_polr0
                    F_ringbuf_dish32_mtaps0_polr0 = F_ringbuf_m0_dish32_polr0
                    F_ringbuf_dish32_mtaps1_polr0 = F_ringbuf_m1_dish32_polr0
                    F_ringbuf_dish32_mtaps2_polr0 = F_ringbuf_m2_dish32_polr0
                    F_ringbuf_dish0_mtaps0_polr1 = F_ringbuf_m0_dish0_polr1
                    F_ringbuf_dish0_mtaps1_polr1 = F_ringbuf_m1_dish0_polr1
                    F_ringbuf_dish0_mtaps2_polr1 = F_ringbuf_m2_dish0_polr1
                    F_ringbuf_dish32_mtaps0_polr1 = F_ringbuf_m0_dish32_polr1
                    F_ringbuf_dish32_mtaps1_polr1 = F_ringbuf_m1_dish32_polr1
                    F_ringbuf_dish32_mtaps2_polr1 = F_ringbuf_m2_dish32_polr1
                end
                let
                    dish = 16
                    F_in_dish0_polr0 = F_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 32 + 0 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 16) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) % 2) * 1040 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 2) % 2) * 520 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 4) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 8) % 2) * 130) + 0x01]
                    F_in_dish32_polr0 = F_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 2) % 2) * 32 + 0 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 16) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) % 2) * 1040 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 2) % 2) * 520 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 4) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 8) % 2) * 130) + 0x01]
                    F_in_dish0_polr1 = F_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 32 + 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 16) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) % 2) * 1040 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 2) % 2) * 520 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 4) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 8) % 2) * 130) + 0x01]
                    F_in_dish32_polr1 = F_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 2) % 2) * 32 + 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 16) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) % 2) * 1040 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 2) % 2) * 520 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 4) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 8) % 2) * 130) + 0x01]
                    (E_cplx0_dish0_polr0, E_cplx1_dish0_polr0, E_cplx0_dish1_polr0, E_cplx1_dish1_polr0) = convert(
                        NTuple{4,Float16x2}, F_in_dish0_polr0
                    )
                    (E_cplx0_dish32_polr0, E_cplx1_dish32_polr0, E_cplx0_dish33_polr0, E_cplx1_dish33_polr0) = convert(
                        NTuple{4,Float16x2}, F_in_dish32_polr0
                    )
                    (E_cplx0_dish0_polr1, E_cplx1_dish0_polr1, E_cplx0_dish1_polr1, E_cplx1_dish1_polr1) = convert(
                        NTuple{4,Float16x2}, F_in_dish0_polr1
                    )
                    (E_cplx0_dish32_polr1, E_cplx1_dish32_polr1, E_cplx0_dish33_polr1, E_cplx1_dish33_polr1) = convert(
                        NTuple{4,Float16x2}, F_in_dish32_polr1
                    )
                    W_m0 = Wpfb_mtaps0
                    W_m1 = Wpfb_mtaps1
                    W_m2 = Wpfb_mtaps2
                    W_m3 = Wpfb_mtaps3
                    E2_cplx0_dish0_polr0 = -W_m3 * E_cplx0_dish0_polr0
                    E2_cplx1_dish0_polr0 = -W_m3 * E_cplx1_dish0_polr0
                    E2_cplx0_dish1_polr0 = -W_m3 * E_cplx0_dish1_polr0
                    E2_cplx1_dish1_polr0 = -W_m3 * E_cplx1_dish1_polr0
                    E2_cplx0_dish32_polr0 = -W_m3 * E_cplx0_dish32_polr0
                    E2_cplx1_dish32_polr0 = -W_m3 * E_cplx1_dish32_polr0
                    E2_cplx0_dish33_polr0 = -W_m3 * E_cplx0_dish33_polr0
                    E2_cplx1_dish33_polr0 = -W_m3 * E_cplx1_dish33_polr0
                    E2_cplx0_dish0_polr1 = -W_m3 * E_cplx0_dish0_polr1
                    E2_cplx1_dish0_polr1 = -W_m3 * E_cplx1_dish0_polr1
                    E2_cplx0_dish1_polr1 = -W_m3 * E_cplx0_dish1_polr1
                    E2_cplx1_dish1_polr1 = -W_m3 * E_cplx1_dish1_polr1
                    E2_cplx0_dish32_polr1 = -W_m3 * E_cplx0_dish32_polr1
                    E2_cplx1_dish32_polr1 = -W_m3 * E_cplx1_dish32_polr1
                    E2_cplx0_dish33_polr1 = -W_m3 * E_cplx0_dish33_polr1
                    E2_cplx1_dish33_polr1 = -W_m3 * E_cplx1_dish33_polr1
                    F_ringbuf_m0_dish0_polr0 = F_ringbuf_dish0_mtaps0_polr0
                    F_ringbuf_m1_dish0_polr0 = F_ringbuf_dish0_mtaps1_polr0
                    F_ringbuf_m2_dish0_polr0 = F_ringbuf_dish0_mtaps2_polr0
                    F_ringbuf_m0_dish32_polr0 = F_ringbuf_dish32_mtaps0_polr0
                    F_ringbuf_m1_dish32_polr0 = F_ringbuf_dish32_mtaps1_polr0
                    F_ringbuf_m2_dish32_polr0 = F_ringbuf_dish32_mtaps2_polr0
                    F_ringbuf_m0_dish0_polr1 = F_ringbuf_dish0_mtaps0_polr1
                    F_ringbuf_m1_dish0_polr1 = F_ringbuf_dish0_mtaps1_polr1
                    F_ringbuf_m2_dish0_polr1 = F_ringbuf_dish0_mtaps2_polr1
                    F_ringbuf_m0_dish32_polr1 = F_ringbuf_dish32_mtaps0_polr1
                    F_ringbuf_m1_dish32_polr1 = F_ringbuf_dish32_mtaps1_polr1
                    F_ringbuf_m2_dish32_polr1 = F_ringbuf_dish32_mtaps2_polr1
                    (E_ringbuf_m0_cplx0_dish0_polr0, E_ringbuf_m0_cplx1_dish0_polr0, E_ringbuf_m0_cplx0_dish1_polr0, E_ringbuf_m0_cplx1_dish1_polr0) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m0_dish0_polr0
                    )
                    (E_ringbuf_m0_cplx0_dish32_polr0, E_ringbuf_m0_cplx1_dish32_polr0, E_ringbuf_m0_cplx0_dish33_polr0, E_ringbuf_m0_cplx1_dish33_polr0) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m0_dish32_polr0
                    )
                    (E_ringbuf_m0_cplx0_dish0_polr1, E_ringbuf_m0_cplx1_dish0_polr1, E_ringbuf_m0_cplx0_dish1_polr1, E_ringbuf_m0_cplx1_dish1_polr1) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m0_dish0_polr1
                    )
                    (E_ringbuf_m0_cplx0_dish32_polr1, E_ringbuf_m0_cplx1_dish32_polr1, E_ringbuf_m0_cplx0_dish33_polr1, E_ringbuf_m0_cplx1_dish33_polr1) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m0_dish32_polr1
                    )
                    E2_cplx0_dish0_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish0_polr0, E2_cplx0_dish0_polr0)
                    E2_cplx1_dish0_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish0_polr0, E2_cplx1_dish0_polr0)
                    E2_cplx0_dish1_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish1_polr0, E2_cplx0_dish1_polr0)
                    E2_cplx1_dish1_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish1_polr0, E2_cplx1_dish1_polr0)
                    E2_cplx0_dish32_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish32_polr0, E2_cplx0_dish32_polr0)
                    E2_cplx1_dish32_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish32_polr0, E2_cplx1_dish32_polr0)
                    E2_cplx0_dish33_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish33_polr0, E2_cplx0_dish33_polr0)
                    E2_cplx1_dish33_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish33_polr0, E2_cplx1_dish33_polr0)
                    E2_cplx0_dish0_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish0_polr1, E2_cplx0_dish0_polr1)
                    E2_cplx1_dish0_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish0_polr1, E2_cplx1_dish0_polr1)
                    E2_cplx0_dish1_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish1_polr1, E2_cplx0_dish1_polr1)
                    E2_cplx1_dish1_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish1_polr1, E2_cplx1_dish1_polr1)
                    E2_cplx0_dish32_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish32_polr1, E2_cplx0_dish32_polr1)
                    E2_cplx1_dish32_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish32_polr1, E2_cplx1_dish32_polr1)
                    E2_cplx0_dish33_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish33_polr1, E2_cplx0_dish33_polr1)
                    E2_cplx1_dish33_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish33_polr1, E2_cplx1_dish33_polr1)
                    (E_ringbuf_m1_cplx0_dish0_polr0, E_ringbuf_m1_cplx1_dish0_polr0, E_ringbuf_m1_cplx0_dish1_polr0, E_ringbuf_m1_cplx1_dish1_polr0) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m1_dish0_polr0
                    )
                    (E_ringbuf_m1_cplx0_dish32_polr0, E_ringbuf_m1_cplx1_dish32_polr0, E_ringbuf_m1_cplx0_dish33_polr0, E_ringbuf_m1_cplx1_dish33_polr0) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m1_dish32_polr0
                    )
                    (E_ringbuf_m1_cplx0_dish0_polr1, E_ringbuf_m1_cplx1_dish0_polr1, E_ringbuf_m1_cplx0_dish1_polr1, E_ringbuf_m1_cplx1_dish1_polr1) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m1_dish0_polr1
                    )
                    (E_ringbuf_m1_cplx0_dish32_polr1, E_ringbuf_m1_cplx1_dish32_polr1, E_ringbuf_m1_cplx0_dish33_polr1, E_ringbuf_m1_cplx1_dish33_polr1) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m1_dish32_polr1
                    )
                    E2_cplx0_dish0_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish0_polr0, E2_cplx0_dish0_polr0)
                    E2_cplx1_dish0_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish0_polr0, E2_cplx1_dish0_polr0)
                    E2_cplx0_dish1_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish1_polr0, E2_cplx0_dish1_polr0)
                    E2_cplx1_dish1_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish1_polr0, E2_cplx1_dish1_polr0)
                    E2_cplx0_dish32_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish32_polr0, E2_cplx0_dish32_polr0)
                    E2_cplx1_dish32_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish32_polr0, E2_cplx1_dish32_polr0)
                    E2_cplx0_dish33_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish33_polr0, E2_cplx0_dish33_polr0)
                    E2_cplx1_dish33_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish33_polr0, E2_cplx1_dish33_polr0)
                    E2_cplx0_dish0_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish0_polr1, E2_cplx0_dish0_polr1)
                    E2_cplx1_dish0_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish0_polr1, E2_cplx1_dish0_polr1)
                    E2_cplx0_dish1_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish1_polr1, E2_cplx0_dish1_polr1)
                    E2_cplx1_dish1_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish1_polr1, E2_cplx1_dish1_polr1)
                    E2_cplx0_dish32_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish32_polr1, E2_cplx0_dish32_polr1)
                    E2_cplx1_dish32_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish32_polr1, E2_cplx1_dish32_polr1)
                    E2_cplx0_dish33_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish33_polr1, E2_cplx0_dish33_polr1)
                    E2_cplx1_dish33_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish33_polr1, E2_cplx1_dish33_polr1)
                    (E_ringbuf_m2_cplx0_dish0_polr0, E_ringbuf_m2_cplx1_dish0_polr0, E_ringbuf_m2_cplx0_dish1_polr0, E_ringbuf_m2_cplx1_dish1_polr0) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m2_dish0_polr0
                    )
                    (E_ringbuf_m2_cplx0_dish32_polr0, E_ringbuf_m2_cplx1_dish32_polr0, E_ringbuf_m2_cplx0_dish33_polr0, E_ringbuf_m2_cplx1_dish33_polr0) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m2_dish32_polr0
                    )
                    (E_ringbuf_m2_cplx0_dish0_polr1, E_ringbuf_m2_cplx1_dish0_polr1, E_ringbuf_m2_cplx0_dish1_polr1, E_ringbuf_m2_cplx1_dish1_polr1) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m2_dish0_polr1
                    )
                    (E_ringbuf_m2_cplx0_dish32_polr1, E_ringbuf_m2_cplx1_dish32_polr1, E_ringbuf_m2_cplx0_dish33_polr1, E_ringbuf_m2_cplx1_dish33_polr1) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m2_dish32_polr1
                    )
                    E2_cplx0_dish0_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish0_polr0, E2_cplx0_dish0_polr0)
                    E2_cplx1_dish0_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish0_polr0, E2_cplx1_dish0_polr0)
                    E2_cplx0_dish1_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish1_polr0, E2_cplx0_dish1_polr0)
                    E2_cplx1_dish1_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish1_polr0, E2_cplx1_dish1_polr0)
                    E2_cplx0_dish32_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish32_polr0, E2_cplx0_dish32_polr0)
                    E2_cplx1_dish32_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish32_polr0, E2_cplx1_dish32_polr0)
                    E2_cplx0_dish33_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish33_polr0, E2_cplx0_dish33_polr0)
                    E2_cplx1_dish33_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish33_polr0, E2_cplx1_dish33_polr0)
                    E2_cplx0_dish0_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish0_polr1, E2_cplx0_dish0_polr1)
                    E2_cplx1_dish0_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish0_polr1, E2_cplx1_dish0_polr1)
                    E2_cplx0_dish1_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish1_polr1, E2_cplx0_dish1_polr1)
                    E2_cplx1_dish1_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish1_polr1, E2_cplx1_dish1_polr1)
                    E2_cplx0_dish32_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish32_polr1, E2_cplx0_dish32_polr1)
                    E2_cplx1_dish32_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish32_polr1, E2_cplx1_dish32_polr1)
                    E2_cplx0_dish33_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish33_polr1, E2_cplx0_dish33_polr1)
                    E2_cplx1_dish33_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish33_polr1, E2_cplx1_dish33_polr1)
                    E2re_dish0_polr0 = E2_cplx0_dish0_polr0
                    E2im_dish0_polr0 = E2_cplx1_dish0_polr0
                    E2re_dish1_polr0 = E2_cplx0_dish1_polr0
                    E2im_dish1_polr0 = E2_cplx1_dish1_polr0
                    E2re_dish32_polr0 = E2_cplx0_dish32_polr0
                    E2im_dish32_polr0 = E2_cplx1_dish32_polr0
                    E2re_dish33_polr0 = E2_cplx0_dish33_polr0
                    E2im_dish33_polr0 = E2_cplx1_dish33_polr0
                    E2re_dish0_polr1 = E2_cplx0_dish0_polr1
                    E2im_dish0_polr1 = E2_cplx1_dish0_polr1
                    E2re_dish1_polr1 = E2_cplx0_dish1_polr1
                    E2im_dish1_polr1 = E2_cplx1_dish1_polr1
                    E2re_dish32_polr1 = E2_cplx0_dish32_polr1
                    E2im_dish32_polr1 = E2_cplx1_dish32_polr1
                    E2re_dish33_polr1 = E2_cplx0_dish33_polr1
                    E2im_dish33_polr1 = E2_cplx1_dish33_polr1
                    Xre = X_cplx0
                    Xim = X_cplx1
                    E3re_dish0_polr0 = muladd(Xre, E2re_dish0_polr0, -Xim * E2im_dish0_polr0)
                    E3re_dish1_polr0 = muladd(Xre, E2re_dish1_polr0, -Xim * E2im_dish1_polr0)
                    E3re_dish32_polr0 = muladd(Xre, E2re_dish32_polr0, -Xim * E2im_dish32_polr0)
                    E3re_dish33_polr0 = muladd(Xre, E2re_dish33_polr0, -Xim * E2im_dish33_polr0)
                    E3re_dish0_polr1 = muladd(Xre, E2re_dish0_polr1, -Xim * E2im_dish0_polr1)
                    E3re_dish1_polr1 = muladd(Xre, E2re_dish1_polr1, -Xim * E2im_dish1_polr1)
                    E3re_dish32_polr1 = muladd(Xre, E2re_dish32_polr1, -Xim * E2im_dish32_polr1)
                    E3re_dish33_polr1 = muladd(Xre, E2re_dish33_polr1, -Xim * E2im_dish33_polr1)
                    E3im_dish0_polr0 = muladd(Xre, E2im_dish0_polr0, Xim * E2re_dish0_polr0)
                    E3im_dish1_polr0 = muladd(Xre, E2im_dish1_polr0, Xim * E2re_dish1_polr0)
                    E3im_dish32_polr0 = muladd(Xre, E2im_dish32_polr0, Xim * E2re_dish32_polr0)
                    E3im_dish33_polr0 = muladd(Xre, E2im_dish33_polr0, Xim * E2re_dish33_polr0)
                    E3im_dish0_polr1 = muladd(Xre, E2im_dish0_polr1, Xim * E2re_dish0_polr1)
                    E3im_dish1_polr1 = muladd(Xre, E2im_dish1_polr1, Xim * E2re_dish1_polr1)
                    E3im_dish32_polr1 = muladd(Xre, E2im_dish32_polr1, Xim * E2re_dish32_polr1)
                    E3im_dish33_polr1 = muladd(Xre, E2im_dish33_polr1, Xim * E2re_dish33_polr1)
                    E3_cplx0_dish0_polr0 = E3re_dish0_polr0
                    E3_cplx1_dish0_polr0 = E3im_dish0_polr0
                    E3_cplx0_dish1_polr0 = E3re_dish1_polr0
                    E3_cplx1_dish1_polr0 = E3im_dish1_polr0
                    E3_cplx0_dish32_polr0 = E3re_dish32_polr0
                    E3_cplx1_dish32_polr0 = E3im_dish32_polr0
                    E3_cplx0_dish33_polr0 = E3re_dish33_polr0
                    E3_cplx1_dish33_polr0 = E3im_dish33_polr0
                    E3_cplx0_dish0_polr1 = E3re_dish0_polr1
                    E3_cplx1_dish0_polr1 = E3im_dish0_polr1
                    E3_cplx0_dish1_polr1 = E3re_dish1_polr1
                    E3_cplx1_dish1_polr1 = E3im_dish1_polr1
                    E3_cplx0_dish32_polr1 = E3re_dish32_polr1
                    E3_cplx1_dish32_polr1 = E3im_dish32_polr1
                    E3_cplx0_dish33_polr1 = E3re_dish33_polr1
                    E3_cplx1_dish33_polr1 = E3im_dish33_polr1
                    XX_cplx0_dish0_polr0 = E3_cplx0_dish0_polr0
                    XX_cplx1_dish0_polr0 = E3_cplx1_dish0_polr0
                    XX_cplx0_dish1_polr0 = E3_cplx0_dish1_polr0
                    XX_cplx1_dish1_polr0 = E3_cplx1_dish1_polr0
                    XX_cplx0_dish32_polr0 = E3_cplx0_dish32_polr0
                    XX_cplx1_dish32_polr0 = E3_cplx1_dish32_polr0
                    XX_cplx0_dish33_polr0 = E3_cplx0_dish33_polr0
                    XX_cplx1_dish33_polr0 = E3_cplx1_dish33_polr0
                    XX_cplx0_dish0_polr1 = E3_cplx0_dish0_polr1
                    XX_cplx1_dish0_polr1 = E3_cplx1_dish0_polr1
                    XX_cplx0_dish1_polr1 = E3_cplx0_dish1_polr1
                    XX_cplx1_dish1_polr1 = E3_cplx1_dish1_polr1
                    XX_cplx0_dish32_polr1 = E3_cplx0_dish32_polr1
                    XX_cplx1_dish32_polr1 = E3_cplx1_dish32_polr1
                    XX_cplx0_dish33_polr1 = E3_cplx0_dish33_polr1
                    XX_cplx1_dish33_polr1 = E3_cplx1_dish33_polr1
                    XXre_dish0_polr0 = XX_cplx0_dish0_polr0
                    XXim_dish0_polr0 = XX_cplx1_dish0_polr0
                    XXre_dish1_polr0 = XX_cplx0_dish1_polr0
                    XXim_dish1_polr0 = XX_cplx1_dish1_polr0
                    XXre_dish32_polr0 = XX_cplx0_dish32_polr0
                    XXim_dish32_polr0 = XX_cplx1_dish32_polr0
                    XXre_dish33_polr0 = XX_cplx0_dish33_polr0
                    XXim_dish33_polr0 = XX_cplx1_dish33_polr0
                    XXre_dish0_polr1 = XX_cplx0_dish0_polr1
                    XXim_dish0_polr1 = XX_cplx1_dish0_polr1
                    XXre_dish1_polr1 = XX_cplx0_dish1_polr1
                    XXim_dish1_polr1 = XX_cplx1_dish1_polr1
                    XXre_dish32_polr1 = XX_cplx0_dish32_polr1
                    XXim_dish32_polr1 = XX_cplx1_dish32_polr1
                    XXre_dish33_polr1 = XX_cplx0_dish33_polr1
                    XXim_dish33_polr1 = XX_cplx1_dish33_polr1
                    XX_cplx_in0_dish0_polr0 = XXre_dish0_polr0
                    XX_cplx_in1_dish0_polr0 = XXim_dish0_polr0
                    XX_cplx_in0_dish1_polr0 = XXre_dish1_polr0
                    XX_cplx_in1_dish1_polr0 = XXim_dish1_polr0
                    XX_cplx_in0_dish32_polr0 = XXre_dish32_polr0
                    XX_cplx_in1_dish32_polr0 = XXim_dish32_polr0
                    XX_cplx_in0_dish33_polr0 = XXre_dish33_polr0
                    XX_cplx_in1_dish33_polr0 = XXim_dish33_polr0
                    XX_cplx_in0_dish0_polr1 = XXre_dish0_polr1
                    XX_cplx_in1_dish0_polr1 = XXim_dish0_polr1
                    XX_cplx_in0_dish1_polr1 = XXre_dish1_polr1
                    XX_cplx_in1_dish1_polr1 = XXim_dish1_polr1
                    XX_cplx_in0_dish32_polr1 = XXre_dish32_polr1
                    XX_cplx_in1_dish32_polr1 = XXim_dish32_polr1
                    XX_cplx_in0_dish33_polr1 = XXre_dish33_polr1
                    XX_cplx_in1_dish33_polr1 = XXim_dish33_polr1
                    WW_cplx0_dish0_polr0 = zero(Float16x2)
                    WW_cplx1_dish0_polr0 = zero(Float16x2)
                    WW_cplx0_dish1_polr0 = zero(Float16x2)
                    WW_cplx1_dish1_polr0 = zero(Float16x2)
                    WW_cplx0_dish32_polr0 = zero(Float16x2)
                    WW_cplx1_dish32_polr0 = zero(Float16x2)
                    WW_cplx0_dish33_polr0 = zero(Float16x2)
                    WW_cplx1_dish33_polr0 = zero(Float16x2)
                    WW_cplx0_dish0_polr1 = zero(Float16x2)
                    WW_cplx1_dish0_polr1 = zero(Float16x2)
                    WW_cplx0_dish1_polr1 = zero(Float16x2)
                    WW_cplx1_dish1_polr1 = zero(Float16x2)
                    WW_cplx0_dish32_polr1 = zero(Float16x2)
                    WW_cplx1_dish32_polr1 = zero(Float16x2)
                    WW_cplx0_dish33_polr1 = zero(Float16x2)
                    WW_cplx1_dish33_polr1 = zero(Float16x2)
                    (WW_cplx0_dish0_polr0, WW_cplx1_dish0_polr0) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish0_polr0, XX_cplx_in1_dish0_polr0),
                        (WW_cplx0_dish0_polr0, WW_cplx1_dish0_polr0),
                    )
                    (WW_cplx0_dish1_polr0, WW_cplx1_dish1_polr0) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish1_polr0, XX_cplx_in1_dish1_polr0),
                        (WW_cplx0_dish1_polr0, WW_cplx1_dish1_polr0),
                    )
                    (WW_cplx0_dish32_polr0, WW_cplx1_dish32_polr0) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish32_polr0, XX_cplx_in1_dish32_polr0),
                        (WW_cplx0_dish32_polr0, WW_cplx1_dish32_polr0),
                    )
                    (WW_cplx0_dish33_polr0, WW_cplx1_dish33_polr0) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish33_polr0, XX_cplx_in1_dish33_polr0),
                        (WW_cplx0_dish33_polr0, WW_cplx1_dish33_polr0),
                    )
                    (WW_cplx0_dish0_polr1, WW_cplx1_dish0_polr1) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish0_polr1, XX_cplx_in1_dish0_polr1),
                        (WW_cplx0_dish0_polr1, WW_cplx1_dish0_polr1),
                    )
                    (WW_cplx0_dish1_polr1, WW_cplx1_dish1_polr1) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish1_polr1, XX_cplx_in1_dish1_polr1),
                        (WW_cplx0_dish1_polr1, WW_cplx1_dish1_polr1),
                    )
                    (WW_cplx0_dish32_polr1, WW_cplx1_dish32_polr1) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish32_polr1, XX_cplx_in1_dish32_polr1),
                        (WW_cplx0_dish32_polr1, WW_cplx1_dish32_polr1),
                    )
                    (WW_cplx0_dish33_polr1, WW_cplx1_dish33_polr1) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish33_polr1, XX_cplx_in1_dish33_polr1),
                        (WW_cplx0_dish33_polr1, WW_cplx1_dish33_polr1),
                    )
                    Γ²re = Γ²_cplx0
                    Γ²im = Γ²_cplx1
                    WWre_dish0_polr0 = WW_cplx0_dish0_polr0
                    WWim_dish0_polr0 = WW_cplx1_dish0_polr0
                    WWre_dish1_polr0 = WW_cplx0_dish1_polr0
                    WWim_dish1_polr0 = WW_cplx1_dish1_polr0
                    WWre_dish32_polr0 = WW_cplx0_dish32_polr0
                    WWim_dish32_polr0 = WW_cplx1_dish32_polr0
                    WWre_dish33_polr0 = WW_cplx0_dish33_polr0
                    WWim_dish33_polr0 = WW_cplx1_dish33_polr0
                    WWre_dish0_polr1 = WW_cplx0_dish0_polr1
                    WWim_dish0_polr1 = WW_cplx1_dish0_polr1
                    WWre_dish1_polr1 = WW_cplx0_dish1_polr1
                    WWim_dish1_polr1 = WW_cplx1_dish1_polr1
                    WWre_dish32_polr1 = WW_cplx0_dish32_polr1
                    WWim_dish32_polr1 = WW_cplx1_dish32_polr1
                    WWre_dish33_polr1 = WW_cplx0_dish33_polr1
                    WWim_dish33_polr1 = WW_cplx1_dish33_polr1
                    ZZre_dish0_polr0 = muladd(Γ²re, WWre_dish0_polr0, -Γ²im * WWim_dish0_polr0)
                    ZZre_dish1_polr0 = muladd(Γ²re, WWre_dish1_polr0, -Γ²im * WWim_dish1_polr0)
                    ZZre_dish32_polr0 = muladd(Γ²re, WWre_dish32_polr0, -Γ²im * WWim_dish32_polr0)
                    ZZre_dish33_polr0 = muladd(Γ²re, WWre_dish33_polr0, -Γ²im * WWim_dish33_polr0)
                    ZZre_dish0_polr1 = muladd(Γ²re, WWre_dish0_polr1, -Γ²im * WWim_dish0_polr1)
                    ZZre_dish1_polr1 = muladd(Γ²re, WWre_dish1_polr1, -Γ²im * WWim_dish1_polr1)
                    ZZre_dish32_polr1 = muladd(Γ²re, WWre_dish32_polr1, -Γ²im * WWim_dish32_polr1)
                    ZZre_dish33_polr1 = muladd(Γ²re, WWre_dish33_polr1, -Γ²im * WWim_dish33_polr1)
                    ZZim_dish0_polr0 = muladd(Γ²re, WWim_dish0_polr0, Γ²im * WWre_dish0_polr0)
                    ZZim_dish1_polr0 = muladd(Γ²re, WWim_dish1_polr0, Γ²im * WWre_dish1_polr0)
                    ZZim_dish32_polr0 = muladd(Γ²re, WWim_dish32_polr0, Γ²im * WWre_dish32_polr0)
                    ZZim_dish33_polr0 = muladd(Γ²re, WWim_dish33_polr0, Γ²im * WWre_dish33_polr0)
                    ZZim_dish0_polr1 = muladd(Γ²re, WWim_dish0_polr1, Γ²im * WWre_dish0_polr1)
                    ZZim_dish1_polr1 = muladd(Γ²re, WWim_dish1_polr1, Γ²im * WWre_dish1_polr1)
                    ZZim_dish32_polr1 = muladd(Γ²re, WWim_dish32_polr1, Γ²im * WWre_dish32_polr1)
                    ZZim_dish33_polr1 = muladd(Γ²re, WWim_dish33_polr1, Γ²im * WWre_dish33_polr1)
                    ZZ_cplx0_dish0_polr0 = ZZre_dish0_polr0
                    ZZ_cplx1_dish0_polr0 = ZZim_dish0_polr0
                    ZZ_cplx0_dish1_polr0 = ZZre_dish1_polr0
                    ZZ_cplx1_dish1_polr0 = ZZim_dish1_polr0
                    ZZ_cplx0_dish32_polr0 = ZZre_dish32_polr0
                    ZZ_cplx1_dish32_polr0 = ZZim_dish32_polr0
                    ZZ_cplx0_dish33_polr0 = ZZre_dish33_polr0
                    ZZ_cplx1_dish33_polr0 = ZZim_dish33_polr0
                    ZZ_cplx0_dish0_polr1 = ZZre_dish0_polr1
                    ZZ_cplx1_dish0_polr1 = ZZim_dish0_polr1
                    ZZ_cplx0_dish1_polr1 = ZZre_dish1_polr1
                    ZZ_cplx1_dish1_polr1 = ZZim_dish1_polr1
                    ZZ_cplx0_dish32_polr1 = ZZre_dish32_polr1
                    ZZ_cplx1_dish32_polr1 = ZZim_dish32_polr1
                    ZZ_cplx0_dish33_polr1 = ZZre_dish33_polr1
                    ZZ_cplx1_dish33_polr1 = ZZim_dish33_polr1
                    ZZre_dish0_polr0 = ZZ_cplx0_dish0_polr0
                    ZZim_dish0_polr0 = ZZ_cplx1_dish0_polr0
                    ZZre_dish1_polr0 = ZZ_cplx0_dish1_polr0
                    ZZim_dish1_polr0 = ZZ_cplx1_dish1_polr0
                    ZZre_dish32_polr0 = ZZ_cplx0_dish32_polr0
                    ZZim_dish32_polr0 = ZZ_cplx1_dish32_polr0
                    ZZre_dish33_polr0 = ZZ_cplx0_dish33_polr0
                    ZZim_dish33_polr0 = ZZ_cplx1_dish33_polr0
                    ZZre_dish0_polr1 = ZZ_cplx0_dish0_polr1
                    ZZim_dish0_polr1 = ZZ_cplx1_dish0_polr1
                    ZZre_dish1_polr1 = ZZ_cplx0_dish1_polr1
                    ZZim_dish1_polr1 = ZZ_cplx1_dish1_polr1
                    ZZre_dish32_polr1 = ZZ_cplx0_dish32_polr1
                    ZZim_dish32_polr1 = ZZ_cplx1_dish32_polr1
                    ZZre_dish33_polr1 = ZZ_cplx0_dish33_polr1
                    ZZim_dish33_polr1 = ZZ_cplx1_dish33_polr1
                    ZZ_cplx_in0_dish0_polr0 = ZZre_dish0_polr0
                    ZZ_cplx_in1_dish0_polr0 = ZZim_dish0_polr0
                    ZZ_cplx_in0_dish1_polr0 = ZZre_dish1_polr0
                    ZZ_cplx_in1_dish1_polr0 = ZZim_dish1_polr0
                    ZZ_cplx_in0_dish32_polr0 = ZZre_dish32_polr0
                    ZZ_cplx_in1_dish32_polr0 = ZZim_dish32_polr0
                    ZZ_cplx_in0_dish33_polr0 = ZZre_dish33_polr0
                    ZZ_cplx_in1_dish33_polr0 = ZZim_dish33_polr0
                    ZZ_cplx_in0_dish0_polr1 = ZZre_dish0_polr1
                    ZZ_cplx_in1_dish0_polr1 = ZZim_dish0_polr1
                    ZZ_cplx_in0_dish1_polr1 = ZZre_dish1_polr1
                    ZZ_cplx_in1_dish1_polr1 = ZZim_dish1_polr1
                    ZZ_cplx_in0_dish32_polr1 = ZZre_dish32_polr1
                    ZZ_cplx_in1_dish32_polr1 = ZZim_dish32_polr1
                    ZZ_cplx_in0_dish33_polr1 = ZZre_dish33_polr1
                    ZZ_cplx_in1_dish33_polr1 = ZZim_dish33_polr1
                    YY_cplx0_dish0_polr0 = zero(Float16x2)
                    YY_cplx1_dish0_polr0 = zero(Float16x2)
                    YY_cplx0_dish1_polr0 = zero(Float16x2)
                    YY_cplx1_dish1_polr0 = zero(Float16x2)
                    YY_cplx0_dish32_polr0 = zero(Float16x2)
                    YY_cplx1_dish32_polr0 = zero(Float16x2)
                    YY_cplx0_dish33_polr0 = zero(Float16x2)
                    YY_cplx1_dish33_polr0 = zero(Float16x2)
                    YY_cplx0_dish0_polr1 = zero(Float16x2)
                    YY_cplx1_dish0_polr1 = zero(Float16x2)
                    YY_cplx0_dish1_polr1 = zero(Float16x2)
                    YY_cplx1_dish1_polr1 = zero(Float16x2)
                    YY_cplx0_dish32_polr1 = zero(Float16x2)
                    YY_cplx1_dish32_polr1 = zero(Float16x2)
                    YY_cplx0_dish33_polr1 = zero(Float16x2)
                    YY_cplx1_dish33_polr1 = zero(Float16x2)
                    (YY_cplx0_dish0_polr0, YY_cplx1_dish0_polr0) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish0_polr0,
                            Γ³_cplx1_cplx_in0_dish0_polr0,
                            Γ³_cplx0_cplx_in1_dish0_polr0,
                            Γ³_cplx1_cplx_in1_dish0_polr0,
                        ),
                        (ZZ_cplx_in0_dish0_polr0, ZZ_cplx_in1_dish0_polr0),
                        (YY_cplx0_dish0_polr0, YY_cplx1_dish0_polr0),
                    )
                    (YY_cplx0_dish1_polr0, YY_cplx1_dish1_polr0) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish1_polr0,
                            Γ³_cplx1_cplx_in0_dish1_polr0,
                            Γ³_cplx0_cplx_in1_dish1_polr0,
                            Γ³_cplx1_cplx_in1_dish1_polr0,
                        ),
                        (ZZ_cplx_in0_dish1_polr0, ZZ_cplx_in1_dish1_polr0),
                        (YY_cplx0_dish1_polr0, YY_cplx1_dish1_polr0),
                    )
                    (YY_cplx0_dish32_polr0, YY_cplx1_dish32_polr0) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish32_polr0,
                            Γ³_cplx1_cplx_in0_dish32_polr0,
                            Γ³_cplx0_cplx_in1_dish32_polr0,
                            Γ³_cplx1_cplx_in1_dish32_polr0,
                        ),
                        (ZZ_cplx_in0_dish32_polr0, ZZ_cplx_in1_dish32_polr0),
                        (YY_cplx0_dish32_polr0, YY_cplx1_dish32_polr0),
                    )
                    (YY_cplx0_dish33_polr0, YY_cplx1_dish33_polr0) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish33_polr0,
                            Γ³_cplx1_cplx_in0_dish33_polr0,
                            Γ³_cplx0_cplx_in1_dish33_polr0,
                            Γ³_cplx1_cplx_in1_dish33_polr0,
                        ),
                        (ZZ_cplx_in0_dish33_polr0, ZZ_cplx_in1_dish33_polr0),
                        (YY_cplx0_dish33_polr0, YY_cplx1_dish33_polr0),
                    )
                    (YY_cplx0_dish0_polr1, YY_cplx1_dish0_polr1) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish0_polr1,
                            Γ³_cplx1_cplx_in0_dish0_polr1,
                            Γ³_cplx0_cplx_in1_dish0_polr1,
                            Γ³_cplx1_cplx_in1_dish0_polr1,
                        ),
                        (ZZ_cplx_in0_dish0_polr1, ZZ_cplx_in1_dish0_polr1),
                        (YY_cplx0_dish0_polr1, YY_cplx1_dish0_polr1),
                    )
                    (YY_cplx0_dish1_polr1, YY_cplx1_dish1_polr1) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish1_polr1,
                            Γ³_cplx1_cplx_in0_dish1_polr1,
                            Γ³_cplx0_cplx_in1_dish1_polr1,
                            Γ³_cplx1_cplx_in1_dish1_polr1,
                        ),
                        (ZZ_cplx_in0_dish1_polr1, ZZ_cplx_in1_dish1_polr1),
                        (YY_cplx0_dish1_polr1, YY_cplx1_dish1_polr1),
                    )
                    (YY_cplx0_dish32_polr1, YY_cplx1_dish32_polr1) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish32_polr1,
                            Γ³_cplx1_cplx_in0_dish32_polr1,
                            Γ³_cplx0_cplx_in1_dish32_polr1,
                            Γ³_cplx1_cplx_in1_dish32_polr1,
                        ),
                        (ZZ_cplx_in0_dish32_polr1, ZZ_cplx_in1_dish32_polr1),
                        (YY_cplx0_dish32_polr1, YY_cplx1_dish32_polr1),
                    )
                    (YY_cplx0_dish33_polr1, YY_cplx1_dish33_polr1) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish33_polr1,
                            Γ³_cplx1_cplx_in0_dish33_polr1,
                            Γ³_cplx0_cplx_in1_dish33_polr1,
                            Γ³_cplx1_cplx_in1_dish33_polr1,
                        ),
                        (ZZ_cplx_in0_dish33_polr1, ZZ_cplx_in1_dish33_polr1),
                        (YY_cplx0_dish33_polr1, YY_cplx1_dish33_polr1),
                    )
                    E4_cplx0_dish0_polr0 = YY_cplx0_dish0_polr0
                    E4_cplx1_dish0_polr0 = YY_cplx1_dish0_polr0
                    E4_cplx0_dish1_polr0 = YY_cplx0_dish1_polr0
                    E4_cplx1_dish1_polr0 = YY_cplx1_dish1_polr0
                    E4_cplx0_dish32_polr0 = YY_cplx0_dish32_polr0
                    E4_cplx1_dish32_polr0 = YY_cplx1_dish32_polr0
                    E4_cplx0_dish33_polr0 = YY_cplx0_dish33_polr0
                    E4_cplx1_dish33_polr0 = YY_cplx1_dish33_polr0
                    E4_cplx0_dish0_polr1 = YY_cplx0_dish0_polr1
                    E4_cplx1_dish0_polr1 = YY_cplx1_dish0_polr1
                    E4_cplx0_dish1_polr1 = YY_cplx0_dish1_polr1
                    E4_cplx1_dish1_polr1 = YY_cplx1_dish1_polr1
                    E4_cplx0_dish32_polr1 = YY_cplx0_dish32_polr1
                    E4_cplx1_dish32_polr1 = YY_cplx1_dish32_polr1
                    E4_cplx0_dish33_polr1 = YY_cplx0_dish33_polr1
                    E4_cplx1_dish33_polr1 = YY_cplx1_dish33_polr1
                    E5_cplx0_dish0_polr0 = Gains * E4_cplx0_dish0_polr0
                    E5_cplx1_dish0_polr0 = Gains * E4_cplx1_dish0_polr0
                    E5_cplx0_dish1_polr0 = Gains * E4_cplx0_dish1_polr0
                    E5_cplx1_dish1_polr0 = Gains * E4_cplx1_dish1_polr0
                    E5_cplx0_dish32_polr0 = Gains * E4_cplx0_dish32_polr0
                    E5_cplx1_dish32_polr0 = Gains * E4_cplx1_dish32_polr0
                    E5_cplx0_dish33_polr0 = Gains * E4_cplx0_dish33_polr0
                    E5_cplx1_dish33_polr0 = Gains * E4_cplx1_dish33_polr0
                    E5_cplx0_dish0_polr1 = Gains * E4_cplx0_dish0_polr1
                    E5_cplx1_dish0_polr1 = Gains * E4_cplx1_dish0_polr1
                    E5_cplx0_dish1_polr1 = Gains * E4_cplx0_dish1_polr1
                    E5_cplx1_dish1_polr1 = Gains * E4_cplx1_dish1_polr1
                    E5_cplx0_dish32_polr1 = Gains * E4_cplx0_dish32_polr1
                    E5_cplx1_dish32_polr1 = Gains * E4_cplx1_dish32_polr1
                    E5_cplx0_dish33_polr1 = Gains * E4_cplx0_dish33_polr1
                    E5_cplx1_dish33_polr1 = Gains * E4_cplx1_dish33_polr1
                    E5_cplx0_dish0_polr0 = clamp(E5_cplx0_dish0_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish0_polr0 = clamp(E5_cplx1_dish0_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish1_polr0 = clamp(E5_cplx0_dish1_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish1_polr0 = clamp(E5_cplx1_dish1_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish32_polr0 = clamp(E5_cplx0_dish32_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish32_polr0 = clamp(E5_cplx1_dish32_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish33_polr0 = clamp(E5_cplx0_dish33_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish33_polr0 = clamp(E5_cplx1_dish33_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish0_polr1 = clamp(E5_cplx0_dish0_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish0_polr1 = clamp(E5_cplx1_dish0_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish1_polr1 = clamp(E5_cplx0_dish1_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish1_polr1 = clamp(E5_cplx1_dish1_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish32_polr1 = clamp(E5_cplx0_dish32_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish32_polr1 = clamp(E5_cplx1_dish32_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish33_polr1 = clamp(E5_cplx0_dish33_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish33_polr1 = clamp(E5_cplx1_dish33_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    F̄_out_dish0_polr0 = Int4x8((
                        E5_cplx0_dish0_polr0, E5_cplx1_dish0_polr0, E5_cplx0_dish1_polr0, E5_cplx1_dish1_polr0
                    ))
                    F̄_out_dish32_polr0 = Int4x8((
                        E5_cplx0_dish32_polr0, E5_cplx1_dish32_polr0, E5_cplx0_dish33_polr0, E5_cplx1_dish33_polr0
                    ))
                    F̄_out_dish0_polr1 = Int4x8((
                        E5_cplx0_dish0_polr1, E5_cplx1_dish0_polr1, E5_cplx0_dish1_polr1, E5_cplx1_dish1_polr1
                    ))
                    F̄_out_dish32_polr1 = Int4x8((
                        E5_cplx0_dish32_polr1, E5_cplx1_dish32_polr1, E5_cplx0_dish33_polr1, E5_cplx1_dish33_polr1
                    ))
                    if true
                        F̄_shared[(((((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 8) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 2) * 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) * 64) ÷ 2) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 32 + 0 + (((((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 16) + 0) + 0x01] =
                            F̄_out_dish0_polr0
                    end
                    if true
                        F̄_shared[(((((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 8) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 2) * 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) * 64) ÷ 2) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 2) % 2) * 32 + 0 + (((((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 4) % 16) + 0) + 0x01] =
                            F̄_out_dish32_polr0
                    end
                    if true
                        F̄_shared[(((((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 8) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 2) * 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) * 64) ÷ 2) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 32 + 16 + (((((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 16) + 0) + 0x01] =
                            F̄_out_dish0_polr1
                    end
                    if true
                        F̄_shared[(((((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 8) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 2) * 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) * 64) ÷ 2) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 2) % 2) * 32 + 16 + (((((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 4) % 16) + 0) + 0x01] =
                            F̄_out_dish32_polr1
                    end
                    F_ringbuf_m0_dish0_polr0 = F_ringbuf_dish0_mtaps0_polr0
                    F_ringbuf_m1_dish0_polr0 = F_ringbuf_dish0_mtaps1_polr0
                    F_ringbuf_m2_dish0_polr0 = F_ringbuf_dish0_mtaps2_polr0
                    F_ringbuf_m0_dish32_polr0 = F_ringbuf_dish32_mtaps0_polr0
                    F_ringbuf_m1_dish32_polr0 = F_ringbuf_dish32_mtaps1_polr0
                    F_ringbuf_m2_dish32_polr0 = F_ringbuf_dish32_mtaps2_polr0
                    F_ringbuf_m0_dish0_polr1 = F_ringbuf_dish0_mtaps0_polr1
                    F_ringbuf_m1_dish0_polr1 = F_ringbuf_dish0_mtaps1_polr1
                    F_ringbuf_m2_dish0_polr1 = F_ringbuf_dish0_mtaps2_polr1
                    F_ringbuf_m0_dish32_polr1 = F_ringbuf_dish32_mtaps0_polr1
                    F_ringbuf_m1_dish32_polr1 = F_ringbuf_dish32_mtaps1_polr1
                    F_ringbuf_m2_dish32_polr1 = F_ringbuf_dish32_mtaps2_polr1
                    F_ringbuf_m0_dish0_polr0 = F_ringbuf_m1_dish0_polr0
                    F_ringbuf_m0_dish32_polr0 = F_ringbuf_m1_dish32_polr0
                    F_ringbuf_m0_dish0_polr1 = F_ringbuf_m1_dish0_polr1
                    F_ringbuf_m0_dish32_polr1 = F_ringbuf_m1_dish32_polr1
                    F_ringbuf_m1_dish0_polr0 = F_ringbuf_m2_dish0_polr0
                    F_ringbuf_m1_dish32_polr0 = F_ringbuf_m2_dish32_polr0
                    F_ringbuf_m1_dish0_polr1 = F_ringbuf_m2_dish0_polr1
                    F_ringbuf_m1_dish32_polr1 = F_ringbuf_m2_dish32_polr1
                    F_ringbuf_m2_dish0_polr0 = F_in_dish0_polr0
                    F_ringbuf_m2_dish32_polr0 = F_in_dish32_polr0
                    F_ringbuf_m2_dish0_polr1 = F_in_dish0_polr1
                    F_ringbuf_m2_dish32_polr1 = F_in_dish32_polr1
                    F_ringbuf_dish0_mtaps0_polr0 = F_ringbuf_m0_dish0_polr0
                    F_ringbuf_dish0_mtaps1_polr0 = F_ringbuf_m1_dish0_polr0
                    F_ringbuf_dish0_mtaps2_polr0 = F_ringbuf_m2_dish0_polr0
                    F_ringbuf_dish32_mtaps0_polr0 = F_ringbuf_m0_dish32_polr0
                    F_ringbuf_dish32_mtaps1_polr0 = F_ringbuf_m1_dish32_polr0
                    F_ringbuf_dish32_mtaps2_polr0 = F_ringbuf_m2_dish32_polr0
                    F_ringbuf_dish0_mtaps0_polr1 = F_ringbuf_m0_dish0_polr1
                    F_ringbuf_dish0_mtaps1_polr1 = F_ringbuf_m1_dish0_polr1
                    F_ringbuf_dish0_mtaps2_polr1 = F_ringbuf_m2_dish0_polr1
                    F_ringbuf_dish32_mtaps0_polr1 = F_ringbuf_m0_dish32_polr1
                    F_ringbuf_dish32_mtaps1_polr1 = F_ringbuf_m1_dish32_polr1
                    F_ringbuf_dish32_mtaps2_polr1 = F_ringbuf_m2_dish32_polr1
                end
                let
                    dish = 32
                    F_in_dish0_polr0 = F_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 32 + 0 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 16) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) % 2) * 1040 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 2) % 2) * 520 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 4) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 8) % 2) * 130) + 0x01]
                    F_in_dish32_polr0 = F_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 2) % 2) * 32 + 0 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 16) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) % 2) * 1040 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 2) % 2) * 520 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 4) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 8) % 2) * 130) + 0x01]
                    F_in_dish0_polr1 = F_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 32 + 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 16) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) % 2) * 1040 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 2) % 2) * 520 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 4) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 8) % 2) * 130) + 0x01]
                    F_in_dish32_polr1 = F_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 2) % 2) * 32 + 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 16) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) % 2) * 1040 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 2) % 2) * 520 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 4) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 8) % 2) * 130) + 0x01]
                    (E_cplx0_dish0_polr0, E_cplx1_dish0_polr0, E_cplx0_dish1_polr0, E_cplx1_dish1_polr0) = convert(
                        NTuple{4,Float16x2}, F_in_dish0_polr0
                    )
                    (E_cplx0_dish32_polr0, E_cplx1_dish32_polr0, E_cplx0_dish33_polr0, E_cplx1_dish33_polr0) = convert(
                        NTuple{4,Float16x2}, F_in_dish32_polr0
                    )
                    (E_cplx0_dish0_polr1, E_cplx1_dish0_polr1, E_cplx0_dish1_polr1, E_cplx1_dish1_polr1) = convert(
                        NTuple{4,Float16x2}, F_in_dish0_polr1
                    )
                    (E_cplx0_dish32_polr1, E_cplx1_dish32_polr1, E_cplx0_dish33_polr1, E_cplx1_dish33_polr1) = convert(
                        NTuple{4,Float16x2}, F_in_dish32_polr1
                    )
                    W_m0 = Wpfb_mtaps0
                    W_m1 = Wpfb_mtaps1
                    W_m2 = Wpfb_mtaps2
                    W_m3 = Wpfb_mtaps3
                    E2_cplx0_dish0_polr0 = -W_m3 * E_cplx0_dish0_polr0
                    E2_cplx1_dish0_polr0 = -W_m3 * E_cplx1_dish0_polr0
                    E2_cplx0_dish1_polr0 = -W_m3 * E_cplx0_dish1_polr0
                    E2_cplx1_dish1_polr0 = -W_m3 * E_cplx1_dish1_polr0
                    E2_cplx0_dish32_polr0 = -W_m3 * E_cplx0_dish32_polr0
                    E2_cplx1_dish32_polr0 = -W_m3 * E_cplx1_dish32_polr0
                    E2_cplx0_dish33_polr0 = -W_m3 * E_cplx0_dish33_polr0
                    E2_cplx1_dish33_polr0 = -W_m3 * E_cplx1_dish33_polr0
                    E2_cplx0_dish0_polr1 = -W_m3 * E_cplx0_dish0_polr1
                    E2_cplx1_dish0_polr1 = -W_m3 * E_cplx1_dish0_polr1
                    E2_cplx0_dish1_polr1 = -W_m3 * E_cplx0_dish1_polr1
                    E2_cplx1_dish1_polr1 = -W_m3 * E_cplx1_dish1_polr1
                    E2_cplx0_dish32_polr1 = -W_m3 * E_cplx0_dish32_polr1
                    E2_cplx1_dish32_polr1 = -W_m3 * E_cplx1_dish32_polr1
                    E2_cplx0_dish33_polr1 = -W_m3 * E_cplx0_dish33_polr1
                    E2_cplx1_dish33_polr1 = -W_m3 * E_cplx1_dish33_polr1
                    F_ringbuf_m0_dish0_polr0 = F_ringbuf_dish0_mtaps0_polr0
                    F_ringbuf_m1_dish0_polr0 = F_ringbuf_dish0_mtaps1_polr0
                    F_ringbuf_m2_dish0_polr0 = F_ringbuf_dish0_mtaps2_polr0
                    F_ringbuf_m0_dish32_polr0 = F_ringbuf_dish32_mtaps0_polr0
                    F_ringbuf_m1_dish32_polr0 = F_ringbuf_dish32_mtaps1_polr0
                    F_ringbuf_m2_dish32_polr0 = F_ringbuf_dish32_mtaps2_polr0
                    F_ringbuf_m0_dish0_polr1 = F_ringbuf_dish0_mtaps0_polr1
                    F_ringbuf_m1_dish0_polr1 = F_ringbuf_dish0_mtaps1_polr1
                    F_ringbuf_m2_dish0_polr1 = F_ringbuf_dish0_mtaps2_polr1
                    F_ringbuf_m0_dish32_polr1 = F_ringbuf_dish32_mtaps0_polr1
                    F_ringbuf_m1_dish32_polr1 = F_ringbuf_dish32_mtaps1_polr1
                    F_ringbuf_m2_dish32_polr1 = F_ringbuf_dish32_mtaps2_polr1
                    (E_ringbuf_m0_cplx0_dish0_polr0, E_ringbuf_m0_cplx1_dish0_polr0, E_ringbuf_m0_cplx0_dish1_polr0, E_ringbuf_m0_cplx1_dish1_polr0) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m0_dish0_polr0
                    )
                    (E_ringbuf_m0_cplx0_dish32_polr0, E_ringbuf_m0_cplx1_dish32_polr0, E_ringbuf_m0_cplx0_dish33_polr0, E_ringbuf_m0_cplx1_dish33_polr0) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m0_dish32_polr0
                    )
                    (E_ringbuf_m0_cplx0_dish0_polr1, E_ringbuf_m0_cplx1_dish0_polr1, E_ringbuf_m0_cplx0_dish1_polr1, E_ringbuf_m0_cplx1_dish1_polr1) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m0_dish0_polr1
                    )
                    (E_ringbuf_m0_cplx0_dish32_polr1, E_ringbuf_m0_cplx1_dish32_polr1, E_ringbuf_m0_cplx0_dish33_polr1, E_ringbuf_m0_cplx1_dish33_polr1) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m0_dish32_polr1
                    )
                    E2_cplx0_dish0_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish0_polr0, E2_cplx0_dish0_polr0)
                    E2_cplx1_dish0_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish0_polr0, E2_cplx1_dish0_polr0)
                    E2_cplx0_dish1_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish1_polr0, E2_cplx0_dish1_polr0)
                    E2_cplx1_dish1_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish1_polr0, E2_cplx1_dish1_polr0)
                    E2_cplx0_dish32_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish32_polr0, E2_cplx0_dish32_polr0)
                    E2_cplx1_dish32_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish32_polr0, E2_cplx1_dish32_polr0)
                    E2_cplx0_dish33_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish33_polr0, E2_cplx0_dish33_polr0)
                    E2_cplx1_dish33_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish33_polr0, E2_cplx1_dish33_polr0)
                    E2_cplx0_dish0_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish0_polr1, E2_cplx0_dish0_polr1)
                    E2_cplx1_dish0_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish0_polr1, E2_cplx1_dish0_polr1)
                    E2_cplx0_dish1_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish1_polr1, E2_cplx0_dish1_polr1)
                    E2_cplx1_dish1_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish1_polr1, E2_cplx1_dish1_polr1)
                    E2_cplx0_dish32_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish32_polr1, E2_cplx0_dish32_polr1)
                    E2_cplx1_dish32_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish32_polr1, E2_cplx1_dish32_polr1)
                    E2_cplx0_dish33_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish33_polr1, E2_cplx0_dish33_polr1)
                    E2_cplx1_dish33_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish33_polr1, E2_cplx1_dish33_polr1)
                    (E_ringbuf_m1_cplx0_dish0_polr0, E_ringbuf_m1_cplx1_dish0_polr0, E_ringbuf_m1_cplx0_dish1_polr0, E_ringbuf_m1_cplx1_dish1_polr0) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m1_dish0_polr0
                    )
                    (E_ringbuf_m1_cplx0_dish32_polr0, E_ringbuf_m1_cplx1_dish32_polr0, E_ringbuf_m1_cplx0_dish33_polr0, E_ringbuf_m1_cplx1_dish33_polr0) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m1_dish32_polr0
                    )
                    (E_ringbuf_m1_cplx0_dish0_polr1, E_ringbuf_m1_cplx1_dish0_polr1, E_ringbuf_m1_cplx0_dish1_polr1, E_ringbuf_m1_cplx1_dish1_polr1) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m1_dish0_polr1
                    )
                    (E_ringbuf_m1_cplx0_dish32_polr1, E_ringbuf_m1_cplx1_dish32_polr1, E_ringbuf_m1_cplx0_dish33_polr1, E_ringbuf_m1_cplx1_dish33_polr1) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m1_dish32_polr1
                    )
                    E2_cplx0_dish0_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish0_polr0, E2_cplx0_dish0_polr0)
                    E2_cplx1_dish0_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish0_polr0, E2_cplx1_dish0_polr0)
                    E2_cplx0_dish1_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish1_polr0, E2_cplx0_dish1_polr0)
                    E2_cplx1_dish1_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish1_polr0, E2_cplx1_dish1_polr0)
                    E2_cplx0_dish32_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish32_polr0, E2_cplx0_dish32_polr0)
                    E2_cplx1_dish32_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish32_polr0, E2_cplx1_dish32_polr0)
                    E2_cplx0_dish33_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish33_polr0, E2_cplx0_dish33_polr0)
                    E2_cplx1_dish33_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish33_polr0, E2_cplx1_dish33_polr0)
                    E2_cplx0_dish0_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish0_polr1, E2_cplx0_dish0_polr1)
                    E2_cplx1_dish0_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish0_polr1, E2_cplx1_dish0_polr1)
                    E2_cplx0_dish1_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish1_polr1, E2_cplx0_dish1_polr1)
                    E2_cplx1_dish1_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish1_polr1, E2_cplx1_dish1_polr1)
                    E2_cplx0_dish32_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish32_polr1, E2_cplx0_dish32_polr1)
                    E2_cplx1_dish32_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish32_polr1, E2_cplx1_dish32_polr1)
                    E2_cplx0_dish33_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish33_polr1, E2_cplx0_dish33_polr1)
                    E2_cplx1_dish33_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish33_polr1, E2_cplx1_dish33_polr1)
                    (E_ringbuf_m2_cplx0_dish0_polr0, E_ringbuf_m2_cplx1_dish0_polr0, E_ringbuf_m2_cplx0_dish1_polr0, E_ringbuf_m2_cplx1_dish1_polr0) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m2_dish0_polr0
                    )
                    (E_ringbuf_m2_cplx0_dish32_polr0, E_ringbuf_m2_cplx1_dish32_polr0, E_ringbuf_m2_cplx0_dish33_polr0, E_ringbuf_m2_cplx1_dish33_polr0) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m2_dish32_polr0
                    )
                    (E_ringbuf_m2_cplx0_dish0_polr1, E_ringbuf_m2_cplx1_dish0_polr1, E_ringbuf_m2_cplx0_dish1_polr1, E_ringbuf_m2_cplx1_dish1_polr1) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m2_dish0_polr1
                    )
                    (E_ringbuf_m2_cplx0_dish32_polr1, E_ringbuf_m2_cplx1_dish32_polr1, E_ringbuf_m2_cplx0_dish33_polr1, E_ringbuf_m2_cplx1_dish33_polr1) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m2_dish32_polr1
                    )
                    E2_cplx0_dish0_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish0_polr0, E2_cplx0_dish0_polr0)
                    E2_cplx1_dish0_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish0_polr0, E2_cplx1_dish0_polr0)
                    E2_cplx0_dish1_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish1_polr0, E2_cplx0_dish1_polr0)
                    E2_cplx1_dish1_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish1_polr0, E2_cplx1_dish1_polr0)
                    E2_cplx0_dish32_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish32_polr0, E2_cplx0_dish32_polr0)
                    E2_cplx1_dish32_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish32_polr0, E2_cplx1_dish32_polr0)
                    E2_cplx0_dish33_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish33_polr0, E2_cplx0_dish33_polr0)
                    E2_cplx1_dish33_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish33_polr0, E2_cplx1_dish33_polr0)
                    E2_cplx0_dish0_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish0_polr1, E2_cplx0_dish0_polr1)
                    E2_cplx1_dish0_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish0_polr1, E2_cplx1_dish0_polr1)
                    E2_cplx0_dish1_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish1_polr1, E2_cplx0_dish1_polr1)
                    E2_cplx1_dish1_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish1_polr1, E2_cplx1_dish1_polr1)
                    E2_cplx0_dish32_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish32_polr1, E2_cplx0_dish32_polr1)
                    E2_cplx1_dish32_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish32_polr1, E2_cplx1_dish32_polr1)
                    E2_cplx0_dish33_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish33_polr1, E2_cplx0_dish33_polr1)
                    E2_cplx1_dish33_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish33_polr1, E2_cplx1_dish33_polr1)
                    E2re_dish0_polr0 = E2_cplx0_dish0_polr0
                    E2im_dish0_polr0 = E2_cplx1_dish0_polr0
                    E2re_dish1_polr0 = E2_cplx0_dish1_polr0
                    E2im_dish1_polr0 = E2_cplx1_dish1_polr0
                    E2re_dish32_polr0 = E2_cplx0_dish32_polr0
                    E2im_dish32_polr0 = E2_cplx1_dish32_polr0
                    E2re_dish33_polr0 = E2_cplx0_dish33_polr0
                    E2im_dish33_polr0 = E2_cplx1_dish33_polr0
                    E2re_dish0_polr1 = E2_cplx0_dish0_polr1
                    E2im_dish0_polr1 = E2_cplx1_dish0_polr1
                    E2re_dish1_polr1 = E2_cplx0_dish1_polr1
                    E2im_dish1_polr1 = E2_cplx1_dish1_polr1
                    E2re_dish32_polr1 = E2_cplx0_dish32_polr1
                    E2im_dish32_polr1 = E2_cplx1_dish32_polr1
                    E2re_dish33_polr1 = E2_cplx0_dish33_polr1
                    E2im_dish33_polr1 = E2_cplx1_dish33_polr1
                    Xre = X_cplx0
                    Xim = X_cplx1
                    E3re_dish0_polr0 = muladd(Xre, E2re_dish0_polr0, -Xim * E2im_dish0_polr0)
                    E3re_dish1_polr0 = muladd(Xre, E2re_dish1_polr0, -Xim * E2im_dish1_polr0)
                    E3re_dish32_polr0 = muladd(Xre, E2re_dish32_polr0, -Xim * E2im_dish32_polr0)
                    E3re_dish33_polr0 = muladd(Xre, E2re_dish33_polr0, -Xim * E2im_dish33_polr0)
                    E3re_dish0_polr1 = muladd(Xre, E2re_dish0_polr1, -Xim * E2im_dish0_polr1)
                    E3re_dish1_polr1 = muladd(Xre, E2re_dish1_polr1, -Xim * E2im_dish1_polr1)
                    E3re_dish32_polr1 = muladd(Xre, E2re_dish32_polr1, -Xim * E2im_dish32_polr1)
                    E3re_dish33_polr1 = muladd(Xre, E2re_dish33_polr1, -Xim * E2im_dish33_polr1)
                    E3im_dish0_polr0 = muladd(Xre, E2im_dish0_polr0, Xim * E2re_dish0_polr0)
                    E3im_dish1_polr0 = muladd(Xre, E2im_dish1_polr0, Xim * E2re_dish1_polr0)
                    E3im_dish32_polr0 = muladd(Xre, E2im_dish32_polr0, Xim * E2re_dish32_polr0)
                    E3im_dish33_polr0 = muladd(Xre, E2im_dish33_polr0, Xim * E2re_dish33_polr0)
                    E3im_dish0_polr1 = muladd(Xre, E2im_dish0_polr1, Xim * E2re_dish0_polr1)
                    E3im_dish1_polr1 = muladd(Xre, E2im_dish1_polr1, Xim * E2re_dish1_polr1)
                    E3im_dish32_polr1 = muladd(Xre, E2im_dish32_polr1, Xim * E2re_dish32_polr1)
                    E3im_dish33_polr1 = muladd(Xre, E2im_dish33_polr1, Xim * E2re_dish33_polr1)
                    E3_cplx0_dish0_polr0 = E3re_dish0_polr0
                    E3_cplx1_dish0_polr0 = E3im_dish0_polr0
                    E3_cplx0_dish1_polr0 = E3re_dish1_polr0
                    E3_cplx1_dish1_polr0 = E3im_dish1_polr0
                    E3_cplx0_dish32_polr0 = E3re_dish32_polr0
                    E3_cplx1_dish32_polr0 = E3im_dish32_polr0
                    E3_cplx0_dish33_polr0 = E3re_dish33_polr0
                    E3_cplx1_dish33_polr0 = E3im_dish33_polr0
                    E3_cplx0_dish0_polr1 = E3re_dish0_polr1
                    E3_cplx1_dish0_polr1 = E3im_dish0_polr1
                    E3_cplx0_dish1_polr1 = E3re_dish1_polr1
                    E3_cplx1_dish1_polr1 = E3im_dish1_polr1
                    E3_cplx0_dish32_polr1 = E3re_dish32_polr1
                    E3_cplx1_dish32_polr1 = E3im_dish32_polr1
                    E3_cplx0_dish33_polr1 = E3re_dish33_polr1
                    E3_cplx1_dish33_polr1 = E3im_dish33_polr1
                    XX_cplx0_dish0_polr0 = E3_cplx0_dish0_polr0
                    XX_cplx1_dish0_polr0 = E3_cplx1_dish0_polr0
                    XX_cplx0_dish1_polr0 = E3_cplx0_dish1_polr0
                    XX_cplx1_dish1_polr0 = E3_cplx1_dish1_polr0
                    XX_cplx0_dish32_polr0 = E3_cplx0_dish32_polr0
                    XX_cplx1_dish32_polr0 = E3_cplx1_dish32_polr0
                    XX_cplx0_dish33_polr0 = E3_cplx0_dish33_polr0
                    XX_cplx1_dish33_polr0 = E3_cplx1_dish33_polr0
                    XX_cplx0_dish0_polr1 = E3_cplx0_dish0_polr1
                    XX_cplx1_dish0_polr1 = E3_cplx1_dish0_polr1
                    XX_cplx0_dish1_polr1 = E3_cplx0_dish1_polr1
                    XX_cplx1_dish1_polr1 = E3_cplx1_dish1_polr1
                    XX_cplx0_dish32_polr1 = E3_cplx0_dish32_polr1
                    XX_cplx1_dish32_polr1 = E3_cplx1_dish32_polr1
                    XX_cplx0_dish33_polr1 = E3_cplx0_dish33_polr1
                    XX_cplx1_dish33_polr1 = E3_cplx1_dish33_polr1
                    XXre_dish0_polr0 = XX_cplx0_dish0_polr0
                    XXim_dish0_polr0 = XX_cplx1_dish0_polr0
                    XXre_dish1_polr0 = XX_cplx0_dish1_polr0
                    XXim_dish1_polr0 = XX_cplx1_dish1_polr0
                    XXre_dish32_polr0 = XX_cplx0_dish32_polr0
                    XXim_dish32_polr0 = XX_cplx1_dish32_polr0
                    XXre_dish33_polr0 = XX_cplx0_dish33_polr0
                    XXim_dish33_polr0 = XX_cplx1_dish33_polr0
                    XXre_dish0_polr1 = XX_cplx0_dish0_polr1
                    XXim_dish0_polr1 = XX_cplx1_dish0_polr1
                    XXre_dish1_polr1 = XX_cplx0_dish1_polr1
                    XXim_dish1_polr1 = XX_cplx1_dish1_polr1
                    XXre_dish32_polr1 = XX_cplx0_dish32_polr1
                    XXim_dish32_polr1 = XX_cplx1_dish32_polr1
                    XXre_dish33_polr1 = XX_cplx0_dish33_polr1
                    XXim_dish33_polr1 = XX_cplx1_dish33_polr1
                    XX_cplx_in0_dish0_polr0 = XXre_dish0_polr0
                    XX_cplx_in1_dish0_polr0 = XXim_dish0_polr0
                    XX_cplx_in0_dish1_polr0 = XXre_dish1_polr0
                    XX_cplx_in1_dish1_polr0 = XXim_dish1_polr0
                    XX_cplx_in0_dish32_polr0 = XXre_dish32_polr0
                    XX_cplx_in1_dish32_polr0 = XXim_dish32_polr0
                    XX_cplx_in0_dish33_polr0 = XXre_dish33_polr0
                    XX_cplx_in1_dish33_polr0 = XXim_dish33_polr0
                    XX_cplx_in0_dish0_polr1 = XXre_dish0_polr1
                    XX_cplx_in1_dish0_polr1 = XXim_dish0_polr1
                    XX_cplx_in0_dish1_polr1 = XXre_dish1_polr1
                    XX_cplx_in1_dish1_polr1 = XXim_dish1_polr1
                    XX_cplx_in0_dish32_polr1 = XXre_dish32_polr1
                    XX_cplx_in1_dish32_polr1 = XXim_dish32_polr1
                    XX_cplx_in0_dish33_polr1 = XXre_dish33_polr1
                    XX_cplx_in1_dish33_polr1 = XXim_dish33_polr1
                    WW_cplx0_dish0_polr0 = zero(Float16x2)
                    WW_cplx1_dish0_polr0 = zero(Float16x2)
                    WW_cplx0_dish1_polr0 = zero(Float16x2)
                    WW_cplx1_dish1_polr0 = zero(Float16x2)
                    WW_cplx0_dish32_polr0 = zero(Float16x2)
                    WW_cplx1_dish32_polr0 = zero(Float16x2)
                    WW_cplx0_dish33_polr0 = zero(Float16x2)
                    WW_cplx1_dish33_polr0 = zero(Float16x2)
                    WW_cplx0_dish0_polr1 = zero(Float16x2)
                    WW_cplx1_dish0_polr1 = zero(Float16x2)
                    WW_cplx0_dish1_polr1 = zero(Float16x2)
                    WW_cplx1_dish1_polr1 = zero(Float16x2)
                    WW_cplx0_dish32_polr1 = zero(Float16x2)
                    WW_cplx1_dish32_polr1 = zero(Float16x2)
                    WW_cplx0_dish33_polr1 = zero(Float16x2)
                    WW_cplx1_dish33_polr1 = zero(Float16x2)
                    (WW_cplx0_dish0_polr0, WW_cplx1_dish0_polr0) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish0_polr0, XX_cplx_in1_dish0_polr0),
                        (WW_cplx0_dish0_polr0, WW_cplx1_dish0_polr0),
                    )
                    (WW_cplx0_dish1_polr0, WW_cplx1_dish1_polr0) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish1_polr0, XX_cplx_in1_dish1_polr0),
                        (WW_cplx0_dish1_polr0, WW_cplx1_dish1_polr0),
                    )
                    (WW_cplx0_dish32_polr0, WW_cplx1_dish32_polr0) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish32_polr0, XX_cplx_in1_dish32_polr0),
                        (WW_cplx0_dish32_polr0, WW_cplx1_dish32_polr0),
                    )
                    (WW_cplx0_dish33_polr0, WW_cplx1_dish33_polr0) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish33_polr0, XX_cplx_in1_dish33_polr0),
                        (WW_cplx0_dish33_polr0, WW_cplx1_dish33_polr0),
                    )
                    (WW_cplx0_dish0_polr1, WW_cplx1_dish0_polr1) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish0_polr1, XX_cplx_in1_dish0_polr1),
                        (WW_cplx0_dish0_polr1, WW_cplx1_dish0_polr1),
                    )
                    (WW_cplx0_dish1_polr1, WW_cplx1_dish1_polr1) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish1_polr1, XX_cplx_in1_dish1_polr1),
                        (WW_cplx0_dish1_polr1, WW_cplx1_dish1_polr1),
                    )
                    (WW_cplx0_dish32_polr1, WW_cplx1_dish32_polr1) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish32_polr1, XX_cplx_in1_dish32_polr1),
                        (WW_cplx0_dish32_polr1, WW_cplx1_dish32_polr1),
                    )
                    (WW_cplx0_dish33_polr1, WW_cplx1_dish33_polr1) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish33_polr1, XX_cplx_in1_dish33_polr1),
                        (WW_cplx0_dish33_polr1, WW_cplx1_dish33_polr1),
                    )
                    Γ²re = Γ²_cplx0
                    Γ²im = Γ²_cplx1
                    WWre_dish0_polr0 = WW_cplx0_dish0_polr0
                    WWim_dish0_polr0 = WW_cplx1_dish0_polr0
                    WWre_dish1_polr0 = WW_cplx0_dish1_polr0
                    WWim_dish1_polr0 = WW_cplx1_dish1_polr0
                    WWre_dish32_polr0 = WW_cplx0_dish32_polr0
                    WWim_dish32_polr0 = WW_cplx1_dish32_polr0
                    WWre_dish33_polr0 = WW_cplx0_dish33_polr0
                    WWim_dish33_polr0 = WW_cplx1_dish33_polr0
                    WWre_dish0_polr1 = WW_cplx0_dish0_polr1
                    WWim_dish0_polr1 = WW_cplx1_dish0_polr1
                    WWre_dish1_polr1 = WW_cplx0_dish1_polr1
                    WWim_dish1_polr1 = WW_cplx1_dish1_polr1
                    WWre_dish32_polr1 = WW_cplx0_dish32_polr1
                    WWim_dish32_polr1 = WW_cplx1_dish32_polr1
                    WWre_dish33_polr1 = WW_cplx0_dish33_polr1
                    WWim_dish33_polr1 = WW_cplx1_dish33_polr1
                    ZZre_dish0_polr0 = muladd(Γ²re, WWre_dish0_polr0, -Γ²im * WWim_dish0_polr0)
                    ZZre_dish1_polr0 = muladd(Γ²re, WWre_dish1_polr0, -Γ²im * WWim_dish1_polr0)
                    ZZre_dish32_polr0 = muladd(Γ²re, WWre_dish32_polr0, -Γ²im * WWim_dish32_polr0)
                    ZZre_dish33_polr0 = muladd(Γ²re, WWre_dish33_polr0, -Γ²im * WWim_dish33_polr0)
                    ZZre_dish0_polr1 = muladd(Γ²re, WWre_dish0_polr1, -Γ²im * WWim_dish0_polr1)
                    ZZre_dish1_polr1 = muladd(Γ²re, WWre_dish1_polr1, -Γ²im * WWim_dish1_polr1)
                    ZZre_dish32_polr1 = muladd(Γ²re, WWre_dish32_polr1, -Γ²im * WWim_dish32_polr1)
                    ZZre_dish33_polr1 = muladd(Γ²re, WWre_dish33_polr1, -Γ²im * WWim_dish33_polr1)
                    ZZim_dish0_polr0 = muladd(Γ²re, WWim_dish0_polr0, Γ²im * WWre_dish0_polr0)
                    ZZim_dish1_polr0 = muladd(Γ²re, WWim_dish1_polr0, Γ²im * WWre_dish1_polr0)
                    ZZim_dish32_polr0 = muladd(Γ²re, WWim_dish32_polr0, Γ²im * WWre_dish32_polr0)
                    ZZim_dish33_polr0 = muladd(Γ²re, WWim_dish33_polr0, Γ²im * WWre_dish33_polr0)
                    ZZim_dish0_polr1 = muladd(Γ²re, WWim_dish0_polr1, Γ²im * WWre_dish0_polr1)
                    ZZim_dish1_polr1 = muladd(Γ²re, WWim_dish1_polr1, Γ²im * WWre_dish1_polr1)
                    ZZim_dish32_polr1 = muladd(Γ²re, WWim_dish32_polr1, Γ²im * WWre_dish32_polr1)
                    ZZim_dish33_polr1 = muladd(Γ²re, WWim_dish33_polr1, Γ²im * WWre_dish33_polr1)
                    ZZ_cplx0_dish0_polr0 = ZZre_dish0_polr0
                    ZZ_cplx1_dish0_polr0 = ZZim_dish0_polr0
                    ZZ_cplx0_dish1_polr0 = ZZre_dish1_polr0
                    ZZ_cplx1_dish1_polr0 = ZZim_dish1_polr0
                    ZZ_cplx0_dish32_polr0 = ZZre_dish32_polr0
                    ZZ_cplx1_dish32_polr0 = ZZim_dish32_polr0
                    ZZ_cplx0_dish33_polr0 = ZZre_dish33_polr0
                    ZZ_cplx1_dish33_polr0 = ZZim_dish33_polr0
                    ZZ_cplx0_dish0_polr1 = ZZre_dish0_polr1
                    ZZ_cplx1_dish0_polr1 = ZZim_dish0_polr1
                    ZZ_cplx0_dish1_polr1 = ZZre_dish1_polr1
                    ZZ_cplx1_dish1_polr1 = ZZim_dish1_polr1
                    ZZ_cplx0_dish32_polr1 = ZZre_dish32_polr1
                    ZZ_cplx1_dish32_polr1 = ZZim_dish32_polr1
                    ZZ_cplx0_dish33_polr1 = ZZre_dish33_polr1
                    ZZ_cplx1_dish33_polr1 = ZZim_dish33_polr1
                    ZZre_dish0_polr0 = ZZ_cplx0_dish0_polr0
                    ZZim_dish0_polr0 = ZZ_cplx1_dish0_polr0
                    ZZre_dish1_polr0 = ZZ_cplx0_dish1_polr0
                    ZZim_dish1_polr0 = ZZ_cplx1_dish1_polr0
                    ZZre_dish32_polr0 = ZZ_cplx0_dish32_polr0
                    ZZim_dish32_polr0 = ZZ_cplx1_dish32_polr0
                    ZZre_dish33_polr0 = ZZ_cplx0_dish33_polr0
                    ZZim_dish33_polr0 = ZZ_cplx1_dish33_polr0
                    ZZre_dish0_polr1 = ZZ_cplx0_dish0_polr1
                    ZZim_dish0_polr1 = ZZ_cplx1_dish0_polr1
                    ZZre_dish1_polr1 = ZZ_cplx0_dish1_polr1
                    ZZim_dish1_polr1 = ZZ_cplx1_dish1_polr1
                    ZZre_dish32_polr1 = ZZ_cplx0_dish32_polr1
                    ZZim_dish32_polr1 = ZZ_cplx1_dish32_polr1
                    ZZre_dish33_polr1 = ZZ_cplx0_dish33_polr1
                    ZZim_dish33_polr1 = ZZ_cplx1_dish33_polr1
                    ZZ_cplx_in0_dish0_polr0 = ZZre_dish0_polr0
                    ZZ_cplx_in1_dish0_polr0 = ZZim_dish0_polr0
                    ZZ_cplx_in0_dish1_polr0 = ZZre_dish1_polr0
                    ZZ_cplx_in1_dish1_polr0 = ZZim_dish1_polr0
                    ZZ_cplx_in0_dish32_polr0 = ZZre_dish32_polr0
                    ZZ_cplx_in1_dish32_polr0 = ZZim_dish32_polr0
                    ZZ_cplx_in0_dish33_polr0 = ZZre_dish33_polr0
                    ZZ_cplx_in1_dish33_polr0 = ZZim_dish33_polr0
                    ZZ_cplx_in0_dish0_polr1 = ZZre_dish0_polr1
                    ZZ_cplx_in1_dish0_polr1 = ZZim_dish0_polr1
                    ZZ_cplx_in0_dish1_polr1 = ZZre_dish1_polr1
                    ZZ_cplx_in1_dish1_polr1 = ZZim_dish1_polr1
                    ZZ_cplx_in0_dish32_polr1 = ZZre_dish32_polr1
                    ZZ_cplx_in1_dish32_polr1 = ZZim_dish32_polr1
                    ZZ_cplx_in0_dish33_polr1 = ZZre_dish33_polr1
                    ZZ_cplx_in1_dish33_polr1 = ZZim_dish33_polr1
                    YY_cplx0_dish0_polr0 = zero(Float16x2)
                    YY_cplx1_dish0_polr0 = zero(Float16x2)
                    YY_cplx0_dish1_polr0 = zero(Float16x2)
                    YY_cplx1_dish1_polr0 = zero(Float16x2)
                    YY_cplx0_dish32_polr0 = zero(Float16x2)
                    YY_cplx1_dish32_polr0 = zero(Float16x2)
                    YY_cplx0_dish33_polr0 = zero(Float16x2)
                    YY_cplx1_dish33_polr0 = zero(Float16x2)
                    YY_cplx0_dish0_polr1 = zero(Float16x2)
                    YY_cplx1_dish0_polr1 = zero(Float16x2)
                    YY_cplx0_dish1_polr1 = zero(Float16x2)
                    YY_cplx1_dish1_polr1 = zero(Float16x2)
                    YY_cplx0_dish32_polr1 = zero(Float16x2)
                    YY_cplx1_dish32_polr1 = zero(Float16x2)
                    YY_cplx0_dish33_polr1 = zero(Float16x2)
                    YY_cplx1_dish33_polr1 = zero(Float16x2)
                    (YY_cplx0_dish0_polr0, YY_cplx1_dish0_polr0) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish0_polr0,
                            Γ³_cplx1_cplx_in0_dish0_polr0,
                            Γ³_cplx0_cplx_in1_dish0_polr0,
                            Γ³_cplx1_cplx_in1_dish0_polr0,
                        ),
                        (ZZ_cplx_in0_dish0_polr0, ZZ_cplx_in1_dish0_polr0),
                        (YY_cplx0_dish0_polr0, YY_cplx1_dish0_polr0),
                    )
                    (YY_cplx0_dish1_polr0, YY_cplx1_dish1_polr0) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish1_polr0,
                            Γ³_cplx1_cplx_in0_dish1_polr0,
                            Γ³_cplx0_cplx_in1_dish1_polr0,
                            Γ³_cplx1_cplx_in1_dish1_polr0,
                        ),
                        (ZZ_cplx_in0_dish1_polr0, ZZ_cplx_in1_dish1_polr0),
                        (YY_cplx0_dish1_polr0, YY_cplx1_dish1_polr0),
                    )
                    (YY_cplx0_dish32_polr0, YY_cplx1_dish32_polr0) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish32_polr0,
                            Γ³_cplx1_cplx_in0_dish32_polr0,
                            Γ³_cplx0_cplx_in1_dish32_polr0,
                            Γ³_cplx1_cplx_in1_dish32_polr0,
                        ),
                        (ZZ_cplx_in0_dish32_polr0, ZZ_cplx_in1_dish32_polr0),
                        (YY_cplx0_dish32_polr0, YY_cplx1_dish32_polr0),
                    )
                    (YY_cplx0_dish33_polr0, YY_cplx1_dish33_polr0) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish33_polr0,
                            Γ³_cplx1_cplx_in0_dish33_polr0,
                            Γ³_cplx0_cplx_in1_dish33_polr0,
                            Γ³_cplx1_cplx_in1_dish33_polr0,
                        ),
                        (ZZ_cplx_in0_dish33_polr0, ZZ_cplx_in1_dish33_polr0),
                        (YY_cplx0_dish33_polr0, YY_cplx1_dish33_polr0),
                    )
                    (YY_cplx0_dish0_polr1, YY_cplx1_dish0_polr1) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish0_polr1,
                            Γ³_cplx1_cplx_in0_dish0_polr1,
                            Γ³_cplx0_cplx_in1_dish0_polr1,
                            Γ³_cplx1_cplx_in1_dish0_polr1,
                        ),
                        (ZZ_cplx_in0_dish0_polr1, ZZ_cplx_in1_dish0_polr1),
                        (YY_cplx0_dish0_polr1, YY_cplx1_dish0_polr1),
                    )
                    (YY_cplx0_dish1_polr1, YY_cplx1_dish1_polr1) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish1_polr1,
                            Γ³_cplx1_cplx_in0_dish1_polr1,
                            Γ³_cplx0_cplx_in1_dish1_polr1,
                            Γ³_cplx1_cplx_in1_dish1_polr1,
                        ),
                        (ZZ_cplx_in0_dish1_polr1, ZZ_cplx_in1_dish1_polr1),
                        (YY_cplx0_dish1_polr1, YY_cplx1_dish1_polr1),
                    )
                    (YY_cplx0_dish32_polr1, YY_cplx1_dish32_polr1) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish32_polr1,
                            Γ³_cplx1_cplx_in0_dish32_polr1,
                            Γ³_cplx0_cplx_in1_dish32_polr1,
                            Γ³_cplx1_cplx_in1_dish32_polr1,
                        ),
                        (ZZ_cplx_in0_dish32_polr1, ZZ_cplx_in1_dish32_polr1),
                        (YY_cplx0_dish32_polr1, YY_cplx1_dish32_polr1),
                    )
                    (YY_cplx0_dish33_polr1, YY_cplx1_dish33_polr1) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish33_polr1,
                            Γ³_cplx1_cplx_in0_dish33_polr1,
                            Γ³_cplx0_cplx_in1_dish33_polr1,
                            Γ³_cplx1_cplx_in1_dish33_polr1,
                        ),
                        (ZZ_cplx_in0_dish33_polr1, ZZ_cplx_in1_dish33_polr1),
                        (YY_cplx0_dish33_polr1, YY_cplx1_dish33_polr1),
                    )
                    E4_cplx0_dish0_polr0 = YY_cplx0_dish0_polr0
                    E4_cplx1_dish0_polr0 = YY_cplx1_dish0_polr0
                    E4_cplx0_dish1_polr0 = YY_cplx0_dish1_polr0
                    E4_cplx1_dish1_polr0 = YY_cplx1_dish1_polr0
                    E4_cplx0_dish32_polr0 = YY_cplx0_dish32_polr0
                    E4_cplx1_dish32_polr0 = YY_cplx1_dish32_polr0
                    E4_cplx0_dish33_polr0 = YY_cplx0_dish33_polr0
                    E4_cplx1_dish33_polr0 = YY_cplx1_dish33_polr0
                    E4_cplx0_dish0_polr1 = YY_cplx0_dish0_polr1
                    E4_cplx1_dish0_polr1 = YY_cplx1_dish0_polr1
                    E4_cplx0_dish1_polr1 = YY_cplx0_dish1_polr1
                    E4_cplx1_dish1_polr1 = YY_cplx1_dish1_polr1
                    E4_cplx0_dish32_polr1 = YY_cplx0_dish32_polr1
                    E4_cplx1_dish32_polr1 = YY_cplx1_dish32_polr1
                    E4_cplx0_dish33_polr1 = YY_cplx0_dish33_polr1
                    E4_cplx1_dish33_polr1 = YY_cplx1_dish33_polr1
                    E5_cplx0_dish0_polr0 = Gains * E4_cplx0_dish0_polr0
                    E5_cplx1_dish0_polr0 = Gains * E4_cplx1_dish0_polr0
                    E5_cplx0_dish1_polr0 = Gains * E4_cplx0_dish1_polr0
                    E5_cplx1_dish1_polr0 = Gains * E4_cplx1_dish1_polr0
                    E5_cplx0_dish32_polr0 = Gains * E4_cplx0_dish32_polr0
                    E5_cplx1_dish32_polr0 = Gains * E4_cplx1_dish32_polr0
                    E5_cplx0_dish33_polr0 = Gains * E4_cplx0_dish33_polr0
                    E5_cplx1_dish33_polr0 = Gains * E4_cplx1_dish33_polr0
                    E5_cplx0_dish0_polr1 = Gains * E4_cplx0_dish0_polr1
                    E5_cplx1_dish0_polr1 = Gains * E4_cplx1_dish0_polr1
                    E5_cplx0_dish1_polr1 = Gains * E4_cplx0_dish1_polr1
                    E5_cplx1_dish1_polr1 = Gains * E4_cplx1_dish1_polr1
                    E5_cplx0_dish32_polr1 = Gains * E4_cplx0_dish32_polr1
                    E5_cplx1_dish32_polr1 = Gains * E4_cplx1_dish32_polr1
                    E5_cplx0_dish33_polr1 = Gains * E4_cplx0_dish33_polr1
                    E5_cplx1_dish33_polr1 = Gains * E4_cplx1_dish33_polr1
                    E5_cplx0_dish0_polr0 = clamp(E5_cplx0_dish0_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish0_polr0 = clamp(E5_cplx1_dish0_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish1_polr0 = clamp(E5_cplx0_dish1_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish1_polr0 = clamp(E5_cplx1_dish1_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish32_polr0 = clamp(E5_cplx0_dish32_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish32_polr0 = clamp(E5_cplx1_dish32_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish33_polr0 = clamp(E5_cplx0_dish33_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish33_polr0 = clamp(E5_cplx1_dish33_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish0_polr1 = clamp(E5_cplx0_dish0_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish0_polr1 = clamp(E5_cplx1_dish0_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish1_polr1 = clamp(E5_cplx0_dish1_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish1_polr1 = clamp(E5_cplx1_dish1_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish32_polr1 = clamp(E5_cplx0_dish32_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish32_polr1 = clamp(E5_cplx1_dish32_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish33_polr1 = clamp(E5_cplx0_dish33_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish33_polr1 = clamp(E5_cplx1_dish33_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    F̄_out_dish0_polr0 = Int4x8((
                        E5_cplx0_dish0_polr0, E5_cplx1_dish0_polr0, E5_cplx0_dish1_polr0, E5_cplx1_dish1_polr0
                    ))
                    F̄_out_dish32_polr0 = Int4x8((
                        E5_cplx0_dish32_polr0, E5_cplx1_dish32_polr0, E5_cplx0_dish33_polr0, E5_cplx1_dish33_polr0
                    ))
                    F̄_out_dish0_polr1 = Int4x8((
                        E5_cplx0_dish0_polr1, E5_cplx1_dish0_polr1, E5_cplx0_dish1_polr1, E5_cplx1_dish1_polr1
                    ))
                    F̄_out_dish32_polr1 = Int4x8((
                        E5_cplx0_dish32_polr1, E5_cplx1_dish32_polr1, E5_cplx0_dish33_polr1, E5_cplx1_dish33_polr1
                    ))
                    if true
                        F̄_shared[(((((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 8) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 2) * 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) * 64) ÷ 2) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 32 + 0 + (((((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 16) + 0) + 0x01] =
                            F̄_out_dish0_polr0
                    end
                    if true
                        F̄_shared[(((((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 8) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 2) * 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) * 64) ÷ 2) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 2) % 2) * 32 + 0 + (((((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 4) % 16) + 0) + 0x01] =
                            F̄_out_dish32_polr0
                    end
                    if true
                        F̄_shared[(((((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 8) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 2) * 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) * 64) ÷ 2) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 32 + 16 + (((((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 16) + 0) + 0x01] =
                            F̄_out_dish0_polr1
                    end
                    if true
                        F̄_shared[(((((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 8) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 2) * 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) * 64) ÷ 2) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 2) % 2) * 32 + 16 + (((((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 4) % 16) + 0) + 0x01] =
                            F̄_out_dish32_polr1
                    end
                    F_ringbuf_m0_dish0_polr0 = F_ringbuf_dish0_mtaps0_polr0
                    F_ringbuf_m1_dish0_polr0 = F_ringbuf_dish0_mtaps1_polr0
                    F_ringbuf_m2_dish0_polr0 = F_ringbuf_dish0_mtaps2_polr0
                    F_ringbuf_m0_dish32_polr0 = F_ringbuf_dish32_mtaps0_polr0
                    F_ringbuf_m1_dish32_polr0 = F_ringbuf_dish32_mtaps1_polr0
                    F_ringbuf_m2_dish32_polr0 = F_ringbuf_dish32_mtaps2_polr0
                    F_ringbuf_m0_dish0_polr1 = F_ringbuf_dish0_mtaps0_polr1
                    F_ringbuf_m1_dish0_polr1 = F_ringbuf_dish0_mtaps1_polr1
                    F_ringbuf_m2_dish0_polr1 = F_ringbuf_dish0_mtaps2_polr1
                    F_ringbuf_m0_dish32_polr1 = F_ringbuf_dish32_mtaps0_polr1
                    F_ringbuf_m1_dish32_polr1 = F_ringbuf_dish32_mtaps1_polr1
                    F_ringbuf_m2_dish32_polr1 = F_ringbuf_dish32_mtaps2_polr1
                    F_ringbuf_m0_dish0_polr0 = F_ringbuf_m1_dish0_polr0
                    F_ringbuf_m0_dish32_polr0 = F_ringbuf_m1_dish32_polr0
                    F_ringbuf_m0_dish0_polr1 = F_ringbuf_m1_dish0_polr1
                    F_ringbuf_m0_dish32_polr1 = F_ringbuf_m1_dish32_polr1
                    F_ringbuf_m1_dish0_polr0 = F_ringbuf_m2_dish0_polr0
                    F_ringbuf_m1_dish32_polr0 = F_ringbuf_m2_dish32_polr0
                    F_ringbuf_m1_dish0_polr1 = F_ringbuf_m2_dish0_polr1
                    F_ringbuf_m1_dish32_polr1 = F_ringbuf_m2_dish32_polr1
                    F_ringbuf_m2_dish0_polr0 = F_in_dish0_polr0
                    F_ringbuf_m2_dish32_polr0 = F_in_dish32_polr0
                    F_ringbuf_m2_dish0_polr1 = F_in_dish0_polr1
                    F_ringbuf_m2_dish32_polr1 = F_in_dish32_polr1
                    F_ringbuf_dish0_mtaps0_polr0 = F_ringbuf_m0_dish0_polr0
                    F_ringbuf_dish0_mtaps1_polr0 = F_ringbuf_m1_dish0_polr0
                    F_ringbuf_dish0_mtaps2_polr0 = F_ringbuf_m2_dish0_polr0
                    F_ringbuf_dish32_mtaps0_polr0 = F_ringbuf_m0_dish32_polr0
                    F_ringbuf_dish32_mtaps1_polr0 = F_ringbuf_m1_dish32_polr0
                    F_ringbuf_dish32_mtaps2_polr0 = F_ringbuf_m2_dish32_polr0
                    F_ringbuf_dish0_mtaps0_polr1 = F_ringbuf_m0_dish0_polr1
                    F_ringbuf_dish0_mtaps1_polr1 = F_ringbuf_m1_dish0_polr1
                    F_ringbuf_dish0_mtaps2_polr1 = F_ringbuf_m2_dish0_polr1
                    F_ringbuf_dish32_mtaps0_polr1 = F_ringbuf_m0_dish32_polr1
                    F_ringbuf_dish32_mtaps1_polr1 = F_ringbuf_m1_dish32_polr1
                    F_ringbuf_dish32_mtaps2_polr1 = F_ringbuf_m2_dish32_polr1
                end
                let
                    dish = 48
                    F_in_dish0_polr0 = F_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 32 + 0 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 16) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) % 2) * 1040 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 2) % 2) * 520 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 4) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 8) % 2) * 130) + 0x01]
                    F_in_dish32_polr0 = F_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 2) % 2) * 32 + 0 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 16) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) % 2) * 1040 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 2) % 2) * 520 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 4) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 8) % 2) * 130) + 0x01]
                    F_in_dish0_polr1 = F_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 32 + 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 16) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) % 2) * 1040 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 2) % 2) * 520 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 4) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 8) % 2) * 130) + 0x01]
                    F_in_dish32_polr1 = F_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 2) % 2) * 32 + 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 4) % 16 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 16) % 2) * 65 + ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) % 2) * 1040 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 2) % 2) * 520 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 4) % 2) * 260 + (((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 8 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4) + ((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64) + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 4) ÷ 8) % 2) * 130) + 0x01]
                    (E_cplx0_dish0_polr0, E_cplx1_dish0_polr0, E_cplx0_dish1_polr0, E_cplx1_dish1_polr0) = convert(
                        NTuple{4,Float16x2}, F_in_dish0_polr0
                    )
                    (E_cplx0_dish32_polr0, E_cplx1_dish32_polr0, E_cplx0_dish33_polr0, E_cplx1_dish33_polr0) = convert(
                        NTuple{4,Float16x2}, F_in_dish32_polr0
                    )
                    (E_cplx0_dish0_polr1, E_cplx1_dish0_polr1, E_cplx0_dish1_polr1, E_cplx1_dish1_polr1) = convert(
                        NTuple{4,Float16x2}, F_in_dish0_polr1
                    )
                    (E_cplx0_dish32_polr1, E_cplx1_dish32_polr1, E_cplx0_dish33_polr1, E_cplx1_dish33_polr1) = convert(
                        NTuple{4,Float16x2}, F_in_dish32_polr1
                    )
                    W_m0 = Wpfb_mtaps0
                    W_m1 = Wpfb_mtaps1
                    W_m2 = Wpfb_mtaps2
                    W_m3 = Wpfb_mtaps3
                    E2_cplx0_dish0_polr0 = -W_m3 * E_cplx0_dish0_polr0
                    E2_cplx1_dish0_polr0 = -W_m3 * E_cplx1_dish0_polr0
                    E2_cplx0_dish1_polr0 = -W_m3 * E_cplx0_dish1_polr0
                    E2_cplx1_dish1_polr0 = -W_m3 * E_cplx1_dish1_polr0
                    E2_cplx0_dish32_polr0 = -W_m3 * E_cplx0_dish32_polr0
                    E2_cplx1_dish32_polr0 = -W_m3 * E_cplx1_dish32_polr0
                    E2_cplx0_dish33_polr0 = -W_m3 * E_cplx0_dish33_polr0
                    E2_cplx1_dish33_polr0 = -W_m3 * E_cplx1_dish33_polr0
                    E2_cplx0_dish0_polr1 = -W_m3 * E_cplx0_dish0_polr1
                    E2_cplx1_dish0_polr1 = -W_m3 * E_cplx1_dish0_polr1
                    E2_cplx0_dish1_polr1 = -W_m3 * E_cplx0_dish1_polr1
                    E2_cplx1_dish1_polr1 = -W_m3 * E_cplx1_dish1_polr1
                    E2_cplx0_dish32_polr1 = -W_m3 * E_cplx0_dish32_polr1
                    E2_cplx1_dish32_polr1 = -W_m3 * E_cplx1_dish32_polr1
                    E2_cplx0_dish33_polr1 = -W_m3 * E_cplx0_dish33_polr1
                    E2_cplx1_dish33_polr1 = -W_m3 * E_cplx1_dish33_polr1
                    F_ringbuf_m0_dish0_polr0 = F_ringbuf_dish0_mtaps0_polr0
                    F_ringbuf_m1_dish0_polr0 = F_ringbuf_dish0_mtaps1_polr0
                    F_ringbuf_m2_dish0_polr0 = F_ringbuf_dish0_mtaps2_polr0
                    F_ringbuf_m0_dish32_polr0 = F_ringbuf_dish32_mtaps0_polr0
                    F_ringbuf_m1_dish32_polr0 = F_ringbuf_dish32_mtaps1_polr0
                    F_ringbuf_m2_dish32_polr0 = F_ringbuf_dish32_mtaps2_polr0
                    F_ringbuf_m0_dish0_polr1 = F_ringbuf_dish0_mtaps0_polr1
                    F_ringbuf_m1_dish0_polr1 = F_ringbuf_dish0_mtaps1_polr1
                    F_ringbuf_m2_dish0_polr1 = F_ringbuf_dish0_mtaps2_polr1
                    F_ringbuf_m0_dish32_polr1 = F_ringbuf_dish32_mtaps0_polr1
                    F_ringbuf_m1_dish32_polr1 = F_ringbuf_dish32_mtaps1_polr1
                    F_ringbuf_m2_dish32_polr1 = F_ringbuf_dish32_mtaps2_polr1
                    (E_ringbuf_m0_cplx0_dish0_polr0, E_ringbuf_m0_cplx1_dish0_polr0, E_ringbuf_m0_cplx0_dish1_polr0, E_ringbuf_m0_cplx1_dish1_polr0) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m0_dish0_polr0
                    )
                    (E_ringbuf_m0_cplx0_dish32_polr0, E_ringbuf_m0_cplx1_dish32_polr0, E_ringbuf_m0_cplx0_dish33_polr0, E_ringbuf_m0_cplx1_dish33_polr0) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m0_dish32_polr0
                    )
                    (E_ringbuf_m0_cplx0_dish0_polr1, E_ringbuf_m0_cplx1_dish0_polr1, E_ringbuf_m0_cplx0_dish1_polr1, E_ringbuf_m0_cplx1_dish1_polr1) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m0_dish0_polr1
                    )
                    (E_ringbuf_m0_cplx0_dish32_polr1, E_ringbuf_m0_cplx1_dish32_polr1, E_ringbuf_m0_cplx0_dish33_polr1, E_ringbuf_m0_cplx1_dish33_polr1) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m0_dish32_polr1
                    )
                    E2_cplx0_dish0_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish0_polr0, E2_cplx0_dish0_polr0)
                    E2_cplx1_dish0_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish0_polr0, E2_cplx1_dish0_polr0)
                    E2_cplx0_dish1_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish1_polr0, E2_cplx0_dish1_polr0)
                    E2_cplx1_dish1_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish1_polr0, E2_cplx1_dish1_polr0)
                    E2_cplx0_dish32_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish32_polr0, E2_cplx0_dish32_polr0)
                    E2_cplx1_dish32_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish32_polr0, E2_cplx1_dish32_polr0)
                    E2_cplx0_dish33_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish33_polr0, E2_cplx0_dish33_polr0)
                    E2_cplx1_dish33_polr0 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish33_polr0, E2_cplx1_dish33_polr0)
                    E2_cplx0_dish0_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish0_polr1, E2_cplx0_dish0_polr1)
                    E2_cplx1_dish0_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish0_polr1, E2_cplx1_dish0_polr1)
                    E2_cplx0_dish1_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish1_polr1, E2_cplx0_dish1_polr1)
                    E2_cplx1_dish1_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish1_polr1, E2_cplx1_dish1_polr1)
                    E2_cplx0_dish32_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish32_polr1, E2_cplx0_dish32_polr1)
                    E2_cplx1_dish32_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish32_polr1, E2_cplx1_dish32_polr1)
                    E2_cplx0_dish33_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx0_dish33_polr1, E2_cplx0_dish33_polr1)
                    E2_cplx1_dish33_polr1 = muladd(+W_m0, E_ringbuf_m0_cplx1_dish33_polr1, E2_cplx1_dish33_polr1)
                    (E_ringbuf_m1_cplx0_dish0_polr0, E_ringbuf_m1_cplx1_dish0_polr0, E_ringbuf_m1_cplx0_dish1_polr0, E_ringbuf_m1_cplx1_dish1_polr0) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m1_dish0_polr0
                    )
                    (E_ringbuf_m1_cplx0_dish32_polr0, E_ringbuf_m1_cplx1_dish32_polr0, E_ringbuf_m1_cplx0_dish33_polr0, E_ringbuf_m1_cplx1_dish33_polr0) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m1_dish32_polr0
                    )
                    (E_ringbuf_m1_cplx0_dish0_polr1, E_ringbuf_m1_cplx1_dish0_polr1, E_ringbuf_m1_cplx0_dish1_polr1, E_ringbuf_m1_cplx1_dish1_polr1) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m1_dish0_polr1
                    )
                    (E_ringbuf_m1_cplx0_dish32_polr1, E_ringbuf_m1_cplx1_dish32_polr1, E_ringbuf_m1_cplx0_dish33_polr1, E_ringbuf_m1_cplx1_dish33_polr1) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m1_dish32_polr1
                    )
                    E2_cplx0_dish0_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish0_polr0, E2_cplx0_dish0_polr0)
                    E2_cplx1_dish0_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish0_polr0, E2_cplx1_dish0_polr0)
                    E2_cplx0_dish1_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish1_polr0, E2_cplx0_dish1_polr0)
                    E2_cplx1_dish1_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish1_polr0, E2_cplx1_dish1_polr0)
                    E2_cplx0_dish32_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish32_polr0, E2_cplx0_dish32_polr0)
                    E2_cplx1_dish32_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish32_polr0, E2_cplx1_dish32_polr0)
                    E2_cplx0_dish33_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish33_polr0, E2_cplx0_dish33_polr0)
                    E2_cplx1_dish33_polr0 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish33_polr0, E2_cplx1_dish33_polr0)
                    E2_cplx0_dish0_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish0_polr1, E2_cplx0_dish0_polr1)
                    E2_cplx1_dish0_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish0_polr1, E2_cplx1_dish0_polr1)
                    E2_cplx0_dish1_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish1_polr1, E2_cplx0_dish1_polr1)
                    E2_cplx1_dish1_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish1_polr1, E2_cplx1_dish1_polr1)
                    E2_cplx0_dish32_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish32_polr1, E2_cplx0_dish32_polr1)
                    E2_cplx1_dish32_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish32_polr1, E2_cplx1_dish32_polr1)
                    E2_cplx0_dish33_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx0_dish33_polr1, E2_cplx0_dish33_polr1)
                    E2_cplx1_dish33_polr1 = muladd(-W_m1, E_ringbuf_m1_cplx1_dish33_polr1, E2_cplx1_dish33_polr1)
                    (E_ringbuf_m2_cplx0_dish0_polr0, E_ringbuf_m2_cplx1_dish0_polr0, E_ringbuf_m2_cplx0_dish1_polr0, E_ringbuf_m2_cplx1_dish1_polr0) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m2_dish0_polr0
                    )
                    (E_ringbuf_m2_cplx0_dish32_polr0, E_ringbuf_m2_cplx1_dish32_polr0, E_ringbuf_m2_cplx0_dish33_polr0, E_ringbuf_m2_cplx1_dish33_polr0) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m2_dish32_polr0
                    )
                    (E_ringbuf_m2_cplx0_dish0_polr1, E_ringbuf_m2_cplx1_dish0_polr1, E_ringbuf_m2_cplx0_dish1_polr1, E_ringbuf_m2_cplx1_dish1_polr1) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m2_dish0_polr1
                    )
                    (E_ringbuf_m2_cplx0_dish32_polr1, E_ringbuf_m2_cplx1_dish32_polr1, E_ringbuf_m2_cplx0_dish33_polr1, E_ringbuf_m2_cplx1_dish33_polr1) = convert(
                        NTuple{4,Float16x2}, F_ringbuf_m2_dish32_polr1
                    )
                    E2_cplx0_dish0_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish0_polr0, E2_cplx0_dish0_polr0)
                    E2_cplx1_dish0_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish0_polr0, E2_cplx1_dish0_polr0)
                    E2_cplx0_dish1_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish1_polr0, E2_cplx0_dish1_polr0)
                    E2_cplx1_dish1_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish1_polr0, E2_cplx1_dish1_polr0)
                    E2_cplx0_dish32_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish32_polr0, E2_cplx0_dish32_polr0)
                    E2_cplx1_dish32_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish32_polr0, E2_cplx1_dish32_polr0)
                    E2_cplx0_dish33_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish33_polr0, E2_cplx0_dish33_polr0)
                    E2_cplx1_dish33_polr0 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish33_polr0, E2_cplx1_dish33_polr0)
                    E2_cplx0_dish0_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish0_polr1, E2_cplx0_dish0_polr1)
                    E2_cplx1_dish0_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish0_polr1, E2_cplx1_dish0_polr1)
                    E2_cplx0_dish1_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish1_polr1, E2_cplx0_dish1_polr1)
                    E2_cplx1_dish1_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish1_polr1, E2_cplx1_dish1_polr1)
                    E2_cplx0_dish32_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish32_polr1, E2_cplx0_dish32_polr1)
                    E2_cplx1_dish32_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish32_polr1, E2_cplx1_dish32_polr1)
                    E2_cplx0_dish33_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx0_dish33_polr1, E2_cplx0_dish33_polr1)
                    E2_cplx1_dish33_polr1 = muladd(+W_m2, E_ringbuf_m2_cplx1_dish33_polr1, E2_cplx1_dish33_polr1)
                    E2re_dish0_polr0 = E2_cplx0_dish0_polr0
                    E2im_dish0_polr0 = E2_cplx1_dish0_polr0
                    E2re_dish1_polr0 = E2_cplx0_dish1_polr0
                    E2im_dish1_polr0 = E2_cplx1_dish1_polr0
                    E2re_dish32_polr0 = E2_cplx0_dish32_polr0
                    E2im_dish32_polr0 = E2_cplx1_dish32_polr0
                    E2re_dish33_polr0 = E2_cplx0_dish33_polr0
                    E2im_dish33_polr0 = E2_cplx1_dish33_polr0
                    E2re_dish0_polr1 = E2_cplx0_dish0_polr1
                    E2im_dish0_polr1 = E2_cplx1_dish0_polr1
                    E2re_dish1_polr1 = E2_cplx0_dish1_polr1
                    E2im_dish1_polr1 = E2_cplx1_dish1_polr1
                    E2re_dish32_polr1 = E2_cplx0_dish32_polr1
                    E2im_dish32_polr1 = E2_cplx1_dish32_polr1
                    E2re_dish33_polr1 = E2_cplx0_dish33_polr1
                    E2im_dish33_polr1 = E2_cplx1_dish33_polr1
                    Xre = X_cplx0
                    Xim = X_cplx1
                    E3re_dish0_polr0 = muladd(Xre, E2re_dish0_polr0, -Xim * E2im_dish0_polr0)
                    E3re_dish1_polr0 = muladd(Xre, E2re_dish1_polr0, -Xim * E2im_dish1_polr0)
                    E3re_dish32_polr0 = muladd(Xre, E2re_dish32_polr0, -Xim * E2im_dish32_polr0)
                    E3re_dish33_polr0 = muladd(Xre, E2re_dish33_polr0, -Xim * E2im_dish33_polr0)
                    E3re_dish0_polr1 = muladd(Xre, E2re_dish0_polr1, -Xim * E2im_dish0_polr1)
                    E3re_dish1_polr1 = muladd(Xre, E2re_dish1_polr1, -Xim * E2im_dish1_polr1)
                    E3re_dish32_polr1 = muladd(Xre, E2re_dish32_polr1, -Xim * E2im_dish32_polr1)
                    E3re_dish33_polr1 = muladd(Xre, E2re_dish33_polr1, -Xim * E2im_dish33_polr1)
                    E3im_dish0_polr0 = muladd(Xre, E2im_dish0_polr0, Xim * E2re_dish0_polr0)
                    E3im_dish1_polr0 = muladd(Xre, E2im_dish1_polr0, Xim * E2re_dish1_polr0)
                    E3im_dish32_polr0 = muladd(Xre, E2im_dish32_polr0, Xim * E2re_dish32_polr0)
                    E3im_dish33_polr0 = muladd(Xre, E2im_dish33_polr0, Xim * E2re_dish33_polr0)
                    E3im_dish0_polr1 = muladd(Xre, E2im_dish0_polr1, Xim * E2re_dish0_polr1)
                    E3im_dish1_polr1 = muladd(Xre, E2im_dish1_polr1, Xim * E2re_dish1_polr1)
                    E3im_dish32_polr1 = muladd(Xre, E2im_dish32_polr1, Xim * E2re_dish32_polr1)
                    E3im_dish33_polr1 = muladd(Xre, E2im_dish33_polr1, Xim * E2re_dish33_polr1)
                    E3_cplx0_dish0_polr0 = E3re_dish0_polr0
                    E3_cplx1_dish0_polr0 = E3im_dish0_polr0
                    E3_cplx0_dish1_polr0 = E3re_dish1_polr0
                    E3_cplx1_dish1_polr0 = E3im_dish1_polr0
                    E3_cplx0_dish32_polr0 = E3re_dish32_polr0
                    E3_cplx1_dish32_polr0 = E3im_dish32_polr0
                    E3_cplx0_dish33_polr0 = E3re_dish33_polr0
                    E3_cplx1_dish33_polr0 = E3im_dish33_polr0
                    E3_cplx0_dish0_polr1 = E3re_dish0_polr1
                    E3_cplx1_dish0_polr1 = E3im_dish0_polr1
                    E3_cplx0_dish1_polr1 = E3re_dish1_polr1
                    E3_cplx1_dish1_polr1 = E3im_dish1_polr1
                    E3_cplx0_dish32_polr1 = E3re_dish32_polr1
                    E3_cplx1_dish32_polr1 = E3im_dish32_polr1
                    E3_cplx0_dish33_polr1 = E3re_dish33_polr1
                    E3_cplx1_dish33_polr1 = E3im_dish33_polr1
                    XX_cplx0_dish0_polr0 = E3_cplx0_dish0_polr0
                    XX_cplx1_dish0_polr0 = E3_cplx1_dish0_polr0
                    XX_cplx0_dish1_polr0 = E3_cplx0_dish1_polr0
                    XX_cplx1_dish1_polr0 = E3_cplx1_dish1_polr0
                    XX_cplx0_dish32_polr0 = E3_cplx0_dish32_polr0
                    XX_cplx1_dish32_polr0 = E3_cplx1_dish32_polr0
                    XX_cplx0_dish33_polr0 = E3_cplx0_dish33_polr0
                    XX_cplx1_dish33_polr0 = E3_cplx1_dish33_polr0
                    XX_cplx0_dish0_polr1 = E3_cplx0_dish0_polr1
                    XX_cplx1_dish0_polr1 = E3_cplx1_dish0_polr1
                    XX_cplx0_dish1_polr1 = E3_cplx0_dish1_polr1
                    XX_cplx1_dish1_polr1 = E3_cplx1_dish1_polr1
                    XX_cplx0_dish32_polr1 = E3_cplx0_dish32_polr1
                    XX_cplx1_dish32_polr1 = E3_cplx1_dish32_polr1
                    XX_cplx0_dish33_polr1 = E3_cplx0_dish33_polr1
                    XX_cplx1_dish33_polr1 = E3_cplx1_dish33_polr1
                    XXre_dish0_polr0 = XX_cplx0_dish0_polr0
                    XXim_dish0_polr0 = XX_cplx1_dish0_polr0
                    XXre_dish1_polr0 = XX_cplx0_dish1_polr0
                    XXim_dish1_polr0 = XX_cplx1_dish1_polr0
                    XXre_dish32_polr0 = XX_cplx0_dish32_polr0
                    XXim_dish32_polr0 = XX_cplx1_dish32_polr0
                    XXre_dish33_polr0 = XX_cplx0_dish33_polr0
                    XXim_dish33_polr0 = XX_cplx1_dish33_polr0
                    XXre_dish0_polr1 = XX_cplx0_dish0_polr1
                    XXim_dish0_polr1 = XX_cplx1_dish0_polr1
                    XXre_dish1_polr1 = XX_cplx0_dish1_polr1
                    XXim_dish1_polr1 = XX_cplx1_dish1_polr1
                    XXre_dish32_polr1 = XX_cplx0_dish32_polr1
                    XXim_dish32_polr1 = XX_cplx1_dish32_polr1
                    XXre_dish33_polr1 = XX_cplx0_dish33_polr1
                    XXim_dish33_polr1 = XX_cplx1_dish33_polr1
                    XX_cplx_in0_dish0_polr0 = XXre_dish0_polr0
                    XX_cplx_in1_dish0_polr0 = XXim_dish0_polr0
                    XX_cplx_in0_dish1_polr0 = XXre_dish1_polr0
                    XX_cplx_in1_dish1_polr0 = XXim_dish1_polr0
                    XX_cplx_in0_dish32_polr0 = XXre_dish32_polr0
                    XX_cplx_in1_dish32_polr0 = XXim_dish32_polr0
                    XX_cplx_in0_dish33_polr0 = XXre_dish33_polr0
                    XX_cplx_in1_dish33_polr0 = XXim_dish33_polr0
                    XX_cplx_in0_dish0_polr1 = XXre_dish0_polr1
                    XX_cplx_in1_dish0_polr1 = XXim_dish0_polr1
                    XX_cplx_in0_dish1_polr1 = XXre_dish1_polr1
                    XX_cplx_in1_dish1_polr1 = XXim_dish1_polr1
                    XX_cplx_in0_dish32_polr1 = XXre_dish32_polr1
                    XX_cplx_in1_dish32_polr1 = XXim_dish32_polr1
                    XX_cplx_in0_dish33_polr1 = XXre_dish33_polr1
                    XX_cplx_in1_dish33_polr1 = XXim_dish33_polr1
                    WW_cplx0_dish0_polr0 = zero(Float16x2)
                    WW_cplx1_dish0_polr0 = zero(Float16x2)
                    WW_cplx0_dish1_polr0 = zero(Float16x2)
                    WW_cplx1_dish1_polr0 = zero(Float16x2)
                    WW_cplx0_dish32_polr0 = zero(Float16x2)
                    WW_cplx1_dish32_polr0 = zero(Float16x2)
                    WW_cplx0_dish33_polr0 = zero(Float16x2)
                    WW_cplx1_dish33_polr0 = zero(Float16x2)
                    WW_cplx0_dish0_polr1 = zero(Float16x2)
                    WW_cplx1_dish0_polr1 = zero(Float16x2)
                    WW_cplx0_dish1_polr1 = zero(Float16x2)
                    WW_cplx1_dish1_polr1 = zero(Float16x2)
                    WW_cplx0_dish32_polr1 = zero(Float16x2)
                    WW_cplx1_dish32_polr1 = zero(Float16x2)
                    WW_cplx0_dish33_polr1 = zero(Float16x2)
                    WW_cplx1_dish33_polr1 = zero(Float16x2)
                    (WW_cplx0_dish0_polr0, WW_cplx1_dish0_polr0) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish0_polr0, XX_cplx_in1_dish0_polr0),
                        (WW_cplx0_dish0_polr0, WW_cplx1_dish0_polr0),
                    )
                    (WW_cplx0_dish1_polr0, WW_cplx1_dish1_polr0) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish1_polr0, XX_cplx_in1_dish1_polr0),
                        (WW_cplx0_dish1_polr0, WW_cplx1_dish1_polr0),
                    )
                    (WW_cplx0_dish32_polr0, WW_cplx1_dish32_polr0) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish32_polr0, XX_cplx_in1_dish32_polr0),
                        (WW_cplx0_dish32_polr0, WW_cplx1_dish32_polr0),
                    )
                    (WW_cplx0_dish33_polr0, WW_cplx1_dish33_polr0) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish33_polr0, XX_cplx_in1_dish33_polr0),
                        (WW_cplx0_dish33_polr0, WW_cplx1_dish33_polr0),
                    )
                    (WW_cplx0_dish0_polr1, WW_cplx1_dish0_polr1) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish0_polr1, XX_cplx_in1_dish0_polr1),
                        (WW_cplx0_dish0_polr1, WW_cplx1_dish0_polr1),
                    )
                    (WW_cplx0_dish1_polr1, WW_cplx1_dish1_polr1) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish1_polr1, XX_cplx_in1_dish1_polr1),
                        (WW_cplx0_dish1_polr1, WW_cplx1_dish1_polr1),
                    )
                    (WW_cplx0_dish32_polr1, WW_cplx1_dish32_polr1) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish32_polr1, XX_cplx_in1_dish32_polr1),
                        (WW_cplx0_dish32_polr1, WW_cplx1_dish32_polr1),
                    )
                    (WW_cplx0_dish33_polr1, WW_cplx1_dish33_polr1) = IndexSpaces.mma_m16n8k16(
                        (Γ¹_cplx0_cplx_in0, Γ¹_cplx1_cplx_in0, Γ¹_cplx0_cplx_in1, Γ¹_cplx1_cplx_in1),
                        (XX_cplx_in0_dish33_polr1, XX_cplx_in1_dish33_polr1),
                        (WW_cplx0_dish33_polr1, WW_cplx1_dish33_polr1),
                    )
                    Γ²re = Γ²_cplx0
                    Γ²im = Γ²_cplx1
                    WWre_dish0_polr0 = WW_cplx0_dish0_polr0
                    WWim_dish0_polr0 = WW_cplx1_dish0_polr0
                    WWre_dish1_polr0 = WW_cplx0_dish1_polr0
                    WWim_dish1_polr0 = WW_cplx1_dish1_polr0
                    WWre_dish32_polr0 = WW_cplx0_dish32_polr0
                    WWim_dish32_polr0 = WW_cplx1_dish32_polr0
                    WWre_dish33_polr0 = WW_cplx0_dish33_polr0
                    WWim_dish33_polr0 = WW_cplx1_dish33_polr0
                    WWre_dish0_polr1 = WW_cplx0_dish0_polr1
                    WWim_dish0_polr1 = WW_cplx1_dish0_polr1
                    WWre_dish1_polr1 = WW_cplx0_dish1_polr1
                    WWim_dish1_polr1 = WW_cplx1_dish1_polr1
                    WWre_dish32_polr1 = WW_cplx0_dish32_polr1
                    WWim_dish32_polr1 = WW_cplx1_dish32_polr1
                    WWre_dish33_polr1 = WW_cplx0_dish33_polr1
                    WWim_dish33_polr1 = WW_cplx1_dish33_polr1
                    ZZre_dish0_polr0 = muladd(Γ²re, WWre_dish0_polr0, -Γ²im * WWim_dish0_polr0)
                    ZZre_dish1_polr0 = muladd(Γ²re, WWre_dish1_polr0, -Γ²im * WWim_dish1_polr0)
                    ZZre_dish32_polr0 = muladd(Γ²re, WWre_dish32_polr0, -Γ²im * WWim_dish32_polr0)
                    ZZre_dish33_polr0 = muladd(Γ²re, WWre_dish33_polr0, -Γ²im * WWim_dish33_polr0)
                    ZZre_dish0_polr1 = muladd(Γ²re, WWre_dish0_polr1, -Γ²im * WWim_dish0_polr1)
                    ZZre_dish1_polr1 = muladd(Γ²re, WWre_dish1_polr1, -Γ²im * WWim_dish1_polr1)
                    ZZre_dish32_polr1 = muladd(Γ²re, WWre_dish32_polr1, -Γ²im * WWim_dish32_polr1)
                    ZZre_dish33_polr1 = muladd(Γ²re, WWre_dish33_polr1, -Γ²im * WWim_dish33_polr1)
                    ZZim_dish0_polr0 = muladd(Γ²re, WWim_dish0_polr0, Γ²im * WWre_dish0_polr0)
                    ZZim_dish1_polr0 = muladd(Γ²re, WWim_dish1_polr0, Γ²im * WWre_dish1_polr0)
                    ZZim_dish32_polr0 = muladd(Γ²re, WWim_dish32_polr0, Γ²im * WWre_dish32_polr0)
                    ZZim_dish33_polr0 = muladd(Γ²re, WWim_dish33_polr0, Γ²im * WWre_dish33_polr0)
                    ZZim_dish0_polr1 = muladd(Γ²re, WWim_dish0_polr1, Γ²im * WWre_dish0_polr1)
                    ZZim_dish1_polr1 = muladd(Γ²re, WWim_dish1_polr1, Γ²im * WWre_dish1_polr1)
                    ZZim_dish32_polr1 = muladd(Γ²re, WWim_dish32_polr1, Γ²im * WWre_dish32_polr1)
                    ZZim_dish33_polr1 = muladd(Γ²re, WWim_dish33_polr1, Γ²im * WWre_dish33_polr1)
                    ZZ_cplx0_dish0_polr0 = ZZre_dish0_polr0
                    ZZ_cplx1_dish0_polr0 = ZZim_dish0_polr0
                    ZZ_cplx0_dish1_polr0 = ZZre_dish1_polr0
                    ZZ_cplx1_dish1_polr0 = ZZim_dish1_polr0
                    ZZ_cplx0_dish32_polr0 = ZZre_dish32_polr0
                    ZZ_cplx1_dish32_polr0 = ZZim_dish32_polr0
                    ZZ_cplx0_dish33_polr0 = ZZre_dish33_polr0
                    ZZ_cplx1_dish33_polr0 = ZZim_dish33_polr0
                    ZZ_cplx0_dish0_polr1 = ZZre_dish0_polr1
                    ZZ_cplx1_dish0_polr1 = ZZim_dish0_polr1
                    ZZ_cplx0_dish1_polr1 = ZZre_dish1_polr1
                    ZZ_cplx1_dish1_polr1 = ZZim_dish1_polr1
                    ZZ_cplx0_dish32_polr1 = ZZre_dish32_polr1
                    ZZ_cplx1_dish32_polr1 = ZZim_dish32_polr1
                    ZZ_cplx0_dish33_polr1 = ZZre_dish33_polr1
                    ZZ_cplx1_dish33_polr1 = ZZim_dish33_polr1
                    ZZre_dish0_polr0 = ZZ_cplx0_dish0_polr0
                    ZZim_dish0_polr0 = ZZ_cplx1_dish0_polr0
                    ZZre_dish1_polr0 = ZZ_cplx0_dish1_polr0
                    ZZim_dish1_polr0 = ZZ_cplx1_dish1_polr0
                    ZZre_dish32_polr0 = ZZ_cplx0_dish32_polr0
                    ZZim_dish32_polr0 = ZZ_cplx1_dish32_polr0
                    ZZre_dish33_polr0 = ZZ_cplx0_dish33_polr0
                    ZZim_dish33_polr0 = ZZ_cplx1_dish33_polr0
                    ZZre_dish0_polr1 = ZZ_cplx0_dish0_polr1
                    ZZim_dish0_polr1 = ZZ_cplx1_dish0_polr1
                    ZZre_dish1_polr1 = ZZ_cplx0_dish1_polr1
                    ZZim_dish1_polr1 = ZZ_cplx1_dish1_polr1
                    ZZre_dish32_polr1 = ZZ_cplx0_dish32_polr1
                    ZZim_dish32_polr1 = ZZ_cplx1_dish32_polr1
                    ZZre_dish33_polr1 = ZZ_cplx0_dish33_polr1
                    ZZim_dish33_polr1 = ZZ_cplx1_dish33_polr1
                    ZZ_cplx_in0_dish0_polr0 = ZZre_dish0_polr0
                    ZZ_cplx_in1_dish0_polr0 = ZZim_dish0_polr0
                    ZZ_cplx_in0_dish1_polr0 = ZZre_dish1_polr0
                    ZZ_cplx_in1_dish1_polr0 = ZZim_dish1_polr0
                    ZZ_cplx_in0_dish32_polr0 = ZZre_dish32_polr0
                    ZZ_cplx_in1_dish32_polr0 = ZZim_dish32_polr0
                    ZZ_cplx_in0_dish33_polr0 = ZZre_dish33_polr0
                    ZZ_cplx_in1_dish33_polr0 = ZZim_dish33_polr0
                    ZZ_cplx_in0_dish0_polr1 = ZZre_dish0_polr1
                    ZZ_cplx_in1_dish0_polr1 = ZZim_dish0_polr1
                    ZZ_cplx_in0_dish1_polr1 = ZZre_dish1_polr1
                    ZZ_cplx_in1_dish1_polr1 = ZZim_dish1_polr1
                    ZZ_cplx_in0_dish32_polr1 = ZZre_dish32_polr1
                    ZZ_cplx_in1_dish32_polr1 = ZZim_dish32_polr1
                    ZZ_cplx_in0_dish33_polr1 = ZZre_dish33_polr1
                    ZZ_cplx_in1_dish33_polr1 = ZZim_dish33_polr1
                    YY_cplx0_dish0_polr0 = zero(Float16x2)
                    YY_cplx1_dish0_polr0 = zero(Float16x2)
                    YY_cplx0_dish1_polr0 = zero(Float16x2)
                    YY_cplx1_dish1_polr0 = zero(Float16x2)
                    YY_cplx0_dish32_polr0 = zero(Float16x2)
                    YY_cplx1_dish32_polr0 = zero(Float16x2)
                    YY_cplx0_dish33_polr0 = zero(Float16x2)
                    YY_cplx1_dish33_polr0 = zero(Float16x2)
                    YY_cplx0_dish0_polr1 = zero(Float16x2)
                    YY_cplx1_dish0_polr1 = zero(Float16x2)
                    YY_cplx0_dish1_polr1 = zero(Float16x2)
                    YY_cplx1_dish1_polr1 = zero(Float16x2)
                    YY_cplx0_dish32_polr1 = zero(Float16x2)
                    YY_cplx1_dish32_polr1 = zero(Float16x2)
                    YY_cplx0_dish33_polr1 = zero(Float16x2)
                    YY_cplx1_dish33_polr1 = zero(Float16x2)
                    (YY_cplx0_dish0_polr0, YY_cplx1_dish0_polr0) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish0_polr0,
                            Γ³_cplx1_cplx_in0_dish0_polr0,
                            Γ³_cplx0_cplx_in1_dish0_polr0,
                            Γ³_cplx1_cplx_in1_dish0_polr0,
                        ),
                        (ZZ_cplx_in0_dish0_polr0, ZZ_cplx_in1_dish0_polr0),
                        (YY_cplx0_dish0_polr0, YY_cplx1_dish0_polr0),
                    )
                    (YY_cplx0_dish1_polr0, YY_cplx1_dish1_polr0) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish1_polr0,
                            Γ³_cplx1_cplx_in0_dish1_polr0,
                            Γ³_cplx0_cplx_in1_dish1_polr0,
                            Γ³_cplx1_cplx_in1_dish1_polr0,
                        ),
                        (ZZ_cplx_in0_dish1_polr0, ZZ_cplx_in1_dish1_polr0),
                        (YY_cplx0_dish1_polr0, YY_cplx1_dish1_polr0),
                    )
                    (YY_cplx0_dish32_polr0, YY_cplx1_dish32_polr0) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish32_polr0,
                            Γ³_cplx1_cplx_in0_dish32_polr0,
                            Γ³_cplx0_cplx_in1_dish32_polr0,
                            Γ³_cplx1_cplx_in1_dish32_polr0,
                        ),
                        (ZZ_cplx_in0_dish32_polr0, ZZ_cplx_in1_dish32_polr0),
                        (YY_cplx0_dish32_polr0, YY_cplx1_dish32_polr0),
                    )
                    (YY_cplx0_dish33_polr0, YY_cplx1_dish33_polr0) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish33_polr0,
                            Γ³_cplx1_cplx_in0_dish33_polr0,
                            Γ³_cplx0_cplx_in1_dish33_polr0,
                            Γ³_cplx1_cplx_in1_dish33_polr0,
                        ),
                        (ZZ_cplx_in0_dish33_polr0, ZZ_cplx_in1_dish33_polr0),
                        (YY_cplx0_dish33_polr0, YY_cplx1_dish33_polr0),
                    )
                    (YY_cplx0_dish0_polr1, YY_cplx1_dish0_polr1) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish0_polr1,
                            Γ³_cplx1_cplx_in0_dish0_polr1,
                            Γ³_cplx0_cplx_in1_dish0_polr1,
                            Γ³_cplx1_cplx_in1_dish0_polr1,
                        ),
                        (ZZ_cplx_in0_dish0_polr1, ZZ_cplx_in1_dish0_polr1),
                        (YY_cplx0_dish0_polr1, YY_cplx1_dish0_polr1),
                    )
                    (YY_cplx0_dish1_polr1, YY_cplx1_dish1_polr1) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish1_polr1,
                            Γ³_cplx1_cplx_in0_dish1_polr1,
                            Γ³_cplx0_cplx_in1_dish1_polr1,
                            Γ³_cplx1_cplx_in1_dish1_polr1,
                        ),
                        (ZZ_cplx_in0_dish1_polr1, ZZ_cplx_in1_dish1_polr1),
                        (YY_cplx0_dish1_polr1, YY_cplx1_dish1_polr1),
                    )
                    (YY_cplx0_dish32_polr1, YY_cplx1_dish32_polr1) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish32_polr1,
                            Γ³_cplx1_cplx_in0_dish32_polr1,
                            Γ³_cplx0_cplx_in1_dish32_polr1,
                            Γ³_cplx1_cplx_in1_dish32_polr1,
                        ),
                        (ZZ_cplx_in0_dish32_polr1, ZZ_cplx_in1_dish32_polr1),
                        (YY_cplx0_dish32_polr1, YY_cplx1_dish32_polr1),
                    )
                    (YY_cplx0_dish33_polr1, YY_cplx1_dish33_polr1) = IndexSpaces.mma_m16n8k16(
                        (
                            Γ³_cplx0_cplx_in0_dish33_polr1,
                            Γ³_cplx1_cplx_in0_dish33_polr1,
                            Γ³_cplx0_cplx_in1_dish33_polr1,
                            Γ³_cplx1_cplx_in1_dish33_polr1,
                        ),
                        (ZZ_cplx_in0_dish33_polr1, ZZ_cplx_in1_dish33_polr1),
                        (YY_cplx0_dish33_polr1, YY_cplx1_dish33_polr1),
                    )
                    E4_cplx0_dish0_polr0 = YY_cplx0_dish0_polr0
                    E4_cplx1_dish0_polr0 = YY_cplx1_dish0_polr0
                    E4_cplx0_dish1_polr0 = YY_cplx0_dish1_polr0
                    E4_cplx1_dish1_polr0 = YY_cplx1_dish1_polr0
                    E4_cplx0_dish32_polr0 = YY_cplx0_dish32_polr0
                    E4_cplx1_dish32_polr0 = YY_cplx1_dish32_polr0
                    E4_cplx0_dish33_polr0 = YY_cplx0_dish33_polr0
                    E4_cplx1_dish33_polr0 = YY_cplx1_dish33_polr0
                    E4_cplx0_dish0_polr1 = YY_cplx0_dish0_polr1
                    E4_cplx1_dish0_polr1 = YY_cplx1_dish0_polr1
                    E4_cplx0_dish1_polr1 = YY_cplx0_dish1_polr1
                    E4_cplx1_dish1_polr1 = YY_cplx1_dish1_polr1
                    E4_cplx0_dish32_polr1 = YY_cplx0_dish32_polr1
                    E4_cplx1_dish32_polr1 = YY_cplx1_dish32_polr1
                    E4_cplx0_dish33_polr1 = YY_cplx0_dish33_polr1
                    E4_cplx1_dish33_polr1 = YY_cplx1_dish33_polr1
                    E5_cplx0_dish0_polr0 = Gains * E4_cplx0_dish0_polr0
                    E5_cplx1_dish0_polr0 = Gains * E4_cplx1_dish0_polr0
                    E5_cplx0_dish1_polr0 = Gains * E4_cplx0_dish1_polr0
                    E5_cplx1_dish1_polr0 = Gains * E4_cplx1_dish1_polr0
                    E5_cplx0_dish32_polr0 = Gains * E4_cplx0_dish32_polr0
                    E5_cplx1_dish32_polr0 = Gains * E4_cplx1_dish32_polr0
                    E5_cplx0_dish33_polr0 = Gains * E4_cplx0_dish33_polr0
                    E5_cplx1_dish33_polr0 = Gains * E4_cplx1_dish33_polr0
                    E5_cplx0_dish0_polr1 = Gains * E4_cplx0_dish0_polr1
                    E5_cplx1_dish0_polr1 = Gains * E4_cplx1_dish0_polr1
                    E5_cplx0_dish1_polr1 = Gains * E4_cplx0_dish1_polr1
                    E5_cplx1_dish1_polr1 = Gains * E4_cplx1_dish1_polr1
                    E5_cplx0_dish32_polr1 = Gains * E4_cplx0_dish32_polr1
                    E5_cplx1_dish32_polr1 = Gains * E4_cplx1_dish32_polr1
                    E5_cplx0_dish33_polr1 = Gains * E4_cplx0_dish33_polr1
                    E5_cplx1_dish33_polr1 = Gains * E4_cplx1_dish33_polr1
                    E5_cplx0_dish0_polr0 = clamp(E5_cplx0_dish0_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish0_polr0 = clamp(E5_cplx1_dish0_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish1_polr0 = clamp(E5_cplx0_dish1_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish1_polr0 = clamp(E5_cplx1_dish1_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish32_polr0 = clamp(E5_cplx0_dish32_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish32_polr0 = clamp(E5_cplx1_dish32_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish33_polr0 = clamp(E5_cplx0_dish33_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish33_polr0 = clamp(E5_cplx1_dish33_polr0, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish0_polr1 = clamp(E5_cplx0_dish0_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish0_polr1 = clamp(E5_cplx1_dish0_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish1_polr1 = clamp(E5_cplx0_dish1_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish1_polr1 = clamp(E5_cplx1_dish1_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish32_polr1 = clamp(E5_cplx0_dish32_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish32_polr1 = clamp(E5_cplx1_dish32_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx0_dish33_polr1 = clamp(E5_cplx0_dish33_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    E5_cplx1_dish33_polr1 = clamp(E5_cplx1_dish33_polr1, Float16x2(-7, -7), Float16x2(7, 7))
                    F̄_out_dish0_polr0 = Int4x8((
                        E5_cplx0_dish0_polr0, E5_cplx1_dish0_polr0, E5_cplx0_dish1_polr0, E5_cplx1_dish1_polr0
                    ))
                    F̄_out_dish32_polr0 = Int4x8((
                        E5_cplx0_dish32_polr0, E5_cplx1_dish32_polr0, E5_cplx0_dish33_polr0, E5_cplx1_dish33_polr0
                    ))
                    F̄_out_dish0_polr1 = Int4x8((
                        E5_cplx0_dish0_polr1, E5_cplx1_dish0_polr1, E5_cplx0_dish1_polr1, E5_cplx1_dish1_polr1
                    ))
                    F̄_out_dish32_polr1 = Int4x8((
                        E5_cplx0_dish32_polr1, E5_cplx1_dish32_polr1, E5_cplx0_dish33_polr1, E5_cplx1_dish33_polr1
                    ))
                    if true
                        F̄_shared[(((((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 8) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 2) * 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) * 64) ÷ 2) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 32 + 0 + (((((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 16) + 0) + 0x01] =
                            F̄_out_dish0_polr0
                    end
                    if true
                        F̄_shared[(((((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 8) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 2) * 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) * 64) ÷ 2) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 2) % 2) * 32 + 0 + (((((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 4) % 16) + 0) + 0x01] =
                            F̄_out_dish32_polr0
                    end
                    if true
                        F̄_shared[(((((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 8) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 2) * 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) * 64) ÷ 2) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 2) % 2) * 32 + 16 + (((((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2) ÷ 4) % 16) + 0) + 0x01] =
                            F̄_out_dish0_polr1
                    end
                    if true
                        F̄_shared[(((((((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) * 8) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 2) * 2) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 4) + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 32) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) * 64) ÷ 2) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 2) % 2) * 32 + 16 + (((((IndexSpaces.assume_inrange(t_inner, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 2 + 32) ÷ 4) % 16) + 0) + 0x01] =
                            F̄_out_dish32_polr1
                    end
                    F_ringbuf_m0_dish0_polr0 = F_ringbuf_dish0_mtaps0_polr0
                    F_ringbuf_m1_dish0_polr0 = F_ringbuf_dish0_mtaps1_polr0
                    F_ringbuf_m2_dish0_polr0 = F_ringbuf_dish0_mtaps2_polr0
                    F_ringbuf_m0_dish32_polr0 = F_ringbuf_dish32_mtaps0_polr0
                    F_ringbuf_m1_dish32_polr0 = F_ringbuf_dish32_mtaps1_polr0
                    F_ringbuf_m2_dish32_polr0 = F_ringbuf_dish32_mtaps2_polr0
                    F_ringbuf_m0_dish0_polr1 = F_ringbuf_dish0_mtaps0_polr1
                    F_ringbuf_m1_dish0_polr1 = F_ringbuf_dish0_mtaps1_polr1
                    F_ringbuf_m2_dish0_polr1 = F_ringbuf_dish0_mtaps2_polr1
                    F_ringbuf_m0_dish32_polr1 = F_ringbuf_dish32_mtaps0_polr1
                    F_ringbuf_m1_dish32_polr1 = F_ringbuf_dish32_mtaps1_polr1
                    F_ringbuf_m2_dish32_polr1 = F_ringbuf_dish32_mtaps2_polr1
                    F_ringbuf_m0_dish0_polr0 = F_ringbuf_m1_dish0_polr0
                    F_ringbuf_m0_dish32_polr0 = F_ringbuf_m1_dish32_polr0
                    F_ringbuf_m0_dish0_polr1 = F_ringbuf_m1_dish0_polr1
                    F_ringbuf_m0_dish32_polr1 = F_ringbuf_m1_dish32_polr1
                    F_ringbuf_m1_dish0_polr0 = F_ringbuf_m2_dish0_polr0
                    F_ringbuf_m1_dish32_polr0 = F_ringbuf_m2_dish32_polr0
                    F_ringbuf_m1_dish0_polr1 = F_ringbuf_m2_dish0_polr1
                    F_ringbuf_m1_dish32_polr1 = F_ringbuf_m2_dish32_polr1
                    F_ringbuf_m2_dish0_polr0 = F_in_dish0_polr0
                    F_ringbuf_m2_dish32_polr0 = F_in_dish32_polr0
                    F_ringbuf_m2_dish0_polr1 = F_in_dish0_polr1
                    F_ringbuf_m2_dish32_polr1 = F_in_dish32_polr1
                    F_ringbuf_dish0_mtaps0_polr0 = F_ringbuf_m0_dish0_polr0
                    F_ringbuf_dish0_mtaps1_polr0 = F_ringbuf_m1_dish0_polr0
                    F_ringbuf_dish0_mtaps2_polr0 = F_ringbuf_m2_dish0_polr0
                    F_ringbuf_dish32_mtaps0_polr0 = F_ringbuf_m0_dish32_polr0
                    F_ringbuf_dish32_mtaps1_polr0 = F_ringbuf_m1_dish32_polr0
                    F_ringbuf_dish32_mtaps2_polr0 = F_ringbuf_m2_dish32_polr0
                    F_ringbuf_dish0_mtaps0_polr1 = F_ringbuf_m0_dish0_polr1
                    F_ringbuf_dish0_mtaps1_polr1 = F_ringbuf_m1_dish0_polr1
                    F_ringbuf_dish0_mtaps2_polr1 = F_ringbuf_m2_dish0_polr1
                    F_ringbuf_dish32_mtaps0_polr1 = F_ringbuf_m0_dish32_polr1
                    F_ringbuf_dish32_mtaps1_polr1 = F_ringbuf_m1_dish32_polr1
                    F_ringbuf_dish32_mtaps2_polr1 = F_ringbuf_m2_dish32_polr1
                end
            end
            IndexSpaces.cuda_sync_threads()
            Ē_dish0 = F̄_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) * 64) ÷ 2) % 32) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 16) + 0x01]
            Ē_dish2 = F̄_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) * 64) ÷ 2) % 32) * 65 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + 2) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 64) % 4) * 2081 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + 2) ÷ 4) % 16) + 0x01]
            Ē_dish4 = F̄_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) * 64) ÷ 2) % 32) * 65 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + 4) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 64) % 4) * 2081 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + 4) ÷ 4) % 16) + 0x01]
            Ē_dish6 = F̄_shared[(((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 4) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) * 64) ÷ 2) % 32) * 65 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + 6) ÷ 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256) ÷ 64) % 4) * 2081 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 2) * 8) + 6) ÷ 4) % 16) + 0x01]
            (Ē1_dish0, Ē1_dish2) = (IndexSpaces.get_lo16(Ē_dish0, Ē_dish2), IndexSpaces.get_hi16(Ē_dish0, Ē_dish2))
            (Ē1_dish4, Ē1_dish6) = (IndexSpaces.get_lo16(Ē_dish4, Ē_dish6), IndexSpaces.get_hi16(Ē_dish4, Ē_dish6))
            Ē1lo_dish0 = Ē1_dish0
            Ē1hi_dish0 = Ē1_dish2
            Ē1lo_dish4 = Ē1_dish4
            Ē1hi_dish4 = Ē1_dish6
            Ē1_dish0_freq0 = Ē1lo_dish0
            Ē1_dish0_freq1 = Ē1hi_dish0
            Ē1_dish4_freq0 = Ē1lo_dish4
            Ē1_dish4_freq1 = Ē1hi_dish4
            is_lo_thread = IndexSpaces.cuda_threadidx() & 0x00000008 == 0x00
            (Ē2_dish0_freq0, Ē2_dish0_freq1) = let
                src = if is_lo_thread
                    Ē1_dish0_freq1
                else
                    Ē1_dish0_freq0
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (Ē1_dish0_freq0, dst)
                else
                    (dst, Ē1_dish0_freq1)
                end
            end
            (Ē2_dish4_freq0, Ē2_dish4_freq1) = let
                src = if is_lo_thread
                    Ē1_dish4_freq1
                else
                    Ē1_dish4_freq0
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (Ē1_dish4_freq0, dst)
                else
                    (dst, Ē1_dish4_freq1)
                end
            end
            Ē2lo_dish0 = Ē2_dish0_freq0
            Ē2hi_dish0 = Ē2_dish0_freq1
            Ē2lo_dish4 = Ē2_dish4_freq0
            Ē2hi_dish4 = Ē2_dish4_freq1
            Ē3_dish0 = Ē2lo_dish0
            Ē3_dish8 = Ē2hi_dish0
            Ē3_dish4 = Ē2lo_dish4
            Ē3_dish12 = Ē2hi_dish4
            if ((0 + ((IndexSpaces.cuda_warpidx() ÷ 16) % (2i32)) * 64) + ((IndexSpaces.cuda_warpidx() ÷ 32) % (2i32)) * 128) +
               ((t_outer ÷ 256) % (2i32)) * 256 ≥ 192
                IndexSpaces.unsafe_store4_global!(
                    Ē_memory,
                    let
                        offset = 262144 * T̄min - 786432
                        length = 536870912
                        mod(
                            (
                                (
                                    (
                                        (
                                            (
                                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 16) % 4) * 64 +
                                                ((IndexSpaces.assume_inrange(t_outer, 0, 256, 131072) ÷ 256) % 512) * 256
                                            ) ÷ 64
                                        ) % 2048
                                    ) * 262144 +
                                    (
                                        (
                                            (
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 8) % 4 +
                                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) * 4
                                            ) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) * 64
                                        ) % 8192
                                    ) * 32 +
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 4) % 2) % 2) * 16 +
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 4) * 16) ÷ 4) % 16
                                ) + 0
                            ) + offset,
                            length,
                        )
                    end + 0x01,
                    (Ē3_dish0, Ē3_dish4, Ē3_dish8, Ē3_dish12),
                )
            end
        end
        info = 0
        if true
            info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) % 16) * 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 128) % 128) % 128) * 512 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32) + 0) + 0x01] =
                info
        end
    end
)
