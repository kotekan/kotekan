# Julia source code for the CUDA upchannelizer
# This file has been generated automatically by `upchan.jl`.
# Do not modify this file, your changes will be lost.

@fastmath @inbounds(
    begin #= /localhome/eschnett/src/kotekan/julia/kernels/upchan.jl:1484 =#
        info = 1
        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 256) % 256) * 512 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) % 16) * 32) + 0) + 0x01] =
            info
        if !(
            0i32 ≤ Tmin < 65536 && (
                Tmin ≤ Tmax < 131072 && (
                    (Tmax - Tmin) % 256 == 0i32 &&
                    (0i32 ≤ T̄min < 1024 && (T̄min ≤ T̄max < 2048 && ((T̄max - T̄min) + 3) % 4 == 0i32))
                )
            )
        )
            info = 2
            info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 256) % 256) * 512 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) % 16) * 32) + 0) + 0x01] =
                info
            IndexSpaces.cuda_trap()
        end
        if !(0i32 ≤ Fmin ≤ Fmax ≤ F)
            info = 3
            info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 256) % 256) * 512 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) % 16) * 32) + 0) + 0x01] =
                info
            IndexSpaces.cuda_trap()
        end
        F_ringbuf_dish0_mtap0 = zero(Int4x8)
        F_ringbuf_dish32_mtap0 = zero(Int4x8)
        F_ringbuf_dish64_mtap0 = zero(Int4x8)
        F_ringbuf_dish96_mtap0 = zero(Int4x8)
        F_ringbuf_dish0_mtap1 = zero(Int4x8)
        F_ringbuf_dish32_mtap1 = zero(Int4x8)
        F_ringbuf_dish64_mtap1 = zero(Int4x8)
        F_ringbuf_dish96_mtap1 = zero(Int4x8)
        F_ringbuf_dish0_mtap2 = zero(Int4x8)
        F_ringbuf_dish32_mtap2 = zero(Int4x8)
        F_ringbuf_dish64_mtap2 = zero(Int4x8)
        F_ringbuf_dish96_mtap2 = zero(Int4x8)
        Gains = G_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 2) % 2) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 2) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64) ÷ 2) % 32 + 0x01]
        (Wpfb0, Wpfb1) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            time0 = 0 + thread2time(thread)
            time1 = time0 + 32
            s0 = time0 + 0
            s1 = time1 + 0
            W0 = Wkernel(s0, 4, 64) / 64.0f0
            W1 = Wkernel(s1, 4, 64) / 64.0f0
            (W0, W1)
        end
        Wpfb_m0_t0 = Float16x2(Wpfb0, Wpfb1)
        (Wpfb0, Wpfb1) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            time0 = 0 + thread2time(thread)
            time1 = time0 + 32
            s0 = time0 + 64
            s1 = time1 + 64
            W0 = Wkernel(s0, 4, 64) / 64.0f0
            W1 = Wkernel(s1, 4, 64) / 64.0f0
            (W0, W1)
        end
        Wpfb_m1_t0 = Float16x2(Wpfb0, Wpfb1)
        (Wpfb0, Wpfb1) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            time0 = 0 + thread2time(thread)
            time1 = time0 + 32
            s0 = time0 + 128
            s1 = time1 + 128
            W0 = Wkernel(s0, 4, 64) / 64.0f0
            W1 = Wkernel(s1, 4, 64) / 64.0f0
            (W0, W1)
        end
        Wpfb_m2_t0 = Float16x2(Wpfb0, Wpfb1)
        (Wpfb0, Wpfb1) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            time0 = 0 + thread2time(thread)
            time1 = time0 + 32
            s0 = time0 + 192
            s1 = time1 + 192
            W0 = Wkernel(s0, 4, 64) / 64.0f0
            W1 = Wkernel(s1, 4, 64) / 64.0f0
            (W0, W1)
        end
        Wpfb_m3_t0 = Float16x2(Wpfb0, Wpfb1)
        Wpfb_m0 = Wpfb_m0_t0
        Wpfb_m1 = Wpfb_m1_t0
        Wpfb_m2 = Wpfb_m2_t0
        Wpfb_m3 = Wpfb_m3_t0
        Wpfb_mtap0 = Wpfb_m0
        Wpfb_mtap1 = Wpfb_m1
        Wpfb_mtap2 = Wpfb_m2
        Wpfb_mtap3 = Wpfb_m3
        (X0, X1) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            time0 = thread2time(thread)
            time1 = time0 + 32
            X0 = cispi(((time0 * 63) % 128) / 64.0f0)
            X1 = cispi(((time1 * 63) % 128) / 64.0f0)
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
            @assert 0i32 ≤ timehi0 < 8                    #= /localhome/eschnett/src/kotekan/julia/kernels/upchan.jl:714 =#
            @assert 0i32 ≤ timehi1 < 8                    #= /localhome/eschnett/src/kotekan/julia/kernels/upchan.jl:715 =#
            @assert 0i32 ≤ freqlo < 8                    #= /localhome/eschnett/src/kotekan/julia/kernels/upchan.jl:716 =#
            delta0 = dish == dish_in0
            delta1 = dish == dish_in1
            (Γ¹0, Γ¹1) = (
                delta0 * conj(cispi((((2i32) * timehi0 * freqlo) % 16) / 8.0f0)),
                delta1 * conj(cispi((((2i32) * timehi1 * freqlo) % 16) / 8.0f0)),
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
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            thread0 = (thread ÷ (1i32)) % (2i32)
            thread1 = (thread ÷ (2i32)) % (2i32)
            thread2 = (thread ÷ (4i32)) % (2i32)
            thread3 = (thread ÷ (8i32)) % (2i32)
            thread4 = (thread ÷ (16i32)) % (2i32)
            timelo0 = (4i32) * (0i32) + (2i32) * thread1 + (1i32) * thread0
            timelo1 = (4i32) * (1i32) + (2i32) * thread1 + (1i32) * thread0
            freqlo = (1i32) * thread2 + (2i32) * thread4 + (4i32) * thread3
            @assert 0i32 ≤ timelo0 < 8                    #= /localhome/eschnett/src/kotekan/julia/kernels/upchan.jl:797 =#
            @assert 0i32 ≤ timelo1 < 8                    #= /localhome/eschnett/src/kotekan/julia/kernels/upchan.jl:798 =#
            @assert 0i32 ≤ freqlo < 8                    #= /localhome/eschnett/src/kotekan/julia/kernels/upchan.jl:799 =#
            (Γ²0, Γ²1) = (
                conj(cispi((((2i32) * timelo0 * freqlo) % 128) / 64.0f0)),
                conj(cispi((((2i32) * timelo1 * freqlo) % 128) / 64.0f0)),
            )
            (Γ²0, Γ²1)
        end
        Γ²re = Float16x2(real(Γ²0), real(Γ²1))
        Γ²im = Float16x2(imag(Γ²0), imag(Γ²1))
        Γ²_cplx0 = Γ²re
        Γ²_cplx1 = Γ²im
        (Γ³0, Γ³1) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            thread0 = (thread ÷ (1i32)) % (2i32)
            thread1 = (thread ÷ (2i32)) % (2i32)
            thread2 = (thread ÷ (4i32)) % (2i32)
            thread3 = (thread ÷ (8i32)) % (2i32)
            thread4 = (thread ÷ (16i32)) % (2i32)
            timelo0 = (4i32) * (0i32) + (2i32) * thread1 + (1i32) * thread0
            timelo1 = (4i32) * (1i32) + (2i32) * thread1 + (1i32) * thread0
            dish_in0 = 0i32
            dish_in1 = 0i32
            freqhi = (1i32) * thread2 + (2i32) * thread4 + (4i32) * thread3
            dish = 0i32
            @assert 0i32 ≤ timelo0 < 8                    #= /localhome/eschnett/src/kotekan/julia/kernels/upchan.jl:911 =#
            @assert 0i32 ≤ timelo1 < 8                    #= /localhome/eschnett/src/kotekan/julia/kernels/upchan.jl:912 =#
            @assert 0i32 ≤ freqhi < 8                    #= /localhome/eschnett/src/kotekan/julia/kernels/upchan.jl:913 =#
            delta0 = dish == dish_in0
            delta1 = dish == dish_in1
            (Γ³0, Γ³1) = (
                delta0 * conj(cispi((((2i32) * timelo0 * freqhi) % 16) / 8.0f0)),
                delta1 * conj(cispi((((2i32) * timelo1 * freqhi) % 16) / 8.0f0)),
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
        Γ³_cplx0_cplx_in0_dish0 = Γ³_cplx0_cplx_in0_dish0
        Γ³_cplx0_cplx_in0_dish64 = Γ³_cplx0_cplx_in0_dish0
        Γ³_cplx1_cplx_in0_dish0 = Γ³_cplx1_cplx_in0_dish0
        Γ³_cplx1_cplx_in0_dish64 = Γ³_cplx1_cplx_in0_dish0
        Γ³_cplx0_cplx_in1_dish0 = Γ³_cplx0_cplx_in1_dish0
        Γ³_cplx0_cplx_in1_dish64 = Γ³_cplx0_cplx_in1_dish0
        Γ³_cplx1_cplx_in1_dish0 = Γ³_cplx1_cplx_in1_dish0
        Γ³_cplx1_cplx_in1_dish64 = Γ³_cplx1_cplx_in1_dish0
        Γ³_cplx0_cplx_in0_dish1 = Γ³_cplx0_cplx_in0_dish1
        Γ³_cplx0_cplx_in0_dish65 = Γ³_cplx0_cplx_in0_dish1
        Γ³_cplx1_cplx_in0_dish1 = Γ³_cplx1_cplx_in0_dish1
        Γ³_cplx1_cplx_in0_dish65 = Γ³_cplx1_cplx_in0_dish1
        Γ³_cplx0_cplx_in1_dish1 = Γ³_cplx0_cplx_in1_dish1
        Γ³_cplx0_cplx_in1_dish65 = Γ³_cplx0_cplx_in1_dish1
        Γ³_cplx1_cplx_in1_dish1 = Γ³_cplx1_cplx_in1_dish1
        Γ³_cplx1_cplx_in1_dish65 = Γ³_cplx1_cplx_in1_dish1
        Γ³_cplx0_cplx_in0_dish32 = Γ³_cplx0_cplx_in0_dish32
        Γ³_cplx0_cplx_in0_dish96 = Γ³_cplx0_cplx_in0_dish32
        Γ³_cplx1_cplx_in0_dish32 = Γ³_cplx1_cplx_in0_dish32
        Γ³_cplx1_cplx_in0_dish96 = Γ³_cplx1_cplx_in0_dish32
        Γ³_cplx0_cplx_in1_dish32 = Γ³_cplx0_cplx_in1_dish32
        Γ³_cplx0_cplx_in1_dish96 = Γ³_cplx0_cplx_in1_dish32
        Γ³_cplx1_cplx_in1_dish32 = Γ³_cplx1_cplx_in1_dish32
        Γ³_cplx1_cplx_in1_dish96 = Γ³_cplx1_cplx_in1_dish32
        Γ³_cplx0_cplx_in0_dish33 = Γ³_cplx0_cplx_in0_dish33
        Γ³_cplx0_cplx_in0_dish97 = Γ³_cplx0_cplx_in0_dish33
        Γ³_cplx1_cplx_in0_dish33 = Γ³_cplx1_cplx_in0_dish33
        Γ³_cplx1_cplx_in0_dish97 = Γ³_cplx1_cplx_in0_dish33
        Γ³_cplx0_cplx_in1_dish33 = Γ³_cplx0_cplx_in1_dish33
        Γ³_cplx0_cplx_in1_dish97 = Γ³_cplx0_cplx_in1_dish33
        Γ³_cplx1_cplx_in1_dish33 = Γ³_cplx1_cplx_in1_dish33
        Γ³_cplx1_cplx_in1_dish97 = Γ³_cplx1_cplx_in1_dish33
        for t_outer in 0:256:65535
            Tmin + t_outer ≥ Tmax && break
            (E_dish0_time0, E_dish4_time0, E_dish8_time0, E_dish12_time0) = IndexSpaces.unsafe_load4(
                E_memory,
                let
                    offset = 8192 * Tmin + 128 * Fmin
                    length = 536870912
                    mod(
                        (
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 2) % 2) % 2) * 64 +
                            (
                                (
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 +
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 +
                                    ((0::Int32 ÷ 4) % 4) * 4
                                ) ÷ 4
                            ) % 64 +
                            (
                                ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64) ÷ 64) %
                                64
                            ) * 128 +
                            (
                                (
                                    IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 +
                                    ((0::Int32 ÷ 64) % 4) * 64 +
                                    ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 +
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16 +
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 32
                                ) % 65536
                            ) * 8192
                        ) + offset,
                        length,
                    )
                end + 0x01,
            )
            (E_dish0_time64, E_dish4_time64, E_dish8_time64, E_dish12_time64) = IndexSpaces.unsafe_load4(
                E_memory,
                let
                    offset = 8192 * Tmin + 128 * Fmin
                    length = 536870912
                    mod(
                        (
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 2) % 2) % 2) * 64 +
                            (
                                (
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 +
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 +
                                    ((0::Int32 ÷ 4) % 4) * 4
                                ) ÷ 4
                            ) % 64 +
                            (
                                ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64) ÷ 64) %
                                64
                            ) * 128 +
                            (
                                (
                                    IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 +
                                    ((64::Int32 ÷ 64) % 4) * 64 +
                                    ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 +
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16 +
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 32
                                ) % 65536
                            ) * 8192
                        ) + offset,
                        length,
                    )
                end + 0x01,
            )
            (E_dish0_time128, E_dish4_time128, E_dish8_time128, E_dish12_time128) = IndexSpaces.unsafe_load4(
                E_memory,
                let
                    offset = 8192 * Tmin + 128 * Fmin
                    length = 536870912
                    mod(
                        (
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 2) % 2) % 2) * 64 +
                            (
                                (
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 +
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 +
                                    ((0::Int32 ÷ 4) % 4) * 4
                                ) ÷ 4
                            ) % 64 +
                            (
                                ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64) ÷ 64) %
                                64
                            ) * 128 +
                            (
                                (
                                    IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 +
                                    ((128::Int32 ÷ 64) % 4) * 64 +
                                    ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 +
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16 +
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 32
                                ) % 65536
                            ) * 8192
                        ) + offset,
                        length,
                    )
                end + 0x01,
            )
            (E_dish0_time192, E_dish4_time192, E_dish8_time192, E_dish12_time192) = IndexSpaces.unsafe_load4(
                E_memory,
                let
                    offset = 8192 * Tmin + 128 * Fmin
                    length = 536870912
                    mod(
                        (
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 2) % 2) % 2) * 64 +
                            (
                                (
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 +
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 +
                                    ((0::Int32 ÷ 4) % 4) * 4
                                ) ÷ 4
                            ) % 64 +
                            (
                                ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64) ÷ 64) %
                                64
                            ) * 128 +
                            (
                                (
                                    IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 +
                                    ((192::Int32 ÷ 64) % 4) * 64 +
                                    ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 +
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16 +
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 32
                                ) % 65536
                            ) * 8192
                        ) + offset,
                        length,
                    )
                end + 0x01,
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
            E1_dish0_time32 = E1hi_dish0_time0
            E1_dish4_time0 = E1lo_dish4_time0
            E1_dish4_time32 = E1hi_dish4_time0
            E1_dish0_time64 = E1lo_dish0_time64
            E1_dish0_time96 = E1hi_dish0_time64
            E1_dish4_time64 = E1lo_dish4_time64
            E1_dish4_time96 = E1hi_dish4_time64
            E1_dish0_time128 = E1lo_dish0_time128
            E1_dish0_time160 = E1hi_dish0_time128
            E1_dish4_time128 = E1lo_dish4_time128
            E1_dish4_time160 = E1hi_dish4_time128
            E1_dish0_time192 = E1lo_dish0_time192
            E1_dish0_time224 = E1hi_dish0_time192
            E1_dish4_time192 = E1lo_dish4_time192
            E1_dish4_time224 = E1hi_dish4_time192
            (E2_dish0_time0, E2_dish0_time32) = (
                IndexSpaces.get_lo16(E1_dish0_time0, E1_dish0_time32), IndexSpaces.get_hi16(E1_dish0_time0, E1_dish0_time32)
            )
            (E2_dish4_time0, E2_dish4_time32) = (
                IndexSpaces.get_lo16(E1_dish4_time0, E1_dish4_time32), IndexSpaces.get_hi16(E1_dish4_time0, E1_dish4_time32)
            )
            (E2_dish0_time64, E2_dish0_time96) = (
                IndexSpaces.get_lo16(E1_dish0_time64, E1_dish0_time96), IndexSpaces.get_hi16(E1_dish0_time64, E1_dish0_time96)
            )
            (E2_dish4_time64, E2_dish4_time96) = (
                IndexSpaces.get_lo16(E1_dish4_time64, E1_dish4_time96), IndexSpaces.get_hi16(E1_dish4_time64, E1_dish4_time96)
            )
            (E2_dish0_time128, E2_dish0_time160) = (
                IndexSpaces.get_lo16(E1_dish0_time128, E1_dish0_time160), IndexSpaces.get_hi16(E1_dish0_time128, E1_dish0_time160)
            )
            (E2_dish4_time128, E2_dish4_time160) = (
                IndexSpaces.get_lo16(E1_dish4_time128, E1_dish4_time160), IndexSpaces.get_hi16(E1_dish4_time128, E1_dish4_time160)
            )
            (E2_dish0_time192, E2_dish0_time224) = (
                IndexSpaces.get_lo16(E1_dish0_time192, E1_dish0_time224), IndexSpaces.get_hi16(E1_dish0_time192, E1_dish0_time224)
            )
            (E2_dish4_time192, E2_dish4_time224) = (
                IndexSpaces.get_lo16(E1_dish4_time192, E1_dish4_time224), IndexSpaces.get_hi16(E1_dish4_time192, E1_dish4_time224)
            )
            E2lo_dish0_time0 = E2_dish0_time0
            E2hi_dish0_time0 = E2_dish0_time32
            E2lo_dish4_time0 = E2_dish4_time0
            E2hi_dish4_time0 = E2_dish4_time32
            E2lo_dish0_time64 = E2_dish0_time64
            E2hi_dish0_time64 = E2_dish0_time96
            E2lo_dish4_time64 = E2_dish4_time64
            E2hi_dish4_time64 = E2_dish4_time96
            E2lo_dish0_time128 = E2_dish0_time128
            E2hi_dish0_time128 = E2_dish0_time160
            E2lo_dish4_time128 = E2_dish4_time128
            E2hi_dish4_time128 = E2_dish4_time160
            E2lo_dish0_time192 = E2_dish0_time192
            E2hi_dish0_time192 = E2_dish0_time224
            E2lo_dish4_time192 = E2_dish4_time192
            E2hi_dish4_time192 = E2_dish4_time224
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
            F_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) % 2) * 1040 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((0::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 2) % 2) * 520 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 4) % 2) * 260 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 16) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 8) % 2) * 130 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((0::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0) + 0x01] =
                F_dish0_time0
            F_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) % 2) * 1040 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((2::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 2) % 2) * 520 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 4) % 2) * 260 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 16) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 8) % 2) * 130 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((2::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0) + 0x01] =
                F_dish2_time0
            F_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) % 2) * 1040 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((4::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 2) % 2) * 520 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 4) % 2) * 260 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 16) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 8) % 2) * 130 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((4::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0) + 0x01] =
                F_dish4_time0
            F_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) % 2) * 1040 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((6::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 2) % 2) * 520 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 4) % 2) * 260 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 16) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 8) % 2) * 130 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((6::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0) + 0x01] =
                F_dish6_time0
            F_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) % 2) * 1040 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((0::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 2) % 2) * 520 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 4) % 2) * 260 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 16) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 8) % 2) * 130 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((0::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0) + 0x01] =
                F_dish0_time64
            F_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) % 2) * 1040 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((2::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 2) % 2) * 520 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 4) % 2) * 260 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 16) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 8) % 2) * 130 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((2::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0) + 0x01] =
                F_dish2_time64
            F_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) % 2) * 1040 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((4::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 2) % 2) * 520 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 4) % 2) * 260 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 16) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 8) % 2) * 130 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((4::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0) + 0x01] =
                F_dish4_time64
            F_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) % 2) * 1040 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((6::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 2) % 2) * 520 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 4) % 2) * 260 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 16) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 8) % 2) * 130 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((6::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0) + 0x01] =
                F_dish6_time64
            F_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) % 2) * 1040 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((0::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 2) % 2) * 520 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 4) % 2) * 260 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 16) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 8) % 2) * 130 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((0::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0) + 0x01] =
                F_dish0_time128
            F_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) % 2) * 1040 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((2::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 2) % 2) * 520 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 4) % 2) * 260 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 16) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 8) % 2) * 130 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((2::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0) + 0x01] =
                F_dish2_time128
            F_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) % 2) * 1040 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((4::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 2) % 2) * 520 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 4) % 2) * 260 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 16) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 8) % 2) * 130 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((4::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0) + 0x01] =
                F_dish4_time128
            F_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) % 2) * 1040 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((6::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 2) % 2) * 520 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 4) % 2) * 260 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 16) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 8) % 2) * 130 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((6::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0) + 0x01] =
                F_dish6_time128
            F_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) % 2) * 1040 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((0::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 2) % 2) * 520 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 4) % 2) * 260 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 16) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 8) % 2) * 130 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((0::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0) + 0x01] =
                F_dish0_time192
            F_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) % 2) * 1040 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((2::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 2) % 2) * 520 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 4) % 2) * 260 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 16) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 8) % 2) * 130 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((2::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0) + 0x01] =
                F_dish2_time192
            F_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) % 2) * 1040 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((4::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 2) % 2) * 520 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 4) % 2) * 260 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 16) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 8) % 2) * 130 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((4::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0) + 0x01] =
                F_dish4_time192
            F_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) % 2) * 1040 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((6::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 2) % 2) * 520 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 4) % 2) * 260 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 64) % 4) * 2081 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 16) % 2) * 65 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16) ÷ 8) % 2) * 130 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((6::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0) + 0x01] =
                F_dish6_time192
            IndexSpaces.cuda_sync_threads()
            for t_inner in 0:64:255
                let polr = 0
                    F_ringbuf_polr_dish0_mtap0 = F_ringbuf_dish0_mtap0
                    F_ringbuf_polr_dish32_mtap0 = F_ringbuf_dish32_mtap0
                    F_ringbuf_polr_dish64_mtap0 = F_ringbuf_dish64_mtap0
                    F_ringbuf_polr_dish96_mtap0 = F_ringbuf_dish96_mtap0
                    F_ringbuf_polr_dish0_mtap1 = F_ringbuf_dish0_mtap1
                    F_ringbuf_polr_dish32_mtap1 = F_ringbuf_dish32_mtap1
                    F_ringbuf_polr_dish64_mtap1 = F_ringbuf_dish64_mtap1
                    F_ringbuf_polr_dish96_mtap1 = F_ringbuf_dish96_mtap1
                    F_ringbuf_polr_dish0_mtap2 = F_ringbuf_dish0_mtap2
                    F_ringbuf_polr_dish32_mtap2 = F_ringbuf_dish32_mtap2
                    F_ringbuf_polr_dish64_mtap2 = F_ringbuf_dish64_mtap2
                    F_ringbuf_polr_dish96_mtap2 = F_ringbuf_dish96_mtap2
                    let dish = 0
                        F_ringbuf_polr_dish_mtap0 = F_ringbuf_polr_dish0_mtap0
                        F_ringbuf_polr_dish_mtap1 = F_ringbuf_polr_dish0_mtap1
                        F_ringbuf_polr_dish_mtap2 = F_ringbuf_polr_dish0_mtap2
                        F_in = F_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 4 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 8) % 2) * 1040 + ((((dish::Int32 ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2) ÷ 4) % 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 4 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 8) ÷ 2) % 2) * 520 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 4 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 8) ÷ 4) % 2) * 260 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 4 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 8) ÷ 64) % 4) * 2081 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 4 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 8) ÷ 16) % 2) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 4 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 8) ÷ 8) % 2) * 130 + (((((dish::Int32 ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2) ÷ 2) % 2) * 32) + 0x01]
                        (E_cplx0_dish0, E_cplx1_dish0, E_cplx0_dish1, E_cplx1_dish1) = convert(NTuple{4,Float16x2}, F_in)
                        E2_cplx0_dish0 = zero(E_cplx0_dish0)
                        E2_cplx1_dish0 = zero(E_cplx1_dish0)
                        E2_cplx0_dish1 = zero(E_cplx0_dish1)
                        E2_cplx1_dish1 = zero(E_cplx1_dish1)
                        let mtap = 0
                            W_mtap = Wpfb_mtap0
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap = F_ringbuf_polr_dish_mtap0
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0, E_ringbuf_polr_dish_mtap_cplx1_dish0, E_ringbuf_polr_dish_mtap_cplx0_dish1, E_ringbuf_polr_dish_mtap_cplx1_dish1) = convert(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap
                                )
                                E2_cplx0_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish0, E2_cplx0_dish0
                                )
                                E2_cplx1_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish0, E2_cplx1_dish0
                                )
                                E2_cplx0_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish1, E2_cplx0_dish1
                                )
                                E2_cplx1_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish1, E2_cplx1_dish1
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish0, E2_cplx0_dish0)
                                E2_cplx1_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish0, E2_cplx1_dish0)
                                E2_cplx0_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish1, E2_cplx0_dish1)
                                E2_cplx1_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish1, E2_cplx1_dish1)
                            end
                        end
                        let mtap = 1
                            W_mtap = Wpfb_mtap1
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap = F_ringbuf_polr_dish_mtap1
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0, E_ringbuf_polr_dish_mtap_cplx1_dish0, E_ringbuf_polr_dish_mtap_cplx0_dish1, E_ringbuf_polr_dish_mtap_cplx1_dish1) = convert(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap
                                )
                                E2_cplx0_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish0, E2_cplx0_dish0
                                )
                                E2_cplx1_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish0, E2_cplx1_dish0
                                )
                                E2_cplx0_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish1, E2_cplx0_dish1
                                )
                                E2_cplx1_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish1, E2_cplx1_dish1
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish0, E2_cplx0_dish0)
                                E2_cplx1_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish0, E2_cplx1_dish0)
                                E2_cplx0_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish1, E2_cplx0_dish1)
                                E2_cplx1_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish1, E2_cplx1_dish1)
                            end
                        end
                        let mtap = 2
                            W_mtap = Wpfb_mtap2
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap = F_ringbuf_polr_dish_mtap2
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0, E_ringbuf_polr_dish_mtap_cplx1_dish0, E_ringbuf_polr_dish_mtap_cplx0_dish1, E_ringbuf_polr_dish_mtap_cplx1_dish1) = convert(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap
                                )
                                E2_cplx0_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish0, E2_cplx0_dish0
                                )
                                E2_cplx1_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish0, E2_cplx1_dish0
                                )
                                E2_cplx0_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish1, E2_cplx0_dish1
                                )
                                E2_cplx1_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish1, E2_cplx1_dish1
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish0, E2_cplx0_dish0)
                                E2_cplx1_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish0, E2_cplx1_dish0)
                                E2_cplx0_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish1, E2_cplx0_dish1)
                                E2_cplx1_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish1, E2_cplx1_dish1)
                            end
                        end
                        let mtap = 3
                            W_mtap = Wpfb_mtap3
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap = F_ringbuf_polr_dish_mtap3
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0, E_ringbuf_polr_dish_mtap_cplx1_dish0, E_ringbuf_polr_dish_mtap_cplx0_dish1, E_ringbuf_polr_dish_mtap_cplx1_dish1) = convert(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap
                                )
                                E2_cplx0_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish0, E2_cplx0_dish0
                                )
                                E2_cplx1_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish0, E2_cplx1_dish0
                                )
                                E2_cplx0_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish1, E2_cplx0_dish1
                                )
                                E2_cplx1_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish1, E2_cplx1_dish1
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish0, E2_cplx0_dish0)
                                E2_cplx1_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish0, E2_cplx1_dish0)
                                E2_cplx0_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish1, E2_cplx0_dish1)
                                E2_cplx1_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish1, E2_cplx1_dish1)
                            end
                        end
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
                        F̄_shared[((((((dish::Int32 ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 2) % 2) * 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 2) * 4) ÷ 2) % 32) * 65 + (((((IndexSpaces.assume_inrange(t_inner::Int32, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256) ÷ 64) % 4) * 2081 + (((((dish::Int32 ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2) ÷ 2) % 2) * 32) + 0) + 0x01] =
                            F̄_out
                        F_ringbuf_polr_dish_m0 = F_ringbuf_polr_dish_mtap0
                        F_ringbuf_polr_dish_m1 = F_ringbuf_polr_dish_mtap1
                        F_ringbuf_polr_dish_m2 = F_ringbuf_polr_dish_mtap2
                        F_ringbuf_polr_dish_m0 = F_ringbuf_polr_dish_m1
                        F_ringbuf_polr_dish_m1 = F_ringbuf_polr_dish_m2
                        F_ringbuf_polr_dish_m2 = F_in
                        F_ringbuf_polr_dish_mtap0 = F_ringbuf_polr_dish_m0
                        F_ringbuf_polr_dish_mtap1 = F_ringbuf_polr_dish_m1
                        F_ringbuf_polr_dish_mtap2 = F_ringbuf_polr_dish_m2
                        F_ringbuf_polr_dish0_mtap0 = F_ringbuf_polr_dish_mtap0
                        F_ringbuf_polr_dish0_mtap1 = F_ringbuf_polr_dish_mtap1
                        F_ringbuf_polr_dish0_mtap2 = F_ringbuf_polr_dish_mtap2
                    end
                    let dish = 32
                        F_ringbuf_polr_dish_mtap0 = F_ringbuf_polr_dish32_mtap0
                        F_ringbuf_polr_dish_mtap1 = F_ringbuf_polr_dish32_mtap1
                        F_ringbuf_polr_dish_mtap2 = F_ringbuf_polr_dish32_mtap2
                        F_in = F_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 4 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 8) % 2) * 1040 + ((((dish::Int32 ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2) ÷ 4) % 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 4 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 8) ÷ 2) % 2) * 520 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 4 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 8) ÷ 4) % 2) * 260 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 4 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 8) ÷ 64) % 4) * 2081 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 4 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 8) ÷ 16) % 2) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 4 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 8) ÷ 8) % 2) * 130 + (((((dish::Int32 ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2) ÷ 2) % 2) * 32) + 0x01]
                        (E_cplx0_dish0, E_cplx1_dish0, E_cplx0_dish1, E_cplx1_dish1) = convert(NTuple{4,Float16x2}, F_in)
                        E2_cplx0_dish0 = zero(E_cplx0_dish0)
                        E2_cplx1_dish0 = zero(E_cplx1_dish0)
                        E2_cplx0_dish1 = zero(E_cplx0_dish1)
                        E2_cplx1_dish1 = zero(E_cplx1_dish1)
                        let mtap = 0
                            W_mtap = Wpfb_mtap0
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap = F_ringbuf_polr_dish_mtap0
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0, E_ringbuf_polr_dish_mtap_cplx1_dish0, E_ringbuf_polr_dish_mtap_cplx0_dish1, E_ringbuf_polr_dish_mtap_cplx1_dish1) = convert(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap
                                )
                                E2_cplx0_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish0, E2_cplx0_dish0
                                )
                                E2_cplx1_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish0, E2_cplx1_dish0
                                )
                                E2_cplx0_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish1, E2_cplx0_dish1
                                )
                                E2_cplx1_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish1, E2_cplx1_dish1
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish0, E2_cplx0_dish0)
                                E2_cplx1_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish0, E2_cplx1_dish0)
                                E2_cplx0_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish1, E2_cplx0_dish1)
                                E2_cplx1_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish1, E2_cplx1_dish1)
                            end
                        end
                        let mtap = 1
                            W_mtap = Wpfb_mtap1
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap = F_ringbuf_polr_dish_mtap1
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0, E_ringbuf_polr_dish_mtap_cplx1_dish0, E_ringbuf_polr_dish_mtap_cplx0_dish1, E_ringbuf_polr_dish_mtap_cplx1_dish1) = convert(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap
                                )
                                E2_cplx0_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish0, E2_cplx0_dish0
                                )
                                E2_cplx1_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish0, E2_cplx1_dish0
                                )
                                E2_cplx0_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish1, E2_cplx0_dish1
                                )
                                E2_cplx1_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish1, E2_cplx1_dish1
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish0, E2_cplx0_dish0)
                                E2_cplx1_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish0, E2_cplx1_dish0)
                                E2_cplx0_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish1, E2_cplx0_dish1)
                                E2_cplx1_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish1, E2_cplx1_dish1)
                            end
                        end
                        let mtap = 2
                            W_mtap = Wpfb_mtap2
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap = F_ringbuf_polr_dish_mtap2
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0, E_ringbuf_polr_dish_mtap_cplx1_dish0, E_ringbuf_polr_dish_mtap_cplx0_dish1, E_ringbuf_polr_dish_mtap_cplx1_dish1) = convert(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap
                                )
                                E2_cplx0_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish0, E2_cplx0_dish0
                                )
                                E2_cplx1_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish0, E2_cplx1_dish0
                                )
                                E2_cplx0_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish1, E2_cplx0_dish1
                                )
                                E2_cplx1_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish1, E2_cplx1_dish1
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish0, E2_cplx0_dish0)
                                E2_cplx1_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish0, E2_cplx1_dish0)
                                E2_cplx0_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish1, E2_cplx0_dish1)
                                E2_cplx1_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish1, E2_cplx1_dish1)
                            end
                        end
                        let mtap = 3
                            W_mtap = Wpfb_mtap3
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap = F_ringbuf_polr_dish_mtap3
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0, E_ringbuf_polr_dish_mtap_cplx1_dish0, E_ringbuf_polr_dish_mtap_cplx0_dish1, E_ringbuf_polr_dish_mtap_cplx1_dish1) = convert(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap
                                )
                                E2_cplx0_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish0, E2_cplx0_dish0
                                )
                                E2_cplx1_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish0, E2_cplx1_dish0
                                )
                                E2_cplx0_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish1, E2_cplx0_dish1
                                )
                                E2_cplx1_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish1, E2_cplx1_dish1
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish0, E2_cplx0_dish0)
                                E2_cplx1_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish0, E2_cplx1_dish0)
                                E2_cplx0_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish1, E2_cplx0_dish1)
                                E2_cplx1_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish1, E2_cplx1_dish1)
                            end
                        end
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
                        F̄_shared[((((((dish::Int32 ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 2) % 2) * 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 2) * 4) ÷ 2) % 32) * 65 + (((((IndexSpaces.assume_inrange(t_inner::Int32, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256) ÷ 64) % 4) * 2081 + (((((dish::Int32 ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2) ÷ 2) % 2) * 32) + 0) + 0x01] =
                            F̄_out
                        F_ringbuf_polr_dish_m0 = F_ringbuf_polr_dish_mtap0
                        F_ringbuf_polr_dish_m1 = F_ringbuf_polr_dish_mtap1
                        F_ringbuf_polr_dish_m2 = F_ringbuf_polr_dish_mtap2
                        F_ringbuf_polr_dish_m0 = F_ringbuf_polr_dish_m1
                        F_ringbuf_polr_dish_m1 = F_ringbuf_polr_dish_m2
                        F_ringbuf_polr_dish_m2 = F_in
                        F_ringbuf_polr_dish_mtap0 = F_ringbuf_polr_dish_m0
                        F_ringbuf_polr_dish_mtap1 = F_ringbuf_polr_dish_m1
                        F_ringbuf_polr_dish_mtap2 = F_ringbuf_polr_dish_m2
                        F_ringbuf_polr_dish32_mtap0 = F_ringbuf_polr_dish_mtap0
                        F_ringbuf_polr_dish32_mtap1 = F_ringbuf_polr_dish_mtap1
                        F_ringbuf_polr_dish32_mtap2 = F_ringbuf_polr_dish_mtap2
                    end
                    let dish = 64
                        F_ringbuf_polr_dish_mtap0 = F_ringbuf_polr_dish64_mtap0
                        F_ringbuf_polr_dish_mtap1 = F_ringbuf_polr_dish64_mtap1
                        F_ringbuf_polr_dish_mtap2 = F_ringbuf_polr_dish64_mtap2
                        F_in = F_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 4 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 8) % 2) * 1040 + ((((dish::Int32 ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2) ÷ 4) % 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 4 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 8) ÷ 2) % 2) * 520 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 4 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 8) ÷ 4) % 2) * 260 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 4 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 8) ÷ 64) % 4) * 2081 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 4 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 8) ÷ 16) % 2) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 4 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 8) ÷ 8) % 2) * 130 + (((((dish::Int32 ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2) ÷ 2) % 2) * 32) + 0x01]
                        (E_cplx0_dish0, E_cplx1_dish0, E_cplx0_dish1, E_cplx1_dish1) = convert(NTuple{4,Float16x2}, F_in)
                        E2_cplx0_dish0 = zero(E_cplx0_dish0)
                        E2_cplx1_dish0 = zero(E_cplx1_dish0)
                        E2_cplx0_dish1 = zero(E_cplx0_dish1)
                        E2_cplx1_dish1 = zero(E_cplx1_dish1)
                        let mtap = 0
                            W_mtap = Wpfb_mtap0
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap = F_ringbuf_polr_dish_mtap0
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0, E_ringbuf_polr_dish_mtap_cplx1_dish0, E_ringbuf_polr_dish_mtap_cplx0_dish1, E_ringbuf_polr_dish_mtap_cplx1_dish1) = convert(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap
                                )
                                E2_cplx0_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish0, E2_cplx0_dish0
                                )
                                E2_cplx1_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish0, E2_cplx1_dish0
                                )
                                E2_cplx0_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish1, E2_cplx0_dish1
                                )
                                E2_cplx1_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish1, E2_cplx1_dish1
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish0, E2_cplx0_dish0)
                                E2_cplx1_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish0, E2_cplx1_dish0)
                                E2_cplx0_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish1, E2_cplx0_dish1)
                                E2_cplx1_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish1, E2_cplx1_dish1)
                            end
                        end
                        let mtap = 1
                            W_mtap = Wpfb_mtap1
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap = F_ringbuf_polr_dish_mtap1
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0, E_ringbuf_polr_dish_mtap_cplx1_dish0, E_ringbuf_polr_dish_mtap_cplx0_dish1, E_ringbuf_polr_dish_mtap_cplx1_dish1) = convert(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap
                                )
                                E2_cplx0_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish0, E2_cplx0_dish0
                                )
                                E2_cplx1_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish0, E2_cplx1_dish0
                                )
                                E2_cplx0_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish1, E2_cplx0_dish1
                                )
                                E2_cplx1_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish1, E2_cplx1_dish1
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish0, E2_cplx0_dish0)
                                E2_cplx1_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish0, E2_cplx1_dish0)
                                E2_cplx0_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish1, E2_cplx0_dish1)
                                E2_cplx1_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish1, E2_cplx1_dish1)
                            end
                        end
                        let mtap = 2
                            W_mtap = Wpfb_mtap2
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap = F_ringbuf_polr_dish_mtap2
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0, E_ringbuf_polr_dish_mtap_cplx1_dish0, E_ringbuf_polr_dish_mtap_cplx0_dish1, E_ringbuf_polr_dish_mtap_cplx1_dish1) = convert(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap
                                )
                                E2_cplx0_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish0, E2_cplx0_dish0
                                )
                                E2_cplx1_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish0, E2_cplx1_dish0
                                )
                                E2_cplx0_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish1, E2_cplx0_dish1
                                )
                                E2_cplx1_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish1, E2_cplx1_dish1
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish0, E2_cplx0_dish0)
                                E2_cplx1_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish0, E2_cplx1_dish0)
                                E2_cplx0_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish1, E2_cplx0_dish1)
                                E2_cplx1_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish1, E2_cplx1_dish1)
                            end
                        end
                        let mtap = 3
                            W_mtap = Wpfb_mtap3
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap = F_ringbuf_polr_dish_mtap3
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0, E_ringbuf_polr_dish_mtap_cplx1_dish0, E_ringbuf_polr_dish_mtap_cplx0_dish1, E_ringbuf_polr_dish_mtap_cplx1_dish1) = convert(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap
                                )
                                E2_cplx0_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish0, E2_cplx0_dish0
                                )
                                E2_cplx1_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish0, E2_cplx1_dish0
                                )
                                E2_cplx0_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish1, E2_cplx0_dish1
                                )
                                E2_cplx1_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish1, E2_cplx1_dish1
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish0, E2_cplx0_dish0)
                                E2_cplx1_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish0, E2_cplx1_dish0)
                                E2_cplx0_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish1, E2_cplx0_dish1)
                                E2_cplx1_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish1, E2_cplx1_dish1)
                            end
                        end
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
                        F̄_shared[((((((dish::Int32 ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 2) % 2) * 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 2) * 4) ÷ 2) % 32) * 65 + (((((IndexSpaces.assume_inrange(t_inner::Int32, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256) ÷ 64) % 4) * 2081 + (((((dish::Int32 ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2) ÷ 2) % 2) * 32) + 0) + 0x01] =
                            F̄_out
                        F_ringbuf_polr_dish_m0 = F_ringbuf_polr_dish_mtap0
                        F_ringbuf_polr_dish_m1 = F_ringbuf_polr_dish_mtap1
                        F_ringbuf_polr_dish_m2 = F_ringbuf_polr_dish_mtap2
                        F_ringbuf_polr_dish_m0 = F_ringbuf_polr_dish_m1
                        F_ringbuf_polr_dish_m1 = F_ringbuf_polr_dish_m2
                        F_ringbuf_polr_dish_m2 = F_in
                        F_ringbuf_polr_dish_mtap0 = F_ringbuf_polr_dish_m0
                        F_ringbuf_polr_dish_mtap1 = F_ringbuf_polr_dish_m1
                        F_ringbuf_polr_dish_mtap2 = F_ringbuf_polr_dish_m2
                        F_ringbuf_polr_dish64_mtap0 = F_ringbuf_polr_dish_mtap0
                        F_ringbuf_polr_dish64_mtap1 = F_ringbuf_polr_dish_mtap1
                        F_ringbuf_polr_dish64_mtap2 = F_ringbuf_polr_dish_mtap2
                    end
                    let dish = 96
                        F_ringbuf_polr_dish_mtap0 = F_ringbuf_polr_dish96_mtap0
                        F_ringbuf_polr_dish_mtap1 = F_ringbuf_polr_dish96_mtap1
                        F_ringbuf_polr_dish_mtap2 = F_ringbuf_polr_dish96_mtap2
                        F_in = F_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 4 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 8) % 2) * 1040 + ((((dish::Int32 ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2) ÷ 4) % 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 4 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 8) ÷ 2) % 2) * 520 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 4 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 8) ÷ 4) % 2) * 260 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 4 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 8) ÷ 64) % 4) * 2081 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 4 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 8) ÷ 16) % 2) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 4 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 8) ÷ 8) % 2) * 130 + (((((dish::Int32 ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2) ÷ 2) % 2) * 32) + 0x01]
                        (E_cplx0_dish0, E_cplx1_dish0, E_cplx0_dish1, E_cplx1_dish1) = convert(NTuple{4,Float16x2}, F_in)
                        E2_cplx0_dish0 = zero(E_cplx0_dish0)
                        E2_cplx1_dish0 = zero(E_cplx1_dish0)
                        E2_cplx0_dish1 = zero(E_cplx0_dish1)
                        E2_cplx1_dish1 = zero(E_cplx1_dish1)
                        let mtap = 0
                            W_mtap = Wpfb_mtap0
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap = F_ringbuf_polr_dish_mtap0
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0, E_ringbuf_polr_dish_mtap_cplx1_dish0, E_ringbuf_polr_dish_mtap_cplx0_dish1, E_ringbuf_polr_dish_mtap_cplx1_dish1) = convert(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap
                                )
                                E2_cplx0_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish0, E2_cplx0_dish0
                                )
                                E2_cplx1_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish0, E2_cplx1_dish0
                                )
                                E2_cplx0_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish1, E2_cplx0_dish1
                                )
                                E2_cplx1_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish1, E2_cplx1_dish1
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish0, E2_cplx0_dish0)
                                E2_cplx1_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish0, E2_cplx1_dish0)
                                E2_cplx0_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish1, E2_cplx0_dish1)
                                E2_cplx1_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish1, E2_cplx1_dish1)
                            end
                        end
                        let mtap = 1
                            W_mtap = Wpfb_mtap1
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap = F_ringbuf_polr_dish_mtap1
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0, E_ringbuf_polr_dish_mtap_cplx1_dish0, E_ringbuf_polr_dish_mtap_cplx0_dish1, E_ringbuf_polr_dish_mtap_cplx1_dish1) = convert(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap
                                )
                                E2_cplx0_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish0, E2_cplx0_dish0
                                )
                                E2_cplx1_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish0, E2_cplx1_dish0
                                )
                                E2_cplx0_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish1, E2_cplx0_dish1
                                )
                                E2_cplx1_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish1, E2_cplx1_dish1
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish0, E2_cplx0_dish0)
                                E2_cplx1_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish0, E2_cplx1_dish0)
                                E2_cplx0_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish1, E2_cplx0_dish1)
                                E2_cplx1_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish1, E2_cplx1_dish1)
                            end
                        end
                        let mtap = 2
                            W_mtap = Wpfb_mtap2
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap = F_ringbuf_polr_dish_mtap2
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0, E_ringbuf_polr_dish_mtap_cplx1_dish0, E_ringbuf_polr_dish_mtap_cplx0_dish1, E_ringbuf_polr_dish_mtap_cplx1_dish1) = convert(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap
                                )
                                E2_cplx0_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish0, E2_cplx0_dish0
                                )
                                E2_cplx1_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish0, E2_cplx1_dish0
                                )
                                E2_cplx0_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish1, E2_cplx0_dish1
                                )
                                E2_cplx1_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish1, E2_cplx1_dish1
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish0, E2_cplx0_dish0)
                                E2_cplx1_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish0, E2_cplx1_dish0)
                                E2_cplx0_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish1, E2_cplx0_dish1)
                                E2_cplx1_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish1, E2_cplx1_dish1)
                            end
                        end
                        let mtap = 3
                            W_mtap = Wpfb_mtap3
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap = F_ringbuf_polr_dish_mtap3
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0, E_ringbuf_polr_dish_mtap_cplx1_dish0, E_ringbuf_polr_dish_mtap_cplx0_dish1, E_ringbuf_polr_dish_mtap_cplx1_dish1) = convert(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap
                                )
                                E2_cplx0_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish0, E2_cplx0_dish0
                                )
                                E2_cplx1_dish0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish0, E2_cplx1_dish0
                                )
                                E2_cplx0_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx0_dish1, E2_cplx0_dish1
                                )
                                E2_cplx1_dish1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap, +W_mtap), E_ringbuf_polr_dish_mtap_cplx1_dish1, E2_cplx1_dish1
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish0, E2_cplx0_dish0)
                                E2_cplx1_dish0 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish0, E2_cplx1_dish0)
                                E2_cplx0_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx0_dish1, E2_cplx0_dish1)
                                E2_cplx1_dish1 = muladd(ifelse(isodd(mtap), -W_mtap, +W_mtap), E_cplx1_dish1, E2_cplx1_dish1)
                            end
                        end
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
                        F̄_shared[((((((dish::Int32 ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 2) % 2) * 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 2) * 4) ÷ 2) % 32) * 65 + (((((IndexSpaces.assume_inrange(t_inner::Int32, 0, 64, 256) ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256) ÷ 64) % 4) * 2081 + (((((dish::Int32 ÷ 32) % 4) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2) ÷ 2) % 2) * 32) + 0) + 0x01] =
                            F̄_out
                        F_ringbuf_polr_dish_m0 = F_ringbuf_polr_dish_mtap0
                        F_ringbuf_polr_dish_m1 = F_ringbuf_polr_dish_mtap1
                        F_ringbuf_polr_dish_m2 = F_ringbuf_polr_dish_mtap2
                        F_ringbuf_polr_dish_m0 = F_ringbuf_polr_dish_m1
                        F_ringbuf_polr_dish_m1 = F_ringbuf_polr_dish_m2
                        F_ringbuf_polr_dish_m2 = F_in
                        F_ringbuf_polr_dish_mtap0 = F_ringbuf_polr_dish_m0
                        F_ringbuf_polr_dish_mtap1 = F_ringbuf_polr_dish_m1
                        F_ringbuf_polr_dish_mtap2 = F_ringbuf_polr_dish_m2
                        F_ringbuf_polr_dish96_mtap0 = F_ringbuf_polr_dish_mtap0
                        F_ringbuf_polr_dish96_mtap1 = F_ringbuf_polr_dish_mtap1
                        F_ringbuf_polr_dish96_mtap2 = F_ringbuf_polr_dish_mtap2
                    end
                    F_ringbuf_dish0_mtap0 = F_ringbuf_polr_dish0_mtap0
                    F_ringbuf_dish32_mtap0 = F_ringbuf_polr_dish32_mtap0
                    F_ringbuf_dish64_mtap0 = F_ringbuf_polr_dish64_mtap0
                    F_ringbuf_dish96_mtap0 = F_ringbuf_polr_dish96_mtap0
                    F_ringbuf_dish0_mtap1 = F_ringbuf_polr_dish0_mtap1
                    F_ringbuf_dish32_mtap1 = F_ringbuf_polr_dish32_mtap1
                    F_ringbuf_dish64_mtap1 = F_ringbuf_polr_dish64_mtap1
                    F_ringbuf_dish96_mtap1 = F_ringbuf_polr_dish96_mtap1
                    F_ringbuf_dish0_mtap2 = F_ringbuf_polr_dish0_mtap2
                    F_ringbuf_dish32_mtap2 = F_ringbuf_polr_dish32_mtap2
                    F_ringbuf_dish64_mtap2 = F_ringbuf_polr_dish64_mtap2
                    F_ringbuf_dish96_mtap2 = F_ringbuf_polr_dish96_mtap2
                end
            end
            IndexSpaces.cuda_sync_threads()
            Ē_dish0_time0 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((0::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64) ÷ 2) % 32) * 65 + (((((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256) ÷ 64) % 4) * 2081 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((0::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish2_time0 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((2::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64) ÷ 2) % 32) * 65 + (((((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256) ÷ 64) % 4) * 2081 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((2::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish4_time0 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((4::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64) ÷ 2) % 32) * 65 + (((((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256) ÷ 64) % 4) * 2081 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((4::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish6_time0 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((6::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64) ÷ 2) % 32) * 65 + (((((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256) ÷ 64) % 4) * 2081 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((6::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish0_time64 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((0::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64) ÷ 2) % 32) * 65 + (((((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256) ÷ 64) % 4) * 2081 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((0::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish2_time64 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((2::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64) ÷ 2) % 32) * 65 + (((((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256) ÷ 64) % 4) * 2081 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((2::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish4_time64 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((4::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64) ÷ 2) % 32) * 65 + (((((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256) ÷ 64) % 4) * 2081 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((4::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish6_time64 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((6::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64) ÷ 2) % 32) * 65 + (((((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256) ÷ 64) % 4) * 2081 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((6::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish0_time128 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((0::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64) ÷ 2) % 32) * 65 + (((((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256) ÷ 64) % 4) * 2081 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((0::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish2_time128 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((2::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64) ÷ 2) % 32) * 65 + (((((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256) ÷ 64) % 4) * 2081 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((2::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish4_time128 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((4::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64) ÷ 2) % 32) * 65 + (((((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256) ÷ 64) % 4) * 2081 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((4::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish6_time128 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((6::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64) ÷ 2) % 32) * 65 + (((((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256) ÷ 64) % 4) * 2081 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((6::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish0_time192 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((0::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64) ÷ 2) % 32) * 65 + (((((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256) ÷ 64) % 4) * 2081 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((0::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish2_time192 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((2::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64) ÷ 2) % 32) * 65 + (((((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256) ÷ 64) % 4) * 2081 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((2::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish4_time192 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((4::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64) ÷ 2) % 32) * 65 + (((((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256) ÷ 64) % 4) * 2081 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((4::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish6_time192 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((6::Int32 ÷ 2) % 4) * 2) ÷ 4) % 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64) ÷ 2) % 32) * 65 + (((((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256) ÷ 64) % 4) * 2081 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8 + ((6::Int32 ÷ 2) % 4) * 2) ÷ 2) % 2) * 32) + 0x01]
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
            if ((0::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0i32, 256, 65536) ÷ 256) % 256) * 256 ≥
                192
                IndexSpaces.unsafe_store4!(
                    Ē_memory,
                    let
                        offset = 8192 * T̄min - 24576
                        length = 8388608
                        mod(
                            (
                                (
                                    (
                                        (
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 +
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 +
                                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64
                                        ) % 64
                                    ) * 128 +
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 2) % 2) % 2) * 64 +
                                    (
                                        (
                                            (
                                                ((0::Int32 ÷ 64) % 4) * 64 +
                                                ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256
                                            ) ÷ 64
                                        ) % 1024
                                    ) * 8192 +
                                    (
                                        (
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 +
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 +
                                            ((0::Int32 ÷ 4) % 4) * 4
                                        ) ÷ 4
                                    ) % 64
                                ) + 0
                            ) + offset,
                            length,
                        )
                    end + 0x01,
                    (Ē3_dish0_time0, Ē3_dish4_time0, Ē3_dish8_time0, Ē3_dish12_time0),
                )
            end
            if ((64::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0i32, 256, 65536) ÷ 256) % 256) * 256 ≥
                192
                IndexSpaces.unsafe_store4!(
                    Ē_memory,
                    let
                        offset = 8192 * T̄min - 24576
                        length = 8388608
                        mod(
                            (
                                (
                                    (
                                        (
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 +
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 +
                                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64
                                        ) % 64
                                    ) * 128 +
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 2) % 2) % 2) * 64 +
                                    (
                                        (
                                            (
                                                ((64::Int32 ÷ 64) % 4) * 64 +
                                                ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256
                                            ) ÷ 64
                                        ) % 1024
                                    ) * 8192 +
                                    (
                                        (
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 +
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 +
                                            ((0::Int32 ÷ 4) % 4) * 4
                                        ) ÷ 4
                                    ) % 64
                                ) + 0
                            ) + offset,
                            length,
                        )
                    end + 0x01,
                    (Ē3_dish0_time64, Ē3_dish4_time64, Ē3_dish8_time64, Ē3_dish12_time64),
                )
            end
            if ((128::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0i32, 256, 65536) ÷ 256) % 256) * 256 ≥
                192
                IndexSpaces.unsafe_store4!(
                    Ē_memory,
                    let
                        offset = 8192 * T̄min - 24576
                        length = 8388608
                        mod(
                            (
                                (
                                    (
                                        (
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 +
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 +
                                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64
                                        ) % 64
                                    ) * 128 +
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 2) % 2) % 2) * 64 +
                                    (
                                        (
                                            (
                                                ((128::Int32 ÷ 64) % 4) * 64 +
                                                ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256
                                            ) ÷ 64
                                        ) % 1024
                                    ) * 8192 +
                                    (
                                        (
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 +
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 +
                                            ((0::Int32 ÷ 4) % 4) * 4
                                        ) ÷ 4
                                    ) % 64
                                ) + 0
                            ) + offset,
                            length,
                        )
                    end + 0x01,
                    (Ē3_dish0_time128, Ē3_dish4_time128, Ē3_dish8_time128, Ē3_dish12_time128),
                )
            end
            if ((192::Int32 ÷ 64) % 4) * 64 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0i32, 256, 65536) ÷ 256) % 256) * 256 ≥
                192
                IndexSpaces.unsafe_store4!(
                    Ē_memory,
                    let
                        offset = 8192 * T̄min - 24576
                        length = 8388608
                        mod(
                            (
                                (
                                    (
                                        (
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 +
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 +
                                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 64
                                        ) % 64
                                    ) * 128 +
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 2) % 2) % 2) * 64 +
                                    (
                                        (
                                            (
                                                ((192::Int32 ÷ 64) % 4) * 64 +
                                                ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256
                                            ) ÷ 64
                                        ) % 1024
                                    ) * 8192 +
                                    (
                                        (
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 +
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 +
                                            ((0::Int32 ÷ 4) % 4) * 4
                                        ) ÷ 4
                                    ) % 64
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
        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 256) % 256) * 512 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) % 16) * 32) + 0) + 0x01] =
            info
    end
)
