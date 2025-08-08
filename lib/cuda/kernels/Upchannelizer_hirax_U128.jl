# Julia source code for the CUDA upchannelizer
# This file has been generated automatically by `upchan.jl`.
# Do not modify this file, your changes will be lost.

@fastmath @inbounds(
    begin #= /localhome/eschnett/src/kotekan/julia/kernels/upchan.jl:1490 =#
        info = 1
        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) % 16) * 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 256) % 256) * 512 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32) + 0) + 0x01] =
            info
        if !(
            0i32 ≤ Tmin < 65536 && (
                Tmin ≤ Tmax < 131072 && (
                    (Tmax - Tmin) % 256 == 0i32 &&
                    (0i32 ≤ T̄min < 512 && (T̄min ≤ T̄max < 1024 && ((T̄max - T̄min) + 3) % 2 == 0i32))
                )
            )
        )
            info = 2
            info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) % 16) * 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 256) % 256) * 512 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32) + 0) + 0x01] =
                info
            IndexSpaces.cuda_trap()
        end
        if !(0i32 ≤ Fmin ≤ Fmax ≤ F)
            info = 3
            info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) % 16) * 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 256) % 256) * 512 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32) + 0) + 0x01] =
                info
            IndexSpaces.cuda_trap()
        end
        F_ringbuf_dish0_mtap0_time0 = zero(Int4x8)
        F_ringbuf_dish32_mtap0_time0 = zero(Int4x8)
        F_ringbuf_dish64_mtap0_time0 = zero(Int4x8)
        F_ringbuf_dish96_mtap0_time0 = zero(Int4x8)
        F_ringbuf_dish0_mtap1_time0 = zero(Int4x8)
        F_ringbuf_dish32_mtap1_time0 = zero(Int4x8)
        F_ringbuf_dish64_mtap1_time0 = zero(Int4x8)
        F_ringbuf_dish96_mtap1_time0 = zero(Int4x8)
        F_ringbuf_dish0_mtap2_time0 = zero(Int4x8)
        F_ringbuf_dish32_mtap2_time0 = zero(Int4x8)
        F_ringbuf_dish64_mtap2_time0 = zero(Int4x8)
        F_ringbuf_dish96_mtap2_time0 = zero(Int4x8)
        F_ringbuf_dish0_mtap0_time1 = zero(Int4x8)
        F_ringbuf_dish32_mtap0_time1 = zero(Int4x8)
        F_ringbuf_dish64_mtap0_time1 = zero(Int4x8)
        F_ringbuf_dish96_mtap0_time1 = zero(Int4x8)
        F_ringbuf_dish0_mtap1_time1 = zero(Int4x8)
        F_ringbuf_dish32_mtap1_time1 = zero(Int4x8)
        F_ringbuf_dish64_mtap1_time1 = zero(Int4x8)
        F_ringbuf_dish96_mtap1_time1 = zero(Int4x8)
        F_ringbuf_dish0_mtap2_time1 = zero(Int4x8)
        F_ringbuf_dish32_mtap2_time1 = zero(Int4x8)
        F_ringbuf_dish64_mtap2_time1 = zero(Int4x8)
        F_ringbuf_dish96_mtap2_time1 = zero(Int4x8)
        Gains_freq0 = G_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((0::Int32 ÷ 64) % 2) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 2) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 2) % 2) * 2) ÷ 2) % 64 + 0x01]
        Gains_freq64 = G_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((64::Int32 ÷ 64) % 2) * 64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 2) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 2) % 2) * 2) ÷ 2) % 64 + 0x01]
        (Wpfb0, Wpfb1) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            time0 = 0 + thread2time(thread)
            time1 = time0 + 64
            s0 = time0 + 0
            s1 = time1 + 0
            W0 = Wkernel(s0, 4, 128) / 128.0f0
            W1 = Wkernel(s1, 4, 128) / 128.0f0
            (W0, W1)
        end
        Wpfb_m0_t0 = Float16x2(Wpfb0, Wpfb1)
        (Wpfb0, Wpfb1) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            time0 = 1 + thread2time(thread)
            time1 = time0 + 64
            s0 = time0 + 0
            s1 = time1 + 0
            W0 = Wkernel(s0, 4, 128) / 128.0f0
            W1 = Wkernel(s1, 4, 128) / 128.0f0
            (W0, W1)
        end
        Wpfb_m0_t1 = Float16x2(Wpfb0, Wpfb1)
        (Wpfb0, Wpfb1) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            time0 = 0 + thread2time(thread)
            time1 = time0 + 64
            s0 = time0 + 128
            s1 = time1 + 128
            W0 = Wkernel(s0, 4, 128) / 128.0f0
            W1 = Wkernel(s1, 4, 128) / 128.0f0
            (W0, W1)
        end
        Wpfb_m1_t0 = Float16x2(Wpfb0, Wpfb1)
        (Wpfb0, Wpfb1) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            time0 = 1 + thread2time(thread)
            time1 = time0 + 64
            s0 = time0 + 128
            s1 = time1 + 128
            W0 = Wkernel(s0, 4, 128) / 128.0f0
            W1 = Wkernel(s1, 4, 128) / 128.0f0
            (W0, W1)
        end
        Wpfb_m1_t1 = Float16x2(Wpfb0, Wpfb1)
        (Wpfb0, Wpfb1) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            time0 = 0 + thread2time(thread)
            time1 = time0 + 64
            s0 = time0 + 256
            s1 = time1 + 256
            W0 = Wkernel(s0, 4, 128) / 128.0f0
            W1 = Wkernel(s1, 4, 128) / 128.0f0
            (W0, W1)
        end
        Wpfb_m2_t0 = Float16x2(Wpfb0, Wpfb1)
        (Wpfb0, Wpfb1) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            time0 = 1 + thread2time(thread)
            time1 = time0 + 64
            s0 = time0 + 256
            s1 = time1 + 256
            W0 = Wkernel(s0, 4, 128) / 128.0f0
            W1 = Wkernel(s1, 4, 128) / 128.0f0
            (W0, W1)
        end
        Wpfb_m2_t1 = Float16x2(Wpfb0, Wpfb1)
        (Wpfb0, Wpfb1) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            time0 = 0 + thread2time(thread)
            time1 = time0 + 64
            s0 = time0 + 384
            s1 = time1 + 384
            W0 = Wkernel(s0, 4, 128) / 128.0f0
            W1 = Wkernel(s1, 4, 128) / 128.0f0
            (W0, W1)
        end
        Wpfb_m3_t0 = Float16x2(Wpfb0, Wpfb1)
        (Wpfb0, Wpfb1) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            time0 = 1 + thread2time(thread)
            time1 = time0 + 64
            s0 = time0 + 384
            s1 = time1 + 384
            W0 = Wkernel(s0, 4, 128) / 128.0f0
            W1 = Wkernel(s1, 4, 128) / 128.0f0
            (W0, W1)
        end
        Wpfb_m3_t1 = Float16x2(Wpfb0, Wpfb1)
        Wpfb_m0_time0 = Wpfb_m0_t0
        Wpfb_m0_time1 = Wpfb_m0_t1
        Wpfb_m1_time0 = Wpfb_m1_t0
        Wpfb_m1_time1 = Wpfb_m1_t1
        Wpfb_m2_time0 = Wpfb_m2_t0
        Wpfb_m2_time1 = Wpfb_m2_t1
        Wpfb_m3_time0 = Wpfb_m3_t0
        Wpfb_m3_time1 = Wpfb_m3_t1
        Wpfb_mtap0_time0 = Wpfb_m0_time0
        Wpfb_mtap1_time0 = Wpfb_m1_time0
        Wpfb_mtap2_time0 = Wpfb_m2_time0
        Wpfb_mtap3_time0 = Wpfb_m3_time0
        Wpfb_mtap0_time1 = Wpfb_m0_time1
        Wpfb_mtap1_time1 = Wpfb_m1_time1
        Wpfb_mtap2_time1 = Wpfb_m2_time1
        Wpfb_mtap3_time1 = Wpfb_m3_time1
        (X0, X1) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            time0 = thread2time(thread)
            time1 = time0 + 64
            X0 = cispi(((time0 * 127) % 256) / 128.0f0)
            X1 = cispi(((time1 * 127) % 256) / 128.0f0)
            (X0, X1)
        end
        Xre_time0 = Float16x2(real(X0), real(X1))
        Xre_time1 = Float16x2(real(X0), real(X1))
        Xim_time0 = Float16x2(imag(X0), imag(X1))
        Xim_time1 = Float16x2(imag(X0), imag(X1))
        X_cplx0_time0 = Xre_time0
        X_cplx1_time0 = Xim_time0
        X_cplx0_time1 = Xre_time1
        X_cplx1_time1 = Xim_time1
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
        Γ¹_cplx0_cplx_in0_time0 = Γ¹_cplx0_cplx_in0
        Γ¹_cplx0_cplx_in0_time1 = Γ¹_cplx0_cplx_in0
        Γ¹_cplx1_cplx_in0_time0 = Γ¹_cplx1_cplx_in0
        Γ¹_cplx1_cplx_in0_time1 = Γ¹_cplx1_cplx_in0
        Γ¹_cplx0_cplx_in1_time0 = Γ¹_cplx0_cplx_in1
        Γ¹_cplx0_cplx_in1_time1 = Γ¹_cplx0_cplx_in1
        Γ¹_cplx1_cplx_in1_time0 = Γ¹_cplx1_cplx_in1
        Γ¹_cplx1_cplx_in1_time1 = Γ¹_cplx1_cplx_in1
        (Γ²0, Γ²1) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            thread0 = (thread ÷ (1i32)) % (2i32)
            thread1 = (thread ÷ (2i32)) % (2i32)
            thread2 = (thread ÷ (4i32)) % (2i32)
            thread3 = (thread ÷ (8i32)) % (2i32)
            thread4 = (thread ÷ (16i32)) % (2i32)
            timelo0 = (8i32) * (0i32) + (4i32) * thread1 + (2i32) * thread0
            timelo1 = (8i32) * (1i32) + (4i32) * thread1 + (2i32) * thread0
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
        Γ²_cplx0_time0 = Γ²_cplx0
        Γ²_cplx0_time1 = Γ²_cplx0
        Γ²_cplx1_time0 = Γ²_cplx1
        Γ²_cplx1_time1 = Γ²_cplx1
        (Γ³0, Γ³1) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            thread0 = (thread ÷ (1i32)) % (2i32)
            thread1 = (thread ÷ (2i32)) % (2i32)
            thread2 = (thread ÷ (4i32)) % (2i32)
            thread3 = (thread ÷ (8i32)) % (2i32)
            thread4 = (thread ÷ (16i32)) % (2i32)
            timelo0 = (8i32) * (0i32) + (4i32) * thread1 + (2i32) * thread0
            timelo1 = (8i32) * (1i32) + (4i32) * thread1 + (2i32) * thread0
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
        Γ³_cplx0_cplx_in0_dish0 = Γ³_cplx0_cplx_in0_dish0
        Γ³_cplx0_cplx_in0_dish128 = Γ³_cplx0_cplx_in0_dish0
        Γ³_cplx1_cplx_in0_dish0 = Γ³_cplx1_cplx_in0_dish0
        Γ³_cplx1_cplx_in0_dish128 = Γ³_cplx1_cplx_in0_dish0
        Γ³_cplx0_cplx_in1_dish0 = Γ³_cplx0_cplx_in1_dish0
        Γ³_cplx0_cplx_in1_dish128 = Γ³_cplx0_cplx_in1_dish0
        Γ³_cplx1_cplx_in1_dish0 = Γ³_cplx1_cplx_in1_dish0
        Γ³_cplx1_cplx_in1_dish128 = Γ³_cplx1_cplx_in1_dish0
        Γ³_cplx0_cplx_in0_dish1 = Γ³_cplx0_cplx_in0_dish1
        Γ³_cplx0_cplx_in0_dish129 = Γ³_cplx0_cplx_in0_dish1
        Γ³_cplx1_cplx_in0_dish1 = Γ³_cplx1_cplx_in0_dish1
        Γ³_cplx1_cplx_in0_dish129 = Γ³_cplx1_cplx_in0_dish1
        Γ³_cplx0_cplx_in1_dish1 = Γ³_cplx0_cplx_in1_dish1
        Γ³_cplx0_cplx_in1_dish129 = Γ³_cplx0_cplx_in1_dish1
        Γ³_cplx1_cplx_in1_dish1 = Γ³_cplx1_cplx_in1_dish1
        Γ³_cplx1_cplx_in1_dish129 = Γ³_cplx1_cplx_in1_dish1
        Γ³_cplx0_cplx_in0_dish32 = Γ³_cplx0_cplx_in0_dish32
        Γ³_cplx0_cplx_in0_dish160 = Γ³_cplx0_cplx_in0_dish32
        Γ³_cplx1_cplx_in0_dish32 = Γ³_cplx1_cplx_in0_dish32
        Γ³_cplx1_cplx_in0_dish160 = Γ³_cplx1_cplx_in0_dish32
        Γ³_cplx0_cplx_in1_dish32 = Γ³_cplx0_cplx_in1_dish32
        Γ³_cplx0_cplx_in1_dish160 = Γ³_cplx0_cplx_in1_dish32
        Γ³_cplx1_cplx_in1_dish32 = Γ³_cplx1_cplx_in1_dish32
        Γ³_cplx1_cplx_in1_dish160 = Γ³_cplx1_cplx_in1_dish32
        Γ³_cplx0_cplx_in0_dish33 = Γ³_cplx0_cplx_in0_dish33
        Γ³_cplx0_cplx_in0_dish161 = Γ³_cplx0_cplx_in0_dish33
        Γ³_cplx1_cplx_in0_dish33 = Γ³_cplx1_cplx_in0_dish33
        Γ³_cplx1_cplx_in0_dish161 = Γ³_cplx1_cplx_in0_dish33
        Γ³_cplx0_cplx_in1_dish33 = Γ³_cplx0_cplx_in1_dish33
        Γ³_cplx0_cplx_in1_dish161 = Γ³_cplx0_cplx_in1_dish33
        Γ³_cplx1_cplx_in1_dish33 = Γ³_cplx1_cplx_in1_dish33
        Γ³_cplx1_cplx_in1_dish161 = Γ³_cplx1_cplx_in1_dish33
        Γ³_cplx0_cplx_in0_dish64 = Γ³_cplx0_cplx_in0_dish64
        Γ³_cplx0_cplx_in0_dish192 = Γ³_cplx0_cplx_in0_dish64
        Γ³_cplx1_cplx_in0_dish64 = Γ³_cplx1_cplx_in0_dish64
        Γ³_cplx1_cplx_in0_dish192 = Γ³_cplx1_cplx_in0_dish64
        Γ³_cplx0_cplx_in1_dish64 = Γ³_cplx0_cplx_in1_dish64
        Γ³_cplx0_cplx_in1_dish192 = Γ³_cplx0_cplx_in1_dish64
        Γ³_cplx1_cplx_in1_dish64 = Γ³_cplx1_cplx_in1_dish64
        Γ³_cplx1_cplx_in1_dish192 = Γ³_cplx1_cplx_in1_dish64
        Γ³_cplx0_cplx_in0_dish65 = Γ³_cplx0_cplx_in0_dish65
        Γ³_cplx0_cplx_in0_dish193 = Γ³_cplx0_cplx_in0_dish65
        Γ³_cplx1_cplx_in0_dish65 = Γ³_cplx1_cplx_in0_dish65
        Γ³_cplx1_cplx_in0_dish193 = Γ³_cplx1_cplx_in0_dish65
        Γ³_cplx0_cplx_in1_dish65 = Γ³_cplx0_cplx_in1_dish65
        Γ³_cplx0_cplx_in1_dish193 = Γ³_cplx0_cplx_in1_dish65
        Γ³_cplx1_cplx_in1_dish65 = Γ³_cplx1_cplx_in1_dish65
        Γ³_cplx1_cplx_in1_dish193 = Γ³_cplx1_cplx_in1_dish65
        Γ³_cplx0_cplx_in0_dish96 = Γ³_cplx0_cplx_in0_dish96
        Γ³_cplx0_cplx_in0_dish224 = Γ³_cplx0_cplx_in0_dish96
        Γ³_cplx1_cplx_in0_dish96 = Γ³_cplx1_cplx_in0_dish96
        Γ³_cplx1_cplx_in0_dish224 = Γ³_cplx1_cplx_in0_dish96
        Γ³_cplx0_cplx_in1_dish96 = Γ³_cplx0_cplx_in1_dish96
        Γ³_cplx0_cplx_in1_dish224 = Γ³_cplx0_cplx_in1_dish96
        Γ³_cplx1_cplx_in1_dish96 = Γ³_cplx1_cplx_in1_dish96
        Γ³_cplx1_cplx_in1_dish224 = Γ³_cplx1_cplx_in1_dish96
        Γ³_cplx0_cplx_in0_dish97 = Γ³_cplx0_cplx_in0_dish97
        Γ³_cplx0_cplx_in0_dish225 = Γ³_cplx0_cplx_in0_dish97
        Γ³_cplx1_cplx_in0_dish97 = Γ³_cplx1_cplx_in0_dish97
        Γ³_cplx1_cplx_in0_dish225 = Γ³_cplx1_cplx_in0_dish97
        Γ³_cplx0_cplx_in1_dish97 = Γ³_cplx0_cplx_in1_dish97
        Γ³_cplx0_cplx_in1_dish225 = Γ³_cplx0_cplx_in1_dish97
        Γ³_cplx1_cplx_in1_dish97 = Γ³_cplx1_cplx_in1_dish97
        Γ³_cplx1_cplx_in1_dish225 = Γ³_cplx1_cplx_in1_dish97
        Γ³_cplx0_cplx_in0_dish0_time0 = Γ³_cplx0_cplx_in0_dish0
        Γ³_cplx0_cplx_in0_dish0_time1 = Γ³_cplx0_cplx_in0_dish0
        Γ³_cplx1_cplx_in0_dish0_time0 = Γ³_cplx1_cplx_in0_dish0
        Γ³_cplx1_cplx_in0_dish0_time1 = Γ³_cplx1_cplx_in0_dish0
        Γ³_cplx0_cplx_in1_dish0_time0 = Γ³_cplx0_cplx_in1_dish0
        Γ³_cplx0_cplx_in1_dish0_time1 = Γ³_cplx0_cplx_in1_dish0
        Γ³_cplx1_cplx_in1_dish0_time0 = Γ³_cplx1_cplx_in1_dish0
        Γ³_cplx1_cplx_in1_dish0_time1 = Γ³_cplx1_cplx_in1_dish0
        Γ³_cplx0_cplx_in0_dish1_time0 = Γ³_cplx0_cplx_in0_dish1
        Γ³_cplx0_cplx_in0_dish1_time1 = Γ³_cplx0_cplx_in0_dish1
        Γ³_cplx1_cplx_in0_dish1_time0 = Γ³_cplx1_cplx_in0_dish1
        Γ³_cplx1_cplx_in0_dish1_time1 = Γ³_cplx1_cplx_in0_dish1
        Γ³_cplx0_cplx_in1_dish1_time0 = Γ³_cplx0_cplx_in1_dish1
        Γ³_cplx0_cplx_in1_dish1_time1 = Γ³_cplx0_cplx_in1_dish1
        Γ³_cplx1_cplx_in1_dish1_time0 = Γ³_cplx1_cplx_in1_dish1
        Γ³_cplx1_cplx_in1_dish1_time1 = Γ³_cplx1_cplx_in1_dish1
        Γ³_cplx0_cplx_in0_dish32_time0 = Γ³_cplx0_cplx_in0_dish32
        Γ³_cplx0_cplx_in0_dish32_time1 = Γ³_cplx0_cplx_in0_dish32
        Γ³_cplx1_cplx_in0_dish32_time0 = Γ³_cplx1_cplx_in0_dish32
        Γ³_cplx1_cplx_in0_dish32_time1 = Γ³_cplx1_cplx_in0_dish32
        Γ³_cplx0_cplx_in1_dish32_time0 = Γ³_cplx0_cplx_in1_dish32
        Γ³_cplx0_cplx_in1_dish32_time1 = Γ³_cplx0_cplx_in1_dish32
        Γ³_cplx1_cplx_in1_dish32_time0 = Γ³_cplx1_cplx_in1_dish32
        Γ³_cplx1_cplx_in1_dish32_time1 = Γ³_cplx1_cplx_in1_dish32
        Γ³_cplx0_cplx_in0_dish33_time0 = Γ³_cplx0_cplx_in0_dish33
        Γ³_cplx0_cplx_in0_dish33_time1 = Γ³_cplx0_cplx_in0_dish33
        Γ³_cplx1_cplx_in0_dish33_time0 = Γ³_cplx1_cplx_in0_dish33
        Γ³_cplx1_cplx_in0_dish33_time1 = Γ³_cplx1_cplx_in0_dish33
        Γ³_cplx0_cplx_in1_dish33_time0 = Γ³_cplx0_cplx_in1_dish33
        Γ³_cplx0_cplx_in1_dish33_time1 = Γ³_cplx0_cplx_in1_dish33
        Γ³_cplx1_cplx_in1_dish33_time0 = Γ³_cplx1_cplx_in1_dish33
        Γ³_cplx1_cplx_in1_dish33_time1 = Γ³_cplx1_cplx_in1_dish33
        Γ³_cplx0_cplx_in0_dish64_time0 = Γ³_cplx0_cplx_in0_dish64
        Γ³_cplx0_cplx_in0_dish64_time1 = Γ³_cplx0_cplx_in0_dish64
        Γ³_cplx1_cplx_in0_dish64_time0 = Γ³_cplx1_cplx_in0_dish64
        Γ³_cplx1_cplx_in0_dish64_time1 = Γ³_cplx1_cplx_in0_dish64
        Γ³_cplx0_cplx_in1_dish64_time0 = Γ³_cplx0_cplx_in1_dish64
        Γ³_cplx0_cplx_in1_dish64_time1 = Γ³_cplx0_cplx_in1_dish64
        Γ³_cplx1_cplx_in1_dish64_time0 = Γ³_cplx1_cplx_in1_dish64
        Γ³_cplx1_cplx_in1_dish64_time1 = Γ³_cplx1_cplx_in1_dish64
        Γ³_cplx0_cplx_in0_dish65_time0 = Γ³_cplx0_cplx_in0_dish65
        Γ³_cplx0_cplx_in0_dish65_time1 = Γ³_cplx0_cplx_in0_dish65
        Γ³_cplx1_cplx_in0_dish65_time0 = Γ³_cplx1_cplx_in0_dish65
        Γ³_cplx1_cplx_in0_dish65_time1 = Γ³_cplx1_cplx_in0_dish65
        Γ³_cplx0_cplx_in1_dish65_time0 = Γ³_cplx0_cplx_in1_dish65
        Γ³_cplx0_cplx_in1_dish65_time1 = Γ³_cplx0_cplx_in1_dish65
        Γ³_cplx1_cplx_in1_dish65_time0 = Γ³_cplx1_cplx_in1_dish65
        Γ³_cplx1_cplx_in1_dish65_time1 = Γ³_cplx1_cplx_in1_dish65
        Γ³_cplx0_cplx_in0_dish96_time0 = Γ³_cplx0_cplx_in0_dish96
        Γ³_cplx0_cplx_in0_dish96_time1 = Γ³_cplx0_cplx_in0_dish96
        Γ³_cplx1_cplx_in0_dish96_time0 = Γ³_cplx1_cplx_in0_dish96
        Γ³_cplx1_cplx_in0_dish96_time1 = Γ³_cplx1_cplx_in0_dish96
        Γ³_cplx0_cplx_in1_dish96_time0 = Γ³_cplx0_cplx_in1_dish96
        Γ³_cplx0_cplx_in1_dish96_time1 = Γ³_cplx0_cplx_in1_dish96
        Γ³_cplx1_cplx_in1_dish96_time0 = Γ³_cplx1_cplx_in1_dish96
        Γ³_cplx1_cplx_in1_dish96_time1 = Γ³_cplx1_cplx_in1_dish96
        Γ³_cplx0_cplx_in0_dish97_time0 = Γ³_cplx0_cplx_in0_dish97
        Γ³_cplx0_cplx_in0_dish97_time1 = Γ³_cplx0_cplx_in0_dish97
        Γ³_cplx1_cplx_in0_dish97_time0 = Γ³_cplx1_cplx_in0_dish97
        Γ³_cplx1_cplx_in0_dish97_time1 = Γ³_cplx1_cplx_in0_dish97
        Γ³_cplx0_cplx_in1_dish97_time0 = Γ³_cplx0_cplx_in1_dish97
        Γ³_cplx0_cplx_in1_dish97_time1 = Γ³_cplx0_cplx_in1_dish97
        Γ³_cplx1_cplx_in1_dish97_time0 = Γ³_cplx1_cplx_in1_dish97
        Γ³_cplx1_cplx_in1_dish97_time1 = Γ³_cplx1_cplx_in1_dish97
        Γ³_cplx0_cplx_in0_dish128_time0 = Γ³_cplx0_cplx_in0_dish128
        Γ³_cplx0_cplx_in0_dish128_time1 = Γ³_cplx0_cplx_in0_dish128
        Γ³_cplx1_cplx_in0_dish128_time0 = Γ³_cplx1_cplx_in0_dish128
        Γ³_cplx1_cplx_in0_dish128_time1 = Γ³_cplx1_cplx_in0_dish128
        Γ³_cplx0_cplx_in1_dish128_time0 = Γ³_cplx0_cplx_in1_dish128
        Γ³_cplx0_cplx_in1_dish128_time1 = Γ³_cplx0_cplx_in1_dish128
        Γ³_cplx1_cplx_in1_dish128_time0 = Γ³_cplx1_cplx_in1_dish128
        Γ³_cplx1_cplx_in1_dish128_time1 = Γ³_cplx1_cplx_in1_dish128
        Γ³_cplx0_cplx_in0_dish129_time0 = Γ³_cplx0_cplx_in0_dish129
        Γ³_cplx0_cplx_in0_dish129_time1 = Γ³_cplx0_cplx_in0_dish129
        Γ³_cplx1_cplx_in0_dish129_time0 = Γ³_cplx1_cplx_in0_dish129
        Γ³_cplx1_cplx_in0_dish129_time1 = Γ³_cplx1_cplx_in0_dish129
        Γ³_cplx0_cplx_in1_dish129_time0 = Γ³_cplx0_cplx_in1_dish129
        Γ³_cplx0_cplx_in1_dish129_time1 = Γ³_cplx0_cplx_in1_dish129
        Γ³_cplx1_cplx_in1_dish129_time0 = Γ³_cplx1_cplx_in1_dish129
        Γ³_cplx1_cplx_in1_dish129_time1 = Γ³_cplx1_cplx_in1_dish129
        Γ³_cplx0_cplx_in0_dish160_time0 = Γ³_cplx0_cplx_in0_dish160
        Γ³_cplx0_cplx_in0_dish160_time1 = Γ³_cplx0_cplx_in0_dish160
        Γ³_cplx1_cplx_in0_dish160_time0 = Γ³_cplx1_cplx_in0_dish160
        Γ³_cplx1_cplx_in0_dish160_time1 = Γ³_cplx1_cplx_in0_dish160
        Γ³_cplx0_cplx_in1_dish160_time0 = Γ³_cplx0_cplx_in1_dish160
        Γ³_cplx0_cplx_in1_dish160_time1 = Γ³_cplx0_cplx_in1_dish160
        Γ³_cplx1_cplx_in1_dish160_time0 = Γ³_cplx1_cplx_in1_dish160
        Γ³_cplx1_cplx_in1_dish160_time1 = Γ³_cplx1_cplx_in1_dish160
        Γ³_cplx0_cplx_in0_dish161_time0 = Γ³_cplx0_cplx_in0_dish161
        Γ³_cplx0_cplx_in0_dish161_time1 = Γ³_cplx0_cplx_in0_dish161
        Γ³_cplx1_cplx_in0_dish161_time0 = Γ³_cplx1_cplx_in0_dish161
        Γ³_cplx1_cplx_in0_dish161_time1 = Γ³_cplx1_cplx_in0_dish161
        Γ³_cplx0_cplx_in1_dish161_time0 = Γ³_cplx0_cplx_in1_dish161
        Γ³_cplx0_cplx_in1_dish161_time1 = Γ³_cplx0_cplx_in1_dish161
        Γ³_cplx1_cplx_in1_dish161_time0 = Γ³_cplx1_cplx_in1_dish161
        Γ³_cplx1_cplx_in1_dish161_time1 = Γ³_cplx1_cplx_in1_dish161
        Γ³_cplx0_cplx_in0_dish192_time0 = Γ³_cplx0_cplx_in0_dish192
        Γ³_cplx0_cplx_in0_dish192_time1 = Γ³_cplx0_cplx_in0_dish192
        Γ³_cplx1_cplx_in0_dish192_time0 = Γ³_cplx1_cplx_in0_dish192
        Γ³_cplx1_cplx_in0_dish192_time1 = Γ³_cplx1_cplx_in0_dish192
        Γ³_cplx0_cplx_in1_dish192_time0 = Γ³_cplx0_cplx_in1_dish192
        Γ³_cplx0_cplx_in1_dish192_time1 = Γ³_cplx0_cplx_in1_dish192
        Γ³_cplx1_cplx_in1_dish192_time0 = Γ³_cplx1_cplx_in1_dish192
        Γ³_cplx1_cplx_in1_dish192_time1 = Γ³_cplx1_cplx_in1_dish192
        Γ³_cplx0_cplx_in0_dish193_time0 = Γ³_cplx0_cplx_in0_dish193
        Γ³_cplx0_cplx_in0_dish193_time1 = Γ³_cplx0_cplx_in0_dish193
        Γ³_cplx1_cplx_in0_dish193_time0 = Γ³_cplx1_cplx_in0_dish193
        Γ³_cplx1_cplx_in0_dish193_time1 = Γ³_cplx1_cplx_in0_dish193
        Γ³_cplx0_cplx_in1_dish193_time0 = Γ³_cplx0_cplx_in1_dish193
        Γ³_cplx0_cplx_in1_dish193_time1 = Γ³_cplx0_cplx_in1_dish193
        Γ³_cplx1_cplx_in1_dish193_time0 = Γ³_cplx1_cplx_in1_dish193
        Γ³_cplx1_cplx_in1_dish193_time1 = Γ³_cplx1_cplx_in1_dish193
        Γ³_cplx0_cplx_in0_dish224_time0 = Γ³_cplx0_cplx_in0_dish224
        Γ³_cplx0_cplx_in0_dish224_time1 = Γ³_cplx0_cplx_in0_dish224
        Γ³_cplx1_cplx_in0_dish224_time0 = Γ³_cplx1_cplx_in0_dish224
        Γ³_cplx1_cplx_in0_dish224_time1 = Γ³_cplx1_cplx_in0_dish224
        Γ³_cplx0_cplx_in1_dish224_time0 = Γ³_cplx0_cplx_in1_dish224
        Γ³_cplx0_cplx_in1_dish224_time1 = Γ³_cplx0_cplx_in1_dish224
        Γ³_cplx1_cplx_in1_dish224_time0 = Γ³_cplx1_cplx_in1_dish224
        Γ³_cplx1_cplx_in1_dish224_time1 = Γ³_cplx1_cplx_in1_dish224
        Γ³_cplx0_cplx_in0_dish225_time0 = Γ³_cplx0_cplx_in0_dish225
        Γ³_cplx0_cplx_in0_dish225_time1 = Γ³_cplx0_cplx_in0_dish225
        Γ³_cplx1_cplx_in0_dish225_time0 = Γ³_cplx1_cplx_in0_dish225
        Γ³_cplx1_cplx_in0_dish225_time1 = Γ³_cplx1_cplx_in0_dish225
        Γ³_cplx0_cplx_in1_dish225_time0 = Γ³_cplx0_cplx_in1_dish225
        Γ³_cplx0_cplx_in1_dish225_time1 = Γ³_cplx0_cplx_in1_dish225
        Γ³_cplx1_cplx_in1_dish225_time0 = Γ³_cplx1_cplx_in1_dish225
        Γ³_cplx1_cplx_in1_dish225_time1 = Γ³_cplx1_cplx_in1_dish225
        (Γ⁴0, Γ⁴1) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            thread0 = (thread ÷ (1i32)) % (2i32)
            thread1 = (thread ÷ (2i32)) % (2i32)
            thread2 = (thread ÷ (4i32)) % (2i32)
            thread3 = (thread ÷ (8i32)) % (2i32)
            thread4 = (thread ÷ (16i32)) % (2i32)
            timelo = 1i32
            freqlo0 =
                (1i32) * (0i32) + (2i32) * thread1 + (4i32) * thread0 + (8i32) * thread2 + (16i32) * thread4 + (32i32) * thread3
            freqlo1 =
                (1i32) * (1i32) + (2i32) * thread1 + (4i32) * thread0 + (8i32) * thread2 + (16i32) * thread4 + (32i32) * thread3
            (Γ⁴0, Γ⁴1) = (
                cispi((((-2i32) * timelo * freqlo0) / 128.0f0) % 2.0f0), cispi((((-2i32) * timelo * freqlo1) / 128.0f0) % 2.0f0)
            )
            (Γ⁴0, Γ⁴1)
        end
        Γ⁴re = Float16x2(real(Γ⁴0), real(Γ⁴1))
        Γ⁴im = Float16x2(imag(Γ⁴0), imag(Γ⁴1))
        Γ⁴_cplx0 = Γ⁴re
        Γ⁴_cplx1 = Γ⁴im
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
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128) ÷
                                    128
                                ) % 64
                            ) * 128 +
                            (
                                (
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 +
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 +
                                    ((0::Int32 ÷ 4) % 4) * 4
                                ) ÷ 4
                            ) % 64 +
                            (
                                (
                                    ((0::Int32 ÷ 16) % 2) * 16 +
                                    ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 +
                                    IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 +
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 +
                                    ((0::Int32 ÷ 128) % 2) * 128 +
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 64
                                ) % 65536
                            ) * 8192
                        ) + offset,
                        length,
                    )
                end + 0x01,
            )
            (E_dish0_time16, E_dish4_time16, E_dish8_time16, E_dish12_time16) = IndexSpaces.unsafe_load4(
                E_memory,
                let
                    offset = 8192 * Tmin + 128 * Fmin
                    length = 536870912
                    mod(
                        (
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 2) % 2) % 2) * 64 +
                            (
                                (
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128) ÷
                                    128
                                ) % 64
                            ) * 128 +
                            (
                                (
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 +
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 +
                                    ((0::Int32 ÷ 4) % 4) * 4
                                ) ÷ 4
                            ) % 64 +
                            (
                                (
                                    ((16::Int32 ÷ 16) % 2) * 16 +
                                    ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 +
                                    IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 +
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 +
                                    ((16::Int32 ÷ 128) % 2) * 128 +
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 64
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
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128) ÷
                                    128
                                ) % 64
                            ) * 128 +
                            (
                                (
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 +
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 +
                                    ((0::Int32 ÷ 4) % 4) * 4
                                ) ÷ 4
                            ) % 64 +
                            (
                                (
                                    ((128::Int32 ÷ 16) % 2) * 16 +
                                    ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 +
                                    IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 +
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 +
                                    ((128::Int32 ÷ 128) % 2) * 128 +
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 64
                                ) % 65536
                            ) * 8192
                        ) + offset,
                        length,
                    )
                end + 0x01,
            )
            (E_dish0_time144, E_dish4_time144, E_dish8_time144, E_dish12_time144) = IndexSpaces.unsafe_load4(
                E_memory,
                let
                    offset = 8192 * Tmin + 128 * Fmin
                    length = 536870912
                    mod(
                        (
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 2) % 2) % 2) * 64 +
                            (
                                (
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128) ÷
                                    128
                                ) % 64
                            ) * 128 +
                            (
                                (
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 +
                                    (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 +
                                    ((0::Int32 ÷ 4) % 4) * 4
                                ) ÷ 4
                            ) % 64 +
                            (
                                (
                                    ((144::Int32 ÷ 16) % 2) * 16 +
                                    ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 +
                                    IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 +
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 +
                                    ((144::Int32 ÷ 128) % 2) * 128 +
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 64
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
            (E1_dish0_time16, E1_dish8_time16) = let
                src = if is_lo_thread
                    E_dish8_time16
                else
                    E_dish0_time16
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E_dish0_time16, dst)
                else
                    (dst, E_dish8_time16)
                end
            end
            (E1_dish4_time16, E1_dish12_time16) = let
                src = if is_lo_thread
                    E_dish12_time16
                else
                    E_dish4_time16
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E_dish4_time16, dst)
                else
                    (dst, E_dish12_time16)
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
            (E1_dish0_time144, E1_dish8_time144) = let
                src = if is_lo_thread
                    E_dish8_time144
                else
                    E_dish0_time144
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E_dish0_time144, dst)
                else
                    (dst, E_dish8_time144)
                end
            end
            (E1_dish4_time144, E1_dish12_time144) = let
                src = if is_lo_thread
                    E_dish12_time144
                else
                    E_dish4_time144
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E_dish4_time144, dst)
                else
                    (dst, E_dish12_time144)
                end
            end
            E1lo_dish0_time0 = E1_dish0_time0
            E1hi_dish0_time0 = E1_dish8_time0
            E1lo_dish4_time0 = E1_dish4_time0
            E1hi_dish4_time0 = E1_dish12_time0
            E1lo_dish0_time16 = E1_dish0_time16
            E1hi_dish0_time16 = E1_dish8_time16
            E1lo_dish4_time16 = E1_dish4_time16
            E1hi_dish4_time16 = E1_dish12_time16
            E1lo_dish0_time128 = E1_dish0_time128
            E1hi_dish0_time128 = E1_dish8_time128
            E1lo_dish4_time128 = E1_dish4_time128
            E1hi_dish4_time128 = E1_dish12_time128
            E1lo_dish0_time144 = E1_dish0_time144
            E1hi_dish0_time144 = E1_dish8_time144
            E1lo_dish4_time144 = E1_dish4_time144
            E1hi_dish4_time144 = E1_dish12_time144
            E1_dish0_time0 = E1lo_dish0_time0
            E1_dish0_time64 = E1hi_dish0_time0
            E1_dish4_time0 = E1lo_dish4_time0
            E1_dish4_time64 = E1hi_dish4_time0
            E1_dish0_time16 = E1lo_dish0_time16
            E1_dish0_time80 = E1hi_dish0_time16
            E1_dish4_time16 = E1lo_dish4_time16
            E1_dish4_time80 = E1hi_dish4_time16
            E1_dish0_time128 = E1lo_dish0_time128
            E1_dish0_time192 = E1hi_dish0_time128
            E1_dish4_time128 = E1lo_dish4_time128
            E1_dish4_time192 = E1hi_dish4_time128
            E1_dish0_time144 = E1lo_dish0_time144
            E1_dish0_time208 = E1hi_dish0_time144
            E1_dish4_time144 = E1lo_dish4_time144
            E1_dish4_time208 = E1hi_dish4_time144
            (E2_dish0_time0, E2_dish0_time64) = (
                IndexSpaces.get_lo16(E1_dish0_time0, E1_dish0_time64), IndexSpaces.get_hi16(E1_dish0_time0, E1_dish0_time64)
            )
            (E2_dish4_time0, E2_dish4_time64) = (
                IndexSpaces.get_lo16(E1_dish4_time0, E1_dish4_time64), IndexSpaces.get_hi16(E1_dish4_time0, E1_dish4_time64)
            )
            (E2_dish0_time16, E2_dish0_time80) = (
                IndexSpaces.get_lo16(E1_dish0_time16, E1_dish0_time80), IndexSpaces.get_hi16(E1_dish0_time16, E1_dish0_time80)
            )
            (E2_dish4_time16, E2_dish4_time80) = (
                IndexSpaces.get_lo16(E1_dish4_time16, E1_dish4_time80), IndexSpaces.get_hi16(E1_dish4_time16, E1_dish4_time80)
            )
            (E2_dish0_time128, E2_dish0_time192) = (
                IndexSpaces.get_lo16(E1_dish0_time128, E1_dish0_time192), IndexSpaces.get_hi16(E1_dish0_time128, E1_dish0_time192)
            )
            (E2_dish4_time128, E2_dish4_time192) = (
                IndexSpaces.get_lo16(E1_dish4_time128, E1_dish4_time192), IndexSpaces.get_hi16(E1_dish4_time128, E1_dish4_time192)
            )
            (E2_dish0_time144, E2_dish0_time208) = (
                IndexSpaces.get_lo16(E1_dish0_time144, E1_dish0_time208), IndexSpaces.get_hi16(E1_dish0_time144, E1_dish0_time208)
            )
            (E2_dish4_time144, E2_dish4_time208) = (
                IndexSpaces.get_lo16(E1_dish4_time144, E1_dish4_time208), IndexSpaces.get_hi16(E1_dish4_time144, E1_dish4_time208)
            )
            E2lo_dish0_time0 = E2_dish0_time0
            E2hi_dish0_time0 = E2_dish0_time64
            E2lo_dish4_time0 = E2_dish4_time0
            E2hi_dish4_time0 = E2_dish4_time64
            E2lo_dish0_time16 = E2_dish0_time16
            E2hi_dish0_time16 = E2_dish0_time80
            E2lo_dish4_time16 = E2_dish4_time16
            E2hi_dish4_time16 = E2_dish4_time80
            E2lo_dish0_time128 = E2_dish0_time128
            E2hi_dish0_time128 = E2_dish0_time192
            E2lo_dish4_time128 = E2_dish4_time128
            E2hi_dish4_time128 = E2_dish4_time192
            E2lo_dish0_time144 = E2_dish0_time144
            E2hi_dish0_time144 = E2_dish0_time208
            E2lo_dish4_time144 = E2_dish4_time144
            E2hi_dish4_time144 = E2_dish4_time208
            E2_dish0_time0 = E2lo_dish0_time0
            E2_dish2_time0 = E2hi_dish0_time0
            E2_dish4_time0 = E2lo_dish4_time0
            E2_dish6_time0 = E2hi_dish4_time0
            E2_dish0_time16 = E2lo_dish0_time16
            E2_dish2_time16 = E2hi_dish0_time16
            E2_dish4_time16 = E2lo_dish4_time16
            E2_dish6_time16 = E2hi_dish4_time16
            E2_dish0_time128 = E2lo_dish0_time128
            E2_dish2_time128 = E2hi_dish0_time128
            E2_dish4_time128 = E2lo_dish4_time128
            E2_dish6_time128 = E2hi_dish4_time128
            E2_dish0_time144 = E2lo_dish0_time144
            E2_dish2_time144 = E2hi_dish0_time144
            E2_dish4_time144 = E2lo_dish4_time144
            E2_dish6_time144 = E2hi_dish4_time144
            F_dish0_time0 = E2_dish0_time0
            F_dish2_time0 = E2_dish2_time0
            F_dish4_time0 = E2_dish4_time0
            F_dish6_time0 = E2_dish6_time0
            F_dish0_time16 = E2_dish0_time16
            F_dish2_time16 = E2_dish2_time16
            F_dish4_time16 = E2_dish4_time16
            F_dish6_time16 = E2_dish6_time16
            F_dish0_time128 = E2_dish0_time128
            F_dish2_time128 = E2_dish2_time128
            F_dish4_time128 = E2_dish4_time128
            F_dish6_time128 = E2_dish6_time128
            F_dish0_time144 = E2_dish0_time144
            F_dish2_time144 = E2_dish2_time144
            F_dish4_time144 = E2_dish4_time144
            F_dish6_time144 = E2_dish6_time144
            F_shared[(((((((0::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 16) % 2) * 130 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((0::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((0::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 8) % 2) * 260 + (((((0::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 32) % 2) * 65 + (((((0::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((0::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 4) % 2) * 520 + (((((0::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 2) % 2) * 1040 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((0::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32 + ((((0::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((0::Int32 ÷ 128) % 2) * 128) % 2) * 2080) + 0) + 0x01] =
                F_dish0_time0
            F_shared[(((((((0::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 16) % 2) * 130 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((2::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((0::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 8) % 2) * 260 + (((((0::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 32) % 2) * 65 + (((((0::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((0::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 4) % 2) * 520 + (((((0::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 2) % 2) * 1040 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((2::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32 + ((((0::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((0::Int32 ÷ 128) % 2) * 128) % 2) * 2080) + 0) + 0x01] =
                F_dish2_time0
            F_shared[(((((((0::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 16) % 2) * 130 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((4::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((0::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 8) % 2) * 260 + (((((0::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 32) % 2) * 65 + (((((0::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((0::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 4) % 2) * 520 + (((((0::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 2) % 2) * 1040 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((4::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32 + ((((0::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((0::Int32 ÷ 128) % 2) * 128) % 2) * 2080) + 0) + 0x01] =
                F_dish4_time0
            F_shared[(((((((0::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 16) % 2) * 130 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((6::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((0::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 8) % 2) * 260 + (((((0::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 32) % 2) * 65 + (((((0::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((0::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 4) % 2) * 520 + (((((0::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 2) % 2) * 1040 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((6::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32 + ((((0::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((0::Int32 ÷ 128) % 2) * 128) % 2) * 2080) + 0) + 0x01] =
                F_dish6_time0
            F_shared[(((((((16::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((16::Int32 ÷ 128) % 2) * 128) ÷ 16) % 2) * 130 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((0::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((16::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((16::Int32 ÷ 128) % 2) * 128) ÷ 8) % 2) * 260 + (((((16::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((16::Int32 ÷ 128) % 2) * 128) ÷ 32) % 2) * 65 + (((((16::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((16::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((16::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((16::Int32 ÷ 128) % 2) * 128) ÷ 4) % 2) * 520 + (((((16::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((16::Int32 ÷ 128) % 2) * 128) ÷ 2) % 2) * 1040 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((0::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32 + ((((16::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((16::Int32 ÷ 128) % 2) * 128) % 2) * 2080) + 0) + 0x01] =
                F_dish0_time16
            F_shared[(((((((16::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((16::Int32 ÷ 128) % 2) * 128) ÷ 16) % 2) * 130 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((2::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((16::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((16::Int32 ÷ 128) % 2) * 128) ÷ 8) % 2) * 260 + (((((16::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((16::Int32 ÷ 128) % 2) * 128) ÷ 32) % 2) * 65 + (((((16::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((16::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((16::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((16::Int32 ÷ 128) % 2) * 128) ÷ 4) % 2) * 520 + (((((16::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((16::Int32 ÷ 128) % 2) * 128) ÷ 2) % 2) * 1040 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((2::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32 + ((((16::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((16::Int32 ÷ 128) % 2) * 128) % 2) * 2080) + 0) + 0x01] =
                F_dish2_time16
            F_shared[(((((((16::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((16::Int32 ÷ 128) % 2) * 128) ÷ 16) % 2) * 130 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((4::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((16::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((16::Int32 ÷ 128) % 2) * 128) ÷ 8) % 2) * 260 + (((((16::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((16::Int32 ÷ 128) % 2) * 128) ÷ 32) % 2) * 65 + (((((16::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((16::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((16::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((16::Int32 ÷ 128) % 2) * 128) ÷ 4) % 2) * 520 + (((((16::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((16::Int32 ÷ 128) % 2) * 128) ÷ 2) % 2) * 1040 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((4::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32 + ((((16::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((16::Int32 ÷ 128) % 2) * 128) % 2) * 2080) + 0) + 0x01] =
                F_dish4_time16
            F_shared[(((((((16::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((16::Int32 ÷ 128) % 2) * 128) ÷ 16) % 2) * 130 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((6::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((16::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((16::Int32 ÷ 128) % 2) * 128) ÷ 8) % 2) * 260 + (((((16::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((16::Int32 ÷ 128) % 2) * 128) ÷ 32) % 2) * 65 + (((((16::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((16::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((16::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((16::Int32 ÷ 128) % 2) * 128) ÷ 4) % 2) * 520 + (((((16::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((16::Int32 ÷ 128) % 2) * 128) ÷ 2) % 2) * 1040 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((6::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32 + ((((16::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((16::Int32 ÷ 128) % 2) * 128) % 2) * 2080) + 0) + 0x01] =
                F_dish6_time16
            F_shared[(((((((128::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 16) % 2) * 130 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((0::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((128::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 8) % 2) * 260 + (((((128::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 32) % 2) * 65 + (((((128::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((128::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 4) % 2) * 520 + (((((128::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 2) % 2) * 1040 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((0::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32 + ((((128::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((128::Int32 ÷ 128) % 2) * 128) % 2) * 2080) + 0) + 0x01] =
                F_dish0_time128
            F_shared[(((((((128::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 16) % 2) * 130 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((2::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((128::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 8) % 2) * 260 + (((((128::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 32) % 2) * 65 + (((((128::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((128::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 4) % 2) * 520 + (((((128::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 2) % 2) * 1040 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((2::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32 + ((((128::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((128::Int32 ÷ 128) % 2) * 128) % 2) * 2080) + 0) + 0x01] =
                F_dish2_time128
            F_shared[(((((((128::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 16) % 2) * 130 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((4::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((128::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 8) % 2) * 260 + (((((128::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 32) % 2) * 65 + (((((128::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((128::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 4) % 2) * 520 + (((((128::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 2) % 2) * 1040 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((4::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32 + ((((128::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((128::Int32 ÷ 128) % 2) * 128) % 2) * 2080) + 0) + 0x01] =
                F_dish4_time128
            F_shared[(((((((128::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 16) % 2) * 130 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((6::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((128::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 8) % 2) * 260 + (((((128::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 32) % 2) * 65 + (((((128::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((128::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 4) % 2) * 520 + (((((128::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 2) % 2) * 1040 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((6::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32 + ((((128::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((128::Int32 ÷ 128) % 2) * 128) % 2) * 2080) + 0) + 0x01] =
                F_dish6_time128
            F_shared[(((((((144::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((144::Int32 ÷ 128) % 2) * 128) ÷ 16) % 2) * 130 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((0::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((144::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((144::Int32 ÷ 128) % 2) * 128) ÷ 8) % 2) * 260 + (((((144::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((144::Int32 ÷ 128) % 2) * 128) ÷ 32) % 2) * 65 + (((((144::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((144::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((144::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((144::Int32 ÷ 128) % 2) * 128) ÷ 4) % 2) * 520 + (((((144::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((144::Int32 ÷ 128) % 2) * 128) ÷ 2) % 2) * 1040 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((0::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32 + ((((144::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((144::Int32 ÷ 128) % 2) * 128) % 2) * 2080) + 0) + 0x01] =
                F_dish0_time144
            F_shared[(((((((144::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((144::Int32 ÷ 128) % 2) * 128) ÷ 16) % 2) * 130 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((2::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((144::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((144::Int32 ÷ 128) % 2) * 128) ÷ 8) % 2) * 260 + (((((144::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((144::Int32 ÷ 128) % 2) * 128) ÷ 32) % 2) * 65 + (((((144::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((144::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((144::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((144::Int32 ÷ 128) % 2) * 128) ÷ 4) % 2) * 520 + (((((144::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((144::Int32 ÷ 128) % 2) * 128) ÷ 2) % 2) * 1040 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((2::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32 + ((((144::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((144::Int32 ÷ 128) % 2) * 128) % 2) * 2080) + 0) + 0x01] =
                F_dish2_time144
            F_shared[(((((((144::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((144::Int32 ÷ 128) % 2) * 128) ÷ 16) % 2) * 130 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((4::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((144::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((144::Int32 ÷ 128) % 2) * 128) ÷ 8) % 2) * 260 + (((((144::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((144::Int32 ÷ 128) % 2) * 128) ÷ 32) % 2) * 65 + (((((144::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((144::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((144::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((144::Int32 ÷ 128) % 2) * 128) ÷ 4) % 2) * 520 + (((((144::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((144::Int32 ÷ 128) % 2) * 128) ÷ 2) % 2) * 1040 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((4::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32 + ((((144::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((144::Int32 ÷ 128) % 2) * 128) % 2) * 2080) + 0) + 0x01] =
                F_dish4_time144
            F_shared[(((((((144::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((144::Int32 ÷ 128) % 2) * 128) ÷ 16) % 2) * 130 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((6::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((144::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((144::Int32 ÷ 128) % 2) * 128) ÷ 8) % 2) * 260 + (((((144::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((144::Int32 ÷ 128) % 2) * 128) ÷ 32) % 2) * 65 + (((((144::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((144::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((144::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((144::Int32 ÷ 128) % 2) * 128) ÷ 4) % 2) * 520 + (((((144::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((144::Int32 ÷ 128) % 2) * 128) ÷ 2) % 2) * 1040 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((6::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32 + ((((144::Int32 ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 32 + ((144::Int32 ÷ 128) % 2) * 128) % 2) * 2080) + 0) + 0x01] =
                F_dish6_time144
            IndexSpaces.cuda_sync_threads()
            for t_inner in 0:128:255
                let polr = 0
                    F_ringbuf_polr_dish0_mtap0_time0 = F_ringbuf_dish0_mtap0_time0
                    F_ringbuf_polr_dish32_mtap0_time0 = F_ringbuf_dish32_mtap0_time0
                    F_ringbuf_polr_dish64_mtap0_time0 = F_ringbuf_dish64_mtap0_time0
                    F_ringbuf_polr_dish96_mtap0_time0 = F_ringbuf_dish96_mtap0_time0
                    F_ringbuf_polr_dish0_mtap1_time0 = F_ringbuf_dish0_mtap1_time0
                    F_ringbuf_polr_dish32_mtap1_time0 = F_ringbuf_dish32_mtap1_time0
                    F_ringbuf_polr_dish64_mtap1_time0 = F_ringbuf_dish64_mtap1_time0
                    F_ringbuf_polr_dish96_mtap1_time0 = F_ringbuf_dish96_mtap1_time0
                    F_ringbuf_polr_dish0_mtap2_time0 = F_ringbuf_dish0_mtap2_time0
                    F_ringbuf_polr_dish32_mtap2_time0 = F_ringbuf_dish32_mtap2_time0
                    F_ringbuf_polr_dish64_mtap2_time0 = F_ringbuf_dish64_mtap2_time0
                    F_ringbuf_polr_dish96_mtap2_time0 = F_ringbuf_dish96_mtap2_time0
                    F_ringbuf_polr_dish0_mtap0_time1 = F_ringbuf_dish0_mtap0_time1
                    F_ringbuf_polr_dish32_mtap0_time1 = F_ringbuf_dish32_mtap0_time1
                    F_ringbuf_polr_dish64_mtap0_time1 = F_ringbuf_dish64_mtap0_time1
                    F_ringbuf_polr_dish96_mtap0_time1 = F_ringbuf_dish96_mtap0_time1
                    F_ringbuf_polr_dish0_mtap1_time1 = F_ringbuf_dish0_mtap1_time1
                    F_ringbuf_polr_dish32_mtap1_time1 = F_ringbuf_dish32_mtap1_time1
                    F_ringbuf_polr_dish64_mtap1_time1 = F_ringbuf_dish64_mtap1_time1
                    F_ringbuf_polr_dish96_mtap1_time1 = F_ringbuf_dish96_mtap1_time1
                    F_ringbuf_polr_dish0_mtap2_time1 = F_ringbuf_dish0_mtap2_time1
                    F_ringbuf_polr_dish32_mtap2_time1 = F_ringbuf_dish32_mtap2_time1
                    F_ringbuf_polr_dish64_mtap2_time1 = F_ringbuf_dish64_mtap2_time1
                    F_ringbuf_polr_dish96_mtap2_time1 = F_ringbuf_dish96_mtap2_time1
                    let dish = 0
                        F_ringbuf_polr_dish_mtap0_time0 = F_ringbuf_polr_dish0_mtap0_time0
                        F_ringbuf_polr_dish_mtap1_time0 = F_ringbuf_polr_dish0_mtap1_time0
                        F_ringbuf_polr_dish_mtap2_time0 = F_ringbuf_polr_dish0_mtap2_time0
                        F_ringbuf_polr_dish_mtap0_time1 = F_ringbuf_polr_dish0_mtap0_time1
                        F_ringbuf_polr_dish_mtap1_time1 = F_ringbuf_polr_dish0_mtap1_time1
                        F_ringbuf_polr_dish_mtap2_time1 = F_ringbuf_polr_dish0_mtap2_time1
                        F_in_time0 = F_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 0::Int32 % 2) ÷ 16) % 2) * 130 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 0::Int32 % 2) ÷ 8) % 2) * 260 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 0::Int32 % 2) ÷ 32) % 2) * 65 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 0::Int32 % 2) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 0::Int32 % 2) ÷ 4) % 2) * 520 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 0::Int32 % 2) ÷ 2) % 2) * 1040 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 2) % 2) * 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 0::Int32 % 2) % 2) * 2080) + 0x01]
                        F_in_time1 = F_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 1::Int32 % 2) ÷ 16) % 2) * 130 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 1::Int32 % 2) ÷ 8) % 2) * 260 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 1::Int32 % 2) ÷ 32) % 2) * 65 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 1::Int32 % 2) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 1::Int32 % 2) ÷ 4) % 2) * 520 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 1::Int32 % 2) ÷ 2) % 2) * 1040 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 2) % 2) * 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 1::Int32 % 2) % 2) * 2080) + 0x01]
                        (E_cplx0_dish0_time0, E_cplx1_dish0_time0, E_cplx0_dish1_time0, E_cplx1_dish1_time0) = convert_swapped_withoffset(
                            NTuple{4,Float16x2}, F_in_time0
                        )
                        (E_cplx0_dish0_time1, E_cplx1_dish0_time1, E_cplx0_dish1_time1, E_cplx1_dish1_time1) = convert_swapped_withoffset(
                            NTuple{4,Float16x2}, F_in_time1
                        )
                        E2_cplx0_dish0_time0 = zero(E_cplx0_dish0_time0)
                        E2_cplx1_dish0_time0 = zero(E_cplx1_dish0_time0)
                        E2_cplx0_dish1_time0 = zero(E_cplx0_dish1_time0)
                        E2_cplx1_dish1_time0 = zero(E_cplx1_dish1_time0)
                        E2_cplx0_dish0_time1 = zero(E_cplx0_dish0_time1)
                        E2_cplx1_dish0_time1 = zero(E_cplx1_dish0_time1)
                        E2_cplx0_dish1_time1 = zero(E_cplx0_dish1_time1)
                        E2_cplx1_dish1_time1 = zero(E_cplx1_dish1_time1)
                        let mtap = 0
                            W_mtap_time0 = Wpfb_mtap0_time0
                            W_mtap_time1 = Wpfb_mtap0_time1
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap_time0 = F_ringbuf_polr_dish_mtap0_time0
                                F_ringbuf_polr_dish_mtap_time1 = F_ringbuf_polr_dish_mtap0_time1
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time0, E_ringbuf_polr_dish_mtap_cplx1_dish0_time0, E_ringbuf_polr_dish_mtap_cplx0_dish1_time0, E_ringbuf_polr_dish_mtap_cplx1_dish1_time0) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time0
                                )
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time1, E_ringbuf_polr_dish_mtap_cplx1_dish0_time1, E_ringbuf_polr_dish_mtap_cplx0_dish1_time1, E_ringbuf_polr_dish_mtap_cplx1_dish1_time1) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time1
                                )
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time0,
                                    E2_cplx0_dish0_time0,
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time0,
                                    E2_cplx1_dish0_time0,
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time0,
                                    E2_cplx0_dish1_time0,
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time0,
                                    E2_cplx1_dish1_time0,
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time1,
                                    E2_cplx0_dish0_time1,
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time1,
                                    E2_cplx1_dish0_time1,
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time1,
                                    E2_cplx0_dish1_time1,
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time1,
                                    E2_cplx1_dish1_time1,
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish0_time0, E2_cplx0_dish0_time0
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish0_time0, E2_cplx1_dish0_time0
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish1_time0, E2_cplx0_dish1_time0
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish1_time0, E2_cplx1_dish1_time0
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish0_time1, E2_cplx0_dish0_time1
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish0_time1, E2_cplx1_dish0_time1
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish1_time1, E2_cplx0_dish1_time1
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish1_time1, E2_cplx1_dish1_time1
                                )
                            end
                        end
                        let mtap = 1
                            W_mtap_time0 = Wpfb_mtap1_time0
                            W_mtap_time1 = Wpfb_mtap1_time1
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap_time0 = F_ringbuf_polr_dish_mtap1_time0
                                F_ringbuf_polr_dish_mtap_time1 = F_ringbuf_polr_dish_mtap1_time1
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time0, E_ringbuf_polr_dish_mtap_cplx1_dish0_time0, E_ringbuf_polr_dish_mtap_cplx0_dish1_time0, E_ringbuf_polr_dish_mtap_cplx1_dish1_time0) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time0
                                )
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time1, E_ringbuf_polr_dish_mtap_cplx1_dish0_time1, E_ringbuf_polr_dish_mtap_cplx0_dish1_time1, E_ringbuf_polr_dish_mtap_cplx1_dish1_time1) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time1
                                )
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time0,
                                    E2_cplx0_dish0_time0,
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time0,
                                    E2_cplx1_dish0_time0,
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time0,
                                    E2_cplx0_dish1_time0,
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time0,
                                    E2_cplx1_dish1_time0,
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time1,
                                    E2_cplx0_dish0_time1,
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time1,
                                    E2_cplx1_dish0_time1,
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time1,
                                    E2_cplx0_dish1_time1,
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time1,
                                    E2_cplx1_dish1_time1,
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish0_time0, E2_cplx0_dish0_time0
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish0_time0, E2_cplx1_dish0_time0
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish1_time0, E2_cplx0_dish1_time0
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish1_time0, E2_cplx1_dish1_time0
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish0_time1, E2_cplx0_dish0_time1
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish0_time1, E2_cplx1_dish0_time1
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish1_time1, E2_cplx0_dish1_time1
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish1_time1, E2_cplx1_dish1_time1
                                )
                            end
                        end
                        let mtap = 2
                            W_mtap_time0 = Wpfb_mtap2_time0
                            W_mtap_time1 = Wpfb_mtap2_time1
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap_time0 = F_ringbuf_polr_dish_mtap2_time0
                                F_ringbuf_polr_dish_mtap_time1 = F_ringbuf_polr_dish_mtap2_time1
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time0, E_ringbuf_polr_dish_mtap_cplx1_dish0_time0, E_ringbuf_polr_dish_mtap_cplx0_dish1_time0, E_ringbuf_polr_dish_mtap_cplx1_dish1_time0) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time0
                                )
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time1, E_ringbuf_polr_dish_mtap_cplx1_dish0_time1, E_ringbuf_polr_dish_mtap_cplx0_dish1_time1, E_ringbuf_polr_dish_mtap_cplx1_dish1_time1) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time1
                                )
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time0,
                                    E2_cplx0_dish0_time0,
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time0,
                                    E2_cplx1_dish0_time0,
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time0,
                                    E2_cplx0_dish1_time0,
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time0,
                                    E2_cplx1_dish1_time0,
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time1,
                                    E2_cplx0_dish0_time1,
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time1,
                                    E2_cplx1_dish0_time1,
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time1,
                                    E2_cplx0_dish1_time1,
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time1,
                                    E2_cplx1_dish1_time1,
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish0_time0, E2_cplx0_dish0_time0
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish0_time0, E2_cplx1_dish0_time0
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish1_time0, E2_cplx0_dish1_time0
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish1_time0, E2_cplx1_dish1_time0
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish0_time1, E2_cplx0_dish0_time1
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish0_time1, E2_cplx1_dish0_time1
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish1_time1, E2_cplx0_dish1_time1
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish1_time1, E2_cplx1_dish1_time1
                                )
                            end
                        end
                        let mtap = 3
                            W_mtap_time0 = Wpfb_mtap3_time0
                            W_mtap_time1 = Wpfb_mtap3_time1
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap_time0 = F_ringbuf_polr_dish_mtap3_time0
                                F_ringbuf_polr_dish_mtap_time1 = F_ringbuf_polr_dish_mtap3_time1
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time0, E_ringbuf_polr_dish_mtap_cplx1_dish0_time0, E_ringbuf_polr_dish_mtap_cplx0_dish1_time0, E_ringbuf_polr_dish_mtap_cplx1_dish1_time0) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time0
                                )
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time1, E_ringbuf_polr_dish_mtap_cplx1_dish0_time1, E_ringbuf_polr_dish_mtap_cplx0_dish1_time1, E_ringbuf_polr_dish_mtap_cplx1_dish1_time1) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time1
                                )
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time0,
                                    E2_cplx0_dish0_time0,
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time0,
                                    E2_cplx1_dish0_time0,
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time0,
                                    E2_cplx0_dish1_time0,
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time0,
                                    E2_cplx1_dish1_time0,
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time1,
                                    E2_cplx0_dish0_time1,
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time1,
                                    E2_cplx1_dish0_time1,
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time1,
                                    E2_cplx0_dish1_time1,
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time1,
                                    E2_cplx1_dish1_time1,
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish0_time0, E2_cplx0_dish0_time0
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish0_time0, E2_cplx1_dish0_time0
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish1_time0, E2_cplx0_dish1_time0
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish1_time0, E2_cplx1_dish1_time0
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish0_time1, E2_cplx0_dish0_time1
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish0_time1, E2_cplx1_dish0_time1
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish1_time1, E2_cplx0_dish1_time1
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish1_time1, E2_cplx1_dish1_time1
                                )
                            end
                        end
                        E2re_dish0_time0 = E2_cplx0_dish0_time0
                        E2im_dish0_time0 = E2_cplx1_dish0_time0
                        E2re_dish1_time0 = E2_cplx0_dish1_time0
                        E2im_dish1_time0 = E2_cplx1_dish1_time0
                        E2re_dish0_time1 = E2_cplx0_dish0_time1
                        E2im_dish0_time1 = E2_cplx1_dish0_time1
                        E2re_dish1_time1 = E2_cplx0_dish1_time1
                        E2im_dish1_time1 = E2_cplx1_dish1_time1
                        Xre_time0 = X_cplx0_time0
                        Xim_time0 = X_cplx1_time0
                        Xre_time1 = X_cplx0_time1
                        Xim_time1 = X_cplx1_time1
                        E3re_dish0_time0 = muladd(Xre_time0, E2re_dish0_time0, -Xim_time0 * E2im_dish0_time0)
                        E3re_dish1_time0 = muladd(Xre_time0, E2re_dish1_time0, -Xim_time0 * E2im_dish1_time0)
                        E3re_dish0_time1 = muladd(Xre_time1, E2re_dish0_time1, -Xim_time1 * E2im_dish0_time1)
                        E3re_dish1_time1 = muladd(Xre_time1, E2re_dish1_time1, -Xim_time1 * E2im_dish1_time1)
                        E3im_dish0_time0 = muladd(Xre_time0, E2im_dish0_time0, Xim_time0 * E2re_dish0_time0)
                        E3im_dish1_time0 = muladd(Xre_time0, E2im_dish1_time0, Xim_time0 * E2re_dish1_time0)
                        E3im_dish0_time1 = muladd(Xre_time1, E2im_dish0_time1, Xim_time1 * E2re_dish0_time1)
                        E3im_dish1_time1 = muladd(Xre_time1, E2im_dish1_time1, Xim_time1 * E2re_dish1_time1)
                        E3_cplx0_dish0_time0 = E3re_dish0_time0
                        E3_cplx1_dish0_time0 = E3im_dish0_time0
                        E3_cplx0_dish1_time0 = E3re_dish1_time0
                        E3_cplx1_dish1_time0 = E3im_dish1_time0
                        E3_cplx0_dish0_time1 = E3re_dish0_time1
                        E3_cplx1_dish0_time1 = E3im_dish0_time1
                        E3_cplx0_dish1_time1 = E3re_dish1_time1
                        E3_cplx1_dish1_time1 = E3im_dish1_time1
                        XX_cplx0_dish0_time0 = E3_cplx0_dish0_time0
                        XX_cplx1_dish0_time0 = E3_cplx1_dish0_time0
                        XX_cplx0_dish1_time0 = E3_cplx0_dish1_time0
                        XX_cplx1_dish1_time0 = E3_cplx1_dish1_time0
                        XX_cplx0_dish0_time1 = E3_cplx0_dish0_time1
                        XX_cplx1_dish0_time1 = E3_cplx1_dish0_time1
                        XX_cplx0_dish1_time1 = E3_cplx0_dish1_time1
                        XX_cplx1_dish1_time1 = E3_cplx1_dish1_time1
                        XXre_dish0_time0 = XX_cplx0_dish0_time0
                        XXim_dish0_time0 = XX_cplx1_dish0_time0
                        XXre_dish1_time0 = XX_cplx0_dish1_time0
                        XXim_dish1_time0 = XX_cplx1_dish1_time0
                        XXre_dish0_time1 = XX_cplx0_dish0_time1
                        XXim_dish0_time1 = XX_cplx1_dish0_time1
                        XXre_dish1_time1 = XX_cplx0_dish1_time1
                        XXim_dish1_time1 = XX_cplx1_dish1_time1
                        XX_cplx_in0_dish0_time0 = XXre_dish0_time0
                        XX_cplx_in1_dish0_time0 = XXim_dish0_time0
                        XX_cplx_in0_dish1_time0 = XXre_dish1_time0
                        XX_cplx_in1_dish1_time0 = XXim_dish1_time0
                        XX_cplx_in0_dish0_time1 = XXre_dish0_time1
                        XX_cplx_in1_dish0_time1 = XXim_dish0_time1
                        XX_cplx_in0_dish1_time1 = XXre_dish1_time1
                        XX_cplx_in1_dish1_time1 = XXim_dish1_time1
                        WW_cplx0_dish0_time0 = zero(Float16x2)
                        WW_cplx1_dish0_time0 = zero(Float16x2)
                        WW_cplx0_dish1_time0 = zero(Float16x2)
                        WW_cplx1_dish1_time0 = zero(Float16x2)
                        WW_cplx0_dish0_time1 = zero(Float16x2)
                        WW_cplx1_dish0_time1 = zero(Float16x2)
                        WW_cplx0_dish1_time1 = zero(Float16x2)
                        WW_cplx1_dish1_time1 = zero(Float16x2)
                        (WW_cplx0_dish0_time0, WW_cplx1_dish0_time0) = IndexSpaces.mma_m16n8k16(
                            (Γ¹_cplx0_cplx_in0_time0, Γ¹_cplx1_cplx_in0_time0, Γ¹_cplx0_cplx_in1_time0, Γ¹_cplx1_cplx_in1_time0),
                            (XX_cplx_in0_dish0_time0, XX_cplx_in1_dish0_time0),
                            (WW_cplx0_dish0_time0, WW_cplx1_dish0_time0),
                        )
                        (WW_cplx0_dish1_time0, WW_cplx1_dish1_time0) = IndexSpaces.mma_m16n8k16(
                            (Γ¹_cplx0_cplx_in0_time0, Γ¹_cplx1_cplx_in0_time0, Γ¹_cplx0_cplx_in1_time0, Γ¹_cplx1_cplx_in1_time0),
                            (XX_cplx_in0_dish1_time0, XX_cplx_in1_dish1_time0),
                            (WW_cplx0_dish1_time0, WW_cplx1_dish1_time0),
                        )
                        (WW_cplx0_dish0_time1, WW_cplx1_dish0_time1) = IndexSpaces.mma_m16n8k16(
                            (Γ¹_cplx0_cplx_in0_time1, Γ¹_cplx1_cplx_in0_time1, Γ¹_cplx0_cplx_in1_time1, Γ¹_cplx1_cplx_in1_time1),
                            (XX_cplx_in0_dish0_time1, XX_cplx_in1_dish0_time1),
                            (WW_cplx0_dish0_time1, WW_cplx1_dish0_time1),
                        )
                        (WW_cplx0_dish1_time1, WW_cplx1_dish1_time1) = IndexSpaces.mma_m16n8k16(
                            (Γ¹_cplx0_cplx_in0_time1, Γ¹_cplx1_cplx_in0_time1, Γ¹_cplx0_cplx_in1_time1, Γ¹_cplx1_cplx_in1_time1),
                            (XX_cplx_in0_dish1_time1, XX_cplx_in1_dish1_time1),
                            (WW_cplx0_dish1_time1, WW_cplx1_dish1_time1),
                        )
                        Γ²re_time0 = Γ²_cplx0_time0
                        Γ²im_time0 = Γ²_cplx1_time0
                        Γ²re_time1 = Γ²_cplx0_time1
                        Γ²im_time1 = Γ²_cplx1_time1
                        WWre_dish0_time0 = WW_cplx0_dish0_time0
                        WWim_dish0_time0 = WW_cplx1_dish0_time0
                        WWre_dish1_time0 = WW_cplx0_dish1_time0
                        WWim_dish1_time0 = WW_cplx1_dish1_time0
                        WWre_dish0_time1 = WW_cplx0_dish0_time1
                        WWim_dish0_time1 = WW_cplx1_dish0_time1
                        WWre_dish1_time1 = WW_cplx0_dish1_time1
                        WWim_dish1_time1 = WW_cplx1_dish1_time1
                        ZZre_dish0_time0 = muladd(Γ²re_time0, WWre_dish0_time0, -Γ²im_time0 * WWim_dish0_time0)
                        ZZre_dish1_time0 = muladd(Γ²re_time0, WWre_dish1_time0, -Γ²im_time0 * WWim_dish1_time0)
                        ZZre_dish0_time1 = muladd(Γ²re_time1, WWre_dish0_time1, -Γ²im_time1 * WWim_dish0_time1)
                        ZZre_dish1_time1 = muladd(Γ²re_time1, WWre_dish1_time1, -Γ²im_time1 * WWim_dish1_time1)
                        ZZim_dish0_time0 = muladd(Γ²re_time0, WWim_dish0_time0, Γ²im_time0 * WWre_dish0_time0)
                        ZZim_dish1_time0 = muladd(Γ²re_time0, WWim_dish1_time0, Γ²im_time0 * WWre_dish1_time0)
                        ZZim_dish0_time1 = muladd(Γ²re_time1, WWim_dish0_time1, Γ²im_time1 * WWre_dish0_time1)
                        ZZim_dish1_time1 = muladd(Γ²re_time1, WWim_dish1_time1, Γ²im_time1 * WWre_dish1_time1)
                        ZZ_cplx0_dish0_time0 = ZZre_dish0_time0
                        ZZ_cplx1_dish0_time0 = ZZim_dish0_time0
                        ZZ_cplx0_dish1_time0 = ZZre_dish1_time0
                        ZZ_cplx1_dish1_time0 = ZZim_dish1_time0
                        ZZ_cplx0_dish0_time1 = ZZre_dish0_time1
                        ZZ_cplx1_dish0_time1 = ZZim_dish0_time1
                        ZZ_cplx0_dish1_time1 = ZZre_dish1_time1
                        ZZ_cplx1_dish1_time1 = ZZim_dish1_time1
                        ZZre_dish0_time0 = ZZ_cplx0_dish0_time0
                        ZZim_dish0_time0 = ZZ_cplx1_dish0_time0
                        ZZre_dish1_time0 = ZZ_cplx0_dish1_time0
                        ZZim_dish1_time0 = ZZ_cplx1_dish1_time0
                        ZZre_dish0_time1 = ZZ_cplx0_dish0_time1
                        ZZim_dish0_time1 = ZZ_cplx1_dish0_time1
                        ZZre_dish1_time1 = ZZ_cplx0_dish1_time1
                        ZZim_dish1_time1 = ZZ_cplx1_dish1_time1
                        ZZ_cplx_in0_dish0_time0 = ZZre_dish0_time0
                        ZZ_cplx_in1_dish0_time0 = ZZim_dish0_time0
                        ZZ_cplx_in0_dish1_time0 = ZZre_dish1_time0
                        ZZ_cplx_in1_dish1_time0 = ZZim_dish1_time0
                        ZZ_cplx_in0_dish0_time1 = ZZre_dish0_time1
                        ZZ_cplx_in1_dish0_time1 = ZZim_dish0_time1
                        ZZ_cplx_in0_dish1_time1 = ZZre_dish1_time1
                        ZZ_cplx_in1_dish1_time1 = ZZim_dish1_time1
                        YY_cplx0_dish0_time0 = zero(Float16x2)
                        YY_cplx1_dish0_time0 = zero(Float16x2)
                        YY_cplx0_dish1_time0 = zero(Float16x2)
                        YY_cplx1_dish1_time0 = zero(Float16x2)
                        YY_cplx0_dish0_time1 = zero(Float16x2)
                        YY_cplx1_dish0_time1 = zero(Float16x2)
                        YY_cplx0_dish1_time1 = zero(Float16x2)
                        YY_cplx1_dish1_time1 = zero(Float16x2)
                        (YY_cplx0_dish0_time0, YY_cplx1_dish0_time0) = IndexSpaces.mma_m16n8k16(
                            (
                                Γ³_cplx0_cplx_in0_dish0_time0,
                                Γ³_cplx1_cplx_in0_dish0_time0,
                                Γ³_cplx0_cplx_in1_dish0_time0,
                                Γ³_cplx1_cplx_in1_dish0_time0,
                            ),
                            (ZZ_cplx_in0_dish0_time0, ZZ_cplx_in1_dish0_time0),
                            (YY_cplx0_dish0_time0, YY_cplx1_dish0_time0),
                        )
                        (YY_cplx0_dish1_time0, YY_cplx1_dish1_time0) = IndexSpaces.mma_m16n8k16(
                            (
                                Γ³_cplx0_cplx_in0_dish1_time0,
                                Γ³_cplx1_cplx_in0_dish1_time0,
                                Γ³_cplx0_cplx_in1_dish1_time0,
                                Γ³_cplx1_cplx_in1_dish1_time0,
                            ),
                            (ZZ_cplx_in0_dish1_time0, ZZ_cplx_in1_dish1_time0),
                            (YY_cplx0_dish1_time0, YY_cplx1_dish1_time0),
                        )
                        (YY_cplx0_dish0_time1, YY_cplx1_dish0_time1) = IndexSpaces.mma_m16n8k16(
                            (
                                Γ³_cplx0_cplx_in0_dish0_time1,
                                Γ³_cplx1_cplx_in0_dish0_time1,
                                Γ³_cplx0_cplx_in1_dish0_time1,
                                Γ³_cplx1_cplx_in1_dish0_time1,
                            ),
                            (ZZ_cplx_in0_dish0_time1, ZZ_cplx_in1_dish0_time1),
                            (YY_cplx0_dish0_time1, YY_cplx1_dish0_time1),
                        )
                        (YY_cplx0_dish1_time1, YY_cplx1_dish1_time1) = IndexSpaces.mma_m16n8k16(
                            (
                                Γ³_cplx0_cplx_in0_dish1_time1,
                                Γ³_cplx1_cplx_in0_dish1_time1,
                                Γ³_cplx0_cplx_in1_dish1_time1,
                                Γ³_cplx1_cplx_in1_dish1_time1,
                            ),
                            (ZZ_cplx_in0_dish1_time1, ZZ_cplx_in1_dish1_time1),
                            (YY_cplx0_dish1_time1, YY_cplx1_dish1_time1),
                        )
                        WWW_cplx0_dish0_time0 = YY_cplx0_dish0_time0
                        WWW_cplx1_dish0_time0 = YY_cplx1_dish0_time0
                        WWW_cplx0_dish1_time0 = YY_cplx0_dish1_time0
                        WWW_cplx1_dish1_time0 = YY_cplx1_dish1_time0
                        WWW_cplx0_dish0_time1 = YY_cplx0_dish0_time1
                        WWW_cplx1_dish0_time1 = YY_cplx1_dish0_time1
                        WWW_cplx0_dish1_time1 = YY_cplx0_dish1_time1
                        WWW_cplx1_dish1_time1 = YY_cplx1_dish1_time1
                        WWW_t0_cplx0_dish0 = WWW_cplx0_dish0_time0
                        WWW_t1_cplx0_dish0 = WWW_cplx0_dish0_time1
                        WWW_t0_cplx1_dish0 = WWW_cplx1_dish0_time0
                        WWW_t1_cplx1_dish0 = WWW_cplx1_dish0_time1
                        WWW_t0_cplx0_dish1 = WWW_cplx0_dish1_time0
                        WWW_t1_cplx0_dish1 = WWW_cplx0_dish1_time1
                        WWW_t0_cplx1_dish1 = WWW_cplx1_dish1_time0
                        WWW_t1_cplx1_dish1 = WWW_cplx1_dish1_time1
                        Γ⁴re = Γ⁴_cplx0
                        Γ⁴im = Γ⁴_cplx1
                        WWWre_dish0 = WWW_t1_cplx0_dish0
                        WWWim_dish0 = WWW_t1_cplx1_dish0
                        WWWre_dish1 = WWW_t1_cplx0_dish1
                        WWWim_dish1 = WWW_t1_cplx1_dish1
                        ZZZre_dish0 = muladd(Γ⁴re, WWWre_dish0, -Γ⁴im * WWWim_dish0)
                        ZZZre_dish1 = muladd(Γ⁴re, WWWre_dish1, -Γ⁴im * WWWim_dish1)
                        ZZZim_dish0 = muladd(Γ⁴re, WWWim_dish0, Γ⁴im * WWWre_dish0)
                        ZZZim_dish1 = muladd(Γ⁴re, WWWim_dish1, Γ⁴im * WWWre_dish1)
                        ZZZ_t0_cplx0_dish0 = WWW_t0_cplx0_dish0
                        ZZZ_t0_cplx1_dish0 = WWW_t0_cplx1_dish0
                        ZZZ_t0_cplx0_dish1 = WWW_t0_cplx0_dish1
                        ZZZ_t0_cplx1_dish1 = WWW_t0_cplx1_dish1
                        ZZZ_t1_cplx0_dish0 = ZZZre_dish0
                        ZZZ_t1_cplx1_dish0 = ZZZim_dish0
                        ZZZ_t1_cplx0_dish1 = ZZZre_dish1
                        ZZZ_t1_cplx1_dish1 = ZZZim_dish1
                        YYY_u0_cplx0_dish0 = WWW_t0_cplx0_dish0 + WWW_t1_cplx0_dish0
                        YYY_u0_cplx1_dish0 = WWW_t0_cplx1_dish0 + WWW_t1_cplx1_dish0
                        YYY_u0_cplx0_dish1 = WWW_t0_cplx0_dish1 + WWW_t1_cplx0_dish1
                        YYY_u0_cplx1_dish1 = WWW_t0_cplx1_dish1 + WWW_t1_cplx1_dish1
                        YYY_u1_cplx0_dish0 = WWW_t0_cplx0_dish0 - WWW_t1_cplx0_dish0
                        YYY_u1_cplx1_dish0 = WWW_t0_cplx1_dish0 - WWW_t1_cplx1_dish0
                        YYY_u1_cplx0_dish1 = WWW_t0_cplx0_dish1 - WWW_t1_cplx0_dish1
                        YYY_u1_cplx1_dish1 = WWW_t0_cplx1_dish1 - WWW_t1_cplx1_dish1
                        YYY_cplx0_dish0_freq0 = YYY_u0_cplx0_dish0
                        YYY_cplx0_dish0_freq64 = YYY_u1_cplx0_dish0
                        YYY_cplx1_dish0_freq0 = YYY_u0_cplx1_dish0
                        YYY_cplx1_dish0_freq64 = YYY_u1_cplx1_dish0
                        YYY_cplx0_dish1_freq0 = YYY_u0_cplx0_dish1
                        YYY_cplx0_dish1_freq64 = YYY_u1_cplx0_dish1
                        YYY_cplx1_dish1_freq0 = YYY_u0_cplx1_dish1
                        YYY_cplx1_dish1_freq64 = YYY_u1_cplx1_dish1
                        E4_cplx0_dish0_freq0 = YYY_cplx0_dish0_freq0
                        E4_cplx1_dish0_freq0 = YYY_cplx1_dish0_freq0
                        E4_cplx0_dish1_freq0 = YYY_cplx0_dish1_freq0
                        E4_cplx1_dish1_freq0 = YYY_cplx1_dish1_freq0
                        E4_cplx0_dish0_freq64 = YYY_cplx0_dish0_freq64
                        E4_cplx1_dish0_freq64 = YYY_cplx1_dish0_freq64
                        E4_cplx0_dish1_freq64 = YYY_cplx0_dish1_freq64
                        E4_cplx1_dish1_freq64 = YYY_cplx1_dish1_freq64
                        E5_cplx0_dish0_freq0 = Gains_freq0 * E4_cplx0_dish0_freq0
                        E5_cplx1_dish0_freq0 = Gains_freq0 * E4_cplx1_dish0_freq0
                        E5_cplx0_dish1_freq0 = Gains_freq0 * E4_cplx0_dish1_freq0
                        E5_cplx1_dish1_freq0 = Gains_freq0 * E4_cplx1_dish1_freq0
                        E5_cplx0_dish0_freq64 = Gains_freq64 * E4_cplx0_dish0_freq64
                        E5_cplx1_dish0_freq64 = Gains_freq64 * E4_cplx1_dish0_freq64
                        E5_cplx0_dish1_freq64 = Gains_freq64 * E4_cplx0_dish1_freq64
                        E5_cplx1_dish1_freq64 = Gains_freq64 * E4_cplx1_dish1_freq64
                        E5_cplx0_dish0_freq0 = clamp(E5_cplx0_dish0_freq0, Float16x2(-7, -7), Float16x2(7, 7))
                        E5_cplx1_dish0_freq0 = clamp(E5_cplx1_dish0_freq0, Float16x2(-7, -7), Float16x2(7, 7))
                        E5_cplx0_dish1_freq0 = clamp(E5_cplx0_dish1_freq0, Float16x2(-7, -7), Float16x2(7, 7))
                        E5_cplx1_dish1_freq0 = clamp(E5_cplx1_dish1_freq0, Float16x2(-7, -7), Float16x2(7, 7))
                        E5_cplx0_dish0_freq64 = clamp(E5_cplx0_dish0_freq64, Float16x2(-7, -7), Float16x2(7, 7))
                        E5_cplx1_dish0_freq64 = clamp(E5_cplx1_dish0_freq64, Float16x2(-7, -7), Float16x2(7, 7))
                        E5_cplx0_dish1_freq64 = clamp(E5_cplx0_dish1_freq64, Float16x2(-7, -7), Float16x2(7, 7))
                        E5_cplx1_dish1_freq64 = clamp(E5_cplx1_dish1_freq64, Float16x2(-7, -7), Float16x2(7, 7))
                        F̄_out_freq0 = convert_swapped_withoffset(
                            Int4x8, (E5_cplx0_dish0_freq0, E5_cplx1_dish0_freq0, E5_cplx0_dish1_freq0, E5_cplx1_dish1_freq0)
                        )
                        F̄_out_freq64 = convert_swapped_withoffset(
                            Int4x8, (E5_cplx0_dish0_freq64, E5_cplx1_dish0_freq64, E5_cplx0_dish1_freq64, E5_cplx1_dish1_freq64)
                        )
                        F̄_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 2) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 2) % 2) * 2 + ((0::Int32 ÷ 64) % 2) * 64) ÷ 2) % 64) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 2) % 2) * 32) + 0) + 0x01] =
                            F̄_out_freq0
                        F̄_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 2) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 2) % 2) * 2 + ((64::Int32 ÷ 64) % 2) * 64) ÷ 2) % 64) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 2) % 2) * 32) + 0) + 0x01] =
                            F̄_out_freq64
                        F_ringbuf_polr_dish_m0_time0 = F_ringbuf_polr_dish_mtap0_time0
                        F_ringbuf_polr_dish_m1_time0 = F_ringbuf_polr_dish_mtap1_time0
                        F_ringbuf_polr_dish_m2_time0 = F_ringbuf_polr_dish_mtap2_time0
                        F_ringbuf_polr_dish_m0_time1 = F_ringbuf_polr_dish_mtap0_time1
                        F_ringbuf_polr_dish_m1_time1 = F_ringbuf_polr_dish_mtap1_time1
                        F_ringbuf_polr_dish_m2_time1 = F_ringbuf_polr_dish_mtap2_time1
                        F_ringbuf_polr_dish_m0_time0 = F_ringbuf_polr_dish_m1_time0
                        F_ringbuf_polr_dish_m0_time1 = F_ringbuf_polr_dish_m1_time1
                        F_ringbuf_polr_dish_m1_time0 = F_ringbuf_polr_dish_m2_time0
                        F_ringbuf_polr_dish_m1_time1 = F_ringbuf_polr_dish_m2_time1
                        F_ringbuf_polr_dish_m2_time0 = F_in_time0
                        F_ringbuf_polr_dish_m2_time1 = F_in_time1
                        F_ringbuf_polr_dish_mtap0_time0 = F_ringbuf_polr_dish_m0_time0
                        F_ringbuf_polr_dish_mtap1_time0 = F_ringbuf_polr_dish_m1_time0
                        F_ringbuf_polr_dish_mtap2_time0 = F_ringbuf_polr_dish_m2_time0
                        F_ringbuf_polr_dish_mtap0_time1 = F_ringbuf_polr_dish_m0_time1
                        F_ringbuf_polr_dish_mtap1_time1 = F_ringbuf_polr_dish_m1_time1
                        F_ringbuf_polr_dish_mtap2_time1 = F_ringbuf_polr_dish_m2_time1
                        F_ringbuf_polr_dish0_mtap0_time0 = F_ringbuf_polr_dish_mtap0_time0
                        F_ringbuf_polr_dish0_mtap1_time0 = F_ringbuf_polr_dish_mtap1_time0
                        F_ringbuf_polr_dish0_mtap2_time0 = F_ringbuf_polr_dish_mtap2_time0
                        F_ringbuf_polr_dish0_mtap0_time1 = F_ringbuf_polr_dish_mtap0_time1
                        F_ringbuf_polr_dish0_mtap1_time1 = F_ringbuf_polr_dish_mtap1_time1
                        F_ringbuf_polr_dish0_mtap2_time1 = F_ringbuf_polr_dish_mtap2_time1
                    end
                    let dish = 32
                        F_ringbuf_polr_dish_mtap0_time0 = F_ringbuf_polr_dish32_mtap0_time0
                        F_ringbuf_polr_dish_mtap1_time0 = F_ringbuf_polr_dish32_mtap1_time0
                        F_ringbuf_polr_dish_mtap2_time0 = F_ringbuf_polr_dish32_mtap2_time0
                        F_ringbuf_polr_dish_mtap0_time1 = F_ringbuf_polr_dish32_mtap0_time1
                        F_ringbuf_polr_dish_mtap1_time1 = F_ringbuf_polr_dish32_mtap1_time1
                        F_ringbuf_polr_dish_mtap2_time1 = F_ringbuf_polr_dish32_mtap2_time1
                        F_in_time0 = F_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 0::Int32 % 2) ÷ 16) % 2) * 130 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 0::Int32 % 2) ÷ 8) % 2) * 260 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 0::Int32 % 2) ÷ 32) % 2) * 65 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 0::Int32 % 2) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 0::Int32 % 2) ÷ 4) % 2) * 520 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 0::Int32 % 2) ÷ 2) % 2) * 1040 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 2) % 2) * 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 0::Int32 % 2) % 2) * 2080) + 0x01]
                        F_in_time1 = F_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 1::Int32 % 2) ÷ 16) % 2) * 130 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 1::Int32 % 2) ÷ 8) % 2) * 260 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 1::Int32 % 2) ÷ 32) % 2) * 65 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 1::Int32 % 2) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 1::Int32 % 2) ÷ 4) % 2) * 520 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 1::Int32 % 2) ÷ 2) % 2) * 1040 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 2) % 2) * 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 1::Int32 % 2) % 2) * 2080) + 0x01]
                        (E_cplx0_dish0_time0, E_cplx1_dish0_time0, E_cplx0_dish1_time0, E_cplx1_dish1_time0) = convert_swapped_withoffset(
                            NTuple{4,Float16x2}, F_in_time0
                        )
                        (E_cplx0_dish0_time1, E_cplx1_dish0_time1, E_cplx0_dish1_time1, E_cplx1_dish1_time1) = convert_swapped_withoffset(
                            NTuple{4,Float16x2}, F_in_time1
                        )
                        E2_cplx0_dish0_time0 = zero(E_cplx0_dish0_time0)
                        E2_cplx1_dish0_time0 = zero(E_cplx1_dish0_time0)
                        E2_cplx0_dish1_time0 = zero(E_cplx0_dish1_time0)
                        E2_cplx1_dish1_time0 = zero(E_cplx1_dish1_time0)
                        E2_cplx0_dish0_time1 = zero(E_cplx0_dish0_time1)
                        E2_cplx1_dish0_time1 = zero(E_cplx1_dish0_time1)
                        E2_cplx0_dish1_time1 = zero(E_cplx0_dish1_time1)
                        E2_cplx1_dish1_time1 = zero(E_cplx1_dish1_time1)
                        let mtap = 0
                            W_mtap_time0 = Wpfb_mtap0_time0
                            W_mtap_time1 = Wpfb_mtap0_time1
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap_time0 = F_ringbuf_polr_dish_mtap0_time0
                                F_ringbuf_polr_dish_mtap_time1 = F_ringbuf_polr_dish_mtap0_time1
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time0, E_ringbuf_polr_dish_mtap_cplx1_dish0_time0, E_ringbuf_polr_dish_mtap_cplx0_dish1_time0, E_ringbuf_polr_dish_mtap_cplx1_dish1_time0) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time0
                                )
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time1, E_ringbuf_polr_dish_mtap_cplx1_dish0_time1, E_ringbuf_polr_dish_mtap_cplx0_dish1_time1, E_ringbuf_polr_dish_mtap_cplx1_dish1_time1) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time1
                                )
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time0,
                                    E2_cplx0_dish0_time0,
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time0,
                                    E2_cplx1_dish0_time0,
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time0,
                                    E2_cplx0_dish1_time0,
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time0,
                                    E2_cplx1_dish1_time0,
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time1,
                                    E2_cplx0_dish0_time1,
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time1,
                                    E2_cplx1_dish0_time1,
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time1,
                                    E2_cplx0_dish1_time1,
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time1,
                                    E2_cplx1_dish1_time1,
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish0_time0, E2_cplx0_dish0_time0
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish0_time0, E2_cplx1_dish0_time0
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish1_time0, E2_cplx0_dish1_time0
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish1_time0, E2_cplx1_dish1_time0
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish0_time1, E2_cplx0_dish0_time1
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish0_time1, E2_cplx1_dish0_time1
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish1_time1, E2_cplx0_dish1_time1
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish1_time1, E2_cplx1_dish1_time1
                                )
                            end
                        end
                        let mtap = 1
                            W_mtap_time0 = Wpfb_mtap1_time0
                            W_mtap_time1 = Wpfb_mtap1_time1
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap_time0 = F_ringbuf_polr_dish_mtap1_time0
                                F_ringbuf_polr_dish_mtap_time1 = F_ringbuf_polr_dish_mtap1_time1
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time0, E_ringbuf_polr_dish_mtap_cplx1_dish0_time0, E_ringbuf_polr_dish_mtap_cplx0_dish1_time0, E_ringbuf_polr_dish_mtap_cplx1_dish1_time0) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time0
                                )
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time1, E_ringbuf_polr_dish_mtap_cplx1_dish0_time1, E_ringbuf_polr_dish_mtap_cplx0_dish1_time1, E_ringbuf_polr_dish_mtap_cplx1_dish1_time1) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time1
                                )
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time0,
                                    E2_cplx0_dish0_time0,
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time0,
                                    E2_cplx1_dish0_time0,
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time0,
                                    E2_cplx0_dish1_time0,
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time0,
                                    E2_cplx1_dish1_time0,
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time1,
                                    E2_cplx0_dish0_time1,
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time1,
                                    E2_cplx1_dish0_time1,
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time1,
                                    E2_cplx0_dish1_time1,
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time1,
                                    E2_cplx1_dish1_time1,
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish0_time0, E2_cplx0_dish0_time0
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish0_time0, E2_cplx1_dish0_time0
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish1_time0, E2_cplx0_dish1_time0
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish1_time0, E2_cplx1_dish1_time0
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish0_time1, E2_cplx0_dish0_time1
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish0_time1, E2_cplx1_dish0_time1
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish1_time1, E2_cplx0_dish1_time1
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish1_time1, E2_cplx1_dish1_time1
                                )
                            end
                        end
                        let mtap = 2
                            W_mtap_time0 = Wpfb_mtap2_time0
                            W_mtap_time1 = Wpfb_mtap2_time1
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap_time0 = F_ringbuf_polr_dish_mtap2_time0
                                F_ringbuf_polr_dish_mtap_time1 = F_ringbuf_polr_dish_mtap2_time1
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time0, E_ringbuf_polr_dish_mtap_cplx1_dish0_time0, E_ringbuf_polr_dish_mtap_cplx0_dish1_time0, E_ringbuf_polr_dish_mtap_cplx1_dish1_time0) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time0
                                )
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time1, E_ringbuf_polr_dish_mtap_cplx1_dish0_time1, E_ringbuf_polr_dish_mtap_cplx0_dish1_time1, E_ringbuf_polr_dish_mtap_cplx1_dish1_time1) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time1
                                )
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time0,
                                    E2_cplx0_dish0_time0,
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time0,
                                    E2_cplx1_dish0_time0,
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time0,
                                    E2_cplx0_dish1_time0,
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time0,
                                    E2_cplx1_dish1_time0,
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time1,
                                    E2_cplx0_dish0_time1,
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time1,
                                    E2_cplx1_dish0_time1,
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time1,
                                    E2_cplx0_dish1_time1,
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time1,
                                    E2_cplx1_dish1_time1,
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish0_time0, E2_cplx0_dish0_time0
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish0_time0, E2_cplx1_dish0_time0
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish1_time0, E2_cplx0_dish1_time0
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish1_time0, E2_cplx1_dish1_time0
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish0_time1, E2_cplx0_dish0_time1
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish0_time1, E2_cplx1_dish0_time1
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish1_time1, E2_cplx0_dish1_time1
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish1_time1, E2_cplx1_dish1_time1
                                )
                            end
                        end
                        let mtap = 3
                            W_mtap_time0 = Wpfb_mtap3_time0
                            W_mtap_time1 = Wpfb_mtap3_time1
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap_time0 = F_ringbuf_polr_dish_mtap3_time0
                                F_ringbuf_polr_dish_mtap_time1 = F_ringbuf_polr_dish_mtap3_time1
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time0, E_ringbuf_polr_dish_mtap_cplx1_dish0_time0, E_ringbuf_polr_dish_mtap_cplx0_dish1_time0, E_ringbuf_polr_dish_mtap_cplx1_dish1_time0) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time0
                                )
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time1, E_ringbuf_polr_dish_mtap_cplx1_dish0_time1, E_ringbuf_polr_dish_mtap_cplx0_dish1_time1, E_ringbuf_polr_dish_mtap_cplx1_dish1_time1) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time1
                                )
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time0,
                                    E2_cplx0_dish0_time0,
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time0,
                                    E2_cplx1_dish0_time0,
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time0,
                                    E2_cplx0_dish1_time0,
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time0,
                                    E2_cplx1_dish1_time0,
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time1,
                                    E2_cplx0_dish0_time1,
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time1,
                                    E2_cplx1_dish0_time1,
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time1,
                                    E2_cplx0_dish1_time1,
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time1,
                                    E2_cplx1_dish1_time1,
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish0_time0, E2_cplx0_dish0_time0
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish0_time0, E2_cplx1_dish0_time0
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish1_time0, E2_cplx0_dish1_time0
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish1_time0, E2_cplx1_dish1_time0
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish0_time1, E2_cplx0_dish0_time1
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish0_time1, E2_cplx1_dish0_time1
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish1_time1, E2_cplx0_dish1_time1
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish1_time1, E2_cplx1_dish1_time1
                                )
                            end
                        end
                        E2re_dish0_time0 = E2_cplx0_dish0_time0
                        E2im_dish0_time0 = E2_cplx1_dish0_time0
                        E2re_dish1_time0 = E2_cplx0_dish1_time0
                        E2im_dish1_time0 = E2_cplx1_dish1_time0
                        E2re_dish0_time1 = E2_cplx0_dish0_time1
                        E2im_dish0_time1 = E2_cplx1_dish0_time1
                        E2re_dish1_time1 = E2_cplx0_dish1_time1
                        E2im_dish1_time1 = E2_cplx1_dish1_time1
                        Xre_time0 = X_cplx0_time0
                        Xim_time0 = X_cplx1_time0
                        Xre_time1 = X_cplx0_time1
                        Xim_time1 = X_cplx1_time1
                        E3re_dish0_time0 = muladd(Xre_time0, E2re_dish0_time0, -Xim_time0 * E2im_dish0_time0)
                        E3re_dish1_time0 = muladd(Xre_time0, E2re_dish1_time0, -Xim_time0 * E2im_dish1_time0)
                        E3re_dish0_time1 = muladd(Xre_time1, E2re_dish0_time1, -Xim_time1 * E2im_dish0_time1)
                        E3re_dish1_time1 = muladd(Xre_time1, E2re_dish1_time1, -Xim_time1 * E2im_dish1_time1)
                        E3im_dish0_time0 = muladd(Xre_time0, E2im_dish0_time0, Xim_time0 * E2re_dish0_time0)
                        E3im_dish1_time0 = muladd(Xre_time0, E2im_dish1_time0, Xim_time0 * E2re_dish1_time0)
                        E3im_dish0_time1 = muladd(Xre_time1, E2im_dish0_time1, Xim_time1 * E2re_dish0_time1)
                        E3im_dish1_time1 = muladd(Xre_time1, E2im_dish1_time1, Xim_time1 * E2re_dish1_time1)
                        E3_cplx0_dish0_time0 = E3re_dish0_time0
                        E3_cplx1_dish0_time0 = E3im_dish0_time0
                        E3_cplx0_dish1_time0 = E3re_dish1_time0
                        E3_cplx1_dish1_time0 = E3im_dish1_time0
                        E3_cplx0_dish0_time1 = E3re_dish0_time1
                        E3_cplx1_dish0_time1 = E3im_dish0_time1
                        E3_cplx0_dish1_time1 = E3re_dish1_time1
                        E3_cplx1_dish1_time1 = E3im_dish1_time1
                        XX_cplx0_dish0_time0 = E3_cplx0_dish0_time0
                        XX_cplx1_dish0_time0 = E3_cplx1_dish0_time0
                        XX_cplx0_dish1_time0 = E3_cplx0_dish1_time0
                        XX_cplx1_dish1_time0 = E3_cplx1_dish1_time0
                        XX_cplx0_dish0_time1 = E3_cplx0_dish0_time1
                        XX_cplx1_dish0_time1 = E3_cplx1_dish0_time1
                        XX_cplx0_dish1_time1 = E3_cplx0_dish1_time1
                        XX_cplx1_dish1_time1 = E3_cplx1_dish1_time1
                        XXre_dish0_time0 = XX_cplx0_dish0_time0
                        XXim_dish0_time0 = XX_cplx1_dish0_time0
                        XXre_dish1_time0 = XX_cplx0_dish1_time0
                        XXim_dish1_time0 = XX_cplx1_dish1_time0
                        XXre_dish0_time1 = XX_cplx0_dish0_time1
                        XXim_dish0_time1 = XX_cplx1_dish0_time1
                        XXre_dish1_time1 = XX_cplx0_dish1_time1
                        XXim_dish1_time1 = XX_cplx1_dish1_time1
                        XX_cplx_in0_dish0_time0 = XXre_dish0_time0
                        XX_cplx_in1_dish0_time0 = XXim_dish0_time0
                        XX_cplx_in0_dish1_time0 = XXre_dish1_time0
                        XX_cplx_in1_dish1_time0 = XXim_dish1_time0
                        XX_cplx_in0_dish0_time1 = XXre_dish0_time1
                        XX_cplx_in1_dish0_time1 = XXim_dish0_time1
                        XX_cplx_in0_dish1_time1 = XXre_dish1_time1
                        XX_cplx_in1_dish1_time1 = XXim_dish1_time1
                        WW_cplx0_dish0_time0 = zero(Float16x2)
                        WW_cplx1_dish0_time0 = zero(Float16x2)
                        WW_cplx0_dish1_time0 = zero(Float16x2)
                        WW_cplx1_dish1_time0 = zero(Float16x2)
                        WW_cplx0_dish0_time1 = zero(Float16x2)
                        WW_cplx1_dish0_time1 = zero(Float16x2)
                        WW_cplx0_dish1_time1 = zero(Float16x2)
                        WW_cplx1_dish1_time1 = zero(Float16x2)
                        (WW_cplx0_dish0_time0, WW_cplx1_dish0_time0) = IndexSpaces.mma_m16n8k16(
                            (Γ¹_cplx0_cplx_in0_time0, Γ¹_cplx1_cplx_in0_time0, Γ¹_cplx0_cplx_in1_time0, Γ¹_cplx1_cplx_in1_time0),
                            (XX_cplx_in0_dish0_time0, XX_cplx_in1_dish0_time0),
                            (WW_cplx0_dish0_time0, WW_cplx1_dish0_time0),
                        )
                        (WW_cplx0_dish1_time0, WW_cplx1_dish1_time0) = IndexSpaces.mma_m16n8k16(
                            (Γ¹_cplx0_cplx_in0_time0, Γ¹_cplx1_cplx_in0_time0, Γ¹_cplx0_cplx_in1_time0, Γ¹_cplx1_cplx_in1_time0),
                            (XX_cplx_in0_dish1_time0, XX_cplx_in1_dish1_time0),
                            (WW_cplx0_dish1_time0, WW_cplx1_dish1_time0),
                        )
                        (WW_cplx0_dish0_time1, WW_cplx1_dish0_time1) = IndexSpaces.mma_m16n8k16(
                            (Γ¹_cplx0_cplx_in0_time1, Γ¹_cplx1_cplx_in0_time1, Γ¹_cplx0_cplx_in1_time1, Γ¹_cplx1_cplx_in1_time1),
                            (XX_cplx_in0_dish0_time1, XX_cplx_in1_dish0_time1),
                            (WW_cplx0_dish0_time1, WW_cplx1_dish0_time1),
                        )
                        (WW_cplx0_dish1_time1, WW_cplx1_dish1_time1) = IndexSpaces.mma_m16n8k16(
                            (Γ¹_cplx0_cplx_in0_time1, Γ¹_cplx1_cplx_in0_time1, Γ¹_cplx0_cplx_in1_time1, Γ¹_cplx1_cplx_in1_time1),
                            (XX_cplx_in0_dish1_time1, XX_cplx_in1_dish1_time1),
                            (WW_cplx0_dish1_time1, WW_cplx1_dish1_time1),
                        )
                        Γ²re_time0 = Γ²_cplx0_time0
                        Γ²im_time0 = Γ²_cplx1_time0
                        Γ²re_time1 = Γ²_cplx0_time1
                        Γ²im_time1 = Γ²_cplx1_time1
                        WWre_dish0_time0 = WW_cplx0_dish0_time0
                        WWim_dish0_time0 = WW_cplx1_dish0_time0
                        WWre_dish1_time0 = WW_cplx0_dish1_time0
                        WWim_dish1_time0 = WW_cplx1_dish1_time0
                        WWre_dish0_time1 = WW_cplx0_dish0_time1
                        WWim_dish0_time1 = WW_cplx1_dish0_time1
                        WWre_dish1_time1 = WW_cplx0_dish1_time1
                        WWim_dish1_time1 = WW_cplx1_dish1_time1
                        ZZre_dish0_time0 = muladd(Γ²re_time0, WWre_dish0_time0, -Γ²im_time0 * WWim_dish0_time0)
                        ZZre_dish1_time0 = muladd(Γ²re_time0, WWre_dish1_time0, -Γ²im_time0 * WWim_dish1_time0)
                        ZZre_dish0_time1 = muladd(Γ²re_time1, WWre_dish0_time1, -Γ²im_time1 * WWim_dish0_time1)
                        ZZre_dish1_time1 = muladd(Γ²re_time1, WWre_dish1_time1, -Γ²im_time1 * WWim_dish1_time1)
                        ZZim_dish0_time0 = muladd(Γ²re_time0, WWim_dish0_time0, Γ²im_time0 * WWre_dish0_time0)
                        ZZim_dish1_time0 = muladd(Γ²re_time0, WWim_dish1_time0, Γ²im_time0 * WWre_dish1_time0)
                        ZZim_dish0_time1 = muladd(Γ²re_time1, WWim_dish0_time1, Γ²im_time1 * WWre_dish0_time1)
                        ZZim_dish1_time1 = muladd(Γ²re_time1, WWim_dish1_time1, Γ²im_time1 * WWre_dish1_time1)
                        ZZ_cplx0_dish0_time0 = ZZre_dish0_time0
                        ZZ_cplx1_dish0_time0 = ZZim_dish0_time0
                        ZZ_cplx0_dish1_time0 = ZZre_dish1_time0
                        ZZ_cplx1_dish1_time0 = ZZim_dish1_time0
                        ZZ_cplx0_dish0_time1 = ZZre_dish0_time1
                        ZZ_cplx1_dish0_time1 = ZZim_dish0_time1
                        ZZ_cplx0_dish1_time1 = ZZre_dish1_time1
                        ZZ_cplx1_dish1_time1 = ZZim_dish1_time1
                        ZZre_dish0_time0 = ZZ_cplx0_dish0_time0
                        ZZim_dish0_time0 = ZZ_cplx1_dish0_time0
                        ZZre_dish1_time0 = ZZ_cplx0_dish1_time0
                        ZZim_dish1_time0 = ZZ_cplx1_dish1_time0
                        ZZre_dish0_time1 = ZZ_cplx0_dish0_time1
                        ZZim_dish0_time1 = ZZ_cplx1_dish0_time1
                        ZZre_dish1_time1 = ZZ_cplx0_dish1_time1
                        ZZim_dish1_time1 = ZZ_cplx1_dish1_time1
                        ZZ_cplx_in0_dish0_time0 = ZZre_dish0_time0
                        ZZ_cplx_in1_dish0_time0 = ZZim_dish0_time0
                        ZZ_cplx_in0_dish1_time0 = ZZre_dish1_time0
                        ZZ_cplx_in1_dish1_time0 = ZZim_dish1_time0
                        ZZ_cplx_in0_dish0_time1 = ZZre_dish0_time1
                        ZZ_cplx_in1_dish0_time1 = ZZim_dish0_time1
                        ZZ_cplx_in0_dish1_time1 = ZZre_dish1_time1
                        ZZ_cplx_in1_dish1_time1 = ZZim_dish1_time1
                        YY_cplx0_dish0_time0 = zero(Float16x2)
                        YY_cplx1_dish0_time0 = zero(Float16x2)
                        YY_cplx0_dish1_time0 = zero(Float16x2)
                        YY_cplx1_dish1_time0 = zero(Float16x2)
                        YY_cplx0_dish0_time1 = zero(Float16x2)
                        YY_cplx1_dish0_time1 = zero(Float16x2)
                        YY_cplx0_dish1_time1 = zero(Float16x2)
                        YY_cplx1_dish1_time1 = zero(Float16x2)
                        (YY_cplx0_dish0_time0, YY_cplx1_dish0_time0) = IndexSpaces.mma_m16n8k16(
                            (
                                Γ³_cplx0_cplx_in0_dish0_time0,
                                Γ³_cplx1_cplx_in0_dish0_time0,
                                Γ³_cplx0_cplx_in1_dish0_time0,
                                Γ³_cplx1_cplx_in1_dish0_time0,
                            ),
                            (ZZ_cplx_in0_dish0_time0, ZZ_cplx_in1_dish0_time0),
                            (YY_cplx0_dish0_time0, YY_cplx1_dish0_time0),
                        )
                        (YY_cplx0_dish1_time0, YY_cplx1_dish1_time0) = IndexSpaces.mma_m16n8k16(
                            (
                                Γ³_cplx0_cplx_in0_dish1_time0,
                                Γ³_cplx1_cplx_in0_dish1_time0,
                                Γ³_cplx0_cplx_in1_dish1_time0,
                                Γ³_cplx1_cplx_in1_dish1_time0,
                            ),
                            (ZZ_cplx_in0_dish1_time0, ZZ_cplx_in1_dish1_time0),
                            (YY_cplx0_dish1_time0, YY_cplx1_dish1_time0),
                        )
                        (YY_cplx0_dish0_time1, YY_cplx1_dish0_time1) = IndexSpaces.mma_m16n8k16(
                            (
                                Γ³_cplx0_cplx_in0_dish0_time1,
                                Γ³_cplx1_cplx_in0_dish0_time1,
                                Γ³_cplx0_cplx_in1_dish0_time1,
                                Γ³_cplx1_cplx_in1_dish0_time1,
                            ),
                            (ZZ_cplx_in0_dish0_time1, ZZ_cplx_in1_dish0_time1),
                            (YY_cplx0_dish0_time1, YY_cplx1_dish0_time1),
                        )
                        (YY_cplx0_dish1_time1, YY_cplx1_dish1_time1) = IndexSpaces.mma_m16n8k16(
                            (
                                Γ³_cplx0_cplx_in0_dish1_time1,
                                Γ³_cplx1_cplx_in0_dish1_time1,
                                Γ³_cplx0_cplx_in1_dish1_time1,
                                Γ³_cplx1_cplx_in1_dish1_time1,
                            ),
                            (ZZ_cplx_in0_dish1_time1, ZZ_cplx_in1_dish1_time1),
                            (YY_cplx0_dish1_time1, YY_cplx1_dish1_time1),
                        )
                        WWW_cplx0_dish0_time0 = YY_cplx0_dish0_time0
                        WWW_cplx1_dish0_time0 = YY_cplx1_dish0_time0
                        WWW_cplx0_dish1_time0 = YY_cplx0_dish1_time0
                        WWW_cplx1_dish1_time0 = YY_cplx1_dish1_time0
                        WWW_cplx0_dish0_time1 = YY_cplx0_dish0_time1
                        WWW_cplx1_dish0_time1 = YY_cplx1_dish0_time1
                        WWW_cplx0_dish1_time1 = YY_cplx0_dish1_time1
                        WWW_cplx1_dish1_time1 = YY_cplx1_dish1_time1
                        WWW_t0_cplx0_dish0 = WWW_cplx0_dish0_time0
                        WWW_t1_cplx0_dish0 = WWW_cplx0_dish0_time1
                        WWW_t0_cplx1_dish0 = WWW_cplx1_dish0_time0
                        WWW_t1_cplx1_dish0 = WWW_cplx1_dish0_time1
                        WWW_t0_cplx0_dish1 = WWW_cplx0_dish1_time0
                        WWW_t1_cplx0_dish1 = WWW_cplx0_dish1_time1
                        WWW_t0_cplx1_dish1 = WWW_cplx1_dish1_time0
                        WWW_t1_cplx1_dish1 = WWW_cplx1_dish1_time1
                        Γ⁴re = Γ⁴_cplx0
                        Γ⁴im = Γ⁴_cplx1
                        WWWre_dish0 = WWW_t1_cplx0_dish0
                        WWWim_dish0 = WWW_t1_cplx1_dish0
                        WWWre_dish1 = WWW_t1_cplx0_dish1
                        WWWim_dish1 = WWW_t1_cplx1_dish1
                        ZZZre_dish0 = muladd(Γ⁴re, WWWre_dish0, -Γ⁴im * WWWim_dish0)
                        ZZZre_dish1 = muladd(Γ⁴re, WWWre_dish1, -Γ⁴im * WWWim_dish1)
                        ZZZim_dish0 = muladd(Γ⁴re, WWWim_dish0, Γ⁴im * WWWre_dish0)
                        ZZZim_dish1 = muladd(Γ⁴re, WWWim_dish1, Γ⁴im * WWWre_dish1)
                        ZZZ_t0_cplx0_dish0 = WWW_t0_cplx0_dish0
                        ZZZ_t0_cplx1_dish0 = WWW_t0_cplx1_dish0
                        ZZZ_t0_cplx0_dish1 = WWW_t0_cplx0_dish1
                        ZZZ_t0_cplx1_dish1 = WWW_t0_cplx1_dish1
                        ZZZ_t1_cplx0_dish0 = ZZZre_dish0
                        ZZZ_t1_cplx1_dish0 = ZZZim_dish0
                        ZZZ_t1_cplx0_dish1 = ZZZre_dish1
                        ZZZ_t1_cplx1_dish1 = ZZZim_dish1
                        YYY_u0_cplx0_dish0 = WWW_t0_cplx0_dish0 + WWW_t1_cplx0_dish0
                        YYY_u0_cplx1_dish0 = WWW_t0_cplx1_dish0 + WWW_t1_cplx1_dish0
                        YYY_u0_cplx0_dish1 = WWW_t0_cplx0_dish1 + WWW_t1_cplx0_dish1
                        YYY_u0_cplx1_dish1 = WWW_t0_cplx1_dish1 + WWW_t1_cplx1_dish1
                        YYY_u1_cplx0_dish0 = WWW_t0_cplx0_dish0 - WWW_t1_cplx0_dish0
                        YYY_u1_cplx1_dish0 = WWW_t0_cplx1_dish0 - WWW_t1_cplx1_dish0
                        YYY_u1_cplx0_dish1 = WWW_t0_cplx0_dish1 - WWW_t1_cplx0_dish1
                        YYY_u1_cplx1_dish1 = WWW_t0_cplx1_dish1 - WWW_t1_cplx1_dish1
                        YYY_cplx0_dish0_freq0 = YYY_u0_cplx0_dish0
                        YYY_cplx0_dish0_freq64 = YYY_u1_cplx0_dish0
                        YYY_cplx1_dish0_freq0 = YYY_u0_cplx1_dish0
                        YYY_cplx1_dish0_freq64 = YYY_u1_cplx1_dish0
                        YYY_cplx0_dish1_freq0 = YYY_u0_cplx0_dish1
                        YYY_cplx0_dish1_freq64 = YYY_u1_cplx0_dish1
                        YYY_cplx1_dish1_freq0 = YYY_u0_cplx1_dish1
                        YYY_cplx1_dish1_freq64 = YYY_u1_cplx1_dish1
                        E4_cplx0_dish0_freq0 = YYY_cplx0_dish0_freq0
                        E4_cplx1_dish0_freq0 = YYY_cplx1_dish0_freq0
                        E4_cplx0_dish1_freq0 = YYY_cplx0_dish1_freq0
                        E4_cplx1_dish1_freq0 = YYY_cplx1_dish1_freq0
                        E4_cplx0_dish0_freq64 = YYY_cplx0_dish0_freq64
                        E4_cplx1_dish0_freq64 = YYY_cplx1_dish0_freq64
                        E4_cplx0_dish1_freq64 = YYY_cplx0_dish1_freq64
                        E4_cplx1_dish1_freq64 = YYY_cplx1_dish1_freq64
                        E5_cplx0_dish0_freq0 = Gains_freq0 * E4_cplx0_dish0_freq0
                        E5_cplx1_dish0_freq0 = Gains_freq0 * E4_cplx1_dish0_freq0
                        E5_cplx0_dish1_freq0 = Gains_freq0 * E4_cplx0_dish1_freq0
                        E5_cplx1_dish1_freq0 = Gains_freq0 * E4_cplx1_dish1_freq0
                        E5_cplx0_dish0_freq64 = Gains_freq64 * E4_cplx0_dish0_freq64
                        E5_cplx1_dish0_freq64 = Gains_freq64 * E4_cplx1_dish0_freq64
                        E5_cplx0_dish1_freq64 = Gains_freq64 * E4_cplx0_dish1_freq64
                        E5_cplx1_dish1_freq64 = Gains_freq64 * E4_cplx1_dish1_freq64
                        E5_cplx0_dish0_freq0 = clamp(E5_cplx0_dish0_freq0, Float16x2(-7, -7), Float16x2(7, 7))
                        E5_cplx1_dish0_freq0 = clamp(E5_cplx1_dish0_freq0, Float16x2(-7, -7), Float16x2(7, 7))
                        E5_cplx0_dish1_freq0 = clamp(E5_cplx0_dish1_freq0, Float16x2(-7, -7), Float16x2(7, 7))
                        E5_cplx1_dish1_freq0 = clamp(E5_cplx1_dish1_freq0, Float16x2(-7, -7), Float16x2(7, 7))
                        E5_cplx0_dish0_freq64 = clamp(E5_cplx0_dish0_freq64, Float16x2(-7, -7), Float16x2(7, 7))
                        E5_cplx1_dish0_freq64 = clamp(E5_cplx1_dish0_freq64, Float16x2(-7, -7), Float16x2(7, 7))
                        E5_cplx0_dish1_freq64 = clamp(E5_cplx0_dish1_freq64, Float16x2(-7, -7), Float16x2(7, 7))
                        E5_cplx1_dish1_freq64 = clamp(E5_cplx1_dish1_freq64, Float16x2(-7, -7), Float16x2(7, 7))
                        F̄_out_freq0 = convert_swapped_withoffset(
                            Int4x8, (E5_cplx0_dish0_freq0, E5_cplx1_dish0_freq0, E5_cplx0_dish1_freq0, E5_cplx1_dish1_freq0)
                        )
                        F̄_out_freq64 = convert_swapped_withoffset(
                            Int4x8, (E5_cplx0_dish0_freq64, E5_cplx1_dish0_freq64, E5_cplx0_dish1_freq64, E5_cplx1_dish1_freq64)
                        )
                        F̄_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 2) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 2) % 2) * 2 + ((0::Int32 ÷ 64) % 2) * 64) ÷ 2) % 64) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 2) % 2) * 32) + 0) + 0x01] =
                            F̄_out_freq0
                        F̄_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 2) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 2) % 2) * 2 + ((64::Int32 ÷ 64) % 2) * 64) ÷ 2) % 64) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 2) % 2) * 32) + 0) + 0x01] =
                            F̄_out_freq64
                        F_ringbuf_polr_dish_m0_time0 = F_ringbuf_polr_dish_mtap0_time0
                        F_ringbuf_polr_dish_m1_time0 = F_ringbuf_polr_dish_mtap1_time0
                        F_ringbuf_polr_dish_m2_time0 = F_ringbuf_polr_dish_mtap2_time0
                        F_ringbuf_polr_dish_m0_time1 = F_ringbuf_polr_dish_mtap0_time1
                        F_ringbuf_polr_dish_m1_time1 = F_ringbuf_polr_dish_mtap1_time1
                        F_ringbuf_polr_dish_m2_time1 = F_ringbuf_polr_dish_mtap2_time1
                        F_ringbuf_polr_dish_m0_time0 = F_ringbuf_polr_dish_m1_time0
                        F_ringbuf_polr_dish_m0_time1 = F_ringbuf_polr_dish_m1_time1
                        F_ringbuf_polr_dish_m1_time0 = F_ringbuf_polr_dish_m2_time0
                        F_ringbuf_polr_dish_m1_time1 = F_ringbuf_polr_dish_m2_time1
                        F_ringbuf_polr_dish_m2_time0 = F_in_time0
                        F_ringbuf_polr_dish_m2_time1 = F_in_time1
                        F_ringbuf_polr_dish_mtap0_time0 = F_ringbuf_polr_dish_m0_time0
                        F_ringbuf_polr_dish_mtap1_time0 = F_ringbuf_polr_dish_m1_time0
                        F_ringbuf_polr_dish_mtap2_time0 = F_ringbuf_polr_dish_m2_time0
                        F_ringbuf_polr_dish_mtap0_time1 = F_ringbuf_polr_dish_m0_time1
                        F_ringbuf_polr_dish_mtap1_time1 = F_ringbuf_polr_dish_m1_time1
                        F_ringbuf_polr_dish_mtap2_time1 = F_ringbuf_polr_dish_m2_time1
                        F_ringbuf_polr_dish32_mtap0_time0 = F_ringbuf_polr_dish_mtap0_time0
                        F_ringbuf_polr_dish32_mtap1_time0 = F_ringbuf_polr_dish_mtap1_time0
                        F_ringbuf_polr_dish32_mtap2_time0 = F_ringbuf_polr_dish_mtap2_time0
                        F_ringbuf_polr_dish32_mtap0_time1 = F_ringbuf_polr_dish_mtap0_time1
                        F_ringbuf_polr_dish32_mtap1_time1 = F_ringbuf_polr_dish_mtap1_time1
                        F_ringbuf_polr_dish32_mtap2_time1 = F_ringbuf_polr_dish_mtap2_time1
                    end
                    let dish = 64
                        F_ringbuf_polr_dish_mtap0_time0 = F_ringbuf_polr_dish64_mtap0_time0
                        F_ringbuf_polr_dish_mtap1_time0 = F_ringbuf_polr_dish64_mtap1_time0
                        F_ringbuf_polr_dish_mtap2_time0 = F_ringbuf_polr_dish64_mtap2_time0
                        F_ringbuf_polr_dish_mtap0_time1 = F_ringbuf_polr_dish64_mtap0_time1
                        F_ringbuf_polr_dish_mtap1_time1 = F_ringbuf_polr_dish64_mtap1_time1
                        F_ringbuf_polr_dish_mtap2_time1 = F_ringbuf_polr_dish64_mtap2_time1
                        F_in_time0 = F_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 0::Int32 % 2) ÷ 16) % 2) * 130 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 0::Int32 % 2) ÷ 8) % 2) * 260 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 0::Int32 % 2) ÷ 32) % 2) * 65 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 0::Int32 % 2) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 0::Int32 % 2) ÷ 4) % 2) * 520 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 0::Int32 % 2) ÷ 2) % 2) * 1040 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 2) % 2) * 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 0::Int32 % 2) % 2) * 2080) + 0x01]
                        F_in_time1 = F_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 1::Int32 % 2) ÷ 16) % 2) * 130 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 1::Int32 % 2) ÷ 8) % 2) * 260 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 1::Int32 % 2) ÷ 32) % 2) * 65 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 1::Int32 % 2) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 1::Int32 % 2) ÷ 4) % 2) * 520 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 1::Int32 % 2) ÷ 2) % 2) * 1040 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 2) % 2) * 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 1::Int32 % 2) % 2) * 2080) + 0x01]
                        (E_cplx0_dish0_time0, E_cplx1_dish0_time0, E_cplx0_dish1_time0, E_cplx1_dish1_time0) = convert_swapped_withoffset(
                            NTuple{4,Float16x2}, F_in_time0
                        )
                        (E_cplx0_dish0_time1, E_cplx1_dish0_time1, E_cplx0_dish1_time1, E_cplx1_dish1_time1) = convert_swapped_withoffset(
                            NTuple{4,Float16x2}, F_in_time1
                        )
                        E2_cplx0_dish0_time0 = zero(E_cplx0_dish0_time0)
                        E2_cplx1_dish0_time0 = zero(E_cplx1_dish0_time0)
                        E2_cplx0_dish1_time0 = zero(E_cplx0_dish1_time0)
                        E2_cplx1_dish1_time0 = zero(E_cplx1_dish1_time0)
                        E2_cplx0_dish0_time1 = zero(E_cplx0_dish0_time1)
                        E2_cplx1_dish0_time1 = zero(E_cplx1_dish0_time1)
                        E2_cplx0_dish1_time1 = zero(E_cplx0_dish1_time1)
                        E2_cplx1_dish1_time1 = zero(E_cplx1_dish1_time1)
                        let mtap = 0
                            W_mtap_time0 = Wpfb_mtap0_time0
                            W_mtap_time1 = Wpfb_mtap0_time1
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap_time0 = F_ringbuf_polr_dish_mtap0_time0
                                F_ringbuf_polr_dish_mtap_time1 = F_ringbuf_polr_dish_mtap0_time1
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time0, E_ringbuf_polr_dish_mtap_cplx1_dish0_time0, E_ringbuf_polr_dish_mtap_cplx0_dish1_time0, E_ringbuf_polr_dish_mtap_cplx1_dish1_time0) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time0
                                )
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time1, E_ringbuf_polr_dish_mtap_cplx1_dish0_time1, E_ringbuf_polr_dish_mtap_cplx0_dish1_time1, E_ringbuf_polr_dish_mtap_cplx1_dish1_time1) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time1
                                )
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time0,
                                    E2_cplx0_dish0_time0,
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time0,
                                    E2_cplx1_dish0_time0,
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time0,
                                    E2_cplx0_dish1_time0,
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time0,
                                    E2_cplx1_dish1_time0,
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time1,
                                    E2_cplx0_dish0_time1,
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time1,
                                    E2_cplx1_dish0_time1,
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time1,
                                    E2_cplx0_dish1_time1,
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time1,
                                    E2_cplx1_dish1_time1,
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish0_time0, E2_cplx0_dish0_time0
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish0_time0, E2_cplx1_dish0_time0
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish1_time0, E2_cplx0_dish1_time0
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish1_time0, E2_cplx1_dish1_time0
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish0_time1, E2_cplx0_dish0_time1
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish0_time1, E2_cplx1_dish0_time1
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish1_time1, E2_cplx0_dish1_time1
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish1_time1, E2_cplx1_dish1_time1
                                )
                            end
                        end
                        let mtap = 1
                            W_mtap_time0 = Wpfb_mtap1_time0
                            W_mtap_time1 = Wpfb_mtap1_time1
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap_time0 = F_ringbuf_polr_dish_mtap1_time0
                                F_ringbuf_polr_dish_mtap_time1 = F_ringbuf_polr_dish_mtap1_time1
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time0, E_ringbuf_polr_dish_mtap_cplx1_dish0_time0, E_ringbuf_polr_dish_mtap_cplx0_dish1_time0, E_ringbuf_polr_dish_mtap_cplx1_dish1_time0) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time0
                                )
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time1, E_ringbuf_polr_dish_mtap_cplx1_dish0_time1, E_ringbuf_polr_dish_mtap_cplx0_dish1_time1, E_ringbuf_polr_dish_mtap_cplx1_dish1_time1) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time1
                                )
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time0,
                                    E2_cplx0_dish0_time0,
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time0,
                                    E2_cplx1_dish0_time0,
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time0,
                                    E2_cplx0_dish1_time0,
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time0,
                                    E2_cplx1_dish1_time0,
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time1,
                                    E2_cplx0_dish0_time1,
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time1,
                                    E2_cplx1_dish0_time1,
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time1,
                                    E2_cplx0_dish1_time1,
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time1,
                                    E2_cplx1_dish1_time1,
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish0_time0, E2_cplx0_dish0_time0
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish0_time0, E2_cplx1_dish0_time0
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish1_time0, E2_cplx0_dish1_time0
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish1_time0, E2_cplx1_dish1_time0
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish0_time1, E2_cplx0_dish0_time1
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish0_time1, E2_cplx1_dish0_time1
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish1_time1, E2_cplx0_dish1_time1
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish1_time1, E2_cplx1_dish1_time1
                                )
                            end
                        end
                        let mtap = 2
                            W_mtap_time0 = Wpfb_mtap2_time0
                            W_mtap_time1 = Wpfb_mtap2_time1
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap_time0 = F_ringbuf_polr_dish_mtap2_time0
                                F_ringbuf_polr_dish_mtap_time1 = F_ringbuf_polr_dish_mtap2_time1
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time0, E_ringbuf_polr_dish_mtap_cplx1_dish0_time0, E_ringbuf_polr_dish_mtap_cplx0_dish1_time0, E_ringbuf_polr_dish_mtap_cplx1_dish1_time0) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time0
                                )
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time1, E_ringbuf_polr_dish_mtap_cplx1_dish0_time1, E_ringbuf_polr_dish_mtap_cplx0_dish1_time1, E_ringbuf_polr_dish_mtap_cplx1_dish1_time1) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time1
                                )
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time0,
                                    E2_cplx0_dish0_time0,
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time0,
                                    E2_cplx1_dish0_time0,
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time0,
                                    E2_cplx0_dish1_time0,
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time0,
                                    E2_cplx1_dish1_time0,
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time1,
                                    E2_cplx0_dish0_time1,
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time1,
                                    E2_cplx1_dish0_time1,
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time1,
                                    E2_cplx0_dish1_time1,
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time1,
                                    E2_cplx1_dish1_time1,
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish0_time0, E2_cplx0_dish0_time0
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish0_time0, E2_cplx1_dish0_time0
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish1_time0, E2_cplx0_dish1_time0
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish1_time0, E2_cplx1_dish1_time0
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish0_time1, E2_cplx0_dish0_time1
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish0_time1, E2_cplx1_dish0_time1
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish1_time1, E2_cplx0_dish1_time1
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish1_time1, E2_cplx1_dish1_time1
                                )
                            end
                        end
                        let mtap = 3
                            W_mtap_time0 = Wpfb_mtap3_time0
                            W_mtap_time1 = Wpfb_mtap3_time1
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap_time0 = F_ringbuf_polr_dish_mtap3_time0
                                F_ringbuf_polr_dish_mtap_time1 = F_ringbuf_polr_dish_mtap3_time1
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time0, E_ringbuf_polr_dish_mtap_cplx1_dish0_time0, E_ringbuf_polr_dish_mtap_cplx0_dish1_time0, E_ringbuf_polr_dish_mtap_cplx1_dish1_time0) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time0
                                )
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time1, E_ringbuf_polr_dish_mtap_cplx1_dish0_time1, E_ringbuf_polr_dish_mtap_cplx0_dish1_time1, E_ringbuf_polr_dish_mtap_cplx1_dish1_time1) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time1
                                )
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time0,
                                    E2_cplx0_dish0_time0,
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time0,
                                    E2_cplx1_dish0_time0,
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time0,
                                    E2_cplx0_dish1_time0,
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time0,
                                    E2_cplx1_dish1_time0,
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time1,
                                    E2_cplx0_dish0_time1,
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time1,
                                    E2_cplx1_dish0_time1,
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time1,
                                    E2_cplx0_dish1_time1,
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time1,
                                    E2_cplx1_dish1_time1,
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish0_time0, E2_cplx0_dish0_time0
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish0_time0, E2_cplx1_dish0_time0
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish1_time0, E2_cplx0_dish1_time0
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish1_time0, E2_cplx1_dish1_time0
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish0_time1, E2_cplx0_dish0_time1
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish0_time1, E2_cplx1_dish0_time1
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish1_time1, E2_cplx0_dish1_time1
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish1_time1, E2_cplx1_dish1_time1
                                )
                            end
                        end
                        E2re_dish0_time0 = E2_cplx0_dish0_time0
                        E2im_dish0_time0 = E2_cplx1_dish0_time0
                        E2re_dish1_time0 = E2_cplx0_dish1_time0
                        E2im_dish1_time0 = E2_cplx1_dish1_time0
                        E2re_dish0_time1 = E2_cplx0_dish0_time1
                        E2im_dish0_time1 = E2_cplx1_dish0_time1
                        E2re_dish1_time1 = E2_cplx0_dish1_time1
                        E2im_dish1_time1 = E2_cplx1_dish1_time1
                        Xre_time0 = X_cplx0_time0
                        Xim_time0 = X_cplx1_time0
                        Xre_time1 = X_cplx0_time1
                        Xim_time1 = X_cplx1_time1
                        E3re_dish0_time0 = muladd(Xre_time0, E2re_dish0_time0, -Xim_time0 * E2im_dish0_time0)
                        E3re_dish1_time0 = muladd(Xre_time0, E2re_dish1_time0, -Xim_time0 * E2im_dish1_time0)
                        E3re_dish0_time1 = muladd(Xre_time1, E2re_dish0_time1, -Xim_time1 * E2im_dish0_time1)
                        E3re_dish1_time1 = muladd(Xre_time1, E2re_dish1_time1, -Xim_time1 * E2im_dish1_time1)
                        E3im_dish0_time0 = muladd(Xre_time0, E2im_dish0_time0, Xim_time0 * E2re_dish0_time0)
                        E3im_dish1_time0 = muladd(Xre_time0, E2im_dish1_time0, Xim_time0 * E2re_dish1_time0)
                        E3im_dish0_time1 = muladd(Xre_time1, E2im_dish0_time1, Xim_time1 * E2re_dish0_time1)
                        E3im_dish1_time1 = muladd(Xre_time1, E2im_dish1_time1, Xim_time1 * E2re_dish1_time1)
                        E3_cplx0_dish0_time0 = E3re_dish0_time0
                        E3_cplx1_dish0_time0 = E3im_dish0_time0
                        E3_cplx0_dish1_time0 = E3re_dish1_time0
                        E3_cplx1_dish1_time0 = E3im_dish1_time0
                        E3_cplx0_dish0_time1 = E3re_dish0_time1
                        E3_cplx1_dish0_time1 = E3im_dish0_time1
                        E3_cplx0_dish1_time1 = E3re_dish1_time1
                        E3_cplx1_dish1_time1 = E3im_dish1_time1
                        XX_cplx0_dish0_time0 = E3_cplx0_dish0_time0
                        XX_cplx1_dish0_time0 = E3_cplx1_dish0_time0
                        XX_cplx0_dish1_time0 = E3_cplx0_dish1_time0
                        XX_cplx1_dish1_time0 = E3_cplx1_dish1_time0
                        XX_cplx0_dish0_time1 = E3_cplx0_dish0_time1
                        XX_cplx1_dish0_time1 = E3_cplx1_dish0_time1
                        XX_cplx0_dish1_time1 = E3_cplx0_dish1_time1
                        XX_cplx1_dish1_time1 = E3_cplx1_dish1_time1
                        XXre_dish0_time0 = XX_cplx0_dish0_time0
                        XXim_dish0_time0 = XX_cplx1_dish0_time0
                        XXre_dish1_time0 = XX_cplx0_dish1_time0
                        XXim_dish1_time0 = XX_cplx1_dish1_time0
                        XXre_dish0_time1 = XX_cplx0_dish0_time1
                        XXim_dish0_time1 = XX_cplx1_dish0_time1
                        XXre_dish1_time1 = XX_cplx0_dish1_time1
                        XXim_dish1_time1 = XX_cplx1_dish1_time1
                        XX_cplx_in0_dish0_time0 = XXre_dish0_time0
                        XX_cplx_in1_dish0_time0 = XXim_dish0_time0
                        XX_cplx_in0_dish1_time0 = XXre_dish1_time0
                        XX_cplx_in1_dish1_time0 = XXim_dish1_time0
                        XX_cplx_in0_dish0_time1 = XXre_dish0_time1
                        XX_cplx_in1_dish0_time1 = XXim_dish0_time1
                        XX_cplx_in0_dish1_time1 = XXre_dish1_time1
                        XX_cplx_in1_dish1_time1 = XXim_dish1_time1
                        WW_cplx0_dish0_time0 = zero(Float16x2)
                        WW_cplx1_dish0_time0 = zero(Float16x2)
                        WW_cplx0_dish1_time0 = zero(Float16x2)
                        WW_cplx1_dish1_time0 = zero(Float16x2)
                        WW_cplx0_dish0_time1 = zero(Float16x2)
                        WW_cplx1_dish0_time1 = zero(Float16x2)
                        WW_cplx0_dish1_time1 = zero(Float16x2)
                        WW_cplx1_dish1_time1 = zero(Float16x2)
                        (WW_cplx0_dish0_time0, WW_cplx1_dish0_time0) = IndexSpaces.mma_m16n8k16(
                            (Γ¹_cplx0_cplx_in0_time0, Γ¹_cplx1_cplx_in0_time0, Γ¹_cplx0_cplx_in1_time0, Γ¹_cplx1_cplx_in1_time0),
                            (XX_cplx_in0_dish0_time0, XX_cplx_in1_dish0_time0),
                            (WW_cplx0_dish0_time0, WW_cplx1_dish0_time0),
                        )
                        (WW_cplx0_dish1_time0, WW_cplx1_dish1_time0) = IndexSpaces.mma_m16n8k16(
                            (Γ¹_cplx0_cplx_in0_time0, Γ¹_cplx1_cplx_in0_time0, Γ¹_cplx0_cplx_in1_time0, Γ¹_cplx1_cplx_in1_time0),
                            (XX_cplx_in0_dish1_time0, XX_cplx_in1_dish1_time0),
                            (WW_cplx0_dish1_time0, WW_cplx1_dish1_time0),
                        )
                        (WW_cplx0_dish0_time1, WW_cplx1_dish0_time1) = IndexSpaces.mma_m16n8k16(
                            (Γ¹_cplx0_cplx_in0_time1, Γ¹_cplx1_cplx_in0_time1, Γ¹_cplx0_cplx_in1_time1, Γ¹_cplx1_cplx_in1_time1),
                            (XX_cplx_in0_dish0_time1, XX_cplx_in1_dish0_time1),
                            (WW_cplx0_dish0_time1, WW_cplx1_dish0_time1),
                        )
                        (WW_cplx0_dish1_time1, WW_cplx1_dish1_time1) = IndexSpaces.mma_m16n8k16(
                            (Γ¹_cplx0_cplx_in0_time1, Γ¹_cplx1_cplx_in0_time1, Γ¹_cplx0_cplx_in1_time1, Γ¹_cplx1_cplx_in1_time1),
                            (XX_cplx_in0_dish1_time1, XX_cplx_in1_dish1_time1),
                            (WW_cplx0_dish1_time1, WW_cplx1_dish1_time1),
                        )
                        Γ²re_time0 = Γ²_cplx0_time0
                        Γ²im_time0 = Γ²_cplx1_time0
                        Γ²re_time1 = Γ²_cplx0_time1
                        Γ²im_time1 = Γ²_cplx1_time1
                        WWre_dish0_time0 = WW_cplx0_dish0_time0
                        WWim_dish0_time0 = WW_cplx1_dish0_time0
                        WWre_dish1_time0 = WW_cplx0_dish1_time0
                        WWim_dish1_time0 = WW_cplx1_dish1_time0
                        WWre_dish0_time1 = WW_cplx0_dish0_time1
                        WWim_dish0_time1 = WW_cplx1_dish0_time1
                        WWre_dish1_time1 = WW_cplx0_dish1_time1
                        WWim_dish1_time1 = WW_cplx1_dish1_time1
                        ZZre_dish0_time0 = muladd(Γ²re_time0, WWre_dish0_time0, -Γ²im_time0 * WWim_dish0_time0)
                        ZZre_dish1_time0 = muladd(Γ²re_time0, WWre_dish1_time0, -Γ²im_time0 * WWim_dish1_time0)
                        ZZre_dish0_time1 = muladd(Γ²re_time1, WWre_dish0_time1, -Γ²im_time1 * WWim_dish0_time1)
                        ZZre_dish1_time1 = muladd(Γ²re_time1, WWre_dish1_time1, -Γ²im_time1 * WWim_dish1_time1)
                        ZZim_dish0_time0 = muladd(Γ²re_time0, WWim_dish0_time0, Γ²im_time0 * WWre_dish0_time0)
                        ZZim_dish1_time0 = muladd(Γ²re_time0, WWim_dish1_time0, Γ²im_time0 * WWre_dish1_time0)
                        ZZim_dish0_time1 = muladd(Γ²re_time1, WWim_dish0_time1, Γ²im_time1 * WWre_dish0_time1)
                        ZZim_dish1_time1 = muladd(Γ²re_time1, WWim_dish1_time1, Γ²im_time1 * WWre_dish1_time1)
                        ZZ_cplx0_dish0_time0 = ZZre_dish0_time0
                        ZZ_cplx1_dish0_time0 = ZZim_dish0_time0
                        ZZ_cplx0_dish1_time0 = ZZre_dish1_time0
                        ZZ_cplx1_dish1_time0 = ZZim_dish1_time0
                        ZZ_cplx0_dish0_time1 = ZZre_dish0_time1
                        ZZ_cplx1_dish0_time1 = ZZim_dish0_time1
                        ZZ_cplx0_dish1_time1 = ZZre_dish1_time1
                        ZZ_cplx1_dish1_time1 = ZZim_dish1_time1
                        ZZre_dish0_time0 = ZZ_cplx0_dish0_time0
                        ZZim_dish0_time0 = ZZ_cplx1_dish0_time0
                        ZZre_dish1_time0 = ZZ_cplx0_dish1_time0
                        ZZim_dish1_time0 = ZZ_cplx1_dish1_time0
                        ZZre_dish0_time1 = ZZ_cplx0_dish0_time1
                        ZZim_dish0_time1 = ZZ_cplx1_dish0_time1
                        ZZre_dish1_time1 = ZZ_cplx0_dish1_time1
                        ZZim_dish1_time1 = ZZ_cplx1_dish1_time1
                        ZZ_cplx_in0_dish0_time0 = ZZre_dish0_time0
                        ZZ_cplx_in1_dish0_time0 = ZZim_dish0_time0
                        ZZ_cplx_in0_dish1_time0 = ZZre_dish1_time0
                        ZZ_cplx_in1_dish1_time0 = ZZim_dish1_time0
                        ZZ_cplx_in0_dish0_time1 = ZZre_dish0_time1
                        ZZ_cplx_in1_dish0_time1 = ZZim_dish0_time1
                        ZZ_cplx_in0_dish1_time1 = ZZre_dish1_time1
                        ZZ_cplx_in1_dish1_time1 = ZZim_dish1_time1
                        YY_cplx0_dish0_time0 = zero(Float16x2)
                        YY_cplx1_dish0_time0 = zero(Float16x2)
                        YY_cplx0_dish1_time0 = zero(Float16x2)
                        YY_cplx1_dish1_time0 = zero(Float16x2)
                        YY_cplx0_dish0_time1 = zero(Float16x2)
                        YY_cplx1_dish0_time1 = zero(Float16x2)
                        YY_cplx0_dish1_time1 = zero(Float16x2)
                        YY_cplx1_dish1_time1 = zero(Float16x2)
                        (YY_cplx0_dish0_time0, YY_cplx1_dish0_time0) = IndexSpaces.mma_m16n8k16(
                            (
                                Γ³_cplx0_cplx_in0_dish0_time0,
                                Γ³_cplx1_cplx_in0_dish0_time0,
                                Γ³_cplx0_cplx_in1_dish0_time0,
                                Γ³_cplx1_cplx_in1_dish0_time0,
                            ),
                            (ZZ_cplx_in0_dish0_time0, ZZ_cplx_in1_dish0_time0),
                            (YY_cplx0_dish0_time0, YY_cplx1_dish0_time0),
                        )
                        (YY_cplx0_dish1_time0, YY_cplx1_dish1_time0) = IndexSpaces.mma_m16n8k16(
                            (
                                Γ³_cplx0_cplx_in0_dish1_time0,
                                Γ³_cplx1_cplx_in0_dish1_time0,
                                Γ³_cplx0_cplx_in1_dish1_time0,
                                Γ³_cplx1_cplx_in1_dish1_time0,
                            ),
                            (ZZ_cplx_in0_dish1_time0, ZZ_cplx_in1_dish1_time0),
                            (YY_cplx0_dish1_time0, YY_cplx1_dish1_time0),
                        )
                        (YY_cplx0_dish0_time1, YY_cplx1_dish0_time1) = IndexSpaces.mma_m16n8k16(
                            (
                                Γ³_cplx0_cplx_in0_dish0_time1,
                                Γ³_cplx1_cplx_in0_dish0_time1,
                                Γ³_cplx0_cplx_in1_dish0_time1,
                                Γ³_cplx1_cplx_in1_dish0_time1,
                            ),
                            (ZZ_cplx_in0_dish0_time1, ZZ_cplx_in1_dish0_time1),
                            (YY_cplx0_dish0_time1, YY_cplx1_dish0_time1),
                        )
                        (YY_cplx0_dish1_time1, YY_cplx1_dish1_time1) = IndexSpaces.mma_m16n8k16(
                            (
                                Γ³_cplx0_cplx_in0_dish1_time1,
                                Γ³_cplx1_cplx_in0_dish1_time1,
                                Γ³_cplx0_cplx_in1_dish1_time1,
                                Γ³_cplx1_cplx_in1_dish1_time1,
                            ),
                            (ZZ_cplx_in0_dish1_time1, ZZ_cplx_in1_dish1_time1),
                            (YY_cplx0_dish1_time1, YY_cplx1_dish1_time1),
                        )
                        WWW_cplx0_dish0_time0 = YY_cplx0_dish0_time0
                        WWW_cplx1_dish0_time0 = YY_cplx1_dish0_time0
                        WWW_cplx0_dish1_time0 = YY_cplx0_dish1_time0
                        WWW_cplx1_dish1_time0 = YY_cplx1_dish1_time0
                        WWW_cplx0_dish0_time1 = YY_cplx0_dish0_time1
                        WWW_cplx1_dish0_time1 = YY_cplx1_dish0_time1
                        WWW_cplx0_dish1_time1 = YY_cplx0_dish1_time1
                        WWW_cplx1_dish1_time1 = YY_cplx1_dish1_time1
                        WWW_t0_cplx0_dish0 = WWW_cplx0_dish0_time0
                        WWW_t1_cplx0_dish0 = WWW_cplx0_dish0_time1
                        WWW_t0_cplx1_dish0 = WWW_cplx1_dish0_time0
                        WWW_t1_cplx1_dish0 = WWW_cplx1_dish0_time1
                        WWW_t0_cplx0_dish1 = WWW_cplx0_dish1_time0
                        WWW_t1_cplx0_dish1 = WWW_cplx0_dish1_time1
                        WWW_t0_cplx1_dish1 = WWW_cplx1_dish1_time0
                        WWW_t1_cplx1_dish1 = WWW_cplx1_dish1_time1
                        Γ⁴re = Γ⁴_cplx0
                        Γ⁴im = Γ⁴_cplx1
                        WWWre_dish0 = WWW_t1_cplx0_dish0
                        WWWim_dish0 = WWW_t1_cplx1_dish0
                        WWWre_dish1 = WWW_t1_cplx0_dish1
                        WWWim_dish1 = WWW_t1_cplx1_dish1
                        ZZZre_dish0 = muladd(Γ⁴re, WWWre_dish0, -Γ⁴im * WWWim_dish0)
                        ZZZre_dish1 = muladd(Γ⁴re, WWWre_dish1, -Γ⁴im * WWWim_dish1)
                        ZZZim_dish0 = muladd(Γ⁴re, WWWim_dish0, Γ⁴im * WWWre_dish0)
                        ZZZim_dish1 = muladd(Γ⁴re, WWWim_dish1, Γ⁴im * WWWre_dish1)
                        ZZZ_t0_cplx0_dish0 = WWW_t0_cplx0_dish0
                        ZZZ_t0_cplx1_dish0 = WWW_t0_cplx1_dish0
                        ZZZ_t0_cplx0_dish1 = WWW_t0_cplx0_dish1
                        ZZZ_t0_cplx1_dish1 = WWW_t0_cplx1_dish1
                        ZZZ_t1_cplx0_dish0 = ZZZre_dish0
                        ZZZ_t1_cplx1_dish0 = ZZZim_dish0
                        ZZZ_t1_cplx0_dish1 = ZZZre_dish1
                        ZZZ_t1_cplx1_dish1 = ZZZim_dish1
                        YYY_u0_cplx0_dish0 = WWW_t0_cplx0_dish0 + WWW_t1_cplx0_dish0
                        YYY_u0_cplx1_dish0 = WWW_t0_cplx1_dish0 + WWW_t1_cplx1_dish0
                        YYY_u0_cplx0_dish1 = WWW_t0_cplx0_dish1 + WWW_t1_cplx0_dish1
                        YYY_u0_cplx1_dish1 = WWW_t0_cplx1_dish1 + WWW_t1_cplx1_dish1
                        YYY_u1_cplx0_dish0 = WWW_t0_cplx0_dish0 - WWW_t1_cplx0_dish0
                        YYY_u1_cplx1_dish0 = WWW_t0_cplx1_dish0 - WWW_t1_cplx1_dish0
                        YYY_u1_cplx0_dish1 = WWW_t0_cplx0_dish1 - WWW_t1_cplx0_dish1
                        YYY_u1_cplx1_dish1 = WWW_t0_cplx1_dish1 - WWW_t1_cplx1_dish1
                        YYY_cplx0_dish0_freq0 = YYY_u0_cplx0_dish0
                        YYY_cplx0_dish0_freq64 = YYY_u1_cplx0_dish0
                        YYY_cplx1_dish0_freq0 = YYY_u0_cplx1_dish0
                        YYY_cplx1_dish0_freq64 = YYY_u1_cplx1_dish0
                        YYY_cplx0_dish1_freq0 = YYY_u0_cplx0_dish1
                        YYY_cplx0_dish1_freq64 = YYY_u1_cplx0_dish1
                        YYY_cplx1_dish1_freq0 = YYY_u0_cplx1_dish1
                        YYY_cplx1_dish1_freq64 = YYY_u1_cplx1_dish1
                        E4_cplx0_dish0_freq0 = YYY_cplx0_dish0_freq0
                        E4_cplx1_dish0_freq0 = YYY_cplx1_dish0_freq0
                        E4_cplx0_dish1_freq0 = YYY_cplx0_dish1_freq0
                        E4_cplx1_dish1_freq0 = YYY_cplx1_dish1_freq0
                        E4_cplx0_dish0_freq64 = YYY_cplx0_dish0_freq64
                        E4_cplx1_dish0_freq64 = YYY_cplx1_dish0_freq64
                        E4_cplx0_dish1_freq64 = YYY_cplx0_dish1_freq64
                        E4_cplx1_dish1_freq64 = YYY_cplx1_dish1_freq64
                        E5_cplx0_dish0_freq0 = Gains_freq0 * E4_cplx0_dish0_freq0
                        E5_cplx1_dish0_freq0 = Gains_freq0 * E4_cplx1_dish0_freq0
                        E5_cplx0_dish1_freq0 = Gains_freq0 * E4_cplx0_dish1_freq0
                        E5_cplx1_dish1_freq0 = Gains_freq0 * E4_cplx1_dish1_freq0
                        E5_cplx0_dish0_freq64 = Gains_freq64 * E4_cplx0_dish0_freq64
                        E5_cplx1_dish0_freq64 = Gains_freq64 * E4_cplx1_dish0_freq64
                        E5_cplx0_dish1_freq64 = Gains_freq64 * E4_cplx0_dish1_freq64
                        E5_cplx1_dish1_freq64 = Gains_freq64 * E4_cplx1_dish1_freq64
                        E5_cplx0_dish0_freq0 = clamp(E5_cplx0_dish0_freq0, Float16x2(-7, -7), Float16x2(7, 7))
                        E5_cplx1_dish0_freq0 = clamp(E5_cplx1_dish0_freq0, Float16x2(-7, -7), Float16x2(7, 7))
                        E5_cplx0_dish1_freq0 = clamp(E5_cplx0_dish1_freq0, Float16x2(-7, -7), Float16x2(7, 7))
                        E5_cplx1_dish1_freq0 = clamp(E5_cplx1_dish1_freq0, Float16x2(-7, -7), Float16x2(7, 7))
                        E5_cplx0_dish0_freq64 = clamp(E5_cplx0_dish0_freq64, Float16x2(-7, -7), Float16x2(7, 7))
                        E5_cplx1_dish0_freq64 = clamp(E5_cplx1_dish0_freq64, Float16x2(-7, -7), Float16x2(7, 7))
                        E5_cplx0_dish1_freq64 = clamp(E5_cplx0_dish1_freq64, Float16x2(-7, -7), Float16x2(7, 7))
                        E5_cplx1_dish1_freq64 = clamp(E5_cplx1_dish1_freq64, Float16x2(-7, -7), Float16x2(7, 7))
                        F̄_out_freq0 = convert_swapped_withoffset(
                            Int4x8, (E5_cplx0_dish0_freq0, E5_cplx1_dish0_freq0, E5_cplx0_dish1_freq0, E5_cplx1_dish1_freq0)
                        )
                        F̄_out_freq64 = convert_swapped_withoffset(
                            Int4x8, (E5_cplx0_dish0_freq64, E5_cplx1_dish0_freq64, E5_cplx0_dish1_freq64, E5_cplx1_dish1_freq64)
                        )
                        F̄_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 2) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 2) % 2) * 2 + ((0::Int32 ÷ 64) % 2) * 64) ÷ 2) % 64) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 2) % 2) * 32) + 0) + 0x01] =
                            F̄_out_freq0
                        F̄_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 2) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 2) % 2) * 2 + ((64::Int32 ÷ 64) % 2) * 64) ÷ 2) % 64) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 2) % 2) * 32) + 0) + 0x01] =
                            F̄_out_freq64
                        F_ringbuf_polr_dish_m0_time0 = F_ringbuf_polr_dish_mtap0_time0
                        F_ringbuf_polr_dish_m1_time0 = F_ringbuf_polr_dish_mtap1_time0
                        F_ringbuf_polr_dish_m2_time0 = F_ringbuf_polr_dish_mtap2_time0
                        F_ringbuf_polr_dish_m0_time1 = F_ringbuf_polr_dish_mtap0_time1
                        F_ringbuf_polr_dish_m1_time1 = F_ringbuf_polr_dish_mtap1_time1
                        F_ringbuf_polr_dish_m2_time1 = F_ringbuf_polr_dish_mtap2_time1
                        F_ringbuf_polr_dish_m0_time0 = F_ringbuf_polr_dish_m1_time0
                        F_ringbuf_polr_dish_m0_time1 = F_ringbuf_polr_dish_m1_time1
                        F_ringbuf_polr_dish_m1_time0 = F_ringbuf_polr_dish_m2_time0
                        F_ringbuf_polr_dish_m1_time1 = F_ringbuf_polr_dish_m2_time1
                        F_ringbuf_polr_dish_m2_time0 = F_in_time0
                        F_ringbuf_polr_dish_m2_time1 = F_in_time1
                        F_ringbuf_polr_dish_mtap0_time0 = F_ringbuf_polr_dish_m0_time0
                        F_ringbuf_polr_dish_mtap1_time0 = F_ringbuf_polr_dish_m1_time0
                        F_ringbuf_polr_dish_mtap2_time0 = F_ringbuf_polr_dish_m2_time0
                        F_ringbuf_polr_dish_mtap0_time1 = F_ringbuf_polr_dish_m0_time1
                        F_ringbuf_polr_dish_mtap1_time1 = F_ringbuf_polr_dish_m1_time1
                        F_ringbuf_polr_dish_mtap2_time1 = F_ringbuf_polr_dish_m2_time1
                        F_ringbuf_polr_dish64_mtap0_time0 = F_ringbuf_polr_dish_mtap0_time0
                        F_ringbuf_polr_dish64_mtap1_time0 = F_ringbuf_polr_dish_mtap1_time0
                        F_ringbuf_polr_dish64_mtap2_time0 = F_ringbuf_polr_dish_mtap2_time0
                        F_ringbuf_polr_dish64_mtap0_time1 = F_ringbuf_polr_dish_mtap0_time1
                        F_ringbuf_polr_dish64_mtap1_time1 = F_ringbuf_polr_dish_mtap1_time1
                        F_ringbuf_polr_dish64_mtap2_time1 = F_ringbuf_polr_dish_mtap2_time1
                    end
                    let dish = 96
                        F_ringbuf_polr_dish_mtap0_time0 = F_ringbuf_polr_dish96_mtap0_time0
                        F_ringbuf_polr_dish_mtap1_time0 = F_ringbuf_polr_dish96_mtap1_time0
                        F_ringbuf_polr_dish_mtap2_time0 = F_ringbuf_polr_dish96_mtap2_time0
                        F_ringbuf_polr_dish_mtap0_time1 = F_ringbuf_polr_dish96_mtap0_time1
                        F_ringbuf_polr_dish_mtap1_time1 = F_ringbuf_polr_dish96_mtap1_time1
                        F_ringbuf_polr_dish_mtap2_time1 = F_ringbuf_polr_dish96_mtap2_time1
                        F_in_time0 = F_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 0::Int32 % 2) ÷ 16) % 2) * 130 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 0::Int32 % 2) ÷ 8) % 2) * 260 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 0::Int32 % 2) ÷ 32) % 2) * 65 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 0::Int32 % 2) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 0::Int32 % 2) ÷ 4) % 2) * 520 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 0::Int32 % 2) ÷ 2) % 2) * 1040 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 2) % 2) * 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 0::Int32 % 2) % 2) * 2080) + 0x01]
                        F_in_time1 = F_shared[((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 1::Int32 % 2) ÷ 16) % 2) * 130 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 1::Int32 % 2) ÷ 8) % 2) * 260 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 1::Int32 % 2) ÷ 32) % 2) * 65 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 1::Int32 % 2) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 1::Int32 % 2) ÷ 4) % 2) * 520 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 1::Int32 % 2) ÷ 2) % 2) * 1040 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 2) % 2) * 32 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4) * 2 + ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 4) * 16 + 1::Int32 % 2) % 2) * 2080) + 0x01]
                        (E_cplx0_dish0_time0, E_cplx1_dish0_time0, E_cplx0_dish1_time0, E_cplx1_dish1_time0) = convert_swapped_withoffset(
                            NTuple{4,Float16x2}, F_in_time0
                        )
                        (E_cplx0_dish0_time1, E_cplx1_dish0_time1, E_cplx0_dish1_time1, E_cplx1_dish1_time1) = convert_swapped_withoffset(
                            NTuple{4,Float16x2}, F_in_time1
                        )
                        E2_cplx0_dish0_time0 = zero(E_cplx0_dish0_time0)
                        E2_cplx1_dish0_time0 = zero(E_cplx1_dish0_time0)
                        E2_cplx0_dish1_time0 = zero(E_cplx0_dish1_time0)
                        E2_cplx1_dish1_time0 = zero(E_cplx1_dish1_time0)
                        E2_cplx0_dish0_time1 = zero(E_cplx0_dish0_time1)
                        E2_cplx1_dish0_time1 = zero(E_cplx1_dish0_time1)
                        E2_cplx0_dish1_time1 = zero(E_cplx0_dish1_time1)
                        E2_cplx1_dish1_time1 = zero(E_cplx1_dish1_time1)
                        let mtap = 0
                            W_mtap_time0 = Wpfb_mtap0_time0
                            W_mtap_time1 = Wpfb_mtap0_time1
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap_time0 = F_ringbuf_polr_dish_mtap0_time0
                                F_ringbuf_polr_dish_mtap_time1 = F_ringbuf_polr_dish_mtap0_time1
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time0, E_ringbuf_polr_dish_mtap_cplx1_dish0_time0, E_ringbuf_polr_dish_mtap_cplx0_dish1_time0, E_ringbuf_polr_dish_mtap_cplx1_dish1_time0) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time0
                                )
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time1, E_ringbuf_polr_dish_mtap_cplx1_dish0_time1, E_ringbuf_polr_dish_mtap_cplx0_dish1_time1, E_ringbuf_polr_dish_mtap_cplx1_dish1_time1) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time1
                                )
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time0,
                                    E2_cplx0_dish0_time0,
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time0,
                                    E2_cplx1_dish0_time0,
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time0,
                                    E2_cplx0_dish1_time0,
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time0,
                                    E2_cplx1_dish1_time0,
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time1,
                                    E2_cplx0_dish0_time1,
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time1,
                                    E2_cplx1_dish0_time1,
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time1,
                                    E2_cplx0_dish1_time1,
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time1,
                                    E2_cplx1_dish1_time1,
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish0_time0, E2_cplx0_dish0_time0
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish0_time0, E2_cplx1_dish0_time0
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish1_time0, E2_cplx0_dish1_time0
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish1_time0, E2_cplx1_dish1_time0
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish0_time1, E2_cplx0_dish0_time1
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish0_time1, E2_cplx1_dish0_time1
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish1_time1, E2_cplx0_dish1_time1
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish1_time1, E2_cplx1_dish1_time1
                                )
                            end
                        end
                        let mtap = 1
                            W_mtap_time0 = Wpfb_mtap1_time0
                            W_mtap_time1 = Wpfb_mtap1_time1
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap_time0 = F_ringbuf_polr_dish_mtap1_time0
                                F_ringbuf_polr_dish_mtap_time1 = F_ringbuf_polr_dish_mtap1_time1
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time0, E_ringbuf_polr_dish_mtap_cplx1_dish0_time0, E_ringbuf_polr_dish_mtap_cplx0_dish1_time0, E_ringbuf_polr_dish_mtap_cplx1_dish1_time0) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time0
                                )
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time1, E_ringbuf_polr_dish_mtap_cplx1_dish0_time1, E_ringbuf_polr_dish_mtap_cplx0_dish1_time1, E_ringbuf_polr_dish_mtap_cplx1_dish1_time1) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time1
                                )
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time0,
                                    E2_cplx0_dish0_time0,
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time0,
                                    E2_cplx1_dish0_time0,
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time0,
                                    E2_cplx0_dish1_time0,
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time0,
                                    E2_cplx1_dish1_time0,
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time1,
                                    E2_cplx0_dish0_time1,
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time1,
                                    E2_cplx1_dish0_time1,
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time1,
                                    E2_cplx0_dish1_time1,
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time1,
                                    E2_cplx1_dish1_time1,
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish0_time0, E2_cplx0_dish0_time0
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish0_time0, E2_cplx1_dish0_time0
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish1_time0, E2_cplx0_dish1_time0
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish1_time0, E2_cplx1_dish1_time0
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish0_time1, E2_cplx0_dish0_time1
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish0_time1, E2_cplx1_dish0_time1
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish1_time1, E2_cplx0_dish1_time1
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish1_time1, E2_cplx1_dish1_time1
                                )
                            end
                        end
                        let mtap = 2
                            W_mtap_time0 = Wpfb_mtap2_time0
                            W_mtap_time1 = Wpfb_mtap2_time1
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap_time0 = F_ringbuf_polr_dish_mtap2_time0
                                F_ringbuf_polr_dish_mtap_time1 = F_ringbuf_polr_dish_mtap2_time1
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time0, E_ringbuf_polr_dish_mtap_cplx1_dish0_time0, E_ringbuf_polr_dish_mtap_cplx0_dish1_time0, E_ringbuf_polr_dish_mtap_cplx1_dish1_time0) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time0
                                )
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time1, E_ringbuf_polr_dish_mtap_cplx1_dish0_time1, E_ringbuf_polr_dish_mtap_cplx0_dish1_time1, E_ringbuf_polr_dish_mtap_cplx1_dish1_time1) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time1
                                )
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time0,
                                    E2_cplx0_dish0_time0,
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time0,
                                    E2_cplx1_dish0_time0,
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time0,
                                    E2_cplx0_dish1_time0,
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time0,
                                    E2_cplx1_dish1_time0,
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time1,
                                    E2_cplx0_dish0_time1,
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time1,
                                    E2_cplx1_dish0_time1,
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time1,
                                    E2_cplx0_dish1_time1,
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time1,
                                    E2_cplx1_dish1_time1,
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish0_time0, E2_cplx0_dish0_time0
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish0_time0, E2_cplx1_dish0_time0
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish1_time0, E2_cplx0_dish1_time0
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish1_time0, E2_cplx1_dish1_time0
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish0_time1, E2_cplx0_dish0_time1
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish0_time1, E2_cplx1_dish0_time1
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish1_time1, E2_cplx0_dish1_time1
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish1_time1, E2_cplx1_dish1_time1
                                )
                            end
                        end
                        let mtap = 3
                            W_mtap_time0 = Wpfb_mtap3_time0
                            W_mtap_time1 = Wpfb_mtap3_time1
                            if mtap < 3
                                F_ringbuf_polr_dish_mtap_time0 = F_ringbuf_polr_dish_mtap3_time0
                                F_ringbuf_polr_dish_mtap_time1 = F_ringbuf_polr_dish_mtap3_time1
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time0, E_ringbuf_polr_dish_mtap_cplx1_dish0_time0, E_ringbuf_polr_dish_mtap_cplx0_dish1_time0, E_ringbuf_polr_dish_mtap_cplx1_dish1_time0) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time0
                                )
                                (E_ringbuf_polr_dish_mtap_cplx0_dish0_time1, E_ringbuf_polr_dish_mtap_cplx1_dish0_time1, E_ringbuf_polr_dish_mtap_cplx0_dish1_time1, E_ringbuf_polr_dish_mtap_cplx1_dish1_time1) = convert_swapped_withoffset(
                                    NTuple{4,Float16x2}, F_ringbuf_polr_dish_mtap_time1
                                )
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time0,
                                    E2_cplx0_dish0_time0,
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time0,
                                    E2_cplx1_dish0_time0,
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time0,
                                    E2_cplx0_dish1_time0,
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time0,
                                    E2_cplx1_dish1_time0,
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish0_time1,
                                    E2_cplx0_dish0_time1,
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish0_time1,
                                    E2_cplx1_dish0_time1,
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx0_dish1_time1,
                                    E2_cplx0_dish1_time1,
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1),
                                    E_ringbuf_polr_dish_mtap_cplx1_dish1_time1,
                                    E2_cplx1_dish1_time1,
                                )
                            end
                            if mtap == 3
                                E2_cplx0_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish0_time0, E2_cplx0_dish0_time0
                                )
                                E2_cplx1_dish0_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish0_time0, E2_cplx1_dish0_time0
                                )
                                E2_cplx0_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx0_dish1_time0, E2_cplx0_dish1_time0
                                )
                                E2_cplx1_dish1_time0 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time0, +W_mtap_time0), E_cplx1_dish1_time0, E2_cplx1_dish1_time0
                                )
                                E2_cplx0_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish0_time1, E2_cplx0_dish0_time1
                                )
                                E2_cplx1_dish0_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish0_time1, E2_cplx1_dish0_time1
                                )
                                E2_cplx0_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx0_dish1_time1, E2_cplx0_dish1_time1
                                )
                                E2_cplx1_dish1_time1 = muladd(
                                    ifelse(isodd(mtap), -W_mtap_time1, +W_mtap_time1), E_cplx1_dish1_time1, E2_cplx1_dish1_time1
                                )
                            end
                        end
                        E2re_dish0_time0 = E2_cplx0_dish0_time0
                        E2im_dish0_time0 = E2_cplx1_dish0_time0
                        E2re_dish1_time0 = E2_cplx0_dish1_time0
                        E2im_dish1_time0 = E2_cplx1_dish1_time0
                        E2re_dish0_time1 = E2_cplx0_dish0_time1
                        E2im_dish0_time1 = E2_cplx1_dish0_time1
                        E2re_dish1_time1 = E2_cplx0_dish1_time1
                        E2im_dish1_time1 = E2_cplx1_dish1_time1
                        Xre_time0 = X_cplx0_time0
                        Xim_time0 = X_cplx1_time0
                        Xre_time1 = X_cplx0_time1
                        Xim_time1 = X_cplx1_time1
                        E3re_dish0_time0 = muladd(Xre_time0, E2re_dish0_time0, -Xim_time0 * E2im_dish0_time0)
                        E3re_dish1_time0 = muladd(Xre_time0, E2re_dish1_time0, -Xim_time0 * E2im_dish1_time0)
                        E3re_dish0_time1 = muladd(Xre_time1, E2re_dish0_time1, -Xim_time1 * E2im_dish0_time1)
                        E3re_dish1_time1 = muladd(Xre_time1, E2re_dish1_time1, -Xim_time1 * E2im_dish1_time1)
                        E3im_dish0_time0 = muladd(Xre_time0, E2im_dish0_time0, Xim_time0 * E2re_dish0_time0)
                        E3im_dish1_time0 = muladd(Xre_time0, E2im_dish1_time0, Xim_time0 * E2re_dish1_time0)
                        E3im_dish0_time1 = muladd(Xre_time1, E2im_dish0_time1, Xim_time1 * E2re_dish0_time1)
                        E3im_dish1_time1 = muladd(Xre_time1, E2im_dish1_time1, Xim_time1 * E2re_dish1_time1)
                        E3_cplx0_dish0_time0 = E3re_dish0_time0
                        E3_cplx1_dish0_time0 = E3im_dish0_time0
                        E3_cplx0_dish1_time0 = E3re_dish1_time0
                        E3_cplx1_dish1_time0 = E3im_dish1_time0
                        E3_cplx0_dish0_time1 = E3re_dish0_time1
                        E3_cplx1_dish0_time1 = E3im_dish0_time1
                        E3_cplx0_dish1_time1 = E3re_dish1_time1
                        E3_cplx1_dish1_time1 = E3im_dish1_time1
                        XX_cplx0_dish0_time0 = E3_cplx0_dish0_time0
                        XX_cplx1_dish0_time0 = E3_cplx1_dish0_time0
                        XX_cplx0_dish1_time0 = E3_cplx0_dish1_time0
                        XX_cplx1_dish1_time0 = E3_cplx1_dish1_time0
                        XX_cplx0_dish0_time1 = E3_cplx0_dish0_time1
                        XX_cplx1_dish0_time1 = E3_cplx1_dish0_time1
                        XX_cplx0_dish1_time1 = E3_cplx0_dish1_time1
                        XX_cplx1_dish1_time1 = E3_cplx1_dish1_time1
                        XXre_dish0_time0 = XX_cplx0_dish0_time0
                        XXim_dish0_time0 = XX_cplx1_dish0_time0
                        XXre_dish1_time0 = XX_cplx0_dish1_time0
                        XXim_dish1_time0 = XX_cplx1_dish1_time0
                        XXre_dish0_time1 = XX_cplx0_dish0_time1
                        XXim_dish0_time1 = XX_cplx1_dish0_time1
                        XXre_dish1_time1 = XX_cplx0_dish1_time1
                        XXim_dish1_time1 = XX_cplx1_dish1_time1
                        XX_cplx_in0_dish0_time0 = XXre_dish0_time0
                        XX_cplx_in1_dish0_time0 = XXim_dish0_time0
                        XX_cplx_in0_dish1_time0 = XXre_dish1_time0
                        XX_cplx_in1_dish1_time0 = XXim_dish1_time0
                        XX_cplx_in0_dish0_time1 = XXre_dish0_time1
                        XX_cplx_in1_dish0_time1 = XXim_dish0_time1
                        XX_cplx_in0_dish1_time1 = XXre_dish1_time1
                        XX_cplx_in1_dish1_time1 = XXim_dish1_time1
                        WW_cplx0_dish0_time0 = zero(Float16x2)
                        WW_cplx1_dish0_time0 = zero(Float16x2)
                        WW_cplx0_dish1_time0 = zero(Float16x2)
                        WW_cplx1_dish1_time0 = zero(Float16x2)
                        WW_cplx0_dish0_time1 = zero(Float16x2)
                        WW_cplx1_dish0_time1 = zero(Float16x2)
                        WW_cplx0_dish1_time1 = zero(Float16x2)
                        WW_cplx1_dish1_time1 = zero(Float16x2)
                        (WW_cplx0_dish0_time0, WW_cplx1_dish0_time0) = IndexSpaces.mma_m16n8k16(
                            (Γ¹_cplx0_cplx_in0_time0, Γ¹_cplx1_cplx_in0_time0, Γ¹_cplx0_cplx_in1_time0, Γ¹_cplx1_cplx_in1_time0),
                            (XX_cplx_in0_dish0_time0, XX_cplx_in1_dish0_time0),
                            (WW_cplx0_dish0_time0, WW_cplx1_dish0_time0),
                        )
                        (WW_cplx0_dish1_time0, WW_cplx1_dish1_time0) = IndexSpaces.mma_m16n8k16(
                            (Γ¹_cplx0_cplx_in0_time0, Γ¹_cplx1_cplx_in0_time0, Γ¹_cplx0_cplx_in1_time0, Γ¹_cplx1_cplx_in1_time0),
                            (XX_cplx_in0_dish1_time0, XX_cplx_in1_dish1_time0),
                            (WW_cplx0_dish1_time0, WW_cplx1_dish1_time0),
                        )
                        (WW_cplx0_dish0_time1, WW_cplx1_dish0_time1) = IndexSpaces.mma_m16n8k16(
                            (Γ¹_cplx0_cplx_in0_time1, Γ¹_cplx1_cplx_in0_time1, Γ¹_cplx0_cplx_in1_time1, Γ¹_cplx1_cplx_in1_time1),
                            (XX_cplx_in0_dish0_time1, XX_cplx_in1_dish0_time1),
                            (WW_cplx0_dish0_time1, WW_cplx1_dish0_time1),
                        )
                        (WW_cplx0_dish1_time1, WW_cplx1_dish1_time1) = IndexSpaces.mma_m16n8k16(
                            (Γ¹_cplx0_cplx_in0_time1, Γ¹_cplx1_cplx_in0_time1, Γ¹_cplx0_cplx_in1_time1, Γ¹_cplx1_cplx_in1_time1),
                            (XX_cplx_in0_dish1_time1, XX_cplx_in1_dish1_time1),
                            (WW_cplx0_dish1_time1, WW_cplx1_dish1_time1),
                        )
                        Γ²re_time0 = Γ²_cplx0_time0
                        Γ²im_time0 = Γ²_cplx1_time0
                        Γ²re_time1 = Γ²_cplx0_time1
                        Γ²im_time1 = Γ²_cplx1_time1
                        WWre_dish0_time0 = WW_cplx0_dish0_time0
                        WWim_dish0_time0 = WW_cplx1_dish0_time0
                        WWre_dish1_time0 = WW_cplx0_dish1_time0
                        WWim_dish1_time0 = WW_cplx1_dish1_time0
                        WWre_dish0_time1 = WW_cplx0_dish0_time1
                        WWim_dish0_time1 = WW_cplx1_dish0_time1
                        WWre_dish1_time1 = WW_cplx0_dish1_time1
                        WWim_dish1_time1 = WW_cplx1_dish1_time1
                        ZZre_dish0_time0 = muladd(Γ²re_time0, WWre_dish0_time0, -Γ²im_time0 * WWim_dish0_time0)
                        ZZre_dish1_time0 = muladd(Γ²re_time0, WWre_dish1_time0, -Γ²im_time0 * WWim_dish1_time0)
                        ZZre_dish0_time1 = muladd(Γ²re_time1, WWre_dish0_time1, -Γ²im_time1 * WWim_dish0_time1)
                        ZZre_dish1_time1 = muladd(Γ²re_time1, WWre_dish1_time1, -Γ²im_time1 * WWim_dish1_time1)
                        ZZim_dish0_time0 = muladd(Γ²re_time0, WWim_dish0_time0, Γ²im_time0 * WWre_dish0_time0)
                        ZZim_dish1_time0 = muladd(Γ²re_time0, WWim_dish1_time0, Γ²im_time0 * WWre_dish1_time0)
                        ZZim_dish0_time1 = muladd(Γ²re_time1, WWim_dish0_time1, Γ²im_time1 * WWre_dish0_time1)
                        ZZim_dish1_time1 = muladd(Γ²re_time1, WWim_dish1_time1, Γ²im_time1 * WWre_dish1_time1)
                        ZZ_cplx0_dish0_time0 = ZZre_dish0_time0
                        ZZ_cplx1_dish0_time0 = ZZim_dish0_time0
                        ZZ_cplx0_dish1_time0 = ZZre_dish1_time0
                        ZZ_cplx1_dish1_time0 = ZZim_dish1_time0
                        ZZ_cplx0_dish0_time1 = ZZre_dish0_time1
                        ZZ_cplx1_dish0_time1 = ZZim_dish0_time1
                        ZZ_cplx0_dish1_time1 = ZZre_dish1_time1
                        ZZ_cplx1_dish1_time1 = ZZim_dish1_time1
                        ZZre_dish0_time0 = ZZ_cplx0_dish0_time0
                        ZZim_dish0_time0 = ZZ_cplx1_dish0_time0
                        ZZre_dish1_time0 = ZZ_cplx0_dish1_time0
                        ZZim_dish1_time0 = ZZ_cplx1_dish1_time0
                        ZZre_dish0_time1 = ZZ_cplx0_dish0_time1
                        ZZim_dish0_time1 = ZZ_cplx1_dish0_time1
                        ZZre_dish1_time1 = ZZ_cplx0_dish1_time1
                        ZZim_dish1_time1 = ZZ_cplx1_dish1_time1
                        ZZ_cplx_in0_dish0_time0 = ZZre_dish0_time0
                        ZZ_cplx_in1_dish0_time0 = ZZim_dish0_time0
                        ZZ_cplx_in0_dish1_time0 = ZZre_dish1_time0
                        ZZ_cplx_in1_dish1_time0 = ZZim_dish1_time0
                        ZZ_cplx_in0_dish0_time1 = ZZre_dish0_time1
                        ZZ_cplx_in1_dish0_time1 = ZZim_dish0_time1
                        ZZ_cplx_in0_dish1_time1 = ZZre_dish1_time1
                        ZZ_cplx_in1_dish1_time1 = ZZim_dish1_time1
                        YY_cplx0_dish0_time0 = zero(Float16x2)
                        YY_cplx1_dish0_time0 = zero(Float16x2)
                        YY_cplx0_dish1_time0 = zero(Float16x2)
                        YY_cplx1_dish1_time0 = zero(Float16x2)
                        YY_cplx0_dish0_time1 = zero(Float16x2)
                        YY_cplx1_dish0_time1 = zero(Float16x2)
                        YY_cplx0_dish1_time1 = zero(Float16x2)
                        YY_cplx1_dish1_time1 = zero(Float16x2)
                        (YY_cplx0_dish0_time0, YY_cplx1_dish0_time0) = IndexSpaces.mma_m16n8k16(
                            (
                                Γ³_cplx0_cplx_in0_dish0_time0,
                                Γ³_cplx1_cplx_in0_dish0_time0,
                                Γ³_cplx0_cplx_in1_dish0_time0,
                                Γ³_cplx1_cplx_in1_dish0_time0,
                            ),
                            (ZZ_cplx_in0_dish0_time0, ZZ_cplx_in1_dish0_time0),
                            (YY_cplx0_dish0_time0, YY_cplx1_dish0_time0),
                        )
                        (YY_cplx0_dish1_time0, YY_cplx1_dish1_time0) = IndexSpaces.mma_m16n8k16(
                            (
                                Γ³_cplx0_cplx_in0_dish1_time0,
                                Γ³_cplx1_cplx_in0_dish1_time0,
                                Γ³_cplx0_cplx_in1_dish1_time0,
                                Γ³_cplx1_cplx_in1_dish1_time0,
                            ),
                            (ZZ_cplx_in0_dish1_time0, ZZ_cplx_in1_dish1_time0),
                            (YY_cplx0_dish1_time0, YY_cplx1_dish1_time0),
                        )
                        (YY_cplx0_dish0_time1, YY_cplx1_dish0_time1) = IndexSpaces.mma_m16n8k16(
                            (
                                Γ³_cplx0_cplx_in0_dish0_time1,
                                Γ³_cplx1_cplx_in0_dish0_time1,
                                Γ³_cplx0_cplx_in1_dish0_time1,
                                Γ³_cplx1_cplx_in1_dish0_time1,
                            ),
                            (ZZ_cplx_in0_dish0_time1, ZZ_cplx_in1_dish0_time1),
                            (YY_cplx0_dish0_time1, YY_cplx1_dish0_time1),
                        )
                        (YY_cplx0_dish1_time1, YY_cplx1_dish1_time1) = IndexSpaces.mma_m16n8k16(
                            (
                                Γ³_cplx0_cplx_in0_dish1_time1,
                                Γ³_cplx1_cplx_in0_dish1_time1,
                                Γ³_cplx0_cplx_in1_dish1_time1,
                                Γ³_cplx1_cplx_in1_dish1_time1,
                            ),
                            (ZZ_cplx_in0_dish1_time1, ZZ_cplx_in1_dish1_time1),
                            (YY_cplx0_dish1_time1, YY_cplx1_dish1_time1),
                        )
                        WWW_cplx0_dish0_time0 = YY_cplx0_dish0_time0
                        WWW_cplx1_dish0_time0 = YY_cplx1_dish0_time0
                        WWW_cplx0_dish1_time0 = YY_cplx0_dish1_time0
                        WWW_cplx1_dish1_time0 = YY_cplx1_dish1_time0
                        WWW_cplx0_dish0_time1 = YY_cplx0_dish0_time1
                        WWW_cplx1_dish0_time1 = YY_cplx1_dish0_time1
                        WWW_cplx0_dish1_time1 = YY_cplx0_dish1_time1
                        WWW_cplx1_dish1_time1 = YY_cplx1_dish1_time1
                        WWW_t0_cplx0_dish0 = WWW_cplx0_dish0_time0
                        WWW_t1_cplx0_dish0 = WWW_cplx0_dish0_time1
                        WWW_t0_cplx1_dish0 = WWW_cplx1_dish0_time0
                        WWW_t1_cplx1_dish0 = WWW_cplx1_dish0_time1
                        WWW_t0_cplx0_dish1 = WWW_cplx0_dish1_time0
                        WWW_t1_cplx0_dish1 = WWW_cplx0_dish1_time1
                        WWW_t0_cplx1_dish1 = WWW_cplx1_dish1_time0
                        WWW_t1_cplx1_dish1 = WWW_cplx1_dish1_time1
                        Γ⁴re = Γ⁴_cplx0
                        Γ⁴im = Γ⁴_cplx1
                        WWWre_dish0 = WWW_t1_cplx0_dish0
                        WWWim_dish0 = WWW_t1_cplx1_dish0
                        WWWre_dish1 = WWW_t1_cplx0_dish1
                        WWWim_dish1 = WWW_t1_cplx1_dish1
                        ZZZre_dish0 = muladd(Γ⁴re, WWWre_dish0, -Γ⁴im * WWWim_dish0)
                        ZZZre_dish1 = muladd(Γ⁴re, WWWre_dish1, -Γ⁴im * WWWim_dish1)
                        ZZZim_dish0 = muladd(Γ⁴re, WWWim_dish0, Γ⁴im * WWWre_dish0)
                        ZZZim_dish1 = muladd(Γ⁴re, WWWim_dish1, Γ⁴im * WWWre_dish1)
                        ZZZ_t0_cplx0_dish0 = WWW_t0_cplx0_dish0
                        ZZZ_t0_cplx1_dish0 = WWW_t0_cplx1_dish0
                        ZZZ_t0_cplx0_dish1 = WWW_t0_cplx0_dish1
                        ZZZ_t0_cplx1_dish1 = WWW_t0_cplx1_dish1
                        ZZZ_t1_cplx0_dish0 = ZZZre_dish0
                        ZZZ_t1_cplx1_dish0 = ZZZim_dish0
                        ZZZ_t1_cplx0_dish1 = ZZZre_dish1
                        ZZZ_t1_cplx1_dish1 = ZZZim_dish1
                        YYY_u0_cplx0_dish0 = WWW_t0_cplx0_dish0 + WWW_t1_cplx0_dish0
                        YYY_u0_cplx1_dish0 = WWW_t0_cplx1_dish0 + WWW_t1_cplx1_dish0
                        YYY_u0_cplx0_dish1 = WWW_t0_cplx0_dish1 + WWW_t1_cplx0_dish1
                        YYY_u0_cplx1_dish1 = WWW_t0_cplx1_dish1 + WWW_t1_cplx1_dish1
                        YYY_u1_cplx0_dish0 = WWW_t0_cplx0_dish0 - WWW_t1_cplx0_dish0
                        YYY_u1_cplx1_dish0 = WWW_t0_cplx1_dish0 - WWW_t1_cplx1_dish0
                        YYY_u1_cplx0_dish1 = WWW_t0_cplx0_dish1 - WWW_t1_cplx0_dish1
                        YYY_u1_cplx1_dish1 = WWW_t0_cplx1_dish1 - WWW_t1_cplx1_dish1
                        YYY_cplx0_dish0_freq0 = YYY_u0_cplx0_dish0
                        YYY_cplx0_dish0_freq64 = YYY_u1_cplx0_dish0
                        YYY_cplx1_dish0_freq0 = YYY_u0_cplx1_dish0
                        YYY_cplx1_dish0_freq64 = YYY_u1_cplx1_dish0
                        YYY_cplx0_dish1_freq0 = YYY_u0_cplx0_dish1
                        YYY_cplx0_dish1_freq64 = YYY_u1_cplx0_dish1
                        YYY_cplx1_dish1_freq0 = YYY_u0_cplx1_dish1
                        YYY_cplx1_dish1_freq64 = YYY_u1_cplx1_dish1
                        E4_cplx0_dish0_freq0 = YYY_cplx0_dish0_freq0
                        E4_cplx1_dish0_freq0 = YYY_cplx1_dish0_freq0
                        E4_cplx0_dish1_freq0 = YYY_cplx0_dish1_freq0
                        E4_cplx1_dish1_freq0 = YYY_cplx1_dish1_freq0
                        E4_cplx0_dish0_freq64 = YYY_cplx0_dish0_freq64
                        E4_cplx1_dish0_freq64 = YYY_cplx1_dish0_freq64
                        E4_cplx0_dish1_freq64 = YYY_cplx0_dish1_freq64
                        E4_cplx1_dish1_freq64 = YYY_cplx1_dish1_freq64
                        E5_cplx0_dish0_freq0 = Gains_freq0 * E4_cplx0_dish0_freq0
                        E5_cplx1_dish0_freq0 = Gains_freq0 * E4_cplx1_dish0_freq0
                        E5_cplx0_dish1_freq0 = Gains_freq0 * E4_cplx0_dish1_freq0
                        E5_cplx1_dish1_freq0 = Gains_freq0 * E4_cplx1_dish1_freq0
                        E5_cplx0_dish0_freq64 = Gains_freq64 * E4_cplx0_dish0_freq64
                        E5_cplx1_dish0_freq64 = Gains_freq64 * E4_cplx1_dish0_freq64
                        E5_cplx0_dish1_freq64 = Gains_freq64 * E4_cplx0_dish1_freq64
                        E5_cplx1_dish1_freq64 = Gains_freq64 * E4_cplx1_dish1_freq64
                        E5_cplx0_dish0_freq0 = clamp(E5_cplx0_dish0_freq0, Float16x2(-7, -7), Float16x2(7, 7))
                        E5_cplx1_dish0_freq0 = clamp(E5_cplx1_dish0_freq0, Float16x2(-7, -7), Float16x2(7, 7))
                        E5_cplx0_dish1_freq0 = clamp(E5_cplx0_dish1_freq0, Float16x2(-7, -7), Float16x2(7, 7))
                        E5_cplx1_dish1_freq0 = clamp(E5_cplx1_dish1_freq0, Float16x2(-7, -7), Float16x2(7, 7))
                        E5_cplx0_dish0_freq64 = clamp(E5_cplx0_dish0_freq64, Float16x2(-7, -7), Float16x2(7, 7))
                        E5_cplx1_dish0_freq64 = clamp(E5_cplx1_dish0_freq64, Float16x2(-7, -7), Float16x2(7, 7))
                        E5_cplx0_dish1_freq64 = clamp(E5_cplx0_dish1_freq64, Float16x2(-7, -7), Float16x2(7, 7))
                        E5_cplx1_dish1_freq64 = clamp(E5_cplx1_dish1_freq64, Float16x2(-7, -7), Float16x2(7, 7))
                        F̄_out_freq0 = convert_swapped_withoffset(
                            Int4x8, (E5_cplx0_dish0_freq0, E5_cplx1_dish0_freq0, E5_cplx0_dish1_freq0, E5_cplx1_dish1_freq0)
                        )
                        F̄_out_freq64 = convert_swapped_withoffset(
                            Int4x8, (E5_cplx0_dish0_freq64, E5_cplx1_dish0_freq64, E5_cplx0_dish1_freq64, E5_cplx1_dish1_freq64)
                        )
                        F̄_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 2) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 2) % 2) * 2 + ((0::Int32 ÷ 64) % 2) * 64) ÷ 2) % 64) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 2) % 2) * 32) + 0) + 0x01] =
                            F̄_out_freq0
                        F̄_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((IndexSpaces.assume_inrange(t_inner::Int32, 0, 128, 256) ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 2) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 8 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 2) % 2) * 2 + ((64::Int32 ÷ 64) % 2) * 64) ÷ 2) % 64) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 2 + ((dish::Int32 ÷ 32) % 4) * 32) ÷ 2) % 2) * 32) + 0) + 0x01] =
                            F̄_out_freq64
                        F_ringbuf_polr_dish_m0_time0 = F_ringbuf_polr_dish_mtap0_time0
                        F_ringbuf_polr_dish_m1_time0 = F_ringbuf_polr_dish_mtap1_time0
                        F_ringbuf_polr_dish_m2_time0 = F_ringbuf_polr_dish_mtap2_time0
                        F_ringbuf_polr_dish_m0_time1 = F_ringbuf_polr_dish_mtap0_time1
                        F_ringbuf_polr_dish_m1_time1 = F_ringbuf_polr_dish_mtap1_time1
                        F_ringbuf_polr_dish_m2_time1 = F_ringbuf_polr_dish_mtap2_time1
                        F_ringbuf_polr_dish_m0_time0 = F_ringbuf_polr_dish_m1_time0
                        F_ringbuf_polr_dish_m0_time1 = F_ringbuf_polr_dish_m1_time1
                        F_ringbuf_polr_dish_m1_time0 = F_ringbuf_polr_dish_m2_time0
                        F_ringbuf_polr_dish_m1_time1 = F_ringbuf_polr_dish_m2_time1
                        F_ringbuf_polr_dish_m2_time0 = F_in_time0
                        F_ringbuf_polr_dish_m2_time1 = F_in_time1
                        F_ringbuf_polr_dish_mtap0_time0 = F_ringbuf_polr_dish_m0_time0
                        F_ringbuf_polr_dish_mtap1_time0 = F_ringbuf_polr_dish_m1_time0
                        F_ringbuf_polr_dish_mtap2_time0 = F_ringbuf_polr_dish_m2_time0
                        F_ringbuf_polr_dish_mtap0_time1 = F_ringbuf_polr_dish_m0_time1
                        F_ringbuf_polr_dish_mtap1_time1 = F_ringbuf_polr_dish_m1_time1
                        F_ringbuf_polr_dish_mtap2_time1 = F_ringbuf_polr_dish_m2_time1
                        F_ringbuf_polr_dish96_mtap0_time0 = F_ringbuf_polr_dish_mtap0_time0
                        F_ringbuf_polr_dish96_mtap1_time0 = F_ringbuf_polr_dish_mtap1_time0
                        F_ringbuf_polr_dish96_mtap2_time0 = F_ringbuf_polr_dish_mtap2_time0
                        F_ringbuf_polr_dish96_mtap0_time1 = F_ringbuf_polr_dish_mtap0_time1
                        F_ringbuf_polr_dish96_mtap1_time1 = F_ringbuf_polr_dish_mtap1_time1
                        F_ringbuf_polr_dish96_mtap2_time1 = F_ringbuf_polr_dish_mtap2_time1
                    end
                    F_ringbuf_dish0_mtap0_time0 = F_ringbuf_polr_dish0_mtap0_time0
                    F_ringbuf_dish32_mtap0_time0 = F_ringbuf_polr_dish32_mtap0_time0
                    F_ringbuf_dish64_mtap0_time0 = F_ringbuf_polr_dish64_mtap0_time0
                    F_ringbuf_dish96_mtap0_time0 = F_ringbuf_polr_dish96_mtap0_time0
                    F_ringbuf_dish0_mtap1_time0 = F_ringbuf_polr_dish0_mtap1_time0
                    F_ringbuf_dish32_mtap1_time0 = F_ringbuf_polr_dish32_mtap1_time0
                    F_ringbuf_dish64_mtap1_time0 = F_ringbuf_polr_dish64_mtap1_time0
                    F_ringbuf_dish96_mtap1_time0 = F_ringbuf_polr_dish96_mtap1_time0
                    F_ringbuf_dish0_mtap2_time0 = F_ringbuf_polr_dish0_mtap2_time0
                    F_ringbuf_dish32_mtap2_time0 = F_ringbuf_polr_dish32_mtap2_time0
                    F_ringbuf_dish64_mtap2_time0 = F_ringbuf_polr_dish64_mtap2_time0
                    F_ringbuf_dish96_mtap2_time0 = F_ringbuf_polr_dish96_mtap2_time0
                    F_ringbuf_dish0_mtap0_time1 = F_ringbuf_polr_dish0_mtap0_time1
                    F_ringbuf_dish32_mtap0_time1 = F_ringbuf_polr_dish32_mtap0_time1
                    F_ringbuf_dish64_mtap0_time1 = F_ringbuf_polr_dish64_mtap0_time1
                    F_ringbuf_dish96_mtap0_time1 = F_ringbuf_polr_dish96_mtap0_time1
                    F_ringbuf_dish0_mtap1_time1 = F_ringbuf_polr_dish0_mtap1_time1
                    F_ringbuf_dish32_mtap1_time1 = F_ringbuf_polr_dish32_mtap1_time1
                    F_ringbuf_dish64_mtap1_time1 = F_ringbuf_polr_dish64_mtap1_time1
                    F_ringbuf_dish96_mtap1_time1 = F_ringbuf_polr_dish96_mtap1_time1
                    F_ringbuf_dish0_mtap2_time1 = F_ringbuf_polr_dish0_mtap2_time1
                    F_ringbuf_dish32_mtap2_time1 = F_ringbuf_polr_dish32_mtap2_time1
                    F_ringbuf_dish64_mtap2_time1 = F_ringbuf_polr_dish64_mtap2_time1
                    F_ringbuf_dish96_mtap2_time1 = F_ringbuf_polr_dish96_mtap2_time1
                end
            end
            IndexSpaces.cuda_sync_threads()
            Ē_dish0_freq0_time0 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((0::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((0::Int32 ÷ 64) % 2) * 64) ÷ 2) % 64) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((0::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish2_freq0_time0 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((2::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((0::Int32 ÷ 64) % 2) * 64) ÷ 2) % 64) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((2::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish4_freq0_time0 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((4::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((0::Int32 ÷ 64) % 2) * 64) ÷ 2) % 64) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((4::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish6_freq0_time0 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((6::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((0::Int32 ÷ 64) % 2) * 64) ÷ 2) % 64) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((6::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish0_freq64_time0 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((0::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((64::Int32 ÷ 64) % 2) * 64) ÷ 2) % 64) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((0::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish2_freq64_time0 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((2::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((64::Int32 ÷ 64) % 2) * 64) ÷ 2) % 64) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((2::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish4_freq64_time0 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((4::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((64::Int32 ÷ 64) % 2) * 64) ÷ 2) % 64) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((4::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish6_freq64_time0 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((6::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((0::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((64::Int32 ÷ 64) % 2) * 64) ÷ 2) % 64) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((6::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish0_freq0_time128 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((0::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((0::Int32 ÷ 64) % 2) * 64) ÷ 2) % 64) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((0::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish2_freq0_time128 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((2::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((0::Int32 ÷ 64) % 2) * 64) ÷ 2) % 64) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((2::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish4_freq0_time128 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((4::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((0::Int32 ÷ 64) % 2) * 64) ÷ 2) % 64) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((4::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish6_freq0_time128 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((6::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((0::Int32 ÷ 64) % 2) * 64) ÷ 2) % 64) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((6::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish0_freq64_time128 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((0::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((64::Int32 ÷ 64) % 2) * 64) ÷ 2) % 64) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((0::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish2_freq64_time128 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((2::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((64::Int32 ÷ 64) % 2) * 64) ÷ 2) % 64) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((2::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish4_freq64_time128 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((4::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((64::Int32 ÷ 64) % 2) * 64) ÷ 2) % 64) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((4::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32) + 0x01]
            Ē_dish6_freq64_time128 = F̄_shared[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((6::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 4) % 32 + (((((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 + ((128::Int32 ÷ 128) % 2) * 128) ÷ 128) % 2) * 4161 + (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) * 128 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 2 + ((64::Int32 ÷ 64) % 2) * 64) ÷ 2) % 64) * 65 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 + ((6::Int32 ÷ 2) % 4) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 8) ÷ 2) % 2) * 32) + 0x01]
            (Ē1_dish0_freq0_time0, Ē1_dish2_freq0_time0) = (
                IndexSpaces.get_lo16(Ē_dish0_freq0_time0, Ē_dish2_freq0_time0),
                IndexSpaces.get_hi16(Ē_dish0_freq0_time0, Ē_dish2_freq0_time0),
            )
            (Ē1_dish4_freq0_time0, Ē1_dish6_freq0_time0) = (
                IndexSpaces.get_lo16(Ē_dish4_freq0_time0, Ē_dish6_freq0_time0),
                IndexSpaces.get_hi16(Ē_dish4_freq0_time0, Ē_dish6_freq0_time0),
            )
            (Ē1_dish0_freq64_time0, Ē1_dish2_freq64_time0) = (
                IndexSpaces.get_lo16(Ē_dish0_freq64_time0, Ē_dish2_freq64_time0),
                IndexSpaces.get_hi16(Ē_dish0_freq64_time0, Ē_dish2_freq64_time0),
            )
            (Ē1_dish4_freq64_time0, Ē1_dish6_freq64_time0) = (
                IndexSpaces.get_lo16(Ē_dish4_freq64_time0, Ē_dish6_freq64_time0),
                IndexSpaces.get_hi16(Ē_dish4_freq64_time0, Ē_dish6_freq64_time0),
            )
            (Ē1_dish0_freq0_time128, Ē1_dish2_freq0_time128) = (
                IndexSpaces.get_lo16(Ē_dish0_freq0_time128, Ē_dish2_freq0_time128),
                IndexSpaces.get_hi16(Ē_dish0_freq0_time128, Ē_dish2_freq0_time128),
            )
            (Ē1_dish4_freq0_time128, Ē1_dish6_freq0_time128) = (
                IndexSpaces.get_lo16(Ē_dish4_freq0_time128, Ē_dish6_freq0_time128),
                IndexSpaces.get_hi16(Ē_dish4_freq0_time128, Ē_dish6_freq0_time128),
            )
            (Ē1_dish0_freq64_time128, Ē1_dish2_freq64_time128) = (
                IndexSpaces.get_lo16(Ē_dish0_freq64_time128, Ē_dish2_freq64_time128),
                IndexSpaces.get_hi16(Ē_dish0_freq64_time128, Ē_dish2_freq64_time128),
            )
            (Ē1_dish4_freq64_time128, Ē1_dish6_freq64_time128) = (
                IndexSpaces.get_lo16(Ē_dish4_freq64_time128, Ē_dish6_freq64_time128),
                IndexSpaces.get_hi16(Ē_dish4_freq64_time128, Ē_dish6_freq64_time128),
            )
            Ē1lo_dish0_freq0_time0 = Ē1_dish0_freq0_time0
            Ē1hi_dish0_freq0_time0 = Ē1_dish2_freq0_time0
            Ē1lo_dish4_freq0_time0 = Ē1_dish4_freq0_time0
            Ē1hi_dish4_freq0_time0 = Ē1_dish6_freq0_time0
            Ē1lo_dish0_freq64_time0 = Ē1_dish0_freq64_time0
            Ē1hi_dish0_freq64_time0 = Ē1_dish2_freq64_time0
            Ē1lo_dish4_freq64_time0 = Ē1_dish4_freq64_time0
            Ē1hi_dish4_freq64_time0 = Ē1_dish6_freq64_time0
            Ē1lo_dish0_freq0_time128 = Ē1_dish0_freq0_time128
            Ē1hi_dish0_freq0_time128 = Ē1_dish2_freq0_time128
            Ē1lo_dish4_freq0_time128 = Ē1_dish4_freq0_time128
            Ē1hi_dish4_freq0_time128 = Ē1_dish6_freq0_time128
            Ē1lo_dish0_freq64_time128 = Ē1_dish0_freq64_time128
            Ē1hi_dish0_freq64_time128 = Ē1_dish2_freq64_time128
            Ē1lo_dish4_freq64_time128 = Ē1_dish4_freq64_time128
            Ē1hi_dish4_freq64_time128 = Ē1_dish6_freq64_time128
            Ē1_dish0_freq0_time0 = Ē1lo_dish0_freq0_time0
            Ē1_dish0_freq1_time0 = Ē1hi_dish0_freq0_time0
            Ē1_dish4_freq0_time0 = Ē1lo_dish4_freq0_time0
            Ē1_dish4_freq1_time0 = Ē1hi_dish4_freq0_time0
            Ē1_dish0_freq64_time0 = Ē1lo_dish0_freq64_time0
            Ē1_dish0_freq65_time0 = Ē1hi_dish0_freq64_time0
            Ē1_dish4_freq64_time0 = Ē1lo_dish4_freq64_time0
            Ē1_dish4_freq65_time0 = Ē1hi_dish4_freq64_time0
            Ē1_dish0_freq0_time128 = Ē1lo_dish0_freq0_time128
            Ē1_dish0_freq1_time128 = Ē1hi_dish0_freq0_time128
            Ē1_dish4_freq0_time128 = Ē1lo_dish4_freq0_time128
            Ē1_dish4_freq1_time128 = Ē1hi_dish4_freq0_time128
            Ē1_dish0_freq64_time128 = Ē1lo_dish0_freq64_time128
            Ē1_dish0_freq65_time128 = Ē1hi_dish0_freq64_time128
            Ē1_dish4_freq64_time128 = Ē1lo_dish4_freq64_time128
            Ē1_dish4_freq65_time128 = Ē1hi_dish4_freq64_time128
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
            (Ē2_dish0_freq64_time0, Ē2_dish0_freq65_time0) = let
                src = if is_lo_thread
                    Ē1_dish0_freq65_time0
                else
                    Ē1_dish0_freq64_time0
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (Ē1_dish0_freq64_time0, dst)
                else
                    (dst, Ē1_dish0_freq65_time0)
                end
            end
            (Ē2_dish4_freq64_time0, Ē2_dish4_freq65_time0) = let
                src = if is_lo_thread
                    Ē1_dish4_freq65_time0
                else
                    Ē1_dish4_freq64_time0
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (Ē1_dish4_freq64_time0, dst)
                else
                    (dst, Ē1_dish4_freq65_time0)
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
            (Ē2_dish0_freq64_time128, Ē2_dish0_freq65_time128) = let
                src = if is_lo_thread
                    Ē1_dish0_freq65_time128
                else
                    Ē1_dish0_freq64_time128
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (Ē1_dish0_freq64_time128, dst)
                else
                    (dst, Ē1_dish0_freq65_time128)
                end
            end
            (Ē2_dish4_freq64_time128, Ē2_dish4_freq65_time128) = let
                src = if is_lo_thread
                    Ē1_dish4_freq65_time128
                else
                    Ē1_dish4_freq64_time128
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (Ē1_dish4_freq64_time128, dst)
                else
                    (dst, Ē1_dish4_freq65_time128)
                end
            end
            Ē2lo_dish0_freq0_time0 = Ē2_dish0_freq0_time0
            Ē2hi_dish0_freq0_time0 = Ē2_dish0_freq1_time0
            Ē2lo_dish4_freq0_time0 = Ē2_dish4_freq0_time0
            Ē2hi_dish4_freq0_time0 = Ē2_dish4_freq1_time0
            Ē2lo_dish0_freq64_time0 = Ē2_dish0_freq64_time0
            Ē2hi_dish0_freq64_time0 = Ē2_dish0_freq65_time0
            Ē2lo_dish4_freq64_time0 = Ē2_dish4_freq64_time0
            Ē2hi_dish4_freq64_time0 = Ē2_dish4_freq65_time0
            Ē2lo_dish0_freq0_time128 = Ē2_dish0_freq0_time128
            Ē2hi_dish0_freq0_time128 = Ē2_dish0_freq1_time128
            Ē2lo_dish4_freq0_time128 = Ē2_dish4_freq0_time128
            Ē2hi_dish4_freq0_time128 = Ē2_dish4_freq1_time128
            Ē2lo_dish0_freq64_time128 = Ē2_dish0_freq64_time128
            Ē2hi_dish0_freq64_time128 = Ē2_dish0_freq65_time128
            Ē2lo_dish4_freq64_time128 = Ē2_dish4_freq64_time128
            Ē2hi_dish4_freq64_time128 = Ē2_dish4_freq65_time128
            Ē3_dish0_freq0_time0 = Ē2lo_dish0_freq0_time0
            Ē3_dish8_freq0_time0 = Ē2hi_dish0_freq0_time0
            Ē3_dish4_freq0_time0 = Ē2lo_dish4_freq0_time0
            Ē3_dish12_freq0_time0 = Ē2hi_dish4_freq0_time0
            Ē3_dish0_freq64_time0 = Ē2lo_dish0_freq64_time0
            Ē3_dish8_freq64_time0 = Ē2hi_dish0_freq64_time0
            Ē3_dish4_freq64_time0 = Ē2lo_dish4_freq64_time0
            Ē3_dish12_freq64_time0 = Ē2hi_dish4_freq64_time0
            Ē3_dish0_freq0_time128 = Ē2lo_dish0_freq0_time128
            Ē3_dish8_freq0_time128 = Ē2hi_dish0_freq0_time128
            Ē3_dish4_freq0_time128 = Ē2lo_dish4_freq0_time128
            Ē3_dish12_freq0_time128 = Ē2hi_dish4_freq0_time128
            Ē3_dish0_freq64_time128 = Ē2lo_dish0_freq64_time128
            Ē3_dish8_freq64_time128 = Ē2hi_dish0_freq64_time128
            Ē3_dish4_freq64_time128 = Ē2lo_dish4_freq64_time128
            Ē3_dish12_freq64_time128 = Ē2hi_dish4_freq64_time128
            if ((IndexSpaces.assume_inrange(t_outer::Int32, 0i32, 256, 65536) ÷ 256) % 256) * 256 + ((0::Int32 ÷ 128) % 2) * 128 ≥
                384
                IndexSpaces.unsafe_store4!(
                    Ē_memory,
                    let
                        offset = 16384 * T̄min - 49152
                        length = 8388608
                        mod(
                            (
                                (
                                    (
                                        (
                                            (
                                                ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 +
                                                ((0::Int32 ÷ 128) % 2) * 128
                                            ) ÷ 128
                                        ) % 512
                                    ) * 16384 +
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 2) % 2) % 2) * 64 +
                                    (
                                        (
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 +
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 +
                                            ((0::Int32 ÷ 4) % 4) * 4
                                        ) ÷ 4
                                    ) % 64 +
                                    (
                                        (
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 +
                                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) *
                                            128 +
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 +
                                            ((0::Int32 ÷ 64) % 2) * 64
                                        ) % 128
                                    ) * 128
                                ) + 0
                            ) + offset,
                            length,
                        )
                    end + 0x01,
                    (Ē3_dish0_freq0_time0, Ē3_dish4_freq0_time0, Ē3_dish8_freq0_time0, Ē3_dish12_freq0_time0),
                )
            end
            if ((IndexSpaces.assume_inrange(t_outer::Int32, 0i32, 256, 65536) ÷ 256) % 256) * 256 + ((0::Int32 ÷ 128) % 2) * 128 ≥
                384
                IndexSpaces.unsafe_store4!(
                    Ē_memory,
                    let
                        offset = 16384 * T̄min - 49152
                        length = 8388608
                        mod(
                            (
                                (
                                    (
                                        (
                                            (
                                                ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 +
                                                ((0::Int32 ÷ 128) % 2) * 128
                                            ) ÷ 128
                                        ) % 512
                                    ) * 16384 +
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 2) % 2) % 2) * 64 +
                                    (
                                        (
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 +
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 +
                                            ((0::Int32 ÷ 4) % 4) * 4
                                        ) ÷ 4
                                    ) % 64 +
                                    (
                                        (
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 +
                                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) *
                                            128 +
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 +
                                            ((64::Int32 ÷ 64) % 2) * 64
                                        ) % 128
                                    ) * 128
                                ) + 0
                            ) + offset,
                            length,
                        )
                    end + 0x01,
                    (Ē3_dish0_freq64_time0, Ē3_dish4_freq64_time0, Ē3_dish8_freq64_time0, Ē3_dish12_freq64_time0),
                )
            end
            if ((IndexSpaces.assume_inrange(t_outer::Int32, 0i32, 256, 65536) ÷ 256) % 256) * 256 + ((128::Int32 ÷ 128) % 2) * 128 ≥
                384
                IndexSpaces.unsafe_store4!(
                    Ē_memory,
                    let
                        offset = 16384 * T̄min - 49152
                        length = 8388608
                        mod(
                            (
                                (
                                    (
                                        (
                                            (
                                                ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 +
                                                ((128::Int32 ÷ 128) % 2) * 128
                                            ) ÷ 128
                                        ) % 512
                                    ) * 16384 +
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 2) % 2) % 2) * 64 +
                                    (
                                        (
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 +
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 +
                                            ((0::Int32 ÷ 4) % 4) * 4
                                        ) ÷ 4
                                    ) % 64 +
                                    (
                                        (
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 +
                                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) *
                                            128 +
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 +
                                            ((0::Int32 ÷ 64) % 2) * 64
                                        ) % 128
                                    ) * 128
                                ) + 0
                            ) + offset,
                            length,
                        )
                    end + 0x01,
                    (Ē3_dish0_freq0_time128, Ē3_dish4_freq0_time128, Ē3_dish8_freq0_time128, Ē3_dish12_freq0_time128),
                )
            end
            if ((IndexSpaces.assume_inrange(t_outer::Int32, 0i32, 256, 65536) ÷ 256) % 256) * 256 + ((128::Int32 ÷ 128) % 2) * 128 ≥
                384
                IndexSpaces.unsafe_store4!(
                    Ē_memory,
                    let
                        offset = 16384 * T̄min - 49152
                        length = 8388608
                        mod(
                            (
                                (
                                    (
                                        (
                                            (
                                                ((IndexSpaces.assume_inrange(t_outer::Int32, 0, 256, 65536) ÷ 256) % 256) * 256 +
                                                ((128::Int32 ÷ 128) % 2) * 128
                                            ) ÷ 128
                                        ) % 512
                                    ) * 16384 +
                                    (((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 2) % 2) % 2) * 64 +
                                    (
                                        (
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 2) * 128 +
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 8) * 16 +
                                            ((0::Int32 ÷ 4) % 4) * 4
                                        ) ÷ 4
                                    ) % 64 +
                                    (
                                        (
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 4 +
                                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) ÷ 4) % 64) *
                                            128 +
                                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) * 4 +
                                            ((64::Int32 ÷ 64) % 2) * 64
                                        ) % 128
                                    ) * 128
                                ) + 0
                            ) + offset,
                            length,
                        )
                    end + 0x01,
                    (Ē3_dish0_freq64_time128, Ē3_dish4_freq64_time128, Ē3_dish8_freq64_time128, Ē3_dish12_freq64_time128),
                )
            end
        end
        info = 0
        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 16) % 16) % 16) * 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 256) % 256) * 512 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32) + 0) + 0x01] =
            info
    end
)
