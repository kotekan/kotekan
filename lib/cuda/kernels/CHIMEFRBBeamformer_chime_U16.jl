# Julia source code for CUDA chimefrb beamformer
# This file has been generated automatically by `chimefrb.jl`.
# Do not modify this file, your changes will be lost.

@fastmath @inbounds(
    begin #= /localhome/eschnett/src/kotekan/julia/kernels/chimefrb.jl:1329 =#
        info = 1
        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) % 8) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 256) % 256) * 256) + 0) + 0x01] =
            info
        if !(
            0i32 ≤ Tbarmin < 4096 && (
                Tbarmin ≤ Tbarmax < 8192 && (
                    (Tbarmax - Tbarmin) % 24 == 0i32 && (
                        0i32 ≤ Ttildemin < 1024 &&
                        (Ttildemin ≤ Ttildemax < 2048 && Ttildemax - Ttildemin == (Tbarmax - Tbarmin) ÷ 24)
                    )
                )
            )
        )
            info = 2
            info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) % 8) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 256) % 256) * 256) + 0) + 0x01] =
                info
            IndexSpaces.cuda_trap()
        end
        (
            Γ1_dish0_0_dish1_0_cplx_in_0_cplx_0,
            Γ1_dish0_0_dish1_0_cplx_in_0_cplx_1,
            Γ1_dish0_0_dish1_0_cplx_in_1_cplx_0,
            Γ1_dish0_0_dish1_0_cplx_in_1_cplx_1,
            Γ1_dish0_1_dish1_0_cplx_in_0_cplx_0,
            Γ1_dish0_1_dish1_0_cplx_in_0_cplx_1,
            Γ1_dish0_1_dish1_0_cplx_in_1_cplx_0,
            Γ1_dish0_1_dish1_0_cplx_in_1_cplx_1,
            Γ1_dish0_0_dish1_1_cplx_in_0_cplx_0,
            Γ1_dish0_0_dish1_1_cplx_in_0_cplx_1,
            Γ1_dish0_0_dish1_1_cplx_in_1_cplx_0,
            Γ1_dish0_0_dish1_1_cplx_in_1_cplx_1,
            Γ1_dish0_1_dish1_1_cplx_in_0_cplx_0,
            Γ1_dish0_1_dish1_1_cplx_in_0_cplx_1,
            Γ1_dish0_1_dish1_1_cplx_in_1_cplx_0,
            Γ1_dish0_1_dish1_1_cplx_in_1_cplx_1,
        ) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            thread0 = (thread ÷ (1i32)) % (2i32)
            thread1 = (thread ÷ (2i32)) % (2i32)
            thread2 = (thread ÷ (4i32)) % (2i32)
            thread3 = (thread ÷ (8i32)) % (2i32)
            thread4 = (thread ÷ (16i32)) % (2i32)
            dish0_0 = 0i32
            dish0_1 = 1i32
            dish1_0 = 0i32
            dish1_1 = 1i32
            dish6 = thread1
            dish7 = thread0
            beamp0 = thread2
            beamp1 = thread3
            beamp2 = thread4
            n_dish0_0_dish1_0 = dish0_0 * (1i32) + dish1_0 * (2i32) + dish6 * (64i32) + dish7 * (128i32)
            n_dish0_0_dish1_1 = dish0_0 * (1i32) + dish1_1 * (2i32) + dish6 * (64i32) + dish7 * (128i32)
            n_dish0_1_dish1_0 = dish0_1 * (1i32) + dish1_0 * (2i32) + dish6 * (64i32) + dish7 * (128i32)
            n_dish0_1_dish1_1 = dish0_1 * (1i32) + dish1_1 * (2i32) + dish6 * (64i32) + dish7 * (128i32)
            q = beamp0 * (1i32) + beamp1 * (2i32) + beamp2 * (4i32)
            Γ1_dish0_0_dish1_0 = cispi(((q * n_dish0_0_dish1_0) % (512i32)) * Float32(2 / 512))
            Γ1_dish0_0_dish1_1 = cispi(((q * n_dish0_0_dish1_1) % (512i32)) * Float32(2 / 512))
            Γ1_dish0_1_dish1_0 = cispi(((q * n_dish0_1_dish1_0) % (512i32)) * Float32(2 / 512))
            Γ1_dish0_1_dish1_1 = cispi(((q * n_dish0_1_dish1_1) % (512i32)) * Float32(2 / 512))
            (
                +(Γ1_dish0_0_dish1_0.re),
                +(Γ1_dish0_0_dish1_0.im),
                -(Γ1_dish0_0_dish1_0.im),
                +(Γ1_dish0_0_dish1_0.re),
                +(Γ1_dish0_1_dish1_0.re),
                +(Γ1_dish0_1_dish1_0.im),
                -(Γ1_dish0_1_dish1_0.im),
                +(Γ1_dish0_1_dish1_0.re),
                +(Γ1_dish0_0_dish1_1.re),
                +(Γ1_dish0_0_dish1_1.im),
                -(Γ1_dish0_0_dish1_1.im),
                +(Γ1_dish0_0_dish1_1.re),
                +(Γ1_dish0_1_dish1_1.re),
                +(Γ1_dish0_1_dish1_1.im),
                -(Γ1_dish0_1_dish1_1.im),
                +(Γ1_dish0_1_dish1_1.re),
            )
        end
        Γ1_dish0_0_dish1_0_cplx_0_dish0 = Float16x2(Γ1_dish0_0_dish1_0_cplx_in_0_cplx_0, Γ1_dish0_0_dish1_0_cplx_in_1_cplx_0)
        Γ1_dish0_0_dish1_0_cplx_0_dish4 = Float16x2(Γ1_dish0_0_dish1_0_cplx_in_0_cplx_0, Γ1_dish0_0_dish1_0_cplx_in_1_cplx_0)
        Γ1_dish0_0_dish1_0_cplx_1_dish0 = Float16x2(Γ1_dish0_0_dish1_0_cplx_in_0_cplx_1, Γ1_dish0_0_dish1_0_cplx_in_1_cplx_1)
        Γ1_dish0_0_dish1_0_cplx_1_dish4 = Float16x2(Γ1_dish0_0_dish1_0_cplx_in_0_cplx_1, Γ1_dish0_0_dish1_0_cplx_in_1_cplx_1)
        Γ1_dish0_0_dish1_1_cplx_0_dish0 = Float16x2(Γ1_dish0_0_dish1_1_cplx_in_0_cplx_0, Γ1_dish0_0_dish1_1_cplx_in_1_cplx_0)
        Γ1_dish0_0_dish1_1_cplx_0_dish4 = Float16x2(Γ1_dish0_0_dish1_1_cplx_in_0_cplx_0, Γ1_dish0_0_dish1_1_cplx_in_1_cplx_0)
        Γ1_dish0_0_dish1_1_cplx_1_dish0 = Float16x2(Γ1_dish0_0_dish1_1_cplx_in_0_cplx_1, Γ1_dish0_0_dish1_1_cplx_in_1_cplx_1)
        Γ1_dish0_0_dish1_1_cplx_1_dish4 = Float16x2(Γ1_dish0_0_dish1_1_cplx_in_0_cplx_1, Γ1_dish0_0_dish1_1_cplx_in_1_cplx_1)
        Γ1_dish0_1_dish1_0_cplx_0_dish0 = Float16x2(Γ1_dish0_1_dish1_0_cplx_in_0_cplx_0, Γ1_dish0_1_dish1_0_cplx_in_1_cplx_0)
        Γ1_dish0_1_dish1_0_cplx_0_dish4 = Float16x2(Γ1_dish0_1_dish1_0_cplx_in_0_cplx_0, Γ1_dish0_1_dish1_0_cplx_in_1_cplx_0)
        Γ1_dish0_1_dish1_0_cplx_1_dish0 = Float16x2(Γ1_dish0_1_dish1_0_cplx_in_0_cplx_1, Γ1_dish0_1_dish1_0_cplx_in_1_cplx_1)
        Γ1_dish0_1_dish1_0_cplx_1_dish4 = Float16x2(Γ1_dish0_1_dish1_0_cplx_in_0_cplx_1, Γ1_dish0_1_dish1_0_cplx_in_1_cplx_1)
        Γ1_dish0_1_dish1_1_cplx_0_dish0 = Float16x2(Γ1_dish0_1_dish1_1_cplx_in_0_cplx_0, Γ1_dish0_1_dish1_1_cplx_in_1_cplx_0)
        Γ1_dish0_1_dish1_1_cplx_0_dish4 = Float16x2(Γ1_dish0_1_dish1_1_cplx_in_0_cplx_0, Γ1_dish0_1_dish1_1_cplx_in_1_cplx_0)
        Γ1_dish0_1_dish1_1_cplx_1_dish0 = Float16x2(Γ1_dish0_1_dish1_1_cplx_in_0_cplx_1, Γ1_dish0_1_dish1_1_cplx_in_1_cplx_1)
        Γ1_dish0_1_dish1_1_cplx_1_dish4 = Float16x2(Γ1_dish0_1_dish1_1_cplx_in_0_cplx_1, Γ1_dish0_1_dish1_1_cplx_in_1_cplx_1)
        Γ1_dish0_0_dish1_0_cplx0_dish0 = Γ1_dish0_0_dish1_0_cplx_0_dish0
        Γ1_dish0_0_dish1_0_cplx1_dish0 = Γ1_dish0_0_dish1_0_cplx_1_dish0
        Γ1_dish0_0_dish1_0_cplx0_dish4 = Γ1_dish0_0_dish1_0_cplx_0_dish4
        Γ1_dish0_0_dish1_0_cplx1_dish4 = Γ1_dish0_0_dish1_0_cplx_1_dish4
        Γ1_dish0_0_dish1_1_cplx0_dish0 = Γ1_dish0_0_dish1_1_cplx_0_dish0
        Γ1_dish0_0_dish1_1_cplx1_dish0 = Γ1_dish0_0_dish1_1_cplx_1_dish0
        Γ1_dish0_0_dish1_1_cplx0_dish4 = Γ1_dish0_0_dish1_1_cplx_0_dish4
        Γ1_dish0_0_dish1_1_cplx1_dish4 = Γ1_dish0_0_dish1_1_cplx_1_dish4
        Γ1_dish0_1_dish1_0_cplx0_dish0 = Γ1_dish0_1_dish1_0_cplx_0_dish0
        Γ1_dish0_1_dish1_0_cplx1_dish0 = Γ1_dish0_1_dish1_0_cplx_1_dish0
        Γ1_dish0_1_dish1_0_cplx0_dish4 = Γ1_dish0_1_dish1_0_cplx_0_dish4
        Γ1_dish0_1_dish1_0_cplx1_dish4 = Γ1_dish0_1_dish1_0_cplx_1_dish4
        Γ1_dish0_1_dish1_1_cplx0_dish0 = Γ1_dish0_1_dish1_1_cplx_0_dish0
        Γ1_dish0_1_dish1_1_cplx1_dish0 = Γ1_dish0_1_dish1_1_cplx_1_dish0
        Γ1_dish0_1_dish1_1_cplx0_dish4 = Γ1_dish0_1_dish1_1_cplx_0_dish4
        Γ1_dish0_1_dish1_1_cplx1_dish4 = Γ1_dish0_1_dish1_1_cplx_1_dish4
        Γ1_dish0_0_cplx0_dish0 = Γ1_dish0_0_dish1_0_cplx0_dish0
        Γ1_dish0_0_cplx0_dish2 = Γ1_dish0_0_dish1_1_cplx0_dish0
        Γ1_dish0_0_cplx1_dish0 = Γ1_dish0_0_dish1_0_cplx1_dish0
        Γ1_dish0_0_cplx1_dish2 = Γ1_dish0_0_dish1_1_cplx1_dish0
        Γ1_dish0_0_cplx0_dish4 = Γ1_dish0_0_dish1_0_cplx0_dish4
        Γ1_dish0_0_cplx0_dish6 = Γ1_dish0_0_dish1_1_cplx0_dish4
        Γ1_dish0_0_cplx1_dish4 = Γ1_dish0_0_dish1_0_cplx1_dish4
        Γ1_dish0_0_cplx1_dish6 = Γ1_dish0_0_dish1_1_cplx1_dish4
        Γ1_dish0_1_cplx0_dish0 = Γ1_dish0_1_dish1_0_cplx0_dish0
        Γ1_dish0_1_cplx0_dish2 = Γ1_dish0_1_dish1_1_cplx0_dish0
        Γ1_dish0_1_cplx1_dish0 = Γ1_dish0_1_dish1_0_cplx1_dish0
        Γ1_dish0_1_cplx1_dish2 = Γ1_dish0_1_dish1_1_cplx1_dish0
        Γ1_dish0_1_cplx0_dish4 = Γ1_dish0_1_dish1_0_cplx0_dish4
        Γ1_dish0_1_cplx0_dish6 = Γ1_dish0_1_dish1_1_cplx0_dish4
        Γ1_dish0_1_cplx1_dish4 = Γ1_dish0_1_dish1_0_cplx1_dish4
        Γ1_dish0_1_cplx1_dish6 = Γ1_dish0_1_dish1_1_cplx1_dish4
        Γ1_cplx0_dish0 = Γ1_dish0_0_cplx0_dish0
        Γ1_cplx0_dish1 = Γ1_dish0_1_cplx0_dish0
        Γ1_cplx1_dish0 = Γ1_dish0_0_cplx1_dish0
        Γ1_cplx1_dish1 = Γ1_dish0_1_cplx1_dish0
        Γ1_cplx0_dish2 = Γ1_dish0_0_cplx0_dish2
        Γ1_cplx0_dish3 = Γ1_dish0_1_cplx0_dish2
        Γ1_cplx1_dish2 = Γ1_dish0_0_cplx1_dish2
        Γ1_cplx1_dish3 = Γ1_dish0_1_cplx1_dish2
        Γ1_cplx0_dish4 = Γ1_dish0_0_cplx0_dish4
        Γ1_cplx0_dish5 = Γ1_dish0_1_cplx0_dish4
        Γ1_cplx1_dish4 = Γ1_dish0_0_cplx1_dish4
        Γ1_cplx1_dish5 = Γ1_dish0_1_cplx1_dish4
        Γ1_cplx0_dish6 = Γ1_dish0_0_cplx0_dish6
        Γ1_cplx0_dish7 = Γ1_dish0_1_cplx0_dish6
        Γ1_cplx1_dish6 = Γ1_dish0_0_cplx1_dish6
        Γ1_cplx1_dish7 = Γ1_dish0_1_cplx1_dish6
        (
            Γ2_dish2_0_dish5_0_cplx_0,
            Γ2_dish2_0_dish5_0_cplx_1,
            Γ2_dish2_1_dish5_0_cplx_0,
            Γ2_dish2_1_dish5_0_cplx_1,
            Γ2_dish2_0_dish5_1_cplx_0,
            Γ2_dish2_0_dish5_1_cplx_1,
            Γ2_dish2_1_dish5_1_cplx_0,
            Γ2_dish2_1_dish5_1_cplx_1,
        ) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            thread0 = (thread ÷ (1i32)) % (2i32)
            thread1 = (thread ÷ (2i32)) % (2i32)
            thread2 = (thread ÷ (4i32)) % (2i32)
            thread3 = (thread ÷ (8i32)) % (2i32)
            thread4 = (thread ÷ (16i32)) % (2i32)
            dish2_0 = 0i32
            dish2_1 = 1i32
            dish3 = thread1
            dish4 = thread0
            dish5_0 = 0i32
            dish5_1 = 1i32
            beamp0 = thread2
            beamp1 = thread3
            beamp2 = thread4
            n_dish2_0_dish5_0 = dish2_0 * (4i32) + dish3 * (8i32) + dish4 * (16i32) + dish5_0 * (32i32)
            n_dish2_0_dish5_1 = dish2_0 * (4i32) + dish3 * (8i32) + dish4 * (16i32) + dish5_1 * (32i32)
            n_dish2_1_dish5_0 = dish2_1 * (4i32) + dish3 * (8i32) + dish4 * (16i32) + dish5_0 * (32i32)
            n_dish2_1_dish5_1 = dish2_1 * (4i32) + dish3 * (8i32) + dish4 * (16i32) + dish5_1 * (32i32)
            q = beamp0 * (1i32) + beamp1 * (2i32) + beamp2 * (4i32)
            Γ2_dish2_0_dish5_0 = cispi(((q * n_dish2_0_dish5_0) % (512i32)) * Float32(2 / 512))
            Γ2_dish2_0_dish5_1 = cispi(((q * n_dish2_0_dish5_1) % (512i32)) * Float32(2 / 512))
            Γ2_dish2_1_dish5_0 = cispi(((q * n_dish2_1_dish5_0) % (512i32)) * Float32(2 / 512))
            Γ2_dish2_1_dish5_1 = cispi(((q * n_dish2_1_dish5_1) % (512i32)) * Float32(2 / 512))
            (
                Γ2_dish2_0_dish5_0.re,
                Γ2_dish2_0_dish5_0.im,
                Γ2_dish2_1_dish5_0.re,
                Γ2_dish2_1_dish5_0.im,
                Γ2_dish2_1_dish5_0.re,
                Γ2_dish2_0_dish5_0.im,
                Γ2_dish2_0_dish5_0.re,
                Γ2_dish2_1_dish5_0.im,
                Γ2_dish2_1_dish5_0.re,
                Γ2_dish2_0_dish5_1.im,
                Γ2_dish2_0_dish5_1.re,
                Γ2_dish2_1_dish5_1.im,
                Γ2_dish2_0_dish5_1.re,
                Γ2_dish2_0_dish5_1.im,
                Γ2_dish2_1_dish5_1.re,
                Γ2_dish2_1_dish5_1.im,
            )
        end
        Γ2_dish2_0_cplx_0_dish0 = Float16x2(Γ2_dish2_0_dish5_0_cplx_0, Γ2_dish2_0_dish5_1_cplx_0)
        Γ2_dish2_0_cplx_0_dish1 = Float16x2(Γ2_dish2_0_dish5_0_cplx_0, Γ2_dish2_0_dish5_1_cplx_0)
        Γ2_dish2_0_cplx_0_dish2 = Float16x2(Γ2_dish2_0_dish5_0_cplx_0, Γ2_dish2_0_dish5_1_cplx_0)
        Γ2_dish2_0_cplx_0_dish3 = Float16x2(Γ2_dish2_0_dish5_0_cplx_0, Γ2_dish2_0_dish5_1_cplx_0)
        Γ2_dish2_0_cplx_1_dish0 = Float16x2(Γ2_dish2_0_dish5_0_cplx_1, Γ2_dish2_0_dish5_1_cplx_1)
        Γ2_dish2_0_cplx_1_dish1 = Float16x2(Γ2_dish2_0_dish5_0_cplx_1, Γ2_dish2_0_dish5_1_cplx_1)
        Γ2_dish2_0_cplx_1_dish2 = Float16x2(Γ2_dish2_0_dish5_0_cplx_1, Γ2_dish2_0_dish5_1_cplx_1)
        Γ2_dish2_0_cplx_1_dish3 = Float16x2(Γ2_dish2_0_dish5_0_cplx_1, Γ2_dish2_0_dish5_1_cplx_1)
        Γ2_dish2_1_cplx_0_dish0 = Float16x2(Γ2_dish2_1_dish5_0_cplx_0, Γ2_dish2_1_dish5_1_cplx_0)
        Γ2_dish2_1_cplx_0_dish1 = Float16x2(Γ2_dish2_1_dish5_0_cplx_0, Γ2_dish2_1_dish5_1_cplx_0)
        Γ2_dish2_1_cplx_0_dish2 = Float16x2(Γ2_dish2_1_dish5_0_cplx_0, Γ2_dish2_1_dish5_1_cplx_0)
        Γ2_dish2_1_cplx_0_dish3 = Float16x2(Γ2_dish2_1_dish5_0_cplx_0, Γ2_dish2_1_dish5_1_cplx_0)
        Γ2_dish2_1_cplx_1_dish0 = Float16x2(Γ2_dish2_1_dish5_0_cplx_1, Γ2_dish2_1_dish5_1_cplx_1)
        Γ2_dish2_1_cplx_1_dish1 = Float16x2(Γ2_dish2_1_dish5_0_cplx_1, Γ2_dish2_1_dish5_1_cplx_1)
        Γ2_dish2_1_cplx_1_dish2 = Float16x2(Γ2_dish2_1_dish5_0_cplx_1, Γ2_dish2_1_dish5_1_cplx_1)
        Γ2_dish2_1_cplx_1_dish3 = Float16x2(Γ2_dish2_1_dish5_0_cplx_1, Γ2_dish2_1_dish5_1_cplx_1)
        Γ2_dish2_0_cplx0_dish0 = Γ2_dish2_0_cplx_0_dish0
        Γ2_dish2_0_cplx1_dish0 = Γ2_dish2_0_cplx_1_dish0
        Γ2_dish2_0_cplx0_dish1 = Γ2_dish2_0_cplx_0_dish1
        Γ2_dish2_0_cplx1_dish1 = Γ2_dish2_0_cplx_1_dish1
        Γ2_dish2_0_cplx0_dish2 = Γ2_dish2_0_cplx_0_dish2
        Γ2_dish2_0_cplx1_dish2 = Γ2_dish2_0_cplx_1_dish2
        Γ2_dish2_0_cplx0_dish3 = Γ2_dish2_0_cplx_0_dish3
        Γ2_dish2_0_cplx1_dish3 = Γ2_dish2_0_cplx_1_dish3
        Γ2_dish2_1_cplx0_dish0 = Γ2_dish2_1_cplx_0_dish0
        Γ2_dish2_1_cplx1_dish0 = Γ2_dish2_1_cplx_1_dish0
        Γ2_dish2_1_cplx0_dish1 = Γ2_dish2_1_cplx_0_dish1
        Γ2_dish2_1_cplx1_dish1 = Γ2_dish2_1_cplx_1_dish1
        Γ2_dish2_1_cplx0_dish2 = Γ2_dish2_1_cplx_0_dish2
        Γ2_dish2_1_cplx1_dish2 = Γ2_dish2_1_cplx_1_dish2
        Γ2_dish2_1_cplx0_dish3 = Γ2_dish2_1_cplx_0_dish3
        Γ2_dish2_1_cplx1_dish3 = Γ2_dish2_1_cplx_1_dish3
        Γ2_cplx0_dish0 = Γ2_dish2_0_cplx0_dish0
        Γ2_cplx0_dish4 = Γ2_dish2_1_cplx0_dish0
        Γ2_cplx1_dish0 = Γ2_dish2_0_cplx1_dish0
        Γ2_cplx1_dish4 = Γ2_dish2_1_cplx1_dish0
        Γ2_cplx0_dish1 = Γ2_dish2_0_cplx0_dish1
        Γ2_cplx0_dish5 = Γ2_dish2_1_cplx0_dish1
        Γ2_cplx1_dish1 = Γ2_dish2_0_cplx1_dish1
        Γ2_cplx1_dish5 = Γ2_dish2_1_cplx1_dish1
        Γ2_cplx0_dish2 = Γ2_dish2_0_cplx0_dish2
        Γ2_cplx0_dish6 = Γ2_dish2_1_cplx0_dish2
        Γ2_cplx1_dish2 = Γ2_dish2_0_cplx1_dish2
        Γ2_cplx1_dish6 = Γ2_dish2_1_cplx1_dish2
        Γ2_cplx0_dish3 = Γ2_dish2_0_cplx0_dish3
        Γ2_cplx0_dish7 = Γ2_dish2_1_cplx0_dish3
        Γ2_cplx1_dish3 = Γ2_dish2_0_cplx1_dish3
        Γ2_cplx1_dish7 = Γ2_dish2_1_cplx1_dish3
        (
            Γ3_dish5_0_cplx_in_0_cplx_0,
            Γ3_dish5_0_cplx_in_1_cplx_0,
            Γ3_dish5_0_cplx_in_0_cplx_1,
            Γ3_dish5_0_cplx_in_1_cplx_1,
            Γ3_dish5_1_cplx_in_0_cplx_0,
            Γ3_dish5_1_cplx_in_1_cplx_0,
            Γ3_dish5_1_cplx_in_0_cplx_1,
            Γ3_dish5_1_cplx_in_1_cplx_1,
        ) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            thread0 = (thread ÷ (1i32)) % (2i32)
            thread1 = (thread ÷ (2i32)) % (2i32)
            thread2 = (thread ÷ (4i32)) % (2i32)
            thread3 = (thread ÷ (8i32)) % (2i32)
            thread4 = (thread ÷ (16i32)) % (2i32)
            dish3 = thread1
            dish4 = thread0
            dish5_0 = 0i32
            dish5_1 = 1i32
            beamp3 = thread2
            beamp4 = thread3
            beamp5 = thread4
            n_dish5_0 = dish3 * (8i32) + dish4 * (16i32) + dish5_0 * (32i32)
            n_dish5_1 = dish3 * (8i32) + dish4 * (16i32) + dish5_1 * (32i32)
            q = beamp3 * (8i32) + beamp4 * (16i32) + beamp5 * (32i32)
            Γ3_dish5_0 = cispi(((q * n_dish5_0) % (512i32)) * Float32(2 / 512))
            Γ3_dish5_1 = cispi(((q * n_dish5_1) % (512i32)) * Float32(2 / 512))
            (
                +(Γ3_dish5_0.re),
                +(Γ3_dish5_0.im),
                -(Γ3_dish5_0.im),
                +(Γ3_dish5_0.re),
                +(Γ3_dish5_1.re),
                +(Γ3_dish5_1.im),
                -(Γ3_dish5_1.im),
                +(Γ3_dish5_1.re),
            )
        end
        Γ3_cplx_in_0_cplx_0_dish0 = Float16x2(Γ3_dish5_0_cplx_in_0_cplx_0, Γ3_dish5_1_cplx_in_0_cplx_0)
        Γ3_cplx_in_0_cplx_0_dish1 = Float16x2(Γ3_dish5_0_cplx_in_0_cplx_0, Γ3_dish5_1_cplx_in_0_cplx_0)
        Γ3_cplx_in_0_cplx_0_dish2 = Float16x2(Γ3_dish5_0_cplx_in_0_cplx_0, Γ3_dish5_1_cplx_in_0_cplx_0)
        Γ3_cplx_in_0_cplx_0_dish3 = Float16x2(Γ3_dish5_0_cplx_in_0_cplx_0, Γ3_dish5_1_cplx_in_0_cplx_0)
        Γ3_cplx_in_0_cplx_0_dish4 = Float16x2(Γ3_dish5_0_cplx_in_0_cplx_0, Γ3_dish5_1_cplx_in_0_cplx_0)
        Γ3_cplx_in_0_cplx_0_dish5 = Float16x2(Γ3_dish5_0_cplx_in_0_cplx_0, Γ3_dish5_1_cplx_in_0_cplx_0)
        Γ3_cplx_in_0_cplx_0_dish6 = Float16x2(Γ3_dish5_0_cplx_in_0_cplx_0, Γ3_dish5_1_cplx_in_0_cplx_0)
        Γ3_cplx_in_0_cplx_0_dish7 = Float16x2(Γ3_dish5_0_cplx_in_0_cplx_0, Γ3_dish5_1_cplx_in_0_cplx_0)
        Γ3_cplx_in_0_cplx_1_dish0 = Float16x2(Γ3_dish5_0_cplx_in_0_cplx_1, Γ3_dish5_1_cplx_in_0_cplx_1)
        Γ3_cplx_in_0_cplx_1_dish1 = Float16x2(Γ3_dish5_0_cplx_in_0_cplx_1, Γ3_dish5_1_cplx_in_0_cplx_1)
        Γ3_cplx_in_0_cplx_1_dish2 = Float16x2(Γ3_dish5_0_cplx_in_0_cplx_1, Γ3_dish5_1_cplx_in_0_cplx_1)
        Γ3_cplx_in_0_cplx_1_dish3 = Float16x2(Γ3_dish5_0_cplx_in_0_cplx_1, Γ3_dish5_1_cplx_in_0_cplx_1)
        Γ3_cplx_in_0_cplx_1_dish4 = Float16x2(Γ3_dish5_0_cplx_in_0_cplx_1, Γ3_dish5_1_cplx_in_0_cplx_1)
        Γ3_cplx_in_0_cplx_1_dish5 = Float16x2(Γ3_dish5_0_cplx_in_0_cplx_1, Γ3_dish5_1_cplx_in_0_cplx_1)
        Γ3_cplx_in_0_cplx_1_dish6 = Float16x2(Γ3_dish5_0_cplx_in_0_cplx_1, Γ3_dish5_1_cplx_in_0_cplx_1)
        Γ3_cplx_in_0_cplx_1_dish7 = Float16x2(Γ3_dish5_0_cplx_in_0_cplx_1, Γ3_dish5_1_cplx_in_0_cplx_1)
        Γ3_cplx_in_1_cplx_0_dish0 = Float16x2(Γ3_dish5_0_cplx_in_1_cplx_0, Γ3_dish5_1_cplx_in_1_cplx_0)
        Γ3_cplx_in_1_cplx_0_dish1 = Float16x2(Γ3_dish5_0_cplx_in_1_cplx_0, Γ3_dish5_1_cplx_in_1_cplx_0)
        Γ3_cplx_in_1_cplx_0_dish2 = Float16x2(Γ3_dish5_0_cplx_in_1_cplx_0, Γ3_dish5_1_cplx_in_1_cplx_0)
        Γ3_cplx_in_1_cplx_0_dish3 = Float16x2(Γ3_dish5_0_cplx_in_1_cplx_0, Γ3_dish5_1_cplx_in_1_cplx_0)
        Γ3_cplx_in_1_cplx_0_dish4 = Float16x2(Γ3_dish5_0_cplx_in_1_cplx_0, Γ3_dish5_1_cplx_in_1_cplx_0)
        Γ3_cplx_in_1_cplx_0_dish5 = Float16x2(Γ3_dish5_0_cplx_in_1_cplx_0, Γ3_dish5_1_cplx_in_1_cplx_0)
        Γ3_cplx_in_1_cplx_0_dish6 = Float16x2(Γ3_dish5_0_cplx_in_1_cplx_0, Γ3_dish5_1_cplx_in_1_cplx_0)
        Γ3_cplx_in_1_cplx_0_dish7 = Float16x2(Γ3_dish5_0_cplx_in_1_cplx_0, Γ3_dish5_1_cplx_in_1_cplx_0)
        Γ3_cplx_in_1_cplx_1_dish0 = Float16x2(Γ3_dish5_0_cplx_in_1_cplx_1, Γ3_dish5_1_cplx_in_1_cplx_1)
        Γ3_cplx_in_1_cplx_1_dish1 = Float16x2(Γ3_dish5_0_cplx_in_1_cplx_1, Γ3_dish5_1_cplx_in_1_cplx_1)
        Γ3_cplx_in_1_cplx_1_dish2 = Float16x2(Γ3_dish5_0_cplx_in_1_cplx_1, Γ3_dish5_1_cplx_in_1_cplx_1)
        Γ3_cplx_in_1_cplx_1_dish3 = Float16x2(Γ3_dish5_0_cplx_in_1_cplx_1, Γ3_dish5_1_cplx_in_1_cplx_1)
        Γ3_cplx_in_1_cplx_1_dish4 = Float16x2(Γ3_dish5_0_cplx_in_1_cplx_1, Γ3_dish5_1_cplx_in_1_cplx_1)
        Γ3_cplx_in_1_cplx_1_dish5 = Float16x2(Γ3_dish5_0_cplx_in_1_cplx_1, Γ3_dish5_1_cplx_in_1_cplx_1)
        Γ3_cplx_in_1_cplx_1_dish6 = Float16x2(Γ3_dish5_0_cplx_in_1_cplx_1, Γ3_dish5_1_cplx_in_1_cplx_1)
        Γ3_cplx_in_1_cplx_1_dish7 = Float16x2(Γ3_dish5_0_cplx_in_1_cplx_1, Γ3_dish5_1_cplx_in_1_cplx_1)
        Γ3_cplx_in_0_cplx0_dish0 = Γ3_cplx_in_0_cplx_0_dish0
        Γ3_cplx_in_0_cplx1_dish0 = Γ3_cplx_in_0_cplx_1_dish0
        Γ3_cplx_in_0_cplx0_dish1 = Γ3_cplx_in_0_cplx_0_dish1
        Γ3_cplx_in_0_cplx1_dish1 = Γ3_cplx_in_0_cplx_1_dish1
        Γ3_cplx_in_0_cplx0_dish2 = Γ3_cplx_in_0_cplx_0_dish2
        Γ3_cplx_in_0_cplx1_dish2 = Γ3_cplx_in_0_cplx_1_dish2
        Γ3_cplx_in_0_cplx0_dish3 = Γ3_cplx_in_0_cplx_0_dish3
        Γ3_cplx_in_0_cplx1_dish3 = Γ3_cplx_in_0_cplx_1_dish3
        Γ3_cplx_in_0_cplx0_dish4 = Γ3_cplx_in_0_cplx_0_dish4
        Γ3_cplx_in_0_cplx1_dish4 = Γ3_cplx_in_0_cplx_1_dish4
        Γ3_cplx_in_0_cplx0_dish5 = Γ3_cplx_in_0_cplx_0_dish5
        Γ3_cplx_in_0_cplx1_dish5 = Γ3_cplx_in_0_cplx_1_dish5
        Γ3_cplx_in_0_cplx0_dish6 = Γ3_cplx_in_0_cplx_0_dish6
        Γ3_cplx_in_0_cplx1_dish6 = Γ3_cplx_in_0_cplx_1_dish6
        Γ3_cplx_in_0_cplx0_dish7 = Γ3_cplx_in_0_cplx_0_dish7
        Γ3_cplx_in_0_cplx1_dish7 = Γ3_cplx_in_0_cplx_1_dish7
        Γ3_cplx_in_1_cplx0_dish0 = Γ3_cplx_in_1_cplx_0_dish0
        Γ3_cplx_in_1_cplx1_dish0 = Γ3_cplx_in_1_cplx_1_dish0
        Γ3_cplx_in_1_cplx0_dish1 = Γ3_cplx_in_1_cplx_0_dish1
        Γ3_cplx_in_1_cplx1_dish1 = Γ3_cplx_in_1_cplx_1_dish1
        Γ3_cplx_in_1_cplx0_dish2 = Γ3_cplx_in_1_cplx_0_dish2
        Γ3_cplx_in_1_cplx1_dish2 = Γ3_cplx_in_1_cplx_1_dish2
        Γ3_cplx_in_1_cplx0_dish3 = Γ3_cplx_in_1_cplx_0_dish3
        Γ3_cplx_in_1_cplx1_dish3 = Γ3_cplx_in_1_cplx_1_dish3
        Γ3_cplx_in_1_cplx0_dish4 = Γ3_cplx_in_1_cplx_0_dish4
        Γ3_cplx_in_1_cplx1_dish4 = Γ3_cplx_in_1_cplx_1_dish4
        Γ3_cplx_in_1_cplx0_dish5 = Γ3_cplx_in_1_cplx_0_dish5
        Γ3_cplx_in_1_cplx1_dish5 = Γ3_cplx_in_1_cplx_1_dish5
        Γ3_cplx_in_1_cplx0_dish6 = Γ3_cplx_in_1_cplx_0_dish6
        Γ3_cplx_in_1_cplx1_dish6 = Γ3_cplx_in_1_cplx_1_dish6
        Γ3_cplx_in_1_cplx0_dish7 = Γ3_cplx_in_1_cplx_0_dish7
        Γ3_cplx_in_1_cplx1_dish7 = Γ3_cplx_in_1_cplx_1_dish7
        Γ3_cplx0_cplx_in0_dish0 = Γ3_cplx_in_0_cplx0_dish0
        Γ3_cplx0_cplx_in1_dish0 = Γ3_cplx_in_1_cplx0_dish0
        Γ3_cplx1_cplx_in0_dish0 = Γ3_cplx_in_0_cplx1_dish0
        Γ3_cplx1_cplx_in1_dish0 = Γ3_cplx_in_1_cplx1_dish0
        Γ3_cplx0_cplx_in0_dish1 = Γ3_cplx_in_0_cplx0_dish1
        Γ3_cplx0_cplx_in1_dish1 = Γ3_cplx_in_1_cplx0_dish1
        Γ3_cplx1_cplx_in0_dish1 = Γ3_cplx_in_0_cplx1_dish1
        Γ3_cplx1_cplx_in1_dish1 = Γ3_cplx_in_1_cplx1_dish1
        Γ3_cplx0_cplx_in0_dish2 = Γ3_cplx_in_0_cplx0_dish2
        Γ3_cplx0_cplx_in1_dish2 = Γ3_cplx_in_1_cplx0_dish2
        Γ3_cplx1_cplx_in0_dish2 = Γ3_cplx_in_0_cplx1_dish2
        Γ3_cplx1_cplx_in1_dish2 = Γ3_cplx_in_1_cplx1_dish2
        Γ3_cplx0_cplx_in0_dish3 = Γ3_cplx_in_0_cplx0_dish3
        Γ3_cplx0_cplx_in1_dish3 = Γ3_cplx_in_1_cplx0_dish3
        Γ3_cplx1_cplx_in0_dish3 = Γ3_cplx_in_0_cplx1_dish3
        Γ3_cplx1_cplx_in1_dish3 = Γ3_cplx_in_1_cplx1_dish3
        Γ3_cplx0_cplx_in0_dish4 = Γ3_cplx_in_0_cplx0_dish4
        Γ3_cplx0_cplx_in1_dish4 = Γ3_cplx_in_1_cplx0_dish4
        Γ3_cplx1_cplx_in0_dish4 = Γ3_cplx_in_0_cplx1_dish4
        Γ3_cplx1_cplx_in1_dish4 = Γ3_cplx_in_1_cplx1_dish4
        Γ3_cplx0_cplx_in0_dish5 = Γ3_cplx_in_0_cplx0_dish5
        Γ3_cplx0_cplx_in1_dish5 = Γ3_cplx_in_1_cplx0_dish5
        Γ3_cplx1_cplx_in0_dish5 = Γ3_cplx_in_0_cplx1_dish5
        Γ3_cplx1_cplx_in1_dish5 = Γ3_cplx_in_1_cplx1_dish5
        Γ3_cplx0_cplx_in0_dish6 = Γ3_cplx_in_0_cplx0_dish6
        Γ3_cplx0_cplx_in1_dish6 = Γ3_cplx_in_1_cplx0_dish6
        Γ3_cplx1_cplx_in0_dish6 = Γ3_cplx_in_0_cplx1_dish6
        Γ3_cplx1_cplx_in1_dish6 = Γ3_cplx_in_1_cplx1_dish6
        Γ3_cplx0_cplx_in0_dish7 = Γ3_cplx_in_0_cplx0_dish7
        Γ3_cplx0_cplx_in1_dish7 = Γ3_cplx_in_1_cplx0_dish7
        Γ3_cplx1_cplx_in0_dish7 = Γ3_cplx_in_0_cplx1_dish7
        Γ3_cplx1_cplx_in1_dish7 = Γ3_cplx_in_1_cplx1_dish7
        (Γ4_dish2_0_cplx_0, Γ4_dish2_0_cplx_1, Γ4_dish2_1_cplx_0, Γ4_dish2_1_cplx_1) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            thread0 = (thread ÷ (1i32)) % (2i32)
            thread1 = (thread ÷ (2i32)) % (2i32)
            thread2 = (thread ÷ (4i32)) % (2i32)
            thread3 = (thread ÷ (8i32)) % (2i32)
            thread4 = (thread ÷ (16i32)) % (2i32)
            dish0 = thread1
            dish1 = thread0
            dish2_0 = 0i32
            dish2_1 = 1i32
            beamp3 = thread2
            beamp4 = thread3
            beamp5 = thread4
            n_dish2_0 = dish0 * (1i32) + dish1 * (2i32) + dish2_0 * (4i32)
            n_dish2_1 = dish0 * (1i32) + dish1 * (2i32) + dish2_1 * (4i32)
            q = beamp3 * (8i32) + beamp4 * (16i32) + beamp5 * (32i32)
            Γ4_dish2_0 = cispi(((q * n_dish2_0) % (512i32)) * Float32(2 / 512))
            Γ4_dish2_1 = cispi(((q * n_dish2_1) % (512i32)) * Float32(2 / 512))
            (
                Γ4_dish2_0.re,
                Γ4_dish2_0.im,
                Γ4_dish2_1.re,
                Γ4_dish2_1.im,
                Γ4_dish2_1.re,
                Γ4_dish2_0.im,
                Γ4_dish2_0.re,
                Γ4_dish2_1.im,
                Γ4_dish2_1.re,
            )
        end
        Γ4_cplx_0_beamP0 = Float16x2(Γ4_dish2_0_cplx_0, Γ4_dish2_1_cplx_0)
        Γ4_cplx_0_beamP1 = Float16x2(Γ4_dish2_0_cplx_0, Γ4_dish2_1_cplx_0)
        Γ4_cplx_0_beamP2 = Float16x2(Γ4_dish2_0_cplx_0, Γ4_dish2_1_cplx_0)
        Γ4_cplx_0_beamP3 = Float16x2(Γ4_dish2_0_cplx_0, Γ4_dish2_1_cplx_0)
        Γ4_cplx_0_beamP4 = Float16x2(Γ4_dish2_0_cplx_0, Γ4_dish2_1_cplx_0)
        Γ4_cplx_0_beamP5 = Float16x2(Γ4_dish2_0_cplx_0, Γ4_dish2_1_cplx_0)
        Γ4_cplx_0_beamP6 = Float16x2(Γ4_dish2_0_cplx_0, Γ4_dish2_1_cplx_0)
        Γ4_cplx_0_beamP7 = Float16x2(Γ4_dish2_0_cplx_0, Γ4_dish2_1_cplx_0)
        Γ4_cplx_1_beamP0 = Float16x2(Γ4_dish2_0_cplx_1, Γ4_dish2_1_cplx_1)
        Γ4_cplx_1_beamP1 = Float16x2(Γ4_dish2_0_cplx_1, Γ4_dish2_1_cplx_1)
        Γ4_cplx_1_beamP2 = Float16x2(Γ4_dish2_0_cplx_1, Γ4_dish2_1_cplx_1)
        Γ4_cplx_1_beamP3 = Float16x2(Γ4_dish2_0_cplx_1, Γ4_dish2_1_cplx_1)
        Γ4_cplx_1_beamP4 = Float16x2(Γ4_dish2_0_cplx_1, Γ4_dish2_1_cplx_1)
        Γ4_cplx_1_beamP5 = Float16x2(Γ4_dish2_0_cplx_1, Γ4_dish2_1_cplx_1)
        Γ4_cplx_1_beamP6 = Float16x2(Γ4_dish2_0_cplx_1, Γ4_dish2_1_cplx_1)
        Γ4_cplx_1_beamP7 = Float16x2(Γ4_dish2_0_cplx_1, Γ4_dish2_1_cplx_1)
        Γ4_beamP0_cplx0 = Γ4_cplx_0_beamP0
        Γ4_beamP0_cplx1 = Γ4_cplx_1_beamP0
        Γ4_beamP1_cplx0 = Γ4_cplx_0_beamP1
        Γ4_beamP1_cplx1 = Γ4_cplx_1_beamP1
        Γ4_beamP2_cplx0 = Γ4_cplx_0_beamP2
        Γ4_beamP2_cplx1 = Γ4_cplx_1_beamP2
        Γ4_beamP3_cplx0 = Γ4_cplx_0_beamP3
        Γ4_beamP3_cplx1 = Γ4_cplx_1_beamP3
        Γ4_beamP4_cplx0 = Γ4_cplx_0_beamP4
        Γ4_beamP4_cplx1 = Γ4_cplx_1_beamP4
        Γ4_beamP5_cplx0 = Γ4_cplx_0_beamP5
        Γ4_beamP5_cplx1 = Γ4_cplx_1_beamP5
        Γ4_beamP6_cplx0 = Γ4_cplx_0_beamP6
        Γ4_beamP6_cplx1 = Γ4_cplx_1_beamP6
        Γ4_beamP7_cplx0 = Γ4_cplx_0_beamP7
        Γ4_beamP7_cplx1 = Γ4_cplx_1_beamP7
        (
            Γ5_dish2_0_cplx_in_0_cplx_0,
            Γ5_dish2_0_cplx_in_1_cplx_0,
            Γ5_dish2_0_cplx_in_0_cplx_1,
            Γ5_dish2_0_cplx_in_1_cplx_1,
            Γ5_dish2_1_cplx_in_0_cplx_0,
            Γ5_dish2_1_cplx_in_1_cplx_0,
            Γ5_dish2_1_cplx_in_0_cplx_1,
            Γ5_dish2_1_cplx_in_1_cplx_1,
        ) = let
            thread = IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32)
            thread0 = (thread ÷ (1i32)) % (2i32)
            thread1 = (thread ÷ (2i32)) % (2i32)
            thread2 = (thread ÷ (4i32)) % (2i32)
            thread3 = (thread ÷ (8i32)) % (2i32)
            thread4 = (thread ÷ (16i32)) % (2i32)
            dish0 = thread1
            dish1 = thread0
            dish2_0 = 0i32
            dish2_1 = 1i32
            beamp6 = thread2
            beamp7 = thread3
            beamp8 = thread4
            n_dish2_0 = dish0 * (1i32) + dish1 * (2i32) + dish2_0 * (4i32)
            n_dish2_1 = dish0 * (1i32) + dish1 * (2i32) + dish2_1 * (4i32)
            q = beamp6 * (64i32) + beamp7 * (128i32) + beamp8 * (256i32)
            Γ5_dish2_0 = cispi(((q * n_dish2_0) % (512i32)) * Float32(2 / 512))
            Γ5_dish2_1 = cispi(((q * n_dish2_1) % (512i32)) * Float32(2 / 512))
            (
                +(Γ5_dish2_0.re),
                +(Γ5_dish2_0.im),
                -(Γ5_dish2_0.im),
                +(Γ5_dish2_0.re),
                +(Γ5_dish2_1.re),
                +(Γ5_dish2_1.im),
                -(Γ5_dish2_1.im),
                +(Γ5_dish2_1.re),
            )
        end
        Γ5_cplx_in_0_cplx_0_beamP0 = Float16x2(Γ5_dish2_0_cplx_in_0_cplx_0, Γ5_dish2_1_cplx_in_0_cplx_0)
        Γ5_cplx_in_0_cplx_0_beamP1 = Float16x2(Γ5_dish2_0_cplx_in_0_cplx_0, Γ5_dish2_1_cplx_in_0_cplx_0)
        Γ5_cplx_in_0_cplx_0_beamP2 = Float16x2(Γ5_dish2_0_cplx_in_0_cplx_0, Γ5_dish2_1_cplx_in_0_cplx_0)
        Γ5_cplx_in_0_cplx_0_beamP3 = Float16x2(Γ5_dish2_0_cplx_in_0_cplx_0, Γ5_dish2_1_cplx_in_0_cplx_0)
        Γ5_cplx_in_0_cplx_0_beamP4 = Float16x2(Γ5_dish2_0_cplx_in_0_cplx_0, Γ5_dish2_1_cplx_in_0_cplx_0)
        Γ5_cplx_in_0_cplx_0_beamP5 = Float16x2(Γ5_dish2_0_cplx_in_0_cplx_0, Γ5_dish2_1_cplx_in_0_cplx_0)
        Γ5_cplx_in_0_cplx_0_beamP6 = Float16x2(Γ5_dish2_0_cplx_in_0_cplx_0, Γ5_dish2_1_cplx_in_0_cplx_0)
        Γ5_cplx_in_0_cplx_0_beamP7 = Float16x2(Γ5_dish2_0_cplx_in_0_cplx_0, Γ5_dish2_1_cplx_in_0_cplx_0)
        Γ5_cplx_in_0_cplx_1_beamP0 = Float16x2(Γ5_dish2_0_cplx_in_0_cplx_1, Γ5_dish2_1_cplx_in_0_cplx_1)
        Γ5_cplx_in_0_cplx_1_beamP1 = Float16x2(Γ5_dish2_0_cplx_in_0_cplx_1, Γ5_dish2_1_cplx_in_0_cplx_1)
        Γ5_cplx_in_0_cplx_1_beamP2 = Float16x2(Γ5_dish2_0_cplx_in_0_cplx_1, Γ5_dish2_1_cplx_in_0_cplx_1)
        Γ5_cplx_in_0_cplx_1_beamP3 = Float16x2(Γ5_dish2_0_cplx_in_0_cplx_1, Γ5_dish2_1_cplx_in_0_cplx_1)
        Γ5_cplx_in_0_cplx_1_beamP4 = Float16x2(Γ5_dish2_0_cplx_in_0_cplx_1, Γ5_dish2_1_cplx_in_0_cplx_1)
        Γ5_cplx_in_0_cplx_1_beamP5 = Float16x2(Γ5_dish2_0_cplx_in_0_cplx_1, Γ5_dish2_1_cplx_in_0_cplx_1)
        Γ5_cplx_in_0_cplx_1_beamP6 = Float16x2(Γ5_dish2_0_cplx_in_0_cplx_1, Γ5_dish2_1_cplx_in_0_cplx_1)
        Γ5_cplx_in_0_cplx_1_beamP7 = Float16x2(Γ5_dish2_0_cplx_in_0_cplx_1, Γ5_dish2_1_cplx_in_0_cplx_1)
        Γ5_cplx_in_1_cplx_0_beamP0 = Float16x2(Γ5_dish2_0_cplx_in_1_cplx_0, Γ5_dish2_1_cplx_in_1_cplx_0)
        Γ5_cplx_in_1_cplx_0_beamP1 = Float16x2(Γ5_dish2_0_cplx_in_1_cplx_0, Γ5_dish2_1_cplx_in_1_cplx_0)
        Γ5_cplx_in_1_cplx_0_beamP2 = Float16x2(Γ5_dish2_0_cplx_in_1_cplx_0, Γ5_dish2_1_cplx_in_1_cplx_0)
        Γ5_cplx_in_1_cplx_0_beamP3 = Float16x2(Γ5_dish2_0_cplx_in_1_cplx_0, Γ5_dish2_1_cplx_in_1_cplx_0)
        Γ5_cplx_in_1_cplx_0_beamP4 = Float16x2(Γ5_dish2_0_cplx_in_1_cplx_0, Γ5_dish2_1_cplx_in_1_cplx_0)
        Γ5_cplx_in_1_cplx_0_beamP5 = Float16x2(Γ5_dish2_0_cplx_in_1_cplx_0, Γ5_dish2_1_cplx_in_1_cplx_0)
        Γ5_cplx_in_1_cplx_0_beamP6 = Float16x2(Γ5_dish2_0_cplx_in_1_cplx_0, Γ5_dish2_1_cplx_in_1_cplx_0)
        Γ5_cplx_in_1_cplx_0_beamP7 = Float16x2(Γ5_dish2_0_cplx_in_1_cplx_0, Γ5_dish2_1_cplx_in_1_cplx_0)
        Γ5_cplx_in_1_cplx_1_beamP0 = Float16x2(Γ5_dish2_0_cplx_in_1_cplx_1, Γ5_dish2_1_cplx_in_1_cplx_1)
        Γ5_cplx_in_1_cplx_1_beamP1 = Float16x2(Γ5_dish2_0_cplx_in_1_cplx_1, Γ5_dish2_1_cplx_in_1_cplx_1)
        Γ5_cplx_in_1_cplx_1_beamP2 = Float16x2(Γ5_dish2_0_cplx_in_1_cplx_1, Γ5_dish2_1_cplx_in_1_cplx_1)
        Γ5_cplx_in_1_cplx_1_beamP3 = Float16x2(Γ5_dish2_0_cplx_in_1_cplx_1, Γ5_dish2_1_cplx_in_1_cplx_1)
        Γ5_cplx_in_1_cplx_1_beamP4 = Float16x2(Γ5_dish2_0_cplx_in_1_cplx_1, Γ5_dish2_1_cplx_in_1_cplx_1)
        Γ5_cplx_in_1_cplx_1_beamP5 = Float16x2(Γ5_dish2_0_cplx_in_1_cplx_1, Γ5_dish2_1_cplx_in_1_cplx_1)
        Γ5_cplx_in_1_cplx_1_beamP6 = Float16x2(Γ5_dish2_0_cplx_in_1_cplx_1, Γ5_dish2_1_cplx_in_1_cplx_1)
        Γ5_cplx_in_1_cplx_1_beamP7 = Float16x2(Γ5_dish2_0_cplx_in_1_cplx_1, Γ5_dish2_1_cplx_in_1_cplx_1)
        Γ5_cplx_in_0_beamP0_cplx0 = Γ5_cplx_in_0_cplx_0_beamP0
        Γ5_cplx_in_0_beamP0_cplx1 = Γ5_cplx_in_0_cplx_1_beamP0
        Γ5_cplx_in_0_beamP1_cplx0 = Γ5_cplx_in_0_cplx_0_beamP1
        Γ5_cplx_in_0_beamP1_cplx1 = Γ5_cplx_in_0_cplx_1_beamP1
        Γ5_cplx_in_0_beamP2_cplx0 = Γ5_cplx_in_0_cplx_0_beamP2
        Γ5_cplx_in_0_beamP2_cplx1 = Γ5_cplx_in_0_cplx_1_beamP2
        Γ5_cplx_in_0_beamP3_cplx0 = Γ5_cplx_in_0_cplx_0_beamP3
        Γ5_cplx_in_0_beamP3_cplx1 = Γ5_cplx_in_0_cplx_1_beamP3
        Γ5_cplx_in_0_beamP4_cplx0 = Γ5_cplx_in_0_cplx_0_beamP4
        Γ5_cplx_in_0_beamP4_cplx1 = Γ5_cplx_in_0_cplx_1_beamP4
        Γ5_cplx_in_0_beamP5_cplx0 = Γ5_cplx_in_0_cplx_0_beamP5
        Γ5_cplx_in_0_beamP5_cplx1 = Γ5_cplx_in_0_cplx_1_beamP5
        Γ5_cplx_in_0_beamP6_cplx0 = Γ5_cplx_in_0_cplx_0_beamP6
        Γ5_cplx_in_0_beamP6_cplx1 = Γ5_cplx_in_0_cplx_1_beamP6
        Γ5_cplx_in_0_beamP7_cplx0 = Γ5_cplx_in_0_cplx_0_beamP7
        Γ5_cplx_in_0_beamP7_cplx1 = Γ5_cplx_in_0_cplx_1_beamP7
        Γ5_cplx_in_1_beamP0_cplx0 = Γ5_cplx_in_1_cplx_0_beamP0
        Γ5_cplx_in_1_beamP0_cplx1 = Γ5_cplx_in_1_cplx_1_beamP0
        Γ5_cplx_in_1_beamP1_cplx0 = Γ5_cplx_in_1_cplx_0_beamP1
        Γ5_cplx_in_1_beamP1_cplx1 = Γ5_cplx_in_1_cplx_1_beamP1
        Γ5_cplx_in_1_beamP2_cplx0 = Γ5_cplx_in_1_cplx_0_beamP2
        Γ5_cplx_in_1_beamP2_cplx1 = Γ5_cplx_in_1_cplx_1_beamP2
        Γ5_cplx_in_1_beamP3_cplx0 = Γ5_cplx_in_1_cplx_0_beamP3
        Γ5_cplx_in_1_beamP3_cplx1 = Γ5_cplx_in_1_cplx_1_beamP3
        Γ5_cplx_in_1_beamP4_cplx0 = Γ5_cplx_in_1_cplx_0_beamP4
        Γ5_cplx_in_1_beamP4_cplx1 = Γ5_cplx_in_1_cplx_1_beamP4
        Γ5_cplx_in_1_beamP5_cplx0 = Γ5_cplx_in_1_cplx_0_beamP5
        Γ5_cplx_in_1_beamP5_cplx1 = Γ5_cplx_in_1_cplx_1_beamP5
        Γ5_cplx_in_1_beamP6_cplx0 = Γ5_cplx_in_1_cplx_0_beamP6
        Γ5_cplx_in_1_beamP6_cplx1 = Γ5_cplx_in_1_cplx_1_beamP6
        Γ5_cplx_in_1_beamP7_cplx0 = Γ5_cplx_in_1_cplx_0_beamP7
        Γ5_cplx_in_1_beamP7_cplx1 = Γ5_cplx_in_1_cplx_1_beamP7
        Γ5_beamP0_cplx0_cplx_in0 = Γ5_cplx_in_0_beamP0_cplx0
        Γ5_beamP0_cplx0_cplx_in1 = Γ5_cplx_in_1_beamP0_cplx0
        Γ5_beamP1_cplx0_cplx_in0 = Γ5_cplx_in_0_beamP1_cplx0
        Γ5_beamP1_cplx0_cplx_in1 = Γ5_cplx_in_1_beamP1_cplx0
        Γ5_beamP2_cplx0_cplx_in0 = Γ5_cplx_in_0_beamP2_cplx0
        Γ5_beamP2_cplx0_cplx_in1 = Γ5_cplx_in_1_beamP2_cplx0
        Γ5_beamP3_cplx0_cplx_in0 = Γ5_cplx_in_0_beamP3_cplx0
        Γ5_beamP3_cplx0_cplx_in1 = Γ5_cplx_in_1_beamP3_cplx0
        Γ5_beamP4_cplx0_cplx_in0 = Γ5_cplx_in_0_beamP4_cplx0
        Γ5_beamP4_cplx0_cplx_in1 = Γ5_cplx_in_1_beamP4_cplx0
        Γ5_beamP5_cplx0_cplx_in0 = Γ5_cplx_in_0_beamP5_cplx0
        Γ5_beamP5_cplx0_cplx_in1 = Γ5_cplx_in_1_beamP5_cplx0
        Γ5_beamP6_cplx0_cplx_in0 = Γ5_cplx_in_0_beamP6_cplx0
        Γ5_beamP6_cplx0_cplx_in1 = Γ5_cplx_in_1_beamP6_cplx0
        Γ5_beamP7_cplx0_cplx_in0 = Γ5_cplx_in_0_beamP7_cplx0
        Γ5_beamP7_cplx0_cplx_in1 = Γ5_cplx_in_1_beamP7_cplx0
        Γ5_beamP0_cplx1_cplx_in0 = Γ5_cplx_in_0_beamP0_cplx1
        Γ5_beamP0_cplx1_cplx_in1 = Γ5_cplx_in_1_beamP0_cplx1
        Γ5_beamP1_cplx1_cplx_in0 = Γ5_cplx_in_0_beamP1_cplx1
        Γ5_beamP1_cplx1_cplx_in1 = Γ5_cplx_in_1_beamP1_cplx1
        Γ5_beamP2_cplx1_cplx_in0 = Γ5_cplx_in_0_beamP2_cplx1
        Γ5_beamP2_cplx1_cplx_in1 = Γ5_cplx_in_1_beamP2_cplx1
        Γ5_beamP3_cplx1_cplx_in0 = Γ5_cplx_in_0_beamP3_cplx1
        Γ5_beamP3_cplx1_cplx_in1 = Γ5_cplx_in_1_beamP3_cplx1
        Γ5_beamP4_cplx1_cplx_in0 = Γ5_cplx_in_0_beamP4_cplx1
        Γ5_beamP4_cplx1_cplx_in1 = Γ5_cplx_in_1_beamP4_cplx1
        Γ5_beamP5_cplx1_cplx_in0 = Γ5_cplx_in_0_beamP5_cplx1
        Γ5_beamP5_cplx1_cplx_in1 = Γ5_cplx_in_1_beamP5_cplx1
        Γ5_beamP6_cplx1_cplx_in0 = Γ5_cplx_in_0_beamP6_cplx1
        Γ5_beamP6_cplx1_cplx_in1 = Γ5_cplx_in_1_beamP6_cplx1
        Γ5_beamP7_cplx1_cplx_in0 = Γ5_cplx_in_0_beamP7_cplx1
        Γ5_beamP7_cplx1_cplx_in1 = Γ5_cplx_in_1_beamP7_cplx1
        W_dish0 = W_memory[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 256) % 256) * 2048 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) ÷ 4) % 2) % 2) * 1024 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 8 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 2) % 2) * 64 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 4) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 16 + 0::Int32 % 8) % 1024) + 0x01]
        W_dish1 = W_memory[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 256) % 256) * 2048 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) ÷ 4) % 2) % 2) * 1024 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 8 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 2) % 2) * 64 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 4) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 16 + 1::Int32 % 8) % 1024) + 0x01]
        W_dish2 = W_memory[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 256) % 256) * 2048 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) ÷ 4) % 2) % 2) * 1024 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 8 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 2) % 2) * 64 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 4) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 16 + 2::Int32 % 8) % 1024) + 0x01]
        W_dish3 = W_memory[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 256) % 256) * 2048 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) ÷ 4) % 2) % 2) * 1024 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 8 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 2) % 2) * 64 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 4) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 16 + 3::Int32 % 8) % 1024) + 0x01]
        W_dish4 = W_memory[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 256) % 256) * 2048 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) ÷ 4) % 2) % 2) * 1024 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 8 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 2) % 2) * 64 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 4) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 16 + 4::Int32 % 8) % 1024) + 0x01]
        W_dish5 = W_memory[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 256) % 256) * 2048 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) ÷ 4) % 2) % 2) * 1024 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 8 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 2) % 2) * 64 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 4) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 16 + 5::Int32 % 8) % 1024) + 0x01]
        W_dish6 = W_memory[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 256) % 256) * 2048 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) ÷ 4) % 2) % 2) * 1024 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 8 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 2) % 2) * 64 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 4) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 16 + 6::Int32 % 8) % 1024) + 0x01]
        W_dish7 = W_memory[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 256) % 256) * 2048 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) ÷ 4) % 2) % 2) * 1024 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 8 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 2) % 2) * 64 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 4) * 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 2) * 128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 16 + 7::Int32 % 8) % 1024) + 0x01]
        for time_outer in 0:24:1007
            I_beamQ0 = zero(Float16x2)
            I_beamQ1 = zero(Float16x2)
            I_beamQ2 = zero(Float16x2)
            I_beamQ3 = zero(Float16x2)
            I_beamQ4 = zero(Float16x2)
            I_beamQ5 = zero(Float16x2)
            I_beamQ6 = zero(Float16x2)
            I_beamQ7 = zero(Float16x2)
            for time_inner in 0:1:23
                (E_dish0, E_dish4) = IndexSpaces.unsafe_load2(
                    E_memory,
                    let
                        offset = 131072 * Tbarmin
                        length = 536870912
                        mod(
                            (
                                (
                                    (
                                        ((IndexSpaces.assume_inrange(time_outer::Int32, 0, 24, 1008) ÷ 24) % 42) * 24 +
                                        IndexSpaces.assume_inrange(time_inner::Int32, 0, 1, 24) % 24
                                    ) % 4096
                                ) * 131072 +
                                (
                                    (
                                        ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 16) % 2) * 8 +
                                        ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 2) % 2) * 64 +
                                        ((0::Int32 ÷ 4) % 2) * 4 +
                                        (IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 4) * 256 +
                                        ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 4) % 2) * 32 +
                                        (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 2) * 128 +
                                        ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) ÷ 8) % 2) * 16
                                    ) ÷ 4
                                ) % 256 +
                                (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) ÷ 4) % 2) % 2) * 256 +
                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 256) % 256) * 512
                            ) + offset,
                            length,
                        )
                    end + 0x01,
                )
                (X0_dish0, X0_dish2, X0_dish1, X0_dish3) = convert(NTuple{4,Float16x2}, E_dish0)
                (X0_dish4, X0_dish6, X0_dish5, X0_dish7) = convert(NTuple{4,Float16x2}, E_dish4)
                (X1_dish0, X1_dish2) = (IndexSpaces.get_lo16(X0_dish0, X0_dish2), IndexSpaces.get_hi16(X0_dish0, X0_dish2))
                (X1_dish1, X1_dish3) = (IndexSpaces.get_lo16(X0_dish1, X0_dish3), IndexSpaces.get_hi16(X0_dish1, X0_dish3))
                (X1_dish4, X1_dish6) = (IndexSpaces.get_lo16(X0_dish4, X0_dish6), IndexSpaces.get_hi16(X0_dish4, X0_dish6))
                (X1_dish5, X1_dish7) = (IndexSpaces.get_lo16(X0_dish5, X0_dish7), IndexSpaces.get_hi16(X0_dish5, X0_dish7))
                X_dish0 = complex_mul(W_dish0, X1_dish0)
                X_dish1 = complex_mul(W_dish1, X1_dish1)
                X_dish2 = complex_mul(W_dish2, X1_dish2)
                X_dish3 = complex_mul(W_dish3, X1_dish3)
                X_dish4 = complex_mul(W_dish4, X1_dish4)
                X_dish5 = complex_mul(W_dish5, X1_dish5)
                X_dish6 = complex_mul(W_dish6, X1_dish6)
                X_dish7 = complex_mul(W_dish7, X1_dish7)
                Z1_cplx0_dish0 = zero(Float16x2)
                Z1_cplx1_dish0 = zero(Float16x2)
                Z1_cplx0_dish1 = zero(Float16x2)
                Z1_cplx1_dish1 = zero(Float16x2)
                Z1_cplx0_dish2 = zero(Float16x2)
                Z1_cplx1_dish2 = zero(Float16x2)
                Z1_cplx0_dish3 = zero(Float16x2)
                Z1_cplx1_dish3 = zero(Float16x2)
                Z1_cplx0_dish4 = zero(Float16x2)
                Z1_cplx1_dish4 = zero(Float16x2)
                Z1_cplx0_dish5 = zero(Float16x2)
                Z1_cplx1_dish5 = zero(Float16x2)
                Z1_cplx0_dish6 = zero(Float16x2)
                Z1_cplx1_dish6 = zero(Float16x2)
                Z1_cplx0_dish7 = zero(Float16x2)
                Z1_cplx1_dish7 = zero(Float16x2)
                (Z1_cplx0_dish0, Z1_cplx1_dish0) = IndexSpaces.mma_m16n8k8(
                    (Γ1_cplx0_dish0, Γ1_cplx1_dish0), X_dish0, (Z1_cplx0_dish0, Z1_cplx1_dish0)
                )
                (Z1_cplx0_dish1, Z1_cplx1_dish1) = IndexSpaces.mma_m16n8k8(
                    (Γ1_cplx0_dish1, Γ1_cplx1_dish1), X_dish1, (Z1_cplx0_dish1, Z1_cplx1_dish1)
                )
                (Z1_cplx0_dish2, Z1_cplx1_dish2) = IndexSpaces.mma_m16n8k8(
                    (Γ1_cplx0_dish2, Γ1_cplx1_dish2), X_dish2, (Z1_cplx0_dish2, Z1_cplx1_dish2)
                )
                (Z1_cplx0_dish3, Z1_cplx1_dish3) = IndexSpaces.mma_m16n8k8(
                    (Γ1_cplx0_dish3, Γ1_cplx1_dish3), X_dish3, (Z1_cplx0_dish3, Z1_cplx1_dish3)
                )
                (Z1_cplx0_dish4, Z1_cplx1_dish4) = IndexSpaces.mma_m16n8k8(
                    (Γ1_cplx0_dish4, Γ1_cplx1_dish4), X_dish4, (Z1_cplx0_dish4, Z1_cplx1_dish4)
                )
                (Z1_cplx0_dish5, Z1_cplx1_dish5) = IndexSpaces.mma_m16n8k8(
                    (Γ1_cplx0_dish5, Γ1_cplx1_dish5), X_dish5, (Z1_cplx0_dish5, Z1_cplx1_dish5)
                )
                (Z1_cplx0_dish6, Z1_cplx1_dish6) = IndexSpaces.mma_m16n8k8(
                    (Γ1_cplx0_dish6, Γ1_cplx1_dish6), X_dish6, (Z1_cplx0_dish6, Z1_cplx1_dish6)
                )
                (Z1_cplx0_dish7, Z1_cplx1_dish7) = IndexSpaces.mma_m16n8k8(
                    (Γ1_cplx0_dish7, Γ1_cplx1_dish7), X_dish7, (Z1_cplx0_dish7, Z1_cplx1_dish7)
                )
                Γ2_re_dish0 = Γ2_cplx0_dish0
                Γ2_im_dish0 = Γ2_cplx1_dish0
                Γ2_re_dish1 = Γ2_cplx0_dish1
                Γ2_im_dish1 = Γ2_cplx1_dish1
                Γ2_re_dish2 = Γ2_cplx0_dish2
                Γ2_im_dish2 = Γ2_cplx1_dish2
                Γ2_re_dish3 = Γ2_cplx0_dish3
                Γ2_im_dish3 = Γ2_cplx1_dish3
                Γ2_re_dish4 = Γ2_cplx0_dish4
                Γ2_im_dish4 = Γ2_cplx1_dish4
                Γ2_re_dish5 = Γ2_cplx0_dish5
                Γ2_im_dish5 = Γ2_cplx1_dish5
                Γ2_re_dish6 = Γ2_cplx0_dish6
                Γ2_im_dish6 = Γ2_cplx1_dish6
                Γ2_re_dish7 = Γ2_cplx0_dish7
                Γ2_im_dish7 = Γ2_cplx1_dish7
                Z1_re_dish0 = Z1_cplx0_dish0
                Z1_im_dish0 = Z1_cplx1_dish0
                Z1_re_dish1 = Z1_cplx0_dish1
                Z1_im_dish1 = Z1_cplx1_dish1
                Z1_re_dish2 = Z1_cplx0_dish2
                Z1_im_dish2 = Z1_cplx1_dish2
                Z1_re_dish3 = Z1_cplx0_dish3
                Z1_im_dish3 = Z1_cplx1_dish3
                Z1_re_dish4 = Z1_cplx0_dish4
                Z1_im_dish4 = Z1_cplx1_dish4
                Z1_re_dish5 = Z1_cplx0_dish5
                Z1_im_dish5 = Z1_cplx1_dish5
                Z1_re_dish6 = Z1_cplx0_dish6
                Z1_im_dish6 = Z1_cplx1_dish6
                Z1_re_dish7 = Z1_cplx0_dish7
                Z1_im_dish7 = Z1_cplx1_dish7
                Z2_re_dish0 = muladd(Γ2_re_dish0, Z1_re_dish0, -Γ2_im_dish0 * Z1_im_dish0)
                Z2_re_dish1 = muladd(Γ2_re_dish1, Z1_re_dish1, -Γ2_im_dish1 * Z1_im_dish1)
                Z2_re_dish2 = muladd(Γ2_re_dish2, Z1_re_dish2, -Γ2_im_dish2 * Z1_im_dish2)
                Z2_re_dish3 = muladd(Γ2_re_dish3, Z1_re_dish3, -Γ2_im_dish3 * Z1_im_dish3)
                Z2_re_dish4 = muladd(Γ2_re_dish4, Z1_re_dish4, -Γ2_im_dish4 * Z1_im_dish4)
                Z2_re_dish5 = muladd(Γ2_re_dish5, Z1_re_dish5, -Γ2_im_dish5 * Z1_im_dish5)
                Z2_re_dish6 = muladd(Γ2_re_dish6, Z1_re_dish6, -Γ2_im_dish6 * Z1_im_dish6)
                Z2_re_dish7 = muladd(Γ2_re_dish7, Z1_re_dish7, -Γ2_im_dish7 * Z1_im_dish7)
                Z2_im_dish0 = muladd(Γ2_re_dish0, Z1_im_dish0, +Γ2_im_dish0 * Z1_re_dish0)
                Z2_im_dish1 = muladd(Γ2_re_dish1, Z1_im_dish1, +Γ2_im_dish1 * Z1_re_dish1)
                Z2_im_dish2 = muladd(Γ2_re_dish2, Z1_im_dish2, +Γ2_im_dish2 * Z1_re_dish2)
                Z2_im_dish3 = muladd(Γ2_re_dish3, Z1_im_dish3, +Γ2_im_dish3 * Z1_re_dish3)
                Z2_im_dish4 = muladd(Γ2_re_dish4, Z1_im_dish4, +Γ2_im_dish4 * Z1_re_dish4)
                Z2_im_dish5 = muladd(Γ2_re_dish5, Z1_im_dish5, +Γ2_im_dish5 * Z1_re_dish5)
                Z2_im_dish6 = muladd(Γ2_re_dish6, Z1_im_dish6, +Γ2_im_dish6 * Z1_re_dish6)
                Z2_im_dish7 = muladd(Γ2_re_dish7, Z1_im_dish7, +Γ2_im_dish7 * Z1_re_dish7)
                Z2_cplx0_dish0 = Z2_re_dish0
                Z2_cplx1_dish0 = Z2_im_dish0
                Z2_cplx0_dish1 = Z2_re_dish1
                Z2_cplx1_dish1 = Z2_im_dish1
                Z2_cplx0_dish2 = Z2_re_dish2
                Z2_cplx1_dish2 = Z2_im_dish2
                Z2_cplx0_dish3 = Z2_re_dish3
                Z2_cplx1_dish3 = Z2_im_dish3
                Z2_cplx0_dish4 = Z2_re_dish4
                Z2_cplx1_dish4 = Z2_im_dish4
                Z2_cplx0_dish5 = Z2_re_dish5
                Z2_cplx1_dish5 = Z2_im_dish5
                Z2_cplx0_dish6 = Z2_re_dish6
                Z2_cplx1_dish6 = Z2_im_dish6
                Z2_cplx0_dish7 = Z2_re_dish7
                Z2_cplx1_dish7 = Z2_im_dish7
                Z3′_cplx0_dish0 = zero(Float16x2)
                Z3′_cplx1_dish0 = zero(Float16x2)
                Z3′_cplx0_dish1 = zero(Float16x2)
                Z3′_cplx1_dish1 = zero(Float16x2)
                Z3′_cplx0_dish2 = zero(Float16x2)
                Z3′_cplx1_dish2 = zero(Float16x2)
                Z3′_cplx0_dish3 = zero(Float16x2)
                Z3′_cplx1_dish3 = zero(Float16x2)
                Z3′_cplx0_dish4 = zero(Float16x2)
                Z3′_cplx1_dish4 = zero(Float16x2)
                Z3′_cplx0_dish5 = zero(Float16x2)
                Z3′_cplx1_dish5 = zero(Float16x2)
                Z3′_cplx0_dish6 = zero(Float16x2)
                Z3′_cplx1_dish6 = zero(Float16x2)
                Z3′_cplx0_dish7 = zero(Float16x2)
                Z3′_cplx1_dish7 = zero(Float16x2)
                Z2_re_dish0 = Z2_cplx0_dish0
                Z2_im_dish0 = Z2_cplx1_dish0
                Z2_re_dish1 = Z2_cplx0_dish1
                Z2_im_dish1 = Z2_cplx1_dish1
                Z2_re_dish2 = Z2_cplx0_dish2
                Z2_im_dish2 = Z2_cplx1_dish2
                Z2_re_dish3 = Z2_cplx0_dish3
                Z2_im_dish3 = Z2_cplx1_dish3
                Z2_re_dish4 = Z2_cplx0_dish4
                Z2_im_dish4 = Z2_cplx1_dish4
                Z2_re_dish5 = Z2_cplx0_dish5
                Z2_im_dish5 = Z2_cplx1_dish5
                Z2_re_dish6 = Z2_cplx0_dish6
                Z2_im_dish6 = Z2_cplx1_dish6
                Z2_re_dish7 = Z2_cplx0_dish7
                Z2_im_dish7 = Z2_cplx1_dish7
                Z2_cplx_in0_dish0 = Z2_re_dish0
                Z2_cplx_in1_dish0 = Z2_im_dish0
                Z2_cplx_in0_dish1 = Z2_re_dish1
                Z2_cplx_in1_dish1 = Z2_im_dish1
                Z2_cplx_in0_dish2 = Z2_re_dish2
                Z2_cplx_in1_dish2 = Z2_im_dish2
                Z2_cplx_in0_dish3 = Z2_re_dish3
                Z2_cplx_in1_dish3 = Z2_im_dish3
                Z2_cplx_in0_dish4 = Z2_re_dish4
                Z2_cplx_in1_dish4 = Z2_im_dish4
                Z2_cplx_in0_dish5 = Z2_re_dish5
                Z2_cplx_in1_dish5 = Z2_im_dish5
                Z2_cplx_in0_dish6 = Z2_re_dish6
                Z2_cplx_in1_dish6 = Z2_im_dish6
                Z2_cplx_in0_dish7 = Z2_re_dish7
                Z2_cplx_in1_dish7 = Z2_im_dish7
                (Z3′_cplx0_dish0, Z3′_cplx1_dish0) = IndexSpaces.mma_m16n8k16(
                    (Γ3_cplx0_cplx_in0_dish0, Γ3_cplx1_cplx_in0_dish0, Γ3_cplx0_cplx_in1_dish0, Γ3_cplx1_cplx_in1_dish0),
                    (Z2_cplx_in0_dish0, Z2_cplx_in1_dish0),
                    (Z3′_cplx0_dish0, Z3′_cplx1_dish0),
                )
                (Z3′_cplx0_dish1, Z3′_cplx1_dish1) = IndexSpaces.mma_m16n8k16(
                    (Γ3_cplx0_cplx_in0_dish1, Γ3_cplx1_cplx_in0_dish1, Γ3_cplx0_cplx_in1_dish1, Γ3_cplx1_cplx_in1_dish1),
                    (Z2_cplx_in0_dish1, Z2_cplx_in1_dish1),
                    (Z3′_cplx0_dish1, Z3′_cplx1_dish1),
                )
                (Z3′_cplx0_dish2, Z3′_cplx1_dish2) = IndexSpaces.mma_m16n8k16(
                    (Γ3_cplx0_cplx_in0_dish2, Γ3_cplx1_cplx_in0_dish2, Γ3_cplx0_cplx_in1_dish2, Γ3_cplx1_cplx_in1_dish2),
                    (Z2_cplx_in0_dish2, Z2_cplx_in1_dish2),
                    (Z3′_cplx0_dish2, Z3′_cplx1_dish2),
                )
                (Z3′_cplx0_dish3, Z3′_cplx1_dish3) = IndexSpaces.mma_m16n8k16(
                    (Γ3_cplx0_cplx_in0_dish3, Γ3_cplx1_cplx_in0_dish3, Γ3_cplx0_cplx_in1_dish3, Γ3_cplx1_cplx_in1_dish3),
                    (Z2_cplx_in0_dish3, Z2_cplx_in1_dish3),
                    (Z3′_cplx0_dish3, Z3′_cplx1_dish3),
                )
                (Z3′_cplx0_dish4, Z3′_cplx1_dish4) = IndexSpaces.mma_m16n8k16(
                    (Γ3_cplx0_cplx_in0_dish4, Γ3_cplx1_cplx_in0_dish4, Γ3_cplx0_cplx_in1_dish4, Γ3_cplx1_cplx_in1_dish4),
                    (Z2_cplx_in0_dish4, Z2_cplx_in1_dish4),
                    (Z3′_cplx0_dish4, Z3′_cplx1_dish4),
                )
                (Z3′_cplx0_dish5, Z3′_cplx1_dish5) = IndexSpaces.mma_m16n8k16(
                    (Γ3_cplx0_cplx_in0_dish5, Γ3_cplx1_cplx_in0_dish5, Γ3_cplx0_cplx_in1_dish5, Γ3_cplx1_cplx_in1_dish5),
                    (Z2_cplx_in0_dish5, Z2_cplx_in1_dish5),
                    (Z3′_cplx0_dish5, Z3′_cplx1_dish5),
                )
                (Z3′_cplx0_dish6, Z3′_cplx1_dish6) = IndexSpaces.mma_m16n8k16(
                    (Γ3_cplx0_cplx_in0_dish6, Γ3_cplx1_cplx_in0_dish6, Γ3_cplx0_cplx_in1_dish6, Γ3_cplx1_cplx_in1_dish6),
                    (Z2_cplx_in0_dish6, Z2_cplx_in1_dish6),
                    (Z3′_cplx0_dish6, Z3′_cplx1_dish6),
                )
                (Z3′_cplx0_dish7, Z3′_cplx1_dish7) = IndexSpaces.mma_m16n8k16(
                    (Γ3_cplx0_cplx_in0_dish7, Γ3_cplx1_cplx_in0_dish7, Γ3_cplx0_cplx_in1_dish7, Γ3_cplx1_cplx_in1_dish7),
                    (Z2_cplx_in0_dish7, Z2_cplx_in1_dish7),
                    (Z3′_cplx0_dish7, Z3′_cplx1_dish7),
                )
                (Z3′1_cplx0_dish0, Z3′1_cplx0_dish4) = (
                    IndexSpaces.get_lo16(Z3′_cplx0_dish0, Z3′_cplx0_dish4), IndexSpaces.get_hi16(Z3′_cplx0_dish0, Z3′_cplx0_dish4)
                )
                (Z3′1_cplx1_dish0, Z3′1_cplx1_dish4) = (
                    IndexSpaces.get_lo16(Z3′_cplx1_dish0, Z3′_cplx1_dish4), IndexSpaces.get_hi16(Z3′_cplx1_dish0, Z3′_cplx1_dish4)
                )
                (Z3′1_cplx0_dish1, Z3′1_cplx0_dish5) = (
                    IndexSpaces.get_lo16(Z3′_cplx0_dish1, Z3′_cplx0_dish5), IndexSpaces.get_hi16(Z3′_cplx0_dish1, Z3′_cplx0_dish5)
                )
                (Z3′1_cplx1_dish1, Z3′1_cplx1_dish5) = (
                    IndexSpaces.get_lo16(Z3′_cplx1_dish1, Z3′_cplx1_dish5), IndexSpaces.get_hi16(Z3′_cplx1_dish1, Z3′_cplx1_dish5)
                )
                (Z3′1_cplx0_dish2, Z3′1_cplx0_dish6) = (
                    IndexSpaces.get_lo16(Z3′_cplx0_dish2, Z3′_cplx0_dish6), IndexSpaces.get_hi16(Z3′_cplx0_dish2, Z3′_cplx0_dish6)
                )
                (Z3′1_cplx1_dish2, Z3′1_cplx1_dish6) = (
                    IndexSpaces.get_lo16(Z3′_cplx1_dish2, Z3′_cplx1_dish6), IndexSpaces.get_hi16(Z3′_cplx1_dish2, Z3′_cplx1_dish6)
                )
                (Z3′1_cplx0_dish3, Z3′1_cplx0_dish7) = (
                    IndexSpaces.get_lo16(Z3′_cplx0_dish3, Z3′_cplx0_dish7), IndexSpaces.get_hi16(Z3′_cplx0_dish3, Z3′_cplx0_dish7)
                )
                (Z3′1_cplx1_dish3, Z3′1_cplx1_dish7) = (
                    IndexSpaces.get_lo16(Z3′_cplx1_dish3, Z3′_cplx1_dish7), IndexSpaces.get_hi16(Z3′_cplx1_dish3, Z3′_cplx1_dish7)
                )
                Z3′1_beam0_0_cplx0_dish0 = Z3′1_cplx0_dish0
                Z3′1_beam0_1_cplx0_dish0 = Z3′1_cplx0_dish4
                Z3′1_beam0_0_cplx1_dish0 = Z3′1_cplx1_dish0
                Z3′1_beam0_1_cplx1_dish0 = Z3′1_cplx1_dish4
                Z3′1_beam0_0_cplx0_dish1 = Z3′1_cplx0_dish1
                Z3′1_beam0_1_cplx0_dish1 = Z3′1_cplx0_dish5
                Z3′1_beam0_0_cplx1_dish1 = Z3′1_cplx1_dish1
                Z3′1_beam0_1_cplx1_dish1 = Z3′1_cplx1_dish5
                Z3′1_beam0_0_cplx0_dish2 = Z3′1_cplx0_dish2
                Z3′1_beam0_1_cplx0_dish2 = Z3′1_cplx0_dish6
                Z3′1_beam0_0_cplx1_dish2 = Z3′1_cplx1_dish2
                Z3′1_beam0_1_cplx1_dish2 = Z3′1_cplx1_dish6
                Z3′1_beam0_0_cplx0_dish3 = Z3′1_cplx0_dish3
                Z3′1_beam0_1_cplx0_dish3 = Z3′1_cplx0_dish7
                Z3′1_beam0_0_cplx1_dish3 = Z3′1_cplx1_dish3
                Z3′1_beam0_1_cplx1_dish3 = Z3′1_cplx1_dish7
                Z3′1_beamP0_cplx0_dish0 = Z3′1_beam0_0_cplx0_dish0
                Z3′1_beamP1_cplx0_dish0 = Z3′1_beam0_1_cplx0_dish0
                Z3′1_beamP0_cplx1_dish0 = Z3′1_beam0_0_cplx1_dish0
                Z3′1_beamP1_cplx1_dish0 = Z3′1_beam0_1_cplx1_dish0
                Z3′1_beamP0_cplx0_dish1 = Z3′1_beam0_0_cplx0_dish1
                Z3′1_beamP1_cplx0_dish1 = Z3′1_beam0_1_cplx0_dish1
                Z3′1_beamP0_cplx1_dish1 = Z3′1_beam0_0_cplx1_dish1
                Z3′1_beamP1_cplx1_dish1 = Z3′1_beam0_1_cplx1_dish1
                Z3′1_beamP0_cplx0_dish2 = Z3′1_beam0_0_cplx0_dish2
                Z3′1_beamP1_cplx0_dish2 = Z3′1_beam0_1_cplx0_dish2
                Z3′1_beamP0_cplx1_dish2 = Z3′1_beam0_0_cplx1_dish2
                Z3′1_beamP1_cplx1_dish2 = Z3′1_beam0_1_cplx1_dish2
                Z3′1_beamP0_cplx0_dish3 = Z3′1_beam0_0_cplx0_dish3
                Z3′1_beamP1_cplx0_dish3 = Z3′1_beam0_1_cplx0_dish3
                Z3′1_beamP0_cplx1_dish3 = Z3′1_beam0_0_cplx1_dish3
                Z3′1_beamP1_cplx1_dish3 = Z3′1_beam0_1_cplx1_dish3
                is_lo_thread = IndexSpaces.cuda_threadidx() & 0x00000001 == 0x00
                (Z3′2_beamP0_cplx0_dish0, Z3′2_beamP0_cplx0_dish2) = let
                    src = if is_lo_thread
                        Z3′1_beamP0_cplx0_dish2
                    else
                        Z3′1_beamP0_cplx0_dish0
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000001)
                    if is_lo_thread
                        (Z3′1_beamP0_cplx0_dish0, dst)
                    else
                        (dst, Z3′1_beamP0_cplx0_dish2)
                    end
                end
                (Z3′2_beamP1_cplx0_dish0, Z3′2_beamP1_cplx0_dish2) = let
                    src = if is_lo_thread
                        Z3′1_beamP1_cplx0_dish2
                    else
                        Z3′1_beamP1_cplx0_dish0
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000001)
                    if is_lo_thread
                        (Z3′1_beamP1_cplx0_dish0, dst)
                    else
                        (dst, Z3′1_beamP1_cplx0_dish2)
                    end
                end
                (Z3′2_beamP0_cplx1_dish0, Z3′2_beamP0_cplx1_dish2) = let
                    src = if is_lo_thread
                        Z3′1_beamP0_cplx1_dish2
                    else
                        Z3′1_beamP0_cplx1_dish0
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000001)
                    if is_lo_thread
                        (Z3′1_beamP0_cplx1_dish0, dst)
                    else
                        (dst, Z3′1_beamP0_cplx1_dish2)
                    end
                end
                (Z3′2_beamP1_cplx1_dish0, Z3′2_beamP1_cplx1_dish2) = let
                    src = if is_lo_thread
                        Z3′1_beamP1_cplx1_dish2
                    else
                        Z3′1_beamP1_cplx1_dish0
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000001)
                    if is_lo_thread
                        (Z3′1_beamP1_cplx1_dish0, dst)
                    else
                        (dst, Z3′1_beamP1_cplx1_dish2)
                    end
                end
                (Z3′2_beamP0_cplx0_dish1, Z3′2_beamP0_cplx0_dish3) = let
                    src = if is_lo_thread
                        Z3′1_beamP0_cplx0_dish3
                    else
                        Z3′1_beamP0_cplx0_dish1
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000001)
                    if is_lo_thread
                        (Z3′1_beamP0_cplx0_dish1, dst)
                    else
                        (dst, Z3′1_beamP0_cplx0_dish3)
                    end
                end
                (Z3′2_beamP1_cplx0_dish1, Z3′2_beamP1_cplx0_dish3) = let
                    src = if is_lo_thread
                        Z3′1_beamP1_cplx0_dish3
                    else
                        Z3′1_beamP1_cplx0_dish1
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000001)
                    if is_lo_thread
                        (Z3′1_beamP1_cplx0_dish1, dst)
                    else
                        (dst, Z3′1_beamP1_cplx0_dish3)
                    end
                end
                (Z3′2_beamP0_cplx1_dish1, Z3′2_beamP0_cplx1_dish3) = let
                    src = if is_lo_thread
                        Z3′1_beamP0_cplx1_dish3
                    else
                        Z3′1_beamP0_cplx1_dish1
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000001)
                    if is_lo_thread
                        (Z3′1_beamP0_cplx1_dish1, dst)
                    else
                        (dst, Z3′1_beamP0_cplx1_dish3)
                    end
                end
                (Z3′2_beamP1_cplx1_dish1, Z3′2_beamP1_cplx1_dish3) = let
                    src = if is_lo_thread
                        Z3′1_beamP1_cplx1_dish3
                    else
                        Z3′1_beamP1_cplx1_dish1
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000001)
                    if is_lo_thread
                        (Z3′1_beamP1_cplx1_dish1, dst)
                    else
                        (dst, Z3′1_beamP1_cplx1_dish3)
                    end
                end
                Z3′2_beam1_0_beamP0_cplx0_dish0 = Z3′2_beamP0_cplx0_dish0
                Z3′2_beam1_1_beamP0_cplx0_dish0 = Z3′2_beamP0_cplx0_dish2
                Z3′2_beam1_0_beamP1_cplx0_dish0 = Z3′2_beamP1_cplx0_dish0
                Z3′2_beam1_1_beamP1_cplx0_dish0 = Z3′2_beamP1_cplx0_dish2
                Z3′2_beam1_0_beamP0_cplx1_dish0 = Z3′2_beamP0_cplx1_dish0
                Z3′2_beam1_1_beamP0_cplx1_dish0 = Z3′2_beamP0_cplx1_dish2
                Z3′2_beam1_0_beamP1_cplx1_dish0 = Z3′2_beamP1_cplx1_dish0
                Z3′2_beam1_1_beamP1_cplx1_dish0 = Z3′2_beamP1_cplx1_dish2
                Z3′2_beam1_0_beamP0_cplx0_dish1 = Z3′2_beamP0_cplx0_dish1
                Z3′2_beam1_1_beamP0_cplx0_dish1 = Z3′2_beamP0_cplx0_dish3
                Z3′2_beam1_0_beamP1_cplx0_dish1 = Z3′2_beamP1_cplx0_dish1
                Z3′2_beam1_1_beamP1_cplx0_dish1 = Z3′2_beamP1_cplx0_dish3
                Z3′2_beam1_0_beamP0_cplx1_dish1 = Z3′2_beamP0_cplx1_dish1
                Z3′2_beam1_1_beamP0_cplx1_dish1 = Z3′2_beamP0_cplx1_dish3
                Z3′2_beam1_0_beamP1_cplx1_dish1 = Z3′2_beamP1_cplx1_dish1
                Z3′2_beam1_1_beamP1_cplx1_dish1 = Z3′2_beamP1_cplx1_dish3
                Z3′2_beamP0_cplx0_dish0 = Z3′2_beam1_0_beamP0_cplx0_dish0
                Z3′2_beamP2_cplx0_dish0 = Z3′2_beam1_1_beamP0_cplx0_dish0
                Z3′2_beamP1_cplx0_dish0 = Z3′2_beam1_0_beamP1_cplx0_dish0
                Z3′2_beamP3_cplx0_dish0 = Z3′2_beam1_1_beamP1_cplx0_dish0
                Z3′2_beamP0_cplx1_dish0 = Z3′2_beam1_0_beamP0_cplx1_dish0
                Z3′2_beamP2_cplx1_dish0 = Z3′2_beam1_1_beamP0_cplx1_dish0
                Z3′2_beamP1_cplx1_dish0 = Z3′2_beam1_0_beamP1_cplx1_dish0
                Z3′2_beamP3_cplx1_dish0 = Z3′2_beam1_1_beamP1_cplx1_dish0
                Z3′2_beamP0_cplx0_dish1 = Z3′2_beam1_0_beamP0_cplx0_dish1
                Z3′2_beamP2_cplx0_dish1 = Z3′2_beam1_1_beamP0_cplx0_dish1
                Z3′2_beamP1_cplx0_dish1 = Z3′2_beam1_0_beamP1_cplx0_dish1
                Z3′2_beamP3_cplx0_dish1 = Z3′2_beam1_1_beamP1_cplx0_dish1
                Z3′2_beamP0_cplx1_dish1 = Z3′2_beam1_0_beamP0_cplx1_dish1
                Z3′2_beamP2_cplx1_dish1 = Z3′2_beam1_1_beamP0_cplx1_dish1
                Z3′2_beamP1_cplx1_dish1 = Z3′2_beam1_0_beamP1_cplx1_dish1
                Z3′2_beamP3_cplx1_dish1 = Z3′2_beam1_1_beamP1_cplx1_dish1
                is_lo_thread = IndexSpaces.cuda_threadidx() & 0x00000002 == 0x00
                (Z3_beamP0_cplx0_dish0, Z3_beamP0_cplx0_dish1) = let
                    src = if is_lo_thread
                        Z3′2_beamP0_cplx0_dish1
                    else
                        Z3′2_beamP0_cplx0_dish0
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                    if is_lo_thread
                        (Z3′2_beamP0_cplx0_dish0, dst)
                    else
                        (dst, Z3′2_beamP0_cplx0_dish1)
                    end
                end
                (Z3_beamP1_cplx0_dish0, Z3_beamP1_cplx0_dish1) = let
                    src = if is_lo_thread
                        Z3′2_beamP1_cplx0_dish1
                    else
                        Z3′2_beamP1_cplx0_dish0
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                    if is_lo_thread
                        (Z3′2_beamP1_cplx0_dish0, dst)
                    else
                        (dst, Z3′2_beamP1_cplx0_dish1)
                    end
                end
                (Z3_beamP2_cplx0_dish0, Z3_beamP2_cplx0_dish1) = let
                    src = if is_lo_thread
                        Z3′2_beamP2_cplx0_dish1
                    else
                        Z3′2_beamP2_cplx0_dish0
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                    if is_lo_thread
                        (Z3′2_beamP2_cplx0_dish0, dst)
                    else
                        (dst, Z3′2_beamP2_cplx0_dish1)
                    end
                end
                (Z3_beamP3_cplx0_dish0, Z3_beamP3_cplx0_dish1) = let
                    src = if is_lo_thread
                        Z3′2_beamP3_cplx0_dish1
                    else
                        Z3′2_beamP3_cplx0_dish0
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                    if is_lo_thread
                        (Z3′2_beamP3_cplx0_dish0, dst)
                    else
                        (dst, Z3′2_beamP3_cplx0_dish1)
                    end
                end
                (Z3_beamP0_cplx1_dish0, Z3_beamP0_cplx1_dish1) = let
                    src = if is_lo_thread
                        Z3′2_beamP0_cplx1_dish1
                    else
                        Z3′2_beamP0_cplx1_dish0
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                    if is_lo_thread
                        (Z3′2_beamP0_cplx1_dish0, dst)
                    else
                        (dst, Z3′2_beamP0_cplx1_dish1)
                    end
                end
                (Z3_beamP1_cplx1_dish0, Z3_beamP1_cplx1_dish1) = let
                    src = if is_lo_thread
                        Z3′2_beamP1_cplx1_dish1
                    else
                        Z3′2_beamP1_cplx1_dish0
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                    if is_lo_thread
                        (Z3′2_beamP1_cplx1_dish0, dst)
                    else
                        (dst, Z3′2_beamP1_cplx1_dish1)
                    end
                end
                (Z3_beamP2_cplx1_dish0, Z3_beamP2_cplx1_dish1) = let
                    src = if is_lo_thread
                        Z3′2_beamP2_cplx1_dish1
                    else
                        Z3′2_beamP2_cplx1_dish0
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                    if is_lo_thread
                        (Z3′2_beamP2_cplx1_dish0, dst)
                    else
                        (dst, Z3′2_beamP2_cplx1_dish1)
                    end
                end
                (Z3_beamP3_cplx1_dish0, Z3_beamP3_cplx1_dish1) = let
                    src = if is_lo_thread
                        Z3′2_beamP3_cplx1_dish1
                    else
                        Z3′2_beamP3_cplx1_dish0
                    end
                    dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                    if is_lo_thread
                        (Z3′2_beamP3_cplx1_dish0, dst)
                    else
                        (dst, Z3′2_beamP3_cplx1_dish1)
                    end
                end
                Z3_beam2_0_beamP0_cplx0 = Z3_beamP0_cplx0_dish0
                Z3_beam2_1_beamP0_cplx0 = Z3_beamP0_cplx0_dish1
                Z3_beam2_0_beamP1_cplx0 = Z3_beamP1_cplx0_dish0
                Z3_beam2_1_beamP1_cplx0 = Z3_beamP1_cplx0_dish1
                Z3_beam2_0_beamP2_cplx0 = Z3_beamP2_cplx0_dish0
                Z3_beam2_1_beamP2_cplx0 = Z3_beamP2_cplx0_dish1
                Z3_beam2_0_beamP3_cplx0 = Z3_beamP3_cplx0_dish0
                Z3_beam2_1_beamP3_cplx0 = Z3_beamP3_cplx0_dish1
                Z3_beam2_0_beamP0_cplx1 = Z3_beamP0_cplx1_dish0
                Z3_beam2_1_beamP0_cplx1 = Z3_beamP0_cplx1_dish1
                Z3_beam2_0_beamP1_cplx1 = Z3_beamP1_cplx1_dish0
                Z3_beam2_1_beamP1_cplx1 = Z3_beamP1_cplx1_dish1
                Z3_beam2_0_beamP2_cplx1 = Z3_beamP2_cplx1_dish0
                Z3_beam2_1_beamP2_cplx1 = Z3_beamP2_cplx1_dish1
                Z3_beam2_0_beamP3_cplx1 = Z3_beamP3_cplx1_dish0
                Z3_beam2_1_beamP3_cplx1 = Z3_beamP3_cplx1_dish1
                Z3_beamP0_cplx0 = Z3_beam2_0_beamP0_cplx0
                Z3_beamP4_cplx0 = Z3_beam2_1_beamP0_cplx0
                Z3_beamP1_cplx0 = Z3_beam2_0_beamP1_cplx0
                Z3_beamP5_cplx0 = Z3_beam2_1_beamP1_cplx0
                Z3_beamP2_cplx0 = Z3_beam2_0_beamP2_cplx0
                Z3_beamP6_cplx0 = Z3_beam2_1_beamP2_cplx0
                Z3_beamP3_cplx0 = Z3_beam2_0_beamP3_cplx0
                Z3_beamP7_cplx0 = Z3_beam2_1_beamP3_cplx0
                Z3_beamP0_cplx1 = Z3_beam2_0_beamP0_cplx1
                Z3_beamP4_cplx1 = Z3_beam2_1_beamP0_cplx1
                Z3_beamP1_cplx1 = Z3_beam2_0_beamP1_cplx1
                Z3_beamP5_cplx1 = Z3_beam2_1_beamP1_cplx1
                Z3_beamP2_cplx1 = Z3_beam2_0_beamP2_cplx1
                Z3_beamP6_cplx1 = Z3_beam2_1_beamP2_cplx1
                Z3_beamP3_cplx1 = Z3_beam2_0_beamP3_cplx1
                Z3_beamP7_cplx1 = Z3_beam2_1_beamP3_cplx1
                Γ4_re_beamP0 = Γ4_beamP0_cplx0
                Γ4_im_beamP0 = Γ4_beamP0_cplx1
                Γ4_re_beamP1 = Γ4_beamP1_cplx0
                Γ4_im_beamP1 = Γ4_beamP1_cplx1
                Γ4_re_beamP2 = Γ4_beamP2_cplx0
                Γ4_im_beamP2 = Γ4_beamP2_cplx1
                Γ4_re_beamP3 = Γ4_beamP3_cplx0
                Γ4_im_beamP3 = Γ4_beamP3_cplx1
                Γ4_re_beamP4 = Γ4_beamP4_cplx0
                Γ4_im_beamP4 = Γ4_beamP4_cplx1
                Γ4_re_beamP5 = Γ4_beamP5_cplx0
                Γ4_im_beamP5 = Γ4_beamP5_cplx1
                Γ4_re_beamP6 = Γ4_beamP6_cplx0
                Γ4_im_beamP6 = Γ4_beamP6_cplx1
                Γ4_re_beamP7 = Γ4_beamP7_cplx0
                Γ4_im_beamP7 = Γ4_beamP7_cplx1
                Z3_re_beamP0 = Z3_beamP0_cplx0
                Z3_im_beamP0 = Z3_beamP0_cplx1
                Z3_re_beamP1 = Z3_beamP1_cplx0
                Z3_im_beamP1 = Z3_beamP1_cplx1
                Z3_re_beamP2 = Z3_beamP2_cplx0
                Z3_im_beamP2 = Z3_beamP2_cplx1
                Z3_re_beamP3 = Z3_beamP3_cplx0
                Z3_im_beamP3 = Z3_beamP3_cplx1
                Z3_re_beamP4 = Z3_beamP4_cplx0
                Z3_im_beamP4 = Z3_beamP4_cplx1
                Z3_re_beamP5 = Z3_beamP5_cplx0
                Z3_im_beamP5 = Z3_beamP5_cplx1
                Z3_re_beamP6 = Z3_beamP6_cplx0
                Z3_im_beamP6 = Z3_beamP6_cplx1
                Z3_re_beamP7 = Z3_beamP7_cplx0
                Z3_im_beamP7 = Z3_beamP7_cplx1
                Z4_re_beamP0 = muladd(Γ4_re_beamP0, Z3_re_beamP0, -Γ4_im_beamP0 * Z3_im_beamP0)
                Z4_re_beamP1 = muladd(Γ4_re_beamP1, Z3_re_beamP1, -Γ4_im_beamP1 * Z3_im_beamP1)
                Z4_re_beamP2 = muladd(Γ4_re_beamP2, Z3_re_beamP2, -Γ4_im_beamP2 * Z3_im_beamP2)
                Z4_re_beamP3 = muladd(Γ4_re_beamP3, Z3_re_beamP3, -Γ4_im_beamP3 * Z3_im_beamP3)
                Z4_re_beamP4 = muladd(Γ4_re_beamP4, Z3_re_beamP4, -Γ4_im_beamP4 * Z3_im_beamP4)
                Z4_re_beamP5 = muladd(Γ4_re_beamP5, Z3_re_beamP5, -Γ4_im_beamP5 * Z3_im_beamP5)
                Z4_re_beamP6 = muladd(Γ4_re_beamP6, Z3_re_beamP6, -Γ4_im_beamP6 * Z3_im_beamP6)
                Z4_re_beamP7 = muladd(Γ4_re_beamP7, Z3_re_beamP7, -Γ4_im_beamP7 * Z3_im_beamP7)
                Z4_im_beamP0 = muladd(Γ4_re_beamP0, Z3_im_beamP0, +Γ4_im_beamP0 * Z3_re_beamP0)
                Z4_im_beamP1 = muladd(Γ4_re_beamP1, Z3_im_beamP1, +Γ4_im_beamP1 * Z3_re_beamP1)
                Z4_im_beamP2 = muladd(Γ4_re_beamP2, Z3_im_beamP2, +Γ4_im_beamP2 * Z3_re_beamP2)
                Z4_im_beamP3 = muladd(Γ4_re_beamP3, Z3_im_beamP3, +Γ4_im_beamP3 * Z3_re_beamP3)
                Z4_im_beamP4 = muladd(Γ4_re_beamP4, Z3_im_beamP4, +Γ4_im_beamP4 * Z3_re_beamP4)
                Z4_im_beamP5 = muladd(Γ4_re_beamP5, Z3_im_beamP5, +Γ4_im_beamP5 * Z3_re_beamP5)
                Z4_im_beamP6 = muladd(Γ4_re_beamP6, Z3_im_beamP6, +Γ4_im_beamP6 * Z3_re_beamP6)
                Z4_im_beamP7 = muladd(Γ4_re_beamP7, Z3_im_beamP7, +Γ4_im_beamP7 * Z3_re_beamP7)
                Z4_beamP0_cplx0 = Z4_re_beamP0
                Z4_beamP0_cplx1 = Z4_im_beamP0
                Z4_beamP1_cplx0 = Z4_re_beamP1
                Z4_beamP1_cplx1 = Z4_im_beamP1
                Z4_beamP2_cplx0 = Z4_re_beamP2
                Z4_beamP2_cplx1 = Z4_im_beamP2
                Z4_beamP3_cplx0 = Z4_re_beamP3
                Z4_beamP3_cplx1 = Z4_im_beamP3
                Z4_beamP4_cplx0 = Z4_re_beamP4
                Z4_beamP4_cplx1 = Z4_im_beamP4
                Z4_beamP5_cplx0 = Z4_re_beamP5
                Z4_beamP5_cplx1 = Z4_im_beamP5
                Z4_beamP6_cplx0 = Z4_re_beamP6
                Z4_beamP6_cplx1 = Z4_im_beamP6
                Z4_beamP7_cplx0 = Z4_re_beamP7
                Z4_beamP7_cplx1 = Z4_im_beamP7
                Y′_beamP0_cplx0 = zero(Float16x2)
                Y′_beamP1_cplx0 = zero(Float16x2)
                Y′_beamP2_cplx0 = zero(Float16x2)
                Y′_beamP3_cplx0 = zero(Float16x2)
                Y′_beamP4_cplx0 = zero(Float16x2)
                Y′_beamP5_cplx0 = zero(Float16x2)
                Y′_beamP6_cplx0 = zero(Float16x2)
                Y′_beamP7_cplx0 = zero(Float16x2)
                Y′_beamP0_cplx1 = zero(Float16x2)
                Y′_beamP1_cplx1 = zero(Float16x2)
                Y′_beamP2_cplx1 = zero(Float16x2)
                Y′_beamP3_cplx1 = zero(Float16x2)
                Y′_beamP4_cplx1 = zero(Float16x2)
                Y′_beamP5_cplx1 = zero(Float16x2)
                Y′_beamP6_cplx1 = zero(Float16x2)
                Y′_beamP7_cplx1 = zero(Float16x2)
                Z4_re_beamP0 = Z4_beamP0_cplx0
                Z4_im_beamP0 = Z4_beamP0_cplx1
                Z4_re_beamP1 = Z4_beamP1_cplx0
                Z4_im_beamP1 = Z4_beamP1_cplx1
                Z4_re_beamP2 = Z4_beamP2_cplx0
                Z4_im_beamP2 = Z4_beamP2_cplx1
                Z4_re_beamP3 = Z4_beamP3_cplx0
                Z4_im_beamP3 = Z4_beamP3_cplx1
                Z4_re_beamP4 = Z4_beamP4_cplx0
                Z4_im_beamP4 = Z4_beamP4_cplx1
                Z4_re_beamP5 = Z4_beamP5_cplx0
                Z4_im_beamP5 = Z4_beamP5_cplx1
                Z4_re_beamP6 = Z4_beamP6_cplx0
                Z4_im_beamP6 = Z4_beamP6_cplx1
                Z4_re_beamP7 = Z4_beamP7_cplx0
                Z4_im_beamP7 = Z4_beamP7_cplx1
                Z4_beamP0_cplx_in0 = Z4_re_beamP0
                Z4_beamP0_cplx_in1 = Z4_im_beamP0
                Z4_beamP1_cplx_in0 = Z4_re_beamP1
                Z4_beamP1_cplx_in1 = Z4_im_beamP1
                Z4_beamP2_cplx_in0 = Z4_re_beamP2
                Z4_beamP2_cplx_in1 = Z4_im_beamP2
                Z4_beamP3_cplx_in0 = Z4_re_beamP3
                Z4_beamP3_cplx_in1 = Z4_im_beamP3
                Z4_beamP4_cplx_in0 = Z4_re_beamP4
                Z4_beamP4_cplx_in1 = Z4_im_beamP4
                Z4_beamP5_cplx_in0 = Z4_re_beamP5
                Z4_beamP5_cplx_in1 = Z4_im_beamP5
                Z4_beamP6_cplx_in0 = Z4_re_beamP6
                Z4_beamP6_cplx_in1 = Z4_im_beamP6
                Z4_beamP7_cplx_in0 = Z4_re_beamP7
                Z4_beamP7_cplx_in1 = Z4_im_beamP7
                (Y′_beamP0_cplx0, Y′_beamP0_cplx1) = IndexSpaces.mma_m16n8k16(
                    (Γ5_beamP0_cplx0_cplx_in0, Γ5_beamP0_cplx1_cplx_in0, Γ5_beamP0_cplx0_cplx_in1, Γ5_beamP0_cplx1_cplx_in1),
                    (Z4_beamP0_cplx_in0, Z4_beamP0_cplx_in1),
                    (Y′_beamP0_cplx0, Y′_beamP0_cplx1),
                )
                (Y′_beamP1_cplx0, Y′_beamP1_cplx1) = IndexSpaces.mma_m16n8k16(
                    (Γ5_beamP1_cplx0_cplx_in0, Γ5_beamP1_cplx1_cplx_in0, Γ5_beamP1_cplx0_cplx_in1, Γ5_beamP1_cplx1_cplx_in1),
                    (Z4_beamP1_cplx_in0, Z4_beamP1_cplx_in1),
                    (Y′_beamP1_cplx0, Y′_beamP1_cplx1),
                )
                (Y′_beamP2_cplx0, Y′_beamP2_cplx1) = IndexSpaces.mma_m16n8k16(
                    (Γ5_beamP2_cplx0_cplx_in0, Γ5_beamP2_cplx1_cplx_in0, Γ5_beamP2_cplx0_cplx_in1, Γ5_beamP2_cplx1_cplx_in1),
                    (Z4_beamP2_cplx_in0, Z4_beamP2_cplx_in1),
                    (Y′_beamP2_cplx0, Y′_beamP2_cplx1),
                )
                (Y′_beamP3_cplx0, Y′_beamP3_cplx1) = IndexSpaces.mma_m16n8k16(
                    (Γ5_beamP3_cplx0_cplx_in0, Γ5_beamP3_cplx1_cplx_in0, Γ5_beamP3_cplx0_cplx_in1, Γ5_beamP3_cplx1_cplx_in1),
                    (Z4_beamP3_cplx_in0, Z4_beamP3_cplx_in1),
                    (Y′_beamP3_cplx0, Y′_beamP3_cplx1),
                )
                (Y′_beamP4_cplx0, Y′_beamP4_cplx1) = IndexSpaces.mma_m16n8k16(
                    (Γ5_beamP4_cplx0_cplx_in0, Γ5_beamP4_cplx1_cplx_in0, Γ5_beamP4_cplx0_cplx_in1, Γ5_beamP4_cplx1_cplx_in1),
                    (Z4_beamP4_cplx_in0, Z4_beamP4_cplx_in1),
                    (Y′_beamP4_cplx0, Y′_beamP4_cplx1),
                )
                (Y′_beamP5_cplx0, Y′_beamP5_cplx1) = IndexSpaces.mma_m16n8k16(
                    (Γ5_beamP5_cplx0_cplx_in0, Γ5_beamP5_cplx1_cplx_in0, Γ5_beamP5_cplx0_cplx_in1, Γ5_beamP5_cplx1_cplx_in1),
                    (Z4_beamP5_cplx_in0, Z4_beamP5_cplx_in1),
                    (Y′_beamP5_cplx0, Y′_beamP5_cplx1),
                )
                (Y′_beamP6_cplx0, Y′_beamP6_cplx1) = IndexSpaces.mma_m16n8k16(
                    (Γ5_beamP6_cplx0_cplx_in0, Γ5_beamP6_cplx1_cplx_in0, Γ5_beamP6_cplx0_cplx_in1, Γ5_beamP6_cplx1_cplx_in1),
                    (Z4_beamP6_cplx_in0, Z4_beamP6_cplx_in1),
                    (Y′_beamP6_cplx0, Y′_beamP6_cplx1),
                )
                (Y′_beamP7_cplx0, Y′_beamP7_cplx1) = IndexSpaces.mma_m16n8k16(
                    (Γ5_beamP7_cplx0_cplx_in0, Γ5_beamP7_cplx1_cplx_in0, Γ5_beamP7_cplx0_cplx_in1, Γ5_beamP7_cplx1_cplx_in1),
                    (Z4_beamP7_cplx_in0, Z4_beamP7_cplx_in1),
                    (Y′_beamP7_cplx0, Y′_beamP7_cplx1),
                )
                (Y_beamP0_cplx0, Y_beamP1_cplx0) = (
                    IndexSpaces.get_lo16(Y′_beamP0_cplx0, Y′_beamP1_cplx0), IndexSpaces.get_hi16(Y′_beamP0_cplx0, Y′_beamP1_cplx0)
                )
                (Y_beamP2_cplx0, Y_beamP3_cplx0) = (
                    IndexSpaces.get_lo16(Y′_beamP2_cplx0, Y′_beamP3_cplx0), IndexSpaces.get_hi16(Y′_beamP2_cplx0, Y′_beamP3_cplx0)
                )
                (Y_beamP4_cplx0, Y_beamP5_cplx0) = (
                    IndexSpaces.get_lo16(Y′_beamP4_cplx0, Y′_beamP5_cplx0), IndexSpaces.get_hi16(Y′_beamP4_cplx0, Y′_beamP5_cplx0)
                )
                (Y_beamP6_cplx0, Y_beamP7_cplx0) = (
                    IndexSpaces.get_lo16(Y′_beamP6_cplx0, Y′_beamP7_cplx0), IndexSpaces.get_hi16(Y′_beamP6_cplx0, Y′_beamP7_cplx0)
                )
                (Y_beamP0_cplx1, Y_beamP1_cplx1) = (
                    IndexSpaces.get_lo16(Y′_beamP0_cplx1, Y′_beamP1_cplx1), IndexSpaces.get_hi16(Y′_beamP0_cplx1, Y′_beamP1_cplx1)
                )
                (Y_beamP2_cplx1, Y_beamP3_cplx1) = (
                    IndexSpaces.get_lo16(Y′_beamP2_cplx1, Y′_beamP3_cplx1), IndexSpaces.get_hi16(Y′_beamP2_cplx1, Y′_beamP3_cplx1)
                )
                (Y_beamP4_cplx1, Y_beamP5_cplx1) = (
                    IndexSpaces.get_lo16(Y′_beamP4_cplx1, Y′_beamP5_cplx1), IndexSpaces.get_hi16(Y′_beamP4_cplx1, Y′_beamP5_cplx1)
                )
                (Y_beamP6_cplx1, Y_beamP7_cplx1) = (
                    IndexSpaces.get_lo16(Y′_beamP6_cplx1, Y′_beamP7_cplx1), IndexSpaces.get_hi16(Y′_beamP6_cplx1, Y′_beamP7_cplx1)
                )
                Y_beam3_0_beamP0_cplx0 = Y_beamP0_cplx0
                Y_beam3_1_beamP0_cplx0 = Y_beamP1_cplx0
                Y_beam3_0_beamP2_cplx0 = Y_beamP2_cplx0
                Y_beam3_1_beamP2_cplx0 = Y_beamP3_cplx0
                Y_beam3_0_beamP4_cplx0 = Y_beamP4_cplx0
                Y_beam3_1_beamP4_cplx0 = Y_beamP5_cplx0
                Y_beam3_0_beamP6_cplx0 = Y_beamP6_cplx0
                Y_beam3_1_beamP6_cplx0 = Y_beamP7_cplx0
                Y_beam3_0_beamP0_cplx1 = Y_beamP0_cplx1
                Y_beam3_1_beamP0_cplx1 = Y_beamP1_cplx1
                Y_beam3_0_beamP2_cplx1 = Y_beamP2_cplx1
                Y_beam3_1_beamP2_cplx1 = Y_beamP3_cplx1
                Y_beam3_0_beamP4_cplx1 = Y_beamP4_cplx1
                Y_beam3_1_beamP4_cplx1 = Y_beamP5_cplx1
                Y_beam3_0_beamP6_cplx1 = Y_beamP6_cplx1
                Y_beam3_1_beamP6_cplx1 = Y_beamP7_cplx1
                Y_beamP0_cplx0 = Y_beam3_0_beamP0_cplx0
                Y_beamP8_cplx0 = Y_beam3_1_beamP0_cplx0
                Y_beamP2_cplx0 = Y_beam3_0_beamP2_cplx0
                Y_beamP10_cplx0 = Y_beam3_1_beamP2_cplx0
                Y_beamP4_cplx0 = Y_beam3_0_beamP4_cplx0
                Y_beamP12_cplx0 = Y_beam3_1_beamP4_cplx0
                Y_beamP6_cplx0 = Y_beam3_0_beamP6_cplx0
                Y_beamP14_cplx0 = Y_beam3_1_beamP6_cplx0
                Y_beamP0_cplx1 = Y_beam3_0_beamP0_cplx1
                Y_beamP8_cplx1 = Y_beam3_1_beamP0_cplx1
                Y_beamP2_cplx1 = Y_beam3_0_beamP2_cplx1
                Y_beamP10_cplx1 = Y_beam3_1_beamP2_cplx1
                Y_beamP4_cplx1 = Y_beam3_0_beamP4_cplx1
                Y_beamP12_cplx1 = Y_beam3_1_beamP4_cplx1
                Y_beamP6_cplx1 = Y_beam3_0_beamP6_cplx1
                Y_beamP14_cplx1 = Y_beam3_1_beamP6_cplx1
                Y_shared[(((((((0::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 4) * 256) ÷ 256) % 4) * 512 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) ÷ 4) % 2) % 2) * 2048 + ((0::Int32 % 2) % 2) * 32 + ((((0::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0) + 0x01] =
                    Y_beamP0_cplx0
                Y_shared[(((((((2::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 4) * 256) ÷ 256) % 4) * 512 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) ÷ 4) % 2) % 2) * 2048 + ((0::Int32 % 2) % 2) * 32 + ((((2::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0) + 0x01] =
                    Y_beamP2_cplx0
                Y_shared[(((((((4::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 4) * 256) ÷ 256) % 4) * 512 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) ÷ 4) % 2) % 2) * 2048 + ((0::Int32 % 2) % 2) * 32 + ((((4::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0) + 0x01] =
                    Y_beamP4_cplx0
                Y_shared[(((((((6::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 4) * 256) ÷ 256) % 4) * 512 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) ÷ 4) % 2) % 2) * 2048 + ((0::Int32 % 2) % 2) * 32 + ((((6::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0) + 0x01] =
                    Y_beamP6_cplx0
                Y_shared[(((((((8::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 4) * 256) ÷ 256) % 4) * 512 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) ÷ 4) % 2) % 2) * 2048 + ((0::Int32 % 2) % 2) * 32 + ((((8::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0) + 0x01] =
                    Y_beamP8_cplx0
                Y_shared[(((((((10::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 4) * 256) ÷ 256) % 4) * 512 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) ÷ 4) % 2) % 2) * 2048 + ((0::Int32 % 2) % 2) * 32 + ((((10::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0) + 0x01] =
                    Y_beamP10_cplx0
                Y_shared[(((((((12::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 4) * 256) ÷ 256) % 4) * 512 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) ÷ 4) % 2) % 2) * 2048 + ((0::Int32 % 2) % 2) * 32 + ((((12::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0) + 0x01] =
                    Y_beamP12_cplx0
                Y_shared[(((((((14::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 4) * 256) ÷ 256) % 4) * 512 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) ÷ 4) % 2) % 2) * 2048 + ((0::Int32 % 2) % 2) * 32 + ((((14::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0) + 0x01] =
                    Y_beamP14_cplx0
                Y_shared[(((((((0::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 4) * 256) ÷ 256) % 4) * 512 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) ÷ 4) % 2) % 2) * 2048 + ((1::Int32 % 2) % 2) * 32 + ((((0::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0) + 0x01] =
                    Y_beamP0_cplx1
                Y_shared[(((((((2::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 4) * 256) ÷ 256) % 4) * 512 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) ÷ 4) % 2) % 2) * 2048 + ((1::Int32 % 2) % 2) * 32 + ((((2::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0) + 0x01] =
                    Y_beamP2_cplx1
                Y_shared[(((((((4::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 4) * 256) ÷ 256) % 4) * 512 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) ÷ 4) % 2) % 2) * 2048 + ((1::Int32 % 2) % 2) * 32 + ((((4::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0) + 0x01] =
                    Y_beamP4_cplx1
                Y_shared[(((((((6::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 4) * 256) ÷ 256) % 4) * 512 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) ÷ 4) % 2) % 2) * 2048 + ((1::Int32 % 2) % 2) * 32 + ((((6::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0) + 0x01] =
                    Y_beamP6_cplx1
                Y_shared[(((((((8::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 4) * 256) ÷ 256) % 4) * 512 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) ÷ 4) % 2) % 2) * 2048 + ((1::Int32 % 2) % 2) * 32 + ((((8::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0) + 0x01] =
                    Y_beamP8_cplx1
                Y_shared[(((((((10::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 4) * 256) ÷ 256) % 4) * 512 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) ÷ 4) % 2) % 2) * 2048 + ((1::Int32 % 2) % 2) * 32 + ((((10::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0) + 0x01] =
                    Y_beamP10_cplx1
                Y_shared[(((((((12::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 4) * 256) ÷ 256) % 4) * 512 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) ÷ 4) % 2) % 2) * 2048 + ((1::Int32 % 2) % 2) * 32 + ((((12::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0) + 0x01] =
                    Y_beamP12_cplx1
                Y_shared[(((((((14::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 4) * 256) ÷ 256) % 4) * 512 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) ÷ 4) % 2) % 2) * 2048 + ((1::Int32 % 2) % 2) * 32 + ((((14::Int32 ÷ 2) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0) + 0x01] =
                    Y_beamP14_cplx1
                IndexSpaces.cuda_sync_threads()
                X_cplx0_dish0_polr0 = Y_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + (((((0::Int32 ÷ 256) % 4) * 256) ÷ 256) % 4) * 512 + ((0::Int32 % 2) % 2) * 2048 + ((0::Int32 % 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0x01]
                X_cplx1_dish0_polr0 = Y_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + (((((0::Int32 ÷ 256) % 4) * 256) ÷ 256) % 4) * 512 + ((0::Int32 % 2) % 2) * 2048 + ((1::Int32 % 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0x01]
                X_cplx0_dish256_polr0 = Y_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + (((((256::Int32 ÷ 256) % 4) * 256) ÷ 256) % 4) * 512 + ((0::Int32 % 2) % 2) * 2048 + ((0::Int32 % 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0x01]
                X_cplx1_dish256_polr0 = Y_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + (((((256::Int32 ÷ 256) % 4) * 256) ÷ 256) % 4) * 512 + ((0::Int32 % 2) % 2) * 2048 + ((1::Int32 % 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0x01]
                X_cplx0_dish512_polr0 = Y_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + (((((512::Int32 ÷ 256) % 4) * 256) ÷ 256) % 4) * 512 + ((0::Int32 % 2) % 2) * 2048 + ((0::Int32 % 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0x01]
                X_cplx1_dish512_polr0 = Y_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + (((((512::Int32 ÷ 256) % 4) * 256) ÷ 256) % 4) * 512 + ((0::Int32 % 2) % 2) * 2048 + ((1::Int32 % 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0x01]
                X_cplx0_dish768_polr0 = Y_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + (((((768::Int32 ÷ 256) % 4) * 256) ÷ 256) % 4) * 512 + ((0::Int32 % 2) % 2) * 2048 + ((0::Int32 % 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0x01]
                X_cplx1_dish768_polr0 = Y_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + (((((768::Int32 ÷ 256) % 4) * 256) ÷ 256) % 4) * 512 + ((0::Int32 % 2) % 2) * 2048 + ((1::Int32 % 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0x01]
                X_cplx0_dish0_polr1 = Y_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + (((((0::Int32 ÷ 256) % 4) * 256) ÷ 256) % 4) * 512 + ((1::Int32 % 2) % 2) * 2048 + ((0::Int32 % 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0x01]
                X_cplx1_dish0_polr1 = Y_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + (((((0::Int32 ÷ 256) % 4) * 256) ÷ 256) % 4) * 512 + ((1::Int32 % 2) % 2) * 2048 + ((1::Int32 % 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0x01]
                X_cplx0_dish256_polr1 = Y_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + (((((256::Int32 ÷ 256) % 4) * 256) ÷ 256) % 4) * 512 + ((1::Int32 % 2) % 2) * 2048 + ((0::Int32 % 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0x01]
                X_cplx1_dish256_polr1 = Y_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + (((((256::Int32 ÷ 256) % 4) * 256) ÷ 256) % 4) * 512 + ((1::Int32 % 2) % 2) * 2048 + ((1::Int32 % 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0x01]
                X_cplx0_dish512_polr1 = Y_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + (((((512::Int32 ÷ 256) % 4) * 256) ÷ 256) % 4) * 512 + ((1::Int32 % 2) % 2) * 2048 + ((0::Int32 % 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0x01]
                X_cplx1_dish512_polr1 = Y_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + (((((512::Int32 ÷ 256) % 4) * 256) ÷ 256) % 4) * 512 + ((1::Int32 % 2) % 2) * 2048 + ((1::Int32 % 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0x01]
                X_cplx0_dish768_polr1 = Y_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + (((((768::Int32 ÷ 256) % 4) * 256) ÷ 256) % 4) * 512 + ((1::Int32 % 2) % 2) * 2048 + ((0::Int32 % 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0x01]
                X_cplx1_dish768_polr1 = Y_shared[(((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 8) * 64 + (((((768::Int32 ÷ 256) % 4) * 256) ÷ 256) % 4) * 512 + ((1::Int32 % 2) % 2) * 2048 + ((1::Int32 % 2) % 2) * 32 + (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 16) % 32) + 0x01]
                X_re_dish0_polr0 = X_cplx0_dish0_polr0
                X_im_dish0_polr0 = X_cplx1_dish0_polr0
                X_re_dish256_polr0 = X_cplx0_dish256_polr0
                X_im_dish256_polr0 = X_cplx1_dish256_polr0
                X_re_dish512_polr0 = X_cplx0_dish512_polr0
                X_im_dish512_polr0 = X_cplx1_dish512_polr0
                X_re_dish768_polr0 = X_cplx0_dish768_polr0
                X_im_dish768_polr0 = X_cplx1_dish768_polr0
                X_re_dish0_polr1 = X_cplx0_dish0_polr1
                X_im_dish0_polr1 = X_cplx1_dish0_polr1
                X_re_dish256_polr1 = X_cplx0_dish256_polr1
                X_im_dish256_polr1 = X_cplx1_dish256_polr1
                X_re_dish512_polr1 = X_cplx0_dish512_polr1
                X_im_dish512_polr1 = X_cplx1_dish512_polr1
                X_re_dish768_polr1 = X_cplx0_dish768_polr1
                X_im_dish768_polr1 = X_cplx1_dish768_polr1
                X_dish0_re_polr0 = X_re_dish0_polr0
                X_dish1_re_polr0 = X_re_dish256_polr0
                X_dish2_re_polr0 = X_re_dish512_polr0
                X_dish3_re_polr0 = X_re_dish768_polr0
                X_dish0_re_polr1 = X_re_dish0_polr1
                X_dish1_re_polr1 = X_re_dish256_polr1
                X_dish2_re_polr1 = X_re_dish512_polr1
                X_dish3_re_polr1 = X_re_dish768_polr1
                X_dish0_im_polr0 = X_im_dish0_polr0
                X_dish1_im_polr0 = X_im_dish256_polr0
                X_dish2_im_polr0 = X_im_dish512_polr0
                X_dish3_im_polr0 = X_im_dish768_polr0
                X_dish0_im_polr1 = X_im_dish0_polr1
                X_dish1_im_polr1 = X_im_dish256_polr1
                X_dish2_im_polr1 = X_im_dish512_polr1
                X_dish3_im_polr1 = X_im_dish768_polr1
                Y_beamQ0_re_polr0 = X_dish0_re_polr0 + +X_dish1_re_polr0 + +X_dish2_re_polr0 + +X_dish3_re_polr0
                Y_beamQ0_re_polr1 = X_dish0_re_polr1 + +X_dish1_re_polr1 + +X_dish2_re_polr1 + +X_dish3_re_polr1
                Y_beamQ0_im_polr0 = X_dish0_im_polr0 + +X_dish1_im_polr0 + +X_dish2_im_polr0 + +X_dish3_im_polr0
                Y_beamQ0_im_polr1 = X_dish0_im_polr1 + +X_dish1_im_polr1 + +X_dish2_im_polr1 + +X_dish3_im_polr1
                Y_beamQ1_re_polr0 = muladd(
                    +Float16x2(0.70703125f0, 0.70703125f0),
                    X_dish1_re_polr0,
                    muladd(-Float16x2(0.70703125f0, 0.70703125f0), X_dish3_im_polr0, X_dish0_re_polr0 + -X_dish2_im_polr0),
                )
                Y_beamQ1_re_polr1 = muladd(
                    +Float16x2(0.70703125f0, 0.70703125f0),
                    X_dish1_re_polr1,
                    muladd(-Float16x2(0.70703125f0, 0.70703125f0), X_dish3_im_polr1, X_dish0_re_polr1 + -X_dish2_im_polr1),
                )
                Y_beamQ1_im_polr0 = muladd(
                    +Float16x2(0.70703125f0, 0.70703125f0),
                    X_dish1_im_polr0,
                    muladd(+Float16x2(0.70703125f0, 0.70703125f0), X_dish3_re_polr0, X_dish0_im_polr0 + +X_dish2_re_polr0),
                )
                Y_beamQ1_im_polr1 = muladd(
                    +Float16x2(0.70703125f0, 0.70703125f0),
                    X_dish1_im_polr1,
                    muladd(+Float16x2(0.70703125f0, 0.70703125f0), X_dish3_re_polr1, X_dish0_im_polr1 + +X_dish2_re_polr1),
                )
                Y_beamQ2_re_polr0 = X_dish0_re_polr0 + -X_dish1_im_polr0 + -X_dish2_re_polr0 + +X_dish3_im_polr0
                Y_beamQ2_re_polr1 = X_dish0_re_polr1 + -X_dish1_im_polr1 + -X_dish2_re_polr1 + +X_dish3_im_polr1
                Y_beamQ2_im_polr0 = X_dish0_im_polr0 + +X_dish1_re_polr0 + -X_dish2_im_polr0 + -X_dish3_re_polr0
                Y_beamQ2_im_polr1 = X_dish0_im_polr1 + +X_dish1_re_polr1 + -X_dish2_im_polr1 + -X_dish3_re_polr1
                Y_beamQ3_re_polr0 = muladd(
                    -Float16x2(0.70703125f0, 0.70703125f0),
                    X_dish1_im_polr0,
                    muladd(+Float16x2(0.70703125f0, 0.70703125f0), X_dish3_re_polr0, X_dish0_re_polr0 + +X_dish2_im_polr0),
                )
                Y_beamQ3_re_polr1 = muladd(
                    -Float16x2(0.70703125f0, 0.70703125f0),
                    X_dish1_im_polr1,
                    muladd(+Float16x2(0.70703125f0, 0.70703125f0), X_dish3_re_polr1, X_dish0_re_polr1 + +X_dish2_im_polr1),
                )
                Y_beamQ3_im_polr0 = muladd(
                    +Float16x2(0.70703125f0, 0.70703125f0),
                    X_dish1_re_polr0,
                    muladd(+Float16x2(0.70703125f0, 0.70703125f0), X_dish3_im_polr0, X_dish0_im_polr0 + -X_dish2_re_polr0),
                )
                Y_beamQ3_im_polr1 = muladd(
                    +Float16x2(0.70703125f0, 0.70703125f0),
                    X_dish1_re_polr1,
                    muladd(+Float16x2(0.70703125f0, 0.70703125f0), X_dish3_im_polr1, X_dish0_im_polr1 + -X_dish2_re_polr1),
                )
                Y_beamQ4_re_polr0 = X_dish0_re_polr0 + -X_dish1_re_polr0 + +X_dish2_re_polr0 + -X_dish3_re_polr0
                Y_beamQ4_re_polr1 = X_dish0_re_polr1 + -X_dish1_re_polr1 + +X_dish2_re_polr1 + -X_dish3_re_polr1
                Y_beamQ4_im_polr0 = X_dish0_im_polr0 + -X_dish1_im_polr0 + +X_dish2_im_polr0 + -X_dish3_im_polr0
                Y_beamQ4_im_polr1 = X_dish0_im_polr1 + -X_dish1_im_polr1 + +X_dish2_im_polr1 + -X_dish3_im_polr1
                Y_beamQ5_re_polr0 = muladd(
                    -Float16x2(0.70703125f0, 0.70703125f0),
                    X_dish1_re_polr0,
                    muladd(+Float16x2(0.70703125f0, 0.70703125f0), X_dish3_im_polr0, X_dish0_re_polr0 + -X_dish2_im_polr0),
                )
                Y_beamQ5_re_polr1 = muladd(
                    -Float16x2(0.70703125f0, 0.70703125f0),
                    X_dish1_re_polr1,
                    muladd(+Float16x2(0.70703125f0, 0.70703125f0), X_dish3_im_polr1, X_dish0_re_polr1 + -X_dish2_im_polr1),
                )
                Y_beamQ5_im_polr0 = muladd(
                    -Float16x2(0.70703125f0, 0.70703125f0),
                    X_dish1_im_polr0,
                    muladd(-Float16x2(0.70703125f0, 0.70703125f0), X_dish3_re_polr0, X_dish0_im_polr0 + +X_dish2_re_polr0),
                )
                Y_beamQ5_im_polr1 = muladd(
                    -Float16x2(0.70703125f0, 0.70703125f0),
                    X_dish1_im_polr1,
                    muladd(-Float16x2(0.70703125f0, 0.70703125f0), X_dish3_re_polr1, X_dish0_im_polr1 + +X_dish2_re_polr1),
                )
                Y_beamQ6_re_polr0 = X_dish0_re_polr0 + +X_dish1_im_polr0 + -X_dish2_re_polr0 + -X_dish3_im_polr0
                Y_beamQ6_re_polr1 = X_dish0_re_polr1 + +X_dish1_im_polr1 + -X_dish2_re_polr1 + -X_dish3_im_polr1
                Y_beamQ6_im_polr0 = X_dish0_im_polr0 + -X_dish1_re_polr0 + -X_dish2_im_polr0 + +X_dish3_re_polr0
                Y_beamQ6_im_polr1 = X_dish0_im_polr1 + -X_dish1_re_polr1 + -X_dish2_im_polr1 + +X_dish3_re_polr1
                Y_beamQ7_re_polr0 = muladd(
                    +Float16x2(0.70703125f0, 0.70703125f0),
                    X_dish1_im_polr0,
                    muladd(-Float16x2(0.70703125f0, 0.70703125f0), X_dish3_re_polr0, X_dish0_re_polr0 + +X_dish2_im_polr0),
                )
                Y_beamQ7_re_polr1 = muladd(
                    +Float16x2(0.70703125f0, 0.70703125f0),
                    X_dish1_im_polr1,
                    muladd(-Float16x2(0.70703125f0, 0.70703125f0), X_dish3_re_polr1, X_dish0_re_polr1 + +X_dish2_im_polr1),
                )
                Y_beamQ7_im_polr0 = muladd(
                    -Float16x2(0.70703125f0, 0.70703125f0),
                    X_dish1_re_polr0,
                    muladd(-Float16x2(0.70703125f0, 0.70703125f0), X_dish3_im_polr0, X_dish0_im_polr0 + -X_dish2_re_polr0),
                )
                Y_beamQ7_im_polr1 = muladd(
                    -Float16x2(0.70703125f0, 0.70703125f0),
                    X_dish1_re_polr1,
                    muladd(-Float16x2(0.70703125f0, 0.70703125f0), X_dish3_im_polr1, X_dish0_im_polr1 + -X_dish2_re_polr1),
                )
                Y_re_beamQ0_polr0 = Y_beamQ0_re_polr0
                Y_re_beamQ1_polr0 = Y_beamQ1_re_polr0
                Y_re_beamQ2_polr0 = Y_beamQ2_re_polr0
                Y_re_beamQ3_polr0 = Y_beamQ3_re_polr0
                Y_re_beamQ4_polr0 = Y_beamQ4_re_polr0
                Y_re_beamQ5_polr0 = Y_beamQ5_re_polr0
                Y_re_beamQ6_polr0 = Y_beamQ6_re_polr0
                Y_re_beamQ7_polr0 = Y_beamQ7_re_polr0
                Y_re_beamQ0_polr1 = Y_beamQ0_re_polr1
                Y_re_beamQ1_polr1 = Y_beamQ1_re_polr1
                Y_re_beamQ2_polr1 = Y_beamQ2_re_polr1
                Y_re_beamQ3_polr1 = Y_beamQ3_re_polr1
                Y_re_beamQ4_polr1 = Y_beamQ4_re_polr1
                Y_re_beamQ5_polr1 = Y_beamQ5_re_polr1
                Y_re_beamQ6_polr1 = Y_beamQ6_re_polr1
                Y_re_beamQ7_polr1 = Y_beamQ7_re_polr1
                Y_im_beamQ0_polr0 = Y_beamQ0_im_polr0
                Y_im_beamQ1_polr0 = Y_beamQ1_im_polr0
                Y_im_beamQ2_polr0 = Y_beamQ2_im_polr0
                Y_im_beamQ3_polr0 = Y_beamQ3_im_polr0
                Y_im_beamQ4_polr0 = Y_beamQ4_im_polr0
                Y_im_beamQ5_polr0 = Y_beamQ5_im_polr0
                Y_im_beamQ6_polr0 = Y_beamQ6_im_polr0
                Y_im_beamQ7_polr0 = Y_beamQ7_im_polr0
                Y_im_beamQ0_polr1 = Y_beamQ0_im_polr1
                Y_im_beamQ1_polr1 = Y_beamQ1_im_polr1
                Y_im_beamQ2_polr1 = Y_beamQ2_im_polr1
                Y_im_beamQ3_polr1 = Y_beamQ3_im_polr1
                Y_im_beamQ4_polr1 = Y_beamQ4_im_polr1
                Y_im_beamQ5_polr1 = Y_beamQ5_im_polr1
                Y_im_beamQ6_polr1 = Y_beamQ6_im_polr1
                Y_im_beamQ7_polr1 = Y_beamQ7_im_polr1
                Y_polr0_re_beamQ0 = Y_re_beamQ0_polr0
                Y_polr1_re_beamQ0 = Y_re_beamQ0_polr1
                Y_polr0_re_beamQ1 = Y_re_beamQ1_polr0
                Y_polr1_re_beamQ1 = Y_re_beamQ1_polr1
                Y_polr0_re_beamQ2 = Y_re_beamQ2_polr0
                Y_polr1_re_beamQ2 = Y_re_beamQ2_polr1
                Y_polr0_re_beamQ3 = Y_re_beamQ3_polr0
                Y_polr1_re_beamQ3 = Y_re_beamQ3_polr1
                Y_polr0_re_beamQ4 = Y_re_beamQ4_polr0
                Y_polr1_re_beamQ4 = Y_re_beamQ4_polr1
                Y_polr0_re_beamQ5 = Y_re_beamQ5_polr0
                Y_polr1_re_beamQ5 = Y_re_beamQ5_polr1
                Y_polr0_re_beamQ6 = Y_re_beamQ6_polr0
                Y_polr1_re_beamQ6 = Y_re_beamQ6_polr1
                Y_polr0_re_beamQ7 = Y_re_beamQ7_polr0
                Y_polr1_re_beamQ7 = Y_re_beamQ7_polr1
                Y_polr0_im_beamQ0 = Y_im_beamQ0_polr0
                Y_polr1_im_beamQ0 = Y_im_beamQ0_polr1
                Y_polr0_im_beamQ1 = Y_im_beamQ1_polr0
                Y_polr1_im_beamQ1 = Y_im_beamQ1_polr1
                Y_polr0_im_beamQ2 = Y_im_beamQ2_polr0
                Y_polr1_im_beamQ2 = Y_im_beamQ2_polr1
                Y_polr0_im_beamQ3 = Y_im_beamQ3_polr0
                Y_polr1_im_beamQ3 = Y_im_beamQ3_polr1
                Y_polr0_im_beamQ4 = Y_im_beamQ4_polr0
                Y_polr1_im_beamQ4 = Y_im_beamQ4_polr1
                Y_polr0_im_beamQ5 = Y_im_beamQ5_polr0
                Y_polr1_im_beamQ5 = Y_im_beamQ5_polr1
                Y_polr0_im_beamQ6 = Y_im_beamQ6_polr0
                Y_polr1_im_beamQ6 = Y_im_beamQ6_polr1
                Y_polr0_im_beamQ7 = Y_im_beamQ7_polr0
                Y_polr1_im_beamQ7 = Y_im_beamQ7_polr1
                I_beamQ0 = muladd(
                    Float16x2(0.005207062f0, 0.005207062f0),
                    muladd(
                        Y_polr1_im_beamQ0,
                        Y_polr1_im_beamQ0,
                        muladd(
                            Y_polr1_re_beamQ0,
                            Y_polr1_re_beamQ0,
                            muladd(Y_polr0_im_beamQ0, Y_polr0_im_beamQ0, Y_polr0_re_beamQ0 * Y_polr0_re_beamQ0),
                        ),
                    ),
                    I_beamQ0,
                )
                I_beamQ1 = muladd(
                    Float16x2(0.005207062f0, 0.005207062f0),
                    muladd(
                        Y_polr1_im_beamQ1,
                        Y_polr1_im_beamQ1,
                        muladd(
                            Y_polr1_re_beamQ1,
                            Y_polr1_re_beamQ1,
                            muladd(Y_polr0_im_beamQ1, Y_polr0_im_beamQ1, Y_polr0_re_beamQ1 * Y_polr0_re_beamQ1),
                        ),
                    ),
                    I_beamQ1,
                )
                I_beamQ2 = muladd(
                    Float16x2(0.005207062f0, 0.005207062f0),
                    muladd(
                        Y_polr1_im_beamQ2,
                        Y_polr1_im_beamQ2,
                        muladd(
                            Y_polr1_re_beamQ2,
                            Y_polr1_re_beamQ2,
                            muladd(Y_polr0_im_beamQ2, Y_polr0_im_beamQ2, Y_polr0_re_beamQ2 * Y_polr0_re_beamQ2),
                        ),
                    ),
                    I_beamQ2,
                )
                I_beamQ3 = muladd(
                    Float16x2(0.005207062f0, 0.005207062f0),
                    muladd(
                        Y_polr1_im_beamQ3,
                        Y_polr1_im_beamQ3,
                        muladd(
                            Y_polr1_re_beamQ3,
                            Y_polr1_re_beamQ3,
                            muladd(Y_polr0_im_beamQ3, Y_polr0_im_beamQ3, Y_polr0_re_beamQ3 * Y_polr0_re_beamQ3),
                        ),
                    ),
                    I_beamQ3,
                )
                I_beamQ4 = muladd(
                    Float16x2(0.005207062f0, 0.005207062f0),
                    muladd(
                        Y_polr1_im_beamQ4,
                        Y_polr1_im_beamQ4,
                        muladd(
                            Y_polr1_re_beamQ4,
                            Y_polr1_re_beamQ4,
                            muladd(Y_polr0_im_beamQ4, Y_polr0_im_beamQ4, Y_polr0_re_beamQ4 * Y_polr0_re_beamQ4),
                        ),
                    ),
                    I_beamQ4,
                )
                I_beamQ5 = muladd(
                    Float16x2(0.005207062f0, 0.005207062f0),
                    muladd(
                        Y_polr1_im_beamQ5,
                        Y_polr1_im_beamQ5,
                        muladd(
                            Y_polr1_re_beamQ5,
                            Y_polr1_re_beamQ5,
                            muladd(Y_polr0_im_beamQ5, Y_polr0_im_beamQ5, Y_polr0_re_beamQ5 * Y_polr0_re_beamQ5),
                        ),
                    ),
                    I_beamQ5,
                )
                I_beamQ6 = muladd(
                    Float16x2(0.005207062f0, 0.005207062f0),
                    muladd(
                        Y_polr1_im_beamQ6,
                        Y_polr1_im_beamQ6,
                        muladd(
                            Y_polr1_re_beamQ6,
                            Y_polr1_re_beamQ6,
                            muladd(Y_polr0_im_beamQ6, Y_polr0_im_beamQ6, Y_polr0_re_beamQ6 * Y_polr0_re_beamQ6),
                        ),
                    ),
                    I_beamQ6,
                )
                I_beamQ7 = muladd(
                    Float16x2(0.005207062f0, 0.005207062f0),
                    muladd(
                        Y_polr1_im_beamQ7,
                        Y_polr1_im_beamQ7,
                        muladd(
                            Y_polr1_re_beamQ7,
                            Y_polr1_re_beamQ7,
                            muladd(Y_polr0_im_beamQ7, Y_polr0_im_beamQ7, Y_polr0_re_beamQ7 * Y_polr0_re_beamQ7),
                        ),
                    ),
                    I_beamQ7,
                )
                IndexSpaces.cuda_sync_threads()
            end
            I_memory[let
                offset = 524288 * Ttildemin
                length = 536870912
                mod((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 256) % 256) * 2048 + (((((IndexSpaces.assume_inrange(time_outer::Int32, 0, 24, 1008) ÷ 24) % 42) * 24) ÷ 24) % 1024) * 524288 + ((0::Int32 % 8) % 8) * 256) + 0) + offset, length)
            end + 0x01] = I_beamQ0
            I_memory[let
                offset = 524288 * Ttildemin
                length = 536870912
                mod((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 256) % 256) * 2048 + (((((IndexSpaces.assume_inrange(time_outer::Int32, 0, 24, 1008) ÷ 24) % 42) * 24) ÷ 24) % 1024) * 524288 + ((1::Int32 % 8) % 8) * 256) + 0) + offset, length)
            end + 0x01] = I_beamQ1
            I_memory[let
                offset = 524288 * Ttildemin
                length = 536870912
                mod((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 256) % 256) * 2048 + (((((IndexSpaces.assume_inrange(time_outer::Int32, 0, 24, 1008) ÷ 24) % 42) * 24) ÷ 24) % 1024) * 524288 + ((2::Int32 % 8) % 8) * 256) + 0) + offset, length)
            end + 0x01] = I_beamQ2
            I_memory[let
                offset = 524288 * Ttildemin
                length = 536870912
                mod((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 256) % 256) * 2048 + (((((IndexSpaces.assume_inrange(time_outer::Int32, 0, 24, 1008) ÷ 24) % 42) * 24) ÷ 24) % 1024) * 524288 + ((3::Int32 % 8) % 8) * 256) + 0) + offset, length)
            end + 0x01] = I_beamQ3
            I_memory[let
                offset = 524288 * Ttildemin
                length = 536870912
                mod((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 256) % 256) * 2048 + (((((IndexSpaces.assume_inrange(time_outer::Int32, 0, 24, 1008) ÷ 24) % 42) * 24) ÷ 24) % 1024) * 524288 + ((4::Int32 % 8) % 8) * 256) + 0) + offset, length)
            end + 0x01] = I_beamQ4
            I_memory[let
                offset = 524288 * Ttildemin
                length = 536870912
                mod((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 256) % 256) * 2048 + (((((IndexSpaces.assume_inrange(time_outer::Int32, 0, 24, 1008) ÷ 24) % 42) * 24) ÷ 24) % 1024) * 524288 + ((5::Int32 % 8) % 8) * 256) + 0) + offset, length)
            end + 0x01] = I_beamQ5
            I_memory[let
                offset = 524288 * Ttildemin
                length = 536870912
                mod((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 256) % 256) * 2048 + (((((IndexSpaces.assume_inrange(time_outer::Int32, 0, 24, 1008) ÷ 24) % 42) * 24) ÷ 24) % 1024) * 524288 + ((6::Int32 % 8) % 8) * 256) + 0) + offset, length)
            end + 0x01] = I_beamQ6
            I_memory[let
                offset = 524288 * Ttildemin
                length = 536870912
                mod((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) * 2 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) * 16) ÷ 2) % 256 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 256) % 256) * 2048 + (((((IndexSpaces.assume_inrange(time_outer::Int32, 0, 24, 1008) ÷ 24) % 42) * 24) ÷ 24) % 1024) * 524288 + ((7::Int32 % 8) % 8) * 256) + 0) + offset, length)
            end + 0x01] = I_beamQ7
        end
        info = 0
        info_memory[((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx()::Int32, 0, 8) % 8) % 8) * 32 + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx()::Int32, 0, 32) % 32) % 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx()::Int32, 0, 256) % 256) % 256) * 256) + 0) + 0x01] =
            info
    end
)
