@fastmath @inbounds(
    begin #= /home/eschnett/src/kotekan/julia/kernels/xpose.jl:319 =#
        info = 1
        if true
            info_memory[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) % 16) * 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) % 16) % 16) * 512) + 0) + 0x01] =
                info
        end
        for loop in 0:1:255
            (Ein_register0, Ein_register1, Ein_register2, Ein_register3) = IndexSpaces.unsafe_load4_global(
                Ein_memory,
                (
                    (
                        (
                            (
                                (
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 8) * 2 +
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 2) % 8) * 4096
                                ) + (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) ÷ 16
                        ) % 2048
                    ) * 32768 +
                    (
                        (
                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8 +
                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 16
                        ) ÷ 4
                    ) % 2 +
                    (
                        (
                            (
                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8 +
                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 16
                            ) ÷ 8
                        ) % 32
                    ) * 32 +
                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) % 16) % 16) * 2048 +
                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 2) % 2) * 1024 +
                    (
                        (
                            (
                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 8) * 2 +
                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 2) % 8) * 4096
                            ) + (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                        ) % 16
                    ) * 2
                ) + 1i32,
            )
            (Ein_register4, Ein_register5, Ein_register6, Ein_register7) = IndexSpaces.unsafe_load4_global(
                Ein_memory,
                (
                    (
                        (
                            (
                                (
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 8) * 2 +
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 2) % 8) * 4096
                                ) + (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) ÷ 16
                        ) % 2048
                    ) * 32768 +
                    (
                        (
                            (32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) +
                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 16
                        ) ÷ 4
                    ) % 2 +
                    (
                        (
                            (
                                (32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) +
                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 16
                            ) ÷ 8
                        ) % 32
                    ) * 32 +
                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) % 16) % 16) * 2048 +
                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 2) % 2) * 1024 +
                    (
                        (
                            (
                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 8) * 2 +
                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 2) % 8) * 4096
                            ) + (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                        ) % 16
                    ) * 2
                ) + 1i32,
            )
            (Ein_register8, Ein_register9, Ein_register10, Ein_register11) = IndexSpaces.unsafe_load4_global(
                Ein_memory,
                (
                    (
                        (
                            (
                                (
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 8) * 2 +
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 2) % 8) * 4096
                                ) + (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) ÷ 16
                        ) % 2048
                    ) * 32768 +
                    (
                        (
                            (64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) +
                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 16
                        ) ÷ 4
                    ) % 2 +
                    (
                        (
                            (
                                (64 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) +
                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 16
                            ) ÷ 8
                        ) % 32
                    ) * 32 +
                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) % 16) % 16) * 2048 +
                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 2) % 2) * 1024 +
                    (
                        (
                            (
                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 8) * 2 +
                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 2) % 8) * 4096
                            ) + (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                        ) % 16
                    ) * 2
                ) + 1i32,
            )
            (Ein_register12, Ein_register13, Ein_register14, Ein_register15) = IndexSpaces.unsafe_load4_global(
                Ein_memory,
                (
                    (
                        (
                            (
                                (
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 8) * 2 +
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 2) % 8) * 4096
                                ) + (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) ÷ 16
                        ) % 2048
                    ) * 32768 +
                    (
                        (
                            (96 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) +
                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 16
                        ) ÷ 4
                    ) % 2 +
                    (
                        (
                            (
                                (96 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) +
                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 16
                            ) ÷ 8
                        ) % 32
                    ) * 32 +
                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) % 16) % 16) * 2048 +
                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 2) % 2) * 1024 +
                    (
                        (
                            (
                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 8) * 2 +
                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 2) % 8) * 4096
                            ) + (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                        ) % 16
                    ) * 2
                ) + 1i32,
            )
            (Ein_register16, Ein_register17, Ein_register18, Ein_register19) = IndexSpaces.unsafe_load4_global(
                Ein_memory,
                (
                    (
                        (
                            (
                                (
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 8) * 2 +
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 2) % 8) * 4096
                                ) + (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) ÷ 16
                        ) % 2048
                    ) * 32768 +
                    (
                        (
                            (128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) +
                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 16
                        ) ÷ 4
                    ) % 2 +
                    (
                        (
                            (
                                (128 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) +
                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 16
                            ) ÷ 8
                        ) % 32
                    ) * 32 +
                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) % 16) % 16) * 2048 +
                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 2) % 2) * 1024 +
                    (
                        (
                            (
                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 8) * 2 +
                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 2) % 8) * 4096
                            ) + (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                        ) % 16
                    ) * 2
                ) + 1i32,
            )
            (Ein_register20, Ein_register21, Ein_register22, Ein_register23) = IndexSpaces.unsafe_load4_global(
                Ein_memory,
                (
                    (
                        (
                            (
                                (
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 8) * 2 +
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 2) % 8) * 4096
                                ) + (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) ÷ 16
                        ) % 2048
                    ) * 32768 +
                    (
                        (
                            (160 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) +
                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 16
                        ) ÷ 4
                    ) % 2 +
                    (
                        (
                            (
                                (160 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) +
                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 16
                            ) ÷ 8
                        ) % 32
                    ) * 32 +
                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) % 16) % 16) * 2048 +
                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 2) % 2) * 1024 +
                    (
                        (
                            (
                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 8) * 2 +
                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 2) % 8) * 4096
                            ) + (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                        ) % 16
                    ) * 2
                ) + 1i32,
            )
            (Ein_register24, Ein_register25, Ein_register26, Ein_register27) = IndexSpaces.unsafe_load4_global(
                Ein_memory,
                (
                    (
                        (
                            (
                                (
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 8) * 2 +
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 2) % 8) * 4096
                                ) + (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) ÷ 16
                        ) % 2048
                    ) * 32768 +
                    (
                        (
                            (192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) +
                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 16
                        ) ÷ 4
                    ) % 2 +
                    (
                        (
                            (
                                (192 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) +
                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 16
                            ) ÷ 8
                        ) % 32
                    ) * 32 +
                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) % 16) % 16) * 2048 +
                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 2) % 2) * 1024 +
                    (
                        (
                            (
                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 8) * 2 +
                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 2) % 8) * 4096
                            ) + (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                        ) % 16
                    ) * 2
                ) + 1i32,
            )
            (Ein_register28, Ein_register29, Ein_register30, Ein_register31) = IndexSpaces.unsafe_load4_global(
                Ein_memory,
                (
                    (
                        (
                            (
                                (
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 8) * 2 +
                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 2) % 8) * 4096
                                ) + (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) ÷ 16
                        ) % 2048
                    ) * 32768 +
                    (
                        (
                            (224 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) +
                            (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 16
                        ) ÷ 4
                    ) % 2 +
                    (
                        (
                            (
                                (224 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2) * 8) +
                                (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 2) * 16
                            ) ÷ 8
                        ) % 32
                    ) * 32 +
                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) % 16) % 16) * 2048 +
                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 2) % 2) * 1024 +
                    (
                        (
                            (
                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 2) % 8) * 2 +
                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 2) % 8) * 4096
                            ) + (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                        ) % 16
                    ) * 2
                ) + 1i32,
            )
            is_lo_thread = IndexSpaces.cuda_threadidx() & 0x00000010 == 0x00
            (E2_register0, E2_register2) = let
                src = if is_lo_thread
                    Ein_register2
                else
                    Ein_register0
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register0, dst)
                else
                    (dst, Ein_register2)
                end
            end
            (E2_register1, E2_register3) = let
                src = if is_lo_thread
                    Ein_register3
                else
                    Ein_register1
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register1, dst)
                else
                    (dst, Ein_register3)
                end
            end
            (E2_register4, E2_register6) = let
                src = if is_lo_thread
                    Ein_register6
                else
                    Ein_register4
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register4, dst)
                else
                    (dst, Ein_register6)
                end
            end
            (E2_register5, E2_register7) = let
                src = if is_lo_thread
                    Ein_register7
                else
                    Ein_register5
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register5, dst)
                else
                    (dst, Ein_register7)
                end
            end
            (E2_register8, E2_register10) = let
                src = if is_lo_thread
                    Ein_register10
                else
                    Ein_register8
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register8, dst)
                else
                    (dst, Ein_register10)
                end
            end
            (E2_register9, E2_register11) = let
                src = if is_lo_thread
                    Ein_register11
                else
                    Ein_register9
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register9, dst)
                else
                    (dst, Ein_register11)
                end
            end
            (E2_register12, E2_register14) = let
                src = if is_lo_thread
                    Ein_register14
                else
                    Ein_register12
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register12, dst)
                else
                    (dst, Ein_register14)
                end
            end
            (E2_register13, E2_register15) = let
                src = if is_lo_thread
                    Ein_register15
                else
                    Ein_register13
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register13, dst)
                else
                    (dst, Ein_register15)
                end
            end
            (E2_register16, E2_register18) = let
                src = if is_lo_thread
                    Ein_register18
                else
                    Ein_register16
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register16, dst)
                else
                    (dst, Ein_register18)
                end
            end
            (E2_register17, E2_register19) = let
                src = if is_lo_thread
                    Ein_register19
                else
                    Ein_register17
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register17, dst)
                else
                    (dst, Ein_register19)
                end
            end
            (E2_register20, E2_register22) = let
                src = if is_lo_thread
                    Ein_register22
                else
                    Ein_register20
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register20, dst)
                else
                    (dst, Ein_register22)
                end
            end
            (E2_register21, E2_register23) = let
                src = if is_lo_thread
                    Ein_register23
                else
                    Ein_register21
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register21, dst)
                else
                    (dst, Ein_register23)
                end
            end
            (E2_register24, E2_register26) = let
                src = if is_lo_thread
                    Ein_register26
                else
                    Ein_register24
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register24, dst)
                else
                    (dst, Ein_register26)
                end
            end
            (E2_register25, E2_register27) = let
                src = if is_lo_thread
                    Ein_register27
                else
                    Ein_register25
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register25, dst)
                else
                    (dst, Ein_register27)
                end
            end
            (E2_register28, E2_register30) = let
                src = if is_lo_thread
                    Ein_register30
                else
                    Ein_register28
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register28, dst)
                else
                    (dst, Ein_register30)
                end
            end
            (E2_register29, E2_register31) = let
                src = if is_lo_thread
                    Ein_register31
                else
                    Ein_register29
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register29, dst)
                else
                    (dst, Ein_register31)
                end
            end
            is_lo_thread = IndexSpaces.cuda_threadidx() & 0x00000002 == 0x00
            (E3_register0, E3_register4) = let
                src = if is_lo_thread
                    E2_register4
                else
                    E2_register0
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register0, dst)
                else
                    (dst, E2_register4)
                end
            end
            (E3_register1, E3_register5) = let
                src = if is_lo_thread
                    E2_register5
                else
                    E2_register1
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register1, dst)
                else
                    (dst, E2_register5)
                end
            end
            (E3_register2, E3_register6) = let
                src = if is_lo_thread
                    E2_register6
                else
                    E2_register2
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register2, dst)
                else
                    (dst, E2_register6)
                end
            end
            (E3_register3, E3_register7) = let
                src = if is_lo_thread
                    E2_register7
                else
                    E2_register3
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register3, dst)
                else
                    (dst, E2_register7)
                end
            end
            (E3_register8, E3_register12) = let
                src = if is_lo_thread
                    E2_register12
                else
                    E2_register8
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register8, dst)
                else
                    (dst, E2_register12)
                end
            end
            (E3_register9, E3_register13) = let
                src = if is_lo_thread
                    E2_register13
                else
                    E2_register9
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register9, dst)
                else
                    (dst, E2_register13)
                end
            end
            (E3_register10, E3_register14) = let
                src = if is_lo_thread
                    E2_register14
                else
                    E2_register10
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register10, dst)
                else
                    (dst, E2_register14)
                end
            end
            (E3_register11, E3_register15) = let
                src = if is_lo_thread
                    E2_register15
                else
                    E2_register11
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register11, dst)
                else
                    (dst, E2_register15)
                end
            end
            (E3_register16, E3_register20) = let
                src = if is_lo_thread
                    E2_register20
                else
                    E2_register16
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register16, dst)
                else
                    (dst, E2_register20)
                end
            end
            (E3_register17, E3_register21) = let
                src = if is_lo_thread
                    E2_register21
                else
                    E2_register17
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register17, dst)
                else
                    (dst, E2_register21)
                end
            end
            (E3_register18, E3_register22) = let
                src = if is_lo_thread
                    E2_register22
                else
                    E2_register18
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register18, dst)
                else
                    (dst, E2_register22)
                end
            end
            (E3_register19, E3_register23) = let
                src = if is_lo_thread
                    E2_register23
                else
                    E2_register19
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register19, dst)
                else
                    (dst, E2_register23)
                end
            end
            (E3_register24, E3_register28) = let
                src = if is_lo_thread
                    E2_register28
                else
                    E2_register24
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register24, dst)
                else
                    (dst, E2_register28)
                end
            end
            (E3_register25, E3_register29) = let
                src = if is_lo_thread
                    E2_register29
                else
                    E2_register25
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register25, dst)
                else
                    (dst, E2_register29)
                end
            end
            (E3_register26, E3_register30) = let
                src = if is_lo_thread
                    E2_register30
                else
                    E2_register26
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register26, dst)
                else
                    (dst, E2_register30)
                end
            end
            (E3_register27, E3_register31) = let
                src = if is_lo_thread
                    E2_register31
                else
                    E2_register27
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register27, dst)
                else
                    (dst, E2_register31)
                end
            end
            is_lo_thread = IndexSpaces.cuda_threadidx() & 0x00000004 == 0x00
            (E4_register0, E4_register8) = let
                src = if is_lo_thread
                    E3_register8
                else
                    E3_register0
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register0, dst)
                else
                    (dst, E3_register8)
                end
            end
            (E4_register1, E4_register9) = let
                src = if is_lo_thread
                    E3_register9
                else
                    E3_register1
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register1, dst)
                else
                    (dst, E3_register9)
                end
            end
            (E4_register2, E4_register10) = let
                src = if is_lo_thread
                    E3_register10
                else
                    E3_register2
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register2, dst)
                else
                    (dst, E3_register10)
                end
            end
            (E4_register3, E4_register11) = let
                src = if is_lo_thread
                    E3_register11
                else
                    E3_register3
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register3, dst)
                else
                    (dst, E3_register11)
                end
            end
            (E4_register4, E4_register12) = let
                src = if is_lo_thread
                    E3_register12
                else
                    E3_register4
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register4, dst)
                else
                    (dst, E3_register12)
                end
            end
            (E4_register5, E4_register13) = let
                src = if is_lo_thread
                    E3_register13
                else
                    E3_register5
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register5, dst)
                else
                    (dst, E3_register13)
                end
            end
            (E4_register6, E4_register14) = let
                src = if is_lo_thread
                    E3_register14
                else
                    E3_register6
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register6, dst)
                else
                    (dst, E3_register14)
                end
            end
            (E4_register7, E4_register15) = let
                src = if is_lo_thread
                    E3_register15
                else
                    E3_register7
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register7, dst)
                else
                    (dst, E3_register15)
                end
            end
            (E4_register16, E4_register24) = let
                src = if is_lo_thread
                    E3_register24
                else
                    E3_register16
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register16, dst)
                else
                    (dst, E3_register24)
                end
            end
            (E4_register17, E4_register25) = let
                src = if is_lo_thread
                    E3_register25
                else
                    E3_register17
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register17, dst)
                else
                    (dst, E3_register25)
                end
            end
            (E4_register18, E4_register26) = let
                src = if is_lo_thread
                    E3_register26
                else
                    E3_register18
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register18, dst)
                else
                    (dst, E3_register26)
                end
            end
            (E4_register19, E4_register27) = let
                src = if is_lo_thread
                    E3_register27
                else
                    E3_register19
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register19, dst)
                else
                    (dst, E3_register27)
                end
            end
            (E4_register20, E4_register28) = let
                src = if is_lo_thread
                    E3_register28
                else
                    E3_register20
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register20, dst)
                else
                    (dst, E3_register28)
                end
            end
            (E4_register21, E4_register29) = let
                src = if is_lo_thread
                    E3_register29
                else
                    E3_register21
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register21, dst)
                else
                    (dst, E3_register29)
                end
            end
            (E4_register22, E4_register30) = let
                src = if is_lo_thread
                    E3_register30
                else
                    E3_register22
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register22, dst)
                else
                    (dst, E3_register30)
                end
            end
            (E4_register23, E4_register31) = let
                src = if is_lo_thread
                    E3_register31
                else
                    E3_register23
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register23, dst)
                else
                    (dst, E3_register31)
                end
            end
            is_lo_thread = IndexSpaces.cuda_threadidx() & 0x00000008 == 0x00
            (Eout_register0, Eout_register16) = let
                src = if is_lo_thread
                    E4_register16
                else
                    E4_register0
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register0, dst)
                else
                    (dst, E4_register16)
                end
            end
            (Eout_register1, Eout_register17) = let
                src = if is_lo_thread
                    E4_register17
                else
                    E4_register1
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register1, dst)
                else
                    (dst, E4_register17)
                end
            end
            (Eout_register2, Eout_register18) = let
                src = if is_lo_thread
                    E4_register18
                else
                    E4_register2
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register2, dst)
                else
                    (dst, E4_register18)
                end
            end
            (Eout_register3, Eout_register19) = let
                src = if is_lo_thread
                    E4_register19
                else
                    E4_register3
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register3, dst)
                else
                    (dst, E4_register19)
                end
            end
            (Eout_register4, Eout_register20) = let
                src = if is_lo_thread
                    E4_register20
                else
                    E4_register4
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register4, dst)
                else
                    (dst, E4_register20)
                end
            end
            (Eout_register5, Eout_register21) = let
                src = if is_lo_thread
                    E4_register21
                else
                    E4_register5
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register5, dst)
                else
                    (dst, E4_register21)
                end
            end
            (Eout_register6, Eout_register22) = let
                src = if is_lo_thread
                    E4_register22
                else
                    E4_register6
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register6, dst)
                else
                    (dst, E4_register22)
                end
            end
            (Eout_register7, Eout_register23) = let
                src = if is_lo_thread
                    E4_register23
                else
                    E4_register7
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register7, dst)
                else
                    (dst, E4_register23)
                end
            end
            (Eout_register8, Eout_register24) = let
                src = if is_lo_thread
                    E4_register24
                else
                    E4_register8
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register8, dst)
                else
                    (dst, E4_register24)
                end
            end
            (Eout_register9, Eout_register25) = let
                src = if is_lo_thread
                    E4_register25
                else
                    E4_register9
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register9, dst)
                else
                    (dst, E4_register25)
                end
            end
            (Eout_register10, Eout_register26) = let
                src = if is_lo_thread
                    E4_register26
                else
                    E4_register10
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register10, dst)
                else
                    (dst, E4_register26)
                end
            end
            (Eout_register11, Eout_register27) = let
                src = if is_lo_thread
                    E4_register27
                else
                    E4_register11
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register11, dst)
                else
                    (dst, E4_register27)
                end
            end
            (Eout_register12, Eout_register28) = let
                src = if is_lo_thread
                    E4_register28
                else
                    E4_register12
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register12, dst)
                else
                    (dst, E4_register28)
                end
            end
            (Eout_register13, Eout_register29) = let
                src = if is_lo_thread
                    E4_register29
                else
                    E4_register13
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register13, dst)
                else
                    (dst, E4_register29)
                end
            end
            (Eout_register14, Eout_register30) = let
                src = if is_lo_thread
                    E4_register30
                else
                    E4_register14
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register14, dst)
                else
                    (dst, E4_register30)
                end
            end
            (Eout_register15, Eout_register31) = let
                src = if is_lo_thread
                    E4_register31
                else
                    E4_register15
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register15, dst)
                else
                    (dst, E4_register31)
                end
            end
            if true
                IndexSpaces.unsafe_store4_global!(
                    Eout_memory,
                    (
                        (
                            (
                                (
                                    (
                                        ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 2) % 8) * 4096 +
                                        (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                                    ) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2
                                ) % 32768
                            ) * 2048 +
                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) % 16) % 16) * 128 +
                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 2) % 2) * 64 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) ÷ 4) % 64
                        ) + 0
                    ) + 0x01,
                    (Eout_register0, Eout_register1, Eout_register2, Eout_register3),
                )
            end
            if true
                IndexSpaces.unsafe_store4_global!(
                    Eout_memory,
                    (
                        (
                            (
                                (
                                    (
                                        (2 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 2) % 8) * 4096) +
                                        (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                                    ) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2
                                ) % 32768
                            ) * 2048 +
                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) % 16) % 16) * 128 +
                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 2) % 2) * 64 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) ÷ 4) % 64
                        ) + 0
                    ) + 0x01,
                    (Eout_register4, Eout_register5, Eout_register6, Eout_register7),
                )
            end
            if true
                IndexSpaces.unsafe_store4_global!(
                    Eout_memory,
                    (
                        (
                            (
                                (
                                    (
                                        (4 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 2) % 8) * 4096) +
                                        (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                                    ) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2
                                ) % 32768
                            ) * 2048 +
                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) % 16) % 16) * 128 +
                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 2) % 2) * 64 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) ÷ 4) % 64
                        ) + 0
                    ) + 0x01,
                    (Eout_register8, Eout_register9, Eout_register10, Eout_register11),
                )
            end
            if true
                IndexSpaces.unsafe_store4_global!(
                    Eout_memory,
                    (
                        (
                            (
                                (
                                    (
                                        (6 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 2) % 8) * 4096) +
                                        (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                                    ) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2
                                ) % 32768
                            ) * 2048 +
                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) % 16) % 16) * 128 +
                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 2) % 2) * 64 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) ÷ 4) % 64
                        ) + 0
                    ) + 0x01,
                    (Eout_register12, Eout_register13, Eout_register14, Eout_register15),
                )
            end
            if true
                IndexSpaces.unsafe_store4_global!(
                    Eout_memory,
                    (
                        (
                            (
                                (
                                    (
                                        (8 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 2) % 8) * 4096) +
                                        (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                                    ) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2
                                ) % 32768
                            ) * 2048 +
                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) % 16) % 16) * 128 +
                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 2) % 2) * 64 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) ÷ 4) % 64
                        ) + 0
                    ) + 0x01,
                    (Eout_register16, Eout_register17, Eout_register18, Eout_register19),
                )
            end
            if true
                IndexSpaces.unsafe_store4_global!(
                    Eout_memory,
                    (
                        (
                            (
                                (
                                    (
                                        (10 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 2) % 8) * 4096) +
                                        (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                                    ) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2
                                ) % 32768
                            ) * 2048 +
                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) % 16) % 16) * 128 +
                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 2) % 2) * 64 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) ÷ 4) % 64
                        ) + 0
                    ) + 0x01,
                    (Eout_register20, Eout_register21, Eout_register22, Eout_register23),
                )
            end
            if true
                IndexSpaces.unsafe_store4_global!(
                    Eout_memory,
                    (
                        (
                            (
                                (
                                    (
                                        (12 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 2) % 8) * 4096) +
                                        (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                                    ) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2
                                ) % 32768
                            ) * 2048 +
                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) % 16) % 16) * 128 +
                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 2) % 2) * 64 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) ÷ 4) % 64
                        ) + 0
                    ) + 0x01,
                    (Eout_register24, Eout_register25, Eout_register26, Eout_register27),
                )
            end
            if true
                IndexSpaces.unsafe_store4_global!(
                    Eout_memory,
                    (
                        (
                            (
                                (
                                    (
                                        (14 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) ÷ 2) % 8) * 4096) +
                                        (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                                    ) + (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) ÷ 16) % 2
                                ) % 32768
                            ) * 2048 +
                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) % 16) % 16) * 128 +
                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 2) % 2) * 64 +
                            (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 16) * 16) ÷ 4) % 64
                        ) + 0
                    ) + 0x01,
                    (Eout_register28, Eout_register29, Eout_register30, Eout_register31),
                )
            end
        end
        info = 0
        if true
            info_memory[(((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(), 0, 32) % 32) % 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) % 16) % 16) * 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) % 16) % 16) * 512) + 0) + 0x01] =
                info
        end
    end
)
