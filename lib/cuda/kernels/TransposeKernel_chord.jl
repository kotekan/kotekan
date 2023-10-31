@fastmath @inbounds(
    begin #= /home/eschnett/src/kotekan/julia/kernels/xpose.jl:263 =#
        info = 1
        info_memory[((IndexSpaces.assume_inrange(
            IndexSpaces.cuda_threadidx(),
            0,
            32,
        )%32)%32+((IndexSpaces.assume_inrange(
            IndexSpaces.cuda_warpidx(),
            0,
            16,
        )%16)%16)*32+((IndexSpaces.assume_inrange(
            IndexSpaces.cuda_blockidx(),
            0,
            16,
        )%16)%16)*512)+0+0x01] = info
        for loop = 0:1:255
            (Ein_register0, Ein_register1, Ein_register2, Ein_register3) =
                IndexSpaces.unsafe_load4_global(
                    Ein_memory,
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_warpidx(),
                                    0,
                                    16,
                                ) % 2
                            ) % 2
                        ) * 2048 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_warpidx(),
                                                0,
                                                16,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 4096 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 2
                                ) +
                                (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) % 16
                        ) * 2 +
                        (
                            (
                                (
                                    (
                                        IndexSpaces.assume_inrange(
                                            IndexSpaces.cuda_threadidx(),
                                            0,
                                            32,
                                        ) % 2
                                    ) * 16 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 16
                                        ) % 2
                                    ) * 8
                                ) ÷ 8
                            ) % 64
                        ) * 32 +
                        (
                            (
                                (
                                    IndexSpaces.assume_inrange(
                                        IndexSpaces.cuda_threadidx(),
                                        0,
                                        32,
                                    ) % 2
                                ) * 16 +
                                (
                                    (
                                        IndexSpaces.assume_inrange(
                                            IndexSpaces.cuda_threadidx(),
                                            0,
                                            32,
                                        ) ÷ 16
                                    ) % 2
                                ) * 8
                            ) ÷ 4
                        ) % 2 +
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_blockidx(),
                                    0,
                                    16,
                                ) % 16
                            ) % 16
                        ) * 4096 +
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_warpidx(),
                                                    0,
                                                    16,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 4096 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 2
                                    ) +
                                    (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) *
                                    16
                                ) ÷ 16
                            ) % 2048
                        ) * 65536
                    ) + 1i32,
                )
            (Ein_register4, Ein_register5, Ein_register6, Ein_register7) =
                IndexSpaces.unsafe_load4_global(
                    Ein_memory,
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_warpidx(),
                                    0,
                                    16,
                                ) % 2
                            ) % 2
                        ) * 2048 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_warpidx(),
                                                0,
                                                16,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 4096 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 2
                                ) +
                                (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) % 16
                        ) * 2 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) % 2
                                        ) * 16 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 16
                                            ) % 2
                                        ) * 8
                                    ) + 32
                                ) ÷ 8
                            ) % 64
                        ) * 32 +
                        (
                            (
                                (
                                    (
                                        IndexSpaces.assume_inrange(
                                            IndexSpaces.cuda_threadidx(),
                                            0,
                                            32,
                                        ) % 2
                                    ) * 16 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 16
                                        ) % 2
                                    ) * 8
                                ) + 32
                            ) ÷ 4
                        ) % 2 +
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_blockidx(),
                                    0,
                                    16,
                                ) % 16
                            ) % 16
                        ) * 4096 +
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_warpidx(),
                                                    0,
                                                    16,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 4096 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 2
                                    ) +
                                    (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) *
                                    16
                                ) ÷ 16
                            ) % 2048
                        ) * 65536
                    ) + 1i32,
                )
            (Ein_register8, Ein_register9, Ein_register10, Ein_register11) =
                IndexSpaces.unsafe_load4_global(
                    Ein_memory,
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_warpidx(),
                                    0,
                                    16,
                                ) % 2
                            ) % 2
                        ) * 2048 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_warpidx(),
                                                0,
                                                16,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 4096 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 2
                                ) +
                                (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) % 16
                        ) * 2 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) % 2
                                        ) * 16 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 16
                                            ) % 2
                                        ) * 8
                                    ) + 64
                                ) ÷ 8
                            ) % 64
                        ) * 32 +
                        (
                            (
                                (
                                    (
                                        IndexSpaces.assume_inrange(
                                            IndexSpaces.cuda_threadidx(),
                                            0,
                                            32,
                                        ) % 2
                                    ) * 16 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 16
                                        ) % 2
                                    ) * 8
                                ) + 64
                            ) ÷ 4
                        ) % 2 +
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_blockidx(),
                                    0,
                                    16,
                                ) % 16
                            ) % 16
                        ) * 4096 +
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_warpidx(),
                                                    0,
                                                    16,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 4096 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 2
                                    ) +
                                    (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) *
                                    16
                                ) ÷ 16
                            ) % 2048
                        ) * 65536
                    ) + 1i32,
                )
            (Ein_register12, Ein_register13, Ein_register14, Ein_register15) =
                IndexSpaces.unsafe_load4_global(
                    Ein_memory,
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_warpidx(),
                                    0,
                                    16,
                                ) % 2
                            ) % 2
                        ) * 2048 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_warpidx(),
                                                0,
                                                16,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 4096 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 2
                                ) +
                                (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) % 16
                        ) * 2 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) % 2
                                        ) * 16 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 16
                                            ) % 2
                                        ) * 8
                                    ) + 96
                                ) ÷ 8
                            ) % 64
                        ) * 32 +
                        (
                            (
                                (
                                    (
                                        IndexSpaces.assume_inrange(
                                            IndexSpaces.cuda_threadidx(),
                                            0,
                                            32,
                                        ) % 2
                                    ) * 16 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 16
                                        ) % 2
                                    ) * 8
                                ) + 96
                            ) ÷ 4
                        ) % 2 +
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_blockidx(),
                                    0,
                                    16,
                                ) % 16
                            ) % 16
                        ) * 4096 +
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_warpidx(),
                                                    0,
                                                    16,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 4096 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 2
                                    ) +
                                    (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) *
                                    16
                                ) ÷ 16
                            ) % 2048
                        ) * 65536
                    ) + 1i32,
                )
            (Ein_register16, Ein_register17, Ein_register18, Ein_register19) =
                IndexSpaces.unsafe_load4_global(
                    Ein_memory,
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_warpidx(),
                                    0,
                                    16,
                                ) % 2
                            ) % 2
                        ) * 2048 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_warpidx(),
                                                0,
                                                16,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 4096 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 2
                                ) +
                                (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) % 16
                        ) * 2 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) % 2
                                        ) * 16 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 16
                                            ) % 2
                                        ) * 8
                                    ) + 128
                                ) ÷ 8
                            ) % 64
                        ) * 32 +
                        (
                            (
                                (
                                    (
                                        IndexSpaces.assume_inrange(
                                            IndexSpaces.cuda_threadidx(),
                                            0,
                                            32,
                                        ) % 2
                                    ) * 16 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 16
                                        ) % 2
                                    ) * 8
                                ) + 128
                            ) ÷ 4
                        ) % 2 +
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_blockidx(),
                                    0,
                                    16,
                                ) % 16
                            ) % 16
                        ) * 4096 +
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_warpidx(),
                                                    0,
                                                    16,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 4096 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 2
                                    ) +
                                    (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) *
                                    16
                                ) ÷ 16
                            ) % 2048
                        ) * 65536
                    ) + 1i32,
                )
            (Ein_register20, Ein_register21, Ein_register22, Ein_register23) =
                IndexSpaces.unsafe_load4_global(
                    Ein_memory,
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_warpidx(),
                                    0,
                                    16,
                                ) % 2
                            ) % 2
                        ) * 2048 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_warpidx(),
                                                0,
                                                16,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 4096 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 2
                                ) +
                                (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) % 16
                        ) * 2 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) % 2
                                        ) * 16 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 16
                                            ) % 2
                                        ) * 8
                                    ) + 160
                                ) ÷ 8
                            ) % 64
                        ) * 32 +
                        (
                            (
                                (
                                    (
                                        IndexSpaces.assume_inrange(
                                            IndexSpaces.cuda_threadidx(),
                                            0,
                                            32,
                                        ) % 2
                                    ) * 16 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 16
                                        ) % 2
                                    ) * 8
                                ) + 160
                            ) ÷ 4
                        ) % 2 +
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_blockidx(),
                                    0,
                                    16,
                                ) % 16
                            ) % 16
                        ) * 4096 +
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_warpidx(),
                                                    0,
                                                    16,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 4096 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 2
                                    ) +
                                    (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) *
                                    16
                                ) ÷ 16
                            ) % 2048
                        ) * 65536
                    ) + 1i32,
                )
            (Ein_register24, Ein_register25, Ein_register26, Ein_register27) =
                IndexSpaces.unsafe_load4_global(
                    Ein_memory,
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_warpidx(),
                                    0,
                                    16,
                                ) % 2
                            ) % 2
                        ) * 2048 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_warpidx(),
                                                0,
                                                16,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 4096 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 2
                                ) +
                                (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) % 16
                        ) * 2 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) % 2
                                        ) * 16 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 16
                                            ) % 2
                                        ) * 8
                                    ) + 192
                                ) ÷ 8
                            ) % 64
                        ) * 32 +
                        (
                            (
                                (
                                    (
                                        IndexSpaces.assume_inrange(
                                            IndexSpaces.cuda_threadidx(),
                                            0,
                                            32,
                                        ) % 2
                                    ) * 16 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 16
                                        ) % 2
                                    ) * 8
                                ) + 192
                            ) ÷ 4
                        ) % 2 +
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_blockidx(),
                                    0,
                                    16,
                                ) % 16
                            ) % 16
                        ) * 4096 +
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_warpidx(),
                                                    0,
                                                    16,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 4096 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 2
                                    ) +
                                    (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) *
                                    16
                                ) ÷ 16
                            ) % 2048
                        ) * 65536
                    ) + 1i32,
                )
            (Ein_register28, Ein_register29, Ein_register30, Ein_register31) =
                IndexSpaces.unsafe_load4_global(
                    Ein_memory,
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_warpidx(),
                                    0,
                                    16,
                                ) % 2
                            ) % 2
                        ) * 2048 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_warpidx(),
                                                0,
                                                16,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 4096 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 2
                                ) +
                                (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) % 16
                        ) * 2 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) % 2
                                        ) * 16 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 16
                                            ) % 2
                                        ) * 8
                                    ) + 224
                                ) ÷ 8
                            ) % 64
                        ) * 32 +
                        (
                            (
                                (
                                    (
                                        IndexSpaces.assume_inrange(
                                            IndexSpaces.cuda_threadidx(),
                                            0,
                                            32,
                                        ) % 2
                                    ) * 16 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 16
                                        ) % 2
                                    ) * 8
                                ) + 224
                            ) ÷ 4
                        ) % 2 +
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_blockidx(),
                                    0,
                                    16,
                                ) % 16
                            ) % 16
                        ) * 4096 +
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_warpidx(),
                                                    0,
                                                    16,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 4096 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 2
                                    ) +
                                    (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) *
                                    16
                                ) ÷ 16
                            ) % 2048
                        ) * 65536
                    ) + 1i32,
                )
            (Ein_register32, Ein_register33, Ein_register34, Ein_register35) =
                IndexSpaces.unsafe_load4_global(
                    Ein_memory,
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_warpidx(),
                                    0,
                                    16,
                                ) % 2
                            ) % 2
                        ) * 2048 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_warpidx(),
                                                0,
                                                16,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 4096 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 2
                                ) +
                                (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) % 16
                        ) * 2 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) % 2
                                        ) * 16 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 16
                                            ) % 2
                                        ) * 8
                                    ) + 256
                                ) ÷ 8
                            ) % 64
                        ) * 32 +
                        (
                            (
                                (
                                    (
                                        IndexSpaces.assume_inrange(
                                            IndexSpaces.cuda_threadidx(),
                                            0,
                                            32,
                                        ) % 2
                                    ) * 16 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 16
                                        ) % 2
                                    ) * 8
                                ) + 256
                            ) ÷ 4
                        ) % 2 +
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_blockidx(),
                                    0,
                                    16,
                                ) % 16
                            ) % 16
                        ) * 4096 +
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_warpidx(),
                                                    0,
                                                    16,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 4096 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 2
                                    ) +
                                    (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) *
                                    16
                                ) ÷ 16
                            ) % 2048
                        ) * 65536
                    ) + 1i32,
                )
            (Ein_register36, Ein_register37, Ein_register38, Ein_register39) =
                IndexSpaces.unsafe_load4_global(
                    Ein_memory,
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_warpidx(),
                                    0,
                                    16,
                                ) % 2
                            ) % 2
                        ) * 2048 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_warpidx(),
                                                0,
                                                16,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 4096 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 2
                                ) +
                                (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) % 16
                        ) * 2 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) % 2
                                        ) * 16 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 16
                                            ) % 2
                                        ) * 8
                                    ) + 288
                                ) ÷ 8
                            ) % 64
                        ) * 32 +
                        (
                            (
                                (
                                    (
                                        IndexSpaces.assume_inrange(
                                            IndexSpaces.cuda_threadidx(),
                                            0,
                                            32,
                                        ) % 2
                                    ) * 16 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 16
                                        ) % 2
                                    ) * 8
                                ) + 288
                            ) ÷ 4
                        ) % 2 +
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_blockidx(),
                                    0,
                                    16,
                                ) % 16
                            ) % 16
                        ) * 4096 +
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_warpidx(),
                                                    0,
                                                    16,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 4096 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 2
                                    ) +
                                    (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) *
                                    16
                                ) ÷ 16
                            ) % 2048
                        ) * 65536
                    ) + 1i32,
                )
            (Ein_register40, Ein_register41, Ein_register42, Ein_register43) =
                IndexSpaces.unsafe_load4_global(
                    Ein_memory,
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_warpidx(),
                                    0,
                                    16,
                                ) % 2
                            ) % 2
                        ) * 2048 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_warpidx(),
                                                0,
                                                16,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 4096 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 2
                                ) +
                                (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) % 16
                        ) * 2 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) % 2
                                        ) * 16 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 16
                                            ) % 2
                                        ) * 8
                                    ) + 320
                                ) ÷ 8
                            ) % 64
                        ) * 32 +
                        (
                            (
                                (
                                    (
                                        IndexSpaces.assume_inrange(
                                            IndexSpaces.cuda_threadidx(),
                                            0,
                                            32,
                                        ) % 2
                                    ) * 16 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 16
                                        ) % 2
                                    ) * 8
                                ) + 320
                            ) ÷ 4
                        ) % 2 +
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_blockidx(),
                                    0,
                                    16,
                                ) % 16
                            ) % 16
                        ) * 4096 +
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_warpidx(),
                                                    0,
                                                    16,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 4096 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 2
                                    ) +
                                    (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) *
                                    16
                                ) ÷ 16
                            ) % 2048
                        ) * 65536
                    ) + 1i32,
                )
            (Ein_register44, Ein_register45, Ein_register46, Ein_register47) =
                IndexSpaces.unsafe_load4_global(
                    Ein_memory,
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_warpidx(),
                                    0,
                                    16,
                                ) % 2
                            ) % 2
                        ) * 2048 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_warpidx(),
                                                0,
                                                16,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 4096 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 2
                                ) +
                                (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) % 16
                        ) * 2 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) % 2
                                        ) * 16 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 16
                                            ) % 2
                                        ) * 8
                                    ) + 352
                                ) ÷ 8
                            ) % 64
                        ) * 32 +
                        (
                            (
                                (
                                    (
                                        IndexSpaces.assume_inrange(
                                            IndexSpaces.cuda_threadidx(),
                                            0,
                                            32,
                                        ) % 2
                                    ) * 16 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 16
                                        ) % 2
                                    ) * 8
                                ) + 352
                            ) ÷ 4
                        ) % 2 +
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_blockidx(),
                                    0,
                                    16,
                                ) % 16
                            ) % 16
                        ) * 4096 +
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_warpidx(),
                                                    0,
                                                    16,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 4096 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 2
                                    ) +
                                    (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) *
                                    16
                                ) ÷ 16
                            ) % 2048
                        ) * 65536
                    ) + 1i32,
                )
            (Ein_register48, Ein_register49, Ein_register50, Ein_register51) =
                IndexSpaces.unsafe_load4_global(
                    Ein_memory,
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_warpidx(),
                                    0,
                                    16,
                                ) % 2
                            ) % 2
                        ) * 2048 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_warpidx(),
                                                0,
                                                16,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 4096 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 2
                                ) +
                                (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) % 16
                        ) * 2 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) % 2
                                        ) * 16 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 16
                                            ) % 2
                                        ) * 8
                                    ) + 384
                                ) ÷ 8
                            ) % 64
                        ) * 32 +
                        (
                            (
                                (
                                    (
                                        IndexSpaces.assume_inrange(
                                            IndexSpaces.cuda_threadidx(),
                                            0,
                                            32,
                                        ) % 2
                                    ) * 16 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 16
                                        ) % 2
                                    ) * 8
                                ) + 384
                            ) ÷ 4
                        ) % 2 +
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_blockidx(),
                                    0,
                                    16,
                                ) % 16
                            ) % 16
                        ) * 4096 +
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_warpidx(),
                                                    0,
                                                    16,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 4096 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 2
                                    ) +
                                    (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) *
                                    16
                                ) ÷ 16
                            ) % 2048
                        ) * 65536
                    ) + 1i32,
                )
            (Ein_register52, Ein_register53, Ein_register54, Ein_register55) =
                IndexSpaces.unsafe_load4_global(
                    Ein_memory,
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_warpidx(),
                                    0,
                                    16,
                                ) % 2
                            ) % 2
                        ) * 2048 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_warpidx(),
                                                0,
                                                16,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 4096 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 2
                                ) +
                                (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) % 16
                        ) * 2 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) % 2
                                        ) * 16 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 16
                                            ) % 2
                                        ) * 8
                                    ) + 416
                                ) ÷ 8
                            ) % 64
                        ) * 32 +
                        (
                            (
                                (
                                    (
                                        IndexSpaces.assume_inrange(
                                            IndexSpaces.cuda_threadidx(),
                                            0,
                                            32,
                                        ) % 2
                                    ) * 16 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 16
                                        ) % 2
                                    ) * 8
                                ) + 416
                            ) ÷ 4
                        ) % 2 +
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_blockidx(),
                                    0,
                                    16,
                                ) % 16
                            ) % 16
                        ) * 4096 +
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_warpidx(),
                                                    0,
                                                    16,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 4096 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 2
                                    ) +
                                    (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) *
                                    16
                                ) ÷ 16
                            ) % 2048
                        ) * 65536
                    ) + 1i32,
                )
            (Ein_register56, Ein_register57, Ein_register58, Ein_register59) =
                IndexSpaces.unsafe_load4_global(
                    Ein_memory,
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_warpidx(),
                                    0,
                                    16,
                                ) % 2
                            ) % 2
                        ) * 2048 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_warpidx(),
                                                0,
                                                16,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 4096 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 2
                                ) +
                                (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) % 16
                        ) * 2 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) % 2
                                        ) * 16 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 16
                                            ) % 2
                                        ) * 8
                                    ) + 448
                                ) ÷ 8
                            ) % 64
                        ) * 32 +
                        (
                            (
                                (
                                    (
                                        IndexSpaces.assume_inrange(
                                            IndexSpaces.cuda_threadidx(),
                                            0,
                                            32,
                                        ) % 2
                                    ) * 16 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 16
                                        ) % 2
                                    ) * 8
                                ) + 448
                            ) ÷ 4
                        ) % 2 +
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_blockidx(),
                                    0,
                                    16,
                                ) % 16
                            ) % 16
                        ) * 4096 +
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_warpidx(),
                                                    0,
                                                    16,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 4096 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 2
                                    ) +
                                    (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) *
                                    16
                                ) ÷ 16
                            ) % 2048
                        ) * 65536
                    ) + 1i32,
                )
            (Ein_register60, Ein_register61, Ein_register62, Ein_register63) =
                IndexSpaces.unsafe_load4_global(
                    Ein_memory,
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_warpidx(),
                                    0,
                                    16,
                                ) % 2
                            ) % 2
                        ) * 2048 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_warpidx(),
                                                0,
                                                16,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 4096 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 2
                                ) +
                                (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) % 16
                        ) * 2 +
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) % 2
                                        ) * 16 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 16
                                            ) % 2
                                        ) * 8
                                    ) + 480
                                ) ÷ 8
                            ) % 64
                        ) * 32 +
                        (
                            (
                                (
                                    (
                                        IndexSpaces.assume_inrange(
                                            IndexSpaces.cuda_threadidx(),
                                            0,
                                            32,
                                        ) % 2
                                    ) * 16 +
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_threadidx(),
                                                0,
                                                32,
                                            ) ÷ 16
                                        ) % 2
                                    ) * 8
                                ) + 480
                            ) ÷ 4
                        ) % 2 +
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_blockidx(),
                                    0,
                                    16,
                                ) % 16
                            ) % 16
                        ) * 4096 +
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_warpidx(),
                                                    0,
                                                    16,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 4096 +
                                        (
                                            (
                                                IndexSpaces.assume_inrange(
                                                    IndexSpaces.cuda_threadidx(),
                                                    0,
                                                    32,
                                                ) ÷ 2
                                            ) % 8
                                        ) * 2
                                    ) +
                                    (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) *
                                    16
                                ) ÷ 16
                            ) % 2048
                        ) * 65536
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
            (E2_register32, E2_register34) = let
                src = if is_lo_thread
                    Ein_register34
                else
                    Ein_register32
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register32, dst)
                else
                    (dst, Ein_register34)
                end
            end
            (E2_register33, E2_register35) = let
                src = if is_lo_thread
                    Ein_register35
                else
                    Ein_register33
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register33, dst)
                else
                    (dst, Ein_register35)
                end
            end
            (E2_register36, E2_register38) = let
                src = if is_lo_thread
                    Ein_register38
                else
                    Ein_register36
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register36, dst)
                else
                    (dst, Ein_register38)
                end
            end
            (E2_register37, E2_register39) = let
                src = if is_lo_thread
                    Ein_register39
                else
                    Ein_register37
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register37, dst)
                else
                    (dst, Ein_register39)
                end
            end
            (E2_register40, E2_register42) = let
                src = if is_lo_thread
                    Ein_register42
                else
                    Ein_register40
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register40, dst)
                else
                    (dst, Ein_register42)
                end
            end
            (E2_register41, E2_register43) = let
                src = if is_lo_thread
                    Ein_register43
                else
                    Ein_register41
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register41, dst)
                else
                    (dst, Ein_register43)
                end
            end
            (E2_register44, E2_register46) = let
                src = if is_lo_thread
                    Ein_register46
                else
                    Ein_register44
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register44, dst)
                else
                    (dst, Ein_register46)
                end
            end
            (E2_register45, E2_register47) = let
                src = if is_lo_thread
                    Ein_register47
                else
                    Ein_register45
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register45, dst)
                else
                    (dst, Ein_register47)
                end
            end
            (E2_register48, E2_register50) = let
                src = if is_lo_thread
                    Ein_register50
                else
                    Ein_register48
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register48, dst)
                else
                    (dst, Ein_register50)
                end
            end
            (E2_register49, E2_register51) = let
                src = if is_lo_thread
                    Ein_register51
                else
                    Ein_register49
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register49, dst)
                else
                    (dst, Ein_register51)
                end
            end
            (E2_register52, E2_register54) = let
                src = if is_lo_thread
                    Ein_register54
                else
                    Ein_register52
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register52, dst)
                else
                    (dst, Ein_register54)
                end
            end
            (E2_register53, E2_register55) = let
                src = if is_lo_thread
                    Ein_register55
                else
                    Ein_register53
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register53, dst)
                else
                    (dst, Ein_register55)
                end
            end
            (E2_register56, E2_register58) = let
                src = if is_lo_thread
                    Ein_register58
                else
                    Ein_register56
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register56, dst)
                else
                    (dst, Ein_register58)
                end
            end
            (E2_register57, E2_register59) = let
                src = if is_lo_thread
                    Ein_register59
                else
                    Ein_register57
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register57, dst)
                else
                    (dst, Ein_register59)
                end
            end
            (E2_register60, E2_register62) = let
                src = if is_lo_thread
                    Ein_register62
                else
                    Ein_register60
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register60, dst)
                else
                    (dst, Ein_register62)
                end
            end
            (E2_register61, E2_register63) = let
                src = if is_lo_thread
                    Ein_register63
                else
                    Ein_register61
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (Ein_register61, dst)
                else
                    (dst, Ein_register63)
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
            (E3_register32, E3_register36) = let
                src = if is_lo_thread
                    E2_register36
                else
                    E2_register32
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register32, dst)
                else
                    (dst, E2_register36)
                end
            end
            (E3_register33, E3_register37) = let
                src = if is_lo_thread
                    E2_register37
                else
                    E2_register33
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register33, dst)
                else
                    (dst, E2_register37)
                end
            end
            (E3_register34, E3_register38) = let
                src = if is_lo_thread
                    E2_register38
                else
                    E2_register34
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register34, dst)
                else
                    (dst, E2_register38)
                end
            end
            (E3_register35, E3_register39) = let
                src = if is_lo_thread
                    E2_register39
                else
                    E2_register35
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register35, dst)
                else
                    (dst, E2_register39)
                end
            end
            (E3_register40, E3_register44) = let
                src = if is_lo_thread
                    E2_register44
                else
                    E2_register40
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register40, dst)
                else
                    (dst, E2_register44)
                end
            end
            (E3_register41, E3_register45) = let
                src = if is_lo_thread
                    E2_register45
                else
                    E2_register41
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register41, dst)
                else
                    (dst, E2_register45)
                end
            end
            (E3_register42, E3_register46) = let
                src = if is_lo_thread
                    E2_register46
                else
                    E2_register42
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register42, dst)
                else
                    (dst, E2_register46)
                end
            end
            (E3_register43, E3_register47) = let
                src = if is_lo_thread
                    E2_register47
                else
                    E2_register43
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register43, dst)
                else
                    (dst, E2_register47)
                end
            end
            (E3_register48, E3_register52) = let
                src = if is_lo_thread
                    E2_register52
                else
                    E2_register48
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register48, dst)
                else
                    (dst, E2_register52)
                end
            end
            (E3_register49, E3_register53) = let
                src = if is_lo_thread
                    E2_register53
                else
                    E2_register49
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register49, dst)
                else
                    (dst, E2_register53)
                end
            end
            (E3_register50, E3_register54) = let
                src = if is_lo_thread
                    E2_register54
                else
                    E2_register50
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register50, dst)
                else
                    (dst, E2_register54)
                end
            end
            (E3_register51, E3_register55) = let
                src = if is_lo_thread
                    E2_register55
                else
                    E2_register51
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register51, dst)
                else
                    (dst, E2_register55)
                end
            end
            (E3_register56, E3_register60) = let
                src = if is_lo_thread
                    E2_register60
                else
                    E2_register56
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register56, dst)
                else
                    (dst, E2_register60)
                end
            end
            (E3_register57, E3_register61) = let
                src = if is_lo_thread
                    E2_register61
                else
                    E2_register57
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register57, dst)
                else
                    (dst, E2_register61)
                end
            end
            (E3_register58, E3_register62) = let
                src = if is_lo_thread
                    E2_register62
                else
                    E2_register58
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register58, dst)
                else
                    (dst, E2_register62)
                end
            end
            (E3_register59, E3_register63) = let
                src = if is_lo_thread
                    E2_register63
                else
                    E2_register59
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000002)
                if is_lo_thread
                    (E2_register59, dst)
                else
                    (dst, E2_register63)
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
            (E4_register32, E4_register40) = let
                src = if is_lo_thread
                    E3_register40
                else
                    E3_register32
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register32, dst)
                else
                    (dst, E3_register40)
                end
            end
            (E4_register33, E4_register41) = let
                src = if is_lo_thread
                    E3_register41
                else
                    E3_register33
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register33, dst)
                else
                    (dst, E3_register41)
                end
            end
            (E4_register34, E4_register42) = let
                src = if is_lo_thread
                    E3_register42
                else
                    E3_register34
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register34, dst)
                else
                    (dst, E3_register42)
                end
            end
            (E4_register35, E4_register43) = let
                src = if is_lo_thread
                    E3_register43
                else
                    E3_register35
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register35, dst)
                else
                    (dst, E3_register43)
                end
            end
            (E4_register36, E4_register44) = let
                src = if is_lo_thread
                    E3_register44
                else
                    E3_register36
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register36, dst)
                else
                    (dst, E3_register44)
                end
            end
            (E4_register37, E4_register45) = let
                src = if is_lo_thread
                    E3_register45
                else
                    E3_register37
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register37, dst)
                else
                    (dst, E3_register45)
                end
            end
            (E4_register38, E4_register46) = let
                src = if is_lo_thread
                    E3_register46
                else
                    E3_register38
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register38, dst)
                else
                    (dst, E3_register46)
                end
            end
            (E4_register39, E4_register47) = let
                src = if is_lo_thread
                    E3_register47
                else
                    E3_register39
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register39, dst)
                else
                    (dst, E3_register47)
                end
            end
            (E4_register48, E4_register56) = let
                src = if is_lo_thread
                    E3_register56
                else
                    E3_register48
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register48, dst)
                else
                    (dst, E3_register56)
                end
            end
            (E4_register49, E4_register57) = let
                src = if is_lo_thread
                    E3_register57
                else
                    E3_register49
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register49, dst)
                else
                    (dst, E3_register57)
                end
            end
            (E4_register50, E4_register58) = let
                src = if is_lo_thread
                    E3_register58
                else
                    E3_register50
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register50, dst)
                else
                    (dst, E3_register58)
                end
            end
            (E4_register51, E4_register59) = let
                src = if is_lo_thread
                    E3_register59
                else
                    E3_register51
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register51, dst)
                else
                    (dst, E3_register59)
                end
            end
            (E4_register52, E4_register60) = let
                src = if is_lo_thread
                    E3_register60
                else
                    E3_register52
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register52, dst)
                else
                    (dst, E3_register60)
                end
            end
            (E4_register53, E4_register61) = let
                src = if is_lo_thread
                    E3_register61
                else
                    E3_register53
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register53, dst)
                else
                    (dst, E3_register61)
                end
            end
            (E4_register54, E4_register62) = let
                src = if is_lo_thread
                    E3_register62
                else
                    E3_register54
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register54, dst)
                else
                    (dst, E3_register62)
                end
            end
            (E4_register55, E4_register63) = let
                src = if is_lo_thread
                    E3_register63
                else
                    E3_register55
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000004)
                if is_lo_thread
                    (E3_register55, dst)
                else
                    (dst, E3_register63)
                end
            end
            is_lo_thread = IndexSpaces.cuda_threadidx() & 0x00000008 == 0x00
            (E5_register0, E5_register16) = let
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
            (E5_register1, E5_register17) = let
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
            (E5_register2, E5_register18) = let
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
            (E5_register3, E5_register19) = let
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
            (E5_register4, E5_register20) = let
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
            (E5_register5, E5_register21) = let
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
            (E5_register6, E5_register22) = let
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
            (E5_register7, E5_register23) = let
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
            (E5_register8, E5_register24) = let
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
            (E5_register9, E5_register25) = let
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
            (E5_register10, E5_register26) = let
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
            (E5_register11, E5_register27) = let
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
            (E5_register12, E5_register28) = let
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
            (E5_register13, E5_register29) = let
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
            (E5_register14, E5_register30) = let
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
            (E5_register15, E5_register31) = let
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
            (E5_register32, E5_register48) = let
                src = if is_lo_thread
                    E4_register48
                else
                    E4_register32
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register32, dst)
                else
                    (dst, E4_register48)
                end
            end
            (E5_register33, E5_register49) = let
                src = if is_lo_thread
                    E4_register49
                else
                    E4_register33
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register33, dst)
                else
                    (dst, E4_register49)
                end
            end
            (E5_register34, E5_register50) = let
                src = if is_lo_thread
                    E4_register50
                else
                    E4_register34
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register34, dst)
                else
                    (dst, E4_register50)
                end
            end
            (E5_register35, E5_register51) = let
                src = if is_lo_thread
                    E4_register51
                else
                    E4_register35
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register35, dst)
                else
                    (dst, E4_register51)
                end
            end
            (E5_register36, E5_register52) = let
                src = if is_lo_thread
                    E4_register52
                else
                    E4_register36
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register36, dst)
                else
                    (dst, E4_register52)
                end
            end
            (E5_register37, E5_register53) = let
                src = if is_lo_thread
                    E4_register53
                else
                    E4_register37
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register37, dst)
                else
                    (dst, E4_register53)
                end
            end
            (E5_register38, E5_register54) = let
                src = if is_lo_thread
                    E4_register54
                else
                    E4_register38
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register38, dst)
                else
                    (dst, E4_register54)
                end
            end
            (E5_register39, E5_register55) = let
                src = if is_lo_thread
                    E4_register55
                else
                    E4_register39
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register39, dst)
                else
                    (dst, E4_register55)
                end
            end
            (E5_register40, E5_register56) = let
                src = if is_lo_thread
                    E4_register56
                else
                    E4_register40
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register40, dst)
                else
                    (dst, E4_register56)
                end
            end
            (E5_register41, E5_register57) = let
                src = if is_lo_thread
                    E4_register57
                else
                    E4_register41
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register41, dst)
                else
                    (dst, E4_register57)
                end
            end
            (E5_register42, E5_register58) = let
                src = if is_lo_thread
                    E4_register58
                else
                    E4_register42
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register42, dst)
                else
                    (dst, E4_register58)
                end
            end
            (E5_register43, E5_register59) = let
                src = if is_lo_thread
                    E4_register59
                else
                    E4_register43
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register43, dst)
                else
                    (dst, E4_register59)
                end
            end
            (E5_register44, E5_register60) = let
                src = if is_lo_thread
                    E4_register60
                else
                    E4_register44
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register44, dst)
                else
                    (dst, E4_register60)
                end
            end
            (E5_register45, E5_register61) = let
                src = if is_lo_thread
                    E4_register61
                else
                    E4_register45
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register45, dst)
                else
                    (dst, E4_register61)
                end
            end
            (E5_register46, E5_register62) = let
                src = if is_lo_thread
                    E4_register62
                else
                    E4_register46
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register46, dst)
                else
                    (dst, E4_register62)
                end
            end
            (E5_register47, E5_register63) = let
                src = if is_lo_thread
                    E4_register63
                else
                    E4_register47
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000008)
                if is_lo_thread
                    (E4_register47, dst)
                else
                    (dst, E4_register63)
                end
            end
            is_lo_thread = IndexSpaces.cuda_threadidx() & 0x00000010 == 0x00
            (Eout_register0, Eout_register32) = let
                src = if is_lo_thread
                    E5_register32
                else
                    E5_register0
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register0, dst)
                else
                    (dst, E5_register32)
                end
            end
            (Eout_register1, Eout_register33) = let
                src = if is_lo_thread
                    E5_register33
                else
                    E5_register1
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register1, dst)
                else
                    (dst, E5_register33)
                end
            end
            (Eout_register2, Eout_register34) = let
                src = if is_lo_thread
                    E5_register34
                else
                    E5_register2
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register2, dst)
                else
                    (dst, E5_register34)
                end
            end
            (Eout_register3, Eout_register35) = let
                src = if is_lo_thread
                    E5_register35
                else
                    E5_register3
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register3, dst)
                else
                    (dst, E5_register35)
                end
            end
            (Eout_register4, Eout_register36) = let
                src = if is_lo_thread
                    E5_register36
                else
                    E5_register4
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register4, dst)
                else
                    (dst, E5_register36)
                end
            end
            (Eout_register5, Eout_register37) = let
                src = if is_lo_thread
                    E5_register37
                else
                    E5_register5
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register5, dst)
                else
                    (dst, E5_register37)
                end
            end
            (Eout_register6, Eout_register38) = let
                src = if is_lo_thread
                    E5_register38
                else
                    E5_register6
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register6, dst)
                else
                    (dst, E5_register38)
                end
            end
            (Eout_register7, Eout_register39) = let
                src = if is_lo_thread
                    E5_register39
                else
                    E5_register7
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register7, dst)
                else
                    (dst, E5_register39)
                end
            end
            (Eout_register8, Eout_register40) = let
                src = if is_lo_thread
                    E5_register40
                else
                    E5_register8
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register8, dst)
                else
                    (dst, E5_register40)
                end
            end
            (Eout_register9, Eout_register41) = let
                src = if is_lo_thread
                    E5_register41
                else
                    E5_register9
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register9, dst)
                else
                    (dst, E5_register41)
                end
            end
            (Eout_register10, Eout_register42) = let
                src = if is_lo_thread
                    E5_register42
                else
                    E5_register10
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register10, dst)
                else
                    (dst, E5_register42)
                end
            end
            (Eout_register11, Eout_register43) = let
                src = if is_lo_thread
                    E5_register43
                else
                    E5_register11
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register11, dst)
                else
                    (dst, E5_register43)
                end
            end
            (Eout_register12, Eout_register44) = let
                src = if is_lo_thread
                    E5_register44
                else
                    E5_register12
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register12, dst)
                else
                    (dst, E5_register44)
                end
            end
            (Eout_register13, Eout_register45) = let
                src = if is_lo_thread
                    E5_register45
                else
                    E5_register13
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register13, dst)
                else
                    (dst, E5_register45)
                end
            end
            (Eout_register14, Eout_register46) = let
                src = if is_lo_thread
                    E5_register46
                else
                    E5_register14
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register14, dst)
                else
                    (dst, E5_register46)
                end
            end
            (Eout_register15, Eout_register47) = let
                src = if is_lo_thread
                    E5_register47
                else
                    E5_register15
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register15, dst)
                else
                    (dst, E5_register47)
                end
            end
            (Eout_register16, Eout_register48) = let
                src = if is_lo_thread
                    E5_register48
                else
                    E5_register16
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register16, dst)
                else
                    (dst, E5_register48)
                end
            end
            (Eout_register17, Eout_register49) = let
                src = if is_lo_thread
                    E5_register49
                else
                    E5_register17
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register17, dst)
                else
                    (dst, E5_register49)
                end
            end
            (Eout_register18, Eout_register50) = let
                src = if is_lo_thread
                    E5_register50
                else
                    E5_register18
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register18, dst)
                else
                    (dst, E5_register50)
                end
            end
            (Eout_register19, Eout_register51) = let
                src = if is_lo_thread
                    E5_register51
                else
                    E5_register19
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register19, dst)
                else
                    (dst, E5_register51)
                end
            end
            (Eout_register20, Eout_register52) = let
                src = if is_lo_thread
                    E5_register52
                else
                    E5_register20
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register20, dst)
                else
                    (dst, E5_register52)
                end
            end
            (Eout_register21, Eout_register53) = let
                src = if is_lo_thread
                    E5_register53
                else
                    E5_register21
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register21, dst)
                else
                    (dst, E5_register53)
                end
            end
            (Eout_register22, Eout_register54) = let
                src = if is_lo_thread
                    E5_register54
                else
                    E5_register22
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register22, dst)
                else
                    (dst, E5_register54)
                end
            end
            (Eout_register23, Eout_register55) = let
                src = if is_lo_thread
                    E5_register55
                else
                    E5_register23
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register23, dst)
                else
                    (dst, E5_register55)
                end
            end
            (Eout_register24, Eout_register56) = let
                src = if is_lo_thread
                    E5_register56
                else
                    E5_register24
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register24, dst)
                else
                    (dst, E5_register56)
                end
            end
            (Eout_register25, Eout_register57) = let
                src = if is_lo_thread
                    E5_register57
                else
                    E5_register25
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register25, dst)
                else
                    (dst, E5_register57)
                end
            end
            (Eout_register26, Eout_register58) = let
                src = if is_lo_thread
                    E5_register58
                else
                    E5_register26
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register26, dst)
                else
                    (dst, E5_register58)
                end
            end
            (Eout_register27, Eout_register59) = let
                src = if is_lo_thread
                    E5_register59
                else
                    E5_register27
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register27, dst)
                else
                    (dst, E5_register59)
                end
            end
            (Eout_register28, Eout_register60) = let
                src = if is_lo_thread
                    E5_register60
                else
                    E5_register28
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register28, dst)
                else
                    (dst, E5_register60)
                end
            end
            (Eout_register29, Eout_register61) = let
                src = if is_lo_thread
                    E5_register61
                else
                    E5_register29
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register29, dst)
                else
                    (dst, E5_register61)
                end
            end
            (Eout_register30, Eout_register62) = let
                src = if is_lo_thread
                    E5_register62
                else
                    E5_register30
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register30, dst)
                else
                    (dst, E5_register62)
                end
            end
            (Eout_register31, Eout_register63) = let
                src = if is_lo_thread
                    E5_register63
                else
                    E5_register31
                end
                dst = IndexSpaces.cuda_shfl_xor_sync(0xffffffff, src, 0x00000010)
                if is_lo_thread
                    (E5_register31, dst)
                else
                    (dst, E5_register63)
                end
            end
            IndexSpaces.unsafe_store4_global!(
                Eout_memory,
                (
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) %
                            2
                        ) % 2
                    ) * 128 +
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) %
                            16
                        ) % 16
                    ) * 256 +
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_threadidx(),
                                    0,
                                    32,
                                ) % 32
                            ) * 16
                        ) ÷ 4
                    ) % 128 +
                    (
                        (
                            (
                                (
                                    IndexSpaces.assume_inrange(
                                        IndexSpaces.cuda_warpidx(),
                                        0,
                                        16,
                                    ) ÷ 2
                                ) % 8
                            ) * 4096 +
                            (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                        ) % 32768
                    ) * 4096
                ) +
                0 +
                0x01,
                (Eout_register0, Eout_register1, Eout_register2, Eout_register3),
            )
            IndexSpaces.unsafe_store4_global!(
                Eout_memory,
                (
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) %
                            2
                        ) % 2
                    ) * 128 +
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) %
                            16
                        ) % 16
                    ) * 256 +
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_threadidx(),
                                    0,
                                    32,
                                ) % 32
                            ) * 16
                        ) ÷ 4
                    ) % 128 +
                    (
                        (
                            (
                                (
                                    (
                                        IndexSpaces.assume_inrange(
                                            IndexSpaces.cuda_warpidx(),
                                            0,
                                            16,
                                        ) ÷ 2
                                    ) % 8
                                ) * 4096 + 2
                            ) + (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                        ) % 32768
                    ) * 4096
                ) +
                0 +
                0x01,
                (Eout_register4, Eout_register5, Eout_register6, Eout_register7),
            )
            IndexSpaces.unsafe_store4_global!(
                Eout_memory,
                (
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) %
                            2
                        ) % 2
                    ) * 128 +
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) %
                            16
                        ) % 16
                    ) * 256 +
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_threadidx(),
                                    0,
                                    32,
                                ) % 32
                            ) * 16
                        ) ÷ 4
                    ) % 128 +
                    (
                        (
                            (
                                (
                                    (
                                        IndexSpaces.assume_inrange(
                                            IndexSpaces.cuda_warpidx(),
                                            0,
                                            16,
                                        ) ÷ 2
                                    ) % 8
                                ) * 4096 + 4
                            ) + (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                        ) % 32768
                    ) * 4096
                ) +
                0 +
                0x01,
                (Eout_register8, Eout_register9, Eout_register10, Eout_register11),
            )
            IndexSpaces.unsafe_store4_global!(
                Eout_memory,
                (
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) %
                            2
                        ) % 2
                    ) * 128 +
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) %
                            16
                        ) % 16
                    ) * 256 +
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_threadidx(),
                                    0,
                                    32,
                                ) % 32
                            ) * 16
                        ) ÷ 4
                    ) % 128 +
                    (
                        (
                            (
                                (
                                    (
                                        IndexSpaces.assume_inrange(
                                            IndexSpaces.cuda_warpidx(),
                                            0,
                                            16,
                                        ) ÷ 2
                                    ) % 8
                                ) * 4096 + 6
                            ) + (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                        ) % 32768
                    ) * 4096
                ) +
                0 +
                0x01,
                (Eout_register12, Eout_register13, Eout_register14, Eout_register15),
            )
            IndexSpaces.unsafe_store4_global!(
                Eout_memory,
                (
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) %
                            2
                        ) % 2
                    ) * 128 +
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) %
                            16
                        ) % 16
                    ) * 256 +
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_threadidx(),
                                    0,
                                    32,
                                ) % 32
                            ) * 16
                        ) ÷ 4
                    ) % 128 +
                    (
                        (
                            (
                                (
                                    (
                                        IndexSpaces.assume_inrange(
                                            IndexSpaces.cuda_warpidx(),
                                            0,
                                            16,
                                        ) ÷ 2
                                    ) % 8
                                ) * 4096 + 8
                            ) + (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                        ) % 32768
                    ) * 4096
                ) +
                0 +
                0x01,
                (Eout_register16, Eout_register17, Eout_register18, Eout_register19),
            )
            IndexSpaces.unsafe_store4_global!(
                Eout_memory,
                (
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) %
                            2
                        ) % 2
                    ) * 128 +
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) %
                            16
                        ) % 16
                    ) * 256 +
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_threadidx(),
                                    0,
                                    32,
                                ) % 32
                            ) * 16
                        ) ÷ 4
                    ) % 128 +
                    (
                        (
                            (
                                (
                                    (
                                        IndexSpaces.assume_inrange(
                                            IndexSpaces.cuda_warpidx(),
                                            0,
                                            16,
                                        ) ÷ 2
                                    ) % 8
                                ) * 4096 + 10
                            ) + (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                        ) % 32768
                    ) * 4096
                ) +
                0 +
                0x01,
                (Eout_register20, Eout_register21, Eout_register22, Eout_register23),
            )
            IndexSpaces.unsafe_store4_global!(
                Eout_memory,
                (
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) %
                            2
                        ) % 2
                    ) * 128 +
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) %
                            16
                        ) % 16
                    ) * 256 +
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_threadidx(),
                                    0,
                                    32,
                                ) % 32
                            ) * 16
                        ) ÷ 4
                    ) % 128 +
                    (
                        (
                            (
                                (
                                    (
                                        IndexSpaces.assume_inrange(
                                            IndexSpaces.cuda_warpidx(),
                                            0,
                                            16,
                                        ) ÷ 2
                                    ) % 8
                                ) * 4096 + 12
                            ) + (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                        ) % 32768
                    ) * 4096
                ) +
                0 +
                0x01,
                (Eout_register24, Eout_register25, Eout_register26, Eout_register27),
            )
            IndexSpaces.unsafe_store4_global!(
                Eout_memory,
                (
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) %
                            2
                        ) % 2
                    ) * 128 +
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) %
                            16
                        ) % 16
                    ) * 256 +
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_threadidx(),
                                    0,
                                    32,
                                ) % 32
                            ) * 16
                        ) ÷ 4
                    ) % 128 +
                    (
                        (
                            (
                                (
                                    (
                                        IndexSpaces.assume_inrange(
                                            IndexSpaces.cuda_warpidx(),
                                            0,
                                            16,
                                        ) ÷ 2
                                    ) % 8
                                ) * 4096 + 14
                            ) + (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                        ) % 32768
                    ) * 4096
                ) +
                0 +
                0x01,
                (Eout_register28, Eout_register29, Eout_register30, Eout_register31),
            )
            IndexSpaces.unsafe_store4_global!(
                Eout_memory,
                (
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) %
                            2
                        ) % 2
                    ) * 128 +
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) %
                            16
                        ) % 16
                    ) * 256 +
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_threadidx(),
                                    0,
                                    32,
                                ) % 32
                            ) * 16
                        ) ÷ 4
                    ) % 128 +
                    (
                        (
                            (
                                (
                                    (
                                        IndexSpaces.assume_inrange(
                                            IndexSpaces.cuda_warpidx(),
                                            0,
                                            16,
                                        ) ÷ 2
                                    ) % 8
                                ) * 4096 +
                                (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) + 1
                        ) % 32768
                    ) * 4096
                ) +
                0 +
                0x01,
                (Eout_register32, Eout_register33, Eout_register34, Eout_register35),
            )
            IndexSpaces.unsafe_store4_global!(
                Eout_memory,
                (
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) %
                            2
                        ) % 2
                    ) * 128 +
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) %
                            16
                        ) % 16
                    ) * 256 +
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_threadidx(),
                                    0,
                                    32,
                                ) % 32
                            ) * 16
                        ) ÷ 4
                    ) % 128 +
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_warpidx(),
                                                0,
                                                16,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 4096 + 2
                                ) +
                                (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) + 1
                        ) % 32768
                    ) * 4096
                ) +
                0 +
                0x01,
                (Eout_register36, Eout_register37, Eout_register38, Eout_register39),
            )
            IndexSpaces.unsafe_store4_global!(
                Eout_memory,
                (
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) %
                            2
                        ) % 2
                    ) * 128 +
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) %
                            16
                        ) % 16
                    ) * 256 +
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_threadidx(),
                                    0,
                                    32,
                                ) % 32
                            ) * 16
                        ) ÷ 4
                    ) % 128 +
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_warpidx(),
                                                0,
                                                16,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 4096 + 4
                                ) +
                                (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) + 1
                        ) % 32768
                    ) * 4096
                ) +
                0 +
                0x01,
                (Eout_register40, Eout_register41, Eout_register42, Eout_register43),
            )
            IndexSpaces.unsafe_store4_global!(
                Eout_memory,
                (
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) %
                            2
                        ) % 2
                    ) * 128 +
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) %
                            16
                        ) % 16
                    ) * 256 +
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_threadidx(),
                                    0,
                                    32,
                                ) % 32
                            ) * 16
                        ) ÷ 4
                    ) % 128 +
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_warpidx(),
                                                0,
                                                16,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 4096 + 6
                                ) +
                                (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) + 1
                        ) % 32768
                    ) * 4096
                ) +
                0 +
                0x01,
                (Eout_register44, Eout_register45, Eout_register46, Eout_register47),
            )
            IndexSpaces.unsafe_store4_global!(
                Eout_memory,
                (
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) %
                            2
                        ) % 2
                    ) * 128 +
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) %
                            16
                        ) % 16
                    ) * 256 +
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_threadidx(),
                                    0,
                                    32,
                                ) % 32
                            ) * 16
                        ) ÷ 4
                    ) % 128 +
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_warpidx(),
                                                0,
                                                16,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 4096 + 8
                                ) +
                                (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) + 1
                        ) % 32768
                    ) * 4096
                ) +
                0 +
                0x01,
                (Eout_register48, Eout_register49, Eout_register50, Eout_register51),
            )
            IndexSpaces.unsafe_store4_global!(
                Eout_memory,
                (
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) %
                            2
                        ) % 2
                    ) * 128 +
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) %
                            16
                        ) % 16
                    ) * 256 +
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_threadidx(),
                                    0,
                                    32,
                                ) % 32
                            ) * 16
                        ) ÷ 4
                    ) % 128 +
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_warpidx(),
                                                0,
                                                16,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 4096 + 10
                                ) +
                                (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) + 1
                        ) % 32768
                    ) * 4096
                ) +
                0 +
                0x01,
                (Eout_register52, Eout_register53, Eout_register54, Eout_register55),
            )
            IndexSpaces.unsafe_store4_global!(
                Eout_memory,
                (
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) %
                            2
                        ) % 2
                    ) * 128 +
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) %
                            16
                        ) % 16
                    ) * 256 +
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_threadidx(),
                                    0,
                                    32,
                                ) % 32
                            ) * 16
                        ) ÷ 4
                    ) % 128 +
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_warpidx(),
                                                0,
                                                16,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 4096 + 12
                                ) +
                                (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) + 1
                        ) % 32768
                    ) * 4096
                ) +
                0 +
                0x01,
                (Eout_register56, Eout_register57, Eout_register58, Eout_register59),
            )
            IndexSpaces.unsafe_store4_global!(
                Eout_memory,
                (
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) %
                            2
                        ) % 2
                    ) * 128 +
                    (
                        (
                            IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(), 0, 16) %
                            16
                        ) % 16
                    ) * 256 +
                    (
                        (
                            (
                                IndexSpaces.assume_inrange(
                                    IndexSpaces.cuda_threadidx(),
                                    0,
                                    32,
                                ) % 32
                            ) * 16
                        ) ÷ 4
                    ) % 128 +
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            IndexSpaces.assume_inrange(
                                                IndexSpaces.cuda_warpidx(),
                                                0,
                                                16,
                                            ) ÷ 2
                                        ) % 8
                                    ) * 4096 + 14
                                ) +
                                (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16
                            ) + 1
                        ) % 32768
                    ) * 4096
                ) +
                0 +
                0x01,
                (Eout_register60, Eout_register61, Eout_register62, Eout_register63),
            )
        end
        info = 0
        info_memory[((IndexSpaces.assume_inrange(
            IndexSpaces.cuda_threadidx(),
            0,
            32,
        )%32)%32+((IndexSpaces.assume_inrange(
            IndexSpaces.cuda_warpidx(),
            0,
            16,
        )%16)%16)*32+((IndexSpaces.assume_inrange(
            IndexSpaces.cuda_blockidx(),
            0,
            16,
        )%16)%16)*512)+0+0x01] = info
    end
)
