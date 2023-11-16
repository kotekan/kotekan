@fastmath @inbounds(begin #= /home/eschnett/src/kotekan/julia/kernels/xpose.jl:263 =#
                        info = 1
                        info_memory[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(),
                        0,
                        32) % 32) % 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(),
                        0,
                        128) % 128) % 128) * 512 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(),
                        0,
                        16) % 16) % 16) * 32) + 0 + 0x01] = info
                        for loop in 0:1:255
                            (Ein_register0, Ein_register1, Ein_register2, Ein_register3) = IndexSpaces.unsafe_load4_global(Ein_memory,
                                                                                                                           ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(),
                                                                                                                                                            0,
                                                                                                                                                            32) ÷
                                                                                                                                 16) %
                                                                                                                                2) *
                                                                                                                               8 +
                                                                                                                               (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(),
                                                                                                                                                           0,
                                                                                                                                                           32) %
                                                                                                                                2) *
                                                                                                                               16) ÷
                                                                                                                              8) %
                                                                                                                             8) *
                                                                                                                            32 +
                                                                                                                            (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(),
                                                                                                                                                            0,
                                                                                                                                                            16) ÷
                                                                                                                                 2) %
                                                                                                                                8) *
                                                                                                                               4096 +
                                                                                                                               ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(),
                                                                                                                                                            0,
                                                                                                                                                            32) ÷
                                                                                                                                 2) %
                                                                                                                                8) *
                                                                                                                               2) +
                                                                                                                              (IndexSpaces.assume_inrange(loop,
                                                                                                                                                          0,
                                                                                                                                                          1,
                                                                                                                                                          256) %
                                                                                                                               256) *
                                                                                                                              16) %
                                                                                                                             16) *
                                                                                                                            2 +
                                                                                                                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(),
                                                                                                                                                         0,
                                                                                                                                                         16) %
                                                                                                                              2) %
                                                                                                                             2) *
                                                                                                                            256 +
                                                                                                                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(),
                                                                                                                                                         0,
                                                                                                                                                         128) %
                                                                                                                              128) %
                                                                                                                             128) *
                                                                                                                            512 +
                                                                                                                            ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(),
                                                                                                                                                             0,
                                                                                                                                                             16) ÷
                                                                                                                                  2) %
                                                                                                                                 8) *
                                                                                                                                4096 +
                                                                                                                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(),
                                                                                                                                                             0,
                                                                                                                                                             32) ÷
                                                                                                                                  2) %
                                                                                                                                 8) *
                                                                                                                                2) +
                                                                                                                               (IndexSpaces.assume_inrange(loop,
                                                                                                                                                           0,
                                                                                                                                                           1,
                                                                                                                                                           256) %
                                                                                                                                256) *
                                                                                                                               16) ÷
                                                                                                                              16) %
                                                                                                                             2048) *
                                                                                                                            65536 +
                                                                                                                            ((((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(),
                                                                                                                                                           0,
                                                                                                                                                           32) ÷
                                                                                                                                16) %
                                                                                                                               2) *
                                                                                                                              8 +
                                                                                                                              (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(),
                                                                                                                                                          0,
                                                                                                                                                          32) %
                                                                                                                               2) *
                                                                                                                              16) ÷
                                                                                                                             4) % 2) +
                                                                                                                           1i32)
                            (Ein_register4, Ein_register5, Ein_register6, Ein_register7) = IndexSpaces.unsafe_load4_global(Ein_memory,
                                                                                                                           (((((32 +
                                                                                                                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(),
                                                                                                                                                             0,
                                                                                                                                                             32) ÷
                                                                                                                                  16) %
                                                                                                                                 2) *
                                                                                                                                8) +
                                                                                                                               (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(),
                                                                                                                                                           0,
                                                                                                                                                           32) %
                                                                                                                                2) *
                                                                                                                               16) ÷
                                                                                                                              8) %
                                                                                                                             8) *
                                                                                                                            32 +
                                                                                                                            (((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(),
                                                                                                                                                            0,
                                                                                                                                                            16) ÷
                                                                                                                                 2) %
                                                                                                                                8) *
                                                                                                                               4096 +
                                                                                                                               ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(),
                                                                                                                                                            0,
                                                                                                                                                            32) ÷
                                                                                                                                 2) %
                                                                                                                                8) *
                                                                                                                               2) +
                                                                                                                              (IndexSpaces.assume_inrange(loop,
                                                                                                                                                          0,
                                                                                                                                                          1,
                                                                                                                                                          256) %
                                                                                                                               256) *
                                                                                                                              16) %
                                                                                                                             16) *
                                                                                                                            2 +
                                                                                                                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(),
                                                                                                                                                         0,
                                                                                                                                                         16) %
                                                                                                                              2) %
                                                                                                                             2) *
                                                                                                                            256 +
                                                                                                                            ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(),
                                                                                                                                                         0,
                                                                                                                                                         128) %
                                                                                                                              128) %
                                                                                                                             128) *
                                                                                                                            512 +
                                                                                                                            ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(),
                                                                                                                                                             0,
                                                                                                                                                             16) ÷
                                                                                                                                  2) %
                                                                                                                                 8) *
                                                                                                                                4096 +
                                                                                                                                ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(),
                                                                                                                                                             0,
                                                                                                                                                             32) ÷
                                                                                                                                  2) %
                                                                                                                                 8) *
                                                                                                                                2) +
                                                                                                                               (IndexSpaces.assume_inrange(loop,
                                                                                                                                                           0,
                                                                                                                                                           1,
                                                                                                                                                           256) %
                                                                                                                                256) *
                                                                                                                               16) ÷
                                                                                                                              16) %
                                                                                                                             2048) *
                                                                                                                            65536 +
                                                                                                                            (((32 +
                                                                                                                               ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(),
                                                                                                                                                            0,
                                                                                                                                                            32) ÷
                                                                                                                                 16) %
                                                                                                                                2) *
                                                                                                                               8) +
                                                                                                                              (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(),
                                                                                                                                                          0,
                                                                                                                                                          32) %
                                                                                                                               2) *
                                                                                                                              16) ÷
                                                                                                                             4) % 2) +
                                                                                                                           1i32)
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
                            is_lo_thread = IndexSpaces.cuda_threadidx() & 0x00000002 == 0x00
                            (Eout_register0, Eout_register4) = let
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
                            (Eout_register1, Eout_register5) = let
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
                            (Eout_register2, Eout_register6) = let
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
                            (Eout_register3, Eout_register7) = let
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
                            IndexSpaces.unsafe_store4_global!(Eout_memory,
                                                              (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) %
                                                                 2) % 2) * 16 +
                                                               (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(),
                                                                                             0,
                                                                                             32) % 4) * 16) ÷ 4) % 16 +
                                                               ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(),
                                                                                            0,
                                                                                            128) % 128) % 128) * 32 +
                                                               ((((((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(),
                                                                                                0,
                                                                                                16) ÷ 2) % 8) * 4096 +
                                                                   ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(),
                                                                                                0,
                                                                                                32) ÷ 4) % 4) * 4) +
                                                                  (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16) +
                                                                 (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(),
                                                                                             0,
                                                                                             32) ÷ 16) % 2) % 32768) * 4096) +
                                                              0 +
                                                              0x01,
                                                              (Eout_register0, Eout_register1, Eout_register2, Eout_register3))
                            IndexSpaces.unsafe_store4_global!(Eout_memory,
                                                              (((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(), 0, 16) %
                                                                 2) % 2) * 16 +
                                                               (((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(),
                                                                                             0,
                                                                                             32) % 4) * 16) ÷ 4) % 16 +
                                                               ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(),
                                                                                            0,
                                                                                            128) % 128) % 128) * 32 +
                                                               (((((2 +
                                                                    ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(),
                                                                                                 0,
                                                                                                 16) ÷ 2) % 8) * 4096) +
                                                                   ((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(),
                                                                                                0,
                                                                                                32) ÷ 4) % 4) * 4) +
                                                                  (IndexSpaces.assume_inrange(loop, 0, 1, 256) % 256) * 16) +
                                                                 (IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(),
                                                                                             0,
                                                                                             32) ÷ 16) % 2) % 32768) * 4096) +
                                                              0 +
                                                              0x01,
                                                              (Eout_register4, Eout_register5, Eout_register6, Eout_register7))
                        end
                        info = 0
                        info_memory[((IndexSpaces.assume_inrange(IndexSpaces.cuda_threadidx(),
                        0,
                        32) % 32) % 32 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_blockidx(),
                        0,
                        128) % 128) % 128) * 512 + ((IndexSpaces.assume_inrange(IndexSpaces.cuda_warpidx(),
                        0,
                        16) % 16) % 16) * 32) + 0 + 0x01] = info
                    end)
