// Cloned from n2k/src_lib/precompute_offsets.cu (see include/n2k_dual/DualCorrelator.hpp
// for provenance). Divergences:
//
//  - The blockId -> (atile, btile) decode is hoisted out of PointerOffsets into
//    dual_enumerate_blocks(), which also applies the block_class_mask: disabled block
//    classes are simply absent from the ptable, and blockIdx.x indexes the KEPT blocks.
//    The full enumeration order matches n2k's _init_block_data() exactly.
//
//  - The ptable has 8 rows per thread (n2k: 6). Row 6 = the warp's input time stride in
//    int32 units (the A and B inputs are separately shaped); row 7 = input select (0=A,
//    1=B). The 'gp' offset is computed against the warp's OWN input: its 128-station tile
//    index is local to that input, and its time stride is that input's emat_tstride.

#include <ksgpu.hpp>

#include <cassert>
#include "../include/n2k_dual/DualCorrelator.hpp"

using namespace std;
using namespace ksgpu;

namespace n2k_dual {
#if 0
}  // editor auto-indent
#endif


int dual_block_class(int atile, int btile, int ntiles_a)
{
    int n_in_a = (atile < ntiles_a ? 1 : 0) + (btile < ntiles_a ? 1 : 0);
    if (n_in_a == 2)
        return BLOCK_CLASS_AA;
    if (n_in_a == 1)
        return BLOCK_CLASS_MIXED;
    return BLOCK_CLASS_BB;
}


vector<pair<int, int>> dual_enumerate_blocks(int ntiles_1d, int ntiles_a, int block_class_mask)
{
    // Full-order reference (must match n2k's _init_block_data): the strict lower triangle
    // walked as (1,0),(2,0),(2,1),(3,0),(3,1),(3,2),... followed by the diagonal
    // (0,0),(1,1),(2,2),... A restricted mask keeps this order and drops disabled blocks.
    vector<pair<int, int>> out;

    for (int a = 1; a < ntiles_1d; a++)
        for (int b = 0; b < a; b++)
            if (block_class_mask & (1 << dual_block_class(a, b, ntiles_a)))
                out.push_back({a, b});

    for (int d = 0; d < ntiles_1d; d++)
        if (block_class_mask & (1 << dual_block_class(d, d, ntiles_a)))
            out.push_back({d, d});

    return out;
}


struct PointerOffsets {
    const DualCorrelatorParams& params;
    const int block_atile; // from dual_enumerate_blocks(), not decoded from blockId
    const int block_btile;
    const int warpId;
    const int laneId;

    // These go into the ptable
    int ap = -1;
    int bp = -1;
    int gp = -1;
    int sp = -1;

    int vi_warp = -1;
    int vk_warp = -1;

    int ts = -1;   // n2k_dual: the warp's input time stride (int32 units)
    int gsel = -1; // n2k_dual: input select (0=A, 1=B)


    __host__ PointerOffsets(const DualCorrelatorParams& params_, int block_atile_,
                            int block_btile_, int warpId_, int laneId_) :
        params(params_),
        block_atile(block_atile_),
        block_btile(block_btile_),
        warpId(warpId_),
        laneId(laneId_)
    {
        assert((laneId >= 0) && (laneId < 32));
        assert((warpId >= 0) && (warpId < DualCorrelatorParams::warps_per_block));
        assert((block_atile >= 0) && (block_atile < params.ntiles_1d));
        assert((block_btile >= 0) && (block_btile <= block_atile));

        _init_prefetch_offsets();
        _init_warp_tiling();
    }


    __host__ void _init_prefetch_offsets()
    {
        // Ultra-weird: the first one is consistently fast, whereas the second one
        // sometimes runs slow!  [comment inherited from n2k]
#if 1
        bool bflag = warpId & 0x1;
        int tpf = (warpId >> 1) * 8;
#else
        bool bflag = warpId >> 2;
        int tpf = (warpId & 0x3) * 8;
#endif

        // Assign 'gp' pointer.
        // We write warp_tpf = 32*t32 + 8*t8
        int t32 = (tpf >> 5);
        int t8 = (tpf >> 3) & 0x3;

        assert(tpf == 32 * t32 + 8 * t8);
        assert(t32 < 2);
        assert(t8 < 4);

        // n2k_dual: this warp reads the 128-station panel of either the A tile or the
        // B tile of the block. Which INPUT that panel lives in, its input-local tile
        // index, and that input's time stride all follow from the global tile index.
        int tile = bflag ? block_btile : block_atile;
        bool is_b = (tile >= params.ntiles_a);

        gsel = is_b ? 1 : 0;
        ts = is_b ? params.emat_tstride_b : params.emat_tstride_a;

        int tile_local = is_b ? (tile - params.ntiles_a) : tile;

        gp = 32 * tile_local; // each 128-station tile corresponds to 32 int32s
        gp += tpf * ts;
        gp += laneId;

        // Assign 'sp' pointer.
        // Each laneId corresponds to 4 stations, so we write (4*laneId) = 8*s8 + s1

        int s8 = laneId >> 1;
        int s1 = (laneId & 0x1) << 2;

        assert(4 * laneId == 8 * s8 + s1);
        assert(s8 < 16);
        assert(s1 < 8);

        sp = bflag * DualCorrelatorParams::shmem_ab_stride;
        sp += (t32 * DualCorrelatorParams::shmem_t32_stride)
              + (t8 * DualCorrelatorParams::shmem_t8_stride);
        sp += (s8 * DualCorrelatorParams::shmem_s8_stride)
              + (s1 * DualCorrelatorParams::shmem_s1_stride);
    }


    __host__ void _init_warp_tiling()
    {
        static_assert(DualCorrelatorParams::warps_per_block == 8);

        // Convert warpId to tile coordinates 0 <= wx < 4 and 0 <= wy < 2.
#if 1
        int wx = warpId & 0x3;
        int wy = warpId >> 2;
#else
        int wx = warpId >> 1;
        int wy = warpId & 0x1;
#endif

        // Assign 'ap' and 'bp' pointers.
        //
        // The shared memory layout has been chosen so that adding the laneId is correct here.
        // From overleaf:
        //    (t0 t1 t2 t3 t4)
        //     <-> (j3 j4 i0 i1 i2)
        //     <-> (t8, t16, s1, s2, s4)
        //     <-> (1, 2, 4, 8, 16) int32s

        ap = (wx * 4 * DualCorrelatorParams::shmem_s8_stride) + laneId; // one x-tile corresponds to 32 stations
        bp = (wy * 8 * DualCorrelatorParams::shmem_s8_stride) + laneId
             + DualCorrelatorParams::shmem_ab_stride; // one y-tile corresponds to 64 stations

        // Output visibility matrix (station indices are GLOBAL, i.e. over A followed by B).
        vi_warp = 128 * block_atile + 32 * wx;
        vk_warp = 128 * block_btile + 64 * wy;
    }
};


shared_ptr<int> precompute_offsets(const DualCorrelatorParams& params)
{
    const auto blocks =
        dual_enumerate_blocks(params.ntiles_1d, params.ntiles_a, params.block_class_mask);

    int nblocks = (int)blocks.size();
    int nwarps = DualCorrelatorParams::warps_per_block;
    constexpr int nrows = DualCorrelatorParams::ptable_nrows;

    assert(nblocks == params.threadblocks_per_freq);

    Array<int> ptable_arr({nblocks, nrows, nwarps, 32}, af_rhost | af_zero);

    for (int blockId = 0; blockId < nblocks; blockId++) {
        for (int warpId = 0; warpId < nwarps; warpId++) {
            for (int laneId = 0; laneId < 32; laneId++) {
                PointerOffsets p(params, blocks[blockId].first, blocks[blockId].second, warpId,
                                 laneId);
                ptable_arr.at({blockId, 0, warpId, laneId}) = p.ap;
                ptable_arr.at({blockId, 1, warpId, laneId}) = p.bp;
                ptable_arr.at({blockId, 2, warpId, laneId}) = p.gp;
                ptable_arr.at({blockId, 3, warpId, laneId}) = p.sp;
                ptable_arr.at({blockId, 4, warpId, laneId}) = p.vi_warp;
                ptable_arr.at({blockId, 5, warpId, laneId}) = p.vk_warp;
                ptable_arr.at({blockId, 6, warpId, laneId}) = p.ts;
                ptable_arr.at({blockId, 7, warpId, laneId}) = p.gsel;
            }
        }
    }

    ptable_arr = ptable_arr.to_gpu();
    return std::static_pointer_cast<int>(ptable_arr.base);
}


} // namespace n2k_dual
