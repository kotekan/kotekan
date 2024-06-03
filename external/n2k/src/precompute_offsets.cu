#include <gputils.hpp>
#include "../include/n2k.hpp"

using namespace std;
using namespace gputils;
using namespace n2k;


namespace n2k {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------


struct PointerOffsets {
    const CorrelatorParams &params;
    const int blockId;
    const int warpId;
    const int laneId;

    // Per-block data
    int block_atile = -1;
    int block_btile = -1;
    
    // These go into the ptable
    int ap = -1;
    int bp = -1;
    int gp = -1;
    int sp = -1;
    int vp = -1;
    int vk_minus_vi = 1000;
    
    
    __host__ PointerOffsets(const CorrelatorParams &params_, int blockId_, int warpId_, int laneId_) :
	params(params_), blockId(blockId_), warpId(warpId_), laneId(laneId_)
    {
	assert((laneId >= 0) && (laneId < 32));
	assert((warpId >= 0) && (warpId < CorrelatorParams::warps_per_block));
	assert((blockId >= 0) && (blockId < params.threadblocks_per_freq));

	_init_block_data();
	_init_prefetch_offsets();
	_init_warp_tiling();
    }

    
    __host__ void _init_block_data()
    {

	// First option is faster.
#if 1
	int nd = params.ntiles_2d_offdiag;

	if (blockId >= nd) {
	    block_atile = blockId - nd;
	    block_btile = blockId - nd;
	    return;
	}
	
	// FIXME lazy
	block_atile = 0;
	block_btile = 1;

	for (int c = 0; c < blockId; c++) {
	    block_atile++;
	    if (block_atile >= block_btile) {
		block_atile = 0;
		block_btile++;
	    }
	}
#else
	// FIXME lazy
	block_atile = 0;
	block_btile = 0;

	for (int c = 0; c < blockId; c++) {
	    if (block_atile < block_btile)
		block_atile++;
	    else {
		block_atile = 0;
		block_btile++;
	    }
	}
#endif
    }

    
    __host__ void _init_prefetch_offsets()
    {
	// Ultra-weird: the first one is consistently fast, whereas the second one
	// sometimes runs slow!
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
	
	assert(tpf == 32*t32 + 8*t8);
	assert(t32 < 2);
	assert(t8 < 4);
	
	gp = 32 * (bflag ? block_btile : block_atile);   // each block_atile/block_btile index corresponds to 128 stations = 32 int32s
	gp += tpf * params.emat_tstride;
	gp += laneId;

	// Assign 'sp' pointer.
	// Each laneId corresponds to 4 stations, so we write (4*laneId) = 8*s8 + s1
	
	int s8 = laneId >> 1;
	int s1 = (laneId & 0x1) << 2;
	
	assert(4*laneId == 8*s8 + s1);
	assert(s8 < 16);
	assert(s1 < 8);
	
	sp = bflag * CorrelatorParams::shmem_ab_stride;
	sp += (t32 * CorrelatorParams::shmem_t32_stride) + (t8 * CorrelatorParams::shmem_t8_stride);
	sp += (s8 * CorrelatorParams::shmem_s8_stride) + (s1 * CorrelatorParams::shmem_s1_stride);
    }


    __host__ void _init_warp_tiling()
    {
	static_assert(CorrelatorParams::warps_per_block == 8);

	// Convert warpId to tile coordinates 0 <= wx < 4 and 0 <= wy < 2.
	// I tried doing this in two different ways, but don't see a significant speed difference.
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

	ap = (wx * 4 * CorrelatorParams::shmem_s8_stride) + laneId;                                      // one x-tile corresponds to 32 stations
	bp = (wy * 8 * CorrelatorParams::shmem_s8_stride) + laneId + CorrelatorParams::shmem_ab_stride;  // one y-tile corresponds to 64 stations

	// Location in visibility matrix
	// t0 t1 t2 t3 t4 <-> k1 k2 k3 i1 i2
	
	int vi_warp = 128*block_atile + 32*wx;
	int vk_warp = 128*block_btile + 64*wy;

	int vi = vi_warp + 2*(laneId >> 3);
	int vk = vk_warp + 2*(laneId & 0x7);

	// Assign 'vp' pointer offset.
	vp = (vi * params.vmat_istride) + (vk * params.vmat_kstride);

	// Assign 'vk_minus_vi' offset.
	vk_minus_vi = vk_warp - vi_warp;
    }
};


shared_ptr<int> precompute_offsets(const CorrelatorParams &params)
{
    int nblocks = params.threadblocks_per_freq;
    int nwarps = CorrelatorParams::warps_per_block;
    
    Array<int> ptable_arr({nblocks,6,nwarps,32}, af_rhost | af_zero);

    for (int blockId = 0; blockId < nblocks; blockId++) {
	for (int warpId = 0; warpId < nwarps; warpId++) {
	    for (int laneId = 0; laneId < 32; laneId++) {
		PointerOffsets p(params, blockId, warpId, laneId);
		ptable_arr.at({blockId,0,warpId,laneId}) = p.ap;
		ptable_arr.at({blockId,1,warpId,laneId}) = p.bp;
		ptable_arr.at({blockId,2,warpId,laneId}) = p.gp;
		ptable_arr.at({blockId,3,warpId,laneId}) = p.sp;
		ptable_arr.at({blockId,4,warpId,laneId}) = p.vp;
		ptable_arr.at({blockId,5,warpId,laneId}) = p.vk_minus_vi;
	    }
	}
    }

    ptable_arr = ptable_arr.to_gpu();
    return ptable_arr.base;
}


}  // namespace n2k
