// I sometimes need to write "shuffling" operations which change the register layout of an array.
//
// These operations can be difficult to test and debug. I started writing a BitMapping class which
// automates testing. The caller just needs to declare the logical <-> physical bit mapping in the
// source and destination arrays.
//
// This file contains a self-contained example: testing a function shuffle_V_16_16() whose input is
//
//   [int32+32 C_{ik}]  (r0 r1 r2 r3) <-> (k0 i3 ReIm k3)   (t0 t1 t2 t3 t4) <-> (k1 k2 i0 i1 i2)
//
// and whose output is
//
//   [int32+32 C_{ik}]  (r0 r1 r2 r3) <-> (i2 i3 i0 i1)   (t0 t1 t2 t3 t4) <-> (ReIm k0 k1 k2 k3)
//
// The current implementation works pretty well, but assumes bit depth 32. Before adding it to
// ksgpu, I want to think about how to generalize to bit depth < 32, since this may involve API
// changes which aren't backwards compatible.

#include <sstream>
#include <iostream>
#include "../include/ksgpu/Array.hpp"
#include "../include/ksgpu/cuda_utils.hpp"

using namespace std;
using namespace ksgpu;


// -------------------------------------------------------------------------------------------------

    
// 'bit' should be a power of two
static __device__ void warp_transpose(int &in0, int &in1, const int bit)
{
    int flag = (threadIdx.x & bit);
    int src = flag ? in0 : in1;
    int dst = __shfl_xor_sync(0xffffffff, src, bit);
    (flag ? in0 : in1) = dst;
}


static __device__ void warp_transpose_4(int in0[4], int in1[4], const int bit)
{
    warp_transpose(in0[0], in1[0], bit);
    warp_transpose(in0[1], in1[1], bit);
    warp_transpose(in0[2], in1[2], bit);
    warp_transpose(in0[3], in1[3], bit);
}


static __device__ int compute_lane_perm()
{
    // t0 t1 t2 t3 t4 <-> k1 k2 i0 i1 i2
    // t0 t1 t2 t3 t4 <-> i0 i2 k1 k2 i1
    
    int t = threadIdx.x;
    
    return (((t & 0x01) << 2) |   // i0
            ((t & 0x10) >> 1) |   // i1
            ((t & 0x02) << 3) |   // i2
            ((t & 0x0c) >> 2));   // k1 k2
}

    
static __device__ void shuffle_Vre_16_8(int V[4], int lane_perm)
{
    // Input:
    //   r0 r1 <-> k0 i3
    //   t0 t1 t2 t3 t4 <-> k1 k2 i0 i1 i2
    
    V[0] = __shfl_sync(0xffffffff, V[0], lane_perm);    
    V[1] = __shfl_sync(0xffffffff, V[1], lane_perm);
    V[2] = __shfl_sync(0xffffffff, V[2], lane_perm);    
    V[3] = __shfl_sync(0xffffffff, V[3], lane_perm);
    
    // After __shfl_sync():
    //   r0 r1 <-> k0 i3
    //   t0 t1 t2 t3 t4 <-> i0 i2 k1 k2 i1
    
    warp_transpose(V[0], V[1], 0x02);
    warp_transpose(V[2], V[3], 0x02);
    
    // Output (swap k0,i2):
    //   r0 r1 <-> i2 i3
    //   t0 t1 t2 t3 t4 <-> i0 k0 k1 k2 i1
}


static __device__ void shuffle_V_16_8(int V[2][4], int lane_perm)
{
    shuffle_Vre_16_8(V[0], lane_perm);
    shuffle_Vre_16_8(V[1], lane_perm);
    
    // After shuffle_Vre_16_8():
    //   r0 r1 r2 <-> i2 i3 ReIm
    //   t0 t1 t2 t3 t4 <-> i0 k0 k1 k2 i1
    
    warp_transpose_4(V[0], V[1], 0x01);
    
    // Output (swap ReIm,i0)
    //   r0 r1 r2 <-> i2 i3 i0
    //   t0 t1 t2 t3 t4 <-> ReIm k0 k1 k2 i1
}


// The 'V0' and 'V1' args correspond to k=0 and k=8.
static __device__ void shuffle_V_16_16(int V0[2][4], int V1[2][4], int lane_perm)
{
    shuffle_V_16_8(V0, lane_perm);
    shuffle_V_16_8(V1, lane_perm);
    
    // After shuffle_V_16_8():
    //   r0 r1 r2 r3 <-> i2 i3 i0 k3
    //   t0 t1 t2 t3 t4 <-> ReIm k0 k1 k2 i1
    
    warp_transpose_4(V0[0], V1[0], 0x10);
    warp_transpose_4(V0[1], V1[1], 0x10);
    
    // Output (swap i1,k3)
    //   r0 r1 r2 r3 <-> i2 i3 i0 i1
    //   t0 t1 t2 t3 t4 <-> ReIm k0 k1 k2 k3
}


// -------------------------------------------------------------------------------------------------
//
// General framework for testing shuffle kernels
// FIXME move to ksgpu, but first add placeholder for bit_depth, including 2-d source array


struct BitMapping
{
    // FIXME currently assumed 
    vector<string> names;
    const int rank;

    BitMapping(const vector<string> &names_)
        : names(names_), rank(names_.size())
    {
        xassert(rank >= 5);
        xassert(rank <= 15);
        
        // No duplicates
        for (int i = 1; i < rank; i++)
            for (int j = 0; j < i; j++)
                xassert(names[i] != names[j]);
    }

    int name_to_index(const string &bit_name) const
    {
        for (int r = 0; r < rank; r++)
            if (names[r] == bit_name)
                return (1 << r);
        throw runtime_error("bit name '" + bit_name + "' not found");
    }

    string index_to_name(int i) const
    {
        xassert(i >= 0);
        xassert(i < (1 << rank));

        stringstream ss;
        ss << "[";
        
        for (int r = 0; r < rank; r++)
            if (i & (1 << r))
                ss << " " << names[r];
        
        ss << " ]";
        return ss.str();
    }

    Array<int> make_src_array_for_testing() const
    {
        const int n = (1 << rank);
        
        Array<int> ret({n}, af_rhost);
        for (int i = 0; i < n; i++)
            ret.data[i] = i;

        return ret.to_gpu();
    }
};


static void test_shuffle_kernel(const Array<int> &dst_, const BitMapping &bm_phys, const BitMapping &bm_in, const BitMapping &bm_out)
{
    xassert(bm_phys.rank == bm_in.rank);
    xassert(bm_phys.rank == bm_out.rank);
    
    int rank = bm_phys.rank;
    int n = (1 << rank);
    int nfail = 0;

    Array<int> dst = dst_.to_host();
    xassert(dst.shape_equals({n}));

    for (int i = 0; i < n; i++) {
        int d_expected = 0;
        for (int r = 0; r < rank; r++)
            if (i & (1 << r))
                d_expected |= bm_in.name_to_index(bm_out.names[r]);

        int d_actual = dst.at({i});
        if (d_actual == d_expected)
            continue;

        cout << "Mismatch found\n"
             << "    at dst_index=" << i
             << ", dst_phys=" << bm_phys.index_to_name(i)
             << ", dst_logical=" << bm_out.index_to_name(i) << "\n"
             << "    got array value " << d_actual
             << ", which came from src_phys=" << bm_phys.index_to_name(d_actual)
             << ", src_logical=" << bm_in.index_to_name(d_actual) << "\n"
             << "    expected array value " << d_expected
             << ", which was at src_phys=" << bm_phys.index_to_name(d_expected)
             << ", src_logical=" << bm_in.index_to_name(d_expected)
             << endl;

        nfail++;
    }

    if (nfail > 0) {
        cout << "shuffle kernel test failed" << endl;
        exit(1);
    }
}


// -------------------------------------------------------------------------------------------------
//
// Test shuffle_V_16_16()


__global__ void kernel_shuffle_V_16_16(int *dst, const int *src)
{
    assert(gridDim.x == 1);
    assert(blockDim.x == 32);
    
    int V0[2][4];
    int V1[2][4];

    int laneId = threadIdx.x & 0x1f;

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            V0[i][j] = src[128*i + 32*j + laneId];
            V1[i][j] = src[128*i + 32*j + laneId + 256];
        }
    }

    int lane_perm = compute_lane_perm();
    shuffle_V_16_16(V0, V1, lane_perm);

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            dst[128*i + 32*j + laneId] = V0[i][j];
            dst[128*i + 32*j + laneId + 256] = V1[i][j];
        }
    }
}


void test_shuffle_V_16_16()
{
    BitMapping bm_phys({"t0","t1","t2","t3","t4","r0","r1","r2","r3"});
    BitMapping bm_in({"k1","k2","i0","i1","i2","k0","i3","ReIm","k3"});
    BitMapping bm_out({"ReIm","k0","k1","k2","k3","i2","i3","i0","i1"});

    Array<int> src = bm_in.make_src_array_for_testing();
    Array<int> dst({src.size}, af_gpu);

    kernel_shuffle_V_16_16 <<<1,32>>> (dst.data, src.data);
    CUDA_PEEK("kernel_shuffle_V_16_16");
    CUDA_CALL(cudaDeviceSynchronize());

    test_shuffle_kernel(dst, bm_phys, bm_in, bm_out);
    cout << "test_shuffle_V_16_16: pass" << endl;
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char  **argv)
{
    test_shuffle_V_16_16();
    return 0;
}
