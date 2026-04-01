#include "../include/ksgpu/test_utils.hpp"
#include "../include/ksgpu/rand_utils.hpp"
#include "../include/ksgpu/xassert.hpp"

#include <cmath>

using namespace std;

namespace ksgpu {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------


// Helper for make_random_shape() and make_random_reshape_compatible_shapes().
inline long make_random_axis(long maxaxis, long &maxsize)
{
    xassert(maxaxis > 0);
    xassert(maxsize > 0);
    
    long n = std::min(maxaxis, maxsize);
    double t = rand_uniform(1.0e-6, log(n+1.0) - 1.0e-6);
    long ret = long(exp(t));  // round down
    maxsize = maxsize / ret;  // round down
    return ret;
}


vector<long> make_random_shape(int ndim, long maxaxis, long maxsize)
{
    if (ndim == 0)
        ndim = rand_int(1, ArrayMaxDim+1);
    
    xassert(ndim > 0);
    xassert(ndim <= ArrayMaxDim);
    xassert(maxsize > 0);

    vector<long> shape(ndim);
    for (int d = 0; d < ndim; d++)
        shape[d] = make_random_axis(maxaxis, maxsize);  // modifies 'maxsize'

    randomly_permute(shape);
    return shape;
}


// -------------------------------------------------------------------------------------------------


vector<long> make_random_strides(int ndim, const long *shape, int ncontig, long nalign)
{
    xassert(ncontig >= 0);
    xassert(ncontig <= ndim);
    xassert(nalign >= 1);

    int nd_strided = ndim - ncontig;
    vector<long> axis_ordering = rand_permutation(nd_strided);
    
    vector<long> strides(ndim);
    long min_stride = 1;

    // These strides are contiguous
    for (int d = ndim-1; d >= nd_strided; d--) {
        xassert(shape[d] > 0);
        strides[d] = min_stride;
        min_stride += (shape[d]-1) * strides[d];
    }

    // These strides are not necessarily contiguous
    for (int i = 0; i < nd_strided; i++) {
        int d = axis_ordering[i];
        xassert(shape[d] > 0);

        // Assign stride (as multiple of nalign)
        long smin = (min_stride + nalign - 1) / nalign;
        long smax = std::max(smin+1, (2*min_stride)/nalign);
        long s = (rand_uniform() < 0.33) ? smin : rand_int(smin,smax+1);
        
        strides[d] = s * nalign;
        min_stride += (shape[d]-1) * strides[d];
    }

    return strides;
}


vector<long> make_random_strides(const vector<long> &shape, int ncontig, long nalign)
{
    return make_random_strides(shape.size(), &shape[0], ncontig, nalign);
}


vector<long> make_contiguous_strides(int ndim, const long *shape)
{
    vector<long> strides(ndim);
    long curr_stride = 1;

    // These strides are contiguous
    for (int d = ndim-1; d >= 0; d--) {
        xassert(shape[d] > 0);
        strides[d] = curr_stride;
        curr_stride += (shape[d]-1) * strides[d];
    }

    return strides;
}


vector<long> make_contiguous_strides(const vector<long> &shape)
{
    return make_contiguous_strides(shape.size(), &shape[0]);
}


// -------------------------------------------------------------------------------------------------


// Helper for make_random_reshape_compatible_shapes()
struct RcBlock
{
    // If dflag==true, then this RcBlock represents a single src axis and multiple dst axes.
    // If dflag==false, then this RcBlock represents multiple src axes and a single dst axis.
    bool dflag = false;
    
    vector<long> bshape;    // multi-axis ("factored") shape
    long bstride = 0;       // single-axis stride
    long bsize = 0;         // single-axis length (= product of 'bshape')

    // Note: the following special case is allowed:
    //  bshape = empty vector
    //  bstrides = empty vector
    //  bsize = 1
};


// Helper for make_random_reshape_compatible_shapes()
struct RcAxis
{
    long len = 0;
    long stride = 0;

    RcAxis() { }
    RcAxis(long len_, long stride_) : len(len_), stride(stride_) { }
};


void make_random_reshape_compatible_shapes(vector<long> &dshape, vector<long> &sshape, vector<long> &sstrides)
{
    long maxaxis = 10;
    long maxsize = 100000;

    // Initialize all members of 'blocks' except 'bstrides'.
    
    vector<RcBlock> blocks;
    uint ddims = 0;
    uint sdims = 0;

    while ((ddims < ArrayMaxDim) || (sdims < ArrayMaxDim)) {
        RcBlock block;
        block.dflag = rand_int(0,2);
        
        uint &fdims = block.dflag ? ddims : sdims;   // "factored" dims
        uint &udims = block.dflag ? sdims : ddims;   // "unfactored" dims

        if ((udims == ArrayMaxDim) && (fdims == 0))
            continue;  // corner case
        if (udims == ArrayMaxDim)
            break;
        
        uint nb = 0;  // number of factored axes
        int nb_max = ArrayMaxDim - fdims;

        if ((nb_max > 0) && (rand_uniform() < 0.95)) {
            nb = rand_int(1, nb_max+1);
            block.bshape = make_random_shape(nb, maxaxis, maxsize);
            xassert(block.bshape.size() == nb);  // should never fail
        }
        
        block.bsize = 1;
        for (uint i = 0; i < nb; i++)
            block.bsize *= block.bshape[i];

        xassert(block.bsize > 0);  // should never fail
        maxsize /= block.bsize;
        
        blocks.push_back(block);
        fdims += nb;
        udims += 1;

        if ((ddims > 0) && (sdims > 0) && (rand_uniform() < 0.2))
            break;
    }
    
    // Should never fail.
    xassert(ddims > 0);
    xassert(sdims > 0);

    randomly_permute(blocks);

    // Initialize 'bstride' members of 'blocks'.
    
    int nblocks = blocks.size();
    vector<long> block_sizes(nblocks);    
    for (int i = 0; i < nblocks; i++)
        block_sizes[i] = blocks[i].bsize;
    
    vector<long> block_strides = make_random_strides(block_sizes);
    for (int i = 0; i < nblocks; i++)
        blocks[i].bstride = block_strides[i];

    randomly_permute(blocks);

    // Now 'blocks' has been fully initialized.
    // Unpack 'blocks' into 'daxes', 'saxes'.

    vector<RcAxis> daxes;
    vector<RcAxis> saxes;

    for (int i = 0; i < nblocks; i++) {
        const RcBlock &block = blocks[i];
        vector<RcAxis> &faxes = block.dflag ? daxes : saxes;
        vector<RcAxis> &uaxes = block.dflag ? saxes : daxes;

        uaxes.push_back({ block.bsize, block.bstride });

        long nf = faxes.size();
        faxes.resize(nf + block.bshape.size());

        long stride = block.bstride;
        for (int j = block.bshape.size()-1; j >= 0; j--) {
            faxes[nf+j].len = block.bshape[j];
            faxes[nf+j].stride = stride;
            stride *= block.bshape[j];
        }
    }

    // Should never fail.
    xassert(daxes.size() == ddims);
    xassert(saxes.size() == sdims);

    // Unpack (daxes,saxes) into (dshape,sshape,sstrides).
    
    dshape.resize(ddims);
    sshape.resize(sdims);
    sstrides.resize(sdims);

    for (uint i = 0; i < ddims; i++)
        dshape[i] = daxes[i].len;
    for (uint i = 0; i < sdims; i++)
        sshape[i] = saxes[i].len;
    for (uint i = 0; i < sdims; i++)
        sstrides[i] = saxes[i].stride;
}


// -------------------------------------------------------------------------------------------------


__global__ void busy_wait_kernel(uint *arr, long niter)
{
    uint x = arr[threadIdx.x];
    for (long i = 0; i < niter; i++) {
        x = (x ^ 0x12345678U);
        x = (x << 5) ^ (x >> 10);
    }
    arr[threadIdx.x] = x;
}

void launch_busy_wait_kernel(Array<uint> &arr, double a40_seconds, cudaStream_t s)
{
    xassert(arr.ndim == 1);
    xassert(arr.size == 32);
    xassert(arr.strides[0] == 1);
    xassert(arr.on_gpu());

    long niter = 1.4e8 * a40_seconds;
    
    busy_wait_kernel <<< 1, 32, 0, s >>>
        (arr.data, niter);
}

}  // namespace ksgpu
