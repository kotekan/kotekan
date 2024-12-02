#include <sstream>
#include <iostream>

#include "../include/gputils/Array.hpp"
#include "../include/gputils/constexpr_functions.hpp"
#include "../include/gputils/cuda_utils.hpp"
#include "../include/gputils/device_mma.hpp"
#include "../include/gputils/test_utils.hpp"

using namespace std;
using namespace gputils;


// -------------------------------------------------------------------------------------------------
//
// __global__ kernel for int4/int8 MMAs.
//
// The 'asrc' array shape is (na, 32*Areg).
// The 'bsrc' array shape is (nb, 32*Breg).
// The 'cdst' array shape is (na, nb, 32*Creg)
//
// This kernel should be launched with (nblocks_x, nblocks_y) = (na, nb),
// and 32 threads/block.


template<void (*F)(int[], const int[], const int[], const int[]), int Areg, int Breg, int Creg>
__global__ void mma_int_kernel(int *cdst, const int *asrc, const int *bsrc)
{
    assert(blockDim.x == 32);

    int ia = blockIdx.x;
    int ib = blockIdx.y;
    int nb = gridDim.y;
    int laneId = threadIdx.x;

    // Add block offsets
    asrc += ia * 32 * Areg;
    bsrc += ib * 32 * Breg;
    cdst += (ia*nb + ib) * 32 * Creg;

    int afrag[Areg];
    int bfrag[Breg];
    int cfrag[Creg];

    for (int r = 0; r < Areg; r++)
	afrag[r] = asrc[32*r + laneId];
    for (int r = 0; r < Breg; r++)
	bfrag[r] = bsrc[32*r + laneId];    
    for (int r = 0; r < Creg; r++)
	cfrag[r] = 0;

    F(cfrag, afrag, bfrag, cfrag);
    
    for (int r = 0; r < Creg; r++)
	cdst[32*r + laneId] = cfrag[r];
}


// -------------------------------------------------------------------------------------------------
//
// __global__ kernel for float16 MMAs
//
// The 'asrc' array shape is (na, 64*Areg).
// The 'bsrc' array shape is (nb, 64*Breg).
// The 'cdst' array shape is (na, nb, 64*Creg)
//
// This kernel should be launched with (nblocks_x, nblocks_y) = (na, nb),
// and 32 threads/block.


__device__ __half2 load_half2(const float *p)
{
    float2 a = *((float2 *) p);
    return __float22half2_rn(a);
}


__device__ void store_half2(float *p, __half2 x)
{
    float2 a = __half22float2(x);
    *((float2 *) p) = a;
}


template<void (*F)(__half2[], const __half2[], const __half2[], const __half2[]), int Areg, int Breg, int Creg>
__global__ void mma_float16_kernel(float *cdst, const float *asrc, const float *bsrc)
{
    assert(blockDim.x == 32);

    int ia = blockIdx.x;
    int ib = blockIdx.y;
    int nb = gridDim.y;
    int laneId = threadIdx.x;

    // Add block offsets
    // Note 64 here (whereas 32 in mma_int32_kernel() above)
    asrc += ia * 64 * Areg;
    bsrc += ib * 64 * Breg;
    cdst += (ia*nb + ib) * 64 * Creg;

    __half2 afrag[Areg];
    __half2 bfrag[Breg];
    __half2 cfrag[Creg];

    for (int r = 0; r < Areg; r++)
	afrag[r] = load_half2(asrc + 64*r + 2*laneId);
    for (int r = 0; r < Breg; r++)
	bfrag[r] = load_half2(bsrc + 64*r + 2*laneId);
    for (int r = 0; r < Creg; r++)
	cfrag[r] = __half2half2(0);

    F(cfrag, afrag, bfrag, cfrag);

    for (int r = 0; r < Creg; r++)
	store_half2(cdst + 64*r + 2*laneId, cfrag[r]);
}


// -------------------------------------------------------------------------------------------------


__host__ int slow_ilog2(int n)
{
    assert(n > 0);
    int i = log2(1.5 * n);
    assert(n == (1 << i));
    return i;
}


template<typename T>
__host__ Array<T> slow_matmul(const Array<T> &a_, const Array<T> &b_)
{
    Array<T> a = a_.to_host();
    Array<T> b = b_.to_host();
    
    assert(a.ndim == 2);
    assert(b.ndim == 2);
    assert(a.shape[1] == b.shape[0]);

    int m = a.shape[0];
    int p = a.shape[1];
    int n = b.shape[1];

    Array<T> c({m,n}, af_rhost);

    for (int i = 0; i < m; i++) {
	for (int j = 0; j < n; j++) {
	    T t = 0;
	    for (int k = 0; k < p; k++)
		t += a.at({i,k}) * b.at({k,j});
	    c.at({i,j}) = t;
	}
    }

    return c;
}


// -------------------------------------------------------------------------------------------------
//
// The MatParams class corresponds to one matrix (out of three: two source matrices and
// one accumulator). There is a little class hierarchy:
//
//   MatParamsBase
//     MatParamsInt
//     MatParamsFloat16
//
// The MatParams class does two things:
//
//   - Keeps track of the register mapping (i.e. mapping between logical and physical bits).
//     This is implemented in the MatParamsBase base class, via the following methods:
//
//        assign_state_bit()
//        show_layout()
//        finalize()
//        pindex_to_rc()
//        rc_to_pindex()
//     
//   - Converts between 2-d "matrices" and 1-d "fragments", which are represented respectively as:
//
//        Array<Dtype> matrix({nrows,ncols})
//        Array<Dtype> fragment({fragment_length})   where Dtype = float or int
//
//     This is implemented in the MatParamsInt and MatParamsFloat16 subclasses, via the following methods:
//
//        make_fragment()
//        make_basis_fragments()
//        unpack_fragment()
//        pack_fragment()
//        test_pack_unpack()
//
// !!! IMPORTANT NOTE !!!
//
//    Throughout the code, the ordering of "physical" bits is:
//      - b (position within a register)
//      - t (thread id)
//      - r (register within a thread)


template<int Nrows, int Ncols, int BitDepth>
struct MatParamsBase
{
    static constexpr int nrows = Nrows;
    static constexpr int ncols = Ncols;
    static constexpr int bit_depth = BitDepth;

    static_assert(constexpr_is_pow2(nrows));
    static_assert(constexpr_is_pow2(ncols));
    
    static constexpr int bits_per_fragment = nrows * ncols * bit_depth;
    static constexpr int registers_per_thread = bits_per_fragment / 1024;

    // A "logical" location is identified by a (row,col).
    static constexpr int num_rowstate_bits = constexpr_ilog2(nrows);
    static constexpr int num_colstate_bits = constexpr_ilog2(ncols);

    // A "physical" location is identified by (b,r,t), where b indexes a "bit" location within
    // a single register, r indexes a register in a thread, and t indexes a thread.
    static constexpr int num_bstate_bits = constexpr_ilog2(32 / bit_depth);
    static constexpr int num_regstate_bits = constexpr_ilog2(registers_per_thread);

    // Physical and logical locations should be parameterized by the same number of state bits.
    static constexpr int num_state_bits = num_rowstate_bits + num_colstate_bits;
    static_assert(num_state_bits == num_bstate_bits + num_regstate_bits + 5);

    // State bit mapping (physical -> logical)
    vector<bool> pbit_isrow;
    vector<int> pbit_lindex;
    bool finalized = false;

    
    MatParamsBase() :
	pbit_isrow(num_state_bits, false),
	pbit_lindex(num_state_bits, -1)
    { }

    
    void assign_state_bit(int b, bool isrow, int index)
    {
	assert(!finalized);
	assert((b >= 0) && (b < num_state_bits));
	assert(this->pbit_lindex[b] < 0);

	assert(index >= 0);
	assert(index < (isrow ? num_rowstate_bits : num_colstate_bits));
	
	this->pbit_isrow[b] = isrow;
	this->pbit_lindex[b] = index;
    }


    void finalize()
    {
	for (int pb = 0; pb < num_state_bits; pb++)
	    assert(pbit_lindex[pb] >= 0);
	
	this->finalized = true;
    }
    
    
    // Helper for show_layout()
    void _show_state_bit(int b, const char *phys_prefix, int phys_index, const char *row_prefix, const char *col_prefix) const
    {
	assert((b >= 0) && (b < num_state_bits));
	cout << "        " << phys_prefix << phys_index << " <-> ";

	if (this->pbit_lindex[b] < 0)
	    cout << "[unassigned]\n";
	else if (this->pbit_isrow[b])
	    cout << row_prefix << this->pbit_lindex[b] << "\n";
	else
	    cout << col_prefix << this->pbit_lindex[b] << "\n";
    }

    
    void show_layout(const char *row_prefix, const char *col_prefix) const
    {
	for (int b = 0; b < num_bstate_bits; b++)
	    _show_state_bit(b, "b", b, row_prefix, col_prefix);
	
	for (int t = 0; t < 5; t++)
	    _show_state_bit(num_bstate_bits + t, "t", t, row_prefix, col_prefix);

	for (int r = 0; r < num_regstate_bits; r++)
	    _show_state_bit(num_bstate_bits + 5 + r, "r", r, row_prefix, col_prefix);
    }


    // Input: a "physical" index 0 <= pindex < (nrows * ncols).
    // Output: pair of logical indices 0 <= rindex <= nrows, and 0 <= cindex < ncols.
    
    __host__ void pindex_to_rc(int pindex, int &rindex, int &cindex) const
    {
	assert(finalized);
	assert((pindex >= 0) && (pindex < nrows * ncols));

	rindex = 0;
	cindex = 0;
	
	for (int b = 0; b < num_state_bits; b++)
	    if (pindex & (1 << b))
		(pbit_isrow[b] ? rindex : cindex) += (1 << pbit_lindex[b]);
    }
    

    // Input: pair of logical indices 0 <= rindex <= nrows, and 0 <= cindex < ncols.
    // Output: a "physical" index 0 <= pindex < (nrows * ncols).
    
    __host__ int rc_to_pindex(int rindex, int cindex) const
    {
	assert(finalized);
	assert((rindex >= 0) && (rindex < nrows));
	assert((cindex >= 0) && (cindex < ncols));

	int pindex = 0;

	for (int b = 0; b < num_state_bits; b++)
	    if ((pbit_isrow[b] ? rindex : cindex) & (1 << pbit_lindex[b]))
		pindex += (1 << b);

	return pindex;
    }
};


// -------------------------------------------------------------------------------------------------


template<int Nrows, int Ncols, int BitDepth>
struct MatParamsInt : public MatParamsBase<Nrows, Ncols, BitDepth>
{
    using Base = MatParamsBase<Nrows, Ncols, BitDepth>;

    // Array datatype for matrices and fragments, see comment above
    using Dtype = int;
    
    using Base::nrows;
    using Base::ncols;
    using Base::bit_depth;
    using Base::num_state_bits;

    static constexpr int fragment_length = Base::bits_per_fragment / 32;


    // Returns 1-d array of length fragment_length.
    __host__ Array<int> make_fragment(int aflags=0) const
    {
	return Array<int> ({fragment_length}, aflags);
    }

    
    // Returns 2-d array of shape (num_state_bits+1, fragment_length).
    static __host__ Array<int> make_basis_fragments()
    {
	Array<int> ret({num_state_bits+1, fragment_length}, af_zero);  // on cpu
	ret.at({0,0}) = 1;

	for (int b = 0; b < num_state_bits; b++) {
	    int bit_index = bit_depth * (1 << b);
	    int int32_index = bit_index / 32;
	    ret.at({b+1,int32_index}) = (1 << (bit_index % 32));
	}
    
	return ret.to_gpu();
    }


    // Converts (fragment_length,) -> (nrows, ncols)
    __host__ Array<int> unpack_fragment(const Array<int> &src_) const
    {
	Array<int> src = src_.to_host();
	assert(src.shape_equals({fragment_length}));
	
	Array<int> dst({nrows, ncols}, af_rhost);

	for (int ir = 0; ir < nrows; ir++) {
	    for (int ic = 0; ic < ncols; ic++) {
		int p = Base::rc_to_pindex(ir, ic);
		int j = p / (32/bit_depth);
		int b = (p*bit_depth) - (32*j);

		dst.at({ir,ic}) = (src.data[j] << (32-bit_depth-b)) >> (32-bit_depth);
	    }
	}

	return dst;
    }


    // Converts (nrows, ncols) -> (fragment_length,)
    __host__ Array<int> pack_fragment(const Array<int> &src_) const
    {
	int smax = (1U << (bit_depth-1)) - 1U;
	int smin = -smax - 1;
    
	Array<int> src = src_.to_host();
	assert(src.shape_equals({nrows, ncols}));

	Array<int> dst({fragment_length}, af_zero);

	for (int ir = 0; ir < nrows; ir++) {
	    for (int ic = 0; ic < ncols; ic++) {
		int p = Base::rc_to_pindex(ir, ic);
		int j = p / (32/bit_depth);
		int b = (p*bit_depth) - (32*j);
		
		int s = src.at({ir,ic});
		assert((s >= smin) && (s <= smax));

		unsigned int us = (s << (32-bit_depth));
		s = (us >> (32-bit_depth-b));
		dst.at({j}) |= s;
	    }
	}

	return dst;
    }

    
    __host__ void test_pack_unpack() const
    {
	for (int iouter = 0; iouter < 10; iouter++) {
	    Array<int> a = make_fragment(af_random);
	    Array<int> b = pack_fragment(unpack_fragment(a));

	    assert(a.shape_equals({fragment_length}));
	    assert(b.shape_equals({fragment_length}));
	    
	    for (int i = 0; i < fragment_length; i++)
		assert(a.data[i] == b.data[i]);
	}
    }    
};


// -------------------------------------------------------------------------------------------------


template<int Nrows, int Ncols>
struct MatParamsFloat16 : MatParamsBase<Nrows, Ncols, 16>
{
    using Base = MatParamsBase<Nrows, Ncols, 16>;

    // Array datatype for matrices and fragments, see comment above
    using Dtype = float;
    
    using Base::nrows;
    using Base::ncols;
    using Base::num_state_bits;

    static constexpr int fragment_length = nrows * ncols;
    static_assert(fragment_length == 64 * Base::registers_per_thread);


    // Returns 1-d array of length fragment_length.
    __host__ Array<float> make_fragment(int aflags=0) const
    {
	return Array<float> ({fragment_length}, aflags);
    }

    
    // Returns 2-d array of shape (num_state_bits+1, fragment_length).
    static __host__ Array<float> make_basis_fragments()
    {
	Array<float> ret({num_state_bits+1, fragment_length}, af_zero);  // on cpu
	ret.at({0,0}) = 1.0;

	for (int b = 0; b < num_state_bits; b++)
	    ret.at({b+1,1<<b}) = 1.0;
    
	return ret.to_gpu();
    }


    // Converts (fragment_length,) -> (nrows, ncols)
    __host__ Array<float> unpack_fragment(const Array<float> &src_) const
    {
	Array<float> src = src_.to_host();
	assert(src.shape_equals({fragment_length}));
	
	Array<float> dst({nrows,ncols}, af_rhost);

	for (int ir = 0; ir < nrows; ir++) {
	    for (int ic = 0; ic < ncols; ic++) {
		int p = Base::rc_to_pindex(ir, ic);
		dst.at({ir,ic}) = src.at({p});
	    }
	}

	return dst;
    }


    // Converts (nrows, ncols) -> (fragment_length,)
    __host__ Array<float> pack_fragment(const Array<float> &src_) const
    {
	Array<float> src = src_.to_host();
	assert(src.shape_equals({nrows, ncols}));

	Array<float> dst({fragment_length}, af_zero);

	for (int ir = 0; ir < nrows; ir++) {
	    for (int ic = 0; ic < ncols; ic++) {
		int p = Base::rc_to_pindex(ir, ic);
		dst.at({p}) = src.at({ir,ic});
	    }
	}

	return dst;
    }
};


// -------------------------------------------------------------------------------------------------
//
// A little unit test for MatParams{Int,Float16}.


template<typename MatParams>
void test_pack_unpack(const MatParams &params)
{
    constexpr int fragment_length = MatParams::fragment_length;
    
    for (int iouter = 0; iouter < 10; iouter++) {
	// Declared 'auto' since Array<MatParams::Dtype> gave an incomprehensible compiler error.
	auto a = params.make_fragment(af_random);
	auto b = params.pack_fragment(params.unpack_fragment(a));
	
	assert(a.shape_equals({fragment_length}));
	assert(b.shape_equals({fragment_length}));
	
	for (int i = 0; i < fragment_length; i++)
	    assert(a.data[i] == b.data[i]);
    }
}


// -------------------------------------------------------------------------------------------------


template<typename Dtype,   // either int or float
	 void (*Kernel)(Dtype *, const Dtype *, const Dtype *),  // __global__ kernel
	 typename AParams,    // MatParams for A-factor
	 typename BParams,    // MatParams for B-factor
	 typename CParams>    // MatParams for C-factor
struct MmaParams
{
    const string name;
    const int verbosity;
    
    AParams aparams;
    BParams bparams;
    CParams cparams;

    static_assert(AParams::ncols == BParams::nrows);
    static_assert(CParams::nrows == AParams::nrows);
    static_assert(CParams::ncols == BParams::ncols);
    
    static constexpr int M = AParams::nrows;
    static constexpr int N = BParams::ncols;
    static constexpr int K = AParams::ncols;

    
    MmaParams(const string &name_, int verbosity_=0)
	: name(name_), verbosity(verbosity_)
    { }

    
    Array<Dtype> run_kernel(const Array<Dtype> &asrc, const Array<Dtype> &bsrc) const
    {
	assert((asrc.ndim >= 1) && (asrc.ndim <= 2));
	assert((bsrc.ndim >= 1) && (bsrc.ndim <= 2));
	assert(asrc.shape[asrc.ndim-1] == AParams::fragment_length);
	assert(bsrc.shape[bsrc.ndim-1] == BParams::fragment_length);
	
	int na = (asrc.ndim > 1) ? asrc.shape[0] : 1;
	int nb = (bsrc.ndim > 1) ? bsrc.shape[0] : 1;

	vector<ssize_t> cshape;
	if (asrc.ndim > 1)
	    cshape.push_back(na);
	if (bsrc.ndim > 1)
	    cshape.push_back(nb);
	cshape.push_back(CParams::fragment_length);
	
	Array<Dtype> agpu = asrc.to_gpu();
	Array<Dtype> bgpu = bsrc.to_gpu();
	Array<Dtype> cdst(cshape, af_gpu);
	
	dim3 nblocks;
	nblocks.x = na;
	nblocks.y = nb;
	nblocks.z = 1;
	
	Kernel <<<nblocks, 32>>> (cdst.data, agpu.data, bgpu.data);
	CUDA_PEEK(name.c_str());

	return cdst.to_host();
    }


    void reverse_engineer()
    {
	constexpr int na = AParams::num_state_bits;
	constexpr int nb = BParams::num_state_bits;

	Array<Dtype> asrc = AParams::make_basis_fragments();
	Array<Dtype> bsrc = BParams::make_basis_fragments();
	
	Array<Dtype> cdst = this->run_kernel(asrc, bsrc);
	assert(cdst.shape_equals({na+1, nb+1, CParams::fragment_length}));

	Array<int> coupling({na+1,nb+1}, af_rhost);
	for (int i = 0; i < na+1; i++) {
	    for (int j = 0; j < nb+1; j++) {
		coupling.at({i,j}) = -1;
		for (int k = 0; k < cdst.shape[2]; k++) {
		    if (cdst.at({i,j,k}) != 0) {
			assert(cdst.at({i,j,k}) == Dtype(1));
			assert(coupling.at({i,j}) < 0);
			coupling.at({i,j}) = k;
		    }
		}
	    }
	}
	
	if (verbosity >= 1) {
	    cout << "Coupling matrix: " << name << endl;
	    for (int i = 0; i < na+1; i++) {
		for (int j = 0; j < nb+1; j++)
		    cout << "  " << coupling.at({i,j});
		cout << endl;
	    }
	}
	    
	assert(coupling.at({0,0}) == 0);
	
	int curr_i = 0;
	int curr_j = 0;
	int curr_k = 0;
    
	for (int ia = 0; ia < na; ia++) {
	    int d = coupling.at({ia+1,0});
	    
	    if (d >= 0) {  // index rows in A, C
		aparams.assign_state_bit(ia, true, curr_i);
		cparams.assign_state_bit(slow_ilog2(d), true, curr_i);
		curr_i++;
		continue;
	    }

	    bool flag = false;
	    
	    for (int ib = 0; ib < nb; ib++) {
		d = coupling.at({ia+1,ib+1});
		if (d < 0)
		    continue;

		// index col in A, row in B
		assert(d == 0);
		assert(!flag);
		aparams.assign_state_bit(ia, false, curr_j);
		bparams.assign_state_bit(ib, true, curr_j);
		flag = true;
		curr_j++;
	    }

	    assert(flag);
	}

	for (int ib = 0; ib < nb; ib++) {
	    int d = coupling.at({0,ib+1});
	    if (d >= 0) {  // index cols in B, C
		bparams.assign_state_bit(ib, false, curr_k);
		cparams.assign_state_bit(slow_ilog2(d), false, curr_k);
		curr_k++;
	    }
	}

	aparams.finalize();
	bparams.finalize();
	cparams.finalize();
	
	for (int ir = 0; ir < na+1; ir++) {
	    int i, ja;
	    int pindex_a = (ir > 0) ? (1 << (ir-1)) : 0;
	    aparams.pindex_to_rc(pindex_a, i, ja);
	    
	    for (int ic = 0; ic < nb+1; ic++) {
		int jb, k;
		int pindex_b = (ic > 0) ? (1 << (ic-1)) : 0;
		bparams.pindex_to_rc(pindex_b, jb, k);

		int d = (ja == jb) ? cparams.rc_to_pindex(i,k) : -1;
		assert(coupling.at({ir,ic}) == d);
	    }
	}
    }

    
    void show_layout() const
    {
	cout << "\n[" << name << "]" << endl;

	cout << "    A-matrix\n";
	aparams.show_layout("i", "j");
	
	cout << "    B-matrix\n";
	bparams.show_layout("j", "k");
	
	cout << "    C-matrix\n";
	cparams.show_layout("i", "k");
    }


    void end_to_end_test() const
    {
	test_pack_unpack(aparams);
	test_pack_unpack(bparams);
	test_pack_unpack(cparams);

	for (int iouter = 0; iouter < 10; iouter++) {
	    Array<Dtype> asrc({AParams::fragment_length}, af_random);
	    Array<Dtype> bsrc({BParams::fragment_length}, af_random);
	    
	    Array<Dtype> cgpu = run_kernel(asrc, bsrc);
	    cgpu = cparams.unpack_fragment(cgpu);

	    Array<Dtype> au = aparams.unpack_fragment(asrc);
	    Array<Dtype> bu = bparams.unpack_fragment(bsrc);
	    Array<Dtype> cu = slow_matmul(au, bu);

	    // Use 'epsabs', 'epsrel' values appropriate for float16.
	    assert_arrays_equal(cu, cgpu, "cpu", "gpu", {"row","col"}, 0.01, 0.01, 15, (verbosity >= 2));
	}

	cout << "end_to_end_test: pass" << endl;
    }
};


// -------------------------------------------------------------------------------------------------


template<void (*F)(int[], const int[], const int[], const int[]), int BitDepth, int M, int N, int K>
static void reverse_engineer_int_mma()
{
    using AParams = MatParamsInt <M, K, BitDepth>;
    using BParams = MatParamsInt <K, N, BitDepth>;
    using CParams = MatParamsInt <M, N, 32>;
    
    constexpr int Areg = AParams::registers_per_thread;
    constexpr int Breg = BParams::registers_per_thread;
    constexpr int Creg = CParams::registers_per_thread;

    stringstream name;
    name << "int" << BitDepth << ", m=" << M << ", n=" << N << ", k=" << K;

    MmaParams<int, mma_int_kernel<F,Areg,Breg,Creg>, AParams, BParams, CParams> params(name.str());
    params.reverse_engineer();
    params.show_layout();
    params.end_to_end_test();
}


template<void (*F)(__half2[], const __half2[], const __half2[], const __half2[]), int M, int N, int K>
static void reverse_engineer_float16_mma()
{
    using AParams = MatParamsFloat16 <M, K>;
    using BParams = MatParamsFloat16 <K, N>;
    using CParams = MatParamsFloat16 <M, N>;
    
    constexpr int Areg = AParams::registers_per_thread;
    constexpr int Breg = BParams::registers_per_thread;
    constexpr int Creg = CParams::registers_per_thread;

    stringstream name;
    name << "float16, m=" << M << ", n=" << N << ", k=" << K;

    MmaParams<float, mma_float16_kernel<F,Areg,Breg,Creg>, AParams, BParams, CParams> params(name.str());
    params.reverse_engineer();
    params.show_layout();
    params.end_to_end_test();
}


int main(int argc, char **argv)
{
    // Implements command-line usage: program [device].
    set_device_from_command_line(argc, argv);

    reverse_engineer_float16_mma <mma_f16_m16_n8_k8, 16, 8, 8> ();
    reverse_engineer_float16_mma <mma_f16_m16_n8_k16, 16, 8, 16> ();
	
    reverse_engineer_int_mma <mma_s4_m8_n8_k32, 4, 8, 8, 32> ();
    reverse_engineer_int_mma <mma_s4_m16_n8_k32, 4, 16, 8, 32> ();
    reverse_engineer_int_mma <mma_s4_m16_n8_k64, 4, 16, 8, 64> ();
    
    reverse_engineer_int_mma <mma_s8_m8_n8_k16, 8, 8, 8, 16> ();
    reverse_engineer_int_mma <mma_s8_m16_n8_k16, 8, 16, 8, 16> ();
    reverse_engineer_int_mma <mma_s8_m16_n8_k32, 8, 16, 8, 32> ();
    
    return 0;
}
