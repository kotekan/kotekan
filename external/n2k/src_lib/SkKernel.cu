#include "../include/n2k/rfi_kernels.hpp"

#include "../include/n2k/internals/bad_feed_mask.hpp"
#include "../include/n2k/internals/interpolation.hpp"
#include "../include/n2k/internals/sk_globals.hpp"

#include <gputils/cuda_utils.hpp>

using namespace std;
using namespace gputils;


namespace n2k {
#if 0
}  // editor auto-indent
#endif


// Helper for reduce_four().
__device__ inline float _reduce_pair(float x, float y, uint bit)
{
    bool upper = (threadIdx.x & bit);
    float src = upper ? x : y;
    float dst = upper ? y : x;
    
    dst += __shfl_sync(FULL_MASK, src, threadIdx.x ^ bit);
    return dst;
}


// reduce_four() returns:
//   (x0 summed over all lanes)  if laneId % 4 == 0
//   (x1 summed over all lanes)  if laneId % 4 == 1
//   (x2 summed over all lanes)  if laneId % 4 == 2
//   (x3 summed over all lanes)  if laneId % 4 == 3

__device__ inline float reduce_four(float x0, float x1, float x2, float x3)
{
    float x01 = _reduce_pair(x0, x1, 1);
    float x23 = _reduce_pair(x2, x3, 1);
    float y = _reduce_pair(x01, x23, 2);
    
    y += __shfl_sync(FULL_MASK, y, threadIdx.x ^ 4);
    y += __shfl_sync(FULL_MASK, y, threadIdx.x ^ 8);
    y += __shfl_sync(FULL_MASK, y, threadIdx.x ^ 16);
    
    return y;
}



// sk_kernel(): based on code by Nada El-Falou
//
// Constraints:
//
//   - Number of stations S is a multiple of 128 (from include/n2k/bad_feed_mask.hpp)
//   - Number of stations S is >= 32 * blockDim.x (from include/n2k/bad_feed_mask.hpp)
//
// Parallelization:
//
//    threadIdx.x <-> station
//    threadIdx.y, blockIdx.y <-> freq
//    threadIdx.z, blockIdx.z <-> time
//
// Shared memory layout:
//
//   uint bf[Nbf];           // temp buffer for load_bad_feed_mask(), Nbf = max(S/32,32)
//   float red[4*W];         // temp reduce buffer, layout (Ws,Wt,Wf,4).
//   uint rfimask[32];       // 32 copies of same 4-byte uint
//   float bsigma_coeffs[ncoeffs];   // currently ~4 KB
//
//   where ncoeffs = (12*bias_nx + 9*sigma_nx)  [ see interpolation.hpp for layout ]
//
// If (bias_nx,sigma_nx) = (128,64), then gmem/shmem footprint for coeffs is 2.25/8.25 KiB.
// This can be compared to the single-threadblock gmem footprint of the S_012 arrays:
// 12 KiB for pathfinder, 96 KiB for full CHORD. (General expr: Wt * Wf * S * 24 bytes)


__global__ void sk_kernel(
    float *out_sk_feed_averaged,         // Shape (T,F,3),
    float *out_sk_single_feed,           // Shape (T,F,3,S), can be NULL
    uint *out_rfimask,                   // Shape (F,(T*Nds)/32), can be NULL
    const ulong *in_S012,                // Shape (T,F,3,S)
    const uint *in_bf_mask,              // Shape (S/4,)
    const float *gmem_bsigma_coeffs,     // Shape (4*bias_nx + sigma_nx,)
    uint rfimask_fstride,                // Only used if (out_rfimask != NULL). NOTE: uint32 stride, not bit stride!
    float sk_rfimask_sigmas,             // RFI masking threshold in "sigmas" (only used if out_rfimask != NULL)
    float single_feed_min_good_frac,     // For single-feed SK-statistic (threshold for validity)
    float feed_averaged_min_good_frac,   // For feed-averaged SK-statistic (threshold for validity)
    float mu_min,                        // For single-feed SK-statistic (threshold for validity)
    float mu_max,                        // For single-feed SK-statistic (threshold for validity)
    int Nds,                             // Downsampling factor used to construct S012 array (before sk_kernel() was called)
    int T,                               // Number of time samples in S012 array (after downsampling by Tds)
    int F,                               // Number of frequency channels
    int S)                               // Number of stations (= 2*dishes)
{
    // Parallelization.
    const uint Wt = blockDim.z;
    const uint Wf = blockDim.y;
    const uint Ws = blockDim.x >> 5;
    
    const uint warpId = (threadIdx.z * Wf * Ws) + (threadIdx.y * Ws) + (threadIdx.x >> 5);
    const uint laneId = (threadIdx.x & 31);
    
    // Shared memory layout.
    extern __shared__ uint shmem_base[];
    const uint Nbf = max(S,1024) >> 5;   // see include/n2k/bad_feed_mask.hpp
    
    uint *shmem_bf = shmem_base;
    float *shmem_red = (float *) (shmem_bf + bf_mask_shmem_nelts(S));
    uint *shmem_rfi = (uint *) (shmem_red + 4*Ws*Wf*Wt);
    float *shmem_bsigma_coeffs = (float *) (shmem_rfi + 32);

    // Copy interpolation coeffs from GPU global -> shared memory.
    // Note: unpack_bias_sigma_coeffs() is defined in include/n2k/interpolation.hpp
    // Note: unpack_bias_sigma_coeffs() calls __syncthreads().
    unpack_bias_sigma_coeffs(gmem_bsigma_coeffs, shmem_bsigma_coeffs);
    
    // Part 1:
    //   - Compute single-feed SK statistics.
    //   - Write out_sk_single_feed (if non-NULL).
    //   - Accumulate single-thread contribution to sum_{w,wsk,wb,wsigma2}.

    // in_ix = per-warp offset in 'in_S012' and 'out_sk_single_feed' arrays,
    // accounting for time/freq but not stations. Both shapes are (T,F,3,S).
    // After shifting by 'in_ix', shapes can be viewed as (3,S).

    int t0 = (blockIdx.z * Wt) + threadIdx.z;
    int f0 = (blockIdx.y * Wf) + threadIdx.y;
    int t1 = (t0 < T) ? t0 : (T-1);
    int f1 = (f0 < F) ? f0 : (F-1);
    
    int in_ix = (t1*F + f1) * (3*S);   // note (t1,f1) not (t0,f0)
    bool write_sf = (out_sk_single_feed != NULL) && (t0 == t1) && (f0 == f1);
    
    in_S012 += in_ix;
    out_sk_single_feed = write_sf ? (out_sk_single_feed + in_ix) : NULL;

    // Reminder: load_bad_feed_mask() (defined in include/n2k/bad_feed_mask.hpp)
    // reads the bad feed mask, and permutes bits so that 'bf' contains relevant
    // bits for this thread.
    
    uint bf = load_bad_feed_mask(in_bf_mask, shmem_bf, S);
    // No __syncthreads() needed here, since 'shmem_bf' is not re-used below.

    // Accumulators (for accumulating single-feed SK into feed-averaged SK)
    float sum_w = 0;
    float sum_wsk = 0;
    float sum_wb = 0;
    float sum_wsigma2 = 0;

    // Loop over stations.
    for (int s = threadIdx.x; s < S; s += blockDim.x) {
	float S0 = in_S012[s];        // ulong -> float
	float S1 = in_S012[s+S];      // ulong -> float
	float S2 = in_S012[s+2*S];    // ulong -> float

	float S0_min = Nds * single_feed_min_good_frac - 0.1f;
	bool sf_valid = (S0 >= S0_min) && (S1 >= mu_min*S0) && (S1 <= mu_max*S0);

	S0 = sf_valid ? S0 : 2.0f;    // If invalid, set to 2.0 to avoid dividing by zero below
	S1 = sf_valid ? S1 : 1.0f;    // If invalid, set to 1.0 to avoid dividing by zero below

	// Interpolation uses variables (x,y).
	float y = 1.0f / S0;      // Note: S0 > 0 (even if !sf_valid)
	float mu = S1 * y;        // Always > 0 (even if !sf_valid)
	float x = logf(mu);       // Note: mu > 0 (even if !sf_valid)

	// Clip x
	constexpr float xmin = sk_globals::xmin;
	constexpr float xmax = sk_globals::xmax;
	x = (x >= xmin) ? x : xmin;
	x = (x <= xmax) ? x : xmax;
	
	// Single-feed (SK, b, sigma).
	// Note: interpolate_bias_gpu() and interpolate_sigma_gpu() are defined in interpolation.hpp
	// FIXME consider making computation of 'sk' more numerically stable.
	
	float b = interpolate_bias_gpu(shmem_bsigma_coeffs, x, y);
	float sigma = interpolate_sigma_gpu(shmem_bsigma_coeffs, x) * sqrtf(y);  // note sqrt(y) = N^(-1/2) here
	float sk = (S0+1)/(S0-1) * (S0*S2/(S1*S1) - 1) - b;
	float sigma2 = sigma * sigma;

	// Write single-feed SK statistics (sk,b,sigma) to 'out_sk_single_feed'.
	// Note: invalid entry is reprsented by (sk,b,sigma) = (0,0,-1).

	if (write_sf) {
	    out_sk_single_feed[s] = sf_valid ? sk : 0.0f;
	    out_sk_single_feed[s+S] = sf_valid ? b : 0.0f;
	    out_sk_single_feed[s+2*S] = sf_valid ? sigma : -1.0f;
	}

	// Accumulate single-feed contribution to feed-averaged SK statistics.
	bool feed_is_good = (bf & 1);
	float w = (sf_valid && feed_is_good) ? S0 : 0.0f;
	
	sum_w += w;
	sum_wsk += w * sk;
	sum_wb += w * b;
	sum_wsigma2 += w*w*sigma2;
	
	// Advance bad feed mask (see bad_feed_mask.hpp)
	bf >>= 1;
    }

    // Part 2: partially reduce sum_{w,wsk,wb,wsigma2} and write to shared memory.
    // ("Partially reduce" means that we have reduced over all lanes in each warp,
    //  but we still need to do a factor-Ws reduction over "x-warps".)

    float x = reduce_four(sum_w, sum_wsk, sum_wb, sum_wsigma2);

    // At this point,
    //   x = (partially reduced sum_w)         if laneId % 4 == 0
    //       (partially reduced sum_wsk)       if laneId % 4 == 1
    //       (partially reduced sum_wb)        if laneId % 4 == 2
    //       (partially reduced sum_wsigma2)   if laneId % 4 == 3
    //
    // Next step is to write x to shared memory ('shmem_red' with shape (Ws,Wt,Wf,4)).

    const uint wt = threadIdx.z;
    const uint wf = threadIdx.y;
    const uint ws = threadIdx.x >> 5;
    const uint s = (ws*Wt*Wf*4) + (wt*Wf*4) + (wf*4) + laneId;

    if (laneId < 4)
	shmem_red[s] = x;

    __syncthreads();

    // Part 3: this part runs on one warp!
    //  - read sum_{w,wsk,wb,wsigma2} from shared memory and fully reduce
    //  - write feed-averaged sk
    //  - write compressed bitmask (Wt*Wf) bits to shared memory.
    //
    // Current code assumes Wt*Wf <= 8, and (Wt*Wf) is a power of two (could be relaxed).
    // FIXME could shave off a few clock cycles in this part.

    if (warpId == 0) {
	// shmem_red has shape (Ws,Wt,Wf,4).
	const int nred = 4*Wt*Wf;
	const int s = laneId & (nred-1);

	float y = 0.0f;
	for (int i = 0; i < Ws; i++)
	    y += shmem_red[s + i*nred];

	// At this point,
	//   y = sum_e N_e               if (laneId < nred) and (laneId % 4 == 0)
	//       sum_e N_e SK_e          if (laneId < nred) and (laneId % 4 == 1)
	//       sum_e N_e b_e           if (laneId < nred) and (laneId % 4 == 2)
	//       sum_e N_e^2 sigma_e^2   if (laneId < nred) and (laneId % 4 == 3)
	//       junk                    if (laneId >= nred)
	
	// Replace sigma^2 -> sigma2 (only for laneId % 4 == 3)
	bool is_sigma2 = (laneId < nred) && ((laneId & 3) == 3);
	float z = sqrtf(is_sigma2 ? y : 0.0f);
	y = is_sigma2 ? z : y;

	// At this point,
	//   y = sum_e N_e                      if (laneId < nred) and (laneId % 4 == 0)
	//       sum_e N_e SK_e                 if (laneId < nred) and (laneId % 4 == 1)
	//       sum_e N_e b_e                  if (laneId < nred) and (laneId % 4 == 2)
	//       sqrt[ sum_e N_e^2 sigma_e^2 ]  if (laneId < nred) and (laneId % 4 == 3)
	//       junk                           if (laneId >= nred)

	// Broadcast (sum_e N_e) from (laneId % 4 == 0) to all lanes.
	float den = __shfl_sync(FULL_MASK, y, laneId & ~3);
	float S0_min = S * Nds * feed_averaged_min_good_frac - 0.1f;
	bool fsum_valid = (den >= S0_min);
	den = fsum_valid ? den : 1.0f;
	y = fsum_valid ? (y/den) : (is_sigma2 ? -1.0f : 0.0f);

	// At this point,
	//   y = junk                                  if (laneId < nred) and (laneId % 4 == 0)
	//       (\tilde SK if fsum_valid else 0)      if (laneId < nred) and (laneId % 4 == 1)
	//       (\tilde b if fsum_valid else 0)       if (laneId < nred) and (laneId % 4 == 2)
	//       (\tilde sigma if fsum_valid else -1)  if (laneId < nred) and (laneId % 4 == 3)
	//       junk                                  if (laneId >= nred)

	// Write feed-averaged SK-statistic to global memory ('out_sk_feed_averaged').
	// Note output array shape = (T,F,3).
	// FIXME could shave off a few clock cycles here.

	uint tf = laneId >> 2;
	uint t = tf / Wf;
	uint f = tf - t*Wf;
	int m = (laneId & 3) - 1;

	t += (blockIdx.z * Wt);
	f += (blockIdx.y * Wf);
	int out_ix = t*(3*F) + (3*f) + m;
	bool write_fsum = (t < T) && (f < F) && (laneId < nred) && (laneId & 3);

	if (write_fsum)
	    out_sk_feed_averaged[out_ix] = y;

	// Now compute RFI mask.
	
	// First we need to get (SK, sigma) onto the same thread.
	// We do this by sending sigma from (laneId % 4 == 3) -> (all lanes).
	float sigma = __shfl_sync(FULL_MASK, y, laneId | 3);

	// Only meaningful if (laneId % 4 == 1).
	bool rfi_good = fsum_valid
	    && (y >= 1.0f - sk_rfimask_sigmas * sigma)
	    && (y <= 1.0f + sk_rfimask_sigmas * sigma);

	// The value of 'rfimask' is meaningful on all lanes, but only
	// bits with (bit index % 4 == 1) in 'rfimask' are meaningful.
	uint rfimask = __ballot_sync(FULL_MASK, rfi_good);
	rfimask &= 0x22222222;  // clear meaningless bits to avoid confusion
	
	// Write bitmask to shared memory (subsequent code must "remember" that
	// the mask is in bit positions % 4 == 1).
	shmem_rfi[laneId] = rfimask;
    }

    if (out_rfimask == NULL)
	return;
    
    __syncthreads();

    // Part 4: Write bitmask to global memory.
    
    // Logical loop over 3-d shape (Nf,Nt,M)
    const int tb = blockIdx.z * Wt;
    const int fb = blockIdx.y * Wf;
    const int Nt = min(Wt, T-tb);
    const int Nf = min(Wf, F-fb);
    const int M = Nds >> 5;      // number of 32-bit outputs per coarse (t,f).
    
    // Loop is actually organized as loop over index 0 <= n < N,
    // where n = f*Nt*M + t*M + m, and N = Nf*Nt*M.
    
    const int threadId = 32*warpId + laneId;
    const int nthreads = 32 * Wt * Wf * Ws;
    const int N = Nf * Nt * M;

    // This way of writing the loop assumes that in the typical case,
    // each thread writes to global memory 0 or 1 times.
    
    for (uint n = threadId; n < N; n += nthreads) {
	// Unpack "flattened" index n to 3-d index (f,t,m).
	uint ft = n / M;
	uint m = n - ft*M;
	uint f = ft / Nt;
	uint t = ft - f*Nt;

	// RFI mask is in (bit positions % 4 == 1), see above.
	int bit = 4 * (t*Wf + f) + 1;
	uint val = (shmem_rfi[laneId] & (1 << bit)) ? 0xffffffffU : 0;

	// Index in out_rfimask (shape (F,T*M) with stride rfimask_fstride).
	int out_ix = (f+fb)*rfimask_fstride + (t+tb)*M + m;
	out_rfimask[out_ix] = val;
    }
}


// -------------------------------------------------------------------------------------------------


SkKernel::SkKernel(const SkKernel::Params &params_, bool check_params)
{
    constexpr int ncoeffs = (4 * sk_globals::bias_nx) + sk_globals::sigma_nx;

    if (check_params)
	SkKernel::check_params(params_);
    
    this->params = params_;
    this->bsigma_coeffs = Array<float> ({ncoeffs}, af_rhost);   // on CPU

    const double *src = sk_globals::get_bsigma_coeffs();
    float *dst = bsigma_coeffs.data;

    for (int i = 0; i < ncoeffs; i++)
	dst[i] = src[i];  // double -> float
    
    this->bsigma_coeffs = bsigma_coeffs.to_gpu();
    CUDA_CALL(cudaGetDevice(&this->device));
}


// Static member function
void SkKernel::check_params(const SkKernel::Params &params)
{
    double single_feed_min_good_frac = params.single_feed_min_good_frac;
    double feed_averaged_min_good_frac = params.feed_averaged_min_good_frac;
    double mu_min = params.mu_min;
    double mu_max = params.mu_max;
    long Nds = params.Nds;
    
    if (Nds <= 0)
	throw runtime_error("SkKernel::Params::Nds was uninitialized or <= 0");
    if ((Nds % 32) != 0)
	throw runtime_error("SkKernel::Params::Nds must be a multiple of 32");

    // Some trivial constraints on thresholds.
    
    if (single_feed_min_good_frac <= 0.0)
	throw runtime_error("SkKernel::Params::single_feed_min_good_frac was uninitialized or <= 0");
    if (feed_averaged_min_good_frac <= 0.0)
	throw runtime_error("SkKernel::Params::feed_averaged_min_good_frac was uninitialized or <= 0");
    if (mu_min <= 0.0)
	throw runtime_error("SkKernel::Params::mu_min was uninitialized or <= 0");
    if (mu_max >= 98.0)
	throw runtime_error("SkKernel::Params::mu_max was uninitialized or <= 0");
    if (mu_min >= mu_max)
	throw runtime_error("SkKernel: expected Params::mu_min < Params::mu_max");
    
    // Some nontrivial constraints deriving from n2k::sk_globals.
    
    constexpr int bias_nmin = n2k::sk_globals::bias_nmin;
    double x = double(bias_nmin) / double(Nds);

    if (Nds < bias_nmin) {
	stringstream ss;
	ss << "SkKernel::Params::Nds=" << Nds
	   << " was specified, and min allowed value is " << bias_nmin
	   << " (= n2k::sk_globals::bias_nmin), since the SK-interpolation table has not been"
	   << " validated for smaller S0-values. This could be improved with some effort.";
	throw runtime_error(ss.str());
    }

    if (single_feed_min_good_frac < x-0.01) {
	stringstream ss;
	ss << "SkKernel::Params::single_feed_min_good_frac=" << single_feed_min_good_frac
	   << " was specified, and min allowed value (for Nds=" << Nds << ") is " << x
	   << " (= n2k::sk_globals::bias_nmin / Nds), since the SK-interpolation table has not"
	   << " been validated for smaller S0-values. This could be improved with some effort.";
	throw runtime_error(ss.str());
    }

    if (mu_min < n2k::sk_globals::mu_min-0.01) {
	stringstream ss;
	ss << "SkKernel::Params::mu_min=" << mu_min
	   << " was specified, and min allowed value is " << mu_min
	   << " (= n2k::sk_globals::mu_min), since the SK-interpolation table has not"
	   << " been validated for smaller mu-values. This could be improved with some effort.";
	throw runtime_error(ss.str());
    }

    if (mu_max > n2k::sk_globals::mu_max+0.01) {
	stringstream ss;
	ss << "SkKernel::Params::mu_max=" << mu_max
	   << " was specified, and max allowed value is " << mu_max
	   << " (= n2k::sk_globals::mu_max), since the SK-interpolation table has not"
	   << " been validated for larger mu-values. This could be improved with some effort.";
	throw runtime_error(ss.str());
    }

    // We don't error-check 'params.sk_rfimask_sigmas' here (check is deferred to launch()).
    // This is because params.sk_rfimask_sigmas is only used if 'out_rfimask' is non-NULL
    // in launch().
}


void SkKernel::launch(
    float *out_sk_feed_averaged,          // Shape (T,F,3)
    float *out_sk_single_feed,            // Shape (T,F,3,S), can be NULL
    uint *out_rfimask,                    // Shape (F,T*Nds/32), can be NULL
    const ulong *in_S012,                 // Shape (T,F,3,S)
    const uint8_t *in_bf_mask,            // Length S (bad feed mask)
    long rfimask_fstride,                 // Only used if (out_rfimask != NULL). NOTE: uint32 stride, not bit stride!
    long T,                               // Number of downsampled times in S012 array
    long F,                               // Number of frequency channels
    long S,                               // Number of stations (= 2 * dishes)
    cudaStream_t stream,
    bool check_params) const
{
    if (check_params)
	SkKernel::check_params(this->params);

    int dev = -2;
    CUDA_CALL(cudaGetDevice(&dev));

    if (dev != this->device)
	throw runtime_error("SkKernel::launch: current CUDA device doesn't match current device when SkKernel constructor was called");
    
    // Check for NULL pointers.
    
    if (!out_sk_feed_averaged)
	throw runtime_error("SkKernel::launch: 'out_sk_feed_averaged' must be non-NULL");
    if (!in_S012)
	throw runtime_error("SkKernel::launch: 'in_S012' must be non-NULL");
    if (!in_bf_mask)
	throw runtime_error("SkKernel::launch: 'in_bf_mask' must be non-NULL");

    // Check integer arguments.
    
    if (T <= 0)
	throw runtime_error("SkKernel::launch: expected T > 0");
    if (F <= 0)
	throw runtime_error("SkKernel::launch: expected F > 0");
    if ((S <= 0) || (S > 8192))
	throw runtime_error("SkKernel::launch: expected 0 < S <= 8192");
    if (3*T*F*S >= INT_MAX)
	throw runtime_error("SkKernel::launch: product T*F*S is too large (32-bit overflow)");
    
    // This constraint comes from load_bad_feed_mask(), and could be relaxed if necessary.
    
    if ((S % 128) != 0)
	throw runtime_error("SkKernel::launch: expected S to be a multiple of 128.");

    // If an RFI bitmask is being computed, check 'rfimask_fstride' and 'sk_rfimask_sigmas' arguments,.
    
    if (out_rfimask != NULL) {
	if (std::abs(rfimask_fstride) < ((T * params.Nds) / 32))
	    throw runtime_error("SkKernel::launch: rfimask_fstride is too small");
	if (F * std::abs(rfimask_fstride) >= INT_MAX)
	    throw runtime_error("SkKernel::launch: product F*rfimask_fstride is too large (32-bit overflow)");
	if (params.sk_rfimask_sigmas <= 0.0)
	    throw runtime_error("SkKernel::launch: expected sk_rfimask_sigmas > 0.0");
    }

    // Assign blockDims.
    // Not much thought put into this!
    // Note that Wt*Wf must be a power of two, and <= 8. (See "Part 3" of kernel above.)
    uint Ws = (S+1023)/1024;  // min value allowed by load_bad_feed_mask()
    uint Wt = 2;
    uint Wf = 2;

    // Assign gridDims.
    uint Bt = (T+Wt-1) / Wt;
    uint Bf = (F+Wf-1) / Wf;

    // Shared memory size.
    constexpr int bnx = sk_globals::bias_nx;
    constexpr int snx = sk_globals::sigma_nx;
    uint shmem_nbytes = bf_mask_shmem_nbytes(S,Ws);
    shmem_nbytes += 16*Ws*Wt*Wf;   // 'shmem_red' has shape (Ws,Wt,Wf,4) and dtype float
    shmem_nbytes += 4*32;          // 'shmem_rfimask' has shape (32,) and dtype uint
    shmem_nbytes += 48*bnx + 36*snx;

    // Launch kernel!
    sk_kernel<<< {1,Bf,Bt}, {32*Ws,Wf,Wt}, shmem_nbytes, stream >>>
	(out_sk_feed_averaged,
	 out_sk_single_feed,
	 out_rfimask,
	 in_S012,
	 (const uint *) in_bf_mask,    // (const uint8_t *) -> (const uint *)
	 this->bsigma_coeffs.data,
	 rfimask_fstride,
	 params.sk_rfimask_sigmas,            // double -> float
	 params.single_feed_min_good_frac,    // double -> float
	 params.feed_averaged_min_good_frac,  // double -> float
	 params.mu_min,                       // double -> float
	 params.mu_max,                       // double -> float
	 params.Nds,                          // long -> int
	 T, F, S);                            // long -> int

    CUDA_PEEK("sk_kernel");
}


void SkKernel::launch(
    Array<float> &out_sk_feed_averaged,   // Shape (T,F,3)
    Array<float> &out_sk_single_feed,     // Either empty array or shape (T,F,3,S)
    Array<uint> &out_rfimask,             // Either empty array or shape (F,T*Nds/32), need not be contiguous
    const Array<ulong> &in_S012,          // Shape (T,F,3,S)
    const Array<uint8_t> &in_bf_mask,     // Length S (bad feed bask)
    cudaStream_t stream) const
{
    check_params(this->params);
    int Nds = this->params.Nds;
    
    // Check 'out_sk_feed_averaged', 'in_S012', 'in_bf_mask' args.
    
    if (out_sk_feed_averaged.ndim != 3)
	throw runtime_error("SkKernel::launch: expected out_sk_feed_averaged.ndim == 3");
    if (!out_sk_feed_averaged.is_fully_contiguous())
	throw runtime_error("SkKernel::launch: expected 'out_sk_feed_averaged' to be fully contiguous");
    if (!out_sk_feed_averaged.on_gpu())
	throw runtime_error("SkKernel::launch: expected 'out_sk_feed_averaged' to be in GPU memory");
    
    if (in_S012.ndim != 4)
	throw runtime_error("SkKernel::launch: expected in_S012.ndim == 4");
    if (!in_S012.is_fully_contiguous())
	throw runtime_error("SkKernel::launch: expected 'in_S012' to be fully contiguous");
    if (!in_S012.on_gpu())
	throw runtime_error("SkKernel::launch: expected 'out_sk_feed_averaged' to be in GPU memory");
    
    if (in_bf_mask.ndim != 1)
	throw runtime_error("SkKernel::launch: expected in_bf_mask.ndim == 1");    
    if (!in_bf_mask.is_fully_contiguous())
	throw runtime_error("SkKernel::launch: expected 'in_bf_mask' to be fully contiguous");
    if (!in_bf_mask.on_gpu())
	throw runtime_error("SkKernel::launch: expected 'out_sk_feed_averaged' to be in GPU memory");
    
    if (out_sk_feed_averaged.shape[0] != in_S012.shape[0])
	throw runtime_error("SkKernel::launch: inconsistent value of T between 'out_sk_feed_averaged' and 'in_S012' arrays");
    if (out_sk_feed_averaged.shape[1] != in_S012.shape[1])
	throw runtime_error("SkKernel::launch: inconsistent value of F between 'out_sk_feed_averaged' and 'in_S012' arrays");
    if (in_S012.shape[3] != in_bf_mask.shape[0])
	throw runtime_error("SkKernel::launch: inconsistent value of S between 'in_S012' and 'in_bf_mask' arrays");
    if (out_sk_feed_averaged.shape[2] != 3)
	throw runtime_error("SkKernel::launch: expected out_sk_feed_averaged.shape == (T,F,3)");
    if (in_S012.shape[2] != 3)
	throw runtime_error("SkKernel::launch: expected out_sk_feed_averaged.shape == (T,F,3)");
    
    long T = in_S012.shape[0];
    long F = in_S012.shape[1];
    long S = in_S012.shape[3];
    long rfimask_fstride = out_rfimask.data ? out_rfimask.strides[0] : 0;

    // Check 'out_sk_single_feed' and 'out_rfimask' arguments.

    if (out_sk_single_feed.data != NULL) {
	if (!out_sk_single_feed.shape_equals({T,F,3,S}))
	    throw runtime_error("SkKernel::launch: 'out_sk_single_feed' array has wrong shape (expected {T,F,3,S}");
	if (!out_sk_single_feed.is_fully_contiguous())
	    throw runtime_error("SkKernel::launch: expected 'out_sk_single_feed' array to be fully contiguous");
	if (!out_sk_single_feed.on_gpu())
	    throw runtime_error("SkKernel::launch: expected 'out_sk_single_feed' to be in GPU memory");
    }

    if (out_rfimask.data != NULL) {
	if (!out_rfimask.shape_equals({F,(T*Nds)/32}))
	    throw runtime_error("SkKernel::launch: 'out_rfimask' array has wrong shape (expected {F,(T*Nds)/32})");
	if (out_rfimask.strides[1] != 1)
	    throw runtime_error("SkKernel::launch: expected inner (time) axis of 'out_rfimask' array to be contiguous");
	if (!out_rfimask.on_gpu())
	    throw runtime_error("SkKernel::launch: expected 'out_rfimask' to be in GPU memory");
    }
    
    this->launch(
        out_sk_feed_averaged.data,
	out_sk_single_feed.data,
	out_rfimask.data,
	in_S012.data,
	in_bf_mask.data,
	rfimask_fstride,
	T, F, S,
	stream,
	false);   // check_params
}


}  // namespace n2k
