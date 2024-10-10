#ifndef _N2K_INTERPOLATION_HPP
#define _N2K_INTERPOLATION_HPP

#include <stdexcept>
#include "device_inlines.hpp"  // bank_conflict_free_load(), roll_forward(), roll_backward()
#include "sk_globals.hpp"

namespace n2k {
#if 0
}  // editor auto-indent
#endif

// This file contains inline functions (mostly __device__ inline) used internally.
// It probably won't be useful "externally" to n2k.


// -------------------------------------------------------------------------------------------------
//
// Host (and __host__ __device__) code
//
// Reminder: x=log(mu)=log(S1/S0), and y=(1/N)=(1/S0).


// t = (-1,0,1,2) returns (y0,y1,y2,y3) respectively.
// This function is tested in src_bin/test-helper-functions.cu.

template<typename T>
__host__ __device__ inline T cubic_interpolate(T t, T y0, T y1, T y2, T y3)
{
    constexpr T one_half = T(1) / T(2);
    constexpr T one_third = T(1) / T(3);
    
    T d01 = (t) * (y1 - y0);
    T d12 = (t-1) * (y2 - y1);

    T c12 = (t) * (y2 - y1);
    T c23 = (t-1) * (y3 - y2);
    
    T d012 = one_half * (t-1) * (c12 - d01);
    T c123 = one_half * (t) * (c23 - d12);
    
    T c0123 = one_third * (t+1) * (c123 - d012);
    return c0123 + d012 + c12 + y1;
}


__host__ inline void _set_up_cpu_interpolation(int &i, double &t, double x, int nx)
{
    constexpr double xmin = sk_globals::xmin;
    constexpr double xmax = sk_globals::xmax;
    constexpr double eps = 1.0e-6;

    t = (x-xmin) / (xmax-xmin) * (nx-1);

    if (t < 1-eps)
	throw std::runtime_error("interpolate_{bias,sigma}_cpu: value of x=log(mu) is too small");
    if (t > nx-2+eps)
	throw std::runtime_error("interpolate_{bias,sigma}_cpu: value of x=log(mu) is too large");

    i = int(t)-1;
    i = std::max(i,0);
    i = std::min(i,nx-4);
    
    t -= (i+1);
    assert(t >= -2*eps);
    assert(t <= 1+2*eps);
}


__host__ inline double interpolate_bias_cpu(double x, double y)
{
    constexpr double ymax = 1.0 / double(sk_globals::bias_nmin);
    const double *btab = sk_globals::get_bsigma_coeffs();

    if (y < 0.0)
	throw std::runtime_error("interpolate_bias_cpu: value of y=1/N is negative");
    if (y > ymax + 1.0e-6)
	throw std::runtime_error("interpolate_bias_cpu: value of y=1/N is too large");
    
    int i;
    double t;
    _set_up_cpu_interpolation(i, t, x, sk_globals::bias_nx);

    double c[4];
    for (int j = 0; j < 4; j++)
	c[j] = btab[4*(i+j)] + btab[4*(i+j)+1]*y + btab[4*(i+j)+2]*y*y + btab[4*(i+j)+3]*y*y*y;

    // Reminder: cubic_interpolate() is defined in include/n2k/interpolation.hpp.
    return cubic_interpolate(t, c[0], c[1], c[2], c[3]);
}


__host__ inline double interpolate_sigma_cpu(double x)
{
    const double *stab = sk_globals::get_bsigma_coeffs() + (4 * sk_globals::bias_nx);
    
    int i;
    double t;
    _set_up_cpu_interpolation(i, t, x, sk_globals::sigma_nx);

    // Reminder: cubic_interpolate() is defined in include/n2k/interpolation.hpp.
    return cubic_interpolate(t, stab[i], stab[i+1], stab[i+2], stab[i+3]);
}


// -------------------------------------------------------------------------------------------------
//
// __device__ inlines


// load_sigma_coeffs(sigma_coeffs, i, c0, c1, c2 ,c3)
//
// 'sigma_coeffs' points to an array of shape (N,8).
// The length-8 axis is a spectator, i.e. all 8 values are equal.
//
// This function is equivalent to:
//   c0 = sigma_coeffs[8*i];
//   c1 = sigma_coeffs[8*i+8];
//   c2 = sigma_coeffs[8*i+16];
//   c3 = sigma_coeffs[8*i+24];
//
// but is guaranteed bank conflict free. This function is tested in
// src_bin/test-helper-functions.cu.


template<bool Debug = false>
__device__ inline void load_sigma_coeffs(const float *sigma_coeffs, int i, float &c0, float &c1, float &c2, float &c3)
{
    int t = threadIdx.x;
    c0 = bank_conflict_free_load<Debug> (sigma_coeffs + ((8*i + t + 24) & ~31) - t + 7);
    c1 = bank_conflict_free_load<Debug> (sigma_coeffs + ((8*i + t + 16) & ~31) - t + 15);
    c2 = bank_conflict_free_load<Debug> (sigma_coeffs + ((8*i + t + 8) & ~31) - t + 23);
    c3 = bank_conflict_free_load<Debug> (sigma_coeffs + ((8*i + t ) & ~31) - t + 31);
    roll_forward(i + (t >> 3), c0, c1, c2, c3);
}


// load_bias_coeffs(bias_coeffs, i, y, c0, c1, c2 ,c3)
//
// 'bias_coeffs' points to an array of shape (N,4,2).
// The length-2 axis is a spectator, i.e. all 2 values are equal.
//
// This function is equivalent to:
//   c0 = sum(bias_coeffs[8*i+2*j] * y^j for j in [0,1,2,3]);
//   c1 = sum(bias_coeffs[8*i+2*j+8] * y^j for j in [0,1,2,3]);
//   c2 = sum(bias_coeffs[8*i+2*j+16] * y^j for j in [0,1,2,3]);
//   c3 = sum(bias_coeffs[8*i+2*j+24] * y^j for j in [0,1,2,3]);
//
// but is guaranteed bank conflict free. This function is tested in
// src_bin/test-helper-functions.cu.


template<bool Debug = false>
__device__ inline float load_bias_inner(const float *coeffs, float y0, float y1, float y2, float y3)
{
    int t = threadIdx.x;
    float c0 = bank_conflict_free_load<Debug> (coeffs + ((t+6) & ~7) + 1-t);
    float c1 = bank_conflict_free_load<Debug> (coeffs + ((t+4) & ~7) + 3-t);
    float c2 = bank_conflict_free_load<Debug> (coeffs + ((t+2) & ~7) + 5-t);
    float c3 = bank_conflict_free_load<Debug> (coeffs + (t & ~7) + 7-t);
    return c0*y0 + c1*y1 + c2*y2 + c3*y3;
}


template<bool Debug = false>
__device__ inline void load_bias_coeffs(const float *bias_coeffs, int i, float y, float &c0, float &c1, float &c2, float &c3)
{
    float y0 = 1.0f;
    float y1 = y;
    float y2 = y*y;
    float y3 = y2*y;
    roll_backward(threadIdx.x >> 1, y0, y1, y2, y3);

    int s = (threadIdx.x >> 3) & 3;
    c0 = load_bias_inner<Debug> (bias_coeffs + 8 * (((i+s+3) & ~3) - s), y0, y1, y2, y3);
    c1 = load_bias_inner<Debug> (bias_coeffs + 8 * (((i+s+2) & ~3) + 1-s), y0, y1, y2, y3);
    c2 = load_bias_inner<Debug> (bias_coeffs + 8 * (((i+s+1) & ~3) + 2-s), y0, y1, y2, y3);
    c3 = load_bias_inner<Debug> (bias_coeffs + 8 * (((i+s) & ~3) + 3-s), y0, y1, y2, y3);
    roll_forward(i+s, c0, c1, c2, c3);
}


// unpack_bias_sigma_coeffs(): copies b/sigma coeffs from global -> shared memory.
//
// The 'gp' pointer should point to global GPU memory, laid out as follows:
//
//   float gmem_bias_coeffs[bias_nx][4];
//   float gmem_sigma_coeffs[sigma_nx];
//   // Total size: (4*bias_nx + sigma_nx) * sizeof(float)
//
// The 'sp' pointer should point to shared memory, laid out as follows:
//
//   float shmem_bias_coeffs[bias_nx][4][2];
//   float shmem_sigma_coeffs[sigma_nx][8];
//   float shmem_tmp_coeffs[4*bias_nx + sigma_nx];
//   // Total size: (12*bias_nx + 9*sigma_nx) * sizeof(float)
//
// If (bias_nx,sigma_nx) = (128,64), then gmem/shmem footprint is 8.25/2.25 KiB.


__device__ void unpack_bias_sigma_coeffs(const float *gp, float *sp)
{
    constexpr int nsh1 = (8 * sk_globals::bias_nx);
    constexpr int nsh = (8 * sk_globals::sigma_nx) + nsh1;
    constexpr int nglo = (4 * sk_globals::bias_nx) + sk_globals::sigma_nx;
    
    int nthreads = blockDim.z * blockDim.y * blockDim.x;
    int threadId = (threadIdx.z * blockDim.y) + threadIdx.y;
    threadId = (threadId * blockDim.x) + threadIdx.x;
    
    // Copy global memory (bsigma_coeffs) to shared memory (shmem_tmp_coeffs)
    for (int i = threadId; i < nglo; i += nthreads)
	sp[i+nsh] = gp[i];

    __syncthreads();

    // "Unpack" (shmem_tmp_coeffs) -> (shmem_bias_coeffs, shmem_sigma_coeffs)
    for (int i = threadId; i < nsh; i += nthreads) {
	int ilo = (i < nsh1) ? i : nsh1;
	int j = (ilo >> 1) + ((i-ilo) >> 3);
	sp[i] = sp[j+nsh];
    }
    
    __syncthreads();
}


__device__ inline float interpolate_bias_gpu(const float *shmem_bsigma_coeffs, float x, float y)
{
    constexpr int nx = sk_globals::bias_nx;   // note bias_nx (not sigma_nx) here
    constexpr float xmin = sk_globals::xmin;
    constexpr float xmax = sk_globals::xmax;
    constexpr float xscal = float(nx-1) / (xmax-xmin);

    x = xscal * (x - sk_globals::xmin);
    int ix = int(x);
    ix = (ix >= 1) ? ix : 1;
    ix = (ix <= nx-3) ? ix : (nx-3);

    float c0, c1, c2, c3;
    load_bias_coeffs(shmem_bsigma_coeffs, ix-1, y, c0, c1, c2, c3);   // note (ix-1) here
    
    return cubic_interpolate<float> (x-ix, c0, c1, c2, c3);   // note (x-ix) here
}


__device__ inline float interpolate_sigma_gpu(const float *shmem_bsigma_coeffs, float x)
{
    constexpr int nx = sk_globals::sigma_nx;   // note sigma_nx (not bias_nx) here
    constexpr float xmin = sk_globals::xmin;
    constexpr float xmax = sk_globals::xmax;
    constexpr float xscal = float(nx-1) / (xmax-xmin);
    constexpr int nb = 8 * sk_globals::bias_nx;
    
    x = xscal * (x - sk_globals::xmin);
    int ix = int(x);
    ix = (ix >= 1) ? ix : 1;
    ix = (ix <= nx-3) ? ix : (nx-3);

    float c0, c1, c2, c3;
    load_sigma_coeffs(shmem_bsigma_coeffs + nb, ix-1, c0, c1, c2, c3);     // note (ix-1) and (... + nb) here.
    
    return cubic_interpolate<float> (x-ix, c0, c1, c2, c3);   // note (x-ix) here
}

    
}  // namespace n2k

#endif // _N2K_INTERPOLATION_HPP
