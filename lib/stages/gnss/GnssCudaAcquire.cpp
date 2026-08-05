#include "GnssCudaAcquire.hpp"

#include "cudaGnssAcquireKernel.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cufft.h>
#include <numeric>
#include <stdexcept>
#include <string>

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("GnssCudaAcquire: ") + what + ": "
                                 + cudaGetErrorString(e));
}
void ckf(cufftResult e, const char* what) {
    if (e != CUFFT_SUCCESS)
        throw std::runtime_error(std::string("GnssCudaAcquire: ") + what + ": cuFFT error "
                                 + std::to_string((int)e));
}

} // namespace

struct GnssCudaAcquire::Impl {
    // device buffers
    float2* dSnap = nullptr;  // [M][n_chan_total]
    float2* dA = nullptr;     // [nc][Mp]   FFT(data), computed once per snapshot
    float2* dHead = nullptr;  // [nc][Mp]
    float2* dTail = nullptr;  // [nc][Mp]
    float2* dR = nullptr;     // [nc][Mp]   materialised replica, then its FFT in place
    float2* dX = nullptr;     // [nc][n_dop][Mp]  wiped product (channel-major: see the header)

    float* dSurf = nullptr;   // [n_dop][Mp][s_cols]
    float* dSgn0 = nullptr;   // [Mp]
    float* dSgn1 = nullptr;   // [Mp]
    int* dCov = nullptr;      // [nc]
    int* dBin = nullptr;      // [nc]
    int* dShift = nullptr;    // [max_dop]
    float2* dTw = nullptr;    // [s_cols/2]
    float2* dVal = nullptr;   // peak {max, sum}
    long long* dIdx = nullptr;
    void* dScratch = nullptr;

    cufftHandle planFwdData = 0; // batch nc, len Mp, contiguous
    cufftHandle planInv = 0;     // batch n_dop for ONE channel, strided out (see header)
    int plan_nd = 0;             // n_dop planInv was built for

    cudaStream_t stream = nullptr;
    bool folded = true;
    size_t bytes = 0;
};

GnssCudaAcquire::GnssCudaAcquire(int nc, int Mp, int M, int n_chan_total, int max_dop, int s_cols)
    : _p(new Impl), _nc(nc), _Mp(Mp), _M(M), _nchan(n_chan_total), _max_dop(max_dop),
      _s_cols(s_cols) {
    if (nc <= 0 || Mp <= 0 || M <= 0 || max_dop <= 0 || s_cols <= 0)
        throw std::runtime_error("GnssCudaAcquire: bad dimensions");
    // The folded aggregate needs a power-of-two fine axis (it is sph/fine_step, both powers of
    // two in every CHORD and airspy geometry). Refuse rather than approximate.
    if ((s_cols & (s_cols - 1)) != 0)
        throw std::runtime_error("GnssCudaAcquire: s_cols must be a power of two");

    auto alloc = [&](void** p, size_t n, const char* what) {
        ck(cudaMalloc(p, n), what);
        _p->bytes += n;
    };
    const size_t cm = (size_t)nc * Mp * sizeof(float2);
    alloc((void**)&_p->dSnap, (size_t)M * n_chan_total * sizeof(float2), "snap");
    alloc((void**)&_p->dA, cm, "A");
    alloc((void**)&_p->dHead, cm, "head");
    alloc((void**)&_p->dTail, cm, "tail");
    alloc((void**)&_p->dR, cm, "R");
    // X holds the wiped product and then, in place, the correlation P[nc][n_dop][Mp]. There is
    // deliberately no second P buffer: the inverse transform runs in place.
    alloc((void**)&_p->dX, (size_t)nc * max_dop * Mp * sizeof(float2), "X");
    alloc((void**)&_p->dSurf, (size_t)max_dop * Mp * s_cols * sizeof(float), "surf");
    alloc((void**)&_p->dSgn0, (size_t)Mp * sizeof(float), "sgn0");
    alloc((void**)&_p->dSgn1, (size_t)Mp * sizeof(float), "sgn1");
    alloc((void**)&_p->dCov, (size_t)nc * sizeof(int), "cov");
    alloc((void**)&_p->dBin, (size_t)nc * sizeof(int), "bin");
    alloc((void**)&_p->dShift, (size_t)max_dop * sizeof(int), "shift");
    alloc((void**)&_p->dTw, (size_t)(s_cols / 2) * sizeof(float2), "twiddle");
    alloc((void**)&_p->dVal, sizeof(float2), "val");
    alloc((void**)&_p->dIdx, sizeof(long long), "idx");
    alloc(&_p->dScratch, gnss_cuda::peak_scratch_bytes((long)max_dop * Mp * s_cols), "scratch");

    ck(cudaStreamCreate(&_p->stream), "stream");

    int n[] = {Mp};
    ckf(cufftPlanMany(&_p->planFwdData, 1, n, nullptr, 1, Mp, nullptr, 1, Mp, CUFFT_C2C, nc),
        "plan fwd data");
    ckf(cufftSetStream(_p->planFwdData, _p->stream), "stream fwd");
}

GnssCudaAcquire::~GnssCudaAcquire() {
    if (_p->planFwdData)
        cufftDestroy(_p->planFwdData);
    if (_p->planInv)
        cufftDestroy(_p->planInv);
    cudaFree(_p->dSnap);
    cudaFree(_p->dA);
    cudaFree(_p->dHead);
    cudaFree(_p->dTail);
    cudaFree(_p->dR);
    cudaFree(_p->dX);

    cudaFree(_p->dSurf);
    cudaFree(_p->dSgn0);
    cudaFree(_p->dSgn1);
    cudaFree(_p->dCov);
    cudaFree(_p->dBin);
    cudaFree(_p->dShift);
    cudaFree(_p->dTw);
    cudaFree(_p->dVal);
    cudaFree(_p->dIdx);
    cudaFree(_p->dScratch);
    if (_p->stream)
        cudaStreamDestroy(_p->stream);
    delete _p;
}

size_t GnssCudaAcquire::device_bytes() const {
    return _p->bytes;
}

void GnssCudaAcquire::set_channels(const std::vector<int>& local, const std::vector<int>& global,
                                   int sph, int fine_step) {
    if ((int)local.size() != _nc || (int)global.size() != _nc)
        throw std::runtime_error("GnssCudaAcquire::set_channels: size != nc");
    ck(cudaMemcpy(_p->dCov, local.data(), (size_t)_nc * sizeof(int), cudaMemcpyHostToDevice),
       "cov");

    // Fold bins. Mirrors gnss::aggregate_accumulate's periodicity argument: with g the gcd of
    // the channel differences, |D|^2 repeats with period sph/g, so s_stored = sph/gcd(g,sph);
    // and when s_cols*s_step == s_stored the ramp collapses to a length-s_cols DFT indexed by
    // (f_c - f_0)/(sph/s_stored) mod s_cols. gg divides every channel difference by
    // construction, so this stays exact integer arithmetic.
    int g = 0;
    for (int c = 1; c < _nc; ++c)
        g = std::gcd(g, std::abs(global[c] - global[0]));
    const int s_stored = sph / std::gcd(g, sph);
    if ((long)_s_cols * (long)fine_step != (long)s_stored)
        throw std::runtime_error("GnssCudaAcquire: s_cols*fine_step != s_stored -- the folded "
                                 "aggregate's precondition fails for this geometry");
    const int gg = sph / s_stored;
    std::vector<int> bin(_nc);
    for (int c = 0; c < _nc; ++c) {
        const long b = ((long)(global[c] - global[0]) / gg) % _s_cols;
        bin[c] = (int)((b % _s_cols + _s_cols) % _s_cols);
    }
    ck(cudaMemcpy(_p->dBin, bin.data(), (size_t)_nc * sizeof(int), cudaMemcpyHostToDevice), "bin");

    std::vector<float2> tw((size_t)_s_cols / 2);
    for (int j = 0; j < _s_cols / 2; ++j) {
        const double a = 2.0 * M_PI * j / _s_cols; // e^{+...}: INVERSE, matching the ramp sign
        tw[j] = make_float2((float)std::cos(a), (float)std::sin(a));
    }
    ck(cudaMemcpy(_p->dTw, tw.data(), tw.size() * sizeof(float2), cudaMemcpyHostToDevice), "tw");
}

bool GnssCudaAcquire::set_doppler_grid(const std::vector<double>& grid, double sample_rate,
                                       int sph) {
    const int nd = (int)grid.size();
    if (nd <= 0 || nd > _max_dop)
        return false;
    const double bin_hz = sample_rate / ((double)_Mp * (double)sph);
    if (!(bin_hz > 0.0))
        return false;
    std::vector<int> shift((size_t)nd);
    for (int d = 0; d < nd; ++d) {
        const double b = grid[d] / bin_hz;
        const double rb = std::round(b);
        // Same tolerance as channel_correlate_into: a grid that is not bin-aligned is a
        // DIFFERENT computation, not a slightly worse one, so this is a hard reject.
        if (std::fabs(b - rb) > 1e-6 * std::max(1.0, std::fabs(rb)))
            return false;
        shift[(size_t)d] = (int)((((long long)rb % _Mp) + _Mp) % _Mp);
    }
    ck(cudaMemcpy(_p->dShift, shift.data(), (size_t)nd * sizeof(int), cudaMemcpyHostToDevice),
       "shift");
    _nd = nd;

    // (Re)plan the inverse: batch over DOPPLER for a fixed channel, writing straight into the
    // [d][q][c] layout. onembed must describe the full output extent or cuFFT clamps the stride.
    if (_p->planInv && _p->plan_nd != nd) {
        cufftDestroy(_p->planInv);
        _p->planInv = 0;
    }
    if (!_p->planInv) {
        int n[] = {_Mp};
        // NATURAL contiguous layout, and the transform runs IN PLACE on X.
        //
        // The first version asked cuFFT for the aggregate's preferred [d][q][c] layout via
        // ostride = nc. That is expressible, and it cost 41 ms against a 2.5 ms aggregate: 8-byte
        // writes at nc*8 = 632-byte stride give every element its own sector. Writing
        // contiguously and letting launch_aggregate_cdq do the transpose through a shared-memory
        // tile moves it to the read side, where it is a pattern rather than 16x write
        // amplification. In-place also drops the separate 0.63 GB P buffer.
        //
        // (Also: a NULL inembed with a non-NULL onembed is not a valid mixture -- cuFFT follows
        // FFTW's rule that NULL inembed means "basic layout, ignore every other stride argument"
        // and returns CUFFT_INVALID_VALUE. That was the first failure here.)
        ckf(cufftPlanMany(&_p->planInv, 1, n, nullptr, 1, _Mp, nullptr, 1, _Mp, CUFFT_C2C, nd),
            "plan inv");
        ckf(cufftSetStream(_p->planInv, _p->stream), "stream inv");
        _p->plan_nd = nd;
    }
    return true;
}

void GnssCudaAcquire::set_snapshot(const std::complex<float>* snap) {
    ck(cudaMemcpyAsync(_p->dSnap, snap, (size_t)_M * _nchan * sizeof(float2),
                       cudaMemcpyHostToDevice, _p->stream),
       "snap upload");
    gnss_cuda::launch_gather_pad(_p->dSnap, _p->dCov, _p->dA, _nchan, _M, _Mp, _nc, _p->stream);
    // FFT(data): depends on neither PRN nor alignment, so it happens exactly once per snapshot.
    ckf(cufftExecC2C(_p->planFwdData, _p->dA, _p->dA, CUFFT_FORWARD), "fwd data");
}

void GnssCudaAcquire::set_replica(const std::vector<std::vector<std::complex<float>>>& head,
                                  const std::vector<std::vector<std::complex<float>>>& tail) {
    if ((int)head.size() != _nc || (int)tail.size() != _nc)
        throw std::runtime_error("GnssCudaAcquire::set_replica: size != nc");
    for (int c = 0; c < _nc; ++c) {
        if ((int)head[c].size() != _Mp || (int)tail[c].size() != _Mp)
            throw std::runtime_error("GnssCudaAcquire::set_replica: row length != Mp");
        ck(cudaMemcpyAsync(_p->dHead + (size_t)c * _Mp, head[c].data(),
                           (size_t)_Mp * sizeof(float2), cudaMemcpyHostToDevice, _p->stream),
           "head");
        ck(cudaMemcpyAsync(_p->dTail + (size_t)c * _Mp, tail[c].data(),
                           (size_t)_Mp * sizeof(float2), cudaMemcpyHostToDevice, _p->stream),
           "tail");
    }
}

void GnssCudaAcquire::clear_surface() {
    ck(cudaMemsetAsync(_p->dSurf, 0, (size_t)_nd * _Mp * _s_cols * sizeof(float), _p->stream),
       "clear surf");
}

void GnssCudaAcquire::accumulate(const float* sgn0, const float* sgn1) {
    if (_nd <= 0)
        throw std::runtime_error("GnssCudaAcquire::accumulate: no Doppler grid set");
    ck(cudaMemcpyAsync(_p->dSgn0, sgn0, (size_t)_Mp * sizeof(float), cudaMemcpyHostToDevice,
                       _p->stream),
       "sgn0");
    ck(cudaMemcpyAsync(_p->dSgn1, sgn1, (size_t)_Mp * sizeof(float), cudaMemcpyHostToDevice,
                       _p->stream),
       "sgn1");

    gnss_cuda::launch_materialise(_p->dHead, _p->dTail, _p->dSgn0, _p->dSgn1, _p->dR, _nc, _Mp,
                                  _p->stream);
    ckf(cufftExecC2C(_p->planFwdData, _p->dR, _p->dR, CUFFT_FORWARD), "fwd replica");

    // X = A(shifted by the Doppler bin) * conj(B), scaled by 1/Mp.
    gnss_cuda::launch_dopmul(_p->dA, _p->dR, _p->dShift, _p->dX, _nc, _nd, _Mp, _p->stream);

    // One inverse call per CHANNEL, each a batch over Doppler, in place and contiguous. X is
    // then P in the [nc][n_dop][Mp] layout the tiled aggregate reads.
    for (int c = 0; c < _nc; ++c) {
        float2* Xc = _p->dX + (size_t)c * _nd * _Mp;
        ckf(cufftExecC2C(_p->planInv, Xc, Xc, CUFFT_INVERSE), "inv correlate");
    }

    gnss_cuda::launch_aggregate_cdq(_p->dX, _p->dSurf, _p->dBin, _p->dTw, _nc, _nd, _Mp, _s_cols,
                                    _p->stream);
}

GnssCudaAcquire::Peak GnssCudaAcquire::peak() const {
    const long n = surface_cells();
    gnss_cuda::launch_peak(_p->dSurf, n, _p->dVal, _p->dIdx, _p->dScratch, _p->stream);
    float2 v;
    long long idx = -1;
    ck(cudaMemcpyAsync(&v, _p->dVal, sizeof(float2), cudaMemcpyDeviceToHost, _p->stream), "val");
    ck(cudaMemcpyAsync(&idx, _p->dIdx, sizeof(long long), cudaMemcpyDeviceToHost, _p->stream),
       "idx");
    ck(cudaStreamSynchronize(_p->stream), "sync peak");
    Peak pk{};
    pk.peak = v.x;
    pk.mean = (double)v.y / (double)n;
    pk.index = idx;
    pk.s = (int)(idx % _s_cols);
    pk.q = (int)((idx / _s_cols) % _Mp);
    pk.d = (int)(idx / ((long)_s_cols * _Mp));
    return pk;
}

void GnssCudaAcquire::download_surface(std::vector<double>& out) const {
    const long n = surface_cells();
    std::vector<float> h((size_t)n);
    ck(cudaStreamSynchronize(_p->stream), "sync dl");
    ck(cudaMemcpy(h.data(), _p->dSurf, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost),
       "surf dl");
    out.assign(h.begin(), h.end());
}
