#ifndef GNSS_CUDA_ACQUIRE_HPP
#define GNSS_CUDA_ACQUIRE_HPP

#include "gnssChannelizedAcquire.hpp" // for AcquisitionResult / AcquisitionSurface

#include <complex>
#include <vector>

struct float2;
struct cudaStream_t_dummy_;
typedef struct CUstream_st* cudaStream_t;

/**
 * Host-side driver for the device-resident GNSS acquisition search (A2 of
 * docs/gnss_gpu_search.md).
 *
 * Owns the cuFFT plans and every device buffer, so a search pass moves 2 MB of snapshot UP and a
 * handful of bytes of peak DOWN -- the 0.63 GB correlation and the 0.51 GB surface never cross
 * PCIe, and the host never scans them. That is the structural change; the 180x on the aggregate
 * (A1) only becomes real once the correlate joins it on the device, because shipping P over the
 * bus costs ~25 ms against the 505 ms the kernel saves.
 *
 * WHAT IT REPLACES: gnss::channelized_accumulate (per-channel correlate + cross-channel
 * aggregate) plus the host-side peak scan, for one (PRN, secondary-code alignment, window).
 *
 * WHAT IT DOES NOT DO: build replicas. The caller hands over `head`/`tail` tables from
 * gnss::ChannelizedReplicaBank exactly as GnssChannelizedSearch already precomputes them, and a
 * per-hop sign pair per alignment. Replica generation stays on the CPU because it is 0.002 s per
 * pass and porting it would mean porting the whole bank.
 *
 * THREE ORDERING CONSTRAINTS worth knowing before touching this:
 *
 *  - The inverse transform is batched over DOPPLER for a FIXED CHANNEL. That is the only
 *    ordering for which both cuFFT strides stay linear AND the output lands directly in the
 *    [d][q][c] layout the aggregate needs (ostride = nc, odist = Mp*nc). Batching over (d,c)
 *    instead forces either nd separate calls or a 1.26 GB transpose pass.
 *  - The Doppler grid MUST be bin-aligned (a whole number of Fs/(Mp*sph) = 62.5 Hz at CHORD).
 *    Then one forward transform of the data serves every trial via a cyclic spectrum shift,
 *    exactly. @ref set_doppler_grid returns false rather than silently doing something else.
 *  - FFT(data) depends on neither PRN nor alignment. @ref set_snapshot computes it once; the
 *    CPU path recomputes it per (PRN, alignment) -- 640 times per snapshot at CHORD scale.
 */
class GnssCudaAcquire {
public:
    /// @param nc            covering channels
    /// @param Mp            replica hop-period == correlation length (3125 at CHORD)
    /// @param M             data window in hops (<= Mp; the rest is zero-padded)
    /// @param n_chan_total  channels in the caller's snapshot row stride
    /// @param max_dop       largest Doppler grid this engine will be asked for (sizes buffers)
    /// @param s_cols        stored fine-lag columns (sph / fine_step)
    GnssCudaAcquire(int nc, int Mp, int M, int n_chan_total, int max_dop, int s_cols);
    ~GnssCudaAcquire();
    GnssCudaAcquire(const GnssCudaAcquire&) = delete;
    GnssCudaAcquire& operator=(const GnssCudaAcquire&) = delete;

    /// Covering set: LOCAL indices into the snapshot row, and the GLOBAL bin of each (which sets
    /// the cross-channel ramp). Both in the same order. Also computes the fold table.
    void set_channels(const std::vector<int>& local, const std::vector<int>& global, int sph,
                      int fine_step);

    /// Bin-aligned Doppler grid. Returns false if any entry is not a whole number of
    /// `sample_rate/(Mp*sph)` -- the caller must then fall back to the CPU path rather than
    /// accept a grid this engine cannot represent exactly.
    bool set_doppler_grid(const std::vector<double>& grid, double sample_rate, int sph);

    /// Upload one snapshot window and transform it. [M][n_chan_total], as the stage holds it.
    void set_snapshot(const std::complex<float>* snap);

    /// Upload one PRN's replica tables, [nc][Mp] each, in covering order.
    void set_replica(const std::vector<std::vector<std::complex<float>>>& head,
                     const std::vector<std::vector<std::complex<float>>>& tail);

    /// Zero the surface (start of a new alignment).
    void clear_surface();

    /// Correlate + aggregate one window of the currently-loaded replica into the surface,
    /// for one secondary-code alignment. `sgn0`/`sgn1` are the hoisted per-hop sign pair, [Mp].
    void accumulate(const float* sgn0, const float* sgn1);

    /// Peak-pick the accumulated surface exactly as the CPU path does, reducing on the DEVICE.
    ///
    /// Reuses gnss::peak_from_reduction for every bit of the cell -> (code phase, Doppler)
    /// arithmetic, including the sub-grid parabolic Doppler refine; only the reduction itself is
    /// GPU work. `dims` must describe this engine's surface, and `doppler_grid` be the grid
    /// handed to @ref set_doppler_grid.
    gnss::AcquisitionResult peak_result(const gnss::AcquisitionSurface& dims,
                                        const std::vector<double>& doppler_grid, double sample_rate,
                                        double chip_rate, long code_length) const;

    /// Peak / mean of the accumulated surface, computed on the device.
    struct Peak {
        double peak;
        double mean;
        long long index; ///< flat index into [n_dop][Mp][s_cols]
        int d, q, s;     ///< decomposed
    };
    Peak peak() const;

    /// Copy the surface back. DIAGNOSTIC ONLY -- this is the 0.51 GB transfer the whole design
    /// exists to avoid. Used by scripts/gnss/acqbench to compare against the CPU reference.
    void download_surface(std::vector<double>& out) const;

    int n_dop() const { return _nd; }
    int s_cols() const { return _s_cols; }
    long surface_cells() const { return (long)_nd * _Mp * _s_cols; }
    /// Bytes of device memory held, so a caller can budget several engines per GPU.
    size_t device_bytes() const;

private:
    struct Impl;
    Impl* _p;
    int _nc, _Mp, _M, _nchan, _max_dop, _s_cols, _nd = 0;
};

#endif // GNSS_CUDA_ACQUIRE_HPP
