#ifndef GNSS_CUDA_ACQUIRE_HPP
#define GNSS_CUDA_ACQUIRE_HPP

#include "gnssChannelizedAcquire.hpp" // for AcquisitionResult / AcquisitionSurface

#include <complex>
#include <vector>

struct float2;
struct cudaStream_t_dummy_;
typedef struct CUstream_st* cudaStream_t;

/**
 * @class GnssCudaAcquire
 * @brief Host driver for the device-resident GNSS acquisition search.
 *
 * Does the work of gnss::channelized_accumulate (per-channel correlate and cross-channel
 * aggregate) and the host-side peak scan, for one (PRN, secondary-code alignment, window) at a
 * time. It owns the cuFFT plans and every device buffer, so a pass uploads the snapshot window
 * and downloads only the peak reduction: the correlation and the surface (~0.6 GB and ~0.5 GB
 * at CHORD's blind-search dimensions) never cross PCIe.
 *
 * It does not build replicas. The caller uploads the per-channel `head`/`tail` tables that
 * GnssChannelizedSearch precomputes from gnss::ChannelizedReplicaBank, and a per-hop sign pair
 * per alignment.
 *
 * Ordering constraints:
 *  - The inverse transform is batched over Doppler for one channel at a time, in place and
 *    contiguous, so the correlation is laid out [nc][n_dop][Mp]. The aggregate
 *    (gnss_cuda::launch_aggregate_cdq) transposes it through shared memory as it reads;
 *    writing the [d][q][c] layout from cuFFT instead (ostride = nc) scatters every write.
 *  - The Doppler grid must be bin-aligned (whole multiples of Fs/(Mp*sph), 62.5 Hz at CHORD).
 *    One forward transform of the data then serves every trial through an exact cyclic
 *    spectrum shift. @ref set_doppler_grid returns false for any other grid.
 *  - FFT(data) depends on neither PRN nor alignment: @ref set_snapshot transforms one window,
 *    and @ref accumulate can run any number of replicas and alignments against it.
 */
class GnssCudaAcquire {
public:
    /// @param nc            covering channels
    /// @param Mp            replica hop-period == correlation length (3125 at CHORD)
    /// @param M             data window in hops (<= Mp; the rest is zero-padded)
    /// @param n_chan_total  channels in the caller's snapshot row stride
    /// @param max_dop       largest Doppler grid this engine will be asked for (sizes buffers)
    /// @param s_cols        stored fine-lag columns, s_stored / fine_step (checked by set_channels)
    GnssCudaAcquire(int nc, int Mp, int M, int n_chan_total, int max_dop, int s_cols);
    ~GnssCudaAcquire();
    GnssCudaAcquire(const GnssCudaAcquire&) = delete;
    GnssCudaAcquire& operator=(const GnssCudaAcquire&) = delete;

    /// Covering set: local indices into the snapshot row, and the global bin of each (which sets
    /// the cross-channel ramp), in the same order. Also builds the fold table; throws unless
    /// s_cols * fine_step equals the geometry's s_stored (the folded aggregate's precondition).
    void set_channels(const std::vector<int>& local, const std::vector<int>& global, int sph,
                      int fine_step);

    /// Bin-aligned Doppler grid. Returns false if the grid is empty, longer than max_dop, or has
    /// an entry that is not a whole multiple of `sample_rate/(Mp*sph)`; the caller then uses the
    /// CPU path.
    bool set_doppler_grid(const std::vector<double>& grid, double sample_rate, int sph);

    /// Upload one snapshot window, [M][n_chan_total] as the stage holds it, and transform it.
    void set_snapshot(const std::complex<float>* snap);

    /// Upload one PRN's replica tables, [nc][Mp] each, in covering order.
    void set_replica(const std::vector<std::vector<std::complex<float>>>& head,
                     const std::vector<std::vector<std::complex<float>>>& tail);

    /// Zero the surface (start of a new alignment).
    void clear_surface();

    /// Correlate the loaded window against the loaded replica at one secondary-code alignment
    /// and add it into the surface. `sgn0`/`sgn1` are the per-hop sign pair, [Mp]:
    /// R = head * sgn0 + tail * sgn1. Throws if no Doppler grid is set.
    void accumulate(const float* sgn0, const float* sgn1);

    /// Peak-pick the accumulated surface as gnss::channelized_peak does, reducing on the device.
    /// The cell -> (code phase, Doppler) arithmetic, including the sub-grid Doppler refine, is
    /// gnss::peak_from_reduction's. `dims` must describe this engine's surface and
    /// `doppler_grid` must be the grid given to @ref set_doppler_grid.
    gnss::AcquisitionResult peak_result(const gnss::AcquisitionSurface& dims,
                                        const std::vector<double>& doppler_grid, double sample_rate,
                                        double chip_rate, long code_length,
                                        bool pairsum_select = false) const;

    /// Peak / mean of the accumulated surface, computed on the device.
    struct Peak {
        double peak;
        double mean;
        long long index; ///< flat index into [n_dop][Mp][s_cols]
        int d, q, s;     ///< decomposed
    };
    Peak peak() const;

    /// Copy the surface back: the large transfer the design avoids, for diagnostics only
    /// (scripts/gnss/acqbench.cpp compares it with the CPU reference).
    void download_surface(std::vector<double>& out) const;

    int n_dop() const {
        return _nd;
    }
    int s_cols() const {
        return _s_cols;
    }
    long surface_cells() const {
        return (long)_nd * _Mp * _s_cols;
    }
    /// Bytes of device memory held, so a caller can budget several engines per GPU.
    size_t device_bytes() const;

private:
    struct Impl;
    Impl* _p;
    int _nc, _Mp, _M, _nchan, _max_dop, _s_cols, _nd = 0;
};

#endif // GNSS_CUDA_ACQUIRE_HPP
