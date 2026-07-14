/**
 * @file
 * @brief Export the per-feed spectral kurtosis as prometheus gauges.
 *  - RfiSKMetrics : public kotekan::Stage
 */

#ifndef RFI_SK_METRICS_HPP
#define RFI_SK_METRICS_HPP

#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "prometheusMetrics.hpp" // for Gauge, MetricFamily
#include "restServer.hpp"        // for connectionInstance

#include <mutex>    // for mutex
#include <stddef.h> // for size_t
#include <string>   // for string
#include <vector>   // for vector

/**
 * @class RfiSKMetrics
 * @brief Export the per-feed (single-feed) spectral kurtosis as prometheus gauges.
 *
 * Consumes the host copy of the second-stage RFI kernel's single-feed SK
 * (cudaRFISKbar ``rfi_SKbar``): float32 with shape
 * [rfi_num_times_bar, num_local_freq, 3, num_polarizations, num_dishes],
 * where the length-3 axis is {SK, b, sigma} and sigma < 0 marks an invalid
 * (masked) cell (see n2k rfi_kernels.hpp).
 *
 * Each frame, every element's SK is averaged over its valid (time, freq)
 * cells and folded into an exponential moving average. Clean Gaussian noise
 * has SK ~= 1. Elements are indexed by the flat [P][D] element index (the
 * same indexing as /updatable_config/bad_inputs).
 *
 * The averages are served two ways: a GET endpoint ``<unique_name>/sk``
 * returning JSON for machine consumers (the bffs feed flagger polls this),
 * and prometheus gauges ``kotekan_rfi_sk_per_feed{element="..."}`` /
 * ``kotekan_rfi_sk_per_feed_valid_frac{element="..."}`` for monitoring
 * dashboards.
 *
 * The single-feed SK is computed by the kernel for every feed regardless of
 * the bad-feed mask, so these outputs keep tracking flagged feeds — an
 * external flagger can watch them and let a recovered feed heal.
 *
 * @par Buffers
 * @buffer in_buf Single-feed SK copied from the GPU.
 *     @buffer_shape [rfi_num_times_bar, num_local_freq, 3, num_polarizations, num_dishes]
 *     @buffer_format float32
 *
 * @conf num_local_freq     Int. Frequencies per frame.
 * @conf num_polarizations  Int. Polarizations; elements = num_polarizations * num_dishes.
 * @conf num_dishes         Int. Dishes.
 * @conf rfi_num_times_bar  Int. Downsampled time samples per frame.
 * @conf ema_frames         Int, default 256. E-folding length of the moving average,
 *                          in frames (~11 s at the pathfinder frame rate).
 *
 * REST Endpoints
 *
 * @endpoint /sk GET Returns ``{"num_elements": N, "ema_frames": M, "sk": [...],
 *                   "valid_frac": [...]}``; the arrays are indexed by element,
 *                   and ``sk`` entries are null until an element's first valid
 *                   measurement.
 *
 * @author James Mertens
 */
class RfiSKMetrics : public kotekan::Stage {
public:
    RfiSKMetrics(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);
    virtual ~RfiSKMetrics();
    void main_thread() override;

    /// GET handler for the /sk endpoint
    void send_sk(kotekan::connectionInstance& conn);

private:
    size_t num_freq;
    size_t num_times;
    size_t num_elements;
    size_t ema_frames;
    double ema_alpha;

    /// Guards the moving averages between main_thread and the /sk endpoint
    std::mutex mtx;

    /// Per-element moving averages (SK is NaN until the first valid frame)
    std::vector<double> ema_sk;
    std::vector<double> ema_valid_frac;

    /// Per-element gauges, looked up once at construction
    std::vector<kotekan::prometheus::Gauge*> sk_gauges;
    std::vector<kotekan::prometheus::Gauge*> valid_frac_gauges;

    Buffer* in_buf;
};

#endif
