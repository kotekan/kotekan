/**
 * @file
 * @brief Decoupled GNSS acquisition (search) stage -- the "search" half of the split.
 *  - GnssAcquire : public kotekan::Stage
 */

#ifndef GNSS_ACQUIRE_HPP
#define GNSS_ACQUIRE_HPP

#include "Config.hpp"
#include "GnssCorrelatorCore.hpp"
#include "GnssTrackState.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "gnssSignal.hpp"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

/**
 * @class GnssAcquire
 * @brief Searches the full PRN x Doppler grid for un-tracked satellites WITHOUT
 *        ever backpressuring the real-time stream.
 *
 * The expensive, latency-tolerant half of the GNSS calibrator. It registers as
 * an ordinary consumer of the voltage buffer but its consumer loop only ever
 * copies a snapshot and immediately releases each frame -- so it drains at line
 * rate and the producer never waits on it. The heavy parallel-code-phase search
 * runs on a separate worker thread over the copied snapshot; while the worker is
 * busy the consumer loop drops every arriving frame. Acquisition latency is thus
 * elastic (a snapshot every so often) while compute is a fixed budget on the
 * worker core -- the key property for deploying on a live pipeline.
 *
 * A snapshot is @c search_blocks contiguous code periods. The worker batches all
 * candidate PRNs (the configured set minus those the tracker already owns,
 * queried via the shared @ref gnss::TrackState), hoisting the per-Doppler data
 * FFT out of the PRN loop. Any PRN whose peak/mean clears @c acquire_snr is
 * handed to the tracker as a @ref gnss::Seed.
 *
 * @par Buffers
 * @buffer in_buf int16 real samples (shared with the GnssTracker).
 * (No output buffer: results flow to the tracker via the shared state.)
 *
 * @conf signal, sample_rate, f_offset   As GpsReplicaCorrelator.
 * @conf prns          Array<int>. PRN universe to search.
 * @conf state_group   String. Shared name pairing this with its GnssTracker.
 * @conf doppler_min/max/step  Search grid, Hz (defaults -6000/6000/250).
 * @conf search_blocks Int (default 200). Contiguous code periods per snapshot
 *                     (the incoherent integration length).
 * @conf acquire_snr   Double (default 9). peak/mean to declare a detection.
 *
 * @author Keith Vanderlinde
 */
class GnssAcquire : public kotekan::Stage {
public:
    GnssAcquire(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);
    ~GnssAcquire() override = default;

    void main_thread() override;

private:
    /// Worker-thread body: search each handed-off snapshot, promote detections.
    void search_worker();
    /// Run one parallel-code-phase search over the current snapshot.
    void search_snapshot();

    Buffer* in_buf;
    int frame_in;

    std::unique_ptr<gnss::CorrelatorCore> core;
    gnss::SignalDescriptor _sig;
    double _sample_rate;
    std::vector<int> _prns;
    int _Ns;

    double _doppler_min, _doppler_max, _doppler_step;
    std::vector<double> _doppler_grid;
    int _search_blocks;
    double _acquire_snr;

    gnss::TrackState* _state;
    int in_samples_per_frame;

    // --- snapshot hand-off (consumer loop -> worker) ---
    std::vector<float> snapshot; ///< search_blocks * Ns real samples
    size_t snap_len;
    std::thread worker;
    std::mutex m;
    std::condition_variable cv;
    bool snap_ready;   ///< a full snapshot is waiting for the worker
    bool worker_busy;  ///< worker is processing (consumer loop drops meanwhile)
    bool worker_stop;  ///< shutdown flag for the worker

    // --- worker-side search scratch ---
    std::vector<float> accum; ///< [n_prn][n_dop][Ns] |corr|^2
};

#endif // GNSS_ACQUIRE_HPP
