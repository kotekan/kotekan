/**
 * @file
 * @brief Subtract a terrestrial transmitter from selected channels of the voltage ring as the
 *        frames are written into it.
 *  - cudaDtvSubtract : public cudaCopyToRingbuffer
 */

#ifndef CUDA_DTV_SUBTRACT_HPP
#define CUDA_DTV_SUBTRACT_HPP

#include "NDArrayBuffer.hpp" // for NDArrayBuffer
#include "cudaCopyToRingbuffer.hpp"
#include "cudaDtvSubtractKernel.hpp" // for DtvOutLayout
#include "dtvRollingFit.hpp"         // for DtvRollOutcome

#include "json.hpp" // for json

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

/**
 * @class cudaDtvSubtract
 * @brief The voltage ring's copy-in with a lagged subtraction on selected channels.
 *
 * Replaces cudaCopyToRingbuffer in the process that fills the GPU voltage ring, so every
 * consumer of the ring (the N2 correlators, the RFI statistics, the GNSS stages) sees the same
 * frame. For each selected channel (a 195 kHz channel inside a DTV station's 6 MHz) it models
 * the inputs as one transmitter timestream through a per-input impulse response
 * (docs/CHORD_DTV_SUBTRACTION_20261102.md, cudaDtvSubtractKernel.hpp): per frame it forms the
 * timestream s = sum_i w_i x_i, correlates every input with s at the configured lags, subtracts
 * sum_tau g_i(tau) s(t - tau) from each input and re-quantises the residual in place. A second
 * source is the same on the residual.
 *
 * The weights w come from an eigen start on request (POST `<unique_name>/start`, {"source": k,
 * "frames": n}: the covariance of the residual entering source k over n frames, its whitened
 * top eigenvector referenced to the strongest input) or from a state file; the impulse
 * responses g come from a state file written by scripts/gnss/dtv_state.py from the lag
 * correlations this command writes (POST `<unique_name>/load`, {"file": path}; also
 * `dtv_state_file` at start). Without weights a channel passes through untouched; with weights
 * but no g it is observed (correlations written, nothing subtracted). POST `/clear`
 * ({"source": k}) drops sources k and above; GET `/status` reports the state.
 *
 * Rolling fit (dtvRollingFit.hpp): when on, the stage keeps sums of the lag correlations on the
 * device that decay with a time constant per source, copies them to the host every
 * `dtv_roll_interval` frames, and a host thread refits g of every subtracting source from them;
 * the next frame uploads the new g. w is untouched; the sums of a source restart when its w
 * changes. A channel whose residual power over an interval exceeds `dtv_roll_max_power` times
 * its input power stops subtracting (act 1) until the next load; a channel whose source-0 fit
 * explains less than `dtv_roll_min_explained` of the input power keeps its g. POST
 * `<unique_name>/rolling` changes the settings ({"enable": bool, "tau_s": [s per source],
 * "interval": frames, "freeze": [0/1 per source], "min_explained", "max_power", "reset": true,
 * "save": path}; "save" writes the state in use as a state file).
 *
 * The output is an NDArrayBuffer `dtv_out_name` (on the device `<dtv_out_name>_buffer`) for
 * cudaOutputData: per frame the header (the int64 seq0 at byte 0 is FrameWindowGate's clock),
 * the weights in use, the per-input powers before and after subtraction, |s|^2 and the lag
 * correlations R_xs and R_ss (cudaDtvSubtractKernel.hpp, DtvOutLayout).
 *
 * @conf As cudaCopyToRingbuffer (input_size, ring_buffer_size, gpu_mem_output, in_buf,
 *       signal_buf) and samples_per_data_set, num_elements, num_local_freq, and:
 * @conf  dtv_local_channels  List of Int. Local frame channel of each selected channel.
 * @conf  dtv_freq_ids        List of Int. Their freq_ids, same order; checked against the
 *                            frame metadata's coarse frequencies on the first frame.
 * @conf  dtv_out_name        String. Output NDArrayBuffer name, e.g. gnss0_dtv.
 * @conf  dtv_lag_min         Int, default -8. Lag of index 0, hops (s(t - lag)).
 * @conf  dtv_n_lag           Int, default 49. Lags per source (lag_min .. lag_min + n_lag - 1,
 *                            within [-16, 63]).
 * @conf  dtv_sources         Int, default 1. Source slots per channel (<= 4).
 * @conf  dtv_state_file      String, default "". State file to load at start.
 * @conf  dtv_eig_iters       Int, default 24. Power iterations of the eigen start.
 * @conf  dtv_roll            Bool, default false. Rolling fit on at start.
 * @conf  dtv_roll_tau_s      List of Float, default [1.5, 4, 8, 8]. Time constant of each
 *                            source's sums, s.
 * @conf  dtv_roll_interval   Int, default 24. Frames between refits.
 * @conf  dtv_roll_min_explained Float, default 0.05.
 * @conf  dtv_roll_max_power  Float, default 2.0.
 */
class cudaDtvSubtract : public cudaCopyToRingbuffer {
public:
    cudaDtvSubtract(kotekan::Config& config, const std::string& unique_name,
                    kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                    int instance_num);
    ~cudaDtvSubtract();

    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;

private:
    /// Host state shared by the command's instances (one per in-flight frame).
    struct Shared {
        std::mutex mtx;
        std::vector<float2> w; ///< [n_chan][n_src][DTV_INPUTS]
        std::vector<float2> g; ///< [n_chan][n_src][n_lag][DTV_INPUTS]
        std::vector<int> act;  ///< [n_chan][n_src]: 0 off, 1 observe, 2 subtract
        bool dirty_w = false;  ///< upload w (and act) at the next frame
        bool dirty_g = false;  ///< upload g (and act) at the next frame
        int start_src = -1;    ///< eigen start in progress for this source (-1: none)
        int start_left = 0;    ///< frames still to accumulate
        int start_total = 0;   ///< frames requested
        std::string state_file;
        std::string last_error;
        long frames = 0;

        // rolling fit settings
        bool roll = false;
        std::vector<double> roll_tau; ///< [n_src] s
        std::vector<int> roll_freeze; ///< [n_src]
        int roll_interval = 24;
        double roll_min_explained = 0.05;
        double roll_max_power = 2.0;
        // rolling fit state
        std::vector<char> roll_reset; ///< [n_chan][n_src] zero these sums at the next frame
        std::vector<long> roll_since; ///< [n_chan][n_src] frames in the sums
        std::vector<char> roll_trip;  ///< [n_chan] stopped by the power check
        long roll_gen = 0;            ///< bumped when w, g or act change outside the refits
        int roll_frames = 0;          ///< frames since the last snapshot
        bool roll_copying = false;    ///< a snapshot is being copied to the host
        bool roll_solving = false;    ///< the refit thread has the snapshot
        bool roll_stop = false;
        float* snap_rxs = nullptr; ///< pinned copies of the sums
        float* snap_rss = nullptr;
        float* snap_pw = nullptr;
        std::vector<int> snap_act;
        std::vector<long> snap_since;
        long snap_gen = 0;
        long snap_frame = 0;
        cudaEvent_t snap_event = nullptr;
        std::condition_variable roll_cv;
        std::thread roll_thread;
        long roll_updates = 0, roll_holds = 0, roll_trips = 0, roll_bad = 0, roll_last = -1;
        double roll_ms = 0.0;
        gnss_cuda::DtvRollOutcome roll_out; ///< the last refit
    };
    static std::shared_ptr<Shared> shared_for(const std::string& name);

    /// Read a state file into the shared mirror; false with the reason on error.
    bool load_state_file(const std::string& path, std::string& err);
    /// Write the w, g and act in use as a state file; false with the reason on error.
    bool save_state_file(const std::string& path, std::string& err);
    /// The refit thread: waits for a snapshot, refits, puts the accepted g in the mirror.
    void roll_main();
    /// Mark the sums of every source of every channel to restart (caller holds the lock).
    void roll_reset_all();
    /// The rolling fit's settings and last refit, for /status and /rolling (caller holds the lock).
    nlohmann::json roll_status() const;

    const int _num_times;
    const int _num_elements;
    const int _num_local_freq;
    const std::vector<std::int32_t> _local_channels;
    const std::vector<std::int32_t> _freq_ids;
    const std::string _out_name;
    const int _lag_min;
    const int _n_lag;
    const int _n_src;
    const int _eig_iters;
    const gnss_cuda::DtvOutLayout _layout;

    NDArrayBuffer<unsigned char, 2> out;
    std::shared_ptr<Shared> _sh;
    std::vector<unsigned char> _hdr_stage;
    std::vector<float> _lam_stage; ///< eigenvalues of the last eigen start, read back
    std::vector<float2> _w_stage;  ///< the device weights after an eigen start, read back
    cudaEvent_t _lam_event = nullptr;
    bool _lam_pending = false;
    int _lam_src = -1;
    bool _checked_freqs = false;
    const bool _owner; ///< instance 0: owns the REST endpoints and the refit thread
};

#endif
