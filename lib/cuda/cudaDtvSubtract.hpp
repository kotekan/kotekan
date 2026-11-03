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

#include <memory>
#include <mutex>
#include <string>
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
    };
    static std::shared_ptr<Shared> shared_for(const std::string& name);

    /// Read a state file into the shared mirror; false with the reason on error.
    bool load_state_file(const std::string& path, std::string& err);

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
};

#endif
