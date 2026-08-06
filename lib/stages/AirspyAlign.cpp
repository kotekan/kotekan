#include "AirspyAlign.hpp"

#include "Config.hpp"          // for Config
#include "NDArray.hpp"         // for GenericNDArray
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "airspyFrameDesc.hpp" // for make_input_desc
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "fftwPlannerLock.hpp" // for fftw_planner_mutex
#include "kotekanLogging.hpp"  // for DEBUG

#include "fmt.hpp" // for compile_string_to_view

#include <algorithm>          // for min
#include <cmath>              // for fabs, sqrt
#include <condition_variable> // for condition_variable
#include <fftw3.h>            // for fftwf_free, fftwf_malloc, fftwf_complex, fftwf_destroy_plan
#include <functional>         // for bind, function, _1
#include <json.hpp>           // for json, basic_json
#include <memory>             // for shared_ptr
#include <mutex>              // for mutex, lock_guard, unique_lock
#include <stdint.h>           // for uint32_t
#include <stdlib.h>           // for free, malloc
#include <string.h>           // for memcpy, memset

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(AirspyAlign);

AirspyAlign::AirspyAlign(Config& config, const std::string& unique_name,
                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&AirspyAlign::main_thread, this)) {

    buf_inA = get_buffer("in_bufA");
    buf_inA->register_consumer(unique_name);
    buf_inB = get_buffer("in_bufB");
    buf_inB->register_consumer(unique_name);

    // Both inputs are int16 1-D airspy sample streams whose descriptors are
    // declared in config (kotekan_buffer: ndarray). require_frame_desc confirms
    // the config declaration is present; this stage reads only the byte length.
    buf_inA->require_frame_desc();
    buf_inB->require_frame_desc();

    _lag_window = config.get_default<uint32_t>(unique_name, "lag_window",
                                               buf_inA->frame_size / sizeof(short));
    localA = (short*)malloc(_lag_window * sizeof(short));
    localB = (short*)malloc(_lag_window * sizeof(short));
}

AirspyAlign::~AirspyAlign() {
    free(localA);
    free(localB);
}

void AirspyAlign::cal_lag_callback(kotekan::connectionInstance& conn) {
    start_callback(conn, false);
}

void AirspyAlign::get_correlation_callback(kotekan::connectionInstance& conn) {
    start_callback(conn, true);
}

void AirspyAlign::start_callback(kotekan::connectionInstance& conn, bool send_corr) {
    {
        std::unique_lock<std::mutex> lock(going_mutex);
        if (going) {
            conn.send_error("Lag capture already in progress.",
                            kotekan::HTTP_RESPONSE::BAD_REQUEST);
            return;
        }
        going = true;
        going_cv.wait(lock, [this] { return !going || stop_thread; });
        if (stop_thread) {
            going = false;
            lock.unlock();
            conn.send_error("Stage shutting down.", kotekan::HTTP_RESPONSE::INTERNAL_ERROR);
            return;
        }
    }

    const uint32_t fft_len = _lag_window * 2;
    const uint32_t n_bins = fft_len / 2 + 1;

    float* samplesA = (float*)fftwf_malloc(sizeof(float) * fft_len);
    fftwf_complex* spectrumA = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * n_bins);
    float* samplesB = (float*)fftwf_malloc(sizeof(float) * fft_len);
    fftwf_complex* spectrumB = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * n_bins);
    float* samplesC = (float*)fftwf_malloc(sizeof(float) * fft_len);
    fftwf_complex* spectrumC = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * n_bins);

    // FFTW's planner is not thread-safe; this callback runs on a REST
    // handler thread and can race other stages' plan create/destroy.
    // Serialize plan creation (fftwf_malloc above is thread-safe).
    fftwf_plan fft_planA, fft_planB, fft_planC;
    {
        std::lock_guard<std::mutex> planner_lock(fftw_planner_mutex());
        fft_planA = fftwf_plan_dft_r2c_1d(fft_len, samplesA, spectrumA, FFTW_ESTIMATE);
        fft_planB = fftwf_plan_dft_r2c_1d(fft_len, samplesB, spectrumB, FFTW_ESTIMATE);
        fft_planC = fftwf_plan_dft_c2r_1d(fft_len, spectrumC, samplesC, FFTW_ESTIMATE);
    }

    for (uint32_t i = 0; i < _lag_window; i++) {
        samplesA[i] = localA[i];
        samplesB[i] = localB[i];
    }
    memset(&samplesA[_lag_window], 0, _lag_window * sizeof(float));
    memset(&samplesB[_lag_window], 0, _lag_window * sizeof(float));

    fftwf_execute(fft_planA);
    fftwf_execute(fft_planB);

    // A x conj(B), bin-by-bin.
    for (uint32_t i = 0; i < n_bins; i++) {
        float Ar = spectrumA[i][0];
        float Ai = spectrumA[i][1];
        float Br = spectrumB[i][0];
        float Bi = spectrumB[i][1];
        spectrumC[i][0] = Ar * Br + Ai * Bi;
        spectrumC[i][1] = Ai * Br - Bi * Ar;
    }
    fftwf_execute(fft_planC);

    float maxval = 0, meanval = 0, var = 0;
    int lag = 0;
    nlohmann::json corr_pos = nlohmann::json::array();
    nlohmann::json corr_neg = nlohmann::json::array();

    const uint32_t edge_truncate = _lag_window * 0.1;
    for (uint32_t i = 0; i < _lag_window - edge_truncate; i++) {
        float norm = (float)_lag_window - i;
        float val = fabs(samplesC[i] / norm);
        if (val > maxval) {
            lag = i;
            maxval = val;
        }
        meanval += val;
        corr_pos.push_back(val);

        // Negative-lag side of the circular correlation: lag -i lives at
        // index (fft_len - i). The modulo keeps i==0 in-bounds (it maps to
        // index 0, i.e. lag 0, the same sample as the positive side) --
        // without it, i==0 reads samplesC[fft_len], one past the end.
        val = fabs(samplesC[(fft_len - i) % fft_len] / norm);
        if (val > maxval) {
            lag = -i;
            maxval = val;
        }
        meanval += val;
        corr_neg.push_back(val);
    }
    meanval /= (_lag_window - edge_truncate) * 2;
    for (uint32_t i = 0; i < _lag_window - edge_truncate; i++) {
        float norm = (float)_lag_window - i;
        float val = fabs(samplesC[i] / norm) - meanval;
        var += val * val;
        val = fabs(samplesC[(fft_len - i) % fft_len] / norm) - meanval;
        var += val * val;
    }
    var /= (_lag_window - edge_truncate) * 2;
    float std_dev = sqrt(var);

    DEBUG("Align: Max: {}, Mean: {}, RMS: {}", maxval, meanval, std_dev);

    nlohmann::json reply;
    if (maxval > meanval + std_dev) {
        DEBUG("Lag found: {}", lag);
        reply["lag"] = lag;
    } else {
        DEBUG("No lag found.");
        reply["lag"] = 0;
    }
    if (send_corr) {
        reply["corr_pos"] = corr_pos;
        reply["corr_neg"] = corr_neg;
    }
    conn.send_json_reply(reply);

    fftwf_free(samplesA);
    fftwf_free(samplesB);
    fftwf_free(samplesC);
    fftwf_free(spectrumA);
    fftwf_free(spectrumB);
    fftwf_free(spectrumC);
    {
        std::lock_guard<std::mutex> planner_lock(fftw_planner_mutex());
        fftwf_destroy_plan(fft_planA);
        fftwf_destroy_plan(fft_planB);
        fftwf_destroy_plan(fft_planC);
    }
}

void AirspyAlign::main_thread() {
    using namespace std::placeholders;
    kotekan::restServer& rest_server = kotekan::restServer::instance();
    rest_server.register_get_callback(unique_name + "/cal_lag",
                                      std::bind(&AirspyAlign::cal_lag_callback, this, _1));
    rest_server.register_get_callback(unique_name + "/get_correlation",
                                      std::bind(&AirspyAlign::get_correlation_callback, this, _1));

    frame_inA = 0;
    frame_inB = 0;
    uint32_t copied = 0;
    const uint32_t samples_per_frame = buf_inA->frame_size / sizeof(short);

    // Assumes the two input buffers are aligned frame-for-frame.
    while (!stop_thread) {
        short* inA = (short*)buf_inA->wait_for_full_frame(unique_name, frame_inA);
        short* inB = (short*)buf_inB->wait_for_full_frame(unique_name, frame_inB);
        if (inA == nullptr || inB == nullptr)
            break;

        bool completed = false;
        {
            std::lock_guard<std::mutex> lock(going_mutex);
            if (going) {
                uint32_t copylen = std::min(_lag_window - copied, samples_per_frame);
                memcpy(localA + copied, inA, copylen * sizeof(short));
                memcpy(localB + copied, inB, copylen * sizeof(short));
                copied += copylen;
                if (copied >= _lag_window) {
                    copied = 0;
                    going = false;
                    completed = true;
                }
            }
        }
        if (completed)
            going_cv.notify_one();

        buf_inA->mark_frame_empty(unique_name, frame_inA);
        buf_inB->mark_frame_empty(unique_name, frame_inB);
        frame_inA = (frame_inA + 1) % buf_inA->num_frames;
        frame_inB = (frame_inB + 1) % buf_inB->num_frames;
    }
    // Wake any REST callback waiting on a pending capture so it doesn't deadlock on shutdown.
    going_cv.notify_all();
}
