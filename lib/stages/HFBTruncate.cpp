#include "HFBTruncate.hpp"

#include "Config.hpp"         // for Config
#include "HFBFrameView.hpp"   // for HFBFrameView
#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"         // for Buffer
#include "kotekanLogging.hpp" // for FATAL_ERROR, DEBUG
#include "truncate.hpp"       // for bit_truncate_float
#include "visUtil.hpp"        // for frameID, modulo

#include "fmt.hpp"      // for compile_string_to_view
#include "gsl-lite.hpp" // for span

#include <cmath>      // for sqrt, abs
#include <cstdint>    // for int32_t, uint32_t
#include <cstring>    // for memset, size_t
#include <functional> // for bind, function
#ifdef __AVX2__
#include <immintrin.h> // for __m256, _mm256_div_ps, _mm256_loadu_ps, _mm256_set1_ps
#endif


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(HFBTruncate);

HFBTruncate::HFBTruncate(Config& config, const std::string& unique_name,
                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&HFBTruncate::main_thread, this)) {

    // Fetch the buffers, register
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    // Get truncation parameters from config
    err_sq_lim = config.get<float>(unique_name, "err_sq_lim");
    if (err_sq_lim < 0)
        FATAL_ERROR("HFBTruncate: config: err_sq_lim should be positive (is %f).", err_sq_lim);
    w_prec = config.get<float>(unique_name, "weight_fixed_precision");
    if (w_prec < 0)
        FATAL_ERROR("HFBTruncate: config: weight_fixed_precision should be positive (is %f).",
                    w_prec);
    hfb_prec = config.get<float>(unique_name, "data_fixed_precision");
    if (hfb_prec < 0)
        FATAL_ERROR("HFBTruncate: config: data_fixed_precision should be positive (is %f).",
                    hfb_prec);
}

void HFBTruncate::main_thread() {

    frameID frame_id(in_buf), output_frame_id(out_buf);
    float err, tr_hfb;
#ifdef __AVX2__
    const float err_init = 0.5 * err_sq_lim;
    const __m256 err_init_vec = _mm256_set1_ps(err_init);
    __m256 err_vec, wgt_vec;
#endif
    int32_t i_vec;
    float* err_all;

    // get the first frame (just to find out about data size)
    // (we don't mark it empty, so it's read again in the main loop)
    if (in_buf->wait_for_full_frame(unique_name, frame_id) == nullptr)
        return;
    auto frame = HFBFrameView(in_buf, frame_id);

    uint32_t data_size = frame.num_beams * frame.num_subfreq;

    // reserve enough memory for all err to be computed per frame
    err_all = (float*)std::malloc(sizeof(float) * data_size);
    std::memset(err_all, 0, sizeof(float) * data_size);

    while (!stop_thread) {
        // Wait for the buffer to be filled with data
        if ((in_buf->wait_for_full_frame(unique_name, frame_id)) == nullptr) {
            break;
        }
        auto frame = HFBFrameView(in_buf, frame_id);

        // Wait for empty frame
        if ((out_buf->wait_for_empty_frame(unique_name, output_frame_id)) == nullptr) {
            break;
        }

        // Copy frame into output buffer
        auto output_frame = HFBFrameView::copy_frame(in_buf, frame_id, out_buf, output_frame_id);

        // truncate absorber data and weights (8 at a time on x86)
        i_vec = 0;
#ifdef __AVX2__
        for (; i_vec < int32_t(data_size) - 7; i_vec += 8) {
            wgt_vec = _mm256_loadu_ps(&output_frame.weight[i_vec]);
            err_vec = _mm256_div_ps(err_init_vec, wgt_vec);
            err_vec = _mm256_sqrt_ps(err_vec);
            _mm256_storeu_ps(err_all + i_vec, err_vec);
        }
#endif
        // scalar path for remaining elements (or all elements on non-x86)
        for (; i_vec < int32_t(data_size); i_vec++)
            err_all[i_vec] = std::sqrt(0.5 / output_frame.weight[i_vec] * err_sq_lim);

#pragma omp parallel for private(err, tr_hfb)
        for (size_t i = 0; i < data_size; i++) {
            // Get truncation precision from weights
            if (output_frame.weight[i] == 0.) {
                zero_weight_found = true;
                err = hfb_prec * std::abs(output_frame.hfb[i]);
            } else {
                err = err_all[i];
            }
            // truncate hfb using weights
            tr_hfb = bit_truncate_float(output_frame.hfb[i], err);
            output_frame.hfb[i] = tr_hfb;
            // truncate weights to fixed precision
            output_frame.weight[i] =
                bit_truncate_float(output_frame.weight[i], w_prec * output_frame.weight[i]);
        }

        if (zero_weight_found) {
            DEBUG("HFBTruncate: Frame {:d} has at least one weight value "
                  "being zero.",
                  frame_id);
            zero_weight_found = false;
        }

        // mark as full
        out_buf->mark_frame_full(unique_name, output_frame_id++);
        // move to next frame
        in_buf->mark_frame_empty(unique_name, frame_id++);
    }
    std::free(err_all);
}
