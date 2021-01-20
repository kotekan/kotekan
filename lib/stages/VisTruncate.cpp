#include "VisTruncate.hpp"

#include "Config.hpp"         // for Config
#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"           // for wait_for_full_frame, allocate_new_metadata_object, mark_fr...
#include "kotekanLogging.hpp" // for DEBUG
#include "truncate.hpp"       // for bit_truncate_float
#include "visBuffer.hpp"      // for VisFrameView
#include "visUtil.hpp"        // for cfloat

#include "gsl-lite.hpp" // for span

#include <atomic>      // for atomic_bool
#include <cmath>       // for abs, sqrt
#include <complex>     // for complex
#include <cstdint>     // for int32_t
#include <cstring>     // for memset, size_t
#include <exception>   // for exception
#include <functional>  // for _Bind_helper<>::type, bind, function
#include <immintrin.h> // for _mm256_broadcast_ss, _mm256_div_ps, _mm256_loadu_ps, _mm25...
#include <mm_malloc.h> // for _mm_free, _mm_malloc
#include <regex>       // for match_results<>::_Base_type
#include <stdexcept>   // for invalid_argument
#include <vector>      // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(VisTruncate);

VisTruncate::VisTruncate(Config& config, const std::string& unique_name,
                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&VisTruncate::main_thread, this)) {

    // Fetch the buffers, register
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    // Get truncation parameters from config
    err_sq_lim = config.get<float>(unique_name, "err_sq_lim");
    if (err_sq_lim < 0)
        FATAL_ERROR("VisTruncate: config: err_sq_lim should be positive (is %f).", err_sq_lim);
    w_prec = config.get<float>(unique_name, "weight_fixed_precision");
    if (w_prec < 0)
        FATAL_ERROR("VisTruncate: config: weight_fixed_precision should be positive (is %f).",
                    w_prec);
    vis_prec = config.get<float>(unique_name, "data_fixed_precision");
    if (vis_prec < 0)
        FATAL_ERROR("VisTruncate: config: data_fixed_precision should be positive (is %f).",
                    vis_prec);
}

void VisTruncate::main_thread() {

    unsigned int frame_id = 0;
    unsigned int output_frame_id = 0;
    const float err_init = 0.5 * err_sq_lim;

    float err_r, err_i;
    const __m256 err_init_vec = _mm256_set1_ps(err_init);
    cfloat tr_vis, tr_evec;
    __m256 err_vec, wgt_vec;
    int32_t i_vec;
    float* err_all;

    // get the first frame (just to find out about num_prod)
    // (we don't mark it empty, so it's read again in the main loop)
    if (wait_for_full_frame(in_buf, unique_name.c_str(), frame_id) == nullptr)
        return;
    auto frame = VisFrameView(in_buf, frame_id);

    // reserve enough memory for all err_r to be computed per frame
    // 32byte-aligned memory allocation (_m256_store_ps() asks for it)
    err_all = (float*)_mm_malloc(sizeof(float) * frame.num_prod, 32);
    std::memset(err_all, 0, sizeof(float) * (frame.num_prod));

    while (!stop_thread) {
        // Wait for the buffer to be filled with data
        if ((wait_for_full_frame(in_buf, unique_name.c_str(), frame_id)) == nullptr) {
            break;
        }
        auto frame = VisFrameView(in_buf, frame_id);

        // Wait for empty frame
        if ((wait_for_empty_frame(out_buf, unique_name.c_str(), output_frame_id)) == nullptr) {
            break;
        }

        // Copy frame into output buffer
        auto output_frame = VisFrameView::copy_frame(in_buf, frame_id, out_buf, output_frame_id);

        // truncate visibilities and weights (8 at a time)
        for (i_vec = 0; i_vec < int32_t(frame.num_prod) - 7; i_vec += 8) {
            wgt_vec = _mm256_loadu_ps(&output_frame.weight[i_vec]);
            err_vec = _mm256_div_ps(err_init_vec, wgt_vec);
            err_vec = _mm256_sqrt_ps(err_vec);
            _mm256_store_ps(err_all + i_vec, err_vec);
        }
        // use std::sqrt for the last few (less than 8)
        for (i_vec = (frame.num_prod < 8) ? 0 : i_vec - 8; i_vec < int32_t(frame.num_prod); i_vec++)
            err_all[i_vec] = std::sqrt(0.5 / output_frame.weight[i_vec] * err_sq_lim);

#pragma omp parallel for private(err_r, err_i, tr_vis)
        for (size_t i = 0; i < frame.num_prod; i++) {
            // Get truncation precision from weights
            if (output_frame.weight[i] == 0.) {
                zero_weight_found = true;
                err_r = vis_prec * std::abs(output_frame.vis[i].real());
                err_i = vis_prec * std::abs(output_frame.vis[i].imag());
            } else {
                err_r = err_all[i];
                err_i = err_r;
            }
            // truncate vis using weights
            tr_vis = {bit_truncate_float(output_frame.vis[i].real(), err_r),
                      bit_truncate_float(output_frame.vis[i].imag(), err_i)};
            output_frame.vis[i] = tr_vis;
            // truncate weights to fixed precision
            output_frame.weight[i] =
                bit_truncate_float(output_frame.weight[i], w_prec * output_frame.weight[i]);
        }
// truncate eigenvectors
#pragma omp parallel for private(err_r, err_i, tr_evec)
        for (size_t i = 0; i < output_frame.evec.size(); i++) {
            // Truncate to fixed precision
            tr_evec = {bit_truncate_float(output_frame.evec[i].real(),
                                          std::abs(vis_prec * output_frame.evec[i].real())),
                       bit_truncate_float(output_frame.evec[i].imag(),
                                          std::abs(vis_prec * output_frame.evec[i].imag()))};
            output_frame.evec[i] = tr_evec;
        }

// truncate gains using same precision as eigenvectors
#pragma omp parallel for private(tr_evec)
        for (size_t i = 0; i < output_frame.gain.size(); i++) {

            tr_evec = {bit_truncate_float(output_frame.gain[i].real(),
                                          std::abs(vis_prec * output_frame.gain[i].real())),
                       bit_truncate_float(output_frame.gain[i].imag(),
                                          std::abs(vis_prec * output_frame.gain[i].imag()))};
            output_frame.gain[i] = tr_evec;
        }

        if (zero_weight_found) {
            DEBUG("VisTruncate: Frame {:d} has at least one weight value "
                  "being zero.",
                  frame_id);
            zero_weight_found = false;
        }

        // mark as full
        mark_frame_full(out_buf, unique_name.c_str(), output_frame_id);
        output_frame_id = (output_frame_id + 1) % out_buf->num_frames;
        // move to next frame
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % in_buf->num_frames;
    }
    _mm_free(err_all);
}
