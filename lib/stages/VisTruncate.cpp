#include "VisTruncate.hpp"

#include "Config.hpp"         // for Config
#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"         // for Buffer
#include "kotekanLogging.hpp" // for FATAL_ERROR, DEBUG
#include "truncate.hpp"       // for bit_truncate_float
#include "visBuffer.hpp"      // for VisFrameView
#include "visUtil.hpp"        // for cfloat

#include "fmt.hpp"      // for compile_string_to_view
#include "gsl-lite.hpp" // for span

#include <cmath>       // for abs, sqrt
#include <complex>     // for complex
#include <cstdint>     // for int32_t
#include <cstring>     // for memset, size_t
#include <functional>  // for bind, function
#ifdef __AVX2__
#include <immintrin.h> // for __m256, _mm256_div_ps, _mm256_loadu_ps, _mm256_set1_ps
#endif


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(VisTruncate);

VisTruncate::VisTruncate(Config& config, const std::string& unique_name,
                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&VisTruncate::main_thread, this)) {

    // Fetch the buffers, register
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

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
    float err_r, err_i;
#ifdef __AVX2__
    const float err_init = 0.5 * err_sq_lim;
    const __m256 err_init_vec = _mm256_set1_ps(err_init);
    __m256 err_vec, wgt_vec;
#endif
    cfloat tr_vis, tr_evec;
    int32_t i_vec;
    float* err_all;

    // get the first frame (just to find out about num_prod)
    // (we don't mark it empty, so it's read again in the main loop)
    if (in_buf->wait_for_full_frame(unique_name, frame_id) == nullptr)
        return;
    auto frame = VisFrameView(in_buf, frame_id);

    // reserve enough memory for all err_r to be computed per frame
    err_all = (float*)std::malloc(sizeof(float) * frame.num_prod);
    std::memset(err_all, 0, sizeof(float) * (frame.num_prod));

    while (!stop_thread) {
        // Wait for the buffer to be filled with data
        if ((in_buf->wait_for_full_frame(unique_name, frame_id)) == nullptr) {
            break;
        }
        auto frame = VisFrameView(in_buf, frame_id);

        // Wait for empty frame
        if ((out_buf->wait_for_empty_frame(unique_name, output_frame_id)) == nullptr) {
            break;
        }

        // Copy frame into output buffer
        auto output_frame = VisFrameView::copy_frame(in_buf, frame_id, out_buf, output_frame_id);

        // truncate visibilities and weights (8 at a time on x86)
        i_vec = 0;
#ifdef __AVX2__
        for (; i_vec < int32_t(frame.num_prod) - 7; i_vec += 8) {
            wgt_vec = _mm256_loadu_ps(&output_frame.weight[i_vec]);
            err_vec = _mm256_div_ps(err_init_vec, wgt_vec);
            err_vec = _mm256_sqrt_ps(err_vec);
            _mm256_storeu_ps(err_all + i_vec, err_vec);
        }
#endif
        // scalar path for remaining elements (or all elements on non-x86)
        for (; i_vec < int32_t(frame.num_prod); i_vec++)
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
        out_buf->mark_frame_full(unique_name, output_frame_id);
        output_frame_id = (output_frame_id + 1) % out_buf->num_frames;
        // move to next frame
        in_buf->mark_frame_empty(unique_name, frame_id);
        frame_id = (frame_id + 1) % in_buf->num_frames;
    }
    std::free(err_all);
}
