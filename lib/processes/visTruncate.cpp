#include "visTruncate.hpp"
#include "errors.h"
#include "visBuffer.hpp"
#include "truncate.hpp"

REGISTER_KOTEKAN_PROCESS(visTruncate);

visTruncate::visTruncate(Config &config, const string& unique_name, bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container, std::bind(&visTruncate::main_thread, this)) {

    // Fetch the buffers, register
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    // Get truncation parameters from config
    err_sq_lim = config.get_float(unique_name, "err_sq_lim");
    w_prec = config.get_float(unique_name, "weight_fixed_precision");
    vis_prec = config.get_float(unique_name, "data_fixed_precision");
}

visTruncate::~visTruncate() {
    double total_time = current_time() - start_time;
    DEBUG("total time %f", total_time);
    DEBUG("wait time %f", wait_time);
    DEBUG("copy time %f", copy_time);
    DEBUG("truncate time %f", truncate_time);
}

void visTruncate::apply_config(uint64_t fpga_seq) {
    (void)fpga_seq;
}

void visTruncate::main_thread() {

    unsigned int frame_id = 0;
    unsigned int output_frame_id = 0;
    float err_r, err_i;
    cfloat tr_vis, tr_evec;

    start_time = current_time();

    while (!stop_thread) {
        last_time = current_time();
        // Wait for the buffer to be filled with data
        if((wait_for_full_frame(in_buf, unique_name.c_str(),
                                        frame_id)) == nullptr) {
            break;
        }
        auto frame = visFrameView(in_buf, frame_id);

        // Wait for empty frame
        if((wait_for_empty_frame(out_buf, unique_name.c_str(),
                                 output_frame_id)) == nullptr) {
            break;
        }
        wait_time += current_time() - last_time;
        last_time = current_time();
        // Copy frame into output buffer
        allocate_new_metadata_object(out_buf, output_frame_id);
        auto output_frame = visFrameView(out_buf, output_frame_id, frame);
        copy_time += current_time() - last_time;

        last_time = current_time();
        // truncate visibilities and weights
        #pragma omp parallel for private(err_r, err_i, tr_vis)
        for (size_t i = 0; i < frame.num_prod; i++) {
            // Get truncation precision from weights
            if (output_frame.weight[i] == 0.) {
                // TODO: should this raise a warning?
                err_r = vis_prec * output_frame.vis[i].real();
                err_i = vis_prec * output_frame.vis[i].imag();
            } else {
                err_r = std::sqrt(0.5 / output_frame.weight[i] * err_sq_lim);
                err_i = err_r;
            }
            // truncate vis using weights
            tr_vis = { bit_truncate_float(output_frame.vis[i].real(), err_r),
                       bit_truncate_float(output_frame.vis[i].imag(), err_i) };
            output_frame.vis[i] = tr_vis;
            // truncate weights to fixed precision
            output_frame.weight[i] = bit_truncate_float(output_frame.weight[i],
                    w_prec * output_frame.weight[i]);
        }
        // truncate eigenvectors
        #pragma omp parallel for private(err_r, err_i, tr_evec)
        for (size_t i = 0; i < output_frame.evec.size(); i++) {
            // Truncate to fixed precision
            tr_evec = {
                bit_truncate_float(output_frame.evec[i].real(),
                        std::abs(vis_prec * output_frame.evec[i].real())),
                bit_truncate_float(output_frame.evec[i].imag(),
                        std::abs(vis_prec * output_frame.evec[i].imag()))
            };
            output_frame.evec[i] = tr_evec;
        }
        truncate_time += current_time() - last_time;

        // mark as full
        mark_frame_full(out_buf, unique_name.c_str(), output_frame_id);
        output_frame_id = (output_frame_id + 1) % out_buf->num_frames;
        // move to next frame
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % in_buf->num_frames;
    }
}

