#include "N2Accumulate.hpp"

#include "Config.hpp"            // for Config
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"         // for Telescope
#include "buffer.hpp"            // for register_producer, Buffer, allocate_new_metadata_object
#include "bufferContainer.hpp"   // for bufferContainer
#include "chordMetadata.hpp"     // for chordMetadata, get_fpga_seq_num
#include "kotekanLogging.hpp"    // for FATAL_ERROR, INFO, logLevel, DEBUG
#include "N2Util.hpp"            // for frameID
#include "N2FrameView.hpp"       // for N2FrameView
#include "prometheusMetrics.hpp" // for Metrics

#include <algorithm>  // for copy, max, fill, copy_backward, equal, transform
#include <assert.h>   // for assert
#include <complex>    // for operator*, complex
#include <sys/time.h> // for TIMEVAL_TO_TIMESPEC
#include <time.h>     // for size_t, timespec
#include <vector>     // for vector


using namespace std::placeholders;

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(N2Accumulate);


N2Accumulate::N2Accumulate(Config& config, const std::string& unique_name,
                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&N2Accumulate::main_thread, this)),
    skipped_frame_counter(Metrics::instance().add_counter(
        "kotekan_N2accumulate_skipped_frame_total", unique_name, {"freq_id", "reason"})) {

    auto& tel = Telescope::instance();

    // Fetch configuration

    // number of frequencies in frame
    _num_freq_in_frame = config.get<int32_t>(unique_name, "num_freq_in_frame");

    // sampling information
    _n_fpga_samples_per_N2_frame = config.get<int32_t>(unique_name, "samples_per_data_set"); // same as in output frame, just coarsened
    _n_fpga_samples_N2_integrates_for = config.get<int32_t>(unique_name, "sub_integration_ntime");
    _n_vis_samples_per_N2_output_frame = _n_fpga_samples_per_N2_frame / _n_fpga_samples_N2_integrates_for;
    
    _n_vis_samples_per_in_frame = _n_vis_samples_per_N2_output_frame;
    _n_fpga_samples_per_vis_sample = _n_fpga_samples_N2_integrates_for;
    _in_frame_duration_nsec = (uint64_t) _n_fpga_samples_per_N2_frame * (uint64_t) tel.seq_length_nsec();
    _in_frame_vis_duration_nsec = _in_frame_duration_nsec / _n_vis_samples_per_in_frame;

    // Number of products sent by the GPU
    _num_elements = config.get<int32_t>(unique_name, "num_elements");
    _num_N2_products = _num_elements*_num_elements; // TODO: Eventually, a blocked matrix might be sent by the gpu
    _num_N2_products_freqs = _num_N2_products * _num_freq_in_frame;
    // Number of products to accumulate
    _num_accum_products = N2::get_num_prod(_num_elements);
    
    // Initializing these here using the computed _num_N2_products_freqs (accumulate the full, blocked matrix x frequencies from the GPU)
    _vis = std::vector<int32_t>(2 * _num_N2_products_freqs, 0); // vis with complex as 2 ints
    _vis_even = std::vector<int32_t>(2 * _num_N2_products_freqs, 0); // store even vis matrix for differencing
    _weights = std::vector<int32_t>(_num_N2_products_freqs, 0); // real-valued weights
    // number of fpga samples, per frequency, in frame
    _n_valid_fpga_samples_in_vis = std::vector<int32_t>(_num_freq_in_frame, 0);
    _n_valid_fpga_samples_in_vis_even = std::vector<int32_t>(_num_freq_in_frame, 0);
    _n_valid_sample_diff_sq_sum = std::vector<int32_t>(_num_freq_in_frame, 0);

    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);

    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);
    // TODO...
    // Make sure output buffer has enough frames (>= # frequencies) and are sized correctly
    // Add other assert()s/checks back
}

void N2Accumulate::main_thread() {

    auto& comp_time_seconds_metric =
        Metrics::instance().add_gauge("kotekan_N2_accum_time", unique_name);
    auto& samples_in_out_frame =
        Metrics::instance().add_gauge("kotekan_samples_in_accumulated_out_frame", unique_name);

    N2::frameID in_frame_id(in_buf);
    N2::frameID out_frame_id(out_buf);

    INFO("Accumulating GPU output for {:s}[{:d}] putting result in {:s}[{:d}]",
            in_buf->buffer_name, in_frame_id, out_buf->buffer_name, out_frame_id);

    // Start time of an output frame (initialize to now)
    timespec output_ts;
    timespec_get(&output_ts, TIME_UTC);
    uint64_t t_output = N2::ts_to_uint64(output_ts);

    size_t vis_samples_in_out_frame = 0;

    while (!stop_thread) {

        // Fetch a new frame and get its sequence id
        DEBUG("Waiting for new input frame {:s}[{:d}].", in_buf->buffer_name, in_frame_id);
        uint8_t* in_frame = in_buf->wait_for_full_frame(unique_name, in_frame_id);
        if (in_frame == nullptr)
            break;
        int32_t* input = (int32_t*)in_frame;
        
        std::shared_ptr<chordMetadata> frame_metadata = get_chord_metadata(in_buf, in_frame_id);
        size_t in_frame_num = frame_metadata->fpga_seq_num / _n_fpga_samples_per_N2_frame;

        // Start and end times of this frame
        bool gps_time_enabled = false;
        // Here we'll just use raw nanoseconds
        uint64_t t_frame_s;
        if (gps_time_enabled) {
            t_frame_s = N2::ts_to_uint64(frame_metadata->gps_time);
        } else {
            // If GPS time is not set, fall back to system time.
            timespec ts;
            TIMEVAL_TO_TIMESPEC( &frame_metadata->first_packet_recv_time,
                &ts );
            t_frame_s = N2::ts_to_uint64(ts);
        }
        // uint64_t t_frame_e = t_frame_s + _in_frame_duration_nsec;
        comp_time_seconds_metric.set(t_frame_s / 1e9);

        // Accumulate each visibility sample in the in_frame
        for (size_t vis_samp_n = 0; vis_samp_n < _n_vis_samples_per_in_frame; ++vis_samp_n) {

            // Start and end times of the visibility matrix sample
            uint64_t t_vis_s = t_frame_s + vis_samp_n*_in_frame_vis_duration_nsec;
            // uint64_t t_vis_e = t_vis_s + _in_frame_vis_duration_nsec;

            // "absolute" vis sample number
            size_t vis_sample_num_abs = in_frame_num*_n_vis_samples_per_in_frame + vis_samp_n;
            
            DEBUG("Accumulating new visibility sample ({:d} of {:d} in frame).",
                vis_samp_n, _n_vis_samples_per_in_frame );
            // DEBUG("   Times are [start, end, out, num] = [{:d}, {:d}, {:d}, {:d}]",
            //     t_vis_s, t_vis_e, t_output, vis_sample_num_abs );


            // Finalize accumulation if the visibility elements are past the output time...
            //  end on an odd frame too so we accumulate weights.
            if(t_vis_s > t_output
                && vis_sample_num_abs % 2 == 1) {

                INFO("Finishing N2Accumulate output frame. Accumulated {:d} visibility samples.",
                    vis_samples_in_out_frame);
                samples_in_out_frame.set(vis_samples_in_out_frame);
                output_and_reset( in_frame_id, out_frame_id );

                t_output += 1000000000L; // TODO: Make this a config parameter. Is there a library for LST?
                vis_samples_in_out_frame = 0;
            }

            // Actual accumulation over
            for (size_t d = 0; d < 2*_num_N2_products_freqs; ++d) {
                _vis[d] += input[d];
            } // d

            // If we're working on an even sample, store it for differencing
            // with an odd sample. If odd, add to the _weights matrix.
            // Potential optimization: copying vis_even is only really
            // necessary if we've started accumulating a new frame
            if (vis_sample_num_abs % 2 == 0) {
                std::copy(input, input + 2*_num_N2_products_freqs, _vis_even.begin());
            } else {
                for (size_t d = 0; d < _num_N2_products_freqs; ++d) {
                    int32_t dr = _vis[2*d + 0] - _vis_even[2*d + 0];
                    int32_t di = _vis[2*d + 1] - _vis_even[2*d + 1];
                    _weights[d] += (dr * dr + di * di);
                } // d
            } // if even/odd

            // Track (frequency-dependent) lost samples
            for (size_t f = 0; f < _num_freq_in_frame; ++f) {

                int32_t lost_fpga_samples = frame_metadata->lost_fpga_samples[f][vis_samp_n];
                int32_t valid_fpga_samples = _n_fpga_samples_per_vis_sample - lost_fpga_samples;
                _n_valid_fpga_samples_in_vis[f] += valid_fpga_samples;

                // Track the lost samples needed for the weights too.
                if (vis_sample_num_abs % 2 == 0) {
                    _n_valid_fpga_samples_in_vis_even[f] = valid_fpga_samples;
                } else {
                    float samples_diff = valid_fpga_samples - _n_valid_fpga_samples_in_vis_even[f];
                    _n_valid_sample_diff_sq_sum[f] += samples_diff*samples_diff;
                } // if even/odd
            } // f
            vis_samples_in_out_frame++;

        } // t (vis samples in frame)

        // Advance to the next frame
        in_buf->mark_frame_empty(unique_name, in_frame_id++);
    }
}

bool N2Accumulate::output_and_reset( N2::frameID &in_frame_id, N2::frameID &out_frame_id )
{
    // Different frame for each frequency
    // But, same metadata
    std::shared_ptr<chordMetadata> chord_frame_metadata = get_chord_metadata(in_buf, in_frame_id);

    // Loop over frequency
    for (size_t f = 0; f < _num_freq_in_frame; ++f) {

        if (out_buf->wait_for_empty_frame(unique_name, out_frame_id) == nullptr) {
            return false;
        }

        DEBUG("Allocating metadata.");
        std::shared_ptr<N2Metadata> meta = alloc_N2_from_chord_metadata(in_buf, in_frame_id,
            out_buf, out_frame_id, config, unique_name, f);
        DEBUG("Creating N2FrameView.");
        N2FrameView out_vis(out_buf, out_frame_id);

        // Sample numbers for normalizing weights
        DEBUG("Computing normalization.");
        float ns = _n_valid_fpga_samples_in_vis[f]; // ns = "number of samples"
        float ins = (ns != 0.0) ? (1.0 / ns) : 0.0;

        // Copy data into buffer.
        // This requires changing from the GPU's blocked format to the triangular format visBuffer expects.
        for (size_t i = 0; i < _num_elements; ++i) {
            for (size_t j = i; j < _num_elements; ++j) {
                size_t d_N2 = i*(_num_elements) + j; // index in the input N2/GPU matrix
                size_t d_accum = N2::cmap(i, j, _num_elements); // index in the output vis matrix

                // Populate the visibility matrix
                N2::cfloat v = {(float)_vis[f*2*_num_N2_products + 2*d_N2 + 1], (float)_vis[f*2*_num_N2_products + 2*d_N2 + 0]}; // TODO: conjugate or no? What does downstream expect?
                out_vis.vis[d_accum] = ins*v;

                // de-bias and populate the weights matrix (with the inverse variance)
                _weights[f*_num_N2_products + d_N2] -= std::norm(v) * _n_valid_sample_diff_sq_sum[f] / ns / ns;
                out_vis.weight[d_accum] = ns*ns / _weights[f*_num_N2_products + d_accum];
            }
        }

        out_buf->mark_frame_full(unique_name, out_frame_id++);
    }

    DEBUG("Wrapping up accumulation buffer output copy.");

    std::fill(_vis.begin(), _vis.end(), 0);
    std::fill(_weights.begin(), _weights.end(), 0);
    std::fill(_n_valid_fpga_samples_in_vis.begin(), _n_valid_fpga_samples_in_vis.end(), 0);
    std::fill(_n_valid_sample_diff_sq_sum.begin(), _n_valid_sample_diff_sq_sum.end(), 0);

    return true;
}

