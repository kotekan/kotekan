#include "fakeVis.hpp"
#include "visBuffer.hpp"
#include "visUtil.hpp"
#include "chimeMetadata.h"
#include <csignal>
#include <time.h>
#include <math.h>
#include <random>
#include <functional>
#include "datasetManager.hpp"
#include "fmt.hpp"


using namespace std::placeholders;


REGISTER_KOTEKAN_PROCESS(fakeVis);
REGISTER_KOTEKAN_PROCESS(replaceVis);


fakeVis::fakeVis(Config &config,
                 const string& unique_name,
                 bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&fakeVis::main_thread, this)) {

    // Fetch any simple configuration
    num_elements = config.get_int(unique_name, "num_elements");
    block_size = config.get_int(unique_name, "block_size");
    num_eigenvectors =  config.get_int(unique_name, "num_ev");

    // Get the output buffer
    std::string buffer_name = config.get_string(unique_name, "out_buf");

    // Fetch the buffer, register it
    out_buf = buffer_container.get_buffer(buffer_name);
    register_producer(out_buf, unique_name.c_str());

    // Get frequency IDs from config
    freq = config.get_array<uint32_t>(unique_name, "freq_ids");

    // Get fill type
    fill_map["default"] = std::bind(&fakeVis::fill_mode_default, this, _1);
    fill_map["fill_ij"] = std::bind(&fakeVis::fill_mode_fill_ij, this, _1);
    fill_map["phase_ij"] = std::bind(&fakeVis::fill_mode_phase_ij, this, _1);
    fill_map["gaussian"] = std::bind(&fakeVis::fill_mode_gaussian, this, _1);
    fill_map["gaussian_random"] = std::bind(&fakeVis::fill_mode_gaussian, this, _1);
    fill_map["chime"] = std::bind(&fakeVis::fill_mode_chime, this, _1);

    mode = config.get_string_default(unique_name, "mode", "default");

    if(fill_map.count(mode) == 0) {
        ERROR("unknown fill type %s", mode.c_str());
        // TODO: exit here
    }
    INFO("Using fill type: %s", mode.c_str());
    fill = fill_map.at(mode);

    if (mode == "gaussian" || mode == "gaussian_random") {
        vis_mean = {
            config.get_float_default(unique_name, "vis_mean_real", 0.),
            config.get_float_default(unique_name, "vis_mean_imag", 0.)
        };
        vis_std = config.get_float_default(unique_name, "vis_std", 1.);

        // initialize random number generation
        if (mode == "gaussian_random") {
            std::random_device rd;
            gen.seed(rd());
        }
    }

    // Get timing and frame params
    cadence = config.get_float(unique_name, "cadence");
    num_frames = config.get_int_default(unique_name, "num_frames", -1);
    wait = config.get_bool_default(unique_name, "wait", true);
    use_dataset_manager = config.get_bool_default(
        unique_name, "use_dataset_manager", false);

    // Get zero_weight option
    zero_weight = config.get_bool_default(unique_name, "zero_weight", false);
}

void fakeVis::apply_config(uint64_t fpga_seq) {

}

void fakeVis::main_thread() {

    unsigned int output_frame_id = 0, frame_count = 0;
    uint64_t fpga_seq = 0;

    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);

    // Calculate the time increments in seq and ctime
    uint64_t delta_seq = (uint64_t)(800e6 / 2048 * cadence);
    uint64_t delta_ns = (uint64_t)(cadence * 1000000000);

    // If configured, register datasetStates to describe the properties of the
    // created stream
    dset_id dataset = 0;
    if (use_dataset_manager) {

        auto& dm = datasetManager::instance();

        std::vector<std::pair<uint32_t, freq_ctype>> fspec;
        std::transform(
            std::begin(freq), std::end(freq), std::back_inserter(fspec),
            [] (const uint32_t& id) -> std::pair<uint32_t, freq_ctype> {
                return {id, {800.0 - 400.0 / 1024 * id, 400.0 / 1024}};
            });
        auto fstate = std::make_unique<freqState>(fspec);

        std::vector<input_ctype> ispec;
        for (uint32_t i = 0; i < num_elements; i++)
            ispec.emplace_back((uint32_t)i, fmt::format("dm_input_{}", i));
        auto istate = std::make_unique<inputState>(ispec, std::move(fstate));

        std::vector<prod_ctype> pspec;
        for (uint16_t i = 0; i < num_elements; i++)
            for (uint16_t j = i; j < num_elements; j++)
                pspec.push_back({i, j});
        auto pstate = std::make_unique<prodState>(pspec, std::move(istate));

        auto s = dm.add_state(std::move(pstate));
        dataset = dm.add_dataset(s.first, -1);  // Register a root state
    }

    while (!stop_thread) {

        double start = current_time();

        for (auto f : freq) {

            DEBUG("Making fake visBuffer for freq=%i, fpga_seq=%i", f, fpga_seq);

            // Wait for the buffer frame to be free
            if (wait_for_empty_frame(out_buf, unique_name.c_str(),
                                     output_frame_id) == nullptr) {
                break;
            }

            // Allocate metadata and get frame
            allocate_new_metadata_object(out_buf, output_frame_id);
            auto output_frame = visFrameView(out_buf, output_frame_id,
                                             num_elements, num_eigenvectors);

            output_frame.dataset_id = dataset;

            // Set the frequency index
            output_frame.freq_id = f;

            // Set the time
            output_frame.time = std::make_tuple(fpga_seq, ts);

            // Set the length and total data
            output_frame.fpga_seq_length = delta_seq;
            output_frame.fpga_seq_total = delta_seq;

            // Fill out the frame with debug info according to the given mode.
            fill(output_frame);

            // gains
            for(uint32_t i = 0; i < num_elements; i++) {
                output_frame.gain[i] = 1;
            }

            // Mark the buffers and move on
            mark_frame_full(out_buf, unique_name.c_str(),
                            output_frame_id);

            // Advance the current frame ids
            output_frame_id = (output_frame_id + 1) % out_buf->num_frames;
        }

        // Increment time
        fpga_seq += delta_seq;
        frame_count++;  // NOTE: frame count increase once for all freq

        // Increment the timespec
        ts.tv_sec += ((ts.tv_nsec + delta_ns) / 1000000000);
        ts.tv_nsec = (ts.tv_nsec + delta_ns) % 1000000000;

        // Cause kotekan to exit if we've hit the maximum number of frames
        if(num_frames > 0 && frame_count >= (unsigned) num_frames) {
            INFO("Reached frame limit [%i frames]. Exiting kotekan...", num_frames);
            std::raise(SIGINT);
            return;
        }

        // If requested sleep for the extra time required to produce a fake vis
        // at the correct cadence
        if(this->wait) {
            double diff = cadence - (current_time() - start);
            timespec ts_diff = double_to_ts(diff);
            nanosleep(&ts_diff, nullptr);
        }
    }
}


void fakeVis::fill_mode_default(visFrameView& frame)
{
    auto out_vis = frame.vis;
    // Set diagonal elements to (0, row)
    for (uint32_t i = 0; i < num_elements; i++) {
        uint32_t pi = cmap(i, i, num_elements);
        out_vis[pi] = {0., (float) i};
    }
    // Save metadata in first few cells
    if ( sizeof(out_vis) < 4 ) {
        WARN("Number of elements (%d) is too small to encode \
                debugging values in fake visibilities", num_elements);
    } else {
        // For simplicity overwrite diagonal if needed
        out_vis[0] = {(float) std::get<0>(frame.time), 0.0};
        out_vis[1] = {(float) ts_to_double(std::get<1>(frame.time)), 0.0};
        out_vis[2] = {(float) frame.freq_id, 0.};
        //out_vis[3] = {(float) output_frame_id, 0.};
    }
    fill_non_vis(frame);
}

void fakeVis::fill_mode_fill_ij(visFrameView& frame)
{
    int ind = 0;
    for(uint32_t i = 0; i < num_elements; i++) {
        for(uint32_t j = i; j < num_elements; j++) {
            frame.vis[ind] = {(float)i, (float)j};
            ind++;
        }
    }
    fill_non_vis(frame);
}

void fakeVis::fill_mode_phase_ij(visFrameView& frame)
{
    int ind = 0;
    for(uint32_t i = 0; i < num_elements; i++) {
        for(uint32_t j = i; j < num_elements; j++) {
            float phase = (float) i - (float) j;
            frame.vis[ind] = {cosf(phase), sinf(phase)};
            ind++;
        }
    }
    fill_non_vis(frame);
}

void fakeVis::fill_mode_gaussian(visFrameView& frame)
{
    // random number generation for gaussian modes
    std::normal_distribution<float> gauss_real{vis_mean.real(), vis_std};
    std::normal_distribution<float> gauss_imag{vis_mean.imag(), vis_std};
    std::normal_distribution<float> gauss(0.1 * vis_std, 0.1 * vis_std);

    // Fill vis
    int ind = 0;
    for(uint32_t i = 0; i < num_elements; i++) {
        for(uint32_t j = i; j < num_elements; j++) {
            frame.vis[ind] = {gauss_real(gen), gauss_imag(gen)};
            ind++;
        }
    }

    // Fill ev
    for (uint32_t i = 0; i < num_eigenvectors; i++) {
        for (uint32_t j = 0; j < num_elements; j++) {
            int k = i * num_elements + j;
            frame.evec[k] = {(float)i, gauss_real(gen)};
        }
        frame.eval[i] = i;
    }
    frame.erms = gauss_real(gen);

    // // Fill weights
    // std::default_random_engine gen;
    // if (mode == "gaussian_random") {
    //     std::random_device rd;
    //     gen.seed(rd());
    // }
    // generate vaguely realistic weights
    ind = 0;
    for(uint32_t i = 0; i < num_elements; i++) {
        for(uint32_t j = i; j < num_elements; j++) {
            frame.weight[ind] = 1. / pow(gauss(gen), 2);
            ind++;
        }
    }

    // Set flags and gains
    std::fill(frame.flags.begin(), frame.flags.end(), 1.0);
    std::fill(frame.gain.begin(), frame.gain.end(), 1.0);
}

void fakeVis::fill_mode_chime(visFrameView& frame)
{
    int ind = 0;
    for(uint32_t i = 0; i < num_elements; i++) {
        for(uint32_t j = i; j < num_elements; j++) {
            int cyl_i = i / 512;
            int cyl_j = j / 512;

            int pos_i = i % 256;
            int pos_j = j % 256;

            frame.vis[ind] = {(float)(cyl_i - cyl_j), (float)(pos_i - pos_j)};
            ind++;
        }
    }
    fill_non_vis(frame);
}

void fakeVis::fill_non_vis(visFrameView& frame)
{
    // Set ev section
    for (uint32_t i = 0; i < num_eigenvectors; i++) {
        for (uint32_t j = 0; j < num_elements; j++) {
            int k = i * num_elements + j;
            frame.evec[k] = {(float)i, (float)j};
        }
        frame.eval[i] = i;
    }
    frame.erms = 1.0;

    // Set weights
    int ind = 0;
    const float weight_fill = zero_weight ? 0.0 : 1.0;
    for(uint32_t i = 0; i < num_elements; i++) {
        for(uint32_t j = i; j < num_elements; j++) {
            frame.weight[ind] = weight_fill;
            ind++;
        }
    }

    // Set flags and gains
    std::fill(frame.flags.begin(), frame.flags.end(), 1.0);
    std::fill(frame.gain.begin(), frame.gain.end(), 1.0);

}

replaceVis::replaceVis(Config& config,
                       const string& unique_name,
                       bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&replaceVis::main_thread, this)) {

    // Setup the input buffer
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Setup the output buffer
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());
}

void replaceVis::apply_config(uint64_t fpga_seq) {
}

void replaceVis::main_thread() {

    unsigned int output_frame_id = 0;
    unsigned int input_frame_id = 0;

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if(wait_for_full_frame(in_buf, unique_name.c_str(),
                               input_frame_id) == nullptr) {
            break;
        }

        // Wait for the output buffer to be empty of data
        if(wait_for_empty_frame(out_buf, unique_name.c_str(),
                                output_frame_id) == nullptr) {
            break;
        }
        // Create view to input frame
        auto input_frame = visFrameView(in_buf, input_frame_id);

        // Copy input frame to output frame and create view
        allocate_new_metadata_object(out_buf, output_frame_id);
        auto output_frame = visFrameView(out_buf,
                                         output_frame_id, input_frame);

        for(uint32_t i = 0; i < output_frame.num_prod; i++) {
            float real = (i % 2 == 0 ?
                          output_frame.freq_id :
                          std::get<0>(output_frame.time));
            float imag = i;

            output_frame.vis[i] = {real, imag};
        }


        // Mark the output buffer and move on
        mark_frame_full(out_buf, unique_name.c_str(), output_frame_id);

        // Mark the input buffer and move on
        mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id);

        // Advance the current frame ids
        output_frame_id = (output_frame_id + 1) % out_buf->num_frames;
        input_frame_id = (input_frame_id + 1) % in_buf->num_frames;
    }
}
