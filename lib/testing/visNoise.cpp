#include "visNoise.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for mark_frame_empty, mark_frame_full, register_consumer, reg...
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for INFO
#include "VisFrameView.hpp"       // for VisFrameView
#include "visUtil.hpp"         // for cfloat

#include "gsl-lite.hpp" // for span

#include <atomic>     // for atomic_bool
#include <cmath>      // for pow
#include <complex>    // for complex
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for invalid_argument, runtime_error
#include <stdint.h>   // for uint32_t
#include <vector>     // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(visNoise);

visNoise::visNoise(Config& config, const std::string& unique_name,
                   bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&visNoise::main_thread, this)) {

    // Setup the buffers
    buf_in = get_buffer("in_buf");
    register_consumer(buf_in, unique_name.c_str());
    buf_out = get_buffer("out_buf");
    register_producer(buf_out, unique_name.c_str());

    _num_elements = config.get<size_t>(unique_name, "num_elements");
    _num_eigenvectors = config.get<size_t>(unique_name, "num_ev");

    _standard_deviation = config.get_default<float>(unique_name, "standard_deviation", 1.0);
    if (_standard_deviation < 0) {
        throw std::invalid_argument("visNoise: standard deviation has to be "
                                    "positive (is "
                                    + std::to_string(_standard_deviation) + ").");
    }

    if (config.get_default<bool>(unique_name, "random", false)) {
        std::random_device rd;
        gen.seed(rd());
        INFO("visNoise: random seed used for init of noise generation.");
    }

    INFO("visNoise: producing gaussian noise with standard deviation {:f}.", _standard_deviation);
}

void visNoise::main_thread() {

    uint32_t frame_id_in = 0;
    uint32_t frame_id_out = 0;

    // random number generation
    std::normal_distribution<float> gauss_vis{0, _standard_deviation};
    std::normal_distribution<float> gauss_weight(0.1 * _standard_deviation,
                                                 0.1 * _standard_deviation);

    while (!stop_thread) {
        // Wait for data in the input buffer
        if ((wait_for_full_frame(buf_in, unique_name.c_str(), frame_id_in)) == nullptr) {
            break;
        }

        // Wait for space in the output buffer
        if (wait_for_empty_frame(buf_out, unique_name.c_str(), frame_id_out) == nullptr) {
            break;
        }
        // Copy frame into output buffer
        auto frame = VisFrameView::copy_frame(buf_in, frame_id_in, buf_out, frame_id_out);

        // Add noise to visibilities
        int ind = 0;
        for (uint32_t i = 0; i < _num_elements; i++) {
            for (uint32_t j = i; j < _num_elements; j++) {
                frame.vis[ind] += std::complex<float>{gauss_vis(gen), gauss_vis(gen)};
                ind++;
            }
        }

        // Add noise to eigenvectors
        for (uint32_t i = 0; i < _num_eigenvectors; i++) {
            for (uint32_t j = 0; j < _num_elements; j++) {
                int k = i * _num_elements + j;
                frame.evec[k] += std::complex<float>{0, gauss_vis(gen)};
            }
        }
        frame.erms += gauss_vis(gen);

        // generate vaguely realistic weights
        ind = 0;
        for (uint32_t i = 0; i < _num_elements; i++) {
            for (uint32_t j = i; j < _num_elements; j++) {
                frame.weight[ind] /= pow(gauss_weight(gen), 2);
                ind++;
            }
        }

        // Mark output frame full and input frame empty
        mark_frame_full(buf_out, unique_name.c_str(), frame_id_out);
        mark_frame_empty(buf_in, unique_name.c_str(), frame_id_in);
        // Move forward one frame
        frame_id_out = (frame_id_out + 1) % buf_out->num_frames;
        frame_id_in = (frame_id_in + 1) % buf_in->num_frames;
    }
}
