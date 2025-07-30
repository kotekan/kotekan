#include "N2FringeStop.hpp"

#include "CHORDTelescope.hpp"    // for CHORDTelescope
#include "Config.hpp"            // for Config
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"            // for mark_frame_empty, allocate_new_metadata_object, mark_fr...
#include "bufferContainer.hpp"   // for bufferContainer
#include "kotekanLogging.hpp"    // for DEBUG
#include "prometheusMetrics.hpp" // for Counter, MetricFamily, Metrics
// #include "visBuffer.hpp"         // for VisFrameView
#include "N2FrameView.hpp"
#include "timeUtil.hpp" //for get_UT1_from_ERA
#include "visUtil.hpp"  // for frameID, modulo, cfloat, operator-, ts_to_double

#include "gsl-lite.hpp" // for span

#include <atomic>     // for atomic_bool
#include <complex>    // for complex
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error
#include <stdint.h>   // for uint32_t, uint64_t, int32_t
#include <time.h>     // for timespec
#include <tuple>      // for get
#include <vector>     // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

#define GIGA 1'000'000'000L

REGISTER_KOTEKAN_STAGE(N2FringeStop);

N2FringeStop::N2FringeStop(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&N2FringeStop::main_thread, this)) {

    // Fetch the buffers, register
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    fringestop_mode = config.get_default<int>(unique_name, "fringestop_mode", 1);
    num_rot_target = config.get_default<int>(unique_name, "num_rot_target", 9000);
    era_target_deg = config.get_default<double>(unique_name, "era_target_deg", 0.0);
    xp_target_as = config.get_default<double>(unique_name, "xp_target_as", 0.0);
    yp_target_as = config.get_default<double>(unique_name, "yp_target_as", 0.0);

    num_elements = 0;
    nprod = num_elements * (num_elements + 1) / 2;
}

void N2FringeStop::main_thread() {

    frameID frame_id(in_buf);
    frameID output_frame_id(out_buf);

    const CHORDTelescope& tel = Telescope::instance().cast<CHORDTelescope>();

    int num_dishes = tel.get_num_dishes();
    std::vector<std::complex<double>> fringe_phase(num_dishes, 1.0);

    int64_t ut1 = get_UT1_from_ERA(num_rot_target, era_target_deg);

    struct EOP eop_target = {.t_inst = ut1,
                             .t_ut1 = ut1,
                             .delta_UT1_inst = 0.0,
                             .ERA_deg = era_target_deg,
                             .xp_as = xp_target_as,
                             .yp_as = yp_target_as};


    while (!stop_thread) {
        // Wait for the buffer to be filled with data
        if ((in_buf->wait_for_full_frame(unique_name, frame_id)) == nullptr) {
            break;
        }

        N2FrameView frame(in_buf, frame_id);

        DEBUG("Input frame - num_elements: {:d}", frame.num_elements);

        size_t num_elements = frame.num_elements;

        DEBUG("ERA: {:f}; ERA_target: {:f}", frame.eop.ERA_deg, era_target_deg);

        struct EOP eop = frame.eop;

        // Wait for an empty frame
        if (out_buf->wait_for_empty_frame(unique_name, output_frame_id) == nullptr) {
            break;
        }

        // Copy frame into output buffer
        auto output_frame = N2FrameView::copy_frame(in_buf, frame_id, out_buf, output_frame_id);


        // Set the target EOP.
        output_frame.eop = eop_target;

        if (fringestop_mode > 0)
            tel.fringestop_phases_1d(frame.freq_Hz, eop, eop_target, fringe_phase);

        size_t idx = 0;
        for (size_t i = 0; i < num_elements; i++) {
            for (size_t j = i; j < num_elements; j++) {

                size_t d_i = i % num_dishes;
                size_t d_j = j % num_dishes;
                if (fringestop_mode == 2)
                    output_frame.vis[idx] = fringe_phase[d_i] * std::conj(fringe_phase[d_j]);
                else
                    output_frame.vis[idx] *= fringe_phase[d_i] * std::conj(fringe_phase[d_j]);

                idx++;
            }
        }

        // Go to next frame
        in_buf->mark_frame_empty(unique_name, frame_id++);

        DEBUG("Output frame - num_elements: {:d}", output_frame.num_elements);
        // mark as full
        out_buf->mark_frame_full(unique_name, output_frame_id++);
    }
}
