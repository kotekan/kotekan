#include "constDataCheck.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"          // for Buffer, mark_frame_empty, register_consumer, wait_for_ful...
#include "bufferContainer.hpp" // for bufferContainer
#include "errors.h"            // for TEST_PASSED
#include "kotekanLogging.hpp"  // for DEBUG, FATAL_ERROR, INFO

#include <atomic>     // for atomic_bool
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error


REGISTER_KOTEKAN_STAGE(constDataCheck);

constDataCheck::constDataCheck(kotekan::Config& config, const std::string& unique_name,
                               kotekan::bufferContainer& buffer_container) :
    kotekan::Stage(config, unique_name, buffer_container,
                   std::bind(&constDataCheck::main_thread, this)) {

    buf = get_buffer("in_buf");
    buf->register_consumer(unique_name);
    ref_real = config.get<std::vector<int32_t>>(unique_name, "real");
    ref_imag = config.get<std::vector<int32_t>>(unique_name, "imag");
    num_frames_to_test = config.get_default<int32_t>(unique_name, "num_frames_to_test", 0);
}

constDataCheck::~constDataCheck() {}

void constDataCheck::main_thread() {

    int frame_id = 0;
    uint8_t* frame = nullptr;
    int num_errors = 0;

    int framect = 0;

    while (!stop_thread) {

        frame = wait_for_full_frame(buf, unique_name.c_str(), frame_id);
        if (frame == nullptr)
            break;

        DEBUG("constDataCheck: Got buffer {:s}[{:d}]", buf->buffer_name, frame_id);

        bool error = false;
        num_errors = 0;
        int rfr = ref_real[framect % ref_real.size()];
        int rfi = ref_imag[framect % ref_imag.size()];

        for (uint32_t i = 0; i < buf->frame_size / sizeof(int32_t); i += 2) {

            int32_t imag = *((int32_t*)&(frame[i * sizeof(int32_t)]));
            int32_t real = *((int32_t*)&(frame[(i + 1) * sizeof(int32_t)]));

            if (real != rfr || imag != rfi) {
                if (num_errors++ < 1000)
                    FATAL_ERROR("{:s}[{:d}][{:d}] != {:d} + {:d}i; actual value: {:d} + {:d}i",
                                buf->buffer_name, frame_id, i / 2, rfr, rfi, real, imag);
                error = true;
            }
        }

        if (!error)
            INFO("The buffer {:s}[{:d}] passed all checks; contains all ({:d} + {:d}i)",
                 buf->buffer_name, frame_id, rfr, rfi);
        //                    ref_real, ref_imag);

        mark_frame_empty(buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % buf->num_frames;
        framect++;

        if (num_frames_to_test == framect)
            TEST_PASSED();
    }
}
