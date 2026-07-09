#include "dropAllFrames.hpp"

#include "Config.hpp"          // for Config
#include "DataType.hpp"        // for DataType
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata, metadata_is_chord, CHORD_META_MAX_DIM, CHO...
#include "kotekanLogging.hpp"  // for FATAL_ERROR, DEBUG, INFO
#include "metadata.hpp"        // for metadataObject
#include "visUtil.hpp"         // for frameID, modulo
#include "waitingForMaxFrames.hpp" // for waiting_for_max_frames

#include "fmt.hpp" // for compile_string_to_view

#include <assert.h>   // for assert
#include <cstdlib>    // for abort, size_t
#include <functional> // for bind, function
#ifdef WITH_OMP
#include <omp.h>      // for omp_get_wtime
#endif
#include <random>     // for uniform_int_distribution, mt19937
#include <stdint.h>   // for int32_t, uint32_t, uint64_t, int64_t
#include <utility>    // for swap
#include <vector>     // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(dropAllFrames);

dropAllFrames::dropAllFrames(Config& config, const std::string& unique_name,
                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&dropAllFrames::main_thread, this)),
    in_buf(get_buffer("in_buf")),
    max_frames(config.get_default<int>(unique_name, "max_frames", -1)) {

    if (max_frames >= 0)
        ++waiting_for_max_frames;

    in_buf->register_consumer(unique_name);
}

void dropAllFrames::main_thread() {

    frameID frame_id(in_buf);
    int num_frames = 0;

    while (!stop_thread) {
        // grab frame
        uint8_t* buff = (uint8_t*)in_buf->wait_for_full_frame(unique_name, frame_id);
        if (buff == nullptr)
            break;

        in_buf->mark_frame_empty(unique_name, frame_id++);

        num_frames++;

        if (max_frames >= 0 && num_frames >= max_frames) {
            INFO("Ate the requested number of frames ({:d}) - exiting", num_frames);
            break;
        }
    }

    if (max_frames >= 0) {
        // Unregister to allow the pipeline to continue, unless I'm the last
        // consumer on this buffer.
        in_buf->unregister_consumer(unique_name, true);

        if (--waiting_for_max_frames == 0) {
            WARN("Shutting down Kotekan");
            exit_kotekan(CLEAN_EXIT);
        }
    }

    DEBUG("exiting");
}
