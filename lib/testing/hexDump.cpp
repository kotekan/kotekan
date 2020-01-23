#include "hexDump.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for mark_frame_empty, register_consumer, wait_for_full_frame
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for DEBUG
#include "util.h"              // for hex_dump

#include <atomic>            // for atomic_bool
#include <cstdint>           // for int32_t
#include <exception>         // for exception
#include <regex>             // for match_results<>::_Base_type
#include <stdexcept>         // for runtime_error
#include <utils/visUtil.hpp> // for frameID, modulo
#include <vector>            // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(hexDump);

STAGE_CONSTRUCTOR(hexDump) {

    // Register as consumer on buffer
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Get some configuration options
    _len = config.get_default<int32_t>(unique_name, "len", 128);
    _offset = config.get_default<int32_t>(unique_name, "offset", 0);

    // Check that we won't read past the end
    if (_offset + _len > in_buf->frame_size) {
        throw std::runtime_error("HexDump: cannot print past end of buffer");
    }
}

hexDump::~hexDump() {}

void hexDump::main_thread() {

    frameID frame_id(in_buf);

    while (!stop_thread) {

        uint8_t* frame = wait_for_full_frame(in_buf, unique_name.c_str(), frame_id);
        if (frame == nullptr)
            break;

        DEBUG("hexDump: Got buffer {:s}[{:d}]", in_buf->buffer_name, frame_id);

        // Prints the hex data to screen
        hex_dump(16, (void*)&frame[_offset], _len);

        mark_frame_empty(in_buf, unique_name.c_str(), frame_id);
        frame_id++;
    }
}
