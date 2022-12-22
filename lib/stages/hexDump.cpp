#include "hexDump.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for mark_frame_empty, register_consumer, wait_for_full_frame
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for DEBUG
#include "util.h"              // for hex_dump

#include <atomic>      // for atomic_bool
#include <cstdint>     // for int32_t
#include <exception>   // for exception
#include <regex>       // for match_results<>::_Base_type
#include <stddef.h>    // for size_t
#include <stdexcept>   // for runtime_error
#include <vector>      // for vector
#include <visUtil.hpp> // for frameID, modulo


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
    if ((size_t)(_offset + _len) > in_buf->frame_size) {
        throw std::runtime_error("HexDump: cannot print past end of buffer");
    }
}

hexDump::~hexDump() {}

#include "unistd.h"
#include "chimeMetadata.hpp"

void hexDump::main_thread() {

    frameID frame_id(in_buf);

    while (!stop_thread) {

        uint8_t* frame = wait_for_full_frame(in_buf, unique_name.c_str(), frame_id);
        if (frame == nullptr)
            break;

        DEBUG("hexDump: Got buffer {:s}[{:d}]", in_buf->buffer_name, frame_id);

        // Prints the hex data to screen
        hex_dump(16, (void*)&frame[_offset], _len);

		sleep(1);
		
        DEBUG("hexDump: Releasing buffer {:s}[{:d}] with metadata 0x{:x} and FPGA seq {:d}",
			  in_buf->buffer_name, frame_id,
			  (long)in_buf->metadata[frame_id],
			  get_fpga_seq_num(in_buf, frame_id));

        mark_frame_empty(in_buf, unique_name.c_str(), frame_id);
        frame_id++;
    }
}
