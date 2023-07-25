#include "recvSingleDishVDIF.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"            // for Buffer, mark_frame_full, register_producer, wait_for_empt...
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for INFO

#include <atomic>     // for atomic_bool
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <regex>      // for match_results<>::_Base_type
#include <stddef.h>   // for size_t
#include <vector>     // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(recvSingleDishVDIF);

recvSingleDishVDIF::recvSingleDishVDIF(Config& config, const std::string& unique_name,
                                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&recvSingleDishVDIF::main_thread, this)) {

    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    // Apply config.
    num_freq = config.get<int>(unique_name, "num_freq");
    orig_port = config.get<uint32_t>(unique_name, "orig_port");
    orig_ip = config.get<std::string>(unique_name, "orig_ip");
}

recvSingleDishVDIF::~recvSingleDishVDIF() {}

void recvSingleDishVDIF::main_thread() {

    int file_num = 0;
    int frame_id = 0;
    uint8_t* frame = nullptr;

    const int vdif_header_len = 32;

    // The *8 is to reduce the number of frames by sending a few per packet
    const int32_t packet_size = (vdif_header_len + num_freq);


    while (!stop_thread) {
        // Get an empty buffer to write into
        frame = wait_for_empty_frame(out_buf, unique_name.c_str(), frame_id);
        if (frame == nullptr)
            break;

        // Send data to remote server.
        // TODO rate limit this output
        for (size_t i = 0; i < out_buf->frame_size / packet_size; ++i) {
            /*
                        int bytes_recvd = sendto(socket_fd,
                                         (void *)&frame[packet_size*i],
                                         packet_size, 0,
                                         (struct sockaddr *)&saddr_remote, saddr_len);

                        if (bytes_sent == -1) {
                            ERROR("Cannot set VDIF packet");
                            break;
                        }

                        if (bytes_sent != packet_size) {
                            ERROR("Did not send full vdif packet.");
                        }
            */
        }

        INFO("recvSingleDishVDIF: marking buffer {:s}[{:d}] as full", out_buf->buffer_name,
             frame_id);
        mark_frame_full(out_buf, unique_name.c_str(), frame_id);

        file_num++;
        frame_id = (frame_id + 1) % out_buf->num_frames;
    }

    // Comment while support for finishing is not yet working.
    // mark_producer_done(&out_buf, 0);
}
