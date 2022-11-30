#include "chrxUplink.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for Buffer, mark_frame_empty, register_consumer, wait_for_ful...
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for ERROR, INFO

#include "fmt.hpp" // for format, fmt

#include <arpa/inet.h>  // for inet_addr
#include <atomic>       // for atomic_bool
#include <errno.h>      // for errno
#include <exception>    // for exception
#include <functional>   // for _Bind_helper<>::type, bind, function
#include <netinet/in.h> // for sockaddr_in, htons, in_addr
#include <regex>        // for match_results<>::_Base_type
#include <strings.h>    // for bzero
#include <sys/socket.h> // for send, connect, socket, AF_INET, SOCK_STREAM
#include <unistd.h>     // for gethostname, ssize_t
#include <vector>       // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using std::string;

REGISTER_KOTEKAN_STAGE(chrxUplink);

chrxUplink::chrxUplink(Config& config, const std::string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&chrxUplink::main_thread, this)) {

    vis_buf = get_buffer("chrx_in_buf");
    register_consumer(vis_buf, unique_name.c_str());
    gate_buf = get_buffer("gate_in_buf");
    register_consumer(gate_buf, unique_name.c_str());
}

chrxUplink::~chrxUplink() {}

// TODO make this more robust to network errors.
void chrxUplink::main_thread() {

    // Apply config.
    char hostname[1024];
    std::string s_port;

    _collection_server_ip = config.get<std::string>(unique_name, "collection_server_ip");
    gethostname(hostname, 1024);

    std::string s_hostname(hostname);
    std::string lastNum = s_hostname.substr(s_hostname.length() - 2, 2);
    s_port = fmt::format(fmt("410{:s}"), lastNum);

    _collection_server_port =
        stoi(s_port); // config.get<int32_t>(unique_name, "collection_server_port");
    _enable_gating = config.get<bool>(unique_name, "enable_gating");

    int buffer_ID = 0;
    uint8_t* vis_frame = nullptr;
    uint8_t* gate_frame = nullptr;

    // Connect to server.
    struct sockaddr_in ch_acq_addr;

    int tcp_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (tcp_fd == -1) {
        ERROR("Could not create socket, errno: {:d}", errno);
        return;
    }

    bzero(&ch_acq_addr, sizeof(ch_acq_addr));
    ch_acq_addr.sin_family = AF_INET;
    ch_acq_addr.sin_addr.s_addr = inet_addr(_collection_server_ip.c_str());
    ch_acq_addr.sin_port = htons(_collection_server_port);

    if (connect(tcp_fd, (struct sockaddr*)&ch_acq_addr, sizeof(ch_acq_addr)) == -1) {
        ERROR("Could not connect to collection server, error: {:d}", errno);
    }
    INFO("Connected to collection server on: {:s}:{:d}", _collection_server_ip,
         _collection_server_port);

    // Wait for, and transmit, full buffers.
    while (!stop_thread) {

        // This call is blocking!
        vis_frame = wait_for_full_frame(vis_buf, unique_name.c_str(), buffer_ID);
        if (vis_frame == nullptr)
            break;

        // INFO("Sending TCP frame to ch_master. frame size: {:d}", vis_buf->frame_size);

        ssize_t bytes_sent = send(tcp_fd, vis_frame, vis_buf->frame_size, 0);
        if (bytes_sent <= 0) {
            ERROR("Could not send frame to chrx, error: {:d}", errno);
            break;
        }
        if ((size_t)bytes_sent != vis_buf->frame_size) {
            ERROR("Could not send all bytes: bytes sent = {:d}; frame_size = {:d}", (int)bytes_sent,
                  vis_buf->frame_size);
            break;
        }
        INFO("Finished sending frame to chrx");

        if (_enable_gating) {
            //            DEBUG("Getting gated buffer");
            gate_frame = wait_for_full_frame(gate_buf, unique_name.c_str(), buffer_ID);
            if (gate_frame == nullptr)
                break;

            //            DEBUG("Sending gated buffer");
            bytes_sent = send(tcp_fd, gate_frame, gate_buf->frame_size, 0);
            if (bytes_sent <= 0) {
                ERROR("Could not send gated date frame to ch_acq, error: {:d}", errno);
                break;
            }
            if ((size_t)bytes_sent != gate_buf->frame_size) {
                ERROR("Could not send all bytes in gated data frame: bytes sent = {:d}; frame_size "
                      "= {:d}",
                      (int)bytes_sent, gate_buf->frame_size);
                break;
            }
            INFO("Finished sending gated data frame to chrx");
            mark_frame_empty(gate_buf, unique_name.c_str(), buffer_ID);
        }

        mark_frame_empty(vis_buf, unique_name.c_str(), buffer_ID);

        buffer_ID = (buffer_ID + 1) % vis_buf->num_frames;
    }
}
