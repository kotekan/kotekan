#include "networkPowerStream.hpp"

#include <arpa/inet.h>          // for htons, inet_addr, inet_aton
#include <netinet/in.h>         // for sockaddr_in, IPPROTO_TCP, IPPROTO_UDP, in_addr
#include <stdlib.h>             // for free, malloc
#include <string.h>             // for memcpy, memset
#include <sys/socket.h>         // for send, AF_INET, socket, MSG_NOSIGNAL, connect, sendto, set...
#include <sys/time.h>           // for timeval, gettimeofday
#include <sys/types.h>          // for uint
#include <unistd.h>             // for close
#include <functional>           // for bind, function
#include <string>               // for allocator, basic_string, operator==, char_traits, string
#include <memory>               // for shared_ptr
#include <stdexcept>            // for runtime_error

#include "Config.hpp"           // for Config
#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE
#include "airspyFrameDesc.hpp"  // for make_power_corr_desc
#include "buffer.hpp"           // for Buffer
#include "bufferContainer.hpp"  // for bufferContainer
#include "kotekanLogging.hpp"   // for ERROR, INFO
#include "fmt.hpp"              // for compile_string_to_view
#include "NDArray.hpp"          // for GenericNDArray

#ifndef MSG_NOSIGNAL
// macOS uses SO_NOSIGPIPE on the socket instead (set up below); make sendto() portable.
#define MSG_NOSIGNAL 0
#endif


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(networkPowerStream);

// Read a per-element config value: either a scalar (broadcast to every element)
// or a length-num_elements integer array (one value per element). Used for
// ``freq`` / ``sample_bw`` so a single stream can describe several sub-bands.
// Values are returned scaled by ``scale`` (e.g. 1e6 for MHz -> Hz); array entries
// are integers (integer MHz is plenty for this metadata), a scalar may be float.
static std::vector<float> read_per_elem(Config& config, const std::string& base,
                                        const std::string& key, int nelem, float scale,
                                        float dflt) {
    std::vector<float> out;
    try {
        std::vector<int> v = config.get<std::vector<int>>(base, key);
        for (int x : v)
            out.push_back((float)x * scale);
    } catch (...) {
        out.assign(1, config.get_default<float>(base, key, dflt) * scale);
    }
    if ((int)out.size() == 1)
        out.assign(nelem, out[0]); // broadcast a scalar to all elements
    if ((int)out.size() != nelem)
        throw std::runtime_error("networkPowerStream: '" + key
                                 + "' must be a scalar or a length-num_elements array");
    return out;
}

networkPowerStream::networkPowerStream(Config& config, const std::string& unique_name,
                                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&networkPowerStream::main_thread, this)) {

    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);

    // PER BUFFER
    times = config.get<int>(unique_name, "samples_per_data_set")
            / config.get<int>(unique_name, "power_integration_length");
    freqs = config.get<int>(unique_name, "num_freq");
    elems = config.get<int>(unique_name, "num_elements");

    // Per integration the upstream emits [freqs float32 bins, 1 uint32 count]
    // per element; we expect ``times`` integrations packed into one frame.
    // The descriptor folds the count word into the float32 array (see
    // airspyFrameDesc.hpp). When ensure_frame_desc has already been called by
    // the producer (simpleAutocorr / SimpleCrosscorr), this becomes a
    // cross-check via FrameDesc::operator==; otherwise it just records
    // the expected layout for any later consumer/producer.
    in_buf->ensure_frame_desc(kotekan_airspy::make_power_corr_desc(elems * times, freqs));

    // Per-element band centre / bandwidth (scalar broadcasts to all elements).
    freq0s = read_per_elem(config, unique_name, "freq", elems, 1e6f, 600.f);
    sample_bws = read_per_elem(config, unique_name, "sample_bw", elems, 1e6f, 200.f);
    // Per-element Stokes/pol id; default XX, YY, XY, YX, ... (-5, -6, -7, ...).
    std::vector<int> st;
    try {
        st = config.get<std::vector<int>>(unique_name, "stokes");
    } catch (...) {
    }
    if (st.empty()) {
        st.resize(elems);
        for (int e = 0; e < elems; e++)
            st[e] = -5 - e;
    } else if ((int)st.size() == 1) {
        st.assign(elems, st[0]);
    }
    if ((int)st.size() != elems)
        throw std::runtime_error("networkPowerStream: 'stokes' must be a scalar "
                                 "or a length-num_elements array");
    stokes_ids = st;
    freq0 = freq0s[0]; // kept for logging / back-compat
    sample_bw = sample_bws[0];

    dest_port = config.get<uint32_t>(unique_name, "destination_port");
    dest_server_ip = config.get<std::string>(unique_name, "destination_ip");
    dest_protocol = config.get<std::string>(unique_name, "destination_protocol");

    atomic_flag_clear(&socket_lock);

    header.protocol_version = 2;                  // per-element freq map + stokes
    header.packet_length = freqs * sizeof(float);
    header.header_length = sizeof(IntensityPacketHeader);
    header.samples_per_packet = freqs;
    header.sample_type = 4; // uint32
    // Timing only: magnitude of the (per-element, all equal) bandwidth. Band
    // direction / inversion is now carried per element in the freq map below.
    float bw0 = sample_bws[0] < 0 ? -sample_bws[0] : sample_bws[0];
    header.raw_cadence = 1 / (bw0 / freqs); // 2.56e-6
    header.num_freqs = freqs;
    header.num_elems = elems;
    header.samples_summed = config.get<int>(unique_name, "power_integration_length");
    header.handshake_idx = -1;
    header.handshake_utc = -1;

    frame_idx = 0;

    // Prevent SIGPIPE on send failure.
    // This is used for MacOS, since linux doesn't have SO_NOSIGPIPE
#ifdef SO_NOSIGPIPE
    int set = 1;
    if (setsockopt(socket_fd, SOL_SOCKET, SO_NOSIGPIPE, (void*)&set, sizeof(int)) < 0) {
        ERROR("bufferSend: setsockopt() NOSIGPIPE ");
    }
#endif
}

networkPowerStream::~networkPowerStream() {}

void networkPowerStream::main_thread() {
    int frame_id = 0;
    uint8_t* frame = nullptr;
    uint packet_length = freqs * sizeof(float) + sizeof(IntensityPacketHeader);
    void* packet_buffer = malloc(packet_length);
    IntensityPacketHeader* packet_header = (IntensityPacketHeader*)packet_buffer;
    float* local_data = (float*)((char*)packet_buffer + sizeof(IntensityPacketHeader));
    struct timeval tv;

    if (dest_protocol == "UDP") {
        // UDP variables
        struct sockaddr_in saddr_remote; /* the libc network address data structure */
        socket_fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (socket_fd == -1) {
            ERROR("Could not create UDP socket for output stream");
            return;
        }

        memset((char*)&saddr_remote, 0, sizeof(sockaddr_in));
        saddr_remote.sin_family = AF_INET;
        saddr_remote.sin_port = htons(dest_port);
        if (inet_aton(dest_server_ip.c_str(), &saddr_remote.sin_addr) == 0) {
            ERROR("Invalid address given for remote server");
            return;
        }
        INFO("{:d} {:s}", dest_port, dest_server_ip);

        while (!stop_thread) {
            // Wait for a full buffer.
            frame = in_buf->wait_for_full_frame(unique_name, frame_id);
            if (frame == nullptr)
                break;

            for (int t = 0; t < times; t++) {
                packet_header->frame_idx = frame_idx++;
                for (int p = 0; p < elems; p++) {
                    packet_header->elem_idx = p;
                    packet_header->samples_summed =
                        ((uint*)frame)[t * elems * (freqs + 1) + p * (freqs + 1) + freqs];
                    memcpy(local_data, frame + (t * elems + p) * (freqs + 1) * sizeof(uint),
                           freqs * sizeof(uint));
                    // Send data to remote server.
                    uint32_t bytes_sent =
                        sendto(socket_fd, packet_buffer, packet_length, MSG_NOSIGNAL,
                               (struct sockaddr*)&saddr_remote, sizeof(sockaddr_in));
                    if (bytes_sent != packet_length)
                        ERROR("SOMETHING WENT WRONG IN UDP TRANSMIT");
                }
            }

            // Mark buffer as empty.
            in_buf->mark_frame_empty(unique_name, frame_id);
            frame_id = (frame_id + 1) % in_buf->num_frames;
        }
    } else if (dest_protocol == "TCP") {
        // TCP variables
        while (!stop_thread) {
            // Wait for a full buffer.
            frame = in_buf->wait_for_full_frame(unique_name, frame_id);
            if (frame == nullptr)
                break;

            while (atomic_flag_test_and_set(&socket_lock)) {
            }
            if (tcp_connected) {
                atomic_flag_clear(&socket_lock);
                for (int t = 0; t < times; t++) {
                    packet_header->frame_idx = frame_idx++;
                    for (int p = 0; p < elems; p++) {
                        packet_header->elem_idx = p;
                        packet_header->samples_summed =
                            ((uint*)frame)[t * elems * (freqs + 1) + p * (freqs + 1) + freqs];
                        memcpy(local_data, frame + (t * elems + p) * (freqs + 1) * sizeof(uint),
                               freqs * sizeof(uint));
                        uint32_t bytes_sent = send(socket_fd, packet_buffer, packet_length, 0);
                        if (bytes_sent != packet_length) {
                            ERROR("Lost TCP connection!");
                            while (atomic_flag_test_and_set(&socket_lock)) {
                            }
                            close(socket_fd);
                            tcp_connected = false;
                            atomic_flag_clear(&socket_lock);
                            break;
                        }
                    }
                }
            } else if (!tcp_connecting) {
                frame_idx += times;
                handshake_idx = frame_idx;
                gettimeofday(&tv, nullptr);
                handshake_utc = tv.tv_sec + tv.tv_usec / 1e6;
                tcp_connecting = true;
                atomic_flag_clear(&socket_lock);
                std::thread(&networkPowerStream::tcpConnect, this).detach();
            } else {
                frame_idx += times;
                atomic_flag_clear(&socket_lock);
            }
            // Mark buffer as empty.
            in_buf->mark_frame_empty(unique_name, frame_id);
            frame_id = (frame_id + 1) % in_buf->num_frames;
        }

    } else
        ERROR("Bad protocol: {:s}\n", dest_protocol);

    free(packet_buffer);
}

void networkPowerStream::tcpConnect() {
    //    INFO("Connecting TCP Power Stream!");
    struct sockaddr_in address;
    address.sin_addr.s_addr = inet_addr(dest_server_ip.c_str());
    address.sin_port = htons(dest_port);
    address.sin_family = AF_INET;

    socket_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (socket_fd == -1) {
        ERROR("Could not create TCP socket for output stream");
        return;
    }

    // TODO: handle errors, make dynamic
    struct timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 200000;

    if (connect(socket_fd, (struct sockaddr*)&address, sizeof(address)) != 0) {
        while (atomic_flag_test_and_set(&socket_lock)) {
        }
        tcp_connecting = false;
        close(socket_fd);
        atomic_flag_clear(&socket_lock);
        return;
    }
    setsockopt(socket_fd, SOL_SOCKET, SO_SNDTIMEO, (char*)&timeout, sizeof(timeout));

    { // put together handshake
        while (atomic_flag_test_and_set(&socket_lock)) {
        }
        header.handshake_idx = handshake_idx;
        header.handshake_utc = handshake_utc;
        atomic_flag_clear(&socket_lock);
        int bytes_sent = send(socket_fd, (void*)&header, sizeof(header), 0);
        if (bytes_sent != sizeof(header)) {
            ERROR("Could not send TCP header for output stream");
            while (atomic_flag_test_and_set(&socket_lock)) {
            }
            close(socket_fd);
            tcp_connected = false;
            atomic_flag_clear(&socket_lock);
            return;
        }
        // Protocol v2: a freq map PER ELEMENT (elems x freqs x [lo, hi] Hz) then
        // one Stokes/pol id per element, so a single stream can carry several
        // sub-bands (e.g. a split 0-1600 MHz feed). Per-element centre/bandwidth
        // come from the config freq/sample_bw arrays; a negative bandwidth means
        // channel 0 is the top of that element's band (inverted).
        int info_size = elems * freqs * 2 * sizeof(float) + elems * sizeof(char);
        void* info = malloc(info_size);
        float* finfo = (float*)info;
        for (int e = 0; e < elems; e++) {
            float f0 = freq0s[e], bw = sample_bws[e];
            for (int f = 0; f < freqs; f++) {
                finfo[2 * (e * freqs + f)] = f0 - bw / 2 + bw / freqs * ((float)f);
                finfo[2 * (e * freqs + f) + 1] = f0 - bw / 2 + bw / freqs * ((float)f + 1);
            }
        }
        // - description of each stream: Stokes / pol id per element.
        //  -8  -7  -6  -5  -4  -3  -2  -1  1   2   3   4
        //  YX  XY  YY  XX  LR  RL  LL  RR  I   Q   U   V
        char* sinfo = (char*)info + elems * freqs * 2 * sizeof(float);
        for (int e = 0; e < elems; e++)
            sinfo[e] = (char)stokes_ids[e];
        bytes_sent = send(socket_fd, info, info_size, 0);
        free(info);
        if (bytes_sent != info_size) {
            ERROR("Could not send TCP header for output stream");
            while (atomic_flag_test_and_set(&socket_lock)) {
            }
            close(socket_fd);
            tcp_connected = false;
            atomic_flag_clear(&socket_lock);
            return;
        }
    }
    while (atomic_flag_test_and_set(&socket_lock)) {
    }
    tcp_connected = true;
    tcp_connecting = false;
    atomic_flag_clear(&socket_lock);
}
