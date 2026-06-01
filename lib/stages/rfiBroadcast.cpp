#include "rfiBroadcast.hpp"

#include "Config.hpp"          // for Config
#include "N2Util.hpp"          // for frameID, modulo
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata, get_chord_metadata
#include "kotekanLogging.hpp"  // for FATAL_ERROR, DEBUG
#include "rfi_functions.hpp"   // for RFIPayload, RFIHeader

#include <algorithm>    // for fill, copy_n, nth_element, fill_n
#include <arpa/inet.h>  // for htons, inet_pton
#include <cerrno>       // for errno
#include <cstring>      // for strerror
#include <functional>   // for bind, function
#include <memory>       // for shared_ptr, __shared_ptr_access
#include <netinet/in.h> // for sockaddr_in, IPPROTO_UDP
#include <sys/socket.h> // for AF_INET, sendto, socket, SOCK_DGRAM
#include <sys/types.h>  // for ssize_t
#include <vector>       // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(rfiBroadcast);

constexpr int BITS_PER_BYTE = 8;
constexpr int RFI_BUF_S = 3;

rfiBroadcast::rfiBroadcast(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&rfiBroadcast::main_thread, this)) {
    // Sort out the input buffers and register as consumers
    sk_tilde_buf = get_buffer("rfi_sktilde_buf");
    sk_tilde_buf->register_consumer(unique_name);

    sk_bar_buf = get_buffer("rfi_skbar_buf");
    sk_bar_buf->register_consumer(unique_name);

    rfi_mask_buf = get_buffer("rfi_mask_buf");
    rfi_mask_buf->register_consumer(unique_name);

    // Standard config params defining the expected buffer shapes
    num_elements = config.get<size_t>(unique_name, "num_elements");
    num_local_freq = config.get<size_t>(unique_name, "num_local_freq");
    num_global_freq = config.get<size_t>(unique_name, "num_global_freq");
    samples_per_data_set = config.get<size_t>(unique_name, "samples_per_data_set");
    rfi_downsampling_factor = config.get<size_t>(unique_name, "rfi_downsampling_factor");
    rfi_second_downsampling_factor =
        config.get<size_t>(unique_name, "rfi_second_downsampling_factor");
    // Stage-specific params
    num_sigma_deviations = config.get<uint16_t>(unique_name, "num_sigma_deviations");
    // Packet/network config params
    frames_per_packet = config.get_default<size_t>(unique_name, "frames_per_packet", 1);
    dest_port = config.get<size_t>(unique_name, "destination_port");
    dest_ip = config.get<std::string>(unique_name, "destination_ip");

    // Require that `samples_per_data_set` is properly represented as a uint1x8
    if (samples_per_data_set % BITS_PER_BYTE != 0) {
        FATAL_ERROR("`samples_per_data_set` is not representable as bits: {:d}/{:d}",
                    samples_per_data_set, BITS_PER_BYTE);
    }

    // Compute the derived number of time samples for sktilde and skbar, respectively
    if (samples_per_data_set % rfi_downsampling_factor != 0) {
        FATAL_ERROR("`rfi_downsampling_factor` does not evenly divide the number of time samples: "
                    "{:d} / {:d}",
                    rfi_downsampling_factor, samples_per_data_set);
    } else {
        _downsampled_samples_per_data_set = samples_per_data_set / rfi_downsampling_factor;
    }
    if (_downsampled_samples_per_data_set % rfi_second_downsampling_factor != 0) {
        FATAL_ERROR("`rfi_second_downsampling_factor` does not evenly divide downsampled time "
                    "samples: {:d} / {:d}",
                    rfi_second_downsampling_factor, _downsampled_samples_per_data_set);
    } else {
        _second_downsampled_samples_per_data_set =
            _downsampled_samples_per_data_set / rfi_second_downsampling_factor;
    }

    // Check that input buffers have the expected frame size
    // per the provided config parameters
    size_t _rfi_mask_expected_frame_size = num_local_freq * samples_per_data_set / BITS_PER_BYTE;
    if (rfi_mask_buf->frame_size != _rfi_mask_expected_frame_size)
        FATAL_ERROR("`rfi_mask_buf` does not have the expected frame size. Expected {:d}, got {:d}",
                    _rfi_mask_expected_frame_size, rfi_mask_buf->frame_size);

    // SK buffers are packed in groups of 3 values (SK, bias, sigma)
    size_t _sktilde_expected_frame_size =
        sizeof(float) * RFI_BUF_S * num_local_freq * _downsampled_samples_per_data_set;
    if (sk_tilde_buf->frame_size != _sktilde_expected_frame_size)
        FATAL_ERROR(
            "`rfi_sktilde_buf` does not have the expected frame size. Expected {:d}, got {:d}",
            _sktilde_expected_frame_size, sk_tilde_buf->frame_size);

    size_t _skbar_expected_frame_size = sizeof(float) * RFI_BUF_S * num_local_freq * num_elements
                                        * _second_downsampled_samples_per_data_set;
    if (sk_bar_buf->frame_size != _skbar_expected_frame_size)
        FATAL_ERROR(
            "`rfi_skbar_buf` does not have the expected frame size. Expected {:d}, got {:d}",
            _skbar_expected_frame_size, sk_bar_buf->frame_size);

    // Set up the UDP socket params and validate the the provided ID is valid
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(dest_port);
    if (inet_pton(AF_INET, dest_ip.data(), &dest_addr.sin_addr) == 0)
        FATAL_ERROR("Invalid destination IP: {:s}", dest_ip);
}

rfiBroadcast::~rfiBroadcast() {}

void rfiBroadcast::main_thread() {
    // Create frame IDs for each input buffer
    N2::frameID sktilde_frame_id(sk_tilde_buf);
    N2::frameID skbar_frame_id(sk_bar_buf);
    N2::frameID rfi_mask_frame_id(rfi_mask_buf);

    // Initialize the payload struct
    RFIPayload payload(num_local_freq, num_elements);

    // Strides for SK tilde and SK bar
    const size_t SE = num_elements * RFI_BUF_S;
    const size_t FS = num_local_freq * RFI_BUF_S;
    const size_t FSE = FS * num_elements;

    // Sort out the rfi mask buffer time stride
    auto rfi_frame_desc = rfi_mask_buf->get_ndarray_frame_desc();
    auto _rfi_frame_rank = rfi_frame_desc->get_rank();
    auto _rfi_frame_dims = rfi_frame_desc->get_extents();
    size_t _rfi_t_hi = _rfi_frame_dims[_rfi_frame_rank - 1];
    size_t _rfi_step = num_local_freq * _rfi_t_hi;

    /// Create the packet header
    RFIHeader header;
    header.set_version();
    // The payload size is set when calling `RFIPayload.serialize_per_freq`
    header.num_elements = static_cast<uint32_t>(num_elements);
    header.samples_per_data_set = static_cast<uint32_t>(samples_per_data_set);
    header.num_total_freq = static_cast<uint32_t>(num_global_freq);
    header.num_local_freq = static_cast<uint32_t>(num_local_freq);
    header.frames_per_packet = static_cast<uint32_t>(frames_per_packet);
    // These get set by the first frame per packet
    header.seq_num = 0;

    // Create the socket
    int socket_handler = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (socket_handler == -1) {
        FATAL_ERROR("Could not create UDP socket for output stream - Error {:d}: {:s}", errno,
                    strerror(errno));
    };
    // Otherwise, connection is successful
    DEBUG("UDP Connection establish: {:d} {:s}", dest_port, dest_ip);
    // Directly track frames since `buffer_depth` for the incoming
    // buffers could be less than `frames_per_packet`
    uint32_t _packet_num_frame_counter = 0;

    while (!stop_thread) {
        // Wait for frames to become available
        DEBUG("Waiting on empty rfi_mask frame.");
        uint8_t* rfi_mask_frame =
            (uint8_t*)rfi_mask_buf->wait_for_full_frame(unique_name, rfi_mask_frame_id);
        if (rfi_mask_frame == nullptr)
            break;

        std::shared_ptr<chordMetadata> meta = get_chord_metadata(rfi_mask_buf, rfi_mask_frame_id);

        // Set additional header metadata, just once per packet
        if (rfi_mask_frame_id % frames_per_packet == 0) {
            header.seq_num = meta->get_fpga_seq_num();
            auto coarse_freq = meta->get_coarse_freq();
            std::copy_n(coarse_freq.begin(), num_local_freq, payload.freq_ids.begin());
        };

        // Sum the rfi mask for each freq. and release, keeping in mind
        // that the rfi_mask has type uint1x8 and split time axis.
        // Iterate frequencies
        for (size_t f = 0; f < num_local_freq; ++f) {
            // Accumulate into an unisigned int
            uint64_t _accum = 0;
            size_t fidx = f * _rfi_t_hi;
            // Iterate outer time samples
            for (size_t t = 0; t < rfi_mask_buf->frame_size; t += _rfi_step) {
                size_t ftidx = fidx + t;
                // Sum over high-cadence time samples
                for (size_t th = 0; th < _rfi_t_hi; ++th) {
                    // Accumulate the count of individual bits per byte
                    _accum += __builtin_popcount(rfi_mask_frame[ftidx + th]);
                }
            }
            // Record as a float because we'll eventually have to normalize
            // into a fraction
            payload.frac_flagged[f] += static_cast<float>(_accum);
        };

        rfi_mask_buf->mark_frame_empty(unique_name, rfi_mask_frame_id);

        // Wait for the sk_tilde frame
        DEBUG("Waiting on empty sk_tilde frame.");
        float* sk_tilde_frame = reinterpret_cast<float*>(
            sk_tilde_buf->wait_for_full_frame(unique_name, sktilde_frame_id));
        if (sk_tilde_frame == nullptr)
            break;

        // Sum SK tilde over time (outer index)
        float* dst_tilde = payload.sktilde_avg.data();

        for (size_t t = 0; t < _downsampled_samples_per_data_set; ++t) {
            // freq_0 for this time sample
            const float* src = sk_tilde_frame + (t * FS);
            // iterate frequencies. stride by RFI_BUF_S to get only
            // the SK value, ignoring sigma and bias
            for (size_t f = 0; f < num_local_freq; ++f) {
                dst_tilde[f] += src[f * RFI_BUF_S];
            }
        }

        sk_tilde_buf->mark_frame_empty(unique_name, sktilde_frame_id);

        DEBUG("Waiting on empty sk_bar frame.");
        float* sk_bar_frame =
            reinterpret_cast<float*>(sk_bar_buf->wait_for_full_frame(unique_name, skbar_frame_id));
        if (sk_bar_frame == nullptr)
            break;

        // Accumulate in time. Outer index is time samples
        float* dst_bar = payload.skbar_avg.data();

        for (size_t t = 0; t < _second_downsampled_samples_per_data_set; ++t) {
            for (size_t f = 0; f < num_local_freq; ++f) {
                // For this frequency and time sample, the first element stride
                // is a full stride for each time sample (FSE) and a stride
                // through all elements and values (SK, sigma, bias) to this
                // frequency
                const float* src = sk_bar_frame + (t * FSE) + (f * SE);
                // target is the same stride through elements, but there's no
                // `S` axis since we're just storing SK
                float* dst_bar_t = dst_bar + f * num_elements;

                for (size_t e = 0; e < num_elements; ++e) {
                    dst_bar_t[e] += src[e];
                }
            }
        }

        sk_bar_buf->mark_frame_empty(unique_name, skbar_frame_id);

        // Increment and wrap to number of frames per packet
        _packet_num_frame_counter = (_packet_num_frame_counter + 1) % frames_per_packet;

        // Send a packet every `frames_per_packet` frames
        if (_packet_num_frame_counter == 0) {
            // Normalize by the number of time samples in the packet.
            // Cast the first operand to a float to get rid of a warning about
            // using integer division in a floating point context, even though we
            // require that these values are evenly divisible
            float divisor = static_cast<float>(frames_per_packet) * samples_per_data_set;
            float second_divisor = divisor / rfi_downsampling_factor;
            float third_divisor = second_divisor / rfi_second_downsampling_factor;

            // Normalize the average SK by the number of time samples, accounting
            // for the fact that the input was already downsampled. Also, convert
            // the number of flagged samples to a fraction.
            for (size_t f = 0; f < num_local_freq; ++f) {
                payload.sktilde_avg[f] /= second_divisor;
                // the RFI mask convention is actually 1 == good, so the
                // fraction flagged is 1.0 - what we computed
                payload.frac_flagged[f] = 1.0 - payload.frac_flagged[f] / divisor;
            }
            for (size_t i = 0; i < num_local_freq * num_elements; ++i) {
                payload.skbar_avg[i] /= third_divisor;
            }

            // Get a vector containing a packet per frequency. Need to send
            // each frequency independently to avoid fragmentation
            auto packets = payload.serialize_per_freq(header);

            for (const auto& packet : packets) {
                // Send and log if something went wrong
                ssize_t bytes_sent = sendto(socket_handler, packet.data(), packet.size(), 0,
                                            (struct sockaddr*)&dest_addr, sizeof(sockaddr_in));

                if (bytes_sent != (ssize_t)packet.size()) {
                    DEBUG("Error sending UDP packet - Error {:d}: {:s}\nExpected to send {:d} "
                          "bytes, actually sent "
                          "{:d}.",
                          errno, strerror(errno), packet.size(), bytes_sent);
                } else {
                    DEBUG("Sent UDP packet with {:d} bytes", bytes_sent);
                }
            }

            // Reset payload members used in accumulation to zero
            std::fill(payload.frac_flagged.begin(), payload.frac_flagged.end(), 0.0f);
            std::fill(payload.sktilde_avg.begin(), payload.sktilde_avg.end(), 0.0f);
            std::fill(payload.skbar_avg.begin(), payload.skbar_avg.end(), 0.0f);
        }
        // Increment frame counters
        sktilde_frame_id++;
        skbar_frame_id++;
        rfi_mask_frame_id++;
    }
}
