#include "rfiBroadcast.hpp"

#include <arpa/inet.h>          // for htons, inet_pton
#include <netinet/in.h>         // for sockaddr_in, IPPROTO_UDP
#include <sys/socket.h>         // for AF_INET, sendto, socket, SOCK_DGRAM
#include <sys/types.h>          // for ssize_t
#include <algorithm>            // for fill, copy_n, nth_element, fill_n
#include <cerrno>               // for errno
#include <cstring>              // for strerror
#include <cmath>                // for abs
#include <functional>           // for bind, function
#include <memory>               // for shared_ptr, __shared_ptr_access
#include <vector>               // for vector

#include "Config.hpp"           // for Config
#include "N2Util.hpp"           // for frameID, modulo
#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"           // for Buffer
#include "bufferContainer.hpp"  // for bufferContainer
#include "chordMetadata.hpp"    // for chordMetadata, get_chord_metadata
#include "kotekanLogging.hpp"   // for FATAL_ERROR, DEBUG
#include "rfi_functions.hpp"    // for RFIPayload, RFIHeader
#include "NDArray.hpp"          // for GenericNDArray
#include "fmt.hpp"              // for compile_string_to_view

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(rfiBroadcast);

constexpr int BITS_PER_BYTE = 8;
constexpr int RFI_BUF_S = 3;
// Conversion between median absolute deviations and sigma for
// a normal distribution
constexpr float MAD_TO_SIGMA = 1.4826;

namespace {
/// Helper to compute array median. Modifies `arr` in-place.
float median(std::vector<float>& arr) {
    auto n = arr.size();
    auto mid = n / 2;

    if (mid == 0) {
        return 0.0f;
    }

    std::nth_element(arr.begin(), arr.begin() + mid, arr.end());
    auto upper = arr[n / 2];

    // array length is odd - return the central value
    if (n % 2 != 0)
        return upper;

    // array length is even
    std::nth_element(arr.begin(), arr.begin() + mid - 1, arr.end());

    return (upper + arr[n / 2 - 1]) / 2.0;
}
}

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

    // Store information of the shape of the skbar array
    const size_t _arr_size = num_local_freq * num_elements;
    // Need an array to store the sum of skbar
    // over time samples, since we don't explicitly assume
    // that there is only one time sample per frame
    std::vector<float> skbar_sum(num_local_freq * num_elements, 0.0f);
    // Also define an array for inplace reordering of
    // skbar_sum when computing the median
    std::vector<float> _median_tmp(num_elements, 0.0f);

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
    header.sk_step = static_cast<uint32_t>(rfi_downsampling_factor);
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
        uint8_t* sk_tilde_frame =
            (uint8_t*)sk_tilde_buf->wait_for_full_frame(unique_name, sktilde_frame_id);
        if (sk_tilde_frame == nullptr)
            break;

        // Sum SK tilde over time for each frequency (second dimension)
        for (size_t f = 0; f < num_local_freq; ++f) {
            float _accum = 0;
            // Sum over time samples, which is the first dimension
            for (size_t t = RFI_BUF_S * f; t < sk_tilde_buf->frame_size;
                 t += RFI_BUF_S * num_local_freq) {
                _accum += (float)sk_tilde_frame[t];
            }
            payload.sktilde_avg[f] = _accum;
        }

        sk_tilde_buf->mark_frame_empty(unique_name, sktilde_frame_id);

        DEBUG("Waiting on empty sk_bar frame.");
        uint8_t* sk_bar_frame =
            (uint8_t*)sk_bar_buf->wait_for_full_frame(unique_name, skbar_frame_id);
        if (sk_bar_frame == nullptr)
            break;

        // Accumulate in time. Outer index is time samples
        // Index each frequency and element
        for (size_t j = 0; j < _arr_size; ++j) {
            // sum over time
            float _accum = 0;
            size_t _elem_id = RFI_BUF_S * j;
            for (size_t i = 0; i < sk_bar_buf->frame_size; i += RFI_BUF_S * _arr_size) {
                // Skip forward by 3 to grab just the SK value
                _accum += sk_bar_frame[_elem_id + i];
            }
            skbar_sum[j] = _accum;
        }

        // Release the frame since we've copied out to skbar_sum
        sk_bar_buf->mark_frame_empty(unique_name, skbar_frame_id);

        // Jump to the start of each frequency block since we need to compute
        // the statistics for each frequency independently
        for (size_t f = 0; f < _arr_size; f += num_elements) {
            // Compute the median of skbar_sum. `median` modifies in-place
            // so copy to a temp array
            std::copy_n(skbar_sum.begin() + f, num_elements, _median_tmp.begin());
            float med = median(_median_tmp);

            // For each element, compute |x - med(x)|
            for (size_t i = 0; i < num_elements; ++i) {
                // subtract the median in-place since this will be used twice
                skbar_sum[f + i] = std::abs(skbar_sum[f + i] - med);
                // copy into the temp array to compute the second median
                _median_tmp[i] = skbar_sum[f + i];
            }
            // Compute the flagging threshold in terms of standard deviations for
            // a normal distribution. med(|x - med(x)|) is the normalization factor
            // for a median absolute deviations metric, but since we just care about
            // the threshold, this is equivalent to normalizing and comparing against
            // `num_sigma_deviations * MAD_TO_SIGMA`, avoiding the intermediate division
            float mad_thresh = num_sigma_deviations * MAD_TO_SIGMA * median(_median_tmp);

            // Finally, increment the bad feed counter if a sample exceeds
            // num_sigma_deviations standard deviations
            for (size_t i = 0; i < num_elements; ++i) {
                if (skbar_sum[f + i] > mad_thresh) {
                    payload.bad_feed_counts[f + i]++;
                }
            }
        }

        // Reset skbar_sum, since it's used in every iteration
        std::fill_n(skbar_sum.begin(), _arr_size, 0.0f);

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

            // Normalize the average SK by the number of time samples, accounting
            // for the fact that the input was already downsampled. Also, convert
            // the number of flagged samples to a fraction.
            for (size_t f = 0; f < num_local_freq; ++f) {
                payload.sktilde_avg[f] /= second_divisor;
                // the RFI mask convention is actually 1 == good, so the
                // fraction flagged is 1.0 - what we computed
                payload.frac_flagged[f] = 1.0 - payload.frac_flagged[f] / divisor;
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
            std::fill(payload.bad_feed_counts.begin(), payload.bad_feed_counts.end(), 0u);
        }
        // Increment frame counters
        sktilde_frame_id++;
        skbar_frame_id++;
        rfi_mask_frame_id++;
    }
}
