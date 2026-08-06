#ifndef RFI_FUNCTIONS_H
#define RFI_FUNCTIONS_H

#include "Telescope.hpp" // for freq_id_t

#include <stdint.h>
#include <vector>

/*
 * @struct RFIHeader
 * @brief A structure that contains the header attached to each rfiBroadcast packet.
 */
struct __attribute__((packed)) RFIHeader {
    /// Fixed version number
    static constexpr uint16_t kVersion = 2;

    /// uint16_t; runtime field, filled from `kVersion` by `set_version`
    uint16_t version;
    /// uint32_t indicating total payload length
    uint32_t payload_length;
    /// uint32_t indicating the number of inputs in the input data.
    uint32_t num_elements;
    /// uint32_t indicating the number of timesteps in each frame.
    uint32_t samples_per_data_set;
    /// uint32_t indicating the total number of frequencies under consideration.
    uint32_t num_total_freq;
    /// uint32_t indicating the number of frequencies in the packet.
    uint32_t num_local_freq;
    /// uint32_t indicating the number of frames which were averaged over.
    uint32_t frames_per_packet;
    /// int64_t containing the FPGA sequence number of the first packet in the average.
    int64_t seq_num;

    /// When constructing a packet
    void set_version() {
        version = kVersion;
    }
};

using frac_flagged_t = float;
using spectral_kurtosis_t = float;

/*
 * @struct RFIPayload
 * @brief A structure defining the payload included in each rfiBroadcast packet.
 */
struct RFIPayload {
    /// List of local frequencies. Has length `num_local_freq`.
    std::vector<freq_id_t> freq_ids;
    /// Fraction of flagged samples per freq. Has length `num_local_freq`.
    std::vector<frac_flagged_t> frac_flagged;
    /// Average SK per frequency. Has length `num_local_freq`.
    std::vector<spectral_kurtosis_t> sktilde_avg;
    /// Average SK per frequency and input. Has length `num_local_freq` * `num_elements`.
    std::vector<spectral_kurtosis_t> skbar_avg;

    // Constructor
    RFIPayload(size_t num_local_freq, size_t num_elements) :
        freq_ids(num_local_freq, freq_id_t{}), frac_flagged(num_local_freq, frac_flagged_t{}),
        sktilde_avg(num_local_freq, spectral_kurtosis_t{}),
        skbar_avg(num_local_freq * num_elements, spectral_kurtosis_t{}) {}

    /// Return a single packet per frequency, which is more trivial to
    /// receive and limits the UDP packet size
    std::vector<std::vector<uint8_t>> serialize_per_freq(RFIHeader header) const {
        // Create a copy of the header with updated freq information
        RFIHeader new_header(header);
        // Compute the total packet size
        uint32_t payload_length =
            (freq_ids.size() * sizeof(freq_id_t) + frac_flagged.size() * sizeof(frac_flagged_t)
             + sktilde_avg.size() * sizeof(spectral_kurtosis_t)
             + skbar_avg.size() * sizeof(spectral_kurtosis_t))
            / header.num_local_freq;

        // Set extra header metadata
        new_header.num_local_freq = 1;
        new_header.payload_length = payload_length;

        // Initialize the vector of empty packets
        std::vector<std::vector<uint8_t>> packets(
            header.num_local_freq, std::vector<uint8_t>(sizeof(new_header) + payload_length));

        // Iterate freqs and package
        for (size_t f = 0; f < header.num_local_freq; ++f) {
            uint8_t* p = packets[f].data();
            // copy the header
            memcpy(p, &new_header, sizeof(new_header));
            p += sizeof(new_header);
            // copy the current frequency index
            memcpy(p, &freq_ids[f], sizeof(freq_id_t));
            p += sizeof(freq_id_t);
            // copy the fraction of flagged samples
            memcpy(p, &frac_flagged[f], sizeof(frac_flagged_t));
            p += sizeof(frac_flagged_t);
            // copy average SKtilde
            memcpy(p, &sktilde_avg[f], sizeof(spectral_kurtosis_t));
            p += sizeof(spectral_kurtosis_t);
            // copy average SKbar
            memcpy(p, skbar_avg.data() + f * header.num_elements,
                   header.num_elements * sizeof(spectral_kurtosis_t));
        }

        return packets;
    };
};

#endif /* RFI_FUNCTIONS_H */
