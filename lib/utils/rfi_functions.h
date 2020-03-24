#ifndef RFI_FUNCTIONS_H
#define RFI_FUNCTIONS_H

#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * @struct RFIHeader
 * @brief A structure that contains the header attached to each rfiBroadcast packet.
 */
struct __attribute__((packed)) RFIHeader {
    /// uint8_t indicating whether or not the SK value was summed over inputs.
    uint8_t rfi_combined;
    /// uint32_t indicating the time intergration length of the SK values.
    uint32_t sk_step;
    /// uint32_t indicating the number of inputs in the input data.
    uint32_t num_elements;
    /// uint32_t indicating the number of timesteps in each frame.
    uint32_t samples_per_data_set;
    /// uint32_t indicating the total number of frequencies under consideration (1024 by default).
    uint32_t num_total_freq;
    /// uint32_t indicating the number of frequencies in the packet.
    uint32_t num_local_freq;
    /// uint32_t indicating the number of frames which were averaged over.
    uint32_t frames_per_packet;
    /// int64_t containing the FPGA sequence number of the first packet in the average.
    int64_t seq_num;
    /// uint16_t holding the current stream ID value.
    uint16_t streamID;
};

#ifdef __cplusplus
}
#endif

#endif /* RFI_FUNCTIONS_H */
