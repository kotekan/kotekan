#ifndef PL_MASK_EXPANDED_TO_COMPACT_HPP
#define PL_MASK_EXPANDED_TO_COMPACT_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <cstdint>
#include <string>

/**
 * @class PLMaskExpandedToCompact
 * @brief Converts an expanded packet loss mask to compact format for the RFI S0 kernel.
 *
 * The expanded PL mask has shape [T/64][F][P][D/8][8] (dim names: Thi64, F, P, D8, Tlo64),
 * where each uint64 holds 64 bits at 1-baseband-sample resolution.
 *
 * The compact PL mask has shape [T/128][(F+3)/4][P][D/8][8] (dim names: T2hi64, F4, P, D8, T2lo64),
 * where each uint64 holds 64 bits at 2-baseband-sample resolution (each bit represents 2 samples).
 *
 * The conversion:
 *   - Time: pairs of expanded uint64s (covering 128 samples) are combined into one compact uint64
 *     by extracting every other bit and packing.
 *   - Frequency: groups of 4 consecutive frequencies are AND'd together.
 *
 * @par Buffers
 * @buffer in_buf Input expanded PL mask buffer
 *     @buffer_format uint64_t pl_mask[T/64][F][E/8]
 *     @buffer_metadata chordMetadata
 * @buffer out_buf Output compact PL mask buffer
 *     @buffer_format uint64_t pl_mask[T/128][(F+3)/4][E/8]
 *     @buffer_metadata chordMetadata
 *
 * @conf num_elements   Int. Total number of elements (num_polarizations * num_dishes).
 * @conf num_local_freq Int. Number of frequency channels.
 * @conf samples_per_data_set Int. Number of time samples per frame (T).
 */
class PLMaskExpandedToCompact : public kotekan::Stage {
public:
    PLMaskExpandedToCompact(kotekan::Config& config, const std::string& unique_name,
                            kotekan::bufferContainer& buffer_container);
    ~PLMaskExpandedToCompact();
    void main_thread() override;

private:
    Buffer* in_buf;
    Buffer* out_buf;

    uint32_t num_elements;
    uint32_t num_local_freq;
    uint32_t samples_per_data_set;
};

#endif // PL_MASK_EXPANDED_TO_COMPACT_HPP
