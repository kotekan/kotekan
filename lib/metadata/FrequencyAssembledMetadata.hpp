// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Kotekan Developers

/****************************************************
* @file FrequencyAssembledMetadata.hpp
* @brief This file declares useful structures and
*        the FrequencyAssembledMetadata structure
*
* @author Mehdi Najafi
* @date   28 AUG 2022
*****************************************************/

#ifndef FREQ_ASSEMBLED_METADATA
#define FREQ_ASSEMBLED_METADATA

#include <cstdint> // integer types
#include <limits>  // std::numeric_limits

/**
 * @struct FrequencyBin
 * @brief Frequency bin for a range of frequency ids.
 * *
 * @par num_frequencies
 *      The number of frequency ids assigned to the bin
 * @par lower_band_frequency, higher_band_frequency
 *      The frequency id range: [low to high]
 *
 * @author Mehdi Najafi
 */
struct FrequencyBin {
    /// The lower frequency range
    uint32_t lower_band_frequency  = std::numeric_limits<uint32_t>::max();
    /// The higher frequency range
    uint32_t higher_band_frequency = std::numeric_limits<uint32_t>::min();
    /// The number of frequencies
    uint32_t num_frequencies;
};

/// add a frequency id to the bin
inline uint32_t frequencyBin_add_frequency_id(FrequencyBin *bin, const uint32_t freq_id) {
    if (bin->lower_band_frequency > freq_id ) bin->lower_band_frequency = freq_id;
    if (bin->higher_band_frequency < freq_id ) bin->higher_band_frequency = freq_id;
    return ++bin->num_frequencies;
}


/**
 * @struct FrequencyAssembledMetadata
 * @brief Frequency assembled beam metadata based on some metadata.
 * *
 * @par BaseMetadata
 *      The base metadata structure to be used
 *
 * @author Mehdi Najafi
 */
template <typename BaseMetadata>
struct FrequencyAssembledMetadata : BaseMetadata, FrequencyBin {
};

template <typename BaseMetadata>
void copy_base_to_frequency_assembled_Metadata(BaseMetadata* base, void *asmb) {
    memcpy((BaseMetadata*)asmb, base, sizeof(BaseMetadata));
}

#endif // FREQ_ASSEMBLED_METADATA
