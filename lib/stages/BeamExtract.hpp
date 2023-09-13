/*****************************************
@file
@brief Stage for extracting one beam from the beamformer output and putting it into a new frame
- BeamExtract : public kotekan::Stage
*****************************************/
#ifndef BEAMEXTRACT_HPP
#define BEAMEXTRACT_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for uint32_t
#include <string>   // for string

/**
 * @class BeamExtract
 * @brief Stage for extracting one beam from the beamformer output and putting it into a new frame.
 *
 * Assumes that the number of polarizations is 2
 *
 * @par Buffers
 * @buffer in_buf The GPU beamformer output buffer
 *     @buffer_format Array of 4+4-bit number with the format of [time][beam][pol]
 *     @buffer_metadata chimeMetadata
 *
 * @buffer out_buf The extracted single beam output buffer
 *     @buffer_format Array of 4+4-bit numbers with format [time][pol]
 *     @buffer_metadata BeamMetadata
 *
 * @conf    num_beams             Int.   The total number of beams in the output.
 * @conf    num_pol               Int.   Must be set to 2 to use this stage
 * @conf    extract_beam          Int.   The beam number to extract from the input set of beams;
 *                                       zero based.
 * @conf    samples_per_data_set  Int.   Number of time samples in a data set
 *
 * @author Andre Renard
 */
class BeamExtract : public kotekan::Stage {
public:
    BeamExtract(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);
    ~BeamExtract();
    void main_thread() override;

private:
    Buffer* in_buf;
    Buffer* out_buf;

    uint32_t _num_beams;
    uint32_t _extract_beam;
    uint32_t _samples_per_data_set;
};


#endif // BEAMEXTRACT_HPP
