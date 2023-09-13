#ifndef KOTEKAN_BEAMINSPECT_HPP
#define KOTEKAN_BEAMINSPECT_HPP

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "bufferContainer.hpp"

#include <string> // for string

/**
 * @class BeamInspect
 * @brief Prints out beam information about the frames in the in_buf
 *
 * This class is intended for testing/debugging.
 *
 * @par Buffers
 * @buffer in_buf The buffer to print the contents of.
 *     @buffer_format 4+4-bit complex voltage (beam) data
 *     @buffer_metadata BeamMetadata
 *
 * @author Andre Renard
 */
class BeamInspect : public kotekan::Stage {
public:
    BeamInspect(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);
    ~BeamInspect();
    void main_thread() override;

private:
    Buffer* in_buf;
};

#endif // KOTEKAN_BEAMINSPECT_HPP
