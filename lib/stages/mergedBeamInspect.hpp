#ifndef KOTEKAN_MERGEDBEAMINSPECT_HPP
#define KOTEKAN_MERGEDBEAMINSPECT_HPP

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "bufferContainer.hpp"

#include <string> // for string

/**
 * @class mergedBeamInspect
 * @brief Prints out beam information about the merged frames in the in_buf
 *
 * This class is intended for testing/debugging.
 *
 * @par Buffers
 * @buffer in_buf The buffer to print the contents of.
 *     @buffer_format 4+4-bit complex voltage (beam) data
 *     @buffer_metadata BeamMetadata
 *
 * @author Jing Luo
 */
class mergedBeamInspect : public kotekan::Stage {
public:
    mergedBeamInspect(kotekan::Config& config, const std::string& unique_name,
                      kotekan::bufferContainer& buffer_container);
    ~mergedBeamInspect();
    void main_thread() override;

private:
    struct Buffer* in_buf;
};

#endif // KOTEKAN_BEAMINSPECT_HPP
