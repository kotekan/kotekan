#ifndef KOTEKAN_BEAMINSPECT_HPP
#define KOTEKAN_BEAMINSPECT_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <string> // for string

/**
 * @class BeamInspect
 * @brief Prints out beam information about the frames in the in_buf
 *
 * This class is intended for testing/debugging: it consumes a beam buffer and logs per-frame
 * pointing, scaling, coarse frequency list, and the first complex sample value. Frames are simply
 * released after printing; no copies are made and no downstream buffer is produced. Use it to
 * sanity-check beam metadata and sample packing from a beamformer before connecting to a
 * downstream consumer.
 *
 * @par Buffers
 * @buffer in_buf The buffer to print the contents of.
 *     @buffer_format 4+4-bit complex voltage (beam) data
 *     @buffer_metadata BeamMetadata
 *
 * @conf in_buf String. Input beam buffer.
 *
 * @par Example
 * @code
 * BeamInspect:
 *   in_buf: beam_gpu0
 * @endcode
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
