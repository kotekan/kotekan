/**
 * @file
 * @brief Sort the incoming tracking bream frames by frequency and time.
 *  - sortBeamBuffer : public kotekan::Stage
 */

#ifndef SORT_BEAM_BUFFER
#define SORT_BEAM_BUFFER

#include "BeamMetadata.hpp" // for BeamMetadata
#include "Config.hpp"       // for Config
#include "Stage.hpp"
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for uint32_t
#include <string>   // for string
#include <vector>   // for vector


using std::vector;


/**
 * @class sortBeamBuffer
 * @brief This stage sort the beam buffer by frequency and time.
 *
 * @par Buffers
 * @buffer in_buf Kotekan single frame tracking beam buffer.
 *     @buffer_format Array of @c chars
 * @buffer out_buf Kotekan sorted tracking beam buffer.
 *     @buffer_format Array of @c uint32_t
 *
 * @author Jing Santiago Luo
 *
 *
 */


class sortBeamBuffer : public kotekan::Stage {
public:
    /// Constructor
    sortBeamBuffer(kotekan::Config& config_, const std::string& unique_name,
                          kotekan::bufferContainer& buffer_container);

    /// Destructor
    virtual ~sortBeamBuffer();

    /// Primary loop to wait for buffers, sort beam buffer.
    void main_thread() override;

private:
    /// The input buffer which has one metadata & voltage block per frame.
    struct Buffer* in_buf;
    /// The out put buffer with sorted frames.
    struct Buffer* out_buf;
    /// queue for sorting the frames.
    uint8_t*** sort_que;
    /// Config variables
    bool has_freq_bin;
    uint32_t samples_per_data_set;
    uint32_t num_freq;
    uint32_t num_freq_per_sorted_frame;
    uint32_t num_time_per_sorted_frame;

};

#endif //SORT_BEAM_BUFFER_HPP

