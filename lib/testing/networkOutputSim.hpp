#ifndef NETWORK_OUTPUT_SIM
#define NETWORK_OUTPUT_SIM

#define SIM_CONSTANT 0
#define SIM_FULL_RANGE 1
#define SIM_SINE 2

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for int32_t
#include <string>   // for string

class networkOutputSim : public kotekan::Stage {
public:
    networkOutputSim(kotekan::Config& config, const std::string& unique_name,
                     kotekan::bufferContainer& buffer_container);
    virtual ~networkOutputSim();
    void main_thread() override;

private:
    /**
     * @brief Generates simulated network output data frames with selectable patterns.
     *
     * Patterns: constant 4+4-bit complex value, full-range ramp over real/imag, or per-frequency
     * complex sine. Frames are marked with FPGA sequence numbers and timestamps in chordMetadata
     * to mimic live GPU output. The stage steps frame IDs by `num_links_in_group` to simulate
     * multiple links interleaving frames.
     *
     * @par Buffers
     * @buffer network_out_buf Output buffer to fill.
     *     @buffer_format 4+4-bit complex voltages [time][freq][elem]
     *     @buffer_metadata chordMetadata
     *
     * @conf network_out_buf     String. Output buffer name.
     * @conf num_links_in_group  Int. Number of interleaved link buffers (frame_id stride).
     * @conf link_id             Int. This link index (initial frame offset).
     * @conf pattern             Int. One of 0=constant, 1=full-range ramp, 2=complex sine.
     * @conf samples_per_data_set Int. Samples per frame.
     * @conf num_local_freq      Int. Number of coarse frequencies per frame.
     * @conf num_elements        Int. Number of elements.
     *
     * @par Example
     * @code
     * networkOutputSim:
     *   network_out_buf: net_out
     *   num_links_in_group: 4
     *   link_id: 0
     *   pattern: 1   # full range
     *   samples_per_data_set: 49152
     *   num_local_freq: 4
     *   num_elements: 2048
     * @endcode
     */
    Buffer* buf;
    int num_links_in_group;
    int link_id;
    int pattern;

    // Config variables.
    int32_t _samples_per_data_set;
    int32_t _num_local_freq;
    int32_t _num_elem;
};

#endif
