#ifndef BUFFER_SWITCH_HPP
#define BUFFER_SWITCH_HPP

#include "Config.hpp"
#include "buffer.h"
#include "bufferContainer.hpp"
#include "bufferMerge.hpp" // for bufferMerge

#include <json.hpp> // for json
#include <map>      // for map
#include <mutex>    // for mutex
#include <stdint.h> // for uint32_t
#include <string>   // for string

/**
 * @brief Selects buffers based on the values in an updatable config endpoint.
 *
 * An example config:
 *
 * buffer_switch:
 * kotekan_stage: bufferSwitch
 * in_bufs:
 *   - network_data_0: gpu_data_buffer_0
 *   - network_data_1: gpu_data_buffer_1
 * out_buf: network_buffer
 * updatable_config: "/buffer_switch/switch_status"
 * switch_status:
 *   kotekan_update_endpoint: "json"
 *   network_data_0: false # Don't merge frames from gpu_data_buffer_0
 *   network_data_1: true  # Merge frames from gpu_data_buffer_1
 *
 * See bufferMerge for more docs.  Requires internal names.
 *
 * @conf updatable_config  String.  JSON pointer to the updatable config block.
 *                                  An example block would be:
 *                                  switch_config:
 *                                      updatable_config: "json"
 *                                      internal_buffer_name_0: true
 *                                      internal_buffer_name_1: false
 *
 * @author Andre Renard
 */
class bufferSwitch : public bufferMerge {
public:
    /// Constructor
    bufferSwitch(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);

    /// Destructor
    ~bufferSwitch() = default;

    /**
     * @brief Selects a buffer if it's internal name is set to true in the
     *        updatable config block.
     *
     * @param internal_name The name given in the config to the buffer
     *                      (not the buffer name)
     * @param in_buf        not used.
     * @param frame_id      not used.
     *
     * @return true if the internal name is set to true in @c enabled_buffers_lock
     */
    virtual bool select_frame(const std::string& internal_name, Buffer* in_buf,
                              uint32_t frame_id) override;

    /// Called by the configUpdater to change which buffers are selected.
    bool enabled_buffers_callback(nlohmann::json& json);

private:
    /// Map of internal names with a true/false value for if we include frames from it
    std::map<std::string, bool> enabled_buffers;

    /// Lock updates to the list of enabled_buffers.
    std::mutex enabled_buffers_lock;
};

#endif
