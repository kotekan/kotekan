#include "bufferSwitch.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "configUpdater.hpp"   // for configUpdater
#include "kotekanLogging.hpp"  // for WARN
#include "visUtil.hpp"         // for frameID  // IWYU pragma: keep

#include "fmt.hpp"  // for format, fmt
#include "json.hpp" // for json, basic_json<>::iterator, basic_json, iter_impl

#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, _Placeholder, bind, _1, placeholders
#include <stdexcept>  // for runtime_error
#include <tuple>      // for get
#include <utility>    // for pair
#include <vector>     // for vector

using nlohmann::json;
using namespace std::placeholders;

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::configUpdater;

REGISTER_KOTEKAN_STAGE(bufferSwitch);

bufferSwitch::bufferSwitch(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    bufferMerge(config, unique_name, buffer_container) {

    // Disabled by default
    for (auto& buffer_info : in_bufs) {
        std::string buffer_name = std::get<0>(buffer_info);
        enabled_buffers[buffer_name] = false;
    }

    configUpdater::instance().subscribe(
        this, std::bind(&bufferSwitch::enabled_buffers_callback, this, _1));
}

bool bufferSwitch::enabled_buffers_callback(json& update) {
    try {
        std::lock_guard<std::mutex> map_lock(enabled_buffers_lock);
        for (json::iterator it = update.begin(); it != update.end(); ++it) {
            std::string key = it.key();

            if (key == "kotekan_update_endpoint")
                continue;

            if (enabled_buffers.count(key) == 0) {
                WARN("Message contains a key we didn't expect: {:s}, request JSON {:s}", key,
                     update.dump());
                return false;
            }

            bool enabled = it.value();
            enabled_buffers.at(key) = enabled;
        }
    } catch (std::exception& e) {
        WARN("bufferSwitch: Failure parsing message. Error: {:s}, Request JSON: {:s}", e.what(),
             update.dump());
        return false;
    }
    return true;
}

bool bufferSwitch::select_frame(const std::string& internal_name, Buffer* in_buf,
                                uint32_t frame_id) {
    (void)in_buf;
    (void)frame_id;
    std::lock_guard<std::mutex> map_lock(enabled_buffers_lock);
    std::map<std::string, bool>::iterator it = enabled_buffers.find(internal_name);
    if (it == enabled_buffers.end()) {
        throw std::runtime_error(
            fmt::format(fmt("No entry for the buffer named: {:s}"), internal_name));
    }
    return it->second;
}
