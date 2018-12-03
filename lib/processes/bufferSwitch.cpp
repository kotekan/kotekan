#include "bufferSwitch.hpp"
#include "json.hpp"
#include "visUtil.hpp"
#include "configUpdater.hpp"

#include <exception>

using nlohmann::json;
using namespace std::placeholders;

REGISTER_KOTEKAN_PROCESS(bufferSwitch);

bufferSwitch::bufferSwitch(Config& config,
                         const string& unique_name,
                         bufferContainer &buffer_container) :
    mergeBuffer(config, unique_name, buffer_container) {

    configUpdater::instance().subscribe(this,
                    std::bind(&bufferSwitch::enabled_buffers_callback, this, _1));

}

bool bufferSwitch::enabled_buffers_callback(json &update) {
    try {
        std::lock_guard<std::mutex> map_lock(enabled_buffers_lock);
        for (json::iterator it = update.begin(); it != update.end(); ++it) {
            std::string key = it.key();

            if (key == "kotekan_update_endpoint")
                continue;

            bool enabled = it.value();
            enabled_buffers[key] = enabled;
        }
    } catch (std::exception& e) {
        WARN("bufferSwitch: Failure parsing message: %s", e.what());
        return false;
    }
    return true;
}

bool bufferSwitch::select_frame(const std::string &internal_name,
                               Buffer * in_buf, uint32_t frame_id) {
    (void)in_buf;
    (void)frame_id;
    std::lock_guard<std::mutex> map_lock(enabled_buffers_lock);
    std::map<std::string, bool>::iterator it = enabled_buffers.find(internal_name);
    if (it == enabled_buffers.end()) {
        throw std::runtime_error("No entry for the buffer named: " + internal_name);
    }
    return it->second;
}