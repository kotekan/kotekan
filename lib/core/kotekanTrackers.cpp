#include "kotekanTrackers.hpp"

#include "fmt.hpp"  // for print, format, fmt
#include "json.hpp" // for json

namespace kotekan {
namespace trackers {

KotekanTrackers::KotekanTrackers() {}

KotekanTrackers::~KotekanTrackers() {
    restServer::instance().remove_get_callback("/trackers");
}

KotekanTrackers& KotekanTrackers::instance() {
    static KotekanTrackers _instance;
    return _instance;
}

void KotekanTrackers::register_with_server(restServer* rest_server) {
    using namespace std::placeholders;
    rest_server->register_get_callback("/trackers",
                                       std::bind(&KotekanTrackers::trackers_callback, this, _1));
}

void KotekanTrackers::trackers_callback(connectionInstance& conn) {
    nlohmann::json trackers_json = {};

    for (auto& it : trackers) {
        trackers_json[it.first] = it.second->get_json();
    }

    conn.send_json_reply(trackers_json);
}

std::shared_ptr<StatTracker> KotekanTrackers::add_tracker(std::string name, std::string unit,
                                                          size_t size, bool is_optimized) {
    if (name.empty()) {
        ERROR_NON_OO("Empty tracker name. Exiting.");
        throw std::runtime_error("Empty tracker name.");
    }
    if (unit.empty()) {
        ERROR_NON_OO("Empty unit for tracker {:s}. Exiting.", name);
        throw std::runtime_error(fmt::format(fmt("Empty unit name: {:s}"), name));
    }

    std::shared_ptr<StatTracker> tracker_ptr =
        std::make_shared<StatTracker>(name, unit, size, is_optimized);

    std::lock_guard<std::mutex> lock(trackers_lock);

    if (trackers.count(name)) {
        ERROR_NON_OO("Duplicate tracker name: {:s}. Exiting.", name);
        throw std::runtime_error(fmt::format(fmt("Duplicate tracker name: {:s}"), name));
    }
    trackers[name] = tracker_ptr;

    return tracker_ptr;
}

} // namespace trackers
} // namespace kotekan
