#include "kotekanTrackers.hpp"

#include "kotekanLogging.hpp" // for ERROR_NON_OO

#include "fmt.hpp"  // for format, fmt
#include "json.hpp" // for basic_json<>::value_type, json, operator<<

#include <chrono>      // for milliseconds, duration_cast, system_clock, system_clock::t...
#include <errno.h>     // for errno
#include <fstream>     // for ofstream, ostream
#include <functional>  // for _Bind_helper<>::type, _Placeholder, bind, _1, placeholders
#include <stdexcept>   // for runtime_error
#include <stdlib.h>    // for exit
#include <type_traits> // for enable_if<>::type
#include <unistd.h>    // for gethostname
#include <utility>     // for pair

namespace kotekan {

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
    rest_server->register_get_callback(
        "/trackers_current", std::bind(&KotekanTrackers::trackers_current_callback, this, _1));
}

void KotekanTrackers::trackers_callback(connectionInstance& conn) {
    nlohmann::json return_json = {};

    for (auto& stage_itr : trackers) {
        for (auto& tracker_itr : trackers[stage_itr.first]) {
            return_json[stage_itr.first][tracker_itr.first] = tracker_itr.second->get_json();
        }
    }

    conn.send_json_reply(return_json);
}

void KotekanTrackers::trackers_current_callback(connectionInstance& conn) {
    nlohmann::json return_json = {};

    for (auto& stage_itr : trackers) {
        for (auto& tracker_itr : trackers[stage_itr.first]) {
            return_json[stage_itr.first][tracker_itr.first] =
                tracker_itr.second->get_current_json();
        }
    }

    conn.send_json_reply(return_json);
}

std::shared_ptr<StatTracker> KotekanTrackers::add_tracker(std::string stage_name,
                                                          std::string tracker_name,
                                                          std::string unit, size_t size,
                                                          bool is_optimized) {
    if (stage_name.empty()) {
        ERROR_NON_OO("Empty stage name. Exiting.");
        throw std::runtime_error("Empty stage name.");
    }
    if (tracker_name.empty()) {
        ERROR_NON_OO("Empty tracker name. Exiting.");
        throw std::runtime_error("Empty tracker name.");
    }
    if (unit.empty()) {
        ERROR_NON_OO("Empty unit for tracker {:s}:{:s}. Exiting.", stage_name, tracker_name);
        throw std::runtime_error(
            fmt::format(fmt("Empty unit name: {:s}:{:s}"), stage_name, tracker_name));
    }

    std::shared_ptr<StatTracker> tracker_ptr =
        std::make_shared<StatTracker>(tracker_name, unit, size, is_optimized);

    std::lock_guard<std::mutex> lock(trackers_lock);

    if (trackers.count(stage_name) && trackers[stage_name].count(tracker_name)) {
        ERROR_NON_OO("Duplicate tracker name: {:s}:{:s}. Exiting.", stage_name, tracker_name);
        throw std::runtime_error(
            fmt::format(fmt("Duplicate tracker name: {:s}:{:s}"), stage_name, tracker_name));
    }
    trackers[stage_name][tracker_name] = tracker_ptr;

    return tracker_ptr;
}

void KotekanTrackers::remove_tracker(std::string stage_name, std::string tracker_name) {
    auto stage_itr = trackers.find(stage_name);
    if (stage_itr != trackers.end()) {
        auto tracker_itr = trackers[stage_itr->first].find(tracker_name);
        if (tracker_itr != trackers[stage_itr->first].end()) {
            trackers[stage_itr->first].erase(tracker_itr);
        }
    }
}

void KotekanTrackers::remove_tracker(std::string stage_name) {
    auto stage_itr = trackers.find(stage_name);
    if (stage_itr != trackers.end()) {
        trackers.erase(stage_itr);
    }
}

void KotekanTrackers::set_path(std::string path) {
    dump_path = path;
}

void KotekanTrackers::dump_trackers() {
    if (dump_path.empty())
        return;

    nlohmann::json return_json = {};

    for (auto& stage_itr : trackers) {
        for (auto& tracker_itr : trackers[stage_itr.first]) {
            return_json[stage_itr.first][tracker_itr.first] = tracker_itr.second->get_json();
        }
    }

    char* host_name = new char[20];
    int ret = gethostname(host_name, sizeof(host_name));
    if (ret == -1) {
        ERROR_NON_OO("Error from gethostname()");
        exit(errno);
    }

    std::chrono::system_clock::time_point timestamp = std::chrono::system_clock::now();
    std::string time = std::to_string(
        std::chrono::duration_cast<std::chrono::milliseconds>(timestamp.time_since_epoch())
            .count());

    std::ofstream dump_file(dump_path + "/" + std::string(host_name) + "_" + time + ".json");
    if (dump_file.is_open()) {
        dump_file << return_json;
        dump_file.close();
    } else {
        ERROR_NON_OO("Cannot read tracker dump path {:s}. Exiting.", dump_path);
        throw std::runtime_error(
            fmt::format(fmt("Cannot read tracker dump path {:s}. Exiting."), dump_path));
    }
}

} // namespace kotekan
