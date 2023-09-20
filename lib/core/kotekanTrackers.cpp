#include "kotekanTrackers.hpp"

#include "kotekanLogging.hpp" // for ERROR_NON_OO

#include "fmt.hpp"  // for format, fmt
#include "json.hpp" // for basic_json<>::value_type, json, operator<<

#include <chrono>     // for system_clock, system_clock::time_point
#include <errno.h>    // for errno
#include <exception>  // for exception
#include <fstream>    // for ofstream, ostream
#include <functional> // for _Bind_helper<>::type, _Placeholder, bind, _1, placeholders
#include <stdexcept>  // for runtime_error
#include <stdio.h>    // for sprintf
#include <stdlib.h>   // for exit
#include <time.h>     // for tm, localtime, time_t
#include <unistd.h>   // for gethostname
#include <utility>    // for pair

namespace kotekan {

KotekanTrackers::KotekanTrackers() {}

KotekanTrackers::~KotekanTrackers() {
    restServer::instance().remove_get_callback("/trackers");
}

KotekanTrackers& KotekanTrackers::private_instance() {
    static KotekanTrackers _instance;
    return _instance;
}

KotekanTrackers& KotekanTrackers::instance() {
    return private_instance();
}

KotekanTrackers& KotekanTrackers::instance(const kotekan::Config& config) {
    KotekanTrackers& kt = private_instance();

    if (config.get_default<bool>("/trackers", "enable_crash_dump", false)) {
        kt.dump_path = config.get_default<std::string>("/trackers", "dump_path", "./");
    }

    return kt;
}

void KotekanTrackers::register_with_server(restServer* rest_server) {
    using namespace std::placeholders;
    rest_server->register_get_callback("/trackers",
                                       std::bind(&KotekanTrackers::trackers_callback, this, _1));
    rest_server->register_get_callback(
        "/trackers_current", std::bind(&KotekanTrackers::trackers_current_callback, this, _1));
}

void KotekanTrackers::set_kotekan_mode_ptr(kotekan::kotekanMode* _kotekan_mode_ptr) {
    kotekan_mode_ptr = _kotekan_mode_ptr;
}

void KotekanTrackers::trackers_callback(connectionInstance& conn) {
    nlohmann::json return_json = {};

    std::lock_guard<std::mutex> lock(trackers_lock);

    for (auto& stage_itr : trackers) {
        for (auto& tracker_itr : trackers[stage_itr.first]) {
            return_json[stage_itr.first][tracker_itr.first] = tracker_itr.second->get_json();
        }
    }

    conn.send_json_reply(return_json);
}

void KotekanTrackers::trackers_current_callback(connectionInstance& conn) {
    nlohmann::json return_json = {};

    std::lock_guard<std::mutex> lock(trackers_lock);

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

    std::lock_guard<std::mutex> lock(trackers_lock);

    auto val1 = trackers.find(stage_name);
    if (val1 != trackers.end()) {
        auto val2 = val1->second.find(tracker_name);
        if (val2 != val1->second.end())
            // Exists, return existing tracker.
            return val2->second;
    }
    // Does not exist, create and store new one
    std::shared_ptr<StatTracker> tracker_ptr =
        std::make_shared<StatTracker>(tracker_name, unit, size, is_optimized);
    trackers[stage_name][tracker_name] = tracker_ptr;

    return tracker_ptr;
}

void KotekanTrackers::remove_tracker(std::string stage_name, std::string tracker_name) {
    std::lock_guard<std::mutex> lock(trackers_lock);

    auto stage_itr = trackers.find(stage_name);
    if (stage_itr != trackers.end()) {
        auto tracker_itr = trackers[stage_itr->first].find(tracker_name);
        if (tracker_itr != trackers[stage_itr->first].end()) {
            trackers[stage_itr->first].erase(tracker_itr);
        }
    }
}

void KotekanTrackers::remove_tracker(std::string stage_name) {
    std::lock_guard<std::mutex> lock(trackers_lock);

    auto stage_itr = trackers.find(stage_name);
    if (stage_itr != trackers.end()) {
        trackers.erase(stage_itr);
    }
}

void KotekanTrackers::dump_trackers() {
    if (dump_path.empty())
        return;

    nlohmann::json return_json = {};

    if (kotekan_mode_ptr != nullptr) {
        return_json["buffers"] = kotekan_mode_ptr->get_buffer_json();
    }

    std::lock_guard<std::mutex> lock(trackers_lock);

    for (auto& stage_itr : trackers) {
        for (auto& tracker_itr : trackers[stage_itr.first]) {
            return_json["trackers"][stage_itr.first][tracker_itr.first] =
                tracker_itr.second->get_json();
        }
    }

    char host_name[20];
    int ret = gethostname(host_name, sizeof(host_name));
    if (ret == -1) {
        ERROR_NON_OO("Error from gethostname()");
        exit(errno);
    }

    std::chrono::system_clock::time_point timestamp = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(timestamp);
    std::tm local_tm;
    std::tm* tm_ptr = localtime_r(&tt, &local_tm);

    if (tm_ptr == nullptr) {
        ERROR_NON_OO("Error from localtime_r()");
        return;
    }

    char time_c[50];
    sprintf(time_c, "%d-%d-%d_%d:%d:%d", local_tm.tm_year + 1900, local_tm.tm_mon + 1,
            local_tm.tm_mday, local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec);
    std::string time_s = time_c;

    std::ofstream dump_file(dump_path + "/" + std::string(host_name) + "_crash_stats_" + time_s
                            + ".json");
    if (dump_file.is_open()) {
        dump_file << return_json;
        dump_file.close();
    } else {
        ERROR_NON_OO("Cannot read tracker dump path {:s}. Exiting.", dump_path);
        throw std::runtime_error(
            fmt::format(fmt("Cannot read tracker dump path {:s}. Exiting."), dump_path));
    }
    INFO_NON_OO("Dumped kotekan statistic tracker (debug) data to {:s}", dump_path);
}

} // namespace kotekan
