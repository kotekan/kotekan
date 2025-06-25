#include "configTracker.hpp"
#include "Config.hpp"
#include "kotekanLogging.hpp"

#include <openssl/md5.h>
#include <sstream>
#include <iomanip>
#include <mutex>

namespace kotekan {

ConfigTracker& ConfigTracker::instance() {
    static ConfigTracker instance;
    return instance;
}

void ConfigTracker::reset() {
    INFO_NON_OO("ConfigTracker: clearing configs.");
    _configs.clear();
}

void ConfigTracker::insertConfig(const nlohmann::json& config_json) {

    // Strip updatable config fields before hashing
    nlohmann::json filtered_json;
    for (auto& [key, value] : config_json.items()) {
        if (key != "updatable_config") {
            filtered_json[key] = value;
        }
    }
    std::string hash = canonicalHash(filtered_json);
    
    // Store the config in the map with its hash
    // Note: This is a raw JSON object, not a Config object.
    {
        std::lock_guard<std::mutex> lock(_lock);

        if (_configs.count(hash)) {
            DEBUG_NON_OO("ConfigTracker: config already exists with hash: {}", hash);
            return;
        }

        _configs.emplace(hash, filtered_json); // no harm if it already exists
    }

    // Update the combined hash
    setCombinedHash();
}

bool ConfigTracker::hasConfig(const std::string& hash) const {
    std::lock_guard<std::mutex> lock(_lock);
    return _configs.count(hash) != 0;
}

const nlohmann::json& ConfigTracker::getConfig(const std::string& hash) const {
    std::lock_guard<std::mutex> lock(_lock);
    auto it = _configs.find(hash);
    if (it == _configs.end())
        throw std::runtime_error("Unknown config hash: " + hash);
    return it->second;
}

std::string ConfigTracker::canonicalHash(const nlohmann::json& config_json) const {
    std::stringstream ss;

    // nlohmann::json::dump() uses an alpha-ordered map for objects, so the
    // config should be serialized in a consistent order.
    // Stick to a string dump, less likely to run into floating point issues.
    // We also remove the "updatable_config" field to ensure that the hash is
    // consistent regardless of whether the config was updated or not.
    nlohmann::json filtered_config;
    for (auto& [key, value] : config_json.items()) {
        if (key != "updatable_config") {
            filtered_config[key] = value;
        }
    }
    ss << filtered_config.dump(-1, '\0', false, nlohmann::json::error_handler_t::strict);

    std::string serialized = ss.str();
    unsigned char md5_result[MD5_DIGEST_LENGTH];

    // The MD5 function is deprecated in openssl 3.0, but we want to
    // maintain compatibility.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    MD5(reinterpret_cast<const unsigned char*>(serialized.c_str()), serialized.size(), md5_result);
#pragma GCC diagnostic pop


    std::stringstream md5_ss;
    for (int i = 0; i < MD5_DIGEST_LENGTH; ++i)
        md5_ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(md5_result[i]);

    return md5_ss.str();
}

std::string ConfigTracker::getCombinedHash() const {
    std::lock_guard<std::mutex> lock(_lock);
    return _combined_hash;
}

void ConfigTracker::setCombinedHash() {

    std::stringstream ss;
    {
        std::lock_guard<std::mutex> lock(_lock);
        for (const auto& [hash, _] : _configs) {
            ss << hash;
        }
    }

    // Get hash of the concatenated hashes

    unsigned char md5_result[MD5_DIGEST_LENGTH];

    // The MD5 function is deprecated in openssl 3.0, but we want to
    // maintain compatibility.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    MD5(reinterpret_cast<const unsigned char*>(ss.str().c_str()), ss.str().size(), md5_result);
#pragma GCC diagnostic pop

    // Convert the MD5 result to a hex string
    std::stringstream md5_ss;
    for (int i = 0; i < MD5_DIGEST_LENGTH; ++i)
        md5_ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(md5_result[i]);
    _combined_hash = md5_ss.str();

    DEBUG_NON_OO("Combined hash set to: {}", _combined_hash);
}


} // namespace kotekan
