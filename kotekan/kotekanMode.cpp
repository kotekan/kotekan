#include "kotekanMode.hpp"
#include "buffer.h"
#include "processFactory.hpp"
#include "metadataFactory.hpp"
#include "bufferFactory.hpp"
#include "restServer.hpp"
#include "configUpdater.hpp"
#include "json.hpp"

kotekanMode::kotekanMode(Config& config_) : config(config_) {
    restServer::instance().register_get_callback("/config", [&] (connectionInstance &conn) {
        conn.send_json_reply(config.get_full_config_json());
    });

#ifdef WITH_SSL
    restServer::instance().register_get_callback("/config_md5sum", [&] (connectionInstance &conn) {
        nlohmann::json reply;
        reply["md5sum"] = config.get_md5sum();
        conn.send_json_reply(reply);
    });
#endif

    restServer::instance().add_aliases_from_config(config);
}

kotekanMode::~kotekanMode() {

    restServer::instance().remove_get_callback("/config");
    restServer::instance().remove_all_aliases();

    for (auto const &process : processes) {
        if (process.second != nullptr) {
            delete process.second;
        }
    }

    for (auto const &buf: buffers) {
        if (buf.second != nullptr) {
            delete_buffer(buf.second);
            free(buf.second);
        }
    }

    for (auto const &metadata_pool : metadata_pools) {
        if (metadata_pool.second != nullptr) {
            delete_metadata_pool(metadata_pool.second);
            free(metadata_pool.second);
        }
    }
}

void kotekanMode::initalize_processes() {

    // Create Metadata Pool
    metadataFactory metadata_factory(config);
    metadata_pools = metadata_factory.build_pools();

    // Create Config Updater
    configUpdater &config_updater = configUpdater::instance();
    config_updater.apply_config(config);

    // Create Buffers
    bufferFactory buffer_factory(config, metadata_pools);
    buffers = buffer_factory.build_buffers();
    buffer_container.set_buffer_map(buffers);

    // Create Processes
    processFactory process_factory(config, buffer_container);
    processes = process_factory.build_processes();

    // Update REST server
    restServer::instance().set_server_affinity(config);
}

void kotekanMode::join() {
    for (auto const &process : processes) {
        INFO("Joining kotekan_process: %s...", process.first.c_str());
        process.second->join();
    }
}

void kotekanMode::start_processes() {
    for (auto const &process : processes) {
        INFO("Starting kotekan_process: %s...", process.first.c_str());
        process.second->start();
    }
}

void kotekanMode::stop_processes() {
    // Send shutdown signal to buffers
    for (auto const &buf : buffers) {
        INFO("Sending shutdown signal to buffer: %s", buf.first.c_str());
        send_shutdown_signal(buf.second);
    }

    for (auto const &process : processes)
        process.second->stop();
}
