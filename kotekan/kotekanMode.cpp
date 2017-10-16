#include "kotekanMode.hpp"
#include "buffer.h"
#include "processFactory.hpp"
#include "metadataFactory.hpp"
#include "bufferFactory.hpp"

kotekanMode::kotekanMode(Config& config_) : config(config_) {

}

kotekanMode::~kotekanMode() {

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

    // Create Buffers
    bufferFactory buffer_factory(config, metadata_pools);
    buffers = buffer_factory.build_buffers();
    buffer_container.set_buffer_map(buffers);

    // Create Processes
    processFactory process_factory(config, buffer_container);
    processes = process_factory.build_processes();

}

void kotekanMode::join() {
    for (auto const &process : processes)
        process.second->join();
}

void kotekanMode::start_processes() {
    for (auto const &process : processes)
        process.second->start();
}

void kotekanMode::stop_processes() {
    for (auto const &process : processes)
        process.second->stop();
}
