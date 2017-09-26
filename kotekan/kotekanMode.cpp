#include "kotekanMode.hpp"
#include "buffer.h"

kotekanMode::kotekanMode(Config& config_) : config(config_) {

}

kotekanMode::~kotekanMode() {

    for (KotekanProcess* process : processes)
        if (process != nullptr)
            delete process;

    for (struct Buffer * buf : buffers) {
        if (buf != nullptr) {
            delete_buffer(buf);
            free(buf);
        }
    }

    for (struct metadataPool * metadata_pool : metadata_pools) {
        if (metadata_pool != nullptr) {
            delete_metadata_pool(metadata_pool);
            free(metadata_pool);
        }
    }
}

void kotekanMode::add_buffer(Buffer* buffer) {
    assert(buffer != nullptr);
    buffers.push_back(buffer);
}

void kotekanMode::add_process(KotekanProcess* process) {
    assert(process != nullptr);
    processes.push_back(process);
}

void kotekanMode::add_metadata_pool(struct metadataPool* metadata_pool) {
    assert(metadata_pool != nullptr);
    metadata_pools.push_back(metadata_pool);
}

void kotekanMode::initalize_processes() {

}

void kotekanMode::join() {
    for (KotekanProcess* process : processes)
        process->join();
}

void kotekanMode::start_processes() {
    for (KotekanProcess* process : processes)
        process->start();
}

void kotekanMode::stop_processes() {
    for (KotekanProcess* process : processes)
        process->stop();
}
