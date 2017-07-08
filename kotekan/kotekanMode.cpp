#include "kotekanMode.hpp"
#include "buffer.c"

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

    for (struct InfoObjectPool * info_object : info_pools) {
        if (info_object != nullptr) {
            delete_info_object_pool(info_object);
            free(info_object);
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

void kotekanMode::add_info_object_pool(InfoObjectPool* info_pool) {
    assert(info_pool != nullptr);
    info_pools.push_back(info_pool);
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
