#include "SampleProcess.hpp"
#include "errors.h"

SampleProcess::SampleProcess(Config &config, const string& unique_name, bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container, std::bind(&SampleProcess::main_thread, this)) {
}

SampleProcess::~SampleProcess() {
}

void SampleProcess::apply_config(uint64_t fpga_seq) {
    (void)fpga_seq;
}

void SampleProcess::main_thread() {
    INFO("Sample Process, reached main_thread!");
    while (!stop_thread) {
        INFO("In thread!");
    }
}

