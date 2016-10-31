#include "SampleProcess.hpp"
#include "errors.h"

SampleProcess::SampleProcess(Config &config) :
    KotekanProcess(config, std::bind(&SampleProcess::main_thread, this)) {
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

