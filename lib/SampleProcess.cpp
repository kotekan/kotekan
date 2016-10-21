#include "SampleProcess.hpp"
#include "errors.h"

SampleProcess::SampleProcess(Config &config) :
    KotekanProcess(config, std::bind(&SampleProcess::main_thread, this)) {
}

SampleProcess::~SampleProcess() {
}

void SampleProcess::main_thread() {
    INFO("Sample Process, reached main_thread!");
    while (!stop_thread) {
        INFO("In thread!");
    }
}

