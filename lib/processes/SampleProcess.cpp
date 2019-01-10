#include "SampleProcess.hpp"

#include "errors.h"

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::KotekanProcess;

REGISTER_KOTEKAN_PROCESS(SampleProcess);

SampleProcess::SampleProcess(Config& config, const string& unique_name,
                             bufferContainer& buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&SampleProcess::main_thread, this)) {}

SampleProcess::~SampleProcess() {}

void SampleProcess::main_thread() {
    INFO("Sample Process, reached main_thread!");
    while (!stop_thread) {
        INFO("In thread!");
    }
}
