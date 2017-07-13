#include "processFactory.hpp"

#include "errors.h"
#include "processes/beamformingPostProcess.hpp"

#include "beamformingPostProcess.hpp"
#include "chrxUplink.hpp"
#include "computeDualpolPower.hpp"
#ifdef WITH_DPDK
    #include "dpdkWrapper.hpp"
#endif
#include "fullPacketDump.hpp"
#include "gpuPostProcess.hpp"
#include "nDiskFileWrite.hpp"
#include "nDiskFileRead.hpp"
#include "networkPowerStream.hpp"
#include "nullProcess.hpp"
#include "pyPlotResult.hpp"
#include "rawFileRead.hpp"
#include "rawFileWrite.hpp"
#include "vdifStream.hpp"
#include "recvSingleDishVDIF.hpp"
#include "streamSingleDishVDIF.hpp"

#include "gpuBeamformSimulate.hpp"
#include "gpuSimulate.hpp"
#include "networkOutputSim.hpp"
#include "simVdifData.hpp"
#include "testDataCheck.hpp"
#include "testDataGen.hpp"
#include "constDataCheck.hpp"

#ifdef WITH_HSA
    #include "hsaProcess.hpp"
#endif
#ifdef WITH_OPENCL
    #include "clProcess.hpp"
#endif

processFactory::processFactory(Config& config,
                               bufferContainer& buffer_container) :
    config(config),
    buffer_container(buffer_container) {

}

processFactory::~processFactory() {
}

vector<KotekanProcess*> processFactory::build_processes() {
    vector<KotekanProcess*> processes;

    // Start parsing tree, put the processes in the "processes" vector
    build_from_tree(processes, config.get_full_config_json(), "");

    return processes;
}

void processFactory::build_from_tree(vector<KotekanProcess*>& processes, json& config_tree, const string& path) {

    for (json::iterator it = config_tree.begin(); it != config_tree.end(); ++it) {
        // If the item isn't an object we can just ignore it.
        if (!it.value().is_object()) {
            continue;
        }

        // Check if this is a kotekan_process block, and if so create the process.
        string process_name = it.value().value("kotekan_process", "none");
        if (process_name != "none") {
            processes.push_back(new_process(process_name, path + "/" + it.key()));
            continue;
        }

        // Recursive part.
        // This is a section/scope not a process block.
        build_from_tree(processes, it.value(), path + "/" + it.key());
    }
}


KotekanProcess* processFactory::new_process(const string& name, const string& location) {
    // I wish there was a better way to do this...  The only thing I can think of
    // involves the compiler preprocessor.

    INFO("Creating process type: %s, at config tree path: %s", name.c_str(), location.c_str());

    // ****** processes directory ******
    if (name == "beamformingPostProcess") {
        return (KotekanProcess *) new beamformingPostProcess(config, location, buffer_container);
    }

    if (name == "chrxUplink") {
        return (KotekanProcess *) new chrxUplink(config, location, buffer_container);
    }

    if (name == "computeDualpolPower") {
        return (KotekanProcess *) new computeDualpolPower(config, location, buffer_container);
    }
#ifdef WITH_DPDK
    if (name == "dpdkWrapper") {
        return (KotekanProcess *) new dpdkWrapper(config, location, buffer_container);
    }
#endif
    if (name == "fullPacketDump") {
        return (KotekanProcess *) new fullPacketDump(config, location, buffer_container);
    }

    if (name == "gpuPostProcess") {
        return (KotekanProcess *) new gpuPostProcess(config, location, buffer_container);
    }

    if (name == "nDiskFileWrite") {
        return (KotekanProcess *) new nDiskFileWrite(config, location, buffer_container);
    }

    if (name == "nDiskFileRead") {
        return (KotekanProcess *) new nDiskFileRead(config, location, buffer_container);
    }

    if (name == "networkPowerStream") {
        return (KotekanProcess *) new networkPowerStream(config, location, buffer_container);
    }

    // Remove this when buffer consumer/producers are dynamic.
    if (name == "nullProcess") {
        return (KotekanProcess *) new nullProcess(config, location, buffer_container);
    }

    if (name == "pyPlotResult") {
        return (KotekanProcess *) new pyPlotResult(config, location, buffer_container);
    }

    if (name == "rawFileRead") {
        return (KotekanProcess *) new rawFileRead(config, location, buffer_container);
    }

    if (name == "vdifStream") {
        return (KotekanProcess *) new vdifStream(config, location, buffer_container);
    }

    if (name == "streamSingleDishVDIF") {
        return (KotekanProcess *) new streamSingleDishVDIF(config, location, buffer_container);
    }

    if (name == "recvSingleDishVDIF") {
        return (KotekanProcess *) new recvSingleDishVDIF(config, location, buffer_container);
    }

    // ****** testing directory ******
    if (name == "gpuBeamformSimulate") {
        return (KotekanProcess *) new gpuBeamformSimulate(config, location, buffer_container);
    }

    if (name == "gpuSimulate") {
        return (KotekanProcess *) new gpuSimulate(config, location, buffer_container);
    }

    if (name == "networkOutputSim") {
        return (KotekanProcess *) new networkOutputSim(config, location, buffer_container);
    }

    if (name == "simVdifData") {
        return (KotekanProcess *) new simVdifData(config, location, buffer_container);
    }

    if (name == "testDataCheck") {
        // TODO This is a template class, how to set template type?
        return (KotekanProcess *) new testDataCheck<int32_t>(config, location, buffer_container);
    }

    if (name == "constDataCheck") {
        // TODO This is a template class, how to set template type?
        return (KotekanProcess *) new constDataCheck(config, location, buffer_container);
    }

    if (name == "testDataGen") {
        return (KotekanProcess *) new testDataGen(config, location, buffer_container);
    }

    // OpenCL
    if (name == "clProcess") {
        #ifdef WITH_OPENCL
            return (KotekanProcess *) new clProcess(config, location, buffer_container);
        #else
            throw std::runtime_error("hsaProcess is not supported on this system");
        #endif
    }

    // HSA
    if (name == "hsaProcess") {
        #ifdef WITH_HSA
            return (KotekanProcess *) new hsaProcess(config, location, buffer_container);
        #else
            throw std::runtime_error("hsaProcess is not supported on this system");
        #endif
    }

    // No process found
    throw std::runtime_error("No process named " + name);
}
