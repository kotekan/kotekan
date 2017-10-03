#include "singleDishModeGpu.hpp"

#include "buffer.h"
#include "hsaProcess.hpp"
#include "chrxUplink.hpp"
#include "gpuPostProcess.hpp"
#include "networkOutputSim.hpp"
#include "vdifStream.hpp"
#include "network_dpdk.h"
#include "util.h"
#include "dpdkWrapper.hpp"
#include "processFactory.hpp"
#include "bufferContainer.hpp"

#include "testDataCheck.hpp"
#include "testDataGen.hpp"
#include "rawFileRead.hpp"
#include "rawFileWrite.hpp"
#include "pyPlotResult.hpp"
#include "simVdifData.hpp"
#include "computeDualpolPower.hpp"
#include "networkPowerStream.hpp"
#include "vdif_functions.h"
#include "dpdkWrapper.hpp"
#include "chimeMetadata.h"

#include <vector>
#include <string>

using std::string;
using std::vector;

singleDishModeGpu::singleDishModeGpu(Config& config) : kotekanMode(config) {
}

singleDishModeGpu::~singleDishModeGpu() {
}

void singleDishModeGpu::initalize_processes() {

    // Config values:
    int num_gpus = config.get_int("/gpu", "num_gpus");
    int num_total_freq = config.get_int("/", "num_freq");
    int num_elements = config.get_int("/", "num_elements");
    int buffer_depth = config.get_int("/", "buffer_depth");
    int num_disks = config.get_int("/raw_capture", "num_disks");

    int integration_length = config.get_int("/", "integration_length");
    int timesteps_in = config.get_int("/", "samples_per_data_set");
    int timesteps_out = timesteps_in / integration_length;

    // Start HSA
    kotekan_hsa_start();

    bufferContainer buffer_container;

    // Create buffers.
    struct Buffer * network_input_buffer[num_gpus];

    // Create gpu output buffers.
    struct Buffer * gpu_output_buffer[num_gpus];

    // Create the shared pool of buffer info objects; used for recording information about a
    // given frame and past between buffers as needed.
    struct metadataPool *pool = create_metadata_pool(5 * num_disks * buffer_depth, sizeof(struct chimeMetadata));
    add_metadata_pool(pool);

    char buffer_name[100];

    for (int i = 0; i < num_gpus; ++i) {

       	    DEBUG("Creating buffers...");

	    snprintf(buffer_name, 100, "vdif_input_buf_%d", i);
	    network_input_buffer[i] = create_buffer(
                                    buffer_depth * num_disks,
                                    timesteps_in * num_elements * (num_total_freq + sizeof(VDIFHeader)),
                                    pool,
                                    buffer_name);
	    add_buffer(network_input_buffer[i]);

	    snprintf(buffer_name, 100, "gpu_output_buffer_%d", i);
	    gpu_output_buffer[i] = create_buffer(
                                    buffer_depth * num_disks,
                                    timesteps_in * num_elements * (num_total_freq + sizeof(VDIFHeader)),
                                    pool,
                                    buffer_name);
	    add_buffer(gpu_output_buffer[i]);
    }

    struct Buffer *output_buffer = create_buffer(
                                        buffer_depth,
                                        timesteps_out * (num_total_freq + 1) * num_elements * sizeof(float),
                                        pool,
                                        "output_power_buf");
    add_buffer(output_buffer);

    processFactory process_factory(config, buffer_container);
    vector<KotekanProcess *> processes = process_factory.build_processes();

    for (auto process: processes) {
        add_process(process);
    }

}
