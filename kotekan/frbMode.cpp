#include "frbMode.hpp"

#include "buffer.h"
#include "hsaProcess.hpp"
#include "chrxUplink.hpp"
#include "gpuPostProcess.hpp"
#include "networkOutputSim.hpp"
#include "nullProcess.hpp"
#include "vdifStream.hpp"
#include "network_dpdk.h"
#include "util.h"
#include "dpdkWrapper.hpp"
#include "processFactory.hpp"
#include "bufferContainer.hpp"
#include "gpuSimulate.hpp"
#include "gpuBeamformSimulate.hpp"

#include <vector>
#include <string>

using std::string;
using std::vector;

frbMode::frbMode(Config& config) : kotekanMode(config) {
}

frbMode::~frbMode() {
}

void frbMode::initalize_processes() {

    // Config values:
    int32_t num_gpus = config.get_int("/gpu", "num_gpus");
    int32_t num_local_freq = config.get_int("/", "num_local_freq");
    int32_t num_elements = config.get_int("/", "num_elements");
    int32_t num_data_sets = config.get_int("/", "num_data_sets");
    int32_t samples_per_data_set = config.get_int("/", "samples_per_data_set");
    int32_t downsample_time = config.get_int("/", "downsample_time");
    int32_t downsample_freq = config.get_int("/", "downsample_freq");
    int32_t buffer_depth = config.get_int("/", "buffer_depth");

    // Start HSA
    kotekan_hsa_start();

    // Create buffers.
    struct Buffer * network_input_buffer[num_gpus];

    // Create gpu output buffers.
    struct Buffer * beamform_output_buffer[num_gpus];

    // CPU simulate output:
    struct Buffer * cpu_output_buffer[num_gpus];

    int output_len = (samples_per_data_set/downsample_time/downsample_freq/2) * num_elements;

    char buffer_name[100];

    struct metadataPool *pool = create_metadata_pool(5 * buffer_depth,
                                                     sizeof(struct chimeMetadata));
    add_metadata_pool(pool);

    for (int i = 0; i < num_gpus; ++i) {

        DEBUG("Creating buffers...");

        snprintf(buffer_name, 100, "gpu_input_buffer_%d", i);
        network_input_buffer[i] = create_buffer(
                                        buffer_depth,
                                        samples_per_data_set * num_elements *
                                        num_local_freq * num_data_sets,
                                        pool,
                                        buffer_name);
        add_buffer(network_input_buffer[i]);

        snprintf(buffer_name, 100, "beamform_output_buffer_%d", i);
        beamform_output_buffer[i] = create_buffer(
                                        buffer_depth,
                                        output_len * num_data_sets * sizeof(float),
                                        pool,
                                        buffer_name);
        add_buffer(beamform_output_buffer[i]);

        snprintf(buffer_name, 100, "cpu_output_buffer_%d", i);
        cpu_output_buffer[i] = create_buffer(
                                        buffer_depth,
                                        output_len * num_data_sets * sizeof(float),
                                        pool,
                                        buffer_name);
        add_buffer(cpu_output_buffer[i]);
    }

    processFactory process_factory(config, buffer_container);
    vector<KotekanProcess *> processes = process_factory.build_processes();

    for (auto process: processes) {
        add_process(process);
    }

}
