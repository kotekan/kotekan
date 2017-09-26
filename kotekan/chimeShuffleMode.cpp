#include "chimeShuffleMode.hpp"

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
#include "accumulate.hpp"

#include <vector>
#include <string>

using std::string;
using std::vector;

chimeShuffleMode::chimeShuffleMode(Config& config) : kotekanMode(config) {
}

chimeShuffleMode::~chimeShuffleMode() {
}

void chimeShuffleMode::initalize_processes() {

    // Config values:
    int32_t num_gpus = config.get_int("/gpu", "num_gpus");
    int32_t num_local_freq = config.get_int("/", "num_local_freq");
    int32_t num_elements = config.get_int("/", "num_elements");
    int32_t block_size = config.get_int("/gpu", "block_size");
    int32_t num_blocks = (int32_t)(num_elements / block_size) *
                         (num_elements / block_size + 1) / 2.;
    int32_t num_data_sets = config.get_int("/", "num_data_sets");
    int32_t samples_per_data_set = config.get_int("/", "samples_per_data_set");
    int32_t buffer_depth = config.get_int("/", "buffer_depth");

    // Start HSA
    kotekan_hsa_start();

    bufferContainer buffer_container;

    // Create buffers.
    struct Buffer * network_input_buffer[num_gpus];
    for (int i = 0; i < num_gpus; ++i) {
        network_input_buffer[i] = (struct Buffer *)malloc(sizeof(struct Buffer));
        add_buffer(network_input_buffer[i]);
    }

    // Create gpu output buffers.
    struct Buffer * gpu_output_buffer[num_gpus];
    for (int i = 0; i < num_gpus; ++i) {
        gpu_output_buffer[i] = (struct Buffer *)malloc(sizeof(struct Buffer));
        add_buffer(gpu_output_buffer[i]);
    }

    // Create file output buffers.
    struct Buffer * file_output_buffer[num_gpus];
    for (int i = 0; i < num_gpus; ++i) {
        file_output_buffer[i] = (struct Buffer *)malloc(sizeof(struct Buffer));
        add_buffer(file_output_buffer[i]);
    }

    // Create the shared pool of buffer info objects; used for recording information about a
    // given frame and past between buffers as needed.
    struct metadataPool *pool = create_metadata_pool(10 * buffer_depth, sizeof(struct chimeMetadata));
    add_metadata_pool(pool);

    int output_len = num_local_freq * num_blocks * (block_size*block_size)*2.;

    char buffer_name[100];

    for (int i = 0; i < num_gpus; ++i) {

        DEBUG("Creating buffers...");

        snprintf(buffer_name, 100, "gpu_input_buffer_%d", i);
        create_buffer(network_input_buffer[i],
                      buffer_depth,
                      samples_per_data_set * num_elements *
                      num_local_freq * num_data_sets,
                      pool,
                      buffer_name);
        buffer_container.add_buffer(buffer_name, network_input_buffer[i]);

        snprintf(buffer_name, 100, "gpu_output_buffer_%d", i);
        create_buffer(gpu_output_buffer[i],
                      buffer_depth,
                      output_len * num_data_sets * sizeof(int32_t),
                      pool,
                      buffer_name);
        buffer_container.add_buffer(buffer_name, gpu_output_buffer[i]);

        snprintf(buffer_name, 100, "file_output_buffer_%d", i);
        create_buffer(file_output_buffer[i],
                      buffer_depth,
                      output_len * num_data_sets * sizeof(int32_t) +
                      sizeof(struct rawGPUFrameHeader),
                      pool,
                      buffer_name);
        buffer_container.add_buffer(buffer_name, file_output_buffer[i]);

    }

    processFactory process_factory(config, buffer_container);
    vector<KotekanProcess *> processes = process_factory.build_processes();

    for (auto process: processes) {
        add_process(process);
    }

}
