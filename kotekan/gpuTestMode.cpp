#include "gpuTestMode.hpp"
#include "buffers.h"
#include "gpuHSAThread.hpp"
#include "chrxUplink.hpp"
#include "gpuPostProcess.hpp"
#include "networkOutputSim.hpp"
#include "nullProcess.hpp"
#include "vdifStream.hpp"
#include "network_dpdk.h"
#include "hccGPUThread.hpp"
#include "util.h"
#include "testDataCheck.hpp"
#include "testDataGen.hpp"
#include "gpuSimulate.hpp"
#include "rawFileRead.hpp"
#include "rawFileWrite.hpp"

#include <vector>
#include <string>

using std::string;
using std::vector;

gpuTestMode::gpuTestMode(Config& config) : kotekanMode(config) {
}

gpuTestMode::~gpuTestMode() {
}

void gpuTestMode::initalize_processes() {
    // Config values:
    int32_t num_gpus = config.get_int("/gpu/num_gpus");
    int32_t num_adjusted_local_freq = config.get_int("/processing/num_adjusted_local_freq");
    int32_t num_adjusted_elements = config.get_int("/processing/num_adjusted_elements");
    int32_t block_size = config.get_int("/gpu/block_size");
    int32_t num_blocks = config.get_int("/gpu/num_blocks");
    int32_t num_data_sets = config.get_int("/processing/num_data_sets");
    int32_t samples_per_data_set = config.get_int("/processing/samples_per_data_set");
    int32_t buffer_depth = config.get_int("/processing/buffer_depth");

    // Start HSA
    kotekan_hsa_start();

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

        // Create simulation output buffers.
    struct Buffer * simulate_output_buffer[num_gpus];
    for (int i = 0; i < num_gpus; ++i) {
        simulate_output_buffer[i] = (struct Buffer *)malloc(sizeof(struct Buffer));
        add_buffer(simulate_output_buffer[i]);
    }

    // Create the shared pool of buffer info objects; used for recording information about a
    // given frame and past between buffers as needed.
    struct InfoObjectPool * pool[1];
    for (int i = 0; i < 1; ++i) {
        pool[i] = (struct InfoObjectPool *)malloc(sizeof(struct InfoObjectPool));
        add_info_object_pool(pool[i]);
    }

    cl_int output_len = num_adjusted_local_freq * num_blocks * (block_size*block_size)*2.;

    char buffer_name[100];

    create_info_pool(pool[0], 5 * buffer_depth,
                                    num_adjusted_local_freq,
                                    num_adjusted_elements);

    for (int i = 0; i < num_gpus; ++i) {

        DEBUG("Creating buffers...");

        int links_per_gpu = config.num_links_per_gpu(i);

        INFO("Num links for gpu[%d] = %d", i, links_per_gpu);

        snprintf(buffer_name, 100, "gpu_input_buffer_%d", i);
        create_buffer(network_input_buffer[i],
                      links_per_gpu * buffer_depth,
                      samples_per_data_set * num_adjusted_elements *
                      num_adjusted_local_freq * num_data_sets,
                      1,
                      2,
                      pool[0],
                      buffer_name);
        host_buffers[i].add_buffer("network_buf", network_input_buffer[i]);

        snprintf(buffer_name, 100, "gpu_output_buffer_%d", i);
        create_buffer(gpu_output_buffer[i],
                      links_per_gpu * buffer_depth,
                      output_len * num_data_sets * sizeof(cl_int),
                      1,
                      1,
                      pool[0],
                      buffer_name);
        host_buffers[i].add_buffer("output_buf", gpu_output_buffer[i]);

        snprintf(buffer_name, 100, "simulate_output_buffer_%d", i);
        create_buffer(simulate_output_buffer[i],
                      links_per_gpu * buffer_depth,
                      output_len * num_data_sets * sizeof(cl_int),
                      1,
                      1,
                      pool[0],
                      buffer_name);

        // TODO better management of the buffers so this list doesn't have to change size...

        // GPU Test
        add_process((KotekanProcess*) new rawFileRead(config, *network_input_buffer[i], true,
                                                        "/data/test_data/", "gpu_input_frame_const", "dat"));
        add_process((KotekanProcess*) new gpuHSAThread(config, host_buffers[i], i));
        add_process((KotekanProcess*) new rawFileRead(config, *simulate_output_buffer[i], false,
                                                        "/data/test_data/", "gpu_sim_output_frame_const", "dat"));
        add_process((KotekanProcess*) new testDataCheck(config, *gpu_output_buffer[i], *simulate_output_buffer[i] ) );

        // Processes to generate test data
        //add_process((KotekanProcess*) new testDataGen(config, *network_input_buffer[i]));
        //add_process((KotekanProcess*) new gpuSimulate(config, *network_input_buffer[i], *simulate_output_buffer[i]));
        //add_process((KotekanProcess*) new rawFileWrite(config, *network_input_buffer[i], "/data/test_data/", "gpu_input_frame_const", "dat"));
        //add_process((KotekanProcess*) new rawFileWrite(config, *simulate_output_buffer[i], "/data/test_data/", "gpu_sim_output_frame_const", "dat"));

    }
}

