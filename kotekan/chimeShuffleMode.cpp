#include "chimeShuffleMode.hpp"

#include "buffers.h"
#include "gpuHSAThread.hpp"
#include "chrxUplink.hpp"
#include "gpuPostProcess.hpp"
#include "networkOutputSim.hpp"
#include "nullProcess.hpp"
#include "vdifStream.hpp"
#include "network_dpdk.h"
#include "util.h"
#include "dpdkWrapper.hpp"

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
    int32_t num_gpus = config.get_int("/gpu/num_gpus");
    int32_t num_total_freq = config.get_int("/processing/num_total_freq");
    int32_t num_elements = config.get_int("/processing/num_elements");
    int32_t num_adjusted_local_freq = config.get_int("/processing/num_adjusted_local_freq");
    int32_t num_adjusted_elements = config.get_int("/processing/num_adjusted_elements");
    int32_t block_size = config.get_int("/gpu/block_size");
    int32_t num_blocks = config.get_int("/gpu/num_blocks");
    int32_t num_data_sets = config.get_int("/processing/num_data_sets");
    int32_t samples_per_data_set = config.get_int("/processing/samples_per_data_set");
    int32_t buffer_depth = config.get_int("/processing/buffer_depth");
    int32_t network_buffer_depth = config.get_int("/ch_master_network/network_buffer_depth");
    bool enable_upload = config.get_bool("/ch_master_network/enable_upload");
    bool enable_gating = config.get_bool("/gating/enable_gating");

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
                      4,
                      1,
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

        // TODO better management of the buffers so this list doesn't have to change size...
        add_process((KotekanProcess*) new gpuHSAThread(config, "hsa", host_buffers[i], i));
    }

    add_process((KotekanProcess *) new dpdkWrapper(config, "dpdk_input", network_input_buffer, "shuffle4") );

    struct Buffer * network_output_buffer = (struct Buffer *)malloc(sizeof(struct Buffer));
    struct Buffer * gated_output_buffer = (struct Buffer *)malloc(sizeof(struct Buffer));
    add_buffer(network_output_buffer);
    add_buffer(gated_output_buffer);

    // TODO this should move to an object
    int num_values = ((num_elements * (num_elements + 1)) / 2 ) * num_total_freq;

    const int tcp_buffer_size = sizeof(struct tcp_frame_header) +
        num_values * sizeof(complex_int_t) +
        num_total_freq * sizeof(struct per_frequency_data) +
        num_total_freq * num_elements * sizeof(struct per_element_data) +
        num_values * sizeof(uint8_t);

    const int gate_buffer_size = sizeof(struct gate_frame_header)
                            + num_values * sizeof(complex_int_t);

    create_buffer(network_output_buffer, network_buffer_depth, tcp_buffer_size,
                  1, 1, pool[0], "network_output_buffer");

    create_buffer(gated_output_buffer, network_buffer_depth, gate_buffer_size,
                  1, 1, pool[0], "gated_output_buffer");

    // The thread which creates output frame.
    gpuPostProcess * gpu_post_process = new gpuPostProcess(config, "gpu_post_process",
                                                            gpu_output_buffer,
                                                            *network_output_buffer,
                                                            *gated_output_buffer);
    add_process((KotekanProcess*)gpu_post_process);

    if (enable_upload) {
        add_process((KotekanProcess*) new chrxUplink(config, "chrx_uplink",
                                            *network_output_buffer, *gated_output_buffer));
    } else {
        // Drop the data.
        add_process((KotekanProcess*) new nullProcess(config, "null", *network_output_buffer));

        if (enable_gating) {
            add_process((KotekanProcess*) new nullProcess(config, "gated_output", *gated_output_buffer));
        }
    }
}
