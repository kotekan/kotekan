#include "dpdkWrapper.hpp"
#include "errors.h"
#include "network_dpdk.h"
#include "util.h"

#include <unistd.h>

dpdkWrapper::dpdkWrapper(Config& config, Buffer* network_input_buffer_[]) :
    KotekanProcess(config, std::bind(&dpdkWrapper::main_thread, this)) {

    apply_config(0);

    network_input_buffer = (struct Buffer **)malloc(_num_fpga_links * sizeof (struct Buffer *));
    for (int i = 0; i < _num_fpga_links; ++i) {
        network_input_buffer[i] = network_input_buffer_[i];
    }
}

dpdkWrapper::~dpdkWrapper() {
    if (network_dpdk_args != nullptr)
        free(network_dpdk_args);

    if (tmp_buffer != nullptr) {
        for (int port = 0; port < _num_fpga_links; ++port) {
            free(tmp_buffer[port]);
        }
        free(tmp_buffer);
    }

    if (network_input_buffer != nullptr)
        free(network_input_buffer);
}

void dpdkWrapper::apply_config(uint64_t fpga_seq) {
    _udp_packet_size = config.get_int("/fpga_network/udp_packet_size");
    _num_data_sets = config.get_int("/processing/num_data_sets");
    _samples_per_data_set = config.get_int("/processing/samples_per_data_set");
    _buffer_depth = config.get_int("/processing/buffer_depth");
    _num_fpga_links = config.get_int("/fpga_network/num_links");
    _timesamples_per_packet = config.get_int("/fpga_network/timesamples_per_packet");
    _num_gpu_frames = config.get_int("/processing/num_gpu_frames");
}

void dpdkWrapper::main_thread() {

    // Create network threads
    pthread_t network_dpdk_t;
    network_dpdk_args = (struct networkDPDKArg *)malloc(sizeof(struct networkDPDKArg));

    tmp_buffer = (struct Buffer ***)malloc(_num_fpga_links * sizeof(struct Buffer *));
    for (int port = 0; port < _num_fpga_links; ++port) {
        tmp_buffer[port] = (struct Buffer **)malloc(NUM_FREQ * sizeof(struct Buffer *));
    }

    // TODO move to function
    int32_t link_ids[_num_fpga_links];
    for (int i = 0; i < _num_fpga_links; ++i) {
        link_ids[i] = 0;
    }

    // Start DPDK
    for (int i = 0; i < _num_fpga_links; ++i) {
        for (int freq = 0; freq < NUM_FREQ; ++freq) {
            tmp_buffer[i][freq] = network_input_buffer[i];
            INFO ("tmp_buffer[%d][%d] = %p", i, freq, tmp_buffer[i][freq]);
        }
        network_dpdk_args->num_links_in_group[i] = 1;
        network_dpdk_args->link_id[i] = link_ids[i];
    }
    network_dpdk_args->buf = tmp_buffer;
    network_dpdk_args->vdif_buf = NULL;
    network_dpdk_args->num_links = _num_fpga_links;
    network_dpdk_args->timesamples_per_packet = _timesamples_per_packet;
    network_dpdk_args->samples_per_data_set = _samples_per_data_set;
    network_dpdk_args->num_data_sets = _num_data_sets;
    network_dpdk_args->num_gpu_frames = _num_gpu_frames;
    network_dpdk_args->udp_packet_size = _udp_packet_size;
    network_dpdk_args->num_lcores = 4;
    network_dpdk_args->num_links_per_lcore = 1;
    network_dpdk_args->port_offset[0] = 0;
    network_dpdk_args->port_offset[1] = 1;
    network_dpdk_args->port_offset[2] = 2;
    network_dpdk_args->port_offset[3] = 3;
    network_dpdk_args->enable_shuffle = 0;
    network_dpdk_args->dump_full_packets = 1;

    CHECK_ERROR( pthread_create(&network_dpdk_t, NULL, &network_dpdk_thread,
                                (void *)network_dpdk_args ) );

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int j = 0; j < 1; j++)
        CPU_SET(j, &cpuset);
    CHECK_ERROR( pthread_setaffinity_np(network_dpdk_t, sizeof(cpu_set_t), &cpuset) );

    while(!stop_thread) {
        sleep(1000);
        // TODO shutdown dpdk here.
    }
}
