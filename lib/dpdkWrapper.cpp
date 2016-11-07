#include "dpdkWrapper.hpp"
#include "errors.h"
#include "network_dpdk.h"
#include "util.h"

#include <unistd.h>

dpdkWrapper::dpdkWrapper(Config& config, Buffer* network_input_buffer_[], string mode) :
    KotekanProcess(config, std::bind(&dpdkWrapper::main_thread, this)), _mode(mode) {

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
    _udp_packet_size = config.get_int("/dpdk/udp_packet_size");
    _num_data_sets = config.get_int("/processing/num_data_sets");
    _samples_per_data_set = config.get_int("/processing/samples_per_data_set");
    _buffer_depth = config.get_int("/processing/buffer_depth");
    _num_fpga_links = config.get_int("/dpdk/num_links");
    _timesamples_per_packet = config.get_int("/dpdk/timesamples_per_packet");
    _num_gpu_frames = config.get_int("/processing/num_gpu_frames");
    _num_lcores = config.get_int("/dpdk/num_lcores");
    _link_map = config.get_int_array("/gpu/link_map");
}

void dpdkWrapper::main_thread() {

    // Create network threads
    pthread_t network_dpdk_t;
    network_dpdk_args = (struct networkDPDKArg *)malloc(sizeof(struct networkDPDKArg));

    tmp_buffer = (struct Buffer ***)malloc(_num_fpga_links * sizeof(struct Buffer *));
    for (int port = 0; port < _num_fpga_links; ++port) {
        tmp_buffer[port] = (struct Buffer **)malloc(NUM_FREQ * sizeof(struct Buffer *));
    }

    // Populate the temp buffer which contains the pointers to the buffers to use.
    // This is a bit ugly because of its complexity,
    // and it's not obvious how it works.
    // TODO clean up.
    if (_mode == "shuffle4") {
        INFO("DPDK mode: shuffle4");
        for (int i = 0; i < _num_fpga_links; ++i) {
            for (int freq = 0; freq < NUM_FREQ; ++freq) {
                tmp_buffer[i][freq] = network_input_buffer[freq];
                INFO ("tmp_buffer[%d][%d] = %p", i, freq, tmp_buffer[i][freq]);
            }
            network_dpdk_args->num_links_in_group[i] = 1;
            network_dpdk_args->link_id[i] = 0;
        }
        network_dpdk_args->enable_shuffle = 1;
        network_dpdk_args->dump_full_packets = 0;

    } else if (_mode == "packet_cap") {
        INFO("DPDK mode: packet_cap");
        for (int i = 0; i < _num_fpga_links; ++i) {
            for (int freq = 0; freq < NUM_FREQ; ++freq) {
                tmp_buffer[i][freq] = network_input_buffer[i];
                INFO ("tmp_buffer[%d][%d] = %p", i, freq, tmp_buffer[i][freq]);
            }
            network_dpdk_args->num_links_in_group[i] = 1;
            network_dpdk_args->link_id[i] = 0;
        }
        network_dpdk_args->enable_shuffle = 1;
        network_dpdk_args->dump_full_packets = 1;

    } else if (_mode == "no_shuffle") {
        INFO("DPDK mode: no_shuffle");
        int current_gpu_id = 0;
        int current_link_id = 0;
        int32_t link_ids[_num_fpga_links];
        for (int i = 0; i < _num_fpga_links; ++i) {
            if (current_gpu_id != _link_map[i]) {
                current_gpu_id = _link_map[i];
                current_link_id = 0;
            }
            link_ids[i] = current_link_id++;
            INFO("link_ids[%d] = %d", i, link_ids[i]);
        }

        for (int i = 0; i < _num_fpga_links; ++i) {
            tmp_buffer[i][0] = network_input_buffer[_link_map[i]];
            network_dpdk_args->num_links_in_group[i] = config.num_links_per_gpu(i);
            network_dpdk_args->link_id[i] = link_ids[i];
        }
        network_dpdk_args->enable_shuffle = 0;
        network_dpdk_args->dump_full_packets = 0;
    } else {
        ERROR("DPDK Mode %s not supported!", _mode.c_str());
        return;
    }

    network_dpdk_args->buf = tmp_buffer;
    network_dpdk_args->vdif_buf = NULL;
    network_dpdk_args->num_links = _num_fpga_links;
    network_dpdk_args->timesamples_per_packet = _timesamples_per_packet;
    network_dpdk_args->samples_per_data_set = _samples_per_data_set;
    network_dpdk_args->num_data_sets = _num_data_sets;
    network_dpdk_args->num_gpu_frames = _num_gpu_frames;
    network_dpdk_args->udp_packet_size = _udp_packet_size;
    network_dpdk_args->num_lcores = _num_lcores;
    network_dpdk_args->num_links_per_lcore = _num_fpga_links / _num_lcores;
    network_dpdk_args->stop_capture = 0;
    for (int i = 0; i < _num_lcores; ++i) {
        network_dpdk_args->port_offset[i] = i * network_dpdk_args->num_links_per_lcore;
    }

    CHECK_ERROR( pthread_create(&network_dpdk_t, NULL, &network_dpdk_thread,
                                (void *)network_dpdk_args ) );

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int j = 0; j < 1; j++)
        CPU_SET(j, &cpuset);
    CHECK_ERROR( pthread_setaffinity_np(network_dpdk_t, sizeof(cpu_set_t), &cpuset) );

    while(!stop_thread) {
        sleep(10);
    }

    // Start exit
    network_dpdk_args->stop_capture = 1;
    int *ret;
    CHECK_ERROR( pthread_join (network_dpdk_t, (void **)  &ret) );
    INFO("DPDK Wrapper finished");
}
