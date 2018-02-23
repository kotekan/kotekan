#include "dpdkWrapper.hpp"
#include "errors.h"
#include "network_dpdk.h"
#include "util.h"
#include "prometheusMetrics.hpp"

#include <unistd.h>

REGISTER_KOTEKAN_PROCESS(dpdkWrapper);

dpdkWrapper::dpdkWrapper(Config& config, const string& unique_name,
                         bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&dpdkWrapper::main_thread, this)) {

    _mode = config.get_string(unique_name, "mode");
    apply_config(0);

    network_input_buffer = (struct Buffer **)malloc(_num_fpga_links * sizeof (struct Buffer *));
    if (_mode == "vdif") {
        network_input_buffer[0] = get_buffer("vdif_buf");
    } else if (_mode == "no_shuffle") {
        for (int i = 0; i < _num_gpus; ++i) {
            network_input_buffer[i] = get_buffer("network_out_buf_" + std::to_string(i));
        }
    } else {
        for (int i = 0; i < _num_fpga_links; ++i) {
            network_input_buffer[i] = get_buffer("network_out_buf_" + std::to_string(i));
        }
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
    _udp_packet_size = config.get_int(unique_name, "udp_packet_size");
    _num_data_sets = config.get_int(unique_name, "num_data_sets");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");
    _buffer_depth = config.get_int(unique_name, "buffer_depth");
    _num_fpga_links = config.get_int(unique_name, "num_links");
    _timesamples_per_packet = config.get_int(unique_name, "timesamples_per_packet");
    _num_gpu_frames = config.get_int(unique_name, "num_gpu_frames");
    _num_lcores = config.get_int(unique_name, "num_lcores");
    if (_mode == "no_shuffle" ) {
        _link_map = config.get_int_array(unique_name, "link_map");
        _num_gpus = config.get_int(unique_name, "num_gpus");
    }
}

void dpdkWrapper::main_thread() {

    // Create network threads
    pthread_t network_dpdk_t;
    network_dpdk_args = (struct networkDPDKArg *)malloc(sizeof(struct networkDPDKArg));

    tmp_buffer = (struct Buffer ***)malloc(_num_fpga_links * sizeof(struct Buffer *));
    for (int port = 0; port < _num_fpga_links; ++port) {
        tmp_buffer[port] = (struct Buffer **)malloc(NUM_FREQ * sizeof(struct Buffer *));
    }

    for (int i = 0; i < _num_fpga_links; ++i) {
        network_dpdk_args->rx_packets_total[i] = 0;
        network_dpdk_args->rx_bytes_total[i] = 0;
        network_dpdk_args->lost_packets_total[i] = 0;
        network_dpdk_args->rx_errors_total[i] = 0;
    }

    // Populate the temp buffer which contains the pointers to the buffers to use.
    // This is a bit ugly because of its complexity,
    // and it's not obvious how it works.
    // TODO clean up.
    network_dpdk_args->fake_stream_ids = 0;
    if (_mode == "shuffle4") {
        INFO("DPDK mode: shuffle4");
        for (int i = 0; i < _num_fpga_links; ++i) {
            strncpy(network_dpdk_args->producer_names[i],
                    (unique_name + "_" + std::to_string(i)).c_str(), 128);

            for (int freq = 0; freq < NUM_FREQ; ++freq) {
                tmp_buffer[i][freq] = network_input_buffer[freq];
                INFO ("tmp_buffer[%d][%d] = %p", i, freq, tmp_buffer[i][freq]);
                register_producer(tmp_buffer[i][freq], network_dpdk_args->producer_names[i]);
            }
            network_dpdk_args->num_links_in_group[i] = 1;
            network_dpdk_args->link_id[i] = 0;
        }
        network_dpdk_args->enable_shuffle = 1;
        network_dpdk_args->dump_full_packets = 0;
        network_dpdk_args->fake_stream_ids = 0;

    } else if (_mode == "packet_cap") {
        INFO("DPDK mode: packet_cap");
        for (int i = 0; i < _num_fpga_links; ++i) {
            tmp_buffer[i][0] = network_input_buffer[i];
            INFO ("tmp_buffer[%d][%d] = %p", i, 0, tmp_buffer[i][0]);
            strncpy(network_dpdk_args->producer_names[i],
                    (unique_name + "_" + std::to_string(i)).c_str(), 128);
            register_producer(tmp_buffer[i][0], network_dpdk_args->producer_names[i]);

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
            network_dpdk_args->num_links_in_group[i] = config.num_links_per_gpu(_link_map[i]);
            network_dpdk_args->link_id[i] = link_ids[i];
            strncpy(network_dpdk_args->producer_names[i],
                    (unique_name + "_" + std::to_string(_link_map[i])).c_str(), 128);
        }
        // Only register one producer per GPU
        current_gpu_id = -1;
        for (int i = 0; i < _num_fpga_links; ++i) {
            if (current_gpu_id != _link_map[i]) {
                current_gpu_id = _link_map[i];
                register_producer(tmp_buffer[i][0], network_dpdk_args->producer_names[i]);
            }
        }
        network_dpdk_args->enable_shuffle = 0;
        network_dpdk_args->dump_full_packets = 0;
    } else if (_mode == "vdif") {
        INFO("DPDK mode: vdif");
        for (int i = 0; i < _num_fpga_links; ++i) {
            network_dpdk_args->link_id[i] = i;
            network_dpdk_args->num_links_in_group[i] = _num_fpga_links;
            strncpy(network_dpdk_args->producer_names[i],
                    (unique_name + "_" + std::to_string(i)).c_str(), 128);
            register_producer(network_input_buffer[0], network_dpdk_args->producer_names[i]);
        }
        network_dpdk_args->enable_shuffle = 0;
        network_dpdk_args->dump_full_packets = 0;
    } else {
        ERROR("DPDK Mode %s not supported!", _mode.c_str());
        return;
    }

    if (_mode == "vdif") {
        network_dpdk_args->buf = NULL;
        network_dpdk_args->vdif_buf = network_input_buffer[0];
    } else {
        network_dpdk_args->buf = tmp_buffer;
        network_dpdk_args->vdif_buf = NULL;
    }

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

    prometheusMetrics &metrics = prometheusMetrics::instance();

    while(!stop_thread) {
        usleep(200000);
        for (int i = 0; i < _num_fpga_links; ++i) {
            // Add packet metrics
            metrics.add_process_metric("rx_packets_total",
                                        unique_name,
                                        network_dpdk_args->rx_packets_total[i],
                                        "port='" + std::to_string(i) + "'");
            metrics.add_process_metric("lost_packets_total",
                                        unique_name,
                                        network_dpdk_args->lost_packets_total[i],
                                        "port='" + std::to_string(i) + "'");
            // Give these metrics in terms of frames as well
            metrics.add_process_metric("rx_frames_total",
                                        unique_name,
                                        network_dpdk_args->rx_packets_total[i] * _timesamples_per_packet,
                                        "port='" + std::to_string(i) + "'");
            metrics.add_process_metric("lost_frames_total",
                                        unique_name,
                                        network_dpdk_args->lost_packets_total[i] * _timesamples_per_packet,
                                        "port='" + std::to_string(i) + "'");

            // Add total rx bytes
            metrics.add_process_metric("rx_bytes_total",
                                        unique_name,
                                        network_dpdk_args->rx_bytes_total[i],
                                        "port='" + std::to_string(i) + "'");

            metrics.add_process_metric("rx_errors_total",
                                        unique_name,
                                        network_dpdk_args->rx_errors_total[i],
                                        "port='" + std::to_string(i) + "'");
        }
    }

    // Start exit
    network_dpdk_args->stop_capture = 1;
    int *ret;
    CHECK_ERROR( pthread_join (network_dpdk_t, (void **)  &ret) );
    INFO("DPDK Wrapper finished");
}
