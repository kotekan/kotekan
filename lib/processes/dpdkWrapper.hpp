#ifndef DPDK_WRAPPER_HPP
#define DPDK_WRAPPER_HPP

#include "KotekanProcess.hpp"

#include <string>
using std::string;

class dpdkWrapper : public KotekanProcess {
public:
    dpdkWrapper(Config& config, const string& unique_name,
                         bufferContainer &buffer_container);
    virtual ~dpdkWrapper();
    void main_thread();
    virtual void apply_config(uint64_t fpga_seq);

private:

    struct networkDPDKArg * network_dpdk_args = nullptr;
    struct Buffer *** tmp_buffer = nullptr;
    // This is the main list of buffers;
    struct Buffer ** network_input_buffer = nullptr;

    // Config options
    int32_t _udp_packet_size;
    int32_t _num_data_sets;
    int32_t _samples_per_data_set;
    int32_t _buffer_depth;
    int32_t _num_fpga_links;
    int32_t _timesamples_per_packet;
    int32_t _num_gpu_frames;
    int32_t _num_lcores;
    int32_t _num_gpus;
    string _mode;
    vector<int32_t> _link_map;
};


#endif /* PACKET_CAP_HPP */