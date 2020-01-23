#ifndef GPU_POST_PROCESS
#define GPU_POST_PROCESS

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // for bufferContainer
#include "restServer.hpp"      // for connectionInstance

#include "json.hpp" // for json

#include <stdint.h>   // for int32_t, uint32_t, int64_t, uint64_t
#include <string>     // for string
#include <sys/time.h> // for timeval
#include <vector>     // for vector


#define MAX_GATE_DESCRIPTION_LEN 127
#define HDF5_NAME_LEN 65

class gpuPostProcess : public kotekan::Stage {
public:
    gpuPostProcess(kotekan::Config& config_, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);
    virtual ~gpuPostProcess();
    void main_thread() override;

    void vis_endpoint(kotekan::connectionInstance& conn, nlohmann::json& json_request);

private:
    struct Buffer** in_buf;
    struct Buffer* out_buf;
    struct Buffer* gate_buf;

    // Config variables
    // Aside (wow this stage needs a lot of configuration options)
    int32_t _num_elem;
    int32_t _num_total_freq;
    int32_t _num_local_freq;
    int32_t _num_data_sets;
    int32_t _samples_per_data_set;
    int32_t _num_gpu_frames;
    int32_t _num_blocks;
    int32_t _block_size;
    std::vector<int32_t> _link_map;
    std::vector<int32_t> _product_remap;
    int32_t* _product_remap_c = nullptr;
    int32_t _num_fpga_links;
    bool _enable_basic_gating;
    int32_t _gate_phase;
    int32_t _gate_cadence;
    int32_t _num_gpus;
};

// A TCP frame contains this header followed by the visibilities, and flags.
// -- HEADER:sizeof(TCP_frame_header) --
// -- VISIBILITIES:n_corr * n_freq * sizeof(complex_int_t) --
// -- PER_FREQUENCY_DATA:n_freq * sizeof(per_frequency_data) --
// -- PER_ELEMENT_DATA:n_freq * n_elem * sizeof(per_element_data) --
// -- VIS_WEIGHT:n_corr * n_freq * sizeof(uint8_t) --
#pragma pack(1)
struct stream_id {
    unsigned int link_id : 8;
    unsigned int slot_id : 8;
    unsigned int crate_id : 8;
    unsigned int reserved : 8;
};

struct tcp_frame_header {
    uint64_t fpga_seq_number;
    uint32_t num_freq;
    uint32_t num_vis; // The number of visibilities per frequency.
    uint32_t num_elements;
    uint32_t num_links; // The number of GPU links in this frame.
    uint32_t num_gates;
    struct timeval cpu_timestamp; // The time stamp as set by the GPU correlator - not accurate!
    double kotekan_version;
    char kotekan_git_hash[64];
};

struct per_frequency_data {
    struct stream_id stream_id;
    uint32_t index; // The position in the FPGA frame which is assoicated with
                    // this frequency.
    uint32_t lost_packet_count;
    uint32_t rfi_count;
};

struct per_element_data {
    uint32_t fpga_adc_count;
    uint32_t fpga_fft_count;
    uint32_t fpga_scalar_count;
};
#pragma pack(0)

#pragma pack(1)
struct gate_frame_header {
    int set_num;
    double folding_period;
    double folding_start;
    int64_t fpga_count_start;
    char description[MAX_GATE_DESCRIPTION_LEN];
    char gate_vis_name[HDF5_NAME_LEN];
    double gate_weight[2]; // TODO This value should be made dynamic
};
#pragma pack(0)

#endif
