#ifndef CL_BEAMFORM_KERNEL_H
#define CL_BEAMFORM_KERNEL_H

#include "Telescope.hpp"
#include "clCommand.hpp"
#include "clDeviceInterface.hpp"

#include <vector>

class clBeamformKernel : public clCommand {
public:
    clBeamformKernel(kotekan::Config& config, const std::string& unique_name,
                     kotekan::bufferContainer& host_buffers, clDeviceInterface& device,
                     int instance_num);
    ~clBeamformKernel();
    virtual void build() override;
    virtual cl_event execute(cl_event pre_event) override;

protected:
    cl_mem device_mask;

    std::vector<int32_t> _element_mask;
    std::vector<int32_t> _product_remap;
    std::vector<int32_t> _inverse_product_remap;
    uint32_t _scale_factor;

private:
    /// Number of elements on the telescope (e.g. 2048 - CHIME, 256 - Pathfinder).
    int32_t _num_elements;
    int32_t _num_data_sets;
    int32_t _samples_per_data_set;
    Buffer* network_buf;

    int32_t num_local_freq;

    // <streamID, freq_map>
    cl_mem get_freq_map(stream_t encoded_stream_id);
    std::map<uint64_t, cl_mem> device_freq_map;
};

#endif // CL_BEAMFORM_KERNEL_H
