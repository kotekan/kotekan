#ifndef CL_BEAMFORM_KERNEL_H
#define CL_BEAMFORM_KERNEL_H

#include "clCommand.hpp"
#include "clDeviceInterface.hpp"

#include <vector>

class clBeamformKernel : public clCommand {
public:
    clBeamformKernel(Config& config, const string& unique_name, bufferContainer& host_buffers,
                     clDeviceInterface& device);
    ~clBeamformKernel();
    virtual void build() override;
    virtual cl_event execute(int param_bufferID, cl_event pre_event) override;

protected:
    cl_mem device_mask;

    vector<int32_t> _element_mask;
    vector<int32_t> _product_remap;
    vector<int32_t> _inverse_product_remap;
    uint32_t _scale_factor;

private:
    /// Number of elements on the telescope (e.g. 2048 - CHIME, 256 - Pathfinder).
    int32_t _num_elements;
    int32_t _num_data_sets;
    int32_t _num_local_freq;
    int32_t _samples_per_data_set;
    Buffer* network_buf;

    // <streamID, freq_map>
    cl_mem get_freq_map(int32_t encoded_stream_id);
    std::map<int32_t, cl_mem> device_freq_map;
};

#endif // CL_BEAMFORM_KERNEL_H
