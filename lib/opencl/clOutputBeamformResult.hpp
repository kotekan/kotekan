
#ifndef OUTPUT_BEAMFORM_RESULT_H
#define OUTPUT_BEAMFORM_RESULT_H

#include "clCommand.hpp"

class clOutputBeamformResult : public clCommand {
public:
    clOutputBeamformResult(kotekan::Config& config, const std::string& unique_name,
                           kotekan::bufferContainer& host_buffers, clDeviceInterface& device,
                           int instance_num);
    ~clOutputBeamformResult();
    int wait_on_precondition() override;
    virtual cl_event execute(cl_event param_PrecedeEvent) override;
    void finalize_frame() override;

protected:
private:
    int32_t _num_local_freq;
    int32_t _num_data_sets;
    int32_t _samples_per_data_set;

    Buffer* output_buffer;
    Buffer* network_buffer;
};

#endif
