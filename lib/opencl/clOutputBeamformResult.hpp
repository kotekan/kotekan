
#ifndef OUTPUT_BEAMFORM_RESULT_H
#define OUTPUT_BEAMFORM_RESULT_H

#include "clCommand.hpp"

class clOutputBeamformResult : public clCommand {
public:
    clOutputBeamformResult(kotekan::Config& config, const std::string& unique_name,
                           kotekan::bufferContainer& host_buffers, clDeviceInterface& device);
    ~clOutputBeamformResult();
    int wait_on_precondition(int gpu_frame_id) override;
    virtual cl_event execute(int param_bufferID, cl_event param_PrecedeEvent) override;
    void finalize_frame(int frame_id) override;

protected:
private:
    int32_t _num_local_freq;
    int32_t _num_data_sets;
    int32_t _samples_per_data_set;

    int32_t output_buffer_execute_id;
    int32_t output_buffer_precondition_id;

    Buffer* output_buffer;
    Buffer* network_buffer;

    int32_t output_buffer_id;
    int32_t network_buffer_id;
};

#endif
