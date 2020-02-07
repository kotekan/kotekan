#ifndef CL_INPUT_DATA_H
#define CL_INPUT_DATA_H

#include "clCommand.hpp"

class clInputData : public clCommand {
public:
    clInputData(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& host_buffers, clDeviceInterface& device);
    ~clInputData();
    int wait_on_precondition(int gpu_frame_id) override;
    cl_event execute(int gpu_frame_id, cl_event pre_event) override;
    void finalize_frame(int frame_id) override;


protected:
    cl_event* data_staged_event;

    int32_t network_buffer_id;
    int32_t network_buffer_precondition_id;
    int32_t network_buffer_finalize_id;
    Buffer* network_buf;
    int32_t input_frame_len;

    int32_t _num_local_freq;
    int32_t _num_elements;
    int32_t _samples_per_data_set;
};

#endif // CL_INPUT_DATA_H
