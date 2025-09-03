#ifndef CL_INPUT_DATA_H
#define CL_INPUT_DATA_H

#include "clCommand.hpp"

class clInputData : public clCommand {
public:
    clInputData(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& host_buffers, clDeviceInterface& device,
                int instance_num);
    ~clInputData();
    int wait_on_precondition() override;
    cl_event execute(cl_event pre_event) override;
    void finalize_frame() override;


protected:
    cl_event* data_staged_event;

    Buffer* network_buf;
    int32_t input_frame_len;

    int32_t _num_local_freq;
    int32_t _num_elements;
    int32_t _samples_per_data_set;
};

#endif // CL_INPUT_DATA_H
