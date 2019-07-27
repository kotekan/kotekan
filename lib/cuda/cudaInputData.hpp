#ifndef CUDA_INPUT_DATA_H
#define CUDA_INPUT_DATA_H

#include "cudaCommand.hpp"

class cudaInputData : public cudaCommand {
public:
    cudaInputData(kotekan::Config& config, const string& unique_name,
                kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device);
    ~cudaInputData();
    int wait_on_precondition(int gpu_frame_id) override;
    cudaEvent_t execute(int gpu_frame_id, cudaEvent_t pre_event) override;
    void finalize_frame(int frame_id) override;


protected:
    cudaEvent_t* data_staged_event;

    int32_t network_buffer_id;
    int32_t network_buffer_precondition_id;
    int32_t network_buffer_finalize_id;
    Buffer* network_buf;
    int32_t input_frame_len;

    int32_t _num_local_freq;
    int32_t _num_elements;
    int32_t _samples_per_data_set;
};

#endif // CUDA_INPUT_DATA_H
