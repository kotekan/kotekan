#ifndef HSA_OUTPUT_DATA_H
#define HSA_OUTPUT_DATA_H

#include <string>

#include "hsaCorrelatorSubframeCommand.hpp"

class hsaOutputData: public hsaCorrelatorSubframeCommand
{
public:

    hsaOutputData(Config &config, const string &unique_name,
                  bufferContainer &host_buffers, hsaDeviceInterface &device);

    virtual ~hsaOutputData();

    int wait_on_precondition(int gpu_frame_id) override;

    hsa_signal_t execute(int gpu_frame_id,
                         hsa_signal_t precede_signal) override;

    void finalize_frame(int frame_id) override;

private:

    /// Use the same consumer/producer name accross subframes, 
    /// but unique for each GPU.
    std::string static_unique_name;

    Buffer * network_buffer;
    Buffer * lost_samples_buf;
    Buffer * output_buffer;

    int32_t network_buffer_id;

    int32_t output_buffer_precondition_id;
    int32_t output_buffer_excute_id;
    int32_t output_buffer_id;

    int32_t lost_samples_buf_id;
};

#endif