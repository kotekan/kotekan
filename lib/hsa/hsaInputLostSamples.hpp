#ifndef HSA_INPUT_LOST_SAMPLES_H
#define HSA_INPUT_LOST_SAMPLES_H

#include "hsaCommand.hpp"

class hsaInputLostSamples: public hsaCommand
{
    public:
        hsaInputLostSamples( Config &config, const string &unique_name,
                      bufferContainer &host_buffers, hsaDeviceInterface &device);
        virtual ~hsaInputLostSamples();
        int wait_on_precondition(int gpu_frame_id) override;
        hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq,
                             hsa_signal_t precede_signal) override;
        void finalize_frame(int frame_id) override;
    private:
        uint32_t lost_samples_buffer_id;
        uint32_t lost_samples_buffer_precondition_id;
        uint32_t lost_samples_buffer_finalize_id;
        Buffer * lost_samples_buf;
        uint32_t input_frame_len;
        uint32_t _num_local_freq;
        uint32_t _num_elements;
        uint32_t _samples_per_data_set;
};

#endif
