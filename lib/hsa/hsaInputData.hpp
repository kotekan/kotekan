#ifndef HSA_INPUT_DATA_H
#define HSA_INPUT_DATA_H

#include "hsaCommand.hpp"

class hsaInputData: public hsaCommand
{
    public:

        hsaInputData( Config &config, const string &unique_name,
                      bufferContainer &host_buffers, hsaDeviceInterface &device);

        virtual ~hsaInputData();

        int wait_on_precondition(int gpu_frame_id) override;

        hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq,
                             hsa_signal_t precede_signal) override;

        void finalize_frame(int frame_id) override;

    private:
        int32_t network_buffer_id;
        int32_t network_buffer_precondition_id;
        int32_t network_buffer_finalize_id;
        Buffer * network_buf;
        int32_t input_frame_len;

        // TODO maybe factor these into a CHIME command object class?
        int32_t _num_local_freq;
        int32_t _num_elements;
        int32_t _samples_per_data_set;

};

#endif