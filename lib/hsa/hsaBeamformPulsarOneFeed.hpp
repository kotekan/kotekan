#ifndef HSA_BEAMFORM_PULSAR_ONE_FEED_H
#define HSA_BEAMFORM_PULSAR_ONE_FEED_H

#include "hsaCommand.hpp"

class hsaBeamformPulsarOneFeed: public hsaCommand
{
public:
    hsaBeamformPulsarOneFeed(const string &kernel_name, const string &kernel_file_name,
                        hsaDeviceInterface &device, Config &config,
                        bufferContainer &host_buffers,
                        const string &unique_name);

    virtual ~hsaBeamformPulsarOneFeed();

    void apply_config(const uint64_t& fpga_seq) override;

    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq,
                         hsa_signal_t precede_signal) override;

private:
    int32_t input_frame_len;
    int32_t output_frame_len;
    int32_t phase_len;

    float * host_phase;

    int32_t _num_elements;
    int32_t _num_pulsar;
    int32_t _num_pol;
    int32_t _one_feed_p0;
    int32_t _one_feed_p1;
    int32_t _samples_per_data_set;
};

#endif
