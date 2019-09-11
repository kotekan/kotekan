#ifndef HSA_ASYNC_COPY_GAIN_H
#define HSA_ASYNC_COPY_GAIN_H

#include "hsaCommand.hpp"

class hsaAsyncCopyGain : public hsaCommand {
public:
    hsaAsyncCopyGain(kotekan::Config& config, const string& unique_name,
                 kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);

    virtual ~hsaAsyncCopyGain();

    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

    void finalize_frame(int frame_id) override;

private:
    struct Buffer* gain_buf;
    int32_t gain_len;
    int32_t gain_buf_precondition_id;
};

#endif
