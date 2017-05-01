#ifndef INPUT_DATA_STAGE_H
#define INPUT_DATA_STAGE_H

#include "gpu_command.h"

class input_data_stage: public gpu_command
{
public:
    input_data_stage(const char* param_name, Config &config);
    ~input_data_stage();
    virtual void build(class device_interface &param_Device) override;
    virtual cl_event execute(int param_bufferID, const uint64_t& fpga_seq, class device_interface& param_Device, cl_event param_PrecedeEvent) override;
    virtual void cleanMe(int param_BufferID) override;
    virtual void freeMe() override;
protected:
    void apply_config(const uint64_t& fpga_seq) override;

    cl_event * data_staged_event;
};

#endif // INITQUEUESEQUENCE_COMMAND_H

