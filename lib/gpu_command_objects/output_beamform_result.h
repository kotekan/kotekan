
#ifndef OUTPUT_BEAMFORM_RESULT_H
#define OUTPUT_BEAMFORM_RESULT_H

#include "gpu_command.h"
#include "callbackdata.h"

class output_beamform_result: public gpu_command
{
public:
    output_beamform_result(const char* param_name, Config &config);
    ~output_beamform_result();
    virtual void build(class device_interface &param_Device) override;
    virtual cl_event execute(int param_bufferID, const uint64_t& fpga_seq, class device_interface &param_Device, cl_event param_PrecedeEvent) override;
protected:
};

#endif

