#ifndef OUTPUT_BEAMFORM_INCOH_RESULT_H
#define OUTPUT_BEAMFORM_INCOH_RESULT_H

#include "clCommand.hpp"
#include "callbackdata.h"

class output_beamform_incoh_result: public clCommand
{
public:
    output_beamform_incoh_result(const char* param_name, Config &config, const string &unique_name);
    ~output_beamform_incoh_result();
    virtual void build(device_interface &param_Device) override;
    virtual cl_event execute(int param_bufferID, class device_interface &param_Device, cl_event param_PrecedeEvent) override;
};

#endif /* OUTPUT_BEAMFORM_INCOH_RESULT_H */

