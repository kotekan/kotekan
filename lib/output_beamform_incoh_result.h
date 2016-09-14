/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   output_beamform_incoh_result.h
 * Author: ian
 *
 * Created on September 8, 2016, 10:45 AM
 */

#ifndef OUTPUT_BEAMFORM_INCOH_RESULT_H
#define OUTPUT_BEAMFORM_INCOH_RESULT_H

#include "gpu_command.h"
#include "callbackdata.h"

class output_beamform_incoh_result: public gpu_command
{
public:
    output_beamform_incoh_result(char* param_name);
    ~output_beamform_incoh_result();
    virtual void build(Config *param_Config, class device_interface &param_Device);
    virtual cl_event execute(int param_bufferID, class device_interface &param_Device, cl_event param_PrecedeEvent);

protected:

};



#endif /* OUTPUT_BEAMFORM_INCOH_RESULT_H */

