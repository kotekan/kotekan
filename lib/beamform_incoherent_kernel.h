/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   beamform_incoherent_kernel.h
 * Author: ian
 *
 * Created on September 7, 2016, 3:24 PM
 */

#ifndef BEAMFORM_INCOHERENT_KERNEL_H
#define BEAMFORM_INCOHERENT_KERNEL_H

#include "gpu_command.h"
#include "device_interface.h"

class beamform_incoherent_kernel: public gpu_command
{
public:
    beamform_incoherent_kernel(char* param_name);
    beamform_incoherent_kernel(char* param_gpuKernel, char* param_name);
    ~beamform_incoherent_kernel();
    virtual void build(Config* param_Config, class device_interface& param_Device);
    virtual cl_event execute(int param_bufferID, class device_interface &param_Device, cl_event param_PrecedeEvent);
protected:
    cl_mem device_mask;
        
};



#endif /* BEAMFORM_INCOHERENT_KERNEL_H */

