#include "dummy_placeholder_kernel.h"

dummy_placeholder_kernel::dummy_placeholder_kernel(char* param_name):gpu_command(param_name){}

dummy_placeholder_kernel::~dummy_placeholder_kernel(){}

void dummy_placeholder_kernel::build(Config* param_Config, class device_interface& param_Device){}

cl_event dummy_placeholder_kernel::execute(int param_bufferID, class device_interface& param_Device, cl_event param_PrecedeEvent)
{return param_PrecedeEvent;}
