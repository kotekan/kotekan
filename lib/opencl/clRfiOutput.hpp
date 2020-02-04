/*
 * @file clRfiOutput.hpp
 * @brief open cl GPU command to read output from RFI kernels.
 *  - clRfiOutput : public gpu_command
 */
#ifndef CL_RFI_OUTPUT_HPP
#define CL_RFI_OUTPUT_HPP

#include "callbackdata.h"
#include "gpu_command.h"

/*
 * @class clRfiOutput
 * @brief ``gpu_command`` which read the results of clRfiInputSum and places in buffer
 *
 * This gpu command read the results of the clRfiInputSum gpu command and places them in a kotekan
 * buffer for further processing.
 *
 * @par GPU Memory
 * @gpu_mem RfiOutputBuffer The kotekan buffer containing output spectral kurtosis estimates
 *      @gpu_mem_type              staging
 *      @gpu_mem_format            Array of @c float
 *
 * @author Jacob Taylor
 */
class clRfiOutput : public gpu_command {
public:
    // Constructor
    clRfiOutput(const char* param_name, kotekan::Config& config, const std::string& unique_name);
    // Destructor
    ~clRfiOutput();
    // Builds gpu command
    virtual void build(device_interface& param_Device) override;
    // Reads gpu memory and places in kotekan buffer
    virtual cl_event execute(int param_bufferID, device_interface& param_Device,
                             cl_event param_PrecedeEvent) override;

protected:
};

#endif
