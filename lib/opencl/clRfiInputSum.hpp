/*
 * @file clRfiInputSum.hpp
 * @brief open cl GPU command to integrate square power across time.
 *  - clRfiInputSum : public gpu_command
 */
#ifndef CL_RFI_INPUT_SUM_HPP
#define CL_RFI_INPUT_SUM_HPP

#include "device_interface.h"
#include "gpu_command.h"
#include "restServer.hpp"

#include <mutex>

/*
 * @class clRfiInputSum
 * @brief ``gpu_command`` which integrates the output of clRfiTimeSum across inputs and computs
 * kurtosis estimates.
 *
 * This gpu command executes the rfi_chime_input_sum.cl or rfi_chime_input_sum_private.cl depending
 * on which is quickest. The kernel reads the output of clRfiTimeSum and integrates it across
 * inputs. Then a kurtosis estimate is computed and placed in the output buffer.
 *
 * @requires_kernel    rfi_chime_input_sum_private.cl
 *
 * @par REST Endpoints
 * @endpoint    /rfi_input_sum_callback/<gpu_id> ``POST`` Update kernel parameters
 *              requires json values      "num_bad_inputs"
 *              update config             "num_bad_inputs"
 *
 * @par GPU Memory
 * @gpu_mem RfiTimeSumBuffer    A gpu memory object which holds the normalized squ$
 *      @gpu_mem_type           static
 *      @gpu_mem_format         Array of @c float
 * @gpu_mem RfiOutputBuffer     The kotekan buffer containing output spectral kurtosis estimates
 *      @gpu_mem_type           staging
 *      @gpu_mem_format         Array of @c float
 * @gpu_mem  num_elements       The total number of elements
 *     @gpu_mem_type            static
 *     @gpu_mem_format          Constant @c uint32_t
 *     @gpu_mem_metadata        none
 * @gpu_mem  M                  The total SK integration length
 *     @gpu_mem_type            static
 *     @gpu_mem_format          Constant @c uint32_t
 *     @gpu_mem_metadata        none
 *
 * @conf   sk_step              Int (default 256). Length of time integration in SK estimate.
 * @conf   bad_inputs           Array of Ints. Used to compute the number of faulty inputs
 *
 * @author Jacob Taylor
 */
class clRfiInputSum : public gpu_command {
public:
    // Constructor
    clRfiInputSum(const char* param_gpuKernel, const char* param_name, kotekan::Config& config,
                  const std::string& unique_name);
    // Destructor
    ~clRfiInputSum();
    // Builds the program/kernel
    virtual void build(device_interface& param_Device) override;
    // Executes the kernel
    virtual cl_event execute(int param_bufferID, device_interface& param_Device,
                             cl_event param_PrecedeEvent) override;
    // Rest Server Callback
    void rest_callback(kotekan::connectionInstance& conn, json& json_request);

private:
    /// Integration length of spectral kurtosis estimate in time
    uint32_t _sk_step;
    /// The total number of faulty inputs
    uint32_t _num_bad_inputs;
    /// The total integration length of the spectral kurtosis estimate
    uint32_t _M;
    /// Flag indicating whether or not the private or local input sum kernel is being used
    bool _use_local_sum;
    /// Mutex for rest server callback
    std::mutex rest_callback_mutex;
    /// String to hold endpoint name
    std::string endpoint;
};

#endif
