/*
 * @file clRfiTimeSum.hpp
 * @brief open cl GPU command to integrate square power across time.
 *  - clRfiTimeSum : public gpu_command
 */
#ifndef CL_RFI_TIME_SUM_HPP
#define CL_RFI_TIME_SUM_HPP

#include "device_interface.h"
#include "gpu_command.h"
#include "restServer.hpp"

#include <mutex>
#include <vector>

/*
 * @class clRfiTimeSum
 * @brief ``gpu_command`` which read the input data and integrates square power values for further
 * processing.
 *
 * This gpu command executes the rfi_chime_timsum_private.cl kernel. The kernel reads input data,
 * computes power and square power values. The kernel integrates those values and outputs a
 * normalized sum of square power values.
 *
 * @requires_kernel    rfi_chime_timesum_private.cl
 *
 * @par REST Endpoints
 * @endpoint    /rfi_time_sum_callback/<gpu_id> ``POST`` Update kernel parameters
 *              requires json values      "bad_inputs"
 *              update config             "bad_inputs"
 *
 * @par GPU Memory
 * @gpu_mem InputBuffer         The kotekan buffer containing input data to be read by the command.
 *      @gpu_mem_type           staging
 *      @gpu_mem_format         Array of @c uint8_t
 * @gpu_mem RfiTimeSumBuffer    A gpu memory object which holds the normalized square power values
 *      @gpu_mem_type           static
 *      @gpu_mem_format         Array of @c float
 * @gpu_mem  InputMask          A mask of faulty inputs of size mask_len
 *     @gpu_mem_type            static
 *     @gpu_mem_format          Array of @c uint8_t
 *     @gpu_mem_metadata        chimeMetadata
 * @gpu_mem  sk_step            The time ingration length (samples)
 *     @gpu_mem_type            static
 *     @gpu_mem_format          Constant @c uint32_t
 *     @gpu_mem_metadata        none
 * @gpu_mem  num_elements       The total number of elements
 *     @gpu_mem_type            static
 *     @gpu_mem_format          Constant @c uint32_t
 *     @gpu_mem_metadata        none
 *
 * @conf   sk_step              Int (default 256). Length of time integration in SK estimate.
 * @conf   bad_inputs           Array of Int The inputs which are currently malfunctioning
 *
 * @author Jacob Taylor
 */
class clRfiTimeSum : public gpu_command {
public:
    // Constructor
    clRfiTimeSum(const char* param_gpuKernel, const char* param_name, kotekan::Config& config,
                 const std::string& unique_name);
    // Destructor
    ~clRfiTimeSum();
    // Builds the program/kernel
    virtual void build(device_interface& param_Device) override;
    // Executes the kernel
    virtual cl_event execute(int param_bufferID, device_interface& param_Device,
                             cl_event param_PrecedeEvent) override;
    // Rest Server Callback
    void rest_callback(kotekan::connectionInstance& conn, json& json_request);

private:
    // RFI parameters
    /// The kurtosis step (How many timesteps per kurtosis estimate)
    uint32_t _sk_step;
    /// A vector holding all of the bad inputs
    vector<int32_t> _bad_inputs;
    /// Mutex for rest server callback
    std::mutex rest_callback_mutex;
    /// A open cl memory object for the input mask
    cl_mem mem_input_mask;
    /// The length of the input mask in bytes
    uint32_t mask_len;
    /// Flag for rebuilding input mask after rest server callback
    bool rebuildInputMask;
    /// The input mask array
    uint8_t* Input_Mask;
    /// String to hold endpoint name
    std::string endpoint;
};

#endif
