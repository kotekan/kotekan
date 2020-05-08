#ifndef CL_INPUT_DATA_H
#define CL_INPUT_DATA_H

#include "clCommand.hpp"
#include "visUtil.hpp"            // for frameID

#include <vector>
#include <tuple>

class clInputData : public clCommand {
public:
    clInputData(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& host_buffers, clDeviceInterface& device);
    ~clInputData();
    int wait_on_precondition(int gpu_frame_id) override;
    cl_event execute(int gpu_frame_id, cl_event pre_event) override;
    void finalize_frame(int frame_id) override;

    std::string get_performance_metric_string();

protected:
    /// Host buffer
    Buffer* in_buf;

    /// Frame ID
    frameID in_buf_id;
    /// Frame ID for precondition
    frameID in_buf_precondition_id;
    /// Frame ID for finilazation
    frameID in_buf_finalize_id;

    /// GPU memory name
    std::string _gpu_memory_name;

    /// Keep track of the OpenCL registered host memory corresponding
    /// to the @c in_buf memory frames.
    std::vector<std::tuple<cl_mem, void*>> opencl_in_buf_frames;
};

#endif // CL_INPUT_DATA_H
