#ifndef CL_OUTPUT_DATA_H
#define CL_OUTPUT_DATA_H

#include "clCommand.hpp"
#include "callbackdata.h"

class clOutputData: public clCommand
{
public:
    clOutputData(Config &config, const string &unique_name,
                      bufferContainer &host_buffers, clDeviceInterface &device);
    ~clOutputData();
    int wait_on_precondition(int gpu_frame_id) override;
    virtual cl_event execute(int buf_frame_id, const uint64_t& fpga_seq, cl_event pre_event) override;
protected:
    int32_t output_buffer_id;
    int32_t output_buffer_precondition_id;
    int32_t output_buffer_finalize_id;
    Buffer * output_buf;

private:
    // Common configuration values (which do not change in a run)
    /// Number of elements on the telescope (e.g. 2048 - CHIME, 256 - Pathfinder).
    int32_t _num_adjusted_elements;
    /// Number of elements on the telescope (e.g. 2048 - CHIME, 256 - Pathfinder).
    int32_t _num_elements;
    /// Number of frequencies per data stream sent to each node.
    int32_t _num_local_freq;
    /// Total samples in each dataset. Must be a value that is a power of 2.
    int32_t _samples_per_data_set;
    /// Number of independent integrations within a single dataset. (eg. 8 means samples_per_data_set/8= amount of integration per dataset.)
    int32_t _num_data_sets;
    /// Number of frequencies per data stream sent to each node.
    int32_t _num_adjusted_local_freq;
    /// Calculated value: num_adjusted_elements/block_size * (num_adjusted_elements/block_size + 1)/2
    int32_t _num_blocks;
    /// This is a kernel tuning parameter for a global work space dimension that sets data sizes for GPU work items.
    int32_t _block_size;
    /// Global buffer depth for all buffers in system. Sets the number of frames to be queued up in each buffer.
    int32_t _buffer_depth;

};

#endif //CL_OUTPUT_DATA_H
