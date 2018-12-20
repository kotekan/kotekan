#ifndef CL_OUTPUT_DATA_ZERO_H
#define CL_OUTPUT_DATA_ZERO_H

#include "clCommand.hpp"

#include <sys/mman.h>

class clOutputDataZero : public clCommand {
public:
    clOutputDataZero(Config& config, const string& unique_name, bufferContainer& host_buffers,
                     clDeviceInterface& device);
    ~clOutputDataZero();
    cl_event execute(int gpu_frame_id, cl_event pre_event) override;

private:
    int32_t output_len;
    void* output_zeros;

    // Common configuration values (which do not change in a run)
    /// Number of elements on the telescope (e.g. 2048 - CHIME, 256 - Pathfinder).
    int32_t _num_elements;
    /// Number of frequencies per data stream sent to each node.
    int32_t _num_local_freq;
    /// Number of independent integrations within a single dataset. (eg. 8 means
    /// samples_per_data_set/8= amount of integration per dataset.)
    int32_t _num_data_sets;
    /// Calculated value: num_adjusted_elements/block_size * (num_adjusted_elements/block_size +
    /// 1)/2
    int32_t _num_blocks;
    /// This is a kernel tuning parameter for a global work space dimension that sets data sizes for
    /// GPU work items.
    int32_t _block_size;
};

#endif // CL_OUTPUT_DATA_ZERO_H
