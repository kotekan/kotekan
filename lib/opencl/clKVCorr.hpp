#ifndef CL_KV_CORR_H
#define CL_KV_CORR_H

#include "clCommand.hpp"
#include "clDeviceInterface.hpp"

class clKVCorr : public clCommand {
public:
    clKVCorr(Config& config, const string& unique_name, bufferContainer& host_buffers,
             clDeviceInterface& device);
    ~clKVCorr();
    virtual void build() override;
    virtual cl_event execute(int gpu_frame_id, cl_event pre_event) override;

protected:
    void defineOutputDataMap();
    cl_int* zeros;

    // Host Buffers
    cl_mem id_x_map;
    cl_mem id_y_map;

private:
    // Common configuration values (which do not change in a run)
    /// Number of elements on the telescope (e.g. 2048 - CHIME, 256 - Pathfinder).
    int32_t _num_elements;
    /// Number of frequencies per data stream sent to each node.
    int32_t _num_local_freq;
    /// Total samples in each dataset. Must be a value that is a power of 2.
    int32_t _samples_per_data_set;
    /// Number of independent integrations within a single dataset. (eg. 8 means
    /// samples_per_data_set/8= amount of integration per dataset.)
    int32_t _num_data_sets;
    /// Calculated value: num_adjusted_elements/block_size * (num_adjusted_elements/block_size +
    /// 1)/2
    int32_t _num_blocks;
    /// This is a kernel tuning parameter for a global work space dimension that sets data sizes for
    /// GPU work items.
    int32_t _block_size;
    /// Allow different options for the input data ordering. Current just 4+4b vs dot4b.
    string _data_format;
    /// This will enable use of the AMD-intrinsic-laden overly-complex & optimized kernel.
    bool _full_complicated;
};

#endif // CL_CORRELATOR_KERNEL
