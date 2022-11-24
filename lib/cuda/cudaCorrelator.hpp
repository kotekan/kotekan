#ifndef KOTEKAN_CUDA_CORRELATOR_CUH
#define KOTEKAN_CUDA_CORRELATOR_CUH

#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"

#include "n2k.hpp"

/**
 * @class cudaCorrelator
 * @brief cudaCommand for doing an N2 correlation, with Cuda code from Kendrick.
 *
 * @author Andre Renard (by kindly telling Dustin Lang what to type)
 *
 */
class cudaCorrelator : public cudaCommand {
public:
    cudaCorrelator(kotekan::Config& config, const std::string& unique_name,
                      kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device);
    ~cudaCorrelator();
    cudaEvent_t execute(int gpu_frame_id, cudaEvent_t pre_event) override;

protected:
private:
    // Common configuration values (which do not change in a run)
    /// Number of elements on the telescope (e.g. 2048 - CHIME, 256 - Pathfinder).
    int32_t _num_elements;
    /// Number of frequencies per data stream sent to each node.
    int32_t _num_local_freq;
    /// Total samples in each dataset. Must be a value that is a power of 2.
    int32_t _samples_per_data_set;
    // Number of time samples into each of the output correlation
    // triangles.  The number of output correlation triangles is the
    // length of the input frame divided by this value.
    int32_t _sub_integration_ntime;

    /// GPU side memory name for the voltage input
    std::string _gpu_mem_voltage;
    /// GPU side memory name for correlator output
    std::string _gpu_mem_correlation_triangle;

    // Cuda kernel wrapper object
    n2k::Correlator n2correlator;
  };

#endif // KOTEKAN_CUDA_CORRELATOR_CUH
