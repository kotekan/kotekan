#ifndef KOTEKAN_CUDA_CORRELATOR_H
#define KOTEKAN_CUDA_CORRELATOR_H

#include <DataType.hpp>
#include <NDArrayRingBuffer.hpp>
#include <cstdint>
#include <cudaCommand.hpp>
#include <cudaDeviceInterface.hpp>
#include <n2k/Correlator.hpp>

/**
 * @class cudaCorrelator
 * @brief cudaCommand for doing an N2 correlation, with Cuda code from Kendrick.
 *
 * @author Andre Renard (by kindly telling Dustin Lang what to type)
 *
 * An example of this stage being used can be found in
 * `config/tests/verify_cuda_n2k.yaml`.
 *
 * A CPU implementation is in `lib/testing/gpuSimulateN2k.hpp`.
 *
 * @par GPU Memory
 * @gpu_mem  gpu_mem_voltage  Input complex voltages of size samples_per_data_set * num_elements *
 * num_local_freq
 *   @gpu_mem_type   staging
 *   @gpu_mem_format Array of @c 4+4-bit complex
 * @gpu_mem  gpu_mem_correlation_triangle  Output complex correlation values of size per frame:
 * (samples_per_data_set / sub_integration_ntimes) * num_freq * num_elements^2 * 2 * sizeof(int32)
 *   @gpu_mem_type   staging
 *   @gpu_mem_format Array of @c int32
 *
 * We assume the input data are stored in a ring buffer, so the signalling buffer must also be
 * specified via the @c in_signal input parameter.
 *
 * @conf   num_elements          Int.  Number of feeds.
 * @conf   num_local_freq        Int.  Number of frequencies.
 * @conf   samples_per_data_set  Int.  Number of time samples per Kotekan block.
 * @conf   sub_integration_ntime Int.  Number of time samples that will be summed into the
 * correlation matrix.
 *
 * Note: While the output is only supposed to fill the upper triangle
 * of the correlation matrices, this implementation fills a few of the
 * below-the-diagonal elements with non-zero values.
 */
class cudaCorrelator : public cudaCommand {
public:
    cudaCorrelator(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device, int inst);
    ~cudaCorrelator();
    int wait_on_precondition() override;
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame() override;

private:
    // Common configuration values (which do not change in a run)
    /// Number of elements on the telescope (aka analog inputs)
    // CHIME            = 2048 (1024 antennas x 2 polarizations),
    // CHORD pathfinder =  128 ( 64  dishes   x 2 polarizations)
    // CHORD pathfinder = 1024 (512  dishes   x 2 polarizations)
    const std::int32_t _num_elements;
    /// Number of frequencies per data stream sent to each node.
    const std::int32_t _num_local_freq;
    /// Total time samples in each dataset. Must be a power of 2.
    const std::int32_t _samples_per_data_set;
    // Number of time samples into each of the output correlation
    // triangles.  The number of output correlation triangles is the
    // length of the input frame divided by this value.
    // Must be a multiple of 256.
    const std::int32_t _sub_integration_ntime;

    /// Name for the voltage input
    const std::string _voltage_name;

    /// Name for the correlation output
    const std::string _n2k_correlation_name;

    // Signalling ring buffer for the input (voltage) data.
    NDArrayRingBuffer<kotekan::int4x2chime_t, 4> voltage;

    // Cuda kernel wrapper object
    n2k::Correlator n2correlator;

    // Placeholder rfi mask
    std::uint32_t* rfimask;
};

#endif // KOTEKAN_CUDA_CORRELATOR_H
