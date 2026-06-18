#ifndef KOTEKAN_CUDA_CORRELATOR_H
#define KOTEKAN_CUDA_CORRELATOR_H

#include "Config.hpp"              // for Config
#include "DataType.hpp"            // for int4x2_swapped_withoffset_t
#include "NDArrayBuffer.hpp"       // for NDArrayBuffer
#include "NDArrayRingBuffer.hpp"   // for NDArrayRingBuffer
#include "bufferContainer.hpp"     // for bufferContainer
#include "cudaCommand.hpp"         // for cudaCommand, cudaPipelineState
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "driver_types.h"          // for cudaEvent_t
#include "n2k/Correlator.hpp"      // for Correlator

#include <cstdint> // for int32_t, uint32_t
#include <string>  // for string, basic_string
#include <vector>  // for vector

/**
 * @class cudaCorrelator
 * @brief cudaCommand for doing an N2 correlation, with CUDA code from Kendrick.
 *
 * @author Andre Renard (by kindly telling Dustin Lang what to type)
 *
 * An example of this stage being used can be found in `config/tests/verify_cuda_n2k.yaml`.
 *
 * A CPU implementation is in `lib/testing/gpuSimulateN2kCorr.hpp`.
 *
 * @par GPU Memory
 * @gpu_mem Input voltage
 *   @gpu_mem_buffer    @c ring
 *   @gpu_mem_quantity  @c E
 *   @gpu_mem_type      @c int4x2_swapped_withoffset
 *   @gpu_mem_dim_name  [@c T][@c F][@c P][@c D]
 *   @gpu_mem_shape     [@c buffer_depth * num_times][@c num_local_freq][@c 2][@c num_elements / 2]
 *   @gpu_mem_metadata  @c chordMetadata
 * @gpu_mem Input RFI mask ringbuffer
 *   @gpu_mem_buffer    @c ring
 *   @gpu_mem_quantity  @c RFImask
 *   @gpu_mem_type      @c uint1x8
 *   @gpu_mem_dim_name  [@c T8hi128][@c F][@c T8lo128]
 *   @gpu_mem_shape     [@c buffer_depth * num_times / 8 / 128][@c num_local_freq][@c 2][@c 128]
 *   @gpu_mem_metadata  @c chordMetadata
 * @gpu_mem Output complex correlation values
 *   num_subintegrations := samples_per_data_set / sub_integration_ntime
 *   blocksize           := 16
 *   linear_num_blocks   := ceil(num_elements / blocksize)
 *   triangle_num_blocks := linear_num_blocks * (linear_num_blocks + 1) / 2
 *   @gpu_mem_buffer    @c standard
 *   @gpu_mem_quantity  @c n2k_correlation
 *   @gpu_mem_type      @c int32
 *   @gpu_mem_dim_name  [@c Tc][@c F][@c DPhi][@c DPlo1][@c DPlo2][@c C]
 *   @gpu_mem_shape     [@c samples_per_data_set / sub_integration_ntimes]
 *                      [@c triangle_num_blocks][@c blocksize][@c blocksize][@c 2]
 *   @gpu_mem_metadata  @c chordMetadata
 * @conf  buffer_depth           Int.     The number of GPU frames used for pipelining commands.
 * @conf  num_times              Int.     Number of time samples per frame.
 * @conf  num_elements           Int.     Number of dish times number of polarizations (2).
 * @conf  num_local_freq         Int.     Number of frequencies handled by this X-Engine node.
 * @conf  sub_integration_ntime  Int.     Number of time samples that will be summed into the
 *                                        correlation matrix.
 * @conf  voltage_name           String.  Base name for the voltage buffers.
 * @conf  rfi_RFImask_name       String.  Base name for the RFI mask buffers.
 * @conf  n2k_correlation_name   String.  Base name for the N2 correlation buffers.
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
    const std::int32_t _buffer_depth;
    const std::int32_t _num_times;
    /// Number of elements on the telescope (aka analog inputs)
    // CHIME            = 2048 (1024 antennas x 2 polarizations),
    // CHORD pathfinder =  128 ( 64  dishes   x 2 polarizations)
    // CHORD pathfinder = 1024 (512  dishes   x 2 polarizations)
    const std::int32_t _num_elements;
    /// Number of frequencies per data stream sent to each node.
    const std::int32_t _num_local_freq;
    // Number of time samples into each of the output correlation
    // triangles.  The number of output correlation triangles is the
    // length of the input frame divided by this value.
    // Must be a multiple of 256.
    const std::int32_t _sub_integration_ntime;

    /// Name for the voltage input
    const std::string _voltage_name;

    /// Name for the rfi mask input
    const std::string _rfi_RFImask_name;

    /// Name for the correlation output
    const std::string _n2k_correlation_name;

    /// Whether to poison the output buffers
    const bool _poison_buffers;

    /// Signaling ring buffer for the input (voltage) data.
    NDArrayRingBuffer<kotekan::int4x2_swapped_withoffset_t, 4> voltage;
    NDArrayRingBuffer<kotekan::uint1x8_t, 3> rfi_RFImask;
    NDArrayBuffer<std::int32_t, 6> n2k_correlation;

    /// Cuda kernel wrapper object
    n2k::Correlator n2correlator;
};

#endif // KOTEKAN_CUDA_CORRELATOR_H
