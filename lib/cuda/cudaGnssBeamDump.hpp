/**
 * @file
 * @brief Coherent beam per selected channel from the voltage ring, written for offline study.
 *  - cudaGnssBeamDump : public cudaCommand
 */

#ifndef CUDA_GNSS_BEAM_DUMP_HPP
#define CUDA_GNSS_BEAM_DUMP_HPP

#include "DataType.hpp"          // for int4x2_swapped_withoffset_t
#include "NDArrayBuffer.hpp"     // for NDArrayBuffer
#include "NDArrayRingBuffer.hpp" // for NDArrayRingBuffer
#include "cudaCommand.hpp"
#include "cudaGnssBeamDumpKernel.hpp" // for BeamDumpLayout

#include <string>
#include <vector>

/**
 * @class cudaGnssBeamDump
 * @brief A self-calibrated beam toward the dominant source, per selected channel, per frame.
 *
 * For each selected channel the command forms the covariance of the 128 inputs over the frame,
 * whitens each input by its own power, takes the top eigenvector (power iterations warm-started
 * from the previous frame), references its phase to one input and sums the inputs with those
 * weights, so the beam reads in the reference input's units. The frame it writes carries the
 * weights, the per-input powers and the two largest eigenvalues beside the samples
 * (cudaGnssBeamDumpKernel.hpp). It is a consumer of the voltage ring like cudaCorrelatorDual,
 * in the same process and on the same stream.
 *
 * The output is an NDArrayBuffer named `beam_name`, on the device as `<beam_name>_buffer`, for
 * cudaOutputData to copy out; its metadata is the ring's with fpga_seq_num set to the frame's
 * first hop. The frame's clock for FrameWindowGate is the int64 seq0 at byte 0.
 *
 * @conf As the ring consumers (buffer_depth, num_times, num_elements, num_local_freq,
 *       voltage_name), and:
 * @conf  beam_name             String. Output NDArrayBuffer name, e.g. gnss0_beam.
 * @conf  beam_local_channels   List of Int. Local frame channel of each selected channel.
 * @conf  beam_freq_ids         List of Int. Their global freq_ids, same order; checked against
 *                              the ring metadata's coarse frequencies on the first frame.
 * @conf  beam_ref_input        Int. The phase-reference input (a live, unflagged input).
 * @conf  beam_eig_iters        Int, default 4. Power iterations per frame per component.
 */
class cudaGnssBeamDump : public cudaCommand {
public:
    cudaGnssBeamDump(kotekan::Config& config, const std::string& unique_name,
                     kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                     int instance_num);
    ~cudaGnssBeamDump();

    int wait_on_precondition() override;
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame() override;

private:
    const int _buffer_depth;
    const int _num_times;
    const int _num_elements;
    const int _num_local_freq;
    const std::string _voltage_name;
    const std::string _beam_name;
    const std::vector<std::int32_t> _local_channels;
    const std::vector<std::int32_t> _freq_ids;
    const int _ref_input;
    const int _eig_iters;
    const gnss_cuda::BeamDumpLayout _layout;

    NDArrayRingBuffer<kotekan::int4x2_swapped_withoffset_t, 4> voltage;
    NDArrayBuffer<unsigned char, 2> beam;

    std::vector<unsigned char> _hdr_stage; // header + freq_ids, per frame
    bool _uploaded_static = false;
    bool _checked_freqs = false;
    long _frames = 0;
};

#endif
