#include "cudaHFB1Accumulate.hpp" // for launch_accumulate_hfb1

#include "Config.hpp"              // for Config
#include "DataType.hpp"            // for float16_t
#include "NDArrayRingBuffer.hpp"   // for NDArrayRingBuffer, extent_t, read_descriptor_t
#include "bufferContainer.hpp"     // for bufferContainer
#include "chordMetadata.hpp"       // for chordMetadata
#include "cudaCommand.hpp"         // for cudaCommand, cudaPipelineState, REGISTER_CUDA_COMMAND
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "cuda_fp16.h"             // for __half
#include "div.hpp"                 // for mod
#include "gpuCommand.hpp"          // for gpuCommandType
#include "kotekanLogging.hpp"      // for DEBUG, FATAL_ERROR

#include <array>          // for array
#include <cstddef>        // for ptrdiff_t
#include <driver_types.h> // for cudaEvent_t
#include <string>         // for string
#include <vector>         // for vector

using kotekan::mod;

/**
 * @class cudaHFB1Accumulate
 * @brief cudaCommand for time-averaging the FRB1 (HFB-1) beams.
 *
 * @author Erik Schnetter
 *
 * Consumes the FRB1 beamformer output and produces the input for the FRB2 stage. Each output frame
 * is the average of `hfb_second_downsampling_factor` consecutive input time samples: the time axis
 * is summed and scaled by `1/hfb_second_downsampling_factor`, collapsing it to a single averaged
 * sample (`num_output_times == 1`). The beam layout (P*Q) and frequency axis are unchanged.
 *
 * @par GPU Memory
 * @gpu_mem Input FRB1 beams
 *   @gpu_mem_buffer       @c ring
 *   @gpu_mem_quantity     @c I
 *   @gpu_mem_type         @c float16
 *   @gpu_mem_dim_name     [@c Ttilde][@c Fbar][@c beamQ][@c beamP]
 * @gpu_mem Output time-averaged beams
 *   @gpu_mem_buffer       @c ring
 *   @gpu_mem_quantity     @c I
 *   @gpu_mem_type         @c float16
 *   @gpu_mem_dim_name     [@c Ttilde][@c Fbar][@c beamQ][@c beamP]
 * @conf  buffer_depth                 Int.  The number of GPU frames used for pipelining.
 * @conf  hfb_second_downsampling_factor  Int.  Number of input time samples to average per output.
 * @conf  num_frequencies              Int.  Number of frequency channels.
 * @conf  frb1_num_beams_P             Int.  Beam dimension P (must satisfy P*Q == 2*256 * 2*4).
 * @conf  frb1_num_beams_Q             Int.  Beam dimension Q.
 * @conf  hfb_downsampling_factor      Int.  FPGA samples per input time sample (Ttilde scaling).
 * @conf  hfb1_beams_name              String.  Base name for the input FRB1 beams buffers.
 * @conf  hfb1_accumulated_beams_name  String.  Base name for the output averaged beams buffers.
 */
class cudaHFB1Accumulate : public cudaCommand {
public:
    cudaHFB1Accumulate(kotekan::Config& config, const std::string& unique_name,
                       kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                       const int instance_num);
    virtual ~cudaHFB1Accumulate();

    int wait_on_precondition() override;
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame() override;

private:
    // The kernel hardcodes the CHIME beam layout P = 2*256, Q = 2*4; the product P*Q must match.
    static constexpr int cuda_num_beams = 2 * 256 * 2 * 4;
    // The kernel always produces a single averaged time sample per frame.
    static constexpr int num_output_times = 1;

    // Parameters
    const int buffer_depth;
    const int hfb_second_downsampling_factor; // input time samples averaged per output sample
    const int num_frequencies;                // frequency channels
    const int frb1_num_beams_P;
    const int frb1_num_beams_Q;
    const int hfb_downsampling_factor; // FPGA samples per input time sample

    // Kotekan buffer names
    const std::string hfb1_beams_name;
    const std::string hfb1_accumulated_beams_name;

    // Buffers. Both keep the Ttilde (time) direction as the slowest dimension; the output keeps it
    // even though only `num_output_times == 1` element is produced per frame.
    NDArrayRingBuffer<float16_t, 4> hfb1_beams;
    NDArrayRingBuffer<float16_t, 4> hfb1_accumulated_beams;
};

REGISTER_CUDA_COMMAND(cudaHFB1Accumulate);

cudaHFB1Accumulate::cudaHFB1Accumulate(kotekan::Config& config, const std::string& unique_name,
                                       kotekan::bufferContainer& host_buffers,
                                       cudaDeviceInterface& device, const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "cudaHFB1Accumulate"),
    // Parameters
    buffer_depth(config.get<int>(unique_name, "buffer_depth")),
    hfb_second_downsampling_factor(config.get<int>(unique_name, "hfb_second_downsampling_factor")),
    num_frequencies(config.get<int>(unique_name, "num_frequencies")),
    frb1_num_beams_P(config.get<int>(unique_name, "frb1_num_beams_P")),
    frb1_num_beams_Q(config.get<int>(unique_name, "frb1_num_beams_Q")),
    hfb_downsampling_factor(config.get<int>(unique_name, "hfb_downsampling_factor")),
    // Buffer names
    hfb1_beams_name(config.get<std::string>(unique_name, "hfb1_beams_name")),
    hfb1_accumulated_beams_name(
        config.get<std::string>(unique_name, "hfb1_accumulated_beams_name")),
    // Buffers. The input holds `hfb_second_downsampling_factor` time samples per frame; the output
    // collapses them to a single averaged sample, so its Ttilde scaling grows by
    // `hfb_second_downsampling_factor`.
    hfb1_beams(hfb1_beams_name, "I",
               std::array<std::ptrdiff_t, 4>{buffer_depth * hfb_second_downsampling_factor
                                                 * num_output_times,
                                             num_frequencies, frb1_num_beams_Q, frb1_num_beams_P},
               std::array<std::string, 4>{"Ttilde", "Fbar", "beamQ", "beamP"},
               {hfb_downsampling_factor, 1, 1, 1}, *this),
    hfb1_accumulated_beams(
        hfb1_accumulated_beams_name, "I",
        std::array<std::ptrdiff_t, 4>{buffer_depth * num_output_times, num_frequencies,
                                      frb1_num_beams_Q, frb1_num_beams_P},
        std::array<std::string, 4>{"Ttilde", "Fbar", "beamQ", "beamP"},
        {hfb_downsampling_factor * hfb_second_downsampling_factor, 1, 1, 1}, *this)
//
{
    if (frb1_num_beams_P * frb1_num_beams_Q != cuda_num_beams)
        FATAL_ERROR(
            "Invalid beam layout: frb1_num_beams_P ({:d}) * frb1_num_beams_Q ({:d}) = {:d}, "
            "but the kernel requires P*Q = {:d}",
            frb1_num_beams_P, frb1_num_beams_Q, frb1_num_beams_P * frb1_num_beams_Q,
            cuda_num_beams);

    hfb1_beams.register_consumer();
    hfb1_accumulated_beams.register_producer();

    set_command_type(gpuCommandType::KERNEL);
}

cudaHFB1Accumulate::~cudaHFB1Accumulate() {}

int cudaHFB1Accumulate::wait_on_precondition() {
    // Average exactly `hfb_second_downsampling_factor` input time samples into one output sample.
    // Claim `hfb_second_downsampling_factor` readable input rows (no overlap), and wait until that
    // many are available.
    DEBUG("Waiting for hfb1_beams input ringbuffer data for frame {:d}...", gpu_frame_id);
    const int in_errcode = hfb1_beams.wait_and_claim_readable([&](const std::ptrdiff_t available) {
        const std::ptrdiff_t claimed =
            available >= hfb_second_downsampling_factor ? hfb_second_downsampling_factor : 0;
        return read_descriptor_t{.claimed = claimed, .read = claimed};
    });
    if (in_errcode < 0)
        return in_errcode;

    const int out_errcode = hfb1_accumulated_beams.wait_for_writable(num_output_times);
    if (out_errcode < 0)
        return out_errcode;

    return 0;
}

cudaEvent_t cudaHFB1Accumulate::execute(cudaPipelineState& /*pipestate*/,
                                        const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();
    record_start_event();

    hfb1_beams.check_metadata();
    hfb1_accumulated_beams.set_metadata(hfb1_beams.get_metadata());

    // Averaging `hfb_second_downsampling_factor` samples collapses the time axis: the output sample
    // spans `hfb_second_downsampling_factor` input samples, so its FPGA time downsampling grows by
    // that factor. (All other metadata is copied by set_metadata above; fpga_seq_num is inherited
    // from the input window start.)
    const std::shared_ptr<const chordMetadata> in_meta = hfb1_beams.get_metadata();
    const std::shared_ptr<chordMetadata> out_meta = hfb1_accumulated_beams.get_metadata();
    out_meta->set_time_downsampling_fpga(in_meta->get_time_downsampling_fpga()
                                         * hfb_second_downsampling_factor);
    hfb1_accumulated_beams.check_metadata();

    // The kernel is not ring-aware: it reads `hfb_second_downsampling_factor` contiguous time rows
    // and writes one output row, both starting at offset 0 of the pointers passed in. Offset the
    // base pointers to the claimed read / write positions. The claimed window cannot wrap the input
    // ring because its size (buffer_depth*hfb_second_downsampling_factor) is a multiple of
    // hfb_second_downsampling_factor and we always claim hfb_second_downsampling_factor rows.
    auto& in_nd = hfb1_beams.get_ndarray();
    auto& out_nd = hfb1_accumulated_beams.get_ndarray();
    const std::ptrdiff_t pos_in = mod(hfb1_beams.get_read_valid().begin(), in_nd.get_extent(0));
    const std::ptrdiff_t pos_out =
        mod(hfb1_accumulated_beams.get_write_valid().begin(), out_nd.get_extent(0));

    // Guard the assumption above: the kernel reads the physical rows
    // [pos_in, pos_in + hfb_second_downsampling_factor) without wrapping, so that window must fit
    // within the ring's slowest dimension.
    if (pos_in + hfb_second_downsampling_factor > in_nd.get_extent(0))
        FATAL_ERROR("hfb1_beams read window would wrap the ring buffer: pos_in ({:d}) + "
                    "hfb_second_downsampling_factor ({:d}) > ring size ({:d}). The kernel is not "
                    "ring-aware; size the ring as a multiple of hfb_second_downsampling_factor.",
                    pos_in, hfb_second_downsampling_factor, in_nd.get_extent(0));

    float16_t* const in_memory = in_nd.data() + pos_in * in_nd.get_stride(0);
    float16_t* const out_memory = out_nd.data() + pos_out * out_nd.get_stride(0);

    launch_accumulate_hfb1(out_memory, in_memory, num_frequencies, hfb_second_downsampling_factor,
                           device.getStream(cuda_stream_id));

    return record_end_event();
}

void cudaHFB1Accumulate::finalize_frame() {
    // Advance the ring buffers
    hfb1_beams.finish_read();
    hfb1_accumulated_beams.finish_write();

    cudaCommand::finalize_frame();
}
