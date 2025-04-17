#include <DataType.hpp>
#include <NDArrayBuffer.hpp>
#include <cassert>
#include <chordMetadata.hpp>
#include <cstddef>
#include <cstring>
#include <cudaCommand.hpp>
#include <div.hpp>
#include <memory>
#include <metadata.hpp>
#include <n2k.hpp>
#include <optional>
#include <ringbuffer.hpp>
#include <string>
#include <tuple>
#include <vector>

class S012Kernel : public cudaCommand {
public:
    S012Kernel(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
               const int instance_num);
    virtual ~S012Kernel();

    int wait_on_precondition() override;
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame() override;

private:
    // Kernel input and output sizes
    std::int64_t num_consumed_elements(std::int64_t num_available_elements) const;
    std::int64_t num_produced_elements(std::int64_t num_available_elements) const;

    std::int64_t num_processed_elements(std::int64_t num_available_elements) const;

    static constexpr const char* bf_mask_name = "bf_mask";
    static constexpr kotekan::DataType bf_mask_type = kotekan::int8;

    static constexpr const char* pl_mask_name = "pl_mask";
    static constexpr kotekan::DataType pl_mask_type = kotekan::uint1x8;

    static constexpr const char* E_name = "E";
    static constexpr kotekan::DataType E_type = kotekan::int4x2chime;

    static constexpr const char* S012_name = "S012";
    static constexpr kotekan::DataType S012_type = kotekan::uint64;

    static constexpr const char* S012tilde_name = "S012tilde";
    static constexpr kotekan::DataType S012tilde_type = kotekan::uint64;

    // Kotekan buffer names
    const std::string bf_mask_memname;
    const std::string pl_mask_memname;
    const std::string E_memname;
    const std::string S012_memname;
    const std::string S012tilde_memname;

    RingBuffer* const pl_mask_input_ringbuf_signal;
    RingBuffer* const voltage_input_ringbuf_signal;
    RingBuffer* const rfi_S012_output_ringbuf_signal;
#warning "IDEA: write rfi_S012tilde into a regular buffer"
    RingBuffer* const rfi_S012tilde_output_ringbuf_signal;

    // Parameters
    const int max_num_times;
    const int num_frequencies;
    const int num_polarizations;
    const int num_dishes;

    const int rfi_downsampling_factor;

    const n2k::SkKernel skKernel;

    // Buffer lengths
    const std::ptrdiff_t bf_mask_length =
        type_total_bytes(bf_mask_type) * num_dishes * num_polarizations;

    // Ringbuffer strides
    const std::ptrdiff_t pl_mask_T128_sample_bytes;
    const std::ptrdiff_t E_T_sample_bytes;
    const std::ptrdiff_t S012_Tcoarse_sample_bytes;
    const std::ptrdiff_t S012tilde_Tcoarse_sample_bytes;

    // How many samples we will process from the input ringbuffer
    // (Set in `wait_for_precondition`, invalid after `finalize_frame`)
    std::ptrdiff_t Tmin, Tmax;

    // How many samples we will produce in the output ringbuffer
    // (Set in `wait_for_precondition`, invalid after `finalize_frame`)
    std::ptrdiff_t Tcoarsemin, Tcoarsemax;
};

REGISTER_CUDA_COMMAND(S012Kernel);

S012Kernel::S012Kernel(kotekan::Config& config, const std::string& unique_name,
                       kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                       const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "S012Kernel"),
    bf_mask_memname(config.get<std::string>(unique_name, "bf_mask")),
    pl_mask_memname(config.get<std::string>(unique_name, "pl_mask")),
    E_memname(config.get<std::string>(unique_name, "gpu_mem_voltage")),
    S012_memname(config.get<std::string>(unique_name, "accumulant")),
    S012tilde_memname(config.get<std::string>(unique_name, "averaged_accumulant")),
    // Find input and output buffers used for signalling ring-buffer state
    pl_mask_input_ringbuf_signal(dynamic_cast<RingBuffer*>(host_buffers.get_generic_buffer(
        config.get<std::string>(unique_name, "pl_mask_in_signal")))),
    voltage_input_ringbuf_signal(dynamic_cast<RingBuffer*>(host_buffers.get_generic_buffer(
        config.get<std::string>(unique_name, "voltage_in_signal")))),
    rfi_S012_output_ringbuf_signal(dynamic_cast<RingBuffer*>(host_buffers.get_generic_buffer(
        config.get<std::string>(unique_name, "rfi_S012_out_signal")))),
    rfi_S012tilde_output_ringbuf_signal(dynamic_cast<RingBuffer*>(host_buffers.get_generic_buffer(
        config.get<std::string>(unique_name, "rfi_S012tilde_out_signal")))),
    max_num_times(config.get<int>(unique_name, "max_num_times")),
    num_frequencies(config.get<int>(unique_name, "num_frequencies")),
    num_polarizations(config.get<int>(unique_name, "num_polarizations")),
    num_dishes(config.get<int>(unique_name, "num_dishes")),
    rfi_downsampling_factor(config.get<int>(unique_name, "rfi_downsampling_factor")),
    skKernel(n2k::SkKernel::Params{
        config.get<double>(unique_name, "rfi_sk_rfimask_sigmas"),
        config.get<double>(unique_name, "rfi_single_feed_min_good_frac"),
        config.get<double>(unique_name, "rfi_feed_averaged_min_good_frac"),
        config.get<double>(unique_name, "rfi_mu_min"),
        config.get<double>(unique_name, "rfi_mu_max"),
        rfi_downsampling_factor,
    }),
    pl_mask_T128_sample_bytes(type_total_bytes(pl_mask_type) * 8 * num_dishes * num_polarizations
                              * num_frequencies / 4),
    E_T_sample_bytes(type_total_bytes(E_type) * num_dishes * num_polarizations * num_frequencies),
    S012_Tcoarse_sample_bytes(type_total_bytes(S012_type) * num_dishes * num_polarizations * 3
                              * num_frequencies),
    S012tilde_Tcoarse_sample_bytes(type_total_bytes(S012tilde_type) * 3 * num_frequencies) {
    // Ensure max_num_times is a power of two
    if (max_num_times <= 0 || ((max_num_times & (max_num_times - 1)) != 0))
        FATAL_ERROR("max_num_times is not a power of 2");

    // For pl_mask_T128_sample_bytes
    if (num_frequencies % 4 != 0)
        FATAL_ERROR("num_frequencies % 4 != 0");
    // if (type_total_bytes(pl_mask_type) * 8 * num_dishes * num_polarizations *
    // num_frequencies
    //         / 4 % 128
    //     != 0)
    //     FATAL_ERROR("type_total_bytes(pl_mask_type) * 8 * num_dishes * num_polarizations
    //     * "
    //                 "num_frequencies / 4 % 128 != 0");

    set_command_type(gpuCommandType::KERNEL);

    if (instance_num == 0) {
        pl_mask_input_ringbuf_signal->register_consumer(unique_name);
        voltage_input_ringbuf_signal->register_consumer(unique_name);
        rfi_S012_output_ringbuf_signal->register_producer(unique_name);
        rfi_S012_output_ringbuf_signal->allocate_new_metadata_object(0);
        rfi_S012tilde_output_ringbuf_signal->register_producer(unique_name);
        rfi_S012tilde_output_ringbuf_signal->allocate_new_metadata_object(0);
    }

    // Add Graphviz entries for the GPU buffers used by this kernel
    gpu_buffers_used.push_back(std::make_tuple(
        bf_mask_memname, true, true, false) /* (name, is_array, does_read, does_write) */);
    gpu_buffers_used.push_back(std::make_tuple(
        pl_mask_memname, true, true, false) /* (name, is_array, does_read, does_write) */);
    gpu_buffers_used.push_back(std::make_tuple(
        E_memname, true, true, false) /* (name, is_array, does_read, does_write) */);
    gpu_buffers_used.push_back(std::make_tuple(S012_memname, true, false,
                                                true) /* (name, is_array, does_read, does_write)
                                                */);
    gpu_buffers_used.push_back(std::make_tuple(S012tilde_memname, true, false,
                                                true) /* (name, is_array, does_read, does_write)
                                                */);
}

S012Kernel::~S012Kernel() {}

std::int64_t S012Kernel::num_consumed_elements(std::int64_t num_available_elements) const {
    return num_produced_elements(num_available_elements) * rfi_downsampling_factor;
}
std::int64_t S012Kernel::num_produced_elements(std::int64_t num_available_elements) const {
    return num_processed_elements(num_available_elements) / rfi_downsampling_factor;
}

std::int64_t S012Kernel::num_processed_elements(std::int64_t num_available_elements) const {
    using std::max; // `max` is the pauper's `lcm`
    const int granularity = max(128, rfi_downsampling_factor);
    assert(granularity % 128 == 0);
    assert(granularity % rfi_downsampling_factor == 0);
    if (num_available_elements < granularity)
        return 0;
    assert(num_available_elements >= granularity);
    return kotekan::round_down(num_available_elements, granularity);
}

int S012Kernel::wait_on_precondition() {
    // Wait for data to be available in input ringbuffers

    // Check available pl_mask samples
    DEBUG("Checking available pl_mask input ringbuffer data for frame {:d}...", gpu_frame_id);
    const std::optional<std::pair<std::ptrdiff_t, std::ptrdiff_t>> pl_mask_peeked =
        pl_mask_input_ringbuf_signal->peek_readable(unique_name, instance_num);
    if (!pl_mask_peeked.has_value())
        return -1;
    std::ptrdiff_t pl_mask_input_bytes = pl_mask_peeked.value().second;
    DEBUG("pl_mask input ring-buffer byte count: {:d}", pl_mask_input_bytes);

    // Check available voltage samples
    DEBUG("Checking available voltage input ringbuffer data for frame {:d}...", gpu_frame_id);
    const std::optional<std::pair<std::ptrdiff_t, std::ptrdiff_t>> voltage_peeked =
        voltage_input_ringbuf_signal->peek_readable(unique_name, instance_num);
    if (!voltage_peeked.has_value())
        return -1;
    std::ptrdiff_t voltage_input_bytes = voltage_peeked.value().second;
    DEBUG("voltage input ring-buffer byte count: {:d}", voltage_input_bytes);

wait_for_data:

    // How many inputs samples are available?
    const std::ptrdiff_t T_available_pl_mask =
        128 * kotekan::div_noremainder(pl_mask_input_bytes, pl_mask_T128_sample_bytes);
    DEBUG("Available pl_mask samples:      T_available_pl_mask: {:d}", T_available_pl_mask);
    const std::ptrdiff_t T_available_voltage =
        kotekan::div_noremainder(voltage_input_bytes, E_T_sample_bytes);
    DEBUG("Available voltage samples:      T_available_voltage: {:d}", T_available_voltage);
    using std::min;
    const std::ptrdiff_t T_available = min(T_available_pl_mask, T_available_voltage);
    DEBUG("Available samples:      T_available: {:d}", T_available);

    // How many inputs will we process and consume?
    const std::ptrdiff_t T_processed = num_processed_elements(T_available);
    const std::ptrdiff_t T_consumed = num_consumed_elements(T_available);
    DEBUG("Will process (samples): T_processed: {:d}", T_processed);
    DEBUG("Will consume (samples): T_consumed:  {:d}", T_consumed);
    assert(T_consumed <= T_processed);
    const std::ptrdiff_t T_consumed2 = num_consumed_elements(T_processed);
    assert(T_consumed2 == T_consumed);

    // Can we make progress?
    if (T_consumed == 0) {
        // We cannot make progress, we need to wait
        DEBUG("We cannot make progress, we need to wait for more input");

        // Wait for pl_mask data if that limits our progress
        if (T_available == T_available_pl_mask) {
            DEBUG("Waiting for pl_mask input ringbuffer data for frame {:d}...", gpu_frame_id);
            const std::optional<std::ptrdiff_t> pl_mask_waited =
                pl_mask_input_ringbuf_signal->wait_without_claiming(unique_name, instance_num,
                                                                    pl_mask_input_bytes + 1);
            DEBUG("Finished waiting for pl_mask input for data frame {:d}.", gpu_frame_id);
            if (!pl_mask_waited.has_value())
                return -1;
            pl_mask_input_bytes = pl_mask_waited.value();
            DEBUG("pl_mask input ring-buffer byte count: {:d}", pl_mask_input_bytes);
        }

        // Wait for voltage data if that limits our progress
        if (T_available == T_available_voltage) {
            DEBUG("Waiting for voltage input ringbuffer data for frame {:d}...", gpu_frame_id);
            const std::optional<std::ptrdiff_t> voltage_waited =
                voltage_input_ringbuf_signal->wait_without_claiming(unique_name, instance_num,
                                                                    voltage_input_bytes + 1);
            DEBUG("Finished waiting for voltage input for data frame {:d}.", gpu_frame_id);
            if (!voltage_waited.has_value())
                return -1;
            voltage_input_bytes = voltage_waited.value();
            DEBUG("voltage input ring-buffer byte count: {:d}", voltage_input_bytes);
        }

        goto wait_for_data;
    }

    // Claim inputs
    assert(T_consumed > 0);
    assert(T_consumed % 128 == 0);
    const std::optional<std::ptrdiff_t> pl_mask_claimed =
        pl_mask_input_ringbuf_signal->wait_and_claim_readable(
            unique_name, instance_num, T_consumed / 128 * pl_mask_T128_sample_bytes);
    if (!pl_mask_claimed.has_value())
        return -1;
    const std::ptrdiff_t pl_mask_input_cursor = pl_mask_claimed.value();
    const std::optional<std::ptrdiff_t> voltage_claimed =
        voltage_input_ringbuf_signal->wait_and_claim_readable(unique_name, instance_num,
                                                              T_consumed * E_T_sample_bytes);
    if (!voltage_claimed.has_value())
        return -1;
    const std::ptrdiff_t voltage_input_cursor = voltage_claimed.value();

    DEBUG("pl_mask input ring-buffer byte offset: {:d}", pl_mask_input_cursor);
    DEBUG("voltage input ring-buffer byte offset: {:d}", voltage_input_cursor);
    Tmin = kotekan::div_noremainder(voltage_input_cursor, E_T_sample_bytes);
    assert(128 * kotekan::div_noremainder(pl_mask_input_cursor, pl_mask_T128_sample_bytes) == Tmin);
    Tmax = Tmin + T_processed;
    const std::ptrdiff_t Tlength = Tmax - Tmin;
    DEBUG("Input samples:");
    DEBUG("    Tmin:    {:d}", Tmin);
    DEBUG("    Tmax:    {:d}", Tmax);
    DEBUG("    Tlength: {:d}", Tlength);

    // How many outputs will we produce?
    const std::ptrdiff_t Tcoarse_produced = num_produced_elements(T_available);
    DEBUG("Will produce (samples): Tcoarse_produced: {:d}", Tcoarse_produced);
    const std::ptrdiff_t Tcoarselength = Tcoarse_produced;

    // to bytes
    const std::ptrdiff_t S012_output_bytes = Tcoarselength * S012_Tcoarse_sample_bytes;
    const std::ptrdiff_t S012tilde_output_bytes = Tcoarselength * S012tilde_Tcoarse_sample_bytes;
    DEBUG("Will produce {:d} S012 output bytes", S012_output_bytes);
    DEBUG("Will produce {:d} S012tilde output bytes", S012tilde_output_bytes);

    // Wait for space to be available in our output ringbuffers...
    DEBUG("Waiting for S012 output ringbuffer space for frame {:d}...", gpu_frame_id);
    const std::optional<std::ptrdiff_t> val_S012_out1 =
        rfi_S012_output_ringbuf_signal->wait_for_writable(unique_name, instance_num,
                                                          S012_output_bytes);
    DEBUG("Finished waiting for S012 output for data frame {:d}.", gpu_frame_id);
    if (!val_S012_out1.has_value())
        return -1;
    const std::ptrdiff_t S012_output_cursor = val_S012_out1.value();
    DEBUG("S012 output ring-buffer byte offset {:d}", S012_output_cursor);

    DEBUG("Waiting for S012tilde output ringbuffer space for frame {:d}...", gpu_frame_id);
    const std::optional<std::ptrdiff_t> val_S012tilde_out2 =
        rfi_S012tilde_output_ringbuf_signal->wait_for_writable(unique_name, instance_num,
                                                               S012tilde_output_bytes);
    DEBUG("Finished waiting for S012tilde output for data frame {:d}.", gpu_frame_id);
    if (!val_S012tilde_out2.has_value())
        return -1;
    const std::ptrdiff_t S012tilde_output_cursor = val_S012tilde_out2.value();
    DEBUG("S012 output ring-buffer byte offset {:d}", S012tilde_output_cursor);

    assert(kotekan::mod(S012_output_cursor, S012_Tcoarse_sample_bytes) == 0);
    assert(kotekan::mod(S012tilde_output_cursor, S012tilde_Tcoarse_sample_bytes) == 0);
    Tcoarsemin = S012_output_cursor / S012_Tcoarse_sample_bytes;
    assert(Tcoarsemin == S012tilde_output_cursor / S012tilde_Tcoarse_sample_bytes);
    Tcoarsemax = Tcoarsemin + Tcoarselength;
    DEBUG("Output samples:");
    DEBUG("    Tcoarsemin:    {:d}", Tcoarsemin);
    DEBUG("    Tcoarsemax:    {:d}", Tcoarsemax);
    DEBUG("    Tcoarselength: {:d}", Tcoarselength);

    return 0;
}

cudaEvent_t S012Kernel::execute(cudaPipelineState& /*pipestate*/,
                                const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();

    record_start_event();
    DEBUG("gpu_frame_id: {}", gpu_frame_id);

    // Check bf_mask metadata
    const std::shared_ptr<metadataObject> bf_mask_mc =
        device.get_gpu_memory_array_metadata(bf_mask_memname, gpu_frame_id);
    assert(bf_mask_mc);
    assert(metadata_is_chord(bf_mask_mc));
    const std::shared_ptr<chordMetadata> bf_mask_meta = get_chord_metadata(bf_mask_mc);
    DEBUG("input bf_mask array: {:s} {:s}", bf_mask_meta->get_type_string(),
          bf_mask_meta->get_dimensions_string());
    assert(std::strncmp(bf_mask_meta->name, bf_mask_name, sizeof bf_mask_meta->name) == 0);
    assert(bf_mask_meta->type == kotekan::int8);
    assert(bf_mask_meta->dims == 2);
    assert(std::strncmp(bf_mask_meta->dim_name[0], "P", sizeof bf_mask_meta->dim_name[0]) == 0);
    assert(std::strncmp(bf_mask_meta->dim_name[1], "D", sizeof bf_mask_meta->dim_name[1]) == 0);
    assert(bf_mask_meta->dim[0] == num_polarizations);
    assert(bf_mask_meta->dim[1] == num_dishes);
    for (int d = bf_mask_meta->dims - 1; d >= 0; --d)
        if (d == bf_mask_meta->dims - 1)
            assert(bf_mask_meta->stride[d] == 1);
        else
            assert(bf_mask_meta->stride[d]
                   == bf_mask_meta->stride[d + 1] * bf_mask_meta->dim[d + 1]);

    // Check pl_mask metadata
    const std::shared_ptr<metadataObject> pl_mask_mc =
        pl_mask_input_ringbuf_signal->get_metadata(0);
    assert(pl_mask_mc);
    assert(metadata_is_chord(pl_mask_mc));
    const std::shared_ptr<chordMetadata> pl_mask_meta = get_chord_metadata(pl_mask_mc);
    DEBUG("input pl_mask array: {:s} {:s}", pl_mask_meta->get_type_string(),
          pl_mask_meta->get_dimensions_string());
    assert(std::strncmp(pl_mask_meta->name, pl_mask_name, sizeof pl_mask_meta->name) == 0);
    assert(pl_mask_meta->type == kotekan::uint1x8);
    assert(pl_mask_meta->dims == 5);
    assert(std::strncmp(pl_mask_meta->dim_name[0], "T16hi8", sizeof pl_mask_meta->dim_name[0])
           == 0);
    assert(std::strncmp(pl_mask_meta->dim_name[1], "F4", sizeof pl_mask_meta->dim_name[1]) == 0);
    assert(std::strncmp(pl_mask_meta->dim_name[2], "P", sizeof pl_mask_meta->dim_name[2]) == 0);
    assert(std::strncmp(pl_mask_meta->dim_name[3], "D", sizeof pl_mask_meta->dim_name[3]) == 0);
    assert(std::strncmp(pl_mask_meta->dim_name[4], "T16lo8", sizeof pl_mask_meta->dim_name[4])
           == 0);
    assert(pl_mask_meta->dim[0] <= max_num_times / 2 / 8 / 8);
    assert(pl_mask_meta->dim[1] == num_frequencies / 4);
    assert(pl_mask_meta->dim[2] == num_polarizations);
    assert(pl_mask_meta->dim[3] == num_dishes);
    assert(pl_mask_meta->dim[4] == 8);
    for (int d = pl_mask_meta->dims - 1; d >= 0; --d)
        if (d == pl_mask_meta->dims - 1)
            assert(pl_mask_meta->stride[d] == 1);
        else
            assert(pl_mask_meta->stride[d]
                   == pl_mask_meta->stride[d + 1] * pl_mask_meta->dim[d + 1]);

    // Check E metadata
    const std::shared_ptr<metadataObject> E_mc = voltage_input_ringbuf_signal->get_metadata(0);
    assert(E_mc);
    assert(metadata_is_chord(E_mc));
    const std::shared_ptr<chordMetadata> E_meta = get_chord_metadata(E_mc);
    DEBUG("input E array: {:s} {:s}", E_meta->get_type_string(), E_meta->get_dimensions_string());
    assert(std::strncmp(E_meta->name, E_name, sizeof E_meta->name) == 0);
    assert(E_meta->type == kotekan::int4x2chime);
    assert(E_meta->dims == 4);
    assert(std::strncmp(E_meta->dim_name[0], "T", sizeof E_meta->dim_name[0]) == 0);
    assert(std::strncmp(E_meta->dim_name[1], "F", sizeof E_meta->dim_name[1]) == 0);
    assert(std::strncmp(E_meta->dim_name[2], "P", sizeof E_meta->dim_name[2]) == 0);
    assert(std::strncmp(E_meta->dim_name[3], "D", sizeof E_meta->dim_name[3]) == 0);
    assert(E_meta->dim[0] <= max_num_times);
    assert(E_meta->dim[1] == num_frequencies);
    assert(E_meta->dim[2] == num_polarizations);
    assert(E_meta->dim[3] == num_dishes);
    for (int d = E_meta->dims - 1; d >= 0; --d)
        if (d == E_meta->dims - 1)
            assert(E_meta->stride[d] == 1);
        else
            assert(E_meta->stride[d] == E_meta->stride[d + 1] * E_meta->dim[d + 1]);

    // Set S012 metadata
    std::shared_ptr<metadataObject> const S012_mc = rfi_S012_output_ringbuf_signal->get_metadata(0);
    std::shared_ptr<chordMetadata> const S012_meta = get_chord_metadata(S012_mc);
    *S012_meta = *pl_mask_meta;
    std::strncpy(S012_meta->name, S012_name, sizeof S012_meta->name);
    S012_meta->type = kotekan::uint64;
    S012_meta->dims = 5;
    std::strncpy(S012_meta->dim_name[0], "Tcoarse", sizeof S012_meta->dim_name[0]);
    std::strncpy(S012_meta->dim_name[1], "F", sizeof S012_meta->dim_name[1]);
    std::strncpy(S012_meta->dim_name[2], "S", sizeof S012_meta->dim_name[2]);
    std::strncpy(S012_meta->dim_name[3], "P", sizeof S012_meta->dim_name[3]);
    std::strncpy(S012_meta->dim_name[4], "D", sizeof S012_meta->dim_name[4]);
    assert(max_num_times % rfi_downsampling_factor == 0);
    S012_meta->dim[0] = max_num_times / rfi_downsampling_factor; // provisionally, overwritten below
    S012_meta->dim[1] = num_frequencies;
    S012_meta->dim[2] = 3;
    S012_meta->dim[3] = num_polarizations;
    S012_meta->dim[4] = num_dishes;
    for (int d = S012_meta->dims - 1; d >= 0; --d)
        if (d == S012_meta->dims - 1)
            S012_meta->stride[d] = 1;
        else
            S012_meta->stride[d] = S012_meta->stride[d + 1] * S012_meta->dim[d + 1];

    // Set S012tilde metadata
    std::shared_ptr<metadataObject> const S012tilde_mc =
        rfi_S012tilde_output_ringbuf_signal->get_metadata(0);
    std::shared_ptr<chordMetadata> const S012tilde_meta = get_chord_metadata(S012tilde_mc);
    *S012tilde_meta = *pl_mask_meta;
    std::strncpy(S012tilde_meta->name, S012tilde_name, sizeof S012tilde_meta->name);
    S012tilde_meta->type = kotekan::uint64;
    S012tilde_meta->dims = 3;
    std::strncpy(S012tilde_meta->dim_name[0], "Tcoarse", sizeof S012tilde_meta->dim_name[0]);
    std::strncpy(S012tilde_meta->dim_name[1], "F", sizeof S012tilde_meta->dim_name[1]);
    std::strncpy(S012tilde_meta->dim_name[2], "S", sizeof S012tilde_meta->dim_name[2]);
    assert(max_num_times % rfi_downsampling_factor == 0);
    S012tilde_meta->dim[0] =
        max_num_times / rfi_downsampling_factor; // provisionally, overwritten below
    S012tilde_meta->dim[1] = num_frequencies;
    S012tilde_meta->dim[2] = 3;
    for (int d = S012tilde_meta->dims - 1; d >= 0; --d)
        if (d == S012tilde_meta->dims - 1)
            S012tilde_meta->stride[d] = 1;
        else
            S012tilde_meta->stride[d] = S012tilde_meta->stride[d + 1] * S012tilde_meta->dim[d + 1];

    // Ringbuffer size
    const std::ptrdiff_t T_ringbuf_pl_mask =
        128 * pl_mask_input_ringbuf_signal->size / pl_mask_T128_sample_bytes;
    const std::ptrdiff_t T_ringbuf_voltage = voltage_input_ringbuf_signal->size / E_T_sample_bytes;
    assert(T_ringbuf_pl_mask == T_ringbuf_voltage);
    const std::ptrdiff_t T_ringbuf = T_ringbuf_pl_mask;
    const std::ptrdiff_t Tcoarse_ringbuf =
        rfi_S012_output_ringbuf_signal->size / S012_Tcoarse_sample_bytes;
    DEBUG("Input ringbuffer size (samples):  {:d}", T_ringbuf);
    DEBUG("Output ringbuffer size (samples): {:d}", Tcoarse_ringbuf);

    const std::ptrdiff_t Tlength = Tmax - Tmin;
    const std::ptrdiff_t Tcoarselength = Tcoarsemax - Tcoarsemin;
    DEBUG("Processed input samples: {:d}", Tlength);
    DEBUG("Produced output samples: {:d}", Tcoarselength);

    // Update metadata
    S012_meta->dim[0] = Tcoarselength;
    assert(S012_meta->dim[0] <= int(max_num_times / rfi_downsampling_factor));
    // Since we use a ring buffer we do not need to update `meta->sample0_offset`
    assert(S012_meta->nfreq >= 0);
    assert(S012_meta->nfreq == E_meta->nfreq);
    for (int freq = 0; freq < S012_meta->nfreq; ++freq) {
        S012_meta->freq_upchan_factor[freq] =
            rfi_downsampling_factor * E_meta->freq_upchan_factor[freq];
        // S012_meta->half_fpga_sample0[freq] = Evar_meta->half_fpga_sample0[freq];
        S012_meta->time_downsampling_fpga[freq] =
            rfi_downsampling_factor * E_meta->time_downsampling_fpga[freq];
    }

    S012tilde_meta->dim[0] = Tcoarselength;
    assert(S012tilde_meta->dim[0] <= int(max_num_times / rfi_downsampling_factor));
    // Since we use a ring buffer we do not need to update `meta->sample0_offset`
    assert(S012tilde_meta->nfreq >= 0);
    assert(S012tilde_meta->nfreq == E_meta->nfreq);
    for (int freq = 0; freq < S012tilde_meta->nfreq; ++freq) {
        S012tilde_meta->freq_upchan_factor[freq] =
            rfi_downsampling_factor * E_meta->freq_upchan_factor[freq];
        // S012tilde_meta->half_fpga_sample0[freq] = Evar_meta->half_fpga_sample0[freq];
        S012tilde_meta->time_downsampling_fpga[freq] =
            rfi_downsampling_factor * E_meta->time_downsampling_fpga[freq];
    }

    const std::uint8_t* const bf_mask_memory =
        static_cast<const std::uint8_t*>(device.get_gpu_memory(bf_mask_memname, bf_mask_length));
    const std::uint8_t* const pl_mask_memory = static_cast<const std::uint8_t*>(
        device.get_gpu_memory(pl_mask_memname, pl_mask_input_ringbuf_signal->size));
    const std::uint8_t* const E_memory = static_cast<const std::uint8_t*>(
        device.get_gpu_memory(E_memname, voltage_input_ringbuf_signal->size));
    std::uint8_t* const S012_memory = static_cast<std::uint8_t*>(
        device.get_gpu_memory(S012_memname, rfi_S012_output_ringbuf_signal->size));
    std::uint8_t* const S012tilde_memory = static_cast<std::uint8_t*>(
        device.get_gpu_memory(S012tilde_memname, rfi_S012tilde_output_ringbuf_signal->size));

    assert(Tmin % 128 == 0);
    const std::ptrdiff_t pl_mask_offset_bytes = pl_mask_T128_sample_bytes * Tmin / 128;
    const std::ptrdiff_t E_offset_bytes = E_T_sample_bytes * Tmin;
    const std::ptrdiff_t S012_offset_bytes = S012_Tcoarse_sample_bytes * Tcoarsemin;
    const std::ptrdiff_t S012tilde_offset_bytes = S012tilde_Tcoarse_sample_bytes * Tcoarsemin;

    const int T = Tmax - Tmin;
    const int Tsize = max_num_times;
    const int Tcoarse = Tcoarsemax - Tcoarsemin;
    assert(max_num_times % rfi_downsampling_factor == 0);
    const int Tcoarsesize = max_num_times / rfi_downsampling_factor;
    const int F_stride = S012_meta->stride[1];
    const int S_stride = S012_meta->stride[2];
    constexpr bool offset_encoded = true;
#warning "TODO"
#if 0
    n2k::launch_s0_kernel(
        static_cast<ulong*>(static_cast<void*>(S012_memory + S012_offset_bytes)),
        static_cast<const ulong*>(static_cast<const void*>(pl_mask_memory + pl_mask_offset_bytes)),
        T, Tmin, Tsize, num_frequencies, num_dishes * num_polarizations, rfi_downsampling_factor,
        F_stride, device.getStream(cuda_stream_id));
    n2k::launch_s12_kernel(
        static_cast<ulong*>(static_cast<void*>(S012_memory + S012_offset_bytes)) + S_stride,
        E_memory + E_offset_bytes, T, Tmin, Tsize, num_frequencies, num_dishes * num_polarizations,
        rfi_downsampling_factor, F_stride, offset_encoded, device.getStream(cuda_stream_id));
    n2k::launch_s012_station_downsample_kernel(
        static_cast<ulong*>(static_cast<void*>(S012tilde_memory + S012tilde_offset_bytes)),
        static_cast<const ulong*>(static_cast<const void*>(S012_memory + S012_offset_bytes)),
        static_cast<const uint8_t*>(static_cast<const void*>(bf_mask_memory)), Tcoarse, Tcoarsemin,
        Tcoarsesize, num_frequencies, num_dishes * num_polarizations,
        device.getStream(cuda_stream_id));
    // skKernel.launch(
    //     float *out_sk_feed_averaged,          // Shape (T,F,3)
    //     nullptr,
    //     uint *out_rfimask,                    // Shape (F,T*Nds/32), can be NULL
    //     const ulong *in_S012,                 // Shape (T,F,3,S)
    //     const uint8_t *in_bf_mask,            // Length S (bad feed mask)
    //     long rfimask_fstride,                 // Only used if (out_rfimask != NULL). NOTE: uint32
    //     stride, not bit stride! long T,                               // Number of downsampled
    //     times in S012 array long Tmin,                            // first time sample in input
    //     array long Tsize,                           // ringbuffer size long F, // Number of
    //     frequency channels long S,                               // Number of stations (= 2 *
    //     dishes) cudaStream_t stream = 0,
    //                 );
#endif

    return record_end_event();
}

void S012Kernel::finalize_frame() {
    const std::ptrdiff_t Tlength = Tmax - Tmin;
    const std::ptrdiff_t Tcoarselength = Tcoarsemax - Tcoarsemin;

    // Advance the input ringbuffer
    const std::ptrdiff_t T_consumed = num_consumed_elements(Tlength);
    DEBUG("Advancing input ringbuffer:");
    DEBUG("    Consumed samples:         {:d}", T_consumed);
    DEBUG("    Consumed pl_mask bytes:   {:d}", T_consumed / 128 * pl_mask_T128_sample_bytes);
    DEBUG("    Consumed voltage bytes:   {:d}", T_consumed * E_T_sample_bytes);
    pl_mask_input_ringbuf_signal->finish_read(unique_name, instance_num,
                                              T_consumed / 128 * pl_mask_T128_sample_bytes);
    voltage_input_ringbuf_signal->finish_read(unique_name, instance_num,
                                              T_consumed * E_T_sample_bytes);

    // Advance the output ringbuffer
    const std::ptrdiff_t Tcoarse_produced = Tcoarselength;
    DEBUG("Advancing output ringbuffer:");
    DEBUG("    Produced samples:         {:d}", Tcoarse_produced);
    DEBUG("    Produced S012 bytes:      {:d}", Tcoarse_produced * S012_Tcoarse_sample_bytes);
    DEBUG("    Produced S012tilde bytes: {:d}", Tcoarse_produced * S012tilde_Tcoarse_sample_bytes);
    rfi_S012_output_ringbuf_signal->finish_write(unique_name, instance_num,
                                                 Tcoarse_produced * S012_Tcoarse_sample_bytes);
    rfi_S012tilde_output_ringbuf_signal->finish_write(
        unique_name, instance_num, Tcoarse_produced * S012tilde_Tcoarse_sample_bytes);

    cudaCommand::finalize_frame();
}
