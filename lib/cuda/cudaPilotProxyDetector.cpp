#include "Config.hpp"               // for Config
#include "DataType.hpp"             // for int4x2_swapped_withoffset_t
#include "NDArray.hpp"              // for NDArray
#include "NDArrayBuffer.hpp"        // for NDArrayBuffer
#include "NDArrayRingBuffer.hpp"    // for NDArrayRingBuffer, read_descriptor_t
#include "bufferContainer.hpp"      // for bufferContainer
#include "chordMetadata.hpp"        // for chordMetadata
#include "cudaCommand.hpp"          // for cudaCommand, cudaCommandState, ...
#include "cudaDeviceInterface.hpp"  // for cudaDeviceInterface
#include "cudaPilotProxyPacker.hpp" // for launch_pilotproxy_pack
#include "cudaUtils.hpp"            // for CHECK_CUDA_ERROR
#include "gpuCommand.hpp"           // for gpuCommandType
#include "kotekanLogging.hpp"       // for FATAL_ERROR, INFO, DEBUG
#include "pilotproxy/f_statistic.h" // for FStat_* (vendored PilotProxy kernel)

#include "fmt.hpp"  // for compile_string_to_view
#include "json.hpp" // for json

#include <array>              // for array
#include <cassert>            // for assert
#include <cstddef>            // for ptrdiff_t
#include <cstdint>            // for uint64_t, int8_t, uint8_t
#include <cuda_runtime_api.h> // for cudaEventCreateWithFlags, cudaStreamWaitEvent
#include <driver_types.h>     // for cudaEvent_t, CUstream_st
#include <fstream>            // for ifstream
#include <functional>         // for function
#include <iterator>           // for istreambuf_iterator
#include <memory>             // for shared_ptr, static_pointer_cast
#include <stdexcept>          // for runtime_error
#include <string>             // for string, stoull
#include <vector>             // for vector

using nlohmann::json;

/**
 * Shared state for the cudaPilotProxyDetector instances: the parsed runtime
 * bundle, the first-frame channel binding, the FStat kernel handles, and the
 * events that order the vendored library's default-stream work against
 * kotekan's per-pipeline stream. One state object is shared by the
 * buffer_depth pipelined command instances; all mutation happens from the
 * gpuProcess main thread, which calls execute() serially, so no locking is
 * needed.
 */
class cudaPilotProxyDetectorState : public cudaCommandState {
public:
    cudaPilotProxyDetectorState(kotekan::Config& config, const std::string& unique_name,
                                kotekan::bufferContainer& host_buffers,
                                cudaDeviceInterface& device) :
        cudaCommandState(config, unique_name, host_buffers, device) {
        const std::string pilot_profiles_path =
            config.get<std::string>(unique_name, "pilot_profiles_path");
        const std::string weights_path = config.get<std::string>(unique_name, "weights_path");
        load_bundle(pilot_profiles_path, weights_path);
    }

    ~cudaPilotProxyDetectorState() {
        for (auto& channel : bound_channels)
            if (channel.fstat_handle)
                FStat_Destroy(channel.fstat_handle);
        if (packed_ready_event)
            cudaEventDestroy(packed_ready_event);
        if (default_stream_done_event)
            cudaEventDestroy(default_stream_done_event);
    }

    /// One pilot channel row of pilot_profiles.json (runtime bundle).
    struct PilotChannelProfile {
        int physical_channel = -1;
        int chord_channel_id = -1;
        double pilot_frequency_hz = 0.0;
        std::ptrdiff_t weight_offset_bytes = 0;
        std::ptrdiff_t weight_nbytes = 0;
        unsigned long long half_threshold_num = 0;
        unsigned long long half_threshold_den = 0;
        // fine_calibration (kernel core 2.3.0 mask epilogue); usable only
        // when status == "calibrated".
        bool fine_calibrated = false;
        int anchor_bin = 0;
        int designated_half_width = 0;
        int cfar_rank = 0;
        unsigned long long multiplier_q16 = 0;
        std::array<unsigned long long, 4> bulk_mask_words{{0, 0, 0, 0}};
    };

    /// A bundle profile bound to a local frequency index at first frame.
    struct BoundChannel {
        int freq_index = -1; // local F index on this node
        const PilotChannelProfile* profile = nullptr;
        bool use_fine_mask = false;
        void* fstat_handle = nullptr;    // FStat_Create handle (d_in bound)
        std::int8_t* d_packed = nullptr; // this channel's packed staging region
    };

    enum class RunState { wait_for_first_frame, running, disabled };

    // Parsed bundle (immutable after construction)
    std::vector<PilotChannelProfile> bundle_profiles;
    std::vector<std::int8_t> weight_bank; // full weights.bin contents (host)
    bool time_reverse_windows = false;    // bundle input_preprocessing flag

    // First-frame binding state (mutated only from the serialized execute path)
    RunState run_state = RunState::wait_for_first_frame;
    std::vector<int> bound_coarse_freq; // coarse_freq at binding, to detect changes
    std::vector<BoundChannel> bound_channels;
    bool logged_disabled = false;

    // Default-stream ordering events (created at binding)
    cudaEvent_t packed_ready_event = nullptr;
    cudaEvent_t default_stream_done_event = nullptr;
    bool default_stream_done_valid = false;

    const std::int8_t* weights_for(const PilotChannelProfile& profile) const {
        return weight_bank.data() + profile.weight_offset_bytes;
    }

private:
    void load_bundle(const std::string& pilot_profiles_path, const std::string& weights_path) {
        std::ifstream profiles_file(pilot_profiles_path);
        if (!profiles_file)
            throw std::runtime_error(
                fmt::format(fmt("cudaPilotProxyDetector: cannot open pilot_profiles_path {:s}"),
                            pilot_profiles_path));
        json pilot_profiles;
        profiles_file >> pilot_profiles;

        std::ifstream weights_file(weights_path, std::ios::binary);
        if (!weights_file)
            throw std::runtime_error(fmt::format(
                fmt("cudaPilotProxyDetector: cannot open weights_path {:s}"), weights_path));
        weight_bank.assign(std::istreambuf_iterator<char>(weights_file),
                           std::istreambuf_iterator<char>());

        // The bundle records whether detector windows must be time-reversed
        // before the kernel. Post-spectral-sense weight banks synthesize
        // exp(-2j*pi*f*k) templates in the true-sense raw frame and assume
        // this adapter flip, so CHORD bundles set it TRUE despite the
        // upright spectral sense (pilot-proxy explicit-baseband-frame
        // contract; see tests/core/test_chord_tone_injection.py upstream).
        const auto& preprocessing = pilot_profiles.at("input_preprocessing");
        time_reverse_windows =
            preprocessing.at("time_reverse_detector_windows_before_kernel").get<bool>();

        int expected_window = 0, expected_terms = 0, expected_bits = 0;
        FStat_GetSpecs(&expected_window, &expected_terms, &expected_bits, nullptr);
        const std::ptrdiff_t profile_nbytes =
            std::ptrdiff_t(expected_terms) * expected_window * sizeof(std::int8_t);

        for (const auto& row : pilot_profiles.at("profiles")) {
            PilotChannelProfile profile;
            profile.physical_channel = row.at("physical_channel").get<int>();
            // chord_channel_id is the kotekan chordMetadata coarse_freq
            // namespace (CHORDTelescope::to_freq_id). Rows without it cannot
            // be bound on a CHORD node and are skipped with a note at
            // binding time.
            if (row.contains("chord_channel_id") && !row.at("chord_channel_id").is_null())
                profile.chord_channel_id = row.at("chord_channel_id").get<int>();
            profile.pilot_frequency_hz = row.at("pilot_frequency_hz").get<double>();
            profile.weight_offset_bytes = row.at("weight_bank_offset_bytes").get<std::ptrdiff_t>();
            profile.weight_nbytes = row.at("weight_bank_nbytes").get<std::ptrdiff_t>();
            profile.half_threshold_num =
                row.at("positive_excess_half_threshold_num").get<unsigned long long>();
            profile.half_threshold_den =
                row.at("positive_excess_half_threshold_den").get<unsigned long long>();
            if (profile.weight_nbytes != profile_nbytes)
                throw std::runtime_error(fmt::format(
                    fmt("cudaPilotProxyDetector: weight_bank_nbytes {:d} does not match the "
                        "compiled kernel contract {:d} for physical channel {:d}"),
                    std::int64_t(profile.weight_nbytes), std::int64_t(profile_nbytes),
                    profile.physical_channel));
            if (profile.weight_offset_bytes < 0
                || profile.weight_offset_bytes + profile.weight_nbytes
                       > std::ptrdiff_t(weight_bank.size()))
                throw std::runtime_error(fmt::format(
                    fmt("cudaPilotProxyDetector: weight bank range [{:d}, {:d}) exceeds "
                        "weights.bin size {:d} for physical channel {:d}"),
                    std::int64_t(profile.weight_offset_bytes),
                    std::int64_t(profile.weight_offset_bytes + profile.weight_nbytes),
                    std::uint64_t(weight_bank.size()), profile.physical_channel));
            if (row.contains("fine_calibration") && row.at("fine_calibration").is_object()) {
                const auto& calibration = row.at("fine_calibration");
                if (calibration.value("status", "") == "calibrated") {
                    profile.fine_calibrated = true;
                    profile.anchor_bin = calibration.at("anchor_bin").get<int>();
                    profile.designated_half_width =
                        calibration.at("designated_half_width").get<int>();
                    profile.cfar_rank = calibration.at("cfar_rank").get<int>();
                    profile.multiplier_q16 =
                        calibration.at("cfar_multiplier_q16").get<unsigned long long>();
                    const auto& words_hex = calibration.at("bulk_mask_words_hex");
                    if (!words_hex.is_array() || words_hex.size() != 4)
                        throw std::runtime_error(
                            "cudaPilotProxyDetector: bulk_mask_words_hex must be 4 hex words");
                    for (int word = 0; word < 4; ++word)
                        profile.bulk_mask_words[word] =
                            std::stoull(words_hex.at(word).get<std::string>(), nullptr, 16);
                }
            }
            bundle_profiles.push_back(profile);
        }
    }
};

/**
 * @class cudaPilotProxyDetector
 * @brief cudaCommand running the PilotProxy ATSC DTV pilot-tone F-statistic
 * detector (vendored in lib/cuda/pilotproxy) on CHORD voltage data.
 *
 * @author Dylan Gormley (PilotProxy detector core); kotekan stage authored
 *         for the pilot-proxy <-> kotekan integration.
 *
 * Consumes the [T, F, P, D] int4x2_swapped_withoffset voltage ring buffer.
 * At the first processed block the stage reads the chordMetadata
 * `coarse_freq` channel identifiers and binds every local frequency whose
 * identifier appears as `chord_channel_id` in the runtime bundle
 * (pilot_profiles.json + weights.bin exported by
 * `pilot-proxy export-runtime-weight-bundle`); the runtime state machine is
 * the one proposed in pilot-proxy's docs/KOTEKAN_INTERFACE_PREP.md:
 *
 *     WAIT_FOR_FIRST_FRAME -> RUNNING   (>= 1 pilot channel on this node)
 *                          -> DISABLED  (no pilot channel on this node)
 *
 * A DISABLED node keeps consuming input and emits all-zero mask/power
 * frames, so downstream consumers see a uniform product. A change of
 * `coarse_freq` during the run is rejected with FATAL_ERROR (the bundle
 * binding is per-run).
 *
 * Per detector block (default 16384 samples = 2 CHORD GPU frames =
 * 83.88608 ms) and per bound channel, the stage:
 *
 *  1. packs the channel's [T, P, D] slice into the kernel's row-major
 *     detector matrix (offset-binary -> two's-complement via XOR 0x88;
 *     stream s = p * num_dishes + d; per-window time reversal when the
 *     bundle's input_preprocessing requests it -- true for CHORD bundles,
 *     whose post-spectral-sense weight templates assume the flip);
 *  2. runs the vendored F-statistic kernel: the fused fine designated-set
 *     CFAR mask (`FStat_Compute_FusedFineMask_U64`, kernel core 2.3.0) when
 *     the channel's bundle `fine_calibration.status` is "calibrated" and
 *     the block has the frozen 128 windows/stream, otherwise the coarse
 *     norm-corrected positive-excess rational mask
 *     (`FStat_Compute_NumDen_Mask_RationalHalf`);
 *  3. records the decision in `dtv_mask` (one byte per local frequency;
 *     1 = pilot detected / reject) and the exact uint64 coarse power
 *     marginals (target, lower ref, upper ref) in `dtv_powers` for
 *     validation against the Python reference path.
 *
 * The vendored library executes on the legacy default CUDA stream; the
 * stage brackets it with events against kotekan's stream (and the legacy
 * default stream's implicit synchronization with blocking streams makes
 * this doubly safe).
 *
 * @par GPU Memory
 * @gpu_mem Input voltage
 *   @gpu_mem_buffer    @c ring
 *   @gpu_mem_quantity  @c E
 *   @gpu_mem_type      @c int4x2_swapped_withoffset
 *   @gpu_mem_dim_name  [@c T][@c F][@c P][@c D]
 *   @gpu_mem_shape     [@c buffer_depth * num_times][@c num_frequencies][@c
 * num_polarizations][@c num_dishes]
 *   @gpu_mem_metadata  @c chordMetadata
 * @gpu_mem Output DTV pilot mask
 *   @gpu_mem_buffer    @c standard
 *   @gpu_mem_quantity  @c dtv_mask
 *   @gpu_mem_type      @c int8
 *   @gpu_mem_dim_name  [@c F]
 *   @gpu_mem_shape     [@c num_frequencies]
 *   @gpu_mem_metadata  @c chordMetadata
 * @gpu_mem Output DTV coarse powers
 *   @gpu_mem_buffer    @c standard
 *   @gpu_mem_quantity  @c dtv_powers
 *   @gpu_mem_type      @c uint64
 *   @gpu_mem_dim_name  [@c F][@c W]
 *   @gpu_mem_shape     [@c num_frequencies][@c 3]
 *   @gpu_mem_metadata  @c chordMetadata
 *
 * @conf buffer_depth               Int. GPU frames used for pipelining.
 * @conf num_times                  Int. Voltage samples per upstream GPU
 *                                  frame (ring advance unit; 8192 for CHORD).
 * @conf samples_per_detector_frame Int. Channelized samples per detector
 *                                  block; multiple of 128. Default 16384
 *                                  (the frozen fine-transform geometry:
 *                                  128 windows/stream).
 * @conf num_frequencies            Int. Local coarse frequencies (F).
 * @conf num_polarizations          Int. Polarizations (2).
 * @conf num_dishes                 Int. Dishes (D).
 * @conf voltage_name               String. Base name of the voltage ring
 *                                  buffer (default "voltage").
 * @conf dtv_mask_name              String. Base name of the mask output
 *                                  buffer (default "dtv_mask").
 * @conf dtv_powers_name            String. Base name of the powers output
 *                                  buffer (default "dtv_powers").
 * @conf pilot_profiles_path        String. Runtime bundle pilot_profiles.json.
 * @conf weights_path               String. Runtime bundle weights.bin.
 * @conf decision_mode              String. "auto" (default; fine mask for
 *                                  calibrated channels, coarse otherwise)
 *                                  or "coarse" (force the coarse rational
 *                                  mask for every channel).
 */
class cudaPilotProxyDetector : public cudaCommand {
public:
    cudaPilotProxyDetector(kotekan::Config& config, const std::string& unique_name,
                           kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                           const int instance_num, const std::shared_ptr<cudaCommandState>& state);
    virtual ~cudaPilotProxyDetector();

    int wait_on_precondition() override;
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame() override;

private:
    using State = cudaPilotProxyDetectorState;

    void bind_first_frame(const std::shared_ptr<const chordMetadata>& metadata);

    std::shared_ptr<State> detector_state;

    // Parameters
    const int buffer_depth;
    const int num_times; // upstream producer samples per GPU frame
    const int samples_per_detector_frame;
    const int num_frequencies;
    const int num_polarizations;
    const int num_dishes;
    const std::string decision_mode; // "auto" or "coarse"

    // Kotekan buffer names
    const std::string voltage_name;
    const std::string dtv_mask_name;
    const std::string dtv_powers_name;

    // Derived geometry
    const int num_streams;             // P * D
    const int detector_window_samples; // K, from the compiled kernel
    const int windows_per_stream;      // samples_per_detector_frame / K
    const int detector_rows;           // num_streams * windows_per_stream

    // Buffers
    NDArrayRingBuffer<kotekan::int4x2_swapped_withoffset_t, 4> voltage;
    NDArrayBuffer<std::int8_t, 1> dtv_mask;
    NDArrayBuffer<std::uint64_t, 2> dtv_powers;
};

REGISTER_CUDA_COMMAND_WITH_STATE(cudaPilotProxyDetector, cudaPilotProxyDetectorState);

static int query_detector_window_samples() {
    int window = 0;
    FStat_GetSpecs(&window, nullptr, nullptr, nullptr);
    return window;
}

cudaPilotProxyDetector::cudaPilotProxyDetector(kotekan::Config& config,
                                               const std::string& unique_name,
                                               kotekan::bufferContainer& host_buffers,
                                               cudaDeviceInterface& device, const int instance_num,
                                               const std::shared_ptr<cudaCommandState>& state) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, state,
                "cudaPilotProxyDetector"),
    detector_state(std::static_pointer_cast<State>(state)),
    // Parameters
    buffer_depth(config.get<int>(unique_name, "buffer_depth")),
    num_times(config.get<int>(unique_name, "num_times")),
    samples_per_detector_frame(
        config.get_default<int>(unique_name, "samples_per_detector_frame", 16384)),
    num_frequencies(config.get<int>(unique_name, "num_frequencies")),
    num_polarizations(config.get<int>(unique_name, "num_polarizations")),
    num_dishes(config.get<int>(unique_name, "num_dishes")),
    decision_mode(config.get_default<std::string>(unique_name, "decision_mode", "auto")),
    // Buffer names
    voltage_name(config.get_default<std::string>(unique_name, "voltage_name", "voltage")),
    dtv_mask_name(config.get_default<std::string>(unique_name, "dtv_mask_name", "dtv_mask")),
    dtv_powers_name(config.get_default<std::string>(unique_name, "dtv_powers_name", "dtv_powers")),
    // Derived geometry
    num_streams(num_polarizations * num_dishes),
    detector_window_samples(query_detector_window_samples()),
    windows_per_stream(samples_per_detector_frame / detector_window_samples),
    detector_rows(num_streams * windows_per_stream),
    // Buffers
    voltage(voltage_name, "E",
            std::array<std::ptrdiff_t, 4>{std::ptrdiff_t(buffer_depth) * num_times, num_frequencies,
                                          num_polarizations, num_dishes},
            std::array<std::string, 4>{"T", "F", "P", "D"},
            std::array<std::ptrdiff_t, 4>{1, 1, 1, 1}, *this),
    dtv_mask(dtv_mask_name, "dtv_mask", std::array<std::ptrdiff_t, 1>{num_frequencies},
             std::array<std::string, 1>{"F"}, std::array<std::ptrdiff_t, 1>{1}, *this),
    dtv_powers(dtv_powers_name, "dtv_powers", std::array<std::ptrdiff_t, 2>{num_frequencies, 3},
               std::array<std::string, 2>{"F", "W"}, std::array<std::ptrdiff_t, 2>{1, 1}, *this)
//
{
    if (decision_mode != "auto" && decision_mode != "coarse")
        FATAL_ERROR("decision_mode must be \"auto\" or \"coarse\"; got {:s}", decision_mode);

    if (samples_per_detector_frame % detector_window_samples != 0)
        FATAL_ERROR("samples_per_detector_frame ({:d}) must be a multiple of the compiled "
                    "detector window ({:d})",
                    samples_per_detector_frame, detector_window_samples);

    // The packer wraps ring indices with a power-of-two bitmask, so the ring
    // size (the slowest, "T" dimension) must be a power of two.
    const std::ptrdiff_t ring_size = voltage.get_ndarray().extent(0);
    if (ring_size <= 0 || (ring_size & (ring_size - 1)) != 0)
        FATAL_ERROR("voltage ring size {:d} is not a power of two", ring_size);
    if (ring_size < samples_per_detector_frame)
        FATAL_ERROR("voltage ring size {:d} is smaller than one detector block ({:d})", ring_size,
                    samples_per_detector_frame);

    int fine_windows = 0;
    FStat_GetFineSpecs(&fine_windows, nullptr, nullptr);
    if (decision_mode == "auto" && windows_per_stream != fine_windows)
        INFO("samples_per_detector_frame ({:d}) gives {:d} windows/stream, not the frozen fine "
             "transform length {:d}; calibrated channels will fall back to the coarse mask",
             samples_per_detector_frame, windows_per_stream, fine_windows);

    voltage.register_consumer();
    dtv_mask.register_producer();
    dtv_powers.register_producer();

    set_command_type(gpuCommandType::KERNEL);
}

cudaPilotProxyDetector::~cudaPilotProxyDetector() {}

int cudaPilotProxyDetector::wait_on_precondition() {
    DEBUG("Waiting for voltage ring buffer data for frame {:d}...", gpu_frame_id);
    const std::ptrdiff_t samples = samples_per_detector_frame;
    const int errcode = voltage.wait_and_claim_readable([&](const std::ptrdiff_t available) {
        if (available < samples)
            return read_descriptor_t{.claimed = 0, .read = 0};
        return read_descriptor_t{.claimed = samples, .read = samples};
    });
    if (errcode < 0)
        return errcode;
    DEBUG("Done waiting for voltage ring buffer data for frame {:d}", gpu_frame_id);
    return 0;
}

void cudaPilotProxyDetector::bind_first_frame(
    const std::shared_ptr<const chordMetadata>& metadata) {
    State& state = *detector_state;

    if (!metadata->has_coarse_freq())
        FATAL_ERROR("voltage metadata does not carry coarse_freq channel identifiers; cannot "
                    "bind the PilotProxy runtime bundle");
    const std::vector<int> coarse_freq = metadata->get_coarse_freq();
    if (int(coarse_freq.size()) != num_frequencies)
        FATAL_ERROR("voltage metadata carries {:d} coarse_freq entries but num_frequencies is "
                    "{:d}",
                    int(coarse_freq.size()), num_frequencies);
    state.bound_coarse_freq = coarse_freq;

    int fine_windows = 0;
    FStat_GetFineSpecs(&fine_windows, nullptr, nullptr);
    const bool fine_geometry_ok = windows_per_stream == fine_windows;

    int num_without_id = 0;
    for (int f = 0; f < num_frequencies; ++f) {
        for (const auto& profile : state.bundle_profiles) {
            if (profile.chord_channel_id < 0) {
                // Counted once below; a bundle without chord ids cannot bind.
                continue;
            }
            if (profile.chord_channel_id != coarse_freq[f])
                continue;
            State::BoundChannel channel;
            channel.freq_index = f;
            channel.profile = &profile;
            channel.use_fine_mask =
                decision_mode == "auto" && profile.fine_calibrated && fine_geometry_ok;
            state.bound_channels.push_back(channel);
            INFO("PilotProxy: bound ATSC physical channel {:d} (pilot {:.6f} MHz, "
                 "chord_channel_id {:d}) to local frequency index {:d}; decision path: {:s}",
                 profile.physical_channel, profile.pilot_frequency_hz * 1e-6,
                 profile.chord_channel_id, f, channel.use_fine_mask ? "fine_mask" : "coarse");
        }
    }
    for (const auto& profile : state.bundle_profiles)
        if (profile.chord_channel_id < 0)
            ++num_without_id;
    if (num_without_id > 0)
        INFO("PilotProxy: {:d} bundle profile(s) carry no chord_channel_id and cannot bind on "
             "this receiver; re-export the bundle from a receiver profile that declares "
             "metadata.channel_id_map",
             num_without_id);

    if (state.bound_channels.empty()) {
        state.run_state = State::RunState::disabled;
        INFO("PilotProxy: no bundle pilot channel among this node's {:d} coarse channels; "
             "detector DISABLED (emitting all-zero masks)",
             num_frequencies);
        return;
    }

    // Shared staging for the packed detector matrices, one region per bound
    // channel. Shared across the pipelined instances: the packer of block
    // N+1 waits on the default_stream_done event of block N.
    const std::ptrdiff_t region_bytes =
        std::ptrdiff_t(detector_rows) * detector_window_samples * sizeof(std::int8_t);
    std::int8_t* const packed_base = static_cast<std::int8_t*>(device.get_gpu_memory(
        "pilotproxy_packed", region_bytes * std::ptrdiff_t(state.bound_channels.size())));

    for (std::size_t i = 0; i < state.bound_channels.size(); ++i) {
        auto& channel = state.bound_channels[i];
        channel.d_packed = packed_base + std::ptrdiff_t(i) * region_bytes;
        channel.fstat_handle = FStat_Create(channel.d_packed, nullptr, detector_rows);
        if (!channel.fstat_handle)
            FATAL_ERROR("FStat_Create failed for physical channel {:d}: {:s}",
                        channel.profile->physical_channel, FStat_LastError());
    }

    CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&state.packed_ready_event, cudaEventDisableTiming));
    CHECK_CUDA_ERROR(
        cudaEventCreateWithFlags(&state.default_stream_done_event, cudaEventDisableTiming));

    state.run_state = State::RunState::running;
}

cudaEvent_t cudaPilotProxyDetector::execute(cudaPipelineState& /*pipestate*/,
                                            const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();
    record_start_event();

    State& state = *detector_state;

    voltage.check_metadata();
    const std::shared_ptr<const chordMetadata> in_metadata = voltage.get_metadata();

    if (state.run_state == State::RunState::wait_for_first_frame)
        bind_first_frame(in_metadata);
    else if (in_metadata->has_coarse_freq()
             && in_metadata->get_coarse_freq() != state.bound_coarse_freq)
        // The bundle binding is per-run; a changing channel identity would
        // silently select the wrong weight profile (see
        // docs/KOTEKAN_INTERFACE_PREP.md "Runtime State Machine").
        FATAL_ERROR("voltage coarse_freq changed after first-frame binding");

    // Output metadata: one mask/power set per detector block. The frame
    // identity key is the ring-buffer read offset (in channelized samples).
    dtv_mask.set_metadata(in_metadata);
    dtv_powers.set_metadata(in_metadata);
    const std::ptrdiff_t block_start_sample = voltage.get_read_valid().begin();
    for (const auto& metadata : {dtv_mask.get_metadata(), dtv_powers.get_metadata()}) {
        metadata->set_fpga_seq_num(block_start_sample);
        metadata->set_time_downsampling_fpga(in_metadata->get_time_downsampling_fpga()
                                             * samples_per_detector_frame);
    }

    std::int8_t* const mask_memory = dtv_mask.get_ndarray().data();
    std::uint64_t* const powers_memory = dtv_powers.get_ndarray().data();
    const cudaStream_t kotekan_stream = device.getStream(cuda_stream_id);

    // Non-pilot channels (and DISABLED nodes) report mask 0 / powers 0.
    CHECK_CUDA_ERROR(cudaMemsetAsync(
        mask_memory, 0, std::size_t(num_frequencies) * sizeof(std::int8_t), kotekan_stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(powers_memory, 0,
                                     std::size_t(num_frequencies) * 3 * sizeof(std::uint64_t),
                                     kotekan_stream));

    if (state.run_state == State::RunState::disabled) {
        if (!state.logged_disabled) {
            DEBUG("PilotProxy detector disabled on this node");
            state.logged_disabled = true;
        }
        return record_end_event();
    }

    const kotekan::int4x2_swapped_withoffset_t* const voltage_memory = voltage.get_ndarray().data();
    const std::ptrdiff_t ring_size = voltage.get_ndarray().extent(0);
    const std::ptrdiff_t ring_pos = block_start_sample % ring_size;

    // Order this block's packer after the previous block's default-stream
    // detector work (the packed staging regions are shared across blocks).
    if (state.default_stream_done_valid)
        CHECK_CUDA_ERROR(cudaStreamWaitEvent(kotekan_stream, state.default_stream_done_event, 0));

    for (const auto& channel : state.bound_channels)
        launch_pilotproxy_pack(channel.d_packed,
                               reinterpret_cast<const std::uint8_t*>(voltage_memory), num_dishes,
                               num_polarizations, num_frequencies, channel.freq_index,
                               samples_per_detector_frame, ring_size, ring_pos,
                               detector_window_samples, state.time_reverse_windows, kotekan_stream);

    // Hand off to the vendored library, which launches on the legacy default
    // stream (stream 0). The events make the ordering explicit; the legacy
    // default stream's implicit synchronization with blocking streams would
    // order these anyway.
    CHECK_CUDA_ERROR(cudaEventRecord(state.packed_ready_event, kotekan_stream));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(/*stream=*/nullptr, state.packed_ready_event, 0));

    // Per-channel scratch: fine-power working accumulator (required by the
    // fused mask entry), an int32 mask cell per channel, and a num/den pair
    // per channel for the coarse entry.
    const std::ptrdiff_t num_bound = std::ptrdiff_t(state.bound_channels.size());
    unsigned long long* const fine_power_scratch = static_cast<unsigned long long*>(
        device.get_gpu_memory("pilotproxy_fine_power_scratch", 3 * 256 * sizeof(std::uint64_t)));
    int* const mask_i32_scratch = static_cast<int*>(
        device.get_gpu_memory("pilotproxy_mask_i32_scratch", num_bound * sizeof(int)));
    unsigned long long* const numden_scratch =
        static_cast<unsigned long long*>(device.get_gpu_memory(
            "pilotproxy_numden_scratch", num_bound * 2 * sizeof(unsigned long long)));

    for (std::ptrdiff_t i = 0; i < num_bound; ++i) {
        const auto& channel = state.bound_channels[i];
        const auto& profile = *channel.profile;
        const InputType* const weights =
            reinterpret_cast<const InputType*>(state.weights_for(profile));
        std::uint64_t* const channel_powers = powers_memory + 3 * channel.freq_index;
        if (channel.use_fine_mask) {
            // Deployed fine designated-set CFAR (kernel core 2.3.0): one
            // launch from packed samples to the mask bit; the coarse power
            // marginals come along as the debug tap for validation.
            FStat_Compute_FusedFineMask_U64(
                channel.fstat_handle, weights, profile.anchor_bin, profile.designated_half_width,
                profile.bulk_mask_words.data(), profile.cfar_rank, profile.multiplier_q16,
                fine_power_scratch, mask_i32_scratch + i,
                reinterpret_cast<unsigned long long*>(channel_powers), nullptr);
        } else {
            // Coarse norm-corrected positive-excess rule: exact uint64
            // powers plus the rational half-threshold mask.
            FStat_Compute_Powers_U64(channel.fstat_handle, weights,
                                     reinterpret_cast<unsigned long long*>(channel_powers));
            FStat_Compute_NumDen_Mask_RationalHalf(
                channel.fstat_handle, weights, profile.half_threshold_num,
                profile.half_threshold_den, numden_scratch + 2 * i, numden_scratch + 2 * i + 1,
                reinterpret_cast<unsigned char*>(mask_memory) + channel.freq_index);
        }
    }
    CHECK_CUDA_ERROR(cudaEventRecord(state.default_stream_done_event, /*stream=*/nullptr));
    state.default_stream_done_valid = true;
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(kotekan_stream, state.default_stream_done_event, 0));

    // Fine-mask channels: fold the int32 mask cells into the int8 mask
    // product. The values are 0/1, so copying the little-endian low byte is
    // exact (both CUDA targets are little-endian).
    for (std::ptrdiff_t i = 0; i < num_bound; ++i) {
        const auto& channel = state.bound_channels[i];
        if (!channel.use_fine_mask)
            continue;
        CHECK_CUDA_ERROR(cudaMemcpyAsync(mask_memory + channel.freq_index, mask_i32_scratch + i, 1,
                                         cudaMemcpyDeviceToDevice, kotekan_stream));
    }

    return record_end_event();
}

void cudaPilotProxyDetector::finalize_frame() {
    voltage.finish_read();
    cudaCommand::finalize_frame();
}
