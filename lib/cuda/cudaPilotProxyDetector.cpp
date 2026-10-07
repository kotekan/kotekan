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

#include <algorithm>          // for find, max
#include <array>              // for array
#include <bitset>             // for bitset
#include <cassert>            // for assert
#include <cctype>             // for isxdigit
#include <cmath>              // for isfinite
#include <cstddef>            // for ptrdiff_t
#include <cstdint>            // for uint64_t, int8_t, uint8_t
#include <cuda_runtime_api.h> // for cudaMemsetAsync, cudaMemcpyAsync
#include <driver_types.h>     // for cudaEvent_t, cudaStream_t
#include <fstream>            // for ifstream
#include <functional>         // for function
#include <iterator>           // for istreambuf_iterator
#include <limits>             // for numeric_limits
#include <memory>             // for shared_ptr, static_pointer_cast
#include <set>                // for set
#include <stdexcept>          // for runtime_error
#include <string>             // for string, stoull
#include <vector>             // for vector

using nlohmann::json;

/**
 * Bundle, channel bindings and CUDA resources shared by pipelined instances.
 * The owning gpuProcess calls execute() serially.
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
    }

    /// One pilot channel row of pilot_profiles.json (runtime bundle).
    struct PilotChannelProfile {
        int physical_channel = -1;
        int chord_channel_id = -1;
        double pilot_frequency_hz = 0.0;
        std::ptrdiff_t weight_offset_bytes = 0;
        std::ptrdiff_t weight_nbytes = 0;
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
        void* fstat_handle = nullptr;    // FStat_Create handle (d_in bound)
        std::int8_t* d_packed = nullptr; // this channel's packed staging region
    };

    enum class RunState { wait_for_first_frame, running, disabled };

    // Parsed bundle (immutable after construction)
    std::vector<PilotChannelProfile> bundle_profiles;
    std::vector<std::int8_t> weight_bank; // full weights.bin contents (host)
    bool time_reverse_windows = false;    // bundle input_preprocessing flag

    // Channel bindings, initialized by the first execute() call.
    RunState run_state = RunState::wait_for_first_frame;
    std::vector<int> bound_coarse_freq; // coarse_freq at binding, to detect changes
    std::vector<BoundChannel> bound_channels;
    bool logged_disabled = false;

    const std::int8_t* weights_for(const PilotChannelProfile& profile) const {
        return weight_bank.data() + profile.weight_offset_bytes;
    }

private:
    template<typename T>
    static T read_integer(const json& object, const char* field) {
        const auto& value = object.at(field);
        bool valid = false;
        if (value.is_number_unsigned()) {
            valid = value.get<unsigned long long>()
                    <= static_cast<unsigned long long>(std::numeric_limits<T>::max());
        } else if (value.is_number_integer()) {
            const auto number = value.get<long long>();
            valid = number < 0
                        ? std::numeric_limits<T>::is_signed
                              && number >= static_cast<long long>(std::numeric_limits<T>::lowest())
                        : static_cast<unsigned long long>(number)
                              <= static_cast<unsigned long long>(std::numeric_limits<T>::max());
        }
        if (!valid)
            throw std::runtime_error(fmt::format(
                fmt("cudaPilotProxyDetector: {:s} must be an integer in the supported range"),
                field));
        return value.get<T>();
    }

    void load_bundle(const std::string& pilot_profiles_path, const std::string& weights_path) {
        std::ifstream profiles_file(pilot_profiles_path);
        if (!profiles_file)
            throw std::runtime_error(
                fmt::format(fmt("cudaPilotProxyDetector: cannot open pilot_profiles_path {:s}"),
                            pilot_profiles_path));
        json pilot_profiles;
        profiles_file >> pilot_profiles;

        if (pilot_profiles.value("schema_version", "") != "pilotproxy_runtime_pilot_profiles_v1")
            throw std::runtime_error(
                "cudaPilotProxyDetector: unsupported or missing pilot_profiles schema_version");
        if (!pilot_profiles.contains("profiles") || !pilot_profiles.at("profiles").is_array()
            || pilot_profiles.at("profiles").empty())
            throw std::runtime_error(
                "cudaPilotProxyDetector: pilot_profiles must be a non-empty array");

        std::ifstream weights_file(weights_path, std::ios::binary);
        if (!weights_file)
            throw std::runtime_error(fmt::format(
                fmt("cudaPilotProxyDetector: cannot open weights_path {:s}"), weights_path));
        weight_bank.assign(std::istreambuf_iterator<char>(weights_file),
                           std::istreambuf_iterator<char>());
        if (weight_bank.empty())
            throw std::runtime_error("cudaPilotProxyDetector: weights.bin is empty");

        // CHORD's exported weight templates require per-window time reversal.
        const auto& preprocessing = pilot_profiles.at("input_preprocessing");
        time_reverse_windows =
            preprocessing.at("time_reverse_detector_windows_before_kernel").get<bool>();

        int expected_window = 0, expected_terms = 0, expected_bits = 0;
        FStat_GetSpecs(&expected_window, &expected_terms, &expected_bits, nullptr);
        const std::ptrdiff_t profile_nbytes =
            std::ptrdiff_t(expected_terms) * expected_window * sizeof(std::int8_t);

        int fine_bins = 0;
        FStat_GetFineSpecs(nullptr, nullptr, &fine_bins);
        std::set<int> physical_channels;
        std::set<int> chord_channel_ids;
        std::ptrdiff_t expected_weight_offset = 0;
        bool has_chord_channel_id = false;

        for (const auto& row : pilot_profiles.at("profiles")) {
            PilotChannelProfile profile;
            profile.physical_channel = read_integer<int>(row, "physical_channel");
            if (profile.physical_channel < 14 || profile.physical_channel > 36)
                throw std::runtime_error(fmt::format(
                    fmt("cudaPilotProxyDetector: physical channel {:d} is outside ATSC 14-36"),
                    profile.physical_channel));
            if (!physical_channels.insert(profile.physical_channel).second)
                throw std::runtime_error(
                    fmt::format(fmt("cudaPilotProxyDetector: duplicate physical channel {:d}"),
                                profile.physical_channel));
            // Match the chordMetadata coarse_freq namespace; skip rows without an ID.
            if (row.contains("chord_channel_id") && !row.at("chord_channel_id").is_null()) {
                profile.chord_channel_id = read_integer<int>(row, "chord_channel_id");
                if (profile.chord_channel_id < 0)
                    throw std::runtime_error(
                        "cudaPilotProxyDetector: chord_channel_id is negative");
                if (!chord_channel_ids.insert(profile.chord_channel_id).second)
                    throw std::runtime_error(
                        fmt::format(fmt("cudaPilotProxyDetector: duplicate chord_channel_id {:d}"),
                                    profile.chord_channel_id));
                has_chord_channel_id = true;
            }
            profile.pilot_frequency_hz = row.at("pilot_frequency_hz").get<double>();
            if (!std::isfinite(profile.pilot_frequency_hz) || profile.pilot_frequency_hz <= 0.0)
                throw std::runtime_error(
                    "cudaPilotProxyDetector: pilot_frequency_hz must be finite and positive");
            profile.weight_offset_bytes =
                read_integer<std::ptrdiff_t>(row, "weight_bank_offset_bytes");
            profile.weight_nbytes = read_integer<std::ptrdiff_t>(row, "weight_bank_nbytes");
            if (profile.weight_nbytes != profile_nbytes)
                throw std::runtime_error(fmt::format(
                    fmt("cudaPilotProxyDetector: weight_bank_nbytes {:d} does not match the "
                        "compiled kernel contract {:d} for physical channel {:d}"),
                    std::int64_t(profile.weight_nbytes), std::int64_t(profile_nbytes),
                    profile.physical_channel));
            if (profile.weight_offset_bytes < 0
                || profile.weight_offset_bytes > std::ptrdiff_t(weight_bank.size())
                || profile.weight_nbytes
                       > std::ptrdiff_t(weight_bank.size()) - profile.weight_offset_bytes)
                throw std::runtime_error(fmt::format(
                    fmt("cudaPilotProxyDetector: weight bank range at offset {:d} with {:d} bytes "
                        "exceeds "
                        "weights.bin size {:d} for physical channel {:d}"),
                    std::int64_t(profile.weight_offset_bytes), std::int64_t(profile.weight_nbytes),
                    std::uint64_t(weight_bank.size()), profile.physical_channel));
            if (profile.weight_offset_bytes != expected_weight_offset)
                throw std::runtime_error(fmt::format(
                    fmt("cudaPilotProxyDetector: weight offset {:d} is not the expected contiguous "
                        "offset {:d} for physical channel {:d}"),
                    std::int64_t(profile.weight_offset_bytes), std::int64_t(expected_weight_offset),
                    profile.physical_channel));
            expected_weight_offset += profile.weight_nbytes;

            if (!row.contains("fine_calibration") || !row.at("fine_calibration").is_object())
                throw std::runtime_error(fmt::format(
                    fmt("cudaPilotProxyDetector: missing fine_calibration object for physical "
                        "channel {:d}"),
                    profile.physical_channel));
            const auto& calibration = row.at("fine_calibration");
            const std::string calibration_status = calibration.at("status").get<std::string>();
            if (calibration.value("decision_version", "") != "fine_decision_v1")
                throw std::runtime_error(fmt::format(
                    fmt("cudaPilotProxyDetector: unsupported fine decision version for physical "
                        "channel {:d}"),
                    profile.physical_channel));
            if (calibration_status != "pending_campaign" && calibration_status != "calibrated")
                throw std::runtime_error(fmt::format(
                    fmt("cudaPilotProxyDetector: unknown fine calibration status {:s} for physical "
                        "channel {:d}"),
                    calibration_status, profile.physical_channel));
            if (calibration_status == "calibrated") {
                if (!FStat_Supports_FusedFineMask())
                    throw std::runtime_error(
                        "cudaPilotProxyDetector: bundle requests calibrated fine masks but the "
                        "compiled detector core does not support them");
                profile.fine_calibrated = true;
                profile.anchor_bin = read_integer<int>(calibration, "anchor_bin");
                profile.designated_half_width =
                    read_integer<int>(calibration, "designated_half_width");
                profile.cfar_rank = read_integer<int>(calibration, "cfar_rank");
                profile.multiplier_q16 =
                    read_integer<unsigned long long>(calibration, "cfar_multiplier_q16");
                if (profile.anchor_bin < 0 || profile.anchor_bin >= fine_bins)
                    throw std::runtime_error(
                        "cudaPilotProxyDetector: fine anchor_bin is out of range");
                if (profile.designated_half_width < 0
                    || profile.designated_half_width >= fine_bins / 2)
                    throw std::runtime_error(
                        "cudaPilotProxyDetector: fine designated_half_width is out of range");
                if (profile.cfar_rank < 0 || profile.cfar_rank >= fine_bins)
                    throw std::runtime_error(
                        "cudaPilotProxyDetector: fine cfar_rank is out of range");
                if (profile.multiplier_q16 == 0)
                    throw std::runtime_error(
                        "cudaPilotProxyDetector: fine cfar_multiplier_q16 must be positive");
                const auto& words_hex = calibration.at("bulk_mask_words_hex");
                if (!words_hex.is_array() || words_hex.size() != 4)
                    throw std::runtime_error(
                        "cudaPilotProxyDetector: bulk_mask_words_hex must be 4 hex words");
                int bulk_population = 0;
                for (int word = 0; word < 4; ++word) {
                    const std::string encoded = words_hex.at(word).get<std::string>();
                    if (encoded.size() < 3 || encoded.size() > 18 || encoded[0] != '0'
                        || (encoded[1] != 'x' && encoded[1] != 'X'))
                        throw std::runtime_error(
                            "cudaPilotProxyDetector: bulk mask words must be uint64 hex strings");
                    for (std::size_t digit = 2; digit < encoded.size(); ++digit)
                        if (!std::isxdigit(static_cast<unsigned char>(encoded[digit])))
                            throw std::runtime_error(
                                "cudaPilotProxyDetector: invalid digit in bulk mask word");
                    std::size_t parsed = 0;
                    profile.bulk_mask_words[word] = std::stoull(encoded, &parsed, 16);
                    if (parsed != encoded.size())
                        throw std::runtime_error(
                            "cudaPilotProxyDetector: trailing data in bulk mask word");
                    bulk_population += std::bitset<64>(profile.bulk_mask_words[word]).count();
                }
                if (profile.cfar_rank >= bulk_population)
                    throw std::runtime_error(
                        "cudaPilotProxyDetector: fine cfar_rank exceeds the bulk population");
                // Exclude the designated set and one independent bin (two fine bins)
                // on each side of the anchor.
                const int exclusion_half_width = std::max(profile.designated_half_width, 2);
                for (int offset = -exclusion_half_width; offset <= exclusion_half_width; ++offset) {
                    const int bin = (profile.anchor_bin + offset + fine_bins) % fine_bins;
                    if ((profile.bulk_mask_words[bin >> 6] >> (bin & 63)) & 1ULL)
                        throw std::runtime_error(
                            "cudaPilotProxyDetector: designated/guard fine bin is present in "
                            "the null-bulk mask");
                }
            }
            bundle_profiles.push_back(profile);
        }
        if (expected_weight_offset != std::ptrdiff_t(weight_bank.size()))
            throw std::runtime_error(fmt::format(
                fmt("cudaPilotProxyDetector: profile table accounts for {:d} weight bytes but "
                    "weights.bin contains {:d}"),
                std::int64_t(expected_weight_offset), std::uint64_t(weight_bank.size())));
        if (!has_chord_channel_id)
            throw std::runtime_error(
                "cudaPilotProxyDetector: bundle carries no chord_channel_id values");
    }
};

/**
 * @class cudaPilotProxyDetector
 * @brief cudaCommand running the PilotProxy ATSC DTV pilot-tone F-statistic
 * detector (vendored in external/pilotproxy) on CHORD voltage data.
 *
 * @author Dylan Gormley
 *
 * Reads packed [T, F, P, D] voltages and binds local coarse-frequency IDs
 * to pilot_profiles.json and weights.bin from a PilotProxy runtime bundle.
 * Bindings are fixed for the run; changing frequency order stops the stage.
 * Frequencies without a pilot entry produce zero mask and power outputs.
 *
 * Each detector block is packed into stream-major rows. Packing removes the
 * offset-binary encoding and reverses each window when the bundle requests it.
 * CHORD uses 8192 samples per block, or 128 windows of 64 samples per stream.
 * Each bound pilot channel uses the fine CFAR mask and must have a calibrated
 * profile; otherwise the stage stops at the first frame. Leave an uncalibrated
 * channel out of the bundle, or list it in permanent_mask_freq_ids.
 *
 * dtv_mask contains one byte per frequency (1 = reject). dtv_powers contains
 * the target, lower-reference and upper-reference coarse powers as uint64.
 * Each handle is bound to this command's stream with FStat_SetStream at first-frame binding, so
 * the packer, the detector kernels and the mask fold are stream-ordered with the pipeline.
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
 * @gpu_mem Output fine rank support (optional)
 *   @gpu_mem_buffer    @c standard
 *   @gpu_mem_quantity  @c dtv_fine_support
 *   @gpu_mem_type      @c int32
 *   @gpu_mem_dim_name  [@c F][@c S]
 *   @gpu_mem_shape     [@c num_frequencies][@c 2]
 *   @gpu_mem_metadata  @c chordMetadata
 *
 * @conf buffer_depth               Int. GPU frames used for pipelining.
 * @conf num_times                  Int. Voltage samples per upstream GPU
 *                                  frame (ring advance unit; 8192 for CHORD).
 * @conf samples_per_detector_frame Int. Channelized samples per detector
 *                                  block; multiple of the compiled core's
 *                                  detector_window_samples (64 for CHORD).
 *                                  Default 8192: one GPU ring frame giving
 *                                  128 windows per stream for the fine transform.
 * @conf num_frequencies            Int. Local coarse frequencies (F).
 * @conf num_polarizations          Int. Polarizations (2).
 * @conf num_dishes                 Int. Dishes (D).
 * @conf voltage_name               String. Base name of the voltage ring
 *                                  buffer (default "voltage").
 * @conf dtv_mask_name              String. Base name of the mask output
 *                                  buffer (default "dtv_mask").
 * @conf dtv_powers_name            String. Base name of the powers output
 *                                  buffer (default "dtv_powers").
 * @conf dtv_fine_support_name      String. Optional int32[F,2] output (default empty).
 *                                  Columns are rank_valid and n_bulk. (-1,-1) means
 *                                  not evaluated; (0,n) means insufficient rank
 *                                  support; (1,n) means valid rank support. This
 *                                  does not certify input health or calibration.
 * @conf permanent_mask_freq_ids    List of unique receiver coarse_freq IDs for
 *                                  the reject-all limit. These bypass detector
 *                                  evaluation; DtvRfiMask applies their mask.
 * @conf pilot_profiles_path        String. Runtime bundle pilot_profiles.json.
 * @conf weights_path               String. Runtime bundle weights.bin.
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
    std::set<int> permanent_mask_freq_ids;

    // Kotekan buffer names
    const std::string voltage_name;
    const std::string dtv_mask_name;
    const std::string dtv_powers_name;
    const std::string dtv_fine_support_name; // empty preserves the legacy output contract

    // Derived geometry
    const int num_streams;             // P * D
    const int detector_window_samples; // K, from the compiled kernel
    const int windows_per_stream;      // samples_per_detector_frame / K
    const int detector_rows;           // num_streams * windows_per_stream

    // Buffers
    NDArrayRingBuffer<kotekan::int4x2_swapped_withoffset_t, 4> voltage;
    NDArrayBuffer<std::int8_t, 1> dtv_mask;
    NDArrayBuffer<std::uint64_t, 2> dtv_powers;
    std::unique_ptr<NDArrayBuffer<std::int32_t, 2>> dtv_fine_support;
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
        config.get_default<int>(unique_name, "samples_per_detector_frame", 8192)),
    num_frequencies(config.get<int>(unique_name, "num_frequencies")),
    num_polarizations(config.get<int>(unique_name, "num_polarizations")),
    num_dishes(config.get<int>(unique_name, "num_dishes")),
    // Buffer names
    voltage_name(config.get_default<std::string>(unique_name, "voltage_name", "voltage")),
    dtv_mask_name(config.get_default<std::string>(unique_name, "dtv_mask_name", "dtv_mask")),
    dtv_powers_name(config.get_default<std::string>(unique_name, "dtv_powers_name", "dtv_powers")),
    dtv_fine_support_name(
        config.get_default<std::string>(unique_name, "dtv_fine_support_name", "")),
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
    const auto permanent = config.get_default<nlohmann::json>(
        unique_name, "permanent_mask_freq_ids", nlohmann::json::array());
    if (!permanent.is_array())
        FATAL_ERROR("permanent_mask_freq_ids must be an array of receiver frequency IDs");
    for (const auto& value : permanent) {
        if (!value.is_number_integer() || value < 0 || value >= 12288)
            FATAL_ERROR("permanent_mask_freq_ids must contain integer receiver IDs in [0,12288)");
        if (!permanent_mask_freq_ids.insert(value.get<int>()).second)
            FATAL_ERROR("permanent_mask_freq_ids must contain unique receiver IDs in [0,12288)");
    }

    if (samples_per_detector_frame <= 0
        || samples_per_detector_frame % detector_window_samples != 0)
        FATAL_ERROR("samples_per_detector_frame ({:d}) must be a multiple of the compiled "
                    "detector window ({:d}) and positive",
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
    if (windows_per_stream != fine_windows)
        FATAL_ERROR("samples_per_detector_frame ({:d}) / detector_window_samples ({:d}) gives "
                    "{:d} windows/stream, but the compiled core's fine transform needs {:d}; "
                    "set samples_per_detector_frame to {:d}",
                    samples_per_detector_frame, detector_window_samples, windows_per_stream,
                    fine_windows, fine_windows * detector_window_samples);

    voltage.register_consumer();
    dtv_mask.register_producer();
    dtv_powers.register_producer();
    if (!dtv_fine_support_name.empty()) {
        if (!FStat_Supports_FusedFineMaskWithSupport())
            FATAL_ERROR("PilotProxy core does not provide fine rank-support outputs");
        dtv_fine_support = std::make_unique<NDArrayBuffer<std::int32_t, 2>>(
            dtv_fine_support_name, "dtv_fine_support",
            std::array<std::ptrdiff_t, 2>{num_frequencies, 2}, std::array<std::string, 2>{"F", "S"},
            std::array<std::ptrdiff_t, 2>{1, 1}, *this);
        dtv_fine_support->register_producer();
    }

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
    for (const int id : permanent_mask_freq_ids)
        if (std::find(coarse_freq.begin(), coarse_freq.end(), id) == coarse_freq.end())
            FATAL_ERROR("permanent_mask_freq_ids contains receiver ID {:d} absent from this node",
                        id);

    int num_without_id = 0;
    for (int f = 0; f < num_frequencies; ++f) {
        if (permanent_mask_freq_ids.count(coarse_freq[f])) {
            INFO("PilotProxy: receiver ID {:d} uses the reject-all limit; detector not evaluated",
                 coarse_freq[f]);
            continue;
        }
        for (const auto& profile : state.bundle_profiles) {
            if (profile.chord_channel_id < 0) {
                // Counted once below; a bundle without chord ids cannot bind.
                continue;
            }
            if (profile.chord_channel_id != coarse_freq[f])
                continue;
            if (!profile.fine_calibrated)
                FATAL_ERROR("PilotProxy: physical channel {:d} is bound on this node but its "
                            "fine calibration is not deployable; remove it from the bundle or "
                            "list receiver ID {:d} in permanent_mask_freq_ids",
                            profile.physical_channel, coarse_freq[f]);
            State::BoundChannel channel;
            channel.freq_index = f;
            channel.profile = &profile;
            state.bound_channels.push_back(channel);
            INFO("PilotProxy: bound ATSC physical channel {:d} (pilot {:.6f} MHz, "
                 "chord_channel_id {:d}) to local frequency index {:d}",
                 profile.physical_channel, profile.pilot_frequency_hz * 1e-6,
                 profile.chord_channel_id, f);
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

    // One staging region per pilot channel, shared by pipelined instances. The packer and
    // the detector kernels are all issued on this command's stream, so block N+1's packer
    // is ordered after block N's detector work without any extra synchronisation.
    const std::ptrdiff_t region_bytes =
        std::ptrdiff_t(detector_rows) * detector_window_samples * sizeof(std::int8_t);
    std::int8_t* const packed_base = static_cast<std::int8_t*>(device.get_gpu_memory(
        unique_name + "/packed", region_bytes * std::ptrdiff_t(state.bound_channels.size())));

    for (std::size_t i = 0; i < state.bound_channels.size(); ++i) {
        auto& channel = state.bound_channels[i];
        channel.d_packed = packed_base + std::ptrdiff_t(i) * region_bytes;
        channel.fstat_handle = FStat_Create(channel.d_packed, nullptr, detector_rows);
        if (!channel.fstat_handle)
            FATAL_ERROR("FStat_Create failed for physical channel {:d}: {:s}",
                        channel.profile->physical_channel, FStat_LastError());
        FStat_SetStream(channel.fstat_handle, device.getStream(cuda_stream_id));
    }

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
    else if (!in_metadata->has_coarse_freq()
             || in_metadata->get_coarse_freq() != state.bound_coarse_freq)
        FATAL_ERROR("voltage coarse_freq changed or disappeared after first-frame binding");

    // Recover the absolute FPGA sequence from the ring-buffer read offset.
    dtv_mask.set_metadata(in_metadata);
    dtv_powers.set_metadata(in_metadata);
    const std::ptrdiff_t block_start_sample = voltage.get_read_valid().begin();
    for (const auto& metadata : {dtv_mask.get_metadata(), dtv_powers.get_metadata()}) {
        metadata->set_fpga_seq_num(in_metadata->get_fpga_seq_num()
                                   + block_start_sample
                                         * in_metadata->get_time_downsampling_fpga());
        metadata->set_time_downsampling_fpga(in_metadata->get_time_downsampling_fpga()
                                             * samples_per_detector_frame);
    }
    if (dtv_fine_support) {
        dtv_fine_support->set_metadata(in_metadata);
        const auto metadata = dtv_fine_support->get_metadata();
        metadata->set_fpga_seq_num(in_metadata->get_fpga_seq_num()
                                   + block_start_sample
                                         * in_metadata->get_time_downsampling_fpga());
        metadata->set_time_downsampling_fpga(in_metadata->get_time_downsampling_fpga()
                                             * samples_per_detector_frame);
    }

    std::int8_t* const mask_memory = dtv_mask.get_ndarray().data();
    std::uint64_t* const powers_memory = dtv_powers.get_ndarray().data();
    const cudaStream_t kotekan_stream = device.getStream(cuda_stream_id);
    std::int32_t* const support_memory =
        dtv_fine_support ? dtv_fine_support->get_ndarray().data() : nullptr;

    // Non-pilot channels (and DISABLED nodes) report mask 0 / powers 0.
    CHECK_CUDA_ERROR(cudaMemsetAsync(
        mask_memory, 0, std::size_t(num_frequencies) * sizeof(std::int8_t), kotekan_stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(powers_memory, 0,
                                     std::size_t(num_frequencies) * 3 * sizeof(std::uint64_t),
                                     kotekan_stream));
    // (-1,-1) means no fine decision: unbound, permanently masked, or disabled.
    if (support_memory)
        CHECK_CUDA_ERROR(cudaMemsetAsync(support_memory, 0xff,
                                         std::size_t(num_frequencies) * 2 * sizeof(std::int32_t),
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

    for (const auto& channel : state.bound_channels)
        launch_pilotproxy_pack(channel.d_packed,
                               reinterpret_cast<const std::uint8_t*>(voltage_memory), num_dishes,
                               num_polarizations, num_frequencies, channel.freq_index,
                               samples_per_detector_frame, ring_size, ring_pos,
                               detector_window_samples, state.time_reverse_windows, kotekan_stream);

    // Scratch: the fine-power working accumulator required by the fused mask
    // entry, and an int32 mask cell per channel.
    const std::ptrdiff_t num_bound = std::ptrdiff_t(state.bound_channels.size());
    unsigned long long* const fine_power_scratch =
        static_cast<unsigned long long*>(device.get_gpu_memory(unique_name + "/fine_power_scratch",
                                                               3 * 256 * sizeof(std::uint64_t)));
    int* const mask_i32_scratch = static_cast<int*>(
        device.get_gpu_memory(unique_name + "/mask_i32_scratch", num_bound * sizeof(int)));

    for (std::ptrdiff_t i = 0; i < num_bound; ++i) {
        const auto& channel = state.bound_channels[i];
        const auto& profile = *channel.profile;
        const InputType* const weights =
            reinterpret_cast<const InputType*>(state.weights_for(profile));
        unsigned long long* const channel_powers =
            reinterpret_cast<unsigned long long*>(powers_memory + 3 * channel.freq_index);
        // Compute the fine CFAR decision and save coarse powers for validation.
        if (support_memory) {
            // Rank support is not packet-loss or feed-health qualification.
            FStat_Compute_FusedFineMaskWithSupport_U64(
                channel.fstat_handle, weights, profile.anchor_bin, profile.designated_half_width,
                profile.bulk_mask_words.data(), profile.cfar_rank, profile.multiplier_q16,
                fine_power_scratch, mask_i32_scratch + i, support_memory + 2 * channel.freq_index,
                support_memory + 2 * channel.freq_index + 1, channel_powers, nullptr);
        } else {
            FStat_Compute_FusedFineMask_U64(
                channel.fstat_handle, weights, profile.anchor_bin, profile.designated_half_width,
                profile.bulk_mask_words.data(), profile.cfar_rank, profile.multiplier_q16,
                fine_power_scratch, mask_i32_scratch + i, channel_powers, nullptr);
        }
        if (const char* error = FStat_LastError(); error != nullptr && error[0] != '\0')
            FATAL_ERROR("PilotProxy fine detector failed for physical channel {:d}: {:s}",
                        profile.physical_channel, error);
    }
    // Fold the int32 mask cells into the int8 mask product. The values are
    // 0/1, so copying the little-endian low byte is exact (both CUDA targets
    // are little-endian).
    for (std::ptrdiff_t i = 0; i < num_bound; ++i) {
        const auto& channel = state.bound_channels[i];
        CHECK_CUDA_ERROR(cudaMemcpyAsync(mask_memory + channel.freq_index, mask_i32_scratch + i, 1,
                                         cudaMemcpyDeviceToDevice, kotekan_stream));
    }

    return record_end_event();
}

void cudaPilotProxyDetector::finalize_frame() {
    voltage.finish_read();
    cudaCommand::finalize_frame();
}
