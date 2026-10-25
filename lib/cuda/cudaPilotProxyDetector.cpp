#include "Config.hpp"                     // for Config
#include "DataType.hpp"                   // for int4x2_swapped_withoffset_t
#include "NDArray.hpp"                    // for NDArray
#include "NDArrayBuffer.hpp"              // for NDArrayBuffer
#include "NDArrayRingBuffer.hpp"          // for NDArrayRingBuffer, read_descriptor_t
#include "bufferContainer.hpp"            // for bufferContainer
#include "chordMetadata.hpp"              // for chordMetadata
#include "cudaCommand.hpp"                // for cudaCommand, cudaCommandState, ...
#include "cudaDeviceInterface.hpp"        // for cudaDeviceInterface
#include "cudaPilotProxyPacker.hpp"       // for launch_pilotproxy_pack
#include "cudaUtils.hpp"                  // for CHECK_CUDA_ERROR
#include "gpuCommand.hpp"                 // for gpuCommandType
#include "kotekanLogging.hpp"             // for FATAL_ERROR, INFO, DEBUG
#include "pilotproxy/pilotproxy_core.hpp" // for PilotProxyCore, pilotproxy_core

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
#include <filesystem>         // for path
#include <fstream>            // for ifstream
#include <functional>         // for function
#include <initializer_list>   // for initializer_list
#include <iterator>           // for istreambuf_iterator
#include <limits>             // for numeric_limits
#include <map>                // for map
#include <memory>             // for shared_ptr, static_pointer_cast
#include <set>                // for set
#include <sstream>            // for istringstream
#include <string>             // for string, stoull
#include <vector>             // for vector

#ifdef WITH_SSL
#include <openssl/evp.h> // for EVP_Digest, EVP_sha256
#endif

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
        cudaCommandState(config, unique_name, host_buffers, device),
        core(pilotproxy_core(config.get<int>(unique_name, "detector_window_samples"))) {
        if (!core)
            FATAL_ERROR("cudaPilotProxyDetector: detector_window_samples must be 64 or 128");
        const std::string pilot_profiles_path =
            config.get<std::string>(unique_name, "pilot_profiles_path");
        const std::string weights_path = config.get<std::string>(unique_name, "weights_path");
        const auto channel_namespace = config.get_default<std::string>(
            unique_name, "receiver_channel_id_namespace", "chord_freq_id");
        if (channel_namespace != "chord_freq_id" && channel_namespace != "chime_freq_id")
            FATAL_ERROR("cudaPilotProxyDetector: unsupported receiver channel namespace");
        load_bundle(pilot_profiles_path, weights_path, channel_namespace);

        // Create handles before the pipeline runs: PilotProxy_Create queries device properties
        // and allocates device scratch. Channels bind at the first frame and reuse the
        // staging region and scratch in command-stream order.
        int window = 0, fine_windows = 0, weight_terms = 0, fine_bins = 0;
        core->GetSpecs(&window, &weight_terms, nullptr, nullptr);
        core->GetFineSpecs(&fine_windows, nullptr, &fine_bins);
        const int detector_rows = config.get<int>(unique_name, "num_polarizations")
                                  * config.get<int>(unique_name, "num_dishes") * fine_windows;
        d_packed = static_cast<std::int8_t*>(device.get_gpu_memory(
            unique_name + "/packed", std::size_t(detector_rows) * window * sizeof(std::int8_t)));
        fine_power_scratch = static_cast<unsigned long long*>(
            device.get_gpu_memory(unique_name + "/fine_power_scratch",
                                  std::size_t(weight_terms) * fine_bins * sizeof(std::uint64_t)));
        mask_i32_scratch = static_cast<int*>(
            device.get_gpu_memory(unique_name + "/mask_i32_scratch", sizeof(int)));
        for (auto& profile : bundle_profiles) {
            if (profile.receiver_channel_id < 0 || !profile.fine_calibrated)
                continue;
            profile.detector_handle = core->Create(d_packed, detector_rows);
            if (!profile.detector_handle)
                FATAL_ERROR("cudaPilotProxyDetector: PilotProxy_Create failed "
                            "for physical channel {:d}: "
                            "{:s}",
                            profile.physical_channel, core->LastError());
        }
    }

    ~cudaPilotProxyDetectorState() {
        for (auto& profile : bundle_profiles)
            if (profile.detector_handle)
                core->Destroy(profile.detector_handle);
    }

    /// One pilot channel row of pilot_profiles.json (runtime bundle).
    struct PilotChannelProfile {
        int physical_channel = -1;
        int receiver_channel_id = -1;
        double pilot_frequency_hz = 0.0;
        std::ptrdiff_t weight_offset_bytes = 0;
        std::ptrdiff_t weight_nbytes = 0;
        // fine_calibration for the mask epilogue; usable only
        // when status == "calibrated".
        bool fine_calibrated = false;
        bool archive_review_only = false;
        int anchor_bin = 0;
        int designated_half_width = 0;
        int cfar_rank = 0;
        unsigned long long multiplier_q16 = 0;
        std::array<unsigned long long, 4> bulk_mask_words{{0, 0, 0, 0}};
        void* detector_handle =
            nullptr; // PilotProxy_Create handle (d_in = d_packed), if calibrated
    };

    /// A bundle profile bound to a local frequency index at first frame.
    struct BoundChannel {
        int freq_index = -1; // local F index on this node
        const PilotChannelProfile* profile = nullptr;
    };

    enum class RunState { wait_for_first_frame, running, disabled };

    const PilotProxyCore* const core; // core built for detector_window_samples

    // Parsed bundle (immutable after construction)
    std::vector<PilotChannelProfile> bundle_profiles;
    std::vector<std::int8_t> weight_bank; // full weights.bin contents (host)
    bool time_reverse_windows = false;    // bundle input_preprocessing flag

    // GPU staging and scratch shared by all channels and instances
    std::int8_t* d_packed = nullptr;
    unsigned long long* fine_power_scratch = nullptr;
    int* mask_i32_scratch = nullptr;

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
    T read_integer(const json& object, const char* field) {
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
            FATAL_ERROR("cudaPilotProxyDetector: {:s} must be an integer in the supported range",
                        field);
        return value.get<T>();
    }

    void load_bundle(const std::string& pilot_profiles_path, const std::string& weights_path,
                     const std::string& channel_namespace) {
        std::ifstream profiles_file(pilot_profiles_path);
        if (!profiles_file)
            FATAL_ERROR("cudaPilotProxyDetector: cannot open pilot_profiles_path {:s}",
                        pilot_profiles_path);
        json pilot_profiles;
        profiles_file >> pilot_profiles;

        if (pilot_profiles.value("schema_version", "") != "pilotproxy_runtime_pilot_profiles_v1")
            FATAL_ERROR(
                "cudaPilotProxyDetector: unsupported or missing pilot_profiles schema_version");
        if (!pilot_profiles.contains("profiles") || !pilot_profiles.at("profiles").is_array()
            || pilot_profiles.at("profiles").empty())
            FATAL_ERROR("cudaPilotProxyDetector: pilot_profiles must be a non-empty array");

        // Legacy CHORD bundles contain only chord_channel_id. Other namespaces require
        // explicit bundle and row identities so an equal integer cannot bind the wrong sky bin.
        const bool explicit_namespace = pilot_profiles.contains("receiver_channel_id_namespace");
        if ((!explicit_namespace && channel_namespace != "chord_freq_id")
            || (explicit_namespace
                && pilot_profiles.at("receiver_channel_id_namespace") != channel_namespace))
            FATAL_ERROR("cudaPilotProxyDetector: receiver channel namespace mismatch");
        const char* alias =
            channel_namespace == "chord_freq_id" ? "chord_channel_id" : "chime_channel_id";

        std::ifstream weights_file(weights_path, std::ios::binary);
        if (!weights_file)
            FATAL_ERROR("cudaPilotProxyDetector: cannot open weights_path {:s}", weights_path);
        weight_bank.assign(std::istreambuf_iterator<char>(weights_file),
                           std::istreambuf_iterator<char>());
        if (weight_bank.empty())
            FATAL_ERROR("cudaPilotProxyDetector: weights.bin is empty");

        // Check the digests as PilotProxy's bundle validator does: sha256sums.txt must list the
        // four bundle files with their SHA-256, and weights_sha256 in pilot_profiles.json and
        // weights.manifest.json must match the loaded weights.
        const auto bundle_dir = std::filesystem::path(pilot_profiles_path).parent_path();
#ifdef WITH_SSL
        const auto sha256 = [](const void* data, std::size_t size) {
            unsigned char digest[EVP_MAX_MD_SIZE];
            unsigned int length = 0;
            EVP_Digest(data, size, digest, &length, EVP_sha256(), nullptr);
            std::string hex;
            for (unsigned int i = 0; i < length; ++i)
                hex += fmt::format(fmt("{:02x}"), digest[i]);
            return hex;
        };
        const std::string sums_path = (bundle_dir / "sha256sums.txt").string();
        std::ifstream sums_file(sums_path);
        if (!sums_file)
            FATAL_ERROR("cudaPilotProxyDetector: cannot open {:s}", sums_path);
        std::map<std::string, std::string> listed;
        for (std::string line; std::getline(sums_file, line);) {
            std::istringstream fields(line);
            std::string digest, name;
            if (!(fields >> digest))
                continue;
            std::getline(fields >> std::ws, name);
            name.erase(name.find_last_not_of(" \t\r") + 1);
            if (name.empty())
                FATAL_ERROR("cudaPilotProxyDetector: invalid line in {:s}: {:s}", sums_path, line);
            listed[name] = digest;
        }
        const std::string weights_digest = sha256(weight_bank.data(), weight_bank.size());
        for (const std::string name : {"detector_contract.json", "pilot_profiles.json",
                                       "weights.bin", "weights.manifest.json"}) {
            if (!listed.count(name))
                FATAL_ERROR("cudaPilotProxyDetector: {:s} is missing from {:s}", name, sums_path);
            // The two files the stage loads are checked at their configured paths.
            std::string path = (bundle_dir / name).string(), actual = weights_digest;
            if (name == "pilot_profiles.json")
                path = pilot_profiles_path;
            if (name == "weights.bin")
                path = weights_path;
            else {
                std::ifstream file(path, std::ios::binary);
                if (!file)
                    FATAL_ERROR("cudaPilotProxyDetector: {:s} lists {:s}, which cannot be read",
                                sums_path, path);
                const std::string bytes(std::istreambuf_iterator<char>(file), {});
                actual = sha256(bytes.data(), bytes.size());
            }
            if (actual != listed[name])
                FATAL_ERROR("cudaPilotProxyDetector: sha256 mismatch for {:s}: the file "
                            "has {:s}, sha256sums.txt lists {:s}",
                            path, actual, listed[name]);
        }
        const auto check_weights_sha256 = [&](const json& payload, const char* name) {
            const json stated = payload.value("weights_sha256", json());
            if (stated != json(weights_digest))
                FATAL_ERROR("cudaPilotProxyDetector: weights_sha256 in {:s} is {:s}, but "
                            "{:s} has {:s}",
                            name, stated.dump(), weights_path, weights_digest);
        };
        check_weights_sha256(pilot_profiles, "pilot_profiles.json");
        const std::string manifest_path = (bundle_dir / "weights.manifest.json").string();
        const json manifest = json::parse(std::ifstream(manifest_path), nullptr, false);
        if (!manifest.is_object())
            FATAL_ERROR("cudaPilotProxyDetector: cannot parse {:s}", manifest_path);
        check_weights_sha256(manifest, "weights.manifest.json");
#else
        FATAL_ERROR("cudaPilotProxyDetector: Kotekan was built without OpenSSL, so the "
                    "runtime bundle in {:s} cannot be checked",
                    bundle_dir.string());
#endif

        // Apply the receiver bundle's per-window time reversal contract.
        const auto& preprocessing = pilot_profiles.at("input_preprocessing");
        time_reverse_windows =
            preprocessing.at("time_reverse_detector_windows_before_kernel").get<bool>();

        int expected_window = 0, expected_terms = 0, expected_bits = 0;
        core->GetSpecs(&expected_window, &expected_terms, &expected_bits, nullptr);
        const std::ptrdiff_t profile_nbytes =
            std::ptrdiff_t(expected_terms) * expected_window * sizeof(std::int8_t);

        int fine_bins = 0;
        core->GetFineSpecs(nullptr, nullptr, &fine_bins);
        std::set<int> physical_channels;
        std::set<int> receiver_channel_ids;
        std::ptrdiff_t expected_weight_offset = 0;
        bool has_receiver_channel_id = false;

        for (const auto& row : pilot_profiles.at("profiles")) {
            PilotChannelProfile profile;
            const auto require = [&](const json& object,
                                     std::initializer_list<const char*> fields) {
                for (const char* field : fields)
                    if (!object.contains(field))
                        FATAL_ERROR("cudaPilotProxyDetector: pilot_profiles entry for "
                                    "physical channel {:d} has no {:s}",
                                    profile.physical_channel, field);
            };
            if (!row.contains("physical_channel"))
                FATAL_ERROR("cudaPilotProxyDetector: pilot_profiles row "
                            "{:d} has no physical_channel",
                            bundle_profiles.size());
            profile.physical_channel = read_integer<int>(row, "physical_channel");
            if (profile.physical_channel < 14 || profile.physical_channel > 36)
                FATAL_ERROR("cudaPilotProxyDetector: physical channel {:d} is outside ATSC 14-36",
                            profile.physical_channel);
            if (!physical_channels.insert(profile.physical_channel).second)
                FATAL_ERROR("cudaPilotProxyDetector: duplicate physical channel {:d}",
                            profile.physical_channel);
            if (!explicit_namespace
                && (row.contains("receiver_channel_id_namespace")
                    || row.contains("receiver_channel_id")))
                FATAL_ERROR("cudaPilotProxyDetector: missing bundle receiver namespace");
            if (explicit_namespace
                && (!row.contains("receiver_channel_id_namespace")
                    || row.at("receiver_channel_id_namespace") != channel_namespace
                    || !row.contains("receiver_channel_id")))
                FATAL_ERROR("cudaPilotProxyDetector: receiver row namespace mismatch");
            const char* id_field = explicit_namespace ? "receiver_channel_id" : alias;
            if (row.contains(alias) && !row.at(alias).is_null()) {
                const int alias_id = read_integer<int>(row, alias);
                if (alias_id < 0)
                    FATAL_ERROR("cudaPilotProxyDetector: {:s} is negative", alias);
            }
            if (explicit_namespace && row.contains(alias) && row.at(alias) != row.at(id_field))
                FATAL_ERROR("cudaPilotProxyDetector: {:s} disagrees with receiver_channel_id",
                            alias);
            if (row.contains(id_field) && !row.at(id_field).is_null()) {
                profile.receiver_channel_id = read_integer<int>(row, id_field);
                if (profile.receiver_channel_id < 0)
                    FATAL_ERROR("cudaPilotProxyDetector: receiver_channel_id is negative");
                if (!receiver_channel_ids.insert(profile.receiver_channel_id).second)
                    FATAL_ERROR("cudaPilotProxyDetector: duplicate {:s} {:d}", alias,
                                profile.receiver_channel_id);
                has_receiver_channel_id = true;
            }
            require(row, {"pilot_frequency_hz", "weight_bank_offset_bytes", "weight_bank_nbytes"});
            profile.pilot_frequency_hz = row.at("pilot_frequency_hz").get<double>();
            if (!std::isfinite(profile.pilot_frequency_hz) || profile.pilot_frequency_hz <= 0.0)
                FATAL_ERROR(
                    "cudaPilotProxyDetector: pilot_frequency_hz must be finite and positive");
            profile.weight_offset_bytes =
                read_integer<std::ptrdiff_t>(row, "weight_bank_offset_bytes");
            profile.weight_nbytes = read_integer<std::ptrdiff_t>(row, "weight_bank_nbytes");
            if (profile.weight_nbytes != profile_nbytes)
                FATAL_ERROR("cudaPilotProxyDetector: weight_bank_nbytes {:d} does not match the "
                            "compiled kernel contract {:d} for physical channel {:d}",
                            std::int64_t(profile.weight_nbytes), std::int64_t(profile_nbytes),
                            profile.physical_channel);
            if (profile.weight_offset_bytes < 0
                || profile.weight_offset_bytes > std::ptrdiff_t(weight_bank.size())
                || profile.weight_nbytes
                       > std::ptrdiff_t(weight_bank.size()) - profile.weight_offset_bytes)
                FATAL_ERROR("cudaPilotProxyDetector: weight bank range at offset "
                            "{:d} with {:d} bytes "
                            "exceeds "
                            "weights.bin size {:d} for physical channel {:d}",
                            std::int64_t(profile.weight_offset_bytes),
                            std::int64_t(profile.weight_nbytes), std::uint64_t(weight_bank.size()),
                            profile.physical_channel);
            if (profile.weight_offset_bytes != expected_weight_offset)
                FATAL_ERROR("cudaPilotProxyDetector: weight offset {:d} is "
                            "not the expected contiguous "
                            "offset {:d} for physical channel {:d}",
                            std::int64_t(profile.weight_offset_bytes),
                            std::int64_t(expected_weight_offset), profile.physical_channel);
            expected_weight_offset += profile.weight_nbytes;

            if (!row.contains("fine_calibration") || !row.at("fine_calibration").is_object())
                FATAL_ERROR("cudaPilotProxyDetector: missing fine_calibration object for physical "
                            "channel {:d}",
                            profile.physical_channel);
            const auto& calibration = row.at("fine_calibration");
            require(calibration, {"status"});
            const std::string calibration_status = calibration.at("status").get<std::string>();
            if (calibration.contains("provenance") && calibration.at("provenance").is_object()) {
                const auto& provenance = calibration.at("provenance");
                if (provenance.contains("threshold_selection")) {
                    const auto& selection = provenance.at("threshold_selection");
                    profile.archive_review_only =
                        selection.is_object()
                        && selection.value("qualification", "") == "archive_review_only";
                }
            }
            if (calibration.value("decision_version", "") != "fine_decision_v1")
                FATAL_ERROR("cudaPilotProxyDetector: unsupported fine "
                            "decision version for physical "
                            "channel {:d}",
                            profile.physical_channel);
            if (calibration_status != "pending_campaign" && calibration_status != "calibrated")
                FATAL_ERROR("cudaPilotProxyDetector: unknown fine "
                            "calibration status {:s} for physical "
                            "channel {:d}",
                            calibration_status, profile.physical_channel);
            if (calibration_status == "calibrated") {
                profile.fine_calibrated = true;
                require(calibration, {"anchor_bin", "designated_half_width", "cfar_rank",
                                      "cfar_multiplier_q16", "bulk_mask_words_hex"});
                profile.anchor_bin = read_integer<int>(calibration, "anchor_bin");
                profile.designated_half_width =
                    read_integer<int>(calibration, "designated_half_width");
                profile.cfar_rank = read_integer<int>(calibration, "cfar_rank");
                profile.multiplier_q16 =
                    read_integer<unsigned long long>(calibration, "cfar_multiplier_q16");
                if (profile.anchor_bin < 0 || profile.anchor_bin >= fine_bins)
                    FATAL_ERROR("cudaPilotProxyDetector: fine anchor_bin is out of range");
                if (profile.designated_half_width < 0
                    || profile.designated_half_width >= fine_bins / 2)
                    FATAL_ERROR(
                        "cudaPilotProxyDetector: fine designated_half_width is out of range");
                if (profile.cfar_rank < 0 || profile.cfar_rank >= fine_bins)
                    FATAL_ERROR("cudaPilotProxyDetector: fine cfar_rank is out of range");
                if (profile.multiplier_q16 == 0)
                    FATAL_ERROR(
                        "cudaPilotProxyDetector: fine cfar_multiplier_q16 must be positive");
                const auto& words_hex = calibration.at("bulk_mask_words_hex");
                if (!words_hex.is_array() || words_hex.size() != 4)
                    FATAL_ERROR("cudaPilotProxyDetector: bulk_mask_words_hex must be 4 hex words");
                int bulk_population = 0;
                for (int word = 0; word < 4; ++word) {
                    const std::string encoded = words_hex.at(word).get<std::string>();
                    if (encoded.size() < 3 || encoded.size() > 18 || encoded[0] != '0'
                        || (encoded[1] != 'x' && encoded[1] != 'X'))
                        FATAL_ERROR(
                            "cudaPilotProxyDetector: bulk mask words must be uint64 hex strings");
                    for (std::size_t digit = 2; digit < encoded.size(); ++digit)
                        if (!std::isxdigit(static_cast<unsigned char>(encoded[digit])))
                            FATAL_ERROR("cudaPilotProxyDetector: invalid digit in bulk mask word");
                    std::size_t parsed = 0;
                    profile.bulk_mask_words[word] = std::stoull(encoded, &parsed, 16);
                    if (parsed != encoded.size())
                        FATAL_ERROR("cudaPilotProxyDetector: trailing data in bulk mask word");
                    bulk_population += std::bitset<64>(profile.bulk_mask_words[word]).count();
                }
                if (profile.cfar_rank >= bulk_population)
                    FATAL_ERROR(
                        "cudaPilotProxyDetector: fine cfar_rank exceeds the bulk population");
                // Exclude the designated set and one independent bin (two fine bins)
                // on each side of the anchor.
                const int exclusion_half_width = std::max(profile.designated_half_width, 2);
                for (int offset = -exclusion_half_width; offset <= exclusion_half_width; ++offset) {
                    const int bin = (profile.anchor_bin + offset + fine_bins) % fine_bins;
                    if ((profile.bulk_mask_words[bin >> 6] >> (bin & 63)) & 1ULL)
                        FATAL_ERROR(
                            "cudaPilotProxyDetector: designated/guard fine bin is present in "
                            "the null-bulk mask");
                }
            }
            bundle_profiles.push_back(profile);
        }
        if (expected_weight_offset != std::ptrdiff_t(weight_bank.size()))
            FATAL_ERROR("cudaPilotProxyDetector: profile table accounts for {:d} weight bytes but "
                        "weights.bin contains {:d}",
                        std::int64_t(expected_weight_offset), std::uint64_t(weight_bank.size()));
        if (!has_receiver_channel_id)
            FATAL_ERROR("cudaPilotProxyDetector: bundle carries no receiver channel IDs");
    }
};

/**
 * @class cudaPilotProxyDetector
 * @brief Detect ATSC DTV pilots in packed receiver voltages with PilotProxy.
 *
 * The detector compares pilot power with a ranked local reference. Its core
 * is vendored in external/pilotproxy.
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
 * A block is 128 windows of detector_window_samples per stream: 8192 samples on CHORD,
 * or 16384 on CHIME.
 * Each bound pilot channel uses the fine CFAR mask and must have a calibrated
 * profile; otherwise the stage stops at the first frame. Leave an uncalibrated
 * channel out of the bundle, or list it in permanent_mask_freq_ids.
 *
 * dtv_mask contains one byte per frequency (1 = reject). dtv_powers contains
 * the target, lower-reference and upper-reference frame powers as uint64; each
 * is the sum of that term's 256 fine powers.
 * The detector handles are created with the command; each bound handle is set to this
 * command's stream with SetStream at first-frame binding, so the packer, the detector
 * kernels and the mask fold are stream-ordered with the pipeline.
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
 * @gpu_mem Output DTV frame powers
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
 *                                  frame (ring advance unit; 8192 for CHORD, 16384 for CHIME).
 * @conf detector_window_samples    Int. Samples per detector window: 64 for CHORD, 128 for
 *                                  CHIME. Must match the bundle.
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
 * @conf pilot_profiles_path        String. Runtime bundle pilot_profiles.json. The
 *                                  bundle's sha256sums.txt, weights.manifest.json and
 *                                  detector_contract.json must be next to it; the
 *                                  digests are checked at load.
 * @conf weights_path               String. Runtime bundle weights.bin.
 * @conf receiver_channel_id_namespace String. coarse_freq namespace: chord_freq_id (default)
 *                                  or chime_freq_id. Must match the bundle.
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
    const int num_frequencies;
    const int num_polarizations;
    const int num_dishes;
    std::set<int> permanent_mask_freq_ids;

    // Kotekan buffer names
    const std::string voltage_name;
    const std::string dtv_mask_name;
    const std::string dtv_powers_name;
    const std::string dtv_fine_support_name; // empty: no fine-support output

    // Block geometry
    const int detector_window_samples;    // K
    const int samples_per_detector_frame; // K * fine windows (8192 on CHORD)

    // Buffers
    NDArrayRingBuffer<kotekan::int4x2_swapped_withoffset_t, 4> voltage;
    NDArrayBuffer<std::int8_t, 1> dtv_mask;
    NDArrayBuffer<std::uint64_t, 2> dtv_powers;
    std::unique_ptr<NDArrayBuffer<std::int32_t, 2>> dtv_fine_support;
};

REGISTER_CUDA_COMMAND_WITH_STATE(cudaPilotProxyDetector, cudaPilotProxyDetectorState);

static int query_window_samples(const PilotProxyCore& core) {
    int window = 0;
    core.GetSpecs(&window, nullptr, nullptr, nullptr);
    return window;
}

static int query_fine_windows(const PilotProxyCore& core) {
    int fine_windows = 0;
    core.GetFineSpecs(&fine_windows, nullptr, nullptr);
    return fine_windows;
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
    num_frequencies(config.get<int>(unique_name, "num_frequencies")),
    num_polarizations(config.get<int>(unique_name, "num_polarizations")),
    num_dishes(config.get<int>(unique_name, "num_dishes")),
    // Buffer names
    voltage_name(config.get_default<std::string>(unique_name, "voltage_name", "voltage")),
    dtv_mask_name(config.get_default<std::string>(unique_name, "dtv_mask_name", "dtv_mask")),
    dtv_powers_name(config.get_default<std::string>(unique_name, "dtv_powers_name", "dtv_powers")),
    dtv_fine_support_name(
        config.get_default<std::string>(unique_name, "dtv_fine_support_name", "")),
    // Block geometry
    detector_window_samples(query_window_samples(*detector_state->core)),
    samples_per_detector_frame(detector_window_samples * query_fine_windows(*detector_state->core)),
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
        if (!value.is_number_integer() || value < 0 || value >= CHORD_META_MAX_FREQ)
            FATAL_ERROR("permanent_mask_freq_ids must contain integer receiver IDs in [0,{:d})",
                        CHORD_META_MAX_FREQ);
        if (!permanent_mask_freq_ids.insert(value.get<int>()).second)
            FATAL_ERROR("permanent_mask_freq_ids must contain unique receiver IDs");
    }

    // The packer wraps ring indices with a power-of-two bitmask, so the ring
    // size (the slowest, "T" dimension) must be a power of two.
    const std::ptrdiff_t ring_size = voltage.get_ndarray().extent(0);
    if (ring_size <= 0 || (ring_size & (ring_size - 1)) != 0)
        FATAL_ERROR("voltage ring size {:d} is not a power of two", ring_size);
    if (ring_size < samples_per_detector_frame)
        FATAL_ERROR("voltage ring size {:d} is smaller than one detector block ({:d})", ring_size,
                    samples_per_detector_frame);

    voltage.register_consumer();
    dtv_mask.register_producer();
    dtv_powers.register_producer();
    if (!dtv_fine_support_name.empty()) {
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
            if (profile.receiver_channel_id < 0) {
                // Counted once below; a bundle without receiver IDs cannot bind.
                continue;
            }
            if (profile.receiver_channel_id != coarse_freq[f])
                continue;
            if (profile.archive_review_only)
                FATAL_ERROR("PilotProxy: receiver ID {:d} has archive-review-only thresholds; "
                            "active detector evaluation requires separately qualified thresholds",
                            coarse_freq[f]);
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
                 "receiver_channel_id {:d}) to local frequency index {:d}",
                 profile.physical_channel, profile.pilot_frequency_hz * 1e-6,
                 profile.receiver_channel_id, f);
        }
    }
    for (const auto& profile : state.bundle_profiles)
        if (profile.receiver_channel_id < 0)
            ++num_without_id;
    if (num_without_id > 0)
        INFO("PilotProxy: {:d} bundle profile(s) carry no receiver_channel_id and cannot bind on "
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

    for (const auto& channel : state.bound_channels)
        state.core->SetStream(channel.profile->detector_handle, device.getStream(cuda_stream_id));

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

    // Pack and detect one channel at a time; the staging region and the
    // scratch are reused in stream order.
    for (const auto& channel : state.bound_channels) {
        const auto& profile = *channel.profile;
        launch_pilotproxy_pack(state.d_packed,
                               reinterpret_cast<const std::uint8_t*>(voltage_memory), num_dishes,
                               num_polarizations, num_frequencies, channel.freq_index,
                               samples_per_detector_frame, ring_size, ring_pos,
                               detector_window_samples, state.time_reverse_windows, kotekan_stream);
        const std::int8_t* const weights = state.weights_for(profile);
        unsigned long long* const channel_powers =
            reinterpret_cast<unsigned long long*>(powers_memory + 3 * channel.freq_index);
        // Compute the fine CFAR decision and save the frame powers for validation.
        if (support_memory) {
            // Rank support is not packet-loss or feed-health qualification.
            state.core->Compute_FusedFineMaskWithSupport_U64(
                profile.detector_handle, weights, profile.anchor_bin, profile.designated_half_width,
                profile.bulk_mask_words.data(), profile.cfar_rank, profile.multiplier_q16,
                state.fine_power_scratch, state.mask_i32_scratch,
                support_memory + 2 * channel.freq_index,
                support_memory + 2 * channel.freq_index + 1, channel_powers, nullptr);
        } else {
            state.core->Compute_FusedFineMask_U64(
                profile.detector_handle, weights, profile.anchor_bin, profile.designated_half_width,
                profile.bulk_mask_words.data(), profile.cfar_rank, profile.multiplier_q16,
                state.fine_power_scratch, state.mask_i32_scratch, channel_powers, nullptr);
        }
        if (const char* error = state.core->LastError(); error != nullptr && error[0] != '\0')
            FATAL_ERROR("PilotProxy fine detector failed for physical channel {:d}: {:s}",
                        profile.physical_channel, error);
        // Fold the int32 mask cell into the int8 mask product. The value is
        // 0/1, so copying the little-endian low byte is exact (both CUDA targets
        // are little-endian).
        CHECK_CUDA_ERROR(cudaMemcpyAsync(mask_memory + channel.freq_index, state.mask_i32_scratch,
                                         1, cudaMemcpyDeviceToDevice, kotekan_stream));
    }

    return record_end_event();
}

void cudaPilotProxyDetector::finalize_frame() {
    voltage.finish_read();
    cudaCommand::finalize_frame();
}
