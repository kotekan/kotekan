#include "Config.hpp"          // for Config
#include "N2Util.hpp"          // for frameID
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "Telescope.hpp"       // for Telescope
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata, get_chord_metadata
#include "configUpdater.hpp"   // for configUpdater
#include "div.hpp"             // for div_noremainder
#include "kotekanLogging.hpp"  // for FATAL_ERROR, DEBUG, INFO
#include "restServer.hpp"      // for restServer, connectionInstance

#include "fmt.hpp"          // for compile_string_to_view
#include "jsonMetadata.hpp" // for MAX_NUM_RFI_THRESHOLDS

#include <assert.h>   // for assert
#include <functional> // for bind, function, placeholders
#include <memory>     // for shared_ptr, __shared_ptr_access
#include <vector>     // for vector


using namespace std::placeholders;

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::connectionInstance;
using kotekan::div_noremainder;
using kotekan::restServer;
using kotekan::Stage;
using N2::frameID;

/**
 * @class RfiFrameMask
 * @brief Compute the mask for second stage RFI excision.
 *
 * num_integrations := samples_per_data_set / sub_integration_ntime
 * num_freq := num_local_freq
 *
 * @par Buffers
 * @buffer  in_buf      SK buffer
 *         @buffer_format   any
 *         @buffer_metadata chordMetadata
 *
 * @buffer  out_buf         Array of bools masking frames.
 *      @buffer_format      NDArray uint8 [num_integrations, num_freq]
 *      @buffer_metadata    chordMetadata
 *
 * @conf    num_local_freq              int64_t Number of frequencies in
 *                                          buffers, required.
 * @conf    samples_per_data_set        int64_t Total number of time samples covered by each
 *                                          input frame. nt_outer in n2k.
 * @conf    sub_integration_ntime       int64_t Number of time samples integrated in each
 *                                          entry in correlation and counts buffers.  n2_inner
 *                                          in n2k.
 *
 * @author  Geoff Ryan
 */
class RfiFrameMask : public kotekan::Stage {
public:
    RfiFrameMask(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);
    ~RfiFrameMask();

    /**
     * @brief The main thread function for RfiFrameMask.
     *
     * This function is responsible for the main logic of the RfiFrameMask class.
     */
    void main_thread() override;

protected:
    bool receive_rfi_excision_enabled(nlohmann::json& json);
    bool receive_rfi_excision_thresholds(nlohmann::json& json);
    void send_rfi_excision_enabled(connectionInstance& conn);
    void send_rfi_excision_thresholds(connectionInstance& conn);

private:
    void update_enabled_and_thresholds(int64_t seq_num);
    void check_sktilde_meta(std::shared_ptr<chordMetadata> meta);

    // Buffers to read/write
    Buffer* in_buf;  /// Buffer containing SK data
    Buffer* out_buf; /// Second stage RFI boolean mask for frames.

    // Parameters saved from the config files
    const int64_t _num_local_freq;
    const int64_t _samples_per_data_set;
    const int64_t _sub_integration_ntime;
    const int64_t _rfi_downsampling_factor;
    const int64_t _rfi_num_times;
    const int64_t _num_integrations;
    const int64_t _num_rfi_per_int;
    const std::string _enabled_config_path;
    const std::string _thresholds_config_path;

    std::mutex update_mutex;
    int64_t _enabled_valid_from_seq;
    int64_t _thresholds_valid_from_seq;
    bool _enabled;
    std::vector<float> _threshold;
    std::vector<float> _fraction;
    int64_t _next_enabled_valid_from_seq;
    int64_t _next_thresholds_valid_from_seq;
    bool _next_enabled;
    std::vector<float> _next_threshold;
    std::vector<float> _next_fraction;
};

REGISTER_KOTEKAN_STAGE(RfiFrameMask);


RfiFrameMask::RfiFrameMask(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&RfiFrameMask::main_thread, this)),
    _num_local_freq(config.get<int64_t>(unique_name, "num_local_freq")),
    _samples_per_data_set(config.get<int64_t>(unique_name, "samples_per_data_set")),
    _sub_integration_ntime(config.get<int64_t>(unique_name, "sub_integration_ntime")),
    _rfi_downsampling_factor(config.get<int64_t>(unique_name, "rfi_downsampling_factor")),
    _rfi_num_times(div_noremainder(_samples_per_data_set, _rfi_downsampling_factor)),
    _num_integrations(div_noremainder(_samples_per_data_set, _sub_integration_ntime)),
    _num_rfi_per_int(div_noremainder(_sub_integration_ntime, _rfi_downsampling_factor)),
    _enabled_config_path(config.get<std::string>(unique_name, "enabled_updatable_config")),
    _thresholds_config_path(config.get<std::string>(unique_name, "thresholds_updatable_config")) {

    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    // Ensure outgoing buffer is standard type
    if (out_buf->buffer_type != "standard")
        FATAL_ERROR("RfiFrameMask out_buf ({:s}) is not of type standard.", out_buf->buffer_name);

    // Sanity checks on initialization
    {
        // number of frequencies in incoming frames from n2k
        if (_num_local_freq <= 0)
            FATAL_ERROR("num_local_freq is not positive: {:d}", _num_local_freq);

        // sampling information
        if (!(_samples_per_data_set > 0))
            FATAL_ERROR("samples_per_data_set is not positve: {:d}", _samples_per_data_set);
        if (!(_sub_integration_ntime > 0))
            FATAL_ERROR("sub_integration_ntime is not positve: {:d}", _sub_integration_ntime);
        if (!(_samples_per_data_set % _sub_integration_ntime == 0))
            FATAL_ERROR(
                "samples_per_data_set ({:d}) is not a multiple of sub_integration_ntime ({:d})",
                _samples_per_data_set, _sub_integration_ntime);

        // RFI downsampling factor checks
        if (!(_rfi_downsampling_factor > 0))
            FATAL_ERROR("rfi_downsampling_factor is not positive: {:d}", _rfi_downsampling_factor);

        if (!(_samples_per_data_set % _rfi_downsampling_factor == 0))
            FATAL_ERROR("samples_per_data_set {} is not a multiple of rfi_downsampling_factor {}",
                        _samples_per_data_set, _rfi_downsampling_factor);
        if (!(_sub_integration_ntime % _rfi_downsampling_factor == 0))
            FATAL_ERROR("sub_integration_ntime {} is not a multiple of rfi_downsampling_factor {}",
                        _sub_integration_ntime, _rfi_downsampling_factor);
    }

    // Ensure incoming buffer frame size is correct
    size_t sk_frame_size = _rfi_num_times * _num_local_freq * 3 * sizeof(float);

    if (in_buf->frame_size != sk_frame_size)
        FATAL_ERROR("RfiFrameMask in_buf ({:s}) has frame size {:d}. Expected {:d}.",
                    in_buf->buffer_name, in_buf->frame_size, sk_frame_size);

    // Set up outgoing frame descriptor
    out_buf->allocate_ndarray_frame_desc(kotekan::uint8, "RFIFrameMask",
                                         {_num_integrations, _num_local_freq}, {"Tc", "F"});

    // Initialize current RFI excision status, the "next" values are taken care of by the
    // configUpdater)
    _enabled = false;
    _threshold = {};
    _fraction = {};
    _enabled_valid_from_seq = 0;
    _thresholds_valid_from_seq = 0;

    // Set up REST endpoints
    INFO("Subscribing {:s} to updatable config.", _enabled_config_path);
    kotekan::configUpdater::instance().subscribe(
        _enabled_config_path, std::bind(&RfiFrameMask::receive_rfi_excision_enabled, this, _1));
    INFO("Subscribing {:s} to updatable config.", _thresholds_config_path);
    kotekan::configUpdater::instance().subscribe(
        _thresholds_config_path,
        std::bind(&RfiFrameMask::receive_rfi_excision_thresholds, this, _1));

    restServer& rest_server = restServer::instance();
    rest_server.register_get_callback(
        unique_name + "/enabled", std::bind(&RfiFrameMask::send_rfi_excision_enabled, this, _1));
    rest_server.register_get_callback(
        unique_name + "/thresholds",
        std::bind(&RfiFrameMask::send_rfi_excision_thresholds, this, _1));
}

RfiFrameMask::~RfiFrameMask() {
    // Must manually remove GET callbacks
    restServer& rest_server = restServer::instance();
    rest_server.remove_get_callback(unique_name + "/enabled");
    rest_server.remove_get_callback(unique_name + "/thresholds");
}

void RfiFrameMask::main_thread() {


    frameID in_frame_id(in_buf);
    frameID out_frame_id(out_buf);

    INFO("Generating RFI Frame Mask from {:s}[{:d}] putting result in {:s}[{:d}]",
         in_buf->buffer_name, in_frame_id, out_buf->buffer_name, out_frame_id);

    std::vector<int64_t> num_sk_exceeds(jsonMetadata::MAX_NUM_RFI_THRESHOLDS);

    while (!stop_thread) {

        // Fetch a new frame and get its sequence id
        DEBUG("Waiting for new SKtilde frame {:s}[{:d}].", in_buf->buffer_name, in_frame_id);
        const float* sktilde = (float*)in_buf->wait_for_full_frame(unique_name, in_frame_id);
        if (sktilde == nullptr)
            break;
        DEBUG("Waiting for new RFIFrameMask frame {:s}[{:d}].", out_buf->buffer_name, out_frame_id);
        uint8_t* frame_mask = (uint8_t*)out_buf->wait_for_empty_frame(unique_name, out_frame_id);
        if (frame_mask == nullptr)
            break;

        // Get metadata for incoming SK frame.
        std::shared_ptr<chordMetadata> in_meta = get_chord_metadata(in_buf, in_frame_id);
        check_sktilde_meta(in_meta);

        int64_t sk_f_stride = 3;
        int64_t sk_t_stride = 3 * _num_local_freq;
        int64_t mask_t_stride = _num_local_freq;

        // update the thresholds and enabled status for this frame.
        int64_t seq_num = in_meta->get_fpga_seq_num();
        update_enabled_and_thresholds(seq_num);

        DEBUG("Computing RFI frame mask");
        for (int64_t t_int = 0; t_int < _num_integrations; t_int++) {

            for (int64_t f = 0; f < _num_local_freq; f++) {

                std::fill(num_sk_exceeds.begin(), num_sk_exceeds.end(), 0);

                if (_enabled) {
                    for (int64_t t_rfi = t_int * _num_rfi_per_int;
                         t_rfi < (t_int + 1) * _num_rfi_per_int; t_rfi++) {
                        int64_t sk_idx = t_rfi * sk_t_stride + f * sk_f_stride;
                        float sk = sktilde[sk_idx + 0];
                        float sigma = sktilde[sk_idx + 2];

                        for (size_t k = 0; k < _threshold.size(); k++) {
                            if (sk > _threshold.at(k) * sigma)
                                num_sk_exceeds.at(k) += 1;
                        } // k
                    } // t_rfi
                }

                int64_t idx = t_int * mask_t_stride + f;

                frame_mask[idx] = 1;

                if (_enabled) {
                    for (size_t k = 0; k < _fraction.size(); k++) {
                        if (num_sk_exceeds.at(k) > _fraction.at(k) * _num_rfi_per_int) {
                            frame_mask[idx] = 0;
                        }
                    } // k
                }
            } // f
        } // t_int

        DEBUG("Setting Metadata");
        out_buf->allocate_new_metadata_object(out_frame_id);
        const std::shared_ptr<chordMetadata> out_meta = get_chord_metadata(out_buf, out_frame_id);

        out_meta->deepCopy(in_meta);
        out_meta->set_from_frame_desc(out_buf->get_ndarray_frame_desc());
        out_meta->set_time_downsampling_fpga(
            div_noremainder(in_meta->get_time_downsampling_fpga(), _rfi_downsampling_factor)
            * _sub_integration_ntime);

        std::vector<std::array<float, 2>> meta_thresholds{};
        for (size_t i = 0; i < _threshold.size(); i++)
            meta_thresholds.push_back({_threshold.at(i), _fraction.at(i)});

        out_meta->set_rfi_frame_excision_enabled(_enabled);
        out_meta->set_rfi_frame_excision_thresholds(meta_thresholds);

        out_meta->check_frame_desc(out_buf->get_ndarray_frame_desc());

        // Advance to the next frame
        in_buf->mark_frame_empty(unique_name, in_frame_id++);
        out_buf->mark_frame_full(unique_name, out_frame_id++);
    }
}

void RfiFrameMask::update_enabled_and_thresholds([[maybe_unused]] int64_t seq_num) {
    std::lock_guard<std::mutex> lock(update_mutex);

    if (seq_num >= _next_enabled_valid_from_seq) {
        _enabled = _next_enabled;
        _enabled_valid_from_seq = _next_enabled_valid_from_seq;
        _next_enabled_valid_from_seq = -1;
    }

    if (seq_num >= _next_thresholds_valid_from_seq) {
        _threshold = _next_threshold;
        _fraction = _next_fraction;
        _thresholds_valid_from_seq = _next_thresholds_valid_from_seq;
        _next_thresholds_valid_from_seq = -1;
    }
}

void RfiFrameMask::check_sktilde_meta(std::shared_ptr<chordMetadata> meta) {
    if (meta->dims != 3)
        FATAL_ERROR("Input buffer has dimension {:d}, should be 3 for sktilde", meta->dims);

    if (meta->dim[0] != _rfi_num_times)
        FATAL_ERROR("Input buffer has dim[0] = {:d}, should be {:d} (rfi_num_times) for sktilde",
                    meta->dim[0], _rfi_num_times);
    if (meta->dim[1] != _num_local_freq)
        FATAL_ERROR("Input buffer has dim[1] = {:d}, should be {:d} (num_local_freq) for sktilde",
                    meta->dim[1], _num_local_freq);
    if (meta->dim[2] != 3)
        FATAL_ERROR("Input buffer has dim[2] = {:d}, should be 3 for sktilde", meta->dim[2]);
}

bool RfiFrameMask::receive_rfi_excision_enabled([[maybe_unused]] nlohmann::json& json) {

    bool new_enabled;
    int64_t new_time_ns;

    try {
        new_enabled = json.at("rfi_frame_excision").get<bool>();
        new_time_ns = json.at("valid_from_time_ns").get<int64_t>();
    } catch (std::exception& e) {
        WARN("RfiFrameMask failed to read update to {:s}: {:s}", _enabled_config_path, e.what());
        return false;
    }

    int64_t seq_num = Telescope::instance().to_seq(nanosec_i64_to_timespec(new_time_ns));

    std::string time_str = fmt::format("t_inst = {:d} s + {:d} ns (seq {:d})",
                                       new_time_ns / 1'000'000, new_time_ns % 1'000'000, seq_num);

    if (new_enabled)
        INFO("Enabling RFI Frame Excision at: {:s}", time_str);
    else
        INFO("Disabling RFI Frame Excision at: {:s}", time_str);

    {
        std::lock_guard<std::mutex> lock(update_mutex);
        _next_enabled = new_enabled;
        _next_enabled_valid_from_seq = seq_num;
    }

    return true;
}

bool RfiFrameMask::receive_rfi_excision_thresholds([[maybe_unused]] nlohmann::json& json) {
    try {
    } catch (std::exception& e) {
        WARN("RfiFrameMask failed to read update to {:s}: {:s}", _thresholds_config_path, e.what());
        return false;
    }

    return true;
}

void RfiFrameMask::send_rfi_excision_enabled([[maybe_unused]] connectionInstance& conn) {}
void RfiFrameMask::send_rfi_excision_thresholds([[maybe_unused]] connectionInstance& conn) {}
