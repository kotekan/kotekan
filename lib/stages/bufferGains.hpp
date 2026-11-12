/**
 * @file
 * @brief Buffer posted per-input complex gains.
 *  - bufferGains : public kotekan::Stage
 */

#ifndef BUFFER_GAINS_HPP
#define BUFFER_GAINS_HPP

#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "prometheusMetrics.hpp" // for Counter

#include "json.hpp" // for json

#include <atomic>             // for atomic
#include <complex>            // for complex
#include <condition_variable> // for condition_variable
#include <mutex>              // for mutex
#include <optional>           // for optional
#include <stddef.h>           // for size_t
#include <string>             // for string
#include <vector>             // for vector

/**
 * @class bufferGains
 * @brief Turn posted per-input complex gains into frames of a gains buffer.
 *
 * Subscribes to an updatable config block, fed by the calibration service through choco. An
 * update carries the gains of every input at a list of frequencies; this stage keeps the ones
 * for its own frequencies and writes them as one frame of @c out_buf, where N2Accumulate and the
 * beamformer weight stages pick them up. A frame is produced for each accepted update, plus one
 * at startup, so a consumer polls the buffer without blocking (`wait_for_full_frame_timeout`
 * with a timeout already passed) and keeps the newest frame it has taken. The gains switch
 * when the frame is produced; a posted `start_time` is not honoured. Unlike loadFeedGains,
 * which reads the gains from files named in the update, this stage takes them inline.
 *
 * An update is a JSON object with these keys (every key of the config block must be present in
 * every POST; keys this stage does not read, such as `start_time` and `weight`, are accepted
 * and ignored):
 * - `update_id`: String, named in log messages.
 * - `freq_id`: Array of Int, the frequency id of each row of `gain`. Must include every
 *   frequency of this instance, in any order; other frequencies are ignored.
 * - `gain`: Object `{"shape": [num_freq, num_elements], "dtype": "complex64",
 *   "encoding": "base64", "data": String}`: the gains, row-major, as base64 of the
 *   little-endian complex64 array. Elements are in the frame's [P][D] order, the order of the
 *   telescope's dish_inputs table. Every value must be finite; bad inputs belong in the bad
 *   feed mask. An empty object `{}` means @c default_gains.
 *
 * The block's initial values arrive through the same path at startup, so a block whose `gain`
 * is `{}` gives @c default_gains until the first POST, and one carrying gains starts from
 * them. A POSTed update that is malformed, has the wrong number of elements, holds a value
 * that is not finite, or lacks one of this instance's frequencies is logged, counted and
 * ignored, so the previous gains stay in force; a bad initial block stops kotekan.
 *
 * This instance's frequencies are the `coarse_freq` of the first frame of @c metadata_source,
 * as for bufferBadInputs; the frames carry them as `coarse_freq`. Until that frame has arrived,
 * updates are queued, a newer one replacing an older one not yet written.
 *
 * @par Buffers
 * @buffer metadata_source  Any buffer with chordMetadata; only its first frame is read.
 *     @buffer_metadata chordMetadata
 * @buffer out_buf  The gains, one frame per update, elements in [P][D] order.
 *     @buffer_format NDArray float32 [num_local_freq, num_polarizations, num_dishes, 2]
 *     @buffer_metadata chordMetadata (quantity "gain", coarse_freq)
 *
 * @conf num_local_freq      Int. Number of frequencies of this instance.
 * @conf num_polarizations   Int.
 * @conf num_dishes          Int.
 * @conf default_gains       Array of two Floats, default [1, 0]. The (real, imag) gain of every
 *                           input until gains are posted.
 * @conf updatable_config/gains  String. Path of the updatable config block.
 *
 * @par Metrics
 * @metric kotekan_buffergains_update_count  Frames produced, i.e. updates applied.
 * @metric kotekan_buffergains_invalid_update_count  POSTed updates ignored as malformed or not
 *     covering this instance.
 *
 * @author James Mertens
 */
class bufferGains : public kotekan::Stage {
public:
    bufferGains(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);

    void main_thread() override;

    /// Callback for the updatable config block; see the class comment for the format.
    bool receive_gains(nlohmann::json& json);

private:
    /// A parsed update.
    struct gainUpdate {
        std::string update_id;
        std::vector<int> freq_id;
        /// [freq_id.size()][num_elements], row-major; empty for default_gains.
        std::vector<std::complex<float>> gain;
        /// True for the config block's initial values.
        bool initial = false;
    };

    /// Parse a posted update; throws std::invalid_argument or std::runtime_error on a malformed
    /// one.
    gainUpdate parse_update(const nlohmann::json& json) const;
    /// The first of `coarse_freq` that `update` has no gains for, if any.
    static std::optional<int> missing_freq(const gainUpdate& update,
                                           const std::vector<int>& coarse_freq);

    Buffer* out_buf;
    Buffer* metadata_source;
    int num_local_freq;
    int num_polarizations;
    int num_dishes;
    size_t num_elements;
    std::complex<float> default_gain;

    /// This instance's frequencies; empty until read from metadata_source.
    std::vector<int> coarse_freq;
    /// The newest update not yet written.
    std::optional<gainUpdate> pending;
    /// Guards coarse_freq and pending.
    std::mutex mtx;
    std::condition_variable pending_cv;

    /// False while the constructor's subscribe() delivers the initial block: a bad update is
    /// fatal then and merely counted afterwards.
    std::atomic<bool> initialised = false;

    kotekan::prometheus::Counter& updates_counter;
    kotekan::prometheus::Counter& invalid_updates_counter;
};

#endif // BUFFER_GAINS_HPP
