#ifndef UPCHANNELIZATION_SCHEDULE_H
#define UPCHANNELIZATION_SCHEDULE_H

#include "Config.hpp"         // for Config
#include "buffer.hpp"         // for Buffer
#include "kotekanLogging.hpp" // for kotekanLogging

#include <assert.h> // for assert
#include <cstddef>  // for size_t
#include <map>      // for map
#include <optional> // for optional
#include <string>   // for basic_string, string
#include <vector>   // for vector

class UpchannelizationSchedule : kotekan::kotekanLogging {
    // Some terminology:
    //
    // - "frequency": real number, measured in MHz
    //
    // - "channel", "frequency id": integer, counting the channels
    //                              created in the F-Engine Fourier
    //                              transform, e.g. 0 to 2048 for
    //                              CHIME or 0 to 8192 for CHORD. Not
    //                              all channels are processed by an
    //                              X-Engine node.
    //
    // - "channel index": the array index where a particular channel
    //                    is stored on a particular X-Engine node,
    //                    usually from a contiguous range, usually
    //                    starting from 0. This labels all the
    //                    channels that are processed by a particular
    //                    X-Engine node.

    // The mapping from "channel" to "frequency" is done via the CHORD
    // Telescope object function `to_freq_MHz`.

    // This is a plain value type, not a singleton. Parsing a schedule
    // is cheap -- a handful of configuration lookups and some small
    // maps -- so every stage that needs a schedule constructs its
    // own.
    //
    // There deliberately is no process-wide instance. A single
    // kotekan process can drive several GPUs, and each GPU handles a
    // different set of coarse frequency channels, so there is no such
    // thing as "the" schedule of a process.
    //
    // The configuration describes the schedule itself: the
    // upchannelization factors, and the range of channels each factor
    // applies to. That part is identical on all nodes and for all
    // GPUs. The set of coarse frequency channels, on the other hand,
    // is local to one GPU; the caller passes it in, taking it e.g.
    // from the metadata of an incoming frame. It must never be read
    // from the configuration.

    ////////////////////////////////////////////////////////////////////////////////

    // The coarse frequency channels handled by this X-Engine
    // instance. This maps "channel index" to "channel".
    const std::vector<int> frequency_channels;
    const std::map<int, int> frequency_channels_to_indices;

    // All upchannelization factors
    const std::vector<int> upchan_factors;

    // Map upchannelization factors to channels and back
    const std::map<int, std::vector<int>> upchan_factors_to_channels; // factor -> channels
    const std::map<int, std::vector<int>> upchan_channels_to_factors; // channel -> factors

    ////////////////////////////////////////////////////////////////////////////////

    // These validate their input and abort with a fatal error if it is
    // malformed. They take `schedule_name` only to name it in error
    // messages; the log prefix is not set yet while they run.
    std::vector<int> make_frequency_channels(const std::string& schedule_name,
                                             const std::vector<int>& coarse_freq) const;
    std::map<int, int> make_frequency_channels_to_indices() const;

    std::vector<int> make_upchan_factors(kotekan::Config& config,
                                         const std::string& schedule_name) const;

    std::map<int, std::vector<int>>
    make_upchan_factors_to_channels(kotekan::Config& config,
                                    const std::string& schedule_name) const;
    std::map<int, std::vector<int>> make_upchan_channels_to_factors() const;

    void output_statistics() const;
    bool invariant() const;

public:
    // Parse an upchannelization schedule.
    //
    // - config, schedule_name: where to find `upchan_factors` and
    //   `upchan_channel_ranges` in the configuration.
    //
    // - coarse_freq: the coarse frequency channels handled here. This
    //   is local to one GPU and thus cannot come from the
    //   configuration.
    //
    // - caller_unique_name: the unique name of the calling stage. The
    //   schedule logs under that name and at that stage's configured
    //   log level, the same way the stage itself does. Several stages
    //   -- and several GPUs -- build their own schedules, so their log
    //   output needs to be told apart.
    UpchannelizationSchedule(kotekan::Config& config, const std::string& schedule_name,
                             const std::vector<int>& coarse_freq,
                             const std::string& caller_unique_name);

    // The coarse frequency channels handled by this X-Engine
    // instance. This maps "channel index" to "channel".
    bool has_frequency_index(const int index) const {
        return index >= 0 && std::size_t(index) < frequency_channels.size();
    }
    int get_frequency_channel(const int index) const {
        assert(has_frequency_index(index));
        return frequency_channels.at(index);
    }
    bool has_frequency_channel(const int channel) const {
        return frequency_channels_to_indices.count(channel);
    }
    int get_frequency_index(const int channel) const {
        assert(has_frequency_channel(channel));
        return frequency_channels_to_indices.at(channel);
    }
    const std::vector<int>& get_frequency_channels() const {
        return frequency_channels;
    }

    // All upchannelization factors
    const std::vector<int>& get_upchan_factors() const {
        return upchan_factors;
    }

    // Map upchannelization factors to channels and back
    const std::vector<int>& get_upchan_channels(const int upchan_factor) const;
    const std::vector<int>& get_upchan_factors(const int channel) const;
};

// Read the coarse frequency channels handled by this GPU from the
// metadata of the first frame of each of `metadata_sources`, in order.
//
// CHORD carries all local channels in a single voltage buffer, CHIME
// splits them over one buffer per channel, so this takes a list and
// concatenates what it finds.
//
// The channels do not change while kotekan runs, so one frame from each
// buffer is enough: this marks that frame empty and unregisters the
// caller as a consumer, leaving the buffers to the rest of the
// pipeline. The caller must have registered as a consumer of every
// buffer beforehand.
//
// Returns an empty optional if kotekan shut down while waiting.
std::optional<std::vector<int>> wait_for_coarse_freq(const std::vector<Buffer*>& metadata_sources,
                                                     const std::string& unique_name);

#endif // #ifndef UPCHANNELIZATION_SCHEDULE_H
