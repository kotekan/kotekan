#ifndef UPCHANNELIZATION_SCHEDULE_H
#define UPCHANNELIZATION_SCHEDULE_H

#include "Config.hpp"         // for Config
#include "kotekanLogging.hpp" // for kotekanLogging

#include <assert.h> // for assert
#include <cstddef>  // for size_t
#include <map>      // for map
#include <set>      // for set
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

    ////////////////////////////////////////////////////////////////////////////////

    const std::string unique_name;
    kotekan::Config& config;

    // The coarse frequency channels handled by this X-Engine
    // instance. This maps "channel index" to "channel".
    const std::vector<int> frequency_channels;
    const std::map<int, int> frequency_channels_to_indices;

    // All upchannelization factors
    const std::set<int> upchan_factors;

    // Map upchannelization factors to channels and back
    const std::map<int, std::set<int>> upchan_factors_to_channels; // factor -> channels
    const std::map<int, std::set<int>> upchan_channels_to_factors; // channel -> factors

    ////////////////////////////////////////////////////////////////////////////////

    std::vector<int> make_frequency_channels() const;
    std::map<int, int> make_frequency_channels_to_indices() const;

    std::set<int> make_upchan_factors() const;

    std::map<int, std::set<int>> make_upchan_factors_to_channels() const;
    std::map<int, std::set<int>> make_upchan_channels_to_factors() const;

    void output_statistics() const;
    bool invariant() const;

public:
    UpchannelizationSchedule(const UpchannelizationSchedule&) = delete;
    UpchannelizationSchedule(UpchannelizationSchedule&&) = delete;
    UpchannelizationSchedule& operator=(const UpchannelizationSchedule&) = delete;
    UpchannelizationSchedule& operator=(UpchannelizationSchedule&&) = delete;

    UpchannelizationSchedule(kotekan::Config& config, const std::string& unique_name = "",
                             const std::vector<int>& coarse_freq = std::vector<int>());

    // Get the singleton instance
    static const UpchannelizationSchedule&
    instance(kotekan::Config& config, const std::string& unique_name = "",
             const std::vector<int>& coarse_freq = std::vector<int>());

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
    const std::set<int>& get_upchan_factors() const {
        return upchan_factors;
    }

    // Map upchannelization factors to channels and back
    const std::set<int>& get_upchan_channels(const int upchan_factor) const;
    const std::set<int>& get_upchan_factors(const int channel) const;
};

#endif // #ifndef UPCHANNELIZATION_SCHEDULE_H
