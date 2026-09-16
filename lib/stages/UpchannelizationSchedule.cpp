#include "UpchannelizationSchedule.hpp"

#include "Telescope.hpp" // for Telescope, freq_id_t

#include "fmt.hpp" // for compile_string_to_view

#include <algorithm> // for find, minmax_element
#include <cstddef>   // for size_t
#include <set>       // for set
#include <sstream>   // for basic_ostream, basic_ostringstream, operator<<, basic_ostream::...

std::vector<int>
UpchannelizationSchedule::make_frequency_channels(const std::string& schedule_name,
                                                  const std::vector<int>& coarse_freq) const {
    // These are passed in by the caller and are thus not validated by
    // the configuration parser. Check them here.
    std::set<int> seen;
    for (const int channel : coarse_freq) {
        if (channel < 0)
            FATAL_ERROR("Upchannelization schedule \"{:s}\": coarse frequency channel {:d} is "
                        "negative",
                        schedule_name, channel);
        if (!seen.insert(channel).second)
            FATAL_ERROR("Upchannelization schedule \"{:s}\": coarse frequency channel {:d} is "
                        "listed more than once",
                        schedule_name, channel);
    }
    return coarse_freq;
}

std::map<int, int> UpchannelizationSchedule::make_frequency_channels_to_indices() const {
    std::map<int, int> channels_to_indices;
    for (std::size_t index = 0; index < frequency_channels.size(); ++index)
        channels_to_indices[frequency_channels.at(index)] = index;
    return channels_to_indices;
}

std::vector<int>
UpchannelizationSchedule::make_upchan_factors(kotekan::Config& config,
                                              const std::string& schedule_name) const {
    const auto upchan_factors = config.get<std::vector<int>>(schedule_name, "upchan_factors");
    for (const int upchan_factor : upchan_factors) {
        if (upchan_factor <= 0 || (upchan_factor & (upchan_factor - 1)) != 0)
            FATAL_ERROR("Upchannelization schedule \"{:s}\": upchannelization factor {:d} is not a "
                        "positive power of two",
                        schedule_name, upchan_factor);
    }
    return upchan_factors;
}

std::map<int, std::vector<int>>
UpchannelizationSchedule::make_upchan_factors_to_channels(kotekan::Config& config,
                                                          const std::string& schedule_name) const {
    std::map<int, std::vector<int>> upchan_factors_to_channels;
    for (const int upchan_factor : upchan_factors) {
        std::ostringstream key;
        key << "upchan_channel_ranges/" << upchan_factor << "";
        const auto& range = config.get<std::vector<int>>(schedule_name, key.str());
        // ranges are [min, max)
        if (range.size() != 2)
            FATAL_ERROR("Upchannelization schedule \"{:s}\": \"{:s}\" must hold exactly 2 elements "
                        "[min, max), found {:d}",
                        schedule_name, key.str(), range.size());
        if (range.at(0) > range.at(1))
            FATAL_ERROR("Upchannelization schedule \"{:s}\": \"{:s}\" is the empty range [{:d}, "
                        "{:d}); the lower bound must not exceed the upper bound",
                        schedule_name, key.str(), range.at(0), range.at(1));
        std::vector<int> channels;
        const auto& upchan_channels = get_frequency_channels();
        // Only include local channels, in local order
        for (const auto channel : upchan_channels) {
            // Only include the requested channels
            if (range.at(0) <= channel && channel < range.at(1))
                channels.push_back(channel);
        }
        upchan_factors_to_channels[upchan_factor] = channels;
    }
    return upchan_factors_to_channels;
}

std::map<int, std::vector<int>> UpchannelizationSchedule::make_upchan_channels_to_factors() const {
    std::map<int, std::vector<int>> upchan_channels_to_factors;
    for (const int upchan_factor : upchan_factors) {
        const auto& channels = get_upchan_channels(upchan_factor);
        for (const int channel : channels)
            upchan_channels_to_factors[channel].push_back(upchan_factor);
    }
    return upchan_channels_to_factors;
}

void UpchannelizationSchedule::output_statistics() const {
    INFO("Upchannelization schedule:");

    {
        std::ostringstream factors;
        bool isfirst = true;
        for (const int upchan_factor : upchan_factors) {
            if (!isfirst)
                factors << ", ";
            isfirst = false;
            factors << upchan_factor;
        }
        INFO("There are {} upchannelization factors: [{}]", upchan_factors.size(), factors.str());
    }

    // Every stage builds its own schedule, so this runs once per stage
    // (and once per GPU). Keep the summary at INFO and the per-channel
    // listing at DEBUG, or the listing crowds out the rest of the
    // startup log.
    int total_output_channels = 0;
    if (frequency_channels.empty()) {
        INFO("There are no local input frequency channels");
    } else {
        const auto minmax =
            std::minmax_element(frequency_channels.begin(), frequency_channels.end());
        INFO("There are {} local input frequency channels, from channel {} to channel {}",
             frequency_channels.size(), *minmax.first, *minmax.second);
    }
    for (std::size_t index = 0; index < frequency_channels.size(); ++index) {
        const freq_id_t channel = frequency_channels.at(index);
        const auto factors = get_upchan_factors(channel);
        std::ostringstream upchannelized;
        if (factors.empty()) {
            upchannelized << "not upchannelized";
            total_output_channels += 1;
        } else {
            upchannelized << "upchannelized by [";
            bool isfirst = true;
            for (const int factor : factors) {
                if (!isfirst)
                    upchannelized << ", ";
                isfirst = false;
                upchannelized << factor;
                total_output_channels += factor;
            }
            upchannelized << "]";
        }
        // Call `Telescope::instance()` here rather than hoisting it out of
        // the loop: in a non-debug build `DEBUG` expands to nothing, and a
        // hoisted variable would be unused.
        DEBUG("    index {}, channel {}, frequency {} MHz, {}", index, channel,
              Telescope::instance().to_freq_MHz(channel), upchannelized.str());
    }
    INFO("There are {} local output frequency channels.", total_output_channels);
}

// Check that the derived lookup tables are consistent with each
// other. Unlike the checks in the `make_*` functions above, a failure
// here indicates a bug in this file, not a malformed schedule, so we
// report the condition that failed.
#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            ERROR("Upchannelization schedule: invariant `{:s}` violated", #cond);                  \
            return false;                                                                          \
        }                                                                                          \
    } while (0)

bool UpchannelizationSchedule::invariant() const {
    const auto& upchan_channels = get_frequency_channels();
    for (int index = 0; index < int(upchan_channels.size()); ++index) {
        CHECK(has_frequency_index(index));
        const int channel = get_frequency_channel(index);
        CHECK(has_frequency_channel(channel));
        CHECK(get_frequency_index(channel) == index);
    }
    for (const int channel : upchan_channels) {
        CHECK(has_frequency_channel(channel));
        const int index = get_frequency_index(channel);
        CHECK(has_frequency_index(index));
        CHECK(get_frequency_channel(index) == channel);
    }

    const auto& upchan_factors = get_upchan_factors();
    for (const int factor : upchan_factors) {
        CHECK(factor > 0);
        CHECK((factor & (factor - 1)) == 0);
    }
    for (const int factor : upchan_factors) {
        const auto& channels = get_upchan_channels(factor);
        for (const int channel : channels) {
            CHECK(has_frequency_channel(channel));
            const auto& factors = get_upchan_factors(channel);
            CHECK(std::find(factors.begin(), factors.end(), factor) != factors.end());
            for (const int factor2 : factors) {
                const auto& channels2 = get_upchan_channels(factor2);
                CHECK(std::find(channels2.begin(), channels2.end(), channel) != channels2.end());
            }
        }
    }
    for (const int channel : upchan_channels) {
        const auto& factors = get_upchan_factors(channel);
        for (const int factor : factors) {
            CHECK(std::find(upchan_factors.begin(), upchan_factors.end(), factor)
                  != upchan_factors.end());
            const auto& channels = get_upchan_channels(factor);
            CHECK(std::find(channels.begin(), channels.end(), channel) != channels.end());
            for (const int channel2 : channels) {
                const auto& factors2 = get_upchan_factors(channel2);
                CHECK(std::find(factors2.begin(), factors2.end(), factor) != factors2.end());
            }
        }
    }

    return true;
}

#undef CHECK

UpchannelizationSchedule::UpchannelizationSchedule(kotekan::Config& config,
                                                   const std::string& schedule_name,
                                                   const std::vector<int>& coarse_freq,
                                                   const std::string& caller_unique_name) :
    kotekan::kotekanLogging(),
    //
    frequency_channels(make_frequency_channels(schedule_name, coarse_freq)),
    frequency_channels_to_indices(make_frequency_channels_to_indices()),
    //
    upchan_factors(make_upchan_factors(config, schedule_name)),
    //
    upchan_factors_to_channels(make_upchan_factors_to_channels(config, schedule_name)),
    upchan_channels_to_factors(make_upchan_channels_to_factors())
//
{
    // Log under the calling stage's name and at its configured log
    // level, just as `Stage` itself does. Without this the schedule
    // would keep `kotekanLogging`'s default level, and its `DEBUG`
    // output could never be enabled.
    set_log_level(config.get<std::string>(caller_unique_name, "log_level"));
    set_log_prefix(caller_unique_name);
    output_statistics();
    assert(invariant());
}

////////////////////////////////////////////////////////////////////////////////

// Map upchannelization factors to channels and back
const std::vector<int>&
UpchannelizationSchedule::get_upchan_channels(const int upchan_factor) const {
    if (!upchan_factors_to_channels.count(upchan_factor)) {
        static const std::vector<int> empty_set;
        return empty_set;
    }
    return upchan_factors_to_channels.at(upchan_factor);
}

const std::vector<int>& UpchannelizationSchedule::get_upchan_factors(const int channel) const {
    if (!upchan_channels_to_factors.count(channel)) {
        static const std::vector<int> empty_set;
        return empty_set;
    }
    return upchan_channels_to_factors.at(channel);
}
