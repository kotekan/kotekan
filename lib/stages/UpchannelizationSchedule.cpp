#include "UpchannelizationSchedule.hpp"

#include <cstddef>        // for size_t
#include <sstream>        // for basic_ostream, basic_ostringstream, operator<<, basic_ostream::...

#include "Telescope.hpp"  // for Telescope, freq_id_t
#include "fmt.hpp"        // for compile_string_to_view

std::vector<int> UpchannelizationSchedule::make_frequency_channels() const {
    const auto frequency_channels = config.get<std::vector<int>>(unique_name, "frequency_channels");
    return frequency_channels;
}

std::map<int, int> UpchannelizationSchedule::make_frequency_channels_to_indices() const {
    std::map<int, int> channels_to_indices;
    for (std::size_t index = 0; index < frequency_channels.size(); ++index)
        channels_to_indices[frequency_channels.at(index)] = index;
    return channels_to_indices;
}

std::set<int> UpchannelizationSchedule::make_upchan_factors() const {
    const auto upchan_factors = config.get<std::vector<int>>(unique_name, "upchan_factors");
    return std::set<int>(upchan_factors.begin(), upchan_factors.end());
}

std::map<int, std::set<int>> UpchannelizationSchedule::make_upchan_factors_to_channels() const {
    std::map<int, std::set<int>> upchan_factors_to_channels;
    // for (const int upchan_factor : upchan_factors) {
    //     std::ostringstream key;
    //     key << "upchan_U" << upchan_factor << "_channels";
    //     const auto channels = config.get<std::vector<int>>(unique_name, key.str());
    //     upchan_factors_to_channels[upchan_factor] = std::set<int>(channels.begin(),
    //     channels.end());
    // }
    for (const int upchan_factor : upchan_factors) {
        std::ostringstream key;
        key << "upchan_channel_ranges/" << upchan_factor << "";
        const auto& range = config.get<std::vector<int>>(unique_name, key.str());
        assert(range.size() == 2);
        std::set<int> channels;
        for (int channel = range.at(0); channel < range.at(1); ++channel) {
            // Only include the local channels
            if (has_frequency_channel(channel))
                channels.insert(channel);
        }
        upchan_factors_to_channels[upchan_factor] = channels;
    }
    return upchan_factors_to_channels;
}

std::map<int, std::set<int>> UpchannelizationSchedule::make_upchan_channels_to_factors() const {
    std::map<int, std::set<int>> upchan_channels_to_factors;
    for (const int upchan_factor : upchan_factors) {
        const auto& channels = get_upchan_channels(upchan_factor);
        for (const int channel : channels)
            upchan_channels_to_factors[channel].insert(upchan_factor);
    }
    return upchan_channels_to_factors;
}

void UpchannelizationSchedule::output_statistics() const {
    const auto& telescope = Telescope::instance();

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

    int total_output_channels = 0;
    INFO("There are {} local input frequency channels", frequency_channels.size());
    for (std::size_t index = 0; index < frequency_channels.size(); ++index) {
        const freq_id_t channel = frequency_channels.at(index);
        const double frequency = telescope.to_freq_MHz(channel);
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
        INFO("    index {}, channel {}, frequency {} MHz, {}", index, channel, frequency,
             upchannelized.str());
    }
    INFO("There are {} local output frequency channels.", total_output_channels);
}

bool UpchannelizationSchedule::invariant() const {
    const auto& upchan_channels = get_frequency_channels();
    for (int index = 0; index < int(upchan_channels.size()); ++index) {
        if (!has_frequency_index(index))
            return false;
        const int channel = get_frequency_channel(index);
        if (!has_frequency_channel(channel))
            return false;
        if (get_frequency_index(channel) != index)
            return false;
    }
    for (const int channel : upchan_channels) {
        if (!has_frequency_channel(channel))
            return false;
        const int index = get_frequency_index(channel);
        if (!has_frequency_index(index))
            return false;
        if (get_frequency_channel(index) != channel)
            return false;
    }

    const auto& upchan_factors = get_upchan_factors();
    for (const int factor : upchan_factors) {
        if (factor <= 0)
            return false;
        if ((factor & (factor - 1)) != 0)
            return false;
    }
    for (const int factor : upchan_factors) {
        const auto& channels = get_upchan_channels(factor);
        for (const int channel : channels) {
            if (!has_frequency_channel(channel))
                return false;
            const auto& factors = get_upchan_factors(channel);
            if (!factors.count(factor))
                return false;
            for (const int factor2 : factors) {
                if (!get_upchan_channels(factor2).count(channel))
                    return false;
            }
        }
    }
    for (const int channel : upchan_channels) {
        const auto& factors = get_upchan_factors(channel);
        for (const int factor : factors) {
            if (!upchan_factors.count(factor))
                return false;
            const auto& channels = get_upchan_channels(factor);
            if (!channels.count(channel))
                return false;
            for (const int channel2 : channels) {
                if (!get_upchan_factors(channel2).count(factor))
                    return false;
            }
        }
    }

    return true;
}

UpchannelizationSchedule::UpchannelizationSchedule(kotekan::Config& config) :
    kotekan::kotekanLogging(),
    //
    unique_name(""), config(config),
    //
    frequency_channels(make_frequency_channels()),
    frequency_channels_to_indices(make_frequency_channels_to_indices()),
    //
    upchan_factors(make_upchan_factors()),
    //
    upchan_factors_to_channels(make_upchan_factors_to_channels()),
    upchan_channels_to_factors(make_upchan_channels_to_factors())
//
{
    output_statistics();
    assert(invariant());
}

const UpchannelizationSchedule& UpchannelizationSchedule::instance(kotekan::Config& config) {
    static const UpchannelizationSchedule the_instance(config);
    return the_instance;
}

////////////////////////////////////////////////////////////////////////////////

// Map upchannelization factors to channels and back
const std::set<int>& UpchannelizationSchedule::get_upchan_channels(const int upchan_factor) const {
    if (!upchan_factors_to_channels.count(upchan_factor)) {
        static const std::set<int> empty_set;
        return empty_set;
    }
    return upchan_factors_to_channels.at(upchan_factor);
}

const std::set<int>& UpchannelizationSchedule::get_upchan_factors(const int channel) const {
    if (!upchan_channels_to_factors.count(channel)) {
        static const std::set<int> empty_set;
        return empty_set;
    }
    return upchan_channels_to_factors.at(channel);
}
