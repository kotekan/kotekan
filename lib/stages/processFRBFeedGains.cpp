#include "processFRBFeedGains.hpp"

#include "Config.hpp"
#include "StageFactory.hpp"
#include "Telescope.hpp"
#include "bufferContainer.hpp"
#include "processFeedGains.hpp"

#include <string>

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_KOTEKAN_STAGE(processFRBFeedGains);

processFRBFeedGains::processFRBFeedGains(Config& config, const std::string& unique_name,
                                         bufferContainer& buffer_container) :
    processFeedGains(config, unique_name, buffer_container) {
    // get the additional config parameters needed for the frame desc
    num_polarizations = config.get<uint32_t>(unique_name, "num_polarizations");
    frb1_swap_MN = config.get_default<bool>(unique_name, "frb1_swap_MN", false);

    // telescope layout
    const int num_dishes_x = Telescope::instance().get_grid_size_x();
    const int num_dishes_y = Telescope::instance().get_grid_size_y();

    num_dishes_M = frb1_swap_MN ? num_dishes_y : num_dishes_x;
    num_dishes_N = frb1_swap_MN ? num_dishes_x : num_dishes_y;
}

processFRBFeedGains::~processFRBFeedGains() {}

void processFRBFeedGains::set_frame_desc(Buffer* buf) {
    buf->require_frame_desc(kotekan::NDArray<kotekan::GetType_t<kotekan::float16>, 5>::describe(
        "W",
        {static_cast<ptrdiff_t>(num_local_freq * upchan_factor),
         static_cast<ptrdiff_t>(num_polarizations), static_cast<ptrdiff_t>(num_dishes_N),
         static_cast<ptrdiff_t>(num_dishes_M), static_cast<ptrdiff_t>(num_components)},
        {"Fbar", "P", "dishN", "dishM", "C"}, {1, 1, 1, 1, 1}));

    // everything below here ends up being the same as the parent class
    freq_upchan_factor = std::vector<int>(num_local_freq * upchan_factor, upchan_factor);
    freq_upchan_index = std::vector<int>(num_local_freq * upchan_factor);
    coarse_freq = std::vector<int>(num_local_freq * upchan_factor, -1);

    // set the actual frequency upchan indices. Assume increasing
    // upchannelized index
    // TODO: this needs to be consistent with the upchannelizer, and
    // potentially configurable
    for (uint64_t f = 0; f < num_local_freq * upchan_factor; ++f) {
        freq_upchan_index[f] = static_cast<int>(f % upchan_factor);
    }
}
