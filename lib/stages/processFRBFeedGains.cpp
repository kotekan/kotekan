#include "processFRBFeedGains.hpp"

#include "Config.hpp"
#include "StageFactory.hpp"
#include "Telescope.hpp"
#include "bufferContainer.hpp"
#include "frb1IntensityBound.hpp"
#include "kotekanLogging.hpp"
#include "processFeedGains.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
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

void processFRBFeedGains::copy_upchannelize_f(const float* src_f, float16_t* dst_f, size_t fid) {
    (void)fid; // unused - coarse frequencies are just copied
    auto scaling_factor = this->scaling_factor;
    for (size_t u = 0; u < upchan_factor; ++u) {
        // copy ell elements from the source into each fine channel
        float16_t* u_ptr = dst_f + u * num_elements * num_components;
        // apply the constant scaling factor
        std::transform(src_f, src_f + num_components * num_elements, u_ptr,
                       [scaling_factor](float v) { return float16_t(v * scaling_factor); });
    }
}

void processFRBFeedGains::check_gains(const float16_t* frame) {
    // Output frame layout: [beam][Fbar][P][dishN][dishM][re/im]
    assert(num_components == 2);
    assert(num_elements == num_polarizations * num_dishes_M * num_dishes_N);
    const std::ptrdiff_t str_freq = std::ptrdiff_t(num_elements) * num_components;
    const std::ptrdiff_t num_freqs = std::ptrdiff_t(num_beams) * num_local_freq * upchan_factor;

    // Find the frequency with the largest worst-case intensity
    std::ptrdiff_t num_bad_freqs = 0;
    std::ptrdiff_t worst_freq = -1;
    double worst_bound = 0;
    for (std::ptrdiff_t freq = 0; freq < num_freqs; ++freq) {
        const double bound = kotekan::frb1_intensity_bound(
            frame + str_freq * freq, num_polarizations, num_dishes_M, num_dishes_N);
        if (bound > kotekan::frb1_intensity_limit)
            ++num_bad_freqs;
        if (bound > worst_bound) {
            worst_freq = freq;
            worst_bound = bound;
        }
    }

    // Don't abort: the gains are updated while the pipeline runs, and the kernel only
    // overflows for strong coherent signals
    if (num_bad_freqs > 0)
        WARN("The FRB1 gains can overflow Float16 for {:d} of {:d} frequencies: frequency {:d} "
             "has a worst-case intensity of {:g}, above the limit {:g}. The gains need to be "
             "reduced by at least a factor {:.3g}.",
             num_bad_freqs, num_freqs, worst_freq, worst_bound, kotekan::frb1_intensity_limit,
             std::sqrt(worst_bound / kotekan::frb1_intensity_limit));
    else
        DEBUG("FRB1 gains: largest worst-case intensity {:g} (frequency {:d}), limit {:g}",
              worst_bound, worst_freq, kotekan::frb1_intensity_limit);
}

void processFRBFeedGains::set_frame_desc(Buffer* buf) {
    // Attach the frame description, or check the declared one
    buf->ensure_frame_desc(kotekan::NDArray<kotekan::GetType_t<kotekan::float16>, 5>::describe(
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
