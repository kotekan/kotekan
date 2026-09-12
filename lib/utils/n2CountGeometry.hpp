#ifndef N2_COUNT_GEOMETRY_HPP
#define N2_COUNT_GEOMETRY_HPP

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace kotekan {

/// Return P*(D/8), or throw if the count kernel does not support it.
inline int n2_count_station_groups(int64_t num_polarizations, int64_t num_dishes) {
    if (num_polarizations <= 0 || num_dishes <= 0)
        throw std::invalid_argument("N2 count geometry requires positive polarizations and dishes");
    if (num_dishes % 8 != 0)
        throw std::invalid_argument("N2 count geometry requires dishes divisible by eight");
    const int64_t groups_per_polarization = num_dishes / 8;
    if (groups_per_polarization > std::numeric_limits<int64_t>::max() / num_polarizations)
        throw std::invalid_argument("N2 count station-group geometry overflow");
    const int64_t groups = num_polarizations * groups_per_polarization;
    if (groups != 16 && groups != 128)
        throw std::invalid_argument("N2 count kernel only supports Sds=16 or Sds=128; got Sds="
                                    + std::to_string(groups) + ".");
    return static_cast<int>(groups);
}

} // namespace kotekan

#endif
