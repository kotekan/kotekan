#include "jsonMetadata.hpp"

#include <cmath>     // for nanf
#include <cstddef>   // for size_t
#include <cstdint>   // for uint32_t, int64_t
#include <stdexcept> // for runtime_error, out_of_range
#include <string>    // for string
#include <vector>    // for vector

namespace jsonMetadata {

namespace {

/// Reads a json array of at most MAX_NUM_BEAMS values into a fixed-size field,
/// marking the remaining entries as unused by setting them to @p unset.
///
/// The fill is unconditional on purpose: it is what tells a consumer which beams
/// are in use (see beamCoord). Skipping it would also leave the tail
/// uninitialized, and to_json always writes all MAX_NUM_BEAMS entries.
template<typename T>
void beam_array_from_json(const nlohmann::json& j, const std::string& key,
                          T (&field)[MAX_NUM_BEAMS], const T unset) {
    const nlohmann::json& values = j.at(key);
    if (values.size() > std::size_t(MAX_NUM_BEAMS))
        throw std::runtime_error("Number of beams in \"" + key + "\" ("
                                 + std::to_string(values.size()) + ") exceeds MAX_NUM_BEAMS ("
                                 + std::to_string(MAX_NUM_BEAMS) + ")");
    std::size_t i = 0;
    for (auto it = values.cbegin(); it != values.cend(); ++it)
        it->get_to(field[i++]);
    for (; i < std::size_t(MAX_NUM_BEAMS); ++i)
        field[i] = unset;
}

} // namespace

void from_json(const nlohmann::json& j, beamCoord& c) {
    beam_array_from_json(j, RIGHT_ASCENSION, c.right_ascension, std::nanf(""));
    beam_array_from_json(j, DECLINATION, c.declination, std::nanf(""));
    beam_array_from_json(j, SCALING, c.scaling, std::uint32_t(0));
}

void to_json(nlohmann::json& j, const beamCoord& c) {
    j[RIGHT_ASCENSION] = std::vector<float>(c.right_ascension, c.right_ascension + MAX_NUM_BEAMS);
    j[DECLINATION] = std::vector<float>(c.declination, c.declination + MAX_NUM_BEAMS);
    j[SCALING] = std::vector<std::uint32_t>(c.scaling, c.scaling + MAX_NUM_BEAMS);
}

} // namespace jsonMetadata

void to_json(nlohmann::json& j, const timeval& tv) {
    j[jsonMetadata::TV_SEC] = static_cast<std::int64_t>(tv.tv_sec);
    j[jsonMetadata::TV_USEC] = static_cast<std::int64_t>(tv.tv_usec);
}

void from_json(const nlohmann::json& j, timeval& tv) {
    const std::int64_t tv_sec = j.at(jsonMetadata::TV_SEC).template get<std::int64_t>();
    const std::int64_t tv_usec = j.at(jsonMetadata::TV_USEC).template get<std::int64_t>();

    // `time_t` and `suseconds_t` can be narrower than int64_t; check that the
    // values survive the conversion instead of silently truncating them.
    const auto sec = static_cast<decltype(tv.tv_sec)>(tv_sec);
    if (static_cast<std::int64_t>(sec) != tv_sec)
        throw std::out_of_range("timeval TV_SEC " + std::to_string(tv_sec)
                                + " does not fit into time_t");
    if (tv_usec < 0 || tv_usec >= 1000000)
        throw std::out_of_range("timeval TV_USEC " + std::to_string(tv_usec)
                                + " is not in [0, 1000000)");
    const auto usec = static_cast<decltype(tv.tv_usec)>(tv_usec);

    tv.tv_sec = sec;
    tv.tv_usec = usec;
}
