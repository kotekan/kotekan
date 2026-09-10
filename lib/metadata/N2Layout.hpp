#ifndef N2LAYOUT_HPP
#define N2LAYOUT_HPP

#include "json.hpp" // for json

#include <cstdint> // for int32_t
#include <string>  // for string

enum class N2Layout : int32_t {
    FullUpperTri = 0,
    RedundantBaselineAvg = 1,
    Autocorrelations = 2,
    InputANDMasked = 3,
    InputORMasked = 4,
    GeneralSubset = 5,
    // A compact frame over the connected elements -- those whose dish type is not
    // Fake, array dishes and RFI antennas alike -- derived from the telescope's dish
    // table. Products and per-element fields index the frame's own dense element
    // axis; each element's identity in the full fiducial order is carried in the
    // descriptor's input_list.
    DishInputs = 6
};

void to_json(nlohmann::json& j, const N2Layout& t);
void from_json(const nlohmann::json& j, N2Layout& t);

inline std::string N2Layout_to_string(N2Layout l) {
    nlohmann::json j = l;
    return j.get<std::string>();
}

#endif // N2LAYOUT_HPP
