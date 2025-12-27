#ifndef N2LAYOUT_HPP
#define N2LAYOUT_HPP

#include "json.hpp" // for json

#include <cstdint> // for int32_t
#include <string>  // for string

enum class N2Layout : int32_t { FullUpperTri = 0, RedundantBaselineAvg = 1, Autocorrelations = 2 };

void to_json(nlohmann::json& j, const N2Layout& t);
void from_json(const nlohmann::json& j, N2Layout& t);

inline std::string N2Layout_to_string(N2Layout l) {
    nlohmann::json j = l;
    return j.get<std::string>();
}

#endif // N2LAYOUT_HPP
