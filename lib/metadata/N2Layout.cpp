#include "N2Layout.hpp"

#include "fmt.hpp"

#include <stdexcept>

void to_json(nlohmann::json& j, const N2Layout& l) {
    switch (l) {
        case N2Layout::FullUpperTri:
            j = "FullUpperTri";
            break;
        case N2Layout::RedundantBaselineAvg:
            j = "RedundantBaselineAvg";
            break;
        case N2Layout::Autocorrelations:
            j = "Autocorrelations";
            break;
        default:
            throw std::runtime_error(
                fmt::format("to_json - unknown N2Layout value: {:s}", static_cast<int32_t>(l)));
            break;
    }
}

void from_json(const nlohmann::json& j, N2Layout& l) {
    if (j == "FullUpperTri")
        l = N2Layout::FullUpperTri;
    else if (j == "RedundantBaselineAvg")
        l = N2Layout::RedundantBaselineAvg;
    else if (j == "Autocorrelations")
        l = N2Layout::Autocorrelations;
    else
        throw std::runtime_error(fmt::format("from_json - unknown N2Layout: {}", j.dump()));
}
