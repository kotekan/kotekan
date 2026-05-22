#include "N2Layout.hpp"

#include <fmt/core.h>  // for format, format_string
#include <json.hpp>    // for operator==, basic_json, json
#include <stdexcept>   // for runtime_error

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
        case N2Layout::InputANDMasked:
            j = "InputANDMasked";
            break;
        case N2Layout::InputORMasked:
            j = "InputORMasked";
            break;
        case N2Layout::GeneralSubset:
            j = "GeneralSubset";
            break;
        default:
            throw std::runtime_error(
                fmt::format("to_json - unknown N2Layout value: {:d}", static_cast<int32_t>(l)));
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
    else if (j == "InputANDMasked")
        l = N2Layout::InputANDMasked;
    else if (j == "InputORMasked")
        l = N2Layout::InputORMasked;
    else if (j == "GeneralSubset")
        l = N2Layout::GeneralSubset;
    else
        throw std::runtime_error(fmt::format("from_json - unknown N2Layout: {}", j.dump()));
}
