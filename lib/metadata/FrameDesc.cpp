#include "FrameDesc.hpp"

#include <stdexcept> // for runtime_error
#include <string>    // for string

#include "N2FrameDesc.hpp"  // for N2FrameDesc
#include "NDArray.hpp"      // for GenericNDArray
#include "fmt.hpp"          // for format

namespace kotekan {

std::shared_ptr<const FrameDesc> FrameDesc::from_json(const nlohmann::json& j) {
    const std::string type = j.at("frame_desc_type").get<std::string>();
    if (type == "ndarray")
        return GenericNDArray::from_json(j);
    if (type == "N2")
        return N2FrameDesc::from_json(j);
    throw std::runtime_error(
        fmt::format(fmt("FrameDesc::from_json: unknown frame_desc_type '{:s}'"), type));
}

} // namespace kotekan
