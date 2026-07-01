#ifndef FRAME_DESC_HPP
#define FRAME_DESC_HPP

#include "Symbol.hpp"

#include "json.hpp" // for json

#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace kotekan {

class FrameDesc {
public:
    virtual ~FrameDesc() = default;

    /// A stored name for the quantity represented by this frame description
    virtual Symbol get_quantity_name() const = 0;

    /// Output the frame description, useful for logging or debugging
    virtual void output_framedesc(std::ostream& os) const = 0;

    /// Describe how this descriptor differs from `other`, for error messages.
    /// The default prints both descriptions; subclasses may return a more
    /// targeted description.
    virtual std::string describe_mismatch(const FrameDesc& other) const {
        std::ostringstream this_os, other_os;
        output_framedesc(this_os);
        other.output_framedesc(other_os);
        return "existing:\n" + this_os.str() + "new:\n" + other_os.str();
    }

    /// Verify compatibility between descriptors
    virtual bool operator==(const FrameDesc& other) const = 0;

    virtual bool operator!=(const FrameDesc& other) const {
        return !(*this == other);
    }

    /// Get the size of the frame in bytes
    virtual size_t get_byte_size() const = 0;

    /// Strict type checking helper
    template<typename T>
    const T* as() const {
        return dynamic_cast<const T*>(this);
    }

    template<typename T>
    T* as() {
        return dynamic_cast<T*>(this);
    }

    /// @name JSON serialization
    /// A descriptor serializes to a self-describing JSON object carrying a
    /// `"frame_desc_type"` discriminator; @c from_json() reads it to reconstruct
    /// the matching subclass. Used to transmit a buffer's descriptor from
    /// bufferSend to bufferRecv so the receiver can validate its config-declared
    /// descriptor against the sender's. This is a one-per-connection control
    /// message, so JSON's convenience is preferred over a compact binary form.
    /// @{

    /// Serialize this descriptor to a JSON object (includes @c frame_desc_type).
    virtual nlohmann::json to_json() const = 0;

    /// Reconstruct a descriptor from JSON written by @c to_json(). Throws
    /// std::runtime_error on an unknown type or malformed/missing fields.
    static std::shared_ptr<const FrameDesc> from_json(const nlohmann::json& j);
    /// @}
};

} // namespace kotekan

#endif // FRAME_DESC_HPP
