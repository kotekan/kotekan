#ifndef FRAME_DESC_HPP
#define FRAME_DESC_HPP

#include "Symbol.hpp"

#include <iostream>
#include <sstream>
#include <string>

namespace kotekan {

class FrameDesc {
public:
    virtual ~FrameDesc() = default;

    // A stored name for the quantity represented by this frame description
    virtual Symbol get_quantity_name() const = 0;

    // Output the frame description, useful for logging or debugging
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

    // Verify compatibility between descriptors
    virtual bool operator==(const FrameDesc& other) const = 0;

    virtual bool operator!=(const FrameDesc& other) const {
        return !(*this == other);
    }

    // Get the size of the frame in bytes
    virtual size_t get_byte_size() const = 0;

    // Strict type checking helper
    template<typename T>
    const T* as() const {
        return dynamic_cast<const T*>(this);
    }

    template<typename T>
    T* as() {
        return dynamic_cast<T*>(this);
    }
};

} // namespace kotekan

#endif // FRAME_DESC_HPP
