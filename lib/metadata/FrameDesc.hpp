#ifndef FRAME_DESC_HPP
#define FRAME_DESC_HPP

#include "Symbol.hpp"

#include <iostream>

namespace kotekan {

class FrameDesc {
public:
    virtual ~FrameDesc() = default;

    // A stored name for the quantity represented by this frame description
    virtual Symbol get_quantity_name() const = 0;

    // Output the array metadata, useful for logging or debugging
    virtual void output_metadata(std::ostream& os) const = 0;

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
