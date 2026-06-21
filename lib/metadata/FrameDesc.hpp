#ifndef FRAME_DESC_HPP
#define FRAME_DESC_HPP

#include "Symbol.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace kotekan {

/// Low-level byte (de)serialization helpers for the FrameDesc wire format.
/// Multi-byte values are written in host byte order (matching the existing
/// bufferSend/bufferRecv header), so the two ends must share an endianness.
namespace wire {

inline char* put_u16(char* p, uint16_t v) {
    std::memcpy(p, &v, sizeof(v));
    return p + sizeof(v);
}
inline char* put_u32(char* p, uint32_t v) {
    std::memcpy(p, &v, sizeof(v));
    return p + sizeof(v);
}
inline char* put_i64(char* p, int64_t v) {
    std::memcpy(p, &v, sizeof(v));
    return p + sizeof(v);
}
inline char* put_str(char* p, const std::string& s) {
    p = put_u32(p, static_cast<uint32_t>(s.size()));
    std::memcpy(p, s.data(), s.size());
    return p + s.size();
}
inline size_t str_size(const std::string& s) {
    return sizeof(uint32_t) + s.size();
}

/// Bounds-checked sequential reader. Every accessor throws std::runtime_error
/// rather than reading past the end, so it is safe on untrusted network input.
class Reader {
public:
    Reader(const char* data, size_t size) : p(data), end(data + size) {}
    uint16_t get_u16() {
        uint16_t v;
        take(&v, sizeof(v));
        return v;
    }
    uint32_t get_u32() {
        uint32_t v;
        take(&v, sizeof(v));
        return v;
    }
    int64_t get_i64() {
        int64_t v;
        take(&v, sizeof(v));
        return v;
    }
    std::string get_str() {
        const uint32_t n = get_u32();
        if (remaining() < n)
            throw std::runtime_error("FrameDesc wire: truncated string");
        std::string s(p, n);
        p += n;
        return s;
    }
    size_t remaining() const {
        return static_cast<size_t>(end - p);
    }
    void require_empty() const {
        if (p != end)
            throw std::runtime_error("FrameDesc wire: trailing bytes after payload");
    }

private:
    void take(void* dst, size_t n) {
        if (remaining() < n)
            throw std::runtime_error("FrameDesc wire: truncated payload");
        std::memcpy(dst, p, n);
        p += n;
    }
    const char* p;
    const char* end;
};

} // namespace wire

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

    /// @name Wire serialization
    /// A descriptor serializes as `[uint32 wire_type][payload]`: the base writes
    /// the tag and delegates the payload to the concrete subclass, and
    /// @c deserialize() reads the tag to reconstruct the matching subclass. Used
    /// to transmit a buffer's descriptor from bufferSend to bufferRecv so the
    /// receiver can validate its config-declared descriptor against the sender's.
    /// @{

    /// Stable tag identifying the concrete descriptor type on the wire.
    enum class WireType : uint32_t { generic_ndarray = 1, n2 = 2 };

    /// The wire tag for this concrete descriptor type.
    virtual WireType wire_type() const = 0;
    /// Number of payload bytes @c serialize_payload() writes (excludes the tag).
    virtual size_t serialized_payload_size() const = 0;
    /// Write @c serialized_payload_size() payload bytes to @p out.
    virtual void serialize_payload(char* out) const = 0;

    /// Total serialized size in bytes (wire tag + payload).
    size_t serialized_size() const {
        return sizeof(uint32_t) + serialized_payload_size();
    }
    /// Serialize as `[uint32 wire_type][payload]` into @p out, which must have
    /// room for at least @c serialized_size() bytes.
    void serialize(char* out) const {
        const uint32_t tag = static_cast<uint32_t>(wire_type());
        std::memcpy(out, &tag, sizeof(tag));
        serialize_payload(out + sizeof(tag));
    }
    /// Reconstruct a descriptor from bytes written by @c serialize(). Throws
    /// std::runtime_error on an unknown tag or malformed/truncated input.
    static std::shared_ptr<const FrameDesc> deserialize(const char* bytes, size_t size);
    /// @}
};

} // namespace kotekan

#endif // FRAME_DESC_HPP
