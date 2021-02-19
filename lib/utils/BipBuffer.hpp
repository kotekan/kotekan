/**
 * @file
 * @brief BipBuffer data structure and associated classes for operating on it safely
 - BipBuffer
 - BipBufferWriter
 - BipWriteReservation
 - BipBufferReader
 - BipReadReservation
 */
#ifndef BIP_BUFFER_HPP
#define BIP_BUFFER_HPP

#include "gsl-lite.hpp" // for span

#include <atomic>   // for atomic_size_t
#include <cstdint>  // for uint8_t
#include <memory>   // for unique_ptr, make_unique
#include <stddef.h> // for size_t

/**
 * @class BipBuffer
 * @brief A single-producer single-consumer circular buffer that always supports writing a
 * contiguous chunk of data.
 *
 * BipBuffer, short of "bipartite buffer", is a data structure first described by Simon Cooke
 * (https://www.codeproject.com/articles/3479/the-bip-buffer-the-circular-buffer-with-a-twist). This
 * version is based on a lockless implementation described by Andrea Lattuada and James Munns
 * (https://andrea.lattuada.me/blog/2019/the-design-and-implementation-of-a-lock-free-ring-buffer-with-contiguous-reservations.html).
 *
 * This design facilitates operating on the buffer by two threads, one doing the writing, and the
 * other the reading. These threads don't work on the buffer directly, but instead use the
 * `BipBufferWriter` and `BipBufferReader` classes, respectively.
 *
 * @author Davor Cubranic
 */
class BipBuffer {
public:
    /**
     * Construct a BipBuffer with the specified size of the underlying memory store.
     */
    BipBuffer(const int size) :
        data(std::make_unique<uint8_t[]>(size)),
        len(size),
        read(0),
        write(0),
        watermark(0) {}

public:
    /// Array for the buffer data
    const std::unique_ptr<uint8_t[]> data;

    /// Length of the buffer data
    const size_t len;

    /// Head of the "read" region, starting at `data`
    std::atomic_size_t read;

    /// Head of the "write" region, starting at `data`
    std::atomic_size_t write;

    /// The value of `write` before it was wrapped back to the beginning
    std::atomic_size_t watermark;
};


/**
 * @class BipWriteReservaton
 * @brief Represents a contiguous region of memory that may be written to, and optionally
 * "committed" to the queue to make it available to the reader.
 *
 * Instances can only be created by the `BipBufferWriter`, but once created
 * provide access to the writeable region through the `start` pointer and
 * `length`.
 *
 * @author Davor Cubranic
 */
struct BipWriteReservation {
    /// Reserved writeable area
    const gsl::span<uint8_t> data;
    /// Length of the reserved writeable area
    const size_t length;

    // Allow buffer Writer access to the private members
    friend class BipBufferWriter;

private:
    /**
     * Constructor called by `BipBufferWriter`, specifying the reservation's member values
     */
    BipWriteReservation(uint8_t* const start, const size_t length, const size_t write,
                        const bool wraparound);

    /// Value of `BipBuffer.write` when the reservation was issued
    const size_t write;
    /// Flag indicating if the reservation jumped ahead of `read`
    const bool wraparound;
};


/**
 * @class BipBufferWriter
 * @brief Provides the interface to request access to continuous writeable regions of BipBuffer
 *
 * @author Davor Cubranic
 */
class BipBufferWriter {
public:
    /**
     * Constructor wrapping an underlying `BipBuffer`
     */
    BipBufferWriter(BipBuffer& buffer) : buffer(buffer) {}

    /// Reserve a contiguous region of memory in the buffer of size `length`
    /// Return nullptr if the buffer does not contain enough free space.
    std::unique_ptr<BipWriteReservation> reserve(const size_t length);

    /// Reserve a contiguous region of memory in the buffer of size up to `length`
    /// Return nullptr if the buffer does not contain no free space.
    std::unique_ptr<BipWriteReservation> reserve_max(const size_t length);

    /// Mark the reserved region available to the reader
    void commit(const BipWriteReservation& r);

private:
    /// The BipBuffer instance that this writer operates on
    BipBuffer& buffer;
};

/**
 * @class BipReadReservation
 * @brief Represents a contiguous region of memory that has the next chunk of committed data that
 * can be read. Once read, the region can released so that it is available to be reused for writing
 * again.
 *
 * Instances can only be created by the `BipBufferReader`, but once created
 * provide access to the readable region through the `start` pointer and
 * `length`.
 *
 * @author Davor Cubranic
 */

struct BipReadReservation {
    /// Reserved readable area
    const gsl::span<const uint8_t> data;
    /// For convenience, length of the reserved readable area
    const size_t length;

    // Allow buffer Writer access to the private members
    friend class BipBufferReader;

private:
    /**
     * Constructor called by `BipBufferReader`, specifying the reservation's member values
     */
    BipReadReservation(uint8_t const* const start, const size_t length, const size_t read,
                       const bool wraparound);

    /// Value of `BipBuffer.read` when the reservation was issued
    const size_t read;
    /// Flag indicating if the reservation jumped back ahead of `write`
    const bool wraparound;
};

/**
 * @class BipBufferReader
 * @brief Provides the interface to request access to continuous readable regions of BipBuffer
 *
 * @author Davor Cubranic
 */

class BipBufferReader {
public:
    /**
     * Constructor wrapping an underlying `BipBuffer`
     */
    BipBufferReader(BipBuffer& buffer) : buffer(buffer) {}

    /**
     * Request a contiguous region of data ready for reading of size `length`
     *
     * @return a unique_ptr to `BipReadReservation` for a readable region of
     * size `length`, or a `nullptr`` if the buffer does not contain enough
     * readable space.
     */
    std::unique_ptr<BipReadReservation> access(const size_t length);

    /**
     * Request a contiguous region of data ready for reading of size up to `length`
     *
     * @return `nullptr`` if the buffer does not contain any readable space, or
     * a unique_ptr to `BipReadReservation` for a readable region of size up to `length`.
     */
    std::unique_ptr<BipReadReservation> access_max(const size_t length);

    /// Mark the reserved region as read and available to the writer
    void advance(const BipReadReservation& r);

private:
    /// The BipBuffer instance that this reader operates on
    BipBuffer& buffer;
};

#endif // BIP_BUFFER_HPP
