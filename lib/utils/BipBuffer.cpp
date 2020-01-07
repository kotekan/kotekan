#include "BipBuffer.hpp"

#include "kotekanLogging.hpp" // for DEBUG2_NON_OO

#include <algorithm> // for min, max
#include <cassert>   // for assert


BipWriteReservation::BipWriteReservation(uint8_t* const start, const size_t length,
                                         const size_t write, const bool wraparound) :
    data(start, length),
    length(length),
    write(write),
    wraparound(wraparound) {}

BipReadReservation::BipReadReservation(uint8_t const* const start, const size_t length,
                                       const size_t read, const bool wraparound) :
    data(start, length),
    length(length),
    read(read),
    wraparound(wraparound) {}

std::unique_ptr<BipWriteReservation> BipBufferWriter::reserve(const size_t length) {
    if (length > buffer.len) {
        return nullptr;
    }

    auto read = buffer.read.load();
    auto write = buffer.write.load();

    if (write >= read) {
        // no wraparound: can write to the end if there is space, or try at the
        // beginning
        auto available = buffer.len - write;
        available &= -(available <= buffer.len);
        if (available >= length) {
            // reserve the space from `write` onwards
            // to advance: buffer.write.store(write + length);
            DEBUG2_NON_OO("reserve {}, starting at {}", length, write);
            return std::unique_ptr<BipWriteReservation>(
                new BipWriteReservation{buffer.data.get() + write, length, write, false});
        } else {
            DEBUG2_NON_OO("Available for wraparound: {}", read);
            if (read > length) {
                DEBUG2_NON_OO("reserve {}, starting at beginning (wraparound); watermark: {}",
                              length, write);
                // to advance:
                // 1. buffer.watermark.store(write);
                // 2. buffer.write.store(length);
                return std::unique_ptr<BipWriteReservation>(
                    new BipWriteReservation{buffer.data.get(), length, write, true});
            } else {
                DEBUG2_NON_OO("Reserving {} failed: not enough space ahead or wraparound", length);
                return nullptr; // no space available
            }
        }
    } else {
        // wraparound case, can write up to before `read`
        auto available = read - write - 1;
        available &= -(available <= read);
        DEBUG2_NON_OO("Available ahead: {} (read: {})", available, read);
        if (available >= length) {
            DEBUG2_NON_OO("reserve {}, starting at {} (up to read: {})", length, write, read);
            // to advance: buffer.write.store(write + length);
            return std::unique_ptr<BipWriteReservation>(
                new BipWriteReservation{buffer.data.get() + write, length, write, false});
        } else {
            DEBUG2_NON_OO("Reserving {} failed: not enough space ahead up to read", length);
            return nullptr; // no space available
        }
    }
}

std::unique_ptr<BipWriteReservation> BipBufferWriter::reserve_max(const size_t length) {
    auto read = buffer.read.load();
    auto write = buffer.write.load();
    if (length == 0) {
        // we can always fullfill a reservation with zero length
        return std::unique_ptr<BipWriteReservation>(
            new BipWriteReservation{buffer.data.get() + write, 0, write, false});
    }

    size_t available;
    if (write >= read) {
        // no wraparound: can write to the end if there is space, or try at the
        // beginning up to before `read`, whichever works better
        size_t available_tail = buffer.len - write;
        available_tail &= -(available_tail <= buffer.len);
        size_t available_head = read - 1;
        available_head &= -(available_head <= read);
        available = std::max(available_head, available_tail);
    } else {
        // wraparound case, can write up to before `read`
        available = read - write - 1;
        available &= -(available <= read);
    }

    DEBUG2_NON_OO("Max available: {}; write: {}, read: {}, watermark: {}", available, write, read,
                  buffer.watermark.load());
    if (available > 0) {
        return reserve(std::min(length, available));
    } else {
        return nullptr; // no space available
    }
}


void BipBufferWriter::commit(const BipWriteReservation& r) {
    assert(r.write == buffer.write.load());
    if (r.wraparound) {
        buffer.watermark.store(r.write);
        buffer.write.store(r.length);
        DEBUG2_NON_OO("Write wraparound to {}; read: {}, watermark {}", r.length,
                      buffer.read.load(), buffer.watermark.load());
    } else {
        DEBUG2_NON_OO("Commit {} to {}; watermark: {}, read: {}", r.length, r.write,
                      buffer.watermark.load(), buffer.read.load());
        // Note: we don't move the watermark because we don't know if `read` has
        // caught up to it when `write` is wrapped around.
        buffer.write.store(r.write + r.length);
    }
}


std::unique_ptr<BipReadReservation> BipBufferReader::access(const size_t length) {
    if (length > buffer.len) {
        return nullptr;
    }

    auto read = buffer.read.load();
    auto write = buffer.write.load();
    auto watermark = buffer.watermark.load();
    if (read <= write) {
        // straightforward case: write is ahead, just catch up to it
        size_t available = write - read;
        if (available < length) {
            return nullptr;
        };

        DEBUG2_NON_OO("access {}, starting at {}", length, read);
        // to advance: buffer.read.store(read + length);
        return std::unique_ptr<BipReadReservation>(
            new BipReadReservation{buffer.data.get() + read, length, read, false});
    } else if (read < watermark) {
        // write has wrapped around, but read still has some catching up to do
        size_t available = watermark - read;
        DEBUG2_NON_OO("Readable to watermark: {}", available);
        if (available < length) {
            return nullptr;
        };

        DEBUG2_NON_OO("access {}, starting at {}; watermark: {}", length, read, watermark);
        // to advance: buffer.read.store(read + length)
        return std::unique_ptr<BipReadReservation>(
            new BipReadReservation{buffer.data.get() + read, length, read, false});
    } else {
        // start from the beginning
        size_t available = write;
        DEBUG2_NON_OO("Readable at the head: {}", available);
        if (available < length) {
            return nullptr;
        };

        // to advance: buffer.read.store(length);
        return std::unique_ptr<BipReadReservation>(
            new BipReadReservation{buffer.data.get(), length, read, true});
    }
}

std::unique_ptr<BipReadReservation> BipBufferReader::access_max(const size_t length) {
    auto read = buffer.read.load();
    if (length == 0) {
        // we can always fullfill a reservation with zero length
        return std::unique_ptr<BipReadReservation>(
            new BipReadReservation{buffer.data.get() + read, 0, read, false});
    }

    auto write = buffer.write.load();
    auto watermark = buffer.watermark.load();
    size_t available;
    if (read <= write) {
        // straightforward case: write is ahead, just catch up to it
        available = write - read;
    } else if (read < watermark) {
        // write has wrapped around, but read still has some catching up to do
        available = watermark - read;
    } else {
        // start from the beginning
        available = write;
    }

    DEBUG2_NON_OO("Max readable: {}; write: {}, read: {}, watermark: {}", available, write, read,
                  buffer.watermark.load());
    if (available > 0) {
        return access(std::min(length, available));
    } else {
        return nullptr; // no space available
    }
}


void BipBufferReader::advance(const BipReadReservation& r) {
    assert(r.read == buffer.read.load()); // basic sanity check to avoid using old reservation

    if (r.wraparound) {
        DEBUG2_NON_OO("Read wraparound to {}; write: {}, watermark {}", r.length,
                      buffer.write.load(), buffer.watermark.load());
        buffer.read.store(r.length);
    } else {
        DEBUG2_NON_OO("Advance by {} from {}; write: {}, watermark {}", r.length, r.read,
                      buffer.write.load(), buffer.watermark.load());
        buffer.read.store(r.read + r.length);
    }
}
