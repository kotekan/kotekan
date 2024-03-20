#include "ringbuffer.hpp"

#include <cassert>
#include <sstream>

typedef std::lock_guard<std::recursive_mutex> buffer_lock;

// This prints out a Python literal used for making plots in post-processing, for debugging
// and illustration purposes.
static void print_py_status(const RingBuffer* const rb) {
    return;
    std::ostringstream write_heads("{");
    for (const auto& [key, value] : rb->write_heads)
        write_heads << "\"" << key << "\": " << value << ", ";
    write_heads << "}";
    std::ostringstream write_next("{");
    for (const auto& [key, value] : rb->write_next)
        write_next << "\"" << key << "\": " << value << ", ";
    write_next << "}";
    std::ostringstream read_heads("{");
    for (const auto& [key, value] : rb->read_heads)
        read_heads << "\"" << key << "\": " << value << ", ";
    read_heads << "}";
    std::ostringstream read_tails("{");
    for (const auto& [key, value] : rb->read_tails)
        read_tails << "\"" << key << "\": " << value << ", ";
    read_tails << "}";
    DEBUG_NON_OO("PY_RB rb_state(\"{:s}\", size={:L}, write_heads={:s}, write_next={:s}, "
                 "first_write_head={:L}, read_heads={:s}, read_tails={:s}, last_read_tail={:L})",
                 rb->buffer_name, rb->size, write_heads.str(), write_next.str(),
                 rb->first_write_head, read_heads.str(), read_tails.str(), rb->last_read_tail);
}

RingBuffer::RingBuffer(std::ptrdiff_t sz, std::shared_ptr<metadataPool> pool,
                       const std::string& _buffer_name, const std::string& _buffer_type) :
    GenericBuffer(_buffer_name, _buffer_type, pool, 1), size(sz), first_write_head(0),
    last_read_tail(0) {
    assert(sz > 0);
}

void RingBuffer::register_producer(const std::string& name) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    if (write_heads.find(name) != write_heads.end())
        throw std::runtime_error(fmt::format(std::locale("en_US.UTF-8"),
                                             "RingBuffer: cannot register producer \"{:s}\" - "
                                             "has already been registered!",
                                             name));
    // Start just after the first element that all other producers have already written.
    write_heads[name] = first_write_head;
    GenericBuffer::register_producer(name);
}

void RingBuffer::register_consumer(const std::string& name) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    if (read_tails.find(name) != read_tails.end())
        throw std::runtime_error(fmt::format(std::locale("en_US.UTF-8"),
                                             "RingBuffer: cannot register consumer \"{:s}\" - "
                                             "has already been registered!",
                                             name));
    // Start at the oldest valid data in the ringbuffer
    read_tails[name] = last_read_tail;
    read_heads[name] = last_read_tail;
    GenericBuffer::register_consumer(name);
}

std::optional<std::ptrdiff_t> RingBuffer::wait_without_claiming(const std::string& name,
                                                                const int inst) {
    // Wait until we can advance the read_head for this consumer
    std::unique_lock<std::recursive_mutex> lock(mutex);
    const std::ptrdiff_t old_first_write_head = first_write_head;
    const std::ptrdiff_t read_head = read_heads[name];
    DEBUG("{:s}", fmt::format(std::locale("en_US.UTF-8"),
                              "wait_without_claiming {:s}[{:d}]: initial bytes available: {:L}",
                              name, inst, first_write_head - read_head));
    while (1) {
        if (shutdown_signal) {
            DEBUG("{:s}",
                  fmt::format(std::locale("en_US.UTF-8"),
                              "wait_without_claiming {:s}[{:d}]: shutting down.", name, inst));
            return std::optional<std::ptrdiff_t>();
        }
        if (first_write_head > old_first_write_head)
            break;
        DEBUG("{:s}", fmt::format(std::locale("en_US.UTF-8"),
                                  "wait_without_claiming {:s}[{:d}]: waiting...", name, inst));
        full_cond.wait(lock);
        DEBUG("{:s}", fmt::format(std::locale("en_US.UTF-8"),
                                  "wait_without_claiming {:s}[{:d}]: waiting done.", name, inst));
    }
    const std::ptrdiff_t sz = first_write_head - read_head;
    DEBUG("{:s}", fmt::format(std::locale("en_US.UTF-8"),
                              "wait_without_claiming {:s}[{:d}]: final bytes available: {:L}", name,
                              inst, sz));
    assert(sz > 0);
    print_py_status(this);
    print_full_status();
    return std::optional<std::ptrdiff_t>(sz);
}

std::optional<std::ptrdiff_t> RingBuffer::wait_and_claim_readable(const std::string& name,
                                                                  const int inst,
                                                                  const std::ptrdiff_t sz) {
    assert(sz > 0);
    // Wait until we can advance the read_head for this consumer
    std::unique_lock<std::recursive_mutex> lock(mutex);
    const std::ptrdiff_t read_head = read_heads[name];
    DEBUG("{:s}",
          fmt::format(std::locale("en_US.UTF-8"),
                      "wait_and_claim_readable {:s}[{:d}]: requested bytes: {:L}, initial bytes "
                      "available: {:L}",
                      name, inst, sz, first_write_head - read_head));
    while (1) {
        if (shutdown_signal) {
            DEBUG("{:s}",
                  fmt::format(std::locale("en_US.UTF-8"),
                              "wait_and_claim_readable {:s}[{:d}]: shutting down.", name, inst));
            return std::optional<std::ptrdiff_t>();
        }
        if (first_write_head - read_head >= sz)
            break;
        DEBUG("{:s}", fmt::format(std::locale("en_US.UTF-8"),
                                  "wait_and_claim_readable {:s}[{:d}]: waiting...", name, inst));
        full_cond.wait(lock);
        DEBUG("{:s}", fmt::format(std::locale("en_US.UTF-8"),
                                  "wait_and_claim_readable {:s}[{:d}]: waiting done.", name, inst));
    }
    DEBUG(
        "{:s}",
        fmt::format(
            std::locale("en_US.UTF-8"),
            "wait_and_claim_readable {:s}[{:d}]: old read position: {:L}, new read position: {:L}",
            name, inst, read_head, read_head + sz));
    read_heads[name] += sz;
    print_py_status(this);
    print_full_status();
    // return the former read_head - that's where the consumer should start reading from.
    return std::optional<std::ptrdiff_t>(read_head);
}

std::optional<std::pair<std::ptrdiff_t, std::ptrdiff_t>>
RingBuffer::wait_and_claim_all_readable(const std::string& name, const int inst) {
    // Wait until we can advance the read_head for this consumer
    std::unique_lock<std::recursive_mutex> lock(mutex);
    const std::ptrdiff_t read_head = read_heads[name];
    DEBUG("{:s}",
          fmt::format(std::locale("en_US.UTF-8"),
                      "wait_and_claim_all_readable {:s}[{:d}]: initial bytes available: {:L}", name,
                      inst, first_write_head - read_head));
    while (1) {
        if (shutdown_signal) {
            DEBUG("{:s}", fmt::format(std::locale("en_US.UTF-8"),
                                      "wait_and_claim_all_readable {:s}[{:d}]: shutting down.",
                                      name, inst));
            return std::optional<std::pair<std::ptrdiff_t, std::ptrdiff_t>>();
        }
        if (first_write_head - read_head > 0)
            break;
        DEBUG("{:s}",
              fmt::format(std::locale("en_US.UTF-8"),
                          "wait_and_claim_all_readable {:s}[{:d}]: waiting...", name, inst));
        full_cond.wait(lock);
        DEBUG("{:s}",
              fmt::format(std::locale("en_US.UTF-8"),
                          "wait_and_claim_all_readable {:s}[{:d}]: waiting done.", name, inst));
    }
    const std::ptrdiff_t sz = first_write_head - read_head;
    DEBUG("{:s}",
          fmt::format(std::locale("en_US.UTF-8"),
                      "wait_and_claim_all_readable {:s}[{:d}]: final bytes available {:L}, old "
                      "read position: {:L}, new read position: {:L}",
                      name, inst, sz, read_head, read_head + sz));
    assert(sz > 0);
    read_heads[name] += sz;
    print_py_status(this);
    print_full_status();
    // return the former read_head - that's where the consumer should start reading from.
    return std::optional<std::pair<std::ptrdiff_t, std::ptrdiff_t>>(std::make_pair(read_head, sz));
}

std::optional<std::pair<std::ptrdiff_t, std::ptrdiff_t>>
RingBuffer::peek_readable(const std::string& name, const int inst) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    if (shutdown_signal) {
        DEBUG("{:s}", fmt::format(std::locale("en_US.UTF-8"),
                                  "peek_readable {:s}[{:d}]: shutting down.", name, inst));
        return std::optional<std::pair<std::ptrdiff_t, std::ptrdiff_t>>();
    }
    const std::ptrdiff_t read_head = read_heads[name];
    const std::ptrdiff_t sz = first_write_head - read_head;
    return std::optional<std::pair<std::ptrdiff_t, std::ptrdiff_t>>(std::make_pair(read_head, sz));
}

void RingBuffer::finish_read(const std::string& name, const int inst, const std::ptrdiff_t sz) {
    DEBUG("{:s}", fmt::format(std::locale("en_US.UTF-8"),
                              "finish_read {:s}[{:d}]: consumed bytes: {:L}", name, inst, sz));
    assert(sz > 0);
    // Advance the last_read_tail for this consumer
    {
        buffer_lock lock(mutex);
        const std::ptrdiff_t read_tail = read_tails[name];
        const std::ptrdiff_t read_head = read_heads[name];
        assert(read_tail + sz <= read_head);
        // Are we (one of) the reader(s) holding on to the oldest data?
        const bool old = (read_tail == last_read_tail);
        read_tails[name] += sz;
        if (old) {
            std::ptrdiff_t oldest = read_tail + sz;
            for (auto& it : read_tails)
                oldest = std::min(oldest, it.second);
            last_read_tail = oldest;
        }
    }
    DEBUG("{:s}", fmt::format(std::locale("en_US.UTF-8"),
                              "finish_read {:s}[{:d}]: new last_read_tail: {:L}", name, inst,
                              last_read_tail));
    print_py_status(this);
    print_full_status();
    empty_cond.notify_all();
}

std::optional<std::ptrdiff_t> RingBuffer::wait_for_writable(const std::string& name, const int inst,
                                                            const std::ptrdiff_t sz) {
    assert(sz > 0);
    std::unique_lock<std::recursive_mutex> lock(mutex);
    DEBUG("{:s}",
          fmt::format(
              std::locale("en_US.UTF-8"),
              "wait_for_writable {:s}[{:d}]: requested bytes: {:L}, initial bytes available: {:L}",
              name, inst, sz, size - (write_next[name] - last_read_tail)));
    while (1) {
        assert(write_next[name] >= last_read_tail);
        if (shutdown_signal) {
            DEBUG("{:s}", fmt::format(std::locale("en_US.UTF-8"),
                                      "wait_for_writable {:s}[{:d}]: shutting down.", name, inst));
            return std::optional<std::ptrdiff_t>();
        }
        if (write_next[name] - last_read_tail + sz <= size)
            break;
        DEBUG("{:s}", fmt::format(std::locale("en_US.UTF-8"),
                                  "wait_for_writable {:s}[{:d}]: waiting...", name, inst));
        empty_cond.wait(lock);
        DEBUG("{:s}", fmt::format(std::locale("en_US.UTF-8"),
                                  "wait_for_writable {:s}[{:d}]: waiting done.", name, inst));
    }
    const std::ptrdiff_t res = write_next[name];
    DEBUG("{:s}",
          fmt::format(std::locale("en_US.UTF-8"),
                      "wait_for_writable {:s}[{:d}]: final bytes available: {:L}, old write "
                      "position: {:L}, new write position: {:L}",
                      name, inst, sz, res, res + sz));
    write_next[name] += sz;
    print_py_status(this);
    print_full_status();
    return std::optional<std::ptrdiff_t>(res);
}

std::optional<std::pair<std::ptrdiff_t, std::ptrdiff_t>>
RingBuffer::get_writable(const std::string& name, const int inst) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    if (shutdown_signal) {
        DEBUG("{:s}", fmt::format(std::locale("en_US.UTF-8"),
                                  "get_writable {:s}[{:d}]: shutting down.", name, inst));
        return std::optional<std::pair<std::ptrdiff_t, std::ptrdiff_t>>();
    }
    const std::ptrdiff_t n = size - (write_next[name] - last_read_tail);
    assert(n >= 0);
    return std::optional<std::pair<std::ptrdiff_t, std::ptrdiff_t>>(
        std::make_pair(write_next[name], n));
}

void RingBuffer::finish_write(const std::string& name, const int inst, const std::ptrdiff_t sz) {
    assert(sz > 0);
    DEBUG("{:s}", fmt::format(std::locale("en_US.UTF-8"),
                              "finish_write {:s}[{:d}]: produced bytes: {:L}", name, inst, sz));
    {
        buffer_lock lock(mutex);
        // print_full_status();
        assert(write_heads[name] >= last_read_tail);
        assert(write_heads[name] + sz - last_read_tail <= size);
        const bool old = (write_heads[name] == first_write_head);
        write_heads[name] += sz;
        if (old) {
            // possibly update first_write_head with the min(write_heads)
            std::ptrdiff_t oldest = write_heads[name];
            for (auto& it : write_heads)
                oldest = std::min(oldest, it.second);
            first_write_head = oldest;
        }
    }
    DEBUG("{:s}", fmt::format(std::locale("en_US.UTF-8"),
                              "finish_write {:s}[{:d}]: new first_write_head: {:L}", name, inst,
                              first_write_head));
    print_py_status(this);
    print_full_status();
    full_cond.notify_all();
}

void RingBuffer::print_full_status() {
    buffer_lock lock(mutex);
    DEBUG("{:s}",
          fmt::format(std::locale("en_US.UTF-8"),
                      "  status: size {:L}, last_read_tail {:L}, first_write_head {:L}, "
                      "available to read: {:L}",
                      size, last_read_tail, first_write_head, first_write_head - last_read_tail));
    for (auto& it : producers) {
        const auto& name = it.second.name;
        DEBUG("{:s}", fmt::format(std::locale("en_US.UTF-8"),
                                  "    producer {:s}: first_write_head {:L}, write_next {:L}", name,
                                  write_heads[name], write_next[name]));
    }
    for (auto& it : consumers) {
        const auto& name = it.second.name;
        DEBUG("{:s}", fmt::format(std::locale("en_US.UTF-8"),
                                  "    consumer {:s}: last_read_tail {:L}, read_head {:L}", name,
                                  read_tails[name], read_heads[name]));
    }
}
