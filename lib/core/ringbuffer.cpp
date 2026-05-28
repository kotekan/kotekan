#include "ringbuffer.hpp"

#include "kotekanLogging.hpp" // for DEBUG2, logLevel, DEBUG_NON_OO

#include "fmt.hpp" // for compile_string_to_view, group_digits, format, format_string

#include <algorithm>          // for min
#include <cassert>            // for assert
#include <condition_variable> // for condition_variable_any
#include <mutex>              // for unique_lock, recursive_mutex, lock_guard
#include <sstream>            // for basic_ostream, operator<<, basic_ostringstream, ostringstream
#include <stdexcept>          // for runtime_error

typedef std::lock_guard<std::recursive_mutex> buffer_lock;

using fmt::group_digits;

// This prints out a Python literal used for making plots in post-processing, for debugging
// and illustration purposes.
[[maybe_unused]] static void print_py_status(const RingBuffer* const rb) {
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
    DEBUG_NON_OO("PY_RB rb_state(\"{:s}\", size={}, write_heads={:s}, write_next={:s}, "
                 "first_write_head={}, read_heads={:s}, read_tails={:s}, last_read_tail={})",
                 rb->buffer_name, group_digits(rb->size), write_heads.str(), write_next.str(),
                 group_digits(rb->first_write_head), read_heads.str(), read_tails.str(),
                 group_digits(rb->last_read_tail));
}

RingBuffer::RingBuffer(std::ptrdiff_t sz, std::shared_ptr<metadataPool> pool,
                       const std::string& _buffer_name, const std::string& _buffer_type) :
    GenericBuffer(_buffer_name, _buffer_type, pool, 1), size(sz), first_write_head(0),
    last_read_tail(0) {
    assert(sz > 0);
}

void RingBuffer::register_producer(const std::string& name) {
    buffer_lock lock(mutex);
    if (write_heads.find(name) != write_heads.end())
        throw std::runtime_error(fmt::format("RingBuffer: cannot register producer \"{:s}\" for "
                                             "ringbuffer \"{:s}\" - has already been registered!",
                                             name, buffer_name));
    // Start just after the first element that all other producers have already written.
    write_heads[name] = first_write_head;
    GenericBuffer::register_producer(name);
}

void RingBuffer::register_consumer(const std::string& name) {
    buffer_lock lock(mutex);
    if (read_tails.find(name) != read_tails.end())
        throw std::runtime_error(fmt::format("RingBuffer: cannot register consumer \"{:s}\" for "
                                             "ringbuffer \"{:s}\" - has already been registered!",
                                             name, buffer_name));
    // Start at the oldest valid data in the ringbuffer
    read_tails[name] = last_read_tail;
    read_heads[name] = last_read_tail;
    GenericBuffer::register_consumer(name);
}

std::optional<std::ptrdiff_t> RingBuffer::wait_without_claiming(const std::string& name,
                                                                [[maybe_unused]] const int inst,
                                                                const std::ptrdiff_t sz) {
    // Wait until we can advance the read_head for this consumer
    std::unique_lock<std::recursive_mutex> lock(mutex);
    const std::ptrdiff_t read_head = read_heads[name];
    DEBUG2("Ringbuffer {:s}: wait_without_claiming({:s}[{:d}]): "
           "requested bytes: {}, initial bytes available: {}",
           buffer_name, name, inst, group_digits(sz), group_digits(first_write_head - read_head));
    DEBUG2("Ringbuffer {:s}: wait_without_claiming({:s}[{:d}]): waiting...", buffer_name, name,
           inst);
    print_full_status();
    full_cond.wait(lock, [&]() { return shutdown_signal || first_write_head - read_head >= sz; });
    DEBUG2("Ringbuffer {:s}: wait_without_claiming({:s}[{:d}]): waiting done.", buffer_name, name,
           inst);
    if (shutdown_signal) {
        return std::optional<std::ptrdiff_t>();
    }
    assert(first_write_head - read_head >= sz);
    const std::ptrdiff_t have_sz = first_write_head - read_head;
    DEBUG2("Ringbuffer {:s}: wait_without_claiming({:s}[{:d}]): "
           "final bytes available: {}",
           buffer_name, name, inst, group_digits(have_sz));
    return std::optional<std::ptrdiff_t>(have_sz);
}

std::optional<std::ptrdiff_t> RingBuffer::wait_and_claim_readable(const std::string& name,
                                                                  [[maybe_unused]] const int inst,
                                                                  const std::ptrdiff_t sz) {
    assert(sz > 0);
    // Wait until we can advance the read_head for this consumer
    std::unique_lock<std::recursive_mutex> lock(mutex);
    const std::ptrdiff_t read_head = read_heads[name];
    DEBUG2("wait_and_claim_readable({:s}[{:d}]): "
           "requested bytes: {}, initial bytes available: {}",
           name, inst, group_digits(sz), group_digits(first_write_head - read_head));
    DEBUG2("wait_and_claim_readable({:s}[{:d}]): waiting...", name, inst);
    print_full_status();
    full_cond.wait(lock, [&]() { return shutdown_signal || first_write_head - read_head >= sz; });
    DEBUG2("wait_and_claim_readable({:s}[{:d}]): waiting done.", name, inst);
    if (shutdown_signal) {
        return std::optional<std::ptrdiff_t>();
    }
    assert(first_write_head - read_head >= sz);
    DEBUG2("wait_and_claim_readable({:s}[{:d}]): final bytes available: {}", name, inst,
           group_digits(sz));
    read_heads[name] += sz;
    // return the former read_head - that's where the consumer should start reading from.
    return std::optional<std::ptrdiff_t>(read_head);
}

std::optional<std::pair<std::ptrdiff_t, std::ptrdiff_t>>
RingBuffer::wait_and_claim_all_readable(const std::string& name, [[maybe_unused]] const int inst) {
    // Wait until we can advance the read_head for this consumer
    std::unique_lock<std::recursive_mutex> lock(mutex);
    const std::ptrdiff_t read_head = read_heads[name];
    DEBUG2("wait_and_claim_all_readable({:s}[{:d}]): "
           "initial bytes available: {}",
           name, inst, group_digits(first_write_head - read_head));
    DEBUG2("wait_and_claim_all_readable({:s}[{:d}]): waiting...", name, inst);
    print_full_status();
    full_cond.wait(lock, [&]() { return shutdown_signal || first_write_head - read_head > 0; });
    DEBUG2("wait_and_claim_all_readable({:s}[{:d}]): waiting done.", name, inst);
    if (shutdown_signal) {
        return std::optional<std::pair<std::ptrdiff_t, std::ptrdiff_t>>();
    }
    assert(first_write_head - read_head > 0);
    const std::ptrdiff_t sz = first_write_head - read_head;
    DEBUG2("wait_and_claim_all_readable({:s}[{:d}]): "
           "final bytes available: {}",
           name, inst, group_digits(sz));
    assert(sz > 0);
    read_heads[name] += sz;
    // return the former read_head - that's where the consumer should start reading from.
    return std::optional<std::pair<std::ptrdiff_t, std::ptrdiff_t>>(std::make_pair(read_head, sz));
}

std::optional<std::pair<std::ptrdiff_t, std::ptrdiff_t>>
RingBuffer::peek_readable(const std::string& name, [[maybe_unused]] const int inst) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    if (shutdown_signal) {
        return std::optional<std::pair<std::ptrdiff_t, std::ptrdiff_t>>();
    }
    const std::ptrdiff_t read_head = read_heads[name];
    const std::ptrdiff_t sz = first_write_head - read_head;
    DEBUG2("peek_readable({:s}[{:d}]): "
           "bytes available: {}",
           name, inst, group_digits(sz));
    return std::optional<std::pair<std::ptrdiff_t, std::ptrdiff_t>>(std::make_pair(read_head, sz));
}

void RingBuffer::finish_read(const std::string& name, [[maybe_unused]] const int inst,
                             const std::ptrdiff_t sz) {
    DEBUG2("finish_read({:s}[{:d}]): "
           "consumed bytes: {}",
           name, inst, group_digits(sz));
    assert(sz > 0);
    // Advance the last_read_tail for this consumer
    {
        std::unique_lock<std::recursive_mutex> lock(mutex);
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
    empty_cond.notify_all();
}

std::optional<std::ptrdiff_t> RingBuffer::wait_for_writable(const std::string& name,
                                                            [[maybe_unused]] const int inst,
                                                            const std::ptrdiff_t sz) {
    assert(sz > 0);
    std::unique_lock<std::recursive_mutex> lock(mutex);
    DEBUG2("wait_for_writable({:s}[{:d}]): "
           "requested bytes: {}, initial bytes available: {}",
           name, inst, group_digits(sz), group_digits(size - (write_next[name] - last_read_tail)));
    DEBUG2("wait_for_writable({:s}[{:d}]): waiting...", name, inst);
    print_full_status();
    empty_cond.wait(
        lock, [&]() { return shutdown_signal || write_next[name] - last_read_tail + sz <= size; });
    DEBUG2("wait_for_writable({:s}[{:d}]): waiting done.", name, inst);
    assert(write_next[name] >= last_read_tail);
    if (shutdown_signal) {
        return std::optional<std::ptrdiff_t>();
    }
    assert(write_next[name] - last_read_tail + sz <= size);
    const std::ptrdiff_t res = write_next[name];
    DEBUG2("wait_for_writable({:s}[{:d}]): "
           "final bytes available: {}",
           name, inst, group_digits(sz));
    write_next[name] += sz;
    return std::optional<std::ptrdiff_t>(res);
}

std::optional<std::pair<std::ptrdiff_t, std::ptrdiff_t>>
RingBuffer::get_writable(const std::string& name, [[maybe_unused]] const int inst) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    if (shutdown_signal) {
        return std::optional<std::pair<std::ptrdiff_t, std::ptrdiff_t>>();
    }
    const std::ptrdiff_t n = size - (write_next[name] - last_read_tail);
    assert(n >= 0);
    DEBUG2("get_writable({:s}[{:d}]): "
           "bytes available: {}",
           name, inst, group_digits(n));
    return std::optional<std::pair<std::ptrdiff_t, std::ptrdiff_t>>(
        std::make_pair(write_next[name], n));
}

void RingBuffer::finish_write(const std::string& name, [[maybe_unused]] const int inst,
                              const std::ptrdiff_t sz) {
    assert(sz > 0);
    std::unique_lock<std::recursive_mutex> lock(mutex);
    DEBUG2("finish_write({:s}[{:d}]): "
           "produced bytes: {}",
           name, inst, group_digits(sz));
    print_full_status();
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
    DEBUG2("finish_write {:s}[{:d}]: new first_write_head: {}", name, inst,
           group_digits(first_write_head));

    if (read_tails.empty()) {
        // if there are no consumers then mark all available data as consumed
        last_read_tail = first_write_head;
    } else {
        // notify consumers
        full_cond.notify_all();
    }
}

void RingBuffer::print_buffer_status() {
    std::ptrdiff_t read_tail, read_head;
    std::ptrdiff_t write_tail, write_head;
    {
        std::unique_lock<std::recursive_mutex> lock(mutex);
        read_tail = last_read_tail;
        read_head =
            read_heads.empty()
                ? last_read_tail
                : std::max_element(read_heads.begin(), read_heads.end(),
                                   [](const auto& a, const auto& b) { return a.second < b.second; })
                      ->second;
        write_tail = first_write_head;
        write_head =
            write_next.empty()
                ? first_write_head
                : std::max_element(write_next.begin(), write_next.end(),
                                   [](const auto& a, const auto& b) { return a.second < b.second; })
                      ->second;
    }
    assert(read_tail <= read_head);
    assert(read_head <= write_tail);
    assert(write_tail <= write_head);
    assert(write_head <= read_tail + size);
    const std::ptrdiff_t emptying_samples = read_head - read_tail;
    const std::ptrdiff_t full_samples = write_tail - read_head;
    const std::ptrdiff_t filling_samples = write_head - write_tail;
    const std::ptrdiff_t empty_samples = read_tail + size - write_head;
    const std::ptrdiff_t total_samples = size;
    const int buffer_display_length = 40;
    using std::lrint;
    const int emptying_count =
        lrint(1.0 * buffer_display_length * emptying_samples / total_samples);
    const int full_count =
        lrint(1.0 * buffer_display_length * (emptying_samples + full_samples) / total_samples)
        - emptying_count;
    const int filling_count =
        lrint(1.0 * buffer_display_length * (emptying_samples + full_samples + filling_samples)
              / total_samples)
        - (emptying_count + full_count);
    const int empty_count =
        lrint(1.0 * buffer_display_length
              * (emptying_samples + full_samples + filling_samples + empty_samples) / total_samples)
        - (emptying_count + full_count + filling_count);
    // const std::string emptying_symbol = "▼";
    // const std::string full_symbol = "█";
    // const std::string filling_symbol = "▲";
    // const std::string empty_symbol = "·";
    const char emptying_symbol = "v";
    const char full_symbol = "X";
    const char filling_symbol = "^";
    const char empty_symbol = ".";
    std::ostringstream buf;
    buf << "[";
    for (int n = 0; n < emptying_count; ++n)
        buf << emptying_symbol;
    for (int n = 0; n < full_count; ++n)
        buf << full_symbol;
    for (int n = 0; n < filling_count; ++n)
        buf << filling_symbol;
    for (int n = 0; n < empty_count; ++n)
        buf << empty_symbol;
    buf << "]";
    INFO_NON_OO("Buffer {:40s}, status: {:s}", buffer_name, buf.str());
}

void RingBuffer::print_full_status() {
    // Don't compute a lot of strings if we aren't going to output the result
    if (get_log_level() < kotekan::logLevel::DEBUG2)
        return;

    std::unique_lock<std::recursive_mutex> lock(mutex);
    static std::recursive_mutex print_mutex;
    std::lock_guard<std::recursive_mutex> print_lock(print_mutex);
    DEBUG2("--------------------- {:s} ---------------------", buffer_name.c_str());
    DEBUG2("{:<40} : {:13.6f} MB", "size", size / 1.0e+6);
    DEBUG2("{:<40} : {:13.6f} MB", "last_read_tail", last_read_tail / 1.0e+6);
    DEBUG2("{:<40} : {:13.6f} MB", "first_write_head", first_write_head / 1.0e+6);
    DEBUG2("{:<40} : {:13.6f} MB", "available to read",
           (first_write_head - last_read_tail) / 1.0e+6);
    DEBUG2("{:<40} : {:13.6f} MB", "free space to write",
           (size - (first_write_head - last_read_tail)) / 1.0e+6);
    DEBUG2("---- Producers ({:d}) ----", producers.size());
    for (auto& it : producers) {
        const auto& name = it.second.name;
        DEBUG2("{:<40} : {:13.6f} MB ... {:.6f} MB ({:.6f} MB)", name.c_str(),
               write_heads[name] / 1.0e+6, write_next[name] / 1.0e+6,
               (write_next[name] - write_heads[name]) / 1.0e+6);
    }
    DEBUG2("---- Consumers ({:d}) ----", consumers.size());
    for (auto& it : consumers) {
        const auto& name = it.second.name;
        DEBUG2("{:<40} : {:13.6f} MB ... {:.6f} MB ({:.6f} MB)", name.c_str(),
               read_tails[name] / 1.0e+6, read_heads[name] / 1.0e+6,
               (read_heads[name] - read_tails[name]) / 1.0e+6);
    }
}
