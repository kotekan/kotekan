#include "ringbuffer.hpp"

typedef std::lock_guard<std::recursive_mutex> buffer_lock;

static void print_py_status(RingBuffer* rb) {
    std::string write_heads = "{";
    for (const auto& [key, value] : rb->write_heads)
        write_heads += "\"" + key + "\": " + std::to_string(value) + ", ";
    write_heads += "}";
    std::string write_next = "{";
    for (const auto& [key, value] : rb->write_next)
        write_next += "\"" + key + "\": " + std::to_string(value) + ", ";
    write_next += "}";
    std::string read_heads = "{";
    for (const auto& [key, value] : rb->read_heads)
        read_heads += "\"" + key + "\": " + std::to_string(value) + ", ";
    read_heads += "}";
    std::string read_tails = "{";
    for (const auto& [key, value] : rb->read_tails)
        read_tails += "\"" + key + "\": " + std::to_string(value) + ", ";
    read_tails += "}";
    DEBUG_NON_OO("PY_RB rb_state(\"{:s}\", size={:d}, write_heads={:s}, write_next={:s}, write_head={:d}, write_tail={:d}, read_heads={:s}, read_tails={:s})", rb->buffer_name, rb->size, write_heads, write_next, rb->write_head, rb->write_tail, read_heads, read_tails);
    //DEBUG_NON_OO("PY_RB rb_state(\"{:s}\", size={:d}, write_heads={:s})", rb->buffer_name, rb->size, write_heads);
}

RingBuffer::RingBuffer(size_t sz, std::shared_ptr<metadataPool> pool,
                       const std::string& _buffer_name, const std::string& _buffer_type) :
    GenericBuffer(_buffer_name, _buffer_type, pool, 1),
    size(sz), write_head(0), write_tail(0) {}

void RingBuffer::register_producer(const std::string& name) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    if (write_heads.find(name) != write_heads.end())
        throw std::runtime_error(fmt::format(fmt("RingBuffer: cannot register producer \"{:s}\" - "
                                                 "has already been registered!"),
                                             name));
    // Start just after the first element that all other producers have already written.
    write_heads[name] = write_head;
    GenericBuffer::register_producer(name);
}

void RingBuffer::register_consumer(const std::string& name) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    if (read_tails.find(name) != read_tails.end())
        throw std::runtime_error(fmt::format(fmt("RingBuffer: cannot register consumer \"{:s}\" - "
                                                 "has already been registered!"),
                                             name));
    // Start at the oldest valid data in the ringbuffer
    read_tails[name] = write_tail;
    read_heads[name] = write_tail;
    GenericBuffer::register_consumer(name);
}

std::optional<size_t> RingBuffer::wait_and_claim_readable(const std::string& name, size_t sz) {
    // Wait until we can advance the read_head for this consumer!
    std::unique_lock<std::recursive_mutex> lock(mutex);
    size_t head = read_heads[name];
    while (1) {
        std::string d = fmt::format(
            std::locale("en_US.UTF-8"),
            "Waiting for input: Want {:L}, Have {:L}.  (Current read head: {:L}, after this read "
            "would be {:L}, current write head: {:L})",
            sz, write_head - head, head, head + sz, write_head);
        DEBUG("{:s}", d);
        if (head + sz <= write_head)
            break;
        if (shutdown_signal)
            break;
        DEBUG("waiting on full condition variable");
        full_cond.wait(lock);
        DEBUG("finished waiting on full condition variable");
    }
    INFO("wait_and_claim_readable {:s} {:L} -> {:L} {:L}", name, sz, head, head % size);
    read_heads[name] += sz;
    if (shutdown_signal)
        return std::optional<size_t>();
    print_py_status(this);
    // return the former read_head - that's where the consumer should start reading from.
    return std::optional<size_t>(head % size);
}

std::optional<std::pair<size_t, size_t>> RingBuffer::peek_readable(const std::string& name) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    if (shutdown_signal)
        return std::optional<std::pair<size_t, size_t>>();
    size_t head = read_heads[name];
    return std::optional<std::pair<size_t, size_t>>(std::make_pair(head % size, write_head - head));
}

void RingBuffer::finish_read(const std::string& name, size_t sz) {
    // Advance the read_tail for this consumer!
    {
        buffer_lock lock(mutex);
        size_t tail = read_tails[name];
        size_t head = read_heads[name];
        assert(tail + sz <= head);
        // Are we (one of) the reader(s) holding on to the oldest data?
        bool old = (tail == write_tail);
        DEBUG("finish_read for \"{:s}\": advancing tail from {:L} by {:L} to {:L}.  old? {:s}",
              name, tail, sz, tail + sz, old ? "yes" : "no");
        read_tails[name] += sz;
        if (old) {
            size_t oldest = tail + sz;
            for (auto& it : read_tails)
                oldest = std::min(oldest, it.second);
            DEBUG("new write_tail: {:L}", oldest);
            write_tail = oldest;
        }
    }
    print_py_status(this);
    INFO("finish_read {:s} {:L} write_tail now {:L}", name, sz, write_tail);
    empty_cond.notify_all();
}

std::optional<size_t> RingBuffer::wait_for_writable(const std::string& name, size_t sz) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    while (1) {
        DEBUG("Waiting to write {:L} elements.  Current write_next {:L} write_head {:L} and "
              "write_tail {:L} (next - tail diff {:L}), space available to write: {:L}",
              sz, write_next[name], write_heads[name], write_tail, write_next[name] - write_tail,
              size - (write_next[name] - write_tail));
        if (write_next[name] - write_tail + sz <= size)
            break;
        if (shutdown_signal)
            break;
        DEBUG("waiting for empty condition...");
        empty_cond.wait(lock);
        DEBUG("done waiting for empty condition");
    }
    if (shutdown_signal)
        return std::optional<size_t>();
    size_t rtn = write_next[name] % size;
    DEBUG("wait_for_writable {:s} {:L} -> {:L} {:L}", name, sz, write_next[name],
          rtn);
    write_next[name] += sz;
    print_py_status(this);
    return std::optional<size_t>(rtn);
}

std::optional<std::pair<size_t, size_t>> RingBuffer::get_writable(const std::string& name) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    if (shutdown_signal)
        return std::optional<std::pair<size_t, size_t>>();
    size_t n = size - (write_next[name] - write_tail);
    return std::optional<std::pair<size_t, size_t>>(std::make_pair(write_next[name] % size, n));
}

void RingBuffer::finish_write(const std::string& name, size_t sz) {
    DEBUG("finish_write {:s} {:L}", name, sz);
    {
        buffer_lock lock(mutex);
        print_full_status();
        assert(write_heads[name] + sz - write_tail <= size);
        bool old = (write_heads[name] == write_head);
        write_heads[name] += sz;
        DEBUG("Wrote {:L}.  Now write head {:L}, tail {:L}, free space: {:L}", sz,
              write_heads[name], write_tail, size - (write_next[name] - write_tail));
        if (old) {
            // possibly update write_head with the min(write_heads)
            size_t oldest = write_heads[name];
            for (auto& it : write_heads)
                oldest = std::min(oldest, it.second);
            DEBUG("new write_head: {:d}", oldest);
            write_head = oldest;
        }
    }
    print_py_status(this);
    // DEBUG("finish_write {:s} {:L} write_head now {:L}", name, sz, write_head);
    full_cond.notify_all();
}

void RingBuffer::print_full_status() {
    buffer_lock lock(mutex);
    INFO("Status:  size {:d}, write_tail {:d} ({:d}), write_head {:d} ({:d}), available to be "
         "read: {:d}",
         size, write_tail, write_tail % size, write_head, write_head % size,
         write_head - write_tail);
    for (auto& it : producers) {
        std::string name = it.second.name;
        INFO("  producer \"{:s}\": write_head {:d} ({:d}), write_next {:d} ({:d})", name,
             write_heads[name], write_heads[name] % size, write_next[name],
             write_next[name] % size);
    }
    for (auto& it : consumers) {
        std::string name = it.second.name;
        INFO("  consumer \"{:s}\": read_tail {:d} ({:d}), read_head {:d} ({:d})", name,
             read_tails[name], read_tails[name] % size, read_heads[name], read_heads[name] % size);
    }
}
