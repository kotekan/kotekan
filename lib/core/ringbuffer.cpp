#include "ringbuffer.hpp"

typedef std::lock_guard<std::recursive_mutex> buffer_lock;

RingBuffer::RingBuffer(size_t sz, metadataPool* pool, const std::string& _buffer_name,
                       const std::string& _buffer_type) :
    GenericBuffer(_buffer_name, _buffer_type, pool, 1),
    size(sz), write_head(0), write_tail(0)
{}

void RingBuffer::register_producer(const std::string& name) {
    assert(producers.size() == 0);
    if (producers.size() > 0)
        throw std::runtime_error(fmt::format(fmt("RingBuffer: cannot register producer \"{:s}\" - "
                                                 "a producer has already been registered."),
                                             name));
    GenericBuffer::register_producer(name);
}

void RingBuffer::register_consumer(const std::string& name) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    if (read_tails.find(name) != read_tails.end())
        throw std::runtime_error(fmt::format(fmt("RingBuffer: cannot register consumer \"{:s}\" - "
                                                 "has already been registered!"), name));
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
        DEBUG("Waiting for input: Want {:d}, Currently at {:d} => total {:d}, vs Written: {:d}",
              sz, head, head+sz, write_head);
        if (head + sz <= write_head)
            break;
        if (shutdown_signal)
            break;
        DEBUG("waiting on full condition variable");
        full_cond.wait(lock);
        DEBUG("finished waiting on full condition variable");
    }
    read_heads[name] += sz;
    if (shutdown_signal)
        return std::optional<size_t>();
    // return the former read_head - that's where the consumer should start reading from.
    return std::optional<size_t>(head % size);
}

void RingBuffer::finish_read(const std::string& name, size_t sz) {
    // Advance the read_tail for this consumer!
     {
        buffer_lock lock(mutex);
        size_t tail = read_tails[name];
        size_t head = read_heads[name];
        assert(tail + sz <= head);
        // Are we (one of the) reader(s) holding on to the oldest data?
        bool old = (tail == write_tail);
        DEBUG("finish_read for \"{:s}\": advancing tail from {:d} by {:d} to {:d}.  old? {:s}",
              name, tail, sz, tail+sz, old ? "yes" : "no");
        read_tails[name] += sz;
        if (old) {
            size_t oldest = tail + sz;
            for (auto& it : read_tails)
                oldest = std::min(oldest, it.second);
            DEBUG("new write_tail: {:d}", oldest);
            write_tail = oldest;
        }
    }
    empty_cond.notify_all();
}

std::optional<size_t> RingBuffer::wait_for_writable(const std::string& producer_name, size_t sz) {
    // assert this is our unique producer?
    (void)producer_name;
    std::unique_lock<std::recursive_mutex> lock(mutex);

    while (1) {
        DEBUG("Waiting to write {:d} elements.  Current write head {:d} and tail {:d} (diff {:d}), space available to write: {:d}",
              sz, write_head, write_tail, write_head - write_tail, size - (write_head - write_tail));
        if (write_head - write_tail + sz <= size)
            break;
        if (shutdown_signal)
            break;
        DEBUG("waiting for empty condition...");
        empty_cond.wait(lock);
        DEBUG("done waiting for empty condition");
    }
    if (shutdown_signal)
        return std::optional<size_t>();
    return std::optional<size_t>(write_head % size);
}

void RingBuffer::finish_write(const std::string& producer_name, size_t sz) {
    // assert this is our unique producer?
    (void)producer_name;
    {
        buffer_lock lock(mutex);
        assert(write_head + sz - write_tail <= size);
        write_head += sz;
        DEBUG("Wrote {:d}.  Now write head {:d}, tail {:d}, free space: {:d}",
              sz, write_head, write_tail, size - (write_head - write_tail));
    }
    full_cond.notify_all();
}
