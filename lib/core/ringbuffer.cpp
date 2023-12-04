#include "ringbuffer.hpp"

typedef std::lock_guard<std::recursive_mutex> buffer_lock;

RingBuffer::RingBuffer(size_t sz, metadataPool* pool, const std::string& _buffer_name,
                       const std::string& _buffer_type) :
    GenericBuffer(_buffer_name, _buffer_type, pool, 1),
    size(sz), elements(0), claimed(0)
// write_cursor(0),
// read_cursor(0)
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
    assert(consumers.size() == 0);
    if (consumers.size() > 0)
        throw std::runtime_error(fmt::format(fmt("RingBuffer: cannot register consumer \"{:s}\" - "
                                                 "a consumer has already been registered."),
                                             name));
    GenericBuffer::register_consumer(name);
}

int RingBuffer::wait_and_claim_readable(const std::string& consumer_name, size_t sz) {
    // assert this is our unique consumer?
    (void)consumer_name;
    std::unique_lock<std::recursive_mutex> lock(mutex);
    while (1) {
        DEBUG("waiting for input: Want {:d}, Available: {:d} - Claimed: {:d}", sz, elements,
              claimed);
        if (elements - claimed >= sz)
            break;
        if (shutdown_signal)
            break;
        // FIXME???
        // if (buf->consumers_done[ID][consumer_id] == 1)
        // break;
        DEBUG("waiting on full condition variable");
        full_cond.wait(lock);
        DEBUG("finished waiting on full condition variable");
    }
    // Claim!
    claimed += sz;
    lock.unlock();

    if (shutdown_signal)
        return -1;
    // FIXME .... + read_cursor here ??
    return 0;
}

void RingBuffer::finish_read(const std::string& consumer_name, size_t sz) {
    // assert this is our unique consumer?
    (void)consumer_name;
    {
        buffer_lock lock(mutex);
        assert(sz >= claimed);
        claimed -= sz;
        elements -= sz;
        DEBUG("Read {:d}.  Now {:d} elements, {:d} claimed", sz, elements, claimed);
        // FIXME ??
        // read_cursor = (read_cursor + sz) % size;
    }
    empty_cond.notify_all();
}

int RingBuffer::wait_for_writable(const std::string& producer_name, size_t sz) {
    // assert this is our unique producer?
    (void)producer_name;
    std::unique_lock<std::recursive_mutex> lock(mutex);

    while (1) {
        DEBUG("waiting to write {:d} elements.  Current elements filled: {:d}", sz, elements);
        if (elements + sz <= size)
            break;
        if (shutdown_signal)
            break;
        DEBUG("waiting for empty condition...");
        empty_cond.wait(lock);
        DEBUG("done waiting for empty condition");
    }
    lock.unlock();
    if (shutdown_signal)
        return -1;
    return 0;
}

void RingBuffer::finish_write(const std::string& producer_name, size_t sz) {
    // assert this is our unique producer?
    (void)producer_name;
    {
        buffer_lock lock(mutex);
        assert(elements + sz <= size);
        elements += sz;
        DEBUG("Wrote {:d}.  Now {:d} elements available", sz, elements);
        // FIXME ??
        // write_cursor = (write_cursor + sz) % size;
    }
    full_cond.notify_all();
}
