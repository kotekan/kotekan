/**
 * @file
 * @brief A core kotekan buffer subclass that synchronizes stages that communicate via a ring
 * buffer.
 *  - RingBuffer
 */
#ifndef RINGBUFFER_H
#define RINGBUFFER_H

#include "buffer.hpp"
#include "kotekanLogging.hpp"
#include "metadata.hpp" // for metadataPool

#include <optional>
#include <utility>

/**
 * @brief A buffer to manage the signalling between stages when those
 * stages want to communicate data using a ring buffer.
 *
 * This class is a little strange in that it doesn't actually allocate
 * any memory!  It was created to connect GPU pipelines that need to
 * communicate via a ring buffer in GPU memory.  This class knows how
 * big the buffer is and controls read and write access for the
 * consumers and producers, but does not actually own the memory.
 *
 * Producers and consumers can produce or consume different numbers of
 * elements, so this is a helpful way of connecting stages that
 * operate at different cadences.
 *
 * In the yaml config file, one of these can be created by setting the
 * @c kotekan_buffer value to @c "ring".
 *
 * @conf ring_buffer_size: the number of elements in the ring buffer
 * @conf metadata_pool The name of the metadata pool to associate with the buffer
 */
class RingBuffer : public GenericBuffer {
public:
    /**
     * @brief Build a new RingBuffer for connecting kotekan stages
     * that want to produce & consume data with different sizes or at
     * different cadences.
     *
     * Multiple consumers and multiple producers are supported.  For
     * multiple producers, the assumptions is that every producer has
     * to finish writing to a region before it is "done"; the idea is
     * that each producer is writing a subset of the data for each
     * time sample.  For multiple consumers, each consumer gets a copy
     * of the data stream.
     *
     * @param ring_size: the number of elements in the ring buffer to be managed
     * @param buffer_name: unique name for this buffer, from the config file declaration
     * @param buffer_type: "ring"
     */
    RingBuffer(size_t ring_size, std::shared_ptr<metadataPool>, const std::string& buffer_name,
               const std::string& buffer_type);
    ~RingBuffer() override {}

    void register_consumer(const std::string& name) override;
    void register_producer(const std::string& name) override;

    void print_full_status() override;

    /**
     * @brief Waits until the given number of elements are free to be written.
     * Must be called by a producer before writing.
     *
     * @return A std::optional<size_t>, where there is a value on
     *   success, and no value if the pipeline is shutting down.  The
     *   value is the write cursor: the offset in the array where the
     *   producer should start writing.
     */
    std::optional<size_t> wait_for_writable(const std::string& producer_name, size_t sz);

    /**
     * @brief Checks many elements are free to be written.
     *
     * @return A std::optional<std::pair<size_t, size_t> >, where there
     * is no value if the pipeline is shutting down, and otherwise, a pair
     * giving the write cursor and number of elements that are writable.
     */
    std::optional<std::pair<size_t, size_t>> get_writable(const std::string& producer_name);

    /**
     * @brief Called by a producer after it has written the given number of
     * elements.  Those elements will becomes available to consumers.
     */
    void finish_write(const std::string& producer_name, size_t sz);

    /**
     * @brief Called by a consumer before reading its next chunk of
     * data.  Waits until the given number of elements have been
     * produced, AND reserves those elements -- they will be treated as
     * unavailable by subsequent @c wait_and_claim_readable calls.
     *
     * @return A value on success, no value if the pipeline is
     * shutting down.  On success, the returned value is the read
     * cursor: the offset in the ring buffer where the consumer should
     * start reading.
     */
    std::optional<size_t> wait_and_claim_readable(const std::string& consumer_name, size_t sz);

    /**
     * @brief Checks many elements are free to be read, BUT does not
     * advance the read head.
     *
     * @return A std::optional<std::pair<size_t, size_t> >, where there
     * is no value if the pipeline is shutting down, and otherwise, a pair
     * giving the read cursor and number of elements that are readable.
     */
    std::optional<std::pair<size_t, size_t>> peek_readable(const std::string& consumer_name);

    /**
     * @brief Called by a consumer after the given number of elements
     * has been read.  This number of elements MUST match the number
     * "reserved" by the @c wait_and_claim_readable() call.
     */
    void finish_read(const std::string& consumer_name, size_t sz);

    // The size of the ring buffer (maximum number of elements in the buffer).
    size_t size;

    // The "write_head" for a producer is the index of the next element to be written,
    // also 1 greater than the index of the last valid element that can be read by consumers.
    // The "wait_for_writable()" will return the "write_head" when enough space is available.
    // The "write_head" advances when "finish_write()" is called.
    std::map<std::string, size_t> write_heads;

    // The overall "write_head" is the min(write_heads), ie, 1 more than the last element that
    // has been written by all producers (and therefore is available to be consumed).
    size_t write_head;

    // "write_tail" is the index of the first valid element that can be read by consumers.
    // The "write_tail" may be updated in "finish_read()" when the last consumer has finished
    // reading a chunk of data.  "write_tail = min(read_tails)".
    size_t write_tail;

    // The *current* "read_head" for a consumer is the next index that client will be told to read.
    // (ie, what the next call to wait_and_claim_readable() will return)
    // (ie, 1 greater than the largest index it is *currently* allowed to read)
    std::map<std::string, size_t> read_heads;

    // The "read_tail" for a consumer is the index of the first item
    // that it has requested to read, but not called "finish_read" on.
    // Elements between the "read_tail" and "read_head" are available to be read.
    // The "read_tail" is advanced when "finish_read()" is called.
    std::map<std::string, size_t> read_tails;
};

#endif
