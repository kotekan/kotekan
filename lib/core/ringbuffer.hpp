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
#include "metadata.h" // for metadataPool

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
     * Currently, only a single producer and single consumer are supported!
     *
     * @param ring_size: the number of elements in the ring buffer to be managed
     * @param buffer_name: unique name for this buffer, from the config file declaration
     * @param buffer_type: "ring"
     */
    RingBuffer(size_t ring_size, metadataPool*, const std::string& buffer_name,
               const std::string& buffer_type);
    ~RingBuffer() override {}

    void register_consumer(const std::string& name) override;
    void register_producer(const std::string& name) override;

    /**
     * @brief Waits until the given number of elements are free to be written.
     * Must be called by a producer before writing.
     *
     * @return 0 on success, -1 if shutting down.
     */
    int wait_for_writable(const std::string& producer_name, size_t sz);

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
     * @return 0 on success, -1 if shutting down.
     */
    int wait_and_claim_readable(const std::string& consumer_name, size_t sz);

    /**
     * @brief Called by a consumer after the given number of elements
     * has been read.  This number of elements MUST match the number
     * "reserved" by the @c wait_and_claim_readable() call.
     */
    void finish_read(const std::string& consumer_name, size_t sz);

    size_t size;
    size_t elements;
    size_t claimed;
    // size_t write_cursor;
    // size_t read_cursor;
};

#endif
