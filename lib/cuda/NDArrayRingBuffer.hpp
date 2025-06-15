#ifndef NDARRAYRINGBUFFER_HPP
#define NDARRAYRINGBUFFER_HPP

#include <DataType.hpp>
#include <NDArray.hpp>
#include <Symbol.hpp>
#include <array>
#include <buffer.hpp>
#include <cassert>
#include <chordMetadata.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cudaCommand.hpp>
#include <cudaDeviceInterface.hpp>
#include <div.hpp>
#include <functional>
#include <gpuDeviceInterface.hpp>
#include <kotekanLogging.hpp>
#include <ostream>
#include <ringbuffer.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using kotekan::div_noremainder;

struct read_descriptor_t {
    std::ptrdiff_t claimed, read;
};

template<typename T, std::size_t D>
class NDArrayRingBuffer : public kotekan::kotekanLogging {
    static_assert(D > 0);

    // // This is a buffer, not just a `std::vector`.
    // // This is used e.g. for the `info` output of the Julia-generated kernels.
    // const bool is_buffer = true;

    // The buffer lives on the device (and not the host).
    const bool is_device_buffer = true;

    // This is a ring buffer.
    const bool is_ringbuffer = true;

    // There is only one copy of the buffer, we don't use multi-buffering.
    // This is used e.g. for data that doesn't vary in time such as the beamforming matrices.
    const bool is_once_buffer = false;

    const std::string buffer_name;        // "official" buffer name (for metadata)
    const std::string buffer_name_host;   // buffer name on host
    const std::string buffer_name_device; // buffer name on device
    const std::string signal_buffer_name; // ringbuffer name

    cudaCommand& cuda_command;
    RingBuffer* const ringbuffer;

    // Begin and end of the valid region where reading/writing is
    // possible. This counts monotonically in elements (not in bytes, not
    // wrapping around).
    //
    // A ringbuffer is attached to only one instance of a command, and
    // it thus doesn't process all data. In other words, the processed
    // regions are not contiguous.
    std::ptrdiff_t begin_write_valid, end_write_valid;
    std::ptrdiff_t begin_read_valid, end_read_valid, end_read_claimed;

    const std::string quantity;
    kotekan::NDArray<T, D> ndarray;

private:
    std::array<std::ptrdiff_t, D> complete_extents(std::array<std::ptrdiff_t, D> extents) const {
        assert(extents[0] == -1);
        std::ptrdiff_t stride = 1;
        for (std::size_t d = 1; d < D; ++d)
            stride *= extents[d];
        const std::ptrdiff_t size = ringbuffer->size;
        assert(size > 0);
        const std::ptrdiff_t stride_bytes = stride * sizeof(T);
        assert(size % stride_bytes == 0);
        extents[0] = size / stride_bytes;
        return extents;
    }

    T* get_buffer_pointer(const std::array<std::ptrdiff_t, D>& extents0) const {
        const std::array<std::ptrdiff_t, D>& extents = complete_extents(extents0);
        std::ptrdiff_t size = 1;
        for (std::size_t d = 0; d < D; ++d)
            size *= extents[d];
        const std::ptrdiff_t size_in_bytes = size * sizeof(T);
        assert(ringbuffer->size == size_in_bytes);
        void* const ptr =
            cuda_command.get_device().get_gpu_memory(buffer_name_device, size_in_bytes);
        return static_cast<T*>(ptr);
    }

public:
    NDArrayRingBuffer(const std::string& buffer_name, const std::string& quantity,
                      const std::array<std::ptrdiff_t, D>& extents,
                      const std::array<kotekan::Symbol, D>& dimnames, cudaCommand& cuda_command) :
        // metadata
        buffer_name(buffer_name),                            // e.g. "bb_beams"
        buffer_name_host("host_" + buffer_name + "_buffer"), // e.g. "host_bb_beams_buffer"
        buffer_name_device(buffer_name + "_buffer"),         // e.g. "bb_beams_buffer"
        signal_buffer_name("host_" + buffer_name + "_ringbuffer"),
        // Buffer
        cuda_command(cuda_command),
        ringbuffer(dynamic_cast<RingBuffer*>(
            cuda_command.get_host_buffers().get_generic_buffer(signal_buffer_name))),
        begin_write_valid(0), end_write_valid(0), begin_read_valid(0), end_read_valid(0),
        end_read_claimed(0),
        // NDArray
        quantity(quantity), // e.g. "J"
        ndarray(complete_extents(extents), dimnames, get_buffer_pointer(extents))
    //
    {}

    NDArrayRingBuffer(const std::string& buffer_name, const std::string& quantity,
                      const std::array<std::ptrdiff_t, D>& extents,
                      const std::array<std::string, D>& dimnames, cudaCommand& cuda_command) :
        NDArrayRingBuffer(buffer_name, quantity, extents, kotekan::strings_to_symbols(dimnames),
                          cuda_command) {}

    NDArrayRingBuffer(const std::string& buffer_name, const std::string& quantity,
                      const std::array<std::ptrdiff_t, D>& extents,
                      const std::array<const char*, D>& dimnames, cudaCommand& cuda_command) :
        NDArrayRingBuffer(buffer_name, quantity, extents, kotekan::strings_to_symbols(dimnames),
                          cuda_command) {}

    virtual ~NDArrayRingBuffer() {}

    // Buffer:

    std::string get_buffer_name() const {
        return buffer_name;
    }
    std::string get_buffer_name_host() const {
        return buffer_name_host;
    }
    std::string get_buffer_name_device() const {
        return buffer_name_device;
    }

    void register_consumer() {
        if (cuda_command.get_instance_num() == 0)
            ringbuffer->register_consumer(cuda_command.get_unique_name());
    }

    // Returns 0 if all is good, -1 if we should terminate
    int wait_and_claim_readable(
        const std::function<read_descriptor_t(std::ptrdiff_t)> calc_read_descriptor) {
        // Check available samples
        DEBUG("Checking available input ringbuffer data...");
        const std::optional<std::pair<std::ptrdiff_t, std::ptrdiff_t>> peeked =
            ringbuffer->peek_readable(cuda_command.get_unique_name(),
                                      cuda_command.get_instance_num());
        if (!peeked.has_value())
            return -1;
        std::ptrdiff_t available_bytes = peeked.value().second;

    wait_for_data:
        const std::ptrdiff_t available_elements =
            div_noremainder(available_bytes, granularity_in_bytes());
        const read_descriptor_t read_descriptor = calc_read_descriptor(available_elements);
        assert(read_descriptor.claimed >= 0);
        assert(read_descriptor.claimed <= read_descriptor.read);
        assert(read_descriptor.read <= available_elements);

        // Can we make progress?
        if (read_descriptor.claimed <= 0) {
            // We cannot make progress, we need to wait
            const std::optional<std::ptrdiff_t> waited = ringbuffer->wait_without_claiming(
                cuda_command.get_unique_name(), cuda_command.get_instance_num(),
                available_bytes + 1);
            if (!waited.has_value())
                return -1;
            available_bytes = waited.value();
            goto wait_for_data;
        }

        // Claim inputs
        assert(read_descriptor.claimed > 0);
        const std::optional<std::ptrdiff_t> claimed = ringbuffer->wait_and_claim_readable(
            cuda_command.get_unique_name(), cuda_command.get_instance_num(),
            read_descriptor.claimed * granularity_in_bytes());
        if (!claimed.has_value())
            return -1;

        begin_read_valid = div_noremainder(claimed.value(), granularity_in_bytes());
        end_read_valid = begin_read_valid + read_descriptor.read;
        end_read_claimed = begin_read_valid + read_descriptor.claimed;

        return 0;
    }

    void finish_read() {
        const std::ptrdiff_t claimed_elements = end_read_claimed - begin_read_valid;
        begin_read_valid = end_read_claimed;
        ringbuffer->finish_read(cuda_command.get_unique_name(), cuda_command.get_instance_num(),
                                claimed_elements * granularity_in_bytes());
    }

    std::ptrdiff_t get_begin_write_valid() const {
        return begin_write_valid;
    }
    std::ptrdiff_t get_end_write_valid() const {
        return end_write_valid;
    }
    std::ptrdiff_t get_begin_read_valid() const {
        return begin_read_valid;
    }
    std::ptrdiff_t get_end_read_valid() const {
        return end_read_valid;
    }
    std::ptrdiff_t get_end_read_claimed() const {
        return end_read_claimed;
    }

    // NDArray:

    std::string get_quantity() const {
        return quantity;
    }

    const kotekan::NDArray<T, D>& get_ndarray() const {
        return ndarray;
    }
    kotekan::NDArray<T, D>& get_ndarray() {
        return ndarray;
    }

private:
    std::ptrdiff_t granularity_in_bytes() const {
        return ndarray.get_stride(0) * ndarray.value_type_size;
    }

public:
    // TODO: make this private
    std::ptrdiff_t length_in_bytes() const {
        return ndarray.get_size() * ndarray.value_type_size;
    }

    // Metadata:

    std::shared_ptr<const chordMetadata> get_metadata() const {
        const std::shared_ptr<const metadataObject> mc = ringbuffer->get_metadata(0);
        assert(mc);
        const std::shared_ptr<const chordMetadata> metadata = get_chord_metadata(mc);
        assert(metadata);
        return metadata;
    }
    std::shared_ptr<chordMetadata> get_metadata() {
        const std::shared_ptr<metadataObject> mc = ringbuffer->get_metadata(0);
        assert(mc);
        const std::shared_ptr<chordMetadata> metadata = get_chord_metadata(mc);
        assert(metadata);
        return metadata;
    }

    void check_metadata() const {
        const std::shared_ptr<const chordMetadata> metadata = get_metadata();
        if (!(metadata->get_name() == quantity))
            ERROR("buffer name: {:s}, quantity: {:s}, metadata name: {:s}", buffer_name, quantity,
                  metadata->get_name());
        assert(metadata->get_name() == quantity);
        assert(metadata->type == ndarray.value_datatype);
        assert(metadata->dims == ndarray.rank);
        for (std::size_t d = 0; d < ndarray.rank; ++d) {
            assert(metadata->get_dimension_name(d) == ndarray.dimname(d));
            // Skip the ringbuffer direction
            if (d > 0) {
                assert(metadata->dim[d] == int(ndarray.extent(d)));
                assert(metadata->stride[d] == ndarray.stride(d));
            }
        }
    }

    // TODO // TODO: Use other NDArrayRingBuffer instead of other metadata
    // TODO void set_metadata(const std::shared_ptr<const chordMetadata>& other_metadata) const {
    // TODO     const std::shared_ptr<metadataObject> mc = device.create_gpu_memory_array_metadata(
    // TODO         buffer_name_device, gpu_frame_id, other_metadata->parent_pool);
    // TODO     const std::shared_ptr<chordMetadata> metadata = get_chord_metadata(mc);
    // TODO     *metadata = *other_metadata;
    // TODO     metadata->set_name(buffer_name);
    // TODO     metadata->type = ndarray.value_datatype;
    // TODO     metadata->dims = ndarray.rank;
    // TODO     for (std::size_t d = 0; d < ndarray.rank; ++d) {
    // TODO         metadata->set_array_dimension(d, ndarray.extent(d),
    // std::string(ndarray.dimname(d)));
    // TODO         metadata->stride[d] = ndarray.stride(d);
    // TODO     }
    // TODO }

    // Poison

    // TODO     // Poison an NDArray buffer
    // TODO     void set_to_poison(const std::uint8_t poison_value) {
    // TODO         // assert(is_buffer);
    // TODO         assert(is_device_buffer);
    // TODO         assert(!is_ringbuffer);
    // TODO #ifdef DEBUGGING
    // TODO         const std::ptrdiff_t buffer_length = length_in_bytes();
    // TODO         void* const buffer_device_ptr = ndarray.data();
    // TODO         assert(buffer_device_ptr);
    // TODO         const cudaStream_t cuda_stream = device.getStream(cuda_stream_id);
    // TODO         CHECK_CUDA_ERROR(
    // TODO             cudaMemsetAsync(buffer_device_ptr, poison_value, buffer_length,
    // cuda_stream));
    // TODO #endif
    // TODO     }
    // TODO
    // TODO     // Check an NDArray buffer for poison
    // TODO     void check_for_poison(const std::uint8_t poison_value) {
    // TODO         // assert(is_buffer);
    // TODO         assert(is_device_buffer);
    // TODO         assert(!is_ringbuffer);
    // TODO #ifdef DEBUGGING
    // TODO         const std::ptrdiff_t buffer_length = length_in_bytes();
    // TODO         const void* const buffer_device_ptr = ndarray.data();
    // TODO         assert(buffer_device_ptr);
    // TODO         std::vector<std::uint8_t> local_data(buffer_length);
    // TODO         CHECK_CUDA_ERROR(cudaMemcpy(local_data.data(), buffer_device_ptr, buffer_length,
    // TODO                                     cudaMemcpyDeviceToHost));
    // TODO         const bool found_error = std::memchr(local_data.data(), poison_value,
    // buffer_length);
    // TODO         if (found_error)
    // TODO             FATAL_ERROR("NDArray buffer {:s} contains poison", buffer_name);
    // TODO #endif
    // TODO     }

    // I/O

    std::ostream& output(std::ostream& os) const {
        return os << "NDArrayRingBuffer<" << ndarray.value_datatype << "," << ndarray.rank << ">("
                  << buffer_name << "," << quantity << ")";
    }

    friend std::ostream& operator<<(std::ostream& os, const NDArrayRingBuffer& rb) {
        return rb.output(os);
    }

    operator std::string() const {
        std::ostringstream buf;
        buf << *this;
        return buf.str();
    }
};

#endif // #ifndef NDARRAYRINGBUFFER_HPP
