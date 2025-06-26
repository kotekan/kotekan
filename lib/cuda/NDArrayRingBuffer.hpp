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
#include <cudaUtils.hpp>
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
using kotekan::mod;

struct read_descriptor_t {
    std::ptrdiff_t claimed, read;
};

template<typename T, std::size_t D>
class NDArrayRingBuffer : public kotekan::kotekanLogging {
    static_assert(D > 0);

    const std::string buffer_name;        // "official" buffer name (for metadata)
    const std::string buffer_name_host;   // buffer name on host
    const std::string buffer_name_device; // buffer name on device
    const std::string signal_buffer_name; // ringbuffer name

    cudaCommand& cuda_command;
    RingBuffer* const ringbuffer;

    const std::string quantity;
    kotekan::NDArray<T, D> ndarray;

    // Begin and end of the valid region where reading/writing is
    // possible. This counts monotonically in elements (not in bytes, not
    // wrapping around).
    //
    // A ringbuffer is attached to only one instance of a command, and
    // it thus doesn't process all data. In other words, the processed
    // regions are not contiguous.
    std::ptrdiff_t begin_write_valid, end_write_valid;
    std::ptrdiff_t begin_read_valid, end_read_valid, end_read_claimed;

private:
    int get_instance_num() const {
        return cuda_command.get_instance_num();
    }

    T* get_buffer_pointer(const std::array<std::ptrdiff_t, D>& extents) const {
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
        // NDArray
        quantity(quantity), // e.g. "J"
        ndarray(extents, dimnames, get_buffer_pointer(extents)),
        // State
        begin_write_valid(0), end_write_valid(0), begin_read_valid(0), end_read_valid(0),
        end_read_claimed(0)
    //
    {
        assert(ringbuffer);
    }

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

    // TODO: Distinguish between input and output buffers, then register automatically
    void register_consumer() {
        if (get_instance_num() == 0) {
            ringbuffer->register_consumer(cuda_command.get_unique_name());
            cuda_command.register_gpu_buffer_user(
                {.name = buffer_name, .is_array = true, .does_read = true, .does_write = false});
        }
    }

    void register_producer() {
        if (get_instance_num() == 0) {
            ringbuffer->register_producer(cuda_command.get_unique_name());
            cuda_command.register_gpu_buffer_user(
                {.name = buffer_name, .is_array = true, .does_read = false, .does_write = true});
        }
    }

    // Returns 0 if all is good, -1 if we should terminate
    int wait_for_writable(const std::ptrdiff_t produced_elements) {
        const std::ptrdiff_t produced_bytes = produced_elements * granularity_in_bytes();
        const std::optional<std::ptrdiff_t> waited = ringbuffer->wait_for_writable(
            cuda_command.get_unique_name(), get_instance_num(), produced_bytes);
        if (!waited.has_value())
            return -1;

        begin_write_valid = div_noremainder(waited.value(), granularity_in_bytes());
        end_write_valid = begin_write_valid + produced_elements;

        return 0;
    }

    // Returns 0 if all is good, -1 if we should terminate
    int wait_and_claim_readable(
        const std::function<read_descriptor_t(std::ptrdiff_t)> calc_read_descriptor) {
        // Check available samples
        const std::optional<std::pair<std::ptrdiff_t, std::ptrdiff_t>> peeked =
            ringbuffer->peek_readable(cuda_command.get_unique_name(), get_instance_num());
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
                cuda_command.get_unique_name(), get_instance_num(), available_bytes + 1);
            if (!waited.has_value())
                return -1;
            available_bytes = waited.value();
            goto wait_for_data;
        }

        // Claim inputs
        assert(read_descriptor.claimed > 0);
        const std::optional<std::ptrdiff_t> claimed =
            ringbuffer->wait_and_claim_readable(cuda_command.get_unique_name(), get_instance_num(),
                                                read_descriptor.claimed * granularity_in_bytes());
        if (!claimed.has_value())
            return -1;

        begin_read_valid = div_noremainder(claimed.value(), granularity_in_bytes());
        end_read_valid = begin_read_valid + read_descriptor.read;
        end_read_claimed = begin_read_valid + read_descriptor.claimed;

        return 0;
    }

    void finish_write() {
        const std::ptrdiff_t written_elements = end_write_valid - begin_write_valid;
        begin_write_valid = end_write_valid;
        ringbuffer->finish_write(cuda_command.get_unique_name(), get_instance_num(),
                                 written_elements * granularity_in_bytes());
    }

    void finish_read() {
        const std::ptrdiff_t claimed_elements = end_read_claimed - begin_read_valid;
        begin_read_valid = end_read_claimed;
        ringbuffer->finish_read(cuda_command.get_unique_name(), get_instance_num(),
                                claimed_elements * granularity_in_bytes());
    }

    // State

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

    std::ptrdiff_t length_in_bytes() const {
        return ndarray.get_size() * ndarray.value_type_size;
    }

public:
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
            // The ring buffer direction is special
            if (d > 0)
                assert(metadata->dim[d] == int(ndarray.extent(d)));
            assert(metadata->stride[d] == ndarray.stride(d));
        }
    }

    void set_metadata(const std::shared_ptr<const chordMetadata>& other_metadata) {
        // const std::shared_ptr<metadataObject> mc =
        //     cuda_command.get_device().create_gpu_memory_array_metadata(buffer_name_device, 0,
        //                                                                other_metadata->parent_pool);
        // const std::shared_ptr<chordMetadata> metadata = get_chord_metadata(mc);
        ringbuffer->allocate_new_metadata_object(0);
        const std::shared_ptr<chordMetadata> metadata = get_metadata();
        *metadata = *other_metadata;
        metadata->set_name(quantity);
        metadata->type = ndarray.value_datatype;
        metadata->dims = ndarray.rank;
        for (std::size_t d = 0; d < ndarray.rank; ++d) {
            // The ring buffer direction is special
            if (d == 0)
                metadata->set_array_dimension(d, other_metadata->dim[d],
                                              std::string(ndarray.dimname(d)));
            else
                metadata->set_array_dimension(d, ndarray.extent(d),
                                              std::string(ndarray.dimname(d)));
            metadata->stride[d] = ndarray.stride(d);
        }
    }

    // Poison

    // Poison an NDArray ring buffer
    void set_to_poison(const std::uint8_t poison_value) {
#ifdef DEBUGGING
        const std::ptrdiff_t T_ringbuf = get_ndarray().extent(0);
        const std::ptrdiff_t T_min = get_begin_write_valid();
        const std::ptrdiff_t T_max = get_end_write_valid();
        const std::ptrdiff_t T_length = T_max - T_min;
        const std::ptrdiff_t T_min_arg = mod(T_min, T_ringbuf);
        const std::ptrdiff_t T_max_arg = mod(T_min, T_ringbuf) + T_length;
        const int num_chunks = T_max_arg <= T_ringbuf ? 1 : 2;
        for (int chunk = 0; chunk < num_chunks; ++chunk) {
            const std::ptrdiff_t T_stride = granularity_in_bytes();
            const std::ptrdiff_t T_offset = chunk == 0 ? T_min_arg : 0;
            const std::ptrdiff_t T_length = num_chunks == 1 ? T_max_arg - T_min_arg
                                            : chunk == 0    ? T_ringbuf - T_min_arg
                                                            : T_max_arg - T_ringbuf;
            CHECK_CUDA_ERROR(cudaMemsetAsync(
                (std::uint8_t*)get_ndarray().data() + T_offset * T_stride, poison_value,
                T_length * T_stride,
                cuda_command.get_device().getStream(cuda_command.get_cuda_stream_id())));
        } // for chunk
#endif
    }

    // Check an NDArray ring buffer for poison
    void check_for_poison(const std::uint8_t poison_value) {
#ifdef DEBUGGING
        const std::ptrdiff_t T_ringbuf = get_ndarray().extent(0);
        const std::ptrdiff_t T_min = get_begin_write_valid();
        const std::ptrdiff_t T_max = get_end_write_valid();
        const std::ptrdiff_t T_length = T_max - T_min;
        const std::ptrdiff_t T_min_arg = mod(T_min, T_ringbuf);
        const std::ptrdiff_t T_max_arg = mod(T_min, T_ringbuf) + T_length;
        const int num_chunks = T_max_arg <= T_ringbuf ? 1 : 2;
        for (int chunk = 0; chunk < num_chunks; ++chunk) {
            const std::ptrdiff_t T_stride = granularity_in_bytes();
            const std::ptrdiff_t T_offset = chunk == 0 ? T_min_arg : 0;
            const std::ptrdiff_t T_length = num_chunks == 1 ? T_max_arg - T_min_arg
                                            : chunk == 0    ? T_ringbuf - T_min_arg
                                                            : T_max_arg - T_ringbuf;
            std::vector<std::uint8_t> local_data(T_length * T_stride, poison_value);
            CHECK_CUDA_ERROR(cudaMemcpy(local_data.data(),
                                        (std::uint8_t*)get_ndarray().data() + T_offset * T_stride,
                                        T_length * T_stride, cudaMemcpyDeviceToHost));
            const bool found_error =
                std::memchr(local_data.data(), poison_value, local_data.size());
            if (found_error) {
                for (std::ptrdiff_t t = 0; t < T_length; ++t) {
                    bool any_error = false;
                    for (std::ptrdiff_t n = 0; n < T_stride; ++n) {
                        const auto val = local_data.at(t * T_stride + n);
                        any_error |= val == 0x00;
                    }
                    if (any_error)
                        DEBUG("    [{}]={:#02x}", t, 0x00);
                }
            }
            if (found_error)
                FATAL_ERROR("NDArray buffer {:s} contains poison", buffer_name);
            assert(!found_error);
        } // for chunk
#endif
    }

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
