#ifndef NDARRAYRINGBUFFER_HPP
#define NDARRAYRINGBUFFER_HPP

#include <DataType.hpp>
#include <NDArray.hpp>
#include <Symbol.hpp>
#include <algorithm>
#include <array>
#include <buffer.hpp>
#include <cassert>
#include <chordMetadata.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cudaCommand.hpp>
#include <cudaDeviceInterface.hpp>
#include <cudaUtils.hpp>
#include <div.hpp>
#include <functional>
#include <gpuDeviceInterface.hpp>
#include <iomanip>
#include <kotekanLogging.hpp>
#include <ostream>
#include <ringbuffer.hpp>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using kotekan::div_noremainder;
using kotekan::mod;

struct read_descriptor_t {
    std::ptrdiff_t claimed, read;
};

struct extent_t {
    std::ptrdiff_t m_begin, m_end;

public:
    extent_t(const extent_t&) = default;
    extent_t(extent_t&&) = default;
    extent_t& operator=(const extent_t&) = default;
    extent_t& operator=(extent_t&&) = default;

    extent_t() : extent_t(0, 0) {}
    extent_t(std::ptrdiff_t begin, std::ptrdiff_t end) : m_begin(begin), m_end(end) {
        assert(size() >= 0);
    }
    std::ptrdiff_t begin() const noexcept {
        return m_begin;
    }
    std::ptrdiff_t end() const noexcept {
        return m_end;
    }
    std::ptrdiff_t size() const noexcept {
        return end() - begin();
    }
};

#warning "TODO: set log level in constructor"
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
    extent_t write_valid, read_valid, read_claimed;

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
        write_valid(0, 0), read_valid(0, 0), read_claimed(0, 0)
    //
    {
        set_log_level(cuda_command.get_log_level());

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
        assert(write_valid.size() == 0);
        assert(read_valid.size() == 0);
        assert(read_claimed.size() == 0);

        const std::ptrdiff_t produced_bytes = produced_elements * granularity_in_bytes();
        const std::optional<std::ptrdiff_t> waited = ringbuffer->wait_for_writable(
            cuda_command.get_unique_name(), get_instance_num(), produced_bytes);
        if (!waited.has_value())
            return -1;

        const std::ptrdiff_t write_begin = div_noremainder(waited.value(), granularity_in_bytes());
        write_valid = extent_t(write_begin, write_begin + produced_elements);
        assert(write_valid.size() > 0);

        return 0;
    }

    void finish_write() {
        assert(write_valid.size() > 0);
        const std::ptrdiff_t written_elements = write_valid.size();
        ringbuffer->finish_write(cuda_command.get_unique_name(), get_instance_num(),
                                 written_elements * granularity_in_bytes());

        write_valid = extent_t(write_valid.end(), write_valid.end());
        assert(write_valid.size() == 0);
    }

    // Returns 0 if all is good, -1 if we should terminate
    int wait_and_claim_readable(
        const std::function<read_descriptor_t(std::ptrdiff_t)>& calc_read_descriptor) {
        assert(write_valid.size() == 0);
#warning "TODO"
        if (!(read_valid.size() == 0)) {
            DEBUG("buffer {:s}, wait_and_claim_readable({:s}[{:d}]): valid read region is not "
                  "empty (begin={:d}, end={:d}, size={:d})",
                  buffer_name, cuda_command.get_unique_name(), get_instance_num(),
                  read_valid.begin(), read_valid.end(), read_valid.size());
            sleep(1);
            FATAL_ERROR("buffer {:s}, wait_and_claim_readable({:s}[{:d}]): valid read region is "
                        "not empty (begin={:d}, end={:d}, size={:d})",
                        buffer_name, cuda_command.get_unique_name(), get_instance_num(),
                        read_valid.begin(), read_valid.end(), read_valid.size());
        }
        assert(read_valid.size() == 0);
        assert(read_claimed.size() == 0);

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

        const std::ptrdiff_t read_begin = div_noremainder(claimed.value(), granularity_in_bytes());
        read_valid = extent_t(read_begin, read_begin + read_descriptor.read);
        read_claimed = extent_t(read_begin, read_begin + read_descriptor.claimed);
        assert(read_valid.size() > 0);
        assert(read_claimed.size() > 0);

        return 0;
    }

    void finish_read() {
        assert(read_valid.size() > 0);
        assert(read_claimed.size() > 0);
        const std::ptrdiff_t claimed_elements = read_claimed.size();
        ringbuffer->finish_read(cuda_command.get_unique_name(), get_instance_num(),
                                claimed_elements * granularity_in_bytes());

        read_valid = extent_t(read_claimed.end(), read_claimed.end());
        read_claimed = extent_t(read_claimed.end(), read_claimed.end());
        assert(read_valid.size() == 0);
        assert(read_claimed.size() == 0);
    }

    // State

    extent_t get_write_valid() const {
#ifdef DEBUGGING
        if (!(write_valid.size() > 0))
            FATAL_ERROR("kernel {:s}, buffer {:s}, get_write_valid: valid write region is empty",
                        cuda_command.get_unique_name(), buffer_name);
        assert(write_valid.size() > 0);
#endif
        return write_valid;
    }
    extent_t get_read_valid() const {
#ifdef DEBUGGING
        if (!(read_valid.size() > 0))
            FATAL_ERROR("kernel {:s}, buffer {:s}, get_read_valid: valid read region is empty",
                        cuda_command.get_unique_name(), buffer_name);
        assert(read_valid.size() > 0);
#endif
        return read_valid;
    }
    extent_t get_read_claimed() const {
#ifdef DEBUGGING
        if (!(read_claimed.size() > 0))
            FATAL_ERROR("kernel {:s}, buffer {:s}, get_read_claimed: claimed read region is empty",
                        cuda_command.get_unique_name(), buffer_name);
        assert(read_claimed.size() > 0);
#endif
        return read_claimed;
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
    void set_to_poison(const std::uint8_t poison_value, const std::ptrdiff_t F_min,
                       const std::ptrdiff_t F_max) {
#ifdef DEBUGGING
        assert(get_write_valid().size() > 0);

        const std::ptrdiff_t F_stride = get_ndarray().get_stride(1);
        assert(F_stride > 0);
        const std::ptrdiff_t F_offset = F_min;
        assert(F_offset >= 0);
        const std::ptrdiff_t F_length = F_max - F_min;
        assert(F_length > 0);

        const std::ptrdiff_t T_ringbuf = get_ndarray().extent(0);
        assert(T_ringbuf > 0);
        const std::ptrdiff_t T_stride = get_ndarray().get_stride(0);
        assert(T_stride > 0);
        const std::ptrdiff_t T_min = get_write_valid().begin();
        assert(T_min >= 0);
        const std::ptrdiff_t T_max = get_write_valid().end();
        assert(T_max > T_min);
        const std::ptrdiff_t T_length = T_max - T_min;
        assert(T_length > 0);

        const std::ptrdiff_t T_min_arg = mod(T_min, T_ringbuf);
        assert(T_min_arg >= 0);
        assert(T_min_arg < T_ringbuf);
        const std::ptrdiff_t T_max_arg = T_min_arg + T_length;
        assert(T_max_arg >= T_min_arg);
        assert(T_max_arg < 2 * T_ringbuf);
        const int num_chunks = T_max_arg <= T_ringbuf ? 1 : 2;
        for (int chunk = 0; chunk < num_chunks; ++chunk) {
            const std::ptrdiff_t T_offset = chunk == 0 ? T_min_arg : 0;
            assert(T_offset >= 0);
            assert(T_offset < T_ringbuf);
            const std::ptrdiff_t T_length = num_chunks == 1 ? T_max_arg - T_min_arg
                                            : chunk == 0    ? T_ringbuf - T_min_arg
                                                            : T_max_arg - T_ringbuf;
            assert(T_length > 0);
            assert(T_offset + T_length <= T_ringbuf);

            const auto stream =
                cuda_command.get_device().getStream(cuda_command.get_cuda_stream_id());
            CHECK_CUDA_ERROR(
                cudaMemset2DAsync(get_ndarray().data() + T_offset * T_stride + F_offset * F_stride,
                                  T_stride * sizeof(T), poison_value,
                                  F_length * F_stride * sizeof(T), T_length, stream));
        } // for chunk
#endif
    }

    void set_to_poison(const std::uint8_t poison_value) {
        set_to_poison(poison_value, 0, get_ndarray().get_extent(1));
    }

    // Check an NDArray ring buffer for poison
    void check_for_poison(const std::uint8_t poison_value, const std::ptrdiff_t F_min,
                          const std::ptrdiff_t F_max) {
#ifdef DEBUGGING
        assert(get_write_valid().size() > 0);

        T poison;
        std::memset(&poison, poison_value, sizeof poison);
        const auto check = [=](const T x) {
            using std::isfinite, kotekan::isfinite;
            if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, float16_t>)
                if (!isfinite(x))
                    return true;
            return std::memcmp(&x, &poison, sizeof poison) == 0;
        };

        const std::ptrdiff_t F_stride = get_ndarray().get_stride(1);
        assert(F_stride > 0);
        const std::ptrdiff_t F_offset = F_min;
        assert(F_offset >= 0);
        const std::ptrdiff_t F_length = F_max - F_min;
        assert(F_length > 0);

        const std::ptrdiff_t T_ringbuf = get_ndarray().extent(0);
        assert(T_ringbuf > 0);
        const std::ptrdiff_t T_stride = get_ndarray().get_stride(0);
        assert(T_stride > 0);
        const std::ptrdiff_t T_min = get_write_valid().begin();
        assert(T_min >= 0);
        const std::ptrdiff_t T_max = get_write_valid().end();
        assert(T_max > T_min);
        const std::ptrdiff_t T_length = T_max - T_min;
        assert(T_length > 0);

        const std::ptrdiff_t T_stride_local = F_length * F_stride;
        std::vector<T> local_data(T_length * T_stride_local, poison);

        const std::ptrdiff_t T_min_arg = mod(T_min, T_ringbuf);
        assert(T_min_arg >= 0);
        assert(T_min_arg < T_ringbuf);
        const std::ptrdiff_t T_max_arg = T_min_arg + T_length;
        assert(T_max_arg >= T_min_arg);
        assert(T_max_arg < 2 * T_ringbuf);
        const int num_chunks = T_max_arg <= T_ringbuf ? 1 : 2;
        for (int chunk = 0; chunk < num_chunks; ++chunk) {
            const std::ptrdiff_t T_offset = chunk == 0 ? T_min_arg : 0;
            assert(T_offset >= 0);
            assert(T_offset < T_ringbuf);
            const std::ptrdiff_t T_length = num_chunks == 1 ? T_max_arg - T_min_arg
                                            : chunk == 0    ? T_ringbuf - T_min_arg
                                                            : T_max_arg - T_ringbuf;
            assert(T_length > 0);
            assert(T_offset + T_length <= T_ringbuf);

            const std::ptrdiff_t T_offset_local = chunk == 0 ? 0 : T_ringbuf - T_min_arg;

            CHECK_CUDA_ERROR(cudaMemcpy2D(
                local_data.data() + T_offset_local * T_stride_local, T_stride_local * sizeof(T),
                get_ndarray().data() + T_offset * T_stride + F_offset * F_stride,
                T_stride * sizeof(T), T_stride_local * sizeof(T), T_length,
                cudaMemcpyDeviceToHost));
        } // for chunk

        const auto first_poison_location =
            std::find_if(local_data.begin(), local_data.end(), check);
        const bool found_error = first_poison_location != local_data.end();
        if (found_error)
            ERROR("NDArray ring buffer {:s} contains poison or a non-finite number at index={:d}",
                  buffer_name, first_poison_location - local_data.begin(), local_data.size());
        if (found_error) {
            for (std::ptrdiff_t t = 0; t < T_length; ++t) {
                for (std::ptrdiff_t f = 0; f < F_length; ++f) {
                    bool any_error = false;
                    for (std::ptrdiff_t n = 0; n < F_stride; ++n) {
                        const auto val = local_data.at(t * T_stride_local + f * F_stride + n);
                        if (true) {
                            if (check(val)) {
                                kotekan::GetType_t<kotekan::uint_from_element_bits(8 * sizeof(T))>
                                    bits;
                                static_assert(sizeof bits == sizeof(T));
                                std::memcpy(&bits, &val, sizeof(T));
                                using kotekan::operator<<;
                                std::cerr << "    [t=" << t << ",f=" << f << ",n=" << n
                                          << "]=" << val << " (0x" << std::hex
                                          << std::setw(2 * sizeof(T)) << std::setfill('0') << bits
                                          << std::setfill(' ') << std::dec << ")" << "\n";
                            }
                        }
                        any_error |= check(val);
                    }
                    if (any_error)
                        ERROR("    poison or non-finite at [t={:d},f={:d}]", t, f);
                }
            }
        }
        if (found_error)
            FATAL_ERROR("NDArray ring buffer {:s} contains poison", buffer_name);
        assert(!found_error);
#endif
    }

    void check_for_poison(const std::uint8_t poison_value) {
        check_for_poison(poison_value, 0, get_ndarray().get_extent(1));
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
