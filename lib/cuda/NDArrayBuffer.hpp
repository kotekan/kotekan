#ifndef NDARRAYBUFFER_HPP
#define NDARRAYBUFFER_HPP

#include <DataType.hpp>
#include <NDArray.hpp>
#include <Symbol.hpp>
#include <algorithm>
#include <array>
#include <buffer.hpp>
#include <cassert>
#include <chordMetadata.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cudaCommand.hpp>
#include <cudaDeviceInterface.hpp>
#include <kotekanLogging.hpp>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

enum buffer_type_t { standard = 0, do_once = 1 << 0 };

template<typename T, std::size_t D>
class NDArrayBuffer : public kotekan::kotekanLogging {
    const std::string buffer_name;        // "official" buffer name (for metadata)
    const std::string buffer_name_host;   // buffer name on host
    const std::string buffer_name_device; // buffer name on device

    const bool is_do_once;

    cudaCommand& cuda_command;

    const std::string quantity;
    kotekan::NDArray<T, D> ndarray;

private:
    int get_instance_num() const {
        return cuda_command.get_instance_num();
    }

    T* get_buffer_pointer(const std::array<std::ptrdiff_t, D>& extents) const {
        std::ptrdiff_t size = 1;
        for (std::size_t d = 0; d < D; ++d)
            size *= extents[d];
        const std::ptrdiff_t size_in_bytes = size * sizeof(T);
        void* const ptr =
            is_do_once ? cuda_command.get_device().get_gpu_memory(buffer_name_device, size_in_bytes)
                       : cuda_command.get_device().get_gpu_memory_array(
                             buffer_name_device, get_instance_num(),
                             cuda_command.get_gpu_buffer_depth(), size_in_bytes);
        return static_cast<T*>(ptr);
    }

public:
    NDArrayBuffer(const std::string& buffer_name, const std::string& quantity,
                  const std::array<std::ptrdiff_t, D>& extents,
                  const std::array<kotekan::Symbol, D>& dimnames, cudaCommand& cuda_command,
                  const buffer_type_t buffer_type = buffer_type_t::standard) :
        // metadata
        buffer_name(buffer_name),                            // e.g. "bb_beams"
        buffer_name_host("host_" + buffer_name + "_buffer"), // e.g. "host_bb_beams_buffer"
        buffer_name_device(buffer_name + "_buffer"),         // e.g. "bb_beams_buffer"
        is_do_once(buffer_type & buffer_type_t::do_once),
        // Buffer
        cuda_command(cuda_command),
        // NDArray
        quantity(quantity), // e.g. "J"
        ndarray(extents, dimnames, get_buffer_pointer(extents))
    //
    {
        set_log_level(cuda_command.get_log_level());
    }

    NDArrayBuffer(const std::string& buffer_name, const std::string& quantity,
                  const std::array<std::ptrdiff_t, D>& extents,
                  const std::array<std::string, D>& dimnames, cudaCommand& cuda_command,
                  const buffer_type_t buffer_type = buffer_type_t::standard) :
        NDArrayBuffer(buffer_name, quantity, extents, kotekan::strings_to_symbols(dimnames),
                      cuda_command, buffer_type) {}

    NDArrayBuffer(const std::string& buffer_name, const std::string& quantity,
                  const std::array<std::ptrdiff_t, D>& extents,
                  const std::array<const char*, D>& dimnames, cudaCommand& cuda_command,
                  const buffer_type_t buffer_type = buffer_type_t::standard) :
        NDArrayBuffer(buffer_name, quantity, extents, kotekan::strings_to_symbols(dimnames),
                      cuda_command, buffer_type) {}

    virtual ~NDArrayBuffer() {}

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
        if (get_instance_num() == 0)
            cuda_command.register_gpu_buffer_user(
                {.name = buffer_name, .is_array = true, .does_read = true, .does_write = false});
    }

    void register_producer() {
        if (get_instance_num() == 0)
            cuda_command.register_gpu_buffer_user(
                {.name = buffer_name, .is_array = true, .does_read = false, .does_write = true});
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
    std::ptrdiff_t length_in_bytes() const {
        return ndarray.get_size() * ndarray.value_type_size;
    }

public:
    // Metadata:

    std::shared_ptr<const chordMetadata> get_metadata() const {
        const std::shared_ptr<const metadataObject> mc =
            cuda_command.get_device().get_gpu_memory_array_metadata(buffer_name_device,
                                                                    get_instance_num());
        const std::shared_ptr<const chordMetadata> metadata = get_chord_metadata(mc);
        return metadata;
    }
    std::shared_ptr<chordMetadata> get_metadata() {
        const std::shared_ptr<metadataObject> mc =
            cuda_command.get_device().get_gpu_memory_array_metadata(buffer_name_device,
                                                                    get_instance_num());
        const std::shared_ptr<chordMetadata> metadata = get_chord_metadata(mc);
        return metadata;
    }

    void check_metadata() const {
        const std::shared_ptr<const chordMetadata> metadata = get_metadata();
        assert(metadata->get_name() == quantity);
        assert(metadata->type == ndarray.value_datatype);
        assert(metadata->dims == ndarray.rank);
        for (std::size_t d = 0; d < ndarray.rank; ++d) {
            assert(metadata->get_dimension_name(d) == ndarray.dimname(d));
            assert(metadata->dim[d] == int(ndarray.extent(d)));
            assert(metadata->stride[d] == ndarray.stride(d));
        }
        // TODO: check `sample0_offset`
    }

    void set_metadata(const std::shared_ptr<const chordMetadata>& other_metadata) const {
        const std::shared_ptr<metadataObject> mc =
            cuda_command.get_device().create_gpu_memory_array_metadata(
                buffer_name_device, get_instance_num(), other_metadata->parent_pool);
        const std::shared_ptr<chordMetadata> metadata = get_chord_metadata(mc);
        *metadata = *other_metadata;
        metadata->set_name(quantity);
        metadata->type = ndarray.value_datatype;
        metadata->dims = ndarray.rank;
        for (std::size_t d = 0; d < ndarray.rank; ++d) {
            metadata->set_array_dimension(d, ndarray.extent(d), std::string(ndarray.dimname(d)));
            metadata->stride[d] = ndarray.stride(d);
        }
        // TODO: set `sample0_offset`
    }

    // Poison

    // Poison an NDArray buffer
    void set_to_poison(const std::uint8_t poison_value) {
#ifdef DEBUGGING
        const std::ptrdiff_t buffer_length = length_in_bytes();
        void* const buffer_device_ptr = ndarray.data();
        assert(buffer_device_ptr);
        const cudaStream_t cuda_stream =
            cuda_command.get_device().getStream(cuda_command.get_cuda_stream_id());
        CHECK_CUDA_ERROR(
            cudaMemsetAsync(buffer_device_ptr, poison_value, buffer_length, cuda_stream));
#endif
    }

    // Check an NDArray buffer for poison
    void check_for_poison(const std::uint8_t poison_value) {
#ifdef DEBUGGING
        T poison;
        std::memset(&poison, poison_value, sizeof poison);
        const auto check = [=](const T x) { return std::memcmp(&x, &poison, sizeof poison) == 0; };
        const std::ptrdiff_t buffer_length = length_in_bytes();
        const void* const buffer_device_ptr = ndarray.data();
        assert(buffer_device_ptr);
        std::vector<T> local_data(buffer_length / sizeof(T), poison);
        CHECK_CUDA_ERROR(cudaMemcpy(local_data.data(), buffer_device_ptr, buffer_length,
                                    cudaMemcpyDeviceToHost));
        const bool found_error =
            std::find_if(local_data.begin(), local_data.end(), check) != local_data.end();
        if (found_error)
            FATAL_ERROR("NDArray buffer {:s} contains poison", buffer_name);
#endif
    }

    // I/O

    std::ostream& output(std::ostream& os) const {
        return os << "NDArrayBuffer<" << ndarray.value_datatype << "," << ndarray.rank << ">("
                  << buffer_name << "," << quantity << ")";
    }

    friend std::ostream& operator<<(std::ostream& os, const NDArrayBuffer& b) {
        return b.output(os);
    }

    operator std::string() const {
        std::ostringstream buf;
        buf << *this;
        return buf.str();
    }
};

#endif // #ifndef NDARRAYBUFFER_HPP
