#ifndef NDARRAYBUFFER_HPP
#define NDARRAYBUFFER_HPP

#include "DataType.hpp"            // for operator<<
#include "NDArray.hpp"             // for NDArray
#include "Symbol.hpp"              // for Symbol, strings_to_symbols, operator==
#include "chordMetadata.hpp"       // for chordMetadata, get_chord_metadata
#include "cudaCommand.hpp"         // for cudaCommand
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "cudaUtils.hpp"           // for CHECK_CUDA_ERROR
#include "kotekanLogging.hpp"      // for kotekanLogging, FATAL_ERROR
#include "metadata.hpp"            // for metadataObject

#include <algorithm>          // for find_if
#include <array>              // for array
#include <cassert>            // for assert
#include <cstddef>            // for ptrdiff_t, size_t
#include <cstdint>            // for uint8_t
#include <cstring>            // for memcmp, memset
#include <cuda_runtime_api.h> // for cudaMemsetAsync, cudaMemcpy
#include <driver_types.h>     // for CUstream_st, cudaMemcpyKind, cudaStream_t
#include <fmt.hpp>            // for compile_string_to_view
#include <memory>             // for shared_ptr, __shared_ptr_access, allocator
#include <sstream>            // for basic_ostream, operator<<, ostream, basic_ostringstream
#include <string>             // for string, basic_string, char_traits, operator+, operator<<
#include <vector>             // for vector

// This affects copying from host to device. A standard buffer is
// copied the usual way. A `do_once` buffer is copied only once, in
// the beginning, and then holds its data. This is used e.g. for
// beamforming phase matrices.
enum buffer_type_t { standard = 0, do_once = 1 << 0 };

// An `NDArrayBuffer` wraps a Kotekan buffer or a GPU buffer and
// associates them with an `NDArray`. (An `NDArray` knows its type,
// rank, and shape, and knows the "names" of its dimensions.) There is
// also a type `NDArrayRingBuffer` wrapping a Kotekan ring buffer.
//
// In a typical scenario, a compute kernel creates an `NDArrayBuffer`
// in its constructor. This simplifies applying "the usual" operations
// to a Kotekan buffer, such as:
// - register producers/consumers
// - check and set metadata, especially those for its type and shape
// - access buffer as `NDArray`, i.e. get a typed pointer to the data
//   or find its shape
//
// Thoughts for the future: It is somewhat tedious to create an
// `NDArrayBuffer` since one needs to know everything about the buffer
// (name, size, type, shape, etc.). Kotekan already knows all this. It
// would be convenient if Kotekan offered a way to directly obtain an
// `NDArrayBuffer`.
//
// Kotekan often can have multiple buffers associated with the same
// data, e.g. a Kotekan buffer on the host and a GPU buffer on the
// device. (Kotekan buffers and GPU buffers use different APIs.)
// `NDArrayBuffer` expects that these names follow a regular pattern:
// - There is an "official" name, e.g. `voltage`
// - Derived from this, the Kotekan host buffer is then called
//   `host_voltage_buffer`
// - Also derived from this, the GPU buffer is then called
//   `voltage_buffer`
template<typename T, std::size_t D>
class NDArrayBuffer : public kotekan::kotekanLogging {
    const std::string buffer_name;        // "official" buffer name (for metadata)
    const std::string buffer_name_host;   // buffer name on host
    const std::string buffer_name_device; // buffer name on device

    const bool is_do_once;

    cudaCommand& cuda_command;

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
    NDArrayBuffer(const std::string& buffer_name, const std::string& quantity_name,
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
        ndarray(quantity_name, extents, dimnames, get_buffer_pointer(extents))
    //
    {
        set_log_level(cuda_command.get_log_level());
    }

    NDArrayBuffer(const std::string& buffer_name, const std::string& quantity_name,
                  const std::array<std::ptrdiff_t, D>& extents,
                  const std::array<std::string, D>& dimnames, cudaCommand& cuda_command,
                  const buffer_type_t buffer_type = buffer_type_t::standard) :
        NDArrayBuffer(buffer_name, quantity_name, extents, kotekan::strings_to_symbols(dimnames),
                      cuda_command, buffer_type) {}

    NDArrayBuffer(const std::string& buffer_name, const std::string& quantity_name,
                  const std::array<std::ptrdiff_t, D>& extents,
                  const std::array<const char*, D>& dimnames, cudaCommand& cuda_command,
                  const buffer_type_t buffer_type = buffer_type_t::standard) :
        NDArrayBuffer(buffer_name, quantity_name, extents, kotekan::strings_to_symbols(dimnames),
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

    bool has_metadata() const {
        const std::shared_ptr<const metadataObject> mc =
            cuda_command.get_device().get_gpu_memory_array_metadata(buffer_name_device,
                                                                    get_instance_num());
        if (!mc)
            return false;
        const std::shared_ptr<const chordMetadata> metadata = get_chord_metadata(mc);
        if (!metadata)
            return false;
        return true;
    }

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
        if (!(metadata->get_name() == ndarray.quantity_name()))
            ERROR("buffer name: {:s}, metadata name: {:s}, quantity_name: {:s}", buffer_name,
                  metadata->get_name(), ndarray.quantity_name());
        assert(metadata->get_name() == ndarray.quantity_name());
        assert(metadata->type == ndarray.value_datatype);
        assert(metadata->dims == ndarray.rank);
        for (std::size_t d = 0; d < ndarray.rank; ++d) {
            if (!(metadata->get_dimension_name(d) == ndarray.dimname(d)))
                ERROR("buffer name: {:s}, dimension: {:d}, metadata dimension name: {:s}, ndarray "
                      "dimname: {:s}",
                      buffer_name, d, metadata->get_dimension_name(d),
                      std::string(ndarray.dimname(d)));
            assert(metadata->get_dimension_name(d) == ndarray.dimname(d));
            assert(metadata->dim[d] == int(ndarray.extent(d)));
            assert(metadata->stride[d] == ndarray.stride(d));
        }
        // TODO: check `fpgq_seq_num`
    }

    void set_metadata(const std::shared_ptr<const chordMetadata>& other_metadata) const {
        const std::shared_ptr<metadataObject> mc =
            cuda_command.get_device().create_gpu_memory_array_metadata(
                buffer_name_device, get_instance_num(), other_metadata->parent_pool);
        const std::shared_ptr<chordMetadata> metadata = get_chord_metadata(mc);
        metadata->deepCopy(other_metadata);
        metadata->set_name(ndarray.quantity_name());
        metadata->type = ndarray.value_datatype;
        metadata->dims = ndarray.rank;
        for (std::size_t d = 0; d < ndarray.rank; ++d) {
            metadata->set_array_dimension(d, ndarray.extent(d), std::string(ndarray.dimname(d)));
            metadata->stride[d] = ndarray.stride(d);
        }
        // TODO: set `fpgq_seq_num`
    }

    // Poison

    // Poison an NDArray buffer
    void set_to_poison(const std::uint8_t poison_value) {
        const std::ptrdiff_t buffer_length = length_in_bytes();
        void* const buffer_device_ptr = ndarray.data();
        assert(buffer_device_ptr);
        const cudaStream_t cuda_stream =
            cuda_command.get_device().getStream(cuda_command.get_cuda_stream_id());
        CHECK_CUDA_ERROR(
            cudaMemsetAsync(buffer_device_ptr, poison_value, buffer_length, cuda_stream));
    }

    // Check an NDArray buffer for poison
    void check_for_poison(const std::uint8_t poison_value) {
        T poison;
        // The cast suppresses a bogus -Wclass-memaccess on GCC.
        std::memset(static_cast<void*>(&poison), poison_value, sizeof poison);
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
    }

    // I/O

    std::ostream& output(std::ostream& os) const {
        return os << "NDArrayBuffer<" << ndarray.value_datatype << "," << ndarray.rank << ">("
                  << buffer_name << "," << ndarray.quantity_name() << ")";
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
