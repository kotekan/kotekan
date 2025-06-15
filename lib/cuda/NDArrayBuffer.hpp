#ifndef NDARRAYBUFFER_HPP
#define NDARRAYBUFFER_HPP

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
#include <gpuDeviceInterface.hpp>
#include <kotekanLogging.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>

template<typename T, std::size_t D>
class NDArrayBuffer : public kotekan::kotekanLogging {

    // // This is a buffer, not just a `std::vector`.
    // // This is used e.g. for the `info` output of the Julia-generated kernels.
    // const bool is_buffer = true;

    // The buffer lives on the device (and not the host).
    const bool is_device_buffer = true;

    // This is a ring buffer.
    const bool is_ringbuffer = false;

    // There is only one copy of the buffer, we don't use multi-buffering.
    // This is used e.g. for data that doesn't vary in time such as the beamforming matrices.
    const bool is_once_buffer = false;

    const std::string buffer_name;        // "official" buffer name (for metadata)
    const std::string buffer_name_host;   // buffer name on host
    const std::string buffer_name_device; // buffer name on device

    cudaCommand& cuda_command;
    const int64_t gpu_frame_id;

    const std::string quantity;
    kotekan::NDArray<T, D> ndarray;

private:
    T* get_buffer_pointer(const std::array<std::ptrdiff_t, D>& extents) const {
        std::ptrdiff_t size = 1;
        for (std::size_t d = 0; d < D; ++d)
            size *= extents[d];
        const std::ptrdiff_t size_in_bytes = size * sizeof(T);
        void* const ptr = cuda_command.get_device().get_gpu_memory_array(
            buffer_name_device, gpu_frame_id, cuda_command.get_gpu_buffer_depth(), size_in_bytes);
        return static_cast<T*>(ptr);
    }

public:
    NDArrayBuffer(const std::string& buffer_name, const std::string& quantity,
                  const std::array<std::ptrdiff_t, D>& extents,
                  const std::array<kotekan::Symbol, D>& dimnames, cudaCommand& cuda_command,
                  const int64_t gpu_frame_id) :
        // metadata
        buffer_name(buffer_name),                            // e.g. "bb_beams"
        buffer_name_host("host_" + buffer_name + "_buffer"), // e.g. "host_bb_beams_buffer"
        buffer_name_device(buffer_name + "_buffer"),         // e.g. "bb_beams_buffer"
        // Buffer
        cuda_command(cuda_command), gpu_frame_id(gpu_frame_id),
        // NDArray
        quantity(quantity), // e.g. "J"
        ndarray(extents, dimnames, get_buffer_pointer(extents))
    //
    {}

    NDArrayBuffer(const std::string& buffer_name, const std::string& quantity,
                  const std::array<std::ptrdiff_t, D>& extents,
                  const std::array<std::string, D>& dimnames, cudaCommand& cuda_command,
                  const int64_t gpu_frame_id) :
        NDArrayBuffer(buffer_name, quantity, extents, kotekan::strings_to_symbols(dimnames),
                      cuda_command, gpu_frame_id) {}

    NDArrayBuffer(const std::string& buffer_name, const std::string& quantity,
                  const std::array<std::ptrdiff_t, D>& extents,
                  const std::array<const char*, D>& dimnames, cudaCommand& cuda_command,
                  const int64_t gpu_frame_id) :
        NDArrayBuffer(buffer_name, quantity, extents, kotekan::strings_to_symbols(dimnames),
                      cuda_command, gpu_frame_id) {}

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
                                                                    gpu_frame_id);
        const std::shared_ptr<const chordMetadata> metadata = get_chord_metadata(mc);
        return metadata;
    }
    std::shared_ptr<chordMetadata> get_metadata() {
        const std::shared_ptr<metadataObject> mc =
            cuda_command.get_device().get_gpu_memory_array_metadata(buffer_name_device,
                                                                    gpu_frame_id);
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
                buffer_name_device, gpu_frame_id, other_metadata->parent_pool);
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

    // TODO template<typename T1, std::size_t D1>
    // TODO void set_metadata(const NDArrayBuffer<T1, D1>& other_buffer) const {
    // TODO     const std::shared_ptr<const metadataObject> mc =
    // TODO         cuda_command.get_device().get_gpu_memory_array_metadata(buffer_name_device,
    // TODO                                                                 gpu_frame_id);
    // TODO     assert(mc);
    // TODO     assert(metadata_is_chord(mc));
    // TODO     const std::shared_ptr<const chordMetadata> metadata = get_chord_metadata(mc);
    // TODO }

    // Poison

    // Poison an NDArray buffer
    void set_to_poison(const std::uint8_t poison_value) {
        // assert(is_buffer);
        assert(is_device_buffer);
        assert(!is_ringbuffer);
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
        // assert(is_buffer);
        assert(is_device_buffer);
        assert(!is_ringbuffer);
#ifdef DEBUGGING
        const std::ptrdiff_t buffer_length = length_in_bytes();
        const void* const buffer_device_ptr = ndarray.data();
        assert(buffer_device_ptr);
        std::vector<std::uint8_t> local_data(buffer_length);
        CHECK_CUDA_ERROR(cudaMemcpy(local_data.data(), buffer_device_ptr, buffer_length,
                                    cudaMemcpyDeviceToHost));
        const bool found_error = std::memchr(local_data.data(), poison_value, buffer_length);
        if (found_error)
            FATAL_ERROR("NDArray buffer {:s} contains poison", buffer_name);
#endif
    }
};

#endif // #ifndef NDARRAYBUFFER_HPP
