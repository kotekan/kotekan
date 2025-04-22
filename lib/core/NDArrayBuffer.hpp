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
#include <cudaDeviceInterface.hpp>
#include <gpuDeviceInterface.hpp>
#include <kotekanLogging.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {
template<std::size_t D>
std::array<kotekan::Symbol, D> strings_to_symbols(const std::array<std::string, D>& strings) {
    std::array<kotekan::Symbol, D> symbols;
    for (std::size_t d = 0; d < D; ++d)
        symbols[d] = strings[d];
    return symbols;
}
template<std::size_t D>
std::array<kotekan::Symbol, D> strings_to_symbols(const std::array<const char*, D>& strings) {
    std::array<kotekan::Symbol, D> symbols;
    for (std::size_t d = 0; d < D; ++d)
        symbols[d] = strings[d];
    return symbols;
}

} // namespace

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

    kotekan::NDArray<T, D> ndarray;

    cudaDeviceInterface& device;
    int cuda_stream_id;
    // GPU buffer depth of a GPU command. Must be the same for all
    // commands in a pipeline. Should probablye be a buffer property
    // rather than a command property.
    int gpu_buffer_depth;
    // GPU frame id. Needed to access the current buffer since we're
    // multi-buffering.
    std::int64_t gpu_frame_id;

public:
    NDArrayBuffer(const std::string& buffer_name, const std::array<std::ptrdiff_t, D>& extents,
                  const std::array<kotekan::Symbol, D>& dimnames, cudaDeviceInterface& device,
                  const int cuda_stream_id, const int gpu_buffer_depth,
                  const std::int64_t gpu_frame_id) :
        // metadata
        buffer_name(buffer_name),                            // e.g. "bb_beams"
        buffer_name_host("host_" + buffer_name + "_buffer"), // e.g. "host_bb_beams_buffer"
        buffer_name_device(buffer_name + "_buffer"),         // e.g. "bb_beams_buffer"
                                                             // NDArray
        ndarray(extents, dimnames, nullptr),
        // Buffer
        device(device), cuda_stream_id(cuda_stream_id), gpu_buffer_depth(gpu_buffer_depth),
        gpu_frame_id(gpu_frame_id)
    //
    {
        void* const ptr = device.get_gpu_memory_array(buffer_name_device, gpu_frame_id,
                                                      gpu_buffer_depth, length_in_bytes());
        ndarray.set_data(static_cast<T*>(ptr));
    }

    NDArrayBuffer(const std::string& buffer_name, const std::array<std::ptrdiff_t, D>& extents,
                  const std::array<std::string, D>& dimnames, cudaDeviceInterface& device,
                  const int cuda_stream_id, const int gpu_buffer_depth,
                  const std::int64_t gpu_frame_id) :
        NDArrayBuffer(buffer_name, extents, strings_to_symbols(dimnames), device, cuda_stream_id,
                      gpu_buffer_depth, gpu_frame_id) {}

    NDArrayBuffer(const std::string& buffer_name, const std::array<std::ptrdiff_t, D>& extents,
                  const std::array<const char*, D>& dimnames, cudaDeviceInterface& device,
                  const int cuda_stream_id, const int gpu_buffer_depth,
                  const std::int64_t gpu_frame_id) :
        NDArrayBuffer(buffer_name, extents, strings_to_symbols(dimnames), device, cuda_stream_id,
                      gpu_buffer_depth, gpu_frame_id) {}

    virtual ~NDArrayBuffer() {}

    // NDArray:

    const kotekan::NDArray<T, D>& get_ndarray() const {
        return ndarray;
    }
    kotekan::NDArray<T, D>& get_ndarray() {
        return ndarray;
    }

    std::ptrdiff_t length_in_bytes() const {
        return get_ndarray().get_size() * sizeof(T);
    }

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

    // Metadata:

    void check_metadata() const {
        const std::shared_ptr<metadataObject> mc =
            device.get_gpu_memory_array_metadata(buffer_name_device, gpu_frame_id);
        assert(mc);
        assert(metadata_is_chord(mc));
        const std::shared_ptr<chordMetadata> metadata = get_chord_metadata(mc);
        assert(metadata->get_name() == buffer_name);
        assert(metadata->type == ndarray.value_datatype);
        assert(metadata->dims == ndarray.rank);
        for (std::size_t d = 0; d < ndarray.rank; ++d) {
            assert(metadata->get_dimension_name(d) == ndarray.dimname(d));
            assert(metadata->dim[d] == int(ndarray.extent(d)));
            assert(metadata->stride[d] == ndarray.stride(d));
        }
    }

    void set_metadata(const std::shared_ptr<chordMetadata>& other_metadata) const {
        std::shared_ptr<metadataObject> const mc = device.create_gpu_memory_array_metadata(
            buffer_name_device, gpu_frame_id, other_metadata->parent_pool);
        std::shared_ptr<chordMetadata> const metadata = get_chord_metadata(mc);
        *metadata = *other_metadata;
        metadata->set_name(buffer_name);
        metadata->type = ndarray.value_datatype;
        metadata->dims = ndarray.rank;
        for (std::size_t d = 0; d < ndarray.rank; ++d) {
            metadata->set_array_dimension(d, ndarray.extent(d), std::string(ndarray.dimname(d)));
            metadata->stride[d] = ndarray.stride(d);
        }
    }

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
        const cudaStream_t cuda_stream = device.getStream(cuda_stream_id);
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
