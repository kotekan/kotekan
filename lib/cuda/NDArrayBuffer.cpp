#include "NDArrayBuffer.hpp"

// TODO: For testing
template class NDArrayBuffer<unsigned char, 2>;
template class NDArrayBuffer<int, 3>;
template class NDArrayBuffer<float, 4>;

const std::shared_ptr<const chordMetadata> get_buffer_metadata(cudaCommand& cuda_command,
                                                               const std::string& buffer_name) {
    const std::string buffer_name_device(buffer_name + "_buffer");
    const std::shared_ptr<const metadataObject> mc =
        cuda_command.get_device().get_gpu_memory_array_metadata(buffer_name_device,
                                                                cuda_command.get_instance_num());
    const std::shared_ptr<const chordMetadata> metadata = get_chord_metadata(mc);
    return metadata;
}
