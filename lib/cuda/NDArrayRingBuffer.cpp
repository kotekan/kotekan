#include "NDArrayRingBuffer.hpp"

std::ostream& operator<<(std::ostream& os, const read_descriptor_t& desc) {
    return os << "read_descriptor_t{claimed:" << desc.claimed << ",read:" << desc.read << "}";
}

std::ostream& operator<<(std::ostream& os, const extent_t& ext) {
    return os << "extent_t{m_begin:" << ext.m_begin << ",m_end:" << ext.m_end << "}";
}

// TODO: For testing
template class NDArrayRingBuffer<unsigned char, 2>;
template class NDArrayRingBuffer<int, 3>;
template class NDArrayRingBuffer<float, 4>;

const std::shared_ptr<const chordMetadata> get_ringbuffer_metadata(cudaCommand& cuda_command,
                                                                   const std::string& buffer_name) {
    const std::string signal_buffer_name("host_" + buffer_name + "_ringbuffer");
    RingBuffer* const ringbuffer(dynamic_cast<RingBuffer*>(
        cuda_command.get_host_buffers().get_generic_buffer(signal_buffer_name)));
    assert(ringbuffer);
    const std::shared_ptr<const metadataObject> mc = ringbuffer->get_metadata(0);
    assert(mc);
    const std::shared_ptr<const chordMetadata> metadata = get_chord_metadata(mc);
    assert(metadata);
    return metadata;
}
