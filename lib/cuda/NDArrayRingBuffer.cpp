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
