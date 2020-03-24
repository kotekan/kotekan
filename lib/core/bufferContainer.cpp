#include "bufferContainer.hpp"

#include "buffer.h"

#include "fmt.hpp" // for format, fmt

#include <stdexcept> // for runtime_error

using std::map;
using std::string;

namespace kotekan {

bufferContainer::bufferContainer() {}

bufferContainer::~bufferContainer() {}

void bufferContainer::add_buffer(const string& name, Buffer* buf) {
    if (buffers.count(name) != 0) {
        throw std::runtime_error(fmt::format(fmt("The buffer named {:s} already exists!"), name));
    }
    buffers[name] = buf;
}

Buffer* bufferContainer::get_buffer(const string& name) {
    if (buffers.count(name) == 0) {
        throw std::runtime_error(fmt::format(fmt("The buffer named {:s} doesn't exist!"), name));
        return nullptr;
    }
    return buffers[name];
}

map<string, Buffer*>& bufferContainer::get_buffer_map() {
    return buffers;
}

void bufferContainer::set_buffer_map(map<string, Buffer*>& buffer_map) {
    buffers = buffer_map;
}

} // namespace kotekan
