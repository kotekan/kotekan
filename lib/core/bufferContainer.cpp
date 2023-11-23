#include "bufferContainer.hpp"

#include "buffer.hpp"

#include "fmt.hpp" // for format, fmt

#include <stdexcept> // for runtime_error

using std::map;
using std::string;

namespace kotekan {

bufferContainer::bufferContainer() {}

bufferContainer::~bufferContainer() {}

void bufferContainer::add_buffer(const string& name, GenericBuffer* buf) {
    if (buffers.count(name) != 0) {
        throw std::runtime_error(fmt::format(fmt("The buffer named {:s} already exists!"), name));
    }
    buffers[name] = buf;
}


Buffer* bufferContainer::get_buffer(const string& name) {
    GenericBuffer* gb = get_generic_buffer(name);
    if (!gb->is_basic())
        throw std::runtime_error(
            fmt::format(fmt("The buffer named {:s} is not a basic Buffer!"), name));
    return dynamic_cast<Buffer*>(gb);
}

GenericBuffer* bufferContainer::get_generic_buffer(const string& name) {
    if (buffers.count(name) == 0)
        throw std::runtime_error(fmt::format(fmt("The buffer named {:s} doesn't exist!"), name));
    GenericBuffer* gb = buffers[name];
    return gb;
}

map<string, GenericBuffer*>& bufferContainer::get_buffer_map() {
    return buffers;
}

map<string, Buffer*> bufferContainer::get_basic_buffer_map() {
    map<string, Buffer*> b;
    for (auto v : buffers)
        if (v.second->is_basic())
            b[v.first] = dynamic_cast<Buffer*>(v.second);
    return b;
}

void bufferContainer::set_buffer_map(map<string, GenericBuffer*>& buffer_map) {
    buffers = buffer_map;
}

} // namespace kotekan
