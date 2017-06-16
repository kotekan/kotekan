#include "bufferContainer.hpp"

bufferContainer::bufferContainer() {

}

bufferContainer::~bufferContainer() {

}

void bufferContainer::add_buffer(const string& name, Buffer* buf) {
    if (buffers.count(name) != 0) {
        throw std::runtime_error("The buffer named " + name + " already exists!");
        return;
    }
    buffers[name] = buf;
}

Buffer* bufferContainer::get_buffer(const string& name) {
    if (buffers.count(name) == 0) {
        throw std::runtime_error("The buffer named " + name + " doesn't exist!");
        return nullptr;
    }
    return buffers[name];
}

