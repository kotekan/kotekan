#include "bufferContainer.hpp"

bufferContainer::bufferContainer() {

}

bufferContainer::~bufferContainer() {

}

void bufferContainer::add_buffer(const string& name, Buffer* buf) {
    buffers[name] = buf;
}

Buffer* bufferContainer::get_buffer(const string& name) {
    return buffers[name];
}

