#ifndef BUFFER_CONTAINER_H
#define BUFFER_CONTAINER_H

#include "buffer.hpp"

#include <map>    // for map
#include <string> // for string


namespace kotekan {

class bufferContainer {

public:
    bufferContainer();
    ~bufferContainer();

    void add_buffer(const std::string& name, GenericBuffer* buf);

    GenericBuffer* get_generic_buffer(const std::string& name);
    Buffer* get_buffer(const std::string& name);

    std::map<std::string, GenericBuffer*>& get_buffer_map();
    void set_buffer_map(std::map<std::string, GenericBuffer*>& buffer_map);

protected:
    std::map<std::string, GenericBuffer*> buffers;
};

} // namespace kotekan

#endif
