#ifndef BUFFER_CONTAINER_H
#define BUFFER_CONTAINER_H

#include "buffer.h"

#include <map>    // for map
#include <string> // for string


namespace kotekan {

class bufferContainer {

public:
    bufferContainer();
    ~bufferContainer();
    void add_buffer(const std::string& name, Buffer* buf);
    Buffer* get_buffer(const std::string& name);
    std::map<std::string, Buffer*>& get_buffer_map();
    void set_buffer_map(std::map<std::string, Buffer*>& buffer_map);

protected:
    std::map<std::string, Buffer*> buffers;
};

} // namespace kotekan

#endif
