#ifndef BUFFER_CONTAINER_H
#define BUFFER_CONTAINER_H

#include "buffer.h"

#include <map>
#include <string>

using std::map;
using std::string;

class bufferContainer {

public:
    bufferContainer();
    ~bufferContainer();
    void add_buffer(const string& name, Buffer* buf);
    Buffer* get_buffer(const string& name);
    map<string, Buffer*>& get_buffer_map();
    void set_buffer_map(map<string, Buffer*>& buffer_map);

protected:
    map<string, Buffer*> buffers;
};

#endif
