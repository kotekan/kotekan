#ifndef BUFFER_CONTAINER_H
#define BUFFER_CONTAINER_H

#include "buffers.h"

#include <string>
#include <map>

using std::string;
using std::map;

class bufferContainer {

public:
    bufferContainer();
    ~bufferContainer();
    void add_buffer(const string &name, Buffer * buf);
    Buffer * get_buffer(const string &name);
    map<string, Buffer*> & get_buffer_map();
protected:
    map<string, Buffer*> buffers;
};

#endif
