#ifndef RAW_FILE_READ_H
#define RAW_FILE_READ_H

#include "buffer.h"
#include "KotekanProcess.hpp"
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>

class rawFileRead : public KotekanProcess {
public:
    rawFileRead(Config& config, const string& unique_name,
                bufferContainer &buffer_container);
    virtual ~rawFileRead();
    void main_thread() override;
private:
    struct Buffer *buf;
    std::string base_dir;
    std::string file_name;
    std::string file_ext;
    // Interrupt Kotekan if run out of files to read
    bool end_interrupt;
};

#endif