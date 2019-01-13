#ifndef RAW_FILE_READ_H
#define RAW_FILE_READ_H

#include "buffer.h"
#include "stage.hpp"

#include <cstdio>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

class rawFileRead : public kotekan::Stage {
public:
    rawFileRead(kotekan::Config& config, const string& unique_name,
                kotekan::bufferContainer& buffer_container);
    virtual ~rawFileRead();
    void main_thread() override;

private:
    struct Buffer* buf;
    std::string base_dir;
    std::string file_name;
    std::string file_ext;
    // Interrupt Kotekan if run out of files to read
    bool end_interrupt;
};

#endif