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
    void apply_config(uint64_t fpga_seq) override;
    void main_thread() override;
private:
    struct Buffer *buf;
    std::string base_dir;
    std::string file_name;
    std::string file_ext;
};

#endif