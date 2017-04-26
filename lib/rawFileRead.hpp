#ifndef RAW_FILE_READ_H
#define RAW_FILE_READ_H

#include "buffers.h"
#include "KotekanProcess.hpp"
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>

class rawFileRead : public KotekanProcess {
public:
    rawFileRead(Config &config,
                const string& unique_name,
                 struct Buffer &buf,
                 bool generate_info_object,
                 bool repeat_frame,
                 const std::string &base_dir,
                 const std::string &file_name,
                 const std::string &file_ext);
    virtual ~rawFileRead();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread();
private:
    struct Buffer &buf;
    void * tmp_buf;
    bool generate_info_object;
    bool repeat_frame;
    std::string base_dir;
    std::string file_name;
    std::string file_ext;
};

#endif