#ifndef N_DISK_FILE_WRITE_H
#define N_DISK_FILE_WRITE_H

#include "buffers.h"
#include "KotekanProcess.hpp"
#include <string>

using std::string;

class nDiskFileWrite : public KotekanProcess {
public:
    nDiskFileWrite(Config &config,
                   const string& unique_name,
                   struct Buffer &buf,
                   int disk_id,
                   const string &dataset_name);
    virtual ~nDiskFileWrite();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread();
private:
    struct Buffer &buf;

    int disk_id;
    int num_disks;

    string dataset_name;
    string disk_base;
    string disk_set;
    bool write_to_disk;
};

#endif
