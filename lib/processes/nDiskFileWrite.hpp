#ifndef N_DISK_FILE_WRITE_H
#define N_DISK_FILE_WRITE_H

#include "buffers.h"
#include "KotekanProcess.hpp"
#include <string>
#include <vector>
#include <thread>

using std::string;

class nDiskFileWrite : public KotekanProcess {
public:
    nDiskFileWrite(Config &config,
                   const string& unique_name,
                   bufferContainer &buffer_containter);
    virtual ~nDiskFileWrite();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread();
private:
    struct Buffer *buf;

    uint32_t disk_id;
    uint32_t num_disks;

    void file_write_thread(int disk_id);
    std::vector<std::thread> file_thread_handles;

    string dataset_name;
    string disk_base;
    string disk_set;
    bool write_to_disk;
    bool first_run;

    void mk_dataset_dir();
    void save_meta_data();
    void copy_gains(const string &gain_file_dir, const string &gain_file_name);
    string instrument_name;
};

#endif
