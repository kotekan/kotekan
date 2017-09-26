#ifndef N_DISK_FILE_READ_H
#define N_DISK_FILE_READ_H

#include "buffer.h"
#include "KotekanProcess.hpp"
#include "vdif_functions.h"

class nDiskFileRead : public KotekanProcess {
public:
    nDiskFileRead(Config &config,
                   const string& unique_name,
                   bufferContainer &buffer_containter);
    ~nDiskFileRead();
	void file_read_thread(int disk_id);
    void apply_config(uint64_t fpga_seq) override;
    void main_thread();
private:
    struct Buffer *buf;
    std::vector<std::thread> file_thread_handles;

    int num_disks;
    int num_elements;
    int num_frequencies;
    string disk_base;
    string disk_set;
    string capture;
    int SK_STEP;
    bool WITH_RFI;
    bool Normalize;
    int THRESHOLD_SENSITIVITY;
};

#endif
