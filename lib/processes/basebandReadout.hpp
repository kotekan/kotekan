#ifndef BASEBAND_READOUT_H
#define BASEBAND_READOUT_H

#include <string>
#include <mutex>

#include "buffer.h"
#include "chimeMetadata.h"
#include "KotekanProcess.hpp"


class basebandDump {
};


class bufferManager {
public:
    bufferManager(Buffer * buf_, int length_);
    ~bufferManager();

    int add_replace_frame(int frame_id);
    basebandDump get_data(int64_t trigger_start_fpga, int64_t trigger_length_fpga);

private:
    Buffer * buf;
    const int length;
    int next_frame, oldest_frame;
    std::vector<std::mutex> frame_locks;
    std::mutex manager_lock;
};


class basebandReadout : public KotekanProcess {
public:
    basebandReadout(Config& config, const string& unique_name,
                    bufferContainer &buffer_container);
    virtual ~basebandReadout();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread() override;
private:
    bufferManager * manager;
    void listen_thread();
    struct Buffer * buf;
    std::string base_dir;
    std::string file_ext;
    int num_frames_buffer;
};



#endif
