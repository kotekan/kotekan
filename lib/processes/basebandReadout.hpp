#ifndef BASEBAND_READOUT_H
#define BASEBAND_READOUT_H

#include <string>
#include <mutex>

#include "buffer.h"
#include "chimeMetadata.h"
#include "KotekanProcess.hpp"


class basebandDump {
public:
    // Initializes the container with all parameters, and allocates memory for data
    // but does not fill in the data.
    basebandDump(
            uint64_t event_id_,
            uint32_t freq_id_,
            uint32_t num_elements_,
            int64_t data_start_fpga_,
            int64_t data_length_fpga_
            );
    ~basebandDump();

    uint64_t event_id;
    uint32_t freq_id;
    uint32_t num_elements;
    int64_t data_start_fpga;
    int64_t data_length_fpga;
    // I think if I change this to a shared pointer I can just pass a copy off to
    // a new thread and the data will persist until the thread ends.
    std::shared_ptr<uint8_t> data;
};


class bufferManager {
public:
    bufferManager(Buffer * buf_, int length_);
    ~bufferManager();

    int add_replace_frame(int frame_id);
    basebandDump get_data(
            uint64_t event_id,
            int64_t trigger_start_fpga,
            int64_t trigger_length_fpga
            );

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
    int num_elements;
    int samples_per_data_set;
};



#endif
