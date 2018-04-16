#ifndef BASEBAND_READOUT_H
#define BASEBAND_READOUT_H

#include <string>
#include <mutex>

#include "gsl-lite.hpp"

#include "buffer.h"
#include "chimeMetadata.h"
#include "KotekanProcess.hpp"
#include "gpsTime.h"


/* A container for baseband data and metadata.
 *
 * The use of a shared pointer to point to an array means that this class is copyable
 * without copying the underlying data buffer. However the memory for the underlying
 * buffer is managed and is deleted when the last copy of the container goes out of
 * scope.
 *
 */
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

    const uint64_t event_id;
    const uint32_t freq_id;
    const uint32_t num_elements;
    const int64_t data_start_fpga;
    const int64_t data_length_fpga;
    // For keeping track of references.
    std::shared_ptr<uint8_t> data_ref;
    // Span used for data access.
    const gsl::span<uint8_t> data;

};



class basebandReadout : public KotekanProcess {
public:
    basebandReadout(Config& config, const string& unique_name,
                    bufferContainer &buffer_container);
    virtual ~basebandReadout();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread() override;
private:
    struct Buffer * buf;
    std::string base_dir;
    std::string file_ext;
    int num_frames_buffer;
    int num_elements;
    int samples_per_data_set;

    int next_frame, oldest_frame;
    std::vector<std::mutex> frame_locks;
    std::mutex manager_lock;

    void listen_thread();
    int add_replace_frame(int frame_id);
    void lock_range(int start_frame, int end_frame);
    void unlock_range(int start_frame, int end_frame);
    basebandDump get_data(
            uint64_t event_id,
            int64_t trigger_start_fpga,
            int64_t trigger_length_fpga
            );

};



#endif
