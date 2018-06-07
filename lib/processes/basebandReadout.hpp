/*****************************************
@file
@brief Processes for triggered baseband recording
- basebandDumpData
- basebandReadout : public KotekanProcess
*****************************************/
#ifndef BASEBAND_READOUT_H
#define BASEBAND_READOUT_H

#include <string>
#include <mutex>
#include <queue>
#include <tuple>

#include "gsl-lite.hpp"

#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>

#include "buffer.h"
#include "chimeMetadata.h"
#include "KotekanProcess.hpp"
#include "gpsTime.h"
#include "baseband_request_manager.hpp"
#include "visUtil.hpp"


#define TARGET_CHUNK_SIZE 1024 * 1024


/**
 * @struct basebandDumpData
 * @brief A container for baseband data and metadata.
 *
 * @note The use of a shared pointer to point to an array means that this class
 *       is copyable without copying the underlying data buffer. However the
 *       memory for the underlying buffer is managed and is deleted when the
 *       last copy of the container goes out of scope.
 *
 * @author Kiyoshi Masui
 */
class basebandDumpData {
    public:
    // Initializes the container with all parameters, and allocates memory for
    // data but does not fill in the data.
    basebandDumpData(
            uint64_t event_id_,
            uint32_t freq_id_,
            uint32_t num_elements_,
            int64_t data_start_fpga_,
            int64_t data_length_fpga_);
    ~basebandDumpData();

    // Metadata.
    const uint64_t event_id;
    const uint32_t freq_id;
    const uint32_t num_elements;
    const int64_t data_start_fpga;
    const int64_t data_length_fpga;
private:
    // Keeps track of references to the underlying data array.
    const std::shared_ptr<uint8_t> data_ref;
public:
    // For data access. Array has length `num_elements * data_length_fpga`.
    const gsl::span<uint8_t> data;

};


/**
 * @class basebandReadout
 * @brief Buffer baseband data and record it to disk upon request.
 *
 * This task manages a kotekan buffer, keeping it mostly full such that it subsets
 * of the data can be written upon triggered request.
 *
 * @par Buffers
 * @buffer in_buf buffer to manage and read. Must be several frames larger than
 *                ``num_frames_buffer`` config parameter.
 *         @buffer_format DPDK baseband ``samples_per_data_set x num_elements`` bytes
 *         @buffer_metadata chimeMetadata
 *
 * @conf  num_elements          Int. The number of elements (i.e. inputs) in the
 *                              correlator data.
 * @conf  samples_per_data_set  Int. The number of time samples in a frame.
 * @conf  num_frames_buffer     Int. Number of buffer frames to simultaneously keep
 *                              full of data.
 * @conf  base_dir              String. Directory for writing triggered dumps.
 * @conf  file_ext              String. File extension for triggered dumps
 *                              (including the '.').
 *
 * @author Kiyoshi Masui, Davor Cubranic
 */
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
    std::vector<input_ctype> inputs;

    int next_frame, oldest_frame;
    std::vector<std::mutex> frame_locks;
    std::mutex manager_lock;
    std::queue<std::tuple<basebandDumpData, std::shared_ptr<BasebandDumpStatus>>> write_q;

    void write_thread();
    void listen_thread(const uint32_t freq_id);
    int add_replace_frame(int frame_id);
    void lock_range(int start_frame, int end_frame);
    void unlock_range(int start_frame, int end_frame);
    basebandDumpData get_data(
            uint64_t event_id,
            int64_t trigger_start_fpga,
            int64_t trigger_length_fpga
            );
};



#endif
