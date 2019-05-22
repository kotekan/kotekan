/*****************************************
@file
@brief Stages for triggered baseband recording
- basebandDumpData
- basebandReadout : public kotekan::Stage
*****************************************/
#ifndef BASEBAND_READOUT_H
#define BASEBAND_READOUT_H

#include "Stage.hpp"
#include "basebandReadoutManager.hpp"
#include "buffer.h"
#include "chimeMetadata.h"
#include "gpsTime.h"
#include "visUtil.hpp"

#include "gsl-lite.hpp"

#include <condition_variable>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <mutex>
#include <queue>
#include <string>
#include <tuple>


constexpr size_t TARGET_CHUNK_SIZE = 1024 * 1024;


/**
 * @struct basebandDumpData
 * @brief A container for baseband data and metadata.
 *
 * @note This class does not own the underlying data buffer, but provides a view
 *       (i.e., a `gsl::span`) to it. Users are responsible for managing the
 *       memory storage.
 *
 * @author Kiyoshi Masui
 */
struct basebandDumpData {
    /// Default constructor used to indicate error
    basebandDumpData();
    /// Initialize the container with all parameters but does not fill in the data.
    basebandDumpData(uint64_t event_id_, uint32_t freq_id_, uint32_t num_elements_,
                     int64_t data_start_fpga_, uint64_t data_length_fpga_,
                     timespec data_start_ctime_, uint8_t* data_ref);

    //@{
    /// Metadata.
    const uint64_t event_id;
    const uint32_t freq_id;
    const uint32_t num_elements;
    const int64_t data_start_fpga;
    const uint64_t data_length_fpga;
    const timespec data_start_ctime;
    //@}
    /// Data access. Array has length `num_elements * data_length_fpga`.
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
 * @conf  max_dump_samples      Int, default 2^30. Maximum number of samples in
 *                              baseband dump. Memory used for dumps limited to
 *                              3 x num_elements x this_number.
 * @conf  num_frames_buffer     Int. Number of buffer frames to simultaneously keep
 *                              full of data. Should be few less than in_buf length.
 * @conf  base_dir              String. Directory name (including trailing '/')
 *                              for writing triggered dumps.
 * @conf  write_throttle        Float, default 0. Add sleep time while writing dumps
 *                              equal to this factor times real time.
 *
 * @author Kiyoshi Masui, Davor Cubranic
 */
class basebandReadout : public kotekan::Stage {
public:
    basebandReadout(kotekan::Config& config, const string& unique_name,
                    kotekan::bufferContainer& buffer_container);
    virtual ~basebandReadout();
    void main_thread() override;

private:
    // settings from the config file
    std::string _base_dir;
    int _num_frames_buffer;
    int _num_elements;
    int _num_local_freq;
    int _samples_per_data_set;
    int64_t _max_dump_samples;
    double _write_throttle;
    std::vector<input_ctype> _inputs;

    struct Buffer* buf;
    int next_frame, oldest_frame;
    std::vector<std::mutex> frame_locks;

    std::mutex manager_lock;

    void listen_thread(const uint32_t freq_id, kotekan::basebandReadoutManager *mgrs[]);
    void write_thread(kotekan::basebandReadoutManager *mgrs[]);
    void write_dump(basebandDumpData data, kotekan::basebandDumpStatus& dump_status,
                    std::mutex& request_mtx);
    int add_replace_frame(int frame_id);
    void lock_range(int start_frame, int end_frame);
    void unlock_range(int start_frame, int end_frame);

    /**
     * @brief Make a private copy of the data from the ring buffer
     *
     * @param event_id unique identifier of the event in the FRB pipeline
     * @param trigger_start_fpga start time, or -1 to use the earliest data available
     * @param trigger_length_fpga number of FPGA samples to include in the dump
     *
     * @return A fully initialized `basebandDumpData` if the call succeeded, or
     * an empty one if the frame data was not availabe for the time requested
     */
    basebandDumpData get_data(uint64_t event_id, int64_t trigger_start_fpga,
                              int64_t trigger_length_fpga);

    /// baseband data array
    const std::unique_ptr<uint8_t[]> baseband_data;

    // the next/current dump to write (reset to nullptr after done)
    std::unique_ptr<basebandDumpData> dump_to_write;
    std::condition_variable ready_to_write;
    std::mutex dump_to_write_mtx;
};

#endif
