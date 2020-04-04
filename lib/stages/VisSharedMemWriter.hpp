/******************************************************************
@file
@brief Stage for writing visibility data to a shared memory region.
- VisSharedMemWriter : public kotekan::Stage
*******************************************************************/
#ifndef VISSHAREDMEMWRITER_HPP
#define VISSHAREDMEMWRITER_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.h"            // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"  // for dset_id_t, fingerprint_t
#include "visBuffer.hpp"       // for visFrameView
#include "visUtil.hpp"         // for time_ctype

#include <cstdint>     // for uint32_t, uint64_t, uint8_t
#include <map>         // for map
#include <semaphore.h> // for sem_t
#include <stdexcept>   // for runtime_error
#include <string>      // for string, operator+

/**
 * @class VisSharedMemWriter
 * @brief Export calibration data out to a shared memory region
 *
 * This stage is an improvement upon visCalWriter. It exports the
 * most recent samples of the calibration data stream to a
 * fixed-length ring-buffer-like shared memory region.
 *
 * The shared memory region is composed of a Structured Data Region
 * (summarises the structure for the data), an Access Record region
 * (records the writer's access to the Data region), and the Data region
 * (ring buffer which contains the data and metadata from the frame).
 *
 * The structure of the shared memory region can be seen in the figure in
 * this issue: https://github.com/kotekan/kotekan/issues/692.
 *
 * The first section of the structured data region contains 0 if if the stage
 * is gracefully shut down. Otherwise, it will contain the number of writes.
 *
 * To ensure the ability to read while the stage is continuously
 * writing to the file, readers can inspect the access record to assess
 * the validity of the data. The access record has a 1:1 correlation to
 * the Data region. Every time the Data is modified, the access record
 * will have the timestamp of modification. While data is written,
 * its access record will be set to -1. Acquiring a semaphore, with the
 * same name as the shared memory region is required before reading
 * from or writing to the access record. The Data region is lock free.
 *
 * This stage writes out the data it receives with minimal processing.
 * Removing certain fields from the output must be done in a prior
 * transformation.
 *
 * A Valve stage should be added before this buffer process
 * to prevent data backing up during semaphore deadlocks.
 *
 * To obtain the metadata about the stream received usage of the datasetManager
 * is required.
 *
 * The dataset ID can be changed in the incoming stream. However, structural
 * parameters that could affect the size of the data cannot be adjusted without
 * triggering a FATAL_ERROR. By default these are `input`, `frequencies`, `products`,
 * `stack`, `gating`, and `metadata`. This list can be *added* to using the config
 * variable `critical_states`. Any state change not considered critical
 * will continue as normal.
 *
 * @par Buffers
 * @buffer in_buf The buffer streaming data to write
 *          @buffer_format visBuffer structured
 *          @buffer_metadata visMetadata
 *
 * @conf    root_path       String. Location in filesystem containing
 *                          shared memory and semaphore.
 * @conf    fname           Name of shared memory region and semaphore.
 * @conf    nsamples        Number of time samples stored in ring buffer.
 * @conf    sem_wait_time   Maximum time that the writer will wait to
 *                          unlock a semaphore. If it hits the limit
 *                          Kotekan will shut down the stage.
 * @conf    critical_states List of strings. A list of state types to consider
 *                          critical. That is, if they change in the incoming
 *                          data stream then Kotekan will shut down.
 *
 * @author Anja Boskovic
 */
class VisSharedMemWriter : public kotekan::Stage {

public:
    VisSharedMemWriter(kotekan::Config& config, const std::string& unique_name,
                       kotekan::bufferContainer& buffer_container);

    ~VisSharedMemWriter();

    void main_thread() override;


protected:
    // Input buffer to read from
    Buffer* in_buf;

    // Semaphore for updating access record
    sem_t* sem;

    // Pointers to shared memory addresses for structured data, access record and ringBuffer
    uint64_t* structured_data_addr;
    int64_t* access_record_addr;
    uint8_t* buf_addr;

    // Parameters that define structure of ring buffer
    //
    struct RingBufferStructure {
        // The number of time samples contained in ring buffer
        uint64_t _ntime;
        // The number of frequencies contained in each time sample
        uint64_t nfreq;
        // The size of each frame (valid byte + metadata + data + page alignment padding)
        uint64_t frame_size;
        // The size of each metadata section
        uint64_t metadata_size;
        // The size of each data section
        uint64_t data_size;
    };

    RingBufferStructure rbs;

    // Counter for the number of writes to the ring buffer
    // Set to 0, upon stage shut-down
    uint64_t num_writes = 0;
    // The semaphore wait time before a timeout, in seconds
    size_t _sem_wait_time;

    // Messages
    // Indicates that the written frame is valid
    const uint8_t valid = 1;
    // Space that the "valid byte" takes up
    size_t valid_size = 4;
    // Indicates that the ring buffer frames at those time_ind and freq_ind are
    // invalid
    const int64_t invalid = -1;

    // The number of elements in the structured data
    const size_t structured_data_num = 6;
    // the size of each element
    const size_t structured_data_size = sizeof(uint64_t);

    // The size of each record address
    const size_t access_record_size = 8;

    // The current position in the ring buffer of the most recent time sample
    modulo<int> cur_pos;

    // Map of indices for time samples and positions within the ring buffer
    std::map<time_ctype, size_t> vis_time_ind_map;

    /**
     * Create a shared memory region, and assign start of location
     * to pointer.
     *
     * @param   shm_name    Name of shared memory region
     * @param    shm_size    Size of shared memory region, in bytes.
     *
     * @return              Pointer to shared memory region of type uint8_t*.
     *
     **/
    uint8_t* assign_memory(std::string shm_name, size_t shm_size);

    /**
     * Write a frame to the shared memory region.
     * This function contains the logic that decides if and where the sample
     * will be indexed, and if any older samples should be dropped.
     *
     * @param frame     visFrameView frame to write
     * @param t         timestamp for sample
     * @param freq_ind  the index for the frequency associated with the frame
     *
     * @return          True if frame successfully written, False otherwise.
     **/
    bool add_sample(const visFrameView& frame, time_ctype t, uint32_t freq_ind);

    /**
     * Writes a frame directly to memory; updates access record.
     *
     * @param   frame       Frame that will be written.
     * @param   time_ind    Time index that it will be written to.
     * @param   freq_ind    Frequency index that it will be written to.
     **/
    void write_to_memory(const visFrameView& frame, uint32_t time_ind, uint32_t freq_ind);

    /**
     * Resets a region of shm to 0, and marks in access record as invalid.
     *
     * @param   time_ind        Time index of region to reset.
     **/
    void reset_memory(uint32_t time_ind);

    /**
     * Waits to acquire access record semaphore.
     **/
    void wait_for_semaphore();

    /**
     * Releases access record semaphore
     **/
    void release_semaphore();

    std::string _root_path, _fname;

    // List of critical states, if changed they will trigger an error
    std::set<std::string> critical_state_types;

    // List of unique dataset ids, whose fingerprints all match
    std::set<dset_id_t> unique_dataset_ids;

    // The stream's keyed fingerprint
    fingerprint_t stream_fingerprint;
};

#endif // VISSHAREDMEMWRITER_HPP
