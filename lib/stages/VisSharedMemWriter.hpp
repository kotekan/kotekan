#ifndef VISSHAREDMEMWRITER_HPP
#define VISSHAREDMEMWRITER_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.h"            // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "visBuffer.hpp"       // for visFrameView
#include "visUtil.hpp"         // for time_ctype
#include "datasetManager.hpp"  // for dset_id_t, fingerprint_t

#include <cstdint>     // for uint32_t, uint64_t, uint8_t
#include <map>         // for map
#include <semaphore.h> // for sem_t
#include <stdexcept>   // for runtime_error
#include <string>      // for string, operator+

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
    struct RingBufferStructure
    {
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

    uint8_t* assign_memory(std::string shm_name, size_t shm_size);

    bool add_sample(const visFrameView& frame, time_ctype t, uint32_t freq_ind);

    void write_to_memory(const visFrameView& frame, uint32_t time_ind, uint32_t freq_ind);

    void reset_memory(uint32_t time_ind);

    void wait_for_semaphore();

    void release_semaphore();

    std::string _root_path, _sem_name, _fname_buf;

    // List of critical states, if changed they will trigger an error
    std::set<std::string> critical_state_types;

    // List of unique dataset ids, whose fingerprints all match
    std::set<dset_id_t> unique_dataset_ids;

    // The stream's keyed fingerprint
    fingerprint_t stream_fingerprint;
};

#endif // VISSHAREDMEMWRITER_HPP
