#ifndef VISSHAREDMEMWRITER_HPP
#define VISSHAREDMEMWRITER_HPP

#include <errno.h>              // for ENOENT, errno
#include <semaphore.h>          // for sem_t
#include <stdio.h>              // for perror, remove
#include <unistd.h>             // for access, F_OK
#include <cstdint>              // for uint32_t, uint64_t, uint8_t
#include <map>                  // for map
#include <stdexcept>            // for runtime_error
#include <string>               // for string, operator+
#include "Config.hpp"           // for Config
#include "Stage.hpp"            // for Stage
#include "buffer.h"             // for Buffer
#include "bufferContainer.hpp"  // for bufferContainer
#include "visBuffer.hpp"        // for visFrameView
#include "visUtil.hpp"          // for time_ctype

class visSharedMemWriter : public kotekan::Stage {

public:
    visSharedMemWriter(kotekan::Config& config, const std::string& unique_name,
             kotekan::bufferContainer& buffer_container);

    ~visSharedMemWriter();

    void main_thread() override;


protected:
    // Input buffer to read from
    Buffer* in_buf;

    // Semaphore for updating access record
    sem_t *sem;

    // Pointers to shared memory addresses for structured data, access record and ringBuffer
    uint64_t* structured_data_addr;
    uint64_t *access_record_addr;
    uint8_t *buf_addr;

    // Parameters that define structure of ring buffer

    // Counter for the number of writes to the ring buffer
    // Set to 0, upon stage shut-down
    uint64_t num_writes = 0;
    // The number of time samples contained in ring buffer
    size_t ntime;
    // The number of frequencies contained in each time sample
    uint64_t nfreq;
    // The size of each frame (valid byte + metadata + data + page alignment padding)
    uint64_t frame_size;
    // The size of each metadata section
    uint64_t metadata_size;
    // The size of each data section
    uint64_t data_size;

    // Messages
    // Indicates that the written frame is valid
    const uint8_t valid = 1;
    // Size of the "valid byte"
    size_t valid_size = 4;
    // Indicates that the ring buffer is being written to for that time sample
    const int64_t in_progress = -1;

    // The number of elements in the structured data
    size_t structured_data_num = 6;

    // The size of each record address
    uint64_t access_record_size;

    // The current position in the ring buffer of the most recent time sample
    uint32_t cur_pos;

    // Map of indices for time samples and positions within the ring buffer
    std::map<time_ctype, size_t> vis_time_ind_map;

    uint8_t* assign_memory(std::string shm_name, size_t shm_size);

    bool add_sample(const visFrameView& frame, time_ctype t, uint32_t freq_ind);

    void write_to_memory(const visFrameView& frame, uint32_t time_ind, uint32_t freq_ind);

    void reset_memory(uint32_t time_ind);

    std::string root_path, sem_name, fname_buf;
};

inline void check_remove(std::string fname) {
    // Check if we need to remove anything
    if (access(fname.c_str(), F_OK) != 0)
        return;
    // Remove
    if (remove(fname.c_str()) != 0) {
        if (errno != ENOENT)
            throw std::runtime_error("Could not remove file " + fname);
    }
}

#endif // VISSHAREDMEMWRITER_HPP
