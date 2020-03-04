#ifndef VISSHAREDMEMWRITER_HPP
#define VISSHAREDMEMWRITER_HPP

#include "Config.hpp"               // for Config
#include "StageFactory.hpp"         // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Stage.hpp"                // for Stage
#include <future>                   // for async
#include "datasetManager.hpp"       // for dset_id_t, datasetManager
#include "datasetState.hpp"         // for freqState
#include "buffer.h"                 // for mark_frame_empty, register_consumer, wait_for_full_frame
#include "bufferContainer.hpp"      // for bufferContainer
#include "visBuffer.hpp"            // for visFrameView
#include "kotekanLogging.hpp"       // for INFO, ERROR, WARN, FATAL_ERROR, DEBUG, logLevel

#include <cstdint>
#include <map>                      // for map
#include <string>                   // for string, operator+
#include <errno.h>                  // for ENOENT, errno

#include <semaphore.h>              // for sem_open, sem_wait, SEM_FAILED, sem_post
#include <sys/types.h>              // Type definitions
#include <sys/time.h>               // Time-related functions
#include <fcntl.h>
#include <unistd.h>                 // System calls, access, F_OK
#include <stdio.h>                  // size_t, remove
#include <sys/stat.h>

#include <sys/mman.h>               //mmap

#define CHECK(EXPR) ({ \
    int _r = EXPR; \
    if (_r == -1) { \
        perror(#EXPR); \
        exit(EXIT_FAILURE); \
    } \
    _r; \
})



class visSharedMemWriter : public kotekan::Stage {

public:
    visSharedMemWriter(kotekan::Config& config, const std::string& unique_name,
             kotekan::bufferContainer& buffer_container);

    ~visSharedMemWriter();

    void main_thread() override;


protected:
    // Input buffer to read from
    Buffer* in_buf;

    // Output every set number of frames
    int _output_period;

    // Semaphore for updating access record
    sem_t *sem;

    // Pointers to shared memory addresses for access record and ringBuffer
    uint64_t *record_addr;
    uint8_t *buf_addr;
    size_t ntime;

    const uint8_t ONE = 1;
    const uint64_t in_progress = -1;

    size_t access_record_size;
    size_t metadata_size;
    size_t data_size;
    size_t frame_size;

    template<typename T>
    T assign_memory(std::string shm_name, int shm_size, T addr);

    void write_to_memory(const visFrameView& frame, size_t index);

    std::string root_path, sem_name, fname_access_record, fname_buf;
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
