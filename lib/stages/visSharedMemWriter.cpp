#include "visSharedMemWriter.hpp"
#include <cxxabi.h>             // for _forced_unwind
#include <fcntl.h>              // for O_CREAT, O_EXCL, O_RDWR
#include <stdlib.h>             // for exit
#include <string.h>             // for memcpy, strerror
#include <sys/mman.h>           // for mmap, shm_open, MAP_FAILED, MAP_SHARED, PROT_READ, PROT_WRITE
#include <sys/stat.h>           // for S_IRUSR, S_IWUSR
#include <sys/types.h>          // for uint
#include <atomic>               // for atomic_bool
#include <exception>            // for exception
#include <functional>           // for _Bind_helper<>::type, bind, function
#include <future>               // for async, future
#include <iterator>             // for reverse_iterator
#include <map>                  // for map, _Rb_tree_iterator
#include <regex>                // for match_results<>::_Base_type
#include <system_error>         // for system_error
#include <tuple>                // for get
#include <utility>              // for pair
#include <vector>               // for vector
#include "Hash.hpp"             // for Hash
#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "datasetManager.hpp"   // for datasetManager
#include "datasetState.hpp"     // for freqState, eigenvalueState, inputState, prodState, stackState
#include "fmt.hpp"              // for format, fmt
#include "kotekanLogging.hpp"   // for INFO, DEBUG, ERROR, FATAL_ERROR
#include "visBuffer.hpp"        // for visFrameView, visMetadata
#include "visUtil.hpp"          // for time_ctype, frameID, ts_to_double

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;


REGISTER_KOTEKAN_STAGE(visSharedMemWriter);

visSharedMemWriter::visSharedMemWriter(Config& config, const std::string& unique_name, bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&visSharedMemWriter::main_thread, this)) {

        // Fetch any simple configuration
        root_path = config.get_default<std::string>(unique_name, "root_path", "/dev/shm/");
        sem_name = config.get_default<std::string>(unique_name, "sem_name", "kotekan");
        fname_access_record = config.get_default<std::string>(unique_name, "fname_access_record", "calBufferAccessRecord");
        fname_buf = config.get_default<std::string>(unique_name, "fname_buf", "calBuffer");
        ntime = config.get_default<size_t>(unique_name, "nsamples", 512);

        // Setup the input vector
        in_buf = get_buffer("in_buf");
        register_consumer(in_buf, unique_name.c_str());

        // Check if any of the old buffer files exist
        DEBUG("Checking for and removing old buffer files...");
        check_remove(root_path + "sem." + sem_name);
        check_remove(root_path + fname_access_record);
        check_remove(root_path + fname_buf);


}

// make a deconstructor in which to deconstruct semaphores and shared memory
visSharedMemWriter::~visSharedMemWriter() {
    // make sure to unlink the semaphore and unmap the mappings
}

// takes the name of a shared memory address opens it, and maps the memory to the provided address pointer
uint8_t* visSharedMemWriter::assign_memory(std::string shm_name, size_t shm_size) {
        uint8_t* addr;

        int fd = shm_open(shm_name.c_str(), (O_CREAT | O_RDWR), (S_IRUSR | S_IWUSR));

        if (fd == -1) {
            throw std::runtime_error(
                fmt::format(fmt("Cannot open shared memory named {:s}: {:s}"), shm_name, strerror(errno)));
        }

        // Resize object to hold buffer
        if (ftruncate(fd, shm_size) == -1) {
            throw std::runtime_error(
                fmt::format(fmt("Failed to expand shared memory named {:s}: {:s}"), shm_name, strerror(errno)));
        }
        INFO("Resized to {} bytes\n", (long) shm_size);

        addr = (uint8_t*) mmap(nullptr, shm_size, (PROT_READ | PROT_WRITE), MAP_SHARED, fd, 0);
        if (addr == MAP_FAILED) {
            throw std::runtime_error(
                fmt::format(fmt("Failed to map shm {:s} to memory: {:s}."), shm_name, strerror(errno)));
        }

        // fd is no longer needed
        if (close(fd) == -1) {
            throw std::runtime_error(
                fmt::format(fmt("Failed to close file descriptor for shm {:s}: {:s}."), shm_name, strerror(errno)));
        }

        return addr;
}

bool visSharedMemWriter::add_sample(const visFrameView& frame, time_ctype t, uint32_t freq_ind) {
    // process the time sample

    if ( vis_time_ind_map.count(t) != 0) {
        // if the time is already indexed, write to memory at that location
        write_to_memory(frame, vis_time_ind_map.at(t), freq_ind);
        return true;
    }
    else if (vis_time_ind_map.size() < ntime) {
        // if there is still empty space in the shared buffer
        // add an additional time index
        cur_pos++;
        vis_time_ind_map[t] = cur_pos;
        write_to_memory(frame, vis_time_ind_map.at(t), freq_ind);
        return true;
    }

    time_ctype max_time = vis_time_ind_map.rbegin()->first;
    time_ctype min_time = vis_time_ind_map.begin()->first;

    if (t < min_time) {
        // this data is older than anything else in the map, so we should
        // just drop it
        INFO("Dropping integration as buffer (FPGA count: {:d}) arrived too late (minimum in pool {:d})", t.fpga_count, min_time.fpga_count);
        return false;
    }

    else if (t > max_time) {
        // we need to drop the oldest time
        // and replace it with the new time index
        cur_pos++;
        reset_memory(cur_pos);
        vis_time_ind_map.erase(min_time);
        vis_time_ind_map[t] = cur_pos;
        write_to_memory(frame, vis_time_ind_map.at(t), freq_ind);
        return true;
    }
    else {
        INFO("Dropping integration as buffer (FPGA count: {:d}) arrived too late (only accepting new times greater than {:d})", t.fpga_count, max_time.fpga_count);
        return false;
    }
}

void visSharedMemWriter::reset_memory(uint32_t time_ind) {

    uint8_t *buf_write_pos = buf_addr + (time_ind * nfreq * frame_size);
    uint64_t *record_write_pos = record_addr + time_ind;

    // each time, we block off an entire time_ind

    if (sem_wait(sem) == -1) {
        FATAL_ERROR("Failed to acquire semaphore {}", sem_name);
        return;
    }
    memcpy(record_write_pos, &in_progress, sizeof(in_progress));
    if (sem_post(sem) == -1) {
        FATAL_ERROR("Failed to post semaphore {}", sem_name);
        return;
    }

    std::vector<char> zeros(nfreq * frame_size, 0);
    char* tmp = zeros.data();
    memcpy(buf_write_pos, &tmp, nfreq * frame_size);

    int64_t invalid = -1;

    if (sem_wait(sem) == -1) {
        FATAL_ERROR("Failed to acquire semaphore {}", sem_name);
        return;
    }
    memcpy(record_write_pos, &invalid, sizeof(invalid));
    if (sem_post(sem) == -1) {
        FATAL_ERROR("Failed to post semaphore {}", sem_name);
        return;
    }
}

void visSharedMemWriter::write_to_memory(const visFrameView& frame, uint32_t time_ind, uint32_t freq_ind) {

    uint8_t *buf_write_pos = buf_addr + ((time_ind * nfreq + freq_ind) * frame_size);
    uint64_t *record_write_pos = record_addr + time_ind;

    // each time, we block off an entire time_ind

    if (sem_wait(sem) == -1) {
        FATAL_ERROR("Failed to acquire semaphore {}", sem_name);
        return;
    }
    memcpy(record_write_pos, &in_progress, sizeof(in_progress));
    if (sem_post(sem) == -1) {
        FATAL_ERROR("Failed to release semaphore {}", sem_name);
        return;
    }

    memcpy(buf_write_pos + 4, frame.metadata(), metadata_size);
    memcpy(buf_write_pos + metadata_size + 4, frame.data(), data_size);
    memcpy(buf_write_pos, &one, 4);

    uint64_t fpga_seq = frame.metadata()->fpga_seq_start;

    if (sem_wait(sem) == -1) {
        FATAL_ERROR("Failed to acquire semaphore {}", sem_name);
        return;
    }

    INFO("Writing fpga_seq {} to index {}", fpga_seq, time_ind);
    memcpy(record_write_pos, &fpga_seq, sizeof(fpga_seq));

    if (sem_post(sem) == -1) {
        FATAL_ERROR("Failed to release semaphore {}", sem_name);
        return;
    }
}


void visSharedMemWriter::main_thread() {
    INFO("Reached main thread");

    frameID frame_id(in_buf);

    cur_pos = modulo<int>(ntime);

    // Create the semaphore, and gain first access to it
    sem = sem_open(
            sem_name.c_str(),
            (O_CREAT | O_EXCL),
            (S_IRUSR | S_IWUSR),
            1
        );

    if (sem == SEM_FAILED) {
        perror("sem_open");
        exit(errno);
    }

    INFO("Semaphore created.\n");

    // Acquire semaphore until shared memory is created
    if (sem_wait(sem) == -1) {
        FATAL_ERROR("Failed to acquire semaphore {}", sem_name);
        return;
    }


    // Set up the structure of the ring buffer shared memory
    // Get one frame for reference
    wait_for_full_frame(in_buf, unique_name.c_str(), frame_id);

    auto frame = visFrameView(in_buf, frame_id);

    // Set up the structure of the access record shared memory
    size_t access_record_size = sizeof(uint64_t);

    record_addr = (uint64_t*) assign_memory(fname_access_record, ntime * access_record_size);

    // Get properties of stream from first frame and datasetManager
    std::map<uint32_t, uint32_t> freq_id_map;
    auto& dm = datasetManager::instance();
    auto sstate_fut = std::async(&datasetManager::dataset_state<stackState>, &dm, frame.dataset_id);
    auto istate_fut = std::async(&datasetManager::dataset_state<inputState>, &dm, frame.dataset_id);
    auto pstate_fut = std::async(&datasetManager::dataset_state<prodState>, &dm, frame.dataset_id);
    auto fstate_fut = std::async(&datasetManager::dataset_state<freqState>, &dm, frame.dataset_id);
    auto evstate_fut = std::async(&datasetManager::dataset_state<eigenvalueState>, &dm, frame.dataset_id);

    const inputState* istate = istate_fut.get();
    const prodState* pstate = pstate_fut.get();
    const stackState* sstate = sstate_fut.get();
    const freqState* fstate = fstate_fut.get();
    const eigenvalueState* evstate = evstate_fut.get();


    if (!istate || !pstate || !fstate) {
        ERROR("Required datasetState not found for dataset ID {}\nThe following required states "
                "were found:\ninputState - {:p}\nprodState - {:p}\nfreqState - {:p}\n",
                frame.dataset_id, (void*)istate, (void*)pstate, (void*)fstate);
        throw std::runtime_error("Could not write to shared memory.");
    }

    // Build the frequency index
    uint ind = 0;
    for (auto& f : fstate->get_freqs())
        freq_id_map[f.first] = ind++;

    // Count the eigenvalue index
    size_t num_ev;
    if (evstate) {
        num_ev = evstate->get_num_ev();
    } else {
        num_ev = 0;
    }

    // Figure out the ring buffer structure
    nfreq = fstate->get_freqs().size();
    size_t ninput = istate->get_inputs().size();
    size_t nvis = sstate ? sstate->get_num_stack() : pstate->get_prods().size();

    // Set the alignment (in kB)
    size_t alignment = 4; // Align on page boundaries

    // Calculate the ring buffer stricture
    auto layout = visFrameView::calculate_buffer_layout(ninput, nvis, num_ev);
    data_size = layout.first;
    metadata_size = sizeof(visMetadata);
    frame_size = _member_alignment(data_size + metadata_size + 4, alignment * 1024);

    // memory_size should be ntime * nfreq * file_frame_size (data + metadata)
    buf_addr = assign_memory(fname_buf, ntime * nfreq * frame_size);
    INFO("Created the shared memory segments\n");
    if (sem_post(sem) == -1) {
        FATAL_ERROR("Failed to release semaphore {}", sem_name);
        return;
    }

    // Get the time and frequency of the frame
    time_ctype t = {std::get<0>(frame.time), ts_to_double(std::get<1>(frame.time))};
    uint32_t freq_ind = freq_id_map.at(frame.freq_id);

    add_sample(frame, t, freq_ind);

    mark_frame_empty(in_buf, unique_name.c_str(), frame_id++);

    // gets called once when kotekan is running
    while (!stop_thread) {


        // Wait for the buffer to be filled with data
        if (wait_for_full_frame(in_buf, unique_name.c_str(), frame_id) == nullptr) {
            break;
        }

        // Get a view of the current frame
        auto frame = visFrameView(in_buf, frame_id);

        // Get the time and frequency of the frame
        time_ctype t = {std::get<0>(frame.time), ts_to_double(std::get<1>(frame.time))};
        uint32_t freq_ind = freq_id_map.at(frame.freq_id);

        add_sample(frame, t, freq_ind);


        // marks the buffer and moves on
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id++);

    }
}
