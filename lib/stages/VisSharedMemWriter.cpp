#include "VisSharedMemWriter.hpp"

#include "Hash.hpp"           // for Hash
#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "datasetManager.hpp" // for datasetManager
#include "datasetState.hpp"   // for freqState, eigenvalueState, inputState, prodState, stackState
#include "kotekanLogging.hpp" // for INFO, DEBUG, ERROR, FATAL_ERROR
#include "prometheusMetrics.hpp" // for Counter, Metrics, MetricFamily, Gauge
#include "visBuffer.hpp"         // for visFrameView, visMetadata
#include "visUtil.hpp"           // for time_ctype, frameID, ts_to_double, current_time

#include "fmt.hpp" // for format, fmt

#include <atomic>       // for atomic_bool
#include <cxxabi.h>     // for _forced_unwind
#include <errno.h>      // for ENOENT, errno
#include <exception>    // for exception
#include <fcntl.h>      // for O_CREAT, O_EXCL, O_RDWR
#include <functional>   // for _Bind_helper<>::type, bind, function
#include <future>       // for async, future
#include <iterator>     // for reverse_iterator
#include <map>          // for map, _Rb_tree_iterator
#include <regex>        // for match_results<>::_Base_type
#include <stdio.h>      // for remove
#include <string.h>     // for memcpy, strerror
#include <sys/mman.h>   // for mmap, shm_open, MAP_FAILED, MAP_SHARED, PROT_READ, PROT_WRITE
#include <sys/stat.h>   // for S_IRUSR, S_IWUSR
#include <sys/types.h>  // for uint
#include <system_error> // for system_error
#include <tuple>        // for get
#include <unistd.h>     // for access, F_OK
#include <utility>      // for pair
#include <vector>       // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;


REGISTER_KOTEKAN_STAGE(VisSharedMemWriter);

void check_remove(std::string name) {
    // Check if we need to remove anything
    if (access(name.c_str(), F_OK) != 0)
        return;
    // Remove
    if (remove(name.c_str()) != 0) {
        if (errno != ENOENT)
            throw std::runtime_error("Could not remove file " + name);
    }
}

VisSharedMemWriter::VisSharedMemWriter(Config& config, const std::string& unique_name,
                                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&VisSharedMemWriter::main_thread, this)),
    dropped_frame_counter(Metrics::instance().add_counter(
        "kotekan_vissharedmemwriter_dropped_frame_total", unique_name, {"freq_id", "reason"})),
    access_record_wait_time_seconds(Metrics::instance().add_gauge(
        "kotekan_access_record_wait_time_seconds", unique_name, {"shm_name"})) {

    // Fetch any simple configuration
    _root_path = config.get_default<std::string>(unique_name, "root_path", "/dev/shm/");
    _name = config.get_default<std::string>(unique_name, "name", "calBuffer");
    rbs._ntime = config.get_default<uint64_t>(unique_name, "num_samples", 512);
    _wait_time = config.get_default<size_t>(unique_name, "wait_time", 120);

    // Set the list of critical states
    critical_state_types = {"frequencies", "inputs",      "products",
                            "stack",       "eigenvalues", "metadata"};
    auto t = config.get_default<std::vector<std::string>>(unique_name, "critical_states", {});
    for (const auto& state : t) {
        if (!FACTORY(datasetState)::exists(state)) {
            FATAL_ERROR("Unknown datasetState type '{}' given as `critical_state`", state);
            return;
        }
        critical_state_types.insert(state);
    }

    // Setup the input vector
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Check if any of the old buffer files exist
    // Remove them, if they do
    DEBUG("Checking for and removing old buffer files...");
    check_remove(_root_path + "sem." + _name);
    check_remove(_root_path + _name);
}

VisSharedMemWriter::~VisSharedMemWriter() {
    // make sure to unlink the semaphore and unmap the mappings
    // We are setting num_writes to 0 in the structured data,
    // to communicate to readers that the ring buffer is not being written to

    wait_for_semaphore();

    num_writes = 0;
    *structured_data_addr = num_writes;

    release_semaphore();
}

void VisSharedMemWriter::wait_for_semaphore() {
    // handles timed waits for semaphores
    // does a standard wait if the system clock is not accessible

#ifdef MAC_OSX
    if (sem_wait(sem) == -1) {
        FATAL_ERROR("Failed to acquire semaphore {}", _name);
        return;
    }
#else
    timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) == -1) {
        WARN("Failed to get system time. {:d} ({:s}) Not using timed semaphores.", errno,
             std::strerror(errno));

        double start_time = current_time();
        if (sem_wait(sem) == -1) {
            FATAL_ERROR("Failed to acquire semaphore {}", _name);
            return;
        }

        wait_time_average.add_sample(current_time() - start_time);
        access_record_wait_time_seconds.labels({_name}).set(wait_time_average.average());
        return;
    }

    ts.tv_sec += _wait_time;
    double start_time = current_time();

    if (sem_timedwait(sem, &ts) == -1) {
        FATAL_ERROR("sem_timedwait() timed out");
        return;
    }

    wait_time_average.add_sample(current_time() - start_time);
    access_record_wait_time_seconds.labels({_name}).set(wait_time_average.average());
    return;
#endif
}

void VisSharedMemWriter::release_semaphore() {
    if (sem_post(sem) == -1) {
        FATAL_ERROR("Failed to post semaphore {}", _name);
    }
    return;
}

uint8_t* VisSharedMemWriter::assign_memory(std::string shm_name, size_t shm_size) {
    // takes the name of a shared memory address, opens a shared memory of the provided size, and
    // maps the memory to a uint8_t* address pointer

    uint8_t* addr;

    int fd = shm_open(shm_name.c_str(), (O_CREAT | O_RDWR), (S_IRUSR | S_IWUSR));

    if (fd == -1) {
        FATAL_ERROR("Cannot open shared memory named {:s}: {:s}", shm_name, strerror(errno));
    }

    // Resize object to hold buffer
    if (ftruncate(fd, shm_size) == -1) {
        FATAL_ERROR("Failed to expand shared memory named {:s}: {:s}", shm_name, strerror(errno));
    }
    INFO("Resized to {} bytes", (long)shm_size);

    addr = (uint8_t*)mmap(nullptr, shm_size, (PROT_READ | PROT_WRITE), MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        FATAL_ERROR("Failed to map shm {:s} to memory: {:s}.", shm_name, strerror(errno));
    }

    // fd is no longer needed
    if (close(fd) == -1) {
        FATAL_ERROR("Failed to close file descriptor for shm {:s}: {:s}.", shm_name,
                    strerror(errno));
    }

    return addr;
}

void VisSharedMemWriter::add_sample(const visFrameView& frame, time_ctype t, uint32_t freq_ind) {
    // calculate the time index for time sample t, add the frame for time sample t at position
    // frequency index
    //
    // curr_pos always points to the ring buffer index for the most recent time

    if (vis_time_ind_map.count(t) != 0) {
        // if the time is already indexed, write to memory at that location
        write_to_memory(frame, vis_time_ind_map.at(t), freq_ind);
        return;
    } else if (vis_time_ind_map.size() == 0) {
        // the first sample added, so we do not increment by 1
        vis_time_ind_map[t] = cur_pos;
        write_to_memory(frame, vis_time_ind_map.at(t), freq_ind);
        return;
    }

    // obtain the most recent and oldest time
    time_ctype max_time = vis_time_ind_map.rbegin()->first;
    time_ctype min_time = vis_time_ind_map.begin()->first;

    if ((t > min_time) && (t < max_time)) {
        // if the time sample is not indexed, and is between the min_time and max_time, we are going
        // to just drop it
        INFO("Dropping integration as buffer (FPGA count: {:d}) arrived too late (only accepting "
             "new times greater than {:d})",
             t.fpga_count, max_time.fpga_count);
        dropped_frame_counter.labels({std::to_string(frame.freq_id), "order"}).inc();
        return;
    } else if (t < min_time) {
        // this data is older than anything else in the map, so we should
        // just drop it
        INFO("Dropping integration as buffer (FPGA count: {:d}) arrived too late (minimum in pool "
             "{:d})",
             t.fpga_count, min_time.fpga_count);
        dropped_frame_counter.labels({std::to_string(frame.freq_id), "late"}).inc();
        return;
    }

    else {
        cur_pos++;
        if (vis_time_ind_map.size() == rbs._ntime) {
            // we need to drop the oldest time
            reset_memory(cur_pos);
            vis_time_ind_map.erase(min_time);
        }

        // and replace it with the new most recent time
        vis_time_ind_map[t] = cur_pos;
        write_to_memory(frame, vis_time_ind_map.at(t), freq_ind);
        return;
    }
}

void VisSharedMemWriter::reset_memory(uint32_t time_ind) {

    // resets all memory at time_ind to 0s

    uint8_t* buf_write_pos = buf_addr + (time_ind * rbs.nfreq * rbs.frame_size);
    int64_t* access_record_write_pos = access_record_addr + (time_ind * rbs.nfreq);

    DEBUG("Resetting access_record memory at position time_ind: {}", time_ind);

    // notify that the entire time_ind is invalid, by setting time_ind in the access record to
    // invalid
    wait_for_semaphore();

    std::fill_n(access_record_write_pos, rbs.nfreq, invalid);

    release_semaphore();

    DEBUG("Resetting ring buffer memory at position time_ind: {}", time_ind);
    // set the full time_ind to 0 in the ring buffer
    memset(buf_write_pos, 0, rbs.nfreq * rbs.frame_size);

    DEBUG("Memory reset");
}

void VisSharedMemWriter::write_to_memory(const visFrameView& frame, uint32_t time_ind,
                                         uint32_t freq_ind) {
    // write frame to ring buffer at time_ind and freq_ind

    uint8_t* buf_write_pos = buf_addr + ((time_ind * rbs.nfreq + freq_ind) * rbs.frame_size);
    int64_t* access_record_write_pos = access_record_addr + (time_ind * rbs.nfreq + freq_ind);

    DEBUG("Writing ringbuffer to time_ind {} and freq_ind {}", time_ind, freq_ind);

    // notify that time_ind and freq_ind are being written to, by setting that
    // location to invalid in the access record
    wait_for_semaphore();

    *access_record_write_pos = invalid;

    release_semaphore();

    // first write the metadata, then the data, then the valid byte
    // add valid_size amount of padding
    *buf_write_pos = valid;
    memcpy(buf_write_pos + valid_size, frame.metadata(), rbs.metadata_size);
    memcpy(buf_write_pos + rbs.metadata_size + valid_size, frame.data(), rbs.data_size);

    // Document the fpga sequence counter for that frame in the access record
    uint64_t fpga_seq = frame.metadata()->fpga_seq_start;


    wait_for_semaphore();

    DEBUG("Writing fpga_seq {} to time index {}", fpga_seq, time_ind);
    *access_record_write_pos = fpga_seq;

    // update num_writes
    num_writes++;
    *structured_data_addr = num_writes;

    release_semaphore();
    return;
}

void VisSharedMemWriter::main_thread() {
    DEBUG("Reached main thread");

    frameID frame_id(in_buf);

    // The current position in the ring buffer of the most recent time sample
    // from 0 -> _ntime
    cur_pos = modulo<int>(rbs._ntime);

    // Create the semaphore, and gain first access to it
    sem = sem_open(_name.c_str(), (O_CREAT | O_EXCL), (S_IRUSR | S_IWUSR), 1);

    if (sem == SEM_FAILED) {
        FATAL_ERROR("Failed to create semaphore {}", _name);
        return;
    }

    DEBUG("Semaphore created.");

    // Acquire semaphore until shared memory is created
    wait_for_semaphore();

    // Set up the structure of the ring buffer shared memory
    // Get one frame for reference
    wait_for_full_frame(in_buf, unique_name.c_str(), frame_id);

    auto frame = visFrameView(in_buf, frame_id);

    // Build the frequency index
    std::map<uint32_t, uint32_t> freq_id_map;
    auto& dm = datasetManager::instance();

    const freqState* freq_state = dm.dataset_state<freqState>(frame.dataset_id);

    if (freq_state == nullptr) {
        FATAL_ERROR("Could not find ancestor of dataset {}. Make sure there is a stage upstream in "
                    "the config, with dataset states. freqState is a nullptr",
                    frame.dataset_id);
        return;
    }

    uint ind = 0;
    for (auto& f : freq_state->get_freqs())
        freq_id_map[f.first] = ind++;

    // Calculate the fingerprint
    stream_fingerprint = dm.fingerprint(frame.dataset_id, critical_state_types);

    unique_dataset_ids.insert(frame.dataset_id);

    // Figure out the ring buffer structure
    rbs.nfreq = freq_state->get_freqs().size();

    // Set the alignment (in kB)
    size_t alignment = 4096; // Align on page boundaries

    // Calculate the ring buffer structure

    rbs.data_size = frame.data_size;
    rbs.metadata_size = sizeof(visMetadata);
    // Alligns the frame along page size
    rbs.frame_size = _member_alignment(rbs.data_size + rbs.metadata_size + valid_size, alignment);

    // memory_size should be _ntime * nfreq * file_frame_size (data + metadata)
    buf_addr = assign_memory(_name, (structured_data_size * structured_data_num)
                                        + (rbs._ntime * rbs.nfreq * access_record_size)
                                        + (rbs._ntime * rbs.nfreq * rbs.frame_size));

    // The elements contained in the structured data and access record are each 64 bytes
    structured_data_addr = (uint64_t*)buf_addr;
    access_record_addr = (int64_t*)(structured_data_addr + structured_data_num);
    buf_addr += (structured_data_size * structured_data_num)
                + (rbs._ntime * rbs.nfreq * access_record_size);

    // Record structure of data
    *structured_data_addr = num_writes;
    *((RingBufferStructure*)(structured_data_addr + 1)) = rbs;

    // initially set the address records with -1
    std::fill_n(access_record_addr, rbs._ntime * rbs.nfreq, invalid);

    INFO("Created the shared memory buffer {}", _name);
    release_semaphore();

    // gets called once when kotekan is running
    while (!stop_thread) {


        // Wait for the buffer to be filled with data
        if (wait_for_full_frame(in_buf, unique_name.c_str(), frame_id) == nullptr) {
            break;
        }

        // Get a view of the current frame
        auto frame = visFrameView(in_buf, frame_id);

        if (frame.data_size != rbs.data_size)
            FATAL_ERROR("Size of data changed mid-stream.");

        // Check that the dataset ID hasn't chaned
        if (unique_dataset_ids.count(frame.dataset_id) == 0) {
            // Check whether the fingerprint has changed
            auto frame_fingerprint =
                datasetManager::instance().fingerprint(frame.dataset_id, critical_state_types);

            if (frame_fingerprint == stream_fingerprint) {
                INFO("Got a new dataset ID={}, with known fingerprint={}", frame.dataset_id,
                     stream_fingerprint);
                unique_dataset_ids.insert(frame.dataset_id);
            } else {
                FATAL_ERROR("Got a new dataset ID={}, but FINGERPRINT HAS CHANGED. Known "
                            "fingerprint={}; Received fingerprint={}",
                            frame.dataset_id, stream_fingerprint, frame_fingerprint);
                return;
            }
        }

        if (frame.data_size != rbs.data_size) {
            FATAL_ERROR(
                "The size of the data has changed. Buffer expects: {}. Current frame's size: {}",
                rbs.data_size, frame.data_size);
            return;
        }

        // Get the time and frequency of the frame
        time_ctype t = {std::get<0>(frame.time), ts_to_double(std::get<1>(frame.time))};
        uint32_t freq_ind = freq_id_map.at(frame.freq_id);

        add_sample(frame, t, freq_ind);


        // marks the buffer and moves on
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id++);
    }
}
