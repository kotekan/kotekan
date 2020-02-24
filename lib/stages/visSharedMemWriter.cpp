#include "visSharedMemWriter.hpp"

#include "errors.h"

#include "fmt.hpp"


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;


REGISTER_KOTEKAN_STAGE(visSharedMemWriter);

visSharedMemWriter::visSharedMemWriter(Config& config, const std::string& unique_name, bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&visSharedMemWriter::main_thread, this)) {

        // Fetch any simple configuration
        root_path = config.get_default<std::string>(unique_name, "root_path", "/dev/shm/");
        sem_name = config.get_default<std::string>(unique_name, "sem_name", "kotekan");
        fname_met = config.get_default<std::string>(unique_name, "fname_met", "calBufferMetadata");
        fname_buf = config.get_default<std::string>(unique_name, "fname_buf", "calBuffer");
        nsamples = config.get_default<size_t>(unique_name, "nsamples", 4096);

        // Setup the input vector
        in_buf = get_buffer("in_buf");
        register_consumer(in_buf, unique_name.c_str());

        // Check if any of the old buffer files exist
        DEBUG("Checking for and removing old buffer files...");
        check_remove(root_path + "sem." + sem_name);
        check_remove(root_path + fname_met);
        check_remove(root_path + fname_buf);

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

        // memory_size should be ntime * nfreq * file_frame_size (data + metadata)
        met_addr = assign_memory<uint64_t*>(fname_met, nsamples * 64, met_addr);
        buf_addr = assign_memory<uint8_t*>(fname_buf, nsamples * 8, buf_addr);
        INFO("Created the shared memory segments\n");

}

// make a deconstructor in which to deconstruct semaphores and shared memory
visSharedMemWriter::~visSharedMemWriter() {
    // make sure to unlink the semaphore and unmap the mappings
}

// takes the name of a shared memory address opens it, and maps the memory to the provided address pointer
template <typename T>
T visSharedMemWriter::assign_memory(std::string shm_name, int shm_size, T addr) {
        int fd = shm_open(shm_name.c_str(), (O_CREAT | O_RDWR), (S_IRUSR | S_IWUSR));

        if (fd == -1) {
            throw std::runtime_error(
                fmt::format(fmt("Cannot open shared memory named {:s}: {:s}"), shm_name, strerror(errno)));
        }

        // Resize object to hold buffer
        CHECK(ftruncate(fd, shm_size));
        INFO("Resized to %ld bytes\n", (long) shm_size);

        addr = (T) mmap(nullptr, shm_size, (PROT_READ | PROT_WRITE), MAP_SHARED, fd, 0);
        if (addr == MAP_FAILED) {
            throw std::runtime_error(
                fmt::format(fmt("Failed to map shm {:s} to memory: {:s}."), shm_name, strerror(errno)));
        }

        // fd is no longer needed
        CHECK(close(fd));

        return addr;
}


void visSharedMemWriter::main_thread() {
    INFO("Reached main thread");

    frameID frame_id(in_buf);

    size_t i = 0;
    struct timeval timestamp;
    uint64_t time_us;
    const uint64_t in_progress = -1;

    // gets called once when kotekan is running
    while (!stop_thread) {


        // Wait for the buffer to be filled with data
        if (wait_for_full_frame(in_buf, unique_name.c_str(), frame_id) == nullptr) {
            break;
        }

        // Get a view of the current frame
        auto frame = visFrameView(in_buf, frame_id);

        // dset_id_t ds_id -> frame.dataset_id
        // Frequency IDs that we are expecting
        /*std::map<uint64_t, uint64_t> freq_id_map;
        auto& dm = datasetManager::instance();
        auto fstate_fut = std::async(&datasetManager::dataset_state<freqState>, &dm, frame.dataset_id);
        const freqState* fstate = fstate_fut.get();

        uint ind = 0;
        for (auto& f : fstate->get_freqs())
            freq_id_map[f.first] = ind++;

        // Get the time and frequency of the frame
        auto ftime = frame.time;
        time_ctype t = {std::get<0>(ftime), ts_to_double(std::get<1>(ftime))};
        uint64_t freq_ind = freq_id_map.at(frame.freq_id); */

        // acq.file_bundle->add_sample(t, freq_ind, frame); then does the writing to disk/buffer


        // class called frame_id in visutil -> does all the cyclic

        // Check whether the next value written is within the buffer bounds
        if (i * sizeof(uint64_t) + sizeof(uint64_t) > nsamples * 64) {
            INFO("Setting back to 0!");
            i = 0;
        }


        CHECK(sem_wait(sem));
        memcpy(met_addr + i, &in_progress, sizeof(uint64_t));
        CHECK(sem_post(sem));

        memcpy(buf_addr + i, frame.data(), sizeof(uint8_t));

        gettimeofday(&timestamp, nullptr);
        time_us = timestamp.tv_sec * 1000000 + timestamp.tv_usec;
        INFO("Data written to location at {} us", time_us);

        CHECK(sem_wait(sem));
        memcpy(met_addr + i, &time_us, sizeof(uint64_t));
        CHECK(sem_post(sem));

        i += 1;

        // marks the buffer and moves on
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id++);

    }
}
