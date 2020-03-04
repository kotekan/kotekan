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
        CHECK(sem_wait(sem));

        // memory_size should be ntime * nfreq * file_frame_size (data + metadata)
        record_addr = assign_memory<uint64_t*>(fname_access_record, ntime * 64, record_addr);
        buf_addr = assign_memory<uint8_t*>(fname_buf, ntime * 8, buf_addr);
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

    // Set up the structure of the access record shared memory
    struct timeval timestamp;
    uint64_t time_us;
    size_t access_record_size = sizeof(time_us);
    const uint64_t in_progress = -1;

    record_addr = assign_memory<uint64_t*>(fname_access_record, ntime * access_record_size, record_addr);

    // Set up the structure of the ring buffer shared memory
    // Get one frame for reference
    wait_for_full_frame(in_buf, unique_name.c_str(), frame_id);

    // Get a view of the current frame
    // TODO: this frame is not getting written anywhere yet.
    auto frame = visFrameView(in_buf, frame_id);

    // dset_id_t ds_id -> frame.dataset_id
    // Frequency IDs that we are expecting
    std::map<uint64_t, uint64_t> freq_id_map;
    auto& dm = datasetManager::instance();

    // Build the frequency index
    auto fstate_fut = std::async(&datasetManager::dataset_state<freqState>, &dm, frame.dataset_id);
    const freqState* fstate = fstate_fut.get();
    uint ind = 0;
    for (auto& f : fstate->get_freqs())
        freq_id_map[f.first] = ind++;


    // Figure out the structure of the ring buffer
    const uint8_t ONE = 1;
    size_t nfreq = fstate->get_freqs().size();
    size_t metadata_size = sizeof(visMetadata);
    size_t data_size = sizeof(uint8_t);
    size_t frame_size = data_size + metadata_size + sizeof(ONE);

    // memory_size should be ntime * nfreq * file_frame_size (data + metadata)
    buf_addr = assign_memory<uint8_t*>(fname_buf, ntime * nfreq * frame_size, buf_addr);
    INFO("Created the shared memory segments\n");
    CHECK(sem_post(sem));

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

        // Get the time and frequency of the frame
        auto ftime = frame.time;
        time_ctype t = {std::get<0>(ftime), ts_to_double(std::get<1>(ftime))};
        uint64_t freq_ind = freq_id_map.at(frame.freq_id);

        // acq.file_bundle->add_sample(t, freq_ind, frame); then does the writing to disk/buffer


        // class called frame_id in visutil -> does all the cyclic

        // Check whether the next value written is within the buffer bounds
        if (i * access_record_size + access_record_size > ntime * access_record_size) {
            INFO("Setting back to 0!");
            i = 0;
        }


        CHECK(sem_wait(sem));
        memcpy(record_addr + i, &in_progress, access_record_size);
        CHECK(sem_post(sem));

        memcpy(buf_addr + i * frame_size, &ONE, sizeof(uint8_t));
        memcpy(buf_addr + i * frame_size + 1, frame.metadata(), metadata_size);
        memcpy(buf_addr + i * frame_size + metadata_size + 1, frame.data(), data_size);

        gettimeofday(&timestamp, nullptr);
        time_us = timestamp.tv_sec * 1000000 + timestamp.tv_usec;
        INFO("Data written to location at {} us", time_us);

        CHECK(sem_wait(sem));
        memcpy(record_addr + i, &time_us, access_record_size);
        CHECK(sem_post(sem));

        ++i;

        // marks the buffer and moves on
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id++);

    }
}
