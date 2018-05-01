#include <libgen.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <csignal>
#include "fmt.hpp"

#include "visRawReader.hpp"

REGISTER_KOTEKAN_PROCESS(visRawReader);

visRawReader::visRawReader(Config &config,
                           const string& unique_name,
                           bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&visRawReader::main_thread, this)) {

    filename = config.get_string(unique_name, "filename");
    
    chunk_size = config.get_int_array(unique_name, "chunk_size");

    // Get the list of buffers that this process shoud connect to
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    // Read the metadata
    std::string md_filename = (filename + ".meta");
    INFO("Reading metadata file: %s", md_filename.c_str());
    struct stat st;
    stat(md_filename.c_str(), &st);
    size_t filesize = st.st_size;
    std::vector<uint8_t> packed_json(filesize);

    std::ifstream metadata_file(md_filename, std::ios::binary);
    metadata_file.read((char *)&packed_json[0], filesize);
    std::cout << packed_json.size() << std::endl;
    json _t = json::from_msgpack(packed_json);
    metadata_file.close();

    // Extract the attributes and index maps
    _metadata = _t["attributes"];
    _times = _t["index_map"]["time"].get<std::vector<time_ctype>>();
    _freqs = _t["index_map"]["freq"].get<std::vector<freq_ctype>>();
    _inputs = _t["index_map"]["input"].get<std::vector<input_ctype>>();
    _prods = _t["index_map"]["prod"].get<std::vector<prod_ctype>>();
    _ev = _t["index_map"]["ev"].get<std::vector<uint32_t>>();

    // Extract the structure
    file_frame_size = _t["structure"]["frame_size"].get<size_t>();
    metadata_size = _t["structure"]["metadata_size"].get<size_t>();
    data_size = _t["structure"]["data_size"].get<size_t>();
    nfreq = _t["structure"]["nfreq"].get<size_t>();
    ntime = _t["structure"]["ntime"].get<size_t>();

    // Check metadata is the correct size 
    if(sizeof(visMetadata) != metadata_size) {
        std::string msg = fmt::format(
            "Metadata in file {} is larger ({} bytes) than visMetadata ({} bytes).",
            filename, metadata_size, sizeof(visMetadata)
        );
        throw std::runtime_error(msg);
    }

    // Check that buffer is large enough
    if(out_buf->frame_size < data_size) {
        std::string msg = fmt::format(
            "Data in file {} is larger ({} bytes) than buffer size ({} bytes).",
            filename, data_size, out_buf->frame_size
        );
        throw std::runtime_error(msg);
    }

    // Open up the data file and mmap it
    INFO("Opening data file: %s", (filename + ".data").c_str());
    if((fd = open((filename + ".data").c_str(), O_RDONLY)) == -1) {
        std::runtime_error(fmt::format("Failed to open file {}: {}.",
                                       filename + ".data", strerror(errno)));
    }
    mapped_file = (uint8_t *)mmap(NULL, ntime * nfreq * file_frame_size,
                                  PROT_READ, MAP_SHARED, fd, 0);
}

visRawReader::~visRawReader() {
    if(munmap(mapped_file, ntime * nfreq * file_frame_size) == -1) {
        std::runtime_error(fmt::format("Failed to unmap file {}: {}.",
                                       filename + ".data", strerror(errno)));
    }

    close(fd);
}

void visRawReader::apply_config(uint64_t fpga_seq) {

}

void visRawReader::read_ahead(int ind) {

    off_t offset = position_map(ind) * file_frame_size;

    madvise(mapped_file + offset, file_frame_size, MADV_WILLNEED);
}

int visRawReader::position_map(int ind) {
    if(time_ordered) {
        int ti = ind % ntime;
        int fi = ind / ntime;
        return ti * nfreq + fi;
    } else {
        return ind;
    }
}

void visRawReader::main_thread() {

    unsigned int frame_id = 0;
    uint8_t * frame;

    int ind = 0, read_ind = 0, file_ind;
    int readahead_blocks = 4;
    size_t nframe = nfreq * ntime;

    // Initial readahead for frames
    for(read_ind = 0; read_ind < readahead_blocks; read_ind++) {
        read_ahead(read_ind);
    }

    while (!stop_thread && ind < nframe) {

        // Wait for the buffer to be filled with data
        if((frame = wait_for_empty_frame(out_buf, unique_name.c_str(),
                                         frame_id)) == nullptr) {
            break;
        }

        // Issue the read ahead request
        if(read_ind < (ntime * nfreq)) {
            read_ahead(read_ind);
        }

        // Get the index into the file
        file_ind = position_map(ind);

        // Allocate the metadata space and copy it from the file
        allocate_new_metadata_object(out_buf, frame_id);
        std::memcpy(out_buf->metadata[frame_id]->metadata,
                    mapped_file + file_ind * file_frame_size + 1, metadata_size);

        // Copy the data from the file
        std::memcpy(frame,
                    mapped_file + file_ind * file_frame_size + metadata_size + 1, data_size);

        // Try and clear out the cached data as we don't need it again
        madvise(mapped_file + file_ind * file_frame_size, file_frame_size, MADV_DONTNEED);

        // Release the frame and advance all the counters
        mark_frame_full(out_buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % out_buf->num_frames;
        read_ind++;
        ind++;
    }

    // Once we've read the file, we should exit kotekan
    std::raise(SIGINT);
}