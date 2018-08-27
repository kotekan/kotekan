#include <libgen.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <csignal>

#include "fmt.hpp"
#include "visUtil.hpp"
#include "visRawReader.hpp"

REGISTER_KOTEKAN_PROCESS(visRawReader);

visRawReader::visRawReader(Config &config,
                           const string& unique_name,
                           bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&visRawReader::main_thread, this)) {

    filename = config.get_string(unique_name, "infile");
    readahead_blocks = config.get_int(unique_name, "readahead_blocks");
    if (config.get_int(unique_name, "readahead_blocks") < 0) {
        throw std::invalid_argument("visRawReader: config: readahead_blocks" \
                "should be positive (is " + std::to_string(config.get_int(
                                unique_name, "readahead_blocks")) + ").");
    }

    chunked = config.exists(unique_name, "chunk_size");
    if (chunked) {
        chunk_size = config.get_int_array(unique_name, "chunk_size");
        if (chunk_size.size() != 3)
            throw std::invalid_argument("Chunk size needs exactly three " \
                    "elements (has " +
                    std::to_string(chunk_size.size()) + ").");
        chunk_t = chunk_size[2];
        chunk_f = chunk_size[0];
        if (chunk_size[0] < 1 || chunk_size[1] < 1 || chunk_size[2] < 1)
            throw std::invalid_argument("visRawReader: config: Chunk size " \
                    "needs to be greater or equal to (1,1,1) (is ("
                    + std::to_string(chunk_size[0]) + ","
                    + std::to_string(chunk_size[1]) + ","
                    + std::to_string(chunk_size[2]) + ")).");
    }

    // Get the list of buffers that this process shoud connect to
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    // Read the metadata
    std::string md_filename = (filename + ".meta");
    INFO("Reading metadata file: %s", md_filename.c_str());
    struct stat st;
    if (stat(md_filename.c_str(), &st) == -1)
        throw std::ios_base::failure("visRawReader: Error reading from " \
                "metadata file: " + md_filename);
    size_t filesize = st.st_size;
    std::vector<uint8_t> packed_json(filesize);

    std::ifstream metadata_file(md_filename, std::ios::binary);
    if (metadata_file) // only read if no error
        metadata_file.read((char *)&packed_json[0], filesize);
    if (!metadata_file) // check if open and read successful
        throw std::ios_base::failure("visRawReader: Error reading from " \
                "metadata file: " + md_filename);
    json _t = json::from_msgpack(packed_json);
    metadata_file.close();

    // Extract the attributes and index maps
    _metadata = _t["attributes"];
    _times = _t["index_map"]["time"].get<std::vector<time_ctype>>();
    _freqs = _t["index_map"]["freq"].get<std::vector<freq_ctype>>();
    _inputs = _t["index_map"]["input"].get<std::vector<input_ctype>>();
    _prods = _t["index_map"]["prod"].get<std::vector<prod_ctype>>();
    _ev = _t["index_map"]["ev"].get<std::vector<uint32_t>>();
    if (_t["index_map"].find("stack") != _t["index_map"].end()) {
        DEBUG("reading stack...");
        _stack = _t["index_map"]["stack"].get<std::vector<stack_ctype>>();
        DEBUG("done.");
    }

    // Extract the structure
    file_frame_size = _t["structure"]["frame_size"].get<size_t>();
    metadata_size = _t["structure"]["metadata_size"].get<size_t>();
    data_size = _t["structure"]["data_size"].get<size_t>();
    nfreq = _t["structure"]["nfreq"].get<size_t>();
    ntime = _t["structure"]["ntime"].get<size_t>();

    if (chunked) {
        // Special case if dimensions less than chunk size
        chunk_f = std::min(chunk_f, nfreq);
        chunk_t = std::min(chunk_t, ntime);

        // Number of elements in a chunked row
        row_size = (chunk_f * chunk_t) * (nfreq / chunk_f)
                 + (nfreq % chunk_f) * chunk_t;
    }

    // Check metadata is the correct size
    if(sizeof(visMetadata) != metadata_size) {
        std::string msg = fmt::format("Metadata in file {} is larger " \
                "({} bytes) than visMetadata ({} bytes).",
            filename, metadata_size, sizeof(visMetadata)
        );
        throw std::runtime_error(msg);
    }

    // Check that buffer is large enough
    if((unsigned int)(out_buf->frame_size) < data_size
            || out_buf->frame_size < 0) {
        std::string msg = fmt::format(
            "Data in file {} is larger ({} bytes) than buffer size ({} bytes).",
            filename, data_size, out_buf->frame_size
        );
        throw std::runtime_error(msg);
    }

    // Open up the data file and mmap it
    INFO("Opening data file: %s", (filename + ".data").c_str());
    if((fd = open((filename + ".data").c_str(), O_RDONLY)) == -1) {
        throw std::runtime_error(fmt::format("Failed to open file {}: {}.",
                                       filename + ".data", strerror(errno)));
    }
    mapped_file = (uint8_t *)mmap(NULL, ntime * nfreq * file_frame_size,
                                  PROT_READ, MAP_SHARED, fd, 0);
    if (mapped_file == MAP_FAILED)
        throw std::runtime_error(fmt::format(
                    "Failed to map file {} to memory: {}.", filename + ".data",
                    strerror(errno)));
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

    if (madvise(mapped_file + offset, file_frame_size, MADV_WILLNEED) == -1)
        DEBUG("madvise failed: %s", strerror(errno));
}

int visRawReader::position_map(int ind) {
    if(chunked) {
        // chunked row index
        int ri = ind / row_size;
        // Special case at edges of time*freq array
        int t_width = std::min(ntime - ri * chunk_t, chunk_t);
        // chunked column index
        int ci = (ind % row_size) / (t_width * chunk_f);
        // edges case
        int f_width = std::min(nfreq - ci * chunk_f, chunk_f);
        // time and frequency indices
        // time is fastest varying
        int fi = ci * chunk_f + ((ind % row_size) %
                 (t_width * f_width)) / t_width;
        int ti = ri * chunk_t + ((ind % row_size) %
                 (t_width * f_width)) % t_width;

        return ti * nfreq + fi;
    } else {
        return ind;
    }
}

void visRawReader::main_thread() {

    size_t frame_id = 0;
    uint8_t * frame;

    size_t ind = 0, read_ind = 0, file_ind;

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

        // Allocate the metadata space
        allocate_new_metadata_object(out_buf, frame_id);

        // Check first byte indicating empty frame
        if (*(mapped_file + file_ind * file_frame_size) != 0) {
            // Copy the metadata from the file
            std::memcpy(out_buf->metadata[frame_id]->metadata, mapped_file
                    + file_ind * file_frame_size + 1, metadata_size);

            // Copy the data from the file
            std::memcpy(frame, mapped_file + file_ind * file_frame_size
                    + metadata_size + 1, data_size);
        } else {
            // Set metadata if file contained an empty frame
			((visMetadata *)(out_buf->metadata[frame_id]->metadata))->num_prod
                = _prods.size();
			((visMetadata *)(out_buf->metadata[frame_id]->metadata))->num_ev
                = _ev.size();
			((visMetadata *)(out_buf->metadata[frame_id]->metadata))->
                num_elements = _inputs.size();
            // Fill data with zeros
            auto frame = visFrameView(out_buf, frame_id);
            std::memset(frame.vis.data(), 0, sizeof(cfloat) * frame.num_prod);
            std::memset(frame.weight.data(), 0, sizeof(float) * frame.num_prod);
            std::memset(frame.eval.data(), 0, sizeof(float) * frame.num_ev);
            std::memset(frame.evec.data(), 0, sizeof(cfloat) * frame.num_ev
                    * frame.num_elements);
            std::memset(frame.gain.data(), 0, sizeof(cfloat) * frame.num_elements);
            std::memset(frame.flags.data(), 0, sizeof(float) * frame.num_elements);
			frame.freq_id = 0;
			frame.dataset_id = 0;
			frame.erms = 0;
			DEBUG("visRawReader: Reading empty frame: %d", frame_id);
        }

        // Try and clear out the cached data as we don't need it again
        if (madvise(mapped_file + file_ind * file_frame_size, file_frame_size,
                    MADV_DONTNEED) == -1)
            DEBUG("madvise failed: %s", strerror(errno));

        // Release the frame and advance all the counters
        mark_frame_full(out_buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % out_buf->num_frames;
        read_ind++;
        ind++;
    }
}
