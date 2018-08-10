#include "visFileRing.hpp"
#include <unistd.h>

// Register the HDF5 file writers
REGISTER_VIS_FILE("ring", visFileRing);

void visFileRing::create_file(const std::string& name,
                             const std::map<std::string, std::string>& metadata,
                             const std::vector<freq_ctype>& freqs,
                             const std::vector<input_ctype>& inputs,
                             const std::vector<prod_ctype>& prods,
                             size_t num_ev, size_t num_time) {

    visFileRaw::create_file(name, metadata, freqs, inputs, prods, num_ev, num_time);

    file_len = num_time;

}


uint32_t visFileRing::extend_time(time_ctype new_time) {

    size_t ntime = num_time();

    // Extend the file until we reach full length
    if (ntime < file_len) {
        cur_pos = visFileRaw::extend_time(new_time);
    } else {
        // Update current position
        cur_pos = (cur_pos + 1) % file_len;

        // Insert new time at current position in file
        times[cur_pos] = new_time;
        // Erase data in this row
        size_t nb = nfreq * frame_size;
        char zeros[frame_size] = { 0 };
        for (size_t i = 0; i < nfreq; i++) {
            int res = TEMP_FAILURE_RETRY(
                pwrite(fd, zeros, frame_size, cur_pos * nb + i * frame_size)
            );

            if(res < 0) {
                ERROR("Write error attempting to write frame at time %d, freq %d: s.",
                      cur_pos, i, strerror(errno));
            }

        }

        // Write metadata file
        write_metadata();

        // Start to flush out older dataset regions
        uint delta_async = 2;
        if(cur_pos > delta_async) {
            flush_raw_async(cur_pos - delta_async);
        }

        // Flush and clear out any really old parts of the datasets
        uint delta_sync = 4;
        if(cur_pos > delta_sync) {
            flush_raw_sync(cur_pos - delta_sync);
        }
    }

    return cur_pos;

}

void visFileRing::write_metadata() {

    // Finalize the metadata file
    file_metadata["structure"]["ntime"] = num_time();
    file_metadata["index_map"]["time"] = times;
    std::vector<uint8_t> t = json::to_msgpack(file_metadata);
    metadata_file.write((const char *)&t[0], t.size());
    metadata_file.close();
}