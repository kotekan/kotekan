#include "visFileRing.hpp"

#include "visFile.hpp" // for REGISTER_VIS_FILE, _factory_aliasvisFile
#include "visUtil.hpp" // for time_ctype

#include "json.hpp" // for basic_json<>::value_type, json

#include <errno.h>     // for errno
#include <fcntl.h>     // for O_CREAT, O_WRONLY
#include <ostream>     // for ofstream, basic_ostream::flush, basic_ostream::seekp, basi...
#include <string.h>    // for size_t, strerror
#include <sys/types.h> // for uint
#include <unistd.h>    // for pwrite, TEMP_FAILURE_RETRY
#include <vector>      // for vector

// Register the HDF5 file writers
REGISTER_VIS_FILE("ring", visFileRing);


visFileRing::visFileRing(const std::string& name, const kotekan::logLevel log_level,
                         const std::map<std::string, std::string>& metadata, dset_id_t dataset,
                         size_t max_time) :
    visFileRaw(name, log_level, metadata, dataset, max_time, O_CREAT | O_WRONLY),
    file_len(max_time){};


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
        std::vector<char> zeros(frame_size, 0);
        for (size_t i = 0; i < nfreq; i++) {
            int res = TEMP_FAILURE_RETRY(
                pwrite(fd, zeros.data(), frame_size, cur_pos * nb + i * frame_size));

            if (res < 0) {
                ERROR("Write error attempting to write frame at time {:d}, freq {:d}: {:s}.",
                      cur_pos, i, strerror(errno));
            }
        }

        // TODO: Are these appropriate in this context?
        // Start to flush out older dataset regions
        uint delta_async = 2;
        if (cur_pos > delta_async) {
            flush_raw_async(cur_pos - delta_async);
        }

        // Flush and clear out any really old parts of the datasets
        uint delta_sync = 4;
        if (cur_pos > delta_sync) {
            flush_raw_sync(cur_pos - delta_sync);
        }
    }

    // Write metadata file
    write_metadata();

    return cur_pos;
}

void visFileRing::write_metadata() {

    // Update the metadata file
    file_metadata["structure"]["ntime"] = num_time();
    file_metadata["index_map"]["time"] = times;
    std::vector<uint8_t> t = nlohmann::json::to_msgpack(file_metadata);
    metadata_file.write((const char*)&t[0], t.size());
    metadata_file.flush();
    // Reset file position for next write
    metadata_file.seekp(0);
}
