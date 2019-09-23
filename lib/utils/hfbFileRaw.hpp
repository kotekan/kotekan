/*****************************************
@file
@brief Raw dump based hyperfine beam output files
- hfbFileRaw
*****************************************/
#ifndef HFB_FILE_RAW_HPP
#define HFB_FILE_RAW_HPP

#include "Config.hpp"
#include "datasetManager.hpp"
#include "prometheusMetrics.hpp"
#include "hfbBuffer.hpp"
#include "visFile.hpp"
#include "visUtil.hpp"

#include <cstdint>
#include <fcntl.h>
#include <fstream>
#include <map>
#include <stddef.h>
#include <string>
#include <sys/types.h>
#include <vector>

using json = nlohmann::json;

/** @brief A hyperfine beam file in raw format.
 *
 * The class creates and manages writes to a hyperfine beam output
 * file. It also manages the lock file.
 *
 * This output has the following structure.
 *
 * The metadata (including the file structure) is written into a `.meta` file
 * which is JSON serialised as `msgpack`. The interpretation should be pretty
 * straightforward of the index_map and attributes sections as these map
 * directly to the HDF5 equivalents. The `structure` section describes how to
 * interpret the final, giving the sizes in bytes of the metadata, data and the
 * overall frame size (which is aligned to a set boundary).
 *
 * The `.data` file contains the raw data output. This is packed as:
 *
 *  - 1st byte is set to `1` if data is present (or is implicitly zero).
 *  - hfbMetadata struct dump
 *  - hfbBuffer dump
 *
 * @author James Willis
 **/
class hfbFileRaw : public visFile {

public:
    ~hfbFileRaw();

    /**
     * @brief Extend the file to a new time sample.
     *
     * @param new_time The new time to add.
     * @return The index of the added time in the file.
     **/
    uint32_t extend_time(time_ctype new_time) override;

    /**
     * @brief Write a sample of data into the file at the given index.
     *
     * @param time_ind Time index to write into.
     * @param freq_ind Frequency index to write into.
     * @param frame Frame to write out.
     **/
    void write_sample(uint32_t time_ind, uint32_t freq_ind, const hfbFrameView& frame) override;

    /**
     * @brief Return the current number of current time samples.
     *
     * @return The current number of time samples.
     **/
    size_t num_time() override;

    /**
     * @brief Remove the time sample from the active set being written to.
     *
     * This explicit flushes the requested time sample and evicts it from the
     * page cache.
     *
     * @param time_ind Sample to cleanup.
     **/
    void deactivate_time(uint32_t time_ind) override;

protected:
    // Implement the create file method
    void create_file(const std::string& name, const kotekan::logLevel log_level,
                     const std::map<std::string, std::string>& metadata, dset_id_t dataset,
                     size_t max_time) override;

    /// Flags used for opening new files
    int oflags = O_CREAT | O_EXCL | O_WRONLY;

    /**
     * @brief  Helper routine for writing data into the file
     *
     * @param offset Offset of the data to write.
     * @param nb     The size of the data in bytes.
     * @param data   The data to write out.
     **/
    bool write_raw(off_t offset, size_t nb, const void* data);

    /**
     * @brief Start an async flush to disk
     *
     * @param ind       The index into the file dataset in time.
     **/
    void flush_raw_async(int ind);

    /**
     * @brief Start a synchronised flush to disk and evict any clean pages.
     *
     * @param ind       The index into the file dataset in time.
     **/
    void flush_raw_sync(int ind);

    // The metadata we will write into the file
    json file_metadata;

    // Save the sizes
    size_t nfreq, data_size, metadata_size, frame_size;

    // File descriptors and related
    int fd;
    std::ofstream metadata_file;
    std::string lock_filename;

    // Keep a list of the times we've seen
    std::vector<time_ctype> times;

    // Align to blocks of this size kB
    size_t alignment;

    // File name (used for debugging)
    std::string _name;

};

#endif
