/*****************************************
@file
@brief Base classes for visibility output files
- visFile
- visFileBundle
*****************************************/
#ifndef VIS_FILE_HPP
#define VIS_FILE_HPP

#include "FrameView.hpp"      // for FrameView
#include "dataset.hpp"        // for dset_id_t
#include "factory.hpp"        // for CREATE_FACTORY, FACTORY, Factory, REGISTER_NAMED_TYPE_WITH...
#include "kotekanLogging.hpp" // for logLevel, kotekanLogging, DEBUG
#include "visUtil.hpp"        // for time_ctype, operator<

#include <cstdint>    // for uint32_t
#include <functional> // for function
#include <map>        // for map, map<>::mapped_type
#include <memory>     // for allocator, shared_ptr, __shared_ptr_access
#include <stddef.h>   // for size_t
#include <string>     // for string, operator+, char_traits
#include <tuple>      // for tie, tuple
#include <utility>    // for pair, forward

/** @brief A base class for files holding correlator data.
 *
 * The class specifies the interface that all correlator file types must follow.
 *
 * File types are expected to create and manage lock files, which should have
 * the format `.<datafilename>.lock`.
 *
 * @author Richard Shaw
 **/
class visFile : public kotekan::kotekanLogging {

public:
    virtual ~visFile() = default;

    /** @brief Create the file.
     *
     * This is the entry point to an abstract visFile factory.
     *
     *  @param  type  Type of the file to write.
     *  @param  name  Name of the file to write
     *  @param  args  Arguments forwarded to create_file.
     **/
    template<typename... CreateArgs>
    static std::shared_ptr<visFile> create(const std::string& type, const std::string& name,
                                           CreateArgs... args);

    /**
     * @brief Extend the file to a new time sample.
     *
     * @param new_time The new time to add.
     * @return The index of the added time in the file.
     **/
    virtual uint32_t extend_time(time_ctype new_time) = 0;

    /**
     * @brief Remove the time sample from the active set being written to.
     *
     * After this is called there should be no more requests to write into
     * this timesample. Implement this method to perform any final cleanup
     * on the file for this sample (e.g. flush, evict pages).
     *
     * @param time_ind Sample to cleanup.
     **/
    virtual void deactivate_time(uint32_t time_ind) {
        (void)time_ind;
        DEBUG("visFile::deactivate_time: called but not implemented.");
    };

    /**
     * @brief Write a sample of data into the file at the given index.
     *
     * @param time_ind Time index to write into.
     * @param freq_ind Frequency index to write into.
     * @param frame Frame to write out.
     **/
    virtual void write_sample(uint32_t time_ind, uint32_t freq_ind, const FrameView& frame) = 0;

    /**
     * @brief Return the current number of current time samples.
     *
     * @return The current number of time samples.
     **/
    virtual size_t num_time() = 0;

protected:
    // Save the size for when we are outside of HDF5 space
    size_t nfreq, nprod, ninput, nev, ntime = 0;
};

CREATE_FACTORY(visFile, const std::string& /*name*/, const kotekan::logLevel /*log_level*/,
               const std::map<std::string, std::string>& /*metadata*/, dset_id_t /*dataset*/,
               size_t /*max_time*/);


#define REGISTER_VIS_FILE(key, T) REGISTER_NAMED_TYPE_WITH_FACTORY(visFile, T, key)


// Abstract factory VisFile creator.
// Forwards on an argument pack. Actual arguments defined on visFile::create_file
template<typename... CreateArgs>
inline std::shared_ptr<visFile> visFile::create(const std::string& type, const std::string& name,
                                                CreateArgs... args) {

    return FACTORY(visFile)::create_shared(type, name, std::forward<CreateArgs>(args)...);
}

/**
 * @brief Manage the set of correlator files being written.
 *
 * This abstraction above visFile allows us to hold open multiple files for
 * writing at the same time. This is needed because we roll over to a new file
 * after a certain number of samples, but in general we may still be waiting on
 * samples to go into the existing file.
 *
 * @author Richard Shaw
 **/
class visFileBundle : public kotekan::kotekanLogging {

public:
    /**
     * Initialise the file bundle
     * @param type Type of the files to write.
     * @param root_path Directory to write into.
     * @param instrument_name Instrument name (e.g. chime)
     * @param acq_type Acquisition type (e.g. corr)
     * @param metadata  Textual metadata to write into the files.
     * @param freq_chunk ID of the frequency chunk being written
     * @param rollover Maximum time length of file.
     * @param window_size Number of "active" timesamples to keep.
     * @param log_level kotekan log level for any logging generated by the visFileBundle instance
     * @param args Arguments passed through to `visFile::visFile`.
     *
     * @warning The directory will not be created if it doesn't exist.
     **/
    template<typename... InitArgs>
    visFileBundle(const std::string& type, const std::string& root_path,
                  const std::string& instrument_name, const std::string& acq_type,
                  const std::map<std::string, std::string>& metadata, int freq_chunk,
                  size_t rollover, size_t window_size, const kotekan::logLevel log_level,
                  InitArgs... args);

    /**
     * Write a new time sample into this set of files
     * @param  new_time  Time of sample
     * @param  args      Arguments passed through to `visFile::write_sample`
     * @return           True if an error occurred while writing
     **/
    template<typename... WriteArgs>
    bool add_sample(time_ctype new_time, WriteArgs&&... args);

    virtual ~visFileBundle();

    /**
     * @brief Get the time of the last update.
     *
     * @return  The time of the last update.
     **/
    time_ctype last_update() const;

protected:
    // Add a file if we need to
    virtual void add_file(time_ctype first_time);

    // Find/create the slot for data at this time to go into
    bool resolve_sample(time_ctype new_time);

    std::map<time_ctype, std::pair<std::shared_ptr<visFile>, uint32_t>> vis_file_map;
    // Thin function to actually create the file
    std::function<std::shared_ptr<visFile>(std::string, std::string, std::string)> mk_file;

    const std::string root_path;

    const std::string instrument_name;
    const std::string acq_type;
    const int freq_chunk;

    size_t rollover;
    size_t window_size;

    std::string acq_name;
    double acq_start_time;

    // Flag to force moving to a new file
    bool change_file = false;
};

// NOTE: in this we need to pass the variadic arguments by value and not attempt
// to forward them. This is because in C++17 we can't capture a variadic
// parameter pack into the lambda perfectly.
template<typename... InitArgs>
inline visFileBundle::visFileBundle(const std::string& type, const std::string& root_path,
                                    const std::string& instrument_name, const std::string& acq_type,
                                    const std::map<std::string, std::string>& metadata,
                                    int freq_chunk, size_t rollover, size_t window_size,
                                    const kotekan::logLevel log_level, InitArgs... args) :
    root_path(root_path),
    instrument_name(instrument_name),
    acq_type(acq_type),
    freq_chunk(freq_chunk),
    rollover(rollover),
    window_size(window_size) {

    set_log_level(log_level);

    // Make a lambda function that creates a file. This is a little convoluted,
    // but is the easiest way of passing on the variadic arguments to the
    // constructor into the file creation.
    mk_file = [type, metadata, log_level, args...](std::string file_name, std::string acq_name,
                                                   std::string root_path) {
        // Add the acq name to the metadata
        auto metadata_acq = metadata;
        metadata_acq["acquisition_name"] = acq_name;

        std::string abspath = root_path + '/' + acq_name + '/' + file_name;
        return visFile::create(type, abspath, log_level, metadata_acq, args...);
    };
}


template<typename... WriteArgs>
inline bool visFileBundle::add_sample(time_ctype new_time, WriteArgs&&... args) {

    if (resolve_sample(new_time)) {
        std::shared_ptr<visFile> file;
        uint32_t ind;
        // We can now safely add the sample into the file
        std::tie(file, ind) = vis_file_map[new_time];
        file->write_sample(ind, std::forward<WriteArgs>(args)...);

        return false;
    } else {
        return true;
    }
}

/**
 * @brief Create a lock file for the given file.
 * @param filename Name of file to lock.
 * @return The name of the lock file.
 **/
std::string create_lockfile(std::string filename);


// Implementation of TEMP_FAILURE_RETRY for file writing which is missing on MacOS
#if defined(__APPLE__)
// Taken from
// https://android.googlesource.com/platform/system/core/+/master/base/include/android-base/macros.h
#ifndef TEMP_FAILURE_RETRY
#define TEMP_FAILURE_RETRY(exp)                                                                    \
    ({                                                                                             \
        decltype(exp) _rc;                                                                         \
        do {                                                                                       \
            _rc = (exp);                                                                           \
        } while (_rc == -1 && errno == EINTR);                                                     \
        _rc;                                                                                       \
    })
#endif
#endif


#endif
