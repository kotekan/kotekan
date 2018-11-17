/*****************************************
@file
@brief Base classes for visibility output files
- visFile
- visFileBundle
*****************************************/
#ifndef VIS_FILE_HPP
#define VIS_FILE_HPP

#include <iostream>
#include <cstdint>
#include <map>
#include <memory>

#include "visBuffer.hpp"
#include "datasetManager.hpp"
#include "visUtil.hpp"
#include "errors.h"
#include "fmt.hpp"

/** @brief A base class for files holding correlator data.
 *
 * The class specifies the interface that all correlator file types must follow.
 *
 * File types are expected to create and manage lock files, which should have
 * the format `.<datafilename>.lock`.
 *
 * @author Richard Shaw
 **/
class visFile {

public:

virtual ~visFile() {};

    /** @brief Create the file.
     *
     * This is the entry point to an abstract visFile factory.
     *
     *  @param type Type of the file to write.
     *  @param name Name of the file to write
     *  @param args Arguments forwarded to create_file.
     **/
    template<typename... CreateArgs>
    static std::shared_ptr<visFile> create(
        const std::string& type,
        const std::string& name,
        CreateArgs&&... args
    );

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
    void deactivate_time(uint32_t time_ind) {};

    /**
     * @brief Write a sample of data into the file at the given index.
     *
     * @param time_ind Time index to write into.
     * @param freq_ind Frequency index to write into.
     * @param frame Frame to write out.
     **/
    virtual void write_sample(uint32_t time_ind, uint32_t freq_ind,
                              const visFrameView& frame) = 0;

    /**
     * @brief Return the current number of current time samples.
     *
     * @return The current number of time samples.
     **/
    virtual size_t num_time() = 0;

    /**
     * @brief Register a compatible visFile type.
     * @param type Name of type.
     **/
    template <typename T>
    static inline int register_file_type(const std::string type);


protected:

    /** @brief Create the file.
     *
     * This variant uses the datasetManager to look up properties of the
     * dataset that we are dealing with.
     *
     *  @param name     Name of the file to write
     *  @param metadata Textual metadata to write into the file.
     *  @param dataset  ID of dataset we are writing.
     *  @param num_ev   Number of eigenvectors to write (0 turns off the
     *                  datasets entirely).
     *  @param max_time Maximum number of times to write into the file.
     **/
    // TODO: decide if the num_ev can be eliminated.
    virtual void create_file(
        const std::string& name,
        const std::map<std::string, std::string>& metadata,
        dset_id_t dataset, size_t num_ev, size_t max_time) = 0;

    // Private constructor to discourage creation of subclasses outside of the
    // create routine
    visFile() {};

    // Save the size for when we are outside of HDF5 space
    size_t nfreq, nprod, ninput, nev, ntime = 0;

private:

    static std::map<std::string, std::function<visFile*()>>&
        _registered_types();

};


// Abstract factory VisFile creator.
// Forwards on an argument pack. Actual arguments defined on visFile::create_file
template<typename... CreateArgs>
inline std::shared_ptr<visFile> visFile::create(
    const std::string& type,
    const std::string& name,
    CreateArgs&&... args
) {

    auto& _type_list = _registered_types();

    if(_type_list.find(type) == _type_list.end()) {
        throw std::runtime_error(
            fmt::format("Cannot create visFile of unknown type {}", type)
        );
    }

    // Lookup the registered file and create an instance
    INFO("Creating file %s of type %s", name.c_str(), type.c_str());
    auto file = std::shared_ptr<visFile>(_type_list[type]());
    file->create_file(name, std::forward<CreateArgs>(args)...);

    return file;
}

// Add a function to the type map that creates a type map.
template<typename T>
inline int visFile::register_file_type(const std::string key) {
    std::cout << "Registering file type: " << key << std::endl;
    _registered_types()[key] = []() { return new T(); };
    return 0;
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
class visFileBundle {

public:

    /**
     * Initialise the file bundle
     * @param root_path Directory to write into.
     * @param inst_name Instrument name (e.g. chime)
     * @param freq_chunk ID of the frequency chunk being written
     * @param rollover Maximum time length of file.
     * @param window_size Number of "active" timesamples to keep.
     * @param ... Arguments passed through to `visFile::visFile`.
     *
     * @warning The directory will not be created if it doesn't exist.
     **/
    template<typename... InitArgs>
    visFileBundle(const std::string& type, const std::string& root_path,
                  const std::string& instrument_name,
                  const std::map<std::string, std::string>& metadata,
                  int freq_chunk,
                  size_t rollover, size_t window_size,
                  InitArgs... args);

    /**
     * Write a new time sample into this set of files
     * @param new_time Time of sample
     * @param ...      Arguments passed through to `visFile::write_sample`
     * @return True if an error occured while writing
     **/
    template<typename... WriteArgs>
    bool add_sample(time_ctype new_time, WriteArgs&&... args);

protected:

    // Add a file if we need to
    virtual void add_file(time_ctype first_time);

    // Find/create the slot for data at this time to go into
    bool resolve_sample(time_ctype new_time);

    std::map<uint64_t, std::tuple<std::shared_ptr<visFile>, uint32_t>> vis_file_map;
    // Thin function to actually create the file
    std::function<std::shared_ptr<visFile>(std::string, std::string, std::string)> mk_file;

    const std::string root_path;

    const std::string instrument_name;
    const int freq_chunk;

    size_t rollover;
    size_t window_size;

    std::string acq_name;
    double acq_start_time;

    // Flag to force moving to a new file
    bool change_file = false;

};

/**
 * @brief Extension to visFileBundle to manage buffer files for the
 *        calibration broker.
 *
 * This version is intended to write to a single file, with a
 * static user defined file name. The file mapping can be cleared
 * so that a new file is written to and the previous one is available
 * for reading. Swapping these files is managed by visCalWriter.
 *
 * @author Tristan Pinsonneault-Marotee
 **/
class visCalFileBundle : public visFileBundle {

public:

    /**
     * Initialise the file bundle
     * @param root_path Directory to write into.
     * @param inst_name Instrument name (e.g. chime)
     * @param freq_chunk ID of the frequency chunk being written
     * @param rollover Maximum time length of file.
     * @param window_size Number of "active" timesamples to keep.
     * @param ... Arguments passed through to `visFile::visFile`.
     *
     * @warning The directory will not be created if it doesn't exist.
     **/
    template<typename... Args>
    visCalFileBundle(Args... args) :
        visFileBundle(args...) {};

    /**
     * Close all files and clear the map.
     **/
    void clear_file_map();

    /**
     * Set the file name to write to.
     **/
    void set_file_name(std::string file_name, std::string acq_name);

    /**
     * Add a new file to the map of open files and let the
     * previous one be flushed out as samples come in.
     **/
    void swap_file(std::string new_fname, std::string new_aname);

protected:

    // Override parent method to use a set file name
    void add_file(time_ctype first_time) override;

    std::string acq_name, file_name;

};


template<typename... InitArgs>
inline visFileBundle::visFileBundle(
    const std::string& type, const std::string& root_path,
    const std::string& instrument_name,
    const std::map<std::string, std::string>& metadata,
    int freq_chunk, size_t rollover, size_t window_size, InitArgs... args
) :
    root_path(root_path),
    instrument_name(instrument_name),
    freq_chunk(freq_chunk),
    rollover(rollover),
    window_size(window_size)
{

    // Make a lambda function that creates a file. This is a little convoluted,
    // but is the easiest way of passing on the variadic arguments to the
    // constructor into the file creation.
    mk_file = [type, metadata, args...](std::string file_name,
                                        std::string acq_name,
                                        std::string root_path) {
        // Add the acq name to the metadata
        auto metadata_acq = metadata;
        metadata_acq["acquisition_name"] = acq_name;

        std::string abspath= root_path + '/' + acq_name.c_str() + '/' + file_name;
        return visFile::create(type, abspath, metadata_acq, args...);
    };
}


template<typename... WriteArgs>
inline bool visFileBundle::add_sample(time_ctype new_time, WriteArgs&&... args) {

    if(resolve_sample(new_time)) {
        std::shared_ptr<visFile> file;
        uint32_t ind;
        // We can now safely add the sample into the file
        std::tie(file, ind) = vis_file_map[new_time.fpga_count];
        file->write_sample(ind, std::forward<WriteArgs>(args)...);

        return false;
    } else {
        return true;
    }
}

//template<typename... InitArgs>
//inline visCalFileBundle::visCalFileBundle(const std::string& type,
//                                   const std::string& root_path,
//                                   const std::string& instrument_name,
//                                   const std::map<std::string, std::string>& metadata,
//                                   int freq_chunk,
//                                   size_t rollover, size_t window_size,
//                                   InitArgs... args) :
//    visFileBundle::visFileBundle(type, root_path, instrument_name, metadata,
//                                 freq_chunk, rollover, window_size, args...) {}
//
/**
 * @brief Create a lock file for the given file.
 * @param filename Name of file to lock.
 * @return The name of the lock file.
 **/
std::string create_lockfile(std::string filename);

#define REGISTER_VIS_FILE(key, T) int _register_ ## T = visFile::register_file_type<T>(key)


// Implementation of TEMP_FAILURE_RETRY for file writing which is missing on MacOS
#if defined( __APPLE__ )
// Taken from
// https://android.googlesource.com/platform/system/core/+/master/base/include/android-base/macros.h
#ifndef TEMP_FAILURE_RETRY
#define TEMP_FAILURE_RETRY(exp)            \
  ({                                       \
    decltype(exp) _rc;                     \
    do {                                   \
      _rc = (exp);                         \
    } while (_rc == -1 && errno == EINTR); \
    _rc;                                   \
  })
#endif
#endif


#endif
