#ifndef VIS_FILE_HPP
#define VIS_FILE_HPP

#include <iostream>
#include <cstdint>
#include <map>

#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>

#include "visUtil.hpp"
#include "errors.h"


/** @brief A CHIME correlator file.
 * 
 * The class creates and manages writes to a CHIME style correlator output
 * file. It also manages the lock file.
 * 
 * @author Richard Shaw
 **/
class visFile {

public:

    /** @brief Create the file.
     * 
     *  @param name Name of the file to write
     *  @param acq_name Name of the acquisition to write
     *  @param root_path Base directory to write the acquisition into
     *  @param inst_name Instrument name (e.g. chime)
     *  @param notes Note about the acquisition
     *  @param weights_type What the visibility weights represent (e.g. 'inverse_var')
     *  @param freqs Frequencies channels that will be in the file
     *  @param inputs Inputs that are in the file
     *  @param prods Products that are in the file.
     *  @param num_ev Number of eigenvectors to write (0 turns off the datasets entirely).
     **/
    visFile(const std::string& name,
            const std::string& acq_name,
            const std::string& root_path,
            const std::string& inst_name,
            const std::string& notes,
            const std::string& weights_type,
            const std::vector<freq_ctype>& freqs,
            const std::vector<input_ctype>& inputs,
            const std::vector<prod_ctype>& prods,
            size_t num_ev);
    ~visFile();

    /**
     * @brief Extend the file to a new time sample.
     * 
     * @param new_time The new time to add.
     * @return The index of the added time in the file.
     **/ 
    uint32_t extend_time(time_ctype new_time);

    /**
     * @brief Write a sample of data into the file at the given index.
     * 
     * @param new_vis Vis data.
     * @param new_weight Weight data.
     * @param new_gcoeff Gain coefficients.
     * @param new_gexp Gain exponents.
     * @param new_eval Eigenvalues.
     * @param new_evec Eigenvectors.
     * @param new_erms RMS after eigenvalue removal.
     **/
    void write_sample(uint32_t time_ind, uint32_t freq_ind,
                      std::vector<cfloat> new_vis,
                      std::vector<float> new_weight,
                      std::vector<cfloat> new_gcoeff,
                      std::vector<int32_t> new_gexp,
                      std::vector<float> new_eval,
                      std::vector<cfloat> new_evec,
                      float new_erms);

    /**
     * @brief Return the current number of current time samples.
     * 
     * @return The current number of time samples.
     **/
    size_t num_time();

private:


    // Create the index maps from the frequencies and the inputs
    void create_index(const std::vector<freq_ctype>& freqs,
                      const std::vector<input_ctype>& inputs,
                      const std::vector<prod_ctype>& prods,
                      size_t num_ev);

    // Create the main visibility holding datasets
    void create_datasets(size_t nfreq, size_t ninput, size_t nprod,
                         size_t num_ev, std::string weights_type);

    // Get datasets
    HighFive::DataSet vis();
    HighFive::DataSet vis_weight();
    HighFive::DataSet gain_coeff();
    HighFive::DataSet gain_exp();
    HighFive::DataSet time();
    HighFive::DataSet eval();
    HighFive::DataSet evec();
    HighFive::DataSet erms();

    // Get dimensions
    size_t num_prod();
    size_t num_input();
    size_t num_freq();
    size_t num_ev();

    // Whether to write eigenvalues or not
    bool write_ev;

    // Pointer to the underlying HighFive file
    std::unique_ptr<HighFive::File> file;

    std::string lock_filename;

};


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
    visFileBundle(const std::string root_path,
                  const std::string instrument_name,
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

private:

    // Add a file if we need to 
    void add_file(time_ctype first_time);

    // Thin function to actually create the file
    std::function<std::shared_ptr<visFile>(std::string, std::string, std::string)> mkFile;

    // Find/create the slot for data at this time to go into
    bool resolve_sample(time_ctype new_time);

    const std::string root_path;
    const std::string instrument_name;
    const int freq_chunk;

    size_t rollover;
    size_t window_size;

    std::string acq_name;
    double acq_start_time;

    std::map<uint64_t, std::tuple<std::shared_ptr<visFile>, uint32_t>> vis_file_map;

};


template<typename... InitArgs>
inline visFileBundle::visFileBundle(const std::string root_path,
                             const std::string instrument_name,
                             int freq_chunk,
                             size_t rollover, size_t window_size,
                             InitArgs... args) :

    root_path(root_path),
    instrument_name(instrument_name),
    freq_chunk(freq_chunk),
    rollover(rollover),
    window_size(window_size)
{
    mkFile = [instrument_name, args...](std::string file_name,
                                        std::string acq_name,
                                        std::string root_path) {
        return std::make_shared<visFile>(file_name, acq_name, root_path,
                                         instrument_name, args...);
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


// These templated functions are needed in order to tell HighFive how the
// various structs are converted into HDF5 datatypes
namespace HighFive {
template <> DataType create_datatype<freq_ctype>();
template <> DataType create_datatype<time_ctype>();
template <> DataType create_datatype<input_ctype>();
template <> DataType create_datatype<prod_ctype>();
template <> DataType create_datatype<cfloat>();
};


#endif
