#ifndef VIS_FILE_RAW_HPP
#define VIS_FILE_RAW_HPP

#include <iostream>
#include <cstdint>

#include "visFile.hpp"
#include "visUtil.hpp"
#include "errors.h"

#include "json.hpp"

using json = nlohmann::json;

/** @brief A CHIME correlator file in raw format.
 * 
 * The class creates and manages writes to a CHIME style correlator output
 * file. It also manages the lock file.
 * 
 * @author Richard Shaw
 **/
class visFileRaw : public visFile {

public:

    ~visFileRaw();

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
                      float new_erms) override;

    /**
     * @brief Return the current number of current time samples.
     * 
     * @return The current number of time samples.
     **/
    size_t num_time() override;


protected:

    // Implement the create_file method
    void create_file(const std::string& name,
                     const std::map<std::string, std::string>& metadata,
                     const std::vector<freq_ctype>& freqs,
                     const std::vector<input_ctype>& inputs,
                     const std::vector<prod_ctype>& prods,
                     size_t num_ev, size_t max_time) override;

    // Create the time axis (separated for overloading)
    virtual void create_time_axis(size_t num_time);

    // Helper to create datasets
    virtual void create_dataset(const std::string& name,
                                const std::vector<std::string>& axes,
                                HighFive::DataType type);

    // Helper function to create an axis
    template<typename T>
    void create_axis(std::string name, const std::vector<T>& axis);

    // Create the index maps from the frequencies and the inputs
    void create_axes(const std::vector<freq_ctype>& freqs,
                     const std::vector<input_ctype>& inputs,
                     const std::vector<prod_ctype>& prods,
                     size_t num_ev, size_t num_time);

    // Create the main visibility holding datasets
    void create_datasets();

    // Whether to write eigenvalues or not
    bool write_ev;

    std::string lock_filename;


    // Calculate offsets into the file for each dataset, and open it
    void setup_raw();

    /**
     * @brief  Helper routine for writing data into the file
     * 
     * @param dset_base Offset of dataset in file
     * @param ind       The index into the file dataset in chunks.
     * @param n         The size of the chunk in elements.
     * @param vec       The data to write out.
     **/
    template<typename T>
    bool write_raw(off_t dset_base, int ind, size_t n, 
                   const std::vector<T>& vec);

    /**
     * @brief  Helper routine for writing data into the file
     * 
     * @param dset_base Offset of dataset in file
     * @param ind       The index into the file dataset in chunks.
     * @param n         The size of the chunk in elements.
     * @param data       The data to write out.
     **/
    template<typename T>
    bool write_raw(off_t dset_base, int ind, size_t n, 
                   const T * data);

    // The metadata we will write into the file
    json file_metadata;


    // Save the size for when we are outside of HDF5 space
    size_t nfreq, nprod, ninput, nev;

    // File descriptor of file.
    int fd;

    // Store offsets into the file for writing
    off_t vis_offset, weight_offset, gcoeff_offset, gexp_offset,
          eval_offset, evec_offset, erms_offset, time_offset;

    // Keep a list of the times we've seen
    std::vector<time_ctype> times;
    
};



#endif