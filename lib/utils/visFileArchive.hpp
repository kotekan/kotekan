#ifndef VIS_FILE_ARCHIVE_HPP
#define VIS_FILE_ARCHIVE_HPP

#include <iostream>
#include <cstdint>

#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>

#include "visBuffer.hpp"
#include "visFile.hpp"
#include "visUtil.hpp"
#include "errors.h"
#include "visFileH5.hpp"  // For HighFive types

/** @brief A CHIME correlator archive file.
 * 
 * The class creates and manages writes to a CHIME style correlator archive
 * file in the standard HDF5 format. It also manages the lock file.
 * 
 * @author Richard Shaw
 **/
class visFileArchive {

public:

    visFileArchive(const std::string& name,
                   const std::map<std::string, std::string>& metadata,
                   const std::vector<time_ctype>& times,
                   const std::vector<freq_ctype>& freqs,
                   const std::vector<input_ctype>& inputs,
                   const std::vector<prod_ctype>& prods,
                   size_t num_ev);

    ~visFileArchive();

    // Write a block in time/freq
    template<typename T>
    void write_block(std::string name, size_t f_ind, size_t t_ind, size_t chunk_f,
                     size_t chunk_t, T* data);


protected:

    // Helper to create datasets
    virtual void create_dataset(const std::string& name,
                                const std::vector<std::string>& axes,
                                HighFive::DataType type);

    // Helper function to create an axis
    template<typename T>
    void create_axis(std::string name, const std::vector<T>& axis);

    // Create the index maps from the frequencies and the inputs
    void create_axes(const std::vector<time_ctype>& times,
                     const std::vector<freq_ctype>& freqs,
                     const std::vector<input_ctype>& inputs,
                     const std::vector<prod_ctype>& prods,
                     size_t num_ev);

    // Create the main visibility holding datasets
    void create_datasets();

    // Get datasets
    HighFive::DataSet dset(const std::string& name);
    size_t length(const std::string& axis_name);

    // Whether to write eigenvalues or not
    bool write_ev;

    // Pointer to the underlying HighFive file
    std::unique_ptr<HighFive::File> file;

    std::string lock_filename;

};


// TODO: these should be included from visFileH5
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
