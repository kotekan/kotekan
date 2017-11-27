#ifndef VIS_FILE_HPP
#define VIS_FILE_HPP

#include <iostream>
#include "errors.h"
#include <highfive/H5File.hpp>

// Structs to represent the datatypes of the index maps
struct freq_ctype {
    double centre;
    double width;
};

struct input_ctype {
    // Allow initialisation from a std::string
    input_ctype(uint16_t id, std::string serial);

    uint16_t chan_id;
    char correlator_input[32];
};

struct time_ctype {
    uint64_t fpga_count;
    double ctime;
};

struct prod_ctype {
    uint16_t input_a;
    uint16_t input_b;
};

struct complex_int {
    int32_t r;
    int32_t i;
};


/// Class to manage access to a CHIME correlator file
///
/// This is only designed with writing data in mind.
class visFile {

public:

    /// Create file (and lock file).
    /// \param name Name of the file to write
    /// \param freqs Frequencies channels that will be in the file
    /// \param inputs Inputs that are in the file
    visFile(const std::string& name,
            const std::string& acq_name,
            const std::string& inst_name,
            const std::string& notes,
            const std::vector<freq_ctype>& freqs,
            const std::vector<input_ctype>& inputs);
    ~visFile();


    /// Write a new time sample into this file
    /// \param new_time Time of sample
    /// \param freq_ind Index of the frequency we are writing
    /// \param new_vis Visibility data for this frequency
    /// \param new_weight Visibility weights for this frequency
    /// \param new_gcoeff Gain coefficient data
    /// \param new_gexp Gain exponent data
    /// \return The number of entries in the time axis
    size_t addSample(time_ctype new_time, uint32_t freq_ind,
                   std::vector<complex_int> new_vis,
                   std::vector<uint8_t> new_weight,
                   std::vector<complex_int> new_gcoeff,
                   std::vector<int32_t> new_gexp);

private:


    // Create the index maps from the frequencies and the inputs
    void createIndex(const std::vector<freq_ctype>& freqs,
                     const std::vector<input_ctype>& inputs);

    // Create the main visibility holding datasets
    void createDatasets(size_t nfreq, size_t ninput, size_t nprod);


    // Pointer to the underlying HighFive file
    std::unique_ptr<HighFive::File> file;

    std::string lock_filename;

};


// These templated functions are needed in order to tell HighFive how the
// various structs are converted into HDF5 datatypes
namespace HighFive {
template <> DataType create_datatype<freq_ctype>();
template <> DataType create_datatype<time_ctype>();
template <> DataType create_datatype<input_ctype>();
template <> DataType create_datatype<prod_ctype>();
template <> DataType create_datatype<complex_int>();
};


#endif
