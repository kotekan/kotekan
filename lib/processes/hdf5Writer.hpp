#ifndef HDF5_WRITER_HPP
#define HDF5_WRITER_HPP

#include "buffer.h"
#include "KotekanProcess.hpp"
#include "errors.h"
#include "util.h"
#include <unistd.h>
#include <highfive/H5File.hpp>

/*
 * Displays CHIME metedata for a given buffer
 * "buf": String with the name of the buffer display metadata info for.
 */

/* Structs to represent the datatypes of the index maps */
typedef struct {
    double centre;
    double width;
} freq_ctype;

typedef struct {
    uint16_t chan_id;
    char correlator_input[32];
} input_ctype;

typedef struct {
    uint64_t fpga_count;
    double ctime;
} time_ctype;

typedef struct {
    uint16_t input_a;
    uint16_t input_b;
} prod_ctype;

typedef struct {
    int32_t r;
    int32_t i;
} complex_int;

/* Class to manage access to a CHIME correlator file

This is only designed with writing data in mind.

 */
class visFile {

public:

     // Create file and lock file
     visFile(const std::string& name,
             const std::vector<freq_ctype>& freqs,
             const std::vector<input_ctype>& inputs);

     // Add a new time sample into this file
     void addSample(time_ctype time, std::vector<complex_int> vis);

 private:


      // Create the index maps from the frequencies and the inputs
      void createIndex(const std::vector<freq_ctype>& freqs,
                       const std::vector<input_ctype>& inputs);

      // Create the main visibility holding datasets
      void createDatasets(size_t nfreq, size_t ninput, size_t nprod);


     // Pointers to the underlying HighFive file and the time varying datasets
     std::unique_ptr<HighFive::File> file;
     std::unique_ptr<HighFive::DataSet> vis;
     std::unique_ptr<HighFive::DataSet> time_imap;
     std::unique_ptr<HighFive::CompoundType> time_h5type;

     bool _index_created = false;

 };


class hdf5Writer : public KotekanProcess {
public:
    hdf5Writer(Config &config,
               const string& unique_name,
               bufferContainer &buffer_container);
    //~hdf5Writer();
    void apply_config(uint64_t fpga_seq) override;

    void main_thread();


private:
    int32_t len;
    int32_t offset;

    size_t num_elements;

    std::unique_ptr<visFile> current_file;

    std::vector<struct Buffer*> buffers;
    std::vector<uint32_t> input_remap;

    std::vector<freq_ctype> freqs;
    std::vector<input_ctype> inputs;

};

namespace HighFive {
template <> DataType create_datatype<freq_ctype>();
template <> DataType create_datatype<time_ctype>();
template <> DataType create_datatype<input_ctype>();
template <> DataType create_datatype<prod_ctype>();
template <> DataType create_datatype<complex_int>();
};

#endif
