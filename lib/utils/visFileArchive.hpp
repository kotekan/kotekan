#ifndef VIS_FILE_ARCHIVE_HPP
#define VIS_FILE_ARCHIVE_HPP

#include "FileArchive.hpp"
#include "kotekanLogging.hpp" // for logLevel, kotekanLogging
#include "visUtil.hpp"        // for freq_ctype, prod_ctype, time_ctype, input_ctype

#include <highfive/H5DataSet.hpp>  // for DataSet
#include <highfive/H5DataType.hpp> // for DataType
#include <highfive/H5File.hpp>     // for File
#include <map>                     // for map
#include <memory>                  // for allocator, unique_ptr
#include <stddef.h>                // for size_t
#include <string>                  // for string
#include <vector>                  // for vector


/** @brief A CHIME correlator archive file.
 *
 * The class creates and manages writes to a CHIME style correlator archive
 * file in the standard HDF5 format. It also manages the lock file.
 *
 * @author Richard Shaw
 **/
class visFileArchive : public FileArchive {

public:
    /**
     * @brief Creates a visFileArchive object.
     *
     * @param name Path of the file to write into (without file extension).
     * @param metadata Metadata attributes.
     * @param times Vector of time indices.
     * @param freqs Vector of frequency indices.
     * @param inputs Vector of input indices.
     * @param prods Vector of product indices.
     * @param num_ev Number of eigenvectors.
     * @param chunk_size HDF5 chunk size (frequencies * products * times).
     * @param comp_alg Compression algorithm to use.
     * @param zstd_comp_lvl ZSTD compression level.
     * @param log_level kotekan log level for any logging generated by the visFileArchive instance
     *
     * @return Instance of visFileArchive.
     **/
    visFileArchive(const std::string& name, const std::map<std::string, std::string>& metadata,
                   const std::vector<time_ctype>& times, const std::vector<freq_ctype>& freqs,
                   const std::vector<input_ctype>& inputs, const std::vector<prod_ctype>& prods,
                   size_t num_ev, std::vector<int> chunk_size, const std::string comp_alg,
                   const uint32_t zstd_comp_lvl, const kotekan::logLevel log_level);
    /**
     * @brief Creates a visFileArchive object.
     *
     * @param name Path of the file to write into (without file extension).
     * @param metadata Metadata attributes.
     * @param times Vector of time indices.
     * @param freqs Vector of frequency indices.
     * @param inputs Vector of input indices.
     * @param prods Vector of product indices.
     * @param stack Vector of stack indices.
     * @param reverse_stack Vector mapping products to stacks.
     * @param num_ev Number of eigenvectors.
     * @param chunk_size HDF5 chunk size (frequencies * products * times).
     * @param comp_alg Compression algorithm to use.
     * @param zstd_comp_lvl ZSTD compression level.
     * @param log_level kotekan log level for any logging generated by the visFileArchive instance
     *
     * @return Instance of visFileArchive.
     **/
    visFileArchive(const std::string& name, const std::map<std::string, std::string>& metadata,
                   const std::vector<time_ctype>& times, const std::vector<freq_ctype>& freqs,
                   const std::vector<input_ctype>& inputs, const std::vector<prod_ctype>& prods,
                   const std::vector<stack_ctype>& stack, std::vector<rstack_ctype>& reverse_stack,
                   size_t num_ev, std::vector<int> chunk_size, const std::string comp_alg,
                   const uint32_t zstd_comp_lvl, const kotekan::logLevel log_level);

    /**
     * @brief Destructor.
     **/
    virtual ~visFileArchive();

    /**
     * @brief Write a block in time/freq.
     *
     * @param name Path of the file to write into (without file extension).
     * @param f_ind Frequency index.
     * @param t_ind Time index.
     * @param chunk_f Size of chunk in frequency dimension.
     * @param chunk_t Size of chunk in time dimension.
     * @param data Pointer to the data.
     **/
    template<typename T>
    void write_block(std::string name, size_t f_ind, size_t t_ind, size_t chunk_f, size_t chunk_t,
                     const T* data);


private:
    // Prepare a file
    void setup_file(const std::string& name, const std::map<std::string, std::string>& metadata,
                    const std::vector<time_ctype>& times, const std::vector<freq_ctype>& freqs,
                    const std::vector<prod_ctype>& prods, size_t num_ev,
                    std::vector<int> chunk_size);

    // Helper to create datasets
    virtual void create_dataset(const std::string& name, const std::vector<std::string>& axes,
                                HighFive::DataType type, const bool& compress);

    // Helper function to create an axis
    template<typename T>
    void create_axis(std::string name, const std::vector<T>& axis);

    // Create the index maps from the frequencies and the inputs
    void create_axes(const std::vector<time_ctype>& times, const std::vector<freq_ctype>& freqs,
                     const std::vector<input_ctype>& inputs, const std::vector<prod_ctype>& prods,
                     size_t num_ev);
    void create_axes(const std::vector<time_ctype>& times, const std::vector<freq_ctype>& freqs,
                     const std::vector<input_ctype>& inputs, const std::vector<prod_ctype>& prods,
                     const std::vector<stack_ctype>& stack, size_t num_ev);

    // Create the main visibility holding datasets
    void create_datasets();

    // Get datasets
    HighFive::DataSet dset(const std::string& name);
    size_t length(const std::string& axis_name);

    // Whether to write eigenvalues or not
    bool write_ev;

    // Description of weights stored in vis_weight dataset
    std::string weight_type;

    // HDF5 chunk size
    std::vector<int> chunk;

    // Pointer to the underlying HighFive file
    std::unique_ptr<HighFive::File> file;

    std::string lock_filename;

    // Whether the products have been compressed via baseline stacking
    bool stacked = false;

    // Shortcut for axes labels
    inline std::string prod_or_stack();
};


inline std::string visFileArchive::prod_or_stack() {
    return stacked ? "stack" : "prod";
}

#endif
