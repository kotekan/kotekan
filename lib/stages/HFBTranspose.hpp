/*****************************************
@file
@brief Transpose HFBFileRaw data and write into HDF5 files.
- HFBTranspose : public Transpose
*****************************************/
#ifndef HFB_TRANSPOSE_HPP
#define HFB_TRANSPOSE_HPP

#include "Config.hpp"          // for Config
#include "HFBFileArchive.hpp"  // for HFBFileArchive
#include "Transpose.hpp"       // for Transpose
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"  // for dset_id_t
#include "visUtil.hpp"         // for cfloat, time_ctype, freq_ctype, input_ctype, prod_ctype

#include <memory>   // for shared_ptr
#include <stddef.h> // for size_t
#include <stdint.h> // for uint32_t
#include <string>   // for string
#include <tuple>    // for tuple
#include <vector>   // for vector


/**
 * @class HFBTranspose
 * @brief Stage to transpose raw HFB data
 *
 * This class inherits from the Transpose base class and transposes raw HFB data.
 * @author James Willis
 */
class HFBTranspose : public Transpose {
public:
    /// Constructor; loads parameters from config
    HFBTranspose(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);
    ~HFBTranspose() = default;

protected:
    /// Request dataset states from the datasetManager and prepare all metadata
    /// that is not already set in the constructor.
    bool get_dataset_state(dset_id_t ds_id) override;

    // Get frame size, fpga_seq_total and dataset_id from HFBFrameView
    std::tuple<size_t, uint64_t, dset_id_t> get_frame_data() override;

    // Create HFBFileArchive
    void create_hdf5_file() override;

    // Copy data into local vectors using HFBFrameView
    void copy_frame_data(uint32_t freq_index, uint32_t time_index) override;

    // Copy flags into local vectors using HFBFrameView
    void copy_flags(uint32_t time_index) override;

    // Write datasets to file
    void write_chunk(size_t t_ind, size_t f_ind) override;

    // Increment between chunks
    void increment_chunk(size_t& t_ind, size_t& f_ind, bool& t_edge, bool& f_edge) override;

private:
    // Datasets to be stored until ready to write
    std::vector<time_ctype> time;
    std::vector<float> hfb;
    std::vector<float> hfb_weight;
    std::vector<float> frac_lost;

    /// The list of frequencies and inputs that get written into the index maps
    /// of the HDF5 files
    std::vector<time_ctype> times;
    std::vector<freq_ctype> freqs;
    std::vector<uint32_t> beams;
    std::vector<uint32_t> sub_freqs;

    /// Number of products to write
    size_t num_beams;
    size_t num_subfreq;

    std::shared_ptr<HFBFileArchive> file;
};

#endif
