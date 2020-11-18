/*****************************************
@file
@brief Transpose VisFileRaw data and write into HDF5 files.
- VisTranspose : public Transpose
*****************************************/
#ifndef VIS_TRANSPOSE_HPP
#define VIS_TRANSPOSE_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.h"            // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"  // for dset_id_t
#include "visFileArchive.hpp"  // for visFileArchive
#include "visFileH5.hpp"
#include "visUtil.hpp" // for cfloat, time_ctype, freq_ctype, input_ctype, prod_ctype
#include "Transpose.hpp"

#include "json.hpp" // for json

#include <chrono>
#include <memory>   // for shared_ptr
#include <stddef.h> // for size_t
#include <stdint.h> // for uint32_t
#include <string>   // for string
#include <vector>   // for vector


/**
 * @class VisTranspose
 * @brief Stage to transpose raw visibility data
 *
 * This class inherits from the RawReader base class and reads raw visibility data
 * @author Tristan Pinsonneault-Marotte, Rick Nitsche
 */
class visTranspose : public Transpose {
public:
    /// Constructor; loads parameters from config
    visTranspose(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);
    ~visTranspose() = default;

protected:
    /// Request dataset states from the datasetManager and prepare all metadata
    /// that is not already set in the constructor.
    bool get_dataset_state(dset_id_t ds_id) override;
   
    // Get frame size, fpga_seq_total and dataset_id from VisFrameView 
    std::tuple<size_t, uint64_t, dset_id_t> get_frame_data() override;
    
    // Create VisFileArchive
    void create_hdf5_file() override;

    // Copy data into local vectors using VisFrameView
    void copy_frame_data(uint32_t freq_index, uint32_t time_index) override;
    
    // Copy flags into local vectors using VisFrameView
    void copy_flags(uint32_t time_index) override;
    
    // Write datasets to file
    void write_chunk() override;
    
    // Increment between chunks
    void increment_chunk() override;

private:
    // Datasets to be stored until ready to write
    std::vector<cfloat> vis;
    std::vector<float> vis_weight;
    std::vector<float> eval;
    std::vector<cfloat> evec;
    std::vector<float> erms;
    std::vector<cfloat> gain;
    std::vector<float> frac_lost;
    std::vector<float> frac_rfi;
    std::vector<float> input_flags;
    std::vector<rstack_ctype> reverse_stack;

    /// The list of frequencies and inputs that get written into the index maps
    /// of the HDF5 files
    std::vector<time_ctype> times;
    std::vector<freq_ctype> freqs;
    std::vector<input_ctype> inputs;
    std::vector<prod_ctype> prods;
    std::vector<uint32_t> ev;
    std::vector<stack_ctype> stack;

    /// Number of products to write
    size_t num_prod;
    size_t num_input;
    size_t num_ev;

    std::shared_ptr<visFileArchive> file;
};

#endif
