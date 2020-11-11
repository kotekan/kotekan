/*****************************************
@file
@brief Read visFileRaw data.
- VisRawReader : public RawReader
*****************************************/
#ifndef _VIS_RAW_READER_HPP
#define _VIS_RAW_READER_HPP

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "buffer.h"
#include "bufferContainer.hpp"
#include "datasetManager.hpp" // for dset_id_t
#include "visUtil.hpp"        // for freq_ctype (ptr only), input_ctype, prod_ctype, rstack_ctype
#include "RawReader.hpp"

#include "json.hpp" // for json

#include <map>      // for map
#include <stddef.h> // for size_t
#include <stdint.h> // for uint32_t, uint8_t
#include <string>   // for string
#include <utility>  // for pair
#include <vector>   // for vector

/**
 * @class VisRawReader
 * @brief Stage to read raw visibility data.
 *
 * This class inherits from the RawReader base class and reads raw visibility data
 * @author Richard Shaw, Tristan Pinsonneault-Marotte, Rick Nitsche
 */
class VisRawReader : public RawReader {

public:
    /// default constructor
    VisRawReader(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);

    ~VisRawReader();

    /**
     * @brief Get the products in the file.
     **/
    const std::vector<prod_ctype>& prods() {
        return _prods;
    }

    /**
     * @brief Get the stack in the file.
     **/
    const std::vector<stack_ctype>& stack() {
        return _stack;
    }

    /**
     * @brief Get the inputs in the file.
     **/
    const std::vector<input_ctype>& inputs() {
        return _inputs;
    }

    /**
     * @brief Get the ev axis in the file.
     **/
    const std::vector<uint32_t>& ev() {
        return _ev;
    }

protected:
    // Create an empty frame
    void create_empty_frame(frameID frame_id) override;
    
    // Get dataset ID
    dset_id_t& get_dataset_id(frameID frame_id) override;

private:
    // The metadata
    std::vector<prod_ctype> _prods;
    std::vector<input_ctype> _inputs;
    std::vector<stack_ctype> _stack;
    std::vector<rstack_ctype> _rstack;
    std::vector<uint32_t> _ev;
    uint32_t _num_stack;
};

#endif
