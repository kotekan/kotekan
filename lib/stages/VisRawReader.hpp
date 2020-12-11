/*****************************************
@file
@brief Read visFileRaw data.
- VisRawReader : public RawReader
*****************************************/
#ifndef _VIS_RAW_READER_HPP
#define _VIS_RAW_READER_HPP

#include "Config.hpp"          // for Config
#include "RawReader.hpp"       // for RawReader
#include "bufferContainer.hpp" // for bufferContainer
#include "visBuffer.hpp"
#include "visUtil.hpp" // for frameID, input_ctype, prod_ctype, stack_ctype, rstack_ctype

#include <stdint.h> // for uint32_t
#include <string>   // for string
#include <vector>   // for vector

/**
 * @class VisRawReader
 * @brief Stage to read raw visibility data.
 *
 * This class inherits from the RawReader base class and reads raw visibility data
 * @author Richard Shaw, Tristan Pinsonneault-Marotte, Rick Nitsche
 */
class VisRawReader : public RawReader<VisFrameView> {

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
