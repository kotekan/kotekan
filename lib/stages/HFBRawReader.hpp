/*****************************************
@file
@brief Read HFBFileRaw data.
- HFBRawReader : public RawReader
*****************************************/
#ifndef _HFB_RAW_READER_HPP
#define _HFB_RAW_READER_HPP

#include "Config.hpp" // for Config
#include "HFBFrameView.hpp"
#include "RawReader.hpp"       // for RawReader
#include "bufferContainer.hpp" // for bufferContainer
#include "visUtil.hpp"         // for frameID

#include <stdint.h> // for uint32_t
#include <string>   // for string
#include <vector>   // for vector

/**
 * @class HFBRawReader
 * @brief Read and stream a raw 21cm absorber file.
 *
 * This class inherits from the RawReader base class and reads raw 21cm absorber data
 * @author James Willis
 */
class HFBRawReader : public RawReader<HFBFrameView> {

public:
    /// default constructor
    HFBRawReader(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);

    ~HFBRawReader();

    /**
     * @brief Get the beams in the file.
     **/
    const std::vector<uint32_t>& beams() {
        return _beams;
    }

    /**
     * @brief Get the sub-frequencies in the file.
     **/
    const std::vector<uint32_t>& subfreqs() {
        return _subfreqs;
    }

protected:
    // Create an empty frame
    void create_empty_frame(frameID frame_id) override;

private:
    // The metadata
    std::vector<uint32_t> _beams;
    std::vector<uint32_t> _subfreqs;
};

#endif
