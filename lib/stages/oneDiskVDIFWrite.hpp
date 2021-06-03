/**
 * @file
 * @brief Write sorted beam buffer to disk
 *  - oneDiskVDIFWrite : public kotekan::Stage
 */


#ifndef ONE_DISK_VDIF_WRITE
#define ONE_DISK_VDIF_WRITE

#include "BeamMetadata.hpp" // for BeamMetadata
#include "Config.hpp"       // for Config
#include "Stage.hpp"
#include "bufferContainer.hpp" // for bufferContainer
#include "vdif_functions.h"   // for VDIFHeader


#include <stdint.h> // for uint32_t
#include <string>   // for string
#include <vector>   // for vector


using std::vector;


/**
 * @class oneDiskVDIFWrite
 * @brief This stage writes the sorted beam buffer to disk in 
 *  vdif format.
 *
 *  @par 
 *  @buffer in_buff kotekan sorted beam buffer in the format of merged buffer.
 *      @buffer_format Array of @c uint32_t
 *  
 *  @author Jing Santiago Luo
 *
 *
 */


struct VDIF_Frame {
    /// VDIF header
    VDIFHeader* vdif_header;
    /// Payload
    uint8_t* payload;
};

class oneDiskVDIFWrite : public kotekan::Stage {
public:
    /// Constructor
    oneDiskVDIFWrite(kotekan::Config& config_, const std::string& unique_name,
                     kotekan::bufferContainer& buffer_container);

    /// Destructor
    virtual ~oneDiskVDIFWrite();
    /// Primary loop to wait for buffers, write vdif format out.
    void main_thread() override;

private:
    /// Input buffer which packs mulitple frequencies comeing at the same time in
    /// one frame. 
    struct Buffer* in_buf;
    /// Buffer for vdif output. 
    VDIF_FRAME* vdif_frame;
    uint32_t vdif_samples_per_frame; 
    uint32_t vdif_freq_per_frame;
    uint32_t time_res_nsec;    
};

#endif // BEAM_VDIF_WRITE_HPP

