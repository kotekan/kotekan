/**
 * @file
 * @brief Sort the incoming tracking bream frames by frequency and time and split by frequencies.
 *  - BeamBufferSort : public kotekan::Stage
 */

#ifndef BEAM_BUFFER_SORT
#define BEAM_BUFFER_SORT

#include "BeamMetadata.hpp" // for BeamMetadata
#include "Config.hpp"       // for Config
#include "Stage.hpp"
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for uint32_t
#include <string>   // for string
#include <vector>   // for vector


using std::vector;


/**
 * @class BeamBufferSort
 * @brief This stage sort the beam buffer by frequency and time and split buffer by frequencies.
 *
 * @par Buffers
 * @buffer in_buf Kotekan single frame tracking beam buffer.
 *     @buffer_format Array of @c chars
 * @buffer out_buf Kotekan sorted tracking beam buffer.
 *     @buffer_format Array of @c uint32_t
 *
 * @author Jing Santiago Luo
 *
 *
 */


class BeamBufferSort : public kotekan::Stage {
public:
    /// Constructor
    BeamBufferSort(kotekan::Config& config_, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);


    /// Destructor
    virtual ~BeamBufferSort();
    /// make a frequency ID'ed metadata
    void make_freq_meta(FreqIDBeamMetadata* out_meta, uint32_t freq_id, 
        uint64_t fpga_seq_start, timespec ctime, uint64_t stream_id,        
        dset_id_t dataset_id, uint32_t beam_number, float ra, float dec,
	uint32_t scaling);
    /// Transfer a non-frequency id'ed metadata to frequency id'ed metadata
    void nonfreq_meta_2_freq_meta(BeamMetadata* nonfreq_meta,  
        FreqIDBeamMetadata* freq_meta, uint32_t freq_id);
    ///Fill up the empty frame.
    void fill_empty_frame(uint32_t time_idx, uint32_t freq_idx, 
        uint64_t fpga_seq_start0, timespec ctime0, uint32_t beam_number, 
	double ra, double dec, double scaling);
    
    /// Primary loop to wait for buffers, sort beam buffer.
    void main_thread() override;

private:
    /// The input buffer which has one metadata & voltage block per frame.
    struct Buffer* in_buf;
    /// The sort_queue
    std::vector<uint8_t*> sort_queue;
    /// The output buffers to put frames into
    std::vector<struct Buffer*> out_bufs;
    /// Start time of one output buffer frame
    std::vector<uint64_t> out_buf_frame0;
    /// Out buff frame 0 metadata.
    std::vector<struct FreqIDBeamMetadata*> frame0_meta;
    /// current time offset in output buffer frame
    std::vector<uint32_t> out_buf_time_offset;
    /// Number of channel in the out put buffer
    std::vector<uint32_t> out_buf_nchan;
    /// Start chan for each out buffer
    std::vector<uint32_t> out_buf_start_chan;
    /// Frame fill up status
    std::vector<vector<uint8_t>> queue_status;
    /// Config variables
    /// If the beam buffer has frequency bin in metadata
    bool has_freq_bin;
    /// samples per single beam buffer frame
    uint32_t samples_per_data_set;
    /// Total number of frequency channel
    uint32_t total_freq_chan;
    /// Use n out put buffer
    uint32_t use_n_out_buffer;
    /// Time resolution for each sample
    double time_resolution;
    /// Number of frequencies per output buffer
    /// number of time frames in the sorting queue.
    int wait_nframes;
    /// The size of the FreqIDBeamMeta
    uint32_t FreqIDBeamMeta_size;
    /// Pad n marginal frames from the start
    uint64_t start_marginal_nframe;
    /// subFrame, which includes the frame metadata, size
    /// in the output merged frame. 
    uint32_t sub_frame_size;
    /// Time span for each subframe.
    long subframe_time_nsec;
};

#endif //BEAM_BUFFER_SORT_HPP
