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
 * @brief This stage sort the beam buffer by frequency and time. The output also splits the sorted data buffer by frequency groups.
 *
 * This stage takes in the tracking beam frames and sort frames by frequeny and time so that 
 * the output data is from the same time and sorted by the frequency channels. 
 * To enable the ndisk writing, the output data can be put into mulitple output buffers by
 * frequency groups. 
 * 
 * @par Buffers
 * @buffer in_buf Kotekan single frame tracking beam buffer.
 *         @buffer_format Array of @c chars
 * @buffer out_bufs                Output buffer array for Kotekan sorted tracking beam 
 *                                 buffer with multiple frequency of data from the same 
 *                                 time. Each buffer in the buffer array contains a 
 *                                 group of data in different frequency but arrives at
 *                                 the same time.  
 *         @buffer_format Vectory Array of @c chars
 *
 * @conf   samples_per_data_set    Uint32_t. Number of time samples in one input frame.
 * @conf   has_freq_bin            Bool (default: false). If the incoming frames has 
 *                                 frequency in the metadata.
 * @conf   total_freq_chan         Uint32_t (default: false). Total frequency channel 
 *                                 of data. 
 * @conf   num_pol                 Uint32_t (default: 2). Number of polarizations.
 * @conf   use_n_out_buffer        Uint32_t. Number of output buffers to use. This is 
 *                                 also related to the number of channels in each buffer.
 * @conf   wait_nframes            Uint32_t (default: 20). Number of frames in the sort queue.
 *                                 Since the incoming data does not always arrive at the same
 *                                 time, it needs a big enough queue to host the incoming data.  
 * @conf   time_resolution         Double (default: 2.56e-06). Timing resoulation for each time
 *                                 sample. 
 * @conf   queue_frame_size        Number of data samples for each queue frame. For the gated
 *                                 data, the data size in the queue does not always has the 
 *                                 same number of data samples with the frame from the correlator.
 *                                 So the total data size in the queue is wait_nframes * queue_frame_size
 * @conf   dump_size               Int. Number of time samples for each output. The output size 
 *                                 should not bigger than 1/3 of total data size in the queue. 
 * @conf   align_start_time        Bool (default: True). If the start the queue align to the integer
 *                                 second of start time. VDIF or other VLBI software requrires the 
 *                                 start time align with the integer seconds.
 * @conf   nchan_buffer0           Uint32_t. Number of channels in output buffer0. If the total channels can
 *                                 not be splited evenly to each out buffer, output buffer0 will take the 
 *                                 remining number of channels and other buffers get the channels evenly. 
 * @conf   nchan_buffer1           Uint32_t. Number of channels in all other outpub buffers. For instance, 
 *                                 1024 channels with 10 output buffers. out_buffer0 gets the first 106 channels
 *                                 and the rest buffers get 102 channels each.   
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
    /// Primary loop to wait for buffers, sort beam buffer.
    void main_thread() override;

private:
    ///Fill up the empty frame.
    void fill_empty_frame(uint32_t time_idx, uint32_t freq_idx,
        uint64_t fpga_seq_start0, timespec ctime0, uint32_t beam_number,
        double ra, double dec, double scaling);
    /// make a frequency ID'ed metadata
    void fill_freq_meta(FreqIDBeamMetadata* out_meta, uint32_t freq_id,
        uint64_t fpga_seq_start, timespec ctime, uint64_t stream_id,
        dset_id_t dataset_id, uint32_t beam_number, float ra, float dec,
        uint32_t scaling);
    /// Transfer a non-frequency id'ed metadata to frequency id'ed metadata
    void nonfreq_meta_2_freq_meta(BeamMetadata* nonfreq_meta,
        FreqIDBeamMetadata* freq_meta, uint32_t freq_id);

    /// The input buffer which has one metadata & voltage block per frame.
    struct Buffer* in_buf;
    /// The sort_queue
    std::vector < std::vector<uint8_t*>> sort_queue;
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
    /// Frame size for the queue
    uint32_t queue_frame_size;
    /// Total number of frequency channel
    uint32_t total_freq_chan;
    /// number of polarizations
    uint32_t num_pol;
    /// Use n out put buffer
    uint32_t use_n_out_buffer;
    /// Time resolution for each sample
    double time_resolution;
    /// Sample time resolution in nano seconds
    double time_resolution_nsec;
    /// The time resolution for the queue frame. 
    /// For gated data, It may not match the data number 
    /// in queue frame. This is used for filling up the 
    /// empty frame.
    double queue_frame_resolution_nsec;
    /// The number of data sample that a queue frame represents
    /// The gated date has gaps in between the fime frames
    /// This is used in filling the empty frames. 
    uint32_t queue_frame_represent_size; 
    /// Number of frequencies per output buffer
    /// number of time frames in the sorting queue.
    int wait_nframes;
    /// The size of the FreqIDBeamMeta
    uint32_t FreqIDBeamMeta_size;
    /// Pad n marginal frames from the start
    uint32_t start_marginal_nframe;
    /// subFrame, which includes the frame metadata, size
    /// in the output merged frame. 
    uint32_t sub_frame_size;
    /// Time span for each subframe.
    long subframe_time_nsec;
    /// The number of samples to dump out to next stage
    int dump_size;
    /// flag to make the output start time align to integer second 
    bool align_start_time;
    /// number of channels for the first output buffer
    uint32_t nchan_buffer0;
    /// number of channels for the rest output buffer
    uint32_t nchan_buffer1;
};

#endif //BEAM_BUFFER_SORT_HPP
