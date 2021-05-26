#include "BeamBufferSort.hpp"

#include "BeamMetadata.hpp"   // for BeamMetadata
#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"           // for allocate_new_metadata_object, copy_metadata, mark_frame_empty
#include "chimeMetadata.hpp"  // for atomic_add_lost_timesamples, zero_lost_samples
#include "kotekanLogging.hpp" // for INFO
#include "tx_utils.hpp"       // for add_nsec

#include <stdlib.h>
#include <atomic>      // for atomic_bool
#include <exception>   // for exception
#include <functional>  // for _Bind_helper<>::type, bind, function
#include <iostream>    // for cout
#include <regex>       // for match_results<>::_Base_type
#include <stdexcept>   // for runtime_error
#include <string>      // for string
#include <visUtil.hpp> // for frameID, modulo

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using std::string;

REGISTER_KOTEKAN_STAGE(BeamBufferSort);

BeamBufferSort::BeamBufferSort(Config& config_, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container, std::bind(&BeamBufferSort::main_thread, this)) {
    uint32_t nchan;
    uint32_t buf_start_f = 0;
    uint32_t buf_nchan;
    uint32_t queue_real_size;
    // Apply config
    samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    has_freq_bin = config.get_default<bool>(unique_name, "has_freq_bin", false);
    total_freq_chan = config.get_default<uint32_t>(unique_name, "total_freq_chan", 1024);
    num_pol = config.get_default<uint32_t>(unique_name, "num_pol", 2);
    use_n_out_buffer = config.get_default<uint32_t>(unique_name, "use_n_out_buffer", total_freq_chan);    
    wait_nframes = config.get_default<int>(unique_name, "wait_nframes", 20);
    time_resolution = config.get_default<double>(unique_name, "time_resolution", 2.56e-6); 
    queue_frame_size =  config.get<uint32_t>(unique_name, "queue_frame_size");
    dump_size = config.get<int>(unique_name, "dump_size");
    start_marginal_nframe = config.get_default<uint32_t>(unique_name, "start_marginal_nframe", 3);
    // The data dump size should be smaller than one third of queue time samples.
    assert(dump_size < wait_nframes / 3 * (int)queue_frame_size);
    FreqIDBeamMeta_size = sizeof(FreqIDBeamMetadata);
    // Total time for one coming frame in nanoseconds.
    subframe_time_nsec = time_resolution * samples_per_data_set * 1e9;
    // Queue frame time resolution
    // TODO this will be changed. 
    queue_frame_resolution_nsec = subframe_time_nsec;
    time_resolution_nsec = time_resolution * 1e9;
    // Number of data samples a queue frame represents.
    // TODO this will be changed.
    queue_frame_represent_size = samples_per_data_set; 
    // Time delay vector for each channel
    std::vector<double> time_delay(total_freq_chan, 0.0); // need to change here
    // Set up buffers
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    // Register the out buffer array
    out_bufs = get_buffer_array("out_bufs");
    for (struct Buffer* out_buf : out_bufs)
        register_producer(out_buf, unique_name.c_str());
    // Check if the request out buffer more than the registered buffers.
    assert(use_n_out_buffer <= out_bufs.size());
    
    // Init sort queue
    queue_real_size = FreqIDBeamMeta_size + queue_frame_size * num_pol;
    //Init sort queue frequency axis
    for (int i = 0; i < wait_nframes; ++i){
	// Build the sort queue as the 2D array pointer
	sort_queue.push_back(std::vector<uint8_t *> (total_freq_chan));
	for (uint32_t j = 0; j < total_freq_chan; j++){
	    sort_queue[i][j] = (uint8_t*)  malloc (sizeof(uint8_t) * queue_real_size);
	}
	queue_status.push_back(std::vector<uint8_t> (total_freq_chan, 0));
    }

    // Assign frequency to the buffers.
    nchan = total_freq_chan / use_n_out_buffer;
    // Check if there is any extra channels
    uint32_t rmder = total_freq_chan % use_n_out_buffer;
    // Assign channels to the output buffer.
    for (uint32_t i = 0; i< use_n_out_buffer; ++i){
        if (i == 0){
	    // First buffer gets more frequency chan
	    buf_nchan = nchan + rmder;
	}
	else{
	    buf_nchan = nchan;
	}
        out_buf_nchan.push_back(buf_nchan);
	out_buf_frame0.push_back(0);
	out_buf_start_chan.push_back(buf_start_f);
	buf_start_f += buf_nchan;
    }
    
}

BeamBufferSort::~BeamBufferSort() {}

void BeamBufferSort::fill_freq_meta(FreqIDBeamMetadata* out_meta, uint32_t freq_id,\ 
    uint64_t fpga_seq_start, timespec ctime, uint64_t stream_id, dset_id_t dataset_id,\
    uint32_t beam_number, float ra, float dec, uint32_t scaling){
    out_meta -> fpga_seq_start = fpga_seq_start;
    out_meta -> ctime = ctime;
    out_meta -> stream_id.id = stream_id;
    out_meta -> dataset_id = dataset_id;
    out_meta -> beam_number = beam_number;
    out_meta -> ra = ra;
    out_meta -> dec = dec;
    out_meta -> scaling = scaling;
    out_meta -> frequency_bin = freq_id;
}

void BeamBufferSort::nonfreq_meta_2_freq_meta(BeamMetadata* nonfreq_meta,\
    FreqIDBeamMetadata* freq_meta, uint32_t freq_id){
    fill_freq_meta(freq_meta, freq_id, nonfreq_meta -> fpga_seq_start,\
        nonfreq_meta-> ctime, nonfreq_meta -> stream_id.id, nonfreq_meta -> dataset_id,\
	nonfreq_meta -> beam_number, nonfreq_meta -> ra, nonfreq_meta -> dec,\
	nonfreq_meta -> scaling);
}

void BeamBufferSort::fill_empty_frame(uint32_t time_idx, uint32_t freq_idx, 
    uint64_t fpga_seq_start0, timespec ctime0, uint32_t beam_number, double ra, 
    double dec, double scaling){
    
    uint64_t fake_fpga_seq_start;
    uint32_t fake_freq_bin;     // For metadata in the empty frame
    timespec fake_ctime;

    fake_fpga_seq_start = fpga_seq_start0 + time_idx *  queue_frame_represent_size;
    fake_ctime = ctime0;
    //INFO("Filling empty time {:d}", fake_ctime.tv_sec);
    add_nsec(fake_ctime, (long)(time_idx * queue_frame_resolution_nsec));
    //INFO("Filling empty time2 {:d}", fake_ctime.tv_sec);
    fake_freq_bin = freq_idx;
    fill_freq_meta((FreqIDBeamMetadata*) &sort_queue[time_idx][freq_idx][0], fake_freq_bin,
        fake_fpga_seq_start, fake_ctime, 0, dset_id_t::null, beam_number, ra, dec, scaling);
    memset(&sort_queue[time_idx][freq_idx][FreqIDBeamMeta_size], 0, queue_frame_size * num_pol);
    queue_status[time_idx][freq_idx] = 1;
	    
}


void BeamBufferSort::main_thread(){
    uint32_t freq_bin;
    int time_offset;
    uint64_t frame_nr;
    uint8_t* output = nullptr;
    uint8_t* input = nullptr;
    uint32_t advance_nframe;
    uint32_t copy_size;
    uint64_t fpga_seq_start;
    uint64_t fpga_seq_start0;
    uint64_t fpga_seq_start_at_dump; 
    uint64_t frame0 = 0;
    uint32_t beam_number;
    uint32_t leftover_in_frame;
    uint32_t empty_out_buff;
    uint32_t dump_frame_count;
    double ra;
    double dec;
    double scaling;
    uint32_t in_frame_dump_offset = 0 ;  // Dump Offset in the incoming frame 
    uint32_t output_buff_offset = 0; 
    uint32_t src_start;
    uint32_t dest_start;

    // The vector for queue start metadata
    std::vector<FreqIDBeamMetadata*> curr_queue0_metadata;
    for (uint32_t i = 0; i < total_freq_chan; i++)
	curr_queue0_metadata.push_back(nullptr);


    timespec ctime0;
    timespec ctime_at_dump;

    FreqIDBeamMetadata* in_metadata = nullptr;
    BeamMetadata* non_freq_metadata = nullptr;

    frameID input_frame_id(in_buf); // Input frame id
    std::vector<frameID> output_frame_ids; 
    for (auto& out_buf : out_bufs){
        output_frame_ids.push_back(frameID(out_buf));
    }
    
    FreqIDBeamMetadata * temp_freq_metadata = (FreqIDBeamMetadata*)malloc(FreqIDBeamMeta_size);

    while (!stop_thread) {
	// Receive a new frame
        input = wait_for_full_frame(in_buf, unique_name.c_str(), input_frame_id);
	if (input == nullptr)
            break;
        
	// Get the metadata for each frame.
        if (has_freq_bin) {
            // Frequency ID is in the metdata.
            in_metadata = (FreqIDBeamMetadata*)get_metadata(in_buf, input_frame_id);
            freq_bin = in_metadata -> frequency_bin;
            fpga_seq_start = in_metadata -> fpga_seq_start;
        }
        else {
	    // Frequency ID is not in the metadata. will add it to the metadata.
            non_freq_metadata = (BeamMetadata*)get_metadata(in_buf, input_frame_id);
            fpga_seq_start = non_freq_metadata -> fpga_seq_start;
            const uint32_t num_freq_per_stream = Telescope::instance().num_freq_per_stream();
            for (uint32_t f = 0; f < num_freq_per_stream; ++f) {
                freq_bin = Telescope::instance().to_freq_id(non_freq_metadata -> stream_id, f);
            }
	    // Build the full Frequency IDed metadata
	    in_metadata = temp_freq_metadata;
	    nonfreq_meta_2_freq_meta(non_freq_metadata, in_metadata, freq_bin);
        }
	// INFO("in come start time {:d} {:d}", in_metadata->ctime.tv_sec, in_metadata->ctime.tv_nsec);

	// Get incoming frame number
	// TODO when it is gated data, this needs to be changed. 
	frame_nr = fpga_seq_start / samples_per_data_set;
	// 1. Check if it is the frist frame, if yes setup the frame0 with marginal nframes.
        if (frame0 == 0){
	    frame0 = frame_nr - start_marginal_nframe;
	    fpga_seq_start0 = frame0 * samples_per_data_set;
	    ctime0 = in_metadata -> ctime;
	    add_nsec(ctime0, -((long)(start_marginal_nframe * subframe_time_nsec)));
	    // for tracking beam, RA DEC Beam and scaling should be the same
	    beam_number = in_metadata -> beam_number;
	    ra = in_metadata -> ra;
	    dec = in_metadata -> dec;
            scaling = in_metadata -> scaling;
	}

	// Compute time offset
        time_offset = frame_nr - frame0;

        // Check if time offset is outside of waiting frames, if yes dump the old frames.
	if (time_offset >= wait_nframes){ // Frame comes beyond the waiting frames, dump old frames.
	    // Decide the number of frames to dump.
	    // dump offset is the position where data has been dump in the frame.
            advance_nframe = time_offset - wait_nframes + 1;
            // Check queue fill stauts, if not, fill frames as zero, since the dump may go beyond 
            // the advance frames, we check one more frame than the advance frame numbers.
            for(uint32_t i = 0; i < advance_nframe + 1; i++){
                for (uint32_t j = 0; j < queue_status[i].size(); j++){
                    if (queue_status[i][j] == 0){
                        fill_empty_frame(i, j, fpga_seq_start0, ctime0, beam_number, ra, dec, scaling);
                    }
                }
            }
	    //INFO("Start dumping advance {:d}", advance_nframe);
            // Get the queue start matedata for each frequency.
            for (uint32_t i = 0; i < total_freq_chan; i++){
                curr_queue0_metadata[i] = (FreqIDBeamMetadata*)&sort_queue[0][i][0];
	    }
	    // Counter for dumped frames
	    dump_frame_count = 0;
	    // time offset in the output buffer
            //INFO("dump frame {:d} {:d}", dump_frame_count, advance_nframe); 
	    //INFO("outbuffoffset {:d}", dump_size);
	    while (dump_frame_count < advance_nframe + 1){
		// in frame dump offset is in the end of queue frame, rotate the queue.
		if (in_frame_dump_offset >= queue_frame_size){
	            // Count the dump frame
		    dump_frame_count++;
	            // Rotate the queue
                    // Reset queue status
                    std::fill(queue_status[0].begin(), queue_status[0].end(), 0);
                    // shift frame0 one time sample down. 
                    // Rotate the sort queue
                    std::rotate(sort_queue.begin(), sort_queue.begin() + 1, sort_queue.end());
                    // Rotate the queue status
                    std::rotate(queue_status.begin(), queue_status.begin() + 1, queue_status.end());
                    frame0 ++;
                    fpga_seq_start0 = frame0 * samples_per_data_set;
		    // rotate the time as well.
                    add_nsec(ctime0, (long)(queue_frame_size * time_resolution_nsec));
                    // Rest the in frame dump offset
		    in_frame_dump_offset = 0;
		    // Get the queue start matedata for each frequency.
		    for (uint32_t i = 0; i < total_freq_chan; i++){
		        curr_queue0_metadata[i] = (FreqIDBeamMetadata*)&sort_queue[0][i][0];
		    }
		} 
	        // Check the queue dump status. In other words, how many data points left 
                // in the frame from last dump.
                // How many data samples can be dumpe in queue time frame0.
                leftover_in_frame = queue_frame_size - in_frame_dump_offset;
		//INFO("left_over_in_frame {:d} {:d} {:d}", leftover_in_frame, queue_frame_size, in_frame_dump_offset);
		//INFO("in frame offset {:d}, out frame offset {:d}", in_frame_dump_offset, output_buff_offset);
		empty_out_buff = dump_size - output_buff_offset;
		//INFO("LEFT {:d} EMPTY {:d}", leftover_in_frame, empty_out_buff);
                
		// Check out put buffer status
		// if output buffer offset gets to the end of output buffer.
                if (empty_out_buff == 0){
		    // reste output buffer offset
		    output_buff_offset = 0;
		    empty_out_buff = dump_size - output_buff_offset;
		}
                // indentified the copy size and make sure there are enough space and 
		// enough data.
		if (empty_out_buff > leftover_in_frame)
		    copy_size = leftover_in_frame;
		else
                    copy_size = empty_out_buff;
                // Since we use nbuffers, dump all nbuffer.
                for (uint32_t k=0; k < out_bufs.size(); k++){
                    // When output buffer offset is zero dump old frame get new frames.
		    if (output_buff_offset == 0){
			// dump the filled output buffer if it is full.
			if (output != nullptr)
			    mark_frame_full(out_bufs.at(k), unique_name.c_str(), output_frame_ids.at(k)++);
			// Get a new output buffer frame
		        output = wait_for_empty_frame(out_bufs.at(k), unique_name.c_str(), output_frame_ids.at(k));
                        if (output == nullptr)
                            break;
                        allocate_new_metadata_object(out_bufs.at(k), output_frame_ids.at(k));
                        MergedBeamMetadata* out_metadata =
                            (MergedBeamMetadata*)get_metadata(out_bufs.at(k), output_frame_ids.at(k));
		    
		        // Fill the merge metadata
                        out_metadata -> sub_frame_pre_frame = out_buf_nchan[k];
                        out_metadata -> sub_frame_metadata_size = FreqIDBeamMeta_size;
                        out_metadata -> sub_frame_data_size = dump_size;
                        out_metadata -> freq_start = out_buf_start_chan[k];
                        out_metadata -> nchan = out_buf_nchan[k];
			// Start fpga seq in the top of frequency band
                        out_metadata -> fpga_seq_start = curr_queue0_metadata[out_buf_start_chan[k]] -> fpga_seq_start + in_frame_dump_offset;
                        out_metadata -> ctime = curr_queue0_metadata[out_buf_start_chan[k]] -> ctime;
			add_nsec(out_metadata -> ctime, (long)(in_frame_dump_offset * time_resolution_nsec));
                        // TODO If we want to dump more than one time frame, this needs to be changed.
                        out_metadata -> nframe = 1;
                        // Compute the start dump for the buffer in the frequency channels
                        // The dump size is not a full frame, we have to select the right 
		    }

                    // copy data for each channel.
                    for (uint32_t ch_id=out_buf_start_chan[k]; ch_id < out_buf_nchan[k] + out_buf_start_chan[k]; ch_id++){
			// Get the metadata for each single input frame
			dest_start = (ch_id - out_buf_start_chan[k]) * (dump_size * num_pol + FreqIDBeamMeta_size);
		        if (output_buff_offset == 0){
                            FreqIDBeamMetadata* dump_frame_metadata = (FreqIDBeamMetadata*)&sort_queue[0][ch_id][0];
		            // Write output meta	
                            FreqIDBeamMetadata* sub_frame_metadata = (FreqIDBeamMetadata*)&output[dest_start];
                            // Compute dump data fpga_seq_start and ctime
                            fpga_seq_start_at_dump = dump_frame_metadata -> fpga_seq_start + in_frame_dump_offset;
                            ctime_at_dump = dump_frame_metadata -> ctime;
                            add_nsec(ctime_at_dump, (long)(in_frame_dump_offset * time_resolution_nsec));
			    //INFO("FPAG_start {:d}, freq {:d} in frame {:d} out frame {:d} RA {:f} dec {:f}", fpga_seq_start_at_dump, ch_id, in_frame_dump_offset, output_buff_offset, dump_frame_metadata-> ra, dump_frame_metadata -> dec);
                            //INFO("C time int {:d} nsec {:d} frame start {:d} time jump {:d}", ctime_at_dump.tv_sec, ctime_at_dump.tv_nsec, dump_frame_metadata -> ctime.tv_sec, (long)(in_frame_dump_offset * time_resolution_nsec));
                            fill_freq_meta(sub_frame_metadata, ch_id, fpga_seq_start_at_dump, ctime_at_dump, dump_frame_metadata -> stream_id.id,\
                                           dump_frame_metadata -> dataset_id, dump_frame_metadata -> beam_number, dump_frame_metadata -> ra, \
				           dump_frame_metadata -> dec, dump_frame_metadata -> scaling);
			}
			
                        // Copy data to out buffer
			src_start = FreqIDBeamMeta_size + in_frame_dump_offset * num_pol;
                        memcpy(&output[dest_start + FreqIDBeamMeta_size + output_buff_offset * num_pol], \
			       &sort_queue[0][ch_id][src_start], copy_size * num_pol);	
	            }
                    
		}
		//INFO("Old offset in {:d} out {:d} copy {:d}", in_frame_dump_offset, output_buff_offset, copy_size);
		in_frame_dump_offset += copy_size;
		output_buff_offset += copy_size;
		//INFO("Updated out offset {:d}", output_buff_offset);

	    }
	    // Update time offset
	    time_offset = frame_nr - frame0;
	}
            
	// Check if the time offset too old
	if (time_offset < 0){ // The incoming frame is too old
            INFO("Drop offset {:d} frame_nr{:d}, frame0 {:d}, freq ID {:d}", time_offset,
                 frame_nr, frame0, freq_bin);
            INFO("Frame {:d} comes too late and will be droped.", frame_nr);
            mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id);
            input_frame_id++;
            continue;
        }
	// Copy in frame to the queue.
        //INFO("In queue {:d} {:d}", time_offset, freq_bin);
	FreqIDBeamMetadata* sub_frame_metadata = (FreqIDBeamMetadata*)&sort_queue[time_offset][freq_bin][0];
	memcpy(sub_frame_metadata, in_metadata, FreqIDBeamMeta_size);
	// copy data
	uint8_t* sub_frame_data = &sort_queue[time_offset][freq_bin][FreqIDBeamMeta_size];
	memcpy(sub_frame_data, input, in_buf->frame_size);
        // Debug line
	FreqIDBeamMetadata* debug_meta = (FreqIDBeamMetadata*)&sort_queue[time_offset][freq_bin][0];
        INFO("Debug RA {:f} Dec: {:f}, scaling: {:d}, beam_num: {:d}, freq_id {:d}, seq_start {:d} stream_id {:d} c_time {:d}\n", 
             debug_meta->ra, debug_meta->dec, debug_meta->scaling,
             debug_meta->beam_number, debug_meta->frequency_bin, debug_meta->fpga_seq_start, 
             debug_meta->stream_id.id, debug_meta -> ctime.tv_sec);	

	// mark fill status is done
	queue_status[time_offset][freq_bin] = 1;
	// Mark in frame empty
	mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id);
        input_frame_id++;
    }
    // Free memory
    // Free queue
    for (int i = 0; i < wait_nframes; ++i){
        for (uint32_t j = 0; j < total_freq_chan; j++){
            free(sort_queue[i][j]);
        }
    // Free temp freqid metadata
    free(temp_freq_metadata);
    }

}
