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
    // Apply config
    samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    has_freq_bin = config.get_default<bool>(unique_name, "has_freq_bin", false);
    total_freq_chan = config.get_default<uint32_t>(unique_name, "total_freq_chan", 1024);
    use_n_out_buffer = config.get_default<uint32_t>(unique_name, "use_n_out_buffer", total_freq_chan);    
    wait_nframes = config.get_default<int>(unique_name, "wait_nframes", 20);
    time_resolution = config.get_default<double>(unique_name, "time_resolution", 2.56e-6);
    FreqIDBeamMeta_size = sizeof(FreqIDBeamMetadata);
    // Total time for one coming frame in nanoseconds.
    subframe_time_nsec = time_resolution * samples_per_data_set * 1000000000;
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
    sub_frame_size = FreqIDBeamMeta_size + in_buf->frame_size;
    //Init sort queue frequency axis
    for (int i = 0; i < wait_nframes; ++i){
        sort_queue.push_back((uint8_t*) malloc (sizeof(uint8_t) * total_freq_chan * sub_frame_size));
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

void BeamBufferSort::make_freq_meta(FreqIDBeamMetadata* out_meta, uint32_t freq_id,\ 
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
    make_freq_meta(freq_meta, freq_id, nonfreq_meta -> fpga_seq_start,\
        nonfreq_meta-> ctime, nonfreq_meta -> stream_id.id, nonfreq_meta -> dataset_id,\
	nonfreq_meta -> beam_number, nonfreq_meta -> ra, nonfreq_meta -> dec,\
	nonfreq_meta -> scaling);
}

void BeamBufferSort::fill_empty_frame(uint32_t time_idx, uint32_t freq_idx, 
    uint64_t fpga_seq_start0, timespec ctime0, uint32_t beam_number, double ra, 
    double dec, double scaling){
    
    uint32_t fill_pos;
    uint64_t fake_fpga_seq_start;
    uint32_t fake_freq_bin;     // For metadata in the empty frame
    timespec fake_ctime;

    fill_pos = freq_idx * sub_frame_size;
    fake_fpga_seq_start = fpga_seq_start0 + time_idx;
    fake_ctime = ctime0;
    add_nsec(fake_ctime, (long)(time_idx * subframe_time_nsec));
    fake_freq_bin = freq_idx;
    make_freq_meta((FreqIDBeamMetadata*) &sort_queue[time_idx][fill_pos], fake_freq_bin,
        fake_fpga_seq_start, fake_ctime, 0, dset_id_t::null, beam_number, ra, dec, scaling);
    memset(&sort_queue[time_idx][fill_pos + FreqIDBeamMeta_size], 0, samples_per_data_set);
    queue_status[time_idx][freq_idx] = 1;
}

void BeamBufferSort::main_thread(){
    uint32_t freq_bin;
    int time_offset;
    uint32_t start_pos;
    uint64_t frame_nr;
    uint8_t* output;
    uint32_t dump_nframe;
    uint32_t dump_start;
    uint32_t dump_size;
    uint64_t fpga_seq_start;
    uint64_t fpga_seq_start0;
    uint64_t frame0 = 0;
    uint32_t beam_number;
    double ra;
    double dec;
    double scaling;

    timespec ctime0;
    
    FreqIDBeamMetadata* in_metadata = nullptr;
    BeamMetadata* non_freq_metadata = nullptr;

    frameID input_frame_id(in_buf); // Input frame id
    std::vector<frameID> output_frame_ids; 
    for (auto& out_buf : out_bufs){
        output_frame_ids.push_back(frameID(out_buf));
    }
    
    while (!stop_thread) {
	// Receive a new frame
        uint8_t* input = wait_for_full_frame(in_buf, unique_name.c_str(), input_frame_id);
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
	    in_metadata = (FreqIDBeamMetadata*)malloc(FreqIDBeamMeta_size);
	    nonfreq_meta_2_freq_meta(non_freq_metadata, in_metadata, freq_bin);
        }

	// Get incoming frame number
	frame_nr = fpga_seq_start / samples_per_data_set;
        ctime0 = in_metadata -> ctime;
	/// Copy the frame to sort queue
	
	// 1. Check if it is the frist frame, if yes setup the frame0 with marginal nframes.
        if (frame0 == 0){
	    frame0 = frame_nr - start_marginal_nframe;
	    fpga_seq_start0 = frame0 * samples_per_data_set;
	    add_nsec(ctime0, (long)(-start_marginal_nframe * subframe_time_nsec));
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
            dump_nframe = time_offset - wait_nframes + 1;
	    INFO("dump_nframe {:d}", dump_nframe);
            for (uint32_t i = 0; i < dump_nframe; i++){ // Get how many frames need to be dumped
	        // Check queue fill stauts, if not, fill frames as zero
		for (uint32_t j = 0; j < queue_status[i].size(); j++){
		    if (queue_status[i][j] == 0){
                        fill_empty_frame(i, j, fpga_seq_start0, ctime0, beam_number, ra, dec, scaling);
		    }
		}
		// Finishing filling empty frame , put frame to the out buffer
		// Since we use nbuffers, dump all nbuffer.
		for (uint32_t k=0; k < out_bufs.size(); k++){
		// Compute the start dump for the buffer
		// TODO maybe dump more time??
		    dump_start = out_buf_start_chan[k] * sub_frame_size;
		    dump_size = out_buf_nchan[k] * sub_frame_size;
		    INFO("Dump start {:d} dump_size {:d}", dump_start, dump_size);
		    output = wait_for_empty_frame(out_bufs.at(k), unique_name.c_str(), output_frame_ids.at(k));
		    if (output == nullptr)
	                break;
		    allocate_new_metadata_object(out_bufs.at(k), output_frame_ids.at(k));
                    MergedBeamMetadata* out_metadata =
                        (MergedBeamMetadata*)get_metadata(out_bufs.at(k), output_frame_ids.at(k));
		    // Fill the merge metadata
                    out_metadata -> sub_frame_pre_frame = out_buf_nchan[k];
                    out_metadata -> sub_frame_metadata_size = FreqIDBeamMeta_size;
                    out_metadata -> sub_frame_data_size = in_buf -> frame_size;
                    out_metadata -> freq_start = out_buf_start_chan[k];
                    out_metadata -> nchan = out_buf_nchan[k];
		    // Copy data to the out frame
                    memcpy(output, &sort_queue[i][dump_start], dump_size);
		    // Debug line
//                    FreqIDBeamMetadata* debug_meta = (FreqIDBeamMetadata*)&output[0];
//                    INFO("Dump Debug RA {:f} Dec: {:f}, scaling: {:d}, beam_num: {:d}, freq_id {:d}, seq_start {:d}\n",
//                         debug_meta->ra, debug_meta->dec, debug_meta->scaling,
//                         debug_meta->beam_number, debug_meta->frequency_bin, debug_meta->fpga_seq_start);

		    // Mark out frame full
		    mark_frame_full(out_bufs.at(k), unique_name.c_str(), output_frame_ids.at(k)++);
		}
		// Finish dump one time
		// Reset queue status
		std::fill(queue_status[i].begin(), queue_status[i].end(), 0);
		// shift frame0 one time sample down. 
		// Rotate the sort queue
                std::rotate(sort_queue.begin(), sort_queue.begin() + 1, sort_queue.end());
		// Reset the queue status of the first time idx to zero
                std::fill(queue_status[0].begin(), queue_status[0].end(), 0);
		// Rotate the queue status
		std::rotate(queue_status.begin(), queue_status.begin() + 1, queue_status.end());
                frame0 ++;
                fpga_seq_start0 = frame0 * samples_per_data_set;
                add_nsec(ctime0, (long)(subframe_time_nsec));
            
                // New time offset
                time_offset = frame_nr - frame0;
	    }
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
	start_pos = freq_bin * sub_frame_size;
	FreqIDBeamMetadata* sub_frame_metadata = (FreqIDBeamMetadata*)&sort_queue[time_offset][start_pos];
	memcpy(sub_frame_metadata, in_metadata, FreqIDBeamMeta_size);
	// copy data
	uint8_t* sub_frame_data = &sort_queue[time_offset][start_pos + FreqIDBeamMeta_size];
	memcpy(sub_frame_data, input, in_buf->frame_size);
        // Debug line
//	FreqIDBeamMetadata* debug_meta = (FreqIDBeamMetadata*)&sort_queue[time_offset][start_pos];
//        INFO("Debug RA {:f} Dec: {:f}, scaling: {:d}, beam_num: {:d}, freq_id {:d}, seq_start {:d}\n", 
//	       debug_meta->ra, debug_meta->dec, debug_meta->scaling,
//               debug_meta->beam_number, debug_meta->frequency_bin, debug_meta->fpga_seq_start);	

	// mark fill status is done
	queue_status[time_offset][freq_bin] = 1;
	// Mark in frame empty
	mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id);
        input_frame_id++;
    }
    //TODO add dump rest    

}
