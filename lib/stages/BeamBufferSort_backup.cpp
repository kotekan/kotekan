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
    uint32_t nsubframe;
    // Apply config
    samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    has_freq_bin = config.get_default<bool>(unique_name, "has_freq_bin", false);
    total_freq_chan = config.get_default<uint32_t>(unique_name, "total_freq_chan", 1024);
    use_n_out_buffer = config.get_default<uint32_t>(unique_name, "use_n_out_buffer", total_freq_chan);    
    wait_nframes = config.get_default<int>(unique_name, "wait_nframes", 20);
    time_resolution = config.get_default<double>(unique_name, "time_resolution", 2.56e-6);
    FreqIDBeamMeta_size = sizeof(FreqIDBeamMetadata);

    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    // Register the out buffer array
    out_bufs = get_buffer_array("out_bufs");
    for (struct Buffer* out_buf : out_bufs)
        register_producer(out_buf, unique_name.c_str());
    // Check if the request out buffer more than the registered buffers.
    INFO("use_n_out_buffer {:d}, out_bufs {:d}", use_n_out_buffer, out_bufs.size());
    assert(use_n_out_buffer <= out_bufs.size());
    uint32_t sub_frame_size = FreqIDBeamMeta_size + in_buf->frame_size;
    // Init the sort queue time axis
    //sort_queue = (uint8_t*) malloc (sizeof(uint8_t*) * wait_nframes);
    // Init sort queue frequency axis
    //for (uint32_t i = 0; i < wait_nframes; ++i){
    //    sort_queue[i] = (uint8_t*) malloc (sizeof(uint8_t) * total_freq_chan * sub_frame_size);
    //}

    // Assign frequency to the buffers.
    for (uint32_t i = 0; i< use_n_out_buffer; ++i){
        nchan = total_freq_chan / use_n_out_buffer; //Fix 
        out_buf_nchan.push_back(nchan);
	out_buf_frame0.push_back(0);
//	time_offset.push_back(0);
	out_buf_start_chan.push_back(i * nchan);
	nsubframe = nchan * wait_nframes;
	out_buf_status.push_back(std::vector<uint8_t>(nsubframe, 0));
	INFO("SIZE {:d}", out_buf_status.size());
	INFO("sub SIZE {:d}", out_buf_status[i].size());
	frame0_meta.push_back((FreqIDBeamMetadata*)malloc(FreqIDBeamMeta_size));
    }
    
    // Check if there is any extra channels
    uint32_t rmder = total_freq_chan % use_n_out_buffer;
    // Add the reminder channels to the last buffer
    out_buf_nchan[use_n_out_buffer - 1] += rmder;
    std::vector<uint8_t> extra(rmder * wait_nframes, 0);
    out_buf_status[use_n_out_buffer - 1].insert(out_buf_status[use_n_out_buffer - 1].end(), \
        extra.begin(), extra.end());
    
    
}

BeamBufferSort::~BeamBufferSort() {}

void BeamBufferSort::make_freq_meta(FreqIDBeamMetadata* out_meta, uint32_t freq_id,    \ 
    uint64_t fpga_seq_start, timespec ctime, uint64_t stream_id, dset_id_t dataset_id, \
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

void BeamBufferSort::nonfreq_meta_2_freq_meta(BeamMetadata* nonfreq_meta,           \
    FreqIDBeamMetadata* freq_meta, uint32_t freq_id){
    make_freq_meta(freq_meta, freq_id, nonfreq_meta -> fpga_seq_start,              \
        nonfreq_meta-> ctime, nonfreq_meta -> stream_id.id, nonfreq_meta -> dataset_id, \
	nonfreq_meta -> beam_number, nonfreq_meta -> ra, nonfreq_meta -> dec,       \
	nonfreq_meta -> scaling);
}

void BeamBufferSort::fill_metadata(FreqIDBeamMetadata* out_meta, FreqIDBeamMetadata* in_meta, \ 
    bool empty_frame, uint32_t frame0, uint32_t freq_offset, int time_offset, uint32_t freq_id,           \
    uint32_t nchan) {
    // Fill up the metadata. If the frame is empty, fill as zeros.
    uint64_t time_sample_offset = (frame0 + time_offset) * samples_per_data_set;
    memcpy(out_meta, in_meta, FreqIDBeamMeta_size);
    if (empty_frame){
        out_meta -> fpga_seq_start = in_meta -> fpga_seq_start + time_sample_offset;
        out_meta -> ctime = in_meta-> ctime;
	add_nsec(out_meta -> ctime, time_sample_offset * time_resolution);
        out_meta -> stream_id.id = 0;
        out_meta -> dataset_id = dset_id_t::null;
	out_meta -> frequency_bin = freq_id;	
    }
    DEBUG2("fpga set {:d} freq_id {:d}", out_meta -> fpga_seq_start, freq_id);
}

void BeamBufferSort::main_thread(){
    uint32_t nchan;
    uint32_t freq_bin;
    uint32_t fake_freq_bin;     // For metadata in the empty frame
    uint32_t fake_chan_offset;  // For metadata in the empty frame
    uint32_t fake_time_offset;  // For metadata in the empty frame
    uint32_t chan_offset;
    int time_offset;
    uint32_t start_pos;
    uint64_t frame_nr;
    uint16_t buf_idx;
    uint8_t* output = nullptr;
    uint8_t* output2 = nullptr;
    uint32_t fill_pos;
    uint64_t fpga_seq_start;
    uint64_t fpga_seq_start0;
    uint64_t frame0 = 0;
    frameID outbuffer2(out_bufs[0]);

    timespec ctime0;
    
    FreqIDBeamMetadata* in_metadata = nullptr;
    BeamMetadata* non_freq_metadata = nullptr;

    frameID input_frame_id(in_buf); // Input frame id
    std::vector<frameID> output_frame_ids; // Out put frame id array.
    for (auto& out_buf : out_bufs){
        output_frame_ids.push_back(frameID(out_buf));
    }
    
    uint32_t sub_frame_size = FreqIDBeamMeta_size + in_buf->frame_size;
    
    while (!stop_thread) {
        uint8_t* input = wait_for_full_frame(in_buf, unique_name.c_str(), input_frame_id);
	if (input == nullptr)
            break;
        
	// Get the metadata for each frame.
        if (has_freq_bin) {
            FreqIDBeamMetadata* in_metadata = (FreqIDBeamMetadata*)get_metadata(in_buf, input_frame_id);
            freq_bin = in_metadata -> frequency_bin;
            fpga_seq_start = in_metadata -> fpga_seq_start;
        }
        else {
            BeamMetadata* non_freq_metadata = (BeamMetadata*)get_metadata(in_buf, input_frame_id);
            fpga_seq_start = non_freq_metadata -> fpga_seq_start;
            const uint32_t num_freq_per_stream = Telescope::instance().num_freq_per_stream();
            for (uint32_t f = 0; f < num_freq_per_stream; ++f) {
                freq_bin = Telescope::instance().to_freq_id(non_freq_metadata -> stream_id, f);
            }
	    // Build the full Frequency IDed metadata
	    in_metadata = (FreqIDBeamMetadata*)malloc(FreqIDBeamMeta_size);
	    nonfreq_meta_2_freq_meta(non_freq_metadata, in_metadata, freq_bin);
        }

	
	frame_nr = fpga_seq_start / samples_per_data_set;
        
	// Copy the frame to sort queue
	// 2. Check if it is the frist frame, if yes setup the frame0 with marginal nframes.
        //if (frame0 == 0){
	//    frame0 = frame_nr - start_marginal_nframe;
	//    fpga_seq_start0 = frame0 * samples_per_data_set; 
	//}
	// Compute time offset
        time_offset = frame_nr - frame0;
        





	// Since the channels are splited evenly for the first n-1 buffer and the last buffer
	// gets the reminder, we use the first buffer's nchan as a start. If buf_idx > use_n_buf,
	// buf_idx = use_n_buf -1
	buf_idx = freq_bin / out_buf_nchan[0]; 
	if (buf_idx > use_n_out_buffer){
	    buf_idx = use_n_out_buffer - 1;
	}
	// Comput the data copy position
        // 1. find the frequency position in the buffer  	
        chan_offset = freq_bin - out_buf_start_chan.at(buf_idx);
	// 2. Check if it is the frist frame
	if (out_buf_frame0.at(buf_idx) == 0){
	    out_buf_frame0.at(buf_idx) = frame_nr; //TODO need to give buffer.	    
	    // First reading frame, init output frame
	    output = wait_for_empty_frame(out_bufs.at(buf_idx), unique_name.c_str(), output_frame_ids.at(buf_idx));
            if (output == nullptr)
                break; 
	    output2 = wait_for_empty_frame(out_bufs.at(buf_idx), unique_name.c_str(), outbuffer2);
	    if (output2 == nullptr)
                break;

	    // Get the metadata for output buffer
            allocate_new_metadata_object(out_bufs.at(buf_idx), output_frame_ids.at(buf_idx));
            MergedBeamMetadata* out_metadata =
                (MergedBeamMetadata*)get_metadata(out_bufs.at(buf_idx), output_frame_ids.at(buf_idx));
            // Fill the merge metadata
            out_metadata -> sub_frame_pre_frame = nchan * wait_nframes;
            out_metadata -> sub_frame_metadata_size = FreqIDBeamMeta_size;
            out_metadata -> sub_frame_data_size = in_buf -> frame_size;
            out_metadata -> freq_start = out_buf_start_chan.at(buf_idx);
            out_metadata -> nchan = out_buf_nchan.at(buf_idx);
	    memcpy(frame0_meta.at(buf_idx), in_metadata, FreqIDBeamMeta_size);
	}
	// 3. Find time offset
	time_offset = frame_nr - out_buf_frame0.at(buf_idx);
	DEBUG2("First time offset {:d} frame_nr{:d}, frame0 {:d} in_id {:d} freqid {:d}", time_offset, 
	     frame_nr, out_buf_frame0.at(buf_idx), input_frame_id, freq_bin);
	// If the time offset is beyond the wait time frames, mark the frame full.
	// We assume when see a frame after `wait_nframes` limit, the n_frames should be filled.
	if (time_offset >= wait_nframes){
            // Check out frame fill status, if not add zeros. 
	    for (uint32_t i = 0; i < out_buf_status.at(buf_idx).size(); i++){
		if (out_buf_status.at(buf_idx).at(i) == 0){ // sub_Frame is not filled.
		    fill_pos = i * sub_frame_size;
		    fake_time_offset = i / out_buf_nchan.at(buf_idx); 
		    fake_chan_offset = i % out_buf_nchan.at(buf_idx);
		    fake_freq_bin = fake_chan_offset + out_buf_start_chan.at(buf_idx);
		    fill_metadata((FreqIDBeamMetadata*)&output.at(fill_pos), frame0_meta.at(buf_idx), 
				   true, out_buf_frame0.at(buf_idx), fake_chan_offset, fake_time_offset, 
				   fake_freq_bin, out_buf_nchan.at(buf_idx));
		    // Fill data to the empty frame.
		    memset(&output.at(fill_pos + FreqIDBeamMeta_size), 0, in_buf->frame_size);
		}
	    }
            mark_frame_full(out_bufs.at(buf_idx), unique_name.c_str(), output_frame_ids.at(buf_idx)++);
	    output = wait_for_empty_frame(out_bufs.at(buf_idx), unique_name.c_str(), output_frame_ids.at(buf_idx));
            // Get the metadata for output buffer
            allocate_new_metadata_object(out_bufs.at(buf_idx), output_frame_ids.at(buf_idx));
            MergedBeamMetadata* out_metadata =
                (MergedBeamMetadata*)get_metadata(out_bufs.at(buf_idx), output_frame_ids.at(buf_idx));
            // Fill the merge metadata
            out_metadata -> sub_frame_pre_frame = nchan * wait_nframes;
            out_metadata -> sub_frame_metadata_size = FreqIDBeamMeta_size;
            out_metadata -> sub_frame_data_size = in_buf -> frame_size;
            out_metadata -> freq_start = out_buf_start_chan.at(buf_idx);
            out_metadata -> nchan = out_buf_nchan.at(buf_idx);
	    // Reset status vector to zero
	    std::fill(out_buf_status.at(buf_idx).begin(), out_buf_status.at(buf_idx).end(), 0);
            // Reset frame0
	    INFO("Rest frame0");
	    out_buf_frame0.at(buf_idx) += wait_nframes;  // Need to give buffer
	    // Recalculate time offset
	    time_offset = frame_nr - out_buf_frame0.at(buf_idx);
	    // Rest frame0 metadata 
            fpga_seq_start0 = wait_nframes * samples_per_data_set;
	    ctime0 = in_metadata -> ctime;
            add_nsec(ctime0, (long)(-time_offset * samples_per_data_set * time_resolution * 1000000000));
	    make_freq_meta(frame0_meta.at(buf_idx), out_buf_start_chan.at(buf_idx), fpga_seq_start0,
			   ctime0, 0, dset_id_t::null, in_metadata -> beam_number, in_metadata -> ra, 
			   in_metadata -> dec, in_metadata -> scaling);
            time_offset = frame_nr - out_buf_frame0.at(buf_idx);
	}
	// Get the metadata for output buffer
        allocate_new_metadata_object(out_bufs.at(buf_idx), output_frame_ids.at(buf_idx));
	MergedBeamMetadata* out_metadata =
                (MergedBeamMetadata*)get_metadata(out_bufs.at(buf_idx), output_frame_ids.at(buf_idx));
	// Fill the merge metadata
	out_metadata -> sub_frame_pre_frame = nchan * wait_nframes;
	out_metadata -> sub_frame_metadata_size = FreqIDBeamMeta_size;
        out_metadata -> sub_frame_data_size = in_buf -> frame_size;
	out_metadata -> freq_start = out_buf_start_chan.at(buf_idx);
	out_metadata -> nchan = out_buf_nchan.at(buf_idx);
        // Rest frame0 meta
        if (time_offset == 0 && chan_offset == 0){ 
	    // The coming frame is the frame0 for the next merged/out frame
	    // Just copy the metadata to the frame0 meta
	    memcpy(frame0_meta.at(buf_idx), in_metadata, FreqIDBeamMeta_size);
	}

        if (time_offset < 0){ // The incoming frame is too old
            INFO("Drop offset {:d} frame_nr{:d}, frame0 {:d}, freq ID {:d}", time_offset, 
	         frame_nr, out_buf_frame0.at(buf_idx), freq_bin);
	    INFO("Frame {:d} comes too late and will be droped.", frame_nr);
	    mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id);
            input_frame_id++;
	    continue;
	}
        // 3. compute the strat_pos, we first put all the frequencies from same time.
	// Then the frequencies for the next time. 
	//INFO("Compute start pos, time_offset {:d}, chan offset {:d} frame_nr {:d} ", time_offset, chan_offset, frame_nr);
	start_pos = (time_offset * out_buf_nchan.at(buf_idx) + chan_offset) * sub_frame_size;
	FreqIDBeamMetadata* sub_frame_metadata = (FreqIDBeamMetadata*)&output.at(start_pos);
	memcpy(sub_frame_metadata, in_metadata, FreqIDBeamMeta_size);
	// copy data
	uint8_t* sub_frame_data = &output.at(start_pos + FreqIDBeamMeta_size);
	memcpy(sub_frame_data, input, in_buf->frame_size);
	// mark fill status is done
	out_buf_status.at(buf_idx).at(start_pos / sub_frame_size) = 1;
	// Mark in frame empty
	mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id);
        input_frame_id++;
    }	    
}
