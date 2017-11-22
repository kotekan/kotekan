#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <functional>
#include <string>

using std::string;

// TODO Where do these live?
# define likely(x)      __builtin_expect(!!(x), 1)
# define unlikely(x)    __builtin_expect(!!(x), 0)

#include "buffer.h"
#include "frbPostProcess.hpp"
#include "Config.hpp"
#include "util.h"
#include "errors.h"
#include "frb_functions.h" 
#include "chimeMetadata.h"
#include "fpga_header_functions.h"

frbPostProcess::frbPostProcess(Config& config_,
        const string& unique_name,
        bufferContainer &buffer_container) :
        KotekanProcess(config_, unique_name, buffer_container,
                       std::bind(&frbPostProcess::main_thread, this)){

    apply_config(0);

    in_buf = (struct Buffer **)malloc(_num_gpus * sizeof (struct Buffer *));
    for (int i = 0; i < _num_gpus; ++i) {
        in_buf[i] = get_buffer("network_input_buffer_" + std::to_string(i));
        register_consumer(in_buf[i], unique_name.c_str());
    }
    frb_buf = get_buffer("frb_out_buf");
    register_producer(frb_buf, unique_name.c_str());

    //Dynamic header
    frb_header_beam_ids = new uint16_t[_nbeams];
    frb_header_coarse_freq_ids = new uint16_t[_nfreq_coarse];
    frb_header_scale = new float[_nbeams * _nfreq_coarse];
    frb_header_offset = new float[_nbeams * _nfreq_coarse];

}

frbPostProcess::~frbPostProcess() {
    free(in_buf);
    free(frb_header_beam_ids);
    free(frb_header_coarse_freq_ids);
    free(frb_header_scale);
    free(frb_header_offset);

}

void frbPostProcess::fill_headers(unsigned char * out_buf,
                  struct FRBHeader * frb_header,
                  const uint64_t fpga_seq_num,
		  const uint16_t num_L1_streams, //256
		  uint16_t * frb_header_coarse_freq_ids,
		  float * frb_header_scale,
		  float * frb_header_offset){  
    // Populate the headers
    //assert(sizeof(struct FRBHeader) == _udp_header_size);

    for (int j=0;j<num_L1_streams; ++j) { //256 streams
        for (int k=0;k<_nbeams;++k) { //the 4 beams are supposed to be consecutive
	    frb_header_beam_ids[k] = j*_nbeams+k;
	}
	for (int i = 0; i < 8; ++i) {  //8 frames in a stream
	    frb_header->fpga_count = fpga_seq_num + 16*_fpga_counts_per_sample * i;
	    memcpy(&out_buf[j*8*_udp_packet_size + i*_udp_packet_size], frb_header, sizeof(struct FRBHeader));

	    int static_header_size = 24;
	    //copy in the dynamic header
	    memcpy(&out_buf[j*8*_udp_packet_size + i*_udp_packet_size + static_header_size], 
		   frb_header_beam_ids, sizeof(uint16_t)*_nbeams);
	    memcpy(&out_buf[j*8*_udp_packet_size + i*_udp_packet_size + static_header_size + 2*_nbeams], 
		   frb_header_coarse_freq_ids, sizeof(uint16_t)*_nfreq_coarse);
	    memcpy(&out_buf[j*8*_udp_packet_size + i*_udp_packet_size + static_header_size + 2*_nbeams + 2*_nfreq_coarse], 
		   frb_header_scale, sizeof(float)*_nbeams*_nfreq_coarse);
	    memcpy(&out_buf[j*8*_udp_packet_size + i*_udp_packet_size + static_header_size + 2*_nbeams + 2*_nfreq_coarse + 4*_nbeams*_nfreq_coarse], 
		   frb_header_offset, sizeof(float)*_nbeams*_nfreq_coarse);
	}
    }
}

void frbPostProcess::apply_config(uint64_t fpga_seq) {
    if (!config.update_needed(fpga_seq))
        return;

    _num_gpus = config.get_int(unique_name, "num_gpus");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");
    _nfreq_coarse = config.get_int(unique_name, "num_gpus"); //4
    _downsample_time = config.get_int(unique_name, "downsample_time");
    _factor_upchan = config.get_int(unique_name, "factor_upchan");
    _factor_upchan_out = config.get_int(unique_name, "factor_upchan_out"); 
    _nbeams = config.get_int(unique_name, "num_beams");
    _timesamples_per_frb_packet = config.get_int(unique_name, "timesamples_per_frb_packet");
    _udp_packet_size = config.get_int(unique_name, "udp_packet_size");
    _udp_header_size = config.get_int(unique_name, "udp_header_size");
    _freq_array = config.get_int_array(unique_name, "freq_array");

    _fpga_counts_per_sample = _downsample_time * _factor_upchan;
      
}

void frbPostProcess::main_thread() {

    uint in_buffer_ID[_num_gpus] ;   //4 of these , cycle through buffer depth
    uint8_t * in_frame[_num_gpus];
    int out_buffer_ID = 0;  
    int startup = 1; //related to the likely & unlikely

    for (int i = 0; i < _num_gpus; ++i) {
        in_buffer_ID[i] = 0;
    }
    const uint32_t num_samples = _samples_per_data_set / _downsample_time / _factor_upchan ; //It is 100 for now, but should be 128.
    uint32_t current_input_location = 0; //goes from 0 to num_samples

    struct FRBHeader frb_header;
    frb_header.protocol_version = 1;     
    frb_header.data_nbytes =  _udp_packet_size - _udp_header_size;
    frb_header.fpga_counts_per_sample =  _fpga_counts_per_sample;
    frb_header.fpga_count = 0 ;  //to be updated in fill_header
    frb_header.nbeams = _nbeams;  //4
    frb_header.nfreq_coarse = _nfreq_coarse; //4
    frb_header.nupfreq = _factor_upchan_out;
    frb_header.ntsamp = _timesamples_per_frb_packet;

    for (int ii=0;ii<_nbeams;++ii){
        frb_header_beam_ids[ii] = 7; //To be overwritten in fill_header
    }
    for (int ii=0;ii<_nfreq_coarse;++ii){
        frb_header_coarse_freq_ids[ii] = 0;;//_freq_array[ii] 
    }
    for (int ii =0; ii<_nbeams * _nfreq_coarse;++ii){
        frb_header_scale[ii] = 1.; 
	frb_header_offset[ii] = 0;
    }

    int frame = 0;
    int in_frame_location = 0; //goes from 0 to 16
    uint64_t fpga_seq_num = 0;

    int num_L1_streams = 1024/_nbeams;

    // Get the first output buffer which will always be id = 0 to start.
    uint8_t * out_frame = wait_for_empty_frame(frb_buf, unique_name.c_str(), out_buffer_ID);
    if (out_frame == NULL) goto end_loop;

    while(!stop_thread) {
        // Get an input buffer, This call is blocking!
        for (int i = 0; i < _num_gpus; ++i) {
	    in_frame[i] = wait_for_full_frame(in_buf[i], unique_name.c_str(), in_buffer_ID[i]);
	    if (in_frame[i] == NULL) goto end_loop;
	    //INFO("GPU Post process got full buffer ID %d for GPU %d", in_buffer_ID[i],i);
	}
        //INFO("frb_post_process; got full set of GPU output buffers");

        uint64_t first_seq_number = get_fpga_seq_num(in_buf[0], in_buffer_ID[0]);

        for (int i = 0; i < _num_gpus; ++i) {
	    assert(first_seq_number ==
		   (uint64_t)get_fpga_seq_num(in_buf[i], in_buffer_ID[i]));

	    stream_id_t stream_id = get_stream_id_t(in_buf[i], in_buffer_ID[i]);
	    frb_header_coarse_freq_ids[i] = get_stream_id(in_buf[i], in_buffer_ID[i]);
	    //This get the freq in MHz but we don't seem to need it
	    //float freq_now = freq_from_bin(bin_number_chime(&stream_id));
	}

        // If this is the first time wait until we get the start of an interger second period.
        if (unlikely(startup == 1)) {
	    // testing sync code
            startup = 0;
            current_input_location = 0;

            // Fill the first output buffer headers
	    fpga_seq_num = first_seq_number;
	    fill_headers((unsigned char*)out_frame,
			 &frb_header,
			 first_seq_number,
			 num_L1_streams,
			 (uint16_t*)frb_header_coarse_freq_ids,
			 (float*)frb_header_scale,
			 (float*)frb_header_offset);
        }

        // This loop which takes data from the input buffer and formats the output.
        if (likely(startup == 0)) {

            for (uint i = current_input_location; i < num_samples; ++i) {
  	        if (in_frame_location == 16) { //last sample
                    in_frame_location = 0;
                    frame++;
                    if (frame == 8) { //last frame
		        frame = 0;
			mark_frame_full(frb_buf, unique_name.c_str(), out_buffer_ID);
                        // Get a new output buffer
                        out_buffer_ID = (out_buffer_ID + 1) % frb_buf->num_frames;
			out_frame = wait_for_empty_frame(frb_buf, unique_name.c_str(), out_buffer_ID);
			if (out_frame == NULL) goto end_loop;
			    // Fill the headers of the new buffer
			    fpga_seq_num += 16*8*_fpga_counts_per_sample;
			    fill_headers((unsigned char*)out_frame,
					 &frb_header,
					 fpga_seq_num,
					 num_L1_streams, 
					 (uint16_t*)frb_header_coarse_freq_ids,
					 (float*)frb_header_scale,
					 (float*)frb_header_offset);
                    } //end if last frame
                } //end if last sample

		unsigned char * out_buf = (unsigned char*)out_frame;
		for (int thread_id = 0; thread_id < _num_gpus; ++thread_id) { //loop the 4 GPUs (input)
		    unsigned char * in_buf_data = (unsigned char *)in_frame[thread_id];
		    for (int stream = 0; stream<num_L1_streams; ++stream) { //loop the output buffer  256 streams
 		        for (int beam = 0; beam<_nbeams; ++beam){   //4 beams
			    for (int freq = 0; freq < _factor_upchan_out; ++freq) { //loop 16
				uint32_t out_index = stream*_udp_packet_size*8 + frame * _udp_packet_size 
				                       + beam*_num_gpus*16*16 
				                       + thread_id*16*16   + freq*16 + in_frame_location + _udp_header_size;
				out_buf[out_index] = in_buf_data[ (stream*_nbeams+beam)*num_samples*16 + i*16 + freq];
			    } //end loop freq
			} //end loop beam
		    } //end loop streams
		} //end loop 4 GPUs
		in_frame_location++;
	    } //end looping 100 time samples 
	    current_input_location = 0;
	} //end if not start up

        // Release the input buffers
	for (int i = 0; i < _num_gpus; ++i) {
	    //release_info_object(in_buf[gpu_id], in_buffer_ID[i]);
	    mark_frame_empty(in_buf[i], unique_name.c_str(), in_buffer_ID[i]);
	    in_buffer_ID[i] = (in_buffer_ID[i] + 1) % in_buf[i]->num_frames;

        }
    } //end stop thread
    end_loop:;
}
