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

# define samples_in_frame 3125
# define num_packet 16

#include "buffer.h"
#include "pulsarPostProcess.hpp"
#include "Config.hpp"
#include "util.h"
#include "errors.h"
#include "vdif_functions.h" 
#include "chimeMetadata.h"
#include "fpga_header_functions.h"

pulsarPostProcess::pulsarPostProcess(Config& config_,
        const string& unique_name,
        bufferContainer &buffer_container) :
        KotekanProcess(config_, unique_name, buffer_container,
                       std::bind(&pulsarPostProcess::main_thread, this)){

    apply_config(0);

    in_buf = (struct Buffer **)malloc(_num_gpus * sizeof (struct Buffer *));
    for (int i = 0; i < _num_gpus; ++i) {
        in_buf[i] = get_buffer("network_input_buffer_" + std::to_string(i));
        register_consumer(in_buf[i], unique_name.c_str());
    }
    pulsar_buf = get_buffer("pulsar_out_buf");
    register_producer(pulsar_buf, unique_name.c_str());

}

pulsarPostProcess::~pulsarPostProcess() {
    free(in_buf);
}

void pulsarPostProcess::fill_headers(unsigned char * out_buf,
                  struct VDIFHeader * vdif_header,
                  const uint64_t fpga_seq_num,
		  struct timeval * time_now,
		  struct psrCoord * psr_coord,
		  uint16_t * freq_ids){
  //    assert(sizeof(struct VDIFHeader) == _udp_header_size);

    for (int f=0;f<_num_gpus;++f) { //4 freq
        vdif_header->thread_id = freq_ids[f];
        for (int psr=0;psr<_num_pulsar; ++psr) { //10 streams
	    vdif_header->eud1 = psr; //beam id
	    for (int i = 0; i < num_packet; ++i) {  //16 frames in a stream
	        uint64_t fpga_now = (fpga_seq_num + samples_in_frame * i);
		vdif_header->eud2 = (fpga_now & (0xFFFFFFFF<<32))>>32 ;
		vdif_header->eud3 = (fpga_now & 0xFFFFFFFF)>>0;
		uint16_t ra_part = (uint16_t)(psr_coord[f].ra[psr]*100);
		uint16_t dec_part = (uint16_t)((psr_coord[f].dec[psr]+90)*100);
		vdif_header->eud4 = ((ra_part<<16) & 0xFFFF0000) + (dec_part & 0xFFFF);
		//if ((i==0) && (psr ==0)) INFO("---fill_header H8 -----gpu=%d  ra_part=%" PRId16 "; Dec=%" PRId16 "; eud4=%" PRIuLEAST32 "; edu3=%" PRIuLEAST32 "\n", f, ra_part, dec_part, vdif_header->eud4, vdif_header->eud3);
		if ((i==0) && (psr ==0)) INFO("---fill_header H8 -----gpu=%d  ra_part=%hd; Dec=%hd; eud4=%" PRIuLEAST32 "\n", f, ra_part, dec_part, vdif_header->eud4);
		vdif_header->seconds = time_now->tv_sec;
		vdif_header->data_frame =  (time_now->tv_usec/1.e6) / (samples_in_frame*2.56e-6);
		memcpy(&out_buf[(f*_num_pulsar+psr)*num_packet*_udp_packet_size + i*_udp_packet_size], vdif_header, sizeof(struct VDIFHeader));
	    }
	}
    }
}

void pulsarPostProcess::apply_config(uint64_t fpga_seq) {
    if (!config.update_needed(fpga_seq))
        return;

    _num_gpus = config.get_int(unique_name, "num_gpus");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");
    _nfreq_coarse = config.get_int(unique_name, "num_gpus"); //4
    _num_pulsar = config.get_int(unique_name, "num_pulsar");
    _num_pol = config.get_int(unique_name, "num_pol");
    _timesamples_per_pulsar_packet = config.get_int(unique_name, "timesamples_per_pulsar_packet");
    _udp_packet_size = config.get_int(unique_name, "udp_pulsar_packet_size");
    _udp_header_size = config.get_int(unique_name, "udp_pulsar_header_size");
      
}

void pulsarPostProcess::main_thread() {

    uint in_buffer_ID[_num_gpus] ;   //4 of these , cycle through buffer depth
    uint8_t * in_frame[_num_gpus];
    int out_buffer_ID = 0;  
    int startup = 1; //related to the likely & unlikely
    uint freq_ids[_num_gpus] ;

    for (int i = 0; i < _num_gpus; ++i) {
        in_buffer_ID[i] = 0;
    }
    uint32_t current_input_location = 0; //goes from 0 to _samples_per_data_set

    struct VDIFHeader vdif_header;
    vdif_header.seconds = 0;  //UD
    vdif_header.legacy = 0;
    vdif_header.invalid = 0;
    vdif_header.data_frame = 0 ;  //UD
    vdif_header.ref_epoch = 36; // First half of 2018. 
    vdif_header.unused = 0;
    vdif_header.frame_len = 5000;
    vdif_header.log_num_chan = 0; //Check ln4=2 or ln1=0 ? 
    vdif_header.vdif_version = 1;
    vdif_header.station_id = 'CH'; //Need to fomrally ask the Vdif community
    vdif_header.thread_id = 0; //UD   freq
    vdif_header.bits_depth = 8 ; //4+4 
    vdif_header.data_type = 1; // Complex
    vdif_header.edv = 0;
    vdif_header.eud1 = 0;  //UD: beam number [0 to 9]
    vdif_header.eud2 = 0;  // UD: fpga count high bit
    vdif_header.eud3 = 0;  // UD: fpga count low bit
    vdif_header.eud4 = 0;  // Ra_int + Ra_dec + Dec_int + Dec_dec ? Source name ? Obs ID?


    int frame = 0;
    int in_frame_location = 0; //goes from 0 to 3125
    uint64_t fpga_seq_num = 0;

    int num_L1_streams = _num_pulsar;

    struct psrCoord psr_coord[_num_gpus];
    // Get the first output buffer which will always be id = 0 to start.
    uint8_t * out_frame = wait_for_empty_frame(pulsar_buf, unique_name.c_str(), out_buffer_ID);
    if (out_frame == NULL) goto end_loop;

    while(!stop_thread) {
        // Get an input buffer, This call is blocking!
        for (int i = 0; i < _num_gpus; ++i) {
	    in_frame[i] = wait_for_full_frame(in_buf[i], unique_name.c_str(), in_buffer_ID[i]);
	    if (in_frame[i] == NULL) goto end_loop;

	    psr_coord[i] = get_psr_coord(in_buf[i], in_buffer_ID[i]);
	    INFO("-----!!!!![postprocess H8]-------bufferID=%d GPU=%d psr_coord=(%f %f) (%f %f) (%f %f) (%f %f)",in_buffer_ID[i], i, psr_coord[i].ra[0], psr_coord[i].dec[0], psr_coord[i].ra[3], psr_coord[i].dec[3], psr_coord[i].ra[7],psr_coord[i].dec[7], psr_coord[i].ra[9], psr_coord[i].dec[9]);

	    //INFO("GPU Post process got full buffer ID %d for GPU %d", in_buffer_ID[i],i);
	}
        //INFO("pulsar_post_process; got full set of GPU output buffers");


        uint64_t first_seq_number = get_fpga_seq_num(in_buf[0], in_buffer_ID[0]);

	//Get time, use system time for now, gps time requires ch_master
	time_now = get_first_packet_recv_time(in_buf[0], in_buffer_ID[0]);

        for (int i = 0; i < _num_gpus; ++i) {
	    assert(first_seq_number ==
		   (uint64_t)get_fpga_seq_num(in_buf[i], in_buffer_ID[i]));

	    stream_id_t stream_id = get_stream_id_t(in_buf[i], in_buffer_ID[i]);
            float freq_now = bin_number_chime(&stream_id);
            freq_ids[i] = freq_now;
	}

        // If this is the first time wait until we get the start of an interger second period.
        if (unlikely(startup == 1)) {
	    // testing sync code
            startup = 0;
            current_input_location = 0;

            // Fill the first output buffer headers
	    fpga_seq_num = first_seq_number;
	    fill_headers((unsigned char*)out_frame,
			 &vdif_header,
			 first_seq_number,
			 &time_now, 
			 psr_coord,
			 (uint16_t*)freq_ids);
        }

        // This loop which takes data from the input buffer and formats the output.
        if (likely(startup == 0)) {

            for (uint i = current_input_location; i < _samples_per_data_set; ++i) {
  	        if (in_frame_location == samples_in_frame) { //last sample
                    in_frame_location = 0;
                    frame++;
                    if (frame == num_packet ) { //last frame
		        frame = 0;
			mark_frame_full(pulsar_buf, unique_name.c_str(), out_buffer_ID);
                        // Get a new output buffer
                        out_buffer_ID = (out_buffer_ID + 1) % pulsar_buf->num_frames;
			out_frame = wait_for_empty_frame(pulsar_buf, unique_name.c_str(), out_buffer_ID);
			if (out_frame == NULL) goto end_loop;
			    // Fill the headers of the new buffer
			    fpga_seq_num += samples_in_frame*num_packet;
			    //gps_time += samples_in_frame*2.56e-6;
			    time_now.tv_usec += samples_in_frame*2.56;
			    if (time_now.tv_usec > 999999) {
			        time_now.tv_usec = time_now.tv_usec % 999999;
			        time_now.tv_sec +=1;
			    }
			    fill_headers((unsigned char*)out_frame,
					 &vdif_header,
					 fpga_seq_num,
					 &time_now,
					 psr_coord,
					 (uint16_t*)freq_ids);
                    } //end if last frame
                } //end if last sample

		unsigned char * out_buf = (unsigned char*)out_frame;
		for (int thread_id = 0; thread_id < _num_gpus; ++thread_id) { //loop the 4 GPUs (input)
		    unsigned char * in_buf_data = (unsigned char *)in_frame[thread_id];
		    for (int psr = 0; psr<_num_pulsar; ++psr) { //loop psr
 		        for (int p=0;p<_num_pol; ++p) {
			    uint32_t out_index = (psr+thread_id*_num_pulsar) *_udp_packet_size*num_packet + frame * _udp_packet_size 
			                          + p*samples_in_frame + in_frame_location + _udp_header_size ;
			    out_buf[out_index] = in_buf_data[i*_num_pulsar*_num_pol + psr*_num_pol + p]; 
			} //end loop pol

			//pad 6 Bytes of 0 to make each packet size divisible by 8
			/*for (i=6282; i<6288;i++){
 			    out_buf[ (psr+thread_id*_num_pulsar) *_udp_packet_size*num_packet + frame * _udp_packet_size + i] = 0;
			    }*/
		    } //end loop psr
		} //end loop 4 GPUs
		in_frame_location++;
	    } //end looping i
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
