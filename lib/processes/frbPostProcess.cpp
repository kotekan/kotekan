// TODO Where do these live?
# define likely(x)      __builtin_expect(!!(x), 1)
# define unlikely(x)    __builtin_expect(!!(x), 0)

#include "frbPostProcess.hpp"

REGISTER_KOTEKAN_PROCESS(frbPostProcess);

frbPostProcess::frbPostProcess(Config& config_,
        const string& unique_name,
        bufferContainer &buffer_container) :
        KotekanProcess(config_, unique_name, buffer_container,
                       std::bind(&frbPostProcess::main_thread, this)){

    apply_config(0);

    in_buf = (struct Buffer **)malloc(_num_gpus * sizeof (struct Buffer *));
    for (int i = 0; i < _num_gpus; ++i) {
        in_buf[i] = get_buffer("in_buf_" + std::to_string(i));
        register_consumer(in_buf[i], unique_name.c_str());
    }
    frb_buf = get_buffer("out_buf");
    register_producer(frb_buf, unique_name.c_str());

    //Dynamic header
    frb_header_beam_ids = new uint16_t[_nbeams];
    frb_header_coarse_freq_ids = new uint16_t[_num_gpus];
    frb_header_scale = new float[_nbeams * _num_gpus];
    frb_header_offset = new float[_nbeams * _num_gpus];

}

frbPostProcess::~frbPostProcess() {
    free(in_buf);
    free(frb_header_beam_ids);
    free(frb_header_coarse_freq_ids);
    free(frb_header_scale);
    free(frb_header_offset);
}

void frbPostProcess::write_header(unsigned char * dest){  
    memcpy(dest, &frb_header, sizeof(struct FRBHeader));
    dest += sizeof(struct FRBHeader);

    memcpy(dest, frb_header_beam_ids, sizeof(uint16_t)*_nbeams);
    dest += sizeof(uint16_t)*_nbeams;

    memcpy(dest, frb_header_coarse_freq_ids, sizeof(uint16_t)*_num_gpus);
    dest += sizeof(uint16_t)*_num_gpus;

    memcpy(dest, frb_header_scale, sizeof(float)*_nbeams*_num_gpus);
    dest += sizeof(float)*_nbeams*_num_gpus;

    memcpy(dest, frb_header_offset, sizeof(float)*_nbeams*_num_gpus);

}

void frbPostProcess::apply_config(uint64_t fpga_seq) {
    if (!config.update_needed(fpga_seq))
        return;

    _num_gpus = config.get_int(unique_name, "num_gpus");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");
    _downsample_time = config.get_int(unique_name, "downsample_time");
    _factor_upchan = config.get_int(unique_name, "factor_upchan");
    _factor_upchan_out = config.get_int(unique_name, "factor_upchan_out"); 
    _nbeams = config.get_int(unique_name, "num_beams");
    _timesamples_per_frb_packet = config.get_int(unique_name, "timesamples_per_frb_packet");
    _incoherent_beam = config.get_bool_default(unique_name, "incoherent_beam",false);

    num_L1_streams = 1024/_nbeams;
    num_samples = _samples_per_data_set / _downsample_time / _factor_upchan;

    fpga_counts_per_sample = _downsample_time * _factor_upchan;
    udp_header_size = sizeof(struct FRBHeader)
                    + sizeof(uint16_t)*_nbeams //beam ids
                    + sizeof(uint16_t)*_num_gpus //freq band ids
                    + sizeof(float)*_nbeams*_num_gpus //scales
                    + sizeof(float)*_nbeams*_num_gpus //offsets
                    ;
    udp_packet_size = _nbeams * _num_gpus * _factor_upchan_out * _timesamples_per_frb_packet + udp_header_size;
}

void frbPostProcess::main_thread() {

    uint in_buffer_ID[_num_gpus] ;   //4 of these , cycle through buffer depth
    uint8_t * in_frame[_num_gpus];
    int out_buffer_ID = 0;  

    for (int i = 0; i < _num_gpus; ++i) {
        in_buffer_ID[i] = 0;
    }

    frb_header.protocol_version = 1;
    frb_header.data_nbytes =  udp_packet_size - udp_header_size;
    frb_header.fpga_counts_per_sample =  fpga_counts_per_sample;
    frb_header.fpga_count = 0 ;  //to be updated in fill_header
    frb_header.nbeams = _nbeams;  //4
    frb_header.nfreq_coarse = _num_gpus; //4
    frb_header.nupfreq = _factor_upchan_out;
    frb_header.ntsamp = _timesamples_per_frb_packet;

    for (int ii=0;ii<_nbeams;++ii){
        frb_header_beam_ids[ii] = 7; //To be overwritten in fill_header
    }
    for (int ii=0;ii<_num_gpus;++ii){
        frb_header_coarse_freq_ids[ii] = 0;;//_freq_array[ii] 
    }
    for (int ii =0; ii<_nbeams * _num_gpus;++ii){
        frb_header_scale[ii] = 1.; 
        frb_header_offset[ii] = 0;
    }

    while(!stop_thread) {
        // Get the next output buffer, id = 0 to start.
        uint8_t * out_frame = wait_for_empty_frame(frb_buf, unique_name.c_str(), out_buffer_ID);
        if (out_frame == NULL) return;
        // Get an input buffer, This call is blocking!
        for (int i = 0; i < _num_gpus; ++i) {
            in_frame[i] = wait_for_full_frame(in_buf[i], unique_name.c_str(), in_buffer_ID[i]);
            if (in_frame[i] == NULL) return;
        }

        frb_header.fpga_count = get_fpga_seq_num(in_buf[0], in_buffer_ID[0]);
        for (int i = 0; i < _num_gpus; ++i) {
            assert(frb_header.fpga_count ==
                   (uint64_t)get_fpga_seq_num(in_buf[i], in_buffer_ID[i]));
            stream_id_t stream_id = get_stream_id_t(in_buf[i], in_buffer_ID[i]);
            frb_header_coarse_freq_ids[i] = bin_number_chime(&stream_id);
        }

        //Sum all the beams together into beam[0] location.
        // WARNING: THIS MODIFIES THE INPUT BUFFER!!!!!
        // POTENTIAL RACE CLOBBER WITH PEER PROCESSES!!
        if (_incoherent_beam){
            int fti = _factor_upchan_out * num_samples / (sizeof(__m256)/sizeof(float));
            for (int thread_id = 0; thread_id < _num_gpus; thread_id++) { //loop 4 GPUs (input)
                float* in_data = (float *)in_frame[thread_id];
                for (int b = 0; b<num_L1_streams*_nbeams; b++) { //loop 1024 beams
                    for (int ft=0; ft < fti; ft++){ //loop over time/freq
                        int idx = ft * (sizeof(__m256)/sizeof(float));
                        int idx_next = b * num_samples * _factor_upchan_out;
                        __m256 _a = _mm256_load_ps(in_data+idx);
                        __m256 _b = _mm256_load_ps(in_data+idx+idx_next);
                        __m256 _c = _mm256_add_ps(_a,_b);
                        _mm256_store_ps(in_data+idx,_c);
                    }
                }
            }
        }

//        in_buf = [stream=256, nbeams=4, nsamples=128, freq=16]
//        out_buf = [stream=256,frames=8,[packet_size]]
//        [packet_size] = [nbeams=4,gpu=4,freq=16,time=16]+header
        float ofs,scl;
        for (uint i = 0; i < num_samples; i+=_timesamples_per_frb_packet) { //loop 128 time samples, in 8 steps
            for (int stream = 0; stream<num_L1_streams; stream++) { //loop 256 streams (output)
                for (int b=0; b<_nbeams;b++){ //loop 4 beams / stream
                    for (int thread_id = 0; thread_id < _num_gpus; thread_id++) { //loop 4 GPUs (input)
                        float* in_data = (float *)in_frame[thread_id];
                        //Calc scale and offset
                        {
                            //KV version
                            float min, max;
                            int idx = ((stream * _nbeams     + b)
                                               * num_samples + i)
                                               * _factor_upchan_out;
                            // AVX2 option, fastest of a few I tried
                            __m256 _cA = _mm256_load_ps(in_data+idx  );
                            __m256 _cB = _mm256_load_ps(in_data+idx+8);
                            __m256 _mx = _mm256_max_ps(_cA,_cB);
                            __m256 _mn = _mm256_min_ps(_cA,_cB);
                            for (int t=1;t<16;t++){
                                idx += 16;
                                _cA = _mm256_load_ps(in_data+idx  );
                                _cB = _mm256_load_ps(in_data+idx+8);
                                _mx = _mm256_max_ps(_mx,_mm256_max_ps(_cA,_cB));
                                _mn = _mm256_min_ps(_mn,_mm256_min_ps(_cA,_cB));
                            }
                            __m128 mx = _mm_max_ps(_mm256_extractf128_ps(_mx,0),_mm256_extractf128_ps(_mx,1));
                            __m128 mn = _mm_min_ps(_mm256_extractf128_ps(_mn,0),_mm256_extractf128_ps(_mn,1));
                            for (int u = 0; u < 3; u++){
                                mx = _mm_max_ps(mx, _mm_shuffle_ps(mx, mx, 0x93));
                                mn = _mm_min_ps(mn, _mm_shuffle_ps(mn, mn, 0x93));
                            }
                            _mm_store_ss(&max,mx);
                            _mm_store_ss(&min,mn);

                            scl = 255. / (max-min);
                            ofs = min;
                            frb_header_scale[b*_num_gpus + thread_id] = scl;
                            frb_header_offset[b*_num_gpus + thread_id] = ofs;
                        }
                        //scale and offset, dump directly into the output buffer
                        __m256 _scl = _mm256_broadcast_ss(&scl);
                        float off=-ofs*scl;
                        __m256 _ofs = _mm256_broadcast_ss(&off);
                        /*
                        int t_per_m = sizeof(__m256) / sizeof(float);
                        for (int t=0; t<_timesamples_per_frb_packet; t+=t_per_m){
                            for (int f=0; f<_factor_upchan_out; f++){
                                uint32_t in_index  = (stream * _nbeams + b) * num_samples * 16 
                                                       + (i + t) * _factor_upchan_out
                                                       + f;
                                __m256 _in = _mm256_set_ps(in_data[in_index+_factor_upchan_out*0],
                                                           in_data[in_index+_factor_upchan_out*1],
                                                           in_data[in_index+_factor_upchan_out*2],
                                                           in_data[in_index+_factor_upchan_out*3],
                                                           in_data[in_index+_factor_upchan_out*4],
                                                           in_data[in_index+_factor_upchan_out*5],
                                                           in_data[in_index+_factor_upchan_out*6],
                                                           in_data[in_index+_factor_upchan_out*7]);
                                __m256 _out = _mm256_fmadd_ps(_in,_scl,_ofs); //now [0-255]
                                __m256i _y = _mm256_cvtps_epi32(_out); // Convert them to 32-bit ints
                                _y = _mm256_packus_epi32(_y, _y);      // Pack down to 16 bits
                                _y = _mm256_packus_epi16(_y, _y);      // Pack down to 8 bits
                                uint32_t out_index = stream* udp_packet_size*num_samples/_timesamples_per_frb_packet
                                                       + (i/_timesamples_per_frb_packet) * udp_packet_size
                                                       + b * _num_gpus* 16*16
                                                       + thread_id * 16*16
                                                       + (f * 16 + t)
                                                       + udp_header_size;
                                *(int*)(out_frame+out_index  ) = _mm256_extract_epi32(_y, 0);
                                *(int*)(out_frame+out_index+4) = _mm256_extract_epi32(_y, 4);
                            }
                        }*/
                        int f_per_m = sizeof(__m256) / sizeof(float);
                        char utr[256], tr[256]; 
                        for (int t=0; t<_timesamples_per_frb_packet; t++){
                            for (int f=0; f<_factor_upchan_out; f+=f_per_m){
                                uint32_t in_index  = (stream * _nbeams + b) * num_samples * 16
                                                       + (i + t) * _factor_upchan_out
                                                       + f;
                                __m256 _in = _mm256_load_ps(in_data+in_index);
                                __m256 _out = _mm256_fmadd_ps(_in,_scl,_ofs); //now [0-255]
                                __m256i _y = _mm256_cvtps_epi32(_out); // Convert them to 32-bit ints
                                _y = _mm256_packus_epi32(_y, _y);      // Pack down to 16 bits
                                _y = _mm256_packus_epi16(_y, _y);      // Pack down to 8 bits
                                ((uint32_t*)utr)[t*16+f+0] = _mm256_extract_epi32(_y, 0);
                                ((uint32_t*)utr)[t*16+f+1] = _mm256_extract_epi32(_y, 4);
                            }
                        }
                        for (int t=0; t<16; t++) for (int f=0; f<16; f++) tr[f*16+t] = utr[t*16+f];
                        uint32_t out_index = stream* udp_packet_size*num_samples/_timesamples_per_frb_packet
                                               + (i/_timesamples_per_frb_packet) * udp_packet_size
                                               + b * _num_gpus* 16*16
                                               + thread_id * 16*16
                                               + udp_header_size;
                        memcpy(out_frame+out_index,tr,16*16);
                        frb_header.fpga_count += fpga_counts_per_sample * _timesamples_per_frb_packet;
                    } //end 4 GPUs
                } // end 4 nbeam
                // Fill the headers of the packet
                uint32_t out_index = stream* udp_packet_size*num_samples/_timesamples_per_frb_packet
                                   + (i/_timesamples_per_frb_packet) * udp_packet_size;
                write_header(&out_frame[out_index]);
            } // end 256 streams
        } //end looping 128 time samples 
        mark_frame_full(frb_buf, unique_name.c_str(), out_buffer_ID);
        out_buffer_ID = (out_buffer_ID + 1) % frb_buf->num_frames;

        // Release the input buffers
        for (int i = 0; i < _num_gpus; ++i) {
            //release_info_object(in_buf[gpu_id], in_buffer_ID[i]);
            mark_frame_empty(in_buf[i], unique_name.c_str(), in_buffer_ID[i]);
            in_buffer_ID[i] = (in_buffer_ID[i] + 1) % in_buf[i]->num_frames;
        }
    } //end stop thread
}
