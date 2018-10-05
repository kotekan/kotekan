#include "autocorrVDIF.hpp"
#include "buffer.h"
#include "errors.h"
#include "nt_memcpy.h"
#include "Config.hpp"
#include "util.h"
#include "vdif_functions.h"
#include "time_tracking.h"
#ifdef MAC_OSX
    #include "osxBindCPU.hpp"
#endif
#include <dirent.h>
#include <sys/socket.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <time.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <memory.h>
#include <unistd.h>
#include <assert.h>
#include <pthread.h>
#include <inttypes.h>
#include <functional>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <math.h>
#include <pmmintrin.h>
#include <immintrin.h>

#define PACKET_LEN (num_freq+VDIF_HEADER_LEN)

REGISTER_KOTEKAN_PROCESS(autocorrVDIF);

autocorrVDIF::autocorrVDIF(Config &config, const string& unique_name,
                        bufferContainer &buffer_container) :
                  KotekanProcess(config, unique_name, buffer_container,
                                 std::bind(&autocorrVDIF::main_thread, this))
{
    buf_in = get_buffer("vdif_in_buf");
    register_consumer(buf_in, unique_name.c_str());
    buf_out = get_buffer("power_out_buf");
    register_producer(buf_out, unique_name.c_str());

    timesteps_in = config.get<int>(unique_name, "samples_per_data_set");
    integration_length = config.get<int>(unique_name, "power_integration_length");
    timesteps_out = timesteps_in / integration_length;
    num_freq = config.get<int>(unique_name, "num_freq");
    num_elem = config.get<int>(unique_name, "num_elements");
    num_pol = config.get<int>(unique_name, "num_pol");

    if (timesteps_in % timesteps_out)
    {
        ERROR("BAD COMBINATION OF BUFFER & INTEGRATION LENGTHS!");
    }

    foldbins=100;
    foldperiod=0.33;
    out = (uint32_t*)malloc(foldbins * num_freq * sizeof(uint32_t));
    outcount = (uint32_t*)malloc(foldbins * num_freq * sizeof(uint32_t));

}


void autocorrVDIF::apply_config(uint64_t fpga_seq) {
}


autocorrVDIF::~autocorrVDIF() {
    free(out);
    free(outcount);
}

void autocorrVDIF::main_thread() {
    srand(time(NULL));

    int buf_in_id=0;
    int buf_out_id=0;

//    const int nthreads=1;
//    int nloop = timesteps_out/nthreads;
//    std::thread this_thread[nthreads];

    while(!stop_thread) {
        in_local = (unsigned char *) wait_for_full_frame(buf_in, unique_name.c_str(), buf_in_id);
        if(in_local == NULL) break;
        out_local = (unsigned char *) wait_for_empty_frame(buf_out, unique_name.c_str(), buf_out_id);
        if(out_local == NULL) break;

        double start_time = e_time();
/*
        for (int j=0; j<nthreads; j++) {
            this_thread[j] = std::thread(&autocorrVDIF::parallelSqSumVdif, this, j, nloop);
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            for (auto &i : config.get<std::vector<int>>(unique_name, "cpu_affinity"))
                CPU_SET(i, &cpuset);
            pthread_setaffinity_np(this_thread[j].native_handle(), sizeof(cpu_set_t), &cpuset);
        }
        for (int j=0; j<nthreads; j++)
            this_thread[j].join();
*/
        if (foldbins){
            //work out how many bins to use
            fastSqSumVdif(in_local, out, outcount);
            float dm=-1.;
            if (dm>0){
                // Dedisperse!
                for (int f=0; f<1024; f++){
                    //Roll & copy in two chunks
                    memcpy(&out_local[f],&out[f],1);
                    memcpy(&out_local[f],&out[f],1);
                }
            } else {
                // Just copy it over
                memcpy(out_local,out,num_freq * foldbins * sizeof(uint32_t));
            }
        } else {
            //integrate fully
            fastSqSumVdif(in_local, out, outcount);
            memcpy(out_local,out,num_freq * sizeof(uint32_t));
        }

        double stop_time = e_time();
        INFO("TIME USED FOR INTEGRATION: %fms\n",(stop_time-start_time)*1000);

        mark_frame_empty(buf_in, unique_name.c_str(), buf_in_id);
        mark_frame_full(buf_out, unique_name.c_str(), buf_out_id);
        buf_in_id = ( buf_in_id + 1 ) % buf_in->num_frames;
        buf_out_id = (buf_out_id + 1) % (buf_out->num_frames);
   }
}

void autocorrVDIF::parallelSqSumVdif(int loop_idx, int loop_length){
//    uint temp_buffer[num_freq*num_elem];
//    for (int i=loop_idx*loop_length; i<(loop_idx+1)*loop_length; i++)
//        fastSqSumVdif(in_local+(i*integration_length*PACKET_LEN*num_elem),
//                temp_buffer, (uint*)(out_local+i*(1+num_freq)*num_elem*sizeof(uint)));
}

#ifdef __AVX2__
inline void autocorrVDIF::fastSqSumVdif(uint8_t *input, uint32_t *output, uint32_t *outcount) {
    memset(output,0,1);

    __m256i _im, _re, _lo, _hi;
    __m256i _pow1, _pow2, _pow3, _pow4;
    __m256i _4b_mask, _offset_8, _pad_0;
    bool first_pass = false;
    int num_chan;
    int vdif_packet_length = 5032;
    int num_threads = 256;
    int samples_per_frameset = 3125;
    int framesets_per_kframe = samples_per_frameset / 625;

    for (int frameset = 0; frameset < framesets_per_kframe; frameset++) {
        for (int thread = 0; thread < num_threads; thread++) {
            const int idx_header = ((frameset) * num_threads +
                                       thread) * vdif_packet_length;
            VDIFHeader *header = (VDIFHeader *)input+idx_header;
            if (header->invalid) continue;
            if (header->bits_depth != 3){
                ERROR("Bad VDIF: I only know how to handle 4b data! (so far...)")
                continue;
            }
            if (first_pass){
                num_chan = (1 << header->log_num_chan);
                samples_per_frameset = (header->frame_len*8 - sizeof(VDIFHeader)) /
                                           num_chan;
            }
            uint8_t *data = input + idx_header + sizeof(VDIFHeader);
            uint8_t *ptr;
            uint8_t local_data[256];

            int bytes_per_sample = (header->bits_depth+1)*2/8;

            int samples_per_iteration = sizeof(__m256i)/bytes_per_sample/num_chan;
            if (samples_per_iteration < 1) samples_per_iteration=1;
            for (int sample = 0; sample < samples_per_frameset; sample++) {

                int chans_per_iteration = sizeof(__m256i)/bytes_per_sample;
                for (int chan = 0; chan < num_chan; chan+= chans_per_iteration) {
                    const int index = (sample*num_chan + chan)*bytes_per_sample;
                    int bytes_into_frame = (sample*num_chan + chan)*bytes_per_sample;
                    if (bytes_into_frame+bytes_per_sample > samples_per_frameset*num_chan) {
                        memcpy(local_data,&data[index],
                                samples_per_frameset*num_chan - bytes_into_frame);
                        ptr = local_data;
                    } else ptr = &data[index];
                    // Load 64 4 bit numbers
                    _im = _mm256_loadu_si256((__m256i const *)ptr);
                    // Shift the high 4-bits to the low 4-bits in each 8 bit block
                    _re = _mm256_srli_epi64(_im, 4); // real

                    // Mask out the lower 4 bits
                    _4b_mask   = _mm256_set1_epi32(0x0f0f0f0f);
                    _re = _mm256_and_si256(_re, _4b_mask); // imag
                    _im = _mm256_and_si256(_im, _4b_mask); // real

                    // This packs (real, imag) (8+8) pairs together
                    _lo = _mm256_unpacklo_epi8(_re, _im);
                    _hi = _mm256_unpackhi_epi8(_re, _im);

                    // subtract 8 to make the 8-bit numbers twos complement
                    _offset_8 = _mm256_set1_epi8(8);
                    _lo = _mm256_sub_epi8(_lo, _offset_8);
                    _hi = _mm256_sub_epi8(_hi, _offset_8);

                    // Take the abs value since the multi is unsigned
                    _lo = _mm256_abs_epi8(_lo);
                    _hi = _mm256_abs_epi8(_hi);

                    // Multiply and add real and imaginary 8+8-bit pairs into 16-bit ints
                    _lo = _mm256_maddubs_epi16(_lo, _lo); // hi
                    _hi = _mm256_maddubs_epi16(_hi, _hi); // lo

                    // Extend to 32-bit
                    _pad_0 = _mm256_set1_epi32(0);
                    _pow1 = _mm256_unpacklo_epi16(_lo, _pad_0);
                    _pow2 = _mm256_unpackhi_epi16(_lo, _pad_0);
                    _pow3 = _mm256_unpacklo_epi16(_hi, _pad_0);
                    _pow4 = _mm256_unpackhi_epi16(_hi, _pad_0);

                    // One or fewer __m256i registers / packet? Unpack
                    if (num_chan*sizeof(uint32_t) <= sizeof(__m256i)){
                        _pow1 = _mm256_add_epi32(_pow1,_pow2);
                        _pow3 = _mm256_add_epi32(_pow3,_pow4);
                        _pow1 = _mm256_add_epi32(_pow1,_pow3);

                        int out_index;
                        if (num_chan*sizeof(uint32_t) == sizeof(__m256i)) //8 channels
                        {
                            if (foldbins){
                                float t=0.;
                                int bin = foldbins * fmod(t,foldperiod);
                                out_index=0;//some calculation
                            } 
                            else
                                out_index=out_index = thread*num_chan + chan;
                            //Load the old power sum
                            _pow2 = _mm256_loadu_si256((__m256i const *)&output[out_index]);
                            _pow1 = _mm256_add_epi32(_pow1,_pow2);

                            //Store new integrated power sum
                            _mm256_storeu_si256((__m256i *)&output[out_index], _pow1);
                        }
                        else if (num_chan*sizeof(uint32_t) == sizeof(__m128i)) //4 channels
                        {
                            //Add the two 8-channel chunks
                            _pow2 = _mm256_permute2f128_ps(_pow1,_pow1,1);
                            _pow1 = _mm256_add_epi32(_pow1,_pow2);
                            __m128 _p1 = _mm256_extractf128_si256(_pow1,0);

                            //Add current power to integrated sum
                            __m128 _p2 = _mm_loadu_si128((__m128i const *)&output[out_index]);
                            _p1 = _mm_add_epi32(_p1,_p2);

                            //Store new integrated power sum
                            _mm_storeu_si128((__m128i *)&output[out_index], _p1);
                        }
                        else throw std::runtime_error("Can't handle 1 or 2 channel data yet");
                    }
                    else if (num_chan % (sizeof(__m256i)/sizeof(uint32_t)) == 0)
                    {
                        throw std::runtime_error("Large num_chan (N*8) not handled yet!");
                        /*
                        int out_index = 1;// * num_freq + freq * 32;
                        //Load Integrated Power
                        ymm4 = _mm256_loadu_si256((__m256i const *)&temp_buf[out_index + 0*8]);
                        ymm5 = _mm256_loadu_si256((__m256i const *)&temp_buf[out_index + 1*8]);
                        ymm6 = _mm256_loadu_si256((__m256i const *)&temp_buf[out_index + 2*8]);
                        ymm7 = _mm256_loadu_si256((__m256i const *)&temp_buf[out_index + 3*8]);

                        //Add current power to integrated sum
                        ymm0 = _mm256_add_epi32(ymm0, ymm4);
                        ymm1 = _mm256_add_epi32(ymm1, ymm5);
                        ymm2 = _mm256_add_epi32(ymm2, ymm6);
                        ymm3 = _mm256_add_epi32(ymm3, ymm7);

                        //Store new integrated power sum
                        _mm256_storeu_si256((__m256i *)&temp_buf[out_index + 0*8], ymm0);
                        _mm256_storeu_si256((__m256i *)&temp_buf[out_index + 1*8], ymm1);
                        _mm256_storeu_si256((__m256i *)&temp_buf[out_index + 2*8], ymm2);
                        _mm256_storeu_si256((__m256i *)&temp_buf[out_index + 3*8], ymm3);
                        */
                    }
                    else{
                        throw std::runtime_error("Can't deal with oddball channel numbers!");
                    }
                }
            }
        }
    }
}

#else
inline void autocorrVDIF::fastSqSumVdif(unsigned char * data, uint * temp_buf, uint *out)
{
    throw std::runtime_error("This system does not support AVX2, fast square-and-sum will not work");
}
#endif

