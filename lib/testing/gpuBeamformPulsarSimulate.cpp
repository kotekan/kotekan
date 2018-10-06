#include "gpuBeamformPulsarSimulate.hpp"
#include "errors.h"
#include <math.h>

#define HI_NIBBLE(b)                    (((b) >> 4) & 0x0F)
#define LO_NIBBLE(b)                    ((b) & 0x0F)

#define PI 3.14159265
#define feed_sep 0.3048
#define light 3.e8
#define Freq_ref 492.125984252
#define freq1 450.

REGISTER_KOTEKAN_PROCESS(gpuBeamformPulsarSimulate);

gpuBeamformPulsarSimulate::gpuBeamformPulsarSimulate(Config& config,
        const string& unique_name,
        bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container, std::bind(&gpuBeamformPulsarSimulate::main_thread, this)) {

    apply_config(0);

    input_buf = get_buffer("network_in_buf");
    register_consumer(input_buf, unique_name.c_str());
    output_buf = get_buffer("beam_out_buf");
    register_producer(output_buf, unique_name.c_str());

    input_len = _samples_per_data_set * _num_elements * 2;
    output_len = _samples_per_data_set * _num_pulsar * _num_pol * 2;

    input_unpacked = (double *)malloc(input_len * sizeof(double));
    phase = (double *) malloc(_num_elements*_num_pulsar*2*sizeof(double));
    cpu_output = (unsigned char *) malloc(output_len*sizeof(unsigned char));
    assert(phase != nullptr);

    //Initiate phase
    int index = 0;
    for (int b=0; b < _num_pulsar; b++){
        for (int n=0; n<_num_elements; n++){
          phase[index++] = b/10.;
          phase[index++] = b/10.;
        }
    }
}

gpuBeamformPulsarSimulate::~gpuBeamformPulsarSimulate() {

    free(input_unpacked);
    free(cpu_output);
    free(phase);

}

void gpuBeamformPulsarSimulate::apply_config(uint64_t fpga_seq) {
    _num_elements = config.get<int32_t>(unique_name, "num_elements");
    _samples_per_data_set = config.get<int32_t>(unique_name,
                                                "samples_per_data_set");
    _num_pulsar = config.get<int32_t>(unique_name, "num_pulsar");
    _num_pol = config.get<int32_t>(unique_name, "num_pol");
}

void gpuBeamformPulsarSimulate::cpu_beamform_pulsar(double *input_unpacked, double *phase, unsigned char *cpu_output, int _samples_per_data_set, int _num_elements, int _num_pulsar, int _num_pol)
{
  float sum_re, sum_im;
  for (int t=0;t<_samples_per_data_set;t++){
    for (int b=0;b<_num_pulsar;b++){
      for (int p=0;p<_num_pol;p++){
        sum_re = 0.0;
        sum_im = 0.0;
        for (int n=0;n<1024;n++){
          //Input and phase both have pol has second fastest
          sum_re += input_unpacked[(t*2048+p*1024+n)*2]*phase[((b*2+p)*1024+n)*2] - input_unpacked[(t*2048+p*1024+n)*2+1]*phase[((b*2+p)*1024+n)*2+1];
          sum_im += input_unpacked[(t*2048+p*1024+n)*2+1]*phase[((b*2+p)*1024+n)*2] + input_unpacked[(t*2048+p*1024+n)*2]*phase[((b*2+p)*1024+n)*2+1];
        }
        cpu_output[(t*_num_pulsar*_num_pol+b*_num_pol+p)*2] = round(sum_re/1024.);
        cpu_output[(t*_num_pulsar*_num_pol+b*_num_pol+p)*2+1] = round(sum_im/1024.);
        //Output has polarization has fastest varying
      }
    }
  }
}


void gpuBeamformPulsarSimulate::main_thread() {
    int input_buf_id = 0;
    int output_buf_id = 0;

    while(!stop_thread) {

        unsigned char * input = (unsigned char *)wait_for_full_frame(input_buf, unique_name.c_str(), input_buf_id);
        if (input == NULL) break;
        unsigned char * output = (unsigned char *)wait_for_empty_frame(output_buf, unique_name.c_str(), output_buf_id);
        if (output == NULL) break;

        for (int i=0;i<input_len;i++){
            input_unpacked[i] = 0.0; //Need this
        }
        for (int i=0;i<output_len;i++){
            cpu_output[i] = 0;
        }

        INFO("Simulating GPU pulsar beamform processing for %s[%d] putting result in %s[%d]",
                input_buf->buffer_name, input_buf_id,
                output_buf->buffer_name, output_buf_id);

        // Unpack and pad the input data
        int dest_idx = 0;
        for (int i = 0; i < input_buf->frame_size; ++i) {
            input_unpacked[dest_idx++] = HI_NIBBLE(input[i])-8;
            input_unpacked[dest_idx++] = LO_NIBBLE(input[i])-8;
        }

	// Beamform 10 pulsars.
        cpu_beamform_pulsar(input_unpacked, phase, cpu_output, _samples_per_data_set, _num_elements, _num_pulsar, _num_pol);

        for (int i = 0; i < output_buf->frame_size; i++) {
	  output[i] = (unsigned char)cpu_output[i];
	    }

        INFO("Simulating GPU pulsar beamform processing done for %s[%d] result is in %s[%d]",
                input_buf->buffer_name, input_buf_id,
                output_buf->buffer_name, output_buf_id);

        //pass_metadata(&input_buf, input_buf_id, &output_buf, output_buf_id);
        mark_frame_empty(input_buf, unique_name.c_str(), input_buf_id);
        mark_frame_full(output_buf, unique_name.c_str(), output_buf_id);

        input_buf_id = (input_buf_id + 1) % input_buf->num_frames;
        output_buf_id = (output_buf_id + 1) % output_buf->num_frames;
    }
}

