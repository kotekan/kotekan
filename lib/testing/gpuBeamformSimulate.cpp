#include "gpuBeamformSimulate.hpp"
#include "errors.h"

#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr
#define HI_NIBBLE(b)                    (((b) >> 4) & 0x0F)  //to reverse packing for CPU array
#define LO_NIBBLE(b)                    ((b) & 0x0F) //to reverse packing for CPU array

#define PI 3.14159265
#define feed_sep 0.3048
#define light 3.e8
#define Freq_ref 492.125984252 //1./ (np.sin(90*D2R) / c / (N_feeds/2) * d * N_feeds *1.e6)

gpuBeamformSimulate::gpuBeamformSimulate(Config& config,
        const string& unique_name,
        bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container, std::bind(&gpuBeamformSimulate::main_thread, this)) {

    apply_config(0);

    input_buf = get_buffer("network_in_buf");
    output_buf = get_buffer("beam_out_buf");

    input_len = _samples_per_data_set * _num_elements * 2;
    // Padded and extracted
    input_len_padded = input_len * 2;
    clamping_output_len = input_len_padded;
    final_output_len = input_len;

    input_unpacked = (double *)malloc(input_len * sizeof(double));
    input_unpacked_padded = (double *)malloc(input_len_padded * sizeof(double));
    clamping_output = (double *)malloc(clamping_output_len * sizeof(double));
    final_output = (double *)malloc(final_output_len * sizeof(double));

    coff = (float *) malloc(16*2*sizeof(float));
    assert(coff != nullptr);
    for (int angle_iter=0; angle_iter < 4; angle_iter++){
        // Not sure what we want for the angles, but let's say 0, 0.1, 0.2 and 0.3 for now.
        // NEED TO FIND OUT
        double anglefrac = 0.1*angle_iter;
        for (int cylinder=0; cylinder < 4; cylinder++){
            coff[angle_iter*4*2 + cylinder*2]     = cos( anglefrac*2*PI*cylinder );
            coff[angle_iter*4*2 + cylinder*2 + 1] = sin( anglefrac*2*PI*cylinder );
        }
    }
}

gpuBeamformSimulate::~gpuBeamformSimulate() {

    free(input_unpacked_padded);
    free(input_unpacked);
    free(clamping_output);
    free(final_output);
    free(coff);
}

void gpuBeamformSimulate::apply_config(uint64_t fpga_seq) {
    _num_elements = config.get_int(unique_name, "num_elements");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");

}

void gpuBeamformSimulate::cpu_beamform_ns(double *data, unsigned long transform_length,  int stop_level)
{
    unsigned long n,m,j,i;
    double wr,wi,theta;
    double tempr,tempi;
    n=transform_length << 1;
    j=1;

    for (i=1;i<n;i+=2) { /* This is the bit-reversal section of the routine. */
        if (j > i) {
            SWAP(data[j-1],data[i-1]); /* Exchange the two complex numbers. */
            SWAP(data[j],data[i]);
        }
        m=transform_length;
        while (m >= 2 && j > m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
    long int step_stop;
    if (stop_level < -1) //neg values mean do the whole sequence; the last stage has pairs half the transform length apart
        step_stop = transform_length/2;
    else
        step_stop =pow(2,stop_level);

    for (long int step_size = 1; step_size <= step_stop ; step_size +=step_size) {
        theta = 3.141592654/(step_size);
        for (int index = 0; index < transform_length; index += step_size*2){ //the trig pattern repeats every step_size*2 entries
            for (int minor_index = 0; minor_index < step_size; minor_index++){ //this inner loop takes cares of the pairs in a set of entries step_size*2 entries
                wr = cos(minor_index*theta);
                wi = sin(minor_index*theta);
                int first_index = (index+minor_index)*2;// *2 for the Re,Im pairs
                int second_index = first_index + step_size*2; //again *2 to account for the pairs
                tempr = wr*data[second_index]-wi*data[second_index+1];
                tempi = wi*data[second_index]+wr*data[second_index+1];
                data[second_index    ]  = data[first_index  ]-tempr;
                data[second_index + 1]  = data[first_index+1]-tempi;
                data[first_index     ] += tempr;
                data[first_index  + 1] += tempi;
            }
        }
    }

}

void gpuBeamformSimulate::cpu_beamform_ew(double *input, double *output, float *Coeff, int nbeamsNS, int nbeamsEW, int npol, int nsamp_in)
{

    int elm_now, out_add;
    for (int t=0;t<nsamp_in;t++){
        for (int p=0;p<npol;p++){
            for (int bEW=0; bEW < nbeamsEW; bEW++){
                for (int bNS=0;bNS < nbeamsNS; bNS++){
                    out_add = (t*2048 + p*1024 + bEW*256 + bNS)*2;
                    for (int elm=0; elm<4; elm++){
                        elm_now = (t*2048 + p*1024 + elm*256 + bNS)*2;

                        //REAL
                        output[out_add]+= input[elm_now]*Coeff[2*(bEW*4+elm)]
                            -input[elm_now + 1]*Coeff[2*(bEW*4+elm)+1];
                        //IMAG
                        output[out_add+1]+=input[elm_now]*Coeff[2*(bEW*4+elm)+1]
                            + input[elm_now+1]*Coeff[2*(bEW*4+elm)];
                    }
                    output[out_add] = output[out_add]/4.;
                    output[out_add+1] = output[out_add+1]/4.;
                }
            }
        }
    }

}

void gpuBeamformSimulate::clamping(double *input, double *output, float freq, int nbeamsNS, int nbeamsEW, int nsamp_in, int npol)
{
    float t, delta_t, Beam_Ref;
    int cl_index;
    float D2R = PI/180.;
    int pad = 2 ;
    int tile=1;
    int nbeams = nbeamsEW*nbeamsNS;
    for (int b=0;b<nbeamsNS; b++){
        Beam_Ref = asin(light*(b-nbeamsNS/2.) / (Freq_ref*1.e6) / (nbeamsNS) /feed_sep) * 180./ PI;
        t = nbeamsNS*pad*(Freq_ref*1.e6)*(feed_sep/light*sin(Beam_Ref*D2R)) + 0.5;
        delta_t = nbeamsNS*pad*(freq*1e6-Freq_ref*1e6) * (feed_sep/light*sin(Beam_Ref*D2R));

        cl_index = (int) floor(t + delta_t) + nbeamsNS*tile*pad/2.;

        if (cl_index < 0)
          cl_index = 0;
        else if (cl_index > nbeamsNS*tile*pad)
          cl_index = nbeamsNS*tile*pad - 1;

        //cl_index = b;
        for (int i=0;i<nsamp_in;i++){
            for (int p=0;p<npol;p++){
                for (int b2 = 0; b2< nbeamsEW; b2++){
                    output[2*(i*npol*nbeamsNS*nbeamsEW + p*nbeams + b2*nbeamsNS + b) ] = input[2*(i*2048*2 + p*1024*2+ b2*512 + cl_index)];
                    output[2*(i*npol*nbeamsNS*nbeamsEW + p*nbeams + b2*nbeamsNS + b) +1] = input[2*(i*2048*2 + p*1024*2 + b2*512 + cl_index) + 1];
                }
            }
        }
    }
}

void gpuBeamformSimulate::main_thread() {
    int input_buf_id = 0;
    int output_buf_id = 0;

    int npol = 2;
    int nbeamsEW = 4;
    int nbeamsNS = 256;
    int nbeams = nbeamsEW*nbeamsNS;
    float freq1 = 450;

    for (;;) {
        get_full_buffer_from_list(input_buf, &input_buf_id, 1);
        wait_for_empty_buffer(output_buf, output_buf_id);

        unsigned char * input = (unsigned char *)input_buf->data[input_buf_id];
        float * output = (float *)output_buf->data[output_buf_id];

        // TODO adjust to allow for more than one frequency.
        // TODO remove all the 32's in here with some kind of constant/define
        INFO("Simulating GPU beamform processing for %s[%d] putting result in %s[%d]",
                input_buf->buffer_name, input_buf_id,
                output_buf->buffer_name, output_buf_id);

        // Unpack and pad the input data
        int dest_idx = 0;
        for (int i = 0; i < input_buf->buffer_size; ++i) {
            input_unpacked[dest_idx++] = HI_NIBBLE(input[i])-8;
            input_unpacked[dest_idx++] = LO_NIBBLE(input[i])-8;
        }

        // Pad to 512
        // TODO this can be simplified a fair bit.
        int index = 0;
        for (int j = 0; j < _samples_per_data_set; j++){
            for (int b = 0; b < nbeamsEW; b++){
                for (int i = 0; i < 512; i++){
                    if (i < 256){
                        input_unpacked_padded[index++] = input_unpacked[2*(j*nbeams + b*nbeamsNS + i)];
                        input_unpacked_padded[index++] = input_unpacked[2*(j*nbeams + b*nbeamsNS + i) + 1];
                    } else{
                        input_unpacked_padded[index++] = 0;
                        input_unpacked_padded[index++] = 0;
                    }
                }
            }
        }

        // Beamform north south.
        for (int i = 0; i < _samples_per_data_set*npol*nbeamsEW; i++){
            cpu_beamform_ns(&input_unpacked_padded[i*512*2], 512, 8);
        }

        // Clam the data
        clamping(input_unpacked_padded, clamping_output, freq1, nbeamsNS, nbeamsEW, _samples_per_data_set, npol);

        cpu_beamform_ew(clamping_output, final_output, coff, nbeamsNS, nbeamsEW, npol, _samples_per_data_set);

        for (int i = 0; i < output_buf->buffer_size; i += sizeof(float)) {
            *((float *)(&output_buf->data[output_buf_id][i])) = (float)final_output[i/sizeof(float)];
        }

        INFO("Simulating GPU beamform processing done for %s[%d] result is in %s[%d]",
                input_buf->buffer_name, input_buf_id,
                output_buf->buffer_name, output_buf_id);

        //move_buffer_info(&input_buf, input_buf_id, &output_buf, output_buf_id);
        mark_buffer_empty(input_buf, input_buf_id);
        mark_buffer_full(output_buf, output_buf_id);

        input_buf_id = (input_buf_id + 1) % input_buf->num_buffers;
        output_buf_id = (output_buf_id + 1) % output_buf->num_buffers;
    }
}

