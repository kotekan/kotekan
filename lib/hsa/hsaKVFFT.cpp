#include "hsaKVFFT.hpp"

#include "Config.hpp"             // for Config
#include "gpuCommand.hpp"         // for gpuCommandType, gpuCommandType::KERNEL
#include "hsaCommand.hpp"         // for kernelParams, KERNEL_EXT, REGISTER_HSA_COMMAND, _facto...
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface, Config

#include "fmt.hpp" // for format, fmt
#include <fftw3.h>

#include <cstdint>   // for int32_t
#include <exception> // for exception
#include <regex>     // for match_results<>::_Base_type
#include <string.h>  // for memcpy, memset
#include <vector>    // for vector

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_HSA_COMMAND(hsaKVFFT);

hsaKVFFT::hsaKVFFT(Config& config, const std::string& unique_name,
                                 bufferContainer& host_buffers, hsaDeviceInterface& device) :
    hsaSubframeCommand(config, unique_name, host_buffers, device, "kv_fft" KERNEL_EXT,
                       "kv_fft.hsaco") {
    command_type = gpuCommandType::KERNEL;

    _num_elements = config.get<int32_t>(unique_name, "num_elements");
    _num_local_freq = config.get<int32_t>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int32_t>(unique_name, "samples_per_data_set");
    input_frame_len = _num_elements * _num_local_freq * _samples_per_data_set;

    // pre-allocate GPU memory
    device.get_gpu_memory_array("input", 0, input_frame_len);
    device.get_gpu_memory_array(fmt::format(fmt("kvfft_{:d}"), _sub_frame_index), 0, 512*2*sizeof(float));//*8*_samples_per_data_set);
}

hsaKVFFT::~hsaKVFFT() {}

hsa_signal_t hsaKVFFT::execute(int gpu_frame_id, hsa_signal_t precede_signal) {

    // Unused parameter, suppress warning
    (void)precede_signal;

    // Set kernel args
    struct __attribute__((aligned(16))) args_t {
        void* input;
        void* output;
    } args;

    memset(&args, 0, sizeof(args));

    // Index past the start of the input for the required sub frame
    args.input =
        (void*)((uint8_t*)device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len)
                + _num_elements * _num_local_freq * _sub_frame_samples * _sub_frame_index);
    args.output = device.get_gpu_memory_array(
        fmt::format(fmt("kvfft_{:d}"), _sub_frame_index), gpu_frame_id, 512*2*sizeof(float));//*8*_samples_per_data_set);

    // Copy kernel args into correct location for GPU
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));

    //int num_ffts = _num_elements * _samples_per_data_set / 256;

    // Set kernel dims
    kernelParams params;
    params.workgroup_size_x = 64;
    params.workgroup_size_y = 1;
    params.workgroup_size_z = 1;
    params.grid_size_x = 64;//*(_num_elements / 256); //full spatial array
    params.grid_size_y = 1;//_samples_per_data_set ; //single time steps
    params.grid_size_z = 1;
    params.num_dims = 2;

    // Should this be zero?
    params.private_segment_size = 0;
    params.group_segment_size = 0;

    signals[gpu_frame_id] = enqueue_kernel(params, gpu_frame_id);

    return signals[gpu_frame_id];
}

void hsaKVFFT::finalize_frame(int frame_id) {
//    static int loop=0;
//    if (loop++>=5) {
//        kotekan_hsa_stop();
//        exit(0);
//    }
//    return hsaSubframeCommand::finalize_frame(frame_id);

    unsigned char *cpu_in = (unsigned char *) hsa_host_malloc(256 * sizeof(char), 0);
    void *gpu_in = (void*)((uint8_t*)device.get_gpu_memory_array("input", frame_id, input_frame_len));
    device.sync_copy_gpu_to_host((void*)cpu_in, gpu_in, 256 * sizeof(char));

    float *cpu_out = (float *) hsa_host_malloc(512 * 2 * sizeof(float), 0);
    void *gpu_out = device.get_gpu_memory_array(
            fmt::format(fmt("kvfft_{:d}"), _sub_frame_index), frame_id, 512*2*sizeof(float));
    device.sync_copy_gpu_to_host((void*)cpu_out,gpu_out, 512 * 2 * sizeof(float));

    float stage_res[10][512][2];

    //initialize CPU calc buffers & do 0th stage
    memset(stage_res, 0, 9*512*sizeof(float));
    for (int i=0; i<256; i++) {
        float im = ((cpu_in[i % 256] & 0x0f)) - 8;
        float re = ((cpu_in[i % 256] & 0xf0) >> 4) - 8;
        stage_res[0][i][0] = im;
        stage_res[0][i][1] = re;

        float twiddle = 2 * 3.1415926535 / 512 * i;
        float ia = sin(twiddle);
        float ra = cos(twiddle);
        stage_res[0][i + 256][0] = im*ra + re*ia;
        stage_res[0][i + 256][1] = re*ra - im*ia;
    }

    int endstage=8;
    //stages
    for (int st = 1; st<9; st++) {
        int s = 9-st-1;
        for (int pp = 0; pp < 256; pp++) {
            int addr_i[2] = {pp + ((pp>>s)<<s),
                             pp + ((pp>>s)<<s) + (1<<s)};
            int addr_o[2] = {2 * pp, 2 * pp + 1};
            if (st != endstage) {
                addr_o[0] = addr_i[0];
                addr_o[1] = addr_i[1];
            }

            stage_res[st][addr_o[0]][0] = stage_res[st-1][addr_i[0]][0] + stage_res[st-1][addr_i[1]][0];
            stage_res[st][addr_o[0]][1] = stage_res[st-1][addr_i[0]][1] + stage_res[st-1][addr_i[1]][1];
            float twiddle = 2 * 3.1415926535 / 512 * pp * (1<<st);
            float ia = sin(twiddle);
            float ra = cos(twiddle);
            float re,im;
            im = stage_res[st-1][addr_i[0]][0] - stage_res[st-1][addr_i[1]][0];
            re = stage_res[st-1][addr_i[0]][1] - stage_res[st-1][addr_i[1]][1];
            stage_res[st][addr_o[1]][0] = re*ia + im*ra;
            stage_res[st][addr_o[1]][1] = re*ra - im*ia;
        }
    }

    int length = 512;
    fftwf_complex *data = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * length);
    fftwf_complex *fdata = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * length);
    fftwf_plan_s *fft_plan = (fftwf_plan_s*)fftwf_plan_dft_1d(length, data, fdata, -1, FFTW_ESTIMATE);

    for (int i = 0; i < length/2; i++) {
        data[i][0] = stage_res[0][i][1];
        data[i][1] = stage_res[0][i][0];
    }
    for (int i=length/2; i< length; i++) {
        data[i][0] = 0;
        data[i][1] = 0;
    }
    fftwf_execute(fft_plan);

    int wrongct = 0;
    for (int i=0; i<512; i++){
//        int st=endstage;
//        if ((stage_res[st][i][0] - cpu_out[i*2])>1e-2 || (stage_res[st][i][1] - cpu_out[i*2+1])>1e-2) {
//            printf("%3i - %6.2f, %6.2f : %6.2f %6.2f \n",i, stage_res[st][i][0], stage_res[st][i][1], cpu_out[i*2],cpu_out[i*2+1]);
        if (((fdata[i][1] - cpu_out[i*2])>0.01) || ((fdata[i][0] - cpu_out[i*2+1])>0.01)){
            printf("%3i - %6.2f, %6.2f : %6.2f %6.2f \n",i, fdata[i][1], fdata[i][0], cpu_out[i*2],cpu_out[i*2+1]);
            wrongct++;
        }
    }
    if (wrongct == 0) printf("Matches!\n");
    else printf("Does not match!\n");

    fftwf_free(data);
    fftwf_free(fdata);
    fftwf_free(fft_plan);

    free(cpu_in);
    free(cpu_out);
}