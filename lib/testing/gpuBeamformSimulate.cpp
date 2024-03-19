#include "gpuBeamformSimulate.hpp"

#include "Config.hpp"       // for Config
#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"
#include "buffer.hpp"          // for Buffer, mark_frame_empty, mark_frame_full, pass_metadata
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for ERROR, INFO

#include <algorithm>   // for copy
#include <assert.h>    // for assert
#include <atomic>      // for atomic_bool
#include <cmath>       // for sin, cos, asin, floor, pow
#include <cstdint>     // for int32_t
#include <exception>   // for exception
#include <functional>  // for _Bind_helper<>::type, bind, function
#include <regex>       // for match_results<>::_Base_type
#include <stdexcept>   // for runtime_error
#include <stdio.h>     // for fclose, fopen, fread, snprintf, FILE
#include <stdlib.h>    // for free, malloc
#include <string.h>    // for memcpy
#include <sys/types.h> // for uint


#define SWAP(a, b)                                                                                 \
    tempr = (a);                                                                                   \
    (a) = (b);                                                                                     \
    (b) = tempr
#define HI_NIBBLE(b) (((b) >> 4) & 0x0F)
#define LO_NIBBLE(b) ((b)&0x0F)

#define PI 3.14159265
#define feed_sep 0.3048
#define light 299792458.

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(gpuBeamformSimulate);

gpuBeamformSimulate::gpuBeamformSimulate(Config& config, const std::string& unique_name,
                                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&gpuBeamformSimulate::main_thread, this)) {

    // Apply config.
    _num_elements = config.get<int32_t>(unique_name, "num_elements");
    _samples_per_data_set = config.get<int32_t>(unique_name, "samples_per_data_set");
    _factor_upchan = config.get<int32_t>(unique_name, "factor_upchan");
    _downsample_time = config.get<int32_t>(unique_name, "downsample_time");
    _downsample_freq = config.get<int32_t>(unique_name, "downsample_freq");
    _reorder_map = config.get<std::vector<int32_t>>(unique_name, "reorder_map");
    _northmost_beam = config.get<float>(unique_name, "northmost_beam");
    Freq_ref = (light * (128) / (sin(_northmost_beam * PI / 180.) * feed_sep * 256)) / 1.e6;
    _ew_spacing = config.get<std::vector<float>>(unique_name, "ew_spacing");
    _factor_upchan = config.get<uint32_t>(unique_name, "factor_upchan");
    _num_frb_total_beams = config.get<int32_t>(unique_name, "num_frb_total_beams");
    _ew_spacing_c = (float*)malloc(4 * sizeof(float));
    for (int i = 0; i < 4; i++) {
        _ew_spacing_c[i] = _ew_spacing[i];
    }
    _gain_dir = config.get<std::string>(unique_name, "gain_dir");
    std::vector<float> dg = {0.0, 0.0}; // re,im
    default_gains = config.get_default<std::vector<float>>(unique_name, "frb_missing_gains", dg);
    scaling = config.get_default<float>(unique_name, "frb_scaling", 1.0);

    input_buf = get_buffer("network_in_buf");
    input_buf->register_consumer(unique_name);
    output_buf = get_buffer("beam_out_buf");
    output_buf->register_producer(unique_name);

    hfb_output_buf = get_buffer("hfb_out_buf");
    hfb_output_buf->register_producer(unique_name);

    input_len = _samples_per_data_set * _num_elements * 2;
    input_len_padded = input_len * 2;
    transposed_len = (_samples_per_data_set + 32) * _num_elements * 2;
    output_len = _num_elements * (_samples_per_data_set / _downsample_time / _downsample_freq / 2);
    hfb_output_len = _num_frb_total_beams * _factor_upchan;

    input_unpacked = (double*)malloc(input_len * sizeof(double));
    input_unpacked_padded = (double*)malloc(input_len_padded * sizeof(double));
    clamping_output = (double*)malloc(input_len * sizeof(double));
    cpu_beamform_output = (double*)malloc(input_len * sizeof(double));
    transposed_output = (double*)malloc(transposed_len * sizeof(double));
    tmp128 = (double*)malloc(_factor_upchan * 2 * sizeof(double));
    cpu_final_output = (float*)malloc(output_len * sizeof(float));
    cpu_hfb_final_output = (float*)malloc(hfb_output_len * sizeof(float));

    cpu_gain = (float*)malloc(2 * 2048 * sizeof(float));

    coff = (float*)malloc(16 * 2 * sizeof(float));
    assert(coff != nullptr);

    // Backward compatibility, array in c
    reorder_map_c = (int*)malloc(512 * sizeof(int));
    for (uint i = 0; i < 512; ++i) {
        reorder_map_c[i] = _reorder_map[i];
    }

    metadata_buf = get_buffer("network_in_buf");
    metadata_buffer_id = 0;
    freq_now = FREQ_ID_NOT_SET;
    freq_MHz = -1;
}

gpuBeamformSimulate::~gpuBeamformSimulate() {
    free(input_unpacked_padded);
    free(input_unpacked);
    free(clamping_output);
    free(cpu_beamform_output);
    free(coff);
    free(cpu_gain);
    free(transposed_output);
    free(tmp128);
    free(cpu_final_output);
    free(cpu_hfb_final_output);
    free(reorder_map_c);
    free(_ew_spacing_c);
}

void gpuBeamformSimulate::reorder(unsigned char* data, int* map) {
    tmp512 = (int*)malloc(2048 * sizeof(int));
    for (int j = 0; j < _samples_per_data_set; j++) {
        for (int i = 0; i < 512; i++) {
            int id = map[i];
            tmp512[i * 4] = data[j * 2048 + (id * 4)];
            tmp512[i * 4 + 1] = data[j * 2048 + (id * 4 + 1)];
            tmp512[i * 4 + 2] = data[j * 2048 + (id * 4 + 2)];
            tmp512[i * 4 + 3] = data[j * 2048 + (id * 4 + 3)];
        }
        for (int i = 0; i < 2048; i++) {
            data[j * 2048 + i] = tmp512[i];
        }
    }
    free(tmp512);
}

void gpuBeamformSimulate::cpu_beamform_ns(double* data, uint64_t transform_length, int stop_level) {
    uint64_t n, m, j, i;
    double wr, wi, theta;
    double tempr, tempi;
    n = transform_length << 1;
    j = 1;

    for (i = 1; i < n; i += 2) { /* This is the bit-reversal section of the routine. */
        if (j > i) {
            SWAP(data[j - 1], data[i - 1]); /* Exchange the two complex numbers. */
            SWAP(data[j], data[i]);
        }
        m = transform_length;
        while (m >= 2 && j > m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
    uint64_t step_stop;
    if (stop_level < -1) // neg values mean do the whole sequence; the last stage has pairs half the
                         // transform length apart
        step_stop = transform_length / 2;
    else
        step_stop = pow(2, stop_level);

    for (uint64_t step_size = 1; step_size <= step_stop; step_size += step_size) {
        theta = -3.141592654 / (step_size);
        for (uint64_t index = 0; index < transform_length; index += step_size * 2) {
            for (uint32_t minor_index = 0; minor_index < step_size; minor_index++) {
                wr = cos(minor_index * theta);
                wi = sin(minor_index * theta);
                int first_index = (index + minor_index) * 2;
                int second_index = first_index + step_size * 2;
                tempr = wr * data[second_index] - wi * data[second_index + 1];
                tempi = wi * data[second_index] + wr * data[second_index + 1];
                data[second_index] = data[first_index] - tempr;
                data[second_index + 1] = data[first_index + 1] - tempi;
                data[first_index] += tempr;
                data[first_index + 1] += tempi;
            }
        }
    }
}

void gpuBeamformSimulate::cpu_beamform_ew(double* input, double* output, float* Coeff, int nbeamsNS,
                                          int nbeamsEW, int npol, int nsamp_in) {

    int elm_now, out_add;
    for (int t = 0; t < nsamp_in; t++) {
        for (int p = 0; p < npol; p++) {
            for (int bEW = 0; bEW < nbeamsEW; bEW++) {
                for (int bNS = 0; bNS < nbeamsNS; bNS++) {
                    out_add = (t * 2048 + p * 1024 + bEW * 256 + bNS) * 2;
                    for (int elm = 0; elm < 4; elm++) {
                        elm_now = (t * 2048 + p * 1024 + elm * 256 + bNS) * 2;

                        // REAL
                        output[out_add] += input[elm_now] * Coeff[2 * (bEW * 4 + elm)]
                                           + input[elm_now + 1] * Coeff[2 * (bEW * 4 + elm) + 1];
                        // IMAG
                        output[out_add + 1] += input[elm_now] * Coeff[2 * (bEW * 4 + elm) + 1]
                                               - input[elm_now + 1] * Coeff[2 * (bEW * 4 + elm)];
                    }
                    output[out_add] = output[out_add] / 4.;
                    output[out_add + 1] = output[out_add + 1] / 4.;
                }
            }
        }
    }
}

void gpuBeamformSimulate::clamping(double* input, double* output, float freq, int nbeamsNS,
                                   int nbeamsEW, int nsamp_in, int npol) {
    float t, delta_t, Beam_Ref;
    int cl_index;
    float D2R = PI / 180.;
    int pad = 2;
    int tile = 1;
    int nbeams = nbeamsEW * nbeamsNS;
    for (int b = 0; b < nbeamsNS; b++) {
        Beam_Ref = asin(light * (b - nbeamsNS / 2.) / (Freq_ref * 1.e6) / (nbeamsNS) / feed_sep)
                   * 180. / PI;
        t = nbeamsNS * pad * (Freq_ref * 1.e6) * (feed_sep / light * sin(Beam_Ref * D2R)) + 0.5;
        delta_t = nbeamsNS * pad * (freq * 1e6 - Freq_ref * 1e6)
                  * (feed_sep / light * sin(Beam_Ref * D2R));
        cl_index = (int)floor(t + delta_t) + nbeamsNS * tile * pad / 2.;

        if (cl_index < 0)
            cl_index = 256 * pad + cl_index;
        else if (cl_index > 256 * pad)
            cl_index = cl_index - 256 * pad;
        cl_index = cl_index - 256;
        if (cl_index < 0)
            cl_index = 256 * pad + cl_index;

        for (int i = 0; i < nsamp_in; i++) {
            for (int p = 0; p < npol; p++) {
                for (int b2 = 0; b2 < nbeamsEW; b2++) {
                    // flip N-S by writing to (255-b)
                    output[2
                           * (i * npol * nbeamsNS * nbeamsEW + p * nbeams + b2 * nbeamsNS
                              + (255 - b))] =
                        input[2 * (i * 2048 * 2 + p * 1024 * 2 + b2 * 512 + cl_index)];
                    output[2
                               * (i * npol * nbeamsNS * nbeamsEW + p * nbeams + b2 * nbeamsNS
                                  + (255 - b))
                           + 1] =
                        input[2 * (i * 2048 * 2 + p * 1024 * 2 + b2 * 512 + cl_index) + 1];
                }
            }
        }
    }
}

void gpuBeamformSimulate::transpose(double* input, double* output, int n_beams, int n_samp) {
    for (int j = 0; j < n_samp; j++) {
        for (int i = 0; i < n_beams; i++) {
            output[(i * (n_samp + 32) + j) * 2] = input[(j * n_beams + i) * 2];
            output[(i * (n_samp + 32) + j) * 2 + 1] = input[(j * n_beams + i) * 2 + 1];
            for (int k = 0; k < 32; k++) {
                output[(i * (n_samp + 32) + (n_samp + k)) * 2] = 0.0;
                output[(i * (n_samp + 32) + (n_samp + k)) * 2 + 1] = 0.0;
            }
        }
    }
}

void gpuBeamformSimulate::upchannelize(double* data, int nn) {
    unsigned long n, mmax, m, j, istep, i;
    double wtemp, wr, wpr, wpi, wi, theta;
    double tempr, tempi;
    n = nn << 1;
    j = 1;
    for (i = 1; i < n; i += 2) { /* bit-reversal section of the routine. */
        if (j > i) {
            SWAP(data[j - 1], data[i - 1]); /* Exchange two complex numbers. */
            SWAP(data[j], data[i]);
        }
        m = nn;
        while (m >= 2 && j > m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
    mmax = 2;
    while (n > mmax) { /* Outer loop executed log2 nn times. */
        istep = mmax << 1;
        theta = (6.28318530717959 / mmax); /* Initialize the trigonometric recurrence. */
        wtemp = sin(0.5 * theta);
        wpr = -2.0 * wtemp * wtemp;
        wpi = sin(theta);
        wr = 1.0;
        wi = 0.0;
        for (m = 1; m < mmax; m += 2) { /* two nested inner loops. */
            for (i = m; i <= n; i += istep) {
                j = i + mmax; /* This is the Danielson-Lanczos formula. */
                tempr = wr * data[j - 1] - wi * data[j];
                tempi = wr * data[j] + wi * data[j - 1];
                data[j - 1] = data[i - 1] - tempr;
                data[j] = data[i] - tempi;
                data[i - 1] += tempr;
                data[i] += tempi;
            }
            wr = (wtemp = wr) * wpr - wi * wpi + wr; /* Trigonometric recurrence. */
            wi = wi * wpr + wtemp * wpi + wi;
        }
        mmax = istep;
    }
}

void gpuBeamformSimulate::main_thread() {

    auto& tel = Telescope::instance();

    int input_buf_id = 0;
    int output_buf_id = 0;

    int npol = 2;
    int nbeamsEW = 4;
    int nbeamsNS = 256;
    int nbeams = nbeamsEW * nbeamsNS;

    while (!stop_thread) {
        unsigned char* input =
            (unsigned char*)input_buf->wait_for_full_frame(unique_name, input_buf_id);
        if (input == nullptr)
            break;

        float* output = (float*)output_buf->wait_for_empty_frame(unique_name, output_buf_id);

        if (output == nullptr)
            break;

        float* hfb_output =
            (float*)hfb_output_buf->wait_for_empty_frame(unique_name, output_buf_id);

        if (hfb_output == nullptr)
            break;

        for (int i = 0; i < input_len; i++) {
            cpu_beamform_output[i] = 0.0; // Need this
            clamping_output[i] = 0.0;     // Maybe don't need this
        }
        for (int i = 0; i < transposed_len; i++) {
            transposed_output[i] = 0.0; // Maybe don't need this
        }
        for (int i = 0; i < output_len; i++) {
            cpu_final_output[i] = 0.0;
        }

        for (int i = 0; i < hfb_output_len; i++)
            cpu_hfb_final_output[i] = 0.f;

        // TODO adjust to allow for more than one frequency.
        // TODO remove all the 32's in here with some kind of constant/define
        INFO("Simulating GPU beamform processing for {:s}[{:d}] putting result in {:s}[{:d}]",
             input_buf->buffer_name, input_buf_id, output_buf->buffer_name, output_buf_id);

        INFO(
            "Simulating GPU hyper fine beam processing for {:s}[{:d}] putting result in {:s}[{:d}]",
            input_buf->buffer_name, input_buf_id, hfb_output_buf->buffer_name, output_buf_id);

        freq_now = tel.to_freq_id(metadata_buf, metadata_buffer_id);
        freq_MHz = tel.to_freq(freq_now);

        // Work out the EW phase coefficients from freq_MHz
        for (int angle_iter = 0; angle_iter < 4; angle_iter++) {
            double anglefrac = sin(_ew_spacing[angle_iter] * PI / 180.);
            for (int cylinder = 0; cylinder < 4; cylinder++) {
                coff[angle_iter * 4 * 2 + cylinder * 2] =
                    cos(2 * PI * anglefrac * cylinder * 22 * freq_MHz * 1.e6 / light);
                coff[angle_iter * 4 * 2 + cylinder * 2 + 1] =
                    sin(2 * PI * anglefrac * cylinder * 22 * freq_MHz * 1.e6 / light);
            }
        }

        FILE* ptr_myfile = nullptr;
        char filename[512];
        snprintf(filename, sizeof(filename), "%s/quick_gains_%04d_reordered.bin", _gain_dir.c_str(),
                 freq_now);
        ptr_myfile = fopen(filename, "rb");

        if (ptr_myfile == nullptr) {
            ERROR("CPU verification code: Cannot open gain file {:s}", filename);
            for (int i = 0; i < 2048; i++) {
                cpu_gain[i * 2] = default_gains[0] * scaling;
                cpu_gain[i * 2 + 1] = default_gains[1] * scaling;
            }
        } else {
            uint32_t read_length = sizeof(float) * 2 * 2048;
            if (fread(cpu_gain, read_length, 1, ptr_myfile) != 1) {
                ERROR("Couldn't read gain file...");
            }
            for (uint32_t i = 0; i < 2048; i++) {
                cpu_gain[i * 2] = cpu_gain[i * 2] * scaling;
                cpu_gain[i * 2 + 1] = cpu_gain[i * 2] * scaling;
            }
            fclose(ptr_myfile);
        }

        // Reorder
        reorder(input, reorder_map_c);

        // Unpack and pad the input data
        int dest_idx = 0;
        for (size_t i = 0; i < input_buf->frame_size; ++i) {
            input_unpacked[dest_idx++] = HI_NIBBLE(input[i]) - 8;
            input_unpacked[dest_idx++] = LO_NIBBLE(input[i]) - 8;
        }

        // Pad to 512
        // TODO this can be simplified a fair bit.
        int index = 0;
        for (int j = 0; j < _samples_per_data_set; j++) {
            for (int p = 0; p < npol; p++) {
                for (int b = 0; b < nbeamsEW; b++) {
                    for (int i = 0; i < 512; i++) {
                        if (i < 256) {
                            // Real
                            input_unpacked_padded[index++] =
                                input_unpacked[2
                                               * (j * npol * nbeams + p * nbeams + b * nbeamsNS
                                                  + i)]
                                    * cpu_gain[(p * nbeams + b * nbeamsNS + i) * 2]
                                + input_unpacked[2
                                                     * (j * npol * nbeams + p * nbeams
                                                        + b * nbeamsNS + i)
                                                 + 1]
                                      * cpu_gain[(p * nbeams + b * nbeamsNS + i) * 2 + 1];
                            // Imag
                            input_unpacked_padded[index++] =
                                input_unpacked[2
                                                   * (j * npol * nbeams + p * nbeams + b * nbeamsNS
                                                      + i)
                                               + 1]
                                    * cpu_gain[(p * nbeams + b * nbeamsNS + i) * 2]
                                - input_unpacked[2
                                                 * (j * npol * nbeams + p * nbeams + b * nbeamsNS
                                                    + i)]
                                      * cpu_gain[(p * nbeams + b * nbeamsNS + i) * 2 + 1];
                        } else {
                            input_unpacked_padded[index++] = 0;
                            input_unpacked_padded[index++] = 0;
                        }
                    }
                }
            }
        }

        // Beamform north south.
        for (int i = 0; i < _samples_per_data_set * npol * nbeamsEW; i++) {
            cpu_beamform_ns(&input_unpacked_padded[i * 512 * 2], 512, 8);
        }

        // Clamp the data
        clamping(input_unpacked_padded, clamping_output, freq_MHz, nbeamsNS, nbeamsEW,
                 _samples_per_data_set, npol);

        // EW brute force beamform
        cpu_beamform_ew(clamping_output, cpu_beamform_output, coff, nbeamsNS, nbeamsEW, npol,
                        _samples_per_data_set);

        // transpose
        transpose(cpu_beamform_output, transposed_output, _num_elements, _samples_per_data_set);

        // Upchannelize; re-use cpu_beamform_output
        for (int b = 0; b < _num_elements; b++) {
            for (int n = 0; n < _samples_per_data_set / _factor_upchan; n++) {
                int index = 0;
                for (int i = 0; i < _factor_upchan; i++) {
                    tmp128[index++] = transposed_output[(b * (_samples_per_data_set + 32)
                                                         + n * _factor_upchan + i)
                                                        * 2];
                    tmp128[index++] = transposed_output
                        [(b * (_samples_per_data_set + 32) + n * _factor_upchan + i) * 2 + 1];
                }
                upchannelize(tmp128, _factor_upchan);
                for (int i = 0; i < _factor_upchan; i++) {
                    cpu_beamform_output[(b * _samples_per_data_set + n * _factor_upchan + i) * 2] =
                        tmp128[i * 2];
                    cpu_beamform_output[(b * _samples_per_data_set + n * _factor_upchan + i) * 2
                                        + 1] = tmp128[i * 2 + 1];
                }
            }
        }

        // 16-bandpass correction
        float BP[16]{0.52225748, 0.58330915, 0.6868705,  0.80121821, 0.89386546, 0.95477358,
                     0.98662733, 0.99942558, 0.99988676, 0.98905127, 0.95874124, 0.90094667,
                     0.81113021, 0.6999944,  0.59367968, 0.52614263};
        float HFB_BP[16] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
                            1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};

        // Downsample
        int nfreq_out = _factor_upchan;
        int nsamp_out = _samples_per_data_set / _factor_upchan / _downsample_time;

        // Loop over every beam
        for (int b = 0; b < 1024; b++) {
            for (int f = 0; f < nfreq_out; f++) {
                float total_sum = 0.0;
                for (int t = 0; t < nsamp_out; t++) {

                    float tmp_real = 0.f, tmp_imag = 0.f, out_sq = 0.f;
                    for (int pp = 0; pp < npol; pp++) {
                        for (int tt = 0; tt < 3; tt++) {
                            const int sample_offset =
                                (pp * 1024 * _samples_per_data_set + b * _samples_per_data_set
                                 + (t * _downsample_time + tt) * _factor_upchan + f)
                                * 2;

                            tmp_real = cpu_beamform_output[sample_offset];
                            tmp_imag = cpu_beamform_output[sample_offset + 1];

                            out_sq += tmp_real * tmp_real + tmp_imag * tmp_imag;
                        } // end for tt
                    }     // end for pol
                    total_sum += out_sq / 6.f / HFB_BP[int((f + 8) % 16)];
                } // end for nsamp

                // JSW TODO: apply bandpass filter
                const int output_offset = b * nfreq_out + ((f + 64) % 128);
                cpu_hfb_final_output[output_offset] = total_sum;
            } // end for freq
        }     // end for beam

        memcpy(hfb_output, cpu_hfb_final_output, hfb_output_buf->frame_size);

        // Downsample
        nfreq_out = _factor_upchan / _downsample_freq;
        nsamp_out = _samples_per_data_set / _factor_upchan / _downsample_time;
        for (int b = 0; b < 1024; b++) {
            for (int t = 0; t < nsamp_out; t++) {
                for (int f = 0; f < nfreq_out; f++) {
                    // FFT shift by (id+8)%16
                    int out_id = b * nsamp_out * nfreq_out + t * nfreq_out + ((f + 8) % 16);
                    float tmp_real = 0.0;
                    float tmp_imag = 0.0;
                    float out_sq = 0.0;
                    for (int pp = 0; pp < npol; pp++) {
                        for (int tt = 0; tt < _downsample_time; tt++) {
                            for (int ff = 0; ff < _downsample_freq; ff++) {
                                tmp_real = cpu_beamform_output[(pp * 1024 * _samples_per_data_set
                                                                + b * _samples_per_data_set
                                                                + (t * _downsample_time + tt)
                                                                      * _factor_upchan
                                                                + (f * _downsample_freq + ff))
                                                               * 2];
                                tmp_imag = cpu_beamform_output[(pp * 1024 * _samples_per_data_set
                                                                + b * _samples_per_data_set
                                                                + (t * _downsample_time + tt)
                                                                      * _factor_upchan
                                                                + (f * _downsample_freq + ff))
                                                                   * 2
                                                               + 1];
                                out_sq += tmp_real * tmp_real + tmp_imag * tmp_imag;
                            } // end for ff
                        }     // end for tt
                    }         // end for pol
                    cpu_final_output[out_id] = out_sq / 48. / BP[int((f + 8) % 16)];

                } // end for freq
            }     // end for time
        }         // end for beam

        memcpy(output, cpu_final_output, output_buf->frame_size);

        INFO("Simulating GPU beamform processing done for {:s}[{:d}] result is in {:s}[{:d}]",
             input_buf->buffer_name, input_buf_id, output_buf->buffer_name, output_buf_id);

        input_buf->pass_metadata(input_buf_id, output_buf, output_buf_id);
        input_buf->mark_frame_empty(unique_name, input_buf_id);
        output_buf->mark_frame_full(unique_name, output_buf_id);
        hfb_output_buf->mark_frame_full(unique_name, output_buf_id);

        input_buf_id = (input_buf_id + 1) % input_buf->num_frames;
        metadata_buffer_id = (metadata_buffer_id + 1) % metadata_buf->num_frames;
        output_buf_id = (output_buf_id + 1) % output_buf->num_frames;
    }
}
