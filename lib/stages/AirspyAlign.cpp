#include "AirspyAlign.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"          // for Buffer, mark_frame_empty, mark_frame_full, register_consumer
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for DEBUG
#include "restServer.hpp"      // for connectionInstance

#include <atomic>      // for atomic_bool
#include <exception>   // for exception
#include <functional>  // for _Bind_helper<>::type, bind, function
#include <regex>       // for match_results<>::_Base_type
#include <stdexcept>   // for runtime_error
#include <stdint.h>    // for uint32_t
#include <stdlib.h>    // for calloc, free
#include <string.h>    // for memset
#include <sys/types.h> // for uint
#include <vector>      // for vector
#include "fftw3.h"


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(AirspyAlign);

AirspyAlign::AirspyAlign(Config& config, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&AirspyAlign::main_thread, this)) {

    buf_inA = get_buffer("in_bufA");
    register_consumer(buf_inA, unique_name.c_str());
//    buf_inB = get_buffer("in_bufB");
//    register_consumer(buf_inB, unique_name.c_str());
//    max_buf_lag = config.get_default<int>(unique_name, "max_buf_lag", 10); 

//    extra_bufA = malloc(buf_inA->frame_size*max_buf_lag);
//    extra_bufB = malloc(buf_inA->frame_size*max_buf_lag);
}

AirspyAlign::~AirspyAlign() {
//    free(extra_bufA);
//    free(extra_bufB);
}


void AirspyAlign::start_callback(kotekan::connectionInstance& conn) {
    nlohmann::json reply;
    reply["Going"] = 1;
    going = true;

    conn.send_json_reply(reply);
}


void AirspyAlign::main_thread() {
    std::string endpoint = unique_name+"/go";
    kotekan::restServer& rest_server = kotekan::restServer::instance();
    using namespace std::placeholders;
    rest_server.register_get_callback(endpoint,
                                      std::bind(&AirspyAlign::start_callback, this, _1));

    short* inA_local;
    short* inB_local;

    frame_inA = 0;
    frame_inB = 0;

    int samples_per_frame = buf_inA->frame_size / (sizeof(short));


    fftwf_complex *spectrumA, *spectrumB, *spectrumC;
    fftwf_complex *samplesA, *samplesB, *samplesC;
    fftwf_plan fft_planA, fft_planB, fft_planC;

    samplesA = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * samples_per_frame/2);
    spectrumA = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * samples_per_frame/2);
    fft_planA = (fftwf_plan_s*)fftwf_plan_dft_1d(samples_per_frame/2, samplesA, spectrumA, -1, FFTW_ESTIMATE);

    samplesB = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * samples_per_frame/2);
    spectrumB = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * samples_per_frame/2);
    fft_planB = (fftwf_plan_s*)fftwf_plan_dft_1d(samples_per_frame/2, samplesB, spectrumB, -1, FFTW_ESTIMATE);

    samplesC = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * samples_per_frame/2);
    spectrumC = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * samples_per_frame/2);
    fft_planC = (fftwf_plan_s*)fftwf_plan_dft_1d(samples_per_frame/2, samplesC, spectrumC, 1, FFTW_ESTIMATE);

    //assumes the arrays are aligned
    while (!stop_thread) {

        inA_local = (short*)wait_for_full_frame(buf_inA, unique_name.c_str(), frame_inA);
//        inB_local = (short*)wait_for_full_frame(buf_inB, unique_name.c_str(), frame_inB);

        if ((inA_local == nullptr))// || (inB_local == nullptr))
            break;

        if (going) {

            for (int i = 0; i < samples_per_frame/2; i++) {
                samplesA[i][0] = inA_local[2 * i];
                samplesA[i][1] = inA_local[2 * i + 1];
//                samplesB[i][0] = inB_local[2 * i];
//                samplesB[i][1] = inB_local[2 * i + 1];
                int o = 1234; //synthetic offset
                samplesB[i][0] = inA_local[2 * (i+o)%(samples_per_frame/2)];
                samplesB[i][1] = inA_local[2 * (i+o)%(samples_per_frame/2) + 1];
            }
            fftwf_execute(fft_planA);
            fftwf_execute(fft_planB);
            for (int i = 0; i < samples_per_frame/2; i++) {
                float Ar = spectrumA[0][i];
                float Ai = spectrumA[1][i];
                float Br = spectrumB[0][i];
                float Bi = spectrumB[1][i];
                //A \times B*
                samplesC[i][0] = Ar * Br + Ai * Bi;
                samplesC[i][1] = Ai * Br - Bi * Ar;
            }
            fftwf_execute(fft_planC);

            float maxval=0, meanval=0;
            int lag = 0;
            for (int i=0; i < samples_per_frame/2; i++){
                float Cr = spectrumC[0][i];
                float Ci = spectrumC[1][i];
                float val = Cr;// + Ci*Ci; //should be mag, but sorting by mag2 works just as well
                if (val > maxval) {
                    lag = i;
                    maxval = val;
                }
                meanval += val;
            }
            meanval /= samples_per_frame/2;
            DEBUG("Align: Max: {}, Mean: {}",maxval, meanval);
            if (maxval / meanval > 10) {
                DEBUG("Lag found!\n");
                DEBUG("Lag {}\n",lag);
                going=false;
            }
            else {
            }
        }

        mark_frame_empty(buf_inA, unique_name.c_str(), frame_inA);
//        mark_frame_empty(buf_inB, unique_name.c_str(), frame_inB);
        frame_inA = (frame_inA + 1) % buf_inA->num_frames;
//        frame_inB = (frame_inB + 1) % buf_inB->num_frames;

    }
}
