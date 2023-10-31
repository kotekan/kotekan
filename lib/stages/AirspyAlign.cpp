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
    buf_inB = get_buffer("in_bufB");
    register_consumer(buf_inB, unique_name.c_str());

    localA = (short *)malloc(buf_inA->frame_size);
    localB = (short *)malloc(buf_inB->frame_size);
}

AirspyAlign::~AirspyAlign() {
    free(localA);
    free(localB);
}


void AirspyAlign::start_callback(kotekan::connectionInstance& conn) {
    nlohmann::json reply;
    going = true;

    while (going) {usleep(1000);}

    int samples_per_frame = buf_inA->frame_size / (sizeof(short));

    fftwf_complex *spectrumA, *spectrumB, *spectrumC;
    fftwf_complex *samplesA, *samplesB, *samplesC;
    fftwf_plan fft_planA, fft_planB, fft_planC;

    samplesA = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * samples_per_frame/2);
    spectrumA = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * samples_per_frame/2);
    fft_planA = (fftwf_plan_s*)fftwf_plan_dft_1d(samples_per_frame/2, samplesA, spectrumA, FFTW_FORWARD, FFTW_ESTIMATE);

    samplesB = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * samples_per_frame/2);
    spectrumB = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * samples_per_frame/2);
    fft_planB = (fftwf_plan_s*)fftwf_plan_dft_1d(samples_per_frame/2, samplesB, spectrumB, FFTW_FORWARD, FFTW_ESTIMATE);

    samplesC = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * samples_per_frame/2);
    spectrumC = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * samples_per_frame/2);
    fft_planC = (fftwf_plan_s*)fftwf_plan_dft_1d(samples_per_frame/2, samplesC, spectrumC, FFTW_BACKWARD, FFTW_ESTIMATE);


    nlohmann::json a = nlohmann::json::array();
    nlohmann::json b = nlohmann::json::array();
    for (int i = 0; i < samples_per_frame/2; i++) {
        samplesA[i][0] = localA[2 * i];
        samplesA[i][1] = localA[2 * i + 1];
        samplesB[i][0] = localB[2 * i];
        samplesB[i][1] = localB[2 * i + 1];
//        int o = 1234; //synthetic offset
//        samplesB[i][0] = inA_local[2 * (i+o)%(samples_per_frame/2)];
//        samplesB[i][1] = inA_local[2 * (i+o)%(samples_per_frame/2) + 1];

        a.push_back(samplesA[i][0]);
        a.push_back(samplesA[i][1]);
        b.push_back(samplesB[i][0]);
        b.push_back(samplesB[i][1]);
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

    float maxval=0, meanval=0, var=0;
    int lag = 0;
    nlohmann::json corr = nlohmann::json::array();
    for (int i=0; i < samples_per_frame/2; i++){
        float val = sqrt(spectrumC[0][i]*spectrumC[0][i] + spectrumC[1][i]*spectrumC[1][i]);
        corr.push_back(val);
        if (val > maxval) {
            lag = i;
            maxval = val;
        }
        meanval += val;
    }
    meanval /= samples_per_frame/2;
    for (int i=0; i < samples_per_frame/2; i++){
        float val = spectrumC[0][i]-meanval;
        var += val*val;
    }
    var /= samples_per_frame/2;
    float std = sqrt(var);
    DEBUG("Align: Max: {}, Mean: {}, RMS: {}",maxval, meanval, std);
    if (maxval > meanval+5*std) {
        DEBUG("Lag found!\n");
        DEBUG("Lag {}\n",lag);
        going=false;
    }
    else {
    }

    fftwf_free(samplesA);
    fftwf_free(samplesB);
    fftwf_free(spectrumA);
    fftwf_free(spectrumC);

    reply["lag"] = lag;
//    reply["a"] = a;
//    reply["b"] = b;
//    reply["corr"] = corr;
    conn.send_json_reply(reply);
}


void AirspyAlign::main_thread() {
    std::string endpoint = unique_name+"/go";
    kotekan::restServer& rest_server = kotekan::restServer::instance();
    using namespace std::placeholders;
    rest_server.register_get_callback(endpoint,
                                      std::bind(&AirspyAlign::start_callback, this, _1));

    short* inA;
    short* inB;

    frame_inA = 0;
    frame_inB = 0;

    //assumes the arrays are aligned
    while (!stop_thread) {

        inA = (short*)wait_for_full_frame(buf_inA, unique_name.c_str(), frame_inA);
        inB = (short*)wait_for_full_frame(buf_inB, unique_name.c_str(), frame_inB);

        if ((inA == nullptr) || (inB == nullptr))
            break;

        if (going) {
            memcpy(localA, inA, buf_inA->frame_size);
            memcpy(localB, inB, buf_inB->frame_size);
            going=false;
        }

        mark_frame_empty(buf_inA, unique_name.c_str(), frame_inA);
        mark_frame_empty(buf_inB, unique_name.c_str(), frame_inB);
        frame_inA = (frame_inA + 1) % buf_inA->num_frames;
        frame_inB = (frame_inB + 1) % buf_inB->num_frames;

    }
}
