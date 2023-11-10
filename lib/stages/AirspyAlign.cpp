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
#include <unistd.h>    // for usleep
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

    lag_window = config.get_default<unsigned int>(unique_name, "lag_window", buf_inA->frame_size/sizeof(short)); //samples
    localA = (short *)malloc(lag_window*sizeof(short));
    localB = (short *)malloc(lag_window*sizeof(short));
}

AirspyAlign::~AirspyAlign() {
    free(localA);
    free(localB);
}


void AirspyAlign::start_callback(kotekan::connectionInstance& conn) {
    nlohmann::json reply;
    going = true;

    while (going) {usleep(1000);}

    nlohmann::json a = nlohmann::json::array();
    nlohmann::json b = nlohmann::json::array();
    nlohmann::json fa = nlohmann::json::array();
    nlohmann::json fb = nlohmann::json::array();
    nlohmann::json fc = nlohmann::json::array();

#ifdef IQ_SAMPLING
    fftwf_complex *spectrumA, *spectrumB, *spectrumC;
    fftwf_complex *samplesA, *samplesB, *samplesC;
    fftwf_plan fft_planA, fft_planB, fft_planC;

    samplesA = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * lag_window/2);
    spectrumA = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * lag_window/2);
    fft_planA = (fftwf_plan_s*)fftwf_plan_dft_1d(lag_window/2, samplesA, spectrumA, FFTW_FORWARD, FFTW_ESTIMATE);

    samplesB = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * lag_window/2);
    spectrumB = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * lag_window/2);
    fft_planB = (fftwf_plan_s*)fftwf_plan_dft_1d(lag_window/2, samplesB, spectrumB, FFTW_FORWARD, FFTW_ESTIMATE);

    samplesC = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * lag_window/2);
    spectrumC = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * lag_window/2);
    fft_planC = (fftwf_plan_s*)fftwf_plan_dft_1d(lag_window/2, spectrumC, samplesC, FFTW_BACKWARD, FFTW_ESTIMATE);

    for (unsigned int i = 0; i < lag_window/2; i++) {
        samplesA[i][0] = localA[2 * i];
        samplesA[i][1] = localA[2 * i + 1];
        samplesB[i][0] = localB[2 * i];
        samplesB[i][1] = localB[2 * i + 1];
        //int o = 1234; //synthetic offset
        //samplesB[i][0] = inA_local[2 * (i+o)%(samples_per_frame/2)];
        //samplesB[i][1] = inA_local[2 * (i+o)%(samples_per_frame/2) + 1];
        a.push_back(samplesA[i][0]);
        a.push_back(samplesA[i][1]);
        b.push_back(samplesB[i][0]);
        b.push_back(samplesB[i][1]);
    }
#else
    fftwf_complex *spectrumA, *spectrumB, *spectrumC;
    float *samplesA, *samplesB, *samplesC;
    fftwf_plan fft_planA, fft_planB, fft_planC;

    samplesA = (float*)fftwf_malloc(sizeof(float) * lag_window);
    spectrumA = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * lag_window/2+1);
    fft_planA = (fftwf_plan_s*)fftwf_plan_dft_r2c_1d(lag_window, samplesA, spectrumA, FFTW_ESTIMATE);

    samplesB = (float*)fftwf_malloc(sizeof(float) * lag_window);
    spectrumB = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * lag_window/2+1);
    fft_planB = (fftwf_plan_s*)fftwf_plan_dft_r2c_1d(lag_window, samplesB, spectrumB, FFTW_ESTIMATE);

    samplesC = (float*)fftwf_malloc(sizeof(float) * lag_window);
    spectrumC = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * lag_window/2+1);
    fft_planC = (fftwf_plan_s*)fftwf_plan_dft_c2r_1d(lag_window, spectrumC, samplesC, FFTW_ESTIMATE);

    for (unsigned int i = 0; i < lag_window; i++) {
        samplesA[i] = localA[i];
//        samplesB[i] = localB[i];
        int o = 12345; //synthetic offset
        samplesB[i] = localA[(i+o)%lag_window];
        a.push_back(samplesA[i]);
        b.push_back(samplesB[i]);
    }
#endif

    fftwf_execute(fft_planA);
    fftwf_execute(fft_planB);

#ifdef IQ_SAMPLING
    for (unsigned int i = 0; i < lag_window/2; i++) {
#else
    for (unsigned int i = 0; i < lag_window/2+1; i++) {
#endif
        float Ar = spectrumA[i][0];
        float Ai = spectrumA[i][1];
        float Br = spectrumB[i][0];
        float Bi = spectrumB[i][1];
        //A \times B*
        spectrumC[i][0] = Ar * Br + Ai * Bi;
        spectrumC[i][1] = Ai * Br - Bi * Ar;

//        fa.push_back(Ar);
//        fa.push_back(Ai);
//        fb.push_back(Br);
//        fb.push_back(Bi);
//        fc.push_back(spectrumC[i][0]);
//        fc.push_back(spectrumC[i][1]);
    }
    fftwf_execute(fft_planC);

    float maxval=0, meanval=0, var=0;
    int lag = 0;
    nlohmann::json corr = nlohmann::json::array();

#ifdef IQ_SAMPLING
    for (unsigned int i=0; i < lag_window/2; i++){
        float val = sqrt(samplesC[0][i]*samplesC[0][i] + samplesC[1][i]*samplesC[1][i]);
        corr.push_back(val);
        if (val > maxval) {
            lag = i;
            maxval = val;
        }
        meanval += val;
    }
    meanval /= lag_window/2;
    for (unsigned int i=0; i < lag_window/2; i++){
        float val = sqrt(samplesC[0][i]*samplesC[0][i] + samplesC[1][i]*samplesC[1][i]);;
        var += val*val;
    }
    var /= lag_window/2;
#else
    for (unsigned int i=0; i < lag_window; i++){
        float val = abs(samplesC[i]);
        corr.push_back(val);
        if (val > maxval) {
            lag = i;
            maxval = val;
        }
        meanval += val;
    }
    meanval /= lag_window;
    for (unsigned int i=0; i < lag_window; i++){
        float val = abs(samplesC[i])-meanval;
        var += val*val;
    }
    var /= lag_window;
#endif

    float std = sqrt(var);
    DEBUG("Align: Max: {}, Mean: {}, RMS: {}",maxval, meanval, std);
    if (maxval > meanval+1*std) {
        DEBUG("Lag found!\n");
        DEBUG("Lag {}\n",lag);

        reply["lag"] = lag;
//        reply["a"] = a;
//        reply["b"] = b;
//        reply["corr"] = corr;
        conn.send_json_reply(reply);
    }
    else {
        DEBUG("NO Lag found!\n");
//        reply["a"] = a;
//        reply["b"] = b;
//        reply["fa"] = fa;
//        reply["fb"] = fb;
//        reply["fc"] = fc;
        reply["lag"] = 0;
//        reply["corr"] = corr;
        conn.send_json_reply(reply);
    }

    fftwf_free(samplesA);
    fftwf_free(samplesB);
    fftwf_free(samplesC);
    fftwf_free(spectrumA);
    fftwf_free(spectrumB);
    fftwf_free(spectrumC);
    fftwf_destroy_plan(fft_planA);
    fftwf_destroy_plan(fft_planB);
    fftwf_destroy_plan(fft_planC);
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
    uint copied = 0; //samples
    uint samples_per_frame = buf_inA->frame_size/sizeof(short);

    //assumes the arrays are aligned
    while (!stop_thread) {

        inA = (short*)wait_for_full_frame(buf_inA, unique_name.c_str(), frame_inA);
        inB = (short*)wait_for_full_frame(buf_inB, unique_name.c_str(), frame_inB);

        if ((inA == nullptr) || (inB == nullptr))
            break;

        if (going) {
            int copylen= ((lag_window-copied) > samples_per_frame) ? samples_per_frame : lag_window-copied;
            memcpy((char*)localA+copied*sizeof(short), inA, copylen*sizeof(short));
            memcpy((char*)localB+copied*sizeof(short), inB, copylen*sizeof(short));
            copied += copylen;
            if (copied >= lag_window) {
                copied=0;
                going=false;
            }
        }

        mark_frame_empty(buf_inA, unique_name.c_str(), frame_inA);
        mark_frame_empty(buf_inB, unique_name.c_str(), frame_inB);
        frame_inA = (frame_inA + 1) % buf_inA->num_frames;
        frame_inB = (frame_inB + 1) % buf_inB->num_frames;

    }
}
