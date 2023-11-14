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

void AirspyAlign::lag_only(kotekan::connectionInstance& conn) {
    this->start_callback(conn,false);
}
void AirspyAlign::lag_n_corr(kotekan::connectionInstance& conn){
    this->start_callback(conn,true);
}


void AirspyAlign::start_callback(kotekan::connectionInstance& conn, bool send_corr) {
    going = true;
    while (going) {usleep(1000);}

    fftwf_complex *spectrumA, *spectrumB, *spectrumC;
    float *samplesA, *samplesB, *samplesC;
    fftwf_plan fft_planA, fft_planB, fft_planC;

    samplesA = (float*)fftwf_malloc(sizeof(float) * lag_window*2);
    spectrumA = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * lag_window*2/2+1);
    fft_planA = (fftwf_plan_s*)fftwf_plan_dft_r2c_1d(lag_window*2, samplesA, spectrumA, FFTW_ESTIMATE);

    samplesB = (float*)fftwf_malloc(sizeof(float) * lag_window*2);
    spectrumB = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * lag_window*2/2+1);
    fft_planB = (fftwf_plan_s*)fftwf_plan_dft_r2c_1d(lag_window*2, samplesB, spectrumB, FFTW_ESTIMATE);

    samplesC = (float*)fftwf_malloc(sizeof(float) * lag_window*2);
    spectrumC = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * lag_window/2*2+1);
    fft_planC = (fftwf_plan_s*)fftwf_plan_dft_c2r_1d(lag_window*2, spectrumC, samplesC, FFTW_ESTIMATE);

    for (unsigned int i = 0; i < lag_window; i++) {
        samplesA[i] = localA[i];
        samplesB[i] = localB[i];
    }
    memset(&samplesA[lag_window],0,lag_window*sizeof(float));
    memset(&samplesB[lag_window],0,lag_window*sizeof(float));

    fftwf_execute(fft_planA);
    fftwf_execute(fft_planB);

    for (unsigned int i = 0; i < lag_window/2*2+1; i++) {
        float Ar = spectrumA[i][0];
        float Ai = spectrumA[i][1];
        float Br = spectrumB[i][0];
        float Bi = spectrumB[i][1];
        //A \times B*
        spectrumC[i][0] = Ar * Br + Ai * Bi;
        spectrumC[i][1] = Ai * Br - Bi * Ar;
    }
    fftwf_execute(fft_planC);

    float maxval=0, meanval=0, var=0;
    int lag = 0;
    nlohmann::json corr_pos = nlohmann::json::array();
    nlohmann::json corr_neg = nlohmann::json::array();

    uint edge_truncate = lag_window*0.1;
    for (unsigned int i=0; i < lag_window-edge_truncate; i++){
        float norm = ((float)lag_window)-i;
        float val = abs(samplesC[i]/norm);
        if (val > maxval) {
            lag = i;
            maxval = val;
        }
        meanval += val;
        corr_pos.push_back(val);

        val = abs(samplesC[2*lag_window-i]/norm);
        if (val > maxval) {
            lag = -i;
            maxval = val;
        }
        meanval += val;
        corr_neg.push_back(val);
    }
    meanval /= (lag_window-edge_truncate)*2;
    for (unsigned int i=0; i < lag_window-edge_truncate; i++){
        float norm = ((float)lag_window)-i;
        float val = abs(samplesC[i]/norm)-meanval;
        var += val*val;
        val = abs(samplesC[2*lag_window-i]/norm)-meanval;
        var += val*val;
    }
    var /= (lag_window-edge_truncate)*2;
    float std = sqrt(var);

    nlohmann::json reply;
    //TODO: messagepack
    DEBUG("Align: Max: {}, Mean: {}, RMS: {}",maxval, meanval, std);
    if (maxval > meanval+1*std) {
        DEBUG("Lag found!\n");
        DEBUG("Lag {}\n",lag);

        reply["lag"] = lag;
        if (send_corr){
            reply["corr_pos"] = corr_pos;
            reply["corr_neg"] = corr_neg;
        }
        conn.send_json_reply(reply);
    }
    else {
        DEBUG("NO Lag found!\n");
        reply["lag"] = 0;
        if (send_corr){
            reply["corr_pos"] = corr_pos;
            reply["corr_neg"] = corr_neg;
        }
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
    std::string endpoint;
    kotekan::restServer& rest_server = kotekan::restServer::instance();
    using namespace std::placeholders;
    endpoint = unique_name+"/cal_lag";
    rest_server.register_get_callback(endpoint,
                                      std::bind(&AirspyAlign::lag_only, this, _1));
    endpoint = unique_name+"/get_correlation";
    rest_server.register_get_callback(endpoint,
                                      std::bind(&AirspyAlign::lag_n_corr, this, _1));

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
