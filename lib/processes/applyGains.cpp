#include "applyGains.hpp"
#include "visBuffer.hpp"
#include "errors.h"
#include "configUpdater.hpp"
#include "visFileH5.hpp"
#include "visUtil.hpp"
#include "prometheusMetrics.hpp"

#include <algorithm>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <sys/stat.h>
#include <csignal>
#include <exception>

using namespace HighFive;
using namespace std::placeholders;



REGISTER_KOTEKAN_PROCESS(applyGains);



applyGains::applyGains(Config& config,
                           const string& unique_name,
                           bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&applyGains::main_thread, this)) {

    apply_config(0);

    // Get the path to gains directory
    gains_dir = config.get_string(unique_name, "gains_dir");

    // Setup the input buffer
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Setup the output buffer
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    // FIFO for gains updates
    gains = updateQueue<std::vector<std::vector<cfloat>>>(num_kept_updates);

    // subscribe to gain timestamp updates
    // Using the new constructor
    configUpdater::instance().subscribe(this,
                std::bind(&applyGains::receive_update, this, 
                          _1));
//    configUpdater::instance().subscribe(updatable_config,
//                              std::bind(&applyGains::receive_update, this,
//                                        std::placeholders::_1));
}

void applyGains::apply_config(uint64_t fpga_seq) {

    // Number of gain versions kept
    num_kept_updates = config.get_uint64(unique_name, "num_kept_updates");
    if (num_kept_updates < 1)
        throw std::invalid_argument("applyGains: config: num_kept_updates has" \
                                    "to equal or greater than one (is "
                                    + std::to_string(num_kept_updates) + ").");

    // Time to blend old and new gains in seconds. Default is 5 minutes. 
    tcombine = config.get_float_default(unique_name, "combine_gains_time", 5*60);

    updatable_config = config.get_string(unique_name, "updatable_config");
}

cfloat applyGains::combine_gains(int idx) {

    if (coef_new == 1) {
        return gain_new[idx];
    } else {
        return coef_new * gain_new[idx] + coef_old * gain_old[idx];
    }
}

bool applyGains::receive_update(nlohmann::json &json) {
    // TODO: need to make sure this is thread safe
    //uint64_t new_ts;
    double new_ts;
    std::string gains_path;
    std::string gtag;
    std::vector<std::vector<cfloat>> gain_read;
    // receive new gains timestamp ("gains_timestamp" might move to "start_time")
    try {
        if (!json.at("gains_timestamp").is_number())
            throw std::invalid_argument("applyGains: received bad gains " \
                                       "timestamp: " +
                                       json.at("gains_timestamp").dump());
        if (json.at("gains_timestamp") < 0)
            throw std::invalid_argument("applyGains: received negative gains " \
                                       "timestamp: " +
                                       json.at("gains_timestamp").dump());
        new_ts = json.at("gains_timestamp");
    } catch (std::exception& e) {
        WARN("%s", e.what());
        return false;
    }
    // receive new gains tag
    try {
        if (!json.at("tag").is_string())
            throw std::invalid_argument("applyGains: received bad gains tag:" \
                                        + json.at("tag").dump());
        gtag = json.at("tag");
    } catch (std::exception& e) {
        WARN("%s", e.what());
        return false;
    }
    // Get the gains for this timestamp
    // TODO: For now, assume the tag is the gain file name.
    gains_path = gains_dir + "/" + gtag + ".hdf5";
    std::cout << "Filepath c++: "+gains_path << std::endl;
    // Read the gains from file
    HighFive::File gains_fl(gains_path, HighFive::File::ReadOnly);
    // Read the dataset and alocates it to the most recent entry of the gain vector
    HighFive::DataSet gains_ds = gains_fl.getDataSet("/gain");
    gains_ds.read(gain_read);
    std::cout << "gain c++: " << gain_read[250][5] << std::endl;

    gains.insert(double_to_ts(new_ts), std::move(gain_read));
    return true;
}

void applyGains::main_thread() {

    unsigned int output_frame_id = 0;
    unsigned int input_frame_id = 0;
    unsigned int freq;
    double tpast;
    double frame_time;

    while (!stop_thread) {


        // Wait for the input buffer to be filled with data
        if(wait_for_full_frame(in_buf, 
                    unique_name.c_str(),input_frame_id) == nullptr) {
            break;
        }

        // Create view to input frame
        auto input_frame = visFrameView(in_buf, input_frame_id);

        // frequency index of this frame
        freq = input_frame.freq_id;
        // Unix time
        frame_time = ts_to_double(std::get<1>(input_frame.time));

        std::pair< timespec, const std::vector<std::vector<cfloat>>* > gainpair1;
        std::pair< timespec, const std::vector<std::vector<cfloat>>* > gainpair2;

        gainpair1 = gains.get_update(double_to_ts(frame_time));
        if (gainpair1.second == NULL) {
            WARN("No gains available.\nKilling kotekan");
            std::raise(SIGINT);
        }
        tpast = frame_time - ts_to_double(gainpair1.first);
        gain_new = (*gainpair1.second)[freq];
//        std::cout << "gain2 c++: " << gain_new[5] << std::endl;
        if (tpast >= tcombine) {
            coef_new = 1;
        } else if (tpast >= 0) {
            gainpair2 = gains.get_update(double_to_ts(frame_time - tcombine));
            if (ts_to_double(gainpair1.first)!=ts_to_double(gainpair2.first)) {
                // If we are not using the very first set of gains,
                // do gains interpolation.
                gain_old = (*gainpair2.second)[freq];
                coef_new = tpast/tcombine;
                coef_old = 1 - coef_new;
            }
        } else {
            // TODO: export prometeus metric saying that there are no gains to apply?
            INFO("Gain timestamp is in the future!")
        }

        if (coef_new==1) {
            INFO("coeff new: %f", coef_new);
            std::cout << "new gain: " << gain_new[5] << std::endl;
        } else {
            INFO("coeff new: %f, coeff old: %f", coef_new, coef_old);
            std::cout << "new gain: " << gain_new[5] << " old gain: " << gain_old[5] << std::endl;
            INFO("tcombine: %f", tcombine);
        }

        // Wait for the output buffer to be empty of data
        if(wait_for_empty_frame(out_buf, unique_name.c_str(),
                                        output_frame_id) == nullptr) {
            break;
        }
        allocate_new_metadata_object(out_buf, output_frame_id);

        // Copy frame and create view
        auto output_frame = visFrameView(out_buf, output_frame_id, input_frame);

        // For now this doesn't try to do any type of check on the
        // ordering of products in vis and elements in gains.
        // Also assumes the ordering of freqs in gains is standard
        int idx = 0;
        for (int ii=0; ii<input_frame.num_elements; ii++) {
            for (int jj=ii; jj<input_frame.num_elements; jj++) {
                // Gains are to be multiplied to vis
                output_frame.vis[idx] = input_frame.vis[idx]
                                        * combine_gains(ii)
                                        * std::conj(combine_gains(jj));
                // Update the wheights.
                // TODO: check that this exponent is correct
                output_frame.weight[idx] /= pow(abs(combine_gains(ii))
                                              * abs(combine_gains(jj)), 2.0);
                idx++;
            }
            // Update the gains.
            output_frame.gain[ii] = combine_gains(ii);
        }

        // Mark the buffers and move on
        mark_frame_full(out_buf, unique_name.c_str(), 
                                                    output_frame_id);
        mark_frame_empty(in_buf, unique_name.c_str(), 
                                                input_frame_id);
        // Advance the current frame ids
        input_frame_id = (input_frame_id + 1) % in_buf->num_frames;
        output_frame_id = (output_frame_id + 1) % out_buf->num_frames;
    }
}
