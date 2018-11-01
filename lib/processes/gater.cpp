#include "gater.hpp"
#include <vector>
#include "chimeMetadata.h"

REGISTER_KOTEKAN_PROCESS(gater);

gater::gater(Config& config, const string& unique_name,
                         bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&gater::main_thread, this)) {

    apply_config(0);

    _samples_per_frame = config.get_int(unique_name, "samples_per_data_set");
    _num_elements = config.get_int(unique_name, "num_elements");

    // Polyco coeffs for 600MHz, for now
    _polyco_coeffs = config.get_double_array(unique_name, "polyco_coeffs");
    // Retrieved in ms, converted to seconds
    _pulse_width = config.get_double_default(unique_name, "pulse_width", 1) /1000;
    _f0 = config.get_double_default(unique_name, "f0", 1);
    _tmid = config.get_double_default(unique_name, "tmid", 1);
    _rphase = config.get_double_default(unique_name, "rphase", 1);
    _dm = config.get_double_default(unique_name, "dm", 1);

    float int_time = config.get_float_default(unique_name, "integration_time", -1.0);

    // If the integration time was set then calculate the number of GPU frames
    // we need to integrate for.
    if(int_time >= 0.0) {
        float frame_length = _samples_per_frame * 2.56e-6;

        // Calculate nearest *even* _number of frames
        _num_gpu_frames = 2 * ((int)(int_time / frame_length) / 2);
        INFO("gater: num_gpu = %d", _num_gpu_frames);
    } else {
        _num_gpu_frames = config.get_int(unique_name, "num_gpu_frames");
    }

    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

}

gater::~gater() {
}

void gater::apply_config(uint64_t fpga_seq) {
}

// NOTE: timespec.tv_nsec is usually treated as unsigned long
timespec gater::add_nsec(timespec temp, long nsec) {
    timespec new_time = temp;
    new_time.tv_sec += nsec / (int) 1e9;
    new_time.tv_nsec += nsec % (int) 1e9;
    if (new_time.tv_nsec < 0) {
        new_time.tv_sec -= 1;
        new_time.tv_nsec += (int) 1e9;
    } else if (new_time.tv_nsec > 1e9) {
        new_time.tv_sec += 1;
        new_time.tv_nsec -= (int) 1e9;
    }
    return new_time;
}

// Assume t1 > t2
timespec gater::get_timespec_diff(timespec t1, timespec t2) {
    long diff_sec = t1.tv_sec - t2.tv_sec;
    long diff_nsec = t1.tv_nsec - t2.tv_nsec;

    if (diff_nsec < 0) {
        diff_sec -= 1;
        diff_nsec += (int) 1e9;
    }

    timespec new_time;
    new_time.tv_sec = diff_sec;
    new_time.tv_nsec = diff_nsec;
    return new_time;
}

double gater::gps_to_mjd(timespec time_gps) {
    // Constant = diff betwen MJD epoch (1950) and GPSTime epoch (1980): 40587
    // Other constant: diff between MJD epoch and Unix epoch (1970): 40587
    return ((double) time_gps.tv_sec/ 86400.) + ((double) time_gps.tv_nsec/ 86400. / 1e9) + 40587;
}

timespec gater::mjd_to_gps(double time_mjd) {
    // Constant = diff betwen MJD epoch (1950) and GPSTime epoch (1980): 40587
    // Other constant: diff between MJD epoch and Unix epoch (1970): 40587
    timespec time_gps;
    time_gps.tv_sec = (unsigned long) ((time_mjd - 40587) * 86400);
    time_gps.tv_nsec = (unsigned long) ((time_mjd - 40587 - time_gps.tv_sec / 86400.) * 86400 * 1e9);
    return time_gps;
}

double gater::get_polyco_phase(double time_mjd) {
    double dt = (time_mjd - _tmid) * 1440;
    return (double)  (_rphase + dt*60*_f0 + _polyco_coeffs[0] + dt*_polyco_coeffs[1]);
}

// Return the corresponding time in MJD of a selected phase, givein theh
double gater::get_polyco_time(double phase) {
    // phase is passed in as an int
    double dt = (double) (phase - _rphase - _polyco_coeffs[0]) / (60.*_f0 + _polyco_coeffs[1]);
    return dt / 1440 + _tmid;
}

// Gets frequency of the current frame in MHz
float gater::get_frequency(struct Buffer* buf, int frame_id) {
    stream_id_t stream_id = get_stream_id_t(buf, frame_id);
    uint32_t bin = bin_number_chime(&stream_id);
    return freq_from_bin(bin);
}

vector<timespec> gater::get_pulse_times(timespec start_time_gps, timespec end_time_gps, float freq) {
    double end_time_mjd = gps_to_mjd(end_time_gps);
    double time_mjd = gps_to_mjd(start_time_gps);
    int prev_phase;
    bool init = true;
    vector<timespec> pulse_times;
    while (time_mjd < end_time_mjd) {
        double time_delay = 4.149*_dm*(1.0 / pow((600.0/1000.0),2) - 1.0 / pow((freq/1000.0),2))/(1000*60*1440);
        // Find the closest phase
        double phase_mjd = get_polyco_phase(time_mjd + time_delay);
        // Strip off the decimal and add one to find next
        int phase = (int) ceil(phase_mjd);
        // Weird bug occurred where we were entering an infinite loop if time_mjd ~= the predicted pulse time
        // Solution: force the phase to increment by one if the phases are the same.
        if (!init && prev_phase == phase)
            phase += 1;
        // Adjust time from polycos with dispersion
        time_mjd = get_polyco_time(phase) - time_delay;
        timespec time = mjd_to_gps(time_mjd);
        // millsecond offset
        pulse_times.push_back(add_nsec(time, (int) -27.0e6));
        prev_phase = phase;
        init = false;
    }
    return pulse_times;
}

void gater::main_thread() {
    int8_t *in_local;
    int8_t *out_pulsar;

    int in_frame_id = 0;
    int pulsar_frame_id = 0;
    int pulsar_i = 0;
    //counts the number of non-zero samples in each pulsar_frame
    int pulse_count = 0;

    int8_t *input;
    int8_t *pulsar = NULL;

    uint64_t init_fpga_seq;
    timespec init_time_gps;
    timespec end_time_gps;
    timespec diff_time_gps;
    timeval init_packet_recv_time;
    float freq;
    int num_pulse = 0;
    int pulsar_samples_left_over = 0;

    bool init = true;

    while(!stop_thread) {
        in_local = (int8_t*) wait_for_full_frame(in_buf, unique_name.c_str(), in_frame_id);
        if (in_local == NULL) break;

        timeval time_v = get_first_packet_recv_time(in_buf, in_frame_id);
        timespec time_s = get_gps_time(in_buf, in_frame_id);
        uint64_t fpga_seq = get_fpga_seq_num(in_buf, in_frame_id);
        uint16_t stream_id = get_stream_id(in_buf, in_frame_id);

        input = (int8_t *) in_local;

        // PUlse from the previous input has a bit of pulse signal in this frame. We should capture it.
        for (int i = 0; i < pulsar_samples_left_over; i++) {
            for (int j = 0; j < _num_elements; ++j) {
                pulsar[pulsar_i * _num_elements + j] = input[i * _num_elements + j];
            }
            pulsar_i = (pulsar_i + 1) % _samples_per_frame;

            // if we have filled up a pulsar frame but the pulse isn't over, set all the metadata and mark it full
            // then get a new frame
            if (pulsar_i == 0) {
                //TODO: possibly add all the lost_timesaples from the frames that had pulsars and then get probability?
                uint64_t lost_samples = (float) get_lost_timesamples(in_buf, in_frame_id)/(float) in_buf->frame_size*pulse_count;
                atomic_add_lost_timesamples(out_buf, pulsar_frame_id, (int32_t) lost_samples);

                mark_frame_full(out_buf, unique_name.c_str(), pulsar_frame_id);
                pulsar_frame_id = (pulsar_frame_id + 1) % out_buf->num_frames;

                //Get a new frame
                out_pulsar = (int8_t*) wait_for_empty_frame(out_buf, unique_name.c_str(), pulsar_frame_id);
                if (out_pulsar == NULL) break;
                pulsar = (int8_t *) out_pulsar;
                INFO("Frame got full.Got new pulsar frame, pulsar_frame_id=%d", pulsar_frame_id);

                pass_metadata(in_buf, in_frame_id, out_buf, pulsar_frame_id);
                pulse_count = 0;
            }
        }
        // After taking care of the leftover pulse signal, we should make sure not to do this for subsequent frames.
        pulsar_samples_left_over = 0;

        if (init) {
            init = false;
            init_fpga_seq = fpga_seq;
            init_packet_recv_time  = time_v;

            //get a new pulsar frame
            INFO("Waiting for new pulsar frame to write into...");
            out_pulsar = (int8_t*) wait_for_empty_frame(out_buf, unique_name.c_str(), pulsar_frame_id);
            if (out_pulsar == NULL) break;
            pulsar = (int8_t *) out_pulsar;

            pass_metadata(in_buf, in_frame_id, out_buf, pulsar_frame_id);
        }

        init_time_gps = time_s;

        //if gps_time is zero then get the time from counting fpga frames
        if (init_time_gps.tv_sec == 0){
            TIMEVAL_TO_TIMESPEC(&init_packet_recv_time, &init_time_gps);
            init_time_gps = add_nsec(init_time_gps, 2560*(fpga_seq - init_fpga_seq));
            INFO("init_time: secs  = %llu", init_time_gps.tv_sec);
            INFO("init_time: nsecs = %llu", init_time_gps.tv_nsec);
        }
        freq = get_frequency(in_buf, in_frame_id);

        end_time_gps = add_nsec(init_time_gps, 2560*_samples_per_frame);
        vector<timespec> pulse_times_gps = get_pulse_times(init_time_gps, end_time_gps, freq);

        for (int v = 0; v < pulse_times_gps.size(); v++){
            INFO("pulse time sec %u", pulse_times_gps.at(v).tv_sec);
            INFO("pulse time nsec %u", pulse_times_gps.at(v).tv_nsec);

            // Assumes pulse_time_gps > init_time_gps
            diff_time_gps = get_timespec_diff(pulse_times_gps.at(v), init_time_gps);

            if (diff_time_gps.tv_sec > 1 || diff_time_gps.tv_nsec > 2.56e3 *  _samples_per_frame) {
              // no pulse in the frame
            } else {
                num_pulse++;
                INFO("num_pulse = %d", num_pulse);
                // convert MJD pulse width to secs -> microsec -> index difference
                int pulse_width_sample = (int) (1e6 * (_pulse_width/2) / 2.56);
                INFO("pulse_width_sample %d", pulse_width_sample);
                // convert the diff between (mid pulse time, first frame time) to an index (starting sample)
                int first_pulse_sample = diff_time_gps.tv_nsec / 2.56e3 - pulse_width_sample;
                timespec first_pulse_time = add_nsec(init_time_gps, first_pulse_sample*2560);
                timespec last_pulse_time = add_nsec(init_time_gps, (first_pulse_sample+2*pulse_width_sample)*2560);

                INFO("first pulse time sec  %d", first_pulse_time.tv_sec);
                INFO("first pulse time nsec  %d", first_pulse_time.tv_nsec);
                INFO("last pulse time sec %d", last_pulse_time.tv_sec);
                INFO("last pulse time nsec %d", last_pulse_time.tv_nsec);


                for (int i = 0; i < 2*pulse_width_sample; i++) {
                    if (first_pulse_sample + i >= _samples_per_frame) { // to avoid segfault
                        pulsar_samples_left_over = 2*pulse_width_sample - i;
                        break;
                    }
                    if (first_pulse_sample + i < 0) break;
                    pulse_count++;
                    for (int j = 0; j < _num_elements; ++j) {
                        pulsar[pulsar_i * _num_elements + j] = input[(first_pulse_sample + i) * _num_elements + j];
                    }
                    pulsar_i = (pulsar_i + 1) % _samples_per_frame;

                    // if we have filled up a pulsar frame but the pulse isn't over, set all the metadata and mark it full
                    // then get a new frame
                    if (pulsar_i == 0) {
                        //TODO: possibly add all the lost_timesaples from the frames that had pulsars and then get probability?
                        uint64_t lost_samples = (float) get_lost_timesamples(in_buf, in_frame_id)/(float) in_buf->frame_size*pulse_count;
                        // This line is prone to segfaulting for some reason related to getting the lost samples from the output buffer
                        INFO("pulsar_frame_id = %d", pulsar_frame_id);
                        atomic_add_lost_timesamples(out_buf, pulsar_frame_id, (int32_t) lost_samples);

                        mark_frame_full(out_buf, unique_name.c_str(), pulsar_frame_id);
                        pulsar_frame_id = (pulsar_frame_id + 1) % out_buf->num_frames;

                        //Get a new frame
                        out_pulsar = (int8_t*) wait_for_empty_frame(out_buf, unique_name.c_str(), pulsar_frame_id);
                        if (out_pulsar == NULL) break;
                        pulsar = (int8_t *) out_pulsar;
                        INFO("Frame got full.Got new pulsar frame, pulsar_frame_id=%d", pulsar_frame_id);

                        pass_metadata(in_buf, in_frame_id, out_buf, pulsar_frame_id);
                        pulse_count = 0;
                    }
                }
            }
        }

        mark_frame_empty(in_buf, unique_name.c_str(), in_frame_id);
        in_frame_id = (in_frame_id + 1) % in_buf->num_frames;


        // We need to synchronize the input gating cadence with the one in visAcc
        // visAcc does the same thing with frame_count, but name collision
        uint frame_count = fpga_seq / _samples_per_frame;
        INFO("frame_count %d", frame_count);

        //if _num_gpu_frames-1 have been sent to accumulate then send the zero-padded
        //pulsar frame as well so than it gets integrated
        if (frame_count % _num_gpu_frames == 0) {
            for (int i = pulsar_i; i<_samples_per_frame; ++i) {
                for (int j = 0; j<_num_elements; ++j) {
                    pulsar[i * _num_elements + j] = 0;
                }
            }
            //TODO:fix lost_timesamples to do the probability thing correctly
            uint64_t lost_samples = (int32_t) _samples_per_frame - pulsar_i;
            atomic_add_lost_timesamples(out_buf, pulsar_frame_id, (int32_t) lost_samples);

            // Mark whatever frame we're sending as the last one in the sequence, so visAccumulate knows to stop.
            // We still send a frame even if there was no actual pulsar data found.
            set_is_last_pulsar_frame(out_buf, pulsar_frame_id, true);
            mark_frame_full(out_buf, unique_name.c_str(), pulsar_frame_id);
            INFO("Sent half-empty frame");

            // Wait for another pulsar frame, where we actually expect to store data
            pulsar_frame_id = (pulsar_frame_id + 1) % out_buf->num_frames;
            INFO("Num pulsar samples, %d", pulsar_i);
            pulsar_i = 0;

            //get a new pulsar frame
            out_pulsar = (int8_t*) wait_for_empty_frame(out_buf, unique_name.c_str(), pulsar_frame_id);
            if (out_pulsar == NULL) break;
            INFO("Sent half-empty farme. Got new pulsar frame");

            pulsar = (int8_t *) out_pulsar;

            pass_metadata(in_buf, in_frame_id, out_buf, pulsar_frame_id);
            pulse_count = 0;

        }

    }
}
