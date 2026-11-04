#include "pulsarNetworkProducer.hpp"

#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"         // for Buffer, allocate_new_metadata_object, mark_frame_full
#include "chordMetadata.hpp"  // for set_first_packet_recv_time, set_fpga_seq_num, set_stream_id
#include "kotekanLogging.hpp" // for INFO
#include "visUtil.hpp"        // for frameID, modulo

#include <atomic>     // for atomic_bool
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error
#include <stdint.h>   // for uint32_t, uint8_t
#include <vector>     // for vector

// Include the classes we will be using
using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

#define DIM(v) (sizeof(v)/sizeof(v[0]))

// Register the stage with the stage factory.
REGISTER_KOTEKAN_STAGE(PulsarNetworkProducer);

PulsarNetworkProducer::PulsarNetworkProducer(Config& config, const std::string& unique_name,
                                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&PulsarNetworkProducer::main_thread, this)) {

    // Register as producer of out_buf
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);
}


PulsarNetworkProducer::~PulsarNetworkProducer() {}

// Framework managed pthread
void PulsarNetworkProducer::main_thread() {

    // Ring buffer pointer
    struct timeval now;
    int frame_id = 0;
    uint64_t seq_num = 0;
    stream_t stream_id;
    stream_id.id = 0; // no idea what that does

#define samples_per_dataset 16384
#define num_pols 2
#define num_freqs 16
#define num_beams 16

    // Until the thread is stopped
    while (!stop_thread) {

        // Acquire frame
        uint8_t* frame = out_buf->wait_for_empty_frame(unique_name, frame_id);
        if (frame == nullptr)
            break;

        out_buf->allocate_new_metadata_object(frame_id);
        get_chord_metadata(out_buf, frame_id)->set_from_frame_desc(out_buf->get_frame_desc<kotekan::GenericNDArray>());
        get_chord_metadata(out_buf, frame_id)->set_fpga_seq_num(seq_num);
        get_chord_metadata(out_buf, frame_id)->set_stream_id(stream_id);

        gettimeofday(&now, nullptr);
        get_chord_metadata(out_buf, frame_id)->set_first_packet_recv_time(now);

        assert(num_beams < MAX_NUM_BEAMS);
        chordMetadata::beamCoord beam_coord = {};
        // just make up some random numbers (not necessarily in physcial range)
        // note: arrays have different lenght to mix things up when repeating
        float right_ascension[] = {
                                    217.80,
                                    82.46,
                                    258.67,
                                    196.04,
                                    176.02,
                                    127.40,
                                    298.89,
                                    90.67,
                                    87.11,
                                    282.91,
                                  };
       float declination[] = {
                               256.52,
                               264.16,
                               49.39,
                               227.15,
                               9.72,
                               21.46,
                               21.73,
                               51.71,
                               326.49,
                             };
      float scaling[] = {
                         3,
                         2,
                         2,
                         2,
                         1,
                         2,
                         2,
                         3,
                    };
        for(int i = 0 ; i < num_beams ; ++i) {
          beam_coord.right_ascension[i] = right_ascension[i % DIM(right_ascension)];
          beam_coord.declination[i] = declination[i % DIM(declination)];
          beam_coord.scaling[i] = scaling[i % DIM(scaling)];
        }
        get_chord_metadata(out_buf, frame_id)->set_beam_coord(beam_coord);

        std::vector<int> coarse_freq(num_freqs);
        for(int i = 0 ; i < num_freqs ; ++i) {
          coarse_freq[i] = 2*i;
        }
        get_chord_metadata(out_buf, frame_id)->set_coarse_freq(coarse_freq);

        // put some data into the frame.
        assert(samples_per_dataset % 4 == 0 && "samples_per_dataset must be a multiple of 4");
        int count = 0;
        for (int beam = 0; beam < num_beams; beam++) {
            for (int freq = 0; freq < num_freqs; freq++) {
                for (int pol = 0; pol < num_pols; pol++) {
                    for (int i = 0; i < samples_per_dataset; i += 4) {
                        int index = i + pol * samples_per_dataset
                                    + freq * num_pols * samples_per_dataset
                                    + beam * num_freqs * num_pols * samples_per_dataset;
                        frame[index + 0] = beam << 4 | freq;
                        frame[index + 1] = pol | (count & 0xff);
                        frame[index + 2] = (i >> 8) & 0xff;
                        frame[index + 3] = (i >> 0) & 0xff;
                        count += 2;
                    }
                }
            }
        }


        // Release frame
        out_buf->mark_frame_full(unique_name, frame_id);

        // Increase the ring pointer
        frame_id = (frame_id + 1) % out_buf->num_frames;
        seq_num += samples_per_dataset;
    }
}
