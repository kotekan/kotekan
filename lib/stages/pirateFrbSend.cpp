#include "pirateFrbSend.hpp"

#include "Config.hpp"            // for Config
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "kotekanLogging.hpp"    // for DEBUG2, ERROR, INFO, DEBUG, WARN
#include "metadata.hpp"          // for metadataObject
#include "XEngineMetadata.hpp"

#include "fmt.hpp" // for compile_string_to_view, format, format_string, fmt
#include "json.hpp" // for json, basic_json, iter_impl

#include <memory>        // for __shared_ptr_access, shared_ptr
#include <arpa/inet.h>   // for htons, inet_addr

using nlohmann::json;

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using pirate::XEngineMetadata;

REGISTER_KOTEKAN_STAGE(pirateFrbSend);

pirateFrbSend::pirateFrbSend(Config& config, const std::string& unique_name,
                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&pirateFrbSend::main_thread, this))
{
    drop_threshold = config.get_default<float>(unique_name, "drop_threshold", 0.6);

    intensity_buf = get_buffer("intensity");
    intensity_buf->register_consumer(unique_name);
    offset_scale_buf = get_buffer("offset_scale");
    offset_scale_buf->register_consumer(unique_name);
    beam_position_buf = get_buffer("beam_positions");
    beam_position_buf->register_consumer(unique_name);
    beam_id_buf = get_buffer("beam_ids");
    beam_id_buf->register_consumer(unique_name);

    frb_chunk_size = config.get<int>(unique_name, "frb_chunk_size");
    if (frb_chunk_size != 256)
        FATAL_ERROR("frb_chunk_size must be 256");
    frb_num_output_times = config.get<int>(unique_name, "frb_num_output_times");
    assert(frb_num_output_times % frb_chunk_size == 0);

    // HACK -- this makes life much easier....
    if (frb_num_output_times != frb_chunk_size)
        FATAL_ERROR("frb_num_output_times must be equal to frb_chunk_size!");

    n_beams = config.get<int>(unique_name, "frb2_num_beams");

    upchan_total_channels = (config.get<int>(unique_name, "upchan_all_max_output_channel") -
                             config.get<int>(unique_name, "upchan_all_min_output_channel"));

    size_t expect_size = sizeof(float16_t) * 2 * (frb_num_output_times / frb_chunk_size) *
        n_beams * upchan_total_channels;
    if (offset_scale_buf->frame_size != expect_size)
        FATAL_ERROR("offset_scale_buf must have size {:d}x{:d}x{:d}x{:d}x{:d} = {:d}; got {:d}",
                    sizeof(float16_t), 2, frb_num_output_times / frb_chunk_size, n_beams,
                    upchan_total_channels, expect_size, offset_scale_buf->frame_size);
    assert(offset_scale_buf->frame_size == expect_size);

    // int4
    expect_size = (n_beams * upchan_total_channels * frb_chunk_size) / 2;
    if (intensity_buf->frame_size != expect_size)
        FATAL_ERROR("intensity_buf must have size {:d}x{:d}x{:d}/2 = {:d}; got {:d}",
                    n_beams, upchan_total_channels, frb_chunk_size, expect_size,
                    intensity_buf->frame_size);
    assert(intensity_buf->frame_size == expect_size);


    /**
beams: [
{beam_id: 0, x: 0.0, y: 0.0, type: Real},
{beam_id: 1, x: 0.1, y: -1.2, type: Real},
{beam_id: 75, x: -0.2, y: 4.8, type: Real},
{beam_id: 4, x: -0.2, y: 4.8, type: Fake},
{beam_id: 5, x: -0.2, y: 4.8, type: Real},
...]

frb_search_nodes: [
  {'ip': '10.0.0.42',
   'port': 9000,
   'beamset': 0,
   'beam_index_min': 0,
   'beam_index_max': 200,
  },
  {'ip': '10.0.0.43',
   'port': 9000,
   'beamset': 1,
   'beam_index_min': 200,
   'beam_index_max': 400,
  }
]
    */
    // ughh duplicated from bufferSend.cpp ...
    uint32_t send_timeout = config.get_default<uint32_t>(unique_name, "send_timeout", 20);
    uint32_t reconnect_time = config.get_default<uint32_t>(unique_name, "reconnect_time", 5);

    json frb_targets = config.get<std::vector<json> >(unique_name, "frb_search_nodes");
    for (json target : frb_targets) {
        if (!target.is_object()) {
            throw std::invalid_argument(fmt::format(fmt("Expect 'frb_targets' to contain a list of dicts")));
        }
        std::string ip = target["ip"].get<std::string>();
        int port = target["port"].get<int>();
        int beamset = target["beamset"].get<int>();
        //std::vector<int> beam_ids = target["beams"].get<std::vector<int> >();
        int beam_index_min = target["beam_index_min"].get<int>();
        int beam_index_max = target["beam_index_max"].get<int>();
        if (beam_index_max > (int)n_beams)
            FATAL_ERROR("beam_index_max ({:d}) must be <= n_beams ({:d})",
                        beam_index_max, n_beams);
        if (beam_index_min >= beam_index_max)
            FATAL_ERROR("beam_index_min ({:d}) must be < beam_index_max ({:d})",
                        beam_index_min, beam_index_max);
        if ((beam_index_min < 0) || (beam_index_max < 0))
            FATAL_ERROR("beam_index_min ({:d}) and beam_index_max ({:d}) must be non-negative",
                        beam_index_min, beam_index_max);

        std::shared_ptr<struct pirateDestination> dest = std::make_shared<pirateDestination>();
        dest->connected = false;
        dest->sent_header = false;
        dest->socket_fd = -1;
        dest->server_ip = ip;
        dest->server_port = port;

        bzero(&dest->server_addr, sizeof(dest->server_addr));
        dest->server_addr.sin_family = AF_INET;
        dest->server_addr.sin_addr.s_addr = inet_addr(dest->server_ip.c_str());
        dest->server_addr.sin_port = htons(dest->server_port);

        dest->reconnect_time = reconnect_time;
        dest->send_timeout = send_timeout;

        dest->beam_index_min = beam_index_min;
        dest->beam_index_max = beam_index_max;
        dest->beamset = beamset;
        
        dests.push_back(dest);
    }
}

pirateFrbSend::~pirateFrbSend() {}

struct pirateNetworkHeader {
    uint32_t magic_number;
    uint32_t config_length;
};


void pirateFrbSend::main_thread() {
    int frame_id = 0;

    // Fire up all the connect threads...
    for (auto dest : dests) {
        std::string name = "pirateFrbSend";
        dest->connect_thread = std::thread(&networkDestination::connect_to_server, std::ref(*dest),
                                           name, std::ref(stop_thread));
    }

    INFO("Waiting for data in beam_positions buffer...");
    uint8_t* beam_ids_ptr = nullptr;
    uint8_t* beam_pos_ptr = nullptr;
    float*   beam_pos = nullptr;
    int16_t* beam_ids = nullptr;
    size_t expect_size;

    beam_pos_ptr = beam_position_buf->wait_for_full_frame(unique_name, frame_id);
    if (beam_pos_ptr == nullptr)
        goto cleanup;

    INFO("Waiting for data in beam_ids buffer...");
    beam_ids_ptr = beam_id_buf->wait_for_full_frame(unique_name, frame_id);
    if (beam_ids_ptr == nullptr)
        goto cleanup;

    // Pull out beam positions and ids...
    beam_pos = reinterpret_cast<float*>(beam_pos_ptr);
    assert(beam_pos);
    expect_size = sizeof(float) * 2 * n_beams;
    assert(beam_position_buf->frame_size == expect_size);

    beam_ids = reinterpret_cast<int16_t*>(beam_ids_ptr);
    assert(beam_ids);
    expect_size = sizeof(int16_t) * n_beams;
    assert(beam_id_buf->frame_size == expect_size);

    // Save the subsets that we are going to send to each of our destinations.
    for (auto dest : dests) {
        int n_dest = dest->beam_index_max - dest->beam_index_min;
        dest->beam_positions.reserve(2 * n_dest);
        dest->beam_positions.insert(dest->beam_positions.begin(),
                                    beam_pos + 2 * dest->beam_index_min,
                                    beam_pos + 2 * dest->beam_index_max);
        dest->beam_ids.reserve(n_dest);
        dest->beam_ids.insert(dest->beam_ids.begin(),
                              beam_ids + dest->beam_index_min,
                              beam_ids + dest->beam_index_max);
    }

    // Release beam position & id buffers
    beam_position_buf->mark_frame_empty(unique_name, frame_id);
    beam_id_buf->mark_frame_empty(unique_name, frame_id);

    while (!stop_thread) {
        INFO("Waiting for frame {:d}...", frame_id);
        uint8_t* intensity_frame = intensity_buf->wait_for_full_frame(unique_name, frame_id);
        if (intensity_frame == nullptr)
            break;
        uint8_t* offset_scale_frame = offset_scale_buf->wait_for_full_frame(unique_name, frame_id);
        if (offset_scale_frame == nullptr)
            break;
        INFO("Got data for frame {:d}", frame_id);


        //FrameDesc get_frame_descr
        //std::shared_ptr<const kotekan::GenericNDArray>
        INFO("getting ndarray");
        auto intensity_nd = intensity_buf->get_ndarray_frame_desc();

        INFO("printing ndarray");
        intensity_nd->output_framedesc(std::cout);

        assert((size_t)intensity_nd->get_extent(0) == n_beams);
        assert((size_t)intensity_nd->get_extent(1) == upchan_total_channels);
        assert((size_t)intensity_nd->get_extent(2) == frb_chunk_size);
        
        uint32_t num_full_frames = intensity_buf->get_num_full_frames();

        // This logic is from sendBuffer, but with drop_frames assumed true.

        INFO("Drop threshold: num_full_frames {:d}, num_frames {:d}, drop threshold {:f}",
             num_full_frames, intensity_buf->num_frames, drop_threshold);
        if ((float)num_full_frames / (float)intensity_buf->num_frames > drop_threshold) {
            // If the number of full frames is high, then we drop some
            // frames, because we likely aren't sending fast enough to
            // keep up with the data rate.
            INFO("Number of full frames in intensity buffer is {:d} (total frames: {:d}), dropping "
                 "frame_id {:d}", num_full_frames, intensity_buf->num_frames, frame_id);
            //dropped_frame_counter.inc();
        } else {
            bool any_connected = false;
            for (auto dest : dests)
                if (dest->connected) {
                    any_connected = true;
                    INFO("Sending frame {:d} to dest {:s}:{:d}",
                         frame_id, dest->server_ip, dest->server_port);
                    if (!send_frame(frame_id, intensity_frame, offset_scale_frame, dest)) {
                        INFO("Failed to send frame {:d} to dest {:s}:{:d}; closing connection",
                             frame_id, dest->server_ip, dest->server_port);
                        dest->close_connection();
                    } else
                        DEBUG("Sent frame {:d} to {:s}:{:d}", frame_id,
                              dest->server_ip, dest->server_port);
                }
            if (!any_connected) {
                // Mark this frame as dropped
                INFO("Dropping intensity frame {:d}, because all connections are down.",
                     frame_id);
                //dropped_frame_counter.inc();
            }
        }

        intensity_buf->mark_frame_empty(unique_name, frame_id);
        offset_scale_buf->mark_frame_empty(unique_name, frame_id);

        frame_id = (frame_id + 1) % intensity_buf->num_frames;
    }

 cleanup:
    INFO("Cleaning up: closing connections...");
    for (auto dest : dests)
        dest->close_connection();
    INFO("Cleaning up: joining TCP threads...");
    for (auto dest : dests)
        dest->connect_thread.join();
    INFO("Exiting");
}

bool pirateFrbSend::send_frame(int frame_id,
                               uint8_t* intensity_frame, uint8_t* offset_scale_frame,
                               std::shared_ptr<pirateDestination> dest) {
    int32_t n = 0;
    int32_t n_sent = 0;

    //pirateDestination* dest = reinterpret_cast<pirateDestination*>(&_dest);

    if (!dest->sent_header) {
        INFO("Sending header");

        struct XEngineMetadata meta;
        meta.version = 2;

        // We need the upchannelization configuration to figure out the upchan zones.
        /*
          See upchan_pathfinder.j2, for instance:

          upchan_channel_ranges:
          64: [0, 1750]
          32: [1750, 2264]
          16: [2264, 2912]
          8: [2912, 3661]
          4: [3661, 4646]
          2: [4646, 5985]
          1: [5985, 8192]

          Where the channel numbers translate to frequencies via
          fengine_pathfinder.j2:

          #     A channel's frequency is   freq = channel * adc_frequency / num_samples_per_frame

          adc_frequency: 3.2e+9           # [Hz]
          num_samples_per_frame: 16384
          
        */

        double adc_freq = config.get<double>(unique_name, "adc_frequency");
        int samples_per_frame = config.get<int>(unique_name, "num_samples_per_frame");

        double MHz = 1e6;

        int channel_low = 0;
        double freq_low = channel_low * adc_freq / samples_per_frame / MHz;
        meta.zone_freq_edges.push_back(freq_low);

        //json upchan_ranges = config.get<std::vector<json> >(unique_name, "upchan_channel_ranges");
 
        //if (!upchan_ranges.is_object()) {
        //throw std::invalid_argument(fmt::format(fmt("Expect 'upchan_channel_ranges' to contain a dict")));
        //}
        //std::map<std::string, std::vector<int> >
        auto upchan_factors = config.get<std::vector<int> >(unique_name, "upchan_factors");
        upchan_factors.insert(upchan_factors.begin(), 1);

        // somehow it can't parse the map with int keys, they have to be strings.
        //auto upchan_ranges = config.get<std::map<int, std::vector<int> > >(unique_name, "upchan_channel_ranges");
        auto upchan_ranges = config.get<std::map<std::string, std::vector<int> > >(unique_name, "upchan_channel_ranges");
        //for (auto& it : upchan_ranges.items()) {
        //for (auto& it : upchan_ranges) {
            //int upchan_factor = it.key().get<int>();
        //int upchan_factor = std::stoi(it.key());
            //std::vector<int> channel_range = it.value().get<std::vector<int> >();

        // argh, keys not generated in yaml order
        //for (const auto& [key, value] : upchan_ranges) {
        //int upchan_factor = std::stoi(key);
        //std::vector<int> channel_range = value;
        while (upchan_factors.size()) {
            int key = upchan_factors.back();
            upchan_factors.pop_back();
            std::vector<int> channel_range = upchan_ranges[std::to_string(key)];
            int upchan_factor = key;

            if (channel_range.size() != 2) {
                throw std::invalid_argument(fmt::format(fmt("Expect 'upchan_channel_ranges' values to be a pair of integers")));
            }
            int lo = channel_range[0];
            int channel_high = channel_range[1];
            if (lo != channel_low) {
                throw std::invalid_argument(fmt::format(fmt("Expect 'upchan_channel_ranges' values to be contiguous: upchan {:d}, lo,hi [{:d}, {:d}], expected lo = {:d}"),
                                                            upchan_factor, lo, channel_high, channel_low));
            }
            double freq_high = channel_high * adc_freq / samples_per_frame / MHz;
            meta.zone_freq_edges.push_back(freq_high);
            int nfreq = upchan_factor * (channel_high - lo);
            meta.zone_nfreq.push_back(nfreq);

            // next one should have:
            channel_low = channel_high;
        }

        meta.freq_channels = config.get<std::vector<long> >(unique_name, "frequency_channels");

        for (int16_t b : dest->beam_ids) {
            INFO("Dest {:s}:{:d} gets beam id {:d}", dest->server_ip, dest->server_port, b);
            meta.beam_ids.push_back(b);
        }
        for (float p : dest->beam_positions)
            meta.beam_positions.push_back(p);

        meta.nbeams = dest->beam_ids.size();

        INFO("XEngineMetadata validate:");
        meta.validate();
        
        //std::string config = "yaml";
        std::string config = meta.to_yaml_string(true);

        INFO("Formatted X-engine metadata for Pirate: {:s}", config);
        
        struct pirateNetworkHeader header;
        header.magic_number = 0xf4bf4b01;
        header.config_length = config.size() + 1;
        size_t header_len = sizeof(struct pirateNetworkHeader);

        DEBUG2("Sending header");
        while ((n = send(dest->socket_fd, &((uint8_t*)&header)[n_sent], header_len - n_sent,
                         MSG_NOSIGNAL))
               > 0) {
            n_sent += n;
        }

        size_t config_len = header.config_length;
        n_sent = 0;
        DEBUG2("Sending config yaml");
        while ((n = send(dest->socket_fd, &((uint8_t*)config.c_str())[n_sent], config_len - n_sent,
                         MSG_NOSIGNAL))
               > 0) {
            n_sent += n;
        }
        dest->sent_header = true;
        INFO("Sent header + yaml (size {:d}) to dest {:s}:{:d}", config.size(), dest->server_ip, dest->server_port);
    }

    // Send 256-time-sample mini-chunks.

    //for (int chunk=0; chunk<(int)(frb_num_output_times / frb_chunk_size); chunk++) {
    /*
    // First the offset & scale:
    size_t chunksize = sizeof(float16_t) * 2 * frb_chunk_size * n_beams * upchan_total_channels;
    n_sent = 0;
    while ((n = send(dest->socket_fd,
    offset_scale_frame + (chunk * chunksize) + n_sent,
    chunksize - n_sent,
    MSG_NOSIGNAL))
    > 0) {
    n_sent += n;
    }
    // Then the intensities:
    chunksize = (n_beams * upchan_total_channels * frb_chunk_size) / 2;
    n_sent = 0;
    while ((n = send(dest->socket_fd,
    intensity_frame + (chunk * chunksize) + n_sent,
    chunksize - n_sent,
    MSG_NOSIGNAL))
    > 0) {
    n_sent += n;
    }
    */
    //}

    // The data come in frb_chunk_size == 256 time sample chunks.
    // The data array shape is Beams, Freqs, Times.
    // So we can just pull out our range of Beam indices for this dest.

    size_t dest_nbeams = dest->beam_index_max - dest->beam_index_min;
    size_t size_per_beam = sizeof(float16_t) * 2 * upchan_total_channels;
    size_t total_size = dest_nbeams * size_per_beam;
    // HACK -- we're assuming contiguous data packing...
    size_t offset = dest->beam_index_min * size_per_beam;

    n_sent = 0;
    while ((n = send(dest->socket_fd,
                     offset_scale_frame + offset + n_sent,
                     total_size - n_sent,
                     MSG_NOSIGNAL))
           > 0) {
        n_sent += n;
    }

    // Then the intensities:
    size_per_beam = upchan_total_channels * frb_chunk_size / 2;
    total_size = dest_nbeams * size_per_beam;
    offset = dest->beam_index_min * size_per_beam;
    
    n_sent = 0;
    while ((n = send(dest->socket_fd,
                     intensity_frame + offset + n_sent,
                     total_size - n_sent,
                     MSG_NOSIGNAL))
           > 0) {
        n_sent += n;
    }
    INFO("Sent frame {:d}: {:d} beams to {:s}:{:d}", frame_id, dest_nbeams, dest->server_ip, dest->server_port);
    
    return true;
}

