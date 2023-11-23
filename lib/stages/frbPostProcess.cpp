#include "frbPostProcess.hpp"

#include "Config.hpp"            // for Config
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"         // for Telescope
#include "buffer.hpp"            // for Buffer, mark_frame_empty, wait_for_full_frame, register...
#include "bufferContainer.hpp"   // for bufferContainer
#include "chimeMetadata.hpp"     // for get_fpga_seq_num
#include "kotekanLogging.hpp"    // for DEBUG, INFO
#include "prometheusMetrics.hpp" // for Metrics, Counter

#include "fmt.hpp" // for format, fmt

#include <algorithm>   // for find, max, min
#include <atomic>      // for atomic_bool
#include <cstdint>     // for int32_t
#include <exception>   // for exception
#include <functional>  // for _Bind_helper<>::type, bind, function
#include <immintrin.h> // for _mm256_broadcast_ss, __m256, _mm256_load_ps, _mm256_min_ps
#include <mm_malloc.h> // for posix_memalign
#include <regex>       // for match_results<>::_Base_type
#include <stdexcept>   // for runtime_error
#include <stdlib.h>    // for free, calloc, malloc
#include <string.h>    // for memcpy, memset
#include <sys/types.h> // for uint
#include <time.h>      // for timespec
#include <xmmintrin.h> // for _mm_max_ps, _mm_min_ps, _mm_store_ss, __m128, _mm_shuff...


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(frbPostProcess);

frbPostProcess::frbPostProcess(Config& config_, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container, std::bind(&frbPostProcess::main_thread, this)),
    masked_packets_counter(kotekan::prometheus::Metrics::instance().add_counter(
        "kotekan_frb_masked_packets_total", unique_name)) {
    // Apply config.
    _num_gpus = config.get<int32_t>(unique_name, "num_gpus");
    _samples_per_data_set = config.get<int32_t>(unique_name, "samples_per_data_set");
    _downsample_time = config.get<int32_t>(unique_name, "downsample_time");
    _factor_upchan = config.get<int32_t>(unique_name, "factor_upchan");
    _factor_upchan_out = config.get<int32_t>(unique_name, "factor_upchan_out");
    _nbeams = config.get<int32_t>(unique_name, "num_beams_per_frb_packet");
    _timesamples_per_frb_packet = config.get<int32_t>(unique_name, "timesamples_per_frb_packet");

    std::vector<int32_t> bd;
    _incoherent_beams =
        config.get_default<std::vector<int32_t>>(unique_name, "incoherent_beams", bd);
    _incoherent_truncation = config.get_default<float>(unique_name, "incoherent_truncation", 1e10);

    num_L1_streams = 1024 / _nbeams;
    num_samples = _samples_per_data_set / _downsample_time / _factor_upchan;

    fpga_counts_per_sample = _downsample_time * _factor_upchan;
    udp_header_size = sizeof(struct FRBHeader) + sizeof(uint16_t) * _nbeams // beam ids
                      + sizeof(uint16_t) * _num_gpus                        // freq band ids
                      + sizeof(float) * _nbeams * _num_gpus                 // scales
                      + sizeof(float) * _nbeams * _num_gpus                 // offsets
        ;
    udp_packet_size =
        _nbeams * _num_gpus * _factor_upchan_out * _timesamples_per_frb_packet + udp_header_size;

    in_buf = (Buffer**)malloc(_num_gpus * sizeof(Buffer*));
    for (int i = 0; i < _num_gpus; ++i) {
        in_buf[i] = get_buffer(fmt::format(fmt("in_buf_{:d}"), i));
        in_buf[i]->register_consumer(unique_name);
    }
    frb_buf = get_buffer("out_buf");
    frb_buf->register_producer(unique_name);

    lost_samples_buf = get_buffer("lost_samples_buf");
    lost_samples_buf->register_consumer(unique_name);
    lost_samples_buf_id = 0;

    // Dynamic header
    frb_header_beam_ids = new uint16_t[_nbeams];
    frb_header_coarse_freq_ids = new uint16_t[_num_gpus];
    frb_header_scale = new float[_nbeams * _num_gpus];
    frb_header_offset = new float[_nbeams * _num_gpus];

    droppacket = (uint8_t*)calloc(num_samples, sizeof(uint8_t));

    if (posix_memalign((void**)&ib, 32,
                       _num_gpus * num_samples * _factor_upchan_out * sizeof(float))) {
        throw std::runtime_error("Couldn't allocate frbPostProcess memory.");
    }
}

frbPostProcess::~frbPostProcess() {
    free(in_buf);
    free(frb_header_beam_ids);
    free(frb_header_coarse_freq_ids);
    free(frb_header_scale);
    free(frb_header_offset);
    free(droppacket);
    free(ib);
}

void frbPostProcess::write_header(unsigned char* dest) {
    memcpy(dest, &frb_header, sizeof(struct FRBHeader));
    dest += sizeof(struct FRBHeader);

    memcpy(dest, frb_header_beam_ids, sizeof(uint16_t) * _nbeams);
    dest += sizeof(uint16_t) * _nbeams;

    memcpy(dest, frb_header_coarse_freq_ids, sizeof(uint16_t) * _num_gpus);
    dest += sizeof(uint16_t) * _num_gpus;

    memcpy(dest, frb_header_scale, sizeof(float) * _nbeams * _num_gpus);
    dest += sizeof(float) * _nbeams * _num_gpus;

    memcpy(dest, frb_header_offset, sizeof(float) * _nbeams * _num_gpus);
}

#ifdef __AVX2__
void frbPostProcess::main_thread() {

    auto& tel = Telescope::instance();

    uint in_buffer_ID[_num_gpus]; // 4 of these , cycle through buffer depth
    uint8_t* in_frame[_num_gpus];
    int out_buffer_ID = 0;

    for (int i = 0; i < _num_gpus; ++i)
        in_buffer_ID[i] = 0;

    frb_header.protocol_version = 2;
    frb_header.data_nbytes = udp_packet_size - udp_header_size;
    frb_header.fpga_counts_per_sample = fpga_counts_per_sample;
    const auto fpga0_ns = tel.to_time(0);
    frb_header.fpga0_ns = fpga0_ns.tv_sec * 1'000'000'000 + fpga0_ns.tv_nsec;
    frb_header.fpga_count = 0;           // to be updated in fill_header
    frb_header.nbeams = _nbeams;         // 4
    frb_header.nfreq_coarse = _num_gpus; // 4
    frb_header.nupfreq = _factor_upchan_out;
    frb_header.ntsamp = _timesamples_per_frb_packet;

    bool startup = true; // True on first pass through the loop

    while (!stop_thread) {
        // Get the next output buffer, id = 0 to start.
        uint8_t* out_frame = frb_buf->wait_for_empty_frame(unique_name, out_buffer_ID);
        if (out_frame == nullptr)
            return;
        // Get an input buffer, This call is blocking!
        for (int i = 0; i < _num_gpus; ++i) {
            in_frame[i] = in_buf[i]->wait_for_full_frame(unique_name, in_buffer_ID[i]);
            if (in_frame[i] == nullptr)
                return;
        }

        // Information on drop packets
        uint8_t* lost_samples_frame =
            lost_samples_buf->wait_for_full_frame(unique_name, lost_samples_buf_id);
        if (lost_samples_frame == nullptr)
            return;

        // Get all input buffers in sync by fpga_seq_no: find the one that's the furthest along, and
        // keep advancing others until they all match. (Keep in mind that advancing one of the
        // others may put it ahead of the current largest fpga_seq_no, in which case we have to
        // repeat the process.)
        int64_t start_fpga_count = frb_header.fpga_count;
        while (!stop_thread) {
            // Find out the amount by which inputs are out of sync
            int64_t max_fpga_count = get_fpga_seq_num(in_buf[0], in_buffer_ID[0]);
            for (int i = 1; i < _num_gpus; i++) {
                max_fpga_count =
                    std::max(max_fpga_count, get_fpga_seq_num(in_buf[i], in_buffer_ID[i]));
            }

            // On startup, we just go with whatever is the smallest input fpga_seq_num.
            // Afterwards, we know what the last frame was, and so can just count from there
            if (startup) {
                start_fpga_count = max_fpga_count;
                for (int i = 1; i < _num_gpus; i++) {
                    start_fpga_count =
                        std::min(start_fpga_count, get_fpga_seq_num(in_buf[i], in_buffer_ID[i]));
                }
                startup = false;
            }

            DEBUG("Syncing min {} -> {} max", start_fpga_count, max_fpga_count);

            // Advance lost_samples buffer by the amount of known dropped frames across all inputs
            for (int i = 0; i < (max_fpga_count - start_fpga_count) / _samples_per_data_set; ++i) {
                INFO("Advance lost_samples");
                lost_samples_buf->mark_frame_empty(unique_name, lost_samples_buf_id);
                lost_samples_buf_id = (lost_samples_buf_id + 1) % lost_samples_buf->num_frames;
                lost_samples_frame =
                    lost_samples_buf->wait_for_full_frame(unique_name, lost_samples_buf_id);
                if (lost_samples_frame == nullptr)
                    return;
            }
            start_fpga_count = max_fpga_count;

            // try to catch up the GPU inputs
            bool fpga_seq_in_sync = true;
            for (int i = 0; i < _num_gpus; ++i) {
                while (max_fpga_count > get_fpga_seq_num(in_buf[i], in_buffer_ID[i])) {
                    INFO("Advance {} from {}", i, get_fpga_seq_num(in_buf[i], in_buffer_ID[i]));
                    in_buf[i]->mark_frame_empty(unique_name, in_buffer_ID[i]);
                    in_buffer_ID[i] = (in_buffer_ID[i] + 1) % in_buf[i]->num_frames;
                    in_frame[i] = in_buf[i]->wait_for_full_frame(unique_name, in_buffer_ID[i]);
                    if (in_frame[i] == nullptr)
                        return;
                }
                if (max_fpga_count != get_fpga_seq_num(in_buf[i], in_buffer_ID[i])) {
                    fpga_seq_in_sync = false;
                }
                frb_header_coarse_freq_ids[i] = tel.to_freq_id(in_buf[i], in_buffer_ID[i]);
            }

            if (fpga_seq_in_sync) {
                frb_header.fpga_count = max_fpga_count;
                DEBUG("Synced to {}", max_fpga_count);
                break;
            }
        }
        if (stop_thread)
            return;

        for (uint t = 0; t < num_samples; t++) {
            // check if drop packet by reading 384 original times, if so flag that t
            droppacket[t] = 0;
            for (int tz = 0; tz < _downsample_time * _factor_upchan; tz++) {
                if (lost_samples_frame[t * _factor_upchan * _downsample_time + tz] == 1) {
                    droppacket[t] = 1;
                    break;
                }
            }
        }

        // Sum all the beams together into ib array.
        if (_incoherent_beams.size() > 0) {
            memset(ib, 0, _num_gpus * num_samples * _factor_upchan_out * sizeof(float));
            for (int thread_id = 0; thread_id < _num_gpus; thread_id++) { // loop 4 GPUs (input)
                float* in_data = (float*)in_frame[thread_id];
                for (uint t = 0; t < num_samples; t++) {
                    float norm = 1. / _nbeams / num_L1_streams;
                    float ce = _incoherent_truncation / norm;
                    __m256 _ce = _mm256_broadcast_ss(&ce);
                    if (droppacket[t] == 1) {
                        // zero output with dropped packet by setting norm to zero
                        norm = 0.0;
                    }
                    __m256 _norm = _mm256_broadcast_ss(&norm);
                    for (int32_t f = 0; f < _factor_upchan_out;
                         f += (sizeof(__m256) / sizeof(float))) { // loop over freq , each +8
                        int idx = t * _factor_upchan_out + f;
                        __m256 _a =
                            _mm256_load_ps(ib + thread_id * num_samples * _factor_upchan_out + idx);
                        for (int b = 0; b < num_L1_streams * _nbeams; b++) {     // loop 1024 beams
                            int idx_next = b * num_samples * _factor_upchan_out; // b*128*16
                            __m256 _b = _mm256_load_ps(in_data + idx + idx_next);
                            // limit the max value in e.g. the coherent beam
                            _b = _mm256_min_ps(_b, _ce);
                            __m256 _c = _mm256_fmadd_ps(_b, _norm, _a); // SUMMING
                            _mm256_store_ps(ib + thread_id * num_samples * _factor_upchan_out + idx,
                                            _c);
                        } // end loop b
                    }     // end loop f
                }         // end loop t
            }
        }

        float ofs, scl, off;
        for (uint T = 0; T < num_samples;
             T += _timesamples_per_frb_packet) {                      // loop 128 time samples, in 8
            for (int stream = 0; stream < num_L1_streams; stream++) { // loop 256 streams (output)
                for (int b = 0; b < _nbeams; b++) {                   // loop 4 beams / stream
                    int beam_id = stream * _nbeams + b;
                    // frb_header_beam_ids[b] = beam_id;
                    // Changing to beam id convention 0->255, 1000->1255, 2000->2255, 3000->3255
                    frb_header_beam_ids[b] = (beam_id) % 256 + (int((beam_id) / 256) * 1000);
                    for (int thread_id = 0; thread_id < _num_gpus; thread_id++) { // loop 4 GPUs
                        float* in_data =
                            ((float*)in_frame[thread_id])
                            + (stream * _nbeams + b) * num_samples * _factor_upchan_out;
                        if (std::find(_incoherent_beams.begin(), _incoherent_beams.end(), beam_id)
                            != _incoherent_beams.end()) {
                            DEBUG("Incoherent beam! Stream {:d}, Beam {:d}; ID {:d}", stream, b,
                                  beam_id);
                            in_data = ib + thread_id * num_samples * _factor_upchan_out;
                        }
                        // pre-set to zero, in case all samples dropped within these 16 t
                        float zero = 0.0;
                        __m256 _mx = _mm256_broadcast_ss(&zero);
                        __m256 _mn = _mm256_broadcast_ss(&zero);
                        __m256 _cA = _mm256_broadcast_ss(&zero);
                        __m256 _cB = _mm256_broadcast_ss(&zero);
                        // AVX2 option, fastest of a few I tried
                        bool firstvalue = true;
                        for (int t = 0; t < _timesamples_per_frb_packet; t++) {
                            if (droppacket[T + t] != 1) {
                                int idx = (T + t) * _factor_upchan_out;
                                _cA = _mm256_load_ps(in_data + idx);     // 8f
                                _cB = _mm256_load_ps(in_data + idx + 8); // 8f
                                if (firstvalue) {
                                    _mx = _mm256_max_ps(_cA, _cB);
                                    _mn = _mm256_min_ps(_cA, _cB);
                                    firstvalue = false;
                                } else {
                                    _mx = _mm256_max_ps(_mx, _mm256_max_ps(_cA, _cB));
                                    _mn = _mm256_min_ps(_mn, _mm256_min_ps(_cA, _cB));
                                }
                            } // end if drop packet
                        }

                        // Calc scale and offset
                        float min, max;
                        __m128 mx = _mm_max_ps(_mm256_extractf128_ps(_mx, 0),
                                               _mm256_extractf128_ps(_mx, 1));
                        __m128 mn = _mm_min_ps(_mm256_extractf128_ps(_mn, 0),
                                               _mm256_extractf128_ps(_mn, 1));
                        for (int u = 0; u < 3; u++) {
                            mx = _mm_max_ps(mx,
                                            _mm_shuffle_ps(mx, mx, 0b10010011)); // 2,1,0,3 = 0x93
                            mn = _mm_min_ps(mn,
                                            _mm_shuffle_ps(mn, mn, 0b10010011)); // 2,1,0,3 = 0x93
                        }
                        _mm_store_ss(&max, mx);
                        _mm_store_ss(&min, mn);
                        if (firstvalue) {
                            // all times dropped within this frb packet
                            frb_header_scale[b * _num_gpus + thread_id] = 0.0;
                            frb_header_offset[b * _num_gpus + thread_id] = 0.0;
                            scl = 0.0;
                            ofs = 0.0;
                            masked_packets_counter.inc();
                        } else {
                            // scale to 1-254 (0 and 255 are both error codes)
                            scl = (253.) / (max - min);
                            ofs = min - 1 / scl; // offset by 1, so 1-254
                            frb_header_scale[b * _num_gpus + thread_id] = 1. / scl;
                            frb_header_offset[b * _num_gpus + thread_id] = ofs;
                        }
                        // Apply scale and offset
                        int f_per_m = sizeof(__m256) / sizeof(float);
                        unsigned char utr[256], tr[256];
                        for (int t = 0; t < _timesamples_per_frb_packet; t++) {
                            if (droppacket[T + t] == 1) {
                                scl = 0.0;
                                off = 0.0;
                            } else {
                                scl = (253.) / (max - min);
                                off = -ofs * scl;
                            }
                            __m256 _scl = _mm256_broadcast_ss(&scl);
                            __m256 _ofs = _mm256_broadcast_ss(&off);
                            for (int f = 0; f < _factor_upchan_out; f += f_per_m) {
                                uint32_t in_index = (T + t) * _factor_upchan_out + f;
                                __m256 _in = _mm256_load_ps(in_data + in_index);
                                __m256 _out =
                                    _mm256_fmadd_ps(_in, _scl, _ofs); // now [0-255]  // APPLY!
                                // extract -- probably a better way to do this...
                                __m256i _y =
                                    _mm256_cvtps_epi32(_out);     // Convert them to 32-bit ints
                                _y = _mm256_packus_epi32(_y, _y); // Pack down to 16 bits
                                _y = _mm256_packus_epi16(_y, _y); // Pack down to 8 bits
                                *(int32_t*)(utr + t * 16 + f) = _mm256_extract_epi32(_y, 0);
                                *(int32_t*)(utr + t * 16 + f + 4) = _mm256_extract_epi32(_y, 4);
                            } // end freq f
                        }     // end time t
                        // transpose
                        for (int t = 0; t < 16; t++)
                            for (int f = 0; f < 16; f++)
                                tr[f * 16 + t] = utr[t * 16 + f];
                        // copy all the data out
                        uint32_t out_index =
                            stream * udp_packet_size * num_samples / _timesamples_per_frb_packet
                            + (T / _timesamples_per_frb_packet) * udp_packet_size
                            + b * _num_gpus * 16 * 16 + thread_id * 16 * 16 + udp_header_size;
                        memcpy(out_frame + out_index, tr, 16 * 16);
                    } // end 4 GPUs
                }     // end 4 nbeam
                // Fill the headers of the packet
                uint32_t out_index =
                    stream * udp_packet_size * num_samples / _timesamples_per_frb_packet
                    + (T / _timesamples_per_frb_packet) * udp_packet_size;
                write_header(&out_frame[out_index]);
            } // end 256 streams
            frb_header.fpga_count += fpga_counts_per_sample * _timesamples_per_frb_packet;
        } // end looping 128 time samples

        frb_buf->mark_frame_full(unique_name, out_buffer_ID);
        out_buffer_ID = (out_buffer_ID + 1) % frb_buf->num_frames;

        lost_samples_buf->mark_frame_empty(unique_name, lost_samples_buf_id);
        lost_samples_buf_id = (lost_samples_buf_id + 1) % lost_samples_buf->num_frames;

        // Release the input buffers
        for (int i = 0; i < _num_gpus; ++i) {
            // release_info_object(in_buf[gpu_id], in_buffer_ID[i]);
            in_buf[i]->mark_frame_empty(unique_name, in_buffer_ID[i]);
            in_buffer_ID[i] = (in_buffer_ID[i] + 1) % in_buf[i]->num_frames;
        }
    } // end stop thread
}
#else
void frbPostProcess::main_thread() {
    ERROR("No AVX2 intrinsics present on this node")
}
#endif
