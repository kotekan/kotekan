#include "compressData.hpp"

#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"         // for wait_for_empty_frame, Buffer, allocate_new_metadata_object
#include "hfbMetadata.hpp"
#include "kotekanLogging.hpp" // for INFO

#include <assert.h>   // for assert
#include <atomic>     // for atomic_bool
#include <chrono>     // for duration, operator-, high_resolution_clock, time_point
#include <csignal>    // for raise, SIGINT
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <regex>      // for match_results<>::_Base_type
#include <stdlib.h>   // for malloc, free, realloc
#include <string.h>   // for memcpy
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(compressData);

compressData::compressData(Config& config_, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container, std::bind(&compressData::main_thread, this)) {

    // Apply config.
    _num_frb_total_beams = config.get<uint32_t>(unique_name, "num_frb_total_beams");
    _num_frames_to_integrate = config.get<uint32_t>(unique_name, "num_frames_to_integrate");
    _factor_upchan = config.get<uint32_t>(unique_name, "factor_upchan");

    in_buf = get_buffer("input_buf");
    register_consumer(in_buf, unique_name.c_str());

    out_buf = get_buffer("output_buf");
    register_producer(out_buf, unique_name.c_str());
}

compressData::~compressData() {
    free(in_buf);
}

int compressData::comparef(const float a, const float b) {
    return (a < b) ? 0 : 1;
}

void compressData::quantize_naive5(const float* in, char* out, const ssize_t n) {
    const float e0 = edges5[0];
    const float e1 = edges5[1];
    const float e2 = edges5[2];
    const float e3 = edges5[3];
    int val;

    for (ssize_t i = 0; i < n; i++) {
        const float v = in[i];
        val = 0;
        val += comparef(v, e0);
        val += comparef(v, e1);
        val += comparef(v, e2);
        val += comparef(v, e3);
        out[i] = (char)val;
    }
}

ssize_t compressData::huff_size_bound(const ssize_t samples_in) {
    ssize_t r = 0;
    ssize_t base = (samples_in * 3) / 8;

    if (((3 * samples_in) % 8) > 0) {
        r = 1;
    }

    return base + r;
}

ssize_t compressData::huff_encode(const char* in, uint32_t* out, const ssize_t n_in) {
    ssize_t iout = 0;    // tracks out index
    ssize_t bit_pos = 0; // tracks bit position within
    uint32_t hval;
    ssize_t this_size, ival;
    uint32_t shuttle;
    uint32_t tmp = 0;
    ssize_t r;
    for (ssize_t i = 0; i < n_in; i++) {
        shuttle = 0;
        ival = (ssize_t)in[i];
        hval = (uint32_t)codes[ival];
        this_size = (ssize_t)lens[ival];
        shuttle = hval << bit_pos;
        tmp += shuttle;

        bit_pos += this_size;

        if (bit_pos >= 32) {
            r = bit_pos - 32;
            out[iout] = tmp;

            tmp = 0;
            if (r > 0) {
                tmp += hval >> (this_size - r);
            }
            iout++;
            bit_pos = r;
        }
    }

    if (bit_pos > 0) {
        out[iout] = tmp;
    }

    return (iout + 1) * sizeof(uint32_t);
}

// assumes a uint32_t "chunk" structure
// provides an upper bound on the number of distinct samples
// coincidentally is the same as an upper bound on the size of the
// resulting unpacked array.
ssize_t compressData::huff_decompress_bound(const ssize_t size_in) {
    return size_in * 4;
}

ssize_t compressData::huff_decode(const uint32_t* in, uint8_t* out, const ssize_t n_in) {
    assert(n_in > 0);

    ssize_t ichunk = 0;
    ssize_t iout = 0;
    ssize_t bpos = 0;
    ssize_t this_len;
    ssize_t imatch;

    // this is lazy - to not deal with bit overrun I use a 64 bit int
    uint64_t v = ((uint64_t)in[0]) + (((uint64_t)in[1]) << 32);
    uint32_t tmp2, tmp3;

    // fix ichunk == n_in - 1 handling
    while (ichunk < n_in - 1) {
        // extract 2 and 3 bit codes starting at bpos
        tmp2 = (uint64_t)((v >> bpos) & 3);
        tmp3 = (uint64_t)((v >> bpos) & 7);

        if (tmp2 == codes32[2]) {
            imatch = 2;
        } else if (tmp2 == codes32[1]) {
            imatch = 1;
        } else if (tmp2 == codes32[3]) {
            imatch = 3;
        } else if (tmp3 == codes32[0]) {
            imatch = 0;
        } else {
            imatch = 4;
        }

        this_len = lens[imatch];
        out[iout] = (uint8_t)imatch;
        // INFO("%d | %d | %d | %d | %d", (uint32_t) out[iout], (uint32_t) verify[iout], bpos, tmp2,
        // tmp3);
        bpos += this_len;
        iout++;

        if (bpos >= 32) {
            ichunk++;
            bpos = bpos - 32;
            v = ((uint64_t)in[ichunk]) + (((uint64_t)in[ichunk + 1]) << 32);
        }
    }

    return iout;
}

void compressData::main_thread() {

    uint in_buffer_ID = 0; // Process only 1 GPU buffer, cycle through buffer depth
    uint8_t* in_frame;
    int out_frame_ID = 0;
    const uint32_t num_elements = _num_frb_total_beams * _factor_upchan;

    // Get the first output buffer which will always be id = 0 to start.
    uint8_t* out_frame = wait_for_empty_frame(out_buf, unique_name.c_str(), out_frame_ID);
    if (out_frame == nullptr)
        goto end_loop;

    while (!stop_thread) {
        // Get an input buffer, This call is blocking!
        in_frame = wait_for_full_frame(in_buf, unique_name.c_str(), in_buffer_ID);
        if (in_frame == nullptr)
            goto end_loop;

        float* input_data = (float*)in_buf->frames[in_buffer_ID];
        char* quant_data = (char*)malloc(num_elements * sizeof(char));

        // Use a simple quantization function - optimization is encouraged!
        quantize_naive5(input_data, quant_data, num_elements);

        // const ssize_t src_size = num_elements * sizeof(char);

        // Compute an upper bound on bit size of encoded huffman data
        const ssize_t max_dst_size = huff_size_bound(num_elements);

        // Allocate a uint32_t "safe" sized array. While this is a bit-level coding, I
        // find that chunking the huffman stream into a larger data type e.g. 32 or 64 bits
        // is helpful for efficiency.

        uint32_t* compressed_data =
            (uint32_t*)malloc(((max_dst_size / sizeof(uint32_t)) + 1) * sizeof(uint32_t));

        // Execute and time the encoding step; returns the byte size of the encoded stream
        // reallocate to realize the space savings (or do whatever you want)
        auto start0 = std::chrono::high_resolution_clock::now();
        const ssize_t compressed_data_size = huff_encode(quant_data, compressed_data, num_elements);
        auto end0 = std::chrono::high_resolution_clock::now();

        // Reallocate to achieve the actual space savings.
        compressed_data = (uint32_t*)realloc(compressed_data, compressed_data_size);
        if (compressed_data == nullptr) {
            INFO("failed to reallocate the data");
        }

        // Perform decompression to verify that this was done properly (also test timing)
        ssize_t n_in = compressed_data_size / sizeof(uint32_t);
        uint8_t* decode_buf = (uint8_t*)malloc(huff_decompress_bound(compressed_data_size));

        // NOTE: here I structure the program to actually determine the number of recovered samples
        // from the compressed stream but I really don't think this is a good idea. I think the
        // better thing to do would be to prepend the number of samples in a fixed-length datatype
        // at the beginning of the huffman stream (e.g. if you're writing it to disk) and just
        // recover the data using this info. Right now there's a degeneracy between samples that
        // correspond to zero and trailing zeroes in the last chunk. This confuses the decoder and
        // no amount of logic on the raw (encoded) sample chunk stream can break this degeneracy.

        auto start1 = std::chrono::high_resolution_clock::now();
        const ssize_t recovered_samples = huff_decode(compressed_data, decode_buf, n_in);
        auto end1 = std::chrono::high_resolution_clock::now();

        // std::cout << "input | output" << std::endl;
        // for(ssize_t i = 0; i < 16; i++){
        // 	std::cout << ((uint8_t) quant_data[i]) << " | " << decode_buf[i] << std::endl;
        // }

        // check every sample

        bool check = true;

        for (ssize_t i = 0; i < recovered_samples; i++) {
            if (((uint8_t)quant_data[i]) != decode_buf[i]) {
                check = false;
                break;
            }
        }

        if (!check) {
            INFO("decompressed samples failed consistency check!");
            std::raise(SIGINT);
        }

        // report efficiency and performance

        std::chrono::duration<double> elapsed0 = end0 - start0;
        std::chrono::duration<double> elapsed1 = end1 - start1;

        INFO("compression rate (MB/s): %f",
             float(num_elements * sizeof(char)) * 1e-6 / elapsed0.count());
        INFO("decompression ratio (to original): %f",
             float(recovered_samples) / float(num_elements));
        INFO("decompression rate (MB/s): %f",
             float(recovered_samples * sizeof(uint8_t)) * 1e-6 / elapsed1.count());

        INFO("Compressed data size: %d for %d elements", compressed_data_size, num_elements);
        INFO("Success! Compression ratio: %f",
             float(compressed_data_size * 8) / (8 * float(num_elements)));
        INFO("Information efficiency (<=1; 1 is optimal): %f",
             (entropy5 * float(num_elements)) / float(compressed_data_size * 8));
        INFO("bit efficiency (can be >1 with a good coding scheme): %f",
             (bitsize * float(num_elements)) / float(compressed_data_size * 8));

        memcpy(out_buf->frames[out_frame_ID], compressed_data, compressed_data_size);

        // Set compressed size in metadata of frame
        allocate_new_metadata_object(out_buf, out_frame_ID);
        set_compressed_data_size_hfb(out_buf, out_frame_ID, compressed_data_size);

        mark_frame_full(out_buf, unique_name.c_str(), out_frame_ID);

        // Get a new output buffer
        out_frame_ID = (out_frame_ID + 1) % out_buf->num_frames;
        out_frame = wait_for_empty_frame(out_buf, unique_name.c_str(), out_frame_ID);
        if (out_frame == nullptr)
            goto end_loop;

        // Release the input buffers
        mark_frame_empty(in_buf, unique_name.c_str(), in_buffer_ID);
        in_buffer_ID = (in_buffer_ID + 1) % in_buf->num_frames;

    } // end stop thread
end_loop:;
}
