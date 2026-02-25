#ifndef TEST_DATA_CHECK_H
#define TEST_DATA_CHECK_H

#include "Config.hpp"          // for Config
#include "N2Metadata.hpp"      // for N2Metadata, get_N2_metadata, metadata_is_N2
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata, get_chord_metadata, metadata_is_chord, CHO...
#include "errors.h"            // for TEST_FAILED, TEST_PASSED
#include "kotekanLogging.hpp"  // for INFO, ERROR, DEBUG, FATAL_ERROR
#include "metadata.hpp"        // for metadataObject
#include "visUtil.hpp"         // for format_nice_string

#include "fmt.hpp" // for compile_string_to_view

#include <algorithm>   // for max
#include <assert.h>    // for assert
#include <cmath>       // for abs
#include <cstdint>     // for uint32_t, int32_t, uint8_t
#include <functional>  // for bind
#include <limits>      // for numeric_limits
#include <memory>      // for shared_ptr, __shared_ptr_access
#include <string.h>    // for strncmp
#include <string>      // for allocator, basic_string, operator!=, string
#include <type_traits> // for enable_if
#include <vector>      // for vector

#define CHECK_META_SCALAR_INT_DIRECT(FIELD, META1, META2, ERR_COUNT, BUF_NAME1, FRAME_ID1,         \
                                     BUF_NAME2, FRAME_ID2)                                         \
    do {                                                                                           \
        DEBUG2("Checking meta field {:s}", #FIELD);                                                \
        if ((META1)->FIELD != (META2)->FIELD) {                                                    \
            ERROR("metadata {:s}[{:d}] {:s} != {:s}[{:d}] {:s}; values: {:d} {:d}", (BUF_NAME1),   \
                  (FRAME_ID1), #FIELD, (BUF_NAME2), (FRAME_ID2), #FIELD, (META1)->FIELD,           \
                  (META2)->FIELD);                                                                 \
            (ERR_COUNT)++;                                                                         \
        }                                                                                          \
    } while (0)

#define CHECK_META_SCALAR_INT(FIELD, META1, META2, ERR_COUNT, BUF_NAME1, FRAME_ID1, BUF_NAME2,     \
                              FRAME_ID2)                                                           \
    do {                                                                                           \
        DEBUG2("Checking meta field {:s}", #FIELD);                                                \
        if ((META1)->get_##FIELD() != (META2)->get_##FIELD()) {                                    \
            ERROR("metadata {:s}[{:d}] {:s} != {:s}[{:d}] {:s}; values: {:d} {:d}", (BUF_NAME1),   \
                  (FRAME_ID1), #FIELD, (BUF_NAME2), (FRAME_ID2), #FIELD, (META1)->get_##FIELD(),   \
                  (META2)->get_##FIELD());                                                         \
            (ERR_COUNT)++;                                                                         \
        }                                                                                          \
    } while (0)

#define CHECK_META_SCALAR_STREAM_T(FIELD, META1, META2, ERR_COUNT, BUF_NAME1, FRAME_ID1,           \
                                   BUF_NAME2, FRAME_ID2)                                           \
    do {                                                                                           \
        DEBUG2("Checking meta field {:s}", #FIELD);                                                \
        if ((META1)->get_##FIELD().id != (META2)->get_##FIELD().id) {                              \
            ERROR("metadata {:s}[{:d}] {:s} != {:s}[{:d}] {:s}; values: {:d} {:d}", (BUF_NAME1),   \
                  (FRAME_ID1), #FIELD, (BUF_NAME2), (FRAME_ID2), #FIELD,                           \
                  (META1)->get_##FIELD().id, (META2)->get_##FIELD().id);                           \
            (ERR_COUNT)++;                                                                         \
        }                                                                                          \
    } while (0)

#define CHECK_META_ARR1_INT(FIELD, LEN, META1, META2, ERR_COUNT, BUF_NAME1, FRAME_ID1, BUF_NAME2,  \
                            FRAME_ID2)                                                             \
    do {                                                                                           \
        DEBUG2("Checking meta field {:s}", #FIELD);                                                \
        for (int meta_idx = 0; meta_idx < (LEN); meta_idx++) {                                     \
            if ((META1)->get_##FIELD()[meta_idx] != (META2)->get_##FIELD()[meta_idx]) {            \
                ERROR(                                                                             \
                    "metadata {:s}[{:d}] {:s}[{:d}] != {:s}[{:d}] {:s}[{:d}]; values: {:d} {:d}",  \
                    (BUF_NAME1), (FRAME_ID1), #FIELD, meta_idx, (BUF_NAME2), (FRAME_ID2), #FIELD,  \
                    meta_idx, (META1)->get_##FIELD()[meta_idx], (META2)->get_##FIELD()[meta_idx]); \
                (ERR_COUNT)++;                                                                     \
            }                                                                                      \
        }                                                                                          \
    } while (0)

#define CHECK_META_ARR1_INT_DIRECT(FIELD, LEN, META1, META2, ERR_COUNT, BUF_NAME1, FRAME_ID1,      \
                                   BUF_NAME2, FRAME_ID2)                                           \
    do {                                                                                           \
        DEBUG2("Checking meta field {:s}", #FIELD);                                                \
        for (int meta_idx = 0; meta_idx < (LEN); meta_idx++) {                                     \
            if ((META1)->FIELD[meta_idx] != (META2)->FIELD[meta_idx]) {                            \
                ERROR(                                                                             \
                    "metadata {:s}[{:d}] {:s}[{:d}] != {:s}[{:d}] {:s}[{:d}]; values: {:d} {:d}",  \
                    (BUF_NAME1), (FRAME_ID1), #FIELD, meta_idx, (BUF_NAME2), (FRAME_ID2), #FIELD,  \
                    meta_idx, (META1)->FIELD[meta_idx], (META2)->FIELD[meta_idx]);                 \
                (ERR_COUNT)++;                                                                     \
            }                                                                                      \
        }                                                                                          \
    } while (0)

#define CHECK_META_SCALAR_STR(FIELD, META1, META2, ERR_COUNT, BUF_NAME1, FRAME_ID1, BUF_NAME2,     \
                              FRAME_ID2)                                                           \
    do {                                                                                           \
        DEBUG2("Checking meta field {:s}", #FIELD);                                                \
        if ((META1)->get_##FIELD() != (META2)->get_##FIELD()) {                                    \
            ERROR("metadata {:s}[{:d}] {:s} != {:s}[{:d}] {:s}; values: {:s} {:s}", (BUF_NAME1),   \
                  (FRAME_ID1), #FIELD, (BUF_NAME2), (FRAME_ID2), #FIELD, (META1)->get_##FIELD(),   \
                  (META2)->get_##FIELD());                                                         \
            (ERR_COUNT)++;                                                                         \
        }                                                                                          \
    } while (0)

#define CHECK_META_ARR1_CSTR_DIRECT(FIELD, ARR_LEN, STR_LEN, META1, META2, ERR_COUNT, BUF_NAME1,   \
                                    FRAME_ID1, BUF_NAME2, FRAME_ID2)                               \
    do {                                                                                           \
        DEBUG2("Checking meta field {:s}", #FIELD);                                                \
        for (int meta_idx = 0; meta_idx < (ARR_LEN); meta_idx++) {                                 \
            if (strncmp((META1)->FIELD[meta_idx], (META2)->FIELD[meta_idx], (STR_LEN))) {          \
                ERROR(                                                                             \
                    "metadata {:s}[{:d}] {:s}[{:d}] != {:s}[{:d}] {:s}[{:d}]; values: {:s} {:s}",  \
                    (BUF_NAME1), (FRAME_ID1), #FIELD, meta_idx, (BUF_NAME2), (FRAME_ID2), #FIELD,  \
                    meta_idx, (META1)->FIELD[meta_idx], (META2)->FIELD[meta_idx]);                 \
                (ERR_COUNT)++;                                                                     \
            }                                                                                      \
        }                                                                                          \
    } while (0)

template<typename A_Type>
class testDataCheck : public kotekan::Stage {
public:
    testDataCheck(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container);
    ~testDataCheck();
    void main_thread() override;

private:
    Buffer* first_buf;
    Buffer* second_buf;
    int num_frames_to_test;
    int max_num_errors_logged;
    double epsilon;
    bool trigger_exit_on_pass;
    bool check_metadata;

    int check_chord_metadata(const std::shared_ptr<const chordMetadata> meta1,
                             const std::shared_ptr<const chordMetadata> meta2, int first_buf_id,
                             int second_buf_id);
    int check_N2_metadata(const std::shared_ptr<const N2Metadata> meta1,
                          const std::shared_ptr<const N2Metadata> meta2, int first_buf_id,
                          int second_buf_id);
};

template<typename A_Type>
testDataCheck<A_Type>::testDataCheck(kotekan::Config& config, const std::string& unique_name,
                                     kotekan::bufferContainer& buffer_container) :
    kotekan::Stage(config, unique_name, buffer_container,
                   std::bind(&testDataCheck::main_thread, this)) {
    first_buf = get_buffer("first_buf");
    first_buf->register_consumer(unique_name);
    second_buf = get_buffer("second_buf");
    second_buf->register_consumer(unique_name);

    num_frames_to_test = config.get_default<int32_t>(unique_name, "num_frames_to_test", 0);
    max_num_errors_logged = config.get_default<int32_t>(unique_name, "max_num_errors_logged", 100);
    epsilon = config.get_default<double>(unique_name, "epsilon",
                                         std::numeric_limits<A_Type>::epsilon() * (A_Type)5.0);
    trigger_exit_on_pass = config.get_default<bool>(unique_name, "trigger_exit_on_pass", true);

    check_metadata = config.get_default<bool>(unique_name, "check_metadata", false);
}

template<typename A_Type>
testDataCheck<A_Type>::~testDataCheck() {}

template<typename A_Type>
typename std::enable_if<!std::numeric_limits<A_Type>::is_integer, bool>::type
almost_equal(A_Type x, A_Type y, double epsilon) {
    // the machine epsilon has to be scaled to the magnitude of the values
    return std::abs(x - y) <= epsilon * std::abs(x + y)
           // unless the result is subnormal
           || std::abs(x - y) < std::numeric_limits<A_Type>::min();
}

template<typename A_Type>
void testDataCheck<A_Type>::main_thread() {

    int first_buf_id = 0, second_buf_id = 0, num_errors = 0, frames = 0;

    assert(first_buf->frame_size == second_buf->frame_size);

    while (!stop_thread) {

        // Get both full frames
        uint8_t* first_frame = first_buf->wait_for_full_frame(unique_name, first_buf_id);
        if (first_frame == nullptr)
            break;
        DEBUG("testDataCheck: Got the first buffer {:s}[{:d}]", first_buf->buffer_name,
              first_buf_id);
        uint8_t* second_frame = second_buf->wait_for_full_frame(unique_name, second_buf_id);
        if (second_frame == nullptr)
            break;
        DEBUG("testDataCheck: Got the second buffer {:s}[{:d}]", second_buf->buffer_name,
              second_buf_id);

        num_errors = 0;

        uint32_t num_elements = first_buf->frame_size / sizeof(A_Type);
        double abs_diff = 0.0;
        double rel_diff = 0.0;
        uint32_t num_nonzero = 0;

        bool use_almost_equal = !std::is_integral_v<A_Type> && (epsilon != 0.0);

        INFO(
            "Checking that the buffers {:s}[{:d}] and {:s}[{:d}] match, this could take a while...",
            first_buf->buffer_name, first_buf_id, second_buf->buffer_name, second_buf_id);

        for (uint32_t i = 0; i < num_elements; ++i) {
            A_Type first_value = *((A_Type*)&(first_frame[i * sizeof(A_Type)]));
            A_Type second_value = *((A_Type*)&(second_frame[i * sizeof(A_Type)]));

            if (use_almost_equal) {
                double v1 = (double)first_value;
                double v2 = (double)second_value;
                abs_diff += std::abs(v1 - v2);
                double absum = (std::abs(v1) + std::abs(v2));
                if (absum != 0.0) {
                    rel_diff += std::abs(v1 - v2) / absum;
                    num_nonzero++;
                }

                if (!almost_equal(v1, v2, epsilon)) {
                    num_errors += 1;
                    if (num_errors < max_num_errors_logged) {
                        ERROR("{:s}[{:d}][{:d}] != {:s}[{:d}][{:d}]; values: ({:f}, {:f}), "
                              "epsilon: {:f}, "
                              "abs(x-y): {:f}, epsilon * abs(x+y): {:f}",
                              first_buf->buffer_name, first_buf_id, i, second_buf->buffer_name,
                              second_buf_id, i, v1, v2, epsilon,
                              std::abs((float)(first_value - second_value)),
                              epsilon * std::abs((float)(first_value + second_value)));
                    }
                }
            } else {
                if (first_value != second_value) {
                    num_errors += 1;
                    if (num_errors < max_num_errors_logged)
                        ERROR("{:s}[{:d}][{:d}] != {:s}[{:d}][{:d}]; values: ({:}, {:})",
                              first_buf->buffer_name, first_buf_id, i, second_buf->buffer_name,
                              second_buf_id, i, format_nice_string(first_value),
                              format_nice_string(second_value));
                }
            }
        }

        if (check_metadata) {

            // Grab 1st metadata
            const std::shared_ptr<const metadataObject> mc1 = first_buf->get_metadata(first_buf_id);
            if (!mc1)
                FATAL_ERROR("Buffer \"{:s}\" frame {:d} does not have metadata",
                            first_buf->buffer_name, first_buf_id);
            assert(mc1);

            // Grab 2nd metadata
            const std::shared_ptr<const metadataObject> mc2 =
                second_buf->get_metadata(second_buf_id);
            if (!mc2)
                FATAL_ERROR("Buffer \"{:s}\" frame {:d} does not have metadata",
                            second_buf->buffer_name, second_buf_id);
            assert(mc2);

            // If both are chord, check them!
            if (metadata_is_chord(mc1) && metadata_is_chord(mc2)) {
                INFO(
                    "Checking that the buffers {:s}[{:d}] and {:s}[{:d}] have matching metadata...",
                    first_buf->buffer_name, first_buf_id, second_buf->buffer_name, second_buf_id);

                const std::shared_ptr<const chordMetadata> meta1 = get_chord_metadata(mc1);
                const std::shared_ptr<const chordMetadata> meta2 = get_chord_metadata(mc2);

                // Count errors in the metadata
                int num_meta_errors =
                    check_chord_metadata(meta1, meta2, first_buf_id, second_buf_id);
                num_errors += num_meta_errors;

                if (num_meta_errors > 0) {
                    INFO("The buffers {:s}[{:d}] and {:s}[{:d}] contained {:d} different metadata "
                         "values.",
                         first_buf->buffer_name, first_buf_id, second_buf->buffer_name,
                         second_buf_id, num_meta_errors);
                } else {
                    INFO("The buffers {:s}[{:d}] and {:s}[{:d}] contained matching metadata.",
                         first_buf->buffer_name, first_buf_id, second_buf->buffer_name,
                         second_buf_id);
                }
            } else if (metadata_is_N2(mc1) && metadata_is_N2(mc2)) {
                INFO("Checking that the buffers {:s}[{:d}] and {:s}[{:d}] have matching N2 "
                     "metadata...",
                     first_buf->buffer_name, first_buf_id, second_buf->buffer_name, second_buf_id);

                const std::shared_ptr<const N2Metadata> meta1 = get_N2_metadata(mc1);
                const std::shared_ptr<const N2Metadata> meta2 = get_N2_metadata(mc2);

                int num_meta_errors = check_N2_metadata(meta1, meta2, first_buf_id, second_buf_id);
                num_errors += num_meta_errors;

                if (num_meta_errors > 0) {
                    INFO("The buffers {:s}[{:d}] and {:s}[{:d}] contained {:d} different N2 "
                         "metadata values.",
                         first_buf->buffer_name, first_buf_id, second_buf->buffer_name,
                         second_buf_id, num_meta_errors);
                } else {
                    INFO("The buffers {:s}[{:d}] and {:s}[{:d}] contained matching N2 metadata.",
                         first_buf->buffer_name, first_buf_id, second_buf->buffer_name,
                         second_buf_id);
                }
            } else {
                FATAL_ERROR("metadata types should both be chord or both be N2!");
            }
        }

        if (num_errors == 0) {
            INFO("The buffers {:s}[{:d}] and {:s}[{:d}] contained values that were equal.",
                 first_buf->buffer_name, first_buf_id, second_buf->buffer_name, second_buf_id);
            if (use_almost_equal) {
                INFO("Compared {:d} elements.  Average absolute difference: {:g}.  Average "
                     "relative difference: {:g}",
                     num_elements, abs_diff / num_elements, rel_diff / num_elements);
                INFO("Nonzero elements: {:d}.  Average absolute difference: {:g}.  Average "
                     "relative difference: {:g}",
                     num_nonzero, abs_diff / std::max((uint32_t)1, num_nonzero),
                     rel_diff / std::max((uint32_t)1, num_nonzero));
            }
        } else {
            INFO("The buffers {:s}[{:d}] and {:s}[{:d}] contained values that were NOT equal!",
                 first_buf->buffer_name, first_buf_id, second_buf->buffer_name, second_buf_id);
            INFO("Test failed, exiting.");
            TEST_FAILED();
        }

        first_buf->mark_frame_empty(unique_name, first_buf_id);
        second_buf->mark_frame_empty(unique_name, second_buf_id);

        first_buf_id = (first_buf_id + 1) % first_buf->num_frames;
        second_buf_id = (second_buf_id + 1) % second_buf->num_frames;
        frames++;


        if (num_frames_to_test == frames) {
            if (num_errors == 0) {
                if (trigger_exit_on_pass) {
                    INFO("Test passed, exiting.");
                    TEST_PASSED();
                } else {
                    INFO("Test passed.");
                }
            }
        } // frames
        
        if (num_errors > 0) {
            break;
        }

    }
}

template<typename A_Type>
int testDataCheck<A_Type>::check_chord_metadata(const std::shared_ptr<const chordMetadata> meta1,
                                                const std::shared_ptr<const chordMetadata> meta2,
                                                int first_buf_id, int second_buf_id) {
    int num_errors = 0;

    // Number of frequencies from metadata (if available)
    int num_freq = meta1->has_nfreq() ? meta1->get_nfreq() : 0;

    /*
     * The relevant metadata fields are a bit of a moving target at the moment (251022),
     * so some of these checks are being commented out.  This function should be updated
     * when the chordMetadata is more stable to cover exactly all the fields.
     */

    // int64_t fpga_seq_num;
    CHECK_META_SCALAR_INT(fpga_seq_num, meta1, meta2, num_errors, first_buf->buffer_name,
                          first_buf_id, second_buf->buffer_name, second_buf_id);

    // int time_downsampling_fpga;
    CHECK_META_SCALAR_INT(time_downsampling_fpga, meta1, meta2, num_errors, first_buf->buffer_name,
                          first_buf_id, second_buf->buffer_name, second_buf_id);

    // TODO: struct timeval first_packet_recv_time;
    // TODO: struct timespec gps_time;

    // uint16_t stream_ID;
    /*
    CHECK_META_SCALAR_STREAM_T(stream_id, meta1, meta2, num_errors,
                               first_buf->buffer_name, first_buf_id,
                               second_buf->buffer_name, second_buf_id);
    */

    // int frame_counter;
    /*
    CHECK_META_SCALAR_INT(frame_counter, meta1, meta2, num_errors, first_buf->buffer_name,
                          first_buf_id, second_buf->buffer_name, second_buf_id);
    */
    // char name[CHORD_META_MAX_DIMNAME]; // "E", "J", "I", etc
    CHECK_META_SCALAR_STR(name, meta1, meta2, num_errors, first_buf->buffer_name, first_buf_id,
                          second_buf->buffer_name, second_buf_id);
    // kotekan::DataType type;

    // int lost_timesamples;
    /*
    CHECK_META_SCALAR_INT(lost_timesamples,
                          meta1, meta2, num_errors,
                          first_buf->buffer_name, first_buf_id,
                          second_buf->buffer_name, second_buf_id);
    */
    // int rfi_flagged_samples;
    /*
    CHECK_META_SCALAR_INT(rfi_flagged_samples,
                          meta1, meta2, num_errors,
                          first_buf->buffer_name, first_buf_id,
                          second_buf->buffer_name, second_buf_id);
    */
    // int dims;
    CHECK_META_SCALAR_INT_DIRECT(dims, meta1, meta2, num_errors, first_buf->buffer_name,
                                 first_buf_id, second_buf->buffer_name, second_buf_id);

    int num_dim = meta1->dims;

    // int dim[CHORD_META_MAX_DIM];
    CHECK_META_ARR1_INT_DIRECT(dim, num_dim, meta1, meta2, num_errors, first_buf->buffer_name,
                               first_buf_id, second_buf->buffer_name, second_buf_id);

    // char dim_name[CHORD_META_MAX_DIM][CHORD_META_MAX_DIMNAME]; // "F", "T", "D", etc
    CHECK_META_ARR1_CSTR_DIRECT(dim_name, num_dim, CHORD_META_MAX_DIMNAME, meta1, meta2, num_errors,
                                first_buf->buffer_name, first_buf_id, second_buf->buffer_name,
                                second_buf_id);

    // int64_t stride[CHORD_META_MAX_DIM];
    CHECK_META_ARR1_INT_DIRECT(stride, num_dim, meta1, meta2, num_errors, first_buf->buffer_name,
                               first_buf_id, second_buf->buffer_name, second_buf_id);

    // int64_t offset;
    CHECK_META_SCALAR_INT_DIRECT(offset, meta1, meta2, num_errors, first_buf->buffer_name,
                                 first_buf_id, second_buf->buffer_name, second_buf_id);


    // int nfreq;
    CHECK_META_SCALAR_INT(nfreq, meta1, meta2, num_errors, first_buf->buffer_name, first_buf_id,
                          second_buf->buffer_name, second_buf_id);

    // int coarse_freq[CHORD_META_MAX_FREQ];
    CHECK_META_ARR1_INT(coarse_freq, num_freq, meta1, meta2, num_errors, first_buf->buffer_name,
                        first_buf_id, second_buf->buffer_name, second_buf_id);

    // int freq_upchan_factor[CHORD_META_MAX_FREQ];
    CHECK_META_ARR1_INT(freq_upchan_factor, num_freq, meta1, meta2, num_errors,
                        first_buf->buffer_name, first_buf_id, second_buf->buffer_name,
                        second_buf_id);

    // int freq_upchan_index[CHORD_META_MAX_FREQ];
    CHECK_META_ARR1_INT(freq_upchan_index, num_freq, meta1, meta2, num_errors,
                        first_buf->buffer_name, first_buf_id, second_buf->buffer_name,
                        second_buf_id);

    // int ndishes;                                  // number of dishes
    CHECK_META_SCALAR_INT_DIRECT(ndishes, meta1, meta2, num_errors, first_buf->buffer_name,
                                 first_buf_id, second_buf->buffer_name, second_buf_id);

    // int n_dish_locations_ew, n_dish_locations_ns; // number of possible dish locations
    CHECK_META_SCALAR_INT_DIRECT(n_dish_locations_ew, meta1, meta2, num_errors,
                                 first_buf->buffer_name, first_buf_id, second_buf->buffer_name,
                                 second_buf_id);
    CHECK_META_SCALAR_INT_DIRECT(n_dish_locations_ns, meta1, meta2, num_errors,
                                 first_buf->buffer_name, first_buf_id, second_buf->buffer_name,
                                 second_buf_id);

    // TODO: int* dish_index; // [non-owning pointer] dish index for a possible dish location, or -1

    return num_errors;
}

template<typename A_Type>
int testDataCheck<A_Type>::check_N2_metadata(const std::shared_ptr<const N2Metadata> meta1,
                                             const std::shared_ptr<const N2Metadata> meta2,
                                             int first_buf_id, int second_buf_id) {
    int num_errors = 0;

    // Frequency identifiers
    CHECK_META_SCALAR_INT_DIRECT(freq_id, meta1, meta2, num_errors, first_buf->buffer_name,
                                 first_buf_id, second_buf->buffer_name, second_buf_id);
    if (meta1->freq_MHz != meta2->freq_MHz) {
        ERROR("metadata {:s}[{:d}] freq_MHz != {:s}[{:d}] freq_MHz; values: {:g} {:g}",
              first_buf->buffer_name, first_buf_id, second_buf->buffer_name, second_buf_id,
              meta1->freq_MHz, meta2->freq_MHz);
        num_errors++;
    }

    // Timing fields
    if (meta1->fpga_start_tick != meta2->fpga_start_tick) {
        ERROR(
            "metadata {:s}[{:d}] fpga_start_tick != {:s}[{:d}] fpga_start_tick; values: {:d} {:d}",
            first_buf->buffer_name, first_buf_id, second_buf->buffer_name, second_buf_id,
            meta1->fpga_start_tick, meta2->fpga_start_tick);
        num_errors++;
    }
    if (meta1->frame_start_time_ns != meta2->frame_start_time_ns) {
        ERROR("metadata {:s}[{:d}] frame_start_time_ns != {:s}[{:d}] frame_start_time_ns; "
              "values: {:d} {:d}",
              first_buf->buffer_name, first_buf_id, second_buf->buffer_name, second_buf_id,
              meta1->frame_start_time_ns, meta2->frame_start_time_ns);
        num_errors++;
    }
    if (meta1->frame_length_fpga_ticks != meta2->frame_length_fpga_ticks) {
        ERROR("metadata {:s}[{:d}] frame_length_fpga_ticks != {:s}[{:d}] frame_length_fpga_ticks; "
              "values: {:d} {:d}",
              first_buf->buffer_name, first_buf_id, second_buf->buffer_name, second_buf_id,
              meta1->frame_length_fpga_ticks, meta2->frame_length_fpga_ticks);
        num_errors++;
    }
    if (meta1->n_valid_fpga_ticks != meta2->n_valid_fpga_ticks) {
        ERROR("metadata {:s}[{:d}] n_valid_fpga_ticks != {:s}[{:d}] n_valid_fpga_ticks; values: "
              "{:d} {:d}",
              first_buf->buffer_name, first_buf_id, second_buf->buffer_name, second_buf_id,
              meta1->n_valid_fpga_ticks, meta2->n_valid_fpga_ticks);
        num_errors++;
    }
    if (meta1->n_rfi_fpga_ticks != meta2->n_rfi_fpga_ticks) {
        ERROR("metadata {:s}[{:d}] n_rfi_fpga_ticks != {:s}[{:d}] n_rfi_fpga_ticks; values: {:d} "
              "{:d}",
              first_buf->buffer_name, first_buf_id, second_buf->buffer_name, second_buf_id,
              meta1->n_rfi_fpga_ticks, meta2->n_rfi_fpga_ticks);
        num_errors++;
    }

    return num_errors;
}

#endif
