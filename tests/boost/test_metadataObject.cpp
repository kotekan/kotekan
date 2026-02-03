#define BOOST_TEST_MODULE "test_metadataObject"

#include <boost/test/included/unit_test.hpp>
#include <chordMetadata.hpp>
#include <iostream>

using namespace kotekan;

BOOST_AUTO_TEST_CASE(test_chordMetadata) {
    std::cout << "Testing chordMetadata class...\n";
    chordMetadata meta0;

    // populate with some values, carefully picked to be random by me
    meta0.set_name("E");
    meta0.type = kotekan::DataType::uint8;

    meta0.dims = 3;
    meta0.set_array_dimension(0, 11229, "D");
    meta0.set_array_dimension(1, 23039, "P");
    meta0.set_array_dimension(2, 18137, "Time");
    meta0.set_strides_simple();
    meta0.offset = 24987; // mostly 0 in code, but still

    // TODO: dish_locations etc

    // TODO: no set_beam_coord yet

    meta0.set_fpga_seq_num(17133);
    meta0.set_frame_counter(19207);

    std::vector<int> coarse_freq = {
        3354,  5121,  1463,  4418,  29134, 28593, 28032, 3725,  508,   31989, 4935,
        22058, 18266, 22437, 15678, 11124, 25677, 19997, 30880, 28490, 3592,  330,
    };
    meta0.set_coarse_freq(coarse_freq);

    meta0.set_rfi_num_bad_inputs(17029);
    meta0.set_rfi_flagged_samples(10090);
    meta0.set_lost_timesamples(18427);

    std::vector<int> freq_upchan_factor = {
        44, 58, 17, 27,  834, 74, 647, 195, 491, 772, 55,
        30, 55, 86, 867, 997, 89, 733, 493, 913, 602, 23,
    };
    meta0.set_freq_upchan_factor(freq_upchan_factor);

    std::vector<int> freq_upchan_index = {
        43, 57, 16, 26,  833, 73, 646, 194, 490, 771, 54,
        29, 54, 85, 866, 996, 88, 732, 492, 912, 601, 22,
    };
    meta0.set_freq_upchan_index(freq_upchan_index);

    meta0.set_time_downsampling_fpga(47);

    timeval tv = {17863, 341};
    meta0.set_first_packet_recv_time(tv);

    stream_t stream_id = {27009};
    meta0.set_stream_id(stream_id);

    dset_id_t dataset_id = hash("Mary had a little lamb");
    meta0.set_dataset_id(dataset_id);

    chordMetadata::beamCoord beam_coord = {.right_ascension =
                                               {
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
                                               },
                                           .declination =
                                               {
                                                   256.52,
                                                   264.16,
                                                   49.39,
                                                   227.15,
                                                   9.72,
                                                   21.46,
                                                   21.73,
                                                   51.71,
                                                   326.49,
                                                   130.62,
                                               },
                                           .scaling = {
                                               3,
                                               2,
                                               2,
                                               2,
                                               1,
                                               2,
                                               2,
                                               3,
                                               1,
                                               3,
                                           }};
    meta0.set_beam_coord(beam_coord);

    // now test the different serializers

    {
        std::cout << "To / from json...\n";

        // serialize to json
        nlohmann::json j = meta0;

        // restore
        chordMetadata meta1;
        from_json(j, meta1);

        // compare
        BOOST_CHECK(meta0 == meta1);

        std::cout << "Success.\n";
    }

    {
        std::cout << "To / from bytes...\n";

        // serialize to bytes
        std::vector<char> bytes(meta0.get_serialized_size());
        size_t written = meta0.serialize(bytes.data());
        BOOST_TEST(written == bytes.size());

        // restore
        chordMetadata meta2;
        size_t read = meta2.set_from_bytes(bytes.data(), bytes.size());
        BOOST_TEST(read == bytes.size());

        // compare
        BOOST_CHECK(meta0 == meta2);

        std::cout << "Success.\n";
    }
}
