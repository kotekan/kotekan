#define BOOST_TEST_MODULE "test_metadataObject"

#include <boost/test/included/unit_test.hpp>
#include <chordMetadata.hpp>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace kotekan;

BOOST_AUTO_TEST_CASE(test_chordMetadata) {
    std::cout << "Testing chordMetadata class...\n";
    chordMetadata meta0;

    // populate with some values, carefully picked to be random by me
    meta0.set_name("E");
    meta0.type = kotekan::DataType::uint8;

    meta0.dims = 3;
    meta0.set_array_dimension(0, 11229, "D", 1);
    meta0.set_array_dimension(1, 23039, "P", 1);
    meta0.set_array_dimension(2, 18137, "Time", 1);
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

// Serializes to bytes and back, and to json and back, and checks that both
// round trips reproduce the original.
static void check_round_trips(chordMetadata& meta) {
    {
        std::vector<char> bytes(meta.get_serialized_size());
        const size_t written = meta.serialize(bytes.data());
        BOOST_TEST(written == bytes.size());

        chordMetadata restored;
        const size_t read = restored.set_from_bytes(bytes.data(), bytes.size());
        BOOST_TEST(read == bytes.size());
        BOOST_CHECK(meta == restored);
    }
    {
        const nlohmann::json j = meta;
        chordMetadata restored;
        from_json(j, restored);
        BOOST_CHECK(meta == restored);
    }
}

BOOST_AUTO_TEST_CASE(test_chordMetadata_unset_per_frequency_arrays) {
    // Serializing without any per-frequency array must work; there used to be an
    // unconditional get_nfreq() which throws when COARSE_FREQ is not set.
    chordMetadata meta;
    meta.set_fpga_seq_num(4711);
    BOOST_CHECK(!meta.has_coarse_freq());

    check_round_trips(meta);

    std::vector<char> bytes(meta.get_serialized_size());
    meta.serialize(bytes.data());
    chordMetadata restored;
    restored.set_from_bytes(bytes.data(), bytes.size());
    BOOST_CHECK(!restored.has_coarse_freq());
    BOOST_CHECK(!restored.has_freq_upchan_factor());
    BOOST_CHECK(!restored.has_freq_upchan_index());
}

BOOST_AUTO_TEST_CASE(test_chordMetadata_inconsistent_per_frequency_arrays) {
    // The byte format stores a single length for the per-frequency arrays, so
    // arrays of differing lengths must be rejected rather than silently resized.
    chordMetadata meta;
    meta.set_coarse_freq(std::vector<int>{1, 2, 3});
    meta.set_freq_upchan_factor(std::vector<int>{1, 1});

    std::vector<char> bytes(meta.get_serialized_size());
    BOOST_CHECK_THROW(meta.serialize(bytes.data()), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(test_chordMetadata_stream_ids_and_rfi_excision) {
    // These three entries used to be dropped by both serializers.
    chordMetadata meta;
    meta.set_stream_ids(std::vector<uint32_t>{7, 11, 13});
    meta.set_rfi_frame_excision_enabled(true);
    meta.set_rfi_frame_excision_thresholds(
        std::vector<std::array<float, 2>>{{1.5f, 0.25f}, {2.5f, 0.75f}});

    check_round_trips(meta);

    std::vector<char> bytes(meta.get_serialized_size());
    meta.serialize(bytes.data());
    chordMetadata restored;
    restored.set_from_bytes(bytes.data(), bytes.size());
    BOOST_CHECK(restored.has_stream_ids());
    BOOST_CHECK(restored.get_stream_ids() == std::vector<uint32_t>({7, 11, 13}));
    BOOST_CHECK(restored.has_rfi_frame_excision_enabled());
    BOOST_CHECK(restored.get_rfi_frame_excision_enabled());
    BOOST_TEST(restored.get_rfi_frame_excision_thresholds().size() == size_t(2));
}

BOOST_AUTO_TEST_CASE(test_chordMetadata_names_are_truncated) {
    // Names are truncated to CHORD_META_MAX_DIMNAME characters. A name of exactly
    // that length fills its field, which is then not NUL-terminated, so it must
    // only ever be read back through get_name()/get_dimension_name().
    const std::string long_name(CHORD_META_MAX_DIMNAME + 7, 'x');
    const std::string full_name(CHORD_META_MAX_DIMNAME, 'y');

    chordMetadata meta;
    meta.set_name(long_name);
    BOOST_TEST(meta.get_name() == std::string(CHORD_META_MAX_DIMNAME, 'x'));

    meta.set_name(full_name);
    BOOST_TEST(meta.get_name() == full_name);
    BOOST_TEST(meta.has_name());

    meta.dims = 1;
    meta.set_array_dimension(0, 4, full_name, 1);
    meta.set_strides_simple();
    BOOST_TEST(meta.get_dimension_name(0) == full_name);

    // A name that fills its field must survive both round trips intact
    check_round_trips(meta);

    // A shorter name is NUL-padded
    meta.set_name("E");
    BOOST_TEST(meta.get_name() == "E");
    for (int i = 1; i < CHORD_META_MAX_DIMNAME; ++i)
        BOOST_TEST(meta.name[i] == '\0');

    // set_dimension_name() rejects out-of-range dimensions with FATAL_ERROR, which
    // shuts kotekan down rather than throwing, so it cannot be checked here.
}

BOOST_AUTO_TEST_CASE(test_chordMetadata_per_frequency_arrays_are_never_empty) {
    // The byte format encodes "unset" in the first element of each per-frequency
    // array, which relies on a set array never being empty.
    chordMetadata meta;
    meta.set_coarse_freq(std::vector<int>{0}); // 0 is a valid frequency index
    meta.set_freq_upchan_factor(std::vector<int>{1});
    meta.set_freq_upchan_index(std::vector<int>{0});

    check_round_trips(meta);

    std::vector<char> bytes(meta.get_serialized_size());
    meta.serialize(bytes.data());
    chordMetadata restored;
    restored.set_from_bytes(bytes.data(), bytes.size());
    BOOST_CHECK(restored.get_coarse_freq() == std::vector<int>({0}));
    BOOST_CHECK(restored.get_freq_upchan_factor() == std::vector<int>({1}));
    BOOST_CHECK(restored.get_freq_upchan_index() == std::vector<int>({0}));
}

BOOST_AUTO_TEST_CASE(test_chordMetadata_unset_array_description) {
    // A default-constructed object has dims == -1; to_json used to build vectors
    // from an inverted iterator range.
    chordMetadata meta;
    BOOST_TEST(meta.dims == -1);
    check_round_trips(meta);
}

BOOST_AUTO_TEST_CASE(test_chordMetadata_max_dims) {
    // dims == CHORD_META_MAX_DIM is a legal rank
    chordMetadata meta;
    meta.dims = CHORD_META_MAX_DIM;
    for (int d = 0; d < CHORD_META_MAX_DIM; ++d)
        meta.set_array_dimension(d, d + 2, "d" + std::to_string(d), 1);
    meta.set_strides_simple();
    check_round_trips(meta);
}

BOOST_AUTO_TEST_CASE(test_chordMetadata_atomic_add_lost_timesamples) {
    chordMetadata meta;
    // Adding to a count that was never set cannot be distinguished from adding to
    // zero, so it is an error rather than an implicit zero.
    BOOST_CHECK_THROW(meta.atomic_add_lost_timesamples(3), std::runtime_error);

    meta.set_lost_timesamples(10);
    meta.atomic_add_lost_timesamples(3);
    BOOST_TEST(meta.get_lost_timesamples() == 13);
}

BOOST_AUTO_TEST_CASE(test_chordMetadata_beam_coord_json_pads_unused_beams) {
    // from_json must fill the entries beyond the ones given, so that to_json
    // never reads indeterminate values.
    nlohmann::json j;
    j[jsonMetadata::RIGHT_ASCENSION] = std::vector<float>{1.0f, 2.0f};
    j[jsonMetadata::DECLINATION] = std::vector<float>{3.0f, 4.0f};
    j[jsonMetadata::SCALING] = std::vector<uint32_t>{5, 6};

    const auto coord = j.template get<jsonMetadata::beamCoord>();
    BOOST_TEST(coord.right_ascension[0] == 1.0f);
    BOOST_TEST(coord.scaling[1] == uint32_t(6));
    for (int b = 2; b < MAX_NUM_BEAMS; ++b) {
        BOOST_CHECK(std::isnan(coord.right_ascension[b]));
        BOOST_CHECK(std::isnan(coord.declination[b]));
        BOOST_TEST(coord.scaling[b] == uint32_t(0));
    }
}

BOOST_AUTO_TEST_CASE(test_timeval_json_range_check) {
    timeval tv = {1234, 5678};
    const nlohmann::json j = tv;
    const timeval restored = j.template get<timeval>();
    BOOST_TEST(restored.tv_sec == tv.tv_sec);
    BOOST_TEST(restored.tv_usec == tv.tv_usec);

    nlohmann::json bad;
    bad[jsonMetadata::TV_SEC] = 1;
    bad[jsonMetadata::TV_USEC] = 1000000; // not a valid microsecond count
    BOOST_CHECK_THROW(bad.template get<timeval>(), std::out_of_range);
}
