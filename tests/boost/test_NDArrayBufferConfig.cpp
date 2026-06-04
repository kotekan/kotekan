#define BOOST_TEST_MODULE "test_NDArrayBufferConfig"

#include <boost/test/included/unit_test.hpp>
#include <cstddef>   // for ptrdiff_t, size_t
#include <map>       // for map
#include <memory>    // for shared_ptr
#include <stdexcept> // for runtime_error
#include <string>    // for string
#include <vector>    // for vector

// the code to test:
#include "Config.hpp"        // for Config
#include "NDArray.hpp"       // for GenericNDArray
#include "Symbol.hpp"        // for Symbol
#include "buffer.hpp"        // for Buffer, is_frame_buffer
#include "bufferFactory.hpp" // for bufferFactory
#include "metadata.hpp"      // for metadataPool

#include "json.hpp" // for json

using kotekan::Config;
using kotekan::GenericNDArray;
using kotekan::Symbol;

using json = nlohmann::json;

// Config::eval evaluates numbers and expression strings in a config scope.
BOOST_AUTO_TEST_CASE(config_eval) {
    json json_config = {{"num_local_freq", 16}, {"samples_per_data_set", 1024}};
    Config config;
    config.update_config(json_config);

    BOOST_CHECK_EQUAL(config.eval<std::ptrdiff_t>("/", json(42)), 42);
    BOOST_CHECK_EQUAL(config.eval<std::ptrdiff_t>("/", json("num_local_freq")), 16);
    BOOST_CHECK_EQUAL(config.eval<std::ptrdiff_t>("/", json("num_local_freq * 2")), 32);
    BOOST_CHECK_EQUAL(
        config.eval<std::ptrdiff_t>("/", json("samples_per_data_set / num_local_freq + 1")), 65);

    // Unknown variables, malformed expressions, and non-evaluable json fail
    BOOST_CHECK_THROW(config.eval<std::ptrdiff_t>("/", json("not_defined + 1")),
                      std::runtime_error);
    BOOST_CHECK_THROW(config.eval<std::ptrdiff_t>("/", json("5 +")), std::runtime_error);
    BOOST_CHECK_THROW(config.eval<std::ptrdiff_t>("/", json::object()), std::runtime_error);
}

// GenericNDArray::from_config builds a frame descriptor from a config block,
// with extent entries evaluated as (scoped) arithmetic expressions.
BOOST_AUTO_TEST_CASE(from_config) {
    json json_config = {{"num_local_freq", 16},
                        {"samples_per_data_set", 1024},
                        {"my_buf",
                         {{"value_type", "uint8"},
                          {"quantity_name", "voltage"},
                          {"extents", {"num_local_freq", "samples_per_data_set / 2", 4}},
                          {"dimnames", {"F", "T", "D"}}}}};
    Config config;
    config.update_config(json_config);

    auto desc = GenericNDArray::from_config(config, "/my_buf");
    BOOST_REQUIRE(desc);
    BOOST_CHECK(desc->get_value_datatype() == kotekan::uint8);
    BOOST_CHECK(desc->get_quantity_name() == Symbol("voltage"));
    BOOST_CHECK_EQUAL(desc->get_rank(), 3);
    BOOST_CHECK_EQUAL(desc->get_extent(0), 16);
    BOOST_CHECK_EQUAL(desc->get_extent(1), 512);
    BOOST_CHECK_EQUAL(desc->get_extent(2), 4);
    BOOST_CHECK(desc->get_dimname(0) == Symbol("F"));
    BOOST_CHECK(desc->get_dimname(1) == Symbol("T"));
    BOOST_CHECK(desc->get_dimname(2) == Symbol("D"));
    BOOST_CHECK_EQUAL(desc->get_byte_size(), 16 * 512 * 4);
}

// Build a Config holding a single buffer block at /my_buf.
static Config make_config(json block) {
    json json_config = {{"my_buf", std::move(block)}};
    Config config;
    config.update_config(json_config);
    return config;
}

// Invalid config blocks fail loudly at construction.
BOOST_AUTO_TEST_CASE(from_config_errors) {
    // unknown value_type
    {
        Config config = make_config({{"value_type", "complex128"},
                                     {"quantity_name", "q"},
                                     {"extents", {4}},
                                     {"dimnames", {"X"}}});
        BOOST_CHECK_THROW(GenericNDArray::from_config(config, "/my_buf"), std::runtime_error);
    }
    // missing key
    {
        Config config =
            make_config({{"value_type", "float32"}, {"extents", {4}}, {"dimnames", {"X"}}});
        BOOST_CHECK_THROW(GenericNDArray::from_config(config, "/my_buf"), std::runtime_error);
    }
    // extents / dimnames size mismatch
    {
        Config config = make_config({{"value_type", "float32"},
                                     {"quantity_name", "q"},
                                     {"extents", {4, 2}},
                                     {"dimnames", {"X"}}});
        BOOST_CHECK_THROW(GenericNDArray::from_config(config, "/my_buf"), std::runtime_error);
    }
    // non-positive extent
    {
        Config config = make_config({{"value_type", "float32"},
                                     {"quantity_name", "q"},
                                     {"extents", {4, 0}},
                                     {"dimnames", {"X", "Y"}}});
        BOOST_CHECK_THROW(GenericNDArray::from_config(config, "/my_buf"), std::runtime_error);
    }
    // empty extents (rank 0)
    {
        Config config = make_config({{"value_type", "float32"},
                                     {"quantity_name", "q"},
                                     {"extents", json::array()},
                                     {"dimnames", json::array()}});
        BOOST_CHECK_THROW(GenericNDArray::from_config(config, "/my_buf"), std::runtime_error);
    }
    // duplicate dimnames
    {
        Config config = make_config({{"value_type", "float32"},
                                     {"quantity_name", "q"},
                                     {"extents", {4, 2}},
                                     {"dimnames", {"X", "X"}}});
        BOOST_CHECK_THROW(GenericNDArray::from_config(config, "/my_buf"), std::runtime_error);
    }
}

// The bufferFactory creates `ndarray` buffers with the frame descriptor
// attached at startup and frame_size derived from it -- and stage-style
// allocate_ndarray_frame_desc calls then act as pure validators.
BOOST_AUTO_TEST_CASE(buffer_factory_ndarray) {
    json json_config = {{"log_level", "off"},
                        {"num_local_freq", 16},
                        {"gain_buffer",
                         {{"kotekan_buffer", "ndarray"},
                          {"value_type", "float32"},
                          {"quantity_name", "gains"},
                          {"extents", {"num_local_freq", 4}},
                          {"dimnames", {"F", "G"}},
                          {"num_frames", 2},
                          {"metadata_pool", "none"}}}};
    Config config;
    config.update_config(json_config);

    std::map<std::string, std::shared_ptr<metadataPool>> pools;
    kotekan::bufferFactory factory(config, pools);
    auto buffers = factory.build_buffers();
    BOOST_REQUIRE_EQUAL(buffers.size(), 1);

    auto* buf = dynamic_cast<Buffer*>(buffers.at("gain_buffer"));
    BOOST_REQUIRE(buf);
    BOOST_CHECK(is_frame_buffer(buf));
    BOOST_CHECK_EQUAL(buf->frame_size, 16 * 4 * sizeof(float));

    // The descriptor is present before any stage would run
    auto desc = buf->get_ndarray_frame_desc();
    BOOST_REQUIRE(desc);
    BOOST_CHECK(desc->get_value_datatype() == kotekan::float32);
    BOOST_CHECK(desc->get_quantity_name() == Symbol("gains"));
    BOOST_CHECK_EQUAL(desc->get_rank(), 2);
    BOOST_CHECK_EQUAL(desc->get_extent(0), 16);
    BOOST_CHECK_EQUAL(desc->get_extent(1), 4);

    // A stage-style declare call now validates against the factory-set
    // descriptor instead of allocating (the migration "auto-degrade"
    // property); a matching call passes through silently.
    buf->allocate_ndarray_frame_desc<float, 2>(Symbol("gains"), {16, 4},
                                               {Symbol("F"), Symbol("G")});
    BOOST_CHECK(buf->get_ndarray_frame_desc() == desc);

    // ... and a MISMATCHING declare call is fatal. (Under the boost test
    // build, FATAL_ERROR throws std::runtime_error via KTK_BOOST_ERR before
    // raising SIGTERM, so the fatal paths are testable with CHECK_THROW.)
    // Same byte size, swapped extents/dimnames -- reaches the field checks:
    BOOST_CHECK_THROW((buf->allocate_ndarray_frame_desc<float, 2>(Symbol("gains"), {4, 16},
                                                                  {Symbol("G"), Symbol("F")})),
                      std::runtime_error);
    // Same byte size and shape, wrong value type:
    BOOST_CHECK_THROW((buf->allocate_ndarray_frame_desc<int32_t, 2>(Symbol("gains"), {16, 4},
                                                                    {Symbol("F"), Symbol("G")})),
                      std::runtime_error);
    // Same byte size and shape, wrong quantity name:
    BOOST_CHECK_THROW((buf->allocate_ndarray_frame_desc<float, 2>(Symbol("not_gains"), {16, 4},
                                                                  {Symbol("F"), Symbol("G")})),
                      std::runtime_error);

    // set_frame_desc with a mismatching descriptor of equal byte size is
    // fatal (previously only a soft ERROR):
    BOOST_CHECK_THROW(
        buf->set_frame_desc(GenericNDArray::create(kotekan::float32, Symbol("gains"), {4, 16},
                                                   {Symbol("G"), Symbol("F")}, nullptr)),
        std::runtime_error);
    // ... as is one with the wrong byte size:
    BOOST_CHECK_THROW(
        buf->set_frame_desc(GenericNDArray::create(kotekan::float32, Symbol("gains"), {16, 8},
                                                   {Symbol("F"), Symbol("G")}, nullptr)),
        std::runtime_error);
    // The recorded descriptor is unchanged by any of the rejected calls
    BOOST_CHECK(buf->get_ndarray_frame_desc() == desc);

    // Setting an equal descriptor again is accepted
    buf->set_frame_desc(GenericNDArray::create(kotekan::float32, Symbol("gains"), {16, 4},
                                               {Symbol("F"), Symbol("G")}, nullptr));
    BOOST_CHECK(buf->get_ndarray_frame_desc() == desc);

    for (auto& [name, b] : buffers)
        delete b;
}
