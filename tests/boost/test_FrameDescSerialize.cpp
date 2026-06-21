#define BOOST_TEST_MODULE "test_FrameDescSerialize"

#include "DataType.hpp"    // for DataType, string_to_type
#include "FrameDesc.hpp"   // for FrameDesc, wire::put_*
#include "N2FrameDesc.hpp" // for N2FrameDesc
#include "N2Layout.hpp"    // for N2Layout
#include "NDArray.hpp"     // for GenericNDArray
#include "Symbol.hpp"      // for Symbol

#include <boost/test/included/unit_test.hpp>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace kotekan;

// Serialize a descriptor and read it back.
static std::shared_ptr<const FrameDesc> round_trip(const FrameDesc& desc) {
    std::vector<char> bytes(desc.serialized_size());
    desc.serialize(bytes.data());
    return FrameDesc::deserialize(bytes.data(), bytes.size());
}

BOOST_AUTO_TEST_CASE(ndarray_round_trip_with_labels) {
    auto desc = GenericNDArray::describe(string_to_type("float32"), Symbol("voltage"), {4, 8, 2},
                                         {Symbol("F"), Symbol("T"), Symbol("P")});
    auto back = round_trip(*desc);
    BOOST_REQUIRE(back);
    BOOST_CHECK(*back == *desc);
}

BOOST_AUTO_TEST_CASE(ndarray_round_trip_packed_type_no_labels) {
    // packed type, labels left unset (empty symbols)
    auto desc = GenericNDArray::describe(string_to_type("int4x2"), Symbol(""), {2048}, {Symbol("")});
    auto back = round_trip(*desc);
    BOOST_REQUIRE(back);
    BOOST_CHECK(*back == *desc);
}

BOOST_AUTO_TEST_CASE(n2_round_trip_full_upper_tri) {
    N2FrameDesc desc(16, 4, N2FrameDesc::get_num_prod(16, N2Layout::FullUpperTri),
                     N2Layout::FullUpperTri);
    auto back = round_trip(desc);
    BOOST_REQUIRE(back);
    BOOST_CHECK(*back == desc);
}

BOOST_AUTO_TEST_CASE(deserialize_rejects_too_small_for_tag) {
    char bytes[2] = {0, 0};
    BOOST_CHECK_THROW(FrameDesc::deserialize(bytes, sizeof(bytes)), std::exception);
}

BOOST_AUTO_TEST_CASE(deserialize_rejects_unknown_tag) {
    char bytes[8] = {0};
    const uint32_t bad_tag = 4242;
    std::memcpy(bytes, &bad_tag, sizeof(bad_tag));
    BOOST_CHECK_THROW(FrameDesc::deserialize(bytes, sizeof(bytes)), std::exception);
}

BOOST_AUTO_TEST_CASE(deserialize_rejects_truncated_payload) {
    auto desc = GenericNDArray::describe(string_to_type("float32"), Symbol("v"), {4}, {Symbol("F")});
    std::vector<char> bytes(desc->serialized_size());
    desc->serialize(bytes.data());
    BOOST_CHECK_THROW(FrameDesc::deserialize(bytes.data(), bytes.size() - 3), std::exception);
}

BOOST_AUTO_TEST_CASE(deserialize_rejects_bad_value_type) {
    // Hand-build an ndarray payload carrying an invalid value_type string.
    std::vector<char> bytes(64);
    char* p = bytes.data();
    const uint32_t tag = static_cast<uint32_t>(FrameDesc::WireType::generic_ndarray);
    std::memcpy(p, &tag, sizeof(tag));
    p += sizeof(tag);
    p = wire::put_str(p, "not_a_real_type");
    p = wire::put_u32(p, 1);  // rank
    p = wire::put_i64(p, 4);  // extent
    p = wire::put_str(p, ""); // quantity_name
    p = wire::put_u32(p, 1);  // dimnames count
    p = wire::put_str(p, ""); // dimname
    BOOST_CHECK_THROW(FrameDesc::deserialize(bytes.data(), p - bytes.data()), std::exception);
}
