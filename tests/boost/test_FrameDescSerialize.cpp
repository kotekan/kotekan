#define BOOST_TEST_MODULE "test_FrameDescSerialize"

#include "DataType.hpp"    // for DataType, string_to_type
#include "FrameDesc.hpp"   // for FrameDesc
#include "N2FrameDesc.hpp" // for N2FrameDesc
#include "N2Layout.hpp"    // for N2Layout
#include "N2Util.hpp"      // for N2::prod_ctype
#include "NDArray.hpp"     // for GenericNDArray
#include "Symbol.hpp"      // for Symbol
#include "json.hpp"        // for json

#include <boost/test/included/unit_test.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace kotekan;

// Serialize a descriptor to JSON text and read it back, exercising the same
// dump/parse path bufferSend/bufferRecv use over the wire.
static std::shared_ptr<const FrameDesc> round_trip(const FrameDesc& desc) {
    const std::string bytes = desc.to_json().dump();
    return FrameDesc::from_json(nlohmann::json::parse(bytes));
}

BOOST_AUTO_TEST_CASE(ndarray_round_trip_with_labels) {
    // Use a non-unit dimscaling so the round trip exercises scaling transmission.
    auto desc = GenericNDArray::describe(string_to_type("float32"), Symbol("voltage"), {4, 8, 2},
                                         {Symbol("F"), Symbol("T"), Symbol("P")}, {1, 2, 1});
    auto back = round_trip(*desc);
    BOOST_REQUIRE(back);
    BOOST_CHECK(*back == *desc);
}

BOOST_AUTO_TEST_CASE(ndarray_round_trip_packed_type_no_labels) {
    // packed type, labels left unset (empty symbols encode as JSON null)
    auto desc =
        GenericNDArray::describe(string_to_type("int4x2"), Symbol(""), {2048}, {Symbol("")}, {1});
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

BOOST_AUTO_TEST_CASE(n2_round_trip_general_subset) {
    // GeneralSubset carries an explicit product_list over the wire.
    std::vector<N2::prod_ctype> products = {{0, 1}, {0, 2}, {3, 5}};
    N2FrameDesc desc(8, 2, static_cast<uint32_t>(products.size()), N2Layout::GeneralSubset,
                     products);
    auto back = round_trip(desc);
    BOOST_REQUIRE(back);
    BOOST_CHECK(*back == desc);
}

BOOST_AUTO_TEST_CASE(from_json_rejects_missing_type) {
    const nlohmann::json j = {{"value_type", "float32"}};
    BOOST_CHECK_THROW(FrameDesc::from_json(j), std::exception);
}

BOOST_AUTO_TEST_CASE(from_json_rejects_unknown_type) {
    const nlohmann::json j = {{"frame_desc_type", "not_a_descriptor"}};
    BOOST_CHECK_THROW(FrameDesc::from_json(j), std::exception);
}

BOOST_AUTO_TEST_CASE(from_json_rejects_bad_value_type) {
    nlohmann::json j;
    j["frame_desc_type"] = "ndarray";
    j["value_type"] = "not_a_real_type";
    j["extents"] = {4};
    j["quantity_name"] = nullptr;
    j["dimnames"] = {nullptr};
    j["dimscalings"] = {1};
    BOOST_CHECK_THROW(FrameDesc::from_json(j), std::exception);
}

BOOST_AUTO_TEST_CASE(from_json_rejects_duplicate_dimnames) {
    nlohmann::json j;
    j["frame_desc_type"] = "ndarray";
    j["value_type"] = "float32";
    j["extents"] = {4, 8};
    j["quantity_name"] = nullptr;
    j["dimnames"] = {"F", "F"}; // duplicate
    j["dimscalings"] = {1, 1};
    BOOST_CHECK_THROW(FrameDesc::from_json(j), std::exception);
}

BOOST_AUTO_TEST_CASE(from_json_rejects_inconsistent_n2_num_products) {
    // FullUpperTri with num_elements=4 implies 10 products; a num_products that
    // disagrees must be rejected rather than stored verbatim into the descriptor.
    nlohmann::json j;
    j["frame_desc_type"] = "N2";
    j["num_elements"] = 4;
    j["num_ev"] = 0;
    j["num_products"] = 99; // wrong; should be 10
    j["n2_layout"] = "FullUpperTri";
    BOOST_CHECK_THROW(FrameDesc::from_json(j), std::exception);
}
