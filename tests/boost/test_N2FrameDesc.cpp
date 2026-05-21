#define BOOST_TEST_MODULE "test_N2FrameDesc"

#include "N2FrameDesc.hpp"
#include "N2Layout.hpp"
#include "N2Util.hpp"

#include <boost/test/included/unit_test.hpp>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

using namespace kotekan;

BOOST_AUTO_TEST_CASE(test_layout_requires_product_list) {
    std::cout << "Testing layout_requires_product_list()...\n";

    // Layouts that do NOT require product list
    BOOST_CHECK_EQUAL(N2FrameDesc::layout_requires_product_list(N2Layout::FullUpperTri), false);
    BOOST_CHECK_EQUAL(N2FrameDesc::layout_requires_product_list(N2Layout::Autocorrelations), false);

    // Layouts that DO require product list
    BOOST_CHECK_EQUAL(N2FrameDesc::layout_requires_product_list(N2Layout::InputANDMasked), true);
    BOOST_CHECK_EQUAL(N2FrameDesc::layout_requires_product_list(N2Layout::InputORMasked), true);
    BOOST_CHECK_EQUAL(N2FrameDesc::layout_requires_product_list(N2Layout::GeneralSubset), true);
    BOOST_CHECK_EQUAL(N2FrameDesc::layout_requires_product_list(N2Layout::RedundantBaselineAvg),
                      true);

    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_note_additional_required_config_param) {
    std::cout << "Testing note_additional_required_config_param()...\n";

    using std::string;

    // Computed layouts - no config param needed
    BOOST_CHECK_EQUAL(
        string(N2FrameDesc::note_additional_required_config_param(N2Layout::FullUpperTri)), "none");
    BOOST_CHECK_EQUAL(
        string(N2FrameDesc::note_additional_required_config_param(N2Layout::Autocorrelations)),
        "none");

    // Input list layouts
    BOOST_CHECK_EQUAL(
        string(N2FrameDesc::note_additional_required_config_param(N2Layout::InputORMasked)),
        "input_list");
    BOOST_CHECK_EQUAL(
        string(N2FrameDesc::note_additional_required_config_param(N2Layout::InputANDMasked)),
        "input_list");

    // Explicit product list layouts
    BOOST_CHECK_EQUAL(
        string(N2FrameDesc::note_additional_required_config_param(N2Layout::GeneralSubset)),
        "product_list");
    BOOST_CHECK_EQUAL(
        string(N2FrameDesc::note_additional_required_config_param(N2Layout::RedundantBaselineAvg)),
        "product_list");

    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_get_num_prod_fulluppertri) {
    std::cout << "Testing get_num_prod() for FullUpperTri...\n";

    // n elements -> n*(n+1)/2 products
    BOOST_CHECK_EQUAL(N2FrameDesc::get_num_prod(4, N2Layout::FullUpperTri), 10);
    BOOST_CHECK_EQUAL(N2FrameDesc::get_num_prod(8, N2Layout::FullUpperTri), 36);
    BOOST_CHECK_EQUAL(N2FrameDesc::get_num_prod(16, N2Layout::FullUpperTri), 136);
    BOOST_CHECK_EQUAL(N2FrameDesc::get_num_prod(64, N2Layout::FullUpperTri), 2080);

    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_get_num_prod_autocorrelations) {
    std::cout << "Testing get_num_prod() for Autocorrelations...\n";

    // n elements -> n products (diagonal only)
    BOOST_CHECK_EQUAL(N2FrameDesc::get_num_prod(4, N2Layout::Autocorrelations), 4);
    BOOST_CHECK_EQUAL(N2FrameDesc::get_num_prod(8, N2Layout::Autocorrelations), 8);
    BOOST_CHECK_EQUAL(N2FrameDesc::get_num_prod(16, N2Layout::Autocorrelations), 16);
    BOOST_CHECK_EQUAL(N2FrameDesc::get_num_prod(64, N2Layout::Autocorrelations), 64);

    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_get_num_prod_throws_for_subset_layouts) {
    std::cout << "Testing get_num_prod() throws for subset layouts without product list...\n";

    // Layouts requiring product list should throw when no product list provided
    BOOST_CHECK_THROW(N2FrameDesc::get_num_prod(8, N2Layout::GeneralSubset), std::runtime_error);
    BOOST_CHECK_THROW(N2FrameDesc::get_num_prod(8, N2Layout::InputANDMasked), std::runtime_error);
    BOOST_CHECK_THROW(N2FrameDesc::get_num_prod(8, N2Layout::InputORMasked), std::runtime_error);
    BOOST_CHECK_THROW(N2FrameDesc::get_num_prod(8, N2Layout::RedundantBaselineAvg),
                      std::runtime_error);

    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_get_num_prod_with_product_list) {
    std::cout << "Testing get_num_prod() with explicit product list...\n";

    // Create a product list
    std::vector<N2::prod_ctype> product_list = {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {1, 2}};

    // With product list, get_num_prod should return the list size for any layout
    BOOST_CHECK_EQUAL(N2FrameDesc::get_num_prod(8, N2Layout::GeneralSubset, product_list), 5);
    BOOST_CHECK_EQUAL(N2FrameDesc::get_num_prod(8, N2Layout::InputANDMasked, product_list), 5);
    BOOST_CHECK_EQUAL(N2FrameDesc::get_num_prod(8, N2Layout::InputORMasked, product_list), 5);

    // Even for layouts that don't require product lists, providing one overrides the computed value
    BOOST_CHECK_EQUAL(N2FrameDesc::get_num_prod(8, N2Layout::FullUpperTri, product_list), 5);
    BOOST_CHECK_EQUAL(N2FrameDesc::get_num_prod(8, N2Layout::Autocorrelations, product_list), 5);

    // Empty product list should fall back to computed values for supported layouts
    std::vector<N2::prod_ctype> empty_list;
    BOOST_CHECK_EQUAL(N2FrameDesc::get_num_prod(8, N2Layout::FullUpperTri, empty_list), 36);
    BOOST_CHECK_EQUAL(N2FrameDesc::get_num_prod(8, N2Layout::Autocorrelations, empty_list), 8);

    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_constructor_fulluppertri) {
    std::cout << "Testing N2FrameDesc constructor for FullUpperTri...\n";

    uint32_t num_elements = 8;
    uint32_t num_ev = 2;
    uint32_t num_prod = N2FrameDesc::get_num_prod(num_elements, N2Layout::FullUpperTri);

    N2FrameDesc desc(num_elements, num_ev, num_prod, N2Layout::FullUpperTri);

    BOOST_CHECK_EQUAL(desc.get_num_elements(), num_elements);
    BOOST_CHECK_EQUAL(desc.get_num_ev(), num_ev);
    BOOST_CHECK_EQUAL(desc.get_num_products(), num_prod);
    BOOST_CHECK(desc.get_n2_layout() == N2Layout::FullUpperTri);
    BOOST_CHECK_EQUAL(desc.get_product_list().size(), num_prod); // Product list generated

    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_constructor_autocorrelations) {
    std::cout << "Testing N2FrameDesc constructor for Autocorrelations...\n";

    uint32_t num_elements = 16;
    uint32_t num_ev = 0;
    uint32_t num_prod = N2FrameDesc::get_num_prod(num_elements, N2Layout::Autocorrelations);

    N2FrameDesc desc(num_elements, num_ev, num_prod, N2Layout::Autocorrelations);

    BOOST_CHECK_EQUAL(desc.get_num_elements(), num_elements);
    BOOST_CHECK_EQUAL(desc.get_num_products(), num_prod);
    BOOST_CHECK(desc.get_n2_layout() == N2Layout::Autocorrelations);
    BOOST_CHECK_EQUAL(desc.get_product_list().size(), num_prod); // Product list generated

    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_constructor_general_subset_with_product_list) {
    std::cout << "Testing N2FrameDesc constructor for GeneralSubset with product list...\n";

    uint32_t num_elements = 8;
    uint32_t num_ev = 2;

    // Create explicit product list: some arbitrary products
    std::vector<N2::prod_ctype> product_list = {{0, 0}, {0, 1}, {1, 1}, {2, 3}, {3, 5}, {7, 7}};

    N2FrameDesc desc(num_elements, num_ev, product_list.size(), N2Layout::GeneralSubset,
                     product_list);

    BOOST_CHECK_EQUAL(desc.get_num_elements(), num_elements);
    BOOST_CHECK_EQUAL(desc.get_num_products(), product_list.size());
    BOOST_CHECK(desc.get_n2_layout() == N2Layout::GeneralSubset);
    BOOST_CHECK_EQUAL(desc.get_product_list().size(), product_list.size());

    // Verify product list contents
    const auto& stored_list = desc.get_product_list();
    for (size_t i = 0; i < product_list.size(); ++i) {
        BOOST_CHECK_EQUAL(stored_list[i].input_a, product_list[i].input_a);
        BOOST_CHECK_EQUAL(stored_list[i].input_b, product_list[i].input_b);
    }

    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_constructor_throws_without_product_list) {
    std::cout << "Testing N2FrameDesc constructor throws without required product list...\n";

    // GeneralSubset without product list should throw
    BOOST_CHECK_THROW(N2FrameDesc(8, 2, 5, N2Layout::GeneralSubset), std::runtime_error);

    // InputANDMasked without product list should throw
    BOOST_CHECK_THROW(N2FrameDesc(8, 2, 5, N2Layout::InputANDMasked), std::runtime_error);

    // InputORMasked without product list should throw
    BOOST_CHECK_THROW(N2FrameDesc(8, 2, 5, N2Layout::InputORMasked), std::runtime_error);

    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_constructor_throws_mismatched_product_count) {
    std::cout << "Testing N2FrameDesc constructor throws on product count mismatch...\n";

    std::vector<N2::prod_ctype> product_list = {{0, 0}, {1, 1}, {2, 2}};

    // num_products (5) != product_list.size() (3) should throw
    BOOST_CHECK_THROW(N2FrameDesc(8, 2, 5, N2Layout::GeneralSubset, product_list),
                      std::runtime_error);

    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_get_product_list_fulluppertri) {
    std::cout << "Testing get_product_list() for FullUpperTri...\n";

    uint32_t num_elements = 4;
    uint32_t num_prod = N2FrameDesc::get_num_prod(num_elements, N2Layout::FullUpperTri);

    N2FrameDesc desc(num_elements, 0, num_prod, N2Layout::FullUpperTri);

    const auto& prods = desc.get_product_list();

    BOOST_CHECK_EQUAL(prods.size(), 10); // 4*5/2 = 10

    // Verify upper triangle order: (0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3),
    // (3,3)
    std::vector<std::pair<uint16_t, uint16_t>> expected = {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {1, 1},
                                                           {1, 2}, {1, 3}, {2, 2}, {2, 3}, {3, 3}};

    for (size_t i = 0; i < expected.size(); ++i) {
        BOOST_CHECK_EQUAL(prods[i].input_a, expected[i].first);
        BOOST_CHECK_EQUAL(prods[i].input_b, expected[i].second);
    }

    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_get_product_list_autocorrelations) {
    std::cout << "Testing get_product_list() for Autocorrelations...\n";

    uint32_t num_elements = 4;
    uint32_t num_prod = N2FrameDesc::get_num_prod(num_elements, N2Layout::Autocorrelations);

    N2FrameDesc desc(num_elements, 0, num_prod, N2Layout::Autocorrelations);

    const auto& prods = desc.get_product_list();

    BOOST_CHECK_EQUAL(prods.size(), 4);

    // Verify diagonal: (0,0), (1,1), (2,2), (3,3)
    for (uint16_t i = 0; i < num_elements; ++i) {
        BOOST_CHECK_EQUAL(prods[i].input_a, i);
        BOOST_CHECK_EQUAL(prods[i].input_b, i);
    }

    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_get_product_list_general_subset) {
    std::cout << "Testing get_product_list() for GeneralSubset...\n";

    std::vector<N2::prod_ctype> product_list = {{0, 0}, {1, 3}, {2, 5}, {7, 7}};

    N2FrameDesc desc(8, 0, product_list.size(), N2Layout::GeneralSubset, product_list);

    const auto& prods = desc.get_product_list();

    BOOST_CHECK_EQUAL(prods.size(), product_list.size());

    // Should return the same product list
    for (size_t i = 0; i < product_list.size(); ++i) {
        BOOST_CHECK_EQUAL(prods[i].input_a, product_list[i].input_a);
        BOOST_CHECK_EQUAL(prods[i].input_b, product_list[i].input_b);
    }

    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_equality_fulluppertri) {
    std::cout << "Testing operator== for FullUpperTri descriptors...\n";

    N2FrameDesc desc1(8, 2, 36, N2Layout::FullUpperTri);
    N2FrameDesc desc2(8, 2, 36, N2Layout::FullUpperTri);
    N2FrameDesc desc3(16, 2, 136, N2Layout::FullUpperTri); // Different num_elements

    BOOST_CHECK(desc1 == desc2);
    BOOST_CHECK(!(desc1 == desc3));

    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_equality_with_product_list) {
    std::cout << "Testing operator== for descriptors with product lists...\n";

    std::vector<N2::prod_ctype> list1 = {{0, 0}, {1, 1}, {2, 2}};
    std::vector<N2::prod_ctype> list2 = {{0, 0}, {1, 1}, {2, 2}};
    std::vector<N2::prod_ctype> list3 = {{0, 0}, {1, 1}, {3, 3}}; // Different product

    N2FrameDesc desc1(8, 0, list1.size(), N2Layout::GeneralSubset, list1);
    N2FrameDesc desc2(8, 0, list2.size(), N2Layout::GeneralSubset, list2);
    N2FrameDesc desc3(8, 0, list3.size(), N2Layout::GeneralSubset, list3);

    BOOST_CHECK(desc1 == desc2);    // Same product list
    BOOST_CHECK(!(desc1 == desc3)); // Different product list

    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_byte_size) {
    std::cout << "Testing get_byte_size()...\n";

    // Frame size depends on num_elements, num_ev, and num_products
    uint32_t num_elements = 8;
    uint32_t num_ev = 2;
    uint32_t num_prod = N2FrameDesc::get_num_prod(num_elements, N2Layout::FullUpperTri);

    N2FrameDesc desc(num_elements, num_ev, num_prod, N2Layout::FullUpperTri);

    size_t expected_size = sizeof(N2::cfloat) * num_prod                // vis
                           + sizeof(float) * num_prod                   // weight
                           + sizeof(float) * num_elements               // flags
                           + sizeof(float) * num_ev                     // eval
                           + sizeof(N2::cfloat) * num_ev * num_elements // evec
                           + sizeof(N2EigenMethod)                      // emethod
                           + sizeof(float)                              // erms
                           + sizeof(float) * 3 // radiometer_chi2 - 3 pol pairs XX, XY, YY

                           + sizeof(N2::cfloat) * num_elements // gain
                           + sizeof(uint8_t) * num_elements;   // mask

    BOOST_CHECK_EQUAL(desc.get_byte_size(), expected_size);

    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_generate_product_list_fulluppertri) {
    std::cout << "Testing generate_product_list() for FullUpperTri...\n";

    uint32_t num_elements = 4;
    auto products = N2FrameDesc::generate_product_list(num_elements, N2Layout::FullUpperTri);

    BOOST_CHECK_EQUAL(products.size(), 10); // 4*5/2 = 10

    // Verify upper triangle order
    std::vector<std::pair<uint16_t, uint16_t>> expected = {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {1, 1},
                                                           {1, 2}, {1, 3}, {2, 2}, {2, 3}, {3, 3}};
    for (size_t i = 0; i < expected.size(); ++i) {
        BOOST_CHECK_EQUAL(products[i].input_a, expected[i].first);
        BOOST_CHECK_EQUAL(products[i].input_b, expected[i].second);
    }

    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_generate_product_list_autocorrelations) {
    std::cout << "Testing generate_product_list() for Autocorrelations...\n";

    uint32_t num_elements = 4;
    auto products = N2FrameDesc::generate_product_list(num_elements, N2Layout::Autocorrelations);

    BOOST_CHECK_EQUAL(products.size(), 4);

    for (uint16_t i = 0; i < num_elements; ++i) {
        BOOST_CHECK_EQUAL(products[i].input_a, i);
        BOOST_CHECK_EQUAL(products[i].input_b, i);
    }

    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_generate_product_list_InputORMasked) {
    std::cout << "Testing generate_product_list() for InputORMasked...\n";

    uint32_t num_elements = 4;
    std::vector<uint16_t> input_list = {0, 2}; // Include inputs 0 and 2

    auto products =
        N2FrameDesc::generate_product_list(num_elements, N2Layout::InputORMasked, input_list);

    // InputORMasked: include product (i,j) if i OR j is in input_list
    // With inputs {0,2} and 4 elements, the products are:
    // (0,0), (0,1), (0,2), (0,3) - all have input 0
    // (1,2), (1,3) - skip (1,1) no match, (1,2) has input 2, skip (1,3) no match
    // (2,2), (2,3) - both have input 2
    // (3,3) - skip, no match
    // Expected: (0,0), (0,1), (0,2), (0,3), (1,2), (2,2), (2,3)
    BOOST_CHECK_EQUAL(products.size(), 7);

    // Verify specific products are present
    std::set<std::pair<uint16_t, uint16_t>> product_set;
    for (const auto& p : products) {
        product_set.insert({p.input_a, p.input_b});
    }
    BOOST_CHECK(product_set.count({0, 0}) > 0);
    BOOST_CHECK(product_set.count({0, 1}) > 0);
    BOOST_CHECK(product_set.count({0, 2}) > 0);
    BOOST_CHECK(product_set.count({0, 3}) > 0);
    BOOST_CHECK(product_set.count({1, 2}) > 0);
    BOOST_CHECK(product_set.count({2, 2}) > 0);
    BOOST_CHECK(product_set.count({2, 3}) > 0);

    // These should NOT be present
    BOOST_CHECK(product_set.count({1, 1}) == 0);
    BOOST_CHECK(product_set.count({1, 3}) == 0);
    BOOST_CHECK(product_set.count({3, 3}) == 0);

    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_generate_product_list_InputANDMasked) {
    std::cout << "Testing generate_product_list() for InputANDMasked...\n";

    uint32_t num_elements = 4;
    std::vector<uint16_t> input_list = {0, 1, 2}; // Include inputs 0, 1, and 2

    auto products =
        N2FrameDesc::generate_product_list(num_elements, N2Layout::InputANDMasked, input_list);

    // InputANDMasked: include product (i,j) if i AND j are both in input_list
    // With inputs {0,1,2} and 4 elements, the products are:
    // (0,0), (0,1), (0,2) - all pairs within {0,1,2}
    // (1,1), (1,2)
    // (2,2)
    // Excludes anything with input 3
    // Expected: (0,0), (0,1), (0,2), (1,1), (1,2), (2,2) = 6 products
    BOOST_CHECK_EQUAL(products.size(), 6);

    // Verify specific products are present
    std::set<std::pair<uint16_t, uint16_t>> product_set;
    for (const auto& p : products) {
        product_set.insert({p.input_a, p.input_b});
    }
    BOOST_CHECK(product_set.count({0, 0}) > 0);
    BOOST_CHECK(product_set.count({0, 1}) > 0);
    BOOST_CHECK(product_set.count({0, 2}) > 0);
    BOOST_CHECK(product_set.count({1, 1}) > 0);
    BOOST_CHECK(product_set.count({1, 2}) > 0);
    BOOST_CHECK(product_set.count({2, 2}) > 0);

    // These should NOT be present (involve input 3)
    BOOST_CHECK(product_set.count({0, 3}) == 0);
    BOOST_CHECK(product_set.count({1, 3}) == 0);
    BOOST_CHECK(product_set.count({2, 3}) == 0);
    BOOST_CHECK(product_set.count({3, 3}) == 0);

    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_generate_product_list_throws_for_unsupported_layout) {
    std::cout << "Testing generate_product_list() throws for unsupported layouts...\n";

    // GeneralSubset and RedundantBaselineAvg are not supported by generate_product_list
    // (product lists for those layouts are read directly from config)
    BOOST_CHECK_THROW(N2FrameDesc::generate_product_list(8, N2Layout::GeneralSubset),
                      std::runtime_error);
    BOOST_CHECK_THROW(N2FrameDesc::generate_product_list(8, N2Layout::RedundantBaselineAvg),
                      std::runtime_error);

    std::cout << "Success.\n";
}
