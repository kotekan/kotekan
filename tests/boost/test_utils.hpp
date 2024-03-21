#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include "visUtil.hpp"

#include <boost/test/included/unit_test.hpp>
#include <locale>
#include <vector>

// Use a boost::test "Global Fixture" to set the locale...
// https://www.boost.org/doc/libs/1_75_0/libs/test/doc/html/boost_test/tests_organization/fixtures/global.html
// Enable this in your test suite by adding:
// BOOST_TEST_GLOBAL_FIXTURE(GlobalFixture_Locale);
struct GlobalFixture_Locale {
    GlobalFixture_Locale() {
        std::cout << "Setting locale (stdout)..." << std::endl;
        BOOST_TEST_MESSAGE("Setting locale...");
        try {
            std::locale::global(std::locale::classic());
        } catch (const std::exception& ex) {
            std::cout << "Exception setting locale (stdout)..." << ex.what() << std::endl;
            BOOST_TEST_MESSAGE("Exception setting locale");
        }
    }
    ~GlobalFixture_Locale() {}
};

struct CompareCTypes {
    void check_equal(const std::vector<input_ctype>& a, const std::vector<input_ctype>& b) {
        BOOST_CHECK(!a.empty() && !b.empty());
        BOOST_CHECK(a.size() == b.size());
        auto ita = a.begin();
        auto itb = b.begin();

        while (ita != a.end() || itb != b.end()) {
            BOOST_CHECK_EQUAL(ita->chan_id, itb->chan_id);
            BOOST_CHECK_EQUAL(ita->correlator_input, itb->correlator_input);
            if (ita != a.end())
                ++ita;
            if (itb != b.end())
                ++itb;
        }
    }

    void check_equal(const std::vector<prod_ctype>& a, const std::vector<prod_ctype>& b) {
        BOOST_CHECK(!a.empty() && !b.empty());
        BOOST_CHECK(a.size() == b.size());
        auto ita = a.begin();
        auto itb = b.begin();

        while (ita != a.end() || itb != b.end()) {
            BOOST_CHECK_EQUAL(ita->input_a, itb->input_a);
            BOOST_CHECK_EQUAL(ita->input_b, itb->input_b);
            if (ita != a.end())
                ++ita;
            if (itb != b.end())
                ++itb;
        }
    }

    void check_equal(const std::vector<std::pair<uint32_t, freq_ctype>>& a,
                     const std::vector<std::pair<uint32_t, freq_ctype>>& b) {
        BOOST_CHECK(!a.empty() && !b.empty());
        BOOST_CHECK(a.size() == b.size());
        auto ita = a.begin();
        auto itb = b.begin();

        while (ita != a.end() || itb != b.end()) {
            BOOST_CHECK_EQUAL(ita->first, itb->first);
            BOOST_CHECK_EQUAL(ita->second.centre, itb->second.centre);
            BOOST_CHECK_EQUAL(ita->second.width, itb->second.width);
            if (ita != a.end())
                ++ita;
            if (itb != b.end())
                ++itb;
        }
    }
};

#endif // TEST_UTILS_HPP
