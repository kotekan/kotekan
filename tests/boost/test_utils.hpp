#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include "visUtil.hpp"

#include <boost/test/included/unit_test.hpp>
#include <vector>

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
