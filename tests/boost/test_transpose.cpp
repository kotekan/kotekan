#define BOOST_TEST_MODULE "test_visTranspose"

#include "visTranspose.hpp" // for strided_copy

#include <boost/test/included/unit_test.hpp> // for BOOST_PP_IIF_1, BOOST_PP_IIF_0, BOOST_PP_BO...
#include <cstdint>                           // for uint32_t, uint8_t
#include <cstdlib>                           // for malloc


BOOST_AUTO_TEST_CASE(_strided_copy) {
    // void strided_copy(T* in, T* out, size_t offset, size_t stride,
    //                      size_t n_val)

    uint32_t in[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    uint32_t* out = (uint32_t*)malloc(10 * sizeof(uint32_t));

    strided_copy<uint32_t>(in, out, 0, 1, 10);

    for (uint8_t i = 0; i < 10; i++)
        BOOST_CHECK_EQUAL(out[i], in[i]);

    strided_copy<uint32_t>(in, out, 0, 2, 5);
    strided_copy<uint32_t>(in, out, 1, 2, 5);

    for (uint8_t i = 0; i < 10; i++)
        BOOST_CHECK_EQUAL(out[i], (uint32_t)(in[i] / 2));
}
