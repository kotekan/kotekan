#define BOOST_TEST_MODULE "test_gnss_cube_write"

#include "GnssCubeWrite.hpp" // for pack_cube_frame, cube_packed_bytes
#include "gnssRecord.hpp"    // for cube_frame_bytes, CUBE_HEADER_BYTES

#include <boost/test/included/unit_test.hpp>
#include <cstdint> // for int32_t, int64_t, uint8_t
#include <cstring> // for memcpy
#include <vector>  // for vector

namespace {

const int MP = 32, MB = 8, NE = 4; // the frame's maxima and antennas
const int NP = 10, NB = 5;         // the chain's slots and bins
const std::vector<int32_t> LIVE = {1, 4, 9};

template<typename T>
void put(std::vector<uint8_t>& f, size_t off, T v) {
    std::memcpy(f.data() + off, &v, sizeof(T));
}

template<typename T>
T get(const std::vector<uint8_t>& f, size_t off) {
    T v;
    std::memcpy(&v, f.data() + off, sizeof(T));
    return v;
}

// A v3 frame: every value encodes its own (array, slot, bin, element) so a misplaced copy shows.
std::vector<uint8_t> make_frame() {
    std::vector<uint8_t> f(gnss::cube_frame_bytes(MP, MB, NE), 0);
    put<int64_t>(f, 0, gnss::CUBE_VERSION);
    put<int64_t>(f, 8, 12345);
    put<int64_t>(f, 40, NP);
    put<int64_t>(f, 48, NB);
    put<int64_t>(f, 56, NE);
    put<int32_t>(f, 88, MP);
    put<int32_t>(f, 92, MB);
    size_t o = gnss::CUBE_HEADER_BYTES;
    for (int s = 0; s < MP; s++)
        put<double>(f, o + 8 * s, 1000.0 + s);
    o += 8 * MP;
    for (int b = 0; b < MB; b++) {
        put<int32_t>(f, o + 4 * b, 2000 + b);
        put<int32_t>(f, o + 4 * (MB + b), 3000 + b);
    }
    o += 8 * MB;
    for (int s : LIVE) {
        put<int32_t>(f, o + 4 * s, 100 + s);          // prn
        put<int32_t>(f, o + 4 * (MP + s), 50 + s);    // n_rec
        put<int32_t>(f, o + 4 * (2 * MP + s), 7 + s); // n_reanchor
    }
    o += 12 * MP;
    for (int a = 0; a < 2; a++, o += 4 * MP * MB)
        for (int s = 0; s < MP; s++)
            for (int b = 0; b < MB; b++)
                put<float>(f, o + 4 * (s * MB + b), 10000.f * (a + 1) + 100.f * s + b);
    for (int a = 0; a < 3; a++, o += 4 * MP * MB * NE)
        for (int s = 0; s < MP; s++)
            for (int b = 0; b < MB; b++)
                for (int e = 0; e < NE; e++)
                    put<float>(f, o + 4 * ((s * MB + b) * NE + e),
                               1e6f * (a + 1) + 1000.f * s + 10.f * b + e);
    BOOST_REQUIRE_EQUAL(o, f.size());
    return f;
}

} // namespace

BOOST_AUTO_TEST_CASE(packs_live_slots_and_actual_bins) {
    std::vector<uint8_t> f = make_frame(), p;
    BOOST_REQUIRE_EQUAL(gnss::pack_cube_frame(f.data(), f.size(), p), "");
    const int NL = LIVE.size();
    BOOST_REQUIRE_EQUAL(p.size(), gnss::cube_packed_bytes(NL, NB, NE));
    BOOST_CHECK_EQUAL(get<int64_t>(p, 0), gnss::CUBE_PACKED_VERSION);
    BOOST_CHECK_EQUAL(get<int64_t>(p, 8), 12345);
    BOOST_CHECK_EQUAL(get<int64_t>(p, 40), NP); // the chain's slot count is kept
    BOOST_CHECK_EQUAL(get<int32_t>(p, 88), NL);
    BOOST_CHECK_EQUAL(get<int32_t>(p, 92), NB);

    size_t o = gnss::CUBE_HEADER_BYTES;
    for (int i = 0; i < NL; i++)
        BOOST_CHECK_EQUAL(get<double>(p, o + 8 * i), 1000.0 + LIVE[i]);
    o += 8 * NL;
    for (int b = 0; b < NB; b++) {
        BOOST_CHECK_EQUAL(get<int32_t>(p, o + 4 * b), 2000 + b);
        BOOST_CHECK_EQUAL(get<int32_t>(p, o + 4 * (NB + b)), 3000 + b);
    }
    o += 8 * NB;
    for (int i = 0; i < NL; i++) {
        BOOST_CHECK_EQUAL(get<int32_t>(p, o + 4 * i), 100 + LIVE[i]);
        BOOST_CHECK_EQUAL(get<int32_t>(p, o + 4 * (NL + i)), 50 + LIVE[i]);
        BOOST_CHECK_EQUAL(get<int32_t>(p, o + 4 * (2 * NL + i)), 7 + LIVE[i]);
        BOOST_CHECK_EQUAL(get<int32_t>(p, o + 4 * (3 * NL + i)), LIVE[i]); // slot
    }
    o += 16 * NL;
    for (int a = 0; a < 2; a++, o += 4 * NL * NB)
        for (int i = 0; i < NL; i++)
            for (int b = 0; b < NB; b++)
                BOOST_CHECK_EQUAL(get<float>(p, o + 4 * (i * NB + b)),
                                  10000.f * (a + 1) + 100.f * LIVE[i] + b);
    for (int a = 0; a < 3; a++, o += 4 * NL * NB * NE)
        for (int i = 0; i < NL; i++)
            for (int b = 0; b < NB; b++)
                for (int e = 0; e < NE; e++)
                    BOOST_CHECK_EQUAL(get<float>(p, o + 4 * ((i * NB + b) * NE + e)),
                                      1e6f * (a + 1) + 1000.f * LIVE[i] + 10.f * b + e);
    BOOST_CHECK_EQUAL(o, p.size());
}

BOOST_AUTO_TEST_CASE(no_live_slot_keeps_the_header) {
    std::vector<uint8_t> f = make_frame(), p;
    size_t n_rec = gnss::CUBE_HEADER_BYTES + 8 * MP + 8 * MB + 4 * MP;
    for (int s : LIVE)
        put<int32_t>(f, n_rec + 4 * s, 0);
    BOOST_REQUIRE_EQUAL(gnss::pack_cube_frame(f.data(), f.size(), p), "");
    BOOST_CHECK_EQUAL(p.size(), gnss::cube_packed_bytes(0, NB, NE));
    BOOST_CHECK_EQUAL(get<int32_t>(p, 88), 0);
}

BOOST_AUTO_TEST_CASE(refuses_what_it_cannot_read) {
    std::vector<uint8_t> f = make_frame(), p;
    BOOST_CHECK_NE(gnss::pack_cube_frame(f.data(), f.size() - 4, p), "");
    put<int64_t>(f, 0, gnss::CUBE_PACKED_VERSION);
    BOOST_CHECK_NE(gnss::pack_cube_frame(f.data(), f.size(), p), "");
    f = make_frame();
    put<int64_t>(f, 48, MB + 1); // n_bin above the frame's maximum
    BOOST_CHECK_NE(gnss::pack_cube_frame(f.data(), f.size(), p), "");
}
