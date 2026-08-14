// telemfmt -- emit ONE task-#59 telemetry frame, straight from the C++ wire structs, so the
// broker's parser can be checked against the definition instead of against a transcription of
// it. Built and run by python/scripts/gnss/test_telem.py; needs no kotekan libraries (the wire
// header is standalone), so the test compiles it on the fly and always tests the CURRENT
// gnssTelem.hpp rather than a binary someone built once.
//
// WHY THIS EXISTS. gnssTelem.hpp is a C++ struct and gnss_broker/telem.py is a struct.Struct
// format string: two independent statements of one layout. Every field-order or padding change
// silently shifts the Python parse, and a shifted parse does not raise -- it yields plausible
// numbers with the wrong meaning, which is the exact failure mode this whole transport exists
// to stop tolerating (see the module header in telem.py). A python-only test can only prove the
// parser is self-consistent. This one makes the C++ the reference.
//
//   usage: telemfmt <out.bin>     writes one frame; prints the layout as JSON on stdout

#include "gnssTelem.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <out.bin>\n", argv[0]);
        return 2;
    }
    const int n_rec = 4, n_prn = 5;

    gnss::TelemHeader h;
    std::memset(&h, 0, sizeof(h));
    h.magic = gnss::TELEM_MAGIC;
    h.version = gnss::TELEM_VERSION;
    h.n_rec = (uint16_t)n_rec;
    h.n_prn = (uint16_t)n_prn;
    h.n_row = (uint16_t)gnss::RECORD_FLOATS;
    h.n_chan = 7;
    h.n_elem = 32;
    h.hops_per_record = 2048;
    h.fft_len = 16384;
    h.win = 1234567ull;
    h.seq = 42ull;
    h.wstart0 = (int64_t)h.win * n_rec * h.hops_per_record * h.fft_len;
    h.utc0 = 1786285988.5;
    // Slot 2 DELIBERATELY MISSING: a dropped record must read as a hole at a known index, not
    // as a shift of everything after it. That distinction is the whole reason slots are
    // addressed rather than appended, so the fixture exercises it.
    h.present = 0b1011u;
    gnss::telem_set_name(h.chain, "gal_e5a");
    gnss::telem_set_name(h.inst, "cx42.1");

    std::vector<float> rows((size_t)n_rec * n_prn * gnss::RECORD_FLOATS, 0.0f);
    for (int r = 0; r < n_rec; ++r)
        for (int p = 0; p < n_prn; ++p) {
            float* row = &rows[gnss::telem_row_offset(r, p, n_prn)];
            for (int f = 0; f < gnss::RECORD_FLOATS; ++f)
                row[f] = (float)(1000 * r + 10 * p + f);
            row[gnss::REC_PRN] = (float)(100 + p);
            row[gnss::REC_P_ENERGY] = (float)(p + 1); // > 0 so every row survives the energy gate
            // UTC is a DOUBLE aliased over two float slots -- the one place a naive
            // float-by-float parse gets a plausible-looking wrong answer.
            *reinterpret_cast<double*>(row + gnss::RECORD_UTC_SLOT) =
                1786285988.5 + 0.0104857 * r + 0.001 * p;
        }

    FILE* f = std::fopen(argv[1], "wb");
    if (!f) {
        std::perror("fopen");
        return 1;
    }
    std::fwrite(&h, sizeof(h), 1, f);
    std::fwrite(rows.data(), sizeof(float), rows.size(), f);
    std::fclose(f);

    std::printf("{\"header_bytes\": %zu, \"telem_header_bytes_const\": %d, \"record_floats\": %d,"
                " \"frame_bytes\": %zu, \"n_rec\": %d, \"n_prn\": %d,"
                " \"off_win\": %zu, \"off_seq\": %zu, \"off_wstart0\": %zu, \"off_utc0\": %zu,"
                " \"off_present\": %zu, \"off_chain\": %zu, \"off_inst\": %zu,"
                " \"magic\": %u, \"version\": %d, \"win\": %llu, \"seq\": %llu,"
                " \"wstart0\": %lld, \"present\": %u,"
                " \"chain\": \"gal_e5a\", \"inst\": \"cx42.1\"}\n",
                sizeof(gnss::TelemHeader), gnss::TELEM_HEADER_BYTES, gnss::RECORD_FLOATS,
                gnss::telem_frame_bytes(n_rec, n_prn), n_rec, n_prn,
                offsetof(gnss::TelemHeader, win), offsetof(gnss::TelemHeader, seq),
                offsetof(gnss::TelemHeader, wstart0), offsetof(gnss::TelemHeader, utc0),
                offsetof(gnss::TelemHeader, present), offsetof(gnss::TelemHeader, chain),
                offsetof(gnss::TelemHeader, inst), gnss::TELEM_MAGIC, gnss::TELEM_VERSION,
                (unsigned long long)h.win, (unsigned long long)h.seq, (long long)h.wstart0,
                h.present);
    return 0;
}
