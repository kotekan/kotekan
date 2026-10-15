// telemfmt: write one telemetry frame from the C++ wire structs, so the broker's parser
// (gnss_broker/telem.py) is checked against gnssTelem.hpp itself rather than a transcription of
// it. python/scripts/gnss/test_telem.py compiles and runs it in setUpClass; it needs only the
// headers in lib/stages/gnss, so the test always exercises the current gnssTelem.hpp.
//
// gnssTelem.hpp and telem.py's struct format strings state one layout twice. A field-order or
// padding change shifts the Python parse without raising, giving plausible numbers with the
// wrong meaning; a Python-only test can prove only that the parser agrees with itself.
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
    // Slot 2 is left empty: a dropped record must read as a hole at its own index, not as a
    // shift of the records after it.
    h.present = 0b1011u;
    gnss::telem_set_name(h.chain, "gal_e5a");
    gnss::telem_set_name(h.inst, "cx42.1");

    const int n_chan = 5;
    // max_chan is the row stride and differs from both n_chan and TELEM_MAX_CHAN, so a reader
    // that strides by either reads every row after the first at the wrong offset. The sixth
    // column stays zero, so a reader that ignores n_chan is caught too.
    const int n_cols = 6;
    h.n_chan = (uint16_t)n_chan;
    h.max_chan = (uint16_t)n_cols;
    h.n_row_total = (uint16_t)gnss::telem_row_floats(n_cols);
    for (int ch = 0; ch < n_chan; ++ch)
        h.chan_id[ch] = (uint16_t)(5972 + 16 * ch); // a real stride-16 comb

    std::vector<float> rows((size_t)n_rec * n_prn * gnss::telem_row_floats(n_cols), 0.0f);
    for (int r = 0; r < n_rec; ++r)
        for (int p = 0; p < n_prn; ++p) {
            float* row = &rows[gnss::telem_row_offset(r, p, n_prn, gnss::telem_row_floats(n_cols))];
            for (int f = 0; f < gnss::RECORD_FLOATS; ++f)
                row[f] = (float)(1000 * r + 10 * p + f);
            row[gnss::REC_PRN] = (float)(100 + p);
            row[gnss::REC_P_ENERGY] = (float)(p + 1); // > 0 so every row survives the energy gate
            // A distinct value per (record, PRN, channel, slot), so a wrong row stride or comb
            // offset reads a number that identifiably belongs to another cell.
            for (int ch = 0; ch < n_chan; ++ch) {
                float* cb = row + gnss::telem_chan_offset(ch);
                cb[gnss::CHAN_RE] = (float)(100000 * r + 1000 * p + 10 * ch + 1);
                cb[gnss::CHAN_IM] = (float)(100000 * r + 1000 * p + 10 * ch + 2);
                cb[gnss::CHAN_ENERGY] = (float)(ch + 1);
                cb[gnss::CHAN_E_RE] = (float)(100000 * r + 1000 * p + 10 * ch + 3);
                cb[gnss::CHAN_E_IM] = (float)(100000 * r + 1000 * p + 10 * ch + 4);
                cb[gnss::CHAN_E_ENERGY] = (float)(ch + 1);
                cb[gnss::CHAN_L_RE] = (float)(100000 * r + 1000 * p + 10 * ch + 5);
                cb[gnss::CHAN_L_IM] = (float)(100000 * r + 1000 * p + 10 * ch + 6);
                cb[gnss::CHAN_L_ENERGY] = (float)(ch + 1);
            }
            // UTC is a double aliased over two float slots, the one field a float-by-float
            // parse gets plausibly wrong.
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

    std::printf("{\"row_floats\": %d, \"max_chan\": %d, \"max_chan_const\": %d,"
                " \"chan_floats\": %d, \"n_chan\": %d,"
                " \"chan_id0\": %d, \"chan_stride\": %zu,"
                " \"header_bytes\": %zu, \"telem_header_bytes_const\": %d, \"record_floats\": %d,"
                " \"frame_bytes\": %zu, \"n_rec\": %d, \"n_prn\": %d,"
                " \"off_win\": %zu, \"off_seq\": %zu, \"off_wstart0\": %zu, \"off_utc0\": %zu,"
                " \"off_present\": %zu, \"off_chain\": %zu, \"off_inst\": %zu,"
                " \"magic\": %u, \"version\": %d, \"win\": %llu, \"seq\": %llu,"
                " \"wstart0\": %lld, \"present\": %u,"
                " \"chain\": \"gal_e5a\", \"inst\": \"cx42.1\"}\n",
                gnss::telem_row_floats(n_cols), n_cols, gnss::TELEM_MAX_CHAN, gnss::CHAN_FLOATS,
                n_chan, (int)h.chan_id[0], gnss::telem_chan_offset(1) - gnss::telem_chan_offset(0),
                sizeof(gnss::TelemHeader), gnss::TELEM_HEADER_BYTES, gnss::RECORD_FLOATS,
                gnss::telem_frame_bytes(n_rec, n_prn, n_cols), n_rec, n_prn,
                offsetof(gnss::TelemHeader, win), offsetof(gnss::TelemHeader, seq),
                offsetof(gnss::TelemHeader, wstart0), offsetof(gnss::TelemHeader, utc0),
                offsetof(gnss::TelemHeader, present), offsetof(gnss::TelemHeader, chain),
                offsetof(gnss::TelemHeader, inst), gnss::TELEM_MAGIC, gnss::TELEM_VERSION,
                (unsigned long long)h.win, (unsigned long long)h.seq, (long long)h.wstart0,
                h.present);
    return 0;
}
