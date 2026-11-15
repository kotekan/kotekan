#include "gnssEplAssemble.hpp"

#include "gnssGpuChain.hpp"
#include "kotekanLogging.hpp"

#include <cmath>
#include <cstring>

namespace gnss_gpu {

int assemble_epl(const CtlView& ctl, const TilesView& tiles, uint8_t* out, size_t out_bytes,
                 int n_prn, int lanes_per_slot, int num_elements, int n_live, int hops_per_record) {
    // The tile geometry of cudaCorrelatorDual::build_tile_selection.
    const int na16 = num_elements / 16;
    const int nlive16 = (n_live + 15) / 16;

    const auto* ihdr = (const gnss_gpu::FrameHdr*)ctl.ptr;
    const auto* iwin = (const int64_t*)(ctl.ptr + gnss_gpu::off_winstart());
    const auto* ictl = (const gnss_gpu::PrnCtl*)(ctl.ptr + gnss_gpu::off_prnctl());
    const size_t ctl_off_energy =
        gnss_gpu::off_prnctl() + sizeof(gnss_gpu::PrnCtl) * gnss_gpu::MAX_REC * n_prn;
    const auto* ienergy = (const double*)(ctl.ptr + ctl_off_energy);

    const int n_rec = ihdr->n_rec;
    const int n_chan = ihdr->n_chan;
    if (out_bytes < frame_bytes(n_prn, n_chan, ROWS_PLAIN, n_live))
        FATAL_ERROR_NON_OO("assemble_epl: out holds {:d} bytes, {:d} PRN x {:d} chan x {:d} elem "
                           "need {:d}",
                           out_bytes, n_prn, n_chan, n_live,
                           frame_bytes(n_prn, n_chan, ROWS_PLAIN, n_live));
    // Only the header, the window starts and PrnCtl are zeroed: the frame is sized for
    // MAX_REC records, and nothing reads a row past n_jobs.
    std::memset(out, 0, gnss_gpu::off_corr(n_prn));
    auto* ohdr = (gnss_gpu::FrameHdr*)out;
    auto* owin = (int64_t*)(out + gnss_gpu::off_winstart());
    auto* octl = (gnss_gpu::PrnCtl*)(out + gnss_gpu::off_prnctl());
    auto* ocorr = (double*)(out + gnss_gpu::off_corr(n_prn)); // [job][chan][elem] x2
    auto* oenergy =
        (double*)(out + gnss_gpu::off_energy(n_prn, n_chan, gnss_gpu::ROWS_PLAIN, n_live));

    // One output record per sub-integration: with sub_integration_ntime = hops_per_record
    // the N^2 emits n_rec visibilities per frame, one per record the injector synthesised
    // (see GnssN2RecordAssemble).
    ohdr->n_rec = n_rec;
    ohdr->n_prn = n_prn;
    ohdr->n_chan = n_chan;
    ohdr->n_rows_spec = gnss_gpu::ROWS_PLAIN;
    ohdr->seq0 = ihdr->seq0;
    ohdr->utc0 = ihdr->utc0;

    int n_out_jobs = 0;
    for (int r = 0; r < n_rec; ++r) {
        owin[r] = iwin[r];
        for (int p = 0; p < n_prn; ++p) {
            const gnss_gpu::PrnCtl& in0 = ictl[(size_t)r * n_prn + p];
            gnss_gpu::PrnCtl& oc = octl[(size_t)r * n_prn + p];
            // Copy the PRN before the run check: a dark slot would otherwise ship 0 ("not
            // stamped"), and the assembler must learn that a slot changed satellite while
            // dark, since that is when it drops the old satellite's element cal. Only the PRN
            // is copied, so a dark slot keeps run = 0 from the memset above.
            oc.prn = in0.prn;
            if (!in0.run)
                continue;
            oc = in0;
            oc.job0 = n_out_jobs * gnss_gpu::ROWS_PLAIN;
            // The pack scales a slot by the energy rows of its first running record of the
            // frame (cudaGnssInject).
            int r_scale = 0;
            while (!ictl[(size_t)r_scale * n_prn + p].run)
                ++r_scale;
            const int job_scale = ictl[(size_t)r_scale * n_prn + p].job0;

            // Rows in lane order, P first: under LANES_PROMPT the E and L rows are formed
            // from the finished P row, so it must be in the frame before they are written.
            static constexpr int ROW_ORDER[gnss_gpu::ROWS_PLAIN] = {
                gnss_gpu::ROW_P, gnss_gpu::ROW_PH, gnss_gpu::ROW_E, gnss_gpu::ROW_L};
            for (int ti = 0; ti < gnss_gpu::ROWS_PLAIN; ++ti) {
                const int t = ROW_ORDER[ti];
                const int lr = gnss_gpu::lane_of_row(lanes_per_slot, t);
                const size_t orow = (size_t)(oc.job0 + t);
                if (lr < 0) {
                    // No lane for this trial: the prompt row rotated per channel by the
                    // kernel's rotation, corr_E = corr_P e^{-i rot}, corr_L = corr_P
                    // e^{+i rot} (PrnCtl::rot and the sign convention there), with the
                    // kernel's own E/L energy rows, which equal the prompt's on this path.
                    const size_t prow = (size_t)(oc.job0 + gnss_gpu::ROW_P);
                    const double sgn = (t == gnss_gpu::ROW_E) ? -1.0 : 1.0;
                    for (int c = 0; c < n_chan && c < tiles.n_chan; ++c) {
                        oenergy[orow * n_chan + c] = ienergy[(size_t)(in0.job0 + t) * n_chan + c];
                        const double cr = std::cos(sgn * (double)in0.rot[c]);
                        const double ci = std::sin(sgn * (double)in0.rot[c]);
                        for (int e = 0; e < n_live; ++e) {
                            const size_t pi = ((prow * n_chan + c) * (size_t)n_live + e) * 2;
                            const size_t oi = ((orow * n_chan + c) * (size_t)n_live + e) * 2;
                            const double re = ocorr[pi], im = ocorr[pi + 1];
                            ocorr[oi + 0] = re * cr - im * ci;
                            ocorr[oi + 1] = re * ci + im * cr;
                        }
                    }
                    continue;
                }
                const int lane = lanes_per_slot * p + lr;
                const int gi = num_elements + lane, ihi = gi >> 4, ilo = gi & 15;

                for (int c = 0; c < n_chan && c < tiles.n_chan; ++c) {
                    const int32_t* slice = tiles.slice(r, c);

                    // The pack's scale, for the E, P and L lanes; the pack takes the P_HEAD
                    // lane's rms over its m_head hops instead of hops_per_record.
                    const double e0 = ienergy[(size_t)(job_scale + t) * n_chan + c];
                    const double rms = (e0 > 0.0) ? std::sqrt(e0 / (double)hops_per_record) : 0.0;
                    const double s = (rms > 0.0) ? 7.0 / (3.0 * rms) : 0.0;

                    // Absolute units, not just the right ratio: the combiner uses energy as
                    // an ML combining weight, so it must be the physical replica energy. Emit
                    // this record's E_R as-is and divide the correlation by s.
                    const double e_rec = ienergy[(size_t)(in0.job0 + t) * n_chan + c];
                    oenergy[orow * n_chan + c] = e_rec;

                    for (int e = 0; e < n_live; ++e) {
                        const int jhi = e >> 4, jlo = e & 15;
                        const int mix_k = (ihi - na16) * nlive16 + jhi;
                        const int32_t* tv = slice + (size_t)mix_k * 512 + 32 * ilo + 2 * jlo;
                        // The replica is conjugated in the pack, so V needs no conjugate
                        // here (see GnssN2RecordAssemble).
                        const size_t oi = ((orow * n_chan + c) * (size_t)n_live + e) * 2;
                        const double inv_s = (s > 0.0) ? 1.0 / s : 0.0;
                        ocorr[oi + 0] = (double)tv[0] * inv_s;
                        ocorr[oi + 1] = (double)tv[1] * inv_s;
                    }
                }
            }
            ++n_out_jobs;
        }
    }
    ohdr->n_jobs = n_out_jobs * gnss_gpu::ROWS_PLAIN;
    return n_out_jobs;
}

int EplShape::n_tile() const {
    // Must mirror cudaCorrelatorDual::build_tile_selection and gen_chord_gnss_config.py's
    // n2_tiles_per_chan. The frame-size check catches a count mismatch; an order mismatch reads
    // as garbage correlations.
    return ((num_elements + num_synth) / 16 - num_elements / 16) * ((n_live + 15) / 16);
}

void EplShape::check(const std::string& who, size_t tiles_frame_size) const {
    if (lanes_per_slot != LANES_FULL && lanes_per_slot != LANES_PROMPT)
        FATAL_ERROR_NON_OO("{:s}: lanes_per_slot {:d} must be {:d} or {:d}", who, lanes_per_slot,
                           LANES_FULL, LANES_PROMPT);
    if (lanes_per_slot == LANES_PROMPT && n_chan > MAX_ROT_CHAN)
        FATAL_ERROR_NON_OO("{:s}: {:d} channels exceed the {:d} PrnCtl::rot carries; "
                           "lanes_per_slot {:d} needs every E/L lane packed",
                           who, n_chan, MAX_ROT_CHAN, lanes_per_slot);
    if (lanes_per_slot * n_prn > num_synth)
        FATAL_ERROR_NON_OO("{:s}: {:d}*n_prn ({:d}) exceeds num_synth ({:d})", who, lanes_per_slot,
                           lanes_per_slot * n_prn, num_synth);

    // The tiles frame is n_rec x n_chan x n_tile x (16x16 complex int32), sized by the config
    // generator. A mismatch means its geometry (n2_tiles_per_chan) and this stage's disagree.
    const size_t need = (size_t)n_rec * n_chan * n_tile() * 512 * sizeof(int32_t);
    if (tiles_frame_size != need)
        FATAL_ERROR_NON_OO("{:s}: tiles_buf frame_size {:d} != {:d} for {:d} rec x {:d} chan x "
                           "{:d} mixed tiles",
                           who, tiles_frame_size, need, n_rec, n_chan, n_tile());
}

int EplShape::assemble(const std::string& who, const uint8_t* ctl, const int32_t* tiles,
                       uint8_t* out, size_t out_bytes) const {
    const auto* hdr = (const FrameHdr*)ctl;
    // The producer's lane count is in the frame; a mismatch would read P's tile as E's.
    if (lanes_per_slot_of(hdr->lanes_per_slot) != lanes_per_slot)
        FATAL_ERROR_NON_OO("{:s}: the control block says {:d} lanes per slot, this stage is "
                           "configured for {:d} -- cudaGnssInject and this stage must carry the "
                           "same lanes_per_slot",
                           who, lanes_per_slot_of(hdr->lanes_per_slot), lanes_per_slot);

    // Every row below n_jobs is written whole, for every channel and element, which needs the
    // frame to have this stage's shape.
    if (hdr->n_prn != n_prn || hdr->n_chan != n_chan || hdr->n_rec != n_rec || hdr->n_rec > MAX_REC)
        FATAL_ERROR_NON_OO("{:s}: the control block is {:d} PRN x {:d} chan x {:d} records, this "
                           "stage {:d} x {:d} x {:d} (at most {:d} records)",
                           who, hdr->n_prn, hdr->n_chan, hdr->n_rec, n_prn, n_chan, n_rec, MAX_REC);
    const TilesView tv{tiles, (size_t)n_chan * n_tile() * 512, 0, n_chan, n_tile()};
    return assemble_epl({ctl}, tv, out, out_bytes, n_prn, lanes_per_slot, num_elements, n_live,
                        hops_per_record);
}

} // namespace gnss_gpu
