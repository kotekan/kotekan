/**
 * @file
 * @brief GuoWang's high-rate (CSK) symbols: each frame's data half despread at its 64 shifts.
 *  - GnssCskDespread : public kotekan::Stage
 */
#ifndef GNSS_CSK_DESPREAD_STAGE_HPP
#define GNSS_CSK_DESPREAD_STAGE_HPP

#include "Config.hpp"                 // for Config
#include "Stage.hpp"                  // for Stage
#include "buffer.hpp"                 // for Buffer
#include "bufferContainer.hpp"        // for bufferContainer
#include "gnssChannelizedReplica.hpp" // for ChannelizedReplicaBank, HopRateFilter
#include "restServer.hpp"             // for connectionInstance

#include "json.hpp" // for json

#include <array>              // for array
#include <complex>            // for complex
#include <condition_variable> // for condition_variable
#include <cstdint>            // for int8_t
#include <cstdio>             // for FILE
#include <deque>              // for deque
#include <map>                // for map
#include <memory>             // for unique_ptr, shared_ptr
#include <mutex>              // for mutex
#include <shared_mutex>       // for shared_mutex
#include <string>             // for string
#include <thread>             // for thread
#include <vector>             // for vector

/**
 * @class GnssCskDespread
 * @brief GuoWang 1202's high-rate channel (MCSK): the second millisecond of every 2 ms frame is
 *        a second code of the ranging code's family, cyclically shifted by k = 0..63 chips, six
 *        bits per frame. For each satellite the tracker holds, every frame's data half is
 *        despread at the 64 shifts and the strongest kept.
 *
 * The GW_1202 tracker's records give each slot's code model (Doppler, code-phase argument and
 * start sample, gnssRecord.hpp); the channelized data come from the CPU search's buffer
 * (GnssChordDequantize: conjugated cfloat, all channels). One pass of the replica bank's
 * adjoint (gnss::ChannelizedReplicaBank::hoprate_adjoint) gives the data half's 2046 chip
 * outputs z_j, under the model of the record that holds the half's middle chip, so the carrier
 * is continuous across it. Shift k is then sum_j csk((j + shift_sign * k) mod 2046) z_j: the 64
 * shifts cost additions only. A satellite's CSK code is its own G2' phase of the same family,
 * paired with the ranging code by a fixed table (@c csk_g2); a slot without one is only
 * captured.
 *
 * Frames are numbered per slot from the first one despread: each next frame's number is the
 * last one's plus its distance in receiver time over 2 ms, rounded, which the Doppler cannot
 * move by half a frame. A restart (PRN change, a gap of a second) renumbers from the frame's
 * receiver time over 2 ms.
 *
 * Both buffers are released at once. Workers take the slots round-robin and drop their oldest
 * records when they fall behind, so the tracker's records and the search's data never wait.
 *
 * @par REST
 * GET `<unique_name>/get_status`: one row per slot with symbols, {prn, csk_g2, snr_db, n_sym,
 * n_drop, utc_ref, id_ref, sym}: sym holds the newest @c symbol_export frames as
 * [id, k1, db1, k2, db2, k3, db3, k4, db4], the four strongest shifts and their power over the
 * mean of the other 60, dB; frame id starts at utc_ref + (id - id_ref) * 2 ms.
 * A slot whose newest symbol is 2 s older than the newest slot's is left out.
 * POST `<unique_name>/capture` {prn, seconds, path}: append each data half's chip outputs for
 * that PRN to @c path for @c seconds, per frame an int64 id, a float64 UTC, a float64 Doppler
 * and 2046 complex64 z_j.
 *
 * @buffer rec_buf  The GW_1202 tracker's records, one record per frame.
 *     @buffer_format n_prn * record_stride(n_elements) float32
 *     @buffer_metadata GnssChanMetadata
 * @buffer chan_buf Channelized data as the CPU search reads it: [hop][channel] cfloat32.
 *     @buffer_format n_hops * channel_ids.size() complex64
 *     @buffer_metadata GnssChanMetadata
 *
 * @conf signal        String. GW_1202.
 * @conf prns          List of int. The tracker's slot -> PRN map at start.
 * @conf f_offset      Double. Carrier offset from the tuned frequency, Hz.
 * @conf channel_ids   List of int. The channel each chan_buf column holds.
 * @conf csk_channels  List of int. The channels to despread (the tracker's channel_ids).
 * @conf csk_g2        List of [prn, g2_phase] (optional). Each satellite's CSK code.
 * @conf shift_sign    Int (default 1). +1: shift k reads chip (j + k) of the CSK code at j.
 * @conf min_snr_db    Double (default 6). A slot is despread while its prompt power stands this
 *                     far over the median slot's.
 * @conf n_workers     Int (default 2). Despread threads.
 * @conf ring_frames   Int (default 16). chan_buf frames kept for the records to catch up.
 * @conf symbol_export Int (default 600). Symbols per slot in get_status.
 * @conf n_elements, hops_per_record, fft_length, sample_rate, spectrum_length, num_taps,
 *       pfb_window, input_type, pfb_time_order  As the tracker's and the search's.
 */
class GnssCskDespread : public kotekan::Stage {
public:
    GnssCskDespread(kotekan::Config& config, const std::string& unique_name,
                    kotekan::bufferContainer& buffer_container);
    ~GnssCskDespread() override;
    void main_thread() override;

private:
    /// One record's model for the slots: what a worker needs once the frame is released.
    struct RecordBatch {
        long long wstart;
        std::vector<int> prn;
        std::vector<float> dop, arg; // Doppler, code-phase argument
        std::vector<double> utc;
        std::vector<char> active; // prompt power over the median slot's by min_snr_db
    };
    /// One chan_buf frame's despread channels, [channel][hop].
    struct ChanFrame {
        long long first_hop = -1;
        int n_hops = 0;
        std::vector<std::vector<std::complex<float>>> data;
    };
    struct Symbol {
        long long id;
        std::array<int, 4> k;
        std::array<float, 4> db;
    };
    struct Slot {
        int prn = 0;
        long long last_id = -1;
        long double last_n = 0.0L; ///< receiver sample of the last despread half's start
        double filt_dop = 1e30;
        gnss::ChannelizedReplicaBank::HopRateFilter filt;
        const std::vector<float>* csk = nullptr; ///< the CSK code twice over, as +-1
        std::deque<Symbol> sym;
        long long n_sym = 0, n_drop = 0;
        double utc_ref = 0.0;
        long long id_ref = 0;
        double snr_db = 0.0;
        FILE* cap = nullptr;
        double cap_until = 0.0;
    };

    void chan_thread();
    void worker(int w);
    void push_record(const float* rec, long long wstart);
    /// Despread one record's frames for slot @p p; worker-owned state, output under _m.
    void despread(Slot& s, const RecordBatch& b, int p, std::vector<std::complex<double>>& z,
                  std::vector<std::vector<std::complex<float>>>& scratch);
    /// Copy hops [h0, h0 + n) of every channel into @p out: 0 when done, 1 when the newest
    /// chan_buf frame ends before them (wait for it), -1 when some are gone or were dropped.
    int gather(long long h0, int n, std::vector<std::vector<std::complex<float>>>& out);
    void get_status_callback(kotekan::connectionInstance& conn);
    void capture_callback(kotekan::connectionInstance& conn, nlohmann::json& body);

    Buffer* _rec_buf;
    Buffer* _chan_buf;
    int _n_prn, _n_elem, _fft_len, _n_cols, _shift_sign, _n_workers, _keep, _max_queue;
    long long _rec_samples;
    double _fs, _min_snr, _carrier;
    std::vector<int> _cols; ///< chan_buf column of each despread channel
    std::unique_ptr<gnss::ChannelizedReplicaBank> _bank;
    std::vector<int> _want;                 ///< the despread channels, the bank's indices
    std::map<int, std::vector<float>> _csk; ///< PRN -> CSK code, two periods of +-1
    std::map<int, int> _csk_g2;

    std::shared_mutex _ring_m;
    std::vector<ChanFrame> _ring;
    size_t _ring_next = 0;
    long long _ring_end = -1; ///< the hop after the newest frame's last
    std::mutex _new_m;
    std::condition_variable _new_cv; ///< a chan_buf frame arrived
    std::thread _chan_th;

    std::mutex _q_m;
    std::condition_variable _q_cv;
    std::vector<std::deque<std::shared_ptr<const RecordBatch>>> _queues;
    std::vector<std::thread> _workers;
    std::vector<double> _pw_avg; ///< each slot's prompt power |P|^2/E, averaged (main thread)
    std::vector<int> _pw_prn;

    std::mutex _m; ///< guards _slots' output (sym, counters) and captures against REST
    std::vector<Slot> _slots;
};

#endif // GNSS_CSK_DESPREAD_STAGE_HPP
