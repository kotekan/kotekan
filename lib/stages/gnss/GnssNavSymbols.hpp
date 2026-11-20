/**
 * @file
 * @brief Data symbols of a pilot chain's data component, decided as the records arrive.
 *  - GnssNavSymbols : public kotekan::Stage
 */

#ifndef GNSS_NAV_SYMBOLS_STAGE_HPP
#define GNSS_NAV_SYMBOLS_STAGE_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "gnssNavSymbols.hpp"  // for NavSymbolTrack
#include "restServer.hpp"      // for connectionInstance

#include <mutex>  // for mutex
#include <string> // for string
#include <vector> // for vector

/**
 * @class GnssNavSymbols
 * @brief Each satellite's data symbols, from a pilot chain's records and those of its data
 *        component (cudaGnssTrack's data block), decided as the records arrive
 *        (gnss::NavSymbolTrack).
 *
 * The two record buffers come from GnssGpuRecordAssemble stages fed by one cudaGnssTrack, so
 * their frames pair up by sample_seq and their slots by PRN. A slot whose pilot or data record
 * has no job, or whose PRN changes, starts a new stream.
 *
 * @par REST
 * GET `<unique_name>/get_status`: one row per slot with decisions, {prn, n_sym, n_restart,
 * snr_db, residual_deg, data_pilot_db, n_records, n_boundary, nav_obs}, nav_obs in the
 * combiner's form (bit_export) holding the newest @c symbol_export decisions: {utc_ref, rec_dt,
 * phase: 0, br, bits: [[index, sign]]}, symbol `index` starting at utc_ref + index * br * rec_dt
 * and utc_ref the oldest one's start. The broker's --cnav-combiner reads it.
 *
 * @buffer pilot_buf The pilot chain's records (gnssRecord.hpp), one record per frame.
 *     @buffer_format n_prn * record_stride(n_elements) float32
 *     @buffer_metadata GnssChanMetadata
 * @buffer data_buf The data component's records, as pilot_buf's.
 *     @buffer_format n_prn * record_stride(n_elements) float32
 *     @buffer_metadata GnssChanMetadata
 *
 * @conf n_prn           Int. PRN slots per record.
 * @conf n_elements      Int. Element blocks per record.
 * @conf hops_per_record Int. Hops per record.
 * @conf fft_length      Int. Samples per hop.
 * @conf sample_rate     Double. Sample rate, Hz.
 * @conf symbol_records  Int. Records per symbol: the data code's period over the record's.
 * @conf offset_deg      Double. The data carrier's phase offset from the pilot's, degrees
 *                       (90 for GPS L5's I5 and Q5).
 * @conf symbol_export   Int (default 1500). Decisions per slot in get_status.
 */
class GnssNavSymbols : public kotekan::Stage {
public:
    GnssNavSymbols(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);
    ~GnssNavSymbols() override;
    void main_thread() override;

private:
    void get_status_callback(kotekan::connectionInstance& conn);
    /// Add one record pair, @p pilot and @p data, to every slot's track.
    void push_records(const float* pilot, const float* data, long long seq);

    Buffer* _pilot_buf;
    Buffer* _data_buf;
    int _n_prn, _n_elem, _symbol_records;
    long long _rec_samples;
    double _rec_dt;

    std::mutex _m; ///< guards _tracks and _prn against get_status
    std::vector<gnss::NavSymbolTrack> _tracks;
    std::vector<int> _prn; ///< the PRN each track follows; 0 = none
};

#endif // GNSS_NAV_SYMBOLS_STAGE_HPP
