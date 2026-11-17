#ifndef GNSS_GPU_RECORD_ASSEMBLE_HPP
#define GNSS_GPU_RECORD_ASSEMBLE_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "gnssEplAssemble.hpp"
#include "gnssGpuRows.hpp"
#include "gnssHostRows.hpp"
#include "gnssRecordChain.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

/**
 * @class GnssGpuRecordAssemble
 * @brief Host tail of the GPU tracking chain: gnssGpuChain frames -> tracker records.
 *
 * Consumes the cudaProcess output (a control block and raw per-channel E/P/L correlations,
 * laid out in gnssGpuChain.hpp). For each PRN it sums the correlations over the PRN's covering
 * channel mask, integrates and applies the carrier-NCO phase (the phase state lives here; the
 * slope f_nco comes in the control block), and writes the gnssRecord.hpp record that the
 * combiner, broker and viewer read. Emits one out_buf frame per record, with the record's
 * start sample (wstart) in GnssChanMetadata::sample_seq. The assembly itself is a
 * gnss::RecordChain; this stage moves its frames. With a @c chains block it hosts one chain per
 * row (see gnss::host_rows), each with its own buffers and thread, under the REST paths, buffer
 * client names, metric labels and board owners of the one-row stage `<key>_n2assemble` it
 * replaces, and serves the rows at `<unique_name>/chains`. With @c workers its rows run on
 * pinned threads instead, each running its rows in order every frame and reading their inputs
 * as one consumer, `<unique_name>/w<k>`: a frame of tiles_buf and ctl_buf that several rows
 * read is released once every worker has assembled its rows' frames from it.
 *
 * A row reads its gnssGpuChain frames from @c in_buf (GnssN2RecordAssemble's output), or, with
 * @c tiles_buf and @c ctl_buf, assembles each one itself from the tiles and the control block
 * into its own frame, as GnssN2RecordAssemble does, and releases both before its first record.
 *
 * @conf in_buf   gnssGpuChain frames from the cudaProcess chain
 * @conf tiles_buf, ctl_buf  instead of in_buf: GnssN2RecordAssemble's inputs, with its
 *                num_elements, num_synth, lanes_per_slot and hops_per_record, and fft_len, the
 *                hop length in samples. Each tiles frame must start on its control block's seq0.
 * @conf gnss_row, fg_total  with tiles_buf and ctl_buf, when they are a GPU half's shared
 *                arrays: the row's gnss_rows entry (gnssGpuRows.hpp) and the tiles' channel
 *                count. The row reads its channels at fg_offset and its control block at
 *                ctl_offset, and FATALs unless that block carries its row tag.
 * @conf out_buf  tracker records: n_prn * record_stride floats, plus the per-channel block
 *                with chan_export
 * @conf prns     initial slot -> PRN map; its length must equal the frame's n_prn
 * @conf sample_rate  rate of the wstart sample counter, Hz
 * @conf record_cp_mod_chips  Float (default 0 = off). Reduce record slot 2 (the code-phase
 *                argument, float32) mod this many chips: the primary code length on a chain
 *                whose baked code is long enough to cost float32 precision (GLO_L2OC_P_OC).
 * @conf elem_proj_sys  String (default "G"). The constellation letter of the chain's PRNs
 *                (G, E, C or R), which names its satellites in the bright-satellite projection.
 * @conf chains   Map (optional). One row per entry; each sets ROW_KEYS itself, alone sets
 *                ROW_ONLY_KEYS, and reads every other key from this block.
 * @conf workers  List (optional, with chains). Worker k, `{rows: [row keys], cpu_affinity:
 *                [cores]}`, runs its rows in that order on one thread bound to its cores, named
 *                for its row if it has one. Every row is in exactly one worker. Rows of one
 *                worker share tiles_buf and ctl_buf both or neither, and never in_buf. They also
 *                share the thread: a row waiting on a full out_buf holds the next one, so they
 *                must not feed one consumer that reads them in lockstep, and on a busy core
 *                they get one thread's share of it.
 */
class GnssGpuRecordAssemble : public kotekan::Stage {
public:
    /// @p clock_s is the chain's clock (gnss::RecordChain); tests pass their own.
    GnssGpuRecordAssemble(kotekan::Config& config, const std::string& unique_name,
                          kotekan::bufferContainer& buffer_container,
                          double (*clock_s)() = gnss::steady_now_s);
    ~GnssGpuRecordAssemble() override;
    void main_thread() override;

    /// Row @p row's own gnssGpuChain frame when it reads tiles_buf and ctl_buf, else null. Stable
    /// only while the row's worker waits for input.
    uint8_t* epl_frame(size_t row) {
        return _rows[row].epl.get();
    }

    /// Keys each row of a @c chains block must set itself.
    static const std::vector<std::string> ROW_KEYS;
    /// Keys only a row of a @c chains block may set.
    static const std::vector<std::string> ROW_ONLY_KEYS;

private:
    struct Row {
        Row(kotekan::Config& config, const gnss::HostRow& row, double (*clock_s)());
        Buffer* in_buf = nullptr;
        int in_id = 0;
        /// With tiles_buf and ctl_buf: their shape, and the frame assembled from them.
        Buffer* tiles_buf = nullptr;
        Buffer* ctl_buf = nullptr;
        int tiles_id = 0, ctl_id = 0;
        /// Its frame holds the next tiles and control-block frames, assembled by its worker with
        /// an earlier row's, and its records are not yet run.
        bool assembled = false;
        gnss_gpu::EplShape epl_shape{};
        /// With gnss_row: the block's offset in ctl_buf and the row tag it must carry; else
        /// 0, the tag of a chain's own control block.
        size_t ctl_offset = 0;
        int64_t tag = 0;
        int64_t fft_len = 0;
        std::unique_ptr<uint8_t[]> epl;
        size_t epl_bytes = 0;
        Buffer* out_buf = nullptr;
        int out_id = 0;
        /// Push leg: each completed window is pushed to cube_buf with its address (chain, window
        /// index, wstart). It never blocks the tracker: if the next frame is not empty (safe to
        /// test, as this row is the only producer) the window is dropped and counted in
        /// _cube_dropped, which every frame carries, so a gap in the window index without a rise
        /// in the count was not dropped here.
        Buffer* cube_out_buf = nullptr;
        int cube_out_id = 0;
        gnss::RecordChain chain;
    };
    const std::vector<gnss::HostRow> _host_rows;
    std::deque<Row> _rows;
    /// Its threads: each runs its rows in order every frame (gnss::host_workers).
    std::vector<gnss::HostWorker> _workers;
    /// The REST paths this host registered.
    std::vector<std::string> _rest_paths;

    /// Register worker @p w as the consumer of its rows' input buffers, once each.
    void register_inputs(const gnss::HostWorker& w);
    void run_worker(size_t w);
    /// run_worker on a thread of its own, counted among the stage's threads.
    void run_pinned(size_t w);
    /// Run row @p row's records of its input frame @p in. Returns false when shutting down.
    bool run_records(size_t row, const uint8_t* in);
    /// As worker @p w, wait for the next tiles and control-block frames of its row at @p pos,
    /// assemble from them the frame of that row and of each later row of @p w that reads the
    /// same two buffers, and release both. Returns false when shutting down.
    bool assemble_frames(const gnss::HostWorker& w, size_t pos);
    /// Take @p row's slice of a GPU half's tiles and control block from its gnss_row.
    void read_gpu_row(kotekan::Config& config, const gnss::HostRow& hr, Row& row);
    /// Pack row @p row's completed window into its cube_buf and mark it full, or count it
    /// dropped if the next frame is not empty.
    void emit_cube_window(size_t row);
};

#endif
