#include "GnssGpuRecordAssemble.hpp"

#include "GnssChanMetadata.hpp"
#include "StageFactory.hpp"
#include "chordMetadata.hpp"    // for chordMetadata, get_chord_metadata
#include "gnssChanPower.hpp"    // for ChanPower
#include "gnssProjSubspace.hpp" // for ProjBoard

#include <sched.h>       // for CPU_SETSIZE
#include <set>           // for set
#include <sys/syscall.h> // for SYS_gettid
#include <unistd.h>      // for syscall

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssGpuRecordAssemble);

const std::vector<std::string> GnssGpuRecordAssemble::ROW_KEYS = {"out_buf", "prns", "channel_ids"};
const std::vector<std::string> GnssGpuRecordAssemble::ROW_ONLY_KEYS = {"beam_cube_chain",
                                                                       "cube_buf",
                                                                       "ctl_buf",
                                                                       "elem_sum_shared_ref",
                                                                       "fg_total",
                                                                       "gnss_row",
                                                                       "in_buf",
                                                                       "lane_transfer_freq_ids",
                                                                       "record_cp_mod_chips",
                                                                       "tiles_buf"};

GnssGpuRecordAssemble::Row::Row(Config& config, const gnss::HostRow& row, double (*clock_s)()) :
    chain(config, row.cfg_path, row.name, gnss::ProjBoard::instance(), gnss::ChanPower::instance(),
          clock_s) {}

GnssGpuRecordAssemble::GnssGpuRecordAssemble(Config& config, const std::string& unique_name,
                                             bufferContainer& buffer_container,
                                             double (*clock_s)()) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssGpuRecordAssemble::main_thread, this)),
    _host_rows(gnss::host_rows(config, unique_name, "_n2assemble", ROW_KEYS, ROW_ONLY_KEYS)) {
    _workers = read_workers(config, unique_name);
    for (const gnss::HostRow& hr : _host_rows) {
        // Exact paths: a row must not take its input from the host block or the root.
        const bool in = config.exists(hr.cfg_path, "in_buf");
        const bool tiles = config.exists(hr.cfg_path, "tiles_buf");
        if (in == tiles || tiles != config.exists(hr.cfg_path, "ctl_buf"))
            FATAL_ERROR("{:s}: set in_buf, or tiles_buf and ctl_buf", hr.name);
        if (config.exists(hr.cfg_path, "gnss_row") != config.exists(hr.cfg_path, "fg_total")
            || (in && config.exists(hr.cfg_path, "gnss_row")))
            FATAL_ERROR("{:s}: gnss_row and fg_total go together, with tiles_buf", hr.name);
        Row& row = _rows.emplace_back(config, hr, clock_s);
        if (in) {
            row.in_buf =
                buffer_container.get_buffer(config.get<std::string>(hr.cfg_path, "in_buf"));
        } else {
            row.tiles_buf =
                buffer_container.get_buffer(config.get<std::string>(hr.cfg_path, "tiles_buf"));
            row.ctl_buf =
                buffer_container.get_buffer(config.get<std::string>(hr.cfg_path, "ctl_buf"));
            const int hops_per_record = config.get<int>(hr.cfg_path, "hops_per_record");
            row.epl_shape = {
                config.get<int>(hr.cfg_path, "num_elements"),
                config.get_default<int>(hr.cfg_path, "num_synth", 128),
                config.get<int>(hr.cfg_path, "n_elements"),
                (int)config.get<std::vector<int>>(hr.cfg_path, "channel_ids").size(),
                (int)config.get<std::vector<int>>(hr.cfg_path, "prns").size(),
                config.get_default<int>(hr.cfg_path, "lanes_per_slot", gnss_gpu::LANES_FULL),
                hops_per_record,
                config.get<int>(hr.cfg_path, "samples_per_data_set") / hops_per_record};
            if (config.exists(hr.cfg_path, "gnss_row"))
                read_gpu_row(config, hr, row);
            row.epl_shape.check(hr.name, row.tiles_buf->frame_size);
            row.fft_len = config.get<int64_t>(hr.cfg_path, "fft_len");
            row.epl_bytes = gnss_gpu::frame_bytes(row.epl_shape.n_prn, row.epl_shape.n_chan,
                                                  gnss_gpu::ROWS_PLAIN, row.epl_shape.n_live);
            row.epl.reset(new uint8_t[row.epl_bytes]);
        }
        row.out_buf = buffer_container.get_buffer(config.get<std::string>(hr.cfg_path, "out_buf"));
        row.out_buf->register_producer(hr.name);
        row.chain.check_out_buf(row.out_buf);
        if (row.chain.cube_on()) {
            const std::string cbuf = config.get<std::string>(hr.cfg_path, "cube_buf");
            row.cube_out_buf = buffer_container.get_buffer(cbuf);
            row.cube_out_buf->register_producer(hr.name);
            row.chain.check_cube_buf(cbuf, row.cube_out_buf);
        }
        for (const auto& v : row.chain.rest_verbs()) {
            if (v.get)
                gnss::rest_register_unique(hr.name + "/" + v.verb, v.get, _rest_paths);
            else
                gnss::rest_register_unique(hr.name + "/" + v.verb, v.post, _rest_paths);
        }
    }
    for (const Worker& w : _workers)
        register_inputs(w);
    gnss::rest_register_chains(unique_name, _host_rows, _rest_paths);
}

std::vector<GnssGpuRecordAssemble::Worker>
GnssGpuRecordAssemble::read_workers(Config& config, const std::string& unique_name) {
    std::vector<Worker> out;
    if (!config.exists(unique_name, "workers")) {
        for (size_t i = 0; i < _host_rows.size(); ++i)
            out.push_back({_host_rows[i].name, {}, {i}});
        return out;
    }
    if (_host_rows[0].key.empty())
        FATAL_ERROR("{:s}: workers needs a chains block", unique_name);
    const nlohmann::json ws = config.get_value(unique_name, "workers");
    if (!ws.is_array() || ws.empty())
        FATAL_ERROR("{:s}: workers must be a non-empty list", unique_name);
    // The worker of each row, ws.size() while it has none.
    std::vector<size_t> of(_host_rows.size(), ws.size());
    for (size_t k = 0; k < ws.size(); ++k) {
        const nlohmann::json& e = ws[k];
        if (!e.is_object() || !e.contains("rows") || !e["rows"].is_array() || e["rows"].empty()
            || !e.contains("cpu_affinity") || !e["cpu_affinity"].is_array()
            || e["cpu_affinity"].empty())
            FATAL_ERROR("{:s}: worker {:d} needs rows and cpu_affinity, each a non-empty list",
                        unique_name, k);
        Worker w{unique_name + "/w" + std::to_string(k), {}, {}};
        for (const nlohmann::json& c : e["cpu_affinity"]) {
            if (!c.is_number_integer() || c.get<int64_t>() < 0 || c.get<int64_t>() >= CPU_SETSIZE)
                FATAL_ERROR("{:s}: worker {:d} has cpu_affinity entry {:s}, not a core",
                            unique_name, k, c.dump());
            w.cpus.push_back(c.get<int>());
        }
        for (const nlohmann::json& key : e["rows"]) {
            size_t i = 0;
            while (i < _host_rows.size()
                   && !(key.is_string() && key.get<std::string>() == _host_rows[i].key))
                ++i;
            if (i == _host_rows.size())
                FATAL_ERROR("{:s}: worker {:d} runs {:s}, which is not a row of chains",
                            unique_name, k, key.dump());
            if (of[i] < ws.size())
                FATAL_ERROR("{:s}: row {:s} is in workers {:d} and {:d}", unique_name,
                            _host_rows[i].key, of[i], k);
            of[i] = k;
            w.rows.push_back(i);
        }
        out.push_back(w);
    }
    for (size_t i = 0; i < _host_rows.size(); ++i)
        if (of[i] == ws.size())
            FATAL_ERROR("{:s}: row {:s} is in no worker", unique_name, _host_rows[i].key);
    return out;
}

void GnssGpuRecordAssemble::register_inputs(const Worker& w) {
    std::set<Buffer*> taken;
    for (size_t a = 0; a < w.rows.size(); ++a) {
        const Row& row = _rows[w.rows[a]];
        bool shared = false;
        for (size_t b = 0; b < a && row.tiles_buf; ++b)
            shared = shared
                     || (_rows[w.rows[b]].tiles_buf == row.tiles_buf
                         && _rows[w.rows[b]].ctl_buf == row.ctl_buf);
        if (shared)
            continue;
        for (Buffer* b : {row.in_buf, row.tiles_buf, row.ctl_buf}) {
            if (b == nullptr)
                continue;
            if (!taken.insert(b).second)
                FATAL_ERROR("{:s}: {:s} reads {:s} for two rows that do not share both tiles_buf "
                            "and ctl_buf",
                            _host_rows[w.rows[a]].name, w.name, b->buffer_name);
            b->register_consumer(w.name);
        }
    }
}

GnssGpuRecordAssemble::~GnssGpuRecordAssemble() {
    gnss::release_host(_host_rows, _rest_paths);
}

void GnssGpuRecordAssemble::main_thread() {
    if (_workers[0].cpus.empty()) {
        gnss::run_rows(_workers.size(), get_cpu_affinity(),
                       std::bind(&GnssGpuRecordAssemble::run_worker, this, std::placeholders::_1));
        return;
    }
    std::vector<std::string> names;
    std::vector<std::vector<int>> cpus;
    for (const Worker& w : _workers) {
        // A worker of one row takes the thread name its row's own stage had.
        names.push_back(w.rows.size() == 1 ? _host_rows[w.rows[0]].name : w.name);
        cpus.push_back(w.cpus);
    }
    gnss::run_workers(names, cpus,
                      std::bind(&GnssGpuRecordAssemble::run_pinned, this, std::placeholders::_1));
}

void GnssGpuRecordAssemble::run_pinned(size_t w) {
    const pid_t tid = syscall(SYS_gettid);
    register_tid(tid);
    run_worker(w);
    unregister_tid(tid);
}

void GnssGpuRecordAssemble::run_worker(size_t w) {
    const Worker& wk = _workers[w];
    while (!stop_thread) {
        for (size_t pos = 0; pos < wk.rows.size(); ++pos) {
            Row& row = _rows[wk.rows[pos]];
            const uint8_t* in = nullptr;
            if (row.in_buf)
                in = row.in_buf->wait_for_full_frame(wk.name, row.in_id);
            else if (row.assembled || assemble_frames(wk, pos))
                in = row.epl.get();
            if (in == nullptr)
                return;
            row.assembled = false;
            if (!run_records(wk.rows[pos], in))
                return;
            if (row.in_buf) {
                row.in_buf->mark_frame_empty(wk.name, row.in_id);
                row.in_id = (row.in_id + 1) % row.in_buf->num_frames;
            }
        }
    }
}

bool GnssGpuRecordAssemble::run_records(size_t i, const uint8_t* in) {
    Row& row = _rows[i];
    const std::string& name = _host_rows[i].name;
    Buffer* out_buf = row.out_buf;
    const int n_rec = row.chain.begin_frame(in);
    for (int r = 0; r < n_rec && !stop_thread; ++r) {
        float* out = (float*)out_buf->wait_for_empty_frame(name, row.out_id);
        if (out == nullptr)
            return false;
        const int64_t wstart = row.chain.assemble_record(r, out);
        emit_cube_window(i);
        if (out_buf->metadata_pool) {
            out_buf->allocate_new_metadata_object(row.out_id);
            get_gnss_chan_metadata(out_buf, row.out_id)->sample_seq = wstart;
        }
        out_buf->mark_frame_full(name, row.out_id);
        row.out_id = (row.out_id + 1) % out_buf->num_frames;
    }
    return true;
}

void GnssGpuRecordAssemble::read_gpu_row(Config& config, const gnss::HostRow& hr, Row& row) {
    const gnss_gpu::GpuRow g =
        gnss_gpu::gpu_row(config.get<nlohmann::json>(hr.cfg_path, "gnss_row"));
    // The tag names the chain the row was cut for; tie it to this row's own name.
    if (hr.name != "/" + g.name + "_n2assemble")
        FATAL_ERROR("{:s}: gnss_row {:s} is another chain's row", hr.name, g.name);
    gnss_gpu::EplShape& sh = row.epl_shape;
    if ((int)g.channels.size() != sh.n_chan || g.n_prn != sh.n_prn || g.lanes != sh.num_synth
        || g.ctl_bytes != gnss_gpu::ctl_bytes(sh.n_prn, sh.n_chan))
        FATAL_ERROR("{:s}: gnss_row {:s} is {:d} channels x {:d} PRN on {:d} lanes with {:d} "
                    "control bytes, this row {:d} x {:d} on {:d} with {:d}",
                    hr.name, g.name, g.channels.size(), g.n_prn, g.lanes, g.ctl_bytes, sh.n_chan,
                    sh.n_prn, sh.num_synth, gnss_gpu::ctl_bytes(sh.n_prn, sh.n_chan));
    if (g.ctl_offset + g.ctl_bytes > row.ctl_buf->frame_size)
        FATAL_ERROR("{:s}: control block bytes [{:d}, {:d}) are not inside ctl_buf's {:d}", hr.name,
                    g.ctl_offset, g.ctl_offset + g.ctl_bytes, row.ctl_buf->frame_size);
    sh.fg_total = config.get<int>(hr.cfg_path, "fg_total");
    sh.fg_offset = g.fg_offset;
    row.ctl_offset = g.ctl_offset;
    row.tag = g.tag();
}

bool GnssGpuRecordAssemble::assemble_frames(const Worker& w, size_t pos) {
    const Row& first = _rows[w.rows[pos]];
    Buffer* const ctl_buf = first.ctl_buf;
    Buffer* const tiles_buf = first.tiles_buf;
    const int ctl_id = first.ctl_id, tiles_id = first.tiles_id;
    const uint8_t* ctl = ctl_buf->wait_for_full_frame(w.name, ctl_id);
    if (ctl == nullptr)
        return false;
    // In a GPU half's shared array each block must be its row's: shape checks cannot tell two
    // rows of one shape apart.
    for (size_t p = pos; p < w.rows.size(); ++p) {
        const Row& row = _rows[w.rows[p]];
        if (row.tiles_buf != tiles_buf)
            continue;
        const int64_t tag = ((const gnss_gpu::FrameHdr*)(ctl + row.ctl_offset))->_pad1;
        if (tag != row.tag)
            FATAL_ERROR("{:s}: the control block at byte {:d} of ctl_buf carries row tag {:#x}, "
                        "this row expects {:#x} (magic, chain id, row, fg_offset, 16 bits each)",
                        _host_rows[w.rows[p]].name, row.ctl_offset, tag, row.tag);
    }
    const int32_t* tiles = (const int32_t*)tiles_buf->wait_for_full_frame(w.name, tiles_id);
    if (tiles == nullptr)
        return false;
    // cudaOutputData fills both from one GPU frame, the tiles with the correlator's metadata: its
    // first hop must be each control block's.
    const std::shared_ptr<chordMetadata> meta = get_chord_metadata(tiles_buf, tiles_id);
    for (; pos < w.rows.size(); ++pos) {
        Row& row = _rows[w.rows[pos]];
        if (row.tiles_buf != tiles_buf)
            continue;
        const std::string& name = _host_rows[w.rows[pos]].name;
        const uint8_t* block = ctl + row.ctl_offset;
        if (!meta || !meta->has_fpga_seq_num())
            FATAL_ERROR("{:s}: tiles frame {:d} carries no CHORD metadata with a sequence number",
                        name, tiles_id);
        const int64_t seq0 = ((const gnss_gpu::FrameHdr*)block)->seq0;
        if (meta->get_fpga_seq_num() * row.fft_len != seq0)
            FATAL_ERROR("{:s}: the tiles frame starts at hop {:d}, its control block at sample "
                        "{:d} (hop {:d} at fft_len {:d}): the two buffers are out of step",
                        name, meta->get_fpga_seq_num(), seq0, seq0 / row.fft_len, row.fft_len);
        row.epl_shape.assemble(name, block, tiles, row.epl.get(), row.epl_bytes);
        row.assembled = true;
        row.ctl_id = (ctl_id + 1) % ctl_buf->num_frames;
        row.tiles_id = (tiles_id + 1) % tiles_buf->num_frames;
    }
    ctl_buf->mark_frame_empty(w.name, ctl_id);
    tiles_buf->mark_frame_empty(w.name, tiles_id);
    return true;
}

void GnssGpuRecordAssemble::emit_cube_window(size_t i) {
    Row& row = _rows[i];
    if (!row.chain.cube_window_ready())
        return;
    // Never blocks (see the header): this stage is in the real-time path, so a full buffer
    // costs the archive a window, never the tracker a record.
    // is_frame_empty() is a safe pre-check because this row is the buffer's only producer.
    if (!row.cube_out_buf->is_frame_empty(row.cube_out_id)) {
        row.chain.take_cube_window(nullptr);
        return;
    }
    const std::string& name = _host_rows[i].name;
    uint8_t* f = row.cube_out_buf->wait_for_empty_frame(name, row.cube_out_id);
    if (f == nullptr)
        return; // shutting down
    const int64_t w0 = row.chain.take_cube_window(f);
    // A frame that feeds bufferSend needs a metadata object: bufferSend drops a frame without
    // one. Stamped like the record leg: the window's first sample, so a consumer after
    // bufferRecv can address the frame without parsing it.
    if (row.cube_out_buf->metadata_pool) {
        row.cube_out_buf->allocate_new_metadata_object(row.cube_out_id);
        if (auto* m = get_gnss_chan_metadata(row.cube_out_buf, row.cube_out_id))
            m->sample_seq = w0;
    }
    row.cube_out_buf->mark_frame_full(name, row.cube_out_id);
    row.cube_out_id = (row.cube_out_id + 1) % row.cube_out_buf->num_frames;
}
