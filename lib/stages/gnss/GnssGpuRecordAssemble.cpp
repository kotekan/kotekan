#include "GnssGpuRecordAssemble.hpp"

#include "GnssChanMetadata.hpp"
#include "StageFactory.hpp"
#include "chordMetadata.hpp"    // for chordMetadata, get_chord_metadata
#include "gnssChanPower.hpp"    // for ChanPower
#include "gnssProjSubspace.hpp" // for ProjBoard

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
            row.in_buf->register_consumer(hr.name);
        } else {
            row.tiles_buf =
                buffer_container.get_buffer(config.get<std::string>(hr.cfg_path, "tiles_buf"));
            row.tiles_buf->register_consumer(hr.name);
            row.ctl_buf =
                buffer_container.get_buffer(config.get<std::string>(hr.cfg_path, "ctl_buf"));
            row.ctl_buf->register_consumer(hr.name);
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
    gnss::rest_register_chains(unique_name, _host_rows, _rest_paths);
}

GnssGpuRecordAssemble::~GnssGpuRecordAssemble() {
    gnss::release_host(_host_rows, _rest_paths);
}

void GnssGpuRecordAssemble::main_thread() {
    gnss::run_rows(_rows.size(), get_cpu_affinity(),
                   std::bind(&GnssGpuRecordAssemble::run_row, this, std::placeholders::_1));
}

void GnssGpuRecordAssemble::run_row(size_t i) {
    Row& row = _rows[i];
    const std::string& name = _host_rows[i].name;
    Buffer* in_buf = row.in_buf;

    while (!stop_thread) {
        const uint8_t* in =
            in_buf ? in_buf->wait_for_full_frame(name, row.in_id) : next_epl_frame(i);
        if (in == nullptr)
            return;
        if (!run_records(i, in))
            return;
        if (in_buf) {
            in_buf->mark_frame_empty(name, row.in_id);
            row.in_id = (row.in_id + 1) % in_buf->num_frames;
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

const uint8_t* GnssGpuRecordAssemble::next_epl_frame(size_t i) {
    Row& row = _rows[i];
    const std::string& name = _host_rows[i].name;
    const uint8_t* ctl = row.ctl_buf->wait_for_full_frame(name, row.ctl_id);
    if (ctl == nullptr)
        return nullptr;
    ctl += row.ctl_offset;
    // In a GPU half's shared array the block must be this row's: shape checks cannot tell two
    // rows of one shape apart.
    const int64_t tag = ((const gnss_gpu::FrameHdr*)ctl)->_pad1;
    if (tag != row.tag)
        FATAL_ERROR("{:s}: the control block at byte {:d} of ctl_buf carries row tag {:#x}, this "
                    "row expects {:#x} (magic, chain id, row, fg_offset, 16 bits each)",
                    name, row.ctl_offset, tag, row.tag);
    const int32_t* tiles = (const int32_t*)row.tiles_buf->wait_for_full_frame(name, row.tiles_id);
    if (tiles == nullptr)
        return nullptr;
    // cudaOutputData fills both from one GPU frame, the tiles with the correlator's metadata: its
    // first hop must be the control block's.
    const std::shared_ptr<chordMetadata> meta = get_chord_metadata(row.tiles_buf, row.tiles_id);
    if (!meta || !meta->has_fpga_seq_num())
        FATAL_ERROR("{:s}: tiles frame {:d} carries no CHORD metadata with a sequence number", name,
                    row.tiles_id);
    const int64_t seq0 = ((const gnss_gpu::FrameHdr*)ctl)->seq0;
    if (meta->get_fpga_seq_num() * row.fft_len != seq0)
        FATAL_ERROR("{:s}: the tiles frame starts at hop {:d}, its control block at sample {:d} "
                    "(hop {:d} at fft_len {:d}): the two buffers are out of step",
                    name, meta->get_fpga_seq_num(), seq0, seq0 / row.fft_len, row.fft_len);
    row.epl_shape.assemble(name, ctl, tiles, row.epl.get(), row.epl_bytes);
    row.ctl_buf->mark_frame_empty(name, row.ctl_id);
    row.ctl_id = (row.ctl_id + 1) % row.ctl_buf->num_frames;
    row.tiles_buf->mark_frame_empty(name, row.tiles_id);
    row.tiles_id = (row.tiles_id + 1) % row.tiles_buf->num_frames;
    return row.epl.get();
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
